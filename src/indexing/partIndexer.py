"""Coordinate file fingerprints, geometry extraction, classification, and storage."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import stat
from typing import Callable, Iterator
from uuid import NAMESPACE_URL, uuid5

from src.errors import CadError, DependencyError
from src.indexing.database import PartDatabase
from src.indexing.partClassifier import classifyPart
from src.models.partMetadata import PartMetadata, classifierVersion, extractorVersion


def fileFingerprint(filePath: Path) -> dict[str, int]:
    fileStat = filePath.stat()
    if not stat.S_ISREG(fileStat.st_mode):
        raise CadError(f"Source is not a regular file: {filePath}")
    return {
        "fileSize": fileStat.st_size,
        "modifiedTimeNs": fileStat.st_mtime_ns,
        "changedTimeNs": fileStat.st_ctime_ns,
    }


def hashFile(filePath: Path) -> str:
    with filePath.open("rb") as sourceFile:
        return hashlib.file_digest(sourceFile, "sha256").hexdigest()


def analyzeStep(filePath: Path, sourceHash: str) -> PartMetadata:
    """The only boundary joining geometry extraction and family classification."""
    try:
        from src.cad.geometryExtractor import extractGeometry
        from src.cad.stepLoader import loadStep
    except ImportError as error:
        raise DependencyError(
            "pythonOCC could not be imported. Run 'conda env create -f environment.yml' "
            "and 'conda activate cad-stage1'. Original error: " + str(error)
        ) from error

    geometry = extractGeometry(loadStep(filePath))
    return PartMetadata(
        **geometry.toDict(),
        partId=uuid5(NAMESPACE_URL, filePath.as_uri()).hex,
        filePath=str(filePath),
        sourceHash=sourceHash,
        shapeFamily=classifyPart(geometry),
        extractedAt=datetime.now(timezone.utc).isoformat(),
    )


def inspectPart(filePath: str | Path) -> PartMetadata:
    """Measure and classify without opening or writing any database."""
    sourcePath = Path(filePath).expanduser().resolve()
    before = fileFingerprint(sourcePath)
    metadata = analyzeStep(sourcePath, hashFile(sourcePath))
    if fileFingerprint(sourcePath) != before:
        raise CadError("Source changed while it was being inspected; retry once the file is stable")
    return metadata


def iterStepFiles(inputPath: str | Path) -> Iterator[Path]:
    """Recurse deterministically, accepting case-insensitive .step and .stp."""
    sourcePath = Path(inputPath).expanduser().resolve()
    if sourcePath.is_file():
        if sourcePath.suffix.lower() not in {".step", ".stp"}:
            raise CadError(f"Unsupported file extension: {sourcePath.suffix}")
        yield sourcePath
        return
    if not sourcePath.is_dir():
        raise CadError(f"Input path does not exist or is not a directory: {sourcePath}")

    def reportWalkError(error):
        raise error

    # Do not follow directory symlinks, which can introduce cycles.
    for directory, directoryNames, fileNames in os.walk(sourcePath, onerror=reportWalkError):
        directoryNames.sort()
        for fileName in sorted(fileNames):
            if Path(fileName).suffix.lower() in {".step", ".stp"}:
                yield Path(directory, fileName).resolve()


@dataclass(frozen=True)
class IndexResult:
    filePath: str
    status: str
    error: str | None = None


@dataclass
class IndexSummary:
    discovered: int = 0
    indexed: int = 0
    skipped: int = 0
    reclassified: int = 0
    failed: int = 0
    skippedFailed: int = 0


def refreshClassification(metadata: PartMetadata) -> PartMetadata:
    """Rule changes can reuse stored geometry without reading a STEP file."""
    if metadata.classifierVersion == classifierVersion:
        return metadata
    return replace(
        metadata, shapeFamily=classifyPart(metadata), classifierVersion=classifierVersion
    )


def indexFile(
    filePath: str | Path,
    database: PartDatabase,
    *,
    retryFailed: bool = False,
    verifyContent: bool = False,
) -> IndexResult:
    sourcePath = Path(filePath).expanduser().resolve()
    canonicalPath = str(sourcePath)
    fingerprint = {"fileSize": -1, "modifiedTimeNs": -1, "changedTimeNs": -1}
    sourceHash = None
    try:
        fingerprint = fileFingerprint(sourcePath)
        fileState = database.getFileState(canonicalPath)
        sameExtractor = fileState is not None and fileState["extractorVersion"] == extractorVersion
        sameFingerprint = fileState is not None and all(
            fileState[key] == value for key, value in fingerprint.items()
        )

        if sameExtractor and sameFingerprint and not verifyContent:
            if fileState["status"] == "failed" and not retryFailed:
                return IndexResult(canonicalPath, "skippedFailed", fileState["error"])
            if fileState["status"] == "indexed":
                storedPart = database.getPart(canonicalPath)
                if storedPart is not None:
                    updatedPart = refreshClassification(storedPart)
                    if updatedPart is not storedPart:
                        database.storePart(updatedPart, fingerprint)
                        return IndexResult(canonicalPath, "reclassified")
                    return IndexResult(canonicalPath, "skipped")

        # Commit this state before any expensive operation: interrupted or failed
        # work must not leave an old version visible in searches.
        database.beginFile(canonicalPath, fingerprint)
        sourceHash = hashFile(sourcePath)
        if fileFingerprint(sourcePath) != fingerprint:
            raise CadError("Source changed while hashing; retry once the file is stable")

        if sameExtractor and fileState["sourceHash"] == sourceHash:
            if fileState["status"] == "failed" and not retryFailed:
                database.recordFailure(canonicalPath, fingerprint, fileState["error"], sourceHash)
                return IndexResult(canonicalPath, "skippedFailed", fileState["error"])
            if fileState["status"] == "indexed":
                storedPart = database.getPart(canonicalPath, includeInactive=True)
                if storedPart is not None:
                    updatedPart = refreshClassification(storedPart)
                    database.storePart(updatedPart, fingerprint)
                    status = "reclassified" if updatedPart is not storedPart else "skipped"
                    return IndexResult(canonicalPath, status)

        metadata = analyzeStep(sourcePath, sourceHash)
        if fileFingerprint(sourcePath) != fingerprint:
            raise CadError("Source changed during geometry extraction; retry once the file is stable")
        database.storePart(metadata, fingerprint)
        return IndexResult(canonicalPath, "indexed")
    except (CadError, OSError) as error:
        database.recordFailure(canonicalPath, fingerprint, str(error), sourceHash)
        return IndexResult(canonicalPath, "failed", str(error))


def indexPath(
    inputPath: str | Path,
    database: PartDatabase,
    *,
    retryFailed: bool = False,
    verifyContent: bool = False,
    onResult: Callable[[IndexResult], None] | None = None,
) -> IndexSummary:
    """Process one file at a time; a bad file does not abort the remaining batch."""
    summary = IndexSummary()
    seenPaths = set()
    for filePath in iterStepFiles(inputPath):
        # File symlinks may resolve to a part already visited in this scan.
        if filePath in seenPaths:
            continue
        seenPaths.add(filePath)
        result = indexFile(filePath, database, retryFailed=retryFailed, verifyContent=verifyContent)
        summary.discovered += 1
        setattr(summary, result.status, getattr(summary, result.status) + 1)
        if onResult is not None:
            onResult(result)
    database.optimize()
    return summary
