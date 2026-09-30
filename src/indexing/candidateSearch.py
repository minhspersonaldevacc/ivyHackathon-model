"""Stage 1 candidates followed by cached Stage 2 descriptor ranking."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from src.descriptors.descriptorPipeline import generateShapeDescriptor
from src.descriptors.descriptorSimilarity import compareTensorBatch, defaultWeights, normalizedWeights, properAxisFlips
from src.descriptors.descriptorStore import DescriptorStore
from src.errors import DependencyError, DescriptorError
from src.indexing.database import PartDatabase
from src.indexing.partClassifier import classifyPart
from src.indexing.partIndexer import fileFingerprint, hashFile
from src.models.sphereDescriptor import DescriptorConfig, SphereDescriptor


@dataclass(frozen=True)
class GeometricMatch:
    partId: str
    distance: float
    axisFlip: tuple[int, int, int]
    cosineSimilarity: float | None = None


@dataclass
class MatchReport:
    matches: list[GeometricMatch]
    metric: str
    requestedCandidates: int
    comparedCandidates: int
    unavailablePartIds: list[str]
    timings: dict
    queryMetadata: dict = field(default_factory=dict)

    def toDict(self):
        return asdict(self)


def rankCandidates(
    queryDescriptor: SphereDescriptor,
    candidatePartIds: Iterable[str],
    store: DescriptorStore,
    topK: int = 50,
    *,
    weights=defaultWeights,
    metric: str = "euclidean",
    axisFlips: bool = True,
    skipMissing: bool = False,
    batchSize: int = 128,
) -> MatchReport:
    """Rank a metadata candidate set in bounded-memory NumPy batches.

    Missing, stale, failed, or incompatible descriptors cause an explicit error
    by default. With skipMissing=True, omitted IDs are included in the report.
    Corrupt stored blobs always raise; they are never silently treated as matches.
    """
    if type(topK) is not int or topK <= 0:
        raise ValueError("topK must be a positive integer")
    if type(batchSize) is not int or not 1 <= batchSize <= 400:
        raise ValueError("batchSize must be an integer between 1 and 400")
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    normalizedWeights(weights)
    # Keep tensor batches near 16 MiB even when callers choose a much denser grid.
    batchSize = min(batchSize, max(1, (16 * 1024 ** 2) // queryDescriptor.tensor.nbytes))
    partIds = list(dict.fromkeys(candidatePartIds))
    if any(not isinstance(partId, str) or not partId for partId in partIds):
        raise ValueError("candidatePartIds must contain nonempty strings")
    startedAt = perf_counter()
    matches, unavailable = [], []
    storageSeconds = comparisonSeconds = 0.0
    for start in range(0, len(partIds), batchSize):
        batch = partIds[start:start + batchSize]
        readAt = perf_counter()
        descriptors = store.loadDescriptors(batch, queryDescriptor.config)
        storageSeconds += perf_counter() - readAt
        availableIds = [partId for partId in batch if partId in descriptors]
        unavailable.extend(partId for partId in batch if partId not in descriptors)
        if not availableIds:
            continue
        compareAt = perf_counter()
        tensors = np.stack([descriptors[partId].tensor for partId in availableIds])
        distances, flipIndices = compareTensorBatch(
            queryDescriptor, tensors, weights=weights, metric=metric, axisFlips=axisFlips
        )
        comparisonSeconds += perf_counter() - compareAt
        matches.extend(
            GeometricMatch(
                partId, float(distance), properAxisFlips[int(flipIndex)],
                1 - float(distance) if metric == "cosine" else None,
            )
            for partId, distance, flipIndex in zip(availableIds, distances, flipIndices)
        )
    if unavailable and not skipMissing:
        raise DescriptorError(
            f"{len(unavailable)} candidate(s) lack a current compatible descriptor; "
            "run build-descriptors with matching settings, or explicitly use skipMissing=True. "
            f"First IDs: {', '.join(unavailable[:5])}"
        )
    comparedCount = len(matches)
    matches.sort(key=lambda match: (match.distance, match.partId))
    elapsed = perf_counter() - startedAt
    return MatchReport(
        matches[:topK], metric, len(partIds), comparedCount, unavailable,
        {"storageSeconds": storageSeconds, "comparisonSeconds": comparisonSeconds,
         "searchTotalSeconds": elapsed,
         "microsecondsPerCandidate": elapsed * 1e6 / comparedCount if comparedCount else 0.0},
    )


def findGeometricMatches(
    queryDescriptor: SphereDescriptor,
    candidatePartIds: Iterable[str],
    topK: int = 50,
    *,
    databasePath: str | Path = "parts.sqlite3",
    weights=defaultWeights,
    metric: str = "euclidean",
    axisFlips: bool = True,
    skipMissing: bool = False,
) -> list[GeometricMatch]:
    with PartDatabase(databasePath, readOnly=True) as database:
        store = DescriptorStore(database, initialize=False)
        return rankCandidates(
            queryDescriptor, candidatePartIds, store, topK, weights=weights,
            metric=metric, axisFlips=axisFlips, skipMissing=skipMissing,
        ).matches


def findMatchesForStep(
    filePath: str | Path, *, databasePath: str | Path = "parts.sqlite3",
    config: DescriptorConfig | None = None, dimensionTolerance: float = 0.1,
    volumeRange: tuple[float, float] | None = None, allFamilies: bool = False,
    topK: int = 50, weights=defaultWeights, metric: str = "euclidean",
    skipMissing: bool = False,
) -> MatchReport:
    """Load only the new query, then use Stage 1 SQL and stored descriptors."""
    config = config or DescriptorConfig()
    try:
        from src.cad.geometryExtractor import extractGeometry
        from src.cad.stepLoader import loadStep
    except ImportError as error:
        raise DependencyError("Activate the pythonOCC environment from environment.yml: " + str(error)) from error
    queryPath = Path(filePath).expanduser().resolve()
    before = fileFingerprint(queryPath)
    startedAt = perf_counter()
    sourceHash = hashFile(queryPath)
    loadAt = perf_counter()
    shape = loadStep(queryPath)
    loadSeconds = perf_counter() - loadAt
    geometry = extractGeometry(shape)
    family = None if allFamilies else classifyPart(geometry)
    queryDescriptor = generateShapeDescriptor(shape, config)
    if fileFingerprint(queryPath) != before:
        raise DescriptorError("Query STEP changed while processing")
    queryDescriptor = replace(
        queryDescriptor, sourceHash=sourceHash,
        timings={**queryDescriptor.timings, "stepLoadSeconds": loadSeconds,
                 "totalSeconds": perf_counter() - startedAt},
    )
    with PartDatabase(databasePath, readOnly=True) as database:
        filterAt = perf_counter()
        candidates = database.findCandidates(
            family, (geometry.sizeX, geometry.sizeY, geometry.sizeZ), dimensionTolerance, volumeRange
        )
        filterSeconds = perf_counter() - filterAt
        store = DescriptorStore(database, initialize=False)
        report = rankCandidates(
            queryDescriptor, (part.partId for part in candidates), store, topK,
            weights=weights, metric=metric, skipMissing=skipMissing,
        )
    report.timings["stage1FilterSeconds"] = filterSeconds
    report.queryMetadata = {
        "filePath": str(queryPath), "shapeFamily": family,
        "dimensionsMm": [geometry.sizeX, geometry.sizeY, geometry.sizeZ],
        "descriptor": queryDescriptor.summary(),
    }
    return report
