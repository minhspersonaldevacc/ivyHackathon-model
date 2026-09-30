"""Offline incremental Stage 3 preprocessing and per-part failure recovery."""

from dataclasses import asdict, dataclass, field
from time import perf_counter

from src.detailedSimilarity.geometryPipeline import buildDetailedGeometryFromStep
from src.errors import CadError
from src.models.detailedGeometry import DetailedGeometryConfig


@dataclass
class GeometryBuildSummary:
    discovered: int = 0
    built: int = 0
    skipped: int = 0
    failed: int = 0
    skippedFailed: int = 0
    elapsedSeconds: float = 0.0
    generationTimings: dict = field(default_factory=dict)

    def toDict(self):
        return asdict(self)


def buildDetailedGeometries(store, config=None, *, retryFailed=False, rebuild=False, limit=None, onResult=None):
    config = config or DetailedGeometryConfig()
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    summary = GeometryBuildSummary()
    startedAt = perf_counter()
    for part, status, sourceHash, previousError in store.iterPartStates(config):
        summary.discovered += 1
        geometry, error = None, previousError
        if not rebuild and sourceHash == part.sourceHash and status == "complete":
            summary.skipped += 1
            resultStatus = "skipped"
        elif not rebuild and not retryFailed and sourceHash == part.sourceHash and status == "failed":
            summary.skippedFailed += 1
            resultStatus = "skippedFailed"
        else:
            if limit is not None and summary.built + summary.failed >= limit:
                summary.discovered -= 1
                break
            store.beginPart(part, config)
            try:
                geometry = buildDetailedGeometryFromStep(part.filePath, config, expectedSourceHash=part.sourceHash)
                store.storeGeometry(part, geometry)
                resultStatus, error = "built", None
                summary.built += 1
                for name, duration in geometry.timings.items():
                    summary.generationTimings[name] = summary.generationTimings.get(name, 0.0) + duration
            except (CadError, OSError, ValueError) as failure:
                resultStatus, error = "failed", str(failure)
                store.recordFailure(part, config, error)
                summary.failed += 1
        if onResult is not None:
            onResult({"partId": part.partId, "filePath": part.filePath, "status": resultStatus,
                      "error": error, "timings": geometry.timings if geometry is not None else None})
    summary.elapsedSeconds = perf_counter() - startedAt
    return summary
