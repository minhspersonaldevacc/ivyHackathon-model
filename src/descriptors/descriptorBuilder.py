"""Offline descriptor preprocessing; unchanged historical parts are never reopened."""

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Callable

from src.descriptors.descriptorPipeline import buildDescriptorFromStep
from src.descriptors.descriptorStore import DescriptorStore
from src.errors import CadError
from src.models.sphereDescriptor import DescriptorConfig


@dataclass
class DescriptorBuildSummary:
    discovered: int = 0
    built: int = 0
    skipped: int = 0
    failed: int = 0
    skippedFailed: int = 0
    elapsedSeconds: float = 0.0
    generationTimings: dict = field(default_factory=dict)

    def toDict(self):
        return asdict(self)


def buildDescriptors(
    store: DescriptorStore,
    config: DescriptorConfig | None = None,
    *,
    retryFailed: bool = False,
    rebuild: bool = False,
    limit: int | None = None,
    onResult: Callable[[dict], None] | None = None,
) -> DescriptorBuildSummary:
    config = config or DescriptorConfig()
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    summary = DescriptorBuildSummary()
    startedAt = perf_counter()
    for part, status, sourceHash, previousError in store.iterPartStates(config):
        summary.discovered += 1
        error, descriptor = previousError, None
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
                descriptor = buildDescriptorFromStep(part.filePath, config, expectedSourceHash=part.sourceHash)
                store.storeDescriptor(part, descriptor)
                resultStatus, error = "built", None
                summary.built += 1
                for name, duration in descriptor.timings.items():
                    summary.generationTimings[name] = summary.generationTimings.get(name, 0.0) + duration
            except (CadError, OSError, ValueError) as failure:
                resultStatus, error = "failed", str(failure)
                summary.failed += 1
                store.recordFailure(part, config, error)
        if onResult is not None:
            onResult({
                "partId": part.partId, "filePath": part.filePath, "status": resultStatus,
                "error": error, "timings": descriptor.timings if descriptor is not None else None,
            })
    summary.elapsedSeconds = perf_counter() - startedAt
    return summary
