"""Raw deterministic distances; no percentage or learned score calibration."""

import numpy as np

from src.geometry.sphericalSampler import axisFlipVariants
from src.models.sphereDescriptor import SphereDescriptor


defaultWeights = (1.0, 1.0, 0.5, 1.5)
properAxisFlips = ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))


def normalizedWeights(weights=defaultWeights) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0) or values.max() <= 0:
        raise ValueError("Provide four finite, nonnegative channel weights with a positive sum")
    scaled = values / values.max()
    return scaled / scaled.sum()


def compareTensorBatch(
    query: SphereDescriptor, candidateTensors: np.ndarray, *,
    weights=defaultWeights, metric: str = "euclidean", axisFlips: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return distances and best flip indices for one compatible tensor batch.

    Euclidean distance = sqrt(mean_over_rays(sum_c(w[c] * error[c]^2) / sum(w))).
    Cosine mode returns 1 - cosine, so both metrics sort in ascending order.
    Stored counts have already been clipped/scaled independently of distances.
    """
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    channelWeights = normalizedWeights(weights)
    candidates = np.asarray(candidateTensors, dtype=np.float32)
    if candidates.ndim != 4 or candidates.shape[1:] != query.tensor.shape:
        raise ValueError("Candidate tensors must have shape (batch, theta, phi, 4)")
    if not np.isfinite(candidates).all():
        raise ValueError("Candidate tensors must contain only finite values")
    best = np.full(len(candidates), np.inf)
    bestFlips = np.zeros(len(candidates), dtype=np.int64)
    variants = axisFlipVariants(query.tensor) if axisFlips else (query.tensor,)
    if metric == "cosine":
        flat = candidates.reshape(len(candidates), -1)
        candidateNorms = np.sqrt(np.einsum("ij,ij->i", flat, flat, dtype=np.float64))
    for flipIndex, variant in enumerate(variants):
        if metric == "euclidean":
            difference = candidates - variant
            squaredErrors = np.einsum(
                "bijk,bijk,k->b", difference, difference, channelWeights,
                dtype=np.float64, optimize=False,
            )
            distances = np.sqrt(squaredErrors / (query.tensor.shape[0] * query.tensor.shape[1]))
        else:
            queryFlat = variant.reshape(-1)
            queryNorm = np.sqrt(np.einsum("i,i->", queryFlat, queryFlat, dtype=np.float64))
            product = candidateNorms * queryNorm
            dot = np.einsum("ij,j->i", flat, queryFlat, dtype=np.float64)
            similarity = np.divide(dot, product, out=np.zeros_like(dot), where=product > 0)
            similarity[(candidateNorms == 0) & (queryNorm == 0)] = 1.0
            distances = 1 - np.clip(similarity, -1, 1)
        improved = distances < best
        best[improved], bestFlips[improved] = distances[improved], flipIndex
    return best, bestFlips


def compareDescriptors(
    first: SphereDescriptor, second: SphereDescriptor, *, weights=defaultWeights,
    metric: str = "euclidean", axisFlips: bool = True,
) -> float:
    if first.config.configKey != second.config.configKey:
        raise ValueError("Descriptors have incompatible sampling, normalization, or meshing settings")
    distances, _ = compareTensorBatch(
        first, second.tensor[None], weights=weights, metric=metric, axisFlips=axisFlips
    )
    return float(distances[0])


def cosineSimilarity(first: SphereDescriptor, second: SphereDescriptor, *, axisFlips: bool = True) -> float:
    """Unweighted flattened-vector cosine; two all-zero vectors have similarity 1."""
    return 1 - compareDescriptors(first, second, metric="cosine", axisFlips=axisFlips)
