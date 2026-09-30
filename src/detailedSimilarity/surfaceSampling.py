"""Repeatable area sampling; no random state and no OpenCascade dependencies."""

import numpy as np

from src.models.mesh import Mesh


def radicalInverse(indices, base):
    values = np.asarray(indices, dtype=np.int64).copy()
    result = np.zeros(values.shape, dtype=np.float64)
    factor = 1.0 / base
    while np.any(values):
        result += (values % base) * factor
        values //= base
        factor /= base
    return result


def sampleSurface(mesh: Mesh, sampleCount: int) -> np.ndarray:
    """Stratify by triangle area and use low-discrepancy barycentric coordinates.

    This approximates area coverage. Separate B-rep face samples in the pipeline
    ensure small faces also contribute to comparison.
    """
    triangles = mesh.vertices[mesh.triangles]
    centers = triangles.mean(axis=1)
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    triangles = triangles[order]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    if np.any(areas <= 0) or not np.isfinite(areas).all():
        raise ValueError("Surface sampling needs finite, positive-area triangles")
    cumulative = np.cumsum(areas)
    triangleIds = np.searchsorted(cumulative, (np.arange(sampleCount) + 0.5) * cumulative[-1] / sampleCount)
    selected = triangles[triangleIds]
    sequence = np.arange(1, sampleCount + 1)
    first = np.sqrt(radicalInverse(sequence, 2))
    second = radicalInverse(sequence, 3)
    return ((1 - first[:, None]) * selected[:, 0] +
            (first * (1 - second))[:, None] * selected[:, 1] +
            (first * second)[:, None] * selected[:, 2])
