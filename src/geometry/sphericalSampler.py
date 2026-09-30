"""Equal-area cells in mu=cos(theta) and azimuth, sampled at cell centers."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def sphericalDirections(thetaResolution: int = 32, phiResolution: int = 64) -> np.ndarray:
    if type(thetaResolution) is not int or type(phiResolution) is not int or thetaResolution < 2 or phiResolution < 4:
        raise ValueError("Use thetaResolution >= 2 and phiResolution >= 4")
    mu = 1 - 2 * (np.arange(thetaResolution) + 0.5) / thetaResolution
    phi = 2 * np.pi * (np.arange(phiResolution) + 0.5) / phiResolution
    radial = np.sqrt(np.maximum(0, 1 - mu ** 2))
    directions = np.empty((thetaResolution, phiResolution, 3), dtype=np.float64)
    directions[..., 0] = radial[:, None] * np.cos(phi)
    directions[..., 1] = radial[:, None] * np.sin(phi)
    directions[..., 2] = mu[:, None]
    directions.setflags(write=False)
    return directions


def axisFlipVariants(tensor: np.ndarray):
    """Four determinant +1 axis flips, mapped exactly on the equal-area grid.

    Odd sign flips would identify mirror images; they are deliberately excluded.
    This handles PCA signs, not arbitrary rotations inside degenerate eigenspaces.
    """
    phiCount = tensor.shape[1]
    if phiCount % 2:
        raise ValueError("An even phiResolution is needed for axis flips")
    phiIndices = np.arange(phiCount)
    yield tensor
    yield tensor[:, (phiIndices + phiCount // 2) % phiCount, :]  # (-x,-y,+z)
    yield tensor[::-1, (phiCount // 2 - phiIndices - 1) % phiCount, :]  # (-x,+y,-z)
    yield tensor[::-1, (-phiIndices - 1) % phiCount, :]  # (+x,-y,-z)
