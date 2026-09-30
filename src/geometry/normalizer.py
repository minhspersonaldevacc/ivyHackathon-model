"""Center, orient, and scale meshes without modifying the original geometry."""

import numpy as np

from src.models.mesh import Mesh, validateClosedMesh


def normalizeMesh(mesh: Mesh, *, degeneracyTolerance: float = 0.001) -> tuple[Mesh, dict]:
    """Use the mesh volume centroid and vertex PCA, retaining the inverse transform.

    Covariance uses unique mesh vertices with equal weights. Tessellation density
    can therefore affect axes. Repeated eigenvalues are reported, not claimed to
    have a unique canonical orientation.
    """
    if not np.isfinite(degeneracyTolerance) or not 0 < degeneracyTolerance < 1:
        raise ValueError("degeneracyTolerance must be finite and between 0 and 1")
    validateClosedMesh(mesh)
    # Shift and scale before integration to avoid cancellation for distant parts.
    reference = mesh.vertices.mean(axis=0)
    initialScale = float(np.max(np.linalg.norm(mesh.vertices - reference, axis=1)))
    if not np.isfinite(initialScale) or initialScale <= 0:
        raise ValueError("Cannot normalize a mesh with zero or non-finite radius")
    local = (mesh.vertices - reference) / initialScale
    trianglePoints = local[mesh.triangles]
    signedVolumes = np.einsum(
        "ij,ij->i", trianglePoints[:, 0], np.cross(trianglePoints[:, 1], trianglePoints[:, 2])
    ) / 6.0
    for bodyId in np.unique(mesh.triangleBodyIds):
        if signedVolumes[mesh.triangleBodyIds == bodyId].sum() <= 0:
            raise ValueError("Each mesh body must have positive volume and outward winding")
    volume = signedVolumes.sum()
    localCentroid = (signedVolumes[:, None] * trianglePoints.sum(axis=1) / 4).sum(axis=0) / volume
    centered = local - localCentroid
    vertexCentered = centered - centered.mean(axis=0)
    covariance = vertexCentered.T @ vertexCentered / len(vertexCentered)
    eigenvalues, axes = np.linalg.eigh(covariance)
    eigenvalues, axes = eigenvalues[::-1], axes[:, ::-1].copy()
    if eigenvalues[-1] <= 0 or not np.isfinite(eigenvalues).all():
        raise ValueError("Mesh covariance is degenerate")

    # Odd moments resolve most signs equivariantly; symmetric projections fall
    # back to the farthest vertex. Comparisons also try all four proper flips.
    projections = centered @ axes
    for axisIndex in range(3):
        projected = projections[:, axisIndex]
        moment = float(np.mean(projected ** 3))
        signReference = moment if abs(moment) > 1e-12 else projected[np.argmax(np.abs(projected))]
        if signReference < 0:
            axes[:, axisIndex] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1

    rotated = centered @ axes
    localRadius = float(np.max(np.linalg.norm(rotated, axis=1)))
    radiusScale = localRadius * initialScale
    centroid = reference + localCentroid * initialScale
    normalized = Mesh(rotated / localRadius, mesh.triangles, mesh.triangleBodyIds)
    relativeGaps = (eigenvalues[:-1] - eigenvalues[1:]) / eigenvalues[0]
    metadata = {
        "centroid": centroid.tolist(), "radiusScale": radiusScale,
        "pcaAxes": axes.tolist(), "pcaEigenvalues": (eigenvalues * initialScale ** 2).tolist(),
        "ambiguousAxisPairs": [[index, index + 1] for index, gap in enumerate(relativeGaps) if gap <= degeneracyTolerance],
        "centroidMethod": "oriented-mesh-volume",
        "covarianceMethod": "equal-weight-unique-vertices",
        "inverseTransform": "world = (normalized * radiusScale) @ pcaAxes.T + centroid",
    }
    if not np.isfinite(centroid).all() or not np.isfinite(radiusScale) or radiusScale <= 0:
        raise ValueError("Normalization transform is non-finite")
    return normalized, metadata
