"""A triangle mesh independent of OpenCascade."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    triangles: np.ndarray
    triangleBodyIds: np.ndarray | None = None

    def __post_init__(self):
        vertices = np.array(self.vertices, dtype=np.float64, copy=True)
        sourceTriangles = np.asarray(self.triangles)
        if sourceTriangles.dtype.kind not in "iu":
            raise ValueError("Triangle indices must be integers")
        triangles = np.array(sourceTriangles, dtype=np.int64, copy=True)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError("vertices must have shape (N, 3), with at least four vertices")
        if not np.isfinite(vertices).all():
            raise ValueError("Mesh vertices must be finite")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 4:
            raise ValueError("triangles must have shape (M, 3), with at least four triangles")
        if triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError("Triangle index outside the vertex array")
        if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
            raise ValueError("Mesh contains a triangle with repeated vertices")
        if self.triangleBodyIds is None:
            bodyIds = np.zeros(len(triangles), dtype=np.int64)
        else:
            sourceIds = np.asarray(self.triangleBodyIds)
            if sourceIds.shape != (len(triangles),) or sourceIds.dtype.kind not in "iu":
                raise ValueError("triangleBodyIds must be an integer array of shape (M,)")
            bodyIds = np.array(sourceIds, dtype=np.int64, copy=True)
            if np.any(bodyIds < 0):
                raise ValueError("Body IDs must be nonnegative")
        for array in (vertices, triangles, bodyIds):
            array.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(self, "triangleBodyIds", bodyIds)


def validateClosedMesh(mesh: Mesh) -> None:
    """Check manifold closure and consistent winding separately for each body."""
    triangles = mesh.triangles
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    bodyIds = np.tile(mesh.triangleBodyIds, 3)
    keys = np.column_stack((bodyIds, np.sort(edges, axis=1)))
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    winding = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if np.any(counts != 2) or np.any(winding != 0):
        raise ValueError("Mesh is not closed and consistently oriented; check tessellation and seam welding")
