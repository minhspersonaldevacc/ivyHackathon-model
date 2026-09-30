"""A deterministic median-split bounding volume hierarchy for triangles."""

from dataclasses import dataclass

import numpy as np

from src.models.mesh import Mesh


@dataclass
class BvhNode:
    lower: np.ndarray
    upper: np.ndarray
    left: int = -1
    right: int = -1
    triangleIds: np.ndarray | None = None


class TriangleBvh:
    def __init__(self, mesh: Mesh, leafSize: int = 32):
        if type(leafSize) is not int or leafSize < 1:
            raise ValueError("leafSize must be a positive integer")
        self.mesh = mesh
        self.points = mesh.vertices[mesh.triangles]
        self.edge1 = self.points[:, 1] - self.points[:, 0]
        self.edge2 = self.points[:, 2] - self.points[:, 0]
        self.normalLengths = np.linalg.norm(np.cross(self.edge1, self.edge2), axis=1)
        if np.any(self.normalLengths <= 0):
            raise ValueError("Cannot build a BVH for zero-area triangles")
        self.triangleLower = self.points.min(axis=1)
        self.triangleUpper = self.points.max(axis=1)
        self.centers = self.points.mean(axis=1)
        self.nodes: list[BvhNode] = []
        self.leafSize = leafSize
        self._build(np.arange(len(mesh.triangles), dtype=np.int64))

    def _build(self, triangleIds: np.ndarray) -> int:
        nodeIndex = len(self.nodes)
        node = BvhNode(self.triangleLower[triangleIds].min(axis=0), self.triangleUpper[triangleIds].max(axis=0))
        self.nodes.append(node)
        if len(triangleIds) <= self.leafSize:
            node.triangleIds = triangleIds
        else:
            axis = int(np.argmax(np.ptp(self.centers[triangleIds], axis=0)))
            order = np.argsort(self.centers[triangleIds, axis], kind="stable")
            ordered = triangleIds[order]
            middle = len(ordered) // 2
            node.left = self._build(ordered[:middle])
            node.right = self._build(ordered[middle:])
        return nodeIndex

    def candidateTriangles(
        self, direction: np.ndarray, origin: np.ndarray, tolerance: float = 1e-7,
    ) -> np.ndarray:
        """Traverse only boxes met by the forward ray; never scan every face."""
        parallel = np.abs(direction) < 1e-15
        inverse = np.divide(1.0, direction, out=np.zeros(3), where=~parallel)
        stack, leaves = [0], []
        while stack:
            node = self.nodes[stack.pop()]
            lower, upper = node.lower - tolerance, node.upper + tolerance
            if np.any(parallel & ((origin < lower) | (origin > upper))):
                continue
            first, last = (lower - origin) * inverse, (upper - origin) * inverse
            near = np.where(parallel, -np.inf, np.minimum(first, last)).max()
            far = np.where(parallel, np.inf, np.maximum(first, last)).min()
            if far < max(near, -tolerance):
                continue
            if node.triangleIds is not None:
                leaves.append(node.triangleIds)
            else:
                stack.extend((node.right, node.left))
        return np.concatenate(leaves) if leaves else np.empty(0, dtype=np.int64)
