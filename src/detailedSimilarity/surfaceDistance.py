"""Point-to-triangle closest points with branch-and-bound over the existing BVH."""

import heapq

import numpy as np

from src.geometry.bvh import TriangleBvh


def closestOnTriangles(point, triangles):
    """Project onto each triangle plane or its three clamped edge segments."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    first, second = b - a, c - a
    normal = np.cross(first, second)
    normalSquared = np.einsum("ij,ij->i", normal, normal)
    offset = point - a
    projection = point - (np.einsum("ij,ij->i", offset, normal) / normalSquared)[:, None] * normal
    projected = projection - a
    d00 = np.einsum("ij,ij->i", first, first)
    d01 = np.einsum("ij,ij->i", first, second)
    d11 = np.einsum("ij,ij->i", second, second)
    d20 = np.einsum("ij,ij->i", projected, first)
    d21 = np.einsum("ij,ij->i", projected, second)
    # Cross-product norm is numerically safer for thin triangles.
    u = (d11 * d20 - d01 * d21) / normalSquared
    v = (d00 * d21 - d01 * d20) / normalSquared
    inside = (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)
    options = [projection]
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        ratio = np.einsum("ij,ij->i", point - start, edge) / np.einsum("ij,ij->i", edge, edge)
        options.append(start + np.clip(ratio, 0, 1)[:, None] * edge)
    points = np.stack(options, axis=1)
    distances = np.sum((points - point) ** 2, axis=2)
    distances[~inside, 0] = np.inf
    triangleIndex, optionIndex = np.unravel_index(np.argmin(distances), distances.shape)
    return points[triangleIndex, optionIndex], float(distances[triangleIndex, optionIndex])


class SurfaceDistanceIndex:
    def __init__(self, mesh):
        self.bvh = TriangleBvh(mesh)

    def distances(self, points):
        result = np.empty(len(points))
        for index, point in enumerate(points):
            bestSquared = np.inf
            queue = [(0.0, 0)]
            while queue:
                bound, nodeIndex = heapq.heappop(queue)
                if bound > bestSquared:
                    break
                node = self.bvh.nodes[nodeIndex]
                if node.triangleIds is not None:
                    _, squared = closestOnTriangles(point, self.bvh.points[node.triangleIds])
                    bestSquared = min(bestSquared, squared)
                else:
                    for childIndex in (node.left, node.right):
                        child = self.bvh.nodes[childIndex]
                        delta = np.maximum(np.maximum(child.lower - point, point - child.upper), 0)
                        squared = float(delta @ delta)
                        if squared <= bestSquared:
                            heapq.heappush(queue, (squared, childIndex))
            result[index] = np.sqrt(bestSquared)
        return result
