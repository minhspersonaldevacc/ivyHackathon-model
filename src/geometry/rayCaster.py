"""Möller–Trumbore intersections and solid intervals, using a triangle BVH."""

from dataclasses import dataclass

import numpy as np

from src.geometry.bvh import TriangleBvh


@dataclass(frozen=True)
class RayIntersections:
    distances: np.ndarray
    thickness: float
    originInside: bool
    trianglesTested: int


def castRay(
    bvh: TriangleBvh, direction: np.ndarray, origin: np.ndarray | None = None,
    *, tolerance: float = 1e-7,
) -> RayIntersections:
    """Return sorted, deduplicated surface crossings and material length.

    Shared triangle edges count once per body. Equal-distance contacts with
    opposite normals on the same body are tangencies and do not toggle occupancy.
    Outward winding lets us infer occupancy at t=0 from the crossings before
    infinity, which is outside. This handles both material and cavity origins.
    """
    direction = np.asarray(direction, dtype=np.float64)
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Ray tolerance must be finite and positive")
    if direction.shape != (3,) or origin.shape != (3,) or not np.isfinite(direction).all() or not np.isfinite(origin).all():
        raise ValueError("Ray origin and direction must be finite 3-vectors")
    if not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Ray direction must be a unit vector")
    triangleIds = bvh.candidateTriangles(direction, origin, tolerance)
    emptyResult = RayIntersections(np.empty(0), 0.0, False, len(triangleIds))
    if len(triangleIds) == 0:
        return emptyResult

    edge1, edge2 = bvh.edge1[triangleIds], bvh.edge2[triangleIds]
    pVector = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, pVector)
    valid = np.abs(determinant) > 1e-12 * bvh.normalLengths[triangleIds]
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=valid)
    tVector = origin - bvh.points[triangleIds, 0]
    u = np.einsum("ij,ij->i", tVector, pVector) * inverse
    qVector = np.cross(tVector, edge1)
    v = (qVector @ direction) * inverse
    distances = np.einsum("ij,ij->i", edge2, qVector) * inverse
    valid &= (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (distances >= -tolerance)
    if not np.any(valid):
        return emptyResult
    distances = np.maximum(distances[valid], 0)
    signs = -np.sign(determinant[valid]).astype(np.int64)  # +1 exit, -1 entry
    bodyIds = bvh.mesh.triangleBodyIds[triangleIds[valid]]
    order = np.argsort(distances, kind="stable")
    distances, signs, bodyIds = distances[order], signs[order], bodyIds[order]

    events = []
    start = 0
    while start < len(distances):
        end = start + 1
        while end < len(distances) and distances[end] - distances[start] <= tolerance:
            end += 1
        crossing = {}
        for bodyId in np.unique(bodyIds[start:end]):
            bodySigns = signs[start:end][bodyIds[start:end] == bodyId]
            if np.all(bodySigns == bodySigns[0]):
                crossing[int(bodyId)] = int(bodySigns[0])
        if crossing:
            events.append((float(np.mean(distances[start:end])), crossing))
        start = end
    if not events:
        return emptyResult

    occupancy = {}
    for _, crossing in events:
        for bodyId, sign in crossing.items():
            occupancy[bodyId] = occupancy.get(bodyId, 0) + sign
    if any(value not in (0, 1) for value in occupancy.values()):
        raise ValueError("Inconsistent mesh ray crossings; check winding or reduce rayTolerance")
    originInside = any(occupancy.values())
    previous, thickness = 0.0, 0.0
    for distance, crossing in events:
        if any(occupancy.values()):
            thickness += distance - previous
        for bodyId, sign in crossing.items():
            occupancy[bodyId] -= sign
            if occupancy[bodyId] not in (0, 1):
                raise ValueError("Mesh crossings do not alternate within a body")
        previous = distance
    return RayIntersections(
        np.asarray([event[0] for event in events]), thickness, originInside, len(triangleIds)
    )
