"""Explainable face matching with explicit penalties for unmatched faces."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def relativeDifference(first, second):
    return abs(first - second) / max(abs(first), abs(second), 1e-12)


def compareFaces(queryFaces, candidateFaces, rotation, translation, candidateScale=1.0):
    """Match surface type, center, area, direction, radius, angle and degree.

    Costs are bounded in [0, 1]. A type mismatch/unmatched face costs 1.
    Adjacency DEGREE is used here; this is not full graph isomorphism. Exported
    face splitting can change the cost even for geometrically identical solids.
    """
    count = max(len(queryFaces), len(candidateFaces))
    costs = np.ones((count, count))
    for queryIndex, queryFace in enumerate(queryFaces):
        center = np.asarray(queryFace.center) @ rotation + translation
        direction = np.asarray(queryFace.direction) @ rotation
        for candidateIndex, candidateFace in enumerate(candidateFaces):
            if queryFace.surfaceType != candidateFace.surfaceType:
                continue
            centerError = min(float(np.linalg.norm(center - np.asarray(candidateFace.center) * candidateScale)) / 2, 1)
            areaError = relativeDifference(queryFace.area, candidateFace.area * candidateScale ** 2)
            directionError = 0.0
            if np.linalg.norm(direction) > 0 and np.linalg.norm(candidateFace.direction) > 0:
                dot = float(np.clip(direction @ np.asarray(candidateFace.direction), -1, 1))
                directionError = (1 - dot) / 2 if queryFace.surfaceType == "plane" else 1 - abs(dot)
            radiusError = relativeDifference(queryFace.radius, candidateFace.radius * candidateScale)
            degreeError = relativeDifference(len(queryFace.adjacentFaces), len(candidateFace.adjacentFaces))
            angleError = min(abs(queryFace.angle - candidateFace.angle) / np.pi, 1)
            costs[queryIndex, candidateIndex] = (0.35 * centerError + 0.2 * areaError +
                                                0.15 * directionError + 0.15 * radiusError +
                                                0.1 * degreeError + 0.05 * angleError)
    queryIds, candidateIds = linear_sum_assignment(costs)
    assigned = costs[queryIds, candidateIds]
    return {"faceDistance": float(assigned.mean()),
            "unmatchedFaceCount": int(np.count_nonzero(assigned >= 1)),
            "queryFaceCount": len(queryFaces), "candidateFaceCount": len(candidateFaces)}
