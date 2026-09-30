"""Proper rotations and trimmed, bidirectional point-to-point ICP."""

from itertools import permutations, product

import numpy as np
from scipy.spatial import cKDTree


def properRotations():
    rotations = []
    for permutation in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            rotation = np.eye(3)[:, permutation] * signs
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
    return rotations


def rigidFit(source, target):
    sourceCenter, targetCenter = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - sourceCenter).T @ (target - targetCenter))
    correction = np.eye(3)
    correction[-1, -1] = 1 if np.linalg.det(u @ vt) > 0 else -1
    rotation = u @ correction @ vt
    return rotation, targetCenter - sourceCenter @ rotation


def sampleError(source, targetTree, target):
    first = targetTree.query(source, workers=1)[0]
    second = cKDTree(source).query(target, workers=1)[0]
    return float((first.mean() + second.mean()) / 2)


def alignSamples(queryPoints, candidatePoints, config):
    """Return the best sampled-surface alignment; scale is fixed by the caller.

    Search 24 proper signed axis permutations, then refine the best starts.
    This is a local optimizer. Symmetry/poor initialization can still produce
    an incorrect alignment. Reflections and scale fitting are excluded.
    """
    targetTree = cKDTree(candidatePoints)
    starts = [(sampleError(queryPoints @ rotation, targetTree, candidatePoints), index, rotation)
              for index, rotation in enumerate(properRotations())]
    starts.sort(key=lambda value: (value[0], value[1]))
    best = None
    for _, startIndex, initialRotation in starts[:config.alignmentStarts]:
        rotation, translation = initialRotation.copy(), np.zeros(3)
        previousError = np.inf
        converged = False
        iterations = 0
        for iteration in range(config.icpIterations):
            transformed = queryPoints @ rotation + translation
            forwardDistances, forwardIds = targetTree.query(transformed, workers=1)
            backwardDistances, backwardIds = cKDTree(transformed).query(candidatePoints, workers=1)
            source = np.concatenate((transformed, transformed[backwardIds]))
            target = np.concatenate((candidatePoints[forwardIds], candidatePoints))
            distances = np.concatenate((forwardDistances, backwardDistances))
            count = max(3, int(len(distances) * config.trimFraction))
            kept = np.argsort(distances, kind="stable")[:count]
            deltaRotation, deltaTranslation = rigidFit(source[kept], target[kept])
            rotation = rotation @ deltaRotation
            translation = translation @ deltaRotation + deltaTranslation
            iterations = iteration + 1
            error = float(distances[kept].mean())
            if abs(previousError - error) < 1e-8:
                converged = True
                break
            previousError = error
        error = sampleError(queryPoints @ rotation + translation, targetTree, candidatePoints)
        if best is None or error < best[0]:
            best = (error, rotation, translation, {"startIndex": startIndex,
                                                  "iterations": iterations, "converged": converged,
                                                  "sampleAlignmentDistance": error})
    return best[1], best[2], best[3]
