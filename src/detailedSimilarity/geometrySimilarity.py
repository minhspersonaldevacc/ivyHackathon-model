"""Detailed geometric measurements independent of STEP loading and SQLite."""

from time import perf_counter

import numpy as np

from src.detailedSimilarity.alignment import alignSamples
from src.detailedSimilarity.faceSimilarity import compareFaces
from src.detailedSimilarity.surfaceDistance import SurfaceDistanceIndex
from src.models.detailedGeometry import DetailedSearchConfig
from src.models.mesh import Mesh


def compareDetailedGeometry(query, candidate, config=None, *, queryIndex=None):
    config = config or DetailedSearchConfig()
    if query.config.configKey != candidate.config.configKey:
        raise ValueError("Detailed geometries have incompatible preprocessing settings")
    startedAt = perf_counter()
    candidateScale = (candidate.normalizationData["radiusScale"] / query.normalizationData["radiusScale"]
                      if config.sizeMode == "physical" else 1.0)
    candidatePoints = candidate.samplePoints * candidateScale
    rotation, translation, alignmentData = alignSamples(query.samplePoints, candidatePoints, config)
    alignedAt = perf_counter()
    candidateMesh = Mesh(candidate.mesh.vertices * candidateScale, candidate.mesh.triangles,
                         candidate.mesh.triangleBodyIds)
    candidateIndex = SurfaceDistanceIndex(candidateMesh)
    queryIndex = queryIndex or SurfaceDistanceIndex(query.mesh)
    builtAt = perf_counter()
    forward = candidateIndex.distances(query.samplePoints @ rotation + translation)
    backward = queryIndex.distances((candidatePoints - translation) @ rotation.T)
    measuredAt = perf_counter()
    meanDistance = float((forward.mean() + backward.mean()) / 2)
    percentileDistance = float((np.percentile(forward, 95) + np.percentile(backward, 95)) / 2)
    faceData = compareFaces(query.faces, candidate.faces, rotation, translation, candidateScale)
    weightSum = config.meanWeight + config.percentileWeight + config.faceWeight
    distance = (config.meanWeight * meanDistance + config.percentileWeight * percentileDistance +
                config.faceWeight * faceData["faceDistance"]) / weightSum
    return {"detailedDistance": float(distance), "surfaceMeanDistance": meanDistance,
            "surfaceP95Distance": percentileDistance, "surfaceMaxDistance": float(max(forward.max(), backward.max())),
            "queryToCandidateMean": float(forward.mean()), "candidateToQueryMean": float(backward.mean()),
            "surfaceMeanMm": meanDistance * query.normalizationData["radiusScale"] if config.sizeMode == "physical" else None,
            "surfaceP95Mm": percentileDistance * query.normalizationData["radiusScale"] if config.sizeMode == "physical" else None,
            **faceData, "alignment": {**alignmentData, "rotation": rotation.tolist(),
                                        "translation": translation.tolist(), "candidateScale": candidateScale,
                                        "convention": "candidateCanonical = queryCanonical @ rotation + translation"},
            "timings": {"alignmentSeconds": alignedAt - startedAt,
                         "distanceIndexSeconds": builtAt - alignedAt,
                         "surfaceComparisonSeconds": measuredAt - builtAt,
                         "faceComparisonSeconds": perf_counter() - measuredAt,
                         "comparisonTotalSeconds": perf_counter() - startedAt}}
