"""Use existing STEP validation/meshing and preserve the original B-rep."""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from src.detailedSimilarity.surfaceSampling import sampleSurface
from src.errors import DependencyError, DetailedSimilarityError
from src.geometry.normalizer import normalizeMesh
from src.indexing.partIndexer import fileFingerprint, hashFile
from src.models.detailedGeometry import DetailedGeometry, DetailedGeometryConfig


def buildDetailedGeometryFromShape(shape, config=None):
    config = config or DetailedGeometryConfig()
    try:
        from src.cad.meshExtractor import extractMesh
        from src.detailedSimilarity.faceExtractor import extractFaceFeatures
    except ImportError as error:
        raise DependencyError("Activate environment.yml for Stage 3 CAD preprocessing: " + str(error)) from error
    startedAt = perf_counter()
    mesh = extractMesh(shape, config.meshConfig)
    meshedAt = perf_counter()
    normalized, normalizationData = normalizeMesh(mesh)
    normalizedAt = perf_counter()
    points = sampleSurface(normalized, config.sampleCount)
    sampledAt = perf_counter()
    try:
        faces, anchors, diagnostics = extractFaceFeatures(shape, normalizationData, config.samplesPerFace)
    except Exception as error:
        # Native wrapper RuntimeErrors must become per-part failures in a bulk
        # build, just like errors in the existing STEP/mesh adapters.
        raise DetailedSimilarityError(f"Cannot extract detailed B-rep faces: {error}") from error
    import OCC
    diagnostics.update({"pythonOccVersion": getattr(OCC, "VERSION", "unknown"), "numpyVersion": np.__version__})
    return DetailedGeometry(normalized, np.concatenate((points, anchors)), faces, normalizationData,
                            config, timings={"triangulationSeconds": meshedAt - startedAt,
                                             "normalizationSeconds": normalizedAt - meshedAt,
                                             "surfaceSamplingSeconds": sampledAt - normalizedAt,
                                             "faceExtractionSeconds": perf_counter() - sampledAt,
                                             "geometryTotalSeconds": perf_counter() - startedAt},
                            diagnostics=diagnostics)


def buildDetailedGeometryFromStep(filePath, config=None, *, expectedSourceHash=None):
    sourcePath = Path(filePath).expanduser().resolve()
    startedAt = perf_counter()
    before = fileFingerprint(sourcePath)
    sourceHash = hashFile(sourcePath)
    if expectedSourceHash is not None and expectedSourceHash != sourceHash:
        raise DetailedSimilarityError("STEP differs from Stage 1; run index on the changed source first")
    try:
        from src.cad.stepLoader import loadStep
    except ImportError as error:
        raise DependencyError("Activate environment.yml for STEP loading: " + str(error)) from error
    loadAt = perf_counter()
    shape = loadStep(sourcePath)
    loadedAt = perf_counter()
    geometry = buildDetailedGeometryFromShape(shape, config)
    if fileFingerprint(sourcePath) != before:
        raise DetailedSimilarityError("STEP changed during detailed preprocessing")
    return replace(geometry, sourceHash=sourceHash,
                   timings={**geometry.timings, "stepLoadSeconds": loadedAt - loadAt,
                            "totalSeconds": perf_counter() - startedAt})
