"""STEP/shape adapters around the mesh-only spherical descriptor algorithm."""

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from src.descriptors.sphereDescriptor import generateSphereDescriptor
from src.errors import DependencyError, DescriptorError
from src.indexing.partIndexer import fileFingerprint, hashFile
from src.models.sphereDescriptor import DescriptorConfig, SphereDescriptor


def generateShapeDescriptor(shape, config: DescriptorConfig | None = None) -> SphereDescriptor:
    config = config or DescriptorConfig()
    try:
        import OCC
        from src.cad.meshExtractor import extractMesh
    except ImportError as error:
        raise DependencyError("Activate the pythonOCC environment from environment.yml: " + str(error)) from error
    startedAt = perf_counter()
    mesh = extractMesh(shape, config)
    meshedAt = perf_counter()
    descriptor = generateSphereDescriptor(mesh, config)
    return replace(
        descriptor,
        timings={
            **descriptor.timings, "triangulationSeconds": meshedAt - startedAt,
            "shapeDescriptorTotalSeconds": perf_counter() - startedAt,
        },
        diagnostics={
            **descriptor.diagnostics, "pythonOccVersion": getattr(OCC, "VERSION", "unknown"),
            "numpyVersion": np.__version__,
        },
    )


def buildDescriptorFromStep(
    filePath: str | Path,
    config: DescriptorConfig | None = None,
    *,
    expectedSourceHash: str | None = None,
) -> SphereDescriptor:
    """Measure one new source. Historical lookup never calls this function."""
    config = config or DescriptorConfig()
    sourcePath = Path(filePath).expanduser().resolve()
    startedAt = perf_counter()
    before = fileFingerprint(sourcePath)
    sourceHash = hashFile(sourcePath)
    hashedAt = perf_counter()
    if expectedSourceHash is not None and sourceHash != expectedSourceHash:
        raise DescriptorError("STEP bytes differ from the Stage 1 index; run index on this source first")
    try:
        from src.cad.stepLoader import loadStep
    except ImportError as error:
        raise DependencyError("Activate the pythonOCC environment from environment.yml: " + str(error)) from error
    shape = loadStep(sourcePath)
    loadedAt = perf_counter()
    descriptor = generateShapeDescriptor(shape, config)
    if fileFingerprint(sourcePath) != before:
        raise DescriptorError("Source changed during descriptor generation; retry with a stable source")
    return replace(
        descriptor, sourceHash=sourceHash,
        timings={
            **descriptor.timings, "sourceHashSeconds": hashedAt - startedAt,
            "stepLoadSeconds": loadedAt - hashedAt, "totalSeconds": perf_counter() - startedAt,
        },
    )
