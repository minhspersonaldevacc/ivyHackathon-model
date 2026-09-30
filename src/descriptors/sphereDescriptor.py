"""Create four-channel, equal-area spherical fingerprints from triangle meshes."""

from time import perf_counter

import numpy as np

from src.geometry.bvh import TriangleBvh
from src.geometry.normalizer import normalizeMesh
from src.geometry.rayCaster import castRay
from src.geometry.sphericalSampler import sphericalDirections
from src.models.mesh import Mesh
from src.models.sphereDescriptor import DescriptorConfig, SphereDescriptor


def generateSphereDescriptor(mesh: Mesh, config: DescriptorConfig | None = None) -> SphereDescriptor:
    config = config or DescriptorConfig()
    startedAt = perf_counter()
    normalized, normalizationData = normalizeMesh(mesh, degeneracyTolerance=config.pcaDegeneracyTolerance)
    normalizedAt = perf_counter()
    bvh = TriangleBvh(normalized, config.leafSize)
    builtAt = perf_counter()
    directions = sphericalDirections(config.thetaResolution, config.phiResolution).reshape(-1, 3)
    tensor = np.zeros((len(directions), 4), dtype=np.float32)
    maxRawCount = clippedRayCount = insideRayCount = trianglesTested = 0
    for rayIndex, direction in enumerate(directions):
        hits = castRay(bvh, direction, tolerance=config.rayTolerance)
        count = len(hits.distances)
        trianglesTested += hits.trianglesTested
        if count == 0:
            continue  # Explicit no-hit convention: all four channels are zero.
        maxRawCount = max(maxRawCount, count)
        clippedRayCount += count > config.countScale
        insideRayCount += hits.originInside
        if hits.distances[-1] > 1 + 10 * config.rayTolerance or hits.thickness > 1 + 10 * config.rayTolerance:
            raise ValueError("A ray extends beyond the normalized unit sphere")
        tensor[rayIndex] = (
            np.clip(hits.distances[0], 0, 1), np.clip(hits.distances[-1], 0, 1),
            min(count, config.countScale) / config.countScale, np.clip(hits.thickness, 0, 1),
        )
    finishedAt = perf_counter()
    return SphereDescriptor(
        tensor.reshape(config.tensorShape), config, normalizationData,
        len(mesh.vertices), len(mesh.triangles),
        timings={
            "normalizationSeconds": normalizedAt - startedAt,
            "bvhBuildSeconds": builtAt - normalizedAt,
            "descriptorGenerationSeconds": finishedAt - builtAt,
            "meshDescriptorTotalSeconds": finishedAt - startedAt,
        },
        diagnostics={
            "maxRawIntersectionCount": maxRawCount, "countClippedRayCount": clippedRayCount,
            "originInsideRayCount": insideRayCount, "rayCount": len(directions),
            "trianglesTested": trianglesTested, "naiveTriangleTests": len(directions) * len(mesh.triangles),
            "bvhNodeCount": len(bvh.nodes),
        },
    )
