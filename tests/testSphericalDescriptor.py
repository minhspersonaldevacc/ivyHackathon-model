"""Analytic mesh cases for PCA, intersections, channels, and deterministic metrics."""

from dataclasses import replace
from math import sqrt
import unittest

import numpy as np

from src.descriptors.descriptorSimilarity import compareDescriptors, cosineSimilarity
from src.descriptors.sphereDescriptor import generateSphereDescriptor
from src.geometry.bvh import TriangleBvh
from src.geometry.normalizer import normalizeMesh
from src.geometry.rayCaster import castRay
from src.geometry.sphericalSampler import axisFlipVariants, sphericalDirections
from src.models.mesh import Mesh, validateClosedMesh
from src.models.sphereDescriptor import DescriptorConfig, SphereDescriptor


def boxMesh(dimensions=(2, 2, 2), center=(0, 0, 0)) -> Mesh:
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float) * np.asarray(dimensions) / 2 + np.asarray(center)
    triangles = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    return Mesh(vertices, triangles)


def combineMeshes(meshes, *, separateBodies=True):
    vertices, triangles, bodyIds = [], [], []
    offset = 0
    for index, mesh in enumerate(meshes):
        vertices.append(mesh.vertices)
        triangles.append(mesh.triangles + offset)
        bodyIds.append(np.full(len(mesh.triangles), index if separateBodies else 0))
        offset += len(mesh.vertices)
    return Mesh(np.concatenate(vertices), np.concatenate(triangles), np.concatenate(bodyIds))


class SphericalGeometryTests(unittest.TestCase):
    def testEqualAreaDirectionsAndExactProperFlips(self):
        directions = sphericalDirections(8, 16)
        np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1, atol=1e-15)
        np.testing.assert_allclose(np.diff(directions[:, 0, 2]), -0.25)
        np.testing.assert_allclose(directions.mean(axis=(0, 1)), 0, atol=1e-15)
        for variant, signs in zip(axisFlipVariants(directions), ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))):
            np.testing.assert_allclose(variant, directions * signs, atol=1e-15)

    def testInsideOriginAndSharedTriangleEdgeCountOnce(self):
        hits = castRay(TriangleBvh(boxMesh(), 2), np.array([1.0, 0, 0]))
        np.testing.assert_allclose(hits.distances, [1])
        self.assertEqual(hits.thickness, 1)
        self.assertTrue(hits.originInside)

    def testOutsideOriginHasEntryExitThickness(self):
        hits = castRay(TriangleBvh(boxMesh()), np.array([1.0, 0, 0]), np.array([-2.0, 0, 0]))
        np.testing.assert_allclose(hits.distances, [1, 3])
        self.assertEqual(hits.thickness, 2)
        self.assertFalse(hits.originInside)

    def testHollowSolidOriginAndThickness(self):
        inner = boxMesh()
        cavity = Mesh(inner.vertices, inner.triangles[:, ::-1])
        hollow = combineMeshes([boxMesh((4, 4, 4)), cavity], separateBodies=False)
        validateClosedMesh(hollow)
        hits = castRay(TriangleBvh(hollow), np.array([1.0, 0, 0]))
        np.testing.assert_allclose(hits.distances, [1, 2])
        self.assertEqual(hits.thickness, 1)
        self.assertFalse(hits.originInside)

    def testDisjointAndOverlappingBodies(self):
        disjoint = combineMeshes([boxMesh(), boxMesh(center=(4, 0, 0))])
        hits = castRay(TriangleBvh(disjoint), np.array([1.0, 0, 0]))
        np.testing.assert_allclose(hits.distances, [1, 3, 5])
        self.assertEqual(hits.thickness, 3)
        overlap = combineMeshes([boxMesh(), boxMesh(center=(0.5, 0, 0))])
        hits = castRay(TriangleBvh(overlap), np.array([1.0, 0, 0]))
        self.assertAlmostEqual(hits.thickness, 1.5)

    def testMissingTangentVertexAndBoundaryRays(self):
        bvh = TriangleBvh(boxMesh(), 2)
        missing = castRay(bvh, np.array([1.0, 0, 0]), np.array([2.0, 2, 2]))
        self.assertEqual(len(missing.distances), 0)
        self.assertEqual(missing.thickness, 0)
        tangent = castRay(bvh, np.array([1.0, 1, 0]) / sqrt(2), np.array([-2.0, 0, 0]))
        self.assertEqual(len(tangent.distances), 0)
        vertex = castRay(bvh, np.ones(3) / sqrt(3))
        np.testing.assert_allclose(vertex.distances, [sqrt(3)])
        boundary = castRay(bvh, np.array([-1.0, 0, 0]), np.array([1.0, 0, 0]))
        np.testing.assert_allclose(boundary.distances, [0, 2])
        self.assertEqual(boundary.thickness, 2)

    def testNormalizationInverseAndNoInputMutation(self):
        mesh = boxMesh((2, 4, 6), (10, 20, 30))
        original = mesh.vertices.copy()
        normalized, data = normalizeMesh(mesh)
        np.testing.assert_allclose(data["centroid"], (10, 20, 30))
        self.assertAlmostEqual(data["radiusScale"], sqrt(14))
        self.assertAlmostEqual(np.max(np.linalg.norm(normalized.vertices, axis=1)), 1)
        np.testing.assert_allclose(
            normalized.vertices * data["radiusScale"] @ np.asarray(data["pcaAxes"]).T + data["centroid"], original
        )
        np.testing.assert_array_equal(mesh.vertices, original)
        self.assertAlmostEqual(np.linalg.det(data["pcaAxes"]), 1)

    def testTranslationScaleRotationInvarianceForDistinctPcaAxes(self):
        mesh = boxMesh((2, 3, 5))
        rotation, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1
        transformed = Mesh(mesh.vertices @ rotation * 7 + (100, -200, 30), mesh.triangles)
        config = DescriptorConfig(thetaResolution=8, phiResolution=16)
        first = generateSphereDescriptor(mesh, config)
        second = generateSphereDescriptor(transformed, config)
        self.assertLess(compareDescriptors(first, second), 1e-6)
        self.assertEqual(first.normalizationData["ambiguousAxisPairs"], [])

    def testDefaultDimensionsAndIndependentChannelScaling(self):
        descriptor = generateSphereDescriptor(boxMesh((2, 3, 5)))
        self.assertEqual(descriptor.tensor.shape, (32, 64, 4))
        self.assertEqual(descriptor.toVector().shape, (8192,))
        np.testing.assert_allclose(descriptor.tensor[..., 2], 1 / 16)
        np.testing.assert_allclose(descriptor.tensor[..., 0], descriptor.tensor[..., 3])
        self.assertTrue(np.all((descriptor.tensor >= 0) & (descriptor.tensor <= 1)))

    def testCountClippingMissingRaysAndDegeneratePcaReport(self):
        mesh = combineMeshes([boxMesh(center=(-3, 0, 0)), boxMesh(center=(3, 0, 0))])
        descriptor = generateSphereDescriptor(mesh, DescriptorConfig(thetaResolution=8, phiResolution=16, countScale=1))
        misses = descriptor.tensor[..., 2] == 0
        self.assertTrue(misses.any())
        self.assertTrue(np.all(descriptor.tensor[misses] == 0))
        self.assertGreater(descriptor.diagnostics["countClippedRayCount"], 0)
        self.assertEqual(descriptor.tensor[..., 2].max(), 1)
        _, cubeData = normalizeMesh(boxMesh())
        self.assertEqual(cubeData["ambiguousAxisPairs"], [[0, 1], [1, 2]])

    def testInvalidMeshAndSettings(self):
        mesh = boxMesh()
        with self.assertRaises(ValueError):
            normalizeMesh(Mesh(mesh.vertices, mesh.triangles[:-1]))
        with self.assertRaises(ValueError):
            normalizeMesh(Mesh(mesh.vertices, mesh.triangles[:, ::-1]))
        for values in ({"phiResolution": 7}, {"thetaResolution": 0}, {"countScale": 0}, {"rayTolerance": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DescriptorConfig(**values)

    def testBvhCullsUnrelatedTriangles(self):
        mesh = combineMeshes([boxMesh(center=(index * 4, 0, 0)) for index in range(100)])
        bvh = TriangleBvh(mesh, 8)
        hits = castRay(bvh, np.array([0.0, 1, 0]))
        self.assertLess(hits.trianglesTested, len(mesh.triangles) / 10)
        np.testing.assert_allclose(hits.distances, [1])


class DescriptorMetricTests(unittest.TestCase):
    def setUp(self):
        self.config = DescriptorConfig(thetaResolution=4, phiResolution=8)
        self.tensor = np.random.default_rng(7).uniform(0.1, 0.9, self.config.tensorShape).astype(np.float32)
        self.descriptor = SphereDescriptor(self.tensor, self.config)

    def testSelfDistanceAndAllProperFlips(self):
        for tensor in axisFlipVariants(self.tensor):
            other = SphereDescriptor(tensor, self.config)
            self.assertAlmostEqual(compareDescriptors(self.descriptor, other), 0)
            self.assertAlmostEqual(cosineSimilarity(self.descriptor, other), 1)
        reflected = SphereDescriptor(self.tensor[:, ::-1, :], self.config)
        self.assertGreater(compareDescriptors(self.descriptor, reflected), 0.1)

    def testWeightsAndKnownErrorMagnitude(self):
        zeros = SphereDescriptor(np.zeros(self.config.tensorShape), self.config)
        ones = SphereDescriptor(np.ones(self.config.tensorShape), self.config)
        self.assertEqual(compareDescriptors(zeros, ones), 1)
        countOnly = np.zeros(self.config.tensorShape)
        countOnly[..., 2] = 1
        countDescriptor = SphereDescriptor(countOnly, self.config)
        self.assertEqual(compareDescriptors(zeros, countDescriptor, weights=(1, 1, 0, 1)), 0)
        self.assertEqual(compareDescriptors(zeros, countDescriptor, weights=(0, 0, 1, 0)), 1)
        self.assertEqual(cosineSimilarity(zeros, zeros), 1)
        self.assertEqual(cosineSimilarity(zeros, ones), 0)

    def testIncompatibleDescriptorsAndBadWeightsFail(self):
        changed = replace(self.descriptor, config=replace(self.config, countScale=8))
        with self.assertRaises(ValueError):
            compareDescriptors(self.descriptor, changed)
        for weights in ((0, 0, 0, 0), (-1, 1, 1, 1), (1, 2), (float("nan"), 1, 1, 1)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                compareDescriptors(self.descriptor, self.descriptor, weights=weights)


if __name__ == "__main__":
    unittest.main()
