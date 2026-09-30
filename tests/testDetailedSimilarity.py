"""Stage 3 metrics, cache lifecycle, and real STEP end-to-end ranking."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.detailedSimilarity.geometryBuilder import buildDetailedGeometries
from src.detailedSimilarity.geometryPipeline import buildDetailedGeometryFromShape, buildDetailedGeometryFromStep
from src.detailedSimilarity.geometryStore import GeometryStore
from src.detailedSimilarity.surfaceDistance import SurfaceDistanceIndex, closestOnTriangles
from src.detailedSimilarity.surfaceSampling import sampleSurface
from src.errors import CadError, DetailedSimilarityError
from src.geometry.normalizer import normalizeMesh
from src.indexing.candidateSearch import GeometricMatch
from src.indexing.database import PartDatabase
from src.models.detailedGeometry import DetailedGeometry, DetailedGeometryConfig, DetailedSearchConfig, FaceFeature
from tests.testDatabase import makeMetadata
from tests.testSphericalDescriptor import boxMesh


hasScipy = importlib.util.find_spec("scipy") is not None
hasOcc = importlib.util.find_spec("OCC") is not None
if hasScipy:
    from src.detailedSimilarity.alignment import properRotations, rigidFit
    from src.detailedSimilarity.detailedSearch import (findDetailedMatchesForStep, rankDetailedCandidates,
                                                      rerankDetailedForStep)
    from src.detailedSimilarity.faceSimilarity import compareFaces
    from src.detailedSimilarity.geometrySimilarity import compareDetailedGeometry
if hasOcc:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
    from src.descriptors.descriptorBuilder import buildDescriptors
    from src.descriptors.descriptorStore import DescriptorStore
    from src.indexing.partIndexer import indexPath
    from src.models.sphereDescriptor import DescriptorConfig
    from tests.testGeometry import writeStep


def cachedBox(config):
    mesh, normalization = normalizeMesh(boxMesh((2, 3, 5)))
    return DetailedGeometry(mesh, sampleSurface(mesh, config.sampleCount),
                            (FaceFeature("plane", 1.0, (0, 0, 0), (1, 0, 0)),), normalization, config)


class SurfaceDistanceTests(unittest.TestCase):
    def testClosestTriangleInteriorEdgesAndVertices(self):
        triangle = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
        for point, expected in (([0.2, 0.3, 2], [0.2, 0.3, 0]),
                                ([0.7, 0.7, 0], [0.5, 0.5, 0]),
                                ([-1, -2, 0], [0, 0, 0])):
            actual, squared = closestOnTriangles(np.asarray(point), triangle)
            np.testing.assert_allclose(actual, expected)
            self.assertAlmostEqual(squared, np.sum((np.asarray(point) - expected) ** 2))

    def testBvhDistancesMatchAnalyticBoxAndBruteForce(self):
        mesh = boxMesh()
        points = np.random.default_rng(18).uniform(-3, 3, size=(100, 3))
        distances = SurfaceDistanceIndex(mesh).distances(points)
        delta = np.maximum(np.abs(points) - 1, 0)
        expected = np.linalg.norm(delta, axis=1)
        inside = np.max(np.abs(points), axis=1) <= 1
        expected[inside] = 1 - np.max(np.abs(points[inside]), axis=1)
        np.testing.assert_allclose(distances, expected, atol=1e-12)
        bruteForce = [np.sqrt(closestOnTriangles(p, mesh.vertices[mesh.triangles])[1]) for p in points]
        np.testing.assert_allclose(distances, bruteForce)

    def testSamplingIsRepeatableAndOnSurface(self):
        mesh = boxMesh()
        points = sampleSurface(mesh, 256)
        np.testing.assert_array_equal(points, sampleSurface(mesh, 256))
        np.testing.assert_allclose(SurfaceDistanceIndex(mesh).distances(points), 0, atol=1e-12)

    def testConfigurationValidation(self):
        for kwargs in ({"sampleCount": 0}, {"samplesPerFace": 0}, {"geometryVersion": 99}, {"linearDeflectionRatio": -1}):
            with self.assertRaises(ValueError):
                DetailedGeometryConfig(**kwargs)
        for kwargs in ({"sizeMode": "unknown"}, {"trimFraction": 0}, {"alignmentStarts": 25},
                       {"icpIterations": -1}, {"meanWeight": float("nan")}):
            with self.assertRaises(ValueError):
                DetailedSearchConfig(**kwargs)


@unittest.skipUnless(hasScipy, "Install SciPy from environment.yml for Stage 3 comparisons")
class DetailedMetricTests(unittest.TestCase):
    def testProperRotationsAndRigidFitDoNotAllowMirrors(self):
        rotations = properRotations()
        self.assertEqual(len(rotations), 24)
        self.assertEqual(len({tuple(r.reshape(-1)) for r in rotations}), 24)
        for rotation in rotations:
            self.assertAlmostEqual(np.linalg.det(rotation), 1)
        source = np.random.default_rng(4).normal(size=(100, 3))
        target = source @ rotations[9] + [2, 3, 4]
        rotation, translation = rigidFit(source, target)
        np.testing.assert_allclose(source @ rotation + translation, target, atol=1e-12)

    def testIdentityAndPhysicalVersusShapeScale(self):
        geometry = cachedBox(DetailedGeometryConfig(sampleCount=64))
        same = compareDetailedGeometry(geometry, geometry)
        self.assertLess(same["detailedDistance"], 1e-8)
        bigger = replace(geometry, normalizationData={**geometry.normalizationData,
                                                     "radiusScale": geometry.normalizationData["radiusScale"] * 2})
        physical = compareDetailedGeometry(geometry, bigger)
        shape = compareDetailedGeometry(geometry, bigger, DetailedSearchConfig(sizeMode="shape"))
        self.assertGreater(physical["detailedDistance"], shape["detailedDistance"])
        self.assertLess(shape["detailedDistance"], 1e-8)
        self.assertIsNone(shape["surfaceMeanMm"])
        with self.assertRaises(ValueError):
            compareDetailedGeometry(geometry, replace(geometry, config=replace(geometry.config, samplesPerFace=4)))

    def testFaceMatchingPenalizesAdditionalOrDifferentFaces(self):
        first = FaceFeature("cylinder", 1, (0, 0, 0), (0, 0, 1), radius=0.1)
        same = compareFaces((first,), (first,), np.eye(3), np.zeros(3))
        extra = compareFaces((first,), (first, first), np.eye(3), np.zeros(3))
        different = compareFaces((first,), (replace(first, surfaceType="plane"),), np.eye(3), np.zeros(3))
        self.assertLess(same["faceDistance"], extra["faceDistance"])
        self.assertLess(same["faceDistance"], different["faceDistance"])
        self.assertEqual(extra["unmatchedFaceCount"], 1)


class GeometryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.rootPath = Path(self.temporary.name)
        self.database = PartDatabase(self.rootPath / "parts.sqlite3")
        self.addCleanup(self.database.close)
        self.part = makeMetadata(str(self.rootPath / "absent.step"))
        self.fingerprint = {"fileSize": 1, "modifiedTimeNs": 1, "changedTimeNs": 1}
        self.database.storePart(self.part, self.fingerprint)
        self.config = DetailedGeometryConfig(sampleCount=64)
        self.geometry = replace(cachedBox(self.config), sourceHash=self.part.sourceHash)
        self.store = GeometryStore(self.database)

    def testAdditiveSchemaRoundTripAndReadonlyNoCadReads(self):
        original = self.database.getPart(self.part.filePath)
        self.store.storeGeometry(self.part, self.geometry)
        with PartDatabase(self.database.databasePath, readOnly=True) as database:
            actual, path = GeometryStore(database, initialize=False).loadGeometries([self.part.partId], self.config)[self.part.partId]
        np.testing.assert_array_equal(actual.mesh.vertices, self.geometry.mesh.vertices)
        np.testing.assert_array_equal(actual.samplePoints, self.geometry.samplePoints)
        self.assertEqual(actual.faces, self.geometry.faces)
        self.assertEqual(path, self.part.filePath)
        self.assertEqual(original, self.database.getPart(self.part.filePath))
        self.assertEqual(self.database.connection.execute("PRAGMA user_version").fetchone()[0], 1)
        self.assertFalse(Path(self.part.filePath).exists())

    def testOnceOnlySourceConfigAndStatusInvalidation(self):
        with patch("src.detailedSimilarity.geometryBuilder.buildDetailedGeometryFromStep", return_value=self.geometry) as generator:
            self.assertEqual(buildDetailedGeometries(self.store, self.config).built, 1)
            self.assertEqual(buildDetailedGeometries(self.store, self.config).skipped, 1)
            self.assertEqual(generator.call_count, 1)
            self.assertEqual(buildDetailedGeometries(self.store, self.config, rebuild=True).built, 1)
        otherConfig = replace(self.config, samplesPerFace=4)
        self.assertEqual(self.store.loadGeometries([self.part.partId], otherConfig), {})
        self.store.storeGeometry(self.part, replace(self.geometry, config=otherConfig))
        self.assertEqual(self.database.connection.execute("SELECT count(*) FROM detailedGeometries").fetchone()[0], 2)
        self.database.beginFile(self.part.filePath, self.fingerprint)
        self.assertEqual(self.store.loadGeometries([self.part.partId], self.config), {})
        self.database.storePart(replace(self.part, sourceHash="b" * 64), self.fingerprint)
        self.assertEqual(self.store.loadGeometries([self.part.partId], self.config), {})
        with self.assertRaises(DetailedSimilarityError):
            self.store.storeGeometry(self.part, self.geometry)

    def testBulkContinuesAfterFailureRetriesAndResumes(self):
        second = replace(self.part, partId="zSecond", filePath=str(self.rootPath / "second.step"))
        self.database.storePart(second, self.fingerprint)
        results = []
        with patch("src.detailedSimilarity.geometryBuilder.buildDetailedGeometryFromStep", side_effect=[CadError("corrupt"), self.geometry]):
            summary = buildDetailedGeometries(self.store, self.config, onResult=results.append)
        self.assertEqual((summary.failed, summary.built), (1, 1))
        self.assertTrue(results[0]["error"])
        with patch("src.detailedSimilarity.geometryBuilder.buildDetailedGeometryFromStep", return_value=self.geometry) as generator:
            summary = buildDetailedGeometries(self.store, self.config)
            self.assertEqual((summary.skippedFailed, summary.skipped), (1, 1))
            generator.assert_not_called()
            self.assertEqual(buildDetailedGeometries(self.store, self.config, retryFailed=True).built, 1)
            self.store.beginPart(self.part, self.config)
            self.assertEqual(buildDetailedGeometries(self.store, self.config).built, 1)

    def testChecksumAndMetadataTamperingAreRejected(self):
        self.store.storeGeometry(self.part, self.geometry)
        with self.database.connection:
            self.database.connection.execute("UPDATE detailedGeometries SET geometryBlob = ?", (b"corrupt",))
        with self.assertRaisesRegex(DetailedSimilarityError, "Invalid detailed geometry"):
            self.store.loadGeometries([self.part.partId], self.config)
        self.store.storeGeometry(self.part, self.geometry)
        with self.database.connection:
            self.database.connection.execute("UPDATE detailedGeometries SET metadataJson = '{}' ")
        with self.assertRaises(DetailedSimilarityError):
            self.store.loadGeometries([self.part.partId], self.config)

    @unittest.skipUnless(hasScipy, "Install SciPy for reranking")
    def testShortlistOnlyMissingPolicyAndPreservedStage2Fields(self):
        self.store.storeGeometry(self.part, self.geometry)
        other = replace(self.part, partId="other", filePath=str(self.rootPath / "other.step"))
        self.database.storePart(other, self.fingerprint)
        bigger = replace(self.geometry, normalizationData={**self.geometry.normalizationData,
                                                          "radiusScale": self.geometry.normalizationData["radiusScale"] * 2})
        self.store.storeGeometry(other, bigger)
        shortlist = [GeometricMatch("other", 0.01, (-1, -1, 1), 0.99),
                     GeometricMatch(self.part.partId, 0.2, (1, 1, 1)),
                     GeometricMatch(self.part.partId, 0.3, (1, 1, 1))]
        with patch("src.detailedSimilarity.detailedSearch.buildDetailedGeometryFromStep", return_value=self.geometry) as generator:
            report = rerankDetailedForStep("query.step", shortlist, topK=1, databasePath=self.database.databasePath)
        generator.assert_called_once()
        self.assertEqual(report.comparedCandidates, 2)
        self.assertEqual(report.matches[0].partId, self.part.partId)
        self.assertEqual(report.matches[0].distance, 0.2)
        self.assertIsInstance(report.matches[0], GeometricMatch)
        missing = [GeometricMatch("missing", 0.1, (1, 1, 1))]
        with self.assertRaises(DetailedSimilarityError):
            rankDetailedCandidates(self.geometry, missing, self.store)
        self.assertEqual(rankDetailedCandidates(self.geometry, missing, self.store, skipMissing=True).unavailablePartIds, ["missing"])
        with patch("src.detailedSimilarity.detailedSearch.buildDetailedGeometryFromStep", side_effect=AssertionError("query reopened")):
            self.assertEqual(rerankDetailedForStep("absent.step", [], databasePath="absent.sqlite3").matches, [])


@unittest.skipUnless(hasOcc and hasScipy, "Install environment.yml for real Stage 3 integration")
class DetailedStepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.rootPath = Path(self.temporary.name)
        self.config = DetailedGeometryConfig(sampleCount=128)
        self.base = BRepPrimAPI_MakeBox(30, 20, 10).Shape()

    def testIdenticalRotatedAndTranslatedModels(self):
        query = buildDetailedGeometryFromShape(self.base, self.config)
        transform = gp_Trsf()
        transform.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(1, 2, 3)), 0.72)
        transform.SetTranslationPart(gp_Vec(100, -50, 20))
        rotated = BRepBuilderAPI_Transform(self.base, transform, True).Shape()
        candidate = buildDetailedGeometryFromShape(rotated, self.config)
        identical = compareDetailedGeometry(query, query)
        transformed = compareDetailedGeometry(query, candidate)
        self.assertLess(identical["detailedDistance"], 1e-8)
        # Broad tolerance for finite samples/local alignment; no universal
        # rotation invariance assertion for all CAD models.
        self.assertLess(transformed["surfaceMeanMm"], 0.5)
        self.assertLess(transformed["faceDistance"], 0.05)
        self.assertGreater(np.linalg.det(np.asarray(transformed["alignment"]["rotation"])), 0.99)

    def testExtraHoleRanksCloserThanUnrelatedCylinder(self):
        query = buildDetailedGeometryFromShape(self.base, self.config)
        hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(7, 8, -1), gp_Dir(0, 0, 1)), 2, 12).Shape()
        changed = buildDetailedGeometryFromShape(BRepAlgoAPI_Cut(self.base, hole).Shape(), self.config)
        different = buildDetailedGeometryFromShape(BRepPrimAPI_MakeCylinder(10, 30).Shape(), self.config)
        sameScore = compareDetailedGeometry(query, query)["detailedDistance"]
        changedScore = compareDetailedGeometry(query, changed)["detailedDistance"]
        differentScore = compareDetailedGeometry(query, different)["detailedDistance"]
        self.assertLess(sameScore, changedScore)
        self.assertLess(changedScore, differentScore)
        self.assertTrue(any(face.surfaceType == "cylinder" for face in changed.faces))
        self.assertEqual(changed.diagnostics["facesWithoutSamples"], 0)

    def testRealOfflineBuildSearchHistoricalFilesNotLoadedAndCli(self):
        historical = self.rootPath / "history"
        historical.mkdir()
        writeStep(self.base, historical / "box.step")
        writeStep(BRepPrimAPI_MakeCylinder(10, 30).Shape(), historical / "cylinder.step")
        queryPath = self.rootPath / "query.step"
        writeStep(self.base, queryPath)
        sphereConfig = DescriptorConfig(thetaResolution=4, phiResolution=8)
        databasePath = self.rootPath / "parts.sqlite3"
        with PartDatabase(databasePath) as database:
            self.assertEqual(indexPath(historical, database).indexed, 2)
            self.assertEqual(buildDescriptors(DescriptorStore(database), sphereConfig).built, 2)
            store = GeometryStore(database)
            self.assertEqual(buildDetailedGeometries(store, self.config).built, 2)
            self.assertEqual(buildDetailedGeometries(store, self.config).skipped, 2)
        # Cached ranking survives unavailable archive files. Only the new query
        # STEP is read by each query preprocessing stage.
        for path in historical.glob("*.step"):
            path.unlink()
        report = findDetailedMatchesForStep(queryPath, databasePath=databasePath, descriptorConfig=sphereConfig,
                                            geometryConfig=self.config, allFamilies=True, dimensionTolerance=1,
                                            stage2TopK=2, topK=1)
        self.assertEqual((report.stage1Candidates, report.stage2Candidates, report.stage3Candidates), (2, 2, 1))
        self.assertTrue(report.matches[0].filePath.endswith("box.step"))
        self.assertLess(report.matches[0].detailedDistance, 1e-8)
        for name in ("cachedGeometryLookupSeconds", "candidateComparisonSeconds", "stage3TotalSeconds"):
            self.assertGreaterEqual(report.timings[name], 0)
        mainPath = Path(__file__).resolve().parents[1] / "main.py"
        result = subprocess.run([sys.executable, str(mainPath), "search-detailed", str(queryPath), "--db", str(databasePath),
                                 "--samples", "128", "--theta-resolution", "4", "--phi-resolution", "8", "--all-families",
                                 "--tolerance", "1", "--stage2-top-k", "2", "--top-k", "1"],
                                capture_output=True, text=True, cwd=self.rootPath)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["stage3Candidates"], 1)
        self.assertTrue(output["matches"][0]["filePath"].endswith("box.step"))
        self.assertIn("sphereDistance", output["matches"][0])

    def testMissingInvalidAndStaleStepFailures(self):
        with self.assertRaises(OSError):
            buildDetailedGeometryFromStep(self.rootPath / "missing.step", self.config)
        path = self.rootPath / "bad.step"
        path.write_text("invalid STEP")
        with self.assertRaises(CadError):
            buildDetailedGeometryFromStep(path, self.config)
        writeStep(self.base, path)
        with self.assertRaisesRegex(DetailedSimilarityError, "Stage 1"):
            buildDetailedGeometryFromStep(path, self.config, expectedSourceHash="b" * 64)
        with patch("src.detailedSimilarity.faceExtractor.extractFaceFeatures", side_effect=RuntimeError("unsupported surface")):
            with self.assertRaisesRegex(DetailedSimilarityError, "unsupported surface"):
                buildDetailedGeometryFromShape(self.base, self.config)


if __name__ == "__main__":
    unittest.main()
