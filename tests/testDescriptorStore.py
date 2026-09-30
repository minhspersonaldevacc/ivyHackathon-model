"""Descriptor persistence, invalidation, candidate ranking, and real STEP integration."""

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

from src.descriptors.descriptorBuilder import buildDescriptors
from src.descriptors.descriptorPipeline import buildDescriptorFromStep
from src.descriptors.descriptorStore import DescriptorStore
from src.errors import CadError, DescriptorError
from src.indexing.candidateSearch import findGeometricMatches, findMatchesForStep, rankCandidates
from src.indexing.database import PartDatabase
from src.indexing.partIndexer import indexFile
from src.models.sphereDescriptor import DescriptorConfig, SphereDescriptor
from tests.testDatabase import makeMetadata


class DescriptorStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempDirectory = TemporaryDirectory()
        self.addCleanup(self.tempDirectory.cleanup)
        self.rootPath = Path(self.tempDirectory.name)
        self.database = PartDatabase(self.rootPath / "parts.sqlite3")
        self.addCleanup(self.database.close)
        self.config = DescriptorConfig(thetaResolution=4, phiResolution=8)
        self.part = makeMetadata(str(self.rootPath / "notOnDisk.step"))
        self.fingerprint = {"fileSize": 1, "modifiedTimeNs": 1, "changedTimeNs": 1}
        self.database.storePart(self.part, self.fingerprint)
        self.store = DescriptorStore(self.database)
        self.descriptor = SphereDescriptor(
            np.random.default_rng(8).random(self.config.tensorShape), self.config,
            sourceHash=self.part.sourceHash,
        )

    def testAdditiveSchemaRoundTripAndReadonlyAccess(self):
        original = self.database.getPart(self.part.filePath)
        self.store.storeDescriptor(self.part, self.descriptor)
        with PartDatabase(self.database.databasePath, readOnly=True) as database:
            loaded = DescriptorStore(database, initialize=False).loadDescriptors([self.part.partId], self.config)[self.part.partId]
        np.testing.assert_array_equal(loaded.tensor, self.descriptor.tensor)
        self.assertEqual(loaded.config.configKey, self.descriptor.config.configKey)
        self.assertEqual(self.database.getPart(self.part.filePath), original)
        self.assertEqual(self.database.connection.execute("PRAGMA user_version").fetchone()[0], 1)

    def testSourceChangesAndFailedStage1EntriesHideDescriptors(self):
        self.store.storeDescriptor(self.part, self.descriptor)
        self.database.beginFile(self.part.filePath, self.fingerprint)
        self.assertEqual(self.store.loadDescriptors([self.part.partId], self.config), {})
        self.database.storePart(replace(self.part, sourceHash="b" * 64), self.fingerprint)
        self.assertEqual(self.store.loadDescriptors([self.part.partId], self.config), {})
        with self.assertRaises(DescriptorError):
            self.store.storeDescriptor(self.part, self.descriptor)

    def testDifferentConfigurationsCoexist(self):
        self.store.storeDescriptor(self.part, self.descriptor)
        otherConfig = replace(self.config, countScale=8)
        other = replace(self.descriptor, config=otherConfig)
        self.store.storeDescriptor(self.part, other)
        self.assertEqual(self.database.connection.execute("SELECT count(*) FROM sphereDescriptors").fetchone()[0], 2)
        self.assertEqual(len(self.store.loadDescriptors([self.part.partId], self.config)), 1)
        self.assertEqual(len(self.store.loadDescriptors([self.part.partId], otherConfig)), 1)

    def testBuildOnlyOnceAndRebuildExplicitly(self):
        with patch("src.descriptors.descriptorBuilder.buildDescriptorFromStep", return_value=self.descriptor) as generator:
            self.assertEqual(buildDescriptors(self.store, self.config).built, 1)
            self.assertEqual(buildDescriptors(self.store, self.config).skipped, 1)
            self.assertEqual(generator.call_count, 1)
            self.assertEqual(buildDescriptors(self.store, self.config, rebuild=True).built, 1)
            self.assertEqual(generator.call_count, 2)
        # The test path does not exist: the cached path cannot open even a stat.
        self.assertFalse(Path(self.part.filePath).exists())

    def testFailureRetryAndInterruptedWork(self):
        with patch("src.descriptors.descriptorBuilder.buildDescriptorFromStep", side_effect=CadError("bad mesh")) as generator:
            self.assertEqual(buildDescriptors(self.store, self.config).failed, 1)
            self.assertEqual(buildDescriptors(self.store, self.config).skippedFailed, 1)
            self.assertEqual(generator.call_count, 1)
            self.assertEqual(buildDescriptors(self.store, self.config, retryFailed=True).failed, 1)
            self.assertEqual(generator.call_count, 2)
        self.store.beginPart(self.part, self.config)
        with patch("src.descriptors.descriptorBuilder.buildDescriptorFromStep", return_value=self.descriptor):
            self.assertEqual(buildDescriptors(self.store, self.config).built, 1)

    def testRankingTopKAndMissingDescriptorPolicy(self):
        self.store.storeDescriptor(self.part, self.descriptor)
        distantPart = replace(self.part, partId="distant", filePath=str(self.rootPath / "distant.step"))
        self.database.storePart(distantPart, self.fingerprint)
        self.store.storeDescriptor(distantPart, replace(self.descriptor, tensor=np.zeros(self.config.tensorShape)))
        matches = findGeometricMatches(
            self.descriptor, ["distant", self.part.partId, self.part.partId], topK=1,
            databasePath=self.database.databasePath,
        )
        self.assertEqual(matches[0].partId, self.part.partId)
        self.assertEqual(matches[0].distance, 0)
        with self.assertRaises(DescriptorError):
            rankCandidates(self.descriptor, [self.part.partId, "missing"], self.store)
        report = rankCandidates(self.descriptor, [self.part.partId, "missing"], self.store, skipMissing=True)
        self.assertEqual(report.comparedCandidates, 1)
        self.assertEqual(report.unavailablePartIds, ["missing"])
        cosine = rankCandidates(self.descriptor, [self.part.partId], self.store, metric="cosine")
        self.assertAlmostEqual(cosine.matches[0].cosineSimilarity, 1)
        self.assertEqual(rankCandidates(self.descriptor, [], self.store).matches, [])

    def testCorruptBlobFailsClearly(self):
        self.store.storeDescriptor(self.part, self.descriptor)
        self.database.connection.execute("UPDATE sphereDescriptors SET tensorBlob = ?", (b"corrupt",))
        self.database.connection.commit()
        with self.assertRaisesRegex(DescriptorError, "Invalid stored descriptor"):
            self.store.loadDescriptors([self.part.partId], self.config)

    def testMissingStoreIsNotCreatedByReadonlySearch(self):
        path = self.rootPath / "stage1Only.sqlite3"
        with PartDatabase(path):
            pass
        with PartDatabase(path, readOnly=True) as database:
            with self.assertRaises(DescriptorError):
                DescriptorStore(database, initialize=False)


hasOcc = importlib.util.find_spec("OCC") is not None
if hasOcc:
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopoDS import topods
    from OCC.Core.gp import gp_Pnt
    from src.cad.meshExtractor import extractMesh
    from src.cad.topology import mapShapes
    from src.descriptors.descriptorPipeline import generateShapeDescriptor
    from tests.testGeometry import writeStep


@unittest.skipUnless(hasOcc, "Install environment.yml for real STEP descriptor tests")
class DescriptorStepTests(unittest.TestCase):
    def setUp(self):
        self.tempDirectory = TemporaryDirectory()
        self.addCleanup(self.tempDirectory.cleanup)
        self.rootPath = Path(self.tempDirectory.name)
        self.filePath = self.rootPath / "box.step"
        writeStep(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), self.filePath)
        self.config = DescriptorConfig(thetaResolution=4, phiResolution=8)

    def testMeshingPreservesOriginalAndAppliesWorldPlacement(self):
        shape = BRepPrimAPI_MakeBox(gp_Pnt(100, 200, 300), 10, 20, 30).Shape()
        faces = mapShapes(shape, TopAbs_FACE)
        before = [BRep_Tool.Triangulation(topods.Face(faces.FindKey(index)), TopLoc_Location()) for index in range(1, faces.Size() + 1)]
        self.assertTrue(all(mesh is None for mesh in before))
        mesh = extractMesh(shape, self.config)
        np.testing.assert_allclose(mesh.vertices.min(axis=0), (100, 200, 300))
        np.testing.assert_allclose(mesh.vertices.max(axis=0), (110, 220, 330))
        for index in range(1, faces.Size() + 1):
            self.assertIsNone(BRep_Tool.Triangulation(topods.Face(faces.FindKey(index)), TopLoc_Location()))

    def testRealStepOfflineBuildSearchAndTimingFields(self):
        with PartDatabase(self.rootPath / "parts.sqlite3") as database:
            self.assertEqual(indexFile(self.filePath, database).status, "indexed")
            original = database.getPart(str(self.filePath))
            store = DescriptorStore(database)
            self.assertEqual(buildDescriptors(store, self.config).built, 1)
            stored = store.loadDescriptors([original.partId], self.config)[original.partId]
            self.assertEqual(database.getPart(str(self.filePath)), original)
            for key in ("stepLoadSeconds", "triangulationSeconds", "normalizationSeconds", "bvhBuildSeconds", "descriptorGenerationSeconds"):
                self.assertGreaterEqual(stored.timings[key], 0)
            report = findMatchesForStep(self.filePath, databasePath=database.databasePath, config=self.config)
            self.assertEqual(report.matches[0].partId, original.partId)
            self.assertAlmostEqual(report.matches[0].distance, 0)
            self.filePath.unlink()
            with patch("src.descriptors.descriptorBuilder.buildDescriptorFromStep", side_effect=AssertionError("cached source loaded")):
                self.assertEqual(buildDescriptors(store, self.config).skipped, 1)
            self.assertEqual(rankCandidates(stored, [original.partId], store).matches[0].distance, 0)

    def testSourceHashMismatchDoesNotPublish(self):
        with self.assertRaisesRegex(DescriptorError, "Stage 1"):
            buildDescriptorFromStep(self.filePath, self.config, expectedSourceHash="wrong")

    def testSphereBvhAccuracyAndPruning(self):
        descriptor = generateShapeDescriptor(BRepPrimAPI_MakeSphere(10).Shape(), self.config)
        np.testing.assert_allclose(descriptor.tensor[..., 0], 1, atol=0.01)
        self.assertLess(descriptor.diagnostics["trianglesTested"], descriptor.diagnostics["naiveTriangleTests"] / 10)

    def testDescriptorCliJsonPngAndNpyWithoutDatabase(self):
        mainPath = Path(__file__).resolve().parents[1] / "main.py"
        result = subprocess.run(
            [sys.executable, str(mainPath), "descriptor", str(self.filePath),
             "--theta-resolution", "4", "--phi-resolution", "8",
             "--save-images", str(self.rootPath / "images"), "--save-tensor", str(self.rootPath / "tensor.npy")],
            capture_output=True, text=True, cwd=self.rootPath, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["descriptorDimensions"], [4, 8, 4])
        for path in summary["channelImages"]:
            self.assertTrue(Path(path).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(np.load(self.rootPath / "tensor.npy", allow_pickle=False).shape, (4, 8, 4))
        self.assertFalse((self.rootPath / "parts.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
