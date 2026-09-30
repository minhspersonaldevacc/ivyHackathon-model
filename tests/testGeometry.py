"""Real OCCT primitives and STEP round trips; no mocked geometry measurements."""

import importlib.util
import json
from math import pi
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.errors import CadError
from src.indexing.database import PartDatabase
from src.indexing.partClassifier import classifyPart
from src.indexing.partIndexer import indexFile, indexPath, inspectPart


hasOcc = importlib.util.find_spec("OCC") is not None
if hasOcc:
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Transform
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
    from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf
    from src.cad.geometryExtractor import extractGeometry


def writeStep(shape, filePath):
    writer = STEPControl_Writer()
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError("Fixture STEP transfer failed")
    if writer.Write(str(filePath)) != IFSelect_RetDone:
        raise RuntimeError("Fixture STEP write failed")


@unittest.skipUnless(hasOcc, "Install environment.yml to run real OpenCascade tests")
class GeometryTests(unittest.TestCase):
    def testBoxPropertiesAndUniqueTopology(self):
        geometry = extractGeometry(BRepPrimAPI_MakeBox(10, 20, 30).Shape())
        self.assertEqual((geometry.sizeX, geometry.sizeY, geometry.sizeZ), (10, 20, 30))
        self.assertAlmostEqual(geometry.volume, 6000)
        self.assertAlmostEqual(geometry.surfaceArea, 2200)
        self.assertEqual((geometry.centerOfMassX, geometry.centerOfMassY, geometry.centerOfMassZ), (5, 10, 15))
        self.assertEqual((geometry.faceCount, geometry.edgeCount, geometry.vertexCount), (6, 12, 8))
        self.assertEqual(geometry.planarFaces, 6)
        self.assertEqual(geometry.planarRatio, 1)
        self.assertEqual(classifyPart(geometry), "block-like")

    def testPlateShaftRingConeAndSphere(self):
        ring = BRepAlgoAPI_Cut(
            BRepPrimAPI_MakeCylinder(10, 3).Shape(),
            BRepPrimAPI_MakeCylinder(5, 3).Shape(),
        ).Shape()
        cases = [
            (BRepPrimAPI_MakeBox(100, 80, 5).Shape(), "plate-like"),
            (BRepPrimAPI_MakeCylinder(5, 80).Shape(), "shaft/turned-like"),
            (ring, "cylindrical/ring-like"),
            (BRepPrimAPI_MakeCone(10, 5, 15).Shape(), "general/unknown"),
            (BRepPrimAPI_MakeSphere(10).Shape(), "general/unknown"),
        ]
        for shape, expected in cases:
            with self.subTest(expected=expected):
                geometry = extractGeometry(shape)
                self.assertEqual(classifyPart(geometry), expected)
                self.assertEqual(geometry.faceCount, geometry.planarFaces + geometry.cylindricalFaces + geometry.conicalFaces + geometry.otherFaces)
                self.assertAlmostEqual(geometry.planarRatio + geometry.cylindricalRatio + geometry.conicalRatio + geometry.otherRatio, 1)
        ringGeometry = extractGeometry(ring)
        self.assertEqual(ringGeometry.cylindricalFaces, 2)
        self.assertAlmostEqual(ringGeometry.volume, pi * (100 - 25) * 3)
        self.assertEqual(extractGeometry(cases[3][0]).conicalFaces, 1)
        self.assertEqual(extractGeometry(cases[4][0]).otherRatio, 1)

    def testAxisPermutationAndTranslation(self):
        shape = BRepPrimAPI_MakeBox(gp_Pnt(100, 200, 300), 10, 20, 30).Shape()
        transform = gp_Trsf()
        transform.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), pi / 2)
        rotated = BRepBuilderAPI_Transform(shape, transform, True).Shape()
        geometry = extractGeometry(rotated)
        for measured, expected in zip((geometry.sizeMin, geometry.sizeMid, geometry.sizeMax), (10, 20, 30)):
            self.assertAlmostEqual(measured, expected)
        self.assertAlmostEqual(geometry.centerOfMassX, -210)
        self.assertAlmostEqual(geometry.centerOfMassY, 105)
        self.assertAlmostEqual(geometry.centerOfMassZ, 315)

    def testMultiBodyMeasuresSolidsAndIgnoresLooseFaces(self):
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        builder.Add(compound, BRepPrimAPI_MakeBox(10, 10, 10).Shape())
        builder.Add(compound, BRepPrimAPI_MakeBox(gp_Pnt(20, 0, 0), 10, 10, 10).Shape())
        looseFace = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(1000, 0, 0), gp_Dir(0, 0, 1)), 0, 50, 0, 50).Shape()
        builder.Add(compound, looseFace)
        geometry = extractGeometry(compound)
        self.assertEqual(geometry.solidCount, 2)
        self.assertAlmostEqual(geometry.volume, 2000)
        self.assertAlmostEqual(geometry.surfaceArea, 1200)
        self.assertAlmostEqual(geometry.sizeMax, 30)
        self.assertAlmostEqual(geometry.centerOfMassX, 15)
        self.assertEqual(geometry.faceCount, 12)
        self.assertEqual(classifyPart(geometry), "general/unknown")

    def testNullSurfaceAndReversedSolidAreRejected(self):
        face = BRepBuilderAPI_MakeFace(gp_Pln(), 0, 10, 0, 10).Shape()
        shapes = [TopoDS_Shape(), face, BRepPrimAPI_MakeBox(10, 20, 30).Shape().Reversed()]
        for shape in shapes:
            with self.subTest(shape=shape), self.assertRaises(CadError):
                extractGeometry(shape)


@unittest.skipUnless(hasOcc, "Install environment.yml to run real OpenCascade tests")
class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.tempDirectory = TemporaryDirectory()
        self.addCleanup(self.tempDirectory.cleanup)
        self.rootPath = Path(self.tempDirectory.name)
        self.partsPath = self.rootPath / "parts"
        self.partsPath.mkdir()
        self.filePath = self.partsPath / "block.STEP"
        writeStep(BRepPrimAPI_MakeBox(10, 20, 30).Shape(), self.filePath)
        self.database = PartDatabase(self.rootPath / "parts.sqlite3")
        self.addCleanup(self.database.close)

    def testStepRoundTripAndUnchangedFilesNeverReopen(self):
        self.assertEqual(indexPath(self.partsPath, self.database).indexed, 1)
        original = self.database.getPart(str(self.filePath))
        self.assertAlmostEqual(original.volume, 6000)
        with patch("src.indexing.partIndexer.hashFile", side_effect=AssertionError("Unchanged source was reopened")), patch("src.indexing.partIndexer.analyzeStep", side_effect=AssertionError("Unchanged source was reprocessed")):
            self.assertEqual(indexPath(self.partsPath, self.database).skipped, 1)
        self.assertEqual(len(self.database.findCandidates("block-like", (30, 10, 20), 0.01)), 1)

    def testTimestampChangesAndContentVerificationReuseGeometry(self):
        indexFile(self.filePath, self.database)
        fileStat = self.filePath.stat()
        os.utime(self.filePath, ns=(fileStat.st_atime_ns, fileStat.st_mtime_ns + 1_000_000_000))
        with patch("src.indexing.partIndexer.analyzeStep", side_effect=AssertionError("Identical bytes were reprocessed")):
            self.assertEqual(indexFile(self.filePath, self.database).status, "skipped")
            self.assertEqual(indexFile(self.filePath, self.database, verifyContent=True).status, "skipped")

    def testChangedFileReplacesMetadataWithStableId(self):
        indexFile(self.filePath, self.database)
        original = self.database.getPart(str(self.filePath))
        writeStep(BRepPrimAPI_MakeBox(100, 80, 5).Shape(), self.filePath)
        self.assertEqual(indexFile(self.filePath, self.database).status, "indexed")
        updated = self.database.getPart(str(self.filePath))
        self.assertEqual(original.partId, updated.partId)
        self.assertNotEqual(original.sourceHash, updated.sourceHash)
        self.assertEqual(updated.shapeFamily, "plate-like")
        self.assertEqual(self.database.findCandidates("block-like", (10, 20, 30), 0.1), [])

    def testCorruptFileIsRememberedAndDoesNotAbortBatch(self):
        badPath = self.partsPath / "broken.stp"
        badPath.write_text("not a STEP file", encoding="utf-8")
        summary = indexPath(self.partsPath, self.database)
        self.assertEqual((summary.indexed, summary.failed), (1, 1))
        with patch("src.indexing.partIndexer.analyzeStep", side_effect=AssertionError("Known failure was reprocessed")):
            self.assertEqual(indexPath(self.partsPath, self.database).skippedFailed, 1)
        self.assertEqual(indexFile(badPath, self.database, retryFailed=True).status, "failed")

    def testChangedCorruptFileHidesOldMetadata(self):
        indexFile(self.filePath, self.database)
        self.filePath.write_text("corrupted after indexing", encoding="utf-8")
        self.assertEqual(indexFile(self.filePath, self.database).status, "failed")
        self.assertEqual(self.database.findCandidates(None, (10, 20, 30), 0.1), [])

    def testClassifierUpgradeReusesStoredGeometry(self):
        indexFile(self.filePath, self.database)
        self.database.connection.execute("UPDATE parts SET classifierVersion = 0, shapeFamily = 'general/unknown'")
        self.database.connection.commit()
        with patch("src.indexing.partIndexer.hashFile", side_effect=AssertionError("Source was reopened")):
            self.assertEqual(indexFile(self.filePath, self.database).status, "reclassified")
        self.assertEqual(self.database.getPart(str(self.filePath)).shapeFamily, "block-like")

    def testSourceMutationDuringExtractionIsNotPublished(self):
        original = inspectPart(self.filePath)

        def mutateSource(filePath, sourceHash):
            filePath.write_text("changed while loading", encoding="utf-8")
            return original

        with patch("src.indexing.partIndexer.analyzeStep", side_effect=mutateSource):
            self.assertEqual(indexFile(self.filePath, self.database).status, "failed")
        self.assertEqual(self.database.findCandidates(None, (10, 20, 30), 0.1), [])

    def testDeclaredInchUnitsConvertToMillimeters(self):
        inchPath = self.partsPath / "inch.stp"
        # Use OCCT's writer to produce valid conversion-based inch units.
        originalUnit = Interface_Static.CVal("write.step.unit")
        try:
            self.assertTrue(Interface_Static.SetCVal("write.step.unit", "INCH"))
            writeStep(BRepPrimAPI_MakeBox(25.4, 50.8, 76.2).Shape(), inchPath)
        finally:
            Interface_Static.SetCVal("write.step.unit", originalUnit)
        self.assertIn("CONVERSION_BASED_UNIT('INCH'", inchPath.read_text(encoding="utf-8"))
        metadata = inspectPart(inchPath)
        self.assertAlmostEqual(metadata.sizeX, 25.4)
        self.assertAlmostEqual(metadata.sizeY, 50.8)
        self.assertAlmostEqual(metadata.sizeZ, 76.2)
        self.assertAlmostEqual(metadata.volume, 6 * 25.4 ** 3)

    def testInspectCliPrintsJsonWithoutCreatingDatabase(self):
        mainPath = Path(__file__).resolve().parents[1] / "main.py"
        result = subprocess.run(
            [sys.executable, str(mainPath), "inspect", str(self.filePath)],
            cwd=self.partsPath, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads(result.stdout)
        self.assertEqual(metadata["shapeFamily"], "block-like")
        self.assertAlmostEqual(metadata["volume"], 6000)
        self.assertFalse((self.partsPath / "parts.sqlite3").exists())

    def testIndexCliKeepsJsonValidWhenStepParsingFails(self):
        (self.partsPath / "broken.stp").write_text("corrupt STEP", encoding="utf-8")
        mainPath = Path(__file__).resolve().parents[1] / "main.py"
        result = subprocess.run(
            [sys.executable, str(mainPath), "index", str(self.partsPath), "--db", str(self.rootPath / "cli.sqlite3")],
            cwd=self.rootPath, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual((summary["indexed"], summary["failed"]), (1, 1))
        self.assertIn("broken.stp", result.stderr)


if __name__ == "__main__":
    unittest.main()
