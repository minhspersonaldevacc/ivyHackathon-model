"""Filtering tests run even in a Python environment without OpenCascade."""

from dataclasses import replace
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from src.errors import DatabaseError
from src.indexing.database import PartDatabase, findCandidates
from src.models.partMetadata import PartMetadata


def makeMetadata(filePath: str, dimensions=(10.0, 20.0, 30.0), family="block-like"):
    sizeX, sizeY, sizeZ = dimensions
    sizeMin, sizeMid, sizeMax = sorted(dimensions)
    return PartMetadata(
        partId=filePath, filePath=filePath, sourceHash="a" * 64,
        sizeX=sizeX, sizeY=sizeY, sizeZ=sizeZ,
        sizeMin=sizeMin, sizeMid=sizeMid, sizeMax=sizeMax,
        volume=sizeX * sizeY * sizeZ, surfaceArea=2 * (sizeX * sizeY + sizeX * sizeZ + sizeY * sizeZ),
        centerOfMassX=sizeX / 2, centerOfMassY=sizeY / 2, centerOfMassZ=sizeZ / 2,
        aspectXY=sizeX / sizeY, aspectXZ=sizeX / sizeZ, aspectYZ=sizeY / sizeZ,
        faceCount=6, edgeCount=12, vertexCount=8, solidCount=1,
        planarFaces=6, cylindricalFaces=0, conicalFaces=0, otherFaces=0,
        planarRatio=1.0, cylindricalRatio=0.0, conicalRatio=0.0, otherRatio=0.0,
        shapeFamily=family, extractedAt="2026-01-01T00:00:00+00:00",
    )


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tempDirectory = TemporaryDirectory()
        self.addCleanup(self.tempDirectory.cleanup)
        self.databasePath = Path(self.tempDirectory.name) / "parts.sqlite3"
        self.database = PartDatabase(self.databasePath)
        self.addCleanup(self.database.close)
        self.fingerprint = {"fileSize": 123, "modifiedTimeNs": 1, "changedTimeNs": 1}

    def store(self, name, dimensions=(10.0, 20.0, 30.0), family="block-like"):
        metadata = makeMetadata(str(Path(self.tempDirectory.name) / name), dimensions, family)
        self.database.storePart(metadata, self.fingerprint)
        return metadata

    def testAxisPermutationAndInclusiveTolerance(self):
        exact = self.store("exact")
        lower = self.store("lower", (9.0, 18.0, 27.0))
        upper = self.store("upper", (11.0, 22.0, 33.0))
        self.store("outsideOneDimension", (10.0, 23.0, 30.0))
        self.store("outsideSize", (90.0, 180.0, 270.0))
        self.store("wrongFamily", family="general/unknown")
        matches = self.database.findCandidates("block-like", (30, 10, 20), 0.10)
        self.assertEqual({match.partId for match in matches}, {exact.partId, lower.partId, upper.partId})

    def testVolumeRangeAndAllFamilies(self):
        exact = self.store("exact")
        general = self.store("general", family="general/unknown")
        self.store("larger", (11.0, 22.0, 33.0))
        matches = findCandidates(None, (10, 20, 30), 0.1, (6000, 6000), databasePath=self.databasePath)
        self.assertEqual({match.partId for match in matches}, {exact.partId, general.partId})

    def testZeroToleranceAndOptionalLimit(self):
        self.store("exact")
        self.store("sameDimensions")
        self.store("slightlyDifferent", (10.00001, 20, 30))
        self.assertEqual(len(self.database.findCandidates(None, (10, 20, 30), 0)), 2)
        self.assertEqual(len(self.database.findCandidates(None, (10, 20, 30), 0, limit=1)), 1)

    def testInvalidSearchInputs(self):
        badArguments = [
            ("not-a-family", (10, 20, 30), 0.1, None),
            (None, (10, 20), 0.1, None),
            (None, (0, 20, 30), 0.1, None),
            (None, (float("nan"), 20, 30), 0.1, None),
            (None, (float("inf"), 20, 30), 0.1, None),
            (None, (10, 20, 30), -0.1, None),
            (None, (10, 20, 30), 10, None),
            (None, (10, 20, 30), float("nan"), None),
            (None, (10, 20, 30), 0.1, (6000, 10)),
            (None, (10, 20, 30), 0.1, (-1, 10)),
            (None, (10, 20, 30), 0.1, (0, float("inf"))),
        ]
        for arguments in badArguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.database.findCandidates(*arguments)

    def testFailuresAndInterruptedChangesHideOldMetadata(self):
        metadata = self.store("part")
        self.database.beginFile(metadata.filePath, self.fingerprint)
        self.assertEqual(self.database.findCandidates(None, (10, 20, 30), 0.1), [])
        self.database.recordFailure(metadata.filePath, self.fingerprint, "bad STEP")
        self.assertIsNone(self.database.getPart(metadata.filePath))
        self.assertIsNotNone(self.database.getPart(metadata.filePath, includeInactive=True))
        self.assertEqual(self.database.findCandidates(None, (10, 20, 30), 0.1), [])
        self.database.storePart(metadata, self.fingerprint)
        self.assertEqual(len(self.database.findCandidates(None, (10, 20, 30), 0.1)), 1)

    def testUpsertKeepsIdentityAndHasNoDuplicate(self):
        original = self.store("part")
        self.database.storePart(replace(original, sourceHash="b" * 64), self.fingerprint)
        matches = self.database.findCandidates(None, (10, 20, 30), 0.1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].partId, original.partId)
        self.assertEqual(matches[0].sourceHash, "b" * 64)

    def testCandidatePlanUsesSizeIndex(self):
        self.store("part")
        query, parameters = self.database.buildCandidateQuery("block-like", (10, 20, 30), 0.1)
        plan = self.database.connection.execute("EXPLAIN QUERY PLAN " + query, parameters).fetchall()
        self.assertTrue(any("USING INDEX partsByFamilyAndSize" in row[3] for row in plan), plan)

    def testReadonlySearchDoesNotCreateDatabase(self):
        missingPath = Path(self.tempDirectory.name) / "missing.sqlite3"
        with self.assertRaises(sqlite3.OperationalError):
            PartDatabase(missingPath, readOnly=True)
        self.assertFalse(missingPath.exists())

    def testUnknownSchemaIsRejected(self):
        futurePath = Path(self.tempDirectory.name) / "future.sqlite3"
        with sqlite3.connect(futurePath) as connection:
            connection.execute("PRAGMA user_version = 999")
        with self.assertRaises(DatabaseError):
            PartDatabase(futurePath)


if __name__ == "__main__":
    unittest.main()
