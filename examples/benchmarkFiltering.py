"""Demonstrate metadata filtering over synthetic rows without installing OCC.

Run: python -m examples.benchmarkFiltering --parts 100000
The temporary database is removed afterwards. These are synthetic box records,
not STEP files or measurements of actual manufacturers' search workloads.
"""

import argparse
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from time import perf_counter

from src.indexing.database import PartDatabase, metadataColumns
from src.indexing.partClassifier import classifyPart
from src.models.partMetadata import GeometryProperties, PartMetadata, extractorVersion


def syntheticParts(partCount: int):
    randomSource = Random(42)
    for partIndex in range(partCount):
        sizeX, sizeY, sizeZ = (50.0, 70.0, 100.0) if partIndex == 0 else sorted(
            randomSource.uniform(5, 500) for _ in range(3)
        )
        geometry = GeometryProperties(
            sizeX=sizeX, sizeY=sizeY, sizeZ=sizeZ,
            sizeMin=sizeX, sizeMid=sizeY, sizeMax=sizeZ,
            volume=sizeX * sizeY * sizeZ, surfaceArea=2 * (sizeX * sizeY + sizeY * sizeZ + sizeX * sizeZ),
            centerOfMassX=sizeX / 2, centerOfMassY=sizeY / 2, centerOfMassZ=sizeZ / 2,
            aspectXY=sizeX / sizeY, aspectXZ=sizeX / sizeZ, aspectYZ=sizeY / sizeZ,
            faceCount=6, edgeCount=12, vertexCount=8, solidCount=1,
            planarFaces=6, cylindricalFaces=0, conicalFaces=0, otherFaces=0,
            planarRatio=1, cylindricalRatio=0, conicalRatio=0, otherRatio=0,
        )
        yield PartMetadata(
            **geometry.toDict(), partId=f"synthetic-{partIndex:06}",
            filePath=f"/synthetic/{partIndex}.step", sourceHash="0" * 64,
            shapeFamily=classifyPart(geometry), extractedAt="synthetic",
        ).toDict()


def benchmarkFiltering(partCount: int = 100000) -> dict:
    if partCount <= 0:
        raise ValueError("partCount must be positive")
    with TemporaryDirectory(prefix="cad-filter-demo-") as directory:
        with PartDatabase(Path(directory) / "parts.sqlite3") as database:
            # Bulk seeding is confined to this synthetic benchmark. Normal
            # ingestion commits each processed file for crash-safe resumption.
            with database.connection:
                database.connection.executemany(
                    "INSERT INTO ingestionFiles VALUES (?, 0, 0, 0, ?, ?, 'indexed', NULL, 'synthetic')",
                    ((f"/synthetic/{partIndex}.step", "0" * 64, extractorVersion) for partIndex in range(partCount)),
                )
                columns = ", ".join(metadataColumns)
                values = ", ".join(f":{column}" for column in metadataColumns)
                database.connection.executemany(
                    f"INSERT INTO parts ({columns}) VALUES ({values})", syntheticParts(partCount)
                )
            database.optimize()
            startedAt = perf_counter()
            candidates = database.findCandidates("block-like", (50, 70, 100), 0.10)
            elapsedMs = (perf_counter() - startedAt) * 1000
            query, parameters = database.buildCandidateQuery("block-like", (50, 70, 100), 0.10)
            queryPlan = [row[3] for row in database.connection.execute("EXPLAIN QUERY PLAN " + query, parameters)]
            return {
                "syntheticParts": partCount,
                "candidates": len(candidates),
                "queryMilliseconds": round(elapsedMs, 3),
                "queryPlan": queryPlan,
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=100000)
    print(json.dumps(benchmarkFiltering(parser.parse_args().parts), indent=2))
