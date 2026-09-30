"""Benchmark ranking 100/1,000/10,000 stored synthetic descriptors.

Run: python -m examples.benchmarkDescriptors
This measures actual SQLite reads, decompression, and four-variant comparisons.
Analytic box fingerprints are used as fixtures; it does not measure STEP ingestion
or retrieval accuracy on a real manufacturing archive. No pythonOCC is required.
"""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from examples.benchmarkFiltering import syntheticParts
from src.descriptors.descriptorStore import DescriptorStore
from src.geometry.sphericalSampler import sphericalDirections
from src.indexing.candidateSearch import rankCandidates
from src.indexing.database import PartDatabase, metadataColumns
from src.models.partMetadata import PartMetadata, extractorVersion
from src.models.sphereDescriptor import DescriptorConfig, SphereDescriptor


def benchmarkDescriptors(
    candidateCounts=(100, 1000, 10000), config: DescriptorConfig | None = None,
    *, metric: str = "euclidean",
) -> dict:
    config = config or DescriptorConfig()
    if not candidateCounts or any(type(count) is not int or count <= 0 for count in candidateCounts):
        raise ValueError("candidateCounts must be a nonempty list of positive integers")
    counts = sorted(set(candidateCounts))
    with TemporaryDirectory(prefix="cad-descriptor-benchmark-") as directory:
        with PartDatabase(Path(directory) / "parts.sqlite3") as database:
            store = DescriptorStore(database)
            seedAt = perf_counter()
            records = list(syntheticParts(max(counts)))
            with database.connection:
                database.connection.executemany(
                    "INSERT INTO ingestionFiles VALUES (?, 0, 0, 0, ?, ?, 'indexed', NULL, 'synthetic')",
                    ((record["filePath"], record["sourceHash"], extractorVersion) for record in records),
                )
                columns = ", ".join(metadataColumns)
                values = ", ".join(f":{column}" for column in metadataColumns)
                database.connection.executemany(f"INSERT INTO parts ({columns}) VALUES ({values})", records)

            directions = sphericalDirections(config.thetaResolution, config.phiResolution)
            query = None
            for record in records:
                # Analytic centered-box first/last radius, with PCA axes longest
                # first. Fixture creation is excluded from the search timings.
                dimensions = np.array([record["sizeMax"], record["sizeMid"], record["sizeMin"]])
                halfSides = dimensions / np.linalg.norm(dimensions)
                radii = 1 / np.max(np.abs(directions) / halfSides, axis=-1)
                tensor = np.stack((radii, radii, np.full_like(radii, 1 / config.countScale), radii), axis=-1)
                descriptor = SphereDescriptor(tensor, config, sourceHash=record["sourceHash"])
                store.storeDescriptor(PartMetadata(**record), descriptor)
                if query is None:
                    query = descriptor
            seedSeconds = perf_counter() - seedAt
            database.optimize()
            reports = []
            partIds = [record["partId"] for record in records]
            for count in counts:
                report = rankCandidates(query, partIds[:count], store, topK=50, metric=metric)
                reports.append({
                    "candidateCount": count, "returnedMatches": len(report.matches),
                    "bestDistance": report.matches[0].distance, **report.timings,
                })
            return {
                "synthetic": True, "config": config.toDict(), "metric": metric,
                "axisFlipVariants": 4, "seedSecondsExcludedFromSearch": seedSeconds,
                "benchmarks": reports,
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--theta-resolution", dest="thetaResolution", type=int, default=32)
    parser.add_argument("--phi-resolution", dest="phiResolution", type=int, default=64)
    parser.add_argument("--metric", choices=("euclidean", "cosine"), default="euclidean")
    options = parser.parse_args()
    print("Seeding a temporary descriptor database; reported search times exclude seeding.", file=sys.stderr)
    print(json.dumps(benchmarkDescriptors(
        options.counts, DescriptorConfig(thetaResolution=options.thetaResolution, phiResolution=options.phiResolution),
        metric=options.metric,
    ), indent=2))
