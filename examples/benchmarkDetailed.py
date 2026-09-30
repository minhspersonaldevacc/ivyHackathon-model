"""Measure actual shortlist reranking and extrapolate an exhaustive cached scan.

Run: python -m examples.benchmarkDetailed query.step --db parts.sqlite3
No exhaustive comparison is performed. Estimates assume identical per-part cost,
which is deliberately only a rough illustration, not a speed guarantee.
"""

import argparse
import json

from src.detailedSimilarity.detailedSearch import findDetailedMatchesForStep
from src.indexing.database import PartDatabase
from src.main import nativeMessagesToStderr
from src.models.detailedGeometry import DetailedGeometryConfig, DetailedSearchConfig
from src.models.sphereDescriptor import DescriptorConfig


def benchmarkDetailed(queryPath, databasePath, *, librarySize=100000, stage2TopK=50,
                      geometryConfig=None, descriptorConfig=None, searchConfig=None):
    if type(librarySize) is not int or librarySize < 1:
        raise ValueError("librarySize must be a positive integer")
    report = findDetailedMatchesForStep(queryPath, databasePath=databasePath, stage2TopK=stage2TopK,
                                       geometryConfig=geometryConfig, descriptorConfig=descriptorConfig,
                                       searchConfig=searchConfig, allFamilies=True, dimensionTolerance=0.5)
    with PartDatabase(databasePath, readOnly=True) as database:
        indexed = database.connection.execute(
            "SELECT count(*) FROM parts p JOIN ingestionFiles f ON f.filePath = p.filePath WHERE f.status = 'indexed'"
        ).fetchone()[0]
    perCandidate = report.timings.get("secondsPerCandidate", 0)
    return {"measuredIndexedParts": indexed, "stage1Candidates": report.stage1Candidates,
            "stage2Candidates": report.stage2Candidates, "comparedCandidates": report.comparedCandidates,
            "timings": report.timings,
            "extrapolation": {"librarySize": librarySize, "secondsPerCandidate": perCandidate,
                              "estimatedExhaustiveCachedComparisonSeconds": perCandidate * librarySize
                              if report.comparedCandidates else None,
                              "shortlistComparisonSeconds": report.timings.get("candidateComparisonSeconds", 0),
                              "assumption": "Same candidate complexity and per-part comparison cost; excludes exhaustive cache reads and offline generation"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--db", default="parts.sqlite3")
    parser.add_argument("--library-size", dest="librarySize", type=int, default=100000)
    parser.add_argument("--stage2-top-k", dest="stage2TopK", type=int, default=50)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--theta-resolution", dest="thetaResolution", type=int, default=32)
    parser.add_argument("--phi-resolution", dest="phiResolution", type=int, default=64)
    options = parser.parse_args()
    with nativeMessagesToStderr():
        result = benchmarkDetailed(options.path, options.db, librarySize=options.librarySize,
                                   stage2TopK=options.stage2TopK,
                                   geometryConfig=DetailedGeometryConfig(sampleCount=options.samples),
                                   descriptorConfig=DescriptorConfig(thetaResolution=options.thetaResolution,
                                                                     phiResolution=options.phiResolution),
                                   searchConfig=DetailedSearchConfig())
    print(json.dumps(result, indent=2, allow_nan=False))
