"""Extend the existing JSON CLI without importing Stage 3 dependencies eagerly."""

import json
from pathlib import Path
import sys


detailedCommands = {"build-detailed-geometry", "detailed-geometry", "compare-detailed", "search-detailed"}


def addDetailedCommands(commands):
    buildParser = commands.add_parser("build-detailed-geometry", help="Precompute missing/stale Stage 3 geometry")
    buildParser.add_argument("--db", default="parts.sqlite3")
    buildParser.add_argument("--retry-failed", dest="retryFailed", action="store_true")
    buildParser.add_argument("--rebuild", action="store_true")
    buildParser.add_argument("--limit", type=int, default=None)
    buildParser.add_argument("--verbose", action="store_true")
    debugParser = commands.add_parser("detailed-geometry", help="Inspect Stage 3 preprocessing without database writes")
    debugParser.add_argument("path")
    compareParser = commands.add_parser("compare-detailed", help="Compare two STEP files for development/evaluation")
    compareParser.add_argument("path")
    compareParser.add_argument("candidatePath")
    searchParser = commands.add_parser("search-detailed", help="Stages 1/2 retrieval then cached Stage 3 reranking")
    searchParser.add_argument("path")
    searchParser.add_argument("--db", default="parts.sqlite3")
    searchParser.add_argument("--stage2-top-k", dest="stage2TopK", type=int, default=50)
    searchParser.add_argument("--top-k", dest="topK", type=int, default=5)
    searchParser.add_argument("--tolerance", type=float, default=0.10)
    searchParser.add_argument("--volume-range", dest="volumeRange", type=float, nargs=2, default=None)
    searchParser.add_argument("--all-families", dest="allFamilies", action="store_true")
    searchParser.add_argument("--sphere-metric", dest="sphereMetric", choices=("euclidean", "cosine"), default="euclidean")
    searchParser.add_argument("--skip-missing", dest="skipMissing", action="store_true", help="Omit and report unavailable Stage 2/3 caches")
    # Identical defaults to the existing Stage 2 CLI. No Stage 2 algorithm change.
    searchParser.add_argument("--theta-resolution", dest="thetaResolution", type=int, default=32)
    searchParser.add_argument("--phi-resolution", dest="phiResolution", type=int, default=64)
    searchParser.add_argument("--count-scale", dest="countScale", type=int, default=16)
    searchParser.add_argument("--linear-deflection-ratio", dest="linearDeflectionRatio", type=float, default=0.002)
    searchParser.add_argument("--angular-deflection", dest="angularDeflection", type=float, default=0.3)
    searchParser.add_argument("--leaf-size", dest="leafSize", type=int, default=32)
    for parser in (buildParser, debugParser, compareParser, searchParser):
        parser.add_argument("--samples", type=int, default=1024, help="Area samples per part, plus samples from each B-rep face")
        parser.add_argument("--samples-per-face", dest="samplesPerFace", type=int, default=8)
        parser.add_argument("--detail-linear-deflection-ratio", dest="detailLinearDeflectionRatio", type=float, default=0.001)
        parser.add_argument("--detail-angular-deflection", dest="detailAngularDeflection", type=float, default=0.2)
    for parser in (compareParser, searchParser):
        parser.add_argument("--size-mode", dest="sizeMode", choices=("physical", "shape"), default="physical")
        parser.add_argument("--alignment-starts", dest="alignmentStarts", type=int, default=4)
        parser.add_argument("--icp-iterations", dest="icpIterations", type=int, default=20)
        parser.add_argument("--trim-fraction", dest="trimFraction", type=float, default=0.9)
        parser.add_argument("--detailed-weights", dest="detailedWeights", type=float, nargs=3, default=(0.5, 0.3, 0.2),
                            metavar=("MEAN", "P95", "FACE"))


def runDetailedCommand(options):
    from src.detailedSimilarity.detailedSearch import findDetailedMatchesForStep
    from src.detailedSimilarity.geometryBuilder import buildDetailedGeometries
    from src.detailedSimilarity.geometryPipeline import buildDetailedGeometryFromStep
    from src.detailedSimilarity.geometrySimilarity import compareDetailedGeometry
    from src.detailedSimilarity.geometryStore import GeometryStore
    from src.indexing.database import PartDatabase
    from src.models.detailedGeometry import DetailedGeometryConfig, DetailedSearchConfig
    from src.models.sphereDescriptor import DescriptorConfig

    geometryConfig = DetailedGeometryConfig(sampleCount=options.samples, samplesPerFace=options.samplesPerFace,
                                            linearDeflectionRatio=options.detailLinearDeflectionRatio,
                                            angularDeflection=options.detailAngularDeflection)
    if options.command == "detailed-geometry":
        return buildDetailedGeometryFromStep(options.path, geometryConfig).summary(), 0
    if options.command == "build-detailed-geometry":
        if not Path(options.db).expanduser().is_file():
            raise ValueError("Stage 1 database does not exist; run index first")

        def reportResult(result):
            if options.verbose or result["error"]:
                print(json.dumps(result, allow_nan=False), file=sys.stderr)

        with PartDatabase(options.db) as database:
            summary = buildDetailedGeometries(GeometryStore(database), geometryConfig,
                                              retryFailed=options.retryFailed, rebuild=options.rebuild,
                                              limit=options.limit, onResult=reportResult)
        return {**summary.toDict(), "config": geometryConfig.toDict(), "configKey": geometryConfig.configKey}, (
            1 if summary.failed or summary.skippedFailed else 0)

    searchConfig = DetailedSearchConfig(sizeMode=options.sizeMode, alignmentStarts=options.alignmentStarts,
                                        icpIterations=options.icpIterations, trimFraction=options.trimFraction,
                                        meanWeight=options.detailedWeights[0], percentileWeight=options.detailedWeights[1],
                                        faceWeight=options.detailedWeights[2])
    if options.command == "compare-detailed":
        query = buildDetailedGeometryFromStep(options.path, geometryConfig)
        candidate = buildDetailedGeometryFromStep(options.candidatePath, geometryConfig)
        return {"queryPath": str(Path(options.path).resolve()),
                "candidatePath": str(Path(options.candidatePath).resolve()),
                "queryPreprocessing": query.timings, "candidatePreprocessing": candidate.timings,
                "searchConfig": searchConfig.toDict(),
                **compareDetailedGeometry(query, candidate, searchConfig)}, 0

    descriptorConfig = DescriptorConfig(thetaResolution=options.thetaResolution, phiResolution=options.phiResolution,
                                         countScale=options.countScale, linearDeflectionRatio=options.linearDeflectionRatio,
                                         angularDeflection=options.angularDeflection, leafSize=options.leafSize)
    report = findDetailedMatchesForStep(options.path, databasePath=options.db, descriptorConfig=descriptorConfig,
                                       geometryConfig=geometryConfig, searchConfig=searchConfig,
                                       stage2TopK=options.stage2TopK, topK=options.topK,
                                       dimensionTolerance=options.tolerance, volumeRange=options.volumeRange,
                                       allFamilies=options.allFamilies, sphereMetric=options.sphereMetric,
                                       skipMissing=options.skipMissing)
    return report.toDict(), 0
