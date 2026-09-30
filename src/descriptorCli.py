"""Add Stage 2 commands without importing NumPy or CAD during Stage 1 use."""

import json
from pathlib import Path
import sys


descriptorCommands = {"build-descriptors", "descriptor", "match"}


def addDescriptorCommands(commands) -> None:
    buildParser = commands.add_parser("build-descriptors", help="Precompute missing/stale spherical descriptors")
    buildParser.add_argument("--db", default="parts.sqlite3")
    buildParser.add_argument("--retry-failed", dest="retryFailed", action="store_true")
    buildParser.add_argument("--rebuild", action="store_true", help="Explicitly regenerate existing descriptors for this configuration")
    buildParser.add_argument("--limit", type=int, default=None, help="Maximum new/retried parts to process")
    buildParser.add_argument("--verbose", action="store_true", help="Emit per-part results and timing JSON to stderr")

    debugParser = commands.add_parser("descriptor", help="Describe one STEP, print diagnostics, optionally save channels")
    debugParser.add_argument("path")
    debugParser.add_argument("--save-images", dest="saveImages", default=None, metavar="DIRECTORY")
    debugParser.add_argument("--save-tensor", dest="saveTensor", default=None, metavar="FILE.npy")

    matchParser = commands.add_parser("match", help="Stage 1 filtering followed by cached Stage 2 ranking")
    matchParser.add_argument("path")
    matchParser.add_argument("--db", default="parts.sqlite3")
    matchParser.add_argument("--top-k", dest="topK", type=int, default=50)
    matchParser.add_argument("--tolerance", type=float, default=0.10, help="Stage 1 relative dimension tolerance")
    matchParser.add_argument("--volume-range", dest="volumeRange", type=float, nargs=2, default=None)
    matchParser.add_argument("--all-families", dest="allFamilies", action="store_true")
    matchParser.add_argument("--metric", choices=("euclidean", "cosine"), default="euclidean")
    matchParser.add_argument("--weights", type=float, nargs=4, default=(1.0, 1.0, 0.5, 1.5), metavar=("FIRST", "LAST", "COUNT", "THICKNESS"))
    matchParser.add_argument("--skip-missing", dest="skipMissing", action="store_true", help="Explicitly omit unavailable descriptors and report their IDs")

    for parser in (buildParser, debugParser, matchParser):
        parser.add_argument("--theta-resolution", dest="thetaResolution", type=int, default=32)
        parser.add_argument("--phi-resolution", dest="phiResolution", type=int, default=64)
        parser.add_argument("--count-scale", dest="countScale", type=int, default=16)
        parser.add_argument("--linear-deflection-ratio", dest="linearDeflectionRatio", type=float, default=0.002)
        parser.add_argument("--angular-deflection", dest="angularDeflection", type=float, default=0.3, help="Meshing angular deflection in radians")
        parser.add_argument("--leaf-size", dest="leafSize", type=int, default=32)


def runDescriptorCommand(options) -> tuple[dict, int]:
    # Lazy imports keep the existing metadata CLI usable without NumPy/pythonOCC.
    from src.descriptors.descriptorBuilder import buildDescriptors
    from src.descriptors.descriptorDebug import saveChannelImages
    from src.descriptors.descriptorPipeline import buildDescriptorFromStep
    from src.descriptors.descriptorStore import DescriptorStore
    from src.indexing.candidateSearch import findMatchesForStep
    from src.indexing.database import PartDatabase
    from src.models.sphereDescriptor import DescriptorConfig

    config = DescriptorConfig(
        thetaResolution=options.thetaResolution, phiResolution=options.phiResolution,
        countScale=options.countScale, linearDeflectionRatio=options.linearDeflectionRatio,
        angularDeflection=options.angularDeflection, leafSize=options.leafSize,
    )
    if options.command == "descriptor":
        descriptor = buildDescriptorFromStep(options.path, config)
        result = descriptor.summary()
        if options.saveImages:
            result["channelImages"] = saveChannelImages(descriptor, options.saveImages)
        if options.saveTensor:
            import numpy as np
            tensorPath = Path(options.saveTensor).expanduser().resolve()
            tensorPath.parent.mkdir(parents=True, exist_ok=True)
            with tensorPath.open("wb") as tensorFile:
                np.save(tensorFile, descriptor.tensor, allow_pickle=False)
            result["tensorPath"] = str(tensorPath)
        return result, 0
    if options.command == "match":
        report = findMatchesForStep(
            options.path, databasePath=options.db, config=config,
            dimensionTolerance=options.tolerance, volumeRange=options.volumeRange,
            allFamilies=options.allFamilies, topK=options.topK, weights=options.weights,
            metric=options.metric, skipMissing=options.skipMissing,
        )
        return report.toDict(), 0

    def reportResult(result):
        if options.verbose or result["error"]:
            print(json.dumps(result, allow_nan=False), file=sys.stderr)

    if not Path(options.db).expanduser().is_file():
        raise ValueError("Stage 1 database does not exist; run index first")
    with PartDatabase(options.db) as database:
        store = DescriptorStore(database)
        summary = buildDescriptors(
            store, config, retryFailed=options.retryFailed, rebuild=options.rebuild,
            limit=options.limit, onResult=reportResult,
        )
    return {**summary.toDict(), "config": config.toDict(), "configKey": config.configKey}, (
        1 if summary.failed or summary.skippedFailed else 0
    )
