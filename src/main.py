"""Command line entry point for indexing, inspecting, and candidate filtering."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
import sqlite3
import sys

from src.errors import CadError, DatabaseError, DependencyError
from src.descriptorCli import addDescriptorCommands, descriptorCommands, runDescriptorCommand
from src.indexing.database import PartDatabase
from src.indexing.partIndexer import IndexResult, indexPath, inspectPart
from src.models.partMetadata import shapeFamilies


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CAD metadata index and spherical descriptor search.")
    commands = parser.add_subparsers(dest="command", required=True)

    indexParser = commands.add_parser("index", help="Index a STEP file or recursively scan a directory")
    indexParser.add_argument("path")
    indexParser.add_argument("--db", default="parts.sqlite3", help="SQLite index path (default: parts.sqlite3)")
    indexParser.add_argument("--retry-failed", dest="retryFailed", action="store_true", help="Retry unchanged files with previous failures")
    indexParser.add_argument("--verify-content", dest="verifyContent", action="store_true", help="Hash even unchanged files; reuse geometry if bytes match")
    indexParser.add_argument("--verbose", action="store_true", help="Report successful and skipped files too")

    inspectParser = commands.add_parser("inspect", help="Print metadata as JSON without writing a database")
    inspectParser.add_argument("path")

    findParser = commands.add_parser("find", help="Filter stored metadata without loading any STEP files")
    findParser.add_argument("--db", default="parts.sqlite3")
    findParser.add_argument("--family", choices=shapeFamilies, default=None, help="Omit to search every family")
    findParser.add_argument("--dimensions", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="Query bounding box lengths in mm; any axis order")
    findParser.add_argument("--tolerance", type=float, default=0.10, help="Relative tolerance from 0 to 1 (default: 0.10 = 10%%)")
    findParser.add_argument("--volume-range", dest="volumeRange", type=float, nargs=2, metavar=("MIN", "MAX"), help="Inclusive volume range in mm³")
    findParser.add_argument("--limit", type=int, default=None, help="Optional truncation by partId; results are not similarity ranked")
    addDescriptorCommands(commands)
    return parser


def printJson(value) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


@contextmanager
def nativeMessagesToStderr():
    """Redirect native stdout while the serial CLI calls OpenCascade.

    Some STEP parser diagnostics bypass the default OCCT messenger. Python's
    redirect_stdout cannot capture those C++ writes, so redirect file descriptor
    1 temporarily. This is process-wide and is used only by the serial CLI.
    """
    sys.stdout.flush()
    savedDescriptor = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(savedDescriptor, 1)
        os.close(savedDescriptor)


def main(arguments: list[str] | None = None) -> int:
    options = buildParser().parse_args(arguments)
    try:
        if options.command in descriptorCommands:
            try:
                with nativeMessagesToStderr():
                    result, exitCode = runDescriptorCommand(options)
            except ImportError as error:
                raise DependencyError("Activate the environment from environment.yml: " + str(error)) from error
            printJson(result)
            return exitCode
        if options.command == "inspect":
            with nativeMessagesToStderr():
                metadata = inspectPart(options.path)
            printJson(metadata.toDict())
            return 0
        if options.command == "find":
            with PartDatabase(options.db, readOnly=True) as database:
                candidates = database.findCandidates(
                    options.family, options.dimensions, options.tolerance,
                    options.volumeRange, limit=options.limit,
                )
            printJson([candidate.toDict() for candidate in candidates])
            return 0

        def reportResult(result: IndexResult):
            if result.error is not None or options.verbose:
                message = f"{result.status}: {result.filePath}"
                if result.error is not None:
                    message += f": {result.error}"
                print(message, file=sys.stderr)

        with PartDatabase(options.db) as database, nativeMessagesToStderr():
            summary = indexPath(
                options.path, database,
                retryFailed=options.retryFailed, verifyContent=options.verifyContent,
                onResult=reportResult,
            )
        printJson(asdict(summary))
        return 1 if summary.failed or summary.skippedFailed else 0
    except (CadError, DatabaseError, DependencyError, OSError, sqlite3.Error, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed records are saved. Rerun index to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
