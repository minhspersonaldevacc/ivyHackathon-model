"""SQLite persistence and candidate filtering; no CAD imports or file reads."""

from dataclasses import fields
from datetime import datetime, timezone
from math import isfinite, nextafter
from pathlib import Path
import sqlite3
from typing import Sequence

from src.errors import DatabaseError
from src.models.partMetadata import PartMetadata, extractorVersion, shapeFamilies


schemaVersion = 1
metadataColumns = tuple(field.name for field in fields(PartMetadata))


class PartDatabase:
    """Use as a context manager. Read-only mode never creates an empty index."""

    def __init__(self, databasePath: str | Path = "parts.sqlite3", *, readOnly: bool = False):
        self.databasePath = Path(databasePath).expanduser().resolve()
        if readOnly:
            self.connection = sqlite3.connect(
                self.databasePath.as_uri() + "?mode=ro", uri=True, timeout=30
            )
        else:
            self.databasePath.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.databasePath, timeout=30)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys = ON")
            actualVersion = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if actualVersion == 0 and not readOnly:
                existingTables = self.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                if existingTables:
                    raise DatabaseError("Refusing to initialize a nonempty database with an unknown schema")
                schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
                self.connection.executescript("BEGIN IMMEDIATE;\n" + schema + "\nCOMMIT;")
                actualVersion = schemaVersion
            if actualVersion != schemaVersion:
                raise DatabaseError(
                    f"Unsupported database schema {actualVersion}; expected {schemaVersion}"
                )
            if not readOnly:
                self.connection.execute("PRAGMA journal_mode = WAL")
        except Exception:
            self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exceptionType, exception, traceback):
        self.close()

    def close(self) -> None:
        self.connection.close()

    def getFileState(self, filePath: str) -> dict | None:
        row = self.connection.execute(
            "SELECT * FROM ingestionFiles WHERE filePath = ?", (filePath,)
        ).fetchone()
        return dict(row) if row is not None else None

    def getPart(self, filePath: str, *, includeInactive: bool = False) -> PartMetadata | None:
        query = """
            SELECT p.* FROM parts AS p
            JOIN ingestionFiles AS f ON f.filePath = p.filePath
            WHERE p.filePath = ?
        """
        if not includeInactive:
            query += " AND f.status = 'indexed'"
        row = self.connection.execute(query, (filePath,)).fetchone()
        return PartMetadata(**dict(row)) if row is not None else None

    def _writeFileState(
        self, filePath: str, fingerprint: dict, status: str,
        sourceHash: str | None = None, error: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO ingestionFiles (
                filePath, fileSize, modifiedTimeNs, changedTimeNs, sourceHash,
                extractorVersion, status, error, lastAttemptAt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filePath) DO UPDATE SET
                fileSize = excluded.fileSize,
                modifiedTimeNs = excluded.modifiedTimeNs,
                changedTimeNs = excluded.changedTimeNs,
                sourceHash = excluded.sourceHash,
                extractorVersion = excluded.extractorVersion,
                status = excluded.status,
                error = excluded.error,
                lastAttemptAt = excluded.lastAttemptAt
            """,
            (
                filePath, fingerprint["fileSize"], fingerprint["modifiedTimeNs"],
                fingerprint["changedTimeNs"], sourceHash, extractorVersion, status,
                error, datetime.now(timezone.utc).isoformat(),
            ),
        )

    def beginFile(self, filePath: str, fingerprint: dict) -> None:
        """Hide any old geometry while a changed source is being processed.

        If ingestion is interrupted, the next run retries this processing entry.
        """
        with self.connection:
            self._writeFileState(filePath, fingerprint, "processing")

    def recordFailure(
        self, filePath: str, fingerprint: dict, error: str, sourceHash: str | None = None,
    ) -> None:
        """Keep failure state and exclude the old record from candidate search."""
        with self.connection:
            self._writeFileState(filePath, fingerprint, "failed", sourceHash, error)

    def storePart(self, metadata: PartMetadata, fingerprint: dict) -> None:
        """Commit geometry and successful file state atomically, retaining partId."""
        columnSql = ", ".join(metadataColumns)
        valueSql = ", ".join(f":{column}" for column in metadataColumns)
        updateSql = ", ".join(
            f"{column} = excluded.{column}"
            for column in metadataColumns if column not in {"partId", "filePath"}
        )
        with self.connection:
            self._writeFileState(metadata.filePath, fingerprint, "indexed", metadata.sourceHash)
            self.connection.execute(
                f"INSERT INTO parts ({columnSql}) VALUES ({valueSql}) "
                f"ON CONFLICT(filePath) DO UPDATE SET {updateSql}",
                metadata.toDict(),
            )

    def findCandidates(
        self,
        shapeFamily: str | None,
        dimensions: Sequence[float],
        dimensionTolerance: float,
        volumeRange: tuple[float, float] | None = None,
        *,
        limit: int | None = None,
    ) -> list[PartMetadata]:
        """Filter stored rows using inclusive, relative dimension tolerances.

        dimensions may be in any axis order. A tolerance of 0.10 means ±10% of
        EACH sorted query dimension. volumeRange is an inclusive mm³ interval.
        None for shapeFamily searches all families. All matches are returned by
        default. An explicit limit truncates by partId, not by similarity rank.
        """
        query, parameters = self.buildCandidateQuery(
            shapeFamily, dimensions, dimensionTolerance, volumeRange, limit=limit
        )
        return [PartMetadata(**dict(row)) for row in self.connection.execute(query, parameters)]

    @staticmethod
    def buildCandidateQuery(
        shapeFamily: str | None,
        dimensions: Sequence[float],
        dimensionTolerance: float,
        volumeRange: tuple[float, float] | None = None,
        *,
        limit: int | None = None,
    ) -> tuple[str, list]:
        """Expose the SQL for EXPLAIN QUERY PLAN without duplicating filtering."""
        if shapeFamily is not None and shapeFamily not in shapeFamilies:
            raise ValueError(f"Unknown shapeFamily: {shapeFamily}")
        try:
            sortedDimensions = sorted(float(value) for value in dimensions)
            tolerance = float(dimensionTolerance)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("dimensions and dimensionTolerance must be numeric") from error
        if len(sortedDimensions) != 3 or not all(
            isfinite(value) and value > 0 for value in sortedDimensions
        ):
            raise ValueError("dimensions must contain exactly three positive, finite lengths in mm")
        if not isfinite(tolerance) or not 0 <= tolerance <= 1:
            raise ValueError("dimensionTolerance must be a fraction from 0 to 1 (0.10 means 10%)")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("limit must be a positive integer or None")

        clauses = ["f.status = 'indexed'"]
        parameters = []
        if shapeFamily is not None:
            clauses.append("p.shapeFamily = ?")
            parameters.append(shapeFamily)
        for column, dimension in zip(("sizeMin", "sizeMid", "sizeMax"), sortedDimensions):
            lower = dimension * (1 - tolerance)
            upper = dimension * (1 + tolerance)
            if not isfinite(upper):
                raise ValueError("Dimension tolerance produces a non-finite upper bound")
            # One ULP outwards protects inclusive decimal boundaries from rounding.
            # Zero tolerance keeps exact stored-value equality.
            if tolerance > 0:
                lower = nextafter(lower, float("-inf"))
                upper = nextafter(upper, float("inf"))
            clauses.append(f"p.{column} BETWEEN ? AND ?")
            parameters.extend((lower, upper))
        if volumeRange is not None:
            try:
                minVolume, maxVolume = (float(value) for value in volumeRange)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("volumeRange must be a (minimum, maximum) pair") from error
            if not all(isfinite(value) for value in (minVolume, maxVolume)) or not 0 <= minVolume <= maxVolume:
                raise ValueError("volumeRange must contain finite values with 0 <= minimum <= maximum")
            clauses.append("p.volume BETWEEN ? AND ?")
            parameters.extend((minVolume, maxVolume))

        query = (
            "SELECT p.* FROM parts AS p "
            "JOIN ingestionFiles AS f ON f.filePath = p.filePath "
            "WHERE " + " AND ".join(clauses) + " ORDER BY p.partId"
        )
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        return query, parameters

    def optimize(self) -> None:
        """Refresh planner statistics after an ingestion batch."""
        self.connection.execute("PRAGMA optimize")


def findCandidates(
    shapeFamily: str | None,
    dimensions: Sequence[float],
    dimensionTolerance: float,
    volumeRange: tuple[float, float] | None = None,
    *,
    databasePath: str | Path = "parts.sqlite3",
    limit: int | None = None,
) -> list[PartMetadata]:
    """Convenience API matching the Stage 1 filtering interface."""
    with PartDatabase(databasePath, readOnly=True) as database:
        return database.findCandidates(
            shapeFamily, dimensions, dimensionTolerance, volumeRange, limit=limit
        )
