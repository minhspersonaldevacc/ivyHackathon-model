"""Compressed float32 tensors in SQLite, keyed by source and full configuration."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zlib

import numpy as np

from src.errors import DescriptorError
from src.indexing.database import PartDatabase
from src.models.partMetadata import PartMetadata
from src.models.sphereDescriptor import DescriptorConfig, SphereDescriptor


class DescriptorStore:
    """Add Stage 2 tables to an existing PartDatabase, or use initialize=False to read."""

    def __init__(self, database: PartDatabase, *, initialize: bool = True):
        self.database = database
        self.connection = database.connection
        exists = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'descriptorSchemaInfo'"
        ).fetchone()
        if exists is None:
            if not initialize:
                raise DescriptorError("No Stage 2 descriptor store; run build-descriptors first")
            schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
            self.connection.executescript("BEGIN IMMEDIATE;\n" + schema + "\nCOMMIT;")
        version = self.connection.execute("SELECT version FROM descriptorSchemaInfo WHERE singleton = 1").fetchone()
        if version is None or version[0] != 1:
            raise DescriptorError("Unsupported Stage 2 descriptor schema version")

    def iterPartStates(self, config: DescriptorConfig, batchSize: int = 256):
        """Page over indexed parts without holding a cursor open during writes."""
        lastId = ""
        while True:
            rows = self.connection.execute(
                """
                SELECT p.*, d.status AS descriptorStatus, d.sourceHash AS descriptorSourceHash,
                       d.error AS descriptorError
                FROM parts AS p JOIN ingestionFiles AS f ON f.filePath = p.filePath
                LEFT JOIN sphereDescriptors AS d ON d.partId = p.partId AND d.configKey = ?
                WHERE f.status = 'indexed' AND p.partId > ? ORDER BY p.partId LIMIT ?
                """, (config.configKey, lastId, batchSize),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                record = dict(row)
                status = record.pop("descriptorStatus")
                sourceHash = record.pop("descriptorSourceHash")
                error = record.pop("descriptorError")
                yield PartMetadata(**record), status, sourceHash, error
            lastId = rows[-1]["partId"]

    def _writeState(self, part: PartMetadata, config: DescriptorConfig, status: str, *, error=None, descriptor=None):
        tensorBlob = tensorHash = metadataJson = None
        if descriptor is not None:
            rawBytes = descriptor.tensor.astype("<f4", copy=False).tobytes(order="C")
            tensorBlob = zlib.compress(rawBytes)
            tensorHash = hashlib.sha256(rawBytes).hexdigest()
            metadataJson = json.dumps(descriptor.metadata(), sort_keys=True, allow_nan=False)
        values = (
            part.partId, config.configKey, part.sourceHash, config.descriptorVersion,
            config.normalizationVersion, config.thetaResolution, config.phiResolution, 4,
            json.dumps(config.toDict(), sort_keys=True, allow_nan=False), status,
            tensorBlob, tensorHash, metadataJson, "zlib-f32-le-v1", error,
            datetime.now(timezone.utc).isoformat(),
        )
        self.connection.execute(
            """
            INSERT INTO sphereDescriptors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(partId, configKey) DO UPDATE SET
                sourceHash = excluded.sourceHash, status = excluded.status,
                tensorBlob = excluded.tensorBlob, tensorHash = excluded.tensorHash,
                metadataJson = excluded.metadataJson, error = excluded.error,
                updatedAt = excluded.updatedAt
            """, values,
        )

    def beginPart(self, part: PartMetadata, config: DescriptorConfig) -> None:
        with self.connection:
            self._writeState(part, config, "processing")

    def recordFailure(self, part: PartMetadata, config: DescriptorConfig, error: str) -> None:
        with self.connection:
            self._writeState(part, config, "failed", error=error)

    def storeDescriptor(self, part: PartMetadata, descriptor: SphereDescriptor) -> None:
        if descriptor.sourceHash != part.sourceHash:
            raise DescriptorError("Descriptor source hash does not match the indexed part")
        with self.connection:
            # Avoid publishing a descriptor if Stage 1 changed while it was built.
            current = self.connection.execute(
                """SELECT p.sourceHash FROM parts p JOIN ingestionFiles f ON f.filePath = p.filePath
                   WHERE p.partId = ? AND f.status = 'indexed'""", (part.partId,),
            ).fetchone()
            if current is None or current[0] != part.sourceHash:
                raise DescriptorError("Indexed part changed during descriptor generation; retry")
            self._writeState(part, descriptor.config, "complete", descriptor=descriptor)

    @staticmethod
    def _decode(row, config: DescriptorConfig) -> SphereDescriptor:
        try:
            if row["storageEncoding"] != "zlib-f32-le-v1" or row["configKey"] != config.configKey:
                raise ValueError("Unsupported encoding or incompatible configuration")
            if json.loads(row["configJson"]) != config.toDict():
                raise ValueError("Stored configuration does not match its key")
            expectedHeader = (config.descriptorVersion, config.normalizationVersion,
                              config.thetaResolution, config.phiResolution, 4)
            actualHeader = tuple(row[name] for name in (
                "descriptorVersion", "normalizationVersion", "thetaResolution", "phiResolution", "channelCount"
            ))
            if actualHeader != expectedHeader:
                raise ValueError("Stored descriptor header is inconsistent")
            expectedSize = int(np.prod(config.tensorShape)) * 4
            decompressor = zlib.decompressobj()
            rawBytes = decompressor.decompress(row["tensorBlob"], expectedSize + 1)
            if len(rawBytes) != expectedSize or not decompressor.eof or decompressor.unused_data:
                raise ValueError("Unexpected descriptor byte count or incomplete compressed data")
            if hashlib.sha256(rawBytes).hexdigest() != row["tensorHash"]:
                raise ValueError("Descriptor checksum mismatch")
            tensor = np.frombuffer(rawBytes, dtype="<f4").reshape(config.tensorShape)
            metadata = json.loads(row["metadataJson"])
            if metadata["sourceHash"] != row["sourceHash"] or metadata["configKey"] != config.configKey:
                raise ValueError("Stored descriptor provenance is inconsistent")
            return SphereDescriptor(
                tensor, config, metadata["normalizationData"], metadata["meshVertexCount"],
                metadata["meshTriangleCount"], metadata["timings"], metadata["diagnostics"], row["sourceHash"],
            )
        except (ValueError, TypeError, KeyError, zlib.error) as error:
            raise DescriptorError(f"Invalid stored descriptor for {row['partId']}: {error}") from error

    def loadDescriptors(self, partIds: list[str], config: DescriptorConfig) -> dict[str, SphereDescriptor]:
        """Load only current, complete descriptors. No source filesystem access."""
        if len(partIds) > 400:
            raise ValueError("Load at most 400 descriptors per batch")
        if not partIds:
            return {}
        placeholders = ",".join("?" for _ in partIds)
        rows = self.connection.execute(
            f"""
            SELECT d.* FROM sphereDescriptors d
            JOIN parts p ON p.partId = d.partId AND p.sourceHash = d.sourceHash
            JOIN ingestionFiles f ON f.filePath = p.filePath
            WHERE d.configKey = ? AND d.status = 'complete' AND f.status = 'indexed'
                  AND d.partId IN ({placeholders})
            """, [config.configKey, *partIds],
        ).fetchall()
        return {row["partId"]: self._decode(row, config) for row in rows}
