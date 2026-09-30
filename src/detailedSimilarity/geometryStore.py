"""Versioned, checksummed NumPy blobs in the existing SQLite index."""

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

from src.errors import DetailedSimilarityError
from src.models.detailedGeometry import DetailedGeometry, FaceFeature
from src.models.mesh import Mesh
from src.models.partMetadata import PartMetadata


maxGeometryBytes = 256 * 1024 ** 2


class GeometryStore:
    def __init__(self, database, *, initialize=True):
        self.database, self.connection = database, database.connection
        exists = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'detailedSchemaInfo'"
        ).fetchone()
        if exists is None:
            if not initialize:
                raise DetailedSimilarityError("No Stage 3 geometry cache; run build-detailed-geometry first")
            schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
            self.connection.executescript("BEGIN IMMEDIATE;\n" + schema + "\nCOMMIT;")
        version = self.connection.execute("SELECT version FROM detailedSchemaInfo WHERE singleton = 1").fetchone()
        if version is None or version[0] != 1:
            raise DetailedSimilarityError("Unsupported Stage 3 cache schema version")

    def iterPartStates(self, config, batchSize=256):
        if type(batchSize) is not int or not 1 <= batchSize <= 400:
            raise ValueError("batchSize must be an integer from 1 to 400")
        lastId = ""
        while True:
            rows = self.connection.execute(
                """SELECT p.*, g.status AS geometryStatus, g.sourceHash AS geometrySourceHash,
                          g.error AS geometryError
                   FROM parts p JOIN ingestionFiles f ON f.filePath = p.filePath
                   LEFT JOIN detailedGeometries g ON g.partId = p.partId AND g.configKey = ?
                   WHERE f.status = 'indexed' AND p.partId > ? ORDER BY p.partId LIMIT ?""",
                (config.configKey, lastId, batchSize),
            ).fetchall()
            if not rows:
                return
            for row in rows:
                values = dict(row)
                status, sourceHash, error = (values.pop(name) for name in
                                            ("geometryStatus", "geometrySourceHash", "geometryError"))
                yield PartMetadata(**values), status, sourceHash, error
            lastId = rows[-1]["partId"]

    def _writeState(self, part, config, status, *, geometry=None, error=None):
        blob = checksum = metadataJson = None
        if geometry is not None:
            arrays = {"vertices": geometry.mesh.vertices, "triangles": geometry.mesh.triangles,
                      "bodyIds": geometry.mesh.triangleBodyIds, "samplePoints": geometry.samplePoints}
            if sum(array.nbytes for array in arrays.values()) > maxGeometryBytes - 4096:
                raise DetailedSimilarityError("Detailed geometry exceeds the 256 MiB cache limit")
            buffer = BytesIO()
            np.savez_compressed(buffer, **arrays)
            blob = buffer.getvalue()
            checksum = hashlib.sha256(blob).hexdigest()
            metadataJson = json.dumps(geometry.metadata(), sort_keys=True, allow_nan=False)
        self.connection.execute(
            """INSERT INTO detailedGeometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(partId, configKey) DO UPDATE SET
                   sourceHash = excluded.sourceHash, status = excluded.status,
                   geometryBlob = excluded.geometryBlob, geometryHash = excluded.geometryHash,
                   metadataJson = excluded.metadataJson, error = excluded.error, updatedAt = excluded.updatedAt""",
            (part.partId, config.configKey, part.sourceHash, config.geometryVersion, config.normalizationVersion,
             json.dumps(config.toDict(), sort_keys=True), status, blob, checksum, metadataJson, "npz-v1", error,
             datetime.now(timezone.utc).isoformat()),
        )

    def beginPart(self, part, config):
        with self.connection:
            self._writeState(part, config, "processing")

    def recordFailure(self, part, config, error):
        with self.connection:
            self._writeState(part, config, "failed", error=error)

    def storeGeometry(self, part, geometry):
        if geometry.sourceHash != part.sourceHash:
            raise DetailedSimilarityError("Detailed geometry source hash does not match Stage 1")
        with self.connection:
            current = self.connection.execute(
                """SELECT p.sourceHash FROM parts p JOIN ingestionFiles f ON f.filePath = p.filePath
                   WHERE p.partId = ? AND f.status = 'indexed'""", (part.partId,),
            ).fetchone()
            if current is None or current[0] != part.sourceHash:
                raise DetailedSimilarityError("Stage 1 part changed during detailed preprocessing")
            self._writeState(part, geometry.config, "complete", geometry=geometry)

    @staticmethod
    def _decode(row, config):
        try:
            if row["storageEncoding"] != "npz-v1" or row["configKey"] != config.configKey:
                raise ValueError("Unsupported encoding/configuration")
            if json.loads(row["configJson"]) != config.toDict():
                raise ValueError("Configuration does not match cache key")
            if (row["geometryVersion"], row["normalizationVersion"]) != (config.geometryVersion, config.normalizationVersion):
                raise ValueError("Inconsistent cache version")
            blob = row["geometryBlob"]
            if len(blob) > maxGeometryBytes or hashlib.sha256(blob).hexdigest() != row["geometryHash"]:
                raise ValueError("Geometry checksum/size mismatch")
            with ZipFile(BytesIO(blob)) as archive:
                if set(archive.namelist()) != {"vertices.npy", "triangles.npy", "bodyIds.npy", "samplePoints.npy"}:
                    raise ValueError("Unexpected geometry arrays")
                if sum(info.file_size for info in archive.infolist()) > maxGeometryBytes:
                    raise ValueError("Decoded geometry exceeds size limit")
            metadata = json.loads(row["metadataJson"])
            if metadata["sourceHash"] != row["sourceHash"] or metadata["configKey"] != config.configKey:
                raise ValueError("Inconsistent geometry provenance")
            if metadata["config"] != config.toDict():
                raise ValueError("Inconsistent metadata configuration")
            faces = tuple(FaceFeature(**{**face, "center": tuple(face["center"]),
                                        "direction": tuple(face["direction"]),
                                        "adjacentFaces": tuple(face["adjacentFaces"])}) for face in metadata["faces"])
            with np.load(BytesIO(blob), allow_pickle=False) as arrays:
                mesh = Mesh(arrays["vertices"], arrays["triangles"], arrays["bodyIds"])
                return DetailedGeometry(mesh, arrays["samplePoints"], faces, metadata["normalizationData"],
                                        config, row["sourceHash"], metadata["timings"], metadata["diagnostics"])
        except (ValueError, TypeError, KeyError, BadZipFile, OSError, EOFError) as error:
            raise DetailedSimilarityError(f"Invalid detailed geometry for {row['partId']}: {error}") from error

    def loadGeometries(self, partIds, config):
        """Read only requested active/source-compatible artifacts; no STEP stat/load."""
        if len(partIds) > 400:
            raise ValueError("Load at most 400 detailed geometries per batch")
        if not partIds:
            return {}
        placeholders = ",".join("?" for _ in partIds)
        rows = self.connection.execute(
            f"""SELECT g.*, p.filePath FROM detailedGeometries g
                JOIN parts p ON p.partId = g.partId AND p.sourceHash = g.sourceHash
                JOIN ingestionFiles f ON f.filePath = p.filePath
                WHERE g.configKey = ? AND g.status = 'complete' AND f.status = 'indexed'
                      AND g.partId IN ({placeholders})""", [config.configKey, *partIds],
        ).fetchall()
        return {row["partId"]: (self._decode(row, config), row["filePath"]) for row in rows}
