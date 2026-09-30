-- Additive Stage 3 cache. Stage 1 and Stage 2 schema versions are unchanged.
CREATE TABLE detailedSchemaInfo (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL
) STRICT;
INSERT INTO detailedSchemaInfo VALUES (1, 1);

CREATE TABLE detailedGeometries (
    partId TEXT NOT NULL REFERENCES parts(partId),
    configKey TEXT NOT NULL,
    sourceHash TEXT NOT NULL,
    geometryVersion INTEGER NOT NULL,
    normalizationVersion INTEGER NOT NULL,
    configJson TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processing', 'complete', 'failed')),
    geometryBlob BLOB,
    geometryHash TEXT,
    metadataJson TEXT,
    storageEncoding TEXT NOT NULL CHECK (storageEncoding = 'npz-v1'),
    error TEXT,
    updatedAt TEXT NOT NULL,
    PRIMARY KEY (partId, configKey),
    CHECK (status != 'complete' OR (geometryBlob IS NOT NULL AND geometryHash IS NOT NULL AND metadataJson IS NOT NULL))
) STRICT;
