-- An additive Stage 2 schema. Stage 1's tables and PRAGMA user_version stay intact.
CREATE TABLE descriptorSchemaInfo (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    version INTEGER NOT NULL
) STRICT;
INSERT INTO descriptorSchemaInfo VALUES (1, 1);

CREATE TABLE sphereDescriptors (
    partId TEXT NOT NULL REFERENCES parts(partId),
    configKey TEXT NOT NULL,
    sourceHash TEXT NOT NULL,
    descriptorVersion INTEGER NOT NULL,
    normalizationVersion INTEGER NOT NULL,
    thetaResolution INTEGER NOT NULL,
    phiResolution INTEGER NOT NULL,
    channelCount INTEGER NOT NULL CHECK (channelCount = 4),
    configJson TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processing', 'complete', 'failed')),
    tensorBlob BLOB,
    tensorHash TEXT,
    metadataJson TEXT,
    storageEncoding TEXT NOT NULL CHECK (storageEncoding = 'zlib-f32-le-v1'),
    error TEXT,
    updatedAt TEXT NOT NULL,
    PRIMARY KEY (partId, configKey),
    CHECK (status != 'complete' OR (tensorBlob IS NOT NULL AND tensorHash IS NOT NULL AND metadataJson IS NOT NULL))
) STRICT;
