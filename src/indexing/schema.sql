-- All geometric values use millimeters, square millimeters, and cubic millimeters.
CREATE TABLE ingestionFiles (
    filePath TEXT PRIMARY KEY,
    fileSize INTEGER NOT NULL,
    modifiedTimeNs INTEGER NOT NULL,
    changedTimeNs INTEGER NOT NULL,
    sourceHash TEXT,
    extractorVersion INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('processing', 'indexed', 'failed')),
    error TEXT,
    lastAttemptAt TEXT NOT NULL
) STRICT;

CREATE TABLE parts (
    partId TEXT PRIMARY KEY,
    filePath TEXT NOT NULL UNIQUE REFERENCES ingestionFiles(filePath),
    sourceHash TEXT NOT NULL,
    sizeX REAL NOT NULL CHECK (sizeX > 0),
    sizeY REAL NOT NULL CHECK (sizeY > 0),
    sizeZ REAL NOT NULL CHECK (sizeZ > 0),
    sizeMin REAL NOT NULL CHECK (sizeMin > 0),
    sizeMid REAL NOT NULL CHECK (sizeMid >= sizeMin),
    sizeMax REAL NOT NULL CHECK (sizeMax >= sizeMid),
    volume REAL NOT NULL CHECK (volume > 0),
    surfaceArea REAL NOT NULL CHECK (surfaceArea > 0),
    centerOfMassX REAL NOT NULL,
    centerOfMassY REAL NOT NULL,
    centerOfMassZ REAL NOT NULL,
    aspectXY REAL NOT NULL CHECK (aspectXY > 0),
    aspectXZ REAL NOT NULL CHECK (aspectXZ > 0),
    aspectYZ REAL NOT NULL CHECK (aspectYZ > 0),
    faceCount INTEGER NOT NULL CHECK (faceCount > 0),
    edgeCount INTEGER NOT NULL CHECK (edgeCount >= 0),
    vertexCount INTEGER NOT NULL CHECK (vertexCount >= 0),
    solidCount INTEGER NOT NULL CHECK (solidCount > 0),
    planarFaces INTEGER NOT NULL CHECK (planarFaces >= 0),
    cylindricalFaces INTEGER NOT NULL CHECK (cylindricalFaces >= 0),
    conicalFaces INTEGER NOT NULL CHECK (conicalFaces >= 0),
    otherFaces INTEGER NOT NULL CHECK (otherFaces >= 0),
    planarRatio REAL NOT NULL CHECK (planarRatio BETWEEN 0 AND 1),
    cylindricalRatio REAL NOT NULL CHECK (cylindricalRatio BETWEEN 0 AND 1),
    conicalRatio REAL NOT NULL CHECK (conicalRatio BETWEEN 0 AND 1),
    otherRatio REAL NOT NULL CHECK (otherRatio BETWEEN 0 AND 1),
    shapeFamily TEXT NOT NULL,
    units TEXT NOT NULL CHECK (units = 'mm'),
    extractorVersion INTEGER NOT NULL,
    classifierVersion INTEGER NOT NULL,
    extractedAt TEXT NOT NULL,
    CHECK (faceCount = planarFaces + cylindricalFaces + conicalFaces + otherFaces)
) STRICT;

-- SQLite seeks on family and the first size range; later bounds are residual
-- filters. Separate indexes support searches without a family and by volume.
CREATE INDEX partsByFamilyAndSize ON parts (shapeFamily, sizeMax, sizeMid, sizeMin, volume);
CREATE INDEX partsBySize ON parts (sizeMax, sizeMid, sizeMin, volume);
CREATE INDEX partsByFamilyAndVolume ON parts (shapeFamily, volume);

PRAGMA user_version = 1;
