"""Serializable records; this module has no OpenCascade dependencies."""

from dataclasses import asdict, dataclass


# Increment these independently when extraction semantics or family rules change.
extractorVersion = 1
classifierVersion = 1

shapeFamilies = (
    "plate-like",
    "block-like",
    "shaft/turned-like",
    "cylindrical/ring-like",
    "general/unknown",
)


@dataclass(frozen=True, kw_only=True)
class GeometryProperties:
    """Lengths and centroid in mm, area in mm², volume in mm³."""

    sizeX: float
    sizeY: float
    sizeZ: float
    sizeMin: float
    sizeMid: float
    sizeMax: float
    volume: float
    surfaceArea: float
    centerOfMassX: float
    centerOfMassY: float
    centerOfMassZ: float
    aspectXY: float
    aspectXZ: float
    aspectYZ: float
    faceCount: int
    edgeCount: int
    vertexCount: int
    solidCount: int
    planarFaces: int
    cylindricalFaces: int
    conicalFaces: int
    otherFaces: int
    planarRatio: float
    cylindricalRatio: float
    conicalRatio: float
    otherRatio: float
    units: str = "mm"

    def toDict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class PartMetadata(GeometryProperties):
    """One record per canonical source path, ready for SQLite or JSON."""

    partId: str
    filePath: str
    sourceHash: str
    shapeFamily: str
    extractedAt: str
    extractorVersion: int = extractorVersion
    classifierVersion: int = classifierVersion
