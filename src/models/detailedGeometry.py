"""Portable Stage 3 geometry and explicit preprocessing/comparison settings."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import numpy as np

from src.models.mesh import Mesh
from src.models.sphereDescriptor import DescriptorConfig


geometryVersion = 1


@dataclass(frozen=True)
class DetailedGeometryConfig:
    sampleCount: int = 1024
    samplesPerFace: int = 8
    linearDeflectionRatio: float = 0.001
    angularDeflection: float = 0.2
    geometryVersion: int = geometryVersion
    normalizationVersion: int = 1

    def __post_init__(self):
        if type(self.sampleCount) is not int or not 32 <= self.sampleCount <= 100000:
            raise ValueError("sampleCount must be an integer from 32 to 100000")
        if type(self.samplesPerFace) is not int or not 1 <= self.samplesPerFace <= 32:
            raise ValueError("samplesPerFace must be an integer from 1 to 32")
        if self.geometryVersion != geometryVersion or self.normalizationVersion != 1:
            raise ValueError("Unsupported detailed geometry/normalization version")
        self.meshConfig  # Reuse Stage 2 validation and meshing semantics.

    @property
    def meshConfig(self):
        return DescriptorConfig(linearDeflectionRatio=self.linearDeflectionRatio,
                                angularDeflection=self.angularDeflection)

    def toDict(self):
        return asdict(self)

    @property
    def configKey(self):
        return hashlib.sha256(json.dumps(self.toDict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class DetailedSearchConfig:
    sizeMode: str = "physical"
    alignmentStarts: int = 4
    icpIterations: int = 20
    trimFraction: float = 0.9
    meanWeight: float = 0.5
    percentileWeight: float = 0.3
    faceWeight: float = 0.2

    def __post_init__(self):
        if self.sizeMode not in {"physical", "shape"}:
            raise ValueError("sizeMode must be 'physical' or 'shape'")
        if type(self.alignmentStarts) is not int or not 1 <= self.alignmentStarts <= 24:
            raise ValueError("alignmentStarts must be an integer from 1 to 24")
        if type(self.icpIterations) is not int or not 0 <= self.icpIterations <= 200:
            raise ValueError("icpIterations must be an integer from 0 to 200")
        if not math.isfinite(self.trimFraction) or not 0.5 <= self.trimFraction <= 1:
            raise ValueError("trimFraction must be between 0.5 and 1")
        weights = (self.meanWeight, self.percentileWeight, self.faceWeight)
        if not all(math.isfinite(w) and 0 <= w <= 100 for w in weights) or sum(weights) <= 0:
            raise ValueError("Detailed weights must be finite, nonnegative, and have a positive sum")

    def toDict(self):
        return asdict(self)


@dataclass(frozen=True)
class FaceFeature:
    surfaceType: str
    area: float
    center: tuple[float, float, float]
    direction: tuple[float, float, float]
    radius: float = 0.0
    angle: float = 0.0
    adjacentFaces: tuple[int, ...] = ()


@dataclass(frozen=True)
class DetailedGeometry:
    mesh: Mesh
    samplePoints: np.ndarray
    faces: tuple[FaceFeature, ...]
    normalizationData: dict
    config: DetailedGeometryConfig = field(default_factory=DetailedGeometryConfig)
    sourceHash: str | None = None
    timings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        points = np.array(self.samplePoints, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < self.config.sampleCount:
            raise ValueError("Detailed samples must have shape (N, 3) and include the area samples")
        if not np.isfinite(points).all():
            raise ValueError("Detailed samples must be finite")
        scale = self.normalizationData.get("radiusScale", 0)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid detailed geometry scale")
        if not self.faces:
            raise ValueError("Detailed geometry needs B-rep face features")
        for face in self.faces:
            values = (*face.center, *face.direction, face.area, face.radius, face.angle)
            if len(face.center) != 3 or len(face.direction) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid face properties")
            if face.area <= 0 or face.radius < 0:
                raise ValueError("Invalid face area or radius")
            if any(type(i) is not int or not 0 <= i < len(self.faces) for i in face.adjacentFaces):
                raise ValueError("Invalid face adjacency")
        points.setflags(write=False)
        object.__setattr__(self, "samplePoints", points)

    def metadata(self):
        return {"config": self.config.toDict(), "configKey": self.config.configKey,
                "sourceHash": self.sourceHash, "normalizationData": self.normalizationData,
                "faces": [asdict(face) for face in self.faces],
                "timings": self.timings, "diagnostics": self.diagnostics}

    def summary(self):
        return {**self.metadata(), "meshVertexCount": len(self.mesh.vertices),
                "triangleCount": len(self.mesh.triangles), "sampleCount": len(self.samplePoints),
                "faceCount": len(self.faces)}
