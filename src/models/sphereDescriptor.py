"""Versioned descriptor configuration and portable NumPy data."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import numpy as np


descriptorVersion = 1
normalizationVersion = 1
channelNames = ("firstDistance", "lastDistance", "intersectionCount", "thickness")


@dataclass(frozen=True)
class DescriptorConfig:
    thetaResolution: int = 32
    phiResolution: int = 64
    countScale: int = 16
    linearDeflectionRatio: float = 0.002
    angularDeflection: float = 0.3
    weldRelativeTolerance: float = 1e-9
    rayTolerance: float = 1e-7
    pcaDegeneracyTolerance: float = 0.001
    leafSize: int = 32
    descriptorVersion: int = descriptorVersion
    normalizationVersion: int = normalizationVersion

    def __post_init__(self):
        for name in ("thetaResolution", "phiResolution", "countScale", "leafSize"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.thetaResolution < 2 or self.phiResolution < 4 or self.phiResolution % 2:
            raise ValueError("Use thetaResolution >= 2 and even phiResolution >= 4 for exact axis-flip mappings")
        if self.thetaResolution * self.phiResolution > 1_000_000:
            raise ValueError("Sampling grid is too large; use at most 1,000,000 directions")
        for name in ("linearDeflectionRatio", "weldRelativeTolerance", "rayTolerance", "pcaDegeneracyTolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if self.weldRelativeTolerance < 1e-15:
            raise ValueError("weldRelativeTolerance must be at least 1e-15 for stable coordinate quantization")
        if not math.isfinite(self.angularDeflection) or not 0 < self.angularDeflection < math.pi:
            raise ValueError("angularDeflection must be in (0, pi) radians")
        if self.descriptorVersion != descriptorVersion or self.normalizationVersion != normalizationVersion:
            raise ValueError("This implementation cannot generate the requested descriptor/normalization version")

    def toDict(self) -> dict:
        return asdict(self)

    @property
    def configKey(self) -> str:
        configJson = json.dumps(self.toDict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(configJson.encode()).hexdigest()

    @property
    def tensorShape(self) -> tuple[int, int, int]:
        return self.thetaResolution, self.phiResolution, 4


@dataclass(frozen=True)
class SphereDescriptor:
    tensor: np.ndarray
    config: DescriptorConfig = field(default_factory=DescriptorConfig)
    normalizationData: dict = field(default_factory=dict)
    meshVertexCount: int = 0
    meshTriangleCount: int = 0
    timings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    sourceHash: str | None = None

    def __post_init__(self):
        tensor = np.array(self.tensor, dtype=np.float32, copy=True, order="C")
        if tensor.shape != self.config.tensorShape:
            raise ValueError(f"Expected descriptor shape {self.config.tensorShape}, got {tensor.shape}")
        if not np.isfinite(tensor).all() or np.any(tensor < 0) or np.any(tensor > 1):
            raise ValueError("All normalized descriptor channels must be finite and within [0, 1]")
        tensor.setflags(write=False)
        object.__setattr__(self, "tensor", tensor)

    def toVector(self) -> np.ndarray:
        """C-order, interleaved channels; the default grid produces 8,192 values.

        A future compressor can consume this vector without changing the mesh or
        sampling APIs. No learned or linear compression is applied here.
        """
        return self.tensor.reshape(-1)

    def metadata(self) -> dict:
        return {
            "config": self.config.toDict(), "configKey": self.config.configKey,
            "descriptorVersion": self.config.descriptorVersion,
            "normalizationVersion": self.config.normalizationVersion,
            "thetaResolution": self.config.thetaResolution,
            "phiResolution": self.config.phiResolution, "channelCount": 4,
            "meshVertexCount": self.meshVertexCount, "meshTriangleCount": self.meshTriangleCount,
            "normalizationData": self.normalizationData, "timings": self.timings,
            "diagnostics": self.diagnostics, "sourceHash": self.sourceHash,
        }

    def summary(self) -> dict:
        return {
            **self.metadata(), "descriptorDimensions": list(self.tensor.shape),
            "vectorLength": self.tensor.size,
            "channels": {
                name: {"min": float(self.tensor[..., index].min()), "max": float(self.tensor[..., index].max())}
                for index, name in enumerate(channelNames)
            },
        }
