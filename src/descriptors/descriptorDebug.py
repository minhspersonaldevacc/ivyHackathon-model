"""Optional grayscale PNG views of normalized channels, using only NumPy/stdlib."""

from pathlib import Path
import struct
import zlib

import numpy as np

from src.models.sphereDescriptor import SphereDescriptor, channelNames


def saveChannelImages(descriptor: SphereDescriptor, outputDirectory: str | Path) -> list[str]:
    """Black=0, white=1; rows are uniform mu, columns are azimuth, not latitude."""
    outputPath = Path(outputDirectory).expanduser().resolve()
    outputPath.mkdir(parents=True, exist_ok=True)

    def pngChunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xffffffff
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    paths = []
    for channelIndex, name in enumerate(channelNames):
        pixels = np.rint(descriptor.tensor[..., channelIndex] * 255).astype(np.uint8)
        pixels = np.repeat(np.repeat(pixels, 4, axis=0), 4, axis=1)
        height, width = pixels.shape
        rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
        png = b"\x89PNG\r\n\x1a\n"
        png += pngChunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        png += pngChunk(b"IDAT", zlib.compress(rows)) + pngChunk(b"IEND", b"")
        filePath = outputPath / f"{name}.png"
        filePath.write_bytes(png)
        paths.append(str(filePath))
    return paths
