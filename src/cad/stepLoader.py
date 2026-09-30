"""STEP transfer into OpenCascade shapes, with explicit millimeter units."""

from pathlib import Path

from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopoDS import TopoDS_Shape

from src.errors import CadError


def loadStep(filePath: str | Path) -> TopoDS_Shape:
    """Read all STEP roots. Solid validation is performed by extractGeometry.

    SetSystemLengthUnit must be called AFTER ReadFile creates the STEP model
    and BEFORE transfer. A value of 1.0 means one millimeter per internal unit.
    The reader converts the STEP file's declared units, including inches.
    """
    sourcePath = Path(filePath).expanduser().resolve()
    if sourcePath.suffix.lower() not in {".step", ".stp"}:
        raise CadError(f"Unsupported file extension: {sourcePath.suffix}")
    if not sourcePath.is_file():
        raise CadError(f"STEP file does not exist or is not a file: {sourcePath}")

    try:
        reader = STEPControl_Reader()
        readStatus = reader.ReadFile(str(sourcePath))
        if readStatus != IFSelect_RetDone:
            raise CadError(f"STEP read failed (OpenCascade status {int(readStatus)})")

        reader.SetSystemLengthUnit(1.0)
        rootCount = reader.NbRootsForTransfer()
        if rootCount == 0:
            raise CadError("STEP file contains no transferable roots")
        transferredCount = reader.TransferRoots()
        if transferredCount != rootCount:
            raise CadError(
                f"Incomplete STEP transfer: {transferredCount} of {rootCount} roots"
            )

        shape = reader.OneShape()
        if shape.IsNull():
            raise CadError("STEP transfer returned an empty shape")
        return shape
    except CadError:
        raise
    except Exception as error:
        # SWIG translates several native OCCT exceptions into Python exceptions.
        raise CadError(f"OpenCascade could not load STEP: {error}") from error
