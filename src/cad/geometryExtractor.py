"""Validate solid bodies and extract properties once, without meshing."""

from math import isfinite

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepCheck import BRepCheck_Analyzer, BRepCheck_NoError, BRepCheck_Shell
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_OUT, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, topods

from src.cad.surfaceClassifier import classifySurfaces
from src.cad.topology import mapShapes
from src.errors import CadError
from src.models.partMetadata import GeometryProperties


def collectSolidProperties(shape: TopoDS_Shape):
    """Build a solids-only compound and sum their unit-density mass properties.

    Loose faces/wires are excluded from ALL measurements. Every solid must be
    valid, closed, finite, and positively oriented; invalid bodies are not silently
    dropped. Bodies are summed, not fused, so overlapping bodies double-count
    their overlap. Avoiding a Boolean union keeps Stage 1 inexpensive.
    """
    if shape.IsNull():
        raise CadError("Cannot extract an empty shape")
    solidMap = mapShapes(shape, TopAbs_SOLID)
    if solidMap.Size() == 0:
        raise CadError("No usable solid bodies: surface-only and wire-only models are unsupported")

    builder = BRep_Builder()
    solidShape = TopoDS_Compound()
    builder.MakeCompound(solidShape)
    totalProperties = GProp_GProps()

    for solidIndex in range(1, solidMap.Size() + 1):
        solid = topods.Solid(solidMap.FindKey(solidIndex))
        if not BRepCheck_Analyzer(solid, True).IsValid():
            raise CadError(f"Solid {solidIndex} failed BRepCheck validation")
        shellMap = mapShapes(solid, TopAbs_SHELL)
        if shellMap.Size() == 0:
            raise CadError(f"Solid {solidIndex} contains no shells")
        for shellIndex in range(1, shellMap.Size() + 1):
            shell = topods.Shell(shellMap.FindKey(shellIndex))
            if BRepCheck_Shell(shell).Closed() != BRepCheck_NoError:
                raise CadError(f"Solid {solidIndex} has an open shell")

        # An infinite point must be outside a finite, outward-oriented solid.
        solidClassifier = BRepClass3d_SolidClassifier(solid)
        solidClassifier.PerformInfinitePoint(1e-7)
        if solidClassifier.State() != TopAbs_OUT:
            raise CadError(f"Solid {solidIndex} is unbounded or incorrectly oriented")

        properties = GProp_GProps()
        # OnlyClosed=True, SkipShared=False, UseTriangulation=False.
        brepgprop.VolumeProperties(solid, properties, True, False, False)
        volume = properties.Mass()
        if not isfinite(volume) or volume <= 0:
            raise CadError(f"Solid {solidIndex} has invalid or zero volume")
        totalProperties.Add(properties)
        builder.Add(solidShape, solid)

    return solidShape, solidMap.Size(), totalProperties


def extractGeometry(shape: TopoDS_Shape) -> GeometryProperties:
    """Measure a shape already transferred in millimeters by loadStep."""
    try:
        solidShape, solidCount, volumeProperties = collectSolidProperties(shape)

        bounds = Bnd_Box()
        bounds.SetGap(0.0)
        # Use the B-rep directly, without existing mesh error or added tolerances.
        brepbndlib.AddOptimal(solidShape, bounds, False, False)
        if bounds.IsVoid() or bounds.IsWhole():
            raise CadError("Solid bodies have no finite bounding box")
        minX, minY, minZ, maxX, maxY, maxZ = bounds.Get()
        sizeX, sizeY, sizeZ = maxX - minX, maxY - minY, maxZ - minZ
        if not all(isfinite(value) and value > 0 for value in (sizeX, sizeY, sizeZ)):
            raise CadError("Solid bodies have non-finite or degenerate dimensions")
        sizeMin, sizeMid, sizeMax = sorted((sizeX, sizeY, sizeZ))

        areaProperties = GProp_GProps()
        # SkipShared=True counts a shared topological face once; no mesh needed.
        brepgprop.SurfaceProperties(solidShape, areaProperties, True, False)
        surfaceArea = areaProperties.Mass()
        volume = volumeProperties.Mass()
        center = volumeProperties.CentreOfMass()
        centerX, centerY, centerZ = center.X(), center.Y(), center.Z()
        if not all(isfinite(value) and value > 0 for value in (surfaceArea, volume)):
            raise CadError("Solid bodies have invalid surface area or volume")
        if not all(isfinite(value) for value in (centerX, centerY, centerZ)):
            raise CadError("Solid bodies have a non-finite center of mass")

        geometry = GeometryProperties(
            sizeX=sizeX, sizeY=sizeY, sizeZ=sizeZ,
            sizeMin=sizeMin, sizeMid=sizeMid, sizeMax=sizeMax,
            volume=volume, surfaceArea=surfaceArea,
            centerOfMassX=centerX, centerOfMassY=centerY, centerOfMassZ=centerZ,
            aspectXY=sizeX / sizeY, aspectXZ=sizeX / sizeZ, aspectYZ=sizeY / sizeZ,
            edgeCount=mapShapes(solidShape, TopAbs_EDGE).Size(),
            vertexCount=mapShapes(solidShape, TopAbs_VERTEX).Size(),
            solidCount=solidCount,
            **classifySurfaces(solidShape),
        )
        if not all(isfinite(value) for value in geometry.toDict().values() if isinstance(value, float)):
            raise CadError("Extracted properties contain non-finite numbers")
        return geometry
    except CadError:
        raise
    except Exception as error:
        raise CadError(f"OpenCascade could not measure the model: {error}") from error
