"""Identify analytic surface types without fitting or comparing surfaces."""

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import TopoDS_Shape, topods

from src.cad.topology import mapShapes


def classifySurfaces(shape: TopoDS_Shape) -> dict[str, int | float]:
    """Count unique faces and return count-based ratios.

    GetType reports the underlying surface's native type. Spheres, tori, splines,
    and unrecognized surfaces go in 'other'; splines are not fitted to cylinders.
    """
    faceMap = mapShapes(shape, TopAbs_FACE)
    counts = {"planarFaces": 0, "cylindricalFaces": 0, "conicalFaces": 0, "otherFaces": 0}
    surfaceKeys = {
        GeomAbs_Plane: "planarFaces",
        GeomAbs_Cylinder: "cylindricalFaces",
        GeomAbs_Cone: "conicalFaces",
    }
    for faceIndex in range(1, faceMap.Size() + 1):
        face = topods.Face(faceMap.FindKey(faceIndex))
        surfaceType = BRepAdaptor_Surface(face, True).GetType()
        counts[surfaceKeys.get(surfaceType, "otherFaces")] += 1

    faceCount = faceMap.Size()
    return {
        **counts,
        "faceCount": faceCount,
        "planarRatio": counts["planarFaces"] / faceCount if faceCount else 0.0,
        "cylindricalRatio": counts["cylindricalFaces"] / faceCount if faceCount else 0.0,
        "conicalRatio": counts["conicalFaces"] / faceCount if faceCount else 0.0,
        "otherRatio": counts["otherFaces"] / faceCount if faceCount else 0.0,
    }
