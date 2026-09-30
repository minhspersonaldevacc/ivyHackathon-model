"""Small helper for counting distinct topological entities."""

from OCC.Core.TopAbs import TopAbs_ShapeEnum
from OCC.Core.TopExp import topexp
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.TopoDS import TopoDS_Shape


def mapShapes(shape: TopoDS_Shape, shapeType: TopAbs_ShapeEnum):
    """Return a fresh, 1-based map, deduplicated by shape identity and location.

    A shared edge appears in multiple face traversals. An indexed map counts it
    once, while preserving separate occurrences at different assembly locations.
    """
    shapeMap = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, shapeType, shapeMap)
    return shapeMap
