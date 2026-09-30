"""Triangulate an independent copy, preserving the original TopoDS_Shape."""

import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods

from src.cad.geometryExtractor import collectSolidProperties
from src.cad.topology import mapShapes
from src.errors import CadError
from src.models.mesh import Mesh
from src.models.sphereDescriptor import DescriptorConfig


def extractMesh(shape, config: DescriptorConfig | None = None) -> Mesh:
    """Extract world coordinates and oriented, zero-based triangle indices.

    Absolute linear deflection is ratio * cbrt(solid volume), so uniform scaling
    changes the meshing tolerance proportionally. No tessellation is attached to
    the caller's B-rep. Meshing is serial for repeatability.
    """
    config = config or DescriptorConfig()
    try:
        solidShape, _, properties = collectSolidProperties(shape)
        copiedShape = BRepBuilderAPI_Copy(solidShape, True, False).Shape()
        linearDeflection = config.linearDeflectionRatio * float(np.cbrt(properties.Mass()))
        mesher = BRepMesh_IncrementalMesh(
            copiedShape, linearDeflection, False, config.angularDeflection, False
        )
        if not mesher.IsDone():
            raise CadError("OpenCascade triangulation did not complete")

        vertices, triangles, bodyIds = [], [], []
        solidMap = mapShapes(copiedShape, TopAbs_SOLID)
        for bodyIndex in range(1, solidMap.Size() + 1):
            faceMap = mapShapes(solidMap.FindKey(bodyIndex), TopAbs_FACE)
            for faceIndex in range(1, faceMap.Size() + 1):
                face = topods.Face(faceMap.FindKey(faceIndex))
                location = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation(face, location)
                if triangulation is None or triangulation.NbTriangles() == 0:
                    raise CadError(f"Face {faceIndex} of body {bodyIndex} has no triangulation")
                offset = len(vertices)
                transform = location.Transformation()
                for nodeIndex in range(1, triangulation.NbNodes() + 1):
                    point = triangulation.Node(nodeIndex).Transformed(transform)
                    vertices.append((point.X(), point.Y(), point.Z()))
                for triangleIndex in range(1, triangulation.NbTriangles() + 1):
                    a, b, c = triangulation.Triangle(triangleIndex).Get()
                    if face.Orientation() == TopAbs_REVERSED:
                        b, c = c, b
                    triangles.append((offset + a - 1, offset + b - 1, offset + c - 1))
                    bodyIds.append(bodyIndex - 1)

        vertices = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64)
        bodyIds = np.asarray(bodyIds, dtype=np.int64)
        if len(vertices) == 0 or len(triangles) == 0:
            raise CadError("Triangulation is empty")

        # OCCT stores each face's mesh separately. Weld seam vertices before PCA
        # so duplicate face-boundary samples do not bias the covariance.
        reference = vertices.min(axis=0)
        extent = float(np.max(np.ptp(vertices, axis=0)))
        if not np.isfinite(extent) or extent <= 0:
            raise CadError("Triangulation has invalid extent")
        keys = np.rint((vertices - reference) / (extent * config.weldRelativeTolerance)).astype(np.int64)
        _, uniqueIndices, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        vertices = vertices[uniqueIndices]
        triangles = inverse[triangles]
        distinct = np.all(np.diff(np.sort(triangles, axis=1), axis=1) != 0, axis=1)
        triangles, bodyIds = triangles[distinct], bodyIds[distinct]
        usedIndices, inverse = np.unique(triangles, return_inverse=True)
        return Mesh(vertices[usedIndices], inverse.reshape(-1, 3), bodyIds)
    except CadError:
        raise
    except Exception as error:
        raise CadError(f"Cannot extract a closed triangle mesh: {error}") from error
