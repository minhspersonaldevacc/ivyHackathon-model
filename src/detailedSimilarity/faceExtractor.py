"""Analytic B-rep face properties and trimmed-face surface coverage."""

import numpy as np

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepTools import breptools
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus)
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_REVERSED
from OCC.Core.TopoDS import topods
from OCC.Core.gp import gp_Pnt2d

from src.cad.geometryExtractor import collectSolidProperties
from src.cad.topology import mapShapes
from src.detailedSimilarity.surfaceSampling import radicalInverse
from src.models.detailedGeometry import FaceFeature


def extractFaceFeatures(shape, normalizationData, samplesPerFace=8):
    solidShape, _, _ = collectSolidProperties(shape)
    faceMap = mapShapes(solidShape, TopAbs_FACE)
    edgeMap = mapShapes(solidShape, TopAbs_EDGE)
    edgeOwners = {}
    neighbors = [set() for _ in range(faceMap.Size())]
    for faceIndex in range(faceMap.Size()):
        edges = mapShapes(faceMap.FindKey(faceIndex + 1), TopAbs_EDGE)
        for edgeIndex in range(1, edges.Size() + 1):
            globalEdgeId = edgeMap.FindIndex(edges.FindKey(edgeIndex))
            edgeOwners.setdefault(globalEdgeId, set()).add(faceIndex)
    for owners in edgeOwners.values():
        for owner in owners:
            neighbors[owner].update(owners - {owner})

    centroid = np.asarray(normalizationData["centroid"])
    axes = np.asarray(normalizationData["pcaAxes"])
    scale = normalizationData["radiusScale"]
    features, samples, coverage = [], [], []
    for faceIndex in range(faceMap.Size()):
        face = topods.Face(faceMap.FindKey(faceIndex + 1))
        properties = GProp_GProps()
        brepgprop.SurfaceProperties(face, properties, True, False)
        center = properties.CentreOfMass()
        normalizedCenter = (np.array((center.X(), center.Y(), center.Z())) - centroid) @ axes / scale
        surface = BRepAdaptor_Surface(face, True)
        kind = surface.GetType()
        radius = angle = 0.0
        direction = np.zeros(3)
        if kind == GeomAbs_Plane:
            surfaceType, axis = "plane", surface.Plane().Axis()
        elif kind == GeomAbs_Cylinder:
            surfaceType, axis = "cylinder", surface.Cylinder().Axis()
            radius = surface.Cylinder().Radius() / scale
        elif kind == GeomAbs_Cone:
            surfaceType, axis = "cone", surface.Cone().Axis()
            radius, angle = surface.Cone().RefRadius() / scale, surface.Cone().SemiAngle()
        elif kind == GeomAbs_Sphere:
            surfaceType, axis = "sphere", None
            radius = surface.Sphere().Radius() / scale
        elif kind == GeomAbs_Torus:
            surfaceType, axis = "torus", surface.Torus().Axis()
            radius = surface.Torus().MajorRadius() / scale
        else:
            surfaceType, axis = "other", None
        if axis is not None:
            vector = axis.Direction()
            direction = np.array((vector.X(), vector.Y(), vector.Z())) @ axes
            if surfaceType == "plane" and face.Orientation() == TopAbs_REVERSED:
                direction *= -1
        features.append(FaceFeature(surfaceType, properties.Mass() / scale ** 2,
                                    tuple(normalizedCenter), tuple(direction), radius, angle,
                                    tuple(sorted(neighbors[faceIndex]))))

        # Value() evaluates exact surfaces; the face classifier rejects trimmed
        # regions (including holes). Coordinates alone are not a visibility test.
        minU, maxU, minV, maxV = breptools.UVBounds(face)
        found = []
        if np.isfinite((minU, maxU, minV, maxV)).all():
            sequence = np.arange(1, samplesPerFace * 64 + 1)
            uValues = minU + radicalInverse(sequence, 2) * (maxU - minU)
            vValues = minV + radicalInverse(sequence, 3) * (maxV - minV)
            for u, v in zip(uValues, vValues):
                status = BRepClass_FaceClassifier(face, gp_Pnt2d(float(u), float(v)), 1e-7).State()
                if status not in (TopAbs_IN, TopAbs_ON):
                    continue
                point = surface.Value(float(u), float(v))
                found.append((point.X(), point.Y(), point.Z()))
                if len(found) == samplesPerFace:
                    break
        coverage.append(len(found))
        samples.extend(found)
    anchors = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    anchors = (anchors - centroid) @ axes / scale
    return tuple(features), anchors, {"faceSampleCounts": coverage,
                                     "facesWithoutSamples": sum(count == 0 for count in coverage)}
