"""Small real STEP fixtures for Stage 3 evaluation; existing files are skipped."""

import argparse
from pathlib import Path

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec


def createDetailedParts(outputPath):
    outputPath = Path(outputPath)
    outputPath.mkdir(parents=True, exist_ok=True)
    block = BRepPrimAPI_MakeBox(30, 20, 10).Shape()
    rotation = gp_Trsf()
    rotation.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1.5707963267948966)
    rotation.SetTranslationPart(gp_Vec(100, 50, 20))
    shapes = {
        "block.step": block,
        "block-transformed.step": BRepBuilderAPI_Transform(block, rotation, True).Shape(),
        "block-hole.step": BRepAlgoAPI_Cut(block, BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(7, 8, -1), gp_Dir(0, 0, 1)), 2, 12).Shape()).Shape(),
        "block-pocket.step": BRepAlgoAPI_Cut(block, BRepPrimAPI_MakeBox(gp_Pnt(5, 5, 7), 8, 6, 4).Shape()).Shape(),
        "block-deep-pocket.step": BRepAlgoAPI_Cut(block, BRepPrimAPI_MakeBox(gp_Pnt(5, 5, 5), 8, 6, 6).Shape()).Shape(),
        "unrelated-cylinder.step": BRepPrimAPI_MakeCylinder(10, 30).Shape(),
    }
    for name, shape in shapes.items():
        filePath = outputPath / name
        if filePath.exists():
            continue
        writer = STEPControl_Writer()
        if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone or writer.Write(str(filePath)) != IFSelect_RetDone:
            raise RuntimeError(f"Cannot write fixture {filePath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputPath", type=Path)
    createDetailedParts(parser.parse_args().outputPath)
