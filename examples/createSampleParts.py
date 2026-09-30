"""Generate a few known solids for exercising the CLI; existing files are skipped.

Run from the project root: python examples/createSampleParts.py ./parts
The Boolean cut creates a ring fixture; it is not used by the indexing pipeline.
"""

import argparse
from pathlib import Path

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer


def createSampleParts(outputPath: Path) -> None:
    outputPath.mkdir(parents=True, exist_ok=True)
    shapes = {
        "block.step": BRepPrimAPI_MakeBox(10, 20, 30).Shape(),
        "plate.step": BRepPrimAPI_MakeBox(100, 80, 5).Shape(),
        "shaft.step": BRepPrimAPI_MakeCylinder(5, 80).Shape(),
        "ring.step": BRepAlgoAPI_Cut(
            BRepPrimAPI_MakeCylinder(10, 3).Shape(), BRepPrimAPI_MakeCylinder(5, 3).Shape()
        ).Shape(),
        "cone.step": BRepPrimAPI_MakeCone(10, 5, 15).Shape(),
        "sphere.step": BRepPrimAPI_MakeSphere(10).Shape(),
    }
    for fileName, shape in shapes.items():
        filePath = outputPath / fileName
        if filePath.exists():
            print(f"Skipped existing file: {filePath}")
            continue
        writer = STEPControl_Writer()
        if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
            raise RuntimeError(f"STEP transfer failed for {filePath}")
        if writer.Write(str(filePath)) != IFSelect_RetDone:
            raise RuntimeError(f"STEP write failed for {filePath}")
        print(f"Created {filePath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create six sample STEP solids in millimeters")
    parser.add_argument("outputPath", nargs="?", type=Path, default=Path("parts"))
    createSampleParts(parser.parse_args().outputPath)
