"""Coarse, deterministic family rules using only extracted numbers."""

from src.models.partMetadata import GeometryProperties


def classifyPart(geometry: GeometryProperties) -> str:
    """Apply rules in priority order; thresholds are deliberately easy to edit.

    Face ratios are counts, not area fractions. These labels describe geometry,
    not a guaranteed manufacturing process. Assemblies stay in general/unknown.
    """
    if geometry.solidCount != 1:
        return "general/unknown"

    small, middle, large = geometry.sizeMin, geometry.sizeMid, geometry.sizeMax
    thinness = small / middle
    elongation = large / middle
    transverseMatch = middle / small
    broadMatch = large / middle
    roundRatio = geometry.cylindricalRatio + geometry.conicalRatio

    # Round stock: two similar transverse dimensions and a long third dimension.
    if roundRatio >= 0.25 and transverseMatch <= 1.25 and elongation >= 3.0:
        return "shaft/turned-like"

    # Cylinders/washers may be either tall or flat. Evaluate before plates so a
    # washer with two planar end faces is not automatically called a plate.
    if geometry.cylindricalRatio >= 0.25 and (
        transverseMatch <= 1.25 or broadMatch <= 1.25
    ):
        return "cylindrical/ring-like"

    if geometry.planarRatio >= 0.60 and thinness <= 0.20:
        return "plate-like"

    if geometry.planarRatio >= 0.75 and large / small <= 3.0:
        return "block-like"

    return "general/unknown"
