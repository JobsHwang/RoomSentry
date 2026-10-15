"""Pure geometry for fixed tiles inferred at zero or 180 degrees.

Regions always refer to the original source. Inference coordinates are first
inverse-rotated inside that tile, then mapped through its exact integer bounds.
No pixels, files, models, box fusion, person counts or identities are accessed.
The caller must independently bind source pixels and the declared transform.
"""

from __future__ import annotations

import math

from baby_monitor.frame_transform import inverse_normalized_box, inverse_normalized_point
from scripts.roi_review_geometry import map_tile_box, tile_review_regions


def _rotation(value: object) -> int:
    if type(value) is not int or value not in (0, 180):
        raise ValueError("tile rotation must be integer 0 or 180")
    return value


def _unit_number(value: object) -> float:
    # Bounds precede isfinite so enormous integers cannot overflow conversion.
    if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
        raise ValueError("tile coordinates and confidence must be finite numbers in [0, 1]")
    return float(value)


def _region(value: object, width: object, height: object) -> dict[str, object]:
    regions = tile_review_regions(width, height)
    if type(value) is not dict or set(value) != {"region_id", "pixel_xyxy"}:
        raise ValueError("region must be a fixed source tile")
    pixels = value["pixel_xyxy"]
    if (
        type(value["region_id"]) is not str
        or type(pixels) is not list
        or len(pixels) != 4
        or any(type(item) is not int for item in pixels)
        or value not in regions
    ):
        raise ValueError("region must be a fixed source tile")
    return {"region_id": value["region_id"], "pixel_xyxy": list(pixels)}


def map_oriented_tile_point(
    x: object,
    y: object,
    *,
    region: object,
    source_width: object,
    source_height: object,
    rotation_degrees: object = 0,
) -> tuple[float, float]:
    """Map one normalized inference-tile point to normalized original source.

    Points may touch the outer boundaries. This does not validate anatomical
    keypoints, assign a person, or establish that the caller rotated any pixels.
    """
    rotation = _rotation(rotation_degrees)
    tile = _region(region, source_width, source_height)
    original_x, original_y = inverse_normalized_point(
        _unit_number(x), _unit_number(y), rotation_degrees=rotation
    )
    left, top, right, bottom = tile["pixel_xyxy"]
    return (
        (left + original_x * (right - left)) / source_width,
        (top + original_y * (bottom - top)) / source_height,
    )


def map_oriented_tile_box(
    box: object,
    *,
    region: object,
    source_width: object,
    source_height: object,
    rotation_degrees: object = 0,
) -> dict[str, object]:
    """Map exactly bbox_xyxy/confidence; reject untransformed extra coordinates.

    Output follows map_tile_box: bbox_xyxy is original-source normalized space,
    crop_bbox_xyxy is the inverse-rotated, original-tile normalized box, and seam
    names refer to that original tile. rotated_tile_bbox_xyxy preserves the
    inference input box; rotation_degrees records its declared orientation.
    Confidence is not adjusted. A seam flag is geometry, not proven truncation.
    """
    rotation = _rotation(rotation_degrees)
    tile = _region(region, source_width, source_height)
    if type(box) is not dict or set(box) != {"bbox_xyxy", "confidence"}:
        raise ValueError("oriented tile box requires only bbox_xyxy and confidence")
    supplied = box["bbox_xyxy"]
    if type(supplied) not in (list, tuple) or len(supplied) != 4:
        raise ValueError("oriented tile box requires four coordinates")
    oriented = [_unit_number(item) for item in supplied]
    x1, y1, x2, y2 = oriented
    if x1 >= x2 or y1 >= y2 or (x2 - x1) * (y2 - y1) == 0:
        raise ValueError("oriented tile box must have positive representable area")
    confidence = _unit_number(box["confidence"])
    original = inverse_normalized_box(oriented, rotation_degrees=rotation)
    result = map_tile_box(
        {"bbox_xyxy": original, "confidence": confidence},
        region=tile,
        source_width=source_width,
        source_height=source_height,
    )
    if rotation == 180:
        # Compare in the inference plane to keep inclusive seam boundaries
        # stable: computing 1 - 0.98 first would round above the 0.02 margin.
        left, top, right, bottom = tile["pixel_xyxy"]
        margin = result["boundary_margin"]
        edges = [
            name
            for name, near in (
                ("left", left > 0 and x2 >= 1 - margin),
                ("top", top > 0 and y2 >= 1 - margin),
                ("right", right < source_width and x1 <= margin),
                ("bottom", bottom < source_height and y1 <= margin),
            )
            if near
        ]
        result["internal_boundary_edges"] = edges
        result["truncated_at_internal_boundary"] = bool(edges)
    result["rotation_degrees"] = rotation
    result["rotated_tile_bbox_xyxy"] = oriented
    return result
