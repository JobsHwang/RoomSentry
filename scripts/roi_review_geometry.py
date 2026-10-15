"""Bounded, standard-library geometry for fixed offline crop review.

No media/model I/O or runtime decision is performed. A boundary flag means only
that a box touches a crop seam; it is not proof that a person was truncated.
"""

from __future__ import annotations

import copy
import math

MAX_DIMENSION = 65_536
MAX_PIXELS = 268_435_456


def _dimensions(width: object, height: object) -> tuple[int, int]:
    for value in (width, height):
        if type(value) is not int or not 1 <= value <= MAX_DIMENSION:
            raise ValueError(f"dimensions must be integers in [1, {MAX_DIMENSION}]")
    if width * height > MAX_PIXELS:
        raise ValueError(f"source area must not exceed {MAX_PIXELS} pixels")
    return width, height


def _unit_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def _bbox(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    result = [_unit_number(item, "bbox_xyxy coordinate") for item in value]
    x1, y1, x2, y2 = result
    if x1 >= x2 or y1 >= y2 or (x2 - x1) * (y2 - y1) == 0:
        raise ValueError("bbox_xyxy must have positive representable area")
    return result


def _validate_metadata(value: object, *, depth: int = 0, budget: list[int]) -> None:
    """Accept bounded JSON-like metadata (tuples also accepted for coordinates)."""
    budget[0] -= 1
    if budget[0] < 0 or depth > 32:
        raise ValueError("box metadata exceeds structural limits")
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) in (list, tuple):
        for item in value:
            _validate_metadata(item, depth=depth + 1, budget=budget)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _validate_metadata(item, depth=depth + 1, budget=budget)
        return
    raise ValueError("box metadata must be finite JSON-like data")


def review_regions(width: object, height: object) -> list[dict[str, object]]:
    """Return full and two overlapping full-height regions, in that order.

    Pixel xyxy uses half-open bounds, suitable for ``image[y1:y2, x1:x2]``.
    Left ends at ceil(3 * width / 5), right begins at floor(2 * width / 5).
    Integer arithmetic keeps both crops equally wide and covers odd/tiny widths;
    nominal 60% widths round outward. For tiny images regions may coincide but
    their IDs remain unique. No resizing or rotation is implied.
    """
    width, height = _dimensions(width, height)
    return [
        {"region_id": "full", "pixel_xyxy": [0, 0, width, height]},
        {"region_id": "left", "pixel_xyxy": [0, 0, (3 * width + 4) // 5, height]},
        {"region_id": "right", "pixel_xyxy": [2 * width // 5, 0, width, height]},
    ]


def map_crop_box(
    box: object,
    *,
    region: object,
    source_width: object,
    source_height: object,
    boundary_margin: object = 0.02,
) -> dict[str, object]:
    """Map one normalized crop box to normalized source coordinates.

    ``region`` must exactly match a fixed region returned by ``review_regions``
    for these source dimensions. ``box`` requires bbox_xyxy and confidence;
    additional finite JSON-like metadata is deep-copied unchanged. Reserved
    output keys are rejected rather than overwritten. Keypoints, if supplied,
    remain in their original coordinate system: only bbox_xyxy is transformed.

    Seam distances use normalized CROP coordinates. A distance <= margin to any
    internal crop boundary sets the flag. Source outer edges never set it. This
    helper neither merges/suppresses boxes nor establishes ground truth, count,
    identity, presence, occlusion, or safety. The caller must verify that the
    crop bytes actually correspond to the declared source/region.
    """
    return _map_box(
        box,
        region=region,
        source_width=source_width,
        source_height=source_height,
        boundary_margin=boundary_margin,
        tiled=False,
    )


def tile_review_regions(width: object, height: object) -> list[dict[str, object]]:
    """Full frame plus fixed 2x2 overlapping tiles, not target-selected crops.

    Each tile spans approximately 60% of each axis, rounded outward. IDs and
    half-open integer bounds remain explicit even when tiny tiles coincide.
    This is geometry for an offline sliced-inference ablation, not SAHI output
    merging or a runtime detection policy.
    """
    width, height = _dimensions(width, height)
    return [
        {"region_id": "full", "pixel_xyxy": [0, 0, width, height]},
        *[
            {"region_id": f"{vertical}_{horizontal}", "pixel_xyxy": [x1, y1, x2, y2]}
            for vertical, y1, y2 in (
                ("top", 0, (3 * height + 4) // 5),
                ("bottom", 2 * height // 5, height),
            )
            for horizontal, x1, x2 in (
                ("left", 0, (3 * width + 4) // 5),
                ("right", 2 * width // 5, width),
            )
        ],
    ]


def map_tile_box(
    box: object,
    *,
    region: object,
    source_width: object,
    source_height: object,
    boundary_margin: object = 0.02,
) -> dict[str, object]:
    """Apply map_crop_box's evidence-only contract to tile_review_regions.

    Only bbox_xyxy is mapped; extra metadata is not transformed. The caller
    must verify crop pixels. Overlapping boxes are retained, not person counts.
    """
    return _map_box(
        box,
        region=region,
        source_width=source_width,
        source_height=source_height,
        boundary_margin=boundary_margin,
        tiled=True,
    )


def _map_box(
    box: object,
    *,
    region: object,
    source_width: object,
    source_height: object,
    boundary_margin: object,
    tiled: bool,
) -> dict[str, object]:
    width, height = _dimensions(source_width, source_height)
    margin = _unit_number(boundary_margin, "boundary_margin")
    if type(region) is not dict or set(region) != {"region_id", "pixel_xyxy"}:
        raise ValueError("region must contain only region_id and pixel_xyxy")
    pixels = region["pixel_xyxy"]
    if (
        type(region["region_id"]) is not str
        or type(pixels) is not list
        or len(pixels) != 4
        or any(type(item) is not int for item in pixels)
        or region not in (tile_review_regions if tiled else review_regions)(width, height)
    ):
        raise ValueError("region must exactly match a fixed source review region")
    reserved = {
        "crop_bbox_xyxy",
        "region",
        "source_dimensions",
        "boundary_margin",
        "truncated_at_internal_boundary",
        "internal_boundary_edges",
        "coordinate_system",
        "additional_metadata_coordinate_system",
    }
    if type(box) is not dict or not {"bbox_xyxy", "confidence"} <= box.keys():
        raise ValueError("box requires bbox_xyxy and confidence")
    if reserved & box.keys():
        raise ValueError("box contains reserved output metadata keys")
    _validate_metadata(box, budget=[10_000])
    crop = _bbox(box["bbox_xyxy"])
    confidence = _unit_number(box["confidence"], "confidence")
    px1, py1, px2, py2 = pixels
    x1, y1, x2, y2 = crop
    mapped = _bbox(
        [
            (px1 + x1 * (px2 - px1)) / width,
            (py1 + y1 * (py2 - py1)) / height,
            (px1 + x2 * (px2 - px1)) / width,
            (py1 + y2 * (py2 - py1)) / height,
        ]
    )
    # Compare upper coordinates directly: 1 - 0.98 can round above 0.02.
    boundaries = [
        ("left", px1 > 0 and x1 <= margin),
        ("top", py1 > 0 and y1 <= margin),
        ("right", px2 < width and x2 >= 1 - margin),
        ("bottom", py2 < height and y2 >= 1 - margin),
    ]
    edges = [name for name, near in boundaries if near]
    result = copy.deepcopy(box)
    result.update(
        bbox_xyxy=mapped,
        confidence=confidence,
        crop_bbox_xyxy=crop,
        region=copy.deepcopy(region),
        source_dimensions={"width": width, "height": height},
        boundary_margin=margin,
        truncated_at_internal_boundary=bool(edges),
        internal_boundary_edges=edges,
        coordinate_system="normalized source frame xyxy",
        additional_metadata_coordinate_system="unchanged from input; not transformed",
    )
    return result
