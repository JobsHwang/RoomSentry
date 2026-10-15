"""Offline head-conditioned crop geometry; model points are not person truth.

No model, media, tracking, role propagation or runtime policy is implemented.
The same declared source-pixel sizes can be applied to paired negative controls.
"""

from __future__ import annotations

from scripts.roi_review_geometry import _bbox, _dimensions

SIDE_PIXELS = (192, 288, 384)


def _head(value: object) -> list[int]:
    if (
        type(value) not in (list, tuple)
        or len(value) != 2
        or any(type(v) is not int or not 0 <= v <= 1000 for v in value)
    ):
        raise ValueError("head must contain two integer source coordinates in [0, 1000]")
    return list(value)


def head_review_regions(head: object, width: object, height: object) -> list[dict]:
    """Three head-centred, clamped square ROIs, without guessed body orientation.

    Integer centres round down. Small sources clamp the side to their shorter
    dimension; coincident regions retain distinct declared size IDs. Half-open
    pixel bounds are for slicing, not a certified full-person box or head label.
    """
    width, height = _dimensions(width, height)
    point = _head(head)
    cx, cy = point[0] * width // 1000, point[1] * height // 1000
    result = []
    for requested in SIDE_PIXELS:
        side = min(requested, width, height)
        x = min(max(0, cx - side // 2), width - side)
        y = min(max(0, cy - side // 2), height - side)
        result.append(
            {
                "region_id": f"head_square_{requested}",
                "head_center": point.copy(),
                "side_pixels": requested,
                "pixel_xyxy": [x, y, x + side, y + side],
            }
        )
    return result


def map_head_roi_box(
    box: object,
    *,
    region: object,
    source_width: object,
    source_height: object,
    rotation: object = 0,
) -> dict:
    """Inverse-map a normalized ROI detection, retaining internal seam flags.

    Accept only an exactly declared region for the supplied head and source.
    Boundary contact is a geometric flag, not proof of truncation or presence.
    """
    width, height = _dimensions(source_width, source_height)
    if (
        type(region) is not dict
        or set(region) != {"region_id", "head_center", "side_pixels", "pixel_xyxy"}
        or type(region["side_pixels"]) is not int
        or type(region["pixel_xyxy"]) is not list
        or len(region["pixel_xyxy"]) != 4
        or any(type(v) is not int for v in region["pixel_xyxy"])
        or region not in head_review_regions(region["head_center"], width, height)
    ):
        raise ValueError("undeclared head-conditioned ROI")
    if type(rotation) is not int or rotation not in (0, 180):
        raise ValueError("rotation must be 0 or 180")
    a, b, c, d = _bbox(box)
    local = _bbox([a, b, c, d] if rotation == 0 else [1 - c, 1 - d, 1 - a, 1 - b])
    x1, y1, x2, y2 = region["pixel_xyxy"]
    source = _bbox(
        [
            (x1 + local[0] * (x2 - x1)) / width,
            (y1 + local[1] * (y2 - y1)) / height,
            (x1 + local[2] * (x2 - x1)) / width,
            (y1 + local[3] * (y2 - y1)) / height,
        ]
    )
    return {
        "bbox_xyxy": source,
        "crop_bbox_xyxy": local,
        "internal_boundary_edges": [
            edge
            for edge, near in (
                ("left", x1 > 0 and local[0] <= 0.02),
                ("top", y1 > 0 and local[1] <= 0.02),
                ("right", x2 < width and local[2] >= 0.98),
                ("bottom", y2 < height and local[3] >= 0.98),
            )
            if near
        ],
    }
