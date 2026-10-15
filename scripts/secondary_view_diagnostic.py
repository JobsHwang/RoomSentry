"""Pure, bounded geometric diagnostics for two views of one source frame.

The caller must map both views into the SAME normalized source-image coordinate
system before calling this helper. Geometry cannot verify that prerequisite,
identify a person, establish ground truth, or establish safe baby supervision.
No media, models, services, networking, file I/O, or runtime decisions are used.
"""

from __future__ import annotations

import math

MAX_BOXES_PER_VIEW = 256


def _unit_number(value: object, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
        or not math.isfinite(value)
        or (positive and value == 0)
    ):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}unit number")
    return float(value)


def _boxes(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)) or len(value) > MAX_BOXES_PER_VIEW:
        raise ValueError(f"{name} must be a list/tuple of at most {MAX_BOXES_PER_VIEW} boxes")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"bbox_xyxy", "confidence"}:
            raise ValueError(f"{name}[{index}] requires only bbox_xyxy and confidence")
        bbox = item["bbox_xyxy"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"{name}[{index}].bbox_xyxy requires four coordinates")
        x1, y1, x2, y2 = [
            _unit_number(coordinate, f"{name}[{index}].bbox_xyxy") for coordinate in bbox
        ]
        if x1 >= x2 or y1 >= y2 or (x2 - x1) * (y2 - y1) == 0:
            raise ValueError(f"{name}[{index}].bbox_xyxy requires positive representable area")
        result.append(
            {
                "index": index,
                "bbox_xyxy": [x1, y1, x2, y2],
                "confidence": _unit_number(item["confidence"], f"{name}[{index}].confidence"),
            }
        )
    return result


def _iou(left: dict[str, object], right: dict[str, object]) -> float:
    ax1, ay1, ax2, ay2 = left["bbox_xyxy"]
    bx1, by1, bx2, by2 = right["bbox_xyxy"]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union


def diagnose_views(
    primary: object,
    secondary: object,
    *,
    minimum_confidence: float,
    iou_threshold: float,
) -> dict[str, object]:
    """Return review evidence without suppressing, merging, or relabeling boxes.

    Each input is a list/tuple of at most 256 dictionaries with exactly
    ``bbox_xyxy`` (four finite normalized xyxy coordinates, positive area) and
    ``confidence`` (finite [0, 1]). Indices always refer to the original input.
    Cross-view matching uses confidence >= the explicit minimum and IoU >= the
    explicit positive threshold. Candidate edges sort by descending IoU, then
    primary and secondary index, and are greedily accepted one-to-one. This is
    deterministic geometric correspondence, NOT an optimal assignment, identity,
    deduplicated person count, ground truth, or a presence decision.

    All input boxes, including below-threshold boxes, remain in the output.
    Within-view pairs at the same IoU threshold are review candidates across ALL
    boxes, not duplicate labels: overlapping boxes can represent different people.
    """
    minimum = _unit_number(minimum_confidence, "minimum_confidence")
    threshold = _unit_number(iou_threshold, "iou_threshold", positive=True)
    primary_boxes = _boxes(primary, "primary")
    secondary_boxes = _boxes(secondary, "secondary")
    eligible_primary = {box["index"] for box in primary_boxes if box["confidence"] >= minimum}
    eligible_secondary = {box["index"] for box in secondary_boxes if box["confidence"] >= minimum}
    edges = []
    for left in primary_boxes:
        for right in secondary_boxes:
            if left["index"] in eligible_primary and right["index"] in eligible_secondary:
                overlap = _iou(left, right)
                if overlap >= threshold:
                    edges.append((overlap, left["index"], right["index"]))
    edges.sort(key=lambda edge: (-edge[0], edge[1], edge[2]))
    used_primary = set()
    used_secondary = set()
    matches = []
    for overlap, left_index, right_index in edges:
        if left_index not in used_primary and right_index not in used_secondary:
            used_primary.add(left_index)
            used_secondary.add(right_index)
            matches.append(
                {"primary_index": left_index, "secondary_index": right_index, "iou": overlap}
            )
    within_view = []
    for view, boxes in (("primary", primary_boxes), ("secondary", secondary_boxes)):
        for index, left in enumerate(boxes):
            for right in boxes[index + 1 :]:
                overlap = _iou(left, right)
                if overlap >= threshold:
                    within_view.append(
                        {
                            "view": view,
                            "first_index": left["index"],
                            "second_index": right["index"],
                            "iou": overlap,
                            "both_at_or_above_minimum_confidence": (
                                left["confidence"] >= minimum and right["confidence"] >= minimum
                            ),
                        }
                    )
    return {
        "schema_version": "secondary-view-diagnostic-v1",
        "offline_diagnostic_only": True,
        "coordinate_system_requirement": "same normalized source frame; caller verified",
        "coordinate_system_verified_by_helper": False,
        "matching_policy": "greedy descending IoU, then ascending original input indices",
        "geometry_is_identity_or_ground_truth": False,
        "within_view_overlap_is_duplicate_label": False,
        "presence": "unknown",
        "true_person_count": None,
        "deployment_approved": False,
        "minimum_confidence": minimum,
        "iou_threshold": threshold,
        "primary_boxes": primary_boxes,
        "secondary_boxes": secondary_boxes,
        "matched_pairs": matches,
        "unmatched_primary_indices": sorted(eligible_primary - used_primary),
        "unmatched_secondary_indices": sorted(eligible_secondary - used_secondary),
        "below_threshold_primary_indices": sorted(
            set(range(len(primary_boxes))) - eligible_primary
        ),
        "below_threshold_secondary_indices": sorted(
            set(range(len(secondary_boxes))) - eligible_secondary
        ),
        "within_view_overlap_candidates": within_view,
    }
