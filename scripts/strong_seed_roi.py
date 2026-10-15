"""Offline, causal strong-box ROI diagnostic. No live entry or role inheritance.

Regions use only the previous native frame. Crop detections are proposals, not
additional people: a prior false positive or a body part can seed a proposal.
"""

from __future__ import annotations

import copy
import math

from scripts.roi_review_geometry import _bbox, _dimensions, _unit_number

REVISION = "previous_native_strong_roi_v1"
SETTINGS = {
    "confidence": 0.35,
    "maximum_seed_age_seconds": 2.0,
    "existing_iou": 0.30,
    "seed_candidate_iou": 0.50,
    "duplicate_intersection_over_smaller": 0.70,
    "context_scale": 2.0,
    "boundary_margin": 0.02,
    "maximum_regions": 4,
}


def _boxes(rows):
    if type(rows) is not list or len(rows) > 100:
        raise ValueError("bounded detector rows required")
    result = []
    for row in rows:
        if type(row) is not dict or set(row) != {"bbox_xyxy", "confidence"}:
            raise ValueError("plain detector geometry required")
        result.append(
            {
                "bbox_xyxy": _bbox(row["bbox_xyxy"]),
                "confidence": _unit_number(row["confidence"], "confidence"),
            }
        )
    return result


def overlap(a, b):
    """IoU and intersection over smaller box, without a body/identity claim."""
    a, b = _bbox(a), _bbox(b)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return intersection / (area_a + area_b - intersection), intersection / min(area_a, area_b)


def _conflict(a, b):
    iou, smaller = overlap(a, b)
    return (
        iou >= SETTINGS["existing_iou"]
        or smaller >= SETTINGS["duplicate_intersection_over_smaller"]
    )


def plan_regions(previous, current, *, width, height, elapsed_seconds, same_generation=True):
    """Uncovered, nonoverlapping strong native seeds only; no extrapolation."""
    previous, current = _boxes(previous), _boxes(current)
    width, height = _dimensions(width, height)
    if (
        type(elapsed_seconds) not in (int, float)
        or not math.isfinite(elapsed_seconds)
        or type(same_generation) is not bool
    ):
        raise ValueError("valid elapsed time and generation decision required")
    regions, decisions = [], []
    for i, seed in enumerate(previous):
        box = seed["bbox_xyxy"]
        reason = (
            "generation_changed"
            if not same_generation
            else "seed_time_invalid"
            if not 0 < elapsed_seconds <= SETTINGS["maximum_seed_age_seconds"]
            else "seed_not_strong"
            if seed["confidence"] < SETTINGS["confidence"]
            else "prior_geometry_ambiguous"
            if any(j != i and _conflict(box, row["bbox_xyxy"]) for j, row in enumerate(previous))
            else "current_box_already_covers_seed"
            if any(_conflict(box, row["bbox_xyxy"]) for row in current)
            else "region_capacity"
            if len(regions) >= SETTINGS["maximum_regions"]
            else None
        )
        decisions.append({"seed_index": i, "reason": reason or "planned"})
        if reason is not None:
            continue
        cx, cy = (box[0] + box[2]) * width / 2, (box[1] + box[3]) * height / 2
        rx = (box[2] - box[0]) * width * SETTINGS["context_scale"] / 2
        ry = (box[3] - box[1]) * height * SETTINGS["context_scale"] / 2
        pixels = [
            max(0, math.floor(cx - rx)),
            max(0, math.floor(cy - ry)),
            min(width, math.ceil(cx + rx)),
            min(height, math.ceil(cy + ry)),
        ]
        if pixels[0] >= pixels[2] or pixels[1] >= pixels[3]:
            raise ValueError("empty ROI")
        regions.append({"seed_index": i, "seed": copy.deepcopy(seed), "pixel_xyxy": pixels})
    return {
        "revision": REVISION,
        "settings": dict(SETTINGS),
        "source_dimensions": [width, height],
        "current": current,
        "regions": regions,
        "seed_decisions": decisions,
    }


def evaluate_crops(plan, crop_rows):
    """Map all crop boxes, reject seams/conflicts, keep native outputs verbatim."""
    if (
        plan.get("revision") != REVISION
        or plan.get("settings") != SETTINGS
        or type(crop_rows) is not list
        or len(crop_rows) != len(plan["regions"])
    ):
        raise ValueError("crop results must match the frozen ROI plan")
    width, height = _dimensions(*plan["source_dimensions"])
    current = _boxes(plan["current"])
    rows = []
    for region, boxes in zip(plan["regions"], crop_rows, strict=True):
        x1, y1, x2, y2 = region["pixel_xyxy"]
        eligible = []
        for index, box in enumerate(_boxes(boxes)):
            a, b, c, d = box["bbox_xyxy"]
            mapped = _bbox(
                [
                    (x1 + a * (x2 - x1)) / width,
                    (y1 + b * (y2 - y1)) / height,
                    (x1 + c * (x2 - x1)) / width,
                    (y1 + d * (y2 - y1)) / height,
                ]
            )
            margin = SETTINGS["boundary_margin"]
            seam = (
                (x1 > 0 and a <= margin)
                or (y1 > 0 and b <= margin)
                or (x2 < width and c >= 1 - margin)
                or (y2 < height and d >= 1 - margin)
            )
            reason = (
                "crop_not_strong"
                if box["confidence"] < SETTINGS["confidence"]
                else "internal_crop_boundary"
                if seam
                else "not_seed_geometry"
                if overlap(mapped, region["seed"]["bbox_xyxy"])[0] < SETTINGS["seed_candidate_iou"]
                else "current_detection_conflict"
                if any(_conflict(mapped, row["bbox_xyxy"]) for row in current)
                else "pending_visible_person_reference"
            )
            row = {
                "seed_index": region["seed_index"],
                "crop_detection_index": index,
                "crop_detection": box,
                "bbox_xyxy": mapped,
                "confidence": box["confidence"],
                "reason": reason,
            }
            rows.append(row)
            if reason == "pending_visible_person_reference":
                eligible.append(row)
        if len(eligible) > 1:
            for row in eligible:
                row["reason"] = "multiple_candidates_for_seed"
    candidates = [r for r in rows if r["reason"] == "pending_visible_person_reference"]
    conflicting = {
        i
        for i, a in enumerate(candidates)
        if any(
            i != j and _conflict(a["bbox_xyxy"], b["bbox_xyxy"]) for j, b in enumerate(candidates)
        )
    }
    for i in conflicting:
        candidates[i]["reason"] = "cross_crop_conflict"
    return {
        "revision": REVISION,
        "native": copy.deepcopy(current),
        "crop_diagnostics": rows,
        "proposals": [
            copy.deepcopy(r)
            for r in candidates
            if r["reason"] == "pending_visible_person_reference"
        ],
        "person_count": None,
        "baby_accuracy_measured": False,
        "role_inheritance": False,
        "deployment_approved": False,
    }
