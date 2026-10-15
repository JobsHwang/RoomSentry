"""Bounded offline evidence for original/180-degree candidates of ONE frame.

Preserves unilateral and weak candidates. Geometric support is not identity,
ground truth, a calibrated score, or permission to suppress a live detection.
No media, model, network, file access or runtime policy changes occur here.
"""

from __future__ import annotations

import math

from scripts.secondary_view_diagnostic import diagnose_views

MAX_BOXES = 100


def _unit(value: object, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError("candidate diagnostic requires finite unit thresholds")
    try:
        valid = math.isfinite(value) and 0 <= value <= 1 and (not positive or value > 0)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError("candidate diagnostic requires finite unit thresholds")
    return float(value)


def diagnose_candidates(
    primary: object,
    rotated: object,
    *,
    support_confidence: float,
    candidate_confidence: float,
    iou_threshold: float,
) -> dict:
    """Map rotated xyxy back to the source and report greedy geometric support.

    Both lists must belong to the SAME source frame; the caller must verify that
    binding and the actual 180-degree transform. All normalized raw boxes remain.
    Only rotated boxes at candidate_confidence participate in cross-view matching,
    so a weak rotated box cannot steal a weak original from a stronger candidate.
    Matching is one-to-one greedy IoU, NOT proof of a unique person correspondence.
    """
    support = _unit(support_confidence)
    candidate = _unit(candidate_confidence)
    overlap = _unit(iou_threshold, positive=True)
    if support > candidate:
        raise ValueError("support threshold must not exceed candidate threshold")
    if any(type(boxes) is not list or len(boxes) > MAX_BOXES for boxes in (primary, rotated)):
        raise ValueError("candidate diagnostic requires bounded lists")
    checked = diagnose_views(primary, rotated, minimum_confidence=support, iou_threshold=overlap)
    original = checked["primary_boxes"]
    mapped = []
    for box in checked["secondary_boxes"]:
        x1, y1, x2, y2 = box["bbox_xyxy"]
        mapped.append(
            {
                "index": box["index"],
                "confidence": box["confidence"],
                "bbox_xyxy": [1 - x2, 1 - y2, 1 - x1, 1 - y1],
            }
        )
    eligible = [box for box in mapped if box["confidence"] >= candidate]
    match = diagnose_views(
        [{k: b[k] for k in ("confidence", "bbox_xyxy")} for b in original],
        [{k: b[k] for k in ("confidence", "bbox_xyxy")} for b in eligible],
        minimum_confidence=support,
        iou_threshold=overlap,
    )
    pairs = []
    for pair in match["matched_pairs"]:
        left = original[pair["primary_index"]]
        right = eligible[pair["secondary_index"]]
        pairs.append(
            {
                "primary_index": left["index"],
                "rotated_index": right["index"],
                "iou": pair["iou"],
                "primary_confidence": left["confidence"],
                "rotated_confidence": right["confidence"],
                "primary_support_kind": "strong" if left["confidence"] >= candidate else "weak",
            }
        )
    supported = {p["rotated_index"] for p in pairs}
    return {
        "version": 1,
        "semantics": "same_frame_original_and_180_candidate_evidence",
        "support_confidence": support,
        "candidate_confidence": candidate,
        "iou_threshold": overlap,
        "matching_policy": match["matching_policy"],
        "source_frame_and_rotation_verified_by_helper": False,
        "source_views_at_box_limit": [
            name
            for name, boxes in (("primary", primary), ("rotated", rotated))
            if len(boxes) == MAX_BOXES
        ],
        "geometric_pairs_are_unique_person_matches": False,
        "cross_view_scores_are_independent": False,
        "primary_boxes": original,
        "rotated_boxes_source_coordinates": mapped,
        "primary_strong_indices": [b["index"] for b in original if b["confidence"] >= candidate],
        "geometric_support_pairs": pairs,
        "rotated_strong_supported_indices": sorted(supported),
        "rotated_strong_unilateral_indices": [
            b["index"] for b in eligible if b["index"] not in supported
        ],
        "primary_below_candidate_indices": [
            b["index"] for b in original if b["confidence"] < candidate
        ],
        "rotated_below_candidate_indices": [
            b["index"] for b in mapped if b["confidence"] < candidate
        ],
        "within_view_overlap_candidates": checked["within_view_overlap_candidates"],
        "unilateral_candidates_suppressed": False,
        "presence": "unknown",
        "true_person_count": None,
        "baby_identity": "unknown",
        "baby_state": "unknown",
        "deployment_approved": False,
    }
