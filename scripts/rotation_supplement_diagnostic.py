"""Offline, fixed-policy rotated-box supplements; never detector or role facts.

Native boxes are kept verbatim in their own lane. A sub-box can be another
person OR another part of the same person: neither size nor containment proves
an extra subject. No model calls, media, I/O, tracking, runtime entry or labels.
"""

from __future__ import annotations

from scripts.individual_coverage_diagnostic import _sha
from scripts.secondary_view_diagnostic import _boxes, _iou

REVISION = "native_reserved_rotation_subbox_v1"
SETTINGS = {
    "confidence": 0.35,
    "seed_iou": 0.50,
    "rotated_overlap_iou": 0.70,
    "minimum_candidate_containment": 0.90,
    "maximum_candidate_seed_area_ratio": 0.60,
}
MAX_BOXES = 100


def _area(box):
    a, b, c, d = box["bbox_xyxy"]
    return (c - a) * (d - b)


def _contained(candidate, seed):
    a, b, c, d = candidate["bbox_xyxy"]
    e, f, g, h = seed["bbox_xyxy"]
    intersection = max(0, min(c, g) - max(a, e)) * max(0, min(d, h) - max(b, f))
    return intersection / _area(candidate)


def _center_inside(candidate, native):
    a, b, c, d = candidate["bbox_xyxy"]
    e, f, g, h = native["bbox_xyxy"]
    return e <= (a + c) / 2 <= g and f <= (b + d) / 2 <= h


def diagnose_supplements(primary, rotated_source, *, source_sha256, rotated_source_sha256):
    """Compare boxes already mapped into the SAME exact original BGR coordinates.

    The hash equality is only a declared binding. A caller must verify actual
    pixels and the inverse transform; this pure helper cannot authenticate them.
    Reserve only degree-one IoU pairs, never choose a greedy winner in a crowd.
    Reject overlap/covered/unanchored sub-boxes, without deleting raw evidence.
    Fixed development settings are not an automatically selected operating point.
    """
    if _sha(source_sha256) != _sha(rotated_source_sha256):
        raise ValueError("rotation supplement source mismatch")
    if any(type(v) is not list or len(v) > MAX_BOXES for v in (primary, rotated_source)):
        raise ValueError("rotation supplement requires bounded lists")
    native, rotated = _boxes(primary, "primary"), _boxes(rotated_source, "rotated_source")
    seeds = [b for b in native if b["confidence"] >= SETTINGS["confidence"]]
    eligible = [b for b in rotated if b["confidence"] >= SETTINGS["confidence"]]
    left = {b["index"]: [] for b in seeds}
    right = {b["index"]: [] for b in eligible}
    for n in seeds:
        for r in eligible:
            if _iou(n, r) >= SETTINGS["seed_iou"]:
                left[n["index"]].append(r["index"])
                right[r["index"]].append(n["index"])
    overlaps = {
        r["index"]
        for r in eligible
        for other in eligible
        if r["index"] != other["index"] and _iou(r, other) >= SETTINGS["rotated_overlap_iou"]
    }
    reserved = {
        n: rs[0]
        for n, rs in left.items()
        if len(rs) == 1 and len(right[rs[0]]) == 1 and rs[0] not in overlaps
    }
    rows = []
    for r in rotated:
        index = r["index"]
        anchors = []
        if r["confidence"] < SETTINGS["confidence"]:
            status = "below_confidence"
        elif index in overlaps:
            status = "rotated_overlap_unresolved"
        elif index in reserved.values():
            status = "reserved_native_correspondence"
        elif right[index]:
            status = "native_iou_correspondence_unresolved"
        elif any(n["index"] not in reserved and _center_inside(r, n) for n in seeds):
            status = "unreserved_native_center_coverage"
        else:
            anchors = [
                n["index"]
                for n in seeds
                if n["index"] in reserved
                and _contained(r, n) >= SETTINGS["minimum_candidate_containment"]
                and _area(r) / _area(n) <= SETTINGS["maximum_candidate_seed_area_ratio"]
            ]
            status = (
                "pending_subject_reference"
                if len(anchors) == 1
                else "ambiguous_native_anchor"
                if anchors
                else "no_qualified_native_anchor"
            )
        rows.append({"rotated_index": index, "status": status, "native_anchor_indices": anchors})
    return {
        "revision": REVISION,
        "settings": dict(SETTINGS),
        "source_sha256": source_sha256,
        "native_boxes": native,
        "rotated_boxes_source_coordinates": rotated,
        "declared_source_hashes_equal": True,
        "pixels_and_transform_verified_by_helper": False,
        "reserved_pairs": [
            {"native_index": n, "rotated_index": r} for n, r in sorted(reserved.items())
        ],
        "rows": rows,
        "pending_rotated_indices": [
            r["rotated_index"] for r in rows if r["status"] == "pending_subject_reference"
        ],
        "native_boxes_modified": False,
        "raw_candidates_suppressed": False,
        "new_person_count": None,
        "presence": "unknown",
        "baby_identity": "unknown",
        "roles_propagated": False,
        "accuracy_measured": False,
        "production_changed": False,
        "deployment_approved": False,
    }
