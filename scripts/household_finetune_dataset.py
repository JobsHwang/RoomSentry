"""Private fine-tune preparation guards, not a detector or an automatic labeler.

Keep role annotations for stratified evaluation; export only person boxes for
the first-layer detector. Never treat a model's zero boxes as a negative label.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime

REVISION = "household_finetune_day_separation_v1"


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("exact lowercase SHA256 required")
    return value


def _time(value):
    if not isinstance(value, str):
        raise ValueError("capture time required")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timezone required")
    return result.astimezone(UTC)


def split_items(items, comparison_sha256s, independent_negative_sha256s=()):
    """Protect entire comparison dates; reserve latest other day for validation.

    This is a coarse leakage guard, not proof of visual/identity independence.
    Inputs are metadata only; review status and labels are never inferred here.
    """
    if not isinstance(items, list) or not 0 < len(items) <= 100_000:
        raise ValueError("bounded nonempty inventory required")
    comparisons = {_sha(x) for x in comparison_sha256s}
    negatives = {_sha(x) for x in independent_negative_sha256s}
    if len(comparisons) != 3 or comparisons & negatives:
        raise ValueError("three disjoint fixed comparison frames required")
    seen_ids, by_png = set(), {}
    for row in items:
        identifier = row["item_id"]
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,96}", identifier) is None
            or identifier in seen_ids
        ):
            raise ValueError("unique safe item ids required")
        seen_ids.add(identifier)
        _time(row["captured_at"])
        png = _sha(row["png_sha256"])
        _sha(row["bgr_sha256"])
        if png in by_png:
            raise ValueError("deduplicate exact source images before partitioning")
        by_png[png] = row
    if not comparisons <= by_png.keys():
        raise ValueError("all fixed comparison frames must be present")
    comparison_days = {_time(by_png[s]["captured_at"]).date() for s in comparisons}
    available_days = sorted(
        {_time(row["captured_at"]).date() for row in items if row["png_sha256"] not in negatives}
        - comparison_days
    )
    validation_day = available_days[-1] if available_days else None
    assigned = []
    for source in items:
        row = copy.deepcopy(source)
        png, day = row["png_sha256"], _time(row["captured_at"]).date()
        row["split"] = (
            "comparison"
            if png in comparisons
            else "independent_negative"
            if png in negatives
            else "quarantine"
            if day in comparison_days
            else "validation"
            if day == validation_day
            else "train"
        )
        row["training_ready"] = False
        row["annotation_status"] = "pending"
        row["negative_confirmed"] = False
        assigned.append(row)
    # Same decoded pixels under a different PNG name must not cross splits.
    pixel_splits = {}
    for row in assigned:
        pixel_splits.setdefault(row["bgr_sha256"], set()).add(row["split"])
    for row in assigned:
        if len(pixel_splits[row["bgr_sha256"]]) > 1 and row["split"] in {"train", "validation"}:
            row["split"] = "quarantine"
            row["quarantine_reason"] = "decoded_pixel_overlap_with_other_split"
    return {
        "revision": REVISION,
        "comparison_dates_utc": sorted(str(d) for d in comparison_days),
        "validation_date_utc": str(validation_day) if validation_day else None,
        "items": assigned,
        "split_counts": dict(Counter(row["split"] for row in assigned)),
        "training_ready": False,
        "visual_near_duplicate_audit_complete": False,
        "accuracy_evaluated": False,
    }


def person_labels(item, annotation, *, allow_assistant_training=False):
    """Return YOLO class-0 rows only after complete visual inventory review.

    Assistant labels may be explicitly admitted to an experimental training
    split, but never silently upgraded to human truth or evaluation labels.
    """
    if item.get("split") not in {"train", "validation", "comparison", "independent_negative"}:
        raise ValueError("quarantined inputs cannot be exported")
    if annotation.get("png_sha256") != _sha(item["png_sha256"]):
        raise ValueError("annotation must bind the exact original PNG")
    if annotation.get("bgr_sha256") != _sha(item["bgr_sha256"]):
        raise ValueError("annotation must bind the decoded pixels")
    if (
        annotation.get("inventory_complete") is not True
        or annotation.get("visually_reviewed") is not True
        or annotation.get("uncertain_regions") != []
    ):
        raise ValueError("complete unambiguous visible-person inventory required")
    provenance = annotation.get("provenance")
    if provenance == "assistant_visual_review":
        if item["split"] != "train" or allow_assistant_training is not True:
            raise ValueError("assistant labels are experimental training only")
        if (
            annotation.get("human_verified") is not False
            or annotation.get("ground_truth") is not False
        ):
            raise ValueError("assistant labels are not human truth")
    elif provenance == "human_visual_review":
        if (
            annotation.get("human_verified") is not True
            or annotation.get("ground_truth") is not True
        ):
            raise ValueError("explicit human reference required")
        if re.fullmatch(r"reviewer_[A-Za-z0-9_-]{1,64}", annotation.get("reviewer_id", "")) is None:
            raise ValueError("anonymous reviewer required")
    else:
        raise ValueError("model predictions and pending annotations cannot be exported")
    people = annotation.get("people")
    if type(people) is not list or len(people) > 16:
        raise ValueError("bounded person list required")
    if not people and annotation.get("negative_kind") != "no_visible_person":
        raise ValueError("empty YOLO output is not an explicitly reviewed negative")
    if people and annotation.get("negative_kind") not in (None, "adult_only"):
        raise ValueError("person labels conflict with negative decision")
    if annotation.get("negative_kind") == "adult_only" and any(
        p.get("role") != "adult" for p in people
    ):
        raise ValueError("adult-only negative requires all visible people to be adults")
    labels, boxes = [], set()
    for person in people:
        if person.get("role") not in {"baby", "adult", "unknown"}:
            raise ValueError("role must not encode a personal identity")
        if person.get("box_scope") != "visible_body" or person.get("occlusion") not in {
            "none",
            "partial",
        }:
            raise ValueError("visible-body geometry required, not guessed hidden anatomy")
        box = person.get("bbox_xyxy")
        if (
            type(box) is not list
            or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= 1
            or not 0 <= box[1] < box[3] <= 1
        ):
            raise ValueError("finite nonempty normalized body box required")
        if tuple(box) in boxes:
            raise ValueError("exact duplicate body annotations rejected")
        boxes.add(tuple(box))
        labels.append(
            "0 "
            + " ".join(
                format(v, ".10f")
                for v in (
                    (box[0] + box[2]) / 2,
                    (box[1] + box[3]) / 2,
                    box[2] - box[0],
                    box[3] - box[1],
                )
            )
        )
    return "\n".join(labels) + ("\n" if labels else "")


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def training_labels(item, annotation, *, training_authorized, allow_assistant_training=False):
    """The training export entry point rejects every reserved split."""
    if training_authorized is not True or item.get("split") != "train":
        raise ValueError("explicit authorization and training split required")
    return person_labels(item, annotation, allow_assistant_training=allow_assistant_training)
