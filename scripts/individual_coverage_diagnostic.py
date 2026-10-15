"""Frame-local weak box-assignment audit; never an official accuracy evaluator.

Reviews describe which anonymous visible subjects a candidate separately locates,
not simply which subjects intersect its rectangle. Geometry alone cannot supply
these labels. No media, model, file I/O, identity, tracking or runtime decisions.
Publisher/human ground-truth evaluation remains in detector_evaluation unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re

from scripts.household_debug_common import safe_id
from scripts.secondary_view_diagnostic import MAX_BOXES_PER_VIEW, _boxes, _unit_number


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("frame/candidate binding requires a lowercase SHA-256")
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _candidates(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)) or len(value) > MAX_BOXES_PER_VIEW:
        raise ValueError("candidates must be a bounded list/tuple")
    ids = []
    boxes = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"detection_id", "bbox_xyxy", "confidence"}:
            raise ValueError("candidate requires detection_id, bbox_xyxy and confidence")
        ids.append(safe_id(item["detection_id"]))
        boxes.append({key: item[key] for key in ("bbox_xyxy", "confidence")})
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate IDs")
    return [
        {"detection_id": identifier, "bbox_xyxy": box["bbox_xyxy"], "confidence": box["confidence"]}
        for identifier, box in zip(ids, _boxes(boxes, "candidates"), strict=True)
    ]


def candidate_fingerprint(frame_sha256: str, candidates: object) -> str:
    """Bind exact ordered, normalized candidate values to caller-supplied frame hash.

    This does not read or authenticate the frame or prove the model used it.
    Caller must verify source provenance before creating the review.
    """
    return _digest({"frame_sha256": _sha(frame_sha256), "candidates": _candidates(candidates)})


def diagnose_individual_coverage(
    frame_sha256: str,
    candidates: object,
    review: object,
    *,
    minimum_confidence: float,
) -> dict[str, object]:
    """Audit separate coverage under explicit synthetic or assistant-weak labels.

    Review fields: frame_sha256, candidates_sha256, provenance (synthetic or
    assistant_weak), reference_ids, reference_inventory_complete, assignments.
    Anonymous reference IDs apply ONLY within this frame. Each assignment has
    detection_id, state, reference_ids. States: single_subject (one reference),
    multi_subject (>=2, not separately localized), background/unknown (none).
    Omitted assignments become unknown, never background. Unknown candidates can
    refer to any subject. Multiple single_subject boxes for the same reference
    count once and remain explicit duplicate-assignment candidates, not deleted.

    'not_separately_localized' is relative to the supplied review and threshold,
    not a confirmed real-world miss. Incomplete inventories cannot establish
    whole-frame coverage. Confidence filtering never drops evidence from output.
    No precision/recall, verified coverage, baby identity or safety verdict.
    """
    frame_sha256 = _sha(frame_sha256)
    boxes = _candidates(candidates)
    minimum = _unit_number(minimum_confidence, "minimum_confidence")
    fingerprint = candidate_fingerprint(frame_sha256, boxes)
    fields = {
        "frame_sha256",
        "candidates_sha256",
        "provenance",
        "reference_ids",
        "reference_inventory_complete",
        "assignments",
    }
    if not isinstance(review, dict) or set(review) != fields:
        raise ValueError("invalid individual review fields")
    if (
        _sha(review["frame_sha256"]) != frame_sha256
        or _sha(review["candidates_sha256"]) != fingerprint
    ):
        raise ValueError("individual review frame/candidates binding mismatch")
    if review["provenance"] not in ("synthetic", "assistant_weak"):
        raise ValueError("individual diagnostic does not accept ground-truth provenance")
    if type(review["reference_inventory_complete"]) is not bool:
        raise ValueError("reference inventory completeness must be boolean")
    refs = review["reference_ids"]
    if not isinstance(refs, (list, tuple)) or len(refs) > MAX_BOXES_PER_VIEW:
        raise ValueError("reference IDs must be a bounded list/tuple")
    refs = [safe_id(item) for item in refs]
    if len(set(refs)) != len(refs):
        raise ValueError("duplicate reference IDs")
    assignments = review["assignments"]
    if not isinstance(assignments, (list, tuple)) or len(assignments) > len(boxes):
        raise ValueError("assignments must be a bounded list/tuple")
    candidate_ids = {box["detection_id"] for box in boxes}
    labels = {}
    for item in assignments:
        if not isinstance(item, dict) or set(item) != {"detection_id", "state", "reference_ids"}:
            raise ValueError("invalid assignment fields")
        identifier = safe_id(item["detection_id"])
        if identifier not in candidate_ids or identifier in labels:
            raise ValueError("assignment has unknown or duplicate candidate ID")
        state, subjects = item["state"], item["reference_ids"]
        if state not in ("single_subject", "multi_subject", "background", "unknown"):
            raise ValueError("invalid assignment state")
        if not isinstance(subjects, (list, tuple)) or len(subjects) > len(refs):
            raise ValueError("assignment reference IDs must be bounded")
        subjects = [safe_id(subject) for subject in subjects]
        if len(set(subjects)) != len(subjects) or not set(subjects) <= set(refs):
            raise ValueError("unknown or duplicate assignment reference")
        if (
            (state == "single_subject" and len(subjects) != 1)
            or (state == "multi_subject" and len(subjects) < 2)
            or (state in ("background", "unknown") and subjects)
        ):
            raise ValueError("assignment state/reference cardinality mismatch")
        labels[identifier] = {"state": state, "reference_ids": subjects}
    rows = []
    for box in boxes:
        identifier = box["detection_id"]
        rows.append(
            {
                **box,
                **labels.get(identifier, {"state": "unknown", "reference_ids": []}),
                "explicitly_reviewed": identifier in labels,
                "eligible": box["confidence"] >= minimum,
            }
        )
    active = [row for row in rows if row["eligible"]]
    unresolved = [row["detection_id"] for row in active if row["state"] == "unknown"]
    singles = {
        ref: [
            row["detection_id"]
            for row in active
            if row["state"] == "single_subject" and ref in row["reference_ids"]
        ]
        for ref in refs
    }
    coverage = [
        {
            "reference_id": ref,
            "single_subject_candidate_ids": singles[ref],
            "status": (
                "separately_localized"
                if singles[ref]
                else "unknown"
                if unresolved
                else "not_separately_localized"
            ),
        }
        for ref in refs
    ]
    all_references_covered = bool(refs) and all(singles.values())
    complete = review["reference_inventory_complete"]
    return {
        "schema_version": "individual-coverage-diagnostic-v1",
        "frame_sha256": frame_sha256,
        "candidates_sha256": fingerprint,
        "review_sha256": _digest(review),
        "provenance": review["provenance"],
        "offline_diagnostic_only": True,
        "model_generated_labels": review["provenance"] == "assistant_weak",
        "human_verified": False,
        "ground_truth": False,
        "official_accuracy_claim": False,
        "deployment_approved": False,
        "separate_individual_coverage_verified": False,
        "identity_semantics": "anonymous_within_frame_only",
        "presence": "unknown",
        "true_person_count": None,
        "minimum_confidence": minimum,
        "reference_inventory_complete": complete,
        "whole_frame_review_status": (
            "unknown_inventory"
            if not complete
            else "empty_reference_inventory"
            if not refs
            else "all_references_separately_localized"
            if all_references_covered
            else "unknown_assignment"
            if unresolved
            else "not_all_references_separately_localized"
        ),
        "separately_localized_reference_count": sum(bool(ids) for ids in singles.values()),
        "reference_coverage": coverage,
        "candidates": rows,
        "unknown_candidate_ids": unresolved,
        "multi_subject_candidate_ids": [
            row["detection_id"] for row in active if row["state"] == "multi_subject"
        ],
        "duplicate_assignment_groups": [
            {"reference_id": ref, "candidate_ids": ids}
            for ref, ids in singles.items()
            if len(ids) > 1
        ],
    }
