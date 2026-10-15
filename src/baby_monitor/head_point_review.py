"""Frame-bound visible-head point reviews, not identity or live-state grants.

Pure standard-library contracts. Human provenance and completeness are caller
declarations; point coverage is not box IoU accuracy, person absence, or proof
that a model recognized an age. Development counterexamples are not holdouts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime

_ID = re.compile(r"[A-Za-z0-9_-]{1,80}")
_SHA = re.compile(r"[0-9a-f]{64}")
MAX_ITEMS = 8
MAX_POINTS = 16


class HeadPointReviewError(ValueError):
    """Unsafe or mismatched offline review input; errors are path-free."""


def _require(condition: bool) -> None:
    if not condition:
        raise HeadPointReviewError("invalid head point review input")


def _fields(value: object, names: set[str]) -> dict:
    _require(type(value) is dict and set(value) == names)
    return value


def _id(value: object) -> str:
    _require(type(value) is str and _ID.fullmatch(value) is not None)
    return value


def _sha(value: object) -> str:
    _require(type(value) is str and _SHA.fullmatch(value) is not None)
    return value


def _unit(value: object) -> float:
    _require(type(value) in (int, float))
    try:
        valid = math.isfinite(value) and 0 <= value <= 1
    except (ValueError, OverflowError):
        valid = False
    _require(valid)
    return float(value)


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def parse_json(raw: bytes) -> object:
    _require(type(raw) is bytes and 0 < len(raw) <= 128 * 1024)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result)
            result[key] = value
        return result

    def constant(_value):
        raise HeadPointReviewError("invalid head point review JSON")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError):
        raise HeadPointReviewError("invalid head point review JSON") from None
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        _require(depth <= 32)
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
    return value


def validate_manifest(value: object) -> dict:
    value = _fields(
        value,
        {"version", "scope", "review_id", "selection_purpose", "blinding_scope", "items"},
    )
    _require(type(value["version"]) is int and value["version"] == 1)
    _require(value["scope"] == "frame_visible_head_points")
    _require(value["selection_purpose"] == "development_counterexamples")
    _require(value["blinding_scope"] == "page_hides_models_and_previous_answers")
    _id(value["review_id"])
    _require(type(value["items"]) is list and 1 <= len(value["items"]) <= MAX_ITEMS)
    ids, frames = set(), set()
    for item in value["items"]:
        item = _fields(
            item,
            {"item_id", "camera_id", "frame_sha256", "frame_received_at", "width", "height"},
        )
        _id(item["item_id"])
        _id(item["camera_id"])
        _sha(item["frame_sha256"])
        _require(item["item_id"] not in ids and item["frame_sha256"] not in frames)
        ids.add(item["item_id"])
        frames.add(item["frame_sha256"])
        for key in ("width", "height"):
            _require(type(item[key]) is int and 1 <= item[key] <= 4096)
        _require(item["width"] * item["height"] <= 4_000_000)
        stamp = item["frame_received_at"]
        _require(type(stamp) is str and len(stamp) <= 64)
        try:
            parsed = datetime.fromisoformat(stamp)
            _require(parsed.utcoffset() is not None and parsed.astimezone(UTC).isoformat() == stamp)
        except (ValueError, OverflowError):
            raise HeadPointReviewError("invalid head point review timestamp") from None
    return value


def validate_review(value: object, manifest: object) -> dict:
    manifest = validate_manifest(manifest)
    value = _fields(
        value,
        {"version", "scope", "manifest_sha256", "reviewer_id", "provenance", "answers"},
    )
    _require(type(value["version"]) is int and value["version"] == 1)
    _require(value["scope"] == manifest["scope"])
    _require(value["manifest_sha256"] == fingerprint(manifest))
    _id(value["reviewer_id"])
    _require(value["reviewer_id"].startswith("reviewer_"))
    _require(value["provenance"] in ("human_user_review", "synthetic"))
    _require(type(value["answers"]) is list and len(value["answers"]) == len(manifest["items"]))
    for answer, item in zip(value["answers"], manifest["items"], strict=True):
        answer = _fields(
            answer,
            {"item_id", "frame_sha256", "decision", "points", "inventory_complete", "reviewed"},
        )
        _require(answer["item_id"] == item["item_id"])
        _require(answer["frame_sha256"] == item["frame_sha256"])
        _require(answer["reviewed"] is True and type(answer["inventory_complete"]) is bool)
        _require(answer["decision"] in ("marked", "no_visible_head", "unknown"))
        points = answer["points"]
        _require(type(points) is list and len(points) <= MAX_POINTS)
        if answer["decision"] == "marked":
            _require(len(points) > 0)
        elif answer["decision"] == "no_visible_head":
            _require(not points and answer["inventory_complete"] is True)
        else:
            _require(not points and answer["inventory_complete"] is False)
        locations = set()
        for index, point in enumerate(points):
            point = _fields(point, {"point_id", "point_xy", "role"})
            _require(point["point_id"] == f"head_{index + 1:02}")
            _require(point["role"] in ("baby", "adult", "unknown"))
            xy = point["point_xy"]
            _require(type(xy) is list and len(xy) == 2)
            location = tuple(_unit(v) for v in xy)
            _require(location not in locations)
            locations.add(location)
    return value


def diagnose_point_coverage(manifest: object, review: object, predictions: object) -> dict:
    """Conservative containment correspondence, never a deployment decision.

    Zero boxes are required as explicit rows. Missing/reordered rows fail
    closed. A box covering two points, or a point in two boxes, is ambiguous,
    not separate-person localization. Unmatched boxes are not called FPs.
    """
    manifest = validate_manifest(manifest)
    review = validate_review(review, manifest)
    _require(type(predictions) is list and len(predictions) == len(manifest["items"]))
    rows = []
    for item, answer, prediction in zip(
        manifest["items"], review["answers"], predictions, strict=True
    ):
        prediction = _fields(prediction, {"item_id", "frame_sha256", "boxes"})
        _require(prediction["item_id"] == item["item_id"])
        _require(prediction["frame_sha256"] == item["frame_sha256"])
        _require(type(prediction["boxes"]) is list and len(prediction["boxes"]) <= 128)
        boxes = []
        for box in prediction["boxes"]:
            _require(type(box) is list and len(box) == 4)
            x1, y1, x2, y2 = (_unit(v) for v in box)
            _require(x1 < x2 and y1 < y2)
            boxes.append((x1, y1, x2, y2))
        links = [
            [
                index
                for index, (x1, y1, x2, y2) in enumerate(boxes)
                if x1 <= point["point_xy"][0] <= x2 and y1 <= point["point_xy"][1] <= y2
            ]
            for point in answer["points"]
        ]
        box_degrees = [
            sum(index in links_for_point for links_for_point in links)
            for index in range(len(boxes))
        ]
        details = []
        for point, candidates in zip(answer["points"], links, strict=True):
            state = (
                "point_not_covered"
                if not candidates
                else "unique_point_containment"
                if len(candidates) == 1 and box_degrees[candidates[0]] == 1
                else "ambiguous_point_containment"
            )
            details.append(
                {"point_id": point["point_id"], "reference_role": point["role"], "state": state}
            )
        rows.append(
            {
                "item_id": item["item_id"],
                "decision": answer["decision"],
                "inventory_complete_declared": answer["inventory_complete"],
                "candidate_count": len(boxes),
                "points": details,
                "candidates_without_marked_point": (
                    sum(degree == 0 for degree in box_degrees)
                    if answer["inventory_complete"]
                    else None
                ),
            }
        )
    return {
        "version": 1,
        "measurement": "visible_head_point_containment_diagnostic",
        "manifest_sha256": fingerprint(manifest),
        "review_sha256": fingerprint(review),
        "predictions_sha256": fingerprint(predictions),
        "human_provenance_declared": review["provenance"] == "human_user_review",
        "reviewer_authenticated": False,
        "rows": rows,
        "formal_box_accuracy": False,
        "age_recognition_evaluated": False,
        "whole_frame_person_absence_established": False,
        "temporal_ground_truth": False,
        "deployment_approved": False,
        "identity_or_live_state_authorized": False,
    }
