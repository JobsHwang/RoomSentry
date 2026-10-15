"""Bounded, pure weak-presence comparisons of two prediction heads.

No file, media, model, or network access occurs here. Callers bind exact stored
head outputs and the weak-label source; this helper validates matching metadata,
not media bytes, annotation provenance, or ground truth. Counts never establish
separate localization of multiple people and never select a deployment winner.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

MAX_ROWS = 32
MAX_BOXES = 1000
MAX_THRESHOLDS = 32
_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _fields(value: object, expected: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("prediction head diagnostic fields are invalid")
    return value


def _identifier(value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError("prediction head diagnostic identifier is invalid")
    return value


def _sha_value(value: object) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError("prediction head diagnostic fingerprint is invalid")
    return value


def _unit(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("prediction head diagnostic number is invalid")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError("prediction head diagnostic number is invalid") from None
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("prediction head diagnostic number is invalid")
    return number


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _labels(value: object) -> dict[str, dict]:
    if not isinstance(value, tuple) or not 1 <= len(value) <= MAX_ROWS:
        raise ValueError("prediction head diagnostic weak label count is invalid")
    result = {}
    frames = set()
    for item in value:
        item = _fields(item, {"review_id", "frame_sha256", "presence"})
        review = _identifier(item["review_id"])
        frame = _sha_value(item["frame_sha256"])
        if item["presence"] not in ("present", "absent", "unknown"):
            raise ValueError("prediction head diagnostic weak presence is invalid")
        if review in result or frame in frames:
            raise ValueError("prediction head diagnostic weak frames must be unique")
        result[review] = item
        frames.add(frame)
    return result


def _branch(value: object, labels: dict[str, dict], label_sha: str) -> dict:
    value = _fields(value, {"branch_id", "labels_source_sha256", "source_confidence", "rows"})
    branch_id = _identifier(value["branch_id"])
    if _sha_value(value["labels_source_sha256"]) != label_sha:
        raise ValueError("prediction head diagnostic weak label source binding differs")
    floor = _unit(value["source_confidence"])
    rows = value["rows"]
    if not isinstance(rows, tuple) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("prediction head diagnostic row count is invalid")
    parsed = {}
    weights = {}
    boxes_total = 0
    for row in rows:
        row = _fields(row, {"model_id", "model_sha256", "review_id", "frame_sha256", "boxes"})
        model = _identifier(row["model_id"])
        weight = _sha_value(row["model_sha256"])
        review = _identifier(row["review_id"])
        frame = _sha_value(row["frame_sha256"])
        if review not in labels or labels[review]["frame_sha256"] != frame:
            raise ValueError("prediction head diagnostic frame and weak review binding differs")
        if (model, review) in parsed:
            raise ValueError("prediction head diagnostic model/frame rows must be unique")
        if model in weights and weights[model] != weight:
            raise ValueError("prediction head diagnostic model weights are inconsistent")
        weights[model] = weight
        boxes = row["boxes"]
        if not isinstance(boxes, tuple):
            raise ValueError("prediction head diagnostic boxes must be a tuple")
        boxes_total += len(boxes)
        if boxes_total > MAX_BOXES:
            raise ValueError("prediction head diagnostic total box count exceeds the bound")
        confidences = []
        for box in boxes:
            box = _fields(box, {"confidence", "bbox_xyxy"})
            confidence = _unit(box["confidence"])
            bbox = box["bbox_xyxy"]
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                raise ValueError("prediction head diagnostic box coordinates are invalid")
            x1, y1, x2, y2 = map(_unit, bbox)
            if x1 >= x2 or y1 >= y2 or (x2 - x1) * (y2 - y1) == 0:
                raise ValueError("prediction head diagnostic box area must be positive")
            if confidence < floor:
                raise ValueError("prediction head diagnostic box is below its stored source floor")
            confidences.append(confidence)
        parsed[model, review] = {
            "confidences": confidences,
            "frame_sha256": frame,
            "model_sha256": weight,
        }
    # Every model is measured against the entire fixed review frame set. Neither
    # branch nor model can silently drop hard frames by taking an intersection.
    for model in weights:
        if {review for row_model, review in parsed if row_model == model} != set(labels):
            raise ValueError("prediction head diagnostic model frame set is incomplete")
    return {"branch_id": branch_id, "floor": floor, "rows": parsed, "weights": weights}


def diagnose_prediction_heads(
    left: dict,
    right: dict,
    *,
    weak_labels: tuple[dict, ...],
    label_source_sha256: str,
    thresholds: tuple[float, ...],
) -> dict[str, object]:
    """Compare fixed raw boxes only against assistant/other weak presence labels.

    Branch envelopes contain ``branch_id``, ``labels_source_sha256``,
    ``source_confidence``, and tuple ``rows``. Each row contains model_id,
    model_sha256, review_id, frame_sha256, and tuple boxes (confidence, bbox_xyxy).
    Each weak label has review_id, frame_sha256, presence. No person-count label
    is accepted. Both branches must cover every model/review pair identically;
    matching keys and SHA values do not independently establish provenance.
    """
    label_sha = _sha_value(label_source_sha256)
    labels = _labels(weak_labels)
    first, second = _branch(left, labels, label_sha), _branch(right, labels, label_sha)
    if first["branch_id"] == second["branch_id"]:
        raise ValueError("prediction head diagnostic branch identifiers must differ")
    if set(first["rows"]) != set(second["rows"]):
        raise ValueError("prediction head diagnostic branch model/frame sets differ")
    for key, row in first["rows"].items():
        other = second["rows"][key]
        if (row["frame_sha256"], row["model_sha256"]) != (
            other["frame_sha256"],
            other["model_sha256"],
        ):
            raise ValueError("prediction head diagnostic branch frame/weight bindings differ")
    if not isinstance(thresholds, tuple) or not 1 <= len(thresholds) <= MAX_THRESHOLDS:
        raise ValueError("prediction head diagnostic threshold count is invalid")
    levels = tuple(_unit(value) for value in thresholds)
    if len(set(levels)) != len(levels) or min(levels) < max(first["floor"], second["floor"]):
        raise ValueError("prediction head diagnostic thresholds repeat or exceed stored evidence")
    models = []
    for model in sorted(first["weights"]):
        points = []
        for threshold in levels:
            branches = []
            for branch in (first, second):
                reference_counts = {kind: 0 for kind in ("present", "absent", "unknown")}
                any_box_counts = reference_counts.copy()
                frames = []
                stored_count = box_count = 0
                for review in sorted(labels):
                    row = branch["rows"][model, review]
                    count = sum(confidence >= threshold for confidence in row["confidences"])
                    presence = labels[review]["presence"]
                    reference_counts[presence] += 1
                    any_box_counts[presence] += count > 0
                    stored_count += len(row["confidences"])
                    box_count += count
                    frames.append(
                        {
                            "review_id": review,
                            "frame_sha256": row["frame_sha256"],
                            "weak_presence": presence,
                            "box_count_at_threshold": count,
                        }
                    )
                branches.append(
                    {
                        "branch_id": branch["branch_id"],
                        "source_confidence": branch["floor"],
                        "stored_raw_box_count": stored_count,
                        "box_count_at_threshold": box_count,
                        "weak_present_frame_count": reference_counts["present"],
                        "weak_absent_frame_count": reference_counts["absent"],
                        "weak_unknown_frame_count": reference_counts["unknown"],
                        "present_any_box_count": any_box_counts["present"],
                        "present_no_box_count": reference_counts["present"]
                        - any_box_counts["present"],
                        "absent_any_box_count": any_box_counts["absent"],
                        "absent_no_box_count": reference_counts["absent"]
                        - any_box_counts["absent"],
                        "unknown_any_box_count": any_box_counts["unknown"],
                        "unknown_no_box_count": reference_counts["unknown"]
                        - any_box_counts["unknown"],
                        "frames": frames,
                    }
                )
            points.append({"confidence_threshold": threshold, "branches": branches})
        models.append(
            {"model_id": model, "model_sha256": first["weights"][model], "points": points}
        )
    report = {
        "schema_version": "prediction-head-weak-presence-diagnostic-v1",
        "left_input_sha256": _hash(left),
        "right_input_sha256": _hash(right),
        "weak_labels_input_sha256": _hash(weak_labels),
        "labels_source_sha256": label_sha,
        "fixed_review_frame_count": len(labels),
        "model_count": len(models),
        "models": models,
        "offline_diagnostic_only": True,
        "weak_presence_consistency_only": True,
        "label_source_bytes_verified_by_helper": False,
        "media_bytes_verified_by_helper": False,
        "model_weights_verified_by_helper": False,
        "ground_truth": False,
        "accuracy_evaluated": False,
        "box_localization_evaluated": False,
        "separate_person_coverage_verified": False,
        "baby_identity_evaluated": False,
        "deployment_approved": False,
        "winner_selected": False,
        "source_exhaustion_verified": False,
        "paths_disclosed": False,
    }
    return {**report, "report_sha256": _hash(report)}
