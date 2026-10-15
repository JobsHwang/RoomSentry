"""Score a complete offline precision/shape matrix, not baby accuracy.

Only metadata is accepted. References are caller declarations; camera identity,
age, box correspondence, independent holdout and deployment are not established.
No model/media I/O or installed-runtime changes occur here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

ARMS = (
    ("fp16_640", (384, 640), "float16"),
    ("fp32_640", (384, 640), "float32"),
    ("fp16_960", (576, 960), "float16"),
    ("fp32_960", (576, 960), "float32"),
)
ROTATIONS = (0, 180)
SOURCE_CONFIDENCE = 0.1
MAX_FRAMES = 32
MAX_BOXES = 100
_ID = re.compile(r"[A-Za-z0-9_-]{1,80}")
_SHA = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("invalid YOLO input diagnostic matrix")


def _hash(value: object) -> str:
    _require(type(value) is str and _SHA.fullmatch(value) is not None)
    return value


def _number(value: object, *, low: float, high: float) -> float:
    _require(type(value) in (int, float))
    try:
        valid = math.isfinite(value) and low <= value <= high
    except (ValueError, OverflowError):
        valid = False
    _require(valid)
    return float(value)


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def score(frames: object, records: object, thresholds: object) -> dict:
    """Require every frame/arm/rotation, including explicit zero-box results.

    Recorded tensor shape/dtype and source/rotated pixel hashes must match the
    declared plan. Hash matching does not authenticate execution or references.
    Threshold sweeps describe retained candidates only. Saturation is explicit;
    the helper cannot recover top-k, preprocessing or detector omissions.
    """
    _require(type(frames) is list and 1 <= len(frames) <= MAX_FRAMES)
    expected_frames, source_hashes = {}, set()
    for frame in frames:
        _require(
            type(frame) is dict
            and set(frame)
            == {"frame_id", "frame_sha256", "pixels_sha256_by_rotation", "reference_presence"}
        )
        identifier = frame["frame_id"]
        _require(type(identifier) is str and _ID.fullmatch(identifier) is not None)
        digest = _hash(frame["frame_sha256"])
        _require(identifier not in expected_frames and digest not in source_hashes)
        source_hashes.add(digest)
        pixels = frame["pixels_sha256_by_rotation"]
        _require(type(pixels) is dict and set(pixels) == {"0", "180"})
        for digest in pixels.values():
            _hash(digest)
        _require(frame["reference_presence"] in ("present", "absent", "unknown"))
        expected_frames[identifier] = frame
    _require(type(thresholds) is list and 1 <= len(thresholds) <= 16)
    thresholds = [_number(t, low=SOURCE_CONFIDENCE, high=1) for t in thresholds]
    _require(thresholds == sorted(set(thresholds)) and thresholds[0] == SOURCE_CONFIDENCE)
    expected_cells = len(frames) * len(ARMS) * len(ROTATIONS)
    _require(type(records) is list and len(records) == expected_cells)
    settings = {name: (shape, dtype) for name, shape, dtype in ARMS}
    cells = {}
    for record in records:
        _require(
            type(record) is dict
            and set(record)
            == {
                "frame_id",
                "frame_sha256",
                "arm",
                "rotation",
                "boxes",
                "elapsed_ms",
                "input_shape",
                "input_dtype",
                "tensor_sha256",
                "pixels_sha256",
                "head_end2end",
            }
        )
        identifier, arm, rotation = record["frame_id"], record["arm"], record["rotation"]
        _require(type(identifier) is str and identifier in expected_frames)
        _require(type(arm) is str and arm in settings)
        _require(type(rotation) is int and rotation in ROTATIONS)
        key = (identifier, arm, rotation)
        _require(key not in cells)
        frame = expected_frames[identifier]
        _require(_hash(record["frame_sha256"]) == frame["frame_sha256"])
        _require(
            _hash(record["pixels_sha256"]) == frame["pixels_sha256_by_rotation"][str(rotation)]
        )
        shape, dtype = settings[arm]
        actual_shape = record["input_shape"]
        _require(
            type(actual_shape) is list
            and len(actual_shape) == 4
            and all(type(v) is int for v in actual_shape)
            and actual_shape == [1, 3, *shape]
        )
        _require(type(record["input_dtype"]) is str and record["input_dtype"] == dtype)
        _require(record["head_end2end"] is True)
        _hash(record["tensor_sha256"])
        _number(record["elapsed_ms"], low=0, high=120_000)
        boxes = record["boxes"]
        _require(type(boxes) is list and len(boxes) <= MAX_BOXES)
        for box in boxes:
            _require(type(box) is dict and set(box) == {"confidence", "bbox_xyxy"})
            _number(box["confidence"], low=SOURCE_CONFIDENCE, high=1)
            xy = box["bbox_xyxy"]
            _require(type(xy) is list and len(xy) == 4)
            x1, y1, x2, y2 = [_number(v, low=0, high=1) for v in xy]
            _require(x1 < x2 and y1 < y2)
        cells[key] = record
    summaries = []
    for arm, _, _ in ARMS:
        for rotation in ROTATIONS:
            for threshold in thresholds:
                confusion = dict.fromkeys(("tp", "fn", "fp", "tn"), 0)
                candidates = 0
                saturation = []
                unknown = 0
                for identifier, frame in expected_frames.items():
                    row = cells[identifier, arm, rotation]
                    count = sum(box["confidence"] >= threshold for box in row["boxes"])
                    candidates += count
                    if len(row["boxes"]) == MAX_BOXES:
                        saturation.append(identifier)
                    label = frame["reference_presence"]
                    if label == "unknown":
                        unknown += 1
                        continue
                    key = (
                        ("tp" if count else "fn")
                        if label == "present"
                        else ("fp" if count else "tn")
                    )
                    confusion[key] += 1
                summaries.append(
                    {
                        "arm": arm,
                        "rotation": rotation,
                        "threshold": threshold,
                        "retained_candidate_boxes": candidates,
                        "reference_presence": confusion,
                        "unknown_reference_frames": unknown,
                        "saturated_source_frames": saturation,
                    }
                )
    precision_changes = []
    for scale in (640, 960):
        for rotation in ROTATIONS:
            for threshold in thresholds:
                changed = []
                for identifier in expected_frames:
                    counts = [
                        sum(
                            b["confidence"] >= threshold
                            for b in cells[identifier, f"fp{bits}_{scale}", rotation]["boxes"]
                        )
                        for bits in (16, 32)
                    ]
                    if counts[0] != counts[1]:
                        changed.append(
                            {
                                "frame_id": identifier,
                                "fp16_boxes": counts[0],
                                "fp32_boxes": counts[1],
                            }
                        )
                precision_changes.append(
                    {
                        "scale": scale,
                        "rotation": rotation,
                        "threshold": threshold,
                        "changed_frames": changed,
                    }
                )
    return {
        "version": 1,
        "semantics": "offline_input_settings_candidate_diagnostic",
        "frames": len(frames),
        "cells": len(cells),
        "frames_sha256": fingerprint(frames),
        "records_sha256": fingerprint(records),
        "source_confidence": SOURCE_CONFIDENCE,
        "summaries": summaries,
        "precision_candidate_count_changes": precision_changes,
        "reference_declarations_authenticated": False,
        "independent_holdout": False,
        "formal_box_accuracy": False,
        "baby_accuracy_verified": False,
        "deployment_approved": False,
        "baby_location": "unknown",
        "baby_state": "unknown",
    }
