"""Pure geometry for an optional head-only ONNX diagnostic, not person facts.

The admitted export emits scores[N,1] and batch/class/y1/x1/y2/x2[N,6]
in fixed-input pixels. Preprocessing must directly resize the entire oriented
image; letterboxed/cropped tensors require a different explicit transform.
No third-party implementation, downloader, or model dependency is included.
"""

from __future__ import annotations

import math

from baby_monitor.frame_transform import inverse_normalized_box, validate_rotation_degrees


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def decode_head_rows(
    scores: list,
    boxes: list,
    *,
    input_width: int,
    input_height: int,
    rotation_degrees: int = 0,
    confidence: float = 0.35,
) -> list[dict]:
    """Map direct-resize head boxes back to normalized original-image space.

    Clipping is limited to valid finite model coordinates. Invalid envelopes,
    class/batch IDs, scores, or inverted boxes fail closed, never become empty
    successful detections. Out-of-frame positive-area boxes are discarded.
    """
    validate_rotation_degrees(rotation_degrees)
    if any(type(v) is not int or not 1 <= v <= 8192 for v in (input_width, input_height)):
        raise ValueError("invalid model input size")
    if not _finite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("invalid head threshold")
    if type(scores) is not list or type(boxes) is not list or len(scores) != len(boxes):
        raise ValueError("invalid head output arrays")
    if len(scores) > 4096:
        raise ValueError("too many head rows")
    result = []
    for score_row, box in zip(scores, boxes, strict=True):
        if type(score_row) is not list or len(score_row) != 1:
            raise ValueError("invalid score row")
        score = score_row[0]
        if not _finite(score) or not 0 <= score <= 1:
            raise ValueError("invalid head score")
        if type(box) is not list or len(box) != 6 or not all(_finite(v) for v in box):
            raise ValueError("invalid head box row")
        batch, category, y1, x1, y2, x2 = box
        if batch != 0 or category != 0 or x1 >= x2 or y1 >= y2:
            raise ValueError("unexpected head class, batch or geometry")
        left, right = max(0, x1) / input_width, min(input_width, x2) / input_width
        top, bottom = max(0, y1) / input_height, min(input_height, y2) / input_height
        if score < confidence or left >= right or top >= bottom:
            continue
        mapped = inverse_normalized_box(
            (left, top, right, bottom), rotation_degrees=rotation_degrees
        )
        result.append({"kind": "head", "confidence": score, "bbox_xyxy": list(mapped)})
    return result
