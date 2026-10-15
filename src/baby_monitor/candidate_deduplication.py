"""Explicit near-identical candidate suppression, never person-count truth.

Callers must supply candidates from one source frame/view. Equal timestamps do
not prove image identity. This optional helper is not enabled in live inference.
Keep the original candidates alongside the suppression trace for evaluation.
"""

from __future__ import annotations

import math

from baby_monitor.domain import Detection


def deduplicate_frame_candidates(
    detections: tuple[Detection, ...], *, minimum_iou: float = 0.95
) -> tuple[tuple[Detection, ...], tuple[tuple[int, int], ...]]:
    """Return retained objects and (suppressed input index, retained index) pairs.

    Only same-label boxes are compared. Stable score order breaks ties by input
    index; suppression is greedy against retained boxes, not transitive clustering.
    High overlap can still describe distinct occluded people, so this is a candidate
    policy for explicit evaluation, not confirmation that two boxes depict one person.
    """
    if (
        type(minimum_iou) not in (int, float)
        or not 0.9 <= minimum_iou <= 1
        or not math.isfinite(minimum_iou)
        or type(detections) is not tuple
        or len(detections) > 100
        or any(not isinstance(item, Detection) for item in detections)
    ):
        raise ValueError("invalid candidate deduplication input")
    if len({item.timestamp_seconds for item in detections}) > 1:
        raise ValueError("candidate timestamps span different frames")
    retained: list[int] = []
    suppressed: list[tuple[int, int]] = []
    for index in sorted(range(len(detections)), key=lambda i: (-detections[i].confidence, i)):
        current = detections[index]
        duplicate = None
        for other_index in retained:
            other = detections[other_index]
            if current.label != other.label:
                continue
            a, b = current.bbox_xyxy, other.bbox_xyxy
            intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0.0, min(a[3], b[3]) - max(a[1], b[1])
            )
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
            if intersection / union >= minimum_iou:
                duplicate = other_index
                break
        if duplicate is None:
            retained.append(index)
        else:
            suppressed.append((index, duplicate))
    return tuple(detections[i] for i in sorted(retained)), tuple(sorted(suppressed))
