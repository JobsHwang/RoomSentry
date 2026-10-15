"""Pure offline comparison of frame-stride and causal source-PTS scheduling.

This debug plan does not run inference or produce a DetectorSamplingLedger.
Selecting a later frame once never recovers frames absent at missed deadlines.
The caller owns reference timestamp provenance; hashes bind values, not media.
"""

from __future__ import annotations

import hashlib
import json
import math

MAX_REFERENCE_FRAMES = 200_000
MAX_PTS_SECONDS = 86_400.0


def _number(value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("PTS sampling diagnostic number is invalid")
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValueError("PTS sampling diagnostic number is invalid") from None
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError("PTS sampling diagnostic number is invalid")
    return number


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _max_gap(values: tuple[float, ...] | list[float]) -> float | None:
    return max(
        (right - left for left, right in zip(values, values[1:], strict=False)), default=None
    )


def _selection(reference: tuple[float, ...], indices: list[int]) -> dict[str, object]:
    pts = [reference[index] for index in indices]
    return {
        "decoded_frame_indices": indices,
        "pts_seconds": pts,
        "actual_sample_count": len(indices),
        "max_adjacent_sample_pts_gap_seconds": _max_gap(pts),
    }


def plan_pts_sampling(
    reference_pts: tuple[float, ...],
    *,
    native_fps: float,
    sample_fps: float,
) -> dict[str, object]:
    """Plan actual reference-frame selections without inventing observations.

    References must be a nonempty tuple of at most 200,000 strictly increasing,
    known finite PTS in [0, 86400]. Native FPS is [0.01, 1000] and requested
    sample FPS [0.01, 10]. Stride uses max(1, round(native_fps / sample_fps)).

    Source-PTS deadlines are integer multiples of 1/sample_fps anchored at zero.
    When a frame reaches the earliest outstanding deadline it is selected once;
    any additional deadlines at or before that same PTS are skipped, counted as
    missed, and never replayed with duplicate frames. This is a causal scheduling
    count, not dropped source-frame count or missing-content duration. Nothing is
    extrapolated beyond the last provided frame. Deadline arithmetic corrects
    product rounding using the same divided deadline value as the causal test.

    The reference SHA hashes its compact JSON array after numeric conversion to
    floats; the report SHA excludes itself. Neither hash validates source media.
    """
    native = _number(native_fps, minimum=0.01, maximum=1000.0)
    requested = _number(sample_fps, minimum=0.01, maximum=10.0)
    if not isinstance(reference_pts, tuple) or not 1 <= len(reference_pts) <= MAX_REFERENCE_FRAMES:
        raise ValueError("PTS sampling diagnostic reference frame count is invalid")
    reference = tuple(
        _number(value, minimum=0.0, maximum=MAX_PTS_SECONDS) for value in reference_pts
    )
    if any(right <= left for left, right in zip(reference, reference[1:], strict=False)):
        raise ValueError("PTS sampling diagnostic requires strictly increasing known PTS")

    step = max(1, round(native / requested))
    stride_indices = list(range(0, len(reference), step))
    deadline_indices = []
    next_deadline_index = 0
    missed_count = 0
    for index, pts in enumerate(reference):
        if pts < next_deadline_index / requested:
            continue
        deadline_indices.append(index)
        previous_deadline_index = next_deadline_index
        next_deadline_index = max(previous_deadline_index + 1, math.floor(pts * requested) + 1)
        # Do not add a clock tolerance or shift timestamps. Floating-point
        # multiplication and division may disagree by one deadline at a boundary.
        while next_deadline_index / requested <= pts:
            next_deadline_index += 1
        while (
            next_deadline_index > previous_deadline_index + 1
            and (next_deadline_index - 1) / requested > pts
        ):
            next_deadline_index -= 1
        missed_count += next_deadline_index - previous_deadline_index - 1

    report = {
        "schema_version": "pts-sampling-diagnostic-v1",
        "reference_pts_sha256": _hash(reference),
        "native_fps": native,
        "sample_fps": requested,
        "source_frame_count": len(reference),
        "source_first_pts_seconds": reference[0],
        "source_last_pts_seconds": reference[-1],
        "source_max_adjacent_pts_gap_seconds": _max_gap(reference),
        "decoded_stride": {"frame_step": step, **_selection(reference, stride_indices)},
        "source_pts_deadline": {
            "anchor_pts_seconds": 0.0,
            **_selection(reference, deadline_indices),
            "missed_deadline_count": missed_count,
            "next_deadline_seconds": next_deadline_index / requested,
        },
        "offline_diagnostic_only": True,
        "accuracy_evaluated": False,
        "ground_truth": False,
        "source_exhaustion_verified": False,
        "deployment_approved": False,
        "equal_stride_ledger_claimed": False,
    }
    return {**report, "report_sha256": _hash(report)}
