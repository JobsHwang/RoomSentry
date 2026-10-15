"""Project actual detector checks into conservative segment-level presence facts.

This is not source-completeness verification or an inference about baby identity.
At most one positive/clear fact per segment preserves segment confirmation counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .detector_sampling import DetectorSamplingLedger
from .domain import Detection
from .room_presence import StreamPresenceSampleState


@dataclass(frozen=True, slots=True)
class PresenceSamplingObservation:
    state: StreamPresenceSampleState
    offset_seconds: float
    reason_code: str | None = None


def project_presence_sampling(
    *,
    detections: tuple[Detection, ...],
    ledger: DetectorSamplingLedger | None,
    duration_seconds: float,
) -> tuple[PresenceSamplingObservation, ...]:
    """Keep acquisition time distinct from scan completion or segment end.

    A known timing/coverage defect prevents negative evidence from clearing a
    room. Known-PTS person evidence is retained separately, followed by a gap
    at the segment boundary, so it cannot conceal the current coverage defect.
    The gap timestamp is an assessment boundary, not a claimed detector call.
    """
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not 0 < duration_seconds <= 86400
    ):
        raise ValueError("presence sampling duration is invalid")
    if not isinstance(detections, tuple) or any(
        not isinstance(item, Detection) for item in detections
    ):
        raise ValueError("presence sampling detections are invalid")
    if ledger is None:
        return (
            PresenceSamplingObservation(
                StreamPresenceSampleState.COVERAGE_GAP,
                duration_seconds,
                "sampling_unavailable",
            ),
        )
    if not isinstance(ledger, DetectorSamplingLedger):
        raise ValueError("presence sampling ledger is invalid")
    ledger.validate_detections(detections)
    if any(sample.timestamp_seconds > duration_seconds for sample in ledger.samples):
        raise ValueError("presence sampling timestamp exceeds segment duration")

    samples = ledger.samples
    reason = None
    if (
        ledger.timestamp_fallback_frame_count
        or ledger.container_pts_regression_count
        or any(
            right.timestamp_seconds <= left.timestamp_seconds
            for left, right in zip(samples, samples[1:], strict=False)
        )
    ):
        reason = "sampling_clock_uncertain"
    elif ledger.native_fps_source != "reported":
        reason = "sampling_cadence_unknown"
    else:
        # Use the executed decoded-frame stride, not an assumed requested FPS.
        # One reported frame (at least 1 ms) allows frame-boundary rounding.
        interval = ledger.frame_step / ledger.native_fps
        tolerance = max(0.001, 1 / ledger.native_fps)
        limit = interval + tolerance
        gaps = [
            right.timestamp_seconds - left.timestamp_seconds
            for left, right in zip(samples, samples[1:], strict=False)
        ]
        if (
            samples[0].timestamp_seconds > limit
            or any(gap > limit for gap in gaps)
            or (
                ledger.max_adjacent_container_pts_gap_seconds is not None
                and ledger.max_adjacent_container_pts_gap_seconds > limit
            )
        ):
            reason = "sampling_time_gap"
        elif duration_seconds - samples[-1].timestamp_seconds > limit:
            reason = "sampling_tail_gap"

    # A regression makes the segment-relative placement unreliable even if a
    # particular invocation retained a PTS. Fallback samples never earn facts.
    positive_offsets = [
        sample.timestamp_seconds
        for sample in samples
        if sample.timestamp_source == "container_pts"
        and not ledger.container_pts_regression_count
        and any(
            item.label == "person"
            for item in detections[
                sample.detection_start : sample.detection_start + sample.detection_count
            ]
        )
    ]
    observations = []
    if positive_offsets:
        observations.append(
            PresenceSamplingObservation(
                StreamPresenceSampleState.PERSON_DETECTED,
                max(positive_offsets),
            )
        )
    elif reason is None:
        observations.append(
            PresenceSamplingObservation(
                StreamPresenceSampleState.CLEAR,
                samples[-1].timestamp_seconds,
            )
        )
    if reason is not None:
        observations.append(
            PresenceSamplingObservation(
                StreamPresenceSampleState.COVERAGE_GAP,
                duration_seconds,
                reason,
            )
        )
    if any(not math.isfinite(item.offset_seconds) for item in observations):
        raise ValueError("presence sampling timestamp is invalid")
    return tuple(observations)
