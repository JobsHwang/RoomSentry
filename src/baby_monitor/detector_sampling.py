"""Path-free evidence of actual detector calls, including calls with no boxes.

A decoder returning false does not distinguish EOF from a decode failure. This
ledger records observations, never a claim that the complete source was seen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .domain import Detection

MAX_DETECTOR_SAMPLES = 100_000
MAX_DECODED_FRAMES = 10_000_000
MAX_SAMPLED_DETECTIONS = 100_000


def validate_nominal_sampling_grid(
    ledger: DetectorSamplingLedger,
    *,
    source_fps: float,
    source_frame_count: int,
) -> None:
    """Admit actual v2 calls to a legacy nominal-frame evaluation denominator.

    Check every invocation, including zero-box and unannotated samples, before
    an evaluator creates nominal empty frames. This is not source completeness
    verification. VFR, sampled fallback clocks, and partial scans require a
    separately aligned evaluation contract, not rounding or invented calls.

    Missing PTS on *unsampled* decoded frames does not invalidate otherwise
    aligned sampled observations; it is not upgraded to known timing here.
    """
    if not isinstance(ledger, DetectorSamplingLedger):
        raise DetectorSamplingError("nominal evaluation requires a sampling ledger")
    _number(source_fps, "nominal source_fps", minimum=0.01, maximum=240)
    _integer(
        source_frame_count, "nominal source_frame_count", minimum=1, maximum=MAX_DECODED_FRAMES
    )
    if ledger.decoded_frame_count != source_frame_count:
        raise DetectorSamplingError(
            "recorded decoded-frame count does not match the declared nominal source count"
        )
    if ledger.native_fps_source != "reported" or not math.isclose(
        ledger.native_fps, source_fps, rel_tol=0, abs_tol=1e-9
    ):
        raise DetectorSamplingError("recorded FPS does not establish the declared nominal grid")
    if ledger.frame_step != max(1, round(source_fps / ledger.requested_sample_fps)):
        raise DetectorSamplingError("recorded sampling step does not match the nominal grid")
    for sample in ledger.samples:
        if sample.timestamp_source != "container_pts":
            raise DetectorSamplingError("nominal evaluation cannot align a fallback sample clock")
        position = sample.timestamp_seconds * source_fps
        if not math.isfinite(position) or abs(position - sample.decoded_frame_index) > 1e-6:
            raise DetectorSamplingError(
                "actual sample timestamp does not match its nominal decoded-frame index"
            )


class DetectorSamplingError(ValueError):
    """The bounded detector sampling evidence is inconsistent or invalid."""


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise DetectorSamplingError(f"detector sampling {field} is invalid")


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectorSamplingError(f"detector sampling {field} is invalid")
    try:
        valid = math.isfinite(value) and minimum <= value <= maximum
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise DetectorSamplingError(f"detector sampling {field} is invalid")


def _fields(value: Any, expected: set[str], field: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise DetectorSamplingError(f"detector sampling {field} fields do not match")


@dataclass(frozen=True, slots=True)
class DetectorSampleFrame:
    """One actual model invocation and its contiguous slice of output boxes."""

    decoded_frame_index: int
    timestamp_seconds: float
    timestamp_source: str
    detection_start: int
    detection_count: int

    def __post_init__(self) -> None:
        _integer(
            self.decoded_frame_index,
            "decoded_frame_index",
            minimum=0,
            maximum=MAX_DECODED_FRAMES - 1,
        )
        _number(self.timestamp_seconds, "timestamp_seconds", minimum=0, maximum=math.inf)
        if self.timestamp_source not in ("container_pts", "nominal_fps"):
            raise DetectorSamplingError("detector sampling timestamp_source is invalid")
        _integer(self.detection_start, "detection_start", minimum=0, maximum=MAX_SAMPLED_DETECTIONS)
        _integer(self.detection_count, "detection_count", minimum=0, maximum=MAX_SAMPLED_DETECTIONS)
        if self.detection_start + self.detection_count > MAX_SAMPLED_DETECTIONS:
            raise DetectorSamplingError("detector sampling detection slice exceeds limit")

    def to_dict(self) -> dict[str, object]:
        return {
            "decoded_frame_index": self.decoded_frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "timestamp_source": self.timestamp_source,
            "detection_start": self.detection_start,
            "detection_count": self.detection_count,
        }

    @classmethod
    def from_dict(cls, value: Any) -> DetectorSampleFrame:
        _fields(
            value,
            {
                "decoded_frame_index",
                "timestamp_seconds",
                "timestamp_source",
                "detection_start",
                "detection_count",
            },
            "sample",
        )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class DetectorSamplingLedger:
    """Bounded decoded-frame sampling evidence, not source completeness proof."""

    native_fps: float
    native_fps_source: str
    requested_sample_fps: float
    frame_step: int
    decoded_frame_count: int
    samples: tuple[DetectorSampleFrame, ...]
    timestamp_fallback_frame_count: int
    max_adjacent_container_pts_gap_seconds: float | None
    container_pts_regression_count: int
    end_reason: str = "read_returned_false"
    source_exhaustion_verified: bool = False

    def __post_init__(self) -> None:
        _number(self.native_fps, "native_fps", minimum=0.01, maximum=1000)
        _number(self.requested_sample_fps, "requested_sample_fps", minimum=0.01, maximum=10)
        if self.native_fps_source not in ("reported", "fallback_25"):
            raise DetectorSamplingError("detector sampling native_fps_source is invalid")
        if self.native_fps_source == "fallback_25" and self.native_fps != 25:
            raise DetectorSamplingError("detector sampling fallback FPS must be 25")
        _integer(self.frame_step, "frame_step", minimum=1, maximum=MAX_DECODED_FRAMES)
        if self.frame_step != max(1, round(self.native_fps / self.requested_sample_fps)):
            raise DetectorSamplingError("detector sampling frame_step does not match FPS")
        _integer(
            self.decoded_frame_count,
            "decoded_frame_count",
            minimum=1,
            maximum=MAX_DECODED_FRAMES,
        )
        if (
            not isinstance(self.samples, tuple)
            or not 1 <= len(self.samples) <= MAX_DETECTOR_SAMPLES
        ):
            raise DetectorSamplingError("detector sampling samples are invalid")
        expected_count = (self.decoded_frame_count - 1) // self.frame_step + 1
        if len(self.samples) != expected_count:
            raise DetectorSamplingError("detector sampling samples do not cover the decoded grid")
        _integer(
            self.timestamp_fallback_frame_count,
            "timestamp_fallback_frame_count",
            minimum=0,
            maximum=self.decoded_frame_count,
        )
        _integer(
            self.container_pts_regression_count,
            "container_pts_regression_count",
            minimum=0,
            maximum=max(0, self.decoded_frame_count - self.timestamp_fallback_frame_count - 1),
        )
        if self.max_adjacent_container_pts_gap_seconds is not None:
            _number(
                self.max_adjacent_container_pts_gap_seconds,
                "max_adjacent_container_pts_gap_seconds",
                minimum=0,
                maximum=math.inf,
            )
            if self.decoded_frame_count - self.timestamp_fallback_frame_count < 2:
                raise DetectorSamplingError("detector sampling PTS gap requires two PTS frames")
        if self.end_reason != "read_returned_false":
            raise DetectorSamplingError("detector sampling end_reason is invalid")
        if self.source_exhaustion_verified is not False:
            raise DetectorSamplingError("detector sampling cannot verify source exhaustion")
        offset = 0
        previous_timestamp = -1.0
        sampled_fallback_count = 0
        for index, sample in enumerate(self.samples):
            if not isinstance(sample, DetectorSampleFrame):
                raise DetectorSamplingError("detector sampling sample is invalid")
            if sample.decoded_frame_index != index * self.frame_step:
                raise DetectorSamplingError("detector sampling sample index does not match grid")
            if sample.detection_start != offset:
                raise DetectorSamplingError(
                    "detector sampling detection offsets are not contiguous"
                )
            if sample.timestamp_seconds < previous_timestamp:
                raise DetectorSamplingError("detector sampling timestamps are not nondecreasing")
            offset += sample.detection_count
            previous_timestamp = sample.timestamp_seconds
            sampled_fallback_count += sample.timestamp_source == "nominal_fps"
        if sampled_fallback_count > self.timestamp_fallback_frame_count:
            raise DetectorSamplingError("detector sampling fallback count omits sampled frames")
        sampled_pts_count = len(self.samples) - sampled_fallback_count
        if self.timestamp_fallback_frame_count > self.decoded_frame_count - sampled_pts_count:
            raise DetectorSamplingError(
                "detector sampling fallback count includes sampled PTS frames"
            )

    def validate_detections(self, detections: tuple[Detection, ...]) -> None:
        """Bind every box to its invocation, without erasing zero-box samples."""
        if not isinstance(detections, tuple) or len(detections) > MAX_SAMPLED_DETECTIONS:
            raise DetectorSamplingError("detector sampling detections are invalid")
        final = self.samples[-1]
        if len(detections) != final.detection_start + final.detection_count:
            raise DetectorSamplingError("detector sampling detection total does not match")
        for sample in self.samples:
            for index in range(
                sample.detection_start, sample.detection_start + sample.detection_count
            ):
                detection = detections[index]
                if not isinstance(detection, Detection):
                    raise DetectorSamplingError("detector sampling detection is invalid")
                if detection.timestamp_seconds != sample.timestamp_seconds:
                    raise DetectorSamplingError(
                        "detector sampling detection timestamp does not match"
                    )

    def to_dict(self) -> dict[str, object]:
        return {
            "native_fps": self.native_fps,
            "native_fps_source": self.native_fps_source,
            "requested_sample_fps": self.requested_sample_fps,
            "frame_step": self.frame_step,
            "decoded_frame_count": self.decoded_frame_count,
            "samples": [sample.to_dict() for sample in self.samples],
            "timestamp_fallback_frame_count": self.timestamp_fallback_frame_count,
            "max_adjacent_container_pts_gap_seconds": self.max_adjacent_container_pts_gap_seconds,
            "container_pts_regression_count": self.container_pts_regression_count,
            "end_reason": self.end_reason,
            "source_exhaustion_verified": self.source_exhaustion_verified,
        }

    @classmethod
    def from_dict(cls, value: Any) -> DetectorSamplingLedger:
        _fields(
            value,
            {
                "native_fps",
                "native_fps_source",
                "requested_sample_fps",
                "frame_step",
                "decoded_frame_count",
                "samples",
                "timestamp_fallback_frame_count",
                "max_adjacent_container_pts_gap_seconds",
                "container_pts_regression_count",
                "end_reason",
                "source_exhaustion_verified",
            },
            "ledger",
        )
        samples = value["samples"]
        if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_DETECTOR_SAMPLES:
            raise DetectorSamplingError("detector sampling samples are invalid")
        return cls(**{**value, "samples": tuple(DetectorSampleFrame.from_dict(s) for s in samples)})
