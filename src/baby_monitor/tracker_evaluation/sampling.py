"""Frozen sampling/loss scenarios on existing boxes, not a camera fault test.

Source timestamps and detection/review rows are never fabricated or retimed.
Omitted inputs are declared explicitly, not reclassified as detector misses.
The same selected batch must be used by every compared tracker.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import UTC

from baby_monitor.object_tracker import DetectionFrameBatch
from baby_monitor.tracker_evaluation.review import TrackerAssociationReview

SCENARIO_NAMES = (
    "publisher_25fps",
    "sampled_5fps",
    "sampled_20fps",
    "adaptive_5_20_5fps",
    "sampled_5fps_drop_every_7th",
    "sampled_5fps_two_second_gap",
)


def _elapsed_us(batch):
    anchor = batch.frames[0].occurred_at.astimezone(UTC)
    values = []
    for frame in batch.frames:
        delta = frame.occurred_at.astimezone(UTC) - anchor
        values.append((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds)
    return values


def _scheduled_indices(elapsed, *, adaptive=False, period_us=200_000):
    """First-frame-at-deadline on a frozen offline schedule, no catch-up.

    The adaptive phases depend on the entire clip span, NOT an online policy.
    """
    result = []
    due, previous_phase = None, None
    span = elapsed[-1]
    for index, value in enumerate(elapsed):
        phase = 0 if not adaptive or value * 3 < span else (1 if value * 3 < span * 2 else 2)
        period = 50_000 if adaptive and phase == 1 else period_us
        if phase != previous_phase:
            due = value
        if value >= due:
            result.append(index)
            due += ((value - due) // period + 1) * period
        previous_phase = phase
    return tuple(result)


def _scenario_selections(batch):
    elapsed = _elapsed_us(batch)
    if elapsed[-1] < 4_000_000:
        raise ValueError("tracker sampling scenarios require at least four seconds")
    if any(b - a != 40_000 for a, b in zip(elapsed, elapsed[1:], strict=False)):
        raise ValueError("publisher_25fps scenario requires original 25fps timeline")
    five = _scheduled_indices(elapsed)
    gap_start = elapsed[-1] // 2
    return (
        tuple(range(len(batch.frames))),
        five,
        _scheduled_indices(elapsed, period_us=50_000),
        _scheduled_indices(elapsed, adaptive=True),
        tuple(value for index, value in enumerate(five) if (index + 1) % 7),
        tuple(value for value in five if not gap_start <= elapsed[value] < gap_start + 2_000_000),
    )


@dataclass(frozen=True, slots=True)
class TrackerSamplingScenario:
    name: str
    source_batch_sha256: str
    source_review_sha256: str
    source_frame_count: int
    source_detection_count: int
    selected_source_indices: tuple[int, ...]
    batch: DetectionFrameBatch
    review: TrackerAssociationReview

    def __post_init__(self):
        if self.name not in SCENARIO_NAMES:
            raise ValueError("invalid tracker sampling scenario name")
        if any(
            type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (self.source_batch_sha256, self.source_review_sha256)
        ):
            raise ValueError("invalid tracker sampling source hash")
        if (
            type(self.source_frame_count) is not int
            or not 1 <= self.source_frame_count <= 5000
            or type(self.source_detection_count) is not int
            or not 0 <= self.source_detection_count <= 10000
        ):
            raise ValueError("invalid tracker sampling source count")
        if (
            not isinstance(self.batch, DetectionFrameBatch)
            or not isinstance(self.review, TrackerAssociationReview)
            or type(self.selected_source_indices) is not tuple
            or len(self.selected_source_indices) != len(self.batch.frames)
            or any(
                type(i) is not int or not 0 <= i < self.source_frame_count
                for i in self.selected_source_indices
            )
            or any(
                b <= a
                for a, b in zip(
                    self.selected_source_indices, self.selected_source_indices[1:], strict=False
                )
            )
        ):
            raise ValueError("invalid tracker sampling selection")
        if sum(len(f.detections) for f in self.batch.frames) > self.source_detection_count:
            raise ValueError("invalid tracker sampling detection count")
        self.review.bind(self.batch)

    def bind_source(self, source_batch, source_review):
        source_review.bind(source_batch)
        if (
            source_batch.fingerprint != self.source_batch_sha256
            or source_review.fingerprint != self.source_review_sha256
            or len(source_batch.frames) != self.source_frame_count
            or sum(len(f.detections) for f in source_batch.frames) != self.source_detection_count
            or self.batch
            != replace(
                source_batch,
                frames=tuple(source_batch.frames[i] for i in self.selected_source_indices),
            )
        ):
            raise ValueError("tracker sampling source mismatch")
        selected_ids = {f.frame_id for f in self.batch.frames}
        expected_review = replace(
            source_review,
            detection_batch_sha256=self.batch.fingerprint,
            assignments=tuple(a for a in source_review.assignments if a.frame_id in selected_ids),
        )
        if self.review != expected_review:
            raise ValueError("tracker sampling review lineage mismatch")
        if (
            self.selected_source_indices
            != _scenario_selections(source_batch)[SCENARIO_NAMES.index(self.name)]
        ):
            raise ValueError("tracker sampling scenario schedule mismatch")

    def to_dict(self):
        elapsed = _elapsed_us(self.batch)
        gaps = [b - a for a, b in zip(elapsed, elapsed[1:], strict=False)]
        return {
            "version": 1,
            "scenario": self.name,
            "source_batch_sha256": self.source_batch_sha256,
            "source_review_sha256": self.source_review_sha256,
            "source_frame_count": self.source_frame_count,
            "source_detection_count": self.source_detection_count,
            "selected_source_indices": list(self.selected_source_indices),
            "selected_frame_count": len(self.batch.frames),
            "selected_detection_count": sum(len(f.detections) for f in self.batch.frames),
            "omitted_input_frame_count": self.source_frame_count - len(self.batch.frames),
            "detection_batch_sha256": self.batch.fingerprint,
            "review_sha256": self.review.fingerprint,
            "minimum_selected_gap_seconds": min(gaps) / 1_000_000 if gaps else None,
            "maximum_selected_gap_seconds": max(gaps) / 1_000_000 if gaps else None,
            "input_timestamps_preserved": True,
            "actual_camera_fault_injected": False,
            "source_pts_continuity_established": False,
            "omitted_inputs_counted_as_missed_detections": False,
            "detection_accuracy_evaluated": False,
            "overall_adapter_selected": False,
        }

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def build_tracker_sampling_scenarios(
    batch: DetectionFrameBatch, review: TrackerAssociationReview
) -> tuple[TrackerSamplingScenario, ...]:
    if not isinstance(batch, DetectionFrameBatch) or not isinstance(
        review, TrackerAssociationReview
    ):
        raise ValueError("invalid tracker sampling inputs")
    review.bind(batch)
    if len(batch.frames) > 5000 or sum(len(f.detections) for f in batch.frames) > 10000:
        raise ValueError("tracker sampling inputs exceed experiment bound")
    selections = _scenario_selections(batch)
    scenarios = []
    source_sha, source_review_sha = batch.fingerprint, review.fingerprint
    for name, indices in zip(SCENARIO_NAMES, selections, strict=True):
        selected_frames = tuple(batch.frames[index] for index in indices)
        selected_batch = replace(batch, frames=selected_frames)
        frame_ids = {frame.frame_id for frame in selected_frames}
        selected_review = replace(
            review,
            detection_batch_sha256=selected_batch.fingerprint,
            assignments=tuple(item for item in review.assignments if item.frame_id in frame_ids),
        )
        selected_review.bind(selected_batch)
        scenario = TrackerSamplingScenario(
            name,
            source_sha,
            source_review_sha,
            len(batch.frames),
            sum(len(f.detections) for f in batch.frames),
            indices,
            selected_batch,
            selected_review,
        )
        scenario.bind_source(batch, review)
        scenarios.append(scenario)
    return tuple(scenarios)
