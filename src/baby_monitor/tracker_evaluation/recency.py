"""Offline recency-cascade candidate; never a live tracker or identity claim.

This deliberately changes the association objective, not the detector inputs:
within each confidence stage, tracks seen on the preceding input get first use
of eligible edges, followed by retained dormant tracks. Empty inputs remain
empty and advance that boundary. Fewer anonymous IDs do not establish accuracy.
"""

from __future__ import annotations

from datetime import datetime

from baby_monitor.motion_tracker import (
    MotionActiveTrack,
    MotionTrackerConfig,
    MotionTrackSample,
    _match,
)
from baby_monitor.object_tracker import DetectionFrameBatch, FrameDetection
from baby_monitor.tracker_evaluation._shared import TrackerEvaluationError, _canonical_sha256
from baby_monitor.tracker_evaluation.prediction import (
    TrackerPredictionAssignment,
    TrackerPredictionBatch,
)

RECENCY_ADAPTER = "offline_recency_cascade_v1"
RECENCY_REVISION = "previous_input_first_then_dormant_v1"


def _cascade(
    tracks: tuple[MotionActiveTrack, ...],
    detections: tuple[FrameDetection, ...],
    *,
    previous_input_at: datetime | None,
    occurred_at: datetime,
    minimum_score: float,
    config: MotionTrackerConfig,
) -> tuple[tuple[int, int], ...]:
    """Keep the original score/gates and deterministic one-to-one matching."""
    result: list[tuple[int, int]] = []
    used: set[int] = set()
    for recent in (True, False):
        track_indexes = tuple(
            i
            for i, track in enumerate(tracks)
            if (track.current.occurred_at == previous_input_at) == recent
        )
        detection_indexes = tuple(i for i in range(len(detections)) if i not in used)
        pairs = _match(
            tuple(tracks[i] for i in track_indexes),
            tuple(detections[i] for i in detection_indexes),
            occurred_at=occurred_at,
            minimum_score=minimum_score,
            config=config,
        )
        for track_index, detection_index in pairs:
            original_detection_index = detection_indexes[detection_index]
            result.append((track_indexes[track_index], original_detection_index))
            used.add(original_detection_index)
    return tuple(sorted(result))


def build_recency_tracker_predictions(
    batch: DetectionFrameBatch, config: MotionTrackerConfig
) -> TrackerPredictionBatch:
    """Replay one fixed batch; not a replacement for persisted motion semantics.

    Confidence staging, lifetime, motion scores, box/confidence values and hard
    limits remain the same. Only priority between recent and dormant tracks
    changes. Prefixes are causal; IDs never use the future complete batch hash.
    """
    if not isinstance(batch, DetectionFrameBatch) or not isinstance(config, MotionTrackerConfig):
        raise TrackerEvaluationError("recency candidate inputs are invalid")
    config_sha = _canonical_sha256({"policy": RECENCY_REVISION, "motion_config": config.to_dict()})
    active: dict[str, MotionActiveTrack] = {}
    issued: set[str] = set()
    assignments: list[TrackerPredictionAssignment] = []
    previous_input_at: datetime | None = None
    for frame in batch.frames:
        if len(frame.detections) > config.max_frame_detections:
            raise TrackerEvaluationError("recency frame exceeds configured detection limit")
        retained = {
            key: track
            for key, track in active.items()
            if (frame.occurred_at - track.current.occurred_at).total_seconds()
            <= config.max_track_age_seconds
        }
        ordered = tuple(retained[key] for key in sorted(retained))
        high = tuple(
            d for d in frame.detections if d.confidence >= config.high_confidence_threshold
        )
        low = tuple(
            d
            for d in frame.detections
            if config.low_confidence_threshold <= d.confidence < config.high_confidence_threshold
        )
        primary = _cascade(
            ordered,
            high,
            previous_input_at=previous_input_at,
            occurred_at=frame.occurred_at,
            minimum_score=config.primary_match_threshold,
            config=config,
        )
        used_tracks = {i for i, _ in primary}
        remaining = tuple(t for i, t in enumerate(ordered) if i not in used_tracks)
        secondary = _cascade(
            remaining,
            low,
            previous_input_at=previous_input_at,
            occurred_at=frame.occurred_at,
            minimum_score=config.secondary_match_threshold,
            config=config,
        )
        by_detection: dict[str, str] = {}
        for tracks, detections, pairs in (
            (ordered, high, primary),
            (remaining, low, secondary),
        ):
            for i, j in pairs:
                track, detection = tracks[i], detections[j]
                retained[track.tracking_id] = track.update(detection, frame.occurred_at)
                by_detection[detection.detection_id] = track.tracking_id
        unmatched = tuple(d for d in frame.detections if d.detection_id not in by_detection)
        if (
            len(retained) + sum(d.confidence >= config.high_confidence_threshold for d in unmatched)
            > config.max_active_tracks
        ):
            raise TrackerEvaluationError("recency active track limit would be exceeded")
        for detection in unmatched:
            tracking_id = (
                "track_"
                + _canonical_sha256(
                    {
                        "adapter": RECENCY_ADAPTER,
                        "revision": RECENCY_REVISION,
                        "config": config_sha,
                        "camera": batch.camera_id,
                        "detector": batch.detector_adapter,
                        "frame": frame.frame_id,
                        "detection": detection.to_dict(),
                    }
                )[:24]
            )
            if tracking_id in issued:
                raise TrackerEvaluationError("recency deterministic ID collision")
            issued.add(tracking_id)
            by_detection[detection.detection_id] = tracking_id
            if detection.confidence >= config.high_confidence_threshold:
                retained[tracking_id] = MotionActiveTrack(
                    tracking_id,
                    detection.label,
                    MotionTrackSample(frame.occurred_at, detection.confidence, detection.bbox_xyxy),
                )
        assignments.extend(
            TrackerPredictionAssignment(
                frame.frame_id, detection.detection_id, by_detection[detection.detection_id]
            )
            for detection in frame.detections
        )
        active = retained
        previous_input_at = frame.occurred_at
    result = TrackerPredictionBatch(
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
        detection_batch_sha256=batch.fingerprint,
        tracker_adapter=RECENCY_ADAPTER,
        tracker_revision=RECENCY_REVISION,
        tracker_config_sha256=config_sha,
        runtime_id="python_stdlib_offline_candidate_v1",
        assignments=tuple(assignments),
    )
    result.bind(batch)
    return result
