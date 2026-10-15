"""Offline candidate for nonpositive motion-prediction dimensions.

Only an invalid raw prediction uses the last observed box. Confidence staging,
global one-to-one assignment, lifetime and detection values are preserved. This
adapter has its own identity and cannot write persistent tracker or role state.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

from baby_monitor.motion_tracker import (
    MotionActiveTrack,
    MotionTrackerConfig,
    MotionTrackerState,
    MotionTrackSample,
    _bbox_from_geometry,
    _geometry,
    _match_score,
    _maximum_weight_assignment,
    _new_tracking_id,
    _predict_bbox,
)
from baby_monitor.object_tracker import DetectionFrame, DetectionFrameBatch
from baby_monitor.tracker_evaluation._shared import TrackerEvaluationError, _canonical_sha256
from baby_monitor.tracker_evaluation.prediction import (
    TrackerPredictionAssignment,
    TrackerPredictionBatch,
)

FALLBACK_ADAPTER = "offline_prediction_fallback_v1"
FALLBACK_REVISION = "invalid_extent_uncontested_latest_bbox_v2"


def _legacy_predict_bbox(track, occurred_at, maximum):
    """Frozen pre-fix predictor for this adapter's original-edge protection."""
    if track.previous is None or maximum == 0:
        return track.current.bbox_xyxy
    history = (track.current.occurred_at - track.previous.occurred_at).total_seconds()
    if history <= 0:
        raise TrackerEvaluationError("prediction fallback history is not chronological")
    horizon = min(maximum, max(0.0, (occurred_at - track.current.occurred_at).total_seconds()))
    previous = _geometry(track.previous.bbox_xyxy)
    current = _geometry(track.current.bbox_xyxy)
    raw = tuple(current[i] + (current[i] - previous[i]) / history * horizon for i in range(4))
    return _bbox_from_geometry(*raw)


def _legacy_match_score(track, detection, *, occurred_at, config):
    # A local sample freezes the old predicted box while reusing unchanged
    # area, distance and IoU gates. Never patch the live predictor globally.
    predicted = _legacy_predict_bbox(track, occurred_at, config.max_prediction_seconds)
    frozen = replace(track, previous=None, current=replace(track.current, bbox_xyxy=predicted))
    return _match_score(frozen, detection, occurred_at=occurred_at, config=config)


def prediction_fallback_required(
    track: MotionActiveTrack, occurred_at: datetime, maximum: float
) -> bool:
    if track.previous is None or maximum == 0:
        return False
    history = (track.current.occurred_at - track.previous.occurred_at).total_seconds()
    if history <= 0:
        raise TrackerEvaluationError("prediction fallback history is not chronological")
    horizon = min(maximum, max(0.0, (occurred_at - track.current.occurred_at).total_seconds()))
    previous = _geometry(track.previous.bbox_xyxy)
    current = _geometry(track.current.bbox_xyxy)
    raw = tuple(current[i] + (current[i] - previous[i]) / history * horizon for i in range(4))
    return not all(math.isfinite(n) for n in raw) or raw[2] <= 0 or raw[3] <= 0


def predict_with_fallback(
    track: MotionActiveTrack, occurred_at: datetime, maximum: float
) -> tuple[float, float, float, float]:
    if prediction_fallback_required(track, occurred_at, maximum):
        return track.current.bbox_xyxy
    return _predict_bbox(track, occurred_at, maximum)


def _fallback_match(tracks, detections, *, occurred_at, minimum_score, config):
    # Removing velocity from a local immutable copy reuses the original score
    # and its area/distance gates; the source track/history is never changed.
    score_tracks = tuple(
        replace(track, previous=None)
        if prediction_fallback_required(track, occurred_at, config.max_prediction_seconds)
        else track
        for track in tracks
    )
    original_scores = [
        [_legacy_match_score(track, d, occurred_at=occurred_at, config=config) for d in detections]
        for track in tracks
    ]
    # A returning dormant track must not steal a detection which an original
    # nondegenerate prediction already qualifies for, including equal scores.
    # Use the existing stage gate; no new recency or confidence threshold.
    protected = [
        any(row[j] > 0 and row[j] + 1e-15 >= minimum_score for row in original_scores)
        for j in range(len(detections))
    ]
    scores = [
        [
            original_scores[i][j]
            if protected[j]
            else _match_score(track, detection, occurred_at=occurred_at, config=config)
            for j, detection in enumerate(detections)
        ]
        for i, track in enumerate(score_tracks)
    ]
    return _maximum_weight_assignment(scores, minimum_score=minimum_score)


class PredictionFallbackTracker:
    """Bounded, in-memory experiment. A supplied checkpoint is comparison-only."""

    def __init__(self, *, config, camera_id, detector_adapter, checkpoint=None):
        if not isinstance(config, MotionTrackerConfig):
            raise TrackerEvaluationError("invalid fallback tracker config")
        identity = MotionTrackerState.empty(
            camera_id=camera_id,
            detector_adapter=detector_adapter,
            config_fingerprint=config.fingerprint,
        )
        if checkpoint is not None and (
            not isinstance(checkpoint, MotionTrackerState)
            or checkpoint.camera_id != camera_id
            or checkpoint.detector_adapter != detector_adapter
            or checkpoint.config_fingerprint != config.fingerprint
            or len(checkpoint.active_tracks) > config.max_active_tracks
        ):
            raise TrackerEvaluationError("fallback checkpoint differs from configured input")
        self.config = config
        self.camera_id = identity.camera_id
        self.detector_adapter = identity.detector_adapter
        self.config_sha256 = _canonical_sha256(
            {"policy": FALLBACK_REVISION, "motion_config": config.to_dict()}
        )
        self.active = () if checkpoint is None else checkpoint.active_tracks
        # Adapter-specific output IDs must not also change equal-score tie
        # order. Keep the original seed IDs as private matching-order keys.
        self.order_keys = {track.tracking_id: track.tracking_id for track in self.active}
        self.last_frame_at = None if checkpoint is None else checkpoint.last_frame_at
        self.last_fallback_ids: tuple[str, ...] = ()

    def process(self, frame: DetectionFrame) -> tuple[TrackerPredictionAssignment, ...]:
        if not isinstance(frame, DetectionFrame) or (
            self.last_frame_at is not None and frame.occurred_at <= self.last_frame_at
        ):
            raise TrackerEvaluationError("fallback input must be a new chronological frame")
        if len(frame.detections) > self.config.max_frame_detections:
            raise TrackerEvaluationError("fallback frame exceeds detection limit")
        retained = {
            track.tracking_id: track
            for track in self.active
            if (frame.occurred_at - track.current.occurred_at).total_seconds()
            <= self.config.max_track_age_seconds
        }
        ordered = tuple(retained[key] for key in sorted(retained, key=self.order_keys.__getitem__))
        high = tuple(
            d for d in frame.detections if d.confidence >= self.config.high_confidence_threshold
        )
        low = tuple(
            d
            for d in frame.detections
            if self.config.low_confidence_threshold
            <= d.confidence
            < self.config.high_confidence_threshold
        )
        primary = _fallback_match(
            ordered,
            high,
            occurred_at=frame.occurred_at,
            minimum_score=self.config.primary_match_threshold,
            config=self.config,
        )
        used = {i for i, _ in primary}
        remaining = tuple(t for i, t in enumerate(ordered) if i not in used)
        secondary = _fallback_match(
            remaining,
            low,
            occurred_at=frame.occurred_at,
            minimum_score=self.config.secondary_match_threshold,
            config=self.config,
        )
        mapping = {}
        for tracks, detections, pairs in ((ordered, high, primary), (remaining, low, secondary)):
            for i, j in pairs:
                track, detection = tracks[i], detections[j]
                retained[track.tracking_id] = track.update(detection, frame.occurred_at)
                mapping[detection.detection_id] = track.tracking_id
        unmatched = tuple(d for d in frame.detections if d.detection_id not in mapping)
        if (
            len(retained)
            + sum(d.confidence >= self.config.high_confidence_threshold for d in unmatched)
            > self.config.max_active_tracks
        ):
            raise TrackerEvaluationError("fallback active track limit would be exceeded")
        issued = {track.tracking_id for track in self.active}
        order_keys = {key: self.order_keys[key] for key in retained}
        source_batch = DetectionFrameBatch(self.camera_id, self.detector_adapter, (frame,))
        for detection in unmatched:
            tracking_id = (
                "track_"
                + _canonical_sha256(
                    {
                        "adapter": FALLBACK_ADAPTER,
                        "revision": FALLBACK_REVISION,
                        "config": self.config_sha256,
                        "camera": self.camera_id,
                        "detector": self.detector_adapter,
                        "frame": frame.frame_id,
                        "detection": detection.to_dict(),
                        "retained": detection.confidence >= self.config.high_confidence_threshold,
                    }
                )[:24]
            )
            if tracking_id in issued:
                raise TrackerEvaluationError("fallback deterministic ID collision")
            issued.add(tracking_id)
            mapping[detection.detection_id] = tracking_id
            if detection.confidence >= self.config.high_confidence_threshold:
                order_keys[tracking_id] = _new_tracking_id(
                    source_batch,
                    self.config,
                    frame_id=frame.frame_id,
                    detection=detection,
                    retained=True,
                )
                retained[tracking_id] = MotionActiveTrack(
                    tracking_id,
                    detection.label,
                    MotionTrackSample(frame.occurred_at, detection.confidence, detection.bbox_xyxy),
                )
        result = tuple(
            TrackerPredictionAssignment(frame.frame_id, d.detection_id, mapping[d.detection_id])
            for d in frame.detections
        )
        fallback_ids = tuple(
            track.tracking_id
            for track in ordered
            if prediction_fallback_required(
                track, frame.occurred_at, self.config.max_prediction_seconds
            )
        )
        self.active = tuple(retained[key] for key in sorted(retained))
        self.order_keys = order_keys
        self.last_frame_at = frame.occurred_at
        self.last_fallback_ids = fallback_ids
        return result


def build_prediction_fallback_predictions(batch, config):
    if not isinstance(batch, DetectionFrameBatch):
        raise TrackerEvaluationError("fallback batch must be a DetectionFrameBatch")
    tracker = PredictionFallbackTracker(
        config=config, camera_id=batch.camera_id, detector_adapter=batch.detector_adapter
    )
    assignments = tuple(row for frame in batch.frames for row in tracker.process(frame))
    result = TrackerPredictionBatch(
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
        detection_batch_sha256=batch.fingerprint,
        tracker_adapter=FALLBACK_ADAPTER,
        tracker_revision=FALLBACK_REVISION,
        tracker_config_sha256=tracker.config_sha256,
        runtime_id="python_stdlib_offline_candidate_v1",
        assignments=assignments,
    )
    result.bind(batch)
    return result
