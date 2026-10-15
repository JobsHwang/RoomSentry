"""Bounded descriptions of actual matching inputs, never new assignments.

An eligible recent edge need not win global one-to-one/confidence-stage matching.
This opt-in metadata explains dormant reuse without asserting person identity or
choosing a different matcher. No pixels, claims, paths, model or network access.
"""

from __future__ import annotations

from baby_monitor.motion_tracker import (
    MOTION_TRACKER_REVISION,
    MotionTrackerConfig,
    MotionTrackerState,
    _match_score,
    _matching_scores,
    _predict_bbox,
    _predict_bbox_without_fallback,
)
from baby_monitor.object_tracker import DetectionFrame
from baby_monitor.temporal_zone import TrackedObjectObservation


def describe_association(state: MotionTrackerState, frame: DetectionFrame, rows, config):
    """Describe before-state and already-issued outputs, without modifying either."""
    if (
        not isinstance(state, MotionTrackerState)
        or not isinstance(frame, DetectionFrame)
        or not isinstance(config, MotionTrackerConfig)
        or state.config_fingerprint != config.fingerprint
        or len(state.active_tracks) > config.max_active_tracks
        or (state.last_frame_at is not None and frame.occurred_at <= state.last_frame_at)
        or len(frame.detections) > config.max_frame_detections
        or not isinstance(rows, tuple)
        or len(rows) != len(frame.detections)
        or any(not isinstance(row, TrackedObjectObservation) for row in rows)
    ):
        raise ValueError("invalid association audit inputs")
    retained = {
        t.tracking_id: t
        for t in state.active_tracks
        if (frame.occurred_at - t.current.occurred_at).total_seconds()
        <= config.max_track_age_seconds
    }
    ordered = tuple(retained[key] for key in sorted(retained))
    high = tuple(d for d in frame.detections if d.confidence >= config.high_confidence_threshold)
    used = {
        row.tracking_id
        for detection, row in zip(frame.detections, rows, strict=True)
        if detection.confidence >= config.high_confidence_threshold and row.tracking_id in retained
    }
    remaining = tuple(track for track in ordered if track.tracking_id not in used)
    low = tuple(
        d
        for d in frame.detections
        if config.low_confidence_threshold <= d.confidence < config.high_confidence_threshold
    )
    stage_scores, original_protected = {}, {}
    for tracks, detections, threshold in (
        (ordered, high, config.primary_match_threshold),
        (remaining, low, config.secondary_match_threshold),
    ):
        scores = _matching_scores(
            tracks,
            detections,
            occurred_at=frame.occurred_at,
            minimum_score=threshold,
            config=config,
        )
        for j, detection in enumerate(detections):
            stage_scores[detection.detection_id] = {
                track.tracking_id: scores[i][j] for i, track in enumerate(tracks)
            }
            original_protected[detection.detection_id] = any(
                score > 0 and score + 1e-15 >= threshold
                for score in (
                    _match_score(
                        track,
                        detection,
                        occurred_at=frame.occurred_at,
                        config=config,
                        use_fallback=False,
                    )
                    for track in tracks
                )
            )
    result = []
    for index, (detection, row) in enumerate(zip(frame.detections, rows, strict=True)):
        if (
            row.camera_id != state.camera_id
            or row.occurred_at != frame.occurred_at
            or (row.label, row.confidence, row.bbox_xyxy)
            != (
                detection.label,
                detection.confidence,
                detection.bbox_xyxy,
            )
        ):
            raise ValueError("association audit output differs from original detection")
        threshold = (
            config.primary_match_threshold
            if detection.confidence >= config.high_confidence_threshold
            else config.secondary_match_threshold
            if detection.confidence >= config.low_confidence_threshold
            else None
        )
        selected = retained.get(row.tracking_id)
        competitors = []
        if threshold is not None:
            for track in retained.values():
                if track.current.occurred_at != state.last_frame_at:
                    continue
                score = stage_scores[detection.detection_id].get(track.tracking_id, 0.0)
                if score + 1e-15 >= threshold:
                    competitors.append({"track_id": track.tracking_id, "match_score": score})
        competitors.sort(key=lambda t: (-t["match_score"], t["track_id"]))
        result.append(
            {
                "detection_index": index,
                "track_id": row.tracking_id,
                "selected_source": "new_anonymous_id"
                if selected is None
                else "previous_input"
                if selected.current.occurred_at == state.last_frame_at
                else "dormant_retained_track",
                "selected_last_sample_age_seconds": None
                if selected is None
                else (frame.occurred_at - selected.current.occurred_at).total_seconds(),
                "selected_prior_bbox_xyxy": None
                if selected is None
                else list(selected.current.bbox_xyxy),
                "selected_predicted_bbox_xyxy": None
                if selected is None
                else list(
                    (
                        _predict_bbox_without_fallback
                        if original_protected.get(detection.detection_id, False)
                        else _predict_bbox
                    )(selected, frame.occurred_at, config.max_prediction_seconds)
                ),
                "selected_match_score": None
                if selected is None
                else stage_scores.get(detection.detection_id, {}).get(selected.tracking_id),
                "stage_threshold": threshold,
                "eligible_previous_input_track_count": len(competitors),
                "eligible_previous_input_tracks": competitors[:8],
                "eligible_previous_input_tracks_truncated": len(competitors) > 8,
            }
        )
    return {
        "version": 1,
        "input_frame_id": frame.frame_id,
        "tracker_revision": MOTION_TRACKER_REVISION,
        "tracker_config_sha256": config.fingerprint,
        "retained_before_count": len(retained),
        "expired_before_count": len(state.active_tracks) - len(retained),
        "detections": result,
        "eligible_edge_does_not_establish_correct_assignment": True,
        "accuracy_measured": False,
    }
