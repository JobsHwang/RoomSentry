"""Replay an actual complete matching checkpoint and its exported input interval.

This verifies the original assignments before comparing one fixed predictor
candidate. It cannot establish person identity or consume pixels/model results.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from baby_monitor.motion_tracker import (
    MOTION_TRACKER_REVISION,
    MotionObjectTracker,
    MotionTrackerConfig,
    MotionTrackerState,
)
from baby_monitor.object_tracker import DetectionFrame, DetectionFrameBatch, FrameDetection
from baby_monitor.tracker_evaluation._shared import (
    TrackerEvaluationError,
    _canonical_sha256,
    _sha256,
)
from baby_monitor.tracker_evaluation.prediction_fallback import (
    FALLBACK_ADAPTER,
    FALLBACK_REVISION,
    PredictionFallbackTracker,
)

ANCHOR = datetime(2000, 1, 1, tzinfo=UTC)


def _inputs(trace):
    if (
        not isinstance(trace, dict)
        or type(trace.get("version")) is not int
        or trace.get("version") != 1
        or trace.get("status") != "bound_interval"
        or type(trace.get("omitted_middle_frames")) is not int
        or trace.get("omitted_middle_frames") != 0
        or not isinstance(trace.get("frames"), list)
        or not 1 <= len(trace["frames"]) <= 256
        or type(trace.get("interval_frames")) is not int
        or trace.get("interval_frames") != len(trace["frames"])
        or type(trace.get("exported_frames")) is not int
        or trace.get("exported_frames") != len(trace["frames"])
    ):
        raise TrackerEvaluationError("checkpoint replay requires a complete bound interval")
    checkpoint = trace.get("association_checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint)
        != {
            "version",
            "before_state",
            "tracker_config",
            "input_frame_id",
            "full_active_track_state",
            "omitted_processed_dedup_frames",
            "pending_delivery_outbox_included",
            "matching_replay_only",
        }
        or (
            type(checkpoint["version"]) is not int
            or checkpoint["version"] != 1
            or checkpoint["full_active_track_state"] is not True
            or checkpoint["pending_delivery_outbox_included"] is not False
            or checkpoint["matching_replay_only"] is not True
            or type(checkpoint["omitted_processed_dedup_frames"]) is not int
            or checkpoint["omitted_processed_dedup_frames"] < 0
        )
    ):
        raise TrackerEvaluationError("complete association checkpoint is unavailable")
    config = MotionTrackerConfig.from_dict(checkpoint["tracker_config"])
    before = MotionTrackerState.from_dict(checkpoint["before_state"])
    if (
        before.camera_id != trace["camera_id"]
        or before.config_fingerprint != config.fingerprint
        or before.pending_observations
        or len(before.processed_frames) > 1
        or len(before.active_tracks) > config.max_active_tracks
    ):
        raise TrackerEvaluationError("association checkpoint source/config differs")
    first = trace["frames"][0]
    if (
        first["generation"] != trace["review_frame_generation"]
        or first["sequence"] != trace["review_frame_sequence"]
        or first["raw_sha256"] != trace["review_raw_bgr_sha256"]
        or first["view"] != trace["detection_view"]
    ):
        raise TrackerEvaluationError("checkpoint does not bind the submitted original input")
    frames, expected = [], []
    prior = None
    for row in trace["frames"]:
        if (
            any(
                type(row.get(k)) is not int or row[k] < 0
                for k in ("generation", "sequence", "ordinal")
            )
            or row.get("view")
            not in (
                ("original",)
                if trace["detection_view"] == "original"
                else ("rotation_180", "rotation_180_scale_fallback")
            )
            or any(
                type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0
                for k in ("received_monotonic_seconds", "completed_monotonic_seconds")
            )
            or row["completed_monotonic_seconds"] < row["received_monotonic_seconds"]
        ):
            raise TrackerEvaluationError("invalid checkpoint interval metadata")
        _sha256(row["raw_sha256"], "checkpoint interval raw SHA")
        audit = row.get("association_audit")
        if (
            not isinstance(audit, dict)
            or audit.get("tracker_revision") != MOTION_TRACKER_REVISION
            or audit.get("tracker_config_sha256") != config.fingerprint
            or row["generation"] != first["generation"]
            or len(row["detections"]) != len(audit.get("detections", []))
        ):
            raise TrackerEvaluationError("checkpoint interval differs from matching audit")
        when = datetime.fromisoformat(row["received_at"])
        if when.tzinfo is None or when.utcoffset() is None:
            raise TrackerEvaluationError("checkpoint receipt UTC is not timezone-aware")
        if prior is not None and (
            row["ordinal"] != prior["ordinal"] + 1
            or row["sequence"] <= prior["sequence"]
            or row["received_monotonic_seconds"] <= prior["received_monotonic_seconds"]
            or when <= datetime.fromisoformat(prior["received_at"])
        ):
            raise TrackerEvaluationError("checkpoint interval has missing or reordered inputs")
        frame = DetectionFrame(
            audit["input_frame_id"],
            ANCHOR + timedelta(seconds=row["received_monotonic_seconds"]),
            tuple(
                FrameDetection(
                    f"d_{i:03}", "person", detection["confidence"], tuple(detection["bbox_xyxy"])
                )
                for i, detection in enumerate(row["detections"])
            ),
        )
        for i, (detection, original) in enumerate(
            zip(row["detections"], audit["detections"], strict=True)
        ):
            if original["detection_index"] != i or original["track_id"] != detection["track_id"]:
                raise TrackerEvaluationError("checkpoint original assignments differ from audit")
            expected.append([frame.frame_id, f"d_{i:03}", detection["track_id"]])
        frames.append(frame)
        prior = row
    if frames[0].frame_id != checkpoint["input_frame_id"]:
        raise TrackerEvaluationError("checkpoint initial frame identity differs")
    batch = DetectionFrameBatch(before.camera_id, before.detector_adapter, tuple(frames))
    return config, before, batch, expected


def _links(assignments):
    grouped = {}
    for frame, detection, track in assignments:
        grouped.setdefault(track, []).append((frame, detection))
    return {(*a, *b) for rows in grouped.values() for a, b in zip(rows, rows[1:], strict=False)}


def compare_prediction_fallback_checkpoint(trace):
    """The complete original matching state is required; cold-start guesses fail."""
    try:
        config, before, batch, expected = _inputs(trace)
        baseline = MotionObjectTracker(
            config=config,
            camera_id=batch.camera_id,
            detector_adapter=batch.detector_adapter,
            state=before,
        )
        candidate = PredictionFallbackTracker(
            config=config,
            camera_id=batch.camera_id,
            detector_adapter=batch.detector_adapter,
            checkpoint=before,
        )
        original_rows, candidate_rows, fallbacks, first_changes = [], [], [], []
        original_origins = {
            t.tracking_id: ["checkpoint", t.tracking_id] for t in before.active_tracks
        }
        candidate_origins = dict(original_origins)
        for row, frame in zip(trace["frames"], batch.frames, strict=True):
            actual = baseline.process(frame)
            proposed = candidate.process(frame)
            actual_ids = [o.tracking_id for o in actual]
            proposed_ids = [o.tracking_id for o in proposed]
            for detection, original_row, proposed_row in zip(
                frame.detections, actual, proposed, strict=True
            ):
                origin = ["input", frame.frame_id, detection.detection_id]
                original_origins.setdefault(original_row.tracking_id, origin)
                candidate_origins.setdefault(proposed_row.tracking_id, origin)
            original_sources = [original_origins[tracking_id] for tracking_id in actual_ids]
            candidate_sources = [candidate_origins[tracking_id] for tracking_id in proposed_ids]
            original_rows.extend(
                [frame.frame_id, d.detection_id, o.tracking_id]
                for d, o in zip(frame.detections, actual, strict=True)
            )
            candidate_rows.extend([o.frame_id, o.detection_id, o.tracking_id] for o in proposed)
            if candidate.last_fallback_ids:
                fallbacks.append(
                    {
                        "sequence": row["sequence"],
                        "ordinal": row["ordinal"],
                        "track_ids": list(candidate.last_fallback_ids),
                    }
                )
            if original_sources != candidate_sources and len(first_changes) < 8:
                first_changes.append(
                    {
                        "sequence": row["sequence"],
                        "ordinal": row["ordinal"],
                        "original_ids": actual_ids,
                        "candidate_ids": proposed_ids,
                        "original_origins": original_sources,
                        "candidate_origins": candidate_sources,
                        "current_geometry_ambiguous": [
                            d["association_ambiguous"] for d in row["detections"]
                        ],
                    }
                )
            baseline.state = replace(
                baseline.state,
                pending_observations=(),
                processed_frames=baseline.state.processed_frames[-1:],
            )
        if original_rows != expected:
            raise TrackerEvaluationError("checkpoint does not reproduce actual original matching")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise TrackerEvaluationError("invalid or nonreproducible association checkpoint") from exc
    original_links, candidate_links = _links(original_rows), _links(candidate_rows)
    return {
        "version": 1,
        "trace_sha256": _canonical_sha256(trace),
        "checkpoint_sha256": _canonical_sha256(trace["association_checkpoint"]),
        "input_batch_sha256": batch.fingerprint,
        "frame_count": len(batch.frames),
        "assignment_count": len(expected),
        "original_assignments_reproduced": True,
        "initial_active_track_count": len(before.active_tracks),
        "candidate_adapter": FALLBACK_ADAPTER,
        "candidate_revision": FALLBACK_REVISION,
        "candidate_config_sha256": candidate.config_sha256,
        "fallback_input_count": len(fallbacks),
        "fallbacks": fallbacks,
        "original_link_count": len(original_links),
        "candidate_link_count": len(candidate_links),
        "shared_link_count": len(original_links & candidate_links),
        "original_only_links": [list(v) for v in sorted(original_links - candidate_links)],
        "candidate_only_links": [list(v) for v in sorted(candidate_links - original_links)],
        "first_assignment_changes": first_changes,
        "original_assignments": original_rows,
        "candidate_assignments": candidate_rows,
        "human_ground_truth": False,
        "accuracy_measured": False,
        "winner_selected": False,
        "live_config_changed": False,
        "role_inheritance": False,
        "media_read": False,
        "extra_model_calls": 0,
    }
