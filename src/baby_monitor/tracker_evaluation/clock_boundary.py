"""Single actual UTC-only reset counterfactual, never role evidence or a live fix.

The empty/reset baseline must reproduce the already-issued assignment and audit.
The independent geometry arm starts from the full discarded matching checkpoint.
It does not replay the whole session, infer identity, or relax dual-clock freshness.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from baby_monitor.live_people_association import describe_association
from baby_monitor.motion_tracker import (
    MOTION_TRACKER_REVISION,
    MotionObjectTracker,
    MotionTrackerConfig,
    MotionTrackerState,
)
from baby_monitor.object_tracker import DetectionFrame
from baby_monitor.tracker_evaluation._shared import (
    TrackerEvaluationError,
    _canonical_sha256,
    _sha256,
)

CLOCK_BOUNDARY_REVISION = "utc_only_geometry_checkpoint_v1"
ANCHOR = datetime(2000, 1, 1, tzinfo=UTC)
FIELDS = {
    "version",
    "camera_id",
    "detection_view",
    "previous_detection_view",
    "boundary",
    "tracker_config",
    "before_state",
    "previous_input",
    "previous_raw_sha256",
    "previous_ordinal",
    "previous_assignment_ids",
    "full_active_track_state",
    "omitted_processed_dedup_frames",
    "pending_delivery_outbox_included",
    "role_evidence_included",
    "matching_replay_only",
    "input_frame",
    "input_raw_sha256",
    "ordinal",
    "original_assignment_ids",
    "original_association_audit",
}


def _aware(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("boundary UTC is not timezone-aware")
    return stamp


def compare_utc_only_reset_checkpoint(checkpoint):
    """Reject incomplete/mixed causes and reproduce actual baseline before compare."""
    try:
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint) != FIELDS
            or type(checkpoint.get("version")) is not int
            or checkpoint["version"] != 1
            or checkpoint.get("full_active_track_state") is not True
            or checkpoint.get("matching_replay_only") is not True
            or checkpoint.get("role_evidence_included") is not False
            or checkpoint.get("pending_delivery_outbox_included") is not False
            or type(checkpoint.get("omitted_processed_dedup_frames")) is not int
            or checkpoint["omitted_processed_dedup_frames"] < 0
            or checkpoint.get("detection_view")
            not in ("original", "rotation_180", "rotation_180_scale_fallback")
            or checkpoint.get("previous_detection_view")
            not in (
                ("original",)
                if checkpoint.get("detection_view") == "original"
                else ("rotation_180", "rotation_180_scale_fallback")
            )
        ):
            raise ValueError("complete reset checkpoint is unavailable")
        boundary = checkpoint["boundary"]
        if (
            type(boundary["version"]) is not int
            or boundary["version"] != 1
            or boundary["reasons"] not in (["utc_gap"], ["non_increasing_utc"])
            or boundary["diagnostic_only"] is not True
            or boundary["camera_exposure_clock_verified"] is not False
        ):
            raise ValueError("not an actual isolated UTC boundary")
        for field in ("previous_generation", "generation", "previous_sequence", "sequence"):
            if type(boundary[field]) is not int or boundary[field] < 0:
                raise ValueError("invalid reset ordering")
        if (
            boundary["previous_generation"] != boundary["generation"]
            or boundary["sequence"] <= boundary["previous_sequence"]
        ):
            raise ValueError("generation/order boundary cannot preserve geometry")
        for field in (
            "previous_received_monotonic_seconds",
            "received_monotonic_seconds",
            "elapsed_receipt_gap_seconds",
            "utc_receipt_gap_seconds",
            "track_gap_seconds",
        ):
            if type(boundary[field]) not in (int, float) or not math.isfinite(boundary[field]):
                raise ValueError("invalid reset clock")
        mono = boundary["received_monotonic_seconds"]
        previous_mono = boundary["previous_received_monotonic_seconds"]
        elapsed = mono - previous_mono
        wall = (
            _aware(boundary["received_at"]) - _aware(boundary["previous_received_at"])
        ).total_seconds()
        limit = boundary["track_gap_seconds"]
        if (
            min(mono, previous_mono) < 0
            or not 0.1 <= limit <= 10
            or not 0 < elapsed <= limit
            or elapsed != boundary["elapsed_receipt_gap_seconds"]
            or wall != boundary["utc_receipt_gap_seconds"]
            or boundary["reasons"]
            != (["utc_gap"] if wall > limit else ["non_increasing_utc"] if wall <= 0 else [])
        ):
            raise ValueError("boundary cause differs from the actual adjacent clocks")
        config = MotionTrackerConfig.from_dict(checkpoint["tracker_config"])
        before = MotionTrackerState.from_dict(checkpoint["before_state"])
        previous = DetectionFrame.from_dict(checkpoint["previous_input"])
        frame = DetectionFrame.from_dict(checkpoint["input_frame"])
        old_parts, new_parts = previous.frame_id.rsplit("_", 3), frame.frame_id.rsplit("_", 3)
        if (
            len(old_parts) != 4
            or len(new_parts) != 4
            or old_parts[0] != new_parts[0]
            or not old_parts[0].startswith("f_")
            or int(new_parts[1]) != int(old_parts[1]) + 1
            or [int(v) for v in old_parts[2:]]
            != [boundary["previous_generation"], boundary["previous_sequence"]]
            or [int(v) for v in new_parts[2:]] != [boundary["generation"], boundary["sequence"]]
        ):
            raise ValueError("reset input session/segment/order differs")
        for field in ("previous_raw_sha256", "input_raw_sha256"):
            _sha256(checkpoint[field], field)
        for field in ("previous_ordinal", "ordinal"):
            if type(checkpoint[field]) is not int or checkpoint[field] < 1:
                raise ValueError("invalid actual input ordinal")
        if (
            checkpoint["ordinal"] != checkpoint["previous_ordinal"] + 1
            or before.camera_id != checkpoint["camera_id"]
            or before.detector_adapter != "live_yolo"
            or before.config_fingerprint != config.fingerprint
            or before.pending_observations
            or len(before.processed_frames) != 1
            or len(before.active_tracks) > config.max_active_tracks
            or max(len(frame.detections), len(previous.detections)) > config.max_frame_detections
            or previous.occurred_at != ANCHOR + timedelta(seconds=previous_mono)
            or frame.occurred_at != ANCHOR + timedelta(seconds=mono)
            or before.last_frame_at != previous.occurred_at
            or before.processed_frames[0].frame_id != previous.frame_id
            or before.processed_frames[0].fingerprint != previous.fingerprint
            or frame.frame_id == previous.frame_id
        ):
            raise ValueError("reset checkpoint/source/adjacent input differs")
        previous_ids = checkpoint["previous_assignment_ids"]
        if (
            not isinstance(previous_ids, list)
            or len(set(previous_ids)) != len(previous_ids)
            or len(previous_ids) != len(previous.detections)
        ):
            raise ValueError("invalid previous input assignments")
        active = {t.tracking_id: t for t in before.active_tracks}
        for detection, track_id in zip(previous.detections, previous_ids, strict=True):
            sample = active[track_id].current
            if active[track_id].label != detection.label or (
                sample.occurred_at,
                sample.confidence,
                sample.bbox_xyxy,
            ) != (previous.occurred_at, detection.confidence, detection.bbox_xyxy):
                raise ValueError("visible assignment does not bind the complete before-state")
        baseline = MotionObjectTracker(
            config=config, camera_id=before.camera_id, detector_adapter=before.detector_adapter
        )
        baseline_before = baseline.state
        original = baseline.process(frame)
        audit = describe_association(baseline_before, frame, original, config)
        original_ids = [o.tracking_id for o in original]
        if (
            original_ids != checkpoint["original_assignment_ids"]
            or audit != checkpoint["original_association_audit"]
            or audit["tracker_revision"] != MOTION_TRACKER_REVISION
        ):
            raise ValueError("reset baseline does not reproduce the actual assignment/audit")
        candidate = MotionObjectTracker(
            config=config,
            camera_id=before.camera_id,
            detector_adapter=before.detector_adapter,
            state=before,
        )
        proposed = candidate.process(frame)
        candidate_ids = [o.tracking_id for o in proposed]
        proposed_audit = describe_association(before, frame, proposed, config)
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise TrackerEvaluationError(
            "invalid or nonreproducible UTC-only reset checkpoint"
        ) from exc
    links = []
    for detection, track_id in zip(frame.detections, candidate_ids, strict=True):
        if track_id in previous_ids:
            links.append(
                [
                    previous.frame_id,
                    previous.detections[previous_ids.index(track_id)].detection_id,
                    frame.frame_id,
                    detection.detection_id,
                ]
            )
    return {
        "version": 1,
        "candidate_revision": CLOCK_BOUNDARY_REVISION,
        "checkpoint_sha256": _canonical_sha256(checkpoint),
        "tracker_config_sha256": config.fingerprint,
        "actual_reset_assignments_reproduced": True,
        "actual_reset_audit_reproduced": True,
        "boundary_reasons": list(boundary["reasons"]),
        "before_active_tracks": len(before.active_tracks),
        "previous_visible_detections": len(previous.detections),
        "boundary_detections": len(frame.detections),
        "original_assignment_ids": original_ids,
        "candidate_assignment_ids": candidate_ids,
        "candidate_association_audit": proposed_audit,
        "candidate_adjacent_geometry_links": links,
        "candidate_dormant_reuses": sum(
            t in active and t not in previous_ids for t in candidate_ids
        ),
        "role_inheritance": False,
        "role_policy_evaluated": False,
        "old_async_review_admitted": False,
        "live_config_changed": False,
        "single_boundary_not_continuous_session": True,
        "raw_pixels_hash_independently_rechecked": False,
        "human_ground_truth": False,
        "accuracy_measured": False,
        "winner_selected": False,
        "media_read": False,
        "extra_model_calls": 0,
    }
