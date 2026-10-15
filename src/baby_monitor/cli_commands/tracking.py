"""Tracker, association review, disagreement review, and temporal zone commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from baby_monitor.cli_support import (  # noqa: F401
    _absolute_runner_executable,
    _aware_datetime,
    _configured_remote_adapter,
    _database_from_path,
    _enqueue_buffered_candidate,
    _json_print,
    _local_adapter,
    _remote_adapter,
    _review_policy,
    _secondary_adapter,
    _storage_override,
    _yolo_detector,
)

from ..caviar_tracking import import_caviar_tracking_ground_truth
from ..motion_tracker import (
    MOTION_TRACKER_ADAPTER_ID,
    MotionObjectTracker,
    MotionTrackerConfig,
    MotionTrackerState,
)
from ..object_tracker import (
    DetectionFrameBatch,
    IoUObjectTracker,
    IoUTrackerConfig,
    IoUTrackerState,
)
from ..source_admission import ExternalSourceAdmission
from ..temporal_zone import (
    TemporalZoneAssembler,
    TemporalZoneConfig,
    TemporalZoneState,
    load_tracked_observations,
)
from ..tracker_evaluation import (
    TrackerAssociationReport,
    TrackerAssociationReview,
    TrackerDisagreementReport,
    TrackerDisagreementReview,
    TrackerPredictionBatch,
    TrackerPredictionComparison,
    build_iou_tracker_predictions,
    build_motion_tracker_predictions,
    compare_tracker_predictions,
    evaluate_tracker_association,
    evaluate_tracker_disagreement,
)


def _tracker_disagreement_inputs(
    args: argparse.Namespace,
) -> tuple[
    DetectionFrameBatch,
    TrackerPredictionBatch,
    TrackerPredictionBatch,
    TrackerPredictionComparison,
]:
    return (
        DetectionFrameBatch.read(args.frames),
        TrackerPredictionBatch.read(args.left),
        TrackerPredictionBatch.read(args.right),
        TrackerPredictionComparison.read(args.comparison),
    )


def _tracker_state_for_batch(
    state_path: str | Path,
    *,
    config: IoUTrackerConfig,
    batch: DetectionFrameBatch,
) -> IoUTrackerState:
    path = Path(state_path).expanduser()
    if path.exists():
        return IoUTrackerState.read(path)
    return IoUTrackerState.empty(
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
        config_fingerprint=config.fingerprint,
    )


def _motion_tracker_state_for_batch(
    state_path: str | Path,
    *,
    config: MotionTrackerConfig,
    batch: DetectionFrameBatch,
) -> MotionTrackerState:
    path = Path(state_path).expanduser()
    if path.exists():
        return MotionTrackerState.read(path)
    return MotionTrackerState.empty(
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
        config_fingerprint=config.fingerprint,
    )


def _temporal_state_for_config(
    state_path: str | Path,
    config: TemporalZoneConfig,
) -> TemporalZoneState:
    path = Path(state_path).expanduser()
    return TemporalZoneState.read(path) if path.exists() else TemporalZoneState.empty(config)


def command_validate_tracked_observations(args: argparse.Namespace) -> int:
    observations = load_tracked_observations(args.path)
    _json_print(
        {
            "valid": True,
            "version": 1,
            "observations": len(observations),
            "paths_disclosed": False,
        }
    )
    return 0


def command_validate_detection_frames(args: argparse.Namespace) -> int:
    batch = DetectionFrameBatch.read(args.path)
    _json_print(
        {
            "valid": True,
            "version": 1,
            "camera_id": batch.camera_id,
            "detector_adapter": batch.detector_adapter,
            "frames": len(batch.frames),
            "detections": sum(len(frame.detections) for frame in batch.frames),
            "paths_disclosed": False,
        }
    )
    return 0


def command_prepare_tracker_review(args: argparse.Namespace) -> int:
    batch = DetectionFrameBatch.read(args.frames)
    review = TrackerAssociationReview.pending(
        batch,
        source_admission_sha256=args.source_admission_sha256,
    )
    review.write_new(args.output)
    _json_print(review.to_public_dict())
    return 0


def command_import_caviar_tracking_ground_truth(args: argparse.Namespace) -> int:
    frames_output = Path(args.frames_output).expanduser()
    review_output = Path(args.review_output).expanduser()
    if frames_output == review_output:
        raise ValueError("CAVIAR frame and review outputs must be different")
    if frames_output.exists() or review_output.exists():
        raise FileExistsError("CAVIAR import outputs must not already exist")
    imported = import_caviar_tracking_ground_truth(
        args.xml,
        args.source_admission,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
    )
    imported.batch.write_new(frames_output)
    imported.review.write_new(review_output)
    _json_print(imported.to_public_dict())
    return 0


def command_validate_tracker_review(args: argparse.Namespace) -> int:
    batch = DetectionFrameBatch.read(args.frames)
    review = TrackerAssociationReview.read(args.review)
    review.bind(batch)
    admission = (
        ExternalSourceAdmission.read(args.source_admission)
        if args.source_admission is not None
        else None
    )
    review.bind_source_admission(admission)
    _json_print(review.to_public_dict())
    return 0


def command_track_iou_predict(args: argparse.Namespace) -> int:
    config = IoUTrackerConfig.read(args.tracker_config)
    batch = DetectionFrameBatch.read(args.frames)
    predictions = build_iou_tracker_predictions(batch, config)
    predictions.write_new(args.output)
    _json_print(predictions.to_public_dict())
    return 0


def command_track_motion_predict(args: argparse.Namespace) -> int:
    config = MotionTrackerConfig.read(args.tracker_config)
    batch = DetectionFrameBatch.read(args.frames)
    predictions = build_motion_tracker_predictions(batch, config)
    predictions.write_new(args.output)
    _json_print(predictions.to_public_dict())
    return 0


def command_validate_tracker_predictions(args: argparse.Namespace) -> int:
    batch = DetectionFrameBatch.read(args.frames)
    predictions = TrackerPredictionBatch.read(args.predictions)
    predictions.bind(batch)
    _json_print(predictions.to_public_dict())
    return 0


def command_compare_tracker_predictions(args: argparse.Namespace) -> int:
    batch = DetectionFrameBatch.read(args.frames)
    left = TrackerPredictionBatch.read(args.left)
    right = TrackerPredictionBatch.read(args.right)
    comparison = compare_tracker_predictions(batch, left, right)
    comparison.write_new(args.output)
    _json_print(comparison.to_public_dict())
    return 0


def command_validate_tracker_comparison(args: argparse.Namespace) -> int:
    comparison = TrackerPredictionComparison.read(args.path)
    _json_print(comparison.to_public_dict())
    return 0


def command_prepare_tracker_disagreement_review(args: argparse.Namespace) -> int:
    batch, left, right, comparison = _tracker_disagreement_inputs(args)
    review = TrackerDisagreementReview.pending(batch, left, right, comparison)
    review.write_new(args.output)
    _json_print(review.to_public_dict())
    return 0


def command_validate_tracker_disagreement_review(args: argparse.Namespace) -> int:
    batch, left, right, comparison = _tracker_disagreement_inputs(args)
    review = TrackerDisagreementReview.read(args.review)
    review.bind(batch, left, right, comparison)
    _json_print(review.to_public_dict())
    return 0


def command_evaluate_tracker_disagreement(args: argparse.Namespace) -> int:
    batch, left, right, comparison = _tracker_disagreement_inputs(args)
    review = TrackerDisagreementReview.read(args.review)
    report = evaluate_tracker_disagreement(
        batch,
        left,
        right,
        comparison,
        review,
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_tracker_disagreement(args: argparse.Namespace) -> int:
    report = TrackerDisagreementReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_evaluate_tracker_association(args: argparse.Namespace) -> int:
    batch = DetectionFrameBatch.read(args.frames)
    review = TrackerAssociationReview.read(args.review)
    predictions = TrackerPredictionBatch.read(args.predictions)
    admission = (
        ExternalSourceAdmission.read(args.source_admission)
        if args.source_admission is not None
        else None
    )
    report = evaluate_tracker_association(
        batch,
        review,
        predictions,
        source_admission=admission,
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_tracker_association(args: argparse.Namespace) -> int:
    report = TrackerAssociationReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_track_iou_run(args: argparse.Namespace) -> int:
    config = IoUTrackerConfig.read(args.tracker_config)
    batch = DetectionFrameBatch.read(args.frames)
    state = _tracker_state_for_batch(args.state, config=config, batch=batch)
    tracker = IoUObjectTracker(
        config=config,
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
        state=state,
    )
    new_frames = 0
    replayed_frames = 0
    observations_added = 0
    for frame in batch.frames:
        already_processed = any(
            item.frame_id == frame.frame_id for item in tracker.state.processed_frames
        )
        observations = tracker.process(frame)
        if already_processed:
            replayed_frames += 1
            continue
        new_frames += 1
        observations_added += len(observations)
        tracker.state.write_atomic(args.state)
    _json_print(
        {
            "version": 1,
            "frames_received": len(batch.frames),
            "new_frames": new_frames,
            "replayed_frames": replayed_frames,
            "observations_added": observations_added,
            "pending_observations": [
                observation.to_dict() for observation in tracker.state.pending_observations
            ],
            "pending_observation_count": len(tracker.state.pending_observations),
            "active_track_count": len(tracker.state.active_tracks),
            "state_written": new_frames > 0,
            "paths_disclosed": False,
        }
    )
    return 0


def command_track_iou_pending(args: argparse.Namespace) -> int:
    state = IoUTrackerState.read(args.state)
    _json_print(
        {
            "version": 1,
            "pending_observations": [
                observation.to_dict() for observation in state.pending_observations
            ],
            "pending_observation_count": len(state.pending_observations),
            "paths_disclosed": False,
        }
    )
    return 0


def command_track_iou_ack(args: argparse.Namespace) -> int:
    state = IoUTrackerState.read(args.state)
    updated = state.acknowledge(tuple(args.observation_id))
    updated.write_atomic(args.state)
    _json_print(
        {
            "version": 1,
            "acknowledged": len(args.observation_id),
            "pending_observation_count": len(updated.pending_observations),
            "state_written": True,
            "paths_disclosed": False,
        }
    )
    return 0


def command_track_iou_zone_run(args: argparse.Namespace) -> int:
    tracker_path = Path(args.tracker_state).expanduser()
    zone_path = Path(args.zone_state).expanduser()
    if tracker_path.resolve() == zone_path.resolve():
        raise ValueError("tracker and temporal zone state paths must differ")
    tracker_state = IoUTrackerState.read(tracker_path)
    zone_config = TemporalZoneConfig.read(args.zone_config)
    zone_state = _temporal_state_for_config(zone_path, zone_config)
    observations = tracker_state.pending_observations
    assembler = TemporalZoneAssembler(config=zone_config, state=zone_state)
    emitted = assembler.process_all(observations)
    if observations:
        assembler.state.write_atomic(zone_path)
        tracker_state = tracker_state.acknowledge(
            tuple(observation.observation_id for observation in observations)
        )
        tracker_state.write_atomic(tracker_path)
    _json_print(
        {
            "version": 1,
            "processed_observations": len(observations),
            "emitted_signals": [signal.to_dict() for signal in emitted],
            "tracker_pending_observation_count": len(tracker_state.pending_observations),
            "zone_pending_signal_count": len(assembler.state.pending_signals),
            "states_written": bool(observations),
            "paths_disclosed": False,
        }
    )
    return 0


def command_track_motion_run(args: argparse.Namespace) -> int:
    config = MotionTrackerConfig.read(args.tracker_config)
    batch = DetectionFrameBatch.read(args.frames)
    state = _motion_tracker_state_for_batch(
        args.state,
        config=config,
        batch=batch,
    )
    tracker = MotionObjectTracker(
        config=config,
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
        state=state,
    )
    new_frames = 0
    replayed_frames = 0
    observations_added = 0
    for frame in batch.frames:
        already_processed = any(
            item.frame_id == frame.frame_id for item in tracker.state.processed_frames
        )
        observations = tracker.process(frame)
        if already_processed:
            replayed_frames += 1
            continue
        new_frames += 1
        observations_added += len(observations)
        tracker.state.write_atomic(args.state)
    _json_print(
        {
            "version": 1,
            "tracker_adapter": MOTION_TRACKER_ADAPTER_ID,
            "frames_received": len(batch.frames),
            "new_frames": new_frames,
            "replayed_frames": replayed_frames,
            "observations_added": observations_added,
            "pending_observations": [
                observation.to_dict() for observation in tracker.state.pending_observations
            ],
            "pending_observation_count": len(tracker.state.pending_observations),
            "active_track_count": len(tracker.state.active_tracks),
            "state_written": new_frames > 0,
            "paths_disclosed": False,
        }
    )
    return 0


def command_track_motion_pending(args: argparse.Namespace) -> int:
    state = MotionTrackerState.read(args.state)
    _json_print(
        {
            "version": 1,
            "tracker_adapter": MOTION_TRACKER_ADAPTER_ID,
            "pending_observations": [
                observation.to_dict() for observation in state.pending_observations
            ],
            "pending_observation_count": len(state.pending_observations),
            "paths_disclosed": False,
        }
    )
    return 0


def command_track_motion_ack(args: argparse.Namespace) -> int:
    state = MotionTrackerState.read(args.state)
    updated = state.acknowledge(tuple(args.observation_id))
    updated.write_atomic(args.state)
    _json_print(
        {
            "version": 1,
            "tracker_adapter": MOTION_TRACKER_ADAPTER_ID,
            "acknowledged": len(args.observation_id),
            "pending_observation_count": len(updated.pending_observations),
            "state_written": True,
            "paths_disclosed": False,
        }
    )
    return 0


def command_track_motion_zone_run(args: argparse.Namespace) -> int:
    tracker_path = Path(args.tracker_state).expanduser()
    zone_path = Path(args.zone_state).expanduser()
    if tracker_path.resolve() == zone_path.resolve():
        raise ValueError("tracker and temporal zone state paths must differ")
    tracker_state = MotionTrackerState.read(tracker_path)
    zone_config = TemporalZoneConfig.read(args.zone_config)
    zone_state = _temporal_state_for_config(zone_path, zone_config)
    observations = tracker_state.pending_observations
    assembler = TemporalZoneAssembler(config=zone_config, state=zone_state)
    emitted = assembler.process_all(observations)
    if observations:
        assembler.state.write_atomic(zone_path)
        tracker_state = tracker_state.acknowledge(
            tuple(observation.observation_id for observation in observations)
        )
        tracker_state.write_atomic(tracker_path)
    _json_print(
        {
            "version": 1,
            "tracker_adapter": MOTION_TRACKER_ADAPTER_ID,
            "processed_observations": len(observations),
            "emitted_signals": [signal.to_dict() for signal in emitted],
            "tracker_pending_observation_count": len(tracker_state.pending_observations),
            "zone_pending_signal_count": len(assembler.state.pending_signals),
            "states_written": bool(observations),
            "paths_disclosed": False,
        }
    )
    return 0


def command_temporal_zone_run(args: argparse.Namespace) -> int:
    config = TemporalZoneConfig.read(args.zone_config)
    observations = load_tracked_observations(args.observations)
    state = _temporal_state_for_config(args.state, config)
    assembler = TemporalZoneAssembler(config=config, state=state)
    emitted = assembler.process_all(observations)
    assembler.state.write_atomic(args.state)
    _json_print(
        {
            "version": 1,
            "processed_observations": len(observations),
            "emitted_signals": [signal.to_dict() for signal in emitted],
            "pending_signal_count": len(assembler.state.pending_signals),
            "state_written": True,
            "paths_disclosed": False,
        }
    )
    return 0


def command_temporal_zone_pending(args: argparse.Namespace) -> int:
    state = TemporalZoneState.read(args.state)
    _json_print(
        {
            "version": 1,
            "pending_signals": [signal.to_dict() for signal in state.pending_signals],
            "pending_signal_count": len(state.pending_signals),
            "paths_disclosed": False,
        }
    )
    return 0


def command_temporal_zone_ack(args: argparse.Namespace) -> int:
    state = TemporalZoneState.read(args.state)
    updated = state.acknowledge(tuple(args.signal_id))
    updated.write_atomic(args.state)
    _json_print(
        {
            "version": 1,
            "acknowledged": len(args.signal_id),
            "pending_signal_count": len(updated.pending_signals),
            "state_written": True,
            "paths_disclosed": False,
        }
    )
    return 0
