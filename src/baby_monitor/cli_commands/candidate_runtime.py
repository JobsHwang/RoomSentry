"""Candidate worker, health, presence, soak, and room presence commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal as signal_module
from datetime import UTC, datetime
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

from ..adapters.detector import (
    Detector,
    FFmpegSceneDetector,
    StaticDetector,
)
from ..adapters.pose import CommandYoloPoseEstimator
from ..analysis_queue import AnalysisQueue
from ..candidate_health import (
    CandidateHealthLifetime,
    CandidateHealthState,
    CandidateWorkerHealth,
    assess_candidate_health,
)
from ..candidate_presence import (
    candidate_health_gap_sample,
    merge_stream_presence_batches,
)
from ..candidate_soak import (
    finish_candidate_soak,
    inspect_candidate_soak,
    start_candidate_soak,
)
from ..candidate_worker import (
    CandidateWorker,
    CandidateWorkerSettings,
    CandidateWorkerStopReason,
)
from ..config import Settings, load_settings
from ..domain import (
    Detection,
)
from ..household_human_presence_audit import HouseholdHumanPoseFusionAuditReport
from ..household_pose_fusion_diagnostic import HouseholdPoseFusionProfile
from ..household_pose_fusion_holdout import (
    HouseholdPoseFusionTemporalHoldoutAttestation,
)
from ..pose_presence_shadow import (
    PosePresenceShadowCameraRule,
    PosePresenceShadowClassifier,
    PosePresenceShadowPolicy,
)
from ..presence_history import PresenceFactStore
from ..room_presence import StreamPresenceBatch, evaluate_room_presence
from ..room_registry import RoomRegistry
from ..runtime_trial import (
    finish_runtime_trial,
    inspect_runtime_trial,
    start_runtime_trial,
)
from ..yolo_pose_model import YoloPoseModelManifest

_SHADOW_DETECTOR_RUNNER_MODULE = "baby_monitor.local_runners.yolo_detector"


def _candidate_detector(
    settings: Settings,
    backend: str,
    *,
    allow_fixture_detector: bool,
    image_size: int | None = None,
    rotation_degrees: int | None = None,
) -> tuple[Detector, str]:
    if backend == "mock":
        if not allow_fixture_detector:
            raise ValueError("the mock candidate detector requires --allow-fixture-detector")
        detector: Detector = StaticDetector(
            [Detection(0.5, "fixture_candidate", 0.5, (0.0, 0.0, 1.0, 1.0))]
        )
        identity: dict[str, object] = {"version": 1, "backend": "mock_fixture"}
    elif backend == "scene":
        detector = FFmpegSceneDetector(
            ffmpeg=settings.media.ffmpeg,
            threshold=settings.detector.scene_threshold,
            max_detections=settings.detector.scene_max_detections,
            timeout_seconds=settings.candidate_worker.detector_timeout_seconds,
        )
        identity = {
            "version": 1,
            "backend": "ffmpeg_scene",
            "threshold": settings.detector.scene_threshold,
            "max_detections": settings.detector.scene_max_detections,
        }
    elif backend in {"yolo", "yolo_command"}:
        detector = _yolo_detector(
            settings,
            subprocess_backend=backend == "yolo_command",
            timeout_seconds=settings.candidate_worker.detector_timeout_seconds,
            image_size=image_size,
            rotation_degrees=rotation_degrees,
        )
        identity = {
            "version": 1,
            "backend": "ultralytics_yolo",
            "execution_boundary": (
                "detector_subprocess_v1" if backend == "yolo_command" else "in_process"
            ),
            "model_id": detector.manifest.model_id,
            "model_revision": detector.manifest.revision,
            "model_sha256": detector.model_sha256,
            "labels": list(settings.detector.labels),
            "confidence": settings.detector.confidence,
            "sample_fps": settings.detector.sample_fps,
            "device": settings.detector.device,
            "image_size": detector.image_size,
            "rotation_degrees": detector.rotation_degrees,
        }
    else:
        raise ValueError(f"unsupported candidate detector backend: {backend}")
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return detector, f"{backend}_v1_{digest[:12]}"


def _candidate_scanner_identity(
    settings: Settings,
    *,
    detector_identity: str,
    presence_only: bool,
) -> str:
    runtime = settings.candidate_worker
    scanner_policy: dict[str, object] = {
        "version": 1,
        "mode": "presence_only" if presence_only else "candidate_events",
        "detector_identity": detector_identity,
        "pre_event_seconds": settings.media.pre_event_seconds,
        "post_event_seconds": settings.media.post_event_seconds,
        "candidate_gap_seconds": settings.detector.candidate_gap_seconds,
        "max_event_span_seconds": runtime.max_event_span_seconds,
        "keyframe_count": settings.media.keyframe_count,
    }
    if not presence_only and runtime.candidate_start_confidence > settings.detector.confidence:
        scanner_policy["candidate_start_confidence"] = runtime.candidate_start_confidence
    scanner_digest = hashlib.sha256(
        json.dumps(scanner_policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"candidate_v1_{scanner_digest[:16]}"


def _pose_presence_shadow_classifier(
    args: argparse.Namespace,
    *,
    settings: Settings,
    backend: str,
    image_size: int,
    rotation_degrees: int,
) -> PosePresenceShadowClassifier | None:
    values = (
        args.pose_presence_shadow_profile,
        args.pose_presence_shadow_attestation,
        args.pose_presence_shadow_audit,
        args.pose_presence_shadow_output,
        args.pose_shadow_detector_manifest,
        args.pose_shadow_detector_weights,
        args.pose_shadow_detector_runner_executable,
        args.pose_shadow_manifest,
        args.pose_shadow_weights,
        args.pose_shadow_runner_executable,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("pose presence shadow arguments must be supplied together")
    if backend != "yolo_command" or args.presence_only or args.presence is None:
        raise ValueError(
            "pose presence shadow requires a normal yolo_command worker with --presence"
        )

    profile = HouseholdPoseFusionProfile.read(args.pose_presence_shadow_profile)
    attestation = HouseholdPoseFusionTemporalHoldoutAttestation.read(
        args.pose_presence_shadow_attestation
    )
    audit = HouseholdHumanPoseFusionAuditReport.read(args.pose_presence_shadow_audit)
    holdout = attestation.holdout_report
    if (
        holdout.profile != profile
        or audit.profile_sha256 != profile.fingerprint
        or audit.temporal_holdout_attestation_sha256 != attestation.fingerprint
        or audit.holdout_report_sha256 != holdout.fingerprint
        or audit.camera_ids != holdout.camera_ids
        or audit.bin_seconds != holdout.bin_seconds
    ):
        raise ValueError("pose presence shadow evidence bindings do not match")
    if image_size != holdout.pose_image_size or rotation_degrees != holdout.pose_rotation_degrees:
        raise ValueError("pose presence shadow image transform does not match frozen evidence")

    shadow_detector = _yolo_detector(
        settings,
        subprocess_backend=True,
        timeout_seconds=settings.candidate_worker.detector_timeout_seconds,
        image_size=image_size,
        rotation_degrees=rotation_degrees,
        confidence=holdout.detector_source_confidence_threshold,
        sample_fps=holdout.detector_requested_sample_fps,
        model_path_override=args.pose_shadow_detector_weights,
        manifest_path_override=args.pose_shadow_detector_manifest,
        runner_command_override=(
            args.pose_shadow_detector_runner_executable,
            "-m",
            _SHADOW_DETECTOR_RUNNER_MODULE,
            "{request}",
        ),
        labels_override=("person",),
        device_override=args.pose_shadow_detector_device,
    )
    pose_manifest = YoloPoseModelManifest.read(args.pose_shadow_manifest)
    pose_runner = Path(args.pose_shadow_runner_executable).expanduser()
    if not pose_runner.is_absolute() or not pose_runner.is_file():
        raise ValueError("pose presence shadow runner must be an absolute local file")
    estimator = CommandYoloPoseEstimator(
        model_path=args.pose_shadow_weights,
        manifest_path=args.pose_shadow_manifest,
        manifest=pose_manifest,
        confidence=holdout.pose_source_confidence_threshold,
        sample_fps=holdout.pose_requested_sample_fps,
        max_frames=holdout.pose_requested_max_frames,
        device=args.pose_shadow_device,
        image_size=image_size,
        rotation_degrees=rotation_degrees,
        command=(
            str(pose_runner),
            "-m",
            "baby_monitor.local_runners.yolo_pose",
            "{request}",
        ),
        timeout_seconds=args.pose_shadow_timeout_seconds,
        execution_lock_path=settings.detector.yolo_execution_lock_path,
        execution_lock_timeout_seconds=(settings.detector.yolo_execution_lock_timeout_seconds),
    )
    policy = PosePresenceShadowPolicy(
        profile_sha256=profile.fingerprint,
        temporal_holdout_attestation_sha256=attestation.fingerprint,
        human_audit_report_sha256=audit.fingerprint,
        detector_adapter=holdout.detector_adapter,
        detector_model_id=holdout.detector_model_id,
        detector_model_revision=holdout.detector_model_revision,
        detector_model_sha256=holdout.detector_model_sha256,
        detector_source_confidence_threshold=(holdout.detector_source_confidence_threshold),
        detector_requested_sample_fps=holdout.detector_requested_sample_fps,
        pose_adapter=holdout.pose_adapter,
        pose_model_id=holdout.pose_model_id,
        pose_model_revision=holdout.pose_model_revision,
        pose_model_sha256=holdout.pose_model_sha256,
        pose_source_confidence_threshold=holdout.pose_source_confidence_threshold,
        pose_requested_sample_fps=holdout.pose_requested_sample_fps,
        pose_requested_max_frames=holdout.pose_requested_max_frames,
        image_size=holdout.pose_image_size,
        rotation_degrees=holdout.pose_rotation_degrees,
        bin_seconds=holdout.bin_seconds,
        cameras=tuple(
            PosePresenceShadowCameraRule(
                **{
                    field: getattr(camera, field)
                    for field in PosePresenceShadowCameraRule.__dataclass_fields__
                }
            )
            for camera in profile.cameras
        ),
    )
    return PosePresenceShadowClassifier(policy, shadow_detector, estimator)


def command_candidate_health_status(args: argparse.Namespace) -> int:
    checked_at = datetime.now(UTC)
    assessments = tuple(
        assess_candidate_health(
            CandidateWorkerHealth.read_private(path),
            checked_at=checked_at,
            stale_after_seconds=args.stale_after_seconds,
            future_tolerance_seconds=args.future_tolerance_seconds,
        )
        for path in args.health
    )
    camera_ids = [item.health.camera_id for item in assessments]
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("candidate health camera IDs must be unique")
    operational = all(item.operational for item in assessments)
    _json_print(
        {
            "version": 1,
            "checked_at": checked_at.isoformat(),
            "status": "operational" if operational else "attention",
            "operational": operational,
            "attention_required": not operational,
            "cameras": [item.to_dict() for item in assessments],
            "media_read": False,
            "paths_disclosed": False,
            "identity_inference": False,
            "risk_assessment": False,
            "notification_authority": False,
        }
    )
    return 0 if operational else 1


def command_candidate_soak_start(args: argparse.Namespace) -> int:
    result = start_candidate_soak(
        ledger_path=args.ledger,
        health_paths=tuple(args.health),
        started_at=datetime.now(UTC),
        expected_duration_seconds=args.expected_duration_seconds,
    )
    _json_print(result)
    return 0


def command_candidate_soak_finish(args: argparse.Namespace) -> int:
    result = finish_candidate_soak(
        ledger_path=args.ledger,
        health_paths=tuple(args.health),
        ended_at=datetime.now(UTC),
    )
    _json_print(result)
    return 0


def command_candidate_soak_status(args: argparse.Namespace) -> int:
    result = inspect_candidate_soak(
        ledger_path=args.ledger,
        checked_at=datetime.now(UTC),
    )
    _json_print(result)
    return 1 if result["attention_required"] else 0


def command_runtime_trial_start(args: argparse.Namespace) -> int:
    result = start_runtime_trial(
        ledger_path=args.ledger,
        health_paths=tuple(args.health),
        database_path=args.database,
        started_at=datetime.now(UTC),
        expected_duration_seconds=args.expected_duration_seconds,
        health_wait_seconds=args.health_wait_seconds,
        shadow_expected=args.shadow_expected,
    )
    _json_print(result)
    return 0


def command_runtime_trial_finish(args: argparse.Namespace) -> int:
    result = finish_runtime_trial(
        ledger_path=args.ledger,
        health_paths=tuple(args.health),
        database_path=args.database,
        ended_at=datetime.now(UTC),
    )
    _json_print(result)
    return 0


def command_runtime_trial_status(args: argparse.Namespace) -> int:
    result = inspect_runtime_trial(
        ledger_path=args.ledger,
        checked_at=datetime.now(UTC),
    )
    _json_print(result)
    return 1 if result["attention_required"] else 0


def command_room_registry_status(args: argparse.Namespace) -> int:
    registry = RoomRegistry.read_private(args.registry)
    _json_print(registry.summary())
    return 0


def command_room_presence_evaluate(args: argparse.Namespace) -> int:
    registry = RoomRegistry.read_private(args.registry)
    batches = tuple(StreamPresenceBatch.read_private(path) for path in args.samples)
    gap_samples = tuple(
        sample
        for sample in (
            candidate_health_gap_sample(CandidateWorkerHealth.read_private(path))
            for path in args.health
        )
        if sample is not None
    )
    batch = merge_stream_presence_batches(batches, extra_samples=gap_samples)
    checked_at = (
        datetime.now(UTC) if args.checked_at is None else datetime.fromisoformat(args.checked_at)
    )
    report = evaluate_room_presence(
        registry=registry,
        batch=batch,
        checked_at=checked_at,
    )
    _json_print(report.to_dict())
    return 0


def command_candidate_presence_health_init(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    backend = args.detector or settings.detector.backend
    _detector, detector_identity = _candidate_detector(
        settings,
        backend,
        allow_fixture_detector=False,
        image_size=args.image_size,
        rotation_degrees=args.rotation_degrees,
    )
    scanner_identity = _candidate_scanner_identity(
        settings,
        detector_identity=detector_identity,
        presence_only=True,
    )
    health_path = Path(args.health).expanduser()
    created = not (health_path.exists() or health_path.is_symlink())
    if created:
        now = datetime.now(UTC)
        health = CandidateWorkerHealth(
            version=2,
            camera_id=args.camera_id,
            scanner_identity=scanner_identity,
            detector_identity=detector_identity,
            checked_at=now,
            state=CandidateHealthState.STOPPED,
            worker_running=False,
            lifetime=CandidateHealthLifetime(updated_at=now),
        )
        health.write_atomic_private(health_path)
    else:
        health = CandidateWorkerHealth.read_private(health_path)
        if (
            health.version not in {2, 3, 4}
            or not isinstance(health.lifetime, CandidateHealthLifetime)
            or health.camera_id != args.camera_id
            or health.scanner_identity != scanner_identity
            or health.detector_identity != detector_identity
            or health.state is not CandidateHealthState.STOPPED
            or health.worker_running
        ):
            raise ValueError("candidate presence health is not a reusable stopped baseline")
    _json_print(
        {
            "version": 1,
            "camera_id": health.camera_id,
            "created": created,
            "state": health.state.value,
            "scanner_identity": health.scanner_identity,
            "detector_identity": health.detector_identity,
            "media_read": False,
            "analysis_queue_touched": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_candidate_worker(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    backend = args.detector or settings.detector.backend
    detector, detector_identity = _candidate_detector(
        settings,
        backend,
        allow_fixture_detector=args.allow_fixture_detector,
        image_size=args.image_size,
        rotation_degrees=args.rotation_degrees,
    )
    manifest_path = Path(args.manifest).expanduser()
    state_path = (
        Path(args.state).expanduser()
        if args.state
        else manifest_path.with_name("candidate-scan-state.json")
    )
    health_path = (
        Path(args.health).expanduser()
        if args.health
        else manifest_path.with_name("candidate-health.json")
    )
    presence_path = Path(args.presence).expanduser() if args.presence else None
    if args.presence_only and presence_path is None:
        raise ValueError("--presence-only requires --presence")
    database_path = Path(args.database) if args.database else settings.storage.database
    if not args.presence_only:
        _database_from_path(database_path)
    shadow_classifier = _pose_presence_shadow_classifier(
        args,
        settings=settings,
        backend=backend,
        image_size=getattr(detector, "image_size", args.image_size or settings.detector.image_size),
        rotation_degrees=getattr(
            detector,
            "rotation_degrees",
            args.rotation_degrees or settings.detector.rotation_degrees,
        ),
    )
    shadow_presence_path = (
        Path(args.pose_presence_shadow_output).expanduser()
        if args.pose_presence_shadow_output
        else None
    )
    if shadow_classifier is not None and not (
        30.0 <= args.pose_shadow_min_interval_seconds <= 3_600.0
    ):
        raise ValueError(
            "pose presence shadow minimum interval must be between 30 and 3600 seconds"
        )
    runtime = settings.candidate_worker
    candidate_start_confidence = max(
        runtime.candidate_start_confidence,
        settings.detector.confidence,
    )
    max_segments = (
        args.max_segments if args.max_segments is not None else runtime.max_segments_per_run
    )
    idle_exit_seconds = (
        args.idle_exit_seconds if args.idle_exit_seconds is not None else runtime.idle_exit_seconds
    )
    max_runtime_seconds = (
        args.max_runtime_seconds
        if args.max_runtime_seconds is not None
        else runtime.max_runtime_seconds
    )
    poll_seconds = args.poll_seconds if args.poll_seconds is not None else runtime.poll_seconds
    scanner_identity = _candidate_scanner_identity(
        settings,
        detector_identity=detector_identity,
        presence_only=args.presence_only,
    )
    worker = CandidateWorker(
        CandidateWorkerSettings(
            manifest_path=manifest_path,
            state_path=state_path,
            database_path=database_path,
            storage=_storage_override(settings, args.output_root),
            detector=detector,
            detector_identity=detector_identity,
            scanner_identity=scanner_identity,
            health_path=health_path,
            presence_path=presence_path,
            presence_store=(
                PresenceFactStore(database_path)
                if presence_path is not None and not args.presence_only
                else None
            ),
            shadow_presence_path=shadow_presence_path,
            shadow_presence_store=(
                PresenceFactStore(database_path) if shadow_classifier is not None else None
            ),
            shadow_presence_classifier=shadow_classifier,
            shadow_min_interval_seconds=(
                args.pose_shadow_min_interval_seconds if shadow_classifier is not None else 0.0
            ),
            pre_event_seconds=settings.media.pre_event_seconds,
            post_event_seconds=settings.media.post_event_seconds,
            candidate_gap_seconds=settings.detector.candidate_gap_seconds,
            candidate_start_confidence=candidate_start_confidence,
            max_event_span_seconds=runtime.max_event_span_seconds,
            keyframe_count=settings.media.keyframe_count,
            max_pending_detections=runtime.max_pending_detections,
            max_segments_per_run=max_segments,
            idle_exit_seconds=idle_exit_seconds,
            max_runtime_seconds=max_runtime_seconds,
            poll_seconds=poll_seconds,
            max_active_jobs=settings.analysis_queue.max_pending,
            emit_candidate_events=not args.presence_only,
            ffmpeg=settings.media.ffmpeg,
            ffprobe=settings.media.ffprobe,
        )
    )
    termination_requested = False
    previous_sigterm: object | None = None

    def handle_sigterm(_signum: int, _frame: object) -> None:
        nonlocal termination_requested
        termination_requested = True
        raise KeyboardInterrupt

    if os.name == "posix":
        previous_sigterm = signal_module.getsignal(signal_module.SIGTERM)
        signal_module.signal(signal_module.SIGTERM, handle_sigterm)
    try:
        summary = worker.run()
    finally:
        if os.name == "posix" and previous_sigterm is not None:
            signal_module.signal(signal_module.SIGTERM, previous_sigterm)
    if termination_requested:
        return 0
    _json_print(
        {
            "detector": backend,
            "detector_identity": detector_identity,
            "scanner_identity": scanner_identity,
            "manifest": str(manifest_path),
            "database": str(database_path),
            "candidate_event_emission": not args.presence_only,
            "candidate_start_confidence": (
                candidate_start_confidence if not args.presence_only else None
            ),
            "analysis_queue_touched": not args.presence_only,
            "worker": summary.to_dict(),
            "queue": (
                AnalysisQueue(database_path).counts().to_dict() if not args.presence_only else None
            ),
        }
    )
    if summary.stop_reason is CandidateWorkerStopReason.INTERRUPTED:
        return 130
    return 3 if summary.queue_full_events else 0
