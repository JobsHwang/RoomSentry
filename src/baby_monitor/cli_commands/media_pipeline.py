"""File and RTSP media processing, buffering, evidence, database, and reporting commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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

from ..adapters.camera import LocalFileCamera
from ..adapters.detector import (
    FFmpegSceneDetector,
    StaticDetector,
)
from ..adapters.media import FFmpegMediaProcessor, FixtureMediaProcessor
from ..analysis_queue import AnalysisQueue
from ..archive_queue import (
    ArchiveQueue,
)
from ..config import Settings, StorageSettings, load_settings
from ..database import EventDatabase
from ..domain import (
    CandidateEvent,
    Detection,
    SignalEvent,
)
from ..event_policy_runtime import EventPolicyDecisionStore
from ..local_perception import run_local_perception
from ..media_health import MediaSourceHealth, assess_media_health
from ..notification import (
    NotificationOutbox,
)
from ..object_tracker import (
    DetectionFrameBatch,
    IoUTrackerConfig,
)
from ..pipeline import PipelineRunner, process_prepared_event
from ..reporting import DailyReportGenerator
from ..ring_buffer import (
    RTSPSegmentBuffer,
    SegmentBufferSettings,
    SegmentManifest,
    extract_buffered_evidence,
)
from ..rtsp_soak import RTSPSoakSettings, run_rtsp_soak
from ..storage import StorageResolver
from ..temporal_zone import (
    TemporalZoneConfig,
)
from ..visual_predicate_shadow_queue import (
    VisualPredicateShadowAttemptQueue,
)


def _pipeline(
    *,
    settings: Settings,
    camera: LocalFileCamera,
    detector: object,
    media: object,
    database: EventDatabase,
    storage: StorageResolver,
) -> PipelineRunner:
    remote = _remote_adapter(settings, database_path=database.path)
    return PipelineRunner(
        camera=camera,
        detector=detector,  # type: ignore[arg-type]
        media=media,  # type: ignore[arg-type]
        local_vlm=_local_adapter(settings),
        secondary_vlm=_secondary_adapter(settings),
        remote_vlm=remote,
        remote_uploads_allowed=settings.remote_vlm.uploads_allowed,
        database=database,
        storage=storage,
        policy=_review_policy(settings),
        candidate_gap_seconds=settings.detector.candidate_gap_seconds,
        pre_event_seconds=settings.media.pre_event_seconds,
        post_event_seconds=settings.media.post_event_seconds,
    )


def command_init_db(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    _database_from_path(database_path)
    AnalysisQueue(database_path).initialize()
    ArchiveQueue(database_path).initialize()
    NotificationOutbox(database_path).initialize()
    EventPolicyDecisionStore(database_path).initialize()
    VisualPredicateShadowAttemptQueue(database_path).initialize()
    _json_print(
        {
            "status": "initialized",
            "database": str(database_path),
            "schemas": [
                "events",
                "analysis_jobs",
                "archive_jobs",
                "remote_review_attempts",
                "recognition_attempts",
                "presence_facts",
                "notification_jobs",
                "event_policy_decisions",
                "visual_predicate_shadow_attempts",
            ],
        }
    )
    return 0


def command_smoke_demo(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser()
    storage_settings = StorageSettings(
        local_root=output_root,
        database=output_root / "events.sqlite3",
        retention_scan_enabled=False,
        retention_delete_enabled=False,
    )
    settings = Settings(storage=storage_settings)
    output_root.mkdir(parents=True, exist_ok=True)
    placeholder = output_root / "smoke-input.placeholder"
    placeholder.write_text("SMOKE DEMO INPUT - NOT VIDEO\n", encoding="utf-8")
    camera = LocalFileCamera(settings.camera.camera_id, placeholder)
    detector = StaticDetector(
        [
            Detection(2.0, "person", 0.81, (0.1, 0.1, 0.5, 0.9)),
            Detection(3.0, "person", 0.84, (0.12, 0.1, 0.52, 0.9)),
        ]
    )
    database = _database_from_path(storage_settings.database)
    storage = StorageResolver(storage_settings)
    started_at = datetime.now(UTC).replace(microsecond=0)
    summary = _pipeline(
        settings=settings,
        camera=camera,
        detector=detector,
        media=FixtureMediaProcessor(frame_count=3),
        database=database,
        storage=storage,
    ).run(source_started_at=started_at)
    report_date = started_at.astimezone(ZoneInfo(settings.timezone)).date()
    report = DailyReportGenerator(
        database=database,
        storage=storage,
        timezone_name=settings.timezone,
    ).generate(report_date)
    _json_print(
        {
            "mode": "smoke_fixture_not_real_video_or_ai",
            "pipeline": {
                "detections": summary.detections,
                "candidates": summary.candidates,
                "stored_event_ids": list(summary.stored_event_ids),
                "secondary_reviews": summary.secondary_reviews,
                "remote_reviews": summary.remote_reviews,
                "pending_remote_reviews": summary.pending_remote_reviews,
                "reused_events": summary.reused_events,
            },
            "database": str(storage_settings.database),
            "report_markdown": str(report.markdown_path),
            "report_html": str(report.html_path),
        }
    )
    return 0


def command_process_file(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    camera_id = args.camera_id or settings.camera.camera_id
    camera = LocalFileCamera(camera_id, args.source)
    tracking_enabled = bool(args.tracker_config or args.zone_config)
    if bool(args.tracker_config) != bool(args.zone_config):
        raise ValueError("--tracker-config and --zone-config must be provided together")
    if args.detection_frames_output and not tracking_enabled:
        raise ValueError("--detection-frames-output requires --tracker-config and --zone-config")
    tracker_config = IoUTrackerConfig.read(args.tracker_config) if tracking_enabled else None
    zone_config = TemporalZoneConfig.read(args.zone_config) if tracking_enabled else None
    backend = args.detector or settings.detector.backend
    if backend == "mock":
        detector = StaticDetector([Detection(5.0, "person", 0.75, (0.0, 0.0, 1.0, 1.0))])
        detector_provenance: dict[str, object] = {
            "version": 1,
            "backend": "mock_fixture",
            "model_loaded": False,
        }
    elif backend == "scene":
        detector = FFmpegSceneDetector(
            ffmpeg=settings.media.ffmpeg,
            threshold=settings.detector.scene_threshold,
            max_detections=settings.detector.scene_max_detections,
        )
        detector_provenance = {
            "version": 1,
            "backend": "ffmpeg_scene",
            "threshold": settings.detector.scene_threshold,
            "max_detections": settings.detector.scene_max_detections,
            "model_loaded": False,
        }
    elif backend in {"yolo", "yolo_command"}:
        detector = _yolo_detector(settings, subprocess_backend=backend == "yolo_command")
        detector_provenance = {
            "version": 1,
            "execution_boundary": (
                "detector_subprocess_v1" if backend == "yolo_command" else "in_process"
            ),
            "backend": detector.manifest.backend,
            "model_id": detector.manifest.model_id,
            "model_revision": detector.manifest.revision,
            "model_sha256": detector.model_sha256,
            "license_spdx": detector.manifest.license_spdx,
            "selected_labels": list(settings.detector.labels),
            "confidence": settings.detector.confidence,
            "sample_fps": settings.detector.sample_fps,
            "device": settings.detector.device,
            "subject_identity_inference": False,
            "paths_disclosed": False,
        }
    else:
        raise ValueError(f"unsupported detector backend: {backend}")
    detector_identity = hashlib.sha256(
        json.dumps(
            detector_provenance,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    detector_adapter = f"{backend}_{detector_identity[:24]}"

    database_path = Path(args.database) if args.database else settings.storage.database
    database = _database_from_path(database_path)
    storage = _storage_override(settings, args.output_root)
    media_processor = FFmpegMediaProcessor(
        ffmpeg=settings.media.ffmpeg,
        ffprobe=settings.media.ffprobe,
        keyframe_count=settings.media.keyframe_count,
    )
    source_metadata = media_processor.probe(camera.source())
    started_at = datetime.now(UTC)
    precomputed_detections: tuple[Detection, ...] | None = None
    detection_frames: DetectionFrameBatch | None = None
    perception_signals: tuple[SignalEvent, ...] = ()
    local_perception: dict[str, object] = {
        "enabled": False,
        "paths_disclosed": False,
    }
    if tracking_enabled:
        assert tracker_config is not None
        assert zone_config is not None
        precomputed_detections = tuple(detector.scan(camera.source()))
        if precomputed_detections:
            perception = run_local_perception(
                precomputed_detections,
                camera_id=camera_id,
                detector_adapter=detector_adapter,
                source_started_at=started_at,
                tracker_config=tracker_config,
                zone_config=zone_config,
            )
            perception_signals = perception.signals
            detection_frames = perception.detection_frames
            if args.detection_frames_output:
                detection_frames.write_new(args.detection_frames_output)
            local_perception = {
                "enabled": True,
                "version": 1,
                "input_contract": "DetectionFrameBatch.v1",
                "tracker": "iou_tracker_v1",
                "temporal_assembler": "temporal_zone_v1",
                "detector_adapter": detector_adapter,
                "detection_frames": len(perception.detection_frames.frames),
                "frame_detections": sum(
                    len(frame.detections) for frame in perception.detection_frames.frames
                ),
                "tracked_observations": len(perception.observations),
                "active_tracks": perception.active_track_count,
                "temporal_signals": len(perception.signals),
                "signal_types": sorted({signal.signal_type.value for signal in perception.signals}),
                "state_mode": "bounded_file_in_memory",
                "detection_frames_saved": bool(args.detection_frames_output),
                "subject_identity_inference": False,
                "paths_disclosed": False,
            }
        else:
            local_perception = {
                "enabled": True,
                "version": 1,
                "input_contract": "DetectionFrameBatch.v1",
                "tracker": "iou_tracker_v1",
                "temporal_assembler": "temporal_zone_v1",
                "detector_adapter": detector_adapter,
                "detection_frames": 0,
                "frame_detections": 0,
                "tracked_observations": 0,
                "active_tracks": 0,
                "temporal_signals": 0,
                "signal_types": [],
                "state_mode": "bounded_file_in_memory",
                "detection_frames_saved": False,
                "subject_identity_inference": False,
                "paths_disclosed": False,
            }
            if args.detection_frames_output:
                raise ValueError(
                    "no detections were available for the requested frame-batch output"
                )
    runner = _pipeline(
        settings=settings,
        camera=camera,
        detector=detector,
        media=media_processor,
        database=database,
        storage=storage,
    )
    summary = runner.run(
        source_started_at=started_at,
        precomputed_detections=precomputed_detections,
        signals=perception_signals,
    )
    stored_events = []
    for event_id in summary.stored_event_ids:
        event = database.get_event(event_id)
        if event is None:
            raise RuntimeError("stored event disappeared from SQLite")
        stored_events.append(
            {
                "event_id": event.event_id,
                "candidate": event.candidate,
                "result": event.result.to_dict(),
                "decision": event.decision.to_dict(),
                "local_provenance": event.local_provenance.to_dict(),
                "local_metrics": (event.local_metrics.to_dict() if event.local_metrics else None),
                "secondary_provenance": (
                    event.secondary_provenance.to_dict() if event.secondary_provenance else None
                ),
                "secondary_metrics": (
                    event.secondary_metrics.to_dict() if event.secondary_metrics else None
                ),
                "remote_provenance": (
                    event.remote_provenance.to_dict() if event.remote_provenance else None
                ),
                "remote_metrics": (
                    event.remote_metrics.to_dict() if event.remote_metrics else None
                ),
                "clip_path": event.clip_path,
            }
        )
    report_date = started_at.astimezone(ZoneInfo(settings.timezone)).date()
    report = DailyReportGenerator(
        database=database,
        storage=storage,
        timezone_name=settings.timezone,
    ).generate(report_date)
    _json_print(
        {
            "detector": backend,
            "detector_provenance": detector_provenance,
            "local_perception": local_perception,
            "local_vlm": _local_adapter(settings).name,
            "source_media": source_metadata.to_dict(),
            "detections": summary.detections,
            "candidates": summary.candidates,
            "stored_event_ids": list(summary.stored_event_ids),
            "events": stored_events,
            "secondary_reviews": summary.secondary_reviews,
            "remote_reviews": summary.remote_reviews,
            "pending_remote_reviews": summary.pending_remote_reviews,
            "reused_events": summary.reused_events,
            "report": {
                "date": report_date.isoformat(),
                "event_count": report.event_count,
                "markdown": str(report.markdown_path),
                "html": str(report.html_path),
            },
        }
    )
    return 0


def command_list_events(args: argparse.Namespace) -> int:
    database = _database_from_path(args.database)
    events = database.list_recent(limit=args.limit)
    _json_print(
        [
            {
                "event_id": event.event_id,
                "camera_id": event.camera_id,
                "occurred_at": event.occurred_at.isoformat(),
                "event_type": event.result.event_type.value,
                "risk_level": event.result.risk_level.value,
                "confidence": event.result.confidence,
                "needs_remote_review": event.decision.needs_remote_review,
                "review_tier": event.decision.review_tier.value,
                "review_reasons": list(event.decision.review_reasons),
                "secondary_used": event.secondary_result is not None,
                "remote_used": event.remote_used,
                "local_provenance": event.local_provenance.to_dict(),
                "local_metrics": (event.local_metrics.to_dict() if event.local_metrics else None),
                "secondary_provenance": (
                    event.secondary_provenance.to_dict() if event.secondary_provenance else None
                ),
                "secondary_metrics": (
                    event.secondary_metrics.to_dict() if event.secondary_metrics else None
                ),
                "remote_provenance": (
                    event.remote_provenance.to_dict() if event.remote_provenance else None
                ),
                "remote_metrics": (
                    event.remote_metrics.to_dict() if event.remote_metrics else None
                ),
                "clip_path": event.clip_path,
            }
            for event in events
        ]
    )
    return 0


def command_report(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database = _database_from_path(args.database or settings.storage.database)
    storage = _storage_override(settings, args.output_root)
    report_date = (
        date.fromisoformat(args.date)
        if args.date
        else datetime.now(ZoneInfo(settings.timezone)).date() + timedelta(days=args.day_offset or 0)
    )
    artifacts = DailyReportGenerator(
        database=database,
        storage=storage,
        timezone_name=settings.timezone,
    ).generate(report_date)
    _json_print(
        {
            "date": report_date.isoformat(),
            "events": artifacts.event_count,
            "markdown": str(artifacts.markdown_path),
            "html": str(artifacts.html_path),
        }
    )
    return 0


def command_rtsp_soak(args: argparse.Namespace) -> int:
    settings = RTSPSoakSettings(
        uri=args.uri,
        duration_seconds=args.seconds,
        startup_timeout_seconds=args.startup_timeout,
        ffmpeg=args.ffmpeg,
    )
    result = run_rtsp_soak(settings)
    payload = result.to_dict()
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        payload["output"] = str(output)
    _json_print(payload)
    return 0 if result.ok else 2


def command_rtsp_buffer(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    buffer_settings = settings.buffer
    result = RTSPSegmentBuffer(
        SegmentBufferSettings(
            uri=args.uri,
            camera_id=args.camera_id or settings.camera.camera_id,
            buffer_root=(
                Path(args.buffer_root).expanduser() if args.buffer_root else buffer_settings.root
            ),
            segment_seconds=buffer_settings.segment_seconds,
            max_segments=buffer_settings.max_segments,
            run_seconds=args.seconds,
            reconnect_initial_seconds=buffer_settings.reconnect_initial_seconds,
            reconnect_max_seconds=buffer_settings.reconnect_max_seconds,
            poll_interval_seconds=buffer_settings.poll_interval_seconds,
            ffmpeg=settings.media.ffmpeg,
            ffprobe=settings.media.ffprobe,
            allow_non_loopback=args.allow_non_loopback,
            eviction_enabled=buffer_settings.eviction_enabled,
            eviction_confirmed=args.confirm_ephemeral_eviction,
        )
    ).run()
    _json_print(result.to_dict())
    return 0 if result.ok else 2


def command_media_health_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    health = MediaSourceHealth.read(args.health)
    assessment = assess_media_health(
        health,
        stale_after_seconds=(
            args.stale_after_seconds
            if args.stale_after_seconds is not None
            else settings.media_health.stale_after_seconds
        ),
        future_tolerance_seconds=(
            args.future_tolerance_seconds
            if args.future_tolerance_seconds is not None
            else settings.media_health.future_tolerance_seconds
        ),
    )
    _json_print(
        {
            "health_path": str(Path(args.health).expanduser()),
            "health": health.to_dict(),
            "assessment": assessment.to_dict(),
        }
    )
    return 0 if assessment.operational else 1


def command_buffer_extract(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    event_time = _aware_datetime(args.trigger_at, field="trigger-at")
    result = extract_buffered_evidence(
        manifest_path=args.manifest,
        event_time=event_time,
        event_id=args.event_id or uuid.uuid4().hex,
        pre_event_seconds=(
            args.pre_seconds if args.pre_seconds is not None else settings.media.pre_event_seconds
        ),
        post_event_seconds=(
            args.post_seconds
            if args.post_seconds is not None
            else settings.media.post_event_seconds
        ),
        keyframe_count=args.keyframes or settings.media.keyframe_count,
        storage=_storage_override(settings, args.output_root),
        ffmpeg=settings.media.ffmpeg,
        ffprobe=settings.media.ffprobe,
    )
    _json_print(result.to_dict())
    return 0


def command_process_buffer_event(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    event_time = _aware_datetime(args.trigger_at, field="trigger-at")
    event_id = args.event_id or uuid.uuid4().hex
    pre_seconds = (
        args.pre_seconds if args.pre_seconds is not None else settings.media.pre_event_seconds
    )
    post_seconds = (
        args.post_seconds if args.post_seconds is not None else settings.media.post_event_seconds
    )
    storage = _storage_override(settings, args.output_root)
    evidence = extract_buffered_evidence(
        manifest_path=args.manifest,
        event_time=event_time,
        event_id=event_id,
        pre_event_seconds=pre_seconds,
        post_event_seconds=post_seconds,
        keyframe_count=args.keyframes or settings.media.keyframe_count,
        storage=storage,
        ffmpeg=settings.media.ffmpeg,
        ffprobe=settings.media.ffprobe,
    )
    manifest = SegmentManifest.read(args.manifest)
    candidate = CandidateEvent(
        event_id=event_id,
        camera_id=manifest.camera_id,
        start_seconds=0.0,
        end_seconds=pre_seconds + post_seconds,
        detections=(),
    )
    database_path = Path(args.database) if args.database else settings.storage.database
    database = _database_from_path(database_path)
    remote = _remote_adapter(settings, database_path=database.path)
    processed = process_prepared_event(
        candidate=candidate,
        occurred_at=event_time,
        source_kind="rtsp",
        source_ref=manifest.source_ref,
        media=evidence.media,
        local_vlm=_local_adapter(settings),
        secondary_vlm=_secondary_adapter(settings),
        remote_vlm=remote,
        remote_uploads_allowed=settings.remote_vlm.uploads_allowed,
        database=database,
        policy=_review_policy(settings),
    )
    stored = database.get_event(event_id)
    if stored is None:
        raise RuntimeError("stored event disappeared from SQLite")
    report_date = event_time.astimezone(ZoneInfo(settings.timezone)).date()
    report = DailyReportGenerator(
        database=database,
        storage=storage,
        timezone_name=settings.timezone,
    ).generate(report_date)
    _json_print(
        {
            "event_id": event_id,
            "local_vlm": _local_adapter(settings).name,
            "secondary_vlm": (
                settings.secondary_vlm.name if processed.secondary_reviewed else None
            ),
            "result": stored.result.to_dict(),
            "decision": stored.decision.to_dict(),
            "evidence": evidence.to_dict(),
            "database": str(database_path),
            "remote_reviewed": processed.remote_reviewed,
            "secondary_reviewed": processed.secondary_reviewed,
            "pending_remote_review": processed.pending_remote_review,
            "reused_existing": processed.reused_existing,
            "report": {
                "date": report_date.isoformat(),
                "event_count": report.event_count,
                "markdown": str(report.markdown_path),
                "html": str(report.html_path),
            },
        }
    )
    return 0


def command_enqueue_buffer_event(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    event_time = _aware_datetime(args.trigger_at, field="trigger-at")
    return _enqueue_buffered_candidate(
        args,
        settings=settings,
        event_time=event_time,
        event_id=args.event_id or uuid.uuid4().hex,
        signals=(),
    )
