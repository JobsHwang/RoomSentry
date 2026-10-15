"""Shared private helpers for the CLI command modules.

These helpers are independent of any concrete command handler so both
``baby_monitor.cli`` and the ``cli_commands`` modules can import them
without cycles.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from baby_monitor.adapters.detector import (
    CommandYoloDetector,
    Detector,
    UltralyticsYoloDetector,
)
from baby_monitor.adapters.vlm import (
    CommandVLMAdapter,
    MockVLMAdapter,
    OpenAICompatibleVLMAdapter,
    VLMAdapter,
)
from baby_monitor.domain import SignalType
from baby_monitor.policy import ReviewPolicy
from baby_monitor.remote_review import RemoteReviewAuditStore
from baby_monitor.yolo_model import YoloModelManifest

from .analysis_queue import AnalysisQueue, QueueFullError
from .config import Settings
from .database import EventDatabase
from .domain import (
    CandidateEvent,
    EventMedia,
    SignalEvent,
)
from .ring_buffer import (
    SegmentManifest,
    extract_buffered_evidence,
)
from .storage import StorageResolver


def _json_print(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _absolute_runner_executable(value: str) -> str:
    if (
        not value
        or "\x00" in value
        or not Path(value).is_absolute()
        or ".." in Path(value).parts
        or str(Path(value)) != value
    ):
        raise argparse.ArgumentTypeError("runner executable must be an absolute path")
    return value


def _database_from_path(path: str | Path) -> EventDatabase:
    database = EventDatabase(path)
    database.initialize()
    return database


def _storage_override(settings: Settings, output_root: str | None) -> StorageResolver:
    if output_root is None:
        return StorageResolver(settings.storage)
    overridden = replace(
        settings.storage,
        local_root=Path(output_root).expanduser(),
        nas_root=None,
        prefer_nas=False,
    )
    return StorageResolver(overridden)


def _aware_datetime(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed


def _enqueue_buffered_candidate(
    args: argparse.Namespace,
    *,
    settings: Settings,
    event_time: datetime,
    event_id: str,
    signals: tuple[SignalEvent, ...],
) -> int:
    pre_seconds = (
        args.pre_seconds if args.pre_seconds is not None else settings.media.pre_event_seconds
    )
    post_seconds = (
        args.post_seconds if args.post_seconds is not None else settings.media.post_event_seconds
    )
    manifest = SegmentManifest.read(args.manifest)
    if any(signal.camera_id != manifest.camera_id for signal in signals):
        raise ValueError("signal camera_id does not match the buffer manifest")
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
    candidate = CandidateEvent(
        event_id=event_id,
        camera_id=manifest.camera_id,
        start_seconds=0.0,
        end_seconds=pre_seconds + post_seconds,
        detections=(),
        signals=signals,
    )
    database_path = Path(args.database) if args.database else settings.storage.database
    _database_from_path(database_path)
    queue = AnalysisQueue(database_path)
    absolute_media = EventMedia(
        clip_path=str(Path(evidence.media.clip_path).resolve()),
        frame_paths=tuple(str(Path(path).resolve()) for path in evidence.media.frame_paths),
        coverage_gaps=evidence.media.coverage_gaps,
        selected_segments=evidence.media.selected_segments,
        version=evidence.media.version,
    )
    storage_root = storage.active_media_root().resolve()
    try:
        enqueued = queue.enqueue(
            candidate=candidate,
            occurred_at=event_time,
            source_kind="rtsp",
            source_ref=manifest.source_ref,
            media=absolute_media,
            storage_root=storage_root,
            max_active=settings.analysis_queue.max_pending,
        )
    except QueueFullError:
        _json_print(
            {
                "queued": False,
                "reason": "queue_full",
                "event_id": event_id,
                "signal_count": len(signals),
                "signal_types": sorted({signal.signal_type.value for signal in signals}),
                "evidence_preserved": True,
                "evidence": evidence.to_dict(),
                "database": str(database_path),
                "queue": queue.counts().to_dict(),
            }
        )
        return 3
    _json_print(
        {
            "queued": True,
            "event_id": event_id,
            "signal_count": len(signals),
            "signal_types": sorted({signal.signal_type.value for signal in signals}),
            "status": enqueued.job.status.value,
            "attempt_count": enqueued.job.attempt_count,
            "reused_existing": enqueued.reused_existing,
            "evidence": evidence.to_dict(),
            "database": str(database_path),
            "queue": queue.counts().to_dict(),
        }
    )
    return 0


def _remote_adapter(
    settings: Settings,
    *,
    database_path: str | Path | None = None,
) -> OpenAICompatibleVLMAdapter | None:
    if not settings.remote_vlm.uploads_allowed:
        return None
    return _configured_remote_adapter(settings, database_path=database_path)


def _local_adapter(settings: Settings) -> VLMAdapter:
    local = settings.local_vlm
    if local.backend == "mock":
        return MockVLMAdapter(name=local.name)
    if local.backend == "command":
        return CommandVLMAdapter(
            name=local.name,
            command=local.command,
            timeout_seconds=local.timeout_seconds,
            model_revision=local.revision or None,
            gpu_device=local.gpu_device,
        )
    raise ValueError(f"unsupported local VLM backend: {local.backend}")


def _secondary_adapter(settings: Settings) -> VLMAdapter | None:
    secondary = settings.secondary_vlm
    if not secondary.enabled:
        return None
    if secondary.backend == "mock":
        return MockVLMAdapter(name=secondary.name)
    if secondary.backend == "command":
        return CommandVLMAdapter(
            name=secondary.name,
            command=secondary.command,
            timeout_seconds=secondary.timeout_seconds,
            model_revision=secondary.revision or None,
            gpu_device=secondary.gpu_device,
        )
    raise ValueError(f"unsupported secondary VLM backend: {secondary.backend}")


def _review_policy(settings: Settings) -> ReviewPolicy:
    policy = settings.review_policy
    return ReviewPolicy(
        confidence_threshold=settings.local_vlm.confidence_threshold,
        signal_frequency_window_seconds=policy.signal_frequency_window_seconds,
        signal_frequency_secondary_threshold=policy.signal_frequency_secondary_threshold,
        priority_signal_types=tuple(SignalType(value) for value in policy.priority_signal_types),
    )


def _yolo_detector(
    settings: Settings,
    *,
    subprocess_backend: bool = False,
    timeout_seconds: float | None = None,
    image_size: int | None = None,
    rotation_degrees: int | None = None,
    confidence: float | None = None,
    sample_fps: float | None = None,
    model_path_override: str | Path | None = None,
    manifest_path_override: str | Path | None = None,
    runner_command_override: Sequence[str] | None = None,
    labels_override: Sequence[str] | None = None,
    device_override: str | None = None,
) -> Detector:
    model_path = (
        settings.detector.yolo_model_path
        if model_path_override is None
        else Path(model_path_override).expanduser()
    )
    manifest_path = (
        settings.detector.yolo_manifest_path
        if manifest_path_override is None
        else Path(manifest_path_override).expanduser()
    )
    if model_path is None:
        raise ValueError("BABY_MONITOR_YOLO_MODEL_PATH must point to a local weight file")
    if manifest_path is None:
        raise ValueError("BABY_MONITOR_YOLO_MANIFEST_PATH must point to a reviewed local manifest")
    manifest = YoloModelManifest.read(manifest_path)
    common = {
        "model_path": model_path,
        "manifest": manifest,
        "labels": (settings.detector.labels if labels_override is None else tuple(labels_override)),
        "confidence": settings.detector.confidence if confidence is None else confidence,
        "sample_fps": settings.detector.sample_fps if sample_fps is None else sample_fps,
        "device": settings.detector.device if device_override is None else device_override,
        "image_size": (settings.detector.image_size if image_size is None else image_size),
        "rotation_degrees": (
            settings.detector.rotation_degrees if rotation_degrees is None else rotation_degrees
        ),
    }
    if subprocess_backend:
        command = (
            settings.detector.yolo_runner_command
            if runner_command_override is None
            else tuple(runner_command_override)
        )
        if not command:
            raise ValueError("detector.yolo_runner_command is required for yolo_command backend")
        return CommandYoloDetector(
            **common,
            manifest_path=manifest_path,
            command=command,
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else settings.detector.yolo_runner_timeout_seconds
            ),
            execution_lock_path=settings.detector.yolo_execution_lock_path,
            execution_lock_timeout_seconds=(settings.detector.yolo_execution_lock_timeout_seconds),
        )
    return UltralyticsYoloDetector(**common)


def _configured_remote_adapter(
    settings: Settings,
    *,
    database_path: str | Path | None = None,
) -> OpenAICompatibleVLMAdapter:
    remote = settings.remote_vlm
    return OpenAICompatibleVLMAdapter(
        base_url=remote.base_url,
        model=remote.model,
        api_key_env=remote.api_key_env,
        enabled=remote.enabled,
        upload_consent=remote.upload_consent,
        max_frames=remote.max_frames,
        timeout_seconds=remote.timeout_seconds,
        max_upload_bytes=remote.max_upload_bytes,
        max_response_bytes=remote.max_response_bytes,
        max_requests_per_day=remote.max_requests_per_day,
        min_interval_seconds=remote.min_interval_seconds,
        audit_store=RemoteReviewAuditStore(database_path or settings.storage.database),
    )
