"""Sequential analysis worker, queue status, GPU gate, and runtime status commands."""

from __future__ import annotations

import argparse
import math
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

from ..analysis_queue import AnalysisQueue
from ..analysis_service import (
    AnalysisDependencies,
    execute_analysis_run_once,
    validate_local_notification_policy,
)
from ..analysis_worker import (
    WorkerAttempt,
    WorkerLoopSettings,
    WorkerStopReason,
    run_analysis_worker,
)
from ..archive_queue import (
    ArchiveQueue,
)
from ..config import Settings, load_settings
from ..event_policy import HouseholdEventPolicy
from ..gpu_guard import probe_gpu_capacity
from ..media_health import MediaSourceHealth, assess_media_health
from ..notification import NotificationOutbox
from ..runtime_status import (
    RuntimeSafetySnapshot,
    analysis_capacity_snapshot,
    assess_archive_queue_health,
    assess_notification_queue_health,
    assess_queue_health,
    build_runtime_status,
    inspect_storage_capacity,
)


def _configured_online_event_policy(settings: Settings) -> HouseholdEventPolicy | None:
    """Load the one policy allowed to feed automatic local notifications."""

    notification = settings.notification
    if not notification.auto_file_enabled:
        return None
    policy = (
        HouseholdEventPolicy.read(notification.event_policy_path)
        if notification.event_policy_path is not None
        else None
    )
    validate_local_notification_policy(settings, policy)
    return policy


def _execute_analysis_run_once(
    settings: Settings,
    database_path: Path,
    event_policy: HouseholdEventPolicy | None,
    *,
    defer_reports: bool = False,
) -> tuple[int, dict[str, object], WorkerAttempt]:
    worker = settings.analysis_worker
    dependencies = AnalysisDependencies(
        local_adapter_factory=lambda: _local_adapter(settings),
        secondary_adapter_factory=lambda: _secondary_adapter(settings),
        remote_adapter_factory=lambda event_database_path: _remote_adapter(
            settings, database_path=event_database_path
        ),
        review_policy_factory=lambda: _review_policy(settings),
        probe_gpu_capacity=lambda: probe_gpu_capacity(
            device=worker.gpu_device,
            min_free_vram_mib=worker.min_free_vram_mib,
            max_utilization_percent=worker.max_gpu_utilization_percent,
            timeout_seconds=worker.gpu_probe_timeout_seconds,
        ),
    )
    return execute_analysis_run_once(
        settings,
        database_path,
        event_policy,
        dependencies=dependencies,
        defer_reports=defer_reports,
    )


def command_runtime_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    checked_at = datetime.now(UTC)
    media = tuple(
        assess_media_health(
            MediaSourceHealth.read(path),
            checked_at=checked_at,
            stale_after_seconds=settings.media_health.stale_after_seconds,
            future_tolerance_seconds=settings.media_health.future_tolerance_seconds,
        )
        for path in args.health
    )
    database_path = Path(args.database) if args.database else settings.storage.database
    queue_snapshot = AnalysisQueue(database_path).snapshot_read_only(now=checked_at)
    queue = assess_queue_health(
        queue_snapshot,
        max_active=settings.analysis_queue.max_pending,
        max_pending_age_seconds=args.max_pending_age_seconds,
    )
    archive_queue = assess_archive_queue_health(
        ArchiveQueue(database_path).snapshot_read_only(now=checked_at),
        max_active=settings.archive_queue.max_pending,
        max_pending_age_seconds=args.max_archive_pending_age_seconds,
    )
    notification_queue = assess_notification_queue_health(
        NotificationOutbox(database_path).snapshot_read_only(now=checked_at),
        enabled=settings.notification.enabled,
        max_active=settings.notification.max_pending,
        max_pending_age_seconds=(
            args.max_notification_pending_age_seconds
            if args.max_notification_pending_age_seconds is not None
            else settings.notification.max_pending_age_seconds
        ),
    )

    if not math.isfinite(args.min_free_gib) or args.min_free_gib <= 0:
        raise ValueError("min-free-gib must be finite and positive")
    minimum_free_bytes = int(args.min_free_gib * (1024**3))
    storage = inspect_storage_capacity(
        Path(args.storage_root) if args.storage_root else settings.storage.local_root,
        minimum_free_bytes=minimum_free_bytes,
    )

    worker = settings.analysis_worker
    gpu_capacity = (
        probe_gpu_capacity(
            device=worker.gpu_device,
            min_free_vram_mib=worker.min_free_vram_mib,
            max_utilization_percent=worker.max_gpu_utilization_percent,
            timeout_seconds=worker.gpu_probe_timeout_seconds,
        )
        if worker.gpu_guard_enabled
        else None
    )
    analysis = analysis_capacity_snapshot(
        gpu_guard_enabled=worker.gpu_guard_enabled,
        capacity=gpu_capacity,
    )
    safety = RuntimeSafetySnapshot(
        remote_enabled=settings.remote_vlm.enabled,
        remote_upload_consent=settings.remote_vlm.upload_consent,
        remote_uploads_allowed=settings.remote_vlm.uploads_allowed,
        notification_enabled=settings.notification.enabled,
        external_notification_delivery_enabled=(settings.notification.external_delivery_enabled),
        retention_scan_enabled=settings.storage.retention_scan_enabled,
        retention_delete_enabled=settings.storage.retention_delete_enabled,
    )
    status = build_runtime_status(
        checked_at=checked_at,
        media=media,
        queue=queue,
        archive_queue=archive_queue,
        notification_queue=notification_queue,
        storage=storage,
        analysis=analysis,
        safety=safety,
    )
    _json_print(status.to_dict())
    return 0 if status.operational else 1


def command_analysis_run_once(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    event_policy = _configured_online_event_policy(settings)
    exit_code, payload, _attempt = _execute_analysis_run_once(
        settings,
        database_path,
        event_policy,
        defer_reports=args.defer_reports,
    )
    _json_print(payload)
    return exit_code


def command_analysis_worker(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    event_policy = _configured_online_event_policy(settings)
    _database_from_path(database_path)
    queue = AnalysisQueue(database_path)
    worker_settings = settings.analysis_worker
    loop_settings = WorkerLoopSettings(
        max_jobs=(args.max_jobs if args.max_jobs is not None else worker_settings.max_jobs_per_run),
        idle_exit_seconds=(
            args.idle_exit_seconds
            if args.idle_exit_seconds is not None
            else worker_settings.idle_exit_seconds
        ),
        max_runtime_seconds=(
            args.max_runtime_seconds
            if args.max_runtime_seconds is not None
            else worker_settings.max_runtime_seconds
        ),
        poll_seconds=(
            args.poll_seconds if args.poll_seconds is not None else worker_settings.poll_seconds
        ),
    )
    summary = run_analysis_worker(
        settings=loop_settings,
        run_once=lambda: _execute_analysis_run_once(
            settings,
            database_path,
            event_policy,
            defer_reports=args.defer_reports,
        )[2],
        queue_snapshot=lambda: queue.snapshot().to_dict(),
    )
    _json_print(
        {
            "status": "stopped",
            "database": str(database_path),
            "worker": summary.to_dict(),
        }
    )
    if summary.stop_reason is WorkerStopReason.INTERRUPTED:
        return 130
    return 2 if summary.failed_jobs else 0


def command_analysis_queue_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    _database_from_path(database_path)
    snapshot = AnalysisQueue(database_path).snapshot_read_only()
    _json_print(
        {
            "database": str(database_path),
            "max_active": settings.analysis_queue.max_pending,
            "queue": snapshot.to_dict(),
        }
    )
    return 0


def command_analysis_gpu_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    worker = settings.analysis_worker
    capacity = probe_gpu_capacity(
        device=worker.gpu_device,
        min_free_vram_mib=worker.min_free_vram_mib,
        max_utilization_percent=worker.max_gpu_utilization_percent,
        timeout_seconds=worker.gpu_probe_timeout_seconds,
    )
    _json_print(
        {
            "guard_enabled": worker.gpu_guard_enabled,
            "gpu": capacity.to_dict(),
        }
    )
    return 0


def command_analysis_retry(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    _database_from_path(database_path)
    queue = AnalysisQueue(database_path)
    job = queue.retry_failed(
        args.event_id,
        max_active=settings.analysis_queue.max_pending,
    )
    _json_print(
        {
            "event_id": job.event_id,
            "status": job.status.value,
            "attempt_count": job.attempt_count,
            "database": str(database_path),
            "queue": queue.counts().to_dict(),
        }
    )
    return 0
