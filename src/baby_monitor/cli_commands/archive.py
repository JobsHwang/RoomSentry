"""Archive plan/copy/queue and retention scan commands."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
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

from ..analysis_worker import (
    WorkerAttempt,
    WorkerAttemptState,
    WorkerLoopSettings,
    WorkerStopReason,
    run_analysis_worker,
)
from ..archive import (
    ArchiveConflictError,
    ArchiveError,
    ArchiveManifest,
    ArchiveUnavailableError,
    FilesystemArchiveAdapter,
    build_archive_manifest,
)
from ..archive_binding import (
    build_event_archive_manifest,
    build_report_archive_manifest,
)
from ..archive_queue import (
    ArchiveJobStatus,
    ArchiveQueue,
    ArchiveQueueFullError,
)
from ..config import Settings, load_settings


def _enqueue_archive_manifest(
    settings: Settings,
    *,
    database_path: Path,
    manifest: ArchiveManifest,
    source_root: str | Path,
) -> tuple[dict[str, object], int]:
    queue = ArchiveQueue(database_path)
    try:
        result = queue.enqueue(
            manifest=manifest,
            source_root=source_root,
            max_active=settings.archive_queue.max_pending,
        )
    except ArchiveQueueFullError:
        snapshot = queue.snapshot()
        return (
            {
                "version": 1,
                "archive_id": manifest.archive_id,
                "queued": False,
                "status": "failed",
                "error_code": "queue_full",
                "queue": snapshot.to_dict(),
                "paths_disclosed": False,
            },
            1,
        )
    snapshot = queue.snapshot()
    return (
        {
            "version": 1,
            "archive_id": result.job.archive_id,
            "queued": True,
            "reused_existing": result.reused_existing,
            "status": result.job.status.value,
            "queue": snapshot.to_dict(),
            "paths_disclosed": False,
        },
        0,
    )


def _archive_run_once(
    settings: Settings,
    *,
    database_path: Path,
    target_root: str,
) -> tuple[dict[str, object], WorkerAttempt]:
    queue = ArchiveQueue(database_path)
    job = queue.claim_next(
        lease_seconds=settings.archive_queue.lease_seconds,
        max_attempts=settings.archive_queue.max_attempts,
    )
    if job is None:
        return (
            {
                "processed": False,
                "status": "idle",
                "queue": queue.snapshot().to_dict(),
                "paths_disclosed": False,
            },
            WorkerAttempt(WorkerAttemptState.IDLE),
        )
    try:
        result = FilesystemArchiveAdapter(
            source_root=job.source_root,
            target_root=target_root,
        ).archive(job.manifest)
    except ArchiveConflictError:
        failed = queue.mark_failed_attempt(
            job.archive_id,
            error_code="archive_conflict",
            retry_delay_seconds=0,
            max_attempts=1,
        )
        attempt = WorkerAttempt(
            WorkerAttemptState.FAILED,
            event_id=job.archive_id,
            error_code="archive_conflict",
        )
        payload_result: dict[str, object] | None = None
    except ArchiveUnavailableError:
        failed = queue.mark_failed_attempt(
            job.archive_id,
            error_code="archive_unavailable",
            retry_delay_seconds=settings.archive_queue.retry_delay_seconds,
            max_attempts=settings.archive_queue.max_attempts,
        )
        attempt = WorkerAttempt(
            (
                WorkerAttemptState.FAILED
                if failed.status is ArchiveJobStatus.FAILED
                else WorkerAttemptState.RETRY_PENDING
            ),
            event_id=job.archive_id,
            error_code="archive_unavailable",
        )
        payload_result = None
    except ArchiveError:
        failed = queue.mark_failed_attempt(
            job.archive_id,
            error_code="archive_source_invalid",
            retry_delay_seconds=0,
            max_attempts=1,
        )
        attempt = WorkerAttempt(
            WorkerAttemptState.FAILED,
            event_id=job.archive_id,
            error_code="archive_source_invalid",
        )
        payload_result = None
    else:
        completed = queue.mark_completed(job.archive_id)
        failed = completed
        attempt = WorkerAttempt(
            WorkerAttemptState.COMPLETED,
            event_id=job.archive_id,
        )
        payload_result = result.to_dict()

    payload = {
        "processed": True,
        "archive_id": job.archive_id,
        "status": failed.status.value,
        "attempt_count": failed.attempt_count,
        "error_code": attempt.error_code,
        "result": payload_result,
        "queue": queue.snapshot().to_dict(),
        "paths_disclosed": False,
    }
    return payload, attempt


def command_retention_scan(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database = _database_from_path(args.database or settings.storage.database)
    cutoff = datetime.now(UTC) - timedelta(days=settings.storage.retention_days)
    candidates = [
        {"event_id": event_id, "clip_path": clip, "frame_paths": list(frames)}
        for event_id, clip, frames in database.iter_media_before(cutoff)
    ]
    _json_print(
        {
            "dry_run": True,
            "deletion_performed": False,
            "retention_days": settings.storage.retention_days,
            "cutoff": cutoff.isoformat(),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "note": "this command never deletes files",
        }
    )
    return 0


def command_archive_plan(args: argparse.Namespace) -> int:
    manifest = build_archive_manifest(
        archive_id=args.archive_id,
        source_root=args.source_root,
        files=tuple(args.file),
    )
    manifest.write_new(args.output)
    _json_print(
        {
            "version": 1,
            "archive_id": manifest.archive_id,
            "items": len(manifest.items),
            "total_bytes": manifest.total_bytes,
            "manifest_written": True,
            "paths_disclosed": False,
        }
    )
    return 0


def command_archive_copy(args: argparse.Namespace) -> int:
    manifest = ArchiveManifest.read(args.manifest)
    result = FilesystemArchiveAdapter(
        source_root=args.source_root,
        target_root=args.target_root,
    ).archive(manifest)
    _json_print(result.to_dict())
    return 0


def command_archive_enqueue(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    manifest = ArchiveManifest.read(args.manifest)
    payload, exit_code = _enqueue_archive_manifest(
        settings,
        database_path=database_path,
        manifest=manifest,
        source_root=args.source_root,
    )
    _json_print(payload)
    return exit_code


def command_archive_enqueue_event(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    database = _database_from_path(database_path)
    manifest = build_event_archive_manifest(
        database=database,
        event_id=args.event_id,
        source_root=args.source_root,
    )
    payload, exit_code = _enqueue_archive_manifest(
        settings,
        database_path=database_path,
        manifest=manifest,
        source_root=args.source_root,
    )
    payload["binding_kind"] = "event_media"
    _json_print(payload)
    return exit_code


def command_archive_enqueue_report(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    report_date = date.fromisoformat(args.date)
    manifest = build_report_archive_manifest(
        report_date=report_date,
        source_root=args.source_root,
    )
    payload, exit_code = _enqueue_archive_manifest(
        settings,
        database_path=database_path,
        manifest=manifest,
        source_root=args.source_root,
    )
    payload["binding_kind"] = "daily_report"
    payload["report_date"] = report_date.isoformat()
    _json_print(payload)
    return exit_code


def command_archive_run_once(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    payload, attempt = _archive_run_once(
        settings,
        database_path=database_path,
        target_root=args.target_root,
    )
    _json_print(payload)
    return 2 if attempt.state is WorkerAttemptState.FAILED else 0


def command_archive_worker(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    worker = settings.archive_worker
    queue = ArchiveQueue(database_path)
    summary = run_analysis_worker(
        settings=WorkerLoopSettings(
            max_jobs=(args.max_jobs if args.max_jobs is not None else worker.max_jobs_per_run),
            idle_exit_seconds=(
                args.idle_exit_seconds
                if args.idle_exit_seconds is not None
                else worker.idle_exit_seconds
            ),
            max_runtime_seconds=(
                args.max_runtime_seconds
                if args.max_runtime_seconds is not None
                else worker.max_runtime_seconds
            ),
            poll_seconds=(
                args.poll_seconds if args.poll_seconds is not None else worker.poll_seconds
            ),
        ),
        run_once=lambda: _archive_run_once(
            settings,
            database_path=database_path,
            target_root=args.target_root,
        )[1],
        queue_snapshot=lambda: queue.snapshot().to_dict(),
    )
    worker_payload = summary.to_dict()
    worker_payload["last_archive_id"] = worker_payload.pop("last_event_id")
    _json_print(
        {
            "version": 1,
            "status": "stopped",
            "worker": worker_payload,
            "paths_disclosed": False,
        }
    )
    if summary.stop_reason is WorkerStopReason.INTERRUPTED:
        return 130
    return 2 if summary.failed_jobs else 0


def command_archive_queue_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    snapshot = ArchiveQueue(database_path).snapshot_read_only()
    _json_print(
        {
            "version": 1,
            "max_active": settings.archive_queue.max_pending,
            "queue": snapshot.to_dict(),
            "paths_disclosed": False,
        }
    )
    return 0


def command_archive_retry(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    queue = ArchiveQueue(database_path)
    job = queue.retry_failed(
        args.archive_id,
        max_active=settings.archive_queue.max_pending,
    )
    _json_print(
        {
            "version": 1,
            "archive_id": job.archive_id,
            "status": job.status.value,
            "attempt_count": job.attempt_count,
            "queue": queue.snapshot().to_dict(),
            "paths_disclosed": False,
        }
    )
    return 0
