"""Explicit CLI hosts for durable, local-only detailed-report refreshes."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ..analysis_worker import WorkerLoopSettings, WorkerStopReason, run_analysis_worker
from ..cli_support import _json_print
from ..config import load_settings
from ..report_queue import ReportRefreshQueue
from ..report_service import execute_report_refresh_once


def _database_path(args: argparse.Namespace) -> Path:
    settings = load_settings(args.config)
    return Path(args.database) if args.database else settings.storage.database


def command_report_refresh_request(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    job = ReportRefreshQueue(database_path).request(
        request_id=f"manual:{args.request_id}",
        local_date=date.fromisoformat(args.date),
        timezone_name=settings.timezone,
        storage_root=Path(args.output_root),
        debounce_seconds=args.debounce_seconds,
    )
    _json_print({"status": "recorded", "report": job.to_public_dict()})
    return 0


def command_report_refresh_run_once(args: argparse.Namespace) -> int:
    code, payload, _attempt = execute_report_refresh_once(
        _database_path(args),
        lease_seconds=args.lease_seconds,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    _json_print(payload)
    return code


def command_report_refresh_worker(args: argparse.Namespace) -> int:
    database_path = _database_path(args)
    queue = ReportRefreshQueue(database_path)
    summary = run_analysis_worker(
        settings=WorkerLoopSettings(
            max_jobs=args.max_jobs,
            idle_exit_seconds=args.idle_exit_seconds,
            max_runtime_seconds=args.max_runtime_seconds,
            poll_seconds=args.poll_seconds,
        ),
        run_once=lambda: execute_report_refresh_once(
            database_path,
            lease_seconds=args.lease_seconds,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )[2],
        queue_snapshot=queue.snapshot_read_only,
    )
    payload = summary.to_dict()
    # Reuse the bounded scheduler, but do not present report IDs as event IDs.
    payload["last_job_id"] = payload.pop("last_event_id")
    _json_print({"status": "stopped", "worker": payload, "paths_disclosed": False})
    if summary.stop_reason is WorkerStopReason.INTERRUPTED:
        return 130
    return 2 if summary.failed_jobs else 0


def command_report_refresh_status(args: argparse.Namespace) -> int:
    _json_print(ReportRefreshQueue(_database_path(args)).snapshot_read_only())
    return 0


def command_report_refresh_retry(args: argparse.Namespace) -> int:
    job = ReportRefreshQueue(_database_path(args)).retry_failed(args.job_id)
    _json_print({"status": "retry_pending", "report": job.to_public_dict()})
    return 0
