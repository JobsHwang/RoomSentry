"""Notification message, outbox, and local test commands."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
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
from ..config import Settings, load_settings
from ..home_notification import run_home_local_notification_test
from ..notification import (
    FileNotificationSink,
    NotificationMessage,
    NotificationOutbox,
    StdoutNotificationSink,
    deliver_notification_once,
)
from ..notification_transports import (
    SMTPNotificationProfile,
    SMTPNotificationSink,
    WebhookNotificationProfile,
    WebhookNotificationSink,
    load_notification_transport_profile,
)


def _require_notifications_enabled(settings: Settings) -> None:
    if not settings.notification.enabled:
        raise ValueError("local notification delivery is disabled")


def _require_notification_sink_enabled(settings: Settings, sink_kind: str) -> None:
    _require_notifications_enabled(settings)
    if sink_kind in {"smtp", "webhook"} and not settings.notification.external_delivery_enabled:
        raise ValueError("external notification delivery is disabled")


def _notification_sink_from_args(args: argparse.Namespace):
    if args.sink == "stdout":
        if args.output_dir is not None or args.profile is not None:
            raise ValueError("stdout notification sink does not accept delivery options")
        return StdoutNotificationSink()
    if args.sink == "file":
        if args.output_dir is None:
            raise ValueError("file notification sink requires --output-dir")
        if args.profile is not None:
            raise ValueError("file notification sink does not accept --profile")
        return FileNotificationSink(args.output_dir)
    if args.output_dir is not None:
        raise ValueError("external notification sink does not accept --output-dir")
    if args.profile is None:
        raise ValueError("external notification sink requires --profile")
    profile = load_notification_transport_profile(args.profile)
    if profile.kind != args.sink or profile.sink_id != args.sink_id:
        raise ValueError("notification transport profile does not match sink selection")
    if isinstance(profile, SMTPNotificationProfile):
        return SMTPNotificationSink(profile)
    if isinstance(profile, WebhookNotificationProfile):
        return WebhookNotificationSink(profile)
    raise ValueError("notification transport profile kind is unsupported")


def command_validate_notification_json(args: argparse.Namespace) -> int:
    message = NotificationMessage.from_json(Path(args.path).read_text(encoding="utf-8"))
    _json_print({"valid": True, "notification": message.to_dict()})
    return 0


def command_notification_transport_preview(args: argparse.Namespace) -> int:
    profile = load_notification_transport_profile(args.profile)
    _json_print(profile.to_preview())
    return 0


def command_notification_enqueue_event(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    _require_notification_sink_enabled(settings, args.sink)
    database_path = Path(args.database) if args.database else settings.storage.database
    database = _database_from_path(database_path)
    event = database.get_event(args.event_id)
    if event is None:
        raise ValueError("notification event does not exist")
    outbox = NotificationOutbox(database_path)
    result = outbox.enqueue(
        event=event,
        sink_kind=args.sink,
        sink_id=args.sink_id,
        max_active=settings.notification.max_pending,
    )
    _json_print(
        {
            "notification_id": result.job.notification_id,
            "event_id": result.job.message.event_id,
            "sink_kind": result.job.sink_kind,
            "sink_id": result.job.sink_id,
            "status": result.job.status.value,
            "review_status": result.job.message.review_status.value,
            "reused_existing": result.reused_existing,
            "contains_media_paths": False,
        }
    )
    return 0


def command_notification_local_test(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    _require_notifications_enabled(settings)
    if not secrets.compare_digest(args.confirmation.upper(), "TEST"):
        raise ValueError("local notification test requires TEST confirmation")
    database_path = Path(args.database) if args.database else settings.storage.database
    receipt = run_home_local_notification_test(
        database_path=database_path,
        target_root=args.output_dir,
        max_pending=settings.notification.max_pending,
        lease_seconds=settings.notification.lease_seconds,
        retry_delay_seconds=settings.notification.retry_delay_seconds,
        max_attempts=settings.notification.max_attempts,
    )
    _json_print(receipt.to_public_dict())
    return 0


def command_notification_run_once(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    _require_notification_sink_enabled(settings, args.sink)
    database_path = Path(args.database) if args.database else settings.storage.database
    _database_from_path(database_path)
    outbox = NotificationOutbox(database_path)
    sink = _notification_sink_from_args(args)
    attempt = deliver_notification_once(
        outbox=outbox,
        sink=sink,
        sink_id=args.sink_id,
        lease_seconds=settings.notification.lease_seconds,
        retry_delay_seconds=settings.notification.retry_delay_seconds,
        max_attempts=settings.notification.max_attempts,
    )
    summary = {"delivery": attempt.to_dict(), "database": str(database_path)}
    if args.sink == "stdout" and attempt.status != "idle":
        print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)
    else:
        _json_print(summary)
    return 1 if attempt.error_code is not None else 0


def command_notification_worker(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    _require_notification_sink_enabled(settings, args.sink)
    database_path = Path(args.database) if args.database else settings.storage.database
    _database_from_path(database_path)
    outbox = NotificationOutbox(database_path)
    sink = _notification_sink_from_args(args)

    def run_once() -> WorkerAttempt:
        attempt = deliver_notification_once(
            outbox=outbox,
            sink=sink,
            sink_id=args.sink_id,
            lease_seconds=settings.notification.lease_seconds,
            retry_delay_seconds=settings.notification.retry_delay_seconds,
            max_attempts=settings.notification.max_attempts,
        )
        if attempt.status == "idle":
            return WorkerAttempt(WorkerAttemptState.IDLE)
        if attempt.status == "delivered":
            state = WorkerAttemptState.COMPLETED
        elif attempt.status == "pending":
            state = WorkerAttemptState.RETRY_PENDING
        else:
            state = WorkerAttemptState.FAILED
        return WorkerAttempt(
            state,
            event_id=attempt.notification_id,
            error_code=attempt.error_code,
        )

    summary = run_analysis_worker(
        settings=WorkerLoopSettings(
            max_jobs=args.max_jobs,
            idle_exit_seconds=args.idle_exit_seconds,
            max_runtime_seconds=args.max_runtime_seconds,
            poll_seconds=args.poll_seconds,
        ),
        run_once=run_once,
        queue_snapshot=lambda: outbox.snapshot_read_only().to_dict(),
    )
    _json_print(
        {
            "status": "stopped",
            "sink_kind": args.sink,
            "sink_id": args.sink_id,
            "database": str(database_path),
            "worker": summary.to_dict(),
        }
    )
    if summary.stop_reason is WorkerStopReason.INTERRUPTED:
        return 130
    return 2 if summary.failed_jobs else 0


def command_notification_queue_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    snapshot = NotificationOutbox(database_path).snapshot_read_only()
    _json_print(
        {
            "enabled": settings.notification.enabled,
            "database": str(database_path),
            "max_active": settings.notification.max_pending,
            "queue": snapshot.to_dict(),
        }
    )
    return 0


def command_notification_retry(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    _require_notifications_enabled(settings)
    database_path = Path(args.database) if args.database else settings.storage.database
    _database_from_path(database_path)
    job = NotificationOutbox(database_path).retry_failed(
        args.notification_id,
        max_active=settings.notification.max_pending,
    )
    _json_print(
        {
            "notification_id": job.notification_id,
            "status": job.status.value,
            "attempt_count": job.attempt_count,
            "database": str(database_path),
        }
    )
    return 0
