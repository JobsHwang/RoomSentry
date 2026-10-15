"""Queued event-analysis use case, shared by command-line and embedded hosts.

Adapter construction and the GPU probe belong to the host. Synchronous reports
remain the default. Explicit deferred mode durably schedules a coalesced refresh
before routing and completion; rendering then has its own retries and no GPU.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from zoneinfo import ZoneInfo

from .adapters.vlm import VLMAdapter
from .analysis_queue import AnalysisQueue
from .analysis_worker import WorkerAttempt, WorkerAttemptState
from .config import Settings
from .database import EventDatabase
from .event_policy import EventPolicyError, HouseholdEventPolicy, policy_event_from_stored_event
from .event_policy_runtime import EventPolicyDecisionStore
from .gpu_guard import GPUCapacity
from .notification import NotificationOutbox, NotificationQueueFullError
from .pipeline import process_prepared_event
from .policy import ReviewPolicy
from .report_queue import ReportRefreshQueue
from .reporting import DailyReportGenerator
from .storage import StorageResolver


@dataclass(frozen=True, slots=True)
class AnalysisDependencies:
    """Lazy host bindings; model factories run only after a job is admitted."""

    local_adapter_factory: Callable[[], VLMAdapter]
    secondary_adapter_factory: Callable[[], VLMAdapter | None]
    remote_adapter_factory: Callable[[Path], VLMAdapter | None]
    review_policy_factory: Callable[[], ReviewPolicy]
    probe_gpu_capacity: Callable[[], GPUCapacity]


def validate_local_notification_policy(
    settings: Settings,
    event_policy: HouseholdEventPolicy | None,
) -> None:
    """Reject incomplete host bindings before claiming evidence or starting work."""

    notification = settings.notification
    if not notification.auto_file_enabled:
        return
    if not isinstance(event_policy, HouseholdEventPolicy):
        raise EventPolicyError("automatic local notifications require an event policy")
    if any(
        sink_id != notification.auto_file_sink_id
        for rule in event_policy.rules
        if rule.local_only
        for sink_id in rule.sink_ids
    ):
        raise EventPolicyError(
            "online event policy local sinks must match the configured automatic file sink"
        )


def execute_analysis_run_once(
    settings: Settings,
    database_path: Path,
    event_policy: HouseholdEventPolicy | None,
    *,
    dependencies: AnalysisDependencies,
    defer_reports: bool = False,
) -> tuple[int, dict[str, object], WorkerAttempt]:
    """Advance one queued event without depending on a command-line host."""

    if type(defer_reports) is not bool:
        raise ValueError("defer_reports must be boolean")
    validate_local_notification_policy(settings, event_policy)
    database = EventDatabase(database_path)
    database.initialize()
    queue = AnalysisQueue(database_path)
    worker = settings.analysis_worker
    if worker.gpu_guard_enabled:
        capacity = dependencies.probe_gpu_capacity()
        if not capacity.allowed:
            payload: dict[str, object] = {
                "status": "deferred",
                "processed": False,
                "reason": capacity.reason,
                "gpu": capacity.to_dict(),
                "database": str(database_path),
                "queue": queue.counts().to_dict(),
            }
            return (
                0,
                payload,
                WorkerAttempt(WorkerAttemptState.DEFERRED),
            )
    job = queue.claim_next(
        lease_seconds=settings.analysis_queue.lease_seconds,
        max_attempts=settings.analysis_queue.max_attempts,
    )
    if job is None:
        payload = {
            "status": "idle",
            "processed": False,
            "database": str(database_path),
            "queue": queue.counts().to_dict(),
        }
        return (
            0,
            payload,
            WorkerAttempt(WorkerAttemptState.IDLE),
        )

    event_policy_decision: dict[str, object] | None = None
    notification_enqueue: dict[str, object] | None = None
    try:
        job.require_local_media()
        local_adapter = dependencies.local_adapter_factory()
        processed = process_prepared_event(
            candidate=job.candidate,
            occurred_at=job.occurred_at,
            source_kind=job.source_kind,
            source_ref=job.source_ref,
            media=job.media,
            local_vlm=local_adapter,
            secondary_vlm=dependencies.secondary_adapter_factory(),
            remote_vlm=dependencies.remote_adapter_factory(database.path),
            remote_uploads_allowed=settings.remote_vlm.uploads_allowed,
            database=database,
            policy=dependencies.review_policy_factory(),
        )
        stored = database.get_event(job.event_id)
        if stored is None:
            raise RuntimeError("processed event disappeared from SQLite")
        report_date = job.occurred_at.astimezone(ZoneInfo(settings.timezone)).date()
        if defer_reports:
            refresh = ReportRefreshQueue(database_path).request(
                request_id="event:" + hashlib.sha256(job.event_id.encode("utf-8")).hexdigest(),
                local_date=report_date,
                timezone_name=settings.timezone,
                storage_root=Path(job.storage_root),
            )
            report_payload: dict[str, object] = {
                "mode": "queued",
                **refresh.to_public_dict(),
            }
        else:
            report_storage = StorageResolver(
                replace(
                    settings.storage,
                    local_root=Path(job.storage_root),
                    nas_root=None,
                    prefer_nas=False,
                    database=database_path,
                )
            )
            report = DailyReportGenerator(
                database=database,
                storage=report_storage,
                timezone_name=settings.timezone,
            ).generate(report_date)
            report_payload = {
                "date": report_date.isoformat(),
                "event_count": report.event_count,
                "markdown": str(report.markdown_path),
                "html": str(report.html_path),
            }
        if settings.notification.auto_file_enabled:
            if event_policy is None:
                raise RuntimeError("automatic local notification policy is unavailable")
            policy_result = EventPolicyDecisionStore(database_path).decide_and_record(
                policy=event_policy,
                event=policy_event_from_stored_event(stored),
            )
            event_policy_decision = policy_result.to_public_dict()
            route = policy_result.record.decision
            if route.action == "suppress":
                notification_enqueue = {
                    "status": "suppressed",
                    "policy_reason": route.reason,
                    "policy_rule_id": route.rule_id,
                }
            elif not route.local_only:
                notification_enqueue = {
                    "status": "not_enqueued",
                    "policy_reason": route.reason,
                    "policy_rule_id": route.rule_id,
                    "error_code": "external_delivery_not_authorized",
                }
            else:
                if len(route.sink_ids) != 1:
                    raise RuntimeError("online event policy local sink binding is invalid")
                sink_id = route.sink_ids[0]
                try:
                    notification_result = NotificationOutbox(database_path).enqueue(
                        event=stored,
                        sink_kind="file",
                        sink_id=sink_id,
                        max_active=settings.notification.max_pending,
                    )
                    notification_enqueue = {
                        "status": notification_result.job.status.value,
                        "notification_id": notification_result.job.notification_id,
                        "sink_kind": "file",
                        "sink_id": sink_id,
                        "reused_existing": notification_result.reused_existing,
                        "policy_reason": route.reason,
                        "policy_rule_id": route.rule_id,
                    }
                except NotificationQueueFullError:
                    notification_enqueue = {
                        "status": "failed",
                        "sink_kind": "file",
                        "sink_id": sink_id,
                        "error_code": "queue_full",
                        "policy_reason": route.reason,
                        "policy_rule_id": route.rule_id,
                    }
        completed = queue.mark_completed(job.event_id)
    except Exception as exc:
        error_code = type(exc).__name__
        failed = queue.mark_failed_attempt(
            job.event_id,
            error_code=error_code,
            retry_delay_seconds=settings.analysis_queue.retry_delay_seconds,
            max_attempts=settings.analysis_queue.max_attempts,
        )
        payload = {
            "status": failed.status.value,
            "processed": False,
            "event_id": job.event_id,
            "attempt_count": failed.attempt_count,
            "error_code": error_code,
            "evidence_preserved": True,
            "database": str(database_path),
            "queue": queue.counts().to_dict(),
        }
        return (
            2,
            payload,
            WorkerAttempt(
                (
                    WorkerAttemptState.RETRY_PENDING
                    if failed.status.value == "pending"
                    else WorkerAttemptState.FAILED
                ),
                event_id=job.event_id,
                error_code=error_code,
            ),
        )

    payload = {
        "status": completed.status.value,
        "processed": True,
        "event_id": job.event_id,
        "attempt_count": completed.attempt_count,
        "local_vlm": local_adapter.name,
        "secondary_vlm": (settings.secondary_vlm.name if processed.secondary_reviewed else None),
        "result": stored.result.to_dict(),
        "decision": stored.decision.to_dict(),
        "remote_reviewed": processed.remote_reviewed,
        "secondary_reviewed": processed.secondary_reviewed,
        "pending_remote_review": processed.pending_remote_review,
        "reused_existing": processed.reused_existing,
        "event_policy_decision": event_policy_decision,
        "notification_enqueue": notification_enqueue,
        "database": str(database_path),
        "report": report_payload,
        "queue": queue.counts().to_dict(),
    }
    return (
        0,
        payload,
        WorkerAttempt(WorkerAttemptState.COMPLETED, event_id=job.event_id),
    )
