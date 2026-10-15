"""Bounded, CPU-only execution of coalesced detailed-report refreshes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .analysis_worker import WorkerAttempt, WorkerAttemptState
from .config import StorageSettings
from .database import EventDatabase
from .local_review_reports import dispatch_local_review_reports
from .report_queue import ReportClaimLostError, ReportQueueError, ReportRefreshQueue
from .reporting import DailyReportGenerator
from .storage import StorageResolver


def execute_report_refresh_once(
    database_path: Path,
    *,
    lease_seconds: int = 300,
    max_attempts: int = 3,
    retry_delay_seconds: int = 60,
    now: datetime | None = None,
) -> tuple[int, dict[str, object], WorkerAttempt]:
    """Refresh one captured revision, never interpreting media or sending alerts."""

    if type(retry_delay_seconds) is not int or not 0 <= retry_delay_seconds <= 86_400:
        raise ValueError("retry_delay_seconds must be between 0 and 86400")
    if type(lease_seconds) is not int or not 1 <= lease_seconds <= 86_400:
        raise ValueError("lease_seconds must be between 1 and 86400")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 100:
        raise ValueError("max_attempts must be between 1 and 100")
    # Recover durable post-review intents even if the CLI exited before handing
    # them to the refresh queue. Neither this dispatcher nor rendering uses VLMs.
    dispatched = dispatch_local_review_reports(database_path, limit=20, now=now)
    queue = ReportRefreshQueue(database_path)
    job = queue.claim_next(lease_seconds=lease_seconds, max_attempts=max_attempts, now=now)
    if job is None:
        if dispatched["pending"]:
            return (
                0,
                {
                    "status": "deferred",
                    "processed": False,
                    "reason": "local_review_report_pending",
                    "report_intents": dispatched,
                },
                WorkerAttempt(WorkerAttemptState.DEFERRED),
            )
        return 0, {"status": "idle", "processed": False}, WorkerAttempt(WorkerAttemptState.IDLE)
    try:
        root = Path(job.storage_root)
        if not root.is_dir() or root.is_symlink() or root.resolve(strict=True) != root:
            raise ReportQueueError("report output root is missing or unsafe")
        database = EventDatabase(database_path)
        database.initialize()
        generator = DailyReportGenerator(
            database=database,
            storage=StorageResolver(
                StorageSettings(
                    local_root=root,
                    nas_root=None,
                    prefer_nas=False,
                    database=database_path,
                )
            ),
            timezone_name=job.timezone_name,
        )
        report = generator.generate(
            job.local_date,
            claim_guard=lambda: queue.assert_claim_current(job),
        )
        completed = queue.complete(job, now=now)
        return (
            0,
            {
                "status": completed.status.value,
                "processed": True,
                "report": completed.to_public_dict(),
                "rendered_revision": job.claimed_revision,
                "event_count": report.event_count,
                "paths_disclosed": False,
                "report_intents": dispatched,
            },
            WorkerAttempt(WorkerAttemptState.COMPLETED, event_id=job.job_id),
        )
    except Exception as exc:
        if not isinstance(exc, ReportClaimLostError):
            try:
                failed = queue.fail(
                    job,
                    error_code=type(exc).__name__,
                    retry_delay_seconds=retry_delay_seconds,
                    max_attempts=max_attempts,
                    now=now,
                )
            except ReportClaimLostError:
                pass
            else:
                return (
                    2,
                    {
                        "status": failed.status.value,
                        "processed": False,
                        "report": failed.to_public_dict(),
                        "error_code": type(exc).__name__,
                        "events_preserved": True,
                        "paths_disclosed": False,
                        "report_intents": dispatched,
                    },
                    WorkerAttempt(
                        WorkerAttemptState.FAILED
                        if failed.status == "failed"
                        else WorkerAttemptState.RETRY_PENDING,
                        event_id=job.job_id,
                        error_code=type(exc).__name__,
                    ),
                )
        return (
            0,
            {
                "status": "deferred",
                "processed": False,
                "reason": "claim_lost",
                "report_intents": dispatched,
            },
            WorkerAttempt(WorkerAttemptState.DEFERRED),
        )
