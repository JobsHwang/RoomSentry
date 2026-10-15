"""Dispatch durable local-review report intents without rendering or model calls."""

from __future__ import annotations

import hashlib
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from .local_review import LocalReviewError, LocalReviewStore
from .report_queue import ReportQueueError, ReportRefreshQueue


def dispatch_local_review_reports(
    database_path: Path,
    *,
    limit: int = 20,
    request_id: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Enqueue a bounded snapshot; pending counts only this snapshot's unacked intents.

    Enqueue precedes acknowledgement. Its deterministic request identifier makes
    a retry after an acknowledgement failure reuse the same queue revision.
    Missing databases/tables are handled read-only by the outbox store; read
    failures propagate rather than appearing as a successful empty dispatch.
    """
    if type(limit) is not int or not 1 <= limit <= 100:
        raise LocalReviewError("invalid_input")
    if now is not None:
        try:
            if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
                raise LocalReviewError("invalid_input")
            now = now.astimezone(UTC)
        except (ValueError, OverflowError):
            raise LocalReviewError("invalid_input") from None
    store = LocalReviewStore(database_path)
    intents = store.pending_report_refreshes(limit=limit, request_id=request_id)
    receipt = {"scanned": len(intents), "enqueued": 0, "pending": 0, "errors_count": 0}
    if not intents:
        return receipt
    queue = ReportRefreshQueue(database_path)
    for intent in intents:
        try:
            job = queue.request(
                request_id="local-review:"
                + hashlib.sha256(intent.request_id.encode("utf-8")).hexdigest(),
                local_date=intent.local_date,
                timezone_name=intent.timezone_name,
                storage_root=intent.storage_root,
                debounce_seconds=0,
                now=now,
            )
            store.mark_report_enqueued(intent.request_id, job_id=job.job_id, now=now)
        except (ReportQueueError, LocalReviewError):
            receipt["pending"] += 1
            receipt["errors_count"] += 1
            # A failed error acknowledgement cannot turn an unacked intent into
            # success; the durable pending intent remains recoverable.
            with suppress(LocalReviewError):
                store.mark_report_failed(
                    intent.request_id, error_code="report_enqueue_failed", now=now
                )
        else:
            receipt["enqueued"] += 1
    return receipt
