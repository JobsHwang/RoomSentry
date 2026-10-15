from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from baby_monitor.adapters.visual_predicate import (
    ResidentCommandVisualPredicateAdapter,
    VisualPredicateAdapter,
)
from baby_monitor.analysis_queue import AnalysisJobStatus, AnalysisQueue
from baby_monitor.visual_predicate_shadow import (
    VisualPredicateShadowStore,
    run_visual_predicate_shadow,
    select_visual_predicate_shadow,
)
from baby_monitor.visual_predicate_shadow_queue import (
    VisualPredicateShadowAttemptQueue,
    VisualPredicateShadowAttemptStatus,
    VisualPredicateShadowQueueFullError,
)
from baby_monitor.visual_predicate_shadow_runtime import (
    VisualPredicateShadowRuntimeSettings,
    admit_private_shadow_frame,
)


class VisualPredicateShadowWorkerStatus(StrEnum):
    IDLE = "idle"
    COMPLETED = "completed"
    RETRY_PENDING = "retry_pending"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowSyncSummary:
    scanned_completed_jobs: int
    scanned_pending_jobs: int
    selected_jobs: int
    enqueued_jobs: int
    skipped_jobs: int
    existing_attempts: int
    queue_full: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "scanned_analysis_jobs": self.scanned_completed_jobs + self.scanned_pending_jobs,
            "scanned_completed_jobs": self.scanned_completed_jobs,
            "scanned_pending_jobs": self.scanned_pending_jobs,
            "selected_jobs": self.selected_jobs,
            "enqueued_jobs": self.enqueued_jobs,
            "skipped_jobs": self.skipped_jobs,
            "existing_attempts": self.existing_attempts,
            "queue_full": self.queue_full,
            "periodic_sample_due": False,
            "paths_disclosed": False,
        }


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowWorkerResult:
    status: VisualPredicateShadowWorkerStatus
    sync: VisualPredicateShadowSyncSummary
    event_id: str | None = None
    attempt_count: int = 0
    error_code: str | None = None
    predicate_count: int = 0
    reused_existing_result: bool = False

    def __post_init__(self) -> None:
        progressed = self.status is not VisualPredicateShadowWorkerStatus.IDLE
        if progressed != bool(self.event_id):
            raise ValueError("progressed visual predicate shadow result requires event_id")
        if (
            self.status
            in {
                VisualPredicateShadowWorkerStatus.RETRY_PENDING,
                VisualPredicateShadowWorkerStatus.FAILED,
            }
        ) != bool(self.error_code):
            raise ValueError("failed visual predicate shadow result requires error_code")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": self.status.value,
            "event_id": self.event_id,
            "attempt_count": self.attempt_count,
            "error_code": self.error_code,
            "predicate_count": self.predicate_count,
            "reused_existing_result": self.reused_existing_result,
            "sync": self.sync.to_dict(),
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
            "paths_disclosed": False,
            "media_metadata_included": False,
        }


def _sync_analysis_jobs(
    *,
    database_path: str | Path,
    runtime: VisualPredicateShadowRuntimeSettings,
    max_active: int,
    statuses: tuple[str, ...],
    scan_limit: int = 1_000,
    now: datetime | None = None,
) -> VisualPredicateShadowSyncSummary:
    if not isinstance(runtime, VisualPredicateShadowRuntimeSettings) or not runtime.enabled:
        raise ValueError("visual predicate shadow is disabled")
    queue = VisualPredicateShadowAttemptQueue(database_path)
    queue.initialize()
    events, existing = queue.analysis_sync_batch(statuses=statuses, limit=scan_limit)
    analysis_queue = AnalysisQueue(database_path)
    selected = 0
    enqueued = 0
    skipped = 0
    queue_full = False
    scanned = {"pending": 0, "completed": 0}
    for event_id, source_status in events:
        scanned[source_status] += 1
        job = analysis_queue.get_read_only(event_id)
        if job is None or job.status.value != source_status:
            raise ValueError("visual predicate shadow analysis binding changed")
        gate = select_visual_predicate_shadow(
            job.candidate,
            predicates=runtime.predicates,
            person_detector_id=job.candidate.detector_id,
            hardware_signal_source_adapters=runtime.hardware_signal_adapters,
            periodic_sample_due=False,
            min_person_confidence=runtime.min_person_confidence,
        )
        if not gate.selected:
            result = queue.record_unselected(gate, now=now)
            skipped += int(not result.reused_existing)
            existing += int(result.reused_existing)
            continue
        selected += 1
        try:
            result = queue.enqueue(gate, max_active=max_active, now=now)
        except VisualPredicateShadowQueueFullError:
            queue_full = True
            break
        enqueued += int(not result.reused_existing)
        existing += int(result.reused_existing)
    return VisualPredicateShadowSyncSummary(
        scanned_completed_jobs=scanned["completed"],
        scanned_pending_jobs=scanned["pending"],
        selected_jobs=selected,
        enqueued_jobs=enqueued,
        skipped_jobs=skipped,
        existing_attempts=existing,
        queue_full=queue_full,
    )


def sync_completed_analysis_jobs(
    *,
    database_path: str | Path,
    runtime: VisualPredicateShadowRuntimeSettings,
    max_active: int,
    scan_limit: int = 1_000,
    now: datetime | None = None,
) -> VisualPredicateShadowSyncSummary:
    return _sync_analysis_jobs(
        database_path=database_path,
        runtime=runtime,
        max_active=max_active,
        statuses=(AnalysisJobStatus.COMPLETED.value,),
        scan_limit=scan_limit,
        now=now,
    )


def sync_shadow_eligible_analysis_jobs(
    *,
    database_path: str | Path,
    runtime: VisualPredicateShadowRuntimeSettings,
    max_active: int,
    scan_limit: int = 1_000,
    now: datetime | None = None,
) -> VisualPredicateShadowSyncSummary:
    """Sync pending evidence and completed detector jobs without mutating the main queue."""
    return _sync_analysis_jobs(
        database_path=database_path,
        runtime=runtime,
        max_active=max_active,
        statuses=(
            AnalysisJobStatus.PENDING.value,
            AnalysisJobStatus.COMPLETED.value,
        ),
        scan_limit=scan_limit,
        now=now,
    )


def run_visual_predicate_shadow_attempt_once(
    *,
    database_path: str | Path,
    runtime: VisualPredicateShadowRuntimeSettings,
    adapter: VisualPredicateAdapter,
    max_active: int = 256,
    lease_seconds: int = 300,
    max_attempts: int = 3,
    retry_delay_seconds: int = 30,
    scan_limit: int = 1_000,
    now: datetime | None = None,
) -> VisualPredicateShadowWorkerResult:
    if not isinstance(runtime, VisualPredicateShadowRuntimeSettings) or not runtime.enabled:
        raise ValueError("visual predicate shadow is disabled")
    if not isinstance(adapter, VisualPredicateAdapter):
        raise ValueError("visual predicate shadow adapter is invalid")
    sync = sync_shadow_eligible_analysis_jobs(
        database_path=database_path,
        runtime=runtime,
        max_active=max_active,
        scan_limit=scan_limit,
        now=now,
    )
    queue = VisualPredicateShadowAttemptQueue(database_path)
    attempt = queue.claim_next(
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
        now=now,
    )
    if attempt is None:
        return VisualPredicateShadowWorkerResult(
            status=VisualPredicateShadowWorkerStatus.IDLE,
            sync=sync,
        )

    try:
        job = AnalysisQueue(database_path).get_read_only(attempt.event_id)
        if job is None or job.status not in {
            AnalysisJobStatus.PENDING,
            AnalysisJobStatus.COMPLETED,
        }:
            raise RuntimeError
        job.require_local_media()
        frames = tuple(
            admit_private_shadow_frame(f"frame_{index:04d}={path}")
            for index, path in enumerate(job.media.frame_paths, start=1)
        )
        summary = run_visual_predicate_shadow(
            gate=attempt.gate,
            frames=frames,
            adapter=adapter,
            store_factory=lambda: VisualPredicateShadowStore(database_path),
            recorded_at=now or datetime.now(UTC),
        )
        completed = queue.mark_completed(attempt.event_id, now=now)
        return VisualPredicateShadowWorkerResult(
            status=VisualPredicateShadowWorkerStatus.COMPLETED,
            sync=sync,
            event_id=completed.event_id,
            attempt_count=completed.attempt_count,
            predicate_count=len(summary.results),
            reused_existing_result=summary.reused_existing,
        )
    except Exception:
        failed = queue.mark_failed_attempt(
            attempt.event_id,
            error_code="shadow_run_failed",
            retry_delay_seconds=retry_delay_seconds,
            max_attempts=max_attempts,
            now=now,
        )
        return VisualPredicateShadowWorkerResult(
            status=(
                VisualPredicateShadowWorkerStatus.RETRY_PENDING
                if failed.status is VisualPredicateShadowAttemptStatus.PENDING
                else VisualPredicateShadowWorkerStatus.FAILED
            ),
            sync=sync,
            event_id=failed.event_id,
            attempt_count=failed.attempt_count,
            error_code="shadow_run_failed",
        )


def close_visual_predicate_shadow_adapter(adapter: VisualPredicateAdapter) -> None:
    if isinstance(adapter, ResidentCommandVisualPredicateAdapter):
        adapter.close()
