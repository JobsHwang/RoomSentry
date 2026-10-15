from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class WorkerAttemptState(StrEnum):
    IDLE = "idle"
    DEFERRED = "deferred"
    COMPLETED = "completed"
    RETRY_PENDING = "retry_pending"
    FAILED = "failed"


class WorkerStopReason(StrEnum):
    MAX_JOBS = "max_jobs"
    IDLE_TIMEOUT = "idle_timeout"
    MAX_RUNTIME = "max_runtime"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class WorkerAttempt:
    state: WorkerAttemptState
    event_id: str | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        progressed = self.state in {
            WorkerAttemptState.COMPLETED,
            WorkerAttemptState.RETRY_PENDING,
            WorkerAttemptState.FAILED,
        }
        if progressed != bool(self.event_id):
            raise ValueError("progressed worker attempts require an event_id")
        if self.error_code is not None and not self.error_code:
            raise ValueError("worker attempt error_code cannot be empty")


@dataclass(frozen=True, slots=True)
class WorkerLoopSettings:
    max_jobs: int = 32
    idle_exit_seconds: float = 30.0
    max_runtime_seconds: float = 3600.0
    poll_seconds: float = 2.0

    def __post_init__(self) -> None:
        if type(self.max_jobs) is not int or not 0 <= self.max_jobs <= 10_000:
            raise ValueError("max_jobs must be between 0 and 10000")
        for value, name, maximum in (
            (self.idle_exit_seconds, "idle_exit_seconds", 86_400.0),
            (self.max_runtime_seconds, "max_runtime_seconds", 604_800.0),
        ):
            if type(value) not in {int, float} or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be between 0 and {maximum:g}")
        if type(self.poll_seconds) not in {int, float} or not 0.01 <= self.poll_seconds <= 60:
            raise ValueError("poll_seconds must be between 0.01 and 60")
        if self.max_jobs == 0 and self.idle_exit_seconds == 0 and self.max_runtime_seconds == 0:
            raise ValueError("analysis worker must have at least one stopping bound")


@dataclass(frozen=True, slots=True)
class WorkerSummary:
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    stop_reason: WorkerStopReason
    attempted_jobs: int
    completed_jobs: int
    retry_pending_jobs: int
    failed_jobs: int
    deferred_checks: int
    idle_checks: int
    last_event_id: str | None
    last_error_code: str | None
    queue: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "ended_at": self.ended_at.astimezone(UTC).isoformat(),
            "duration_seconds": round(self.duration_seconds, 3),
            "stop_reason": self.stop_reason.value,
            "attempted_jobs": self.attempted_jobs,
            "completed_jobs": self.completed_jobs,
            "retry_pending_jobs": self.retry_pending_jobs,
            "failed_jobs": self.failed_jobs,
            "deferred_checks": self.deferred_checks,
            "idle_checks": self.idle_checks,
            "last_event_id": self.last_event_id,
            "last_error_code": self.last_error_code,
            "queue": self.queue,
        }


def run_analysis_worker(
    *,
    settings: WorkerLoopSettings,
    run_once: Callable[[], WorkerAttempt],
    queue_snapshot: Callable[[], dict[str, Any]],
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> WorkerSummary:
    started_at = wall_clock()
    started = monotonic()
    idle_started: float | None = None
    attempted = 0
    completed = 0
    retry_pending = 0
    failed = 0
    deferred = 0
    idle = 0
    last_event_id: str | None = None
    last_error_code: str | None = None
    stop_reason: WorkerStopReason

    while True:
        now = monotonic()
        if settings.max_jobs and attempted >= settings.max_jobs:
            stop_reason = WorkerStopReason.MAX_JOBS
            break
        if settings.max_runtime_seconds and now - started >= settings.max_runtime_seconds:
            stop_reason = WorkerStopReason.MAX_RUNTIME
            break
        if (
            idle_started is not None
            and settings.idle_exit_seconds
            and now - idle_started >= settings.idle_exit_seconds
        ):
            stop_reason = WorkerStopReason.IDLE_TIMEOUT
            break

        try:
            attempt = run_once()
        except KeyboardInterrupt:
            stop_reason = WorkerStopReason.INTERRUPTED
            break

        if attempt.state is WorkerAttemptState.IDLE:
            idle += 1
            if idle_started is None:
                idle_started = monotonic()
        elif attempt.state is WorkerAttemptState.DEFERRED:
            deferred += 1
            idle_started = None
        else:
            attempted += 1
            idle_started = None
            last_event_id = attempt.event_id
            if attempt.error_code is not None:
                last_error_code = attempt.error_code
            if attempt.state is WorkerAttemptState.COMPLETED:
                completed += 1
            elif attempt.state is WorkerAttemptState.RETRY_PENDING:
                retry_pending += 1
            else:
                failed += 1

        after_attempt = monotonic()
        if settings.max_jobs and attempted >= settings.max_jobs:
            continue
        if attempt.state is WorkerAttemptState.IDLE or attempt.state is WorkerAttemptState.DEFERRED:
            wait_for = settings.poll_seconds
            if settings.max_runtime_seconds:
                wait_for = min(
                    wait_for,
                    max(0.0, settings.max_runtime_seconds - (after_attempt - started)),
                )
            if attempt.state is WorkerAttemptState.IDLE and settings.idle_exit_seconds:
                assert idle_started is not None
                wait_for = min(
                    wait_for,
                    max(0.0, settings.idle_exit_seconds - (after_attempt - idle_started)),
                )
            try:
                sleeper(wait_for)
            except KeyboardInterrupt:
                stop_reason = WorkerStopReason.INTERRUPTED
                break

    ended_at = wall_clock()
    return WorkerSummary(
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=max(0.0, monotonic() - started),
        stop_reason=stop_reason,
        attempted_jobs=attempted,
        completed_jobs=completed,
        retry_pending_jobs=retry_pending,
        failed_jobs=failed,
        deferred_checks=deferred,
        idle_checks=idle,
        last_event_id=last_event_id,
        last_error_code=last_error_code,
        queue=queue_snapshot(),
    )
