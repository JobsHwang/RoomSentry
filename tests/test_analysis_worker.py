from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from baby_monitor.analysis_worker import (
    WorkerAttempt,
    WorkerAttemptState,
    WorkerLoopSettings,
    WorkerStopReason,
    run_analysis_worker,
)


class FakeTime:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.started_at = datetime(2026, 8, 3, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.seconds

    def sleep(self, seconds: float) -> None:
        self.seconds += seconds

    def wall_clock(self) -> datetime:
        return self.started_at + timedelta(seconds=self.seconds)


class AnalysisWorkerTests(unittest.TestCase):
    @staticmethod
    def _run(
        attempts: list[WorkerAttempt],
        settings: WorkerLoopSettings,
    ):
        clock = FakeTime()
        iterator = iter(attempts)
        summary = run_analysis_worker(
            settings=settings,
            run_once=lambda: next(iterator),
            queue_snapshot=lambda: {"pending": 0, "completed": 3},
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=clock.wall_clock,
        )
        return summary, clock

    def test_multiple_results_are_counted_and_stop_at_job_bound(self) -> None:
        summary, clock = self._run(
            [
                WorkerAttempt(WorkerAttemptState.COMPLETED, "event_01"),
                WorkerAttempt(
                    WorkerAttemptState.RETRY_PENDING,
                    "event_02",
                    "VLMError",
                ),
                WorkerAttempt(WorkerAttemptState.FAILED, "event_03", "MediaError"),
            ],
            WorkerLoopSettings(
                max_jobs=3,
                idle_exit_seconds=0,
                max_runtime_seconds=30,
                poll_seconds=1,
            ),
        )
        self.assertEqual(WorkerStopReason.MAX_JOBS, summary.stop_reason)
        self.assertEqual(3, summary.attempted_jobs)
        self.assertEqual(1, summary.completed_jobs)
        self.assertEqual(1, summary.retry_pending_jobs)
        self.assertEqual(1, summary.failed_jobs)
        self.assertEqual("event_03", summary.last_event_id)
        self.assertEqual("MediaError", summary.last_error_code)
        self.assertEqual(0.0, clock.seconds)
        self.assertEqual(3, summary.queue["completed"])

    def test_idle_timeout_uses_bounded_polling(self) -> None:
        clock = FakeTime()
        calls = 0

        def idle() -> WorkerAttempt:
            nonlocal calls
            calls += 1
            return WorkerAttempt(WorkerAttemptState.IDLE)

        summary = run_analysis_worker(
            settings=WorkerLoopSettings(
                max_jobs=0,
                idle_exit_seconds=5,
                max_runtime_seconds=30,
                poll_seconds=2,
            ),
            run_once=idle,
            queue_snapshot=dict,
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=clock.wall_clock,
        )
        self.assertEqual(WorkerStopReason.IDLE_TIMEOUT, summary.stop_reason)
        self.assertEqual(3, calls)
        self.assertEqual(3, summary.idle_checks)
        self.assertEqual(5.0, summary.duration_seconds)

    def test_gpu_defer_does_not_count_as_idle_or_attempted_job(self) -> None:
        clock = FakeTime()

        summary = run_analysis_worker(
            settings=WorkerLoopSettings(
                max_jobs=0,
                idle_exit_seconds=1,
                max_runtime_seconds=6,
                poll_seconds=2,
            ),
            run_once=lambda: WorkerAttempt(WorkerAttemptState.DEFERRED),
            queue_snapshot=lambda: {"pending": 1},
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=clock.wall_clock,
        )
        self.assertEqual(WorkerStopReason.MAX_RUNTIME, summary.stop_reason)
        self.assertEqual(3, summary.deferred_checks)
        self.assertEqual(0, summary.idle_checks)
        self.assertEqual(0, summary.attempted_jobs)
        self.assertEqual(1, summary.queue["pending"])

    def test_unbounded_configuration_and_invalid_attempt_identity_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "stopping bound"):
            WorkerLoopSettings(
                max_jobs=0,
                idle_exit_seconds=0,
                max_runtime_seconds=0,
            )
        with self.assertRaisesRegex(ValueError, "event_id"):
            WorkerAttempt(WorkerAttemptState.COMPLETED)
        with self.assertRaisesRegex(ValueError, "event_id"):
            WorkerAttempt(WorkerAttemptState.IDLE, "unexpected")

    def test_keyboard_interrupt_returns_a_health_summary(self) -> None:
        clock = FakeTime()

        def interrupted() -> WorkerAttempt:
            raise KeyboardInterrupt

        summary = run_analysis_worker(
            settings=WorkerLoopSettings(max_jobs=1),
            run_once=interrupted,
            queue_snapshot=lambda: {"running": 1},
            monotonic=clock.monotonic,
            sleeper=clock.sleep,
            wall_clock=clock.wall_clock,
        )
        self.assertEqual(WorkerStopReason.INTERRUPTED, summary.stop_reason)
        self.assertEqual(0, summary.attempted_jobs)
        self.assertEqual(1, summary.queue["running"])


if __name__ == "__main__":
    unittest.main()
