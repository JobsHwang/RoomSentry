from __future__ import annotations

import unittest
from datetime import UTC, datetime

from baby_monitor.candidate_health import (
    CandidateHealthLifetime,
    CandidateHealthState,
    CandidateWorkerHealth,
)
from baby_monitor.candidate_presence import (
    candidate_health_gap_sample,
    merge_stream_presence_batches,
)
from baby_monitor.room_presence import (
    RoomPresenceError,
    StreamPresenceBatch,
    StreamPresenceSampleState,
)


class CandidatePresenceTests(unittest.TestCase):
    NOW = datetime(2026, 8, 20, 6, 0, tzinfo=UTC)

    @classmethod
    def _health(
        cls,
        *,
        state: CandidateHealthState,
        running: bool,
    ) -> CandidateWorkerHealth:
        lifetime = CandidateHealthLifetime.begin_session(None, updated_at=cls.NOW)
        if not running:
            lifetime = lifetime.finish_session(
                updated_at=cls.NOW,
                failed=state is CandidateHealthState.FAILED,
            )
        return CandidateWorkerHealth(
            version=2,
            camera_id="camera_01",
            scanner_identity="candidate_v1_fixture",
            detector_identity="yolo_command_v1_fixture",
            checked_at=cls.NOW,
            state=state,
            worker_running=running,
            last_error_code=(
                "candidate_scan_failed" if state is CandidateHealthState.FAILED else None
            ),
            lifetime=lifetime,
        )

    def test_running_health_adds_no_presence_claim(self) -> None:
        for state in (
            CandidateHealthState.STARTING,
            CandidateHealthState.SCANNING,
            CandidateHealthState.IDLE,
        ):
            with self.subTest(state=state):
                self.assertIsNone(
                    candidate_health_gap_sample(self._health(state=state, running=True))
                )

    def test_stopped_or_failed_health_becomes_identity_free_coverage_gap(self) -> None:
        stopped = candidate_health_gap_sample(
            self._health(state=CandidateHealthState.STOPPED, running=False)
        )
        failed = candidate_health_gap_sample(
            self._health(state=CandidateHealthState.FAILED, running=False)
        )
        assert stopped is not None and failed is not None
        self.assertEqual(StreamPresenceSampleState.COVERAGE_GAP, stopped.state)
        self.assertEqual("candidate_worker_not_running", stopped.reason_code)
        self.assertEqual("candidate_worker_failed", failed.reason_code)
        self.assertIsNone(stopped.detector_id)
        self.assertEqual(64, len(stopped.source_fingerprint))

        merged = merge_stream_presence_batches(
            (StreamPresenceBatch(samples=()),),
            extra_samples=(stopped, failed),
        )
        self.assertEqual(2, len(merged.samples))

    def test_duplicate_health_fact_fails_closed_when_batches_merge(self) -> None:
        sample = candidate_health_gap_sample(
            self._health(state=CandidateHealthState.STOPPED, running=False)
        )
        assert sample is not None
        with self.assertRaisesRegex(RoomPresenceError, "IDs must be unique"):
            merge_stream_presence_batches(
                (StreamPresenceBatch(samples=(sample,)),),
                extra_samples=(sample,),
            )


if __name__ == "__main__":
    unittest.main()
