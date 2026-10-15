from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from baby_monitor.live_monitor import _review_evidence_snapshot


class LiveReviewFreshnessTests(unittest.TestCase):
    def snapshot(self, status="analyzed", *, now=20.0, received=10.0, generation=0, **kwargs):
        attempt = {
            "status": status,
            "frame_received_at": kwargs.get("stamp", "2026-01-01T00:00:00+00:00"),
            "received_monotonic": received,
            "frame_generation": 0,
            "completed_at": None if status == "analyzing" else "2026-01-01T00:00:10+00:00",
        }
        result = _review_evidence_snapshot(
            attempt,
            now=now,
            source_phase=kwargs.get("source_phase", "receiving"),
            source_generation=generation,
            monitor_phase=kwargs.get("monitor_phase", "detecting"),
            now_utc=kwargs.get("now_utc"),
        )
        self.assertFalse(result["model_output_verified"])
        self.assertFalse(result["live_baby_state_established"])
        self.assertNotIn("received_monotonic", result)
        return result

    def test_startup_does_not_import_a_previous_session_result(self):
        result = _review_evidence_snapshot(
            None, now=20, source_phase="receiving", source_generation=0, monitor_phase="detecting"
        )
        self.assertEqual("not_reviewed_this_session", result["status"])
        self.assertIsNone(result["evidence_age_seconds"])
        self.assertIsNone(result["attempt_status"])

    def test_age_is_receipt_age_not_time_since_completion(self):
        result = self.snapshot(now=80)
        self.assertEqual(70, result["evidence_age_seconds"])
        self.assertEqual("stale_completed_review", result["status"])

    def test_diagnostic_recency_boundary_and_ongoing_aging(self):
        self.assertEqual("recent_completed_review", self.snapshot(now=40)["status"])
        self.assertEqual("stale_completed_review", self.snapshot(now=40.001)["status"])
        self.assertEqual("stale_completed_review", self.snapshot(now=1000)["status"])

    def test_analyzing_is_not_completed_output(self):
        result = self.snapshot("analyzing")
        self.assertEqual("analyzing", result["status"])
        self.assertIsNone(result["completed_at"])

    def test_latest_failure_never_falls_back_to_previous_success(self):
        result = self.snapshot("analysis_failed", now=1000)
        self.assertEqual("latest_analysis_failed", result["status"])
        self.assertEqual("analysis_failed", result["attempt_status"])

    def test_unknown_attempt_does_not_get_a_recent_success_label(self):
        self.assertEqual("attempt_unavailable", self.snapshot("invalid")["status"])

    def test_reconnection_invalidates_success_and_inflight_attempt(self):
        for status in ("analyzed", "analyzing", "analysis_failed"):
            with self.subTest(status=status):
                self.assertEqual(
                    "previous_source_generation", self.snapshot(status, generation=1)["status"]
                )

    def test_missing_or_invalid_generation_fails_closed(self):
        for generation in (None, True, -1, "0"):
            with self.subTest(generation=generation):
                self.assertEqual(
                    "source_generation_unavailable", self.snapshot(generation=generation)["status"]
                )

    def test_clock_rollback_and_nonfinite_age_are_unknown(self):
        for now in (9, float("nan"), float("inf"), -float("inf")):
            with self.subTest(now=now):
                result = self.snapshot(now=now)
                self.assertEqual("receipt_clock_unavailable", result["status"])
                self.assertIsNone(result["evidence_age_seconds"])

    def test_wall_clock_catches_old_receipt_despite_young_elapsed_age(self):
        now_utc = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=31)
        result = self.snapshot(now_utc=now_utc)
        self.assertEqual("stale_completed_review", result["status"])
        self.assertEqual(31, result["evidence_age_seconds"])

    def test_elapsed_clock_catches_old_receipt_despite_utc_rollback(self):
        now_utc = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=5)
        result = self.snapshot(now=41, now_utc=now_utc)
        self.assertEqual("stale_completed_review", result["status"])
        self.assertEqual(31, result["evidence_age_seconds"])

    def test_dual_clock_recency_boundary_is_not_rounded_before_check(self):
        base = datetime(2026, 1, 1, tzinfo=UTC)
        self.assertEqual(
            "recent_completed_review", self.snapshot(now_utc=base + timedelta(seconds=30))["status"]
        )
        self.assertEqual(
            "stale_completed_review",
            self.snapshot(now_utc=base + timedelta(seconds=30.000001))["status"],
        )

    def test_invalid_receipt_utc_cannot_publish_current_results(self):
        for stamp in ("invalid", None, 123, "2026-01-01T00:00:00"):
            with self.subTest(stamp=stamp):
                result = self.snapshot(stamp=stamp, now_utc=datetime(2026, 1, 1, tzinfo=UTC))
                self.assertEqual("receipt_clock_unavailable", result["status"])
                self.assertIsNone(result["evidence_age_seconds"])

    def test_future_receipt_or_naive_reader_utc_fails_closed(self):
        for now_utc in (datetime(2025, 12, 31, tzinfo=UTC), datetime(2026, 1, 1), "invalid"):
            result = self.snapshot(now_utc=now_utc)
            self.assertEqual("receipt_clock_unavailable", result["status"])
            self.assertIsNone(result["evidence_age_seconds"])

    def test_utc_does_not_refresh_generation_or_failure(self):
        now_utc = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=5)
        self.assertEqual(
            "latest_analysis_failed", self.snapshot("analysis_failed", now_utc=now_utc)["status"]
        )
        self.assertEqual(
            "previous_source_generation", self.snapshot(generation=1, now_utc=now_utc)["status"]
        )

    def test_disconnected_or_stopped_monitor_never_looks_live(self):
        for phase in ("stopped", "reconnecting", "failed", None):
            with self.subTest(source_phase=phase):
                self.assertEqual(
                    "source_not_receiving", self.snapshot(source_phase=phase)["status"]
                )
        for phase in ("initializing_detector", "stopped", "detector_failed", "cleanup_failed"):
            with self.subTest(monitor_phase=phase):
                self.assertEqual(
                    "monitor_not_detecting", self.snapshot(monitor_phase=phase)["status"]
                )


if __name__ == "__main__":
    unittest.main()
