import unittest
from datetime import UTC, datetime, timedelta

from baby_monitor.live_monitor import _subject_projection


class SubjectRefreshProjectionTests(unittest.TestCase):
    def setUp(self):
        self.previous = {
            "status": "analyzed",
            "frame_received_at": "old",
            "received_monotonic": 10.0,
            "frame_generation": 2,
            "completed_at": "old_done",
        }
        self.attempt = {
            **self.previous,
            "status": "analyzing",
            "received_monotonic": 20.0,
            "frame_received_at": "new",
            "completed_at": None,
        }
        self.claims = [{"age_claim": "baby", "head_center": [100, 200]}]
        self.completed = {"attempt": self.previous, "claims": self.claims}
        self.context = dict(
            now=22.0, source_phase="receiving", source_generation=2, monitor_phase="detecting"
        )

    def project(self):
        return _subject_projection(self.attempt, self.completed, None, **self.context)

    def test_preserves_original_age_and_timestamp_while_refreshing(self):
        result = self.project()
        self.assertEqual(result["subject_claims"], self.claims)
        evidence = result["subject_evidence"]
        self.assertEqual(evidence["frame_received_at"], "old")
        self.assertEqual(evidence["evidence_age_seconds"], 12.0)
        self.assertTrue(evidence["update_in_progress"])

    def test_expiry_does_not_slide_with_new_attempt(self):
        self.context["now"] = 40.001
        self.assertIsNone(self.project()["subject_claims"])

    def test_refresh_fallback_also_expires_on_wall_receipt_age(self):
        base = datetime(2026, 1, 1, tzinfo=UTC)
        self.previous["frame_received_at"] = base.isoformat()
        self.attempt["frame_received_at"] = (base + timedelta(seconds=30)).isoformat()
        self.context["now_utc"] = base + timedelta(seconds=31)
        result = self.project()
        self.assertIsNone(result["subject_claims"])
        self.assertEqual("analyzing", result["subject_evidence"]["status"])
        self.assertFalse(result["subject_evidence"]["update_in_progress"])

    def test_failure_does_not_fall_back(self):
        self.attempt["status"] = "analysis_failed"
        self.assertIsNone(self.project()["subject_claims"])

    def test_generations_must_match_both_frames(self):
        self.previous["frame_generation"] = 1
        self.assertIsNone(self.project()["subject_claims"])
        self.previous["frame_generation"] = 2
        self.attempt["frame_generation"] = 3
        self.assertIsNone(self.project()["subject_claims"])

    def test_stopped_or_disconnected_hides_previous(self):
        self.context["source_phase"] = "reconnecting"
        self.assertIsNone(self.project()["subject_claims"])
        self.context["source_phase"] = "receiving"
        self.context["monitor_phase"] = "stopped"
        self.assertIsNone(self.project()["subject_claims"])

    def test_first_attempt_has_no_fallback(self):
        self.completed = None
        self.assertIsNone(self.project()["subject_claims"])

    def test_new_success_replaces_previous_including_empty(self):
        self.attempt["status"] = "analyzed"
        result = _subject_projection(self.attempt, self.completed, [], **self.context)
        self.assertEqual(result["subject_claims"], [])
        self.assertEqual(result["subject_evidence"]["frame_received_at"], "new")
        self.assertFalse(result["subject_evidence"]["update_in_progress"])
