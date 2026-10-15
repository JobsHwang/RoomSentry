import unittest

from scripts.live_desktop_backend import baby_hint_summary
from tests import test_live_desktop_backend as fixtures


class BabyHintSummaryTests(unittest.TestCase):
    def camera(self, camera, ages):
        return {
            "camera_id": camera,
            "receiving": True,
            "current_subjects": {
                "age_claims": ages,
                "frame_received_at": "2026-01-01T00:00:00+00:00",
                "evidence_age_seconds": 8,
            },
        }

    def test_unavailable_is_not_absence(self):
        result = baby_hint_summary([])
        self.assertEqual(result["status"], "no_fresh_reviews")
        self.assertEqual(len(result["unavailable_cameras"]), 4)
        self.assertFalse(result["absence_established"])

    def test_single_hint_with_partial_coverage(self):
        result = baby_hint_summary([self.camera("camera_04", ["adult", "baby"])])
        self.assertEqual(result["status"], "single_camera_baby_hint")
        self.assertEqual(result["hints"][0]["camera_id"], "camera_04")
        self.assertFalse(result["coverage_complete"])
        self.assertFalse(result["identity_established"])
        self.assertFalse(result["room_established"])

    def test_two_cameras_do_not_choose_winner(self):
        result = baby_hint_summary([self.camera(c, ["baby"]) for c in ("camera_01", "camera_04")])
        self.assertEqual(result["status"], "ambiguous_baby_claims")
        self.assertEqual(len(result["hints"]), 2)

    def test_two_baby_claims_in_one_view_stay_ambiguous(self):
        result = baby_hint_summary([self.camera("camera_04", ["baby", "baby"])])
        self.assertEqual(result["status"], "ambiguous_baby_claims")
        self.assertEqual(result["multiple_baby_claim_cameras"], ["camera_04"])

    def test_full_empty_reviews_do_not_prove_absence(self):
        result = baby_hint_summary([self.camera(f"camera_{i:02}", []) for i in range(1, 5)])
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(result["status"], "no_baby_claim_in_fresh_reviews")
        self.assertFalse(result["absence_established"])

    def test_disconnected_camera_is_excluded(self):
        row = self.camera("camera_04", ["baby"])
        row["receiving"] = False
        self.assertEqual(baby_hint_summary([row])["hints"], [])

    def test_snapshot_early_returns_clear_summary(self):
        fixture = fixtures.LiveDesktopBackendTests()
        fixture.setUp()
        fixture.status["worker_pid"] = 123
        self.assertEqual(fixture.snapshot()["baby_hint_summary"]["hints"], [])

    def test_snapshot_expiration_and_generation_clear_hint(self):
        fixture = fixtures.LiveDesktopBackendTests()
        fixture.setUp()
        data = fixture.subject_fixture()
        self.assertEqual(
            fixture.snapshot()["baby_hint_summary"]["status"], "single_camera_baby_hint"
        )
        data["review_evidence"]["evidence_age_seconds"] = 31
        self.assertEqual(fixture.snapshot()["baby_hint_summary"]["hints"], [])
        data["review_evidence"]["evidence_age_seconds"] = 8
        data["generation"] = 3
        self.assertEqual(fixture.snapshot()["baby_hint_summary"]["hints"], [])

    def test_updating_flag_propagates_without_renewing_age(self):
        fixture = fixtures.LiveDesktopBackendTests()
        fixture.setUp()
        data = fixture.subject_fixture()
        data["review_evidence"]["update_in_progress"] = True
        hint = fixture.snapshot()["baby_hint_summary"]["hints"][0]
        self.assertTrue(hint["update_in_progress"])
        self.assertEqual(hint["evidence_age_seconds"], 8)
        data["review_evidence"]["update_in_progress"] = "true"
        self.assertEqual(fixture.snapshot()["baby_hint_summary"]["hints"], [])
