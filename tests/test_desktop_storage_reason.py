import json
import unittest

from tests import test_live_desktop_backend as fixtures


class DesktopStorageReasonTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.LiveDesktopBackendTests()
        self.fixture.setUp()
        self.storage = self.fixture.status["review_storage"]
        self.storage.update(scan_complete=True, reserve_bytes=12, reason="available")

    def test_capacity_reserve_is_subtracted(self):
        self.assertEqual(self.fixture.snapshot()["storage_writable_budget_bytes"], 888)
        self.storage["used_bytes"] = 999
        self.assertEqual(self.fixture.snapshot()["storage_writable_budget_bytes"], 0)

    def test_incomplete_scan_has_no_exact_remaining_estimate(self):
        self.storage.update(reason="entry_limit", scan_complete=False)
        result = self.fixture.snapshot()
        self.assertEqual(result["storage_reason"], "entry_limit")
        self.assertIsNone(result["storage_writable_budget_bytes"])

    def test_all_fixed_reasons_preserved_and_raw_errors_hidden(self):
        for reason in (
            "available",
            "capacity_limit",
            "entry_limit",
            "disk_space_low",
            "unsafe_symlink",
            "io_error",
        ):
            self.storage["reason"] = reason
            self.assertEqual(self.fixture.snapshot()["storage_reason"], reason)
        self.storage["reason"] = "sensitive_error_path"
        result = self.fixture.snapshot()
        self.assertEqual(result["storage_reason"], "unknown")
        self.assertNotIn("sensitive_error_path", json.dumps(result))

    def test_invalid_or_missing_reserve_does_not_guess(self):
        for reserve in (None, True, -1, "12"):
            self.storage["reserve_bytes"] = reserve
            self.assertIsNone(self.fixture.snapshot()["storage_writable_budget_bytes"])

    def test_storage_pause_does_not_hide_yolo_counters(self):
        self.storage.update(available=False, reason="capacity_limit")
        self.fixture.status["review_phase"] = "storage_deferred"
        result = self.fixture.snapshot()
        self.assertEqual(result["state"], "degraded")
        self.assertEqual(result["detected_frames"], 40)
        self.assertTrue(all(row["receiving"] for row in result["cameras"]))

    def test_invalid_session_clears_old_storage_reason(self):
        self.storage["reason"] = "capacity_limit"
        self.fixture.status["worker_pid"] = 123
        self.assertEqual(self.fixture.snapshot()["storage_reason"], "unknown")
