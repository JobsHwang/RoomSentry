from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.home_control import _report_refresh_status, home_stack_status
from baby_monitor.home_dashboard import render_home_dashboard
from baby_monitor.report_queue import ReportRefreshQueue
from tests.test_home_control import FakeSystemctl


class ReportRefreshHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.database = self.root / "events.sqlite3"
        self.queue = ReportRefreshQueue(self.database)
        self.now = datetime.now(UTC)

    def request(self, *, age: int = 0) -> None:
        self.queue.request(
            "health-test",
            self.now.date(),
            "UTC",
            self.root,
            debounce_seconds=0,
            now=self.now - timedelta(seconds=age),
        )

    def test_missing_and_legacy_database_never_created_or_migrated(self) -> None:
        self.assertEqual("not_configured", _report_refresh_status(None)["status"])
        missing = _report_refresh_status(self.database)
        self.assertEqual("unavailable", missing["status"])
        self.assertTrue(missing["attention_required"])
        self.assertIsNone(missing["counts"])
        self.assertFalse(self.database.exists())
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE legacy (id INTEGER)")
        before = self.database.read_bytes()
        legacy = _report_refresh_status(self.database)
        self.assertEqual("not_initialized", legacy["status"])
        self.assertFalse(legacy["attention_required"])
        self.assertEqual(before, self.database.read_bytes())

    def test_ready_pending_does_not_immediately_warn_or_claim(self) -> None:
        self.request()
        before = self.database.read_bytes()
        result = home_stack_status(runner=FakeSystemctl(), analysis_database=self.database)
        health = result["report_refresh_queue"]
        self.assertEqual("available", health["status"])
        self.assertEqual(1, health["counts"]["pending"])
        self.assertEqual(1, health["ready"])
        self.assertFalse(health["attention_required"])
        self.assertFalse(health["worker_liveness_verified"])
        self.assertFalse(health["media_read"])
        self.assertEqual(before, self.database.read_bytes())
        serialized = json.dumps(health)
        for private in (str(self.root), "health-test", "report_", "unresolved_jobs"):
            self.assertNotIn(private, serialized)

    def test_overdue_pending_and_expired_lease_require_attention(self) -> None:
        self.request(age=3601)
        pending = _report_refresh_status(self.database)
        self.assertTrue(pending["attention_required"])
        claimed = self.queue.claim_next(lease_seconds=1, now=self.now - timedelta(seconds=2))
        self.assertIsNotNone(claimed)
        expired = _report_refresh_status(self.database)
        self.assertEqual(1, expired["expired_leases"])
        self.assertTrue(expired["attention_required"])
        self.assertEqual(1, expired["counts"]["running"])

    def test_failed_requires_attention_and_completion_clears_it(self) -> None:
        self.request()
        claimed = self.queue.claim_next(now=self.now)
        self.queue.fail(claimed, error_code="RenderFailed", max_attempts=1, now=self.now)
        failed = _report_refresh_status(self.database)
        self.assertTrue(failed["attention_required"])
        self.assertEqual(1, failed["counts"]["failed"])
        self.queue.retry_failed(claimed.job_id, now=self.now)
        retry = self.queue.claim_next(now=self.now)
        self.queue.complete(retry, now=self.now)
        completed = _report_refresh_status(self.database)
        self.assertFalse(completed["attention_required"])
        self.assertEqual(1, completed["counts"]["completed"])

    def test_active_lease_and_future_debounce_are_not_overdue(self) -> None:
        self.queue.request(
            "debounced-test",
            self.now.date(),
            "UTC",
            self.root,
            debounce_seconds=30,
            now=self.now,
        )
        pending = _report_refresh_status(self.database)
        self.assertEqual(0, pending["ready"])
        self.assertEqual(0.0, pending["oldest_pending_age_seconds"])
        self.assertFalse(pending["attention_required"])
        self.queue.claim_next(now=self.now + timedelta(seconds=31))
        running = _report_refresh_status(self.database)
        self.assertEqual(1, running["counts"]["running"])
        self.assertEqual(0, running["expired_leases"])
        self.assertFalse(running["attention_required"])

    def test_corruption_is_unavailable_without_exception_details(self) -> None:
        self.database.write_bytes(b"invalid sqlite with private detail")
        result = _report_refresh_status(self.database)
        self.assertEqual("unavailable", result["status"])
        self.assertFalse(result["operational"])
        self.assertTrue(result["attention_required"])
        self.assertNotIn("private", json.dumps(result))
        self.assertNotIn(str(self.database), json.dumps(result))

    def test_one_hour_boundary_excludes_debounce(self) -> None:
        self.request()
        snapshot = self.queue.snapshot_read_only(now=self.now + timedelta(seconds=3600))
        with patch.object(ReportRefreshQueue, "snapshot_read_only", return_value=snapshot):
            self.assertFalse(_report_refresh_status(self.database)["attention_required"])
        snapshot["oldest_pending_age_seconds"] = 3600.01
        with patch.object(ReportRefreshQueue, "snapshot_read_only", return_value=snapshot):
            self.assertTrue(_report_refresh_status(self.database)["attention_required"])
        # A future available_at (debounce or retry delay) is not overdue.
        snapshot["oldest_pending_age_seconds"] = 0.0
        with patch.object(ReportRefreshQueue, "snapshot_read_only", return_value=snapshot):
            self.assertFalse(_report_refresh_status(self.database)["attention_required"])

    def test_dashboard_warns_without_rendering_private_queue_fields(self) -> None:
        self.request(age=3601)
        health = _report_refresh_status(self.database)
        health["unresolved_jobs"] = [{"storage_root": "PRIVATE_SENTINEL"}]
        page = render_home_dashboard(
            {"base_operational": True, "report_refresh_queue": health}
        ).decode()
        self.assertIn("运行中 · 需处理", page)
        self.assertIn("详细日报刷新队列", page)
        self.assertIn("1 个待刷新日", page)
        self.assertIn("1.0 小时", page)
        self.assertNotIn("PRIVATE_SENTINEL", page)
        self.assertNotIn(str(self.root), page)

    def test_dashboard_distinguishes_unknown_from_empty_queue(self) -> None:
        for health, expected in (
            ({"status": "unavailable", "attention_required": True}, "状态不可用"),
            ({"status": "not_initialized"}, "尚未使用异步刷新"),
            ({}, "尚未使用异步刷新"),
        ):
            with self.subTest(expected=expected):
                page = render_home_dashboard({"report_refresh_queue": health}).decode()
                self.assertIn(expected, page)
                self.assertNotIn("0 个待刷新日", page)


if __name__ == "__main__":
    unittest.main()
