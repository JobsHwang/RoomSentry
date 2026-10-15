"""Metadata-only report outbox dispatch; no model, media or notification work."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from baby_monitor.local_review import LocalReviewError, LocalReviewStore
from baby_monitor.local_review_reports import dispatch_local_review_reports
from baby_monitor.report_queue import ReportQueueError, ReportQueueFullError, ReportRefreshQueue
from tests import test_local_review_store as store_fixtures


class LocalReviewReportDispatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "events.sqlite3"
        self.day = date(2026, 9, 12)

    def _intent(self, request_id="review_1", *, root=None, timezone="UTC"):
        return SimpleNamespace(
            request_id=request_id,
            local_date=self.day,
            timezone_name=timezone,
            storage_root=root or self.root,
        )

    def _rows(self, table):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute("SELECT * FROM " + table)]

    def _store(self, *intents):
        store = Mock()
        store.pending_report_refreshes.return_value = tuple(intents)
        return store

    def test_actual_queue_same_day_reviews_coalesce_and_receipt_is_path_free(self):
        store = self._store(self._intent(), self._intent("review_2"))
        with (
            patch("baby_monitor.local_review_reports.LocalReviewStore", return_value=store),
            patch("subprocess.Popen", side_effect=AssertionError("no model or renderer")),
        ):
            receipt = dispatch_local_review_reports(self.path)
        self.assertEqual({"scanned": 2, "enqueued": 2, "pending": 0, "errors_count": 0}, receipt)
        jobs = self._rows("report_refresh_jobs")
        self.assertEqual(1, len(jobs))
        self.assertEqual(2, jobs[0]["requested_revision"])
        self.assertEqual(2, len(self._rows("report_refresh_requests")))
        self.assertEqual(2, store.mark_report_enqueued.call_count)
        self.assertNotIn(str(self.root), json.dumps(receipt))
        self.assertNotIn("review_1", json.dumps(receipt))

    def test_ack_failure_replay_reuses_exact_request_and_revision(self):
        store = self._store(self._intent())
        store.mark_report_enqueued.side_effect = [LocalReviewError("database_unavailable"), None]
        store.mark_report_failed.side_effect = LocalReviewError("database_unavailable")
        with patch("baby_monitor.local_review_reports.LocalReviewStore", return_value=store):
            first = dispatch_local_review_reports(self.path)
            second = dispatch_local_review_reports(self.path)
        self.assertEqual({"scanned": 1, "enqueued": 0, "pending": 1, "errors_count": 1}, first)
        self.assertEqual({"scanned": 1, "enqueued": 1, "pending": 0, "errors_count": 0}, second)
        requests = self._rows("report_refresh_requests")
        self.assertEqual(1, len(requests))
        self.assertEqual(
            "local-review:" + hashlib.sha256(b"review_1").hexdigest(), requests[0]["request_id"]
        )
        self.assertEqual(1, self._rows("report_refresh_jobs")[0]["requested_revision"])
        store.mark_report_failed.assert_called_once_with(
            "review_1", error_code="report_enqueue_failed", now=None
        )

    def test_queue_full_and_private_failure_detail_are_fixed_and_next_intent_continues(self):
        store = self._store(self._intent(), self._intent("review_2"))
        queue = Mock()
        queue.request.side_effect = [
            ReportQueueFullError("PRIVATE_FAILURE " + str(self.root)),
            SimpleNamespace(job_id="job"),
        ]
        with (
            patch("baby_monitor.local_review_reports.LocalReviewStore", return_value=store),
            patch("baby_monitor.local_review_reports.ReportRefreshQueue", return_value=queue),
        ):
            receipt = dispatch_local_review_reports(self.path)
        self.assertEqual({"scanned": 2, "enqueued": 1, "pending": 1, "errors_count": 1}, receipt)
        store.mark_report_failed.assert_called_once_with(
            "review_1", error_code="report_enqueue_failed", now=None
        )
        store.mark_report_enqueued.assert_called_once_with("review_2", job_id="job", now=None)
        self.assertNotIn("PRIVATE_FAILURE", json.dumps(receipt))
        self.assertNotIn(str(self.root), json.dumps(receipt))

    def test_invalid_root_and_timezone_conflict_remain_pending(self):
        queue = ReportRefreshQueue(self.path)
        queue.request("original", self.day, "UTC", self.root)
        store = self._store(
            self._intent(root=self.root / "missing"),
            self._intent("review_2", timezone="Asia/Shanghai"),
        )
        with patch("baby_monitor.local_review_reports.LocalReviewStore", return_value=store):
            receipt = dispatch_local_review_reports(self.path)
        self.assertEqual({"scanned": 2, "enqueued": 0, "pending": 2, "errors_count": 2}, receipt)
        self.assertEqual(1, len(self._rows("report_refresh_requests")))
        self.assertEqual(1, self._rows("report_refresh_jobs")[0]["requested_revision"])
        store.mark_report_enqueued.assert_not_called()
        self.assertEqual(2, store.mark_report_failed.call_count)

    def test_limit_is_strict_before_store_access_and_filter_is_forwarded(self):
        with patch("baby_monitor.local_review_reports.LocalReviewStore") as factory:
            for limit in (True, False, 0, 101, -1, 1.5, None, "20"):
                with self.subTest(limit=limit), self.assertRaises(LocalReviewError) as raised:
                    dispatch_local_review_reports(self.path, limit=limit)
                self.assertEqual("invalid_input", raised.exception.code)
            factory.assert_not_called()
            factory.return_value.pending_report_refreshes.return_value = ()
            receipt = dispatch_local_review_reports(self.path, limit=1, request_id="review_1")
            factory.return_value.pending_report_refreshes.assert_called_once_with(
                limit=1, request_id="review_1"
            )
        self.assertEqual({"scanned": 0, "enqueued": 0, "pending": 0, "errors_count": 0}, receipt)
        self.assertFalse(self.path.exists())

    def test_read_failure_is_not_successful_empty_dispatch(self):
        store = self._store()
        store.pending_report_refreshes.side_effect = LocalReviewError("database_unavailable")
        with (
            patch("baby_monitor.local_review_reports.LocalReviewStore", return_value=store),
            patch("baby_monitor.local_review_reports.ReportRefreshQueue") as queue,
            self.assertRaises(LocalReviewError) as raised,
        ):
            dispatch_local_review_reports(self.path)
        self.assertEqual("database_unavailable", raised.exception.code)
        queue.assert_not_called()

    def test_supplied_clock_reaches_queue_and_ack(self):
        observed = datetime(2026, 9, 12, tzinfo=UTC)
        store = self._store(self._intent())
        with patch("baby_monitor.local_review_reports.LocalReviewStore", return_value=store):
            dispatch_local_review_reports(self.path, now=observed)
        job = self._rows("report_refresh_jobs")[0]
        self.assertEqual(observed.isoformat(), job["available_at"])
        store.mark_report_enqueued.assert_called_once_with(
            "review_1", job_id=job["job_id"], now=observed
        )

    def test_invalid_clock_rejected_before_empty_store_read(self):
        with patch("baby_monitor.local_review_reports.LocalReviewStore") as factory:
            for observed in (True, 0, "2026-09-12", datetime(2026, 9, 12)):
                with self.subTest(now=observed), self.assertRaises(LocalReviewError) as raised:
                    dispatch_local_review_reports(self.path, now=observed)
                self.assertEqual("invalid_input", raised.exception.code)
            factory.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_failure_ack_receives_supplied_clock(self):
        observed = datetime(2026, 9, 12, tzinfo=UTC)
        store = self._store(self._intent())
        with (
            patch("baby_monitor.local_review_reports.LocalReviewStore", return_value=store),
            patch.object(ReportRefreshQueue, "request", side_effect=ReportQueueError("controlled")),
        ):
            receipt = dispatch_local_review_reports(self.path, now=observed)
        self.assertEqual(1, receipt["pending"])
        store.mark_report_failed.assert_called_once_with(
            "review_1", error_code="report_enqueue_failed", now=observed
        )

    def test_unexpected_enqueue_or_failure_ack_exceptions_are_not_swallowed(self):
        for failure in (KeyboardInterrupt(), SystemExit(4), RuntimeError("synthetic unexpected")):
            for during_failure_ack in (False, True):
                store = self._store(self._intent())
                queue = Mock()
                queue.request.side_effect = (
                    ReportQueueError("controlled") if during_failure_ack else failure
                )
                if during_failure_ack:
                    store.mark_report_failed.side_effect = failure
                with (
                    self.subTest(failure=type(failure).__name__, ack=during_failure_ack),
                    patch("baby_monitor.local_review_reports.LocalReviewStore", return_value=store),
                    patch(
                        "baby_monitor.local_review_reports.ReportRefreshQueue", return_value=queue
                    ),
                    self.assertRaises(type(failure)) as raised,
                ):
                    dispatch_local_review_reports(self.path)
                self.assertIs(failure, raised.exception)
                store.mark_report_enqueued.assert_not_called()


class LocalReviewReportStoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Reuse metadata-only fixture setup without inheriting its test methods.
        self.fixture = store_fixtures.LocalReviewStoreTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.path = self.fixture.path
        self.storage_root = self.path.parent / "reports"
        self.storage_root.mkdir(mode=0o700)

    def _complete(self, request_id="report_review_1", event_id="event_1"):
        self.fixture.store.reserve(
            event_id,
            request_id,
            now=self.fixture.now,
            report_storage_root=self.storage_root,
            report_timezone="UTC",
        )
        self.fixture._complete(request_id)

    def test_missing_database_and_legacy_database_do_not_create_or_migrate(self):
        missing = self.path.parent / "missing" / "events.sqlite3"
        self.assertEqual(0, dispatch_local_review_reports(missing)["scanned"])
        self.assertFalse(missing.parent.exists())
        before = self.path.read_bytes()
        self.assertEqual(0, dispatch_local_review_reports(self.path)["scanned"])
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(
            [],
            self.fixture._sql("SELECT name FROM sqlite_master WHERE name LIKE 'report_refresh_%'"),
        )

    def test_success_ack_is_durable_does_not_modify_review_result_or_rerun_work(self):
        self._complete()
        before_event = self.fixture._event()
        before_history = self.fixture._sql("SELECT * FROM local_review_attempts")
        observed = self.fixture.now + timedelta(seconds=2)
        with patch("subprocess.Popen", side_effect=AssertionError("no model or renderer")):
            receipt = dispatch_local_review_reports(self.path, now=observed)
            replay = dispatch_local_review_reports(self.path, now=observed)
        self.assertEqual({"scanned": 1, "enqueued": 1, "pending": 0, "errors_count": 0}, receipt)
        self.assertEqual(0, replay["scanned"])
        self.assertEqual(before_event, self.fixture._event())
        self.assertEqual(before_history, self.fixture._sql("SELECT * FROM local_review_attempts"))
        self.assertEqual((), self.fixture.store.pending_report_refreshes())

    def test_actual_corrupt_database_failure_is_fixed_and_nonmutating(self):
        damaged = self.path.parent / "damaged.sqlite3"
        payload = b"PRIVATE_SYNTHETIC_DATABASE_FAILURE_NOT_SQLITE"
        damaged.write_bytes(payload)
        damaged.chmod(0o600)
        with self.assertRaises(LocalReviewError) as raised:
            dispatch_local_review_reports(damaged)
        self.assertEqual("database_unavailable", raised.exception.code)
        self.assertNotIn(str(damaged), str(raised.exception))
        self.assertNotIn("PRIVATE_SYNTHETIC", str(raised.exception))
        self.assertEqual(payload, damaged.read_bytes())

    def test_queue_full_then_missing_target_preserves_success_and_pending_intent(self):
        self._complete()
        before = self.fixture._event()
        with patch.object(
            ReportRefreshQueue, "request", side_effect=ReportQueueFullError("PRIVATE_FAILURE")
        ):
            first = dispatch_local_review_reports(self.path)
        self.storage_root.rmdir()
        second = dispatch_local_review_reports(self.path)
        self.assertEqual({"scanned": 1, "enqueued": 0, "pending": 1, "errors_count": 1}, first)
        self.assertEqual(first, second)
        self.assertEqual(before, self.fixture._event())
        self.assertEqual("succeeded", self.fixture.store.recent()[0]["status"])
        self.assertEqual(1, len(self.fixture.store.pending_report_refreshes()))
        self.assertNotIn("PRIVATE_FAILURE", json.dumps(first))

    def test_real_ack_failure_recovery_retains_single_queue_revision(self):
        self._complete()
        with patch.object(
            LocalReviewStore,
            "mark_report_enqueued",
            side_effect=LocalReviewError("database_unavailable"),
        ):
            first = dispatch_local_review_reports(self.path)
        second = dispatch_local_review_reports(self.path)
        self.assertEqual(1, first["pending"])
        self.assertEqual(1, second["enqueued"])
        self.assertEqual((), self.fixture.store.pending_report_refreshes())
        self.assertEqual(1, len(self.fixture._sql("SELECT * FROM report_refresh_requests")))
        self.assertEqual(
            1, self.fixture._sql("SELECT * FROM report_refresh_jobs")[0]["requested_revision"]
        )

    def test_actual_outbox_limit_keeps_remainder_then_coalesces_same_day(self):
        self._complete()
        self.fixture._insert("event_2")
        self._complete("report_review_2", "event_2")
        first = dispatch_local_review_reports(self.path, limit=1)
        self.assertEqual(1, first["scanned"])
        self.assertEqual(1, len(self.fixture.store.pending_report_refreshes()))
        second = dispatch_local_review_reports(self.path, request_id="report_review_2")
        self.assertEqual(1, second["enqueued"])
        jobs = self.fixture._sql("SELECT * FROM report_refresh_jobs")
        self.assertEqual(1, len(jobs))
        self.assertEqual(2, jobs[0]["requested_revision"])


if __name__ == "__main__":
    unittest.main()
