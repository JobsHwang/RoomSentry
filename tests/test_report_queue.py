from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from baby_monitor.report_queue import (
    ReportClaimLostError,
    ReportQueueError,
    ReportQueueFullError,
    ReportRefreshQueue,
)


class ReportRefreshQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-report-queue-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.database = self.root / "state" / "events.sqlite3"
        self.queue = ReportRefreshQueue(self.database)
        self.day = date(2026, 9, 8)
        self.now = datetime(2026, 9, 8, 1, tzinfo=UTC)

    def request(self, request_id="event:one", **overrides):
        values = dict(
            request_id=request_id,
            local_date=self.day,
            timezone_name="Asia/Shanghai",
            storage_root=self.root,
            debounce_seconds=0,
            now=self.now,
        )
        values.update(overrides)
        return self.queue.request(**values)

    def claim(self, **overrides):
        values = dict(now=self.now, lease_seconds=10, max_attempts=3)
        values.update(overrides)
        job = self.queue.claim_next(**values)
        self.assertIsNotNone(job)
        return job

    def count_requests(self) -> int:
        with sqlite3.connect(self.database) as connection:
            return connection.execute("SELECT COUNT(*) FROM report_refresh_requests").fetchone()[0]

    def test_missing_and_legacy_read_only_status_never_create_schema_or_files(self) -> None:
        job_id = "report_" + "a" * 64
        self.assertIsNone(self.queue.get_read_only(job_id))
        snapshot = self.queue.snapshot_read_only(now=self.now)
        self.assertFalse(snapshot["table_present"])
        self.assertEqual([], snapshot["unresolved_jobs"])
        self.assertEqual(20, snapshot["unresolved_jobs_limit"])
        self.assertEqual(0, snapshot["counts"]["total"])
        self.assertFalse(self.database.parent.exists())

        self.database.parent.mkdir()
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE existing_data (value TEXT)")
            connection.execute("INSERT INTO existing_data VALUES ('preserved')")
        before = self.database.read_bytes()
        self.assertIsNone(self.queue.get_read_only(job_id))
        legacy_snapshot = self.queue.snapshot_read_only()
        self.assertFalse(legacy_snapshot["table_present"])
        self.assertEqual([], legacy_snapshot["unresolved_jobs"])
        self.assertEqual(before, self.database.read_bytes())

    def test_request_id_is_global_idempotent_and_timezone_cannot_fork_destination(self) -> None:
        first = self.request()
        self.assertEqual(1, first.requested_revision)
        duplicate = self.request(now=self.now + timedelta(hours=1))
        self.assertEqual(first, duplicate)
        other_root = self.root / "other"
        other_root.mkdir()
        for changed in (
            {"storage_root": other_root},
            {"local_date": self.day + timedelta(days=1)},
            {"timezone_name": "UTC"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ReportQueueError):
                self.request(**changed)
        with self.assertRaises(ReportQueueError):
            self.request("new:request", timezone_name="UTC")
        self.assertEqual(first, self.queue.get_read_only(first.job_id))
        self.assertEqual(1, self.count_requests())

    def test_pending_requests_coalesce_without_extending_debounce(self) -> None:
        first = self.request(debounce_seconds=30)
        self.assertIsNone(self.queue.claim_next(now=self.now))
        merged = self.request(
            "event:two", debounce_seconds=300, now=self.now + timedelta(seconds=20)
        )
        self.assertEqual(first.job_id, merged.job_id)
        self.assertEqual(first.available_at, merged.available_at)
        self.assertEqual(2, merged.requested_revision)
        claimed = self.claim(now=self.now + timedelta(seconds=30))
        self.assertEqual(2, claimed.claimed_revision)
        self.assertEqual(1, claimed.attempt_count)

    def test_running_dirty_revision_survives_completion_as_fresh_pending_cycle(self) -> None:
        first = self.request()
        claimed = self.claim()
        self.request("event:two", now=self.now + timedelta(seconds=1))
        self.queue.assert_claim_current(claimed)
        completed = self.queue.complete(claimed, now=self.now + timedelta(seconds=2))
        self.assertEqual("pending", completed.status)
        self.assertEqual(2, completed.requested_revision)
        self.assertEqual(1, completed.completed_revision)
        self.assertEqual(0, completed.attempt_count)
        self.assertIsNone(completed.claim_token)
        with self.assertRaises(ReportClaimLostError):
            self.queue.complete(claimed, now=self.now + timedelta(seconds=3))
        second = self.claim(now=self.now + timedelta(seconds=2))
        self.assertEqual(2, second.claimed_revision)
        self.assertNotEqual(claimed.claim_token, second.claim_token)
        final = self.queue.complete(second, now=self.now + timedelta(seconds=3))
        self.assertEqual("completed", final.status)
        self.assertEqual(2, final.completed_revision)
        self.assertEqual(first.job_id, final.job_id)
        self.assertEqual(2, self.count_requests())

    def test_failures_are_bounded_and_failed_dirty_requests_require_explicit_retry(self) -> None:
        self.request()
        first = self.claim(max_attempts=2)
        pending = self.queue.fail(
            first, error_code="render_failed", retry_delay_seconds=5, max_attempts=2, now=self.now
        )
        self.assertEqual("pending", pending.status)
        self.assertIsNone(self.queue.claim_next(now=self.now + timedelta(seconds=4)))
        second = self.claim(max_attempts=2, now=self.now + timedelta(seconds=5))
        failed = self.queue.fail(
            second,
            error_code="render_failed",
            max_attempts=2,
            now=self.now + timedelta(seconds=6),
        )
        self.assertEqual("failed", failed.status)
        dirty = self.request("event:late", now=self.now + timedelta(seconds=7))
        self.assertEqual("failed", dirty.status)
        self.assertEqual(2, dirty.requested_revision)
        self.assertEqual(2, dirty.attempt_count)
        self.assertIsNone(self.queue.claim_next(now=self.now + timedelta(hours=1)))
        retried = self.queue.retry_failed(failed.job_id, now=self.now + timedelta(hours=1))
        self.assertEqual("pending", retried.status)
        self.assertEqual(0, retried.attempt_count)
        current = self.claim(now=self.now + timedelta(hours=1))
        self.assertEqual(2, current.claimed_revision)
        self.assertEqual(2, self.count_requests())

    def test_expiry_reclaim_fences_old_worker_completion_failure_and_guard(self) -> None:
        self.request()
        stale = self.claim()
        live = self.claim(now=self.now + timedelta(seconds=10))
        self.assertNotEqual(stale.claim_token, live.claim_token)
        self.assertEqual(stale.claimed_revision, live.claimed_revision)
        self.assertEqual(2, live.attempt_count)
        for operation in (
            lambda: self.queue.assert_claim_current(stale),
            lambda: self.queue.complete(stale, now=self.now + timedelta(seconds=11)),
            lambda: self.queue.fail(stale, error_code="stale_worker", now=self.now),
        ):
            with self.assertRaises(ReportClaimLostError):
                operation()
        self.assertEqual(live, self.queue.get_read_only(live.job_id))
        self.assertEqual("completed", self.queue.complete(live, now=self.now).status)

    def test_natural_expiry_without_reclaim_does_not_revoke_ownership(self) -> None:
        self.request()
        claimed = self.claim()
        self.queue.assert_claim_current(claimed)
        completed = self.queue.complete(claimed, now=self.now + timedelta(hours=1))
        self.assertEqual("completed", completed.status)

    def test_expired_final_attempt_becomes_failed_and_clears_token(self) -> None:
        self.request()
        claimed = self.claim(max_attempts=1)
        self.assertIsNone(
            self.queue.claim_next(max_attempts=1, now=self.now + timedelta(seconds=10))
        )
        failed = self.queue.get_read_only(claimed.job_id)
        self.assertEqual("failed", failed.status)
        self.assertIsNone(failed.claim_token)
        self.assertIsNone(failed.lease_until)
        self.assertEqual("lease_expired", failed.last_error_code)
        with self.assertRaises(ReportClaimLostError):
            self.queue.assert_claim_current(claimed)

    def test_capacity_does_not_block_coalescing_and_rejected_requests_do_not_consume_ids(
        self,
    ) -> None:
        first = self.request(max_active=1)
        coalesced = self.request("event:two", max_active=1)
        self.assertEqual(2, coalesced.requested_revision)
        with self.assertRaises(ReportQueueFullError):
            self.request("event:new_day", local_date=self.day + timedelta(days=1), max_active=1)
        self.assertEqual(2, self.count_requests())
        self.queue.complete(self.claim(), now=self.now)
        other = self.request("event:new_day", local_date=self.day + timedelta(days=1), max_active=1)
        with self.assertRaises(ReportQueueFullError):
            self.request("event:reactivate", max_active=1)
        self.assertEqual(3, self.count_requests())
        self.assertEqual("completed", self.queue.get_read_only(first.job_id).status)
        self.queue.complete(self.claim(), now=self.now)
        revived = self.request("event:reactivate", max_active=1)
        self.assertEqual("pending", revived.status)
        self.assertEqual(3, revived.requested_revision)
        self.assertNotEqual(other.job_id, revived.job_id)

    def test_retry_capacity_failure_leaves_failed_job_and_audit_unchanged(self) -> None:
        first = self.request()
        failed = self.queue.fail(self.claim(), error_code="failed", max_attempts=1, now=self.now)
        self.request("event:other", local_date=self.day + timedelta(days=1), max_active=1)
        with self.assertRaises(ReportQueueFullError):
            self.queue.retry_failed(first.job_id, max_active=1, now=self.now)
        self.assertEqual(failed, self.queue.get_read_only(first.job_id))
        self.assertEqual(2, self.count_requests())

    def test_concurrent_requests_merge_exactly_once_and_only_one_worker_claims(self) -> None:
        def request(index: int):
            return self.request("event:concurrent:" + str(index % 8))

        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(request, range(24)))
        self.assertEqual(1, len({job.job_id for job in jobs}))
        job_id = jobs[0].job_id
        self.assertEqual(8, self.queue.get_read_only(job_id).requested_revision)
        self.assertEqual(8, self.count_requests())
        with ThreadPoolExecutor(max_workers=8) as pool:
            claims = list(pool.map(lambda _: self.queue.claim_next(now=self.now), range(8)))
        owners = [job for job in claims if job is not None]
        self.assertEqual(1, len(owners))
        self.assertEqual(8, owners[0].claimed_revision)

    def test_forged_revision_or_destination_cannot_finalize_current_token(self) -> None:
        self.request()
        claimed = self.claim()
        for forged in (
            replace(claimed, claimed_revision=claimed.claimed_revision + 1),
            replace(claimed, storage_root=self.root / "elsewhere"),
            replace(claimed, timezone_name="UTC"),
        ):
            with self.subTest(forged=forged), self.assertRaises(ReportClaimLostError):
                self.queue.complete(forged, now=self.now)
        self.assertEqual(claimed, self.queue.get_read_only(claimed.job_id))

    def test_dst_fold_is_normalized_to_utc_without_changing_local_date_binding(self) -> None:
        timezone = ZoneInfo("America/New_York")
        first_time = datetime(2026, 11, 1, 1, 30, tzinfo=timezone, fold=0)
        second_time = first_time.replace(fold=1)
        first = self.request(
            local_date=date(2026, 11, 1), timezone_name=timezone.key, now=first_time
        )
        second = self.request(
            "event:fold", local_date=date(2026, 11, 1), timezone_name=timezone.key, now=second_time
        )
        self.assertEqual(first.job_id, second.job_id)
        self.assertEqual(date(2026, 11, 1), second.local_date)
        self.assertEqual(UTC, second.available_at.tzinfo)
        self.assertEqual(timedelta(hours=1), second.last_requested_at - first.last_requested_at)
        self.assertEqual(first.available_at, second.available_at)

    def test_invalid_types_paths_and_limits_fail_before_database_creation(self) -> None:
        linked = self.root / "linked"
        linked.symlink_to(self.root, target_is_directory=True)
        for changes in (
            {"request_id": ""},
            {"request_id": "private/request"},
            {"request_id": "a" * 257},
            {"local_date": self.now},
            {"local_date": "2026-09-08"},
            {"local_date": date.max},
            {"timezone_name": "missing/private-zone"},
            {"storage_root": Path("relative")},
            {"storage_root": Path("/")},
            {"storage_root": self.root / "missing"},
            {"storage_root": self.root / ".." / self.root.name},
            {"storage_root": linked},
            {"storage_root": 10},
            {"max_active": True},
            {"max_active": 0},
            {"debounce_seconds": float("nan")},
            {"debounce_seconds": float("inf")},
            {"debounce_seconds": 10**1000},
            {"now": datetime(2026, 1, 1)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ReportQueueError) as caught:
                self.request(**changes)
            self.assertNotIn(str(self.root), str(caught.exception))
            self.assertFalse(self.database.exists())

    def test_snapshot_and_job_receipts_are_path_token_and_request_free(self) -> None:
        job = self.request(request_id="PRIVATE_REQUEST", debounce_seconds=5)
        waiting = self.queue.snapshot_read_only(now=self.now)
        self.assertTrue(waiting["table_present"])
        self.assertEqual(0, waiting["ready"])
        self.assertEqual(1, waiting["dirty_jobs"])
        ready = self.queue.snapshot_read_only(now=self.now + timedelta(seconds=7))
        self.assertEqual(1, ready["ready"])
        self.assertEqual(2.0, ready["oldest_pending_age_seconds"])
        claimed = self.claim(now=self.now + timedelta(seconds=5))
        payload = json.dumps(claimed.to_public_dict()) + repr(claimed) + json.dumps(ready)
        for private in (str(self.root), claimed.claim_token, "PRIVATE_REQUEST"):
            self.assertNotIn(private, payload)
        failed = self.queue.fail(claimed, error_code="render_failed", max_attempts=1, now=self.now)
        snapshot = self.queue.snapshot_read_only(now=self.now)
        self.assertEqual([{"error_code": "render_failed", "count": 1}], snapshot["errors"])
        self.assertEqual(1, snapshot["counts"]["failed"])
        self.assertEqual(job.job_id, failed.job_id)

    def test_unresolved_snapshot_is_bounded_failed_first_private_and_read_only(self) -> None:
        pending = [
            self.request(
                "PRIVATE_REQUEST_" + str(index),
                local_date=self.day + timedelta(days=index),
                debounce_seconds=100,
            )
            for index in range(24)
        ]
        self.request(
            "PRIVATE_FAILED",
            local_date=self.day + timedelta(days=24),
            now=self.now + timedelta(seconds=50),
        )
        failed = self.queue.fail(
            self.claim(now=self.now + timedelta(seconds=50)),
            error_code="render_failed",
            max_attempts=1,
            now=self.now + timedelta(seconds=50),
        )
        self.request(
            "PRIVATE_RUNNING",
            local_date=self.day + timedelta(days=25),
            now=self.now + timedelta(seconds=60),
        )
        running = self.claim(now=self.now + timedelta(seconds=60))
        self.request("PRIVATE_COMPLETED", local_date=self.day + timedelta(days=26))
        completed = self.queue.complete(
            self.claim(now=self.now + timedelta(seconds=60)),
            now=self.now + timedelta(seconds=61),
        )
        before = self.database.read_bytes()
        before_files = set(self.database.parent.iterdir())

        snapshot = self.queue.snapshot_read_only(now=self.now + timedelta(seconds=65))

        self.assertEqual(before, self.database.read_bytes())
        self.assertEqual(before_files, set(self.database.parent.iterdir()))
        unresolved = snapshot["unresolved_jobs"]
        self.assertEqual(20, snapshot["unresolved_jobs_limit"])
        self.assertEqual(20, len(unresolved))
        self.assertEqual(failed.job_id, unresolved[0]["job_id"])
        self.assertEqual("failed", unresolved[0]["status"])
        self.assertEqual(running.job_id, unresolved[1]["job_id"])
        self.assertEqual(
            sorted(job.job_id for job in pending)[:18],
            [job["job_id"] for job in unresolved[2:]],
        )
        self.assertNotIn(completed.job_id, [job["job_id"] for job in unresolved])
        self.assertEqual(27, snapshot["counts"]["total"])
        self.assertEqual(24, snapshot["counts"]["pending"])
        self.assertEqual(1, snapshot["counts"]["failed"])
        public = json.dumps(snapshot)
        for private in (str(self.root), running.claim_token, "PRIVATE_"):
            self.assertNotIn(private, public)

    def test_frozen_utc_day_boundaries_preserve_23_and_25_hour_dst_days(self) -> None:
        for day, hours in ((date(2026, 3, 8), 23), (date(2026, 11, 1), 25)):
            with self.subTest(day=day):
                job = self.request(
                    "event:" + day.isoformat(),
                    local_date=day,
                    timezone_name="America/New_York",
                )
                self.assertEqual(timedelta(hours=hours), job.end_at - job.start_at)
                self.assertIs(UTC, job.start_at.tzinfo)
                self.assertIs(UTC, job.end_at.tzinfo)
                stored = self.queue.get_read_only(job.job_id)
                self.assertEqual(job.start_at, stored.start_at)
                self.assertEqual(job.end_at, stored.end_at)

    def test_changed_timezone_interpretation_cannot_silently_reuse_a_frozen_window(self) -> None:
        job = self.request()
        with (
            patch(
                "baby_monitor.report_queue._day_bounds",
                return_value=(job.start_at + timedelta(hours=1), job.end_at + timedelta(hours=1)),
            ),
            self.assertRaisesRegex(ReportQueueError, "frozen day boundary"),
        ):
            self.queue.get_read_only(job.job_id)
        self.assertEqual(job, self.queue.get_read_only(job.job_id))

    def test_storage_root_must_belong_to_current_user(self) -> None:
        with (
            patch("baby_monitor.report_queue.os.geteuid", return_value=os.geteuid() + 1),
            self.assertRaises(ReportQueueError),
        ):
            self.request()
        self.assertFalse(self.database.exists())

    def test_standalone_creation_is_private_even_with_permissive_umask(self) -> None:
        previous = os.umask(0)
        try:
            self.request()
        finally:
            os.umask(previous)
        self.assertEqual(0o600, self.database.stat().st_mode & 0o777)
        self.assertEqual(0o700, self.database.parent.stat().st_mode & 0o777)

    def test_existing_database_and_directory_permissions_are_not_rewritten(self) -> None:
        self.database.parent.mkdir()
        self.database.parent.chmod(0o755)
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE existing_data (value TEXT)")
        self.database.chmod(0o644)
        self.request()
        self.assertEqual(0o644, self.database.stat().st_mode & 0o777)
        self.assertEqual(0o755, self.database.parent.stat().st_mode & 0o777)

    def test_error_codes_are_sanitized_and_invalid_writes_preserve_claim(self) -> None:
        self.request()
        claimed = self.claim()
        with self.assertRaises(ReportQueueError) as caught:
            self.queue.fail(claimed, error_code=str(self.root), now=self.now)
        self.assertNotIn(str(self.root), str(caught.exception))
        self.assertEqual(claimed, self.queue.get_read_only(claimed.job_id))
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "UPDATE report_refresh_jobs SET last_error_code=?", (str(self.root),)
            )
        with self.assertRaises(ReportQueueError) as caught:
            self.queue.snapshot_read_only(now=self.now)
        self.assertNotIn(str(self.root), str(caught.exception))

    def test_partial_schema_is_not_silently_repaired_or_read_as_empty(self) -> None:
        self.database.parent.mkdir()
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE report_refresh_requests (request_id TEXT)")
        before = self.database.read_bytes()
        with self.assertRaises(ReportQueueError):
            self.queue.snapshot_read_only()
        with self.assertRaises(ReportQueueError):
            self.request()
        self.assertEqual(before, self.database.read_bytes())


if __name__ == "__main__":
    unittest.main()
