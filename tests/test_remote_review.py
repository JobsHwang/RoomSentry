from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.remote_review import (
    RemoteAttemptStatus,
    RemoteBudgetError,
    RemoteReviewAuditStore,
)


class RemoteReviewAuditTests(unittest.TestCase):
    def test_reservation_and_completion_store_only_media_free_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "events.sqlite3"
            store = RemoteReviewAuditStore(database_path)
            store.initialize()
            requested_at = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)
            attempt_id = store.reserve(
                event_id="opaque_event_01",
                requested_at=requested_at,
                target_origin="http://127.0.0.1:18081",
                model="fixture-model",
                frame_count=3,
                frame_bytes=1234,
                max_requests_per_day=20,
                min_interval_seconds=0,
            )
            store.complete(
                attempt_id,
                status=RemoteAttemptStatus.SUCCEEDED,
                completed_at=requested_at + timedelta(milliseconds=25),
                duration_ms=25,
            )

            attempts = store.recent()
            self.assertEqual(1, len(attempts))
            self.assertEqual(RemoteAttemptStatus.SUCCEEDED, attempts[0].status)
            self.assertEqual(3, attempts[0].frame_count)
            self.assertEqual(1234, attempts[0].frame_bytes)
            public = attempts[0].to_public_dict()
            self.assertNotIn("frame_paths", public)
            self.assertNotIn("request_body", public)
            self.assertNotIn("api_key", public)

            with closing(sqlite3.connect(database_path)) as connection, connection:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(remote_review_attempts)"
                    ).fetchall()
                }
            self.assertNotIn("frame_paths", columns)
            self.assertNotIn("request_body", columns)
            self.assertNotIn("api_key", columns)

    def test_daily_budget_and_minimum_interval_are_atomic_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RemoteReviewAuditStore(Path(directory) / "events.sqlite3")
            store.initialize()
            now = datetime(2026, 8, 3, 1, 0, tzinfo=UTC)
            store.reserve(
                event_id="event_01",
                requested_at=now,
                target_origin="https://provider.example",
                model="fixture-model",
                frame_count=1,
                frame_bytes=10,
                max_requests_per_day=2,
                min_interval_seconds=0,
            )
            with self.assertRaisesRegex(RemoteBudgetError, "local call budget") as rate_limited:
                store.reserve(
                    event_id="event_02",
                    requested_at=now + timedelta(seconds=30),
                    target_origin="https://provider.example",
                    model="fixture-model",
                    frame_count=1,
                    frame_bytes=10,
                    max_requests_per_day=2,
                    min_interval_seconds=60,
                )
            self.assertEqual("rate_limited", rate_limited.exception.code)

            store.reserve(
                event_id="event_02",
                requested_at=now + timedelta(seconds=60),
                target_origin="https://provider.example",
                model="fixture-model",
                frame_count=1,
                frame_bytes=10,
                max_requests_per_day=2,
                min_interval_seconds=60,
            )
            with self.assertRaises(RemoteBudgetError) as exhausted:
                store.reserve(
                    event_id="event_03",
                    requested_at=now + timedelta(seconds=120),
                    target_origin="https://provider.example",
                    model="fixture-model",
                    frame_count=1,
                    frame_bytes=10,
                    max_requests_per_day=2,
                    min_interval_seconds=60,
                )
            self.assertEqual("daily_budget_exhausted", exhausted.exception.code)
            self.assertEqual(2, len(store.recent()))


if __name__ == "__main__":
    unittest.main()
