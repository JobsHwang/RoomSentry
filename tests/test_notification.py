from __future__ import annotations

import json
import os
import sqlite3
import stat
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.adapters.vlm import MockVLMAdapter
from baby_monitor.database import EventDatabase, StoredEvent
from baby_monitor.domain import CandidateEvent, EventMedia
from baby_monitor.notification import (
    FileNotificationSink,
    NotificationDeliveryError,
    NotificationJobStatus,
    NotificationMessage,
    NotificationOutbox,
    NotificationQueueError,
    ReviewStatus,
    deliver_notification_once,
)
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy


def _store_event(
    root: Path, *, event_id: str, pending: bool, secondary: bool = False
) -> tuple[EventDatabase, StoredEvent]:
    frame = root / f"PRIVATE_FRAME_{event_id}.jpg"
    clip = root / f"PRIVATE_CLIP_{event_id}.mp4"
    frame.write_bytes(b"fixture")
    clip.write_bytes(b"fixture")
    database = EventDatabase(root / "events.sqlite3")
    process_prepared_event(
        candidate=CandidateEvent(event_id, "camera_01", 0.0, 1.0, ()),
        occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
        source_kind="fixture",
        source_ref="PRIVATE_SOURCE_REF_DO_NOT_NOTIFY",
        media=EventMedia(str(clip), (str(frame),)),
        local_vlm=MockVLMAdapter(),
        secondary_vlm=MockVLMAdapter() if secondary else None,
        database=database,
        policy=ReviewPolicy(confidence_threshold=0.65 if pending else 0.0),
    )
    event = database.get_event(event_id)
    assert event is not None
    return database, event


class NotificationTests(unittest.TestCase):
    def test_pending_local_review_is_not_a_final_notification_or_remote_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, event = _store_event(Path(directory), event_id="pending_local", pending=False)
            message = NotificationMessage.from_event(event, sink_kind="stdout", sink_id="test")
            self.assertTrue(event.decision.needs_local_review)
            self.assertFalse(event.decision.needs_remote_review)
            self.assertFalse(event.result.needs_remote_review)
            self.assertFalse(event.remote_used)
            self.assertEqual(ReviewStatus.PENDING_REVIEW, message.review_status)
            self.assertEqual(event.result.risk_level, message.risk_level)
            self.assertEqual(event.result.confidence, message.confidence)
            self.assertIn("evidence_coverage_unverified", message.review_reasons)

    def test_completed_secondary_is_final_without_certifying_media_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, event = _store_event(
                Path(directory), event_id="completed_local", pending=False, secondary=True
            )
            message = NotificationMessage.from_event(event, sink_kind="stdout", sink_id="test")
            self.assertFalse(event.decision.needs_review)
            self.assertEqual(ReviewStatus.FINAL, message.review_status)
            self.assertFalse(message.remote_used)
            self.assertEqual("unverified", event.evidence.coverage_status)
            self.assertIn("evidence_coverage_unverified", message.review_reasons)

    def test_local_test_message_is_explicit_and_targeted_delivery_skips_older_jobs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "notifications"
            output.mkdir()
            outbox = NotificationOutbox(root / "events.sqlite3")
            started = datetime(2026, 8, 19, 1, 0, tzinfo=UTC)
            older = outbox.enqueue_local_test(
                test_id="1" * 32,
                sink_kind="file",
                sink_id="local_test",
                max_active=4,
                now=started,
            )
            target = outbox.enqueue_local_test(
                test_id="2" * 32,
                sink_kind="file",
                sink_id="local_test",
                max_active=4,
                now=started + timedelta(seconds=1),
            )

            attempt = deliver_notification_once(
                outbox=outbox,
                sink=FileNotificationSink(output),
                sink_id="local_test",
                lease_seconds=30,
                retry_delay_seconds=0,
                max_attempts=2,
                now=started + timedelta(seconds=2),
                notification_id=target.job.notification_id,
            )

            self.assertEqual("delivered", attempt.status)
            self.assertEqual(target.job.notification_id, attempt.notification_id)
            snapshot = outbox.snapshot_read_only(now=started + timedelta(seconds=2))
            self.assertEqual(1, snapshot.counts.pending)
            self.assertEqual(1, snapshot.counts.delivered)
            delivered = NotificationMessage.from_json(
                (output / f"{target.job.notification_id}.json").read_text(encoding="utf-8")
            )
            self.assertEqual("unknown", delivered.event_type.value)
            self.assertEqual("none", delivered.risk_level.value)
            self.assertEqual(("manual_local_notification_test",), delivered.observations)
            self.assertIn("不代表真实监控事件或风险", delivered.summary_zh)
            self.assertFalse((output / f"{older.job.notification_id}.json").exists())

    def test_message_contract_is_strict_and_contains_no_media_or_source_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="PRIVATE_NOTIFICATION_ROOT_") as directory:
            root = Path(directory)
            _, event = _store_event(root, event_id="event_pending", pending=True)
            message = NotificationMessage.from_event(
                event,
                sink_kind="stdout",
                sink_id="local_stdout",
            )
            self.assertEqual(ReviewStatus.PENDING_REVIEW, message.review_status)
            serialized = message.to_json(indent=None)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("PRIVATE_SOURCE_REF_DO_NOT_NOTIFY", serialized)
            self.assertNotIn("clip_path", serialized)
            self.assertNotIn("frame_paths", serialized)
            self.assertNotIn("evidence_frames", serialized)
            self.assertEqual(message, NotificationMessage.from_json(serialized))

            payload = message.to_dict()
            payload["unexpected"] = True
            with self.assertRaisesRegex(ValueError, "fields do not match"):
                NotificationMessage.from_dict(payload)

            schema = json.loads(
                Path("schemas/notification-message-v1.schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(message.to_dict()), set(schema["required"]))
            self.assertEqual(
                {value.value for value in ReviewStatus},
                set(schema["properties"]["review_status"]["enum"]),
            )
            with self.assertRaisesRegex(ValueError, "path or source URI"):
                replace(message, summary_zh="不得发送 rtsp://private.local/live")

    def test_outbox_is_idempotent_recovers_lease_and_requires_explicit_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, event = _store_event(root, event_id="event_queue", pending=False)
            outbox = NotificationOutbox(database.path)
            now = datetime(2026, 8, 3, 1, 0, tzinfo=UTC)
            first = outbox.enqueue(
                event=event,
                sink_kind="file",
                sink_id="local_file",
                max_active=4,
                now=now,
            )
            second = outbox.enqueue(
                event=event,
                sink_kind="file",
                sink_id="local_file",
                max_active=4,
                now=now,
            )
            self.assertFalse(first.reused_existing)
            self.assertTrue(second.reused_existing)
            self.assertEqual(first.job.notification_id, second.job.notification_id)

            claimed = outbox.claim_next(
                sink_kind="file",
                sink_id="local_file",
                lease_seconds=10,
                max_attempts=2,
                now=now,
            )
            assert claimed is not None
            self.assertEqual(1, claimed.attempt_count)
            recovered = outbox.claim_next(
                sink_kind="file",
                sink_id="local_file",
                lease_seconds=10,
                max_attempts=2,
                now=now + timedelta(seconds=11),
            )
            assert recovered is not None
            self.assertEqual(2, recovered.attempt_count)
            failed = outbox.mark_failed_attempt(
                recovered.notification_id,
                error_code="file_io_error",
                retry_delay_seconds=0,
                max_attempts=2,
                now=now + timedelta(seconds=11),
            )
            self.assertEqual(NotificationJobStatus.FAILED, failed.status)
            retried = outbox.retry_failed(
                failed.notification_id,
                max_active=4,
                now=now + timedelta(seconds=12),
            )
            self.assertEqual(NotificationJobStatus.PENDING, retried.status)
            self.assertEqual(0, retried.attempt_count)

            snapshot = outbox.snapshot_read_only(now=now + timedelta(seconds=12))
            self.assertEqual(1, snapshot.counts.pending)
            self.assertEqual(0, snapshot.counts.failed)

    def test_snapshot_rejects_unsafe_error_code_without_disclosing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, event = _store_event(root, event_id="event_unsafe_error", pending=False)
            outbox = NotificationOutbox(database.path)
            now = datetime(2026, 8, 3, 1, 0, tzinfo=UTC)
            outbox.enqueue(
                event=event,
                sink_kind="stdout",
                sink_id="local_stdout",
                max_active=4,
                now=now,
            )
            with closing(sqlite3.connect(database.path)) as connection, connection:
                connection.execute(
                    "UPDATE notification_jobs SET status = 'failed', last_error_code = ?",
                    ("C:\\private\\notification.txt",),
                )

            with self.assertRaisesRegex(
                NotificationQueueError,
                "database is missing or invalid",
            ) as raised:
                outbox.snapshot_read_only(now=now)
            self.assertNotIn("private", str(raised.exception).lower())

    def test_file_sink_is_atomic_idempotent_and_never_overwrites_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "notifications"
            output.mkdir()
            _, event = _store_event(root, event_id="event_file", pending=False)
            message = NotificationMessage.from_event(
                event,
                sink_kind="file",
                sink_id="local_file",
            )
            sink = FileNotificationSink(output.resolve())
            first = sink.deliver(message)
            second = sink.deliver(message)
            self.assertFalse(first.reused_existing)
            self.assertTrue(second.reused_existing)
            target = output / f"{message.notification_id}.json"
            if os.name != "nt":
                self.assertEqual(0o600, stat.S_IMODE(target.stat().st_mode))
            payload = NotificationMessage.from_json(target.read_text(encoding="utf-8"))
            self.assertEqual(message, payload)
            self.assertEqual([], list(output.glob("*.partial")))

            target.write_text("user-owned-conflict", encoding="utf-8")
            with self.assertRaisesRegex(NotificationDeliveryError, "delivery failed") as raised:
                sink.deliver(message)
            self.assertEqual("file_conflict", raised.exception.code)
            self.assertEqual("user-owned-conflict", target.read_text(encoding="utf-8"))

    def test_delivery_failure_is_bounded_then_explicit_retry_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, event = _store_event(root, event_id="event_retry", pending=True)
            outbox = NotificationOutbox(database.path)
            queued = outbox.enqueue(
                event=event,
                sink_kind="file",
                sink_id="local_file",
                max_active=4,
            )
            missing = root / "missing-output"
            first = deliver_notification_once(
                outbox=outbox,
                sink=FileNotificationSink(missing.resolve()),
                sink_id="local_file",
                lease_seconds=10,
                retry_delay_seconds=0,
                max_attempts=2,
            )
            second = deliver_notification_once(
                outbox=outbox,
                sink=FileNotificationSink(missing.resolve()),
                sink_id="local_file",
                lease_seconds=10,
                retry_delay_seconds=0,
                max_attempts=2,
            )
            self.assertEqual(NotificationJobStatus.PENDING.value, first.status)
            self.assertEqual(NotificationJobStatus.FAILED.value, second.status)
            self.assertEqual("file_target_unavailable", second.error_code)

            missing.mkdir()
            outbox.retry_failed(queued.job.notification_id, max_active=4)
            recovered = deliver_notification_once(
                outbox=outbox,
                sink=FileNotificationSink(missing.resolve()),
                sink_id="local_file",
                lease_seconds=10,
                retry_delay_seconds=0,
                max_attempts=2,
            )
            self.assertEqual(NotificationJobStatus.DELIVERED.value, recovered.status)
            self.assertEqual(1, len(list(missing.glob("*.json"))))


if __name__ == "__main__":
    unittest.main()
