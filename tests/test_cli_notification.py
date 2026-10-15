from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.adapters.vlm import MockVLMAdapter
from baby_monitor.cli import main
from baby_monitor.database import EventDatabase
from baby_monitor.domain import CandidateEvent, EventMedia
from baby_monitor.notification import NotificationMessage
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy


def _create_pending_event(root: Path) -> tuple[Path, str]:
    frame = root / "PRIVATE_CLI_FRAME.jpg"
    clip = root / "PRIVATE_CLI_CLIP.mp4"
    frame.write_bytes(b"fixture")
    clip.write_bytes(b"fixture")
    database_path = root / "events.sqlite3"
    event_id = "notification_cli_event"
    process_prepared_event(
        candidate=CandidateEvent(event_id, "camera_01", 0.0, 1.0, ()),
        occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
        source_kind="fixture",
        source_ref="PRIVATE_CLI_SOURCE",
        media=EventMedia(str(clip), (str(frame),)),
        local_vlm=MockVLMAdapter(),
        database=EventDatabase(database_path),
        policy=ReviewPolicy(),
    )
    return database_path, event_id


class NotificationCLITests(unittest.TestCase):
    def test_local_test_requires_confirmation_and_never_claims_event_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database, _event_id = _create_pending_event(root)
            database.chmod(0o600)
            config = root / "notification.toml"
            config.write_text("[notification]\nenabled = true\n", encoding="utf-8")
            output = root / "local-notifications"

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                refused = main(
                    [
                        "notification-local-test",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                        "--output-dir",
                        str(output),
                        "--confirmation",
                        "NO",
                    ]
                )
            self.assertEqual(2, refused)
            self.assertIn("TEST confirmation", stderr.getvalue())
            self.assertFalse(output.exists())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                delivered = main(
                    [
                        "notification-local-test",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                        "--output-dir",
                        str(output),
                        "--confirmation",
                        "TEST",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(0, delivered)
            self.assertEqual("delivered", payload["status"])
            self.assertTrue(payload["test_only"])
            self.assertFalse(payload["external_delivery_used"])
            self.assertFalse(payload["event_authority"])
            self.assertFalse(payload["risk_authority"])
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertEqual(1, len(list(output.glob("*.json"))))

    def test_default_disabled_refuses_before_creating_outbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, event_id = _create_pending_event(root)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "notification-enqueue-event",
                        "--database",
                        str(database),
                        "--event-id",
                        event_id,
                        "--sink",
                        "stdout",
                        "--sink-id",
                        "local_stdout",
                    ]
                )
            self.assertEqual(2, code)
            self.assertIn("disabled", stderr.getvalue())
            with closing(sqlite3.connect(database)) as connection, connection:
                table = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'notification_jobs'"
                ).fetchone()
            self.assertIsNone(table)

    def test_stdout_fixture_delivers_path_free_pending_event_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="PRIVATE_CLI_NOTIFY_ROOT_") as directory:
            root = Path(directory)
            database, event_id = _create_pending_event(root)
            config = root / "notification.toml"
            config.write_text(
                "\n".join(
                    [
                        "[notification]",
                        "enabled = true",
                        "max_pending = 4",
                        "lease_seconds = 30",
                        "retry_delay_seconds = 0",
                        "max_attempts = 2",
                    ]
                ),
                encoding="utf-8",
            )
            enqueue_output = io.StringIO()
            with redirect_stdout(enqueue_output):
                enqueue_code = main(
                    [
                        "notification-enqueue-event",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                        "--event-id",
                        event_id,
                        "--sink",
                        "stdout",
                        "--sink-id",
                        "local_stdout",
                    ]
                )
            self.assertEqual(0, enqueue_code)
            enqueued = json.loads(enqueue_output.getvalue())
            self.assertEqual("pending", enqueued["status"])
            self.assertEqual("pending_review", enqueued["review_status"])
            self.assertFalse(enqueued["contains_media_paths"])

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                run_code = main(
                    [
                        "notification-run-once",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                        "--sink",
                        "stdout",
                        "--sink-id",
                        "local_stdout",
                    ]
                )
            self.assertEqual(0, run_code)
            message = NotificationMessage.from_json(stdout.getvalue().strip())
            self.assertEqual("pending_review", message.review_status.value)
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertNotIn("PRIVATE_CLI_SOURCE", stdout.getvalue())
            delivery = json.loads(stderr.getvalue())
            self.assertEqual("delivered", delivery["delivery"]["status"])

            status_output = io.StringIO()
            with redirect_stdout(status_output):
                status_code = main(
                    [
                        "notification-queue-status",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                    ]
                )
            self.assertEqual(0, status_code)
            status = json.loads(status_output.getvalue())
            self.assertEqual(1, status["queue"]["delivered"])
            self.assertEqual(0, status["queue"]["active"])

            idle_output = io.StringIO()
            with redirect_stdout(idle_output):
                idle_code = main(
                    [
                        "notification-run-once",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                        "--sink",
                        "stdout",
                        "--sink-id",
                        "local_stdout",
                    ]
                )
            self.assertEqual(0, idle_code)
            self.assertEqual("idle", json.loads(idle_output.getvalue())["delivery"]["status"])

    def test_file_notification_worker_delivers_one_job_and_stops_at_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database, event_id = _create_pending_event(root)
            config = root / "notification.toml"
            config.write_text(
                "[notification]\nenabled = true\n",
                encoding="utf-8",
            )
            output = root / "notifications"
            output.mkdir()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    0,
                    main(
                        [
                            "notification-enqueue-event",
                            "--config",
                            str(config),
                            "--database",
                            str(database),
                            "--event-id",
                            event_id,
                            "--sink",
                            "file",
                            "--sink-id",
                            "shadow_local",
                        ]
                    ),
                )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "notification-worker",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                        "--sink",
                        "file",
                        "--sink-id",
                        "shadow_local",
                        "--output-dir",
                        str(output),
                        "--max-jobs",
                        "1",
                        "--idle-exit-seconds",
                        "5",
                        "--max-runtime-seconds",
                        "30",
                        "--poll-seconds",
                        "0.01",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(0, code)
            self.assertEqual("max_jobs", payload["worker"]["stop_reason"])
            self.assertEqual(1, payload["worker"]["completed_jobs"])
            delivered = list(output.glob("*.json"))
            self.assertEqual(1, len(delivered))
            message = NotificationMessage.from_json(delivered[0].read_text(encoding="utf-8"))
            self.assertEqual(event_id, message.event_id)


if __name__ == "__main__":
    unittest.main()
