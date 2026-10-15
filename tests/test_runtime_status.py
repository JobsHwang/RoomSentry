from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.analysis_queue import AnalysisQueue, QueueCounts, QueueSnapshot
from baby_monitor.archive_queue import ArchiveQueue, ArchiveQueueCounts, ArchiveQueueSnapshot
from baby_monitor.cli import main
from baby_monitor.gpu_guard import GPUCapacity
from baby_monitor.media_health import MediaSourceHealth, MediaSourceState
from baby_monitor.notification import (
    NotificationOutbox,
    NotificationQueueCounts,
    NotificationQueueSnapshot,
)
from baby_monitor.runtime_status import (
    assess_archive_queue_health,
    assess_notification_queue_health,
    assess_queue_health,
)


class RuntimeStatusTests(unittest.TestCase):
    @staticmethod
    def _write_health(path: Path, *, running: bool = True) -> None:
        now = datetime.now(UTC)
        state = MediaSourceState.RUNNING if running else MediaSourceState.STOPPED
        MediaSourceHealth(
            camera_id="cam_01",
            source_ref="rtsp://127.0.0.1:8554/test/camera",
            state=state,
            observed_at=now,
            session_started_at=now - timedelta(seconds=2),
            session_ended_at=None if running else now,
            segments=1,
            coverage_seconds=1.0,
            gap_count=0,
            reconnect_count=0,
            capacity_reached=False,
            ffmpeg_exit_code=None,
            last_error_code=None,
        ).write_atomic(path)

    def test_runtime_status_is_operational_read_only_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-runtime-") as directory:
            root = Path(directory).resolve()
            storage = root / "storage"
            storage.mkdir()
            database = root / "new/state/events.sqlite3"
            init_output = io.StringIO()
            with redirect_stdout(init_output):
                init_code = main(["init-db", "--database", str(database)])
            self.assertEqual(0, init_code)
            self.assertEqual(
                [
                    "events",
                    "analysis_jobs",
                    "archive_jobs",
                    "remote_review_attempts",
                    "recognition_attempts",
                    "presence_facts",
                    "notification_jobs",
                    "event_policy_decisions",
                    "visual_predicate_shadow_attempts",
                ],
                json.loads(init_output.getvalue())["schemas"],
            )
            database_before = database.read_bytes()
            mtime_before = database.stat().st_mtime_ns
            health = root / "health.json"
            self._write_health(health)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "runtime-status",
                        "--health",
                        str(health),
                        "--database",
                        str(database),
                        "--storage-root",
                        str(storage),
                        "--min-free-gib",
                        "0.000001",
                    ]
                )
            payload = json.loads(output.getvalue())

            self.assertEqual(0, code)
            self.assertEqual("operational", payload["status"])
            self.assertTrue(payload["operational"])
            self.assertFalse(payload["attention_required"])
            self.assertEqual([], payload["reason_codes"])
            self.assertTrue(payload["queue"]["healthy"])
            self.assertTrue(payload["archive_queue"]["healthy"])
            self.assertTrue(payload["notification_queue"]["healthy"])
            self.assertFalse(payload["notification_queue"]["enabled"])
            self.assertTrue(payload["storage"]["sufficient"])
            self.assertIsNone(payload["analysis"]["admission_allowed"])
            self.assertFalse(payload["safety"]["remote_uploads_allowed"])
            self.assertFalse(payload["safety"]["notification_enabled"])
            self.assertFalse(payload["safety"]["external_notification_delivery_enabled"])
            self.assertFalse(payload["safety"]["retention_delete_enabled"])
            self.assertFalse(payload["paths_disclosed"])
            self.assertNotIn(str(root), output.getvalue())
            self.assertNotIn("rtsp://", output.getvalue())
            self.assertEqual(database_before, database.read_bytes())
            self.assertEqual(mtime_before, database.stat().st_mtime_ns)
            schema = json.loads(
                Path("schemas/runtime-status-v1.schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(schema["required"]), set(payload))
            self.assertEqual(
                set(schema["$defs"]["queue"]["required"]),
                set(payload["queue"]),
            )
            self.assertEqual(
                set(schema["$defs"]["analysis"]["required"]),
                set(payload["analysis"]),
            )
            self.assertEqual(
                set(schema["$defs"]["archiveQueue"]["required"]),
                set(payload["archive_queue"]),
            )
            self.assertEqual(
                set(schema["$defs"]["notificationQueue"]["required"]),
                set(payload["notification_queue"]),
            )

    def test_notification_backlog_is_attention_without_payload_or_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-runtime-") as directory:
            root = Path(directory).resolve()
            storage = root / "storage"
            storage.mkdir()
            database = root / "events.sqlite3"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(0, main(["init-db", "--database", str(database)]))
            health = root / "health.json"
            self._write_health(health)
            snapshot = NotificationQueueSnapshot(
                counts=NotificationQueueCounts(
                    pending=64,
                    running=0,
                    delivered=2,
                    failed=1,
                ),
                ready_pending=64,
                oldest_pending_age_seconds=3_601.0,
                failed_error_counts=(("fixture_failure", 1),),
            )

            output = io.StringIO()
            with (
                patch(
                    "baby_monitor.cli_commands.notification.NotificationOutbox.snapshot_read_only",
                    return_value=snapshot,
                ),
                redirect_stdout(output),
            ):
                code = main(
                    [
                        "runtime-status",
                        "--health",
                        str(health),
                        "--database",
                        str(database),
                        "--storage-root",
                        str(storage),
                        "--min-free-gib",
                        "0.000001",
                    ]
                )
            payload = json.loads(output.getvalue())

            self.assertEqual(1, code)
            self.assertEqual("attention", payload["status"])
            self.assertEqual(
                [
                    "notification_disabled_with_backlog",
                    "notification_queue_at_capacity",
                    "notification_failed_jobs",
                    "notification_pending_too_old",
                ],
                payload["reason_codes"],
            )
            self.assertEqual(
                payload["reason_codes"],
                payload["notification_queue"]["reason_codes"],
            )
            self.assertNotIn(str(root), output.getvalue())
            self.assertNotIn("fixture_failure", json.dumps(payload["reason_codes"]))

    def test_attention_combines_terminal_media_and_low_storage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-runtime-") as directory:
            root = Path(directory).resolve()
            storage = root / "storage"
            storage.mkdir()
            database = root / "events.sqlite3"
            AnalysisQueue(database).initialize()
            ArchiveQueue(database).initialize()
            NotificationOutbox(database).initialize()
            health = root / "health.json"
            self._write_health(health, running=False)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "runtime-status",
                        "--health",
                        str(health),
                        "--database",
                        str(database),
                        "--storage-root",
                        str(storage),
                        "--min-free-gib",
                        "900000",
                    ]
                )
            payload = json.loads(output.getvalue())

            self.assertEqual(1, code)
            self.assertEqual("attention", payload["status"])
            self.assertEqual(
                ["media_not_operational", "storage_low_free_space"],
                payload["reason_codes"],
            )

    def test_busy_guard_is_advisory_and_does_not_break_media_capture(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-runtime-") as directory:
            root = Path(directory).resolve()
            storage = root / "storage"
            storage.mkdir()
            database = root / "events.sqlite3"
            AnalysisQueue(database).initialize()
            ArchiveQueue(database).initialize()
            NotificationOutbox(database).initialize()
            health = root / "health.json"
            self._write_health(health)
            config = root / "config.toml"
            config.write_text(
                "[analysis_worker]\ngpu_guard_enabled = true\n",
                encoding="utf-8",
            )
            capacity = GPUCapacity(
                allowed=False,
                reason="gpu_busy",
                device_index=0,
                memory_total_mib=12_282,
                memory_used_mib=2_000,
                memory_free_mib=10_282,
                utilization_percent=90,
            )

            output = io.StringIO()
            with (
                patch(
                    "baby_monitor.cli_commands.analysis.probe_gpu_capacity", return_value=capacity
                ),
                redirect_stdout(output),
            ):
                code = main(
                    [
                        "runtime-status",
                        "--config",
                        str(config),
                        "--health",
                        str(health),
                        "--database",
                        str(database),
                        "--storage-root",
                        str(storage),
                        "--min-free-gib",
                        "0.000001",
                    ]
                )
            payload = json.loads(output.getvalue())

            self.assertEqual(0, code)
            self.assertTrue(payload["operational"])
            self.assertFalse(payload["analysis"]["admission_allowed"])
            self.assertEqual("gpu_busy", payload["analysis"]["reason_code"])
            self.assertEqual(["analysis_deferred"], payload["advisory_codes"])

    def test_missing_queue_database_returns_two_without_creating_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-runtime-") as directory:
            root = Path(directory).resolve()
            storage = root / "storage"
            storage.mkdir()
            database = root / "missing.sqlite3"
            health = root / "health.json"
            self._write_health(health)
            errors = io.StringIO()

            with redirect_stderr(errors):
                code = main(
                    [
                        "runtime-status",
                        "--health",
                        str(health),
                        "--database",
                        str(database),
                        "--storage-root",
                        str(storage),
                    ]
                )

            self.assertEqual(2, code)
            self.assertFalse(database.exists())
            self.assertIn("database is missing or invalid", errors.getvalue())
            self.assertNotIn(str(root), errors.getvalue())

    def test_missing_notification_schema_fails_read_only_without_migration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-runtime-") as directory:
            root = Path(directory).resolve()
            storage = root / "storage"
            storage.mkdir()
            database = root / "legacy.sqlite3"
            AnalysisQueue(database).initialize()
            ArchiveQueue(database).initialize()
            database_before = database.read_bytes()
            mtime_before = database.stat().st_mtime_ns
            health = root / "health.json"
            self._write_health(health)
            errors = io.StringIO()

            with redirect_stderr(errors):
                code = main(
                    [
                        "runtime-status",
                        "--health",
                        str(health),
                        "--database",
                        str(database),
                        "--storage-root",
                        str(storage),
                    ]
                )

            self.assertEqual(2, code)
            self.assertEqual(database_before, database.read_bytes())
            self.assertEqual(mtime_before, database.stat().st_mtime_ns)
            self.assertIn("notification database is missing or invalid", errors.getvalue())
            self.assertNotIn(str(root), errors.getvalue())

    def test_duplicate_camera_health_is_rejected_without_disclosing_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-runtime-") as directory:
            root = Path(directory).resolve()
            storage = root / "storage"
            storage.mkdir()
            database = root / "events.sqlite3"
            AnalysisQueue(database).initialize()
            ArchiveQueue(database).initialize()
            NotificationOutbox(database).initialize()
            health = root / "health.json"
            self._write_health(health)
            errors = io.StringIO()

            with redirect_stderr(errors):
                code = main(
                    [
                        "runtime-status",
                        "--health",
                        str(health),
                        "--health",
                        str(health),
                        "--database",
                        str(database),
                        "--storage-root",
                        str(storage),
                    ]
                )

            self.assertEqual(2, code)
            self.assertIn("camera_id values must be unique", errors.getvalue())
            self.assertNotIn(str(root), errors.getvalue())

    def test_queue_health_reports_all_backpressure_reasons(self) -> None:
        snapshot = QueueSnapshot(
            counts=QueueCounts(pending=2, running=0, completed=3, failed=1),
            ready_pending=2,
            oldest_pending_age_seconds=3_601.0,
            queued_media_files=2,
            queued_media_bytes=100,
            missing_media_files=1,
            failed_error_counts=(("model_timeout", 1),),
        )
        health = assess_queue_health(
            snapshot,
            max_active=2,
            max_pending_age_seconds=3_600,
        )
        self.assertFalse(health.healthy)
        self.assertEqual(
            (
                "queue_at_capacity",
                "queue_failed_jobs",
                "queue_missing_media",
                "queue_pending_too_old",
            ),
            health.reason_codes,
        )

    def test_archive_queue_health_reports_capacity_failure_and_old_backlog(self) -> None:
        snapshot = ArchiveQueueSnapshot(
            counts=ArchiveQueueCounts(pending=2, running=0, completed=3, failed=1),
            ready_pending=2,
            oldest_pending_age_seconds=86_401.0,
            pending_items=4,
            pending_bytes=1_024,
            failed_error_counts=(("archive_conflict", 1),),
        )
        health = assess_archive_queue_health(
            snapshot,
            max_active=2,
            max_pending_age_seconds=86_400,
        )
        self.assertFalse(health.healthy)
        self.assertEqual(
            (
                "archive_queue_at_capacity",
                "archive_failed_jobs",
                "archive_pending_too_old",
            ),
            health.reason_codes,
        )

    def test_notification_queue_health_reports_disabled_capacity_failure_and_age(self) -> None:
        snapshot = NotificationQueueSnapshot(
            counts=NotificationQueueCounts(
                pending=2,
                running=0,
                delivered=3,
                failed=1,
            ),
            ready_pending=2,
            oldest_pending_age_seconds=3_601.0,
            failed_error_counts=(("delivery_failed", 1),),
        )
        health = assess_notification_queue_health(
            snapshot,
            enabled=False,
            max_active=2,
            max_pending_age_seconds=3_600,
        )
        self.assertFalse(health.healthy)
        self.assertEqual(
            (
                "notification_disabled_with_backlog",
                "notification_queue_at_capacity",
                "notification_failed_jobs",
                "notification_pending_too_old",
            ),
            health.reason_codes,
        )


if __name__ == "__main__":
    unittest.main()
