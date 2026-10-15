from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.archive import ArchiveManifest, build_archive_manifest
from baby_monitor.archive_queue import (
    ArchiveJobStatus,
    ArchiveQueue,
    ArchiveQueueError,
    ArchiveQueueFullError,
)
from baby_monitor.cli import main


class ArchiveQueueTests(unittest.TestCase):
    @staticmethod
    def _bundle(root: Path, name: str) -> tuple[Path, Path, ArchiveManifest]:
        source = root / f"source_{name}"
        event_dir = source / "events/2026/08/03/cam_01"
        event_dir.mkdir(parents=True)
        item = event_dir / f"20260803T010203_000000Z_cam_01_{name}.bin"
        item.write_bytes(("synthetic-" + name).encode("ascii"))
        manifest = build_archive_manifest(
            archive_id=f"archive_{name}",
            source_root=source,
            files=(item,),
            created_at=datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC),
        )
        return source, item, manifest

    @staticmethod
    def _config(path: Path, *, retry_delay: int = 0) -> None:
        path.write_text(
            "\n".join(
                [
                    "[archive_queue]",
                    "max_pending = 8",
                    "lease_seconds = 10",
                    f"retry_delay_seconds = {retry_delay}",
                    "max_attempts = 3",
                    "",
                    "[archive_worker]",
                    "max_jobs_per_run = 8",
                    "idle_exit_seconds = 0",
                    "max_runtime_seconds = 10",
                    "poll_seconds = 0.01",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def test_queue_lifecycle_restart_and_read_only_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-queue-") as directory:
            root = Path(directory).resolve()
            source, item, manifest = self._bundle(root, "one")
            database = root / "events.sqlite3"
            queued_at = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)
            queue = ArchiveQueue(database)
            enqueued = queue.enqueue(
                manifest=manifest,
                source_root=source,
                max_active=4,
                now=queued_at,
            )
            self.assertFalse(enqueued.reused_existing)
            self.assertEqual(ArchiveJobStatus.PENDING, enqueued.job.status)
            duplicate = queue.enqueue(
                manifest=manifest,
                source_root=source,
                max_active=4,
                now=queued_at + timedelta(seconds=1),
            )
            self.assertTrue(duplicate.reused_existing)

            database_before = database.read_bytes()
            mtime_before = database.stat().st_mtime_ns
            snapshot = ArchiveQueue(database).snapshot_read_only(
                now=queued_at + timedelta(seconds=2)
            )
            self.assertEqual(1, snapshot.counts.pending)
            self.assertEqual(1, snapshot.pending_items)
            self.assertEqual(item.stat().st_size, snapshot.pending_bytes)
            self.assertEqual(database_before, database.read_bytes())
            self.assertEqual(mtime_before, database.stat().st_mtime_ns)

            claimed = ArchiveQueue(database).claim_next(
                lease_seconds=10,
                max_attempts=3,
                now=queued_at + timedelta(seconds=3),
            )
            self.assertIsNotNone(claimed)
            assert claimed is not None
            self.assertEqual(ArchiveJobStatus.RUNNING, claimed.status)
            completed = ArchiveQueue(database).mark_completed(
                manifest.archive_id,
                now=queued_at + timedelta(seconds=4),
            )
            self.assertEqual(ArchiveJobStatus.COMPLETED, completed.status)
            self.assertTrue(item.is_file())

    def test_full_queue_tracks_failure_and_explicit_retry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-queue-") as directory:
            root = Path(directory).resolve()
            source_one, _, manifest_one = self._bundle(root, "one")
            source_two, _, manifest_two = self._bundle(root, "two")
            database = root / "events.sqlite3"
            now = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)
            queue = ArchiveQueue(database)
            queue.enqueue(
                manifest=manifest_one,
                source_root=source_one,
                max_active=1,
                now=now,
            )
            with self.assertRaisesRegex(ArchiveQueueFullError, "queue is full"):
                queue.enqueue(
                    manifest=manifest_two,
                    source_root=source_two,
                    max_active=1,
                    now=now,
                )
            snapshot = queue.snapshot(now=now)
            self.assertEqual(1, snapshot.counts.pending)
            self.assertEqual(1, snapshot.counts.failed)
            self.assertEqual({"queue_full": 1}, dict(snapshot.failed_error_counts))
            with self.assertRaises(ArchiveQueueFullError):
                queue.retry_failed(manifest_two.archive_id, max_active=1, now=now)

            claimed = queue.claim_next(lease_seconds=10, max_attempts=3, now=now)
            assert claimed is not None
            queue.mark_completed(claimed.archive_id, now=now)
            retried = queue.retry_failed(manifest_two.archive_id, max_active=1, now=now)
            self.assertEqual(ArchiveJobStatus.PENDING, retried.status)
            self.assertEqual(0, retried.attempt_count)

    def test_expired_lease_retries_then_reaches_attempt_limit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-queue-") as directory:
            root = Path(directory).resolve()
            source, _, manifest = self._bundle(root, "lease")
            database = root / "events.sqlite3"
            now = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)
            queue = ArchiveQueue(database)
            queue.enqueue(
                manifest=manifest,
                source_root=source,
                max_active=2,
                now=now,
            )
            first = queue.claim_next(lease_seconds=10, max_attempts=2, now=now)
            assert first is not None
            second = ArchiveQueue(database).claim_next(
                lease_seconds=10,
                max_attempts=2,
                now=now + timedelta(seconds=11),
            )
            assert second is not None
            self.assertEqual(2, second.attempt_count)
            failed = queue.mark_failed_attempt(
                manifest.archive_id,
                error_code="archive_unavailable",
                retry_delay_seconds=0,
                max_attempts=2,
                now=now + timedelta(seconds=12),
            )
            self.assertEqual(ArchiveJobStatus.FAILED, failed.status)
            self.assertEqual("archive_unavailable", failed.last_error_code)

    def test_cli_offline_job_stays_pending_then_recovers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-queue-") as directory:
            root = Path(directory).resolve()
            source, item, manifest = self._bundle(root, "offline")
            manifest_path = root / "manifest.json"
            manifest.write_new(manifest_path)
            database = root / "events.sqlite3"
            config = root / "config.toml"
            self._config(config)

            enqueue_output = io.StringIO()
            with redirect_stdout(enqueue_output):
                enqueue_code = main(
                    [
                        "archive-enqueue",
                        "--config",
                        str(config),
                        "--manifest",
                        str(manifest_path),
                        "--source-root",
                        str(source),
                        "--database",
                        str(database),
                    ]
                )
            self.assertEqual(0, enqueue_code)
            self.assertTrue(json.loads(enqueue_output.getvalue())["queued"])

            missing_target = root / "nas-offline"
            offline_output = io.StringIO()
            with redirect_stdout(offline_output):
                offline_code = main(
                    [
                        "archive-run-once",
                        "--config",
                        str(config),
                        "--target-root",
                        str(missing_target),
                        "--database",
                        str(database),
                    ]
                )
            offline = json.loads(offline_output.getvalue())
            self.assertEqual(0, offline_code)
            self.assertEqual("pending", offline["status"])
            self.assertEqual("archive_unavailable", offline["error_code"])
            self.assertFalse(missing_target.exists())
            self.assertNotIn(str(root), offline_output.getvalue())

            missing_target.mkdir()
            recovered_output = io.StringIO()
            with redirect_stdout(recovered_output):
                recovered_code = main(
                    [
                        "archive-run-once",
                        "--config",
                        str(config),
                        "--target-root",
                        str(missing_target),
                        "--database",
                        str(database),
                    ]
                )
            recovered = json.loads(recovered_output.getvalue())
            self.assertEqual(0, recovered_code)
            self.assertEqual("completed", recovered["status"])
            self.assertTrue(item.is_file())
            self.assertEqual(1, recovered["result"]["copied_items"])
            self.assertNotIn(str(root), recovered_output.getvalue())

    def test_bounded_worker_processes_jobs_sequentially(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-queue-") as directory:
            root = Path(directory).resolve()
            database = root / "events.sqlite3"
            config = root / "config.toml"
            self._config(config)
            target = root / "target"
            target.mkdir()
            manifests = []
            for name in ("one", "two"):
                source, _, manifest = self._bundle(root, name)
                ArchiveQueue(database).enqueue(
                    manifest=manifest,
                    source_root=source,
                    max_active=8,
                )
                manifests.append(manifest)

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "archive-worker",
                        "--config",
                        str(config),
                        "--target-root",
                        str(target),
                        "--database",
                        str(database),
                        "--max-jobs",
                        "2",
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(0, code)
            self.assertEqual(2, payload["worker"]["completed_jobs"])
            self.assertEqual(0, payload["worker"]["failed_jobs"])
            self.assertEqual(2, payload["worker"]["queue"]["completed"])
            self.assertIn(payload["worker"]["last_archive_id"], [m.archive_id for m in manifests])
            self.assertNotIn(str(root), output.getvalue())

    def test_conflict_is_permanent_and_preserves_unknown_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-queue-") as directory:
            root = Path(directory).resolve()
            source, _, manifest = self._bundle(root, "conflict")
            manifest_path = root / "manifest.json"
            manifest.write_new(manifest_path)
            database = root / "events.sqlite3"
            config = root / "config.toml"
            self._config(config)
            target = root / "target"
            destination = target.joinpath(*manifest.items[0].relative_path.split("/"))
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"unknown")
            ArchiveQueue(database).enqueue(
                manifest=manifest,
                source_root=source,
                max_active=8,
            )

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "archive-run-once",
                        "--config",
                        str(config),
                        "--target-root",
                        str(target),
                        "--database",
                        str(database),
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(2, code)
            self.assertEqual("failed", payload["status"])
            self.assertEqual("archive_conflict", payload["error_code"])
            self.assertEqual(b"unknown", destination.read_bytes())

    def test_read_only_snapshot_rejects_naive_persisted_timestamp(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-queue-") as directory:
            root = Path(directory).resolve()
            source, _, manifest = self._bundle(root, "naive-time")
            database = root / "events.sqlite3"
            queue = ArchiveQueue(database)
            queue.enqueue(
                manifest=manifest,
                source_root=source,
                max_active=8,
            )
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute(
                    "UPDATE archive_jobs SET created_at = ? WHERE archive_id = ?",
                    ("2026-08-03T01:02:03", manifest.archive_id),
                )
            before = database.read_bytes()
            with self.assertRaisesRegex(ArchiveQueueError, "missing or invalid"):
                queue.snapshot_read_only()
            self.assertEqual(before, database.read_bytes())


if __name__ == "__main__":
    unittest.main()
