from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.analysis_queue import (
    QUEUE_SCHEMA_SQL,
    AnalysisJobStatus,
    AnalysisQueue,
    AnalysisQueueError,
    QueueFullError,
)
from baby_monitor.domain import (
    CandidateEvent,
    EventMedia,
    EvidenceGap,
    SignalEvent,
    SignalType,
)


class AnalysisQueueTests(unittest.TestCase):
    @staticmethod
    def _event(root: Path, event_id: str) -> tuple[CandidateEvent, EventMedia]:
        media_root = root / event_id
        media_root.mkdir()
        clip = media_root / "event.mp4"
        clip.write_bytes(b"clip")
        frames = []
        for index in range(3):
            frame = media_root / f"frame_{index}.jpg"
            frame.write_bytes(b"frame")
            frames.append(str(frame))
        return (
            CandidateEvent(event_id, "cam_01", 0.0, 12.0, ()),
            EventMedia(str(clip), tuple(frames)),
        )

    def test_queue_is_bounded_idempotent_and_preserves_media_when_full(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            queue = AnalysisQueue(root / "events.sqlite3")
            occurred_at = datetime(2026, 8, 3, tzinfo=UTC)
            first_candidate, first_media = self._event(root, "event_01")
            first = queue.enqueue(
                candidate=first_candidate,
                occurred_at=occurred_at,
                source_kind="rtsp",
                source_ref="rtsp://127.0.0.1:8554/test",
                media=first_media,
                storage_root=root,
                max_active=1,
            )
            reused = queue.enqueue(
                candidate=first_candidate,
                occurred_at=occurred_at,
                source_kind="rtsp",
                source_ref="rtsp://127.0.0.1:8554/test",
                media=first_media,
                storage_root=root,
                max_active=1,
            )
            second_candidate, second_media = self._event(root, "event_02")
            with self.assertRaisesRegex(QueueFullError, "preserved"):
                queue.enqueue(
                    candidate=second_candidate,
                    occurred_at=occurred_at,
                    source_kind="rtsp",
                    source_ref="rtsp://127.0.0.1:8554/test",
                    media=second_media,
                    storage_root=root,
                    max_active=1,
                )

            self.assertFalse(first.reused_existing)
            self.assertTrue(reused.reused_existing)
            self.assertTrue(Path(second_media.clip_path).is_file())
            self.assertTrue(all(Path(path).is_file() for path in second_media.frame_paths))
            self.assertEqual(1, queue.counts().active)
            self.assertEqual(1, queue.counts().failed)
            snapshot = queue.snapshot()
            self.assertEqual(1, snapshot.ready_pending)
            self.assertEqual(8, snapshot.queued_media_files)
            self.assertGreater(snapshot.queued_media_bytes, 0)
            self.assertEqual(0, snapshot.missing_media_files)
            self.assertEqual((("queue_full", 1),), snapshot.failed_error_counts)
            with self.assertRaises(QueueFullError):
                queue.retry_failed("event_02", max_active=1)
            claimed = queue.claim_next(lease_seconds=10, max_attempts=3)
            assert claimed is not None
            queue.mark_completed(claimed.event_id)
            retried = queue.retry_failed("event_02", max_active=1)
            self.assertEqual(AnalysisJobStatus.PENDING, retried.status)

    def test_expired_lease_is_recovered_and_completed_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = root / "events.sqlite3"
            queue = AnalysisQueue(database)
            candidate, media = self._event(root, "event_01")
            started = datetime(2026, 8, 3, tzinfo=UTC)
            queue.enqueue(
                candidate=candidate,
                occurred_at=started,
                source_kind="file",
                source_ref="fixture.mp4",
                media=media,
                storage_root=root,
                max_active=2,
                now=started,
            )
            first_claim = queue.claim_next(
                lease_seconds=10,
                max_attempts=3,
                now=started,
            )
            restarted = AnalysisQueue(database)
            second_claim = restarted.claim_next(
                lease_seconds=10,
                max_attempts=3,
                now=started + timedelta(seconds=11),
            )
            self.assertIsNotNone(first_claim)
            self.assertIsNotNone(second_claim)
            assert second_claim is not None
            self.assertEqual(2, second_claim.attempt_count)
            completed = restarted.mark_completed(
                second_claim.event_id,
                now=started + timedelta(seconds=12),
            )
            self.assertEqual(AnalysisJobStatus.COMPLETED, completed.status)
            self.assertEqual(0, restarted.counts().active)
            self.assertEqual(1, restarted.counts().completed)

    def test_failures_stop_at_limit_and_manual_retry_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            queue = AnalysisQueue(root / "events.sqlite3")
            candidate, media = self._event(root, "event_01")
            started = datetime(2026, 8, 3, tzinfo=UTC)
            queue.enqueue(
                candidate=candidate,
                occurred_at=started,
                source_kind="file",
                source_ref="fixture.mp4",
                media=media,
                storage_root=root,
                max_active=2,
                now=started,
            )
            first = queue.claim_next(lease_seconds=10, max_attempts=2, now=started)
            assert first is not None
            pending = queue.mark_failed_attempt(
                first.event_id,
                error_code="VLMError",
                retry_delay_seconds=0,
                max_attempts=2,
                now=started,
            )
            self.assertEqual(AnalysisJobStatus.PENDING, pending.status)
            second = queue.claim_next(lease_seconds=10, max_attempts=2, now=started)
            assert second is not None
            failed = queue.mark_failed_attempt(
                second.event_id,
                error_code="VLMError",
                retry_delay_seconds=0,
                max_attempts=2,
                now=started,
            )
            self.assertEqual(AnalysisJobStatus.FAILED, failed.status)
            self.assertIsNone(queue.claim_next(lease_seconds=10, max_attempts=2, now=started))
            retried = queue.retry_failed(failed.event_id, max_active=2, now=started)
            self.assertEqual(AnalysisJobStatus.PENDING, retried.status)
            self.assertEqual(0, retried.attempt_count)

    def test_event_id_collision_and_unredacted_rtsp_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            queue = AnalysisQueue(root / "events.sqlite3")
            candidate, media = self._event(root, "event_01")
            occurred_at = datetime(2026, 8, 3, tzinfo=UTC)
            with self.assertRaisesRegex(AnalysisQueueError, "redacted"):
                queue.enqueue(
                    candidate=candidate,
                    occurred_at=occurred_at,
                    source_kind="rtsp",
                    # release-audit: allow=fixture-credential
                    source_ref="rtsp://user:password@127.0.0.1:8554/test?token=x",
                    media=media,
                    storage_root=root,
                    max_active=2,
                )
            queue.enqueue(
                candidate=candidate,
                occurred_at=occurred_at,
                source_kind="file",
                source_ref="fixture.mp4",
                media=media,
                storage_root=root,
                max_active=2,
            )
            other = CandidateEvent("event_01", "different_cam", 0.0, 12.0, ())
            with self.assertRaisesRegex(AnalysisQueueError, "different"):
                queue.enqueue(
                    candidate=other,
                    occurred_at=occurred_at,
                    source_kind="file",
                    source_ref="fixture.mp4",
                    media=media,
                    storage_root=root,
                    max_active=2,
                )

    def test_signal_fact_survives_sqlite_queue_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            queue = AnalysisQueue(root / "events.sqlite3")
            _candidate, media = self._event(root, "event_signal")
            signal = SignalEvent(
                signal_id="hardware_01",
                camera_id="cam_01",
                occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
                signal_type=SignalType.HARDWARE_CRY,
                source_adapter="fixture",
                confidence=0.9,
                attributes=(("channel", "main"),),
            )
            candidate = CandidateEvent(
                "event_signal",
                "cam_01",
                0.0,
                12.0,
                (),
                (signal,),
            )
            queue.enqueue(
                candidate=candidate,
                occurred_at=signal.occurred_at,
                source_kind="rtsp",
                source_ref="rtsp://127.0.0.1:8554/test",
                media=media,
                storage_root=root,
                max_active=2,
            )
            claimed = AnalysisQueue(root / "events.sqlite3").claim_next(
                lease_seconds=10,
                max_attempts=3,
            )
            assert claimed is not None
            self.assertEqual((signal,), claimed.candidate.signals)

    def test_evidence_coverage_survives_sqlite_queue_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            candidate, base_media = self._event(root, "event_gap")
            started = datetime(2026, 8, 3, tzinfo=UTC)
            gap = EvidenceGap(
                started_at=started,
                ended_at=started + timedelta(seconds=2),
                reason="segment_missing",
            )
            media = EventMedia(
                clip_path=base_media.clip_path,
                frame_paths=base_media.frame_paths,
                coverage_gaps=(gap,),
                selected_segments=2,
            )
            database = root / "events.sqlite3"
            AnalysisQueue(database).enqueue(
                candidate=candidate,
                occurred_at=started,
                source_kind="rtsp",
                source_ref="rtsp://127.0.0.1:8554/test",
                media=media,
                storage_root=root,
                max_active=2,
            )
            claimed = AnalysisQueue(database).claim_next(
                lease_seconds=10,
                max_attempts=3,
            )
            assert claimed is not None
            self.assertEqual(media, claimed.media)
            self.assertFalse(claimed.media.complete_coverage)

    def test_initialize_adds_evidence_column_to_legacy_queue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            legacy_schema = QUEUE_SCHEMA_SQL.replace("    evidence_json TEXT,\n", "")
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.executescript(legacy_schema)
            AnalysisQueue(database).initialize()
            with closing(sqlite3.connect(database)) as connection, connection:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(analysis_jobs)").fetchall()
                }
            self.assertIn("evidence_json", columns)


if __name__ == "__main__":
    unittest.main()
