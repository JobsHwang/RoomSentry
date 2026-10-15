from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.adapters.visual_predicate import (
    ConservativeVisualPredicateAdapter,
    VisualPredicateFrame,
)
from baby_monitor.domain import CandidateEvent, Detection
from baby_monitor.visual_predicate import VisualPredicate
from baby_monitor.visual_predicate_shadow import (
    VisualPredicateShadowStore,
    run_visual_predicate_shadow,
    select_visual_predicate_shadow,
)
from baby_monitor.visual_predicate_shadow_queue import (
    VisualPredicateShadowAttemptQueue,
    VisualPredicateShadowAttemptStatus,
    VisualPredicateShadowQueueError,
    VisualPredicateShadowQueueFullError,
)

_REVISION = "5c78cab61938e73859b63724d9bf5cb88c477eaa"


class VisualPredicateShadowAttemptQueueTests(unittest.TestCase):
    @staticmethod
    def _database(root: Path) -> Path:
        path = root / "events.sqlite3"
        sqlite3.connect(path).close()
        return path

    @staticmethod
    def _gate(event_id: str = "event_01", *, predicate_count: int = 1):
        candidate = CandidateEvent(
            event_id=event_id,
            camera_id="camera_01",
            start_seconds=0.0,
            end_seconds=2.0,
            detections=(
                Detection(
                    timestamp_seconds=1.0,
                    label="person",
                    confidence=0.8,
                    bbox_xyxy=(0.1, 0.1, 0.8, 0.9),
                ),
            ),
            detector_id="yolo11n.sha256_fixture",
        )
        predicates = (
            (
                VisualPredicate.FACE_VISIBILITY,
                VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
            )
            if predicate_count == 2
            else (VisualPredicate.FACE_VISIBILITY,)
        )
        return select_visual_predicate_shadow(
            candidate,
            predicates=predicates,
            person_detector_id=candidate.detector_id,
        )

    def test_missing_database_is_not_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "missing.sqlite3"
            with self.assertRaisesRegex(
                VisualPredicateShadowQueueError,
                "^visual predicate shadow attempt database is unavailable$",
            ):
                VisualPredicateShadowAttemptQueue(path)
            self.assertFalse(path.exists())

    def test_enqueue_is_bounded_idempotent_and_conflict_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            queue = VisualPredicateShadowAttemptQueue(self._database(root))
            now = datetime(2026, 8, 13, tzinfo=UTC)
            first = queue.enqueue(self._gate(), max_active=1, now=now)
            repeated = queue.enqueue(self._gate(), max_active=1, now=now)
            self.assertFalse(first.reused_existing)
            self.assertTrue(repeated.reused_existing)
            self.assertEqual(VisualPredicateShadowAttemptStatus.PENDING, first.attempt.status)

            with self.assertRaisesRegex(VisualPredicateShadowQueueError, "conflicts"):
                queue.enqueue(self._gate(predicate_count=2), max_active=1, now=now)
            with self.assertRaisesRegex(VisualPredicateShadowQueueFullError, "remains tracked"):
                queue.enqueue(self._gate("event_02"), max_active=1, now=now)
            snapshot = queue.snapshot(now=now)
            self.assertEqual(1, snapshot.counts.pending)
            self.assertEqual(1, snapshot.counts.failed)
            self.assertEqual((("queue_full", 1),), snapshot.failed_error_counts)
            self.assertFalse(snapshot.to_dict()["paths_disclosed"])

    def test_tampered_state_combination_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._database(root)
            queue = VisualPredicateShadowAttemptQueue(database)
            now = datetime(2026, 8, 13, tzinfo=UTC)
            queue.enqueue(self._gate(), max_active=2, now=now)
            connection = sqlite3.connect(database)
            connection.execute(
                """
                UPDATE visual_predicate_shadow_attempts
                SET status = 'completed', completed_at = ?
                WHERE event_id = 'event_01'
                """,
                (now.isoformat(),),
            )
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(
                VisualPredicateShadowQueueError,
                "^visual predicate shadow attempt row is invalid$",
            ):
                queue.enqueue(self._gate(), max_active=2, now=now)

    def test_expired_lease_retries_then_stops_at_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._database(root)
            started = datetime(2026, 8, 13, tzinfo=UTC)
            queue = VisualPredicateShadowAttemptQueue(database)
            queue.enqueue(self._gate(), max_active=2, now=started)
            first = queue.claim_next(lease_seconds=10, max_attempts=2, now=started)
            self.assertIsNotNone(first)

            restarted = VisualPredicateShadowAttemptQueue(database)
            second = restarted.claim_next(
                lease_seconds=10,
                max_attempts=2,
                now=started + timedelta(seconds=11),
            )
            self.assertIsNotNone(second)
            assert second is not None
            self.assertEqual(2, second.attempt_count)
            self.assertIsNone(
                restarted.claim_next(
                    lease_seconds=10,
                    max_attempts=2,
                    now=started + timedelta(seconds=22),
                )
            )
            snapshot = restarted.snapshot(now=started + timedelta(seconds=22))
            self.assertEqual(1, snapshot.counts.failed)
            self.assertEqual((("lease_expired", 1),), snapshot.failed_error_counts)

    def test_failures_are_shadow_only_and_manual_retry_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._database(root)
            connection = sqlite3.connect(database)
            connection.execute(
                "CREATE TABLE analysis_jobs (event_id TEXT PRIMARY KEY, status TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO analysis_jobs VALUES ('event_01', 'completed')")
            connection.commit()
            connection.close()

            queue = VisualPredicateShadowAttemptQueue(database)
            now = datetime(2026, 8, 13, tzinfo=UTC)
            queue.enqueue(self._gate(), max_active=2, now=now)
            claimed = queue.claim_next(lease_seconds=10, max_attempts=1, now=now)
            assert claimed is not None
            failed = queue.mark_failed_attempt(
                claimed.event_id,
                error_code="assessment_failed",
                retry_delay_seconds=0,
                max_attempts=1,
                now=now,
            )
            self.assertEqual(VisualPredicateShadowAttemptStatus.FAILED, failed.status)
            connection = sqlite3.connect(database)
            main_status = connection.execute(
                "SELECT status FROM analysis_jobs WHERE event_id = 'event_01'"
            ).fetchone()[0]
            connection.close()
            self.assertEqual("completed", main_status)
            retried = queue.retry_failed("event_01", max_active=2, now=now)
            self.assertEqual(VisualPredicateShadowAttemptStatus.PENDING, retried.status)
            self.assertEqual(0, retried.attempt_count)

    def test_completion_requires_a_gate_bound_atomic_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._database(root)
            now = datetime(2026, 8, 13, tzinfo=UTC)
            gate = self._gate()
            queue = VisualPredicateShadowAttemptQueue(database)
            queue.enqueue(gate, max_active=2, now=now)
            claimed = queue.claim_next(lease_seconds=10, max_attempts=2, now=now)
            assert claimed is not None
            with self.assertRaisesRegex(VisualPredicateShadowQueueError, "result is unavailable"):
                queue.mark_completed(claimed.event_id, now=now)

            frame_path = root / "frame.jpg"
            frame_path.write_bytes(b"fixture")
            run_visual_predicate_shadow(
                gate=gate,
                frames=(
                    VisualPredicateFrame(
                        frame_id="frame_01",
                        sha256="a" * 64,
                        media_type="image/jpeg",
                        path=str(frame_path),
                    ),
                ),
                adapter=ConservativeVisualPredicateAdapter(
                    model_id="microsoft.Mage-VL",
                    model_revision=_REVISION,
                    prompt_version="atomic_shadow_v1",
                ),
                store_factory=lambda: VisualPredicateShadowStore(database),
                recorded_at=now,
            )
            completed = queue.mark_completed(claimed.event_id, now=now)
            self.assertEqual(VisualPredicateShadowAttemptStatus.COMPLETED, completed.status)
            self.assertRegex(completed.result_binding_sha256 or "", r"^[0-9a-f]{64}$")
            public = completed.to_public_dict()
            self.assertFalse(public["paths_disclosed"])
            self.assertNotIn(str(root), json.dumps(public))

            observations = queue.recent_observations_read_only()
            self.assertEqual(1, len(observations))
            self.assertEqual(
                (gate.predicates[0],), tuple(item.predicate for item in observations[0].results)
            )
            observation_public = observations[0].to_dict()
            self.assertFalse(observation_public["media_metadata_included"])
            self.assertNotIn(str(root), json.dumps(observation_public))

            connection = sqlite3.connect(database)
            binding = connection.execute(
                "SELECT binding_json FROM visual_predicate_shadow_runs WHERE event_id = 'event_01'"
            ).fetchone()[0]
            connection.execute(
                "UPDATE visual_predicate_shadow_runs SET binding_json = ? "
                "WHERE event_id = 'event_01'",
                (binding + " ",),
            )
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(
                VisualPredicateShadowQueueError,
                "^visual predicate shadow observation is invalid$",
            ):
                queue.recent_observations_read_only()

    def test_mismatched_result_gate_cannot_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._database(root)
            now = datetime(2026, 8, 13, tzinfo=UTC)
            queue = VisualPredicateShadowAttemptQueue(database)
            gate = self._gate()
            queue.enqueue(gate, max_active=2, now=now)
            claimed = queue.claim_next(lease_seconds=10, max_attempts=2, now=now)
            assert claimed is not None
            other_gate = self._gate(predicate_count=2)
            frame_path = root / "frame.jpg"
            frame_path.write_bytes(b"fixture")
            run_visual_predicate_shadow(
                gate=other_gate,
                frames=(
                    VisualPredicateFrame(
                        frame_id="frame_01",
                        sha256="a" * 64,
                        media_type="image/jpeg",
                        path=str(frame_path),
                    ),
                ),
                adapter=ConservativeVisualPredicateAdapter(
                    model_id="microsoft.Mage-VL",
                    model_revision=_REVISION,
                    prompt_version="atomic_shadow_v1",
                ),
                store_factory=lambda: VisualPredicateShadowStore(database),
                recorded_at=now,
            )
            with self.assertRaisesRegex(VisualPredicateShadowQueueError, "does not match"):
                queue.mark_completed(claimed.event_id, now=now)

    def test_read_only_snapshot_does_not_initialize_missing_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._database(root)
            queue = VisualPredicateShadowAttemptQueue(database)
            with self.assertRaisesRegex(VisualPredicateShadowQueueError, "database is unavailable"):
                queue.snapshot(read_only=True)
            connection = sqlite3.connect(database)
            count = connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'visual_predicate_shadow_attempts'
                """
            ).fetchone()[0]
            connection.close()
            self.assertEqual(0, count)


if __name__ == "__main__":
    unittest.main()
