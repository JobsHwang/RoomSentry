from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.adapters.visual_predicate import ConservativeVisualPredicateAdapter
from baby_monitor.analysis_queue import AnalysisJobStatus, AnalysisQueue
from baby_monitor.domain import (
    CandidateEvent,
    Detection,
    EventMedia,
    SignalEvent,
    SignalType,
)
from baby_monitor.visual_predicate import VisualPredicate
from baby_monitor.visual_predicate_shadow_queue import VisualPredicateShadowAttemptQueue
from baby_monitor.visual_predicate_shadow_runtime import VisualPredicateShadowRuntimeSettings
from baby_monitor.visual_predicate_shadow_worker import (
    VisualPredicateShadowWorkerStatus,
    run_visual_predicate_shadow_attempt_once,
    sync_completed_analysis_jobs,
)

_REVISION = "5c78cab61938e73859b63724d9bf5cb88c477eaa"


class VisualPredicateShadowWorkerTests(unittest.TestCase):
    @staticmethod
    def _runtime() -> VisualPredicateShadowRuntimeSettings:
        return VisualPredicateShadowRuntimeSettings(
            enabled=True,
            backend="conservative",
            predicates=(VisualPredicate.FACE_VISIBILITY,),
            model_revision=_REVISION,
        )

    @staticmethod
    def _queue_completed_analysis(
        root: Path,
        *,
        event_id: str = "event_01",
        detector_id: str | None = "yolo11n.sha256_fixture",
        unsafe_frame: bool = False,
    ) -> Path:
        database = root / "events.sqlite3"
        media_root = root / event_id
        media_root.mkdir()
        clip = media_root / "event.mp4"
        frame = media_root / "frame.jpg"
        clip.write_bytes(b"clip")
        frame.write_bytes(b"frame")
        if not unsafe_frame:
            clip.chmod(0o600)
            frame.chmod(0o600)
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
            detector_id=detector_id,
        )
        queue = AnalysisQueue(database)
        queue.enqueue(
            candidate=candidate,
            occurred_at=datetime(2026, 8, 13, tzinfo=UTC),
            source_kind="file",
            source_ref="fixture.mp4",
            media=EventMedia(str(clip), (str(frame),)),
            storage_root=root,
            max_active=2,
        )
        claimed = queue.claim_next(lease_seconds=10, max_attempts=2)
        assert claimed is not None
        queue.mark_completed(event_id)
        return database

    @staticmethod
    def _adapter() -> ConservativeVisualPredicateAdapter:
        return ConservativeVisualPredicateAdapter(
            model_id="microsoft.Mage-VL",
            model_revision=_REVISION,
            prompt_version="mage_atomic_visible_v1",
        )

    def test_sync_does_not_read_media_and_requires_gate_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._queue_completed_analysis(root, unsafe_frame=True)
            summary = sync_completed_analysis_jobs(
                database_path=database,
                runtime=self._runtime(),
                max_active=2,
            )
            self.assertEqual(1, summary.enqueued_jobs)
            self.assertFalse(summary.queue_full)
            self.assertFalse(summary.to_dict()["periodic_sample_due"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._queue_completed_analysis(root, detector_id=None)
            summary = sync_completed_analysis_jobs(
                database_path=database,
                runtime=self._runtime(),
                max_active=2,
            )
            self.assertEqual(0, summary.selected_jobs)
            self.assertEqual(
                0, VisualPredicateShadowAttemptQueue(database).snapshot().counts.active
            )

    def test_bounded_sync_skips_unselected_without_starving_later_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._queue_completed_analysis(
                root,
                event_id="event_01",
                detector_id=None,
            )
            self._queue_completed_analysis(root, event_id="event_02")
            first = sync_completed_analysis_jobs(
                database_path=database,
                runtime=self._runtime(),
                max_active=2,
                scan_limit=1,
                now=datetime(2026, 8, 13, 1, tzinfo=UTC),
            )
            second = sync_completed_analysis_jobs(
                database_path=database,
                runtime=self._runtime(),
                max_active=2,
                scan_limit=1,
                now=datetime(2026, 8, 13, 1, tzinfo=UTC),
            )
            self.assertEqual(1, first.skipped_jobs)
            self.assertEqual(0, first.enqueued_jobs)
            self.assertEqual(1, second.enqueued_jobs)
            snapshot = VisualPredicateShadowAttemptQueue(database).snapshot()
            self.assertEqual(1, snapshot.counts.skipped)
            self.assertEqual(1, snapshot.counts.pending)

    def test_pending_hardware_event_runs_shadow_without_completing_main_analysis(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            media_root = root / "event_signal"
            media_root.mkdir()
            clip = media_root / "event.mp4"
            frame = media_root / "frame.jpg"
            clip.write_bytes(b"clip")
            frame.write_bytes(b"frame")
            clip.chmod(0o600)
            frame.chmod(0o600)
            occurred_at = datetime(2026, 8, 13, tzinfo=UTC)
            signal = SignalEvent(
                signal_id="signal_01",
                camera_id="camera_01",
                occurred_at=occurred_at,
                signal_type=SignalType.MOTION,
                source_adapter="home_assistant",
                confidence=1.0,
            )
            candidate = CandidateEvent(
                event_id="event_signal",
                camera_id="camera_01",
                start_seconds=0.0,
                end_seconds=2.0,
                detections=(),
                signals=(signal,),
            )
            database = root / "events.sqlite3"
            AnalysisQueue(database).enqueue(
                candidate=candidate,
                occurred_at=occurred_at,
                source_kind="file",
                source_ref="fixture.mp4",
                media=EventMedia(str(clip), (str(frame),)),
                storage_root=root,
                max_active=2,
            )

            result = run_visual_predicate_shadow_attempt_once(
                database_path=database,
                runtime=self._runtime(),
                adapter=self._adapter(),
                now=datetime(2026, 8, 13, 1, tzinfo=UTC),
            )

            self.assertEqual(VisualPredicateShadowWorkerStatus.COMPLETED, result.status)
            self.assertEqual(1, result.sync.scanned_pending_jobs)
            self.assertEqual(0, result.sync.scanned_completed_jobs)
            self.assertFalse(result.to_dict()["risk_inference"])
            self.assertFalse(result.to_dict()["notification_decision"])
            job = AnalysisQueue(database).get_read_only("event_signal")
            assert job is not None
            self.assertEqual(AnalysisJobStatus.PENDING, job.status)

    def test_worker_completes_shadow_without_mutating_main_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._queue_completed_analysis(root)
            result = run_visual_predicate_shadow_attempt_once(
                database_path=database,
                runtime=self._runtime(),
                adapter=self._adapter(),
                now=datetime(2026, 8, 13, 1, tzinfo=UTC),
            )
            self.assertEqual(VisualPredicateShadowWorkerStatus.COMPLETED, result.status)
            self.assertEqual(1, result.predicate_count)
            self.assertFalse(result.to_dict()["risk_inference"])
            job = AnalysisQueue(database).get_read_only("event_01")
            assert job is not None
            self.assertEqual(AnalysisJobStatus.COMPLETED, job.status)
            connection = sqlite3.connect(database)
            self.assertEqual(
                1,
                connection.execute("SELECT COUNT(*) FROM visual_predicate_shadow_runs").fetchone()[
                    0
                ],
            )
            connection.close()

    def test_worker_failure_retries_only_shadow_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._queue_completed_analysis(root, unsafe_frame=True)
            result = run_visual_predicate_shadow_attempt_once(
                database_path=database,
                runtime=self._runtime(),
                adapter=self._adapter(),
                retry_delay_seconds=0,
                max_attempts=1,
                now=datetime(2026, 8, 13, 1, tzinfo=UTC),
            )
            self.assertEqual(VisualPredicateShadowWorkerStatus.FAILED, result.status)
            self.assertEqual("shadow_run_failed", result.error_code)
            job = AnalysisQueue(database).get_read_only("event_01")
            assert job is not None
            self.assertEqual(AnalysisJobStatus.COMPLETED, job.status)

    def test_existing_attempt_keeps_its_original_gate_across_config_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = self._queue_completed_analysis(root)
            first = sync_completed_analysis_jobs(
                database_path=database,
                runtime=self._runtime(),
                max_active=2,
            )
            changed = VisualPredicateShadowRuntimeSettings(
                enabled=True,
                backend="conservative",
                predicates=(VisualPredicate.COVERING_NEAR_FACE_VISIBLE,),
                model_revision=_REVISION,
            )
            second = sync_completed_analysis_jobs(
                database_path=database,
                runtime=changed,
                max_active=2,
            )
            self.assertEqual(1, first.enqueued_jobs)
            self.assertEqual(1, second.existing_attempts)
            attempt = VisualPredicateShadowAttemptQueue(database).claim_next(
                lease_seconds=10,
                max_attempts=2,
            )
            assert attempt is not None
            self.assertEqual((VisualPredicate.FACE_VISIBILITY,), attempt.gate.predicates)


if __name__ == "__main__":
    unittest.main()
