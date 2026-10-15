from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.candidate_health import CandidateHealthState, CandidateWorkerHealth
from baby_monitor.candidate_worker import CandidateWorker, CandidateWorkerError
from baby_monitor.database import EventDatabase
from baby_monitor.detector_protocol import DetectorScanResult
from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from baby_monitor.domain import Detection
from baby_monitor.presence_history import PresenceFactStore, read_presence_day_summary
from baby_monitor.presence_sampling import PresenceSamplingObservation
from baby_monitor.ring_buffer import SegmentManifest
from baby_monitor.room_presence import (
    RoomPresenceError,
    RoomPresencePolicy,
    RoomPresenceStatus,
    StreamPresenceBatch,
    StreamPresenceSampleState,
    evaluate_room_presence,
)
from baby_monitor.room_registry import (
    LogicalStreamDefinition,
    PhysicalCameraDefinition,
    RoomDefinition,
    RoomRegistry,
)
from tests import test_candidate_worker as fixtures


def _result(
    timestamps=(0.0, 0.5, 1.0, 1.5),
    *,
    person_indices=(),
    fallback_indices=(),
) -> DetectorScanResult:
    detections = []
    samples = []
    for index, timestamp in enumerate(timestamps):
        start = len(detections)
        if index in person_indices:
            detections.append(Detection(timestamp, "person", 0.9, (0.1, 0.1, 0.5, 0.9)))
        samples.append(
            DetectorSampleFrame(
                decoded_frame_index=index,
                timestamp_seconds=timestamp,
                timestamp_source="nominal_fps" if index in fallback_indices else "container_pts",
                detection_start=start,
                detection_count=len(detections) - start,
            )
        )
    pts = [value for index, value in enumerate(timestamps) if index not in fallback_indices]
    gaps = [right - left for left, right in zip(pts, pts[1:], strict=False)]
    return DetectorScanResult(
        camera_id="cam_01",
        detector_adapter="fixture",
        model_id="fixture",
        model_revision="v1",
        model_sha256="a" * 64,
        detections=tuple(detections),
        version=2,
        sampling=DetectorSamplingLedger(
            native_fps=2,
            native_fps_source="reported",
            requested_sample_fps=2,
            frame_step=1,
            decoded_frame_count=len(samples),
            samples=tuple(samples),
            timestamp_fallback_frame_count=len(fallback_indices),
            max_adjacent_container_pts_gap_seconds=max(gaps) if gaps else None,
            container_pts_regression_count=0,
        ),
    )


def _registry() -> RoomRegistry:
    return RoomRegistry(
        rooms=(RoomDefinition(room_id="room_01", display_name="房间"),),
        physical_cameras=(
            PhysicalCameraDefinition(physical_camera_id="physical_01", display_name="摄像头"),
        ),
        logical_streams=(
            LogicalStreamDefinition(
                camera_id="cam_01",
                physical_camera_id="physical_01",
                room_id="room_01",
                display_name="视角",
            ),
        ),
    )


class PresenceSamplingIntegrationTests(unittest.TestCase):
    BASE = fixtures.CandidateWorkerTests.BASE

    def _prepare(self, root, result, *, count):
        root.chmod(0o700)
        manifest, _ = fixtures.CandidateWorkerTests._manifest(root / "buffer", count)
        detector = fixtures.MappingDetector({})
        if result is not None:
            detector.scan_result = lambda source: result
        store = PresenceFactStore(root / "events.sqlite3")
        EventDatabase(store.path).initialize()
        store.path.chmod(0o600)
        settings = replace(
            fixtures.CandidateWorkerTests._settings(root, manifest, detector, max_segments=count),
            presence_path=root / "presence.json",
            presence_store=store,
            health_path=root / "candidate-health.json",
        )
        return settings, store

    def _run(self, root, result, *, count=3):
        """Exercise real worker writes and replay, without models or real media."""
        settings, store = self._prepare(root, result, count=count)
        with patch(
            "baby_monitor.candidate_worker.extract_buffered_evidence",
            side_effect=fixtures.CandidateWorkerTests._fake_extract,
        ):
            first = CandidateWorker(settings).scan_once(segment_limit=count)
            self.assertEqual(count, first.scanned_segments)
            batch = StreamPresenceBatch.read_private(settings.presence_path)
            original_bytes = settings.presence_path.read_bytes()
            history = store.fetch_between(self.BASE, self.BASE + timedelta(days=1))
            self.assertEqual(batch.samples, history)
            day = read_presence_day_summary(
                store.path, local_date=self.BASE.date(), timezone_name="UTC"
            )
            self.assertIsNotNone(day)
            self.assertEqual(len(batch.samples), day.sample_count)
            self.assertFalse(day.media_read)
            self.assertFalse(day.subject_identity_inference)
            self.assertFalse(day.risk_assessment)
            self.assertFalse(day.notification_authority)

            # Restart with the persisted scan state: neither backend may gain votes.
            replay = CandidateWorker(settings).scan_once(segment_limit=count)
            self.assertEqual(0, replay.scanned_segments)
            self.assertEqual(original_bytes, settings.presence_path.read_bytes())
            self.assertEqual(history, store.fetch_between(self.BASE, self.BASE + timedelta(days=1)))
            self.assertEqual(
                day,
                read_presence_day_summary(
                    store.path, local_date=self.BASE.date(), timezone_name="UTC"
                ),
            )
        return batch, day

    def _room(self, batch, *, checked_seconds=6, policy=None):
        return evaluate_room_presence(
            registry=_registry(),
            batch=batch,
            checked_at=self.BASE + timedelta(seconds=checked_seconds),
            policy=policy,
        ).rooms[0]

    def _assert_only_gaps(self, batch, day, reason):
        self.assertEqual(3, day.coverage_gap_count)
        self.assertEqual(0, day.clear_count)
        self.assertEqual(0, day.person_detected_count)
        self.assertEqual(
            (StreamPresenceSampleState.COVERAGE_GAP,) * 3,
            tuple(sample.state for sample in batch.samples),
        )
        self.assertEqual({reason}, {sample.reason_code for sample in batch.samples})
        self.assertEqual(
            tuple(self.BASE + timedelta(seconds=seconds) for seconds in (2, 4, 6)),
            tuple(sample.observed_at for sample in batch.samples),
        )
        self.assertEqual(RoomPresenceStatus.COVERAGE_GAP, self._room(batch).status)

    def test_legacy_empty_without_ledger_cannot_clear_room_or_daily_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            batch, day = self._run(Path(directory), None)
            self._assert_only_gaps(batch, day, "sampling_unavailable")

    def test_truncated_zero_box_sampling_cannot_accumulate_clear_confirmations(self):
        with tempfile.TemporaryDirectory() as directory:
            batch, day = self._run(Path(directory), _result((0.0,)))
            self._assert_only_gaps(batch, day, "sampling_tail_gap")

    def test_uncertain_zero_box_clocks_cannot_clear_room(self):
        cases = (
            (_result(fallback_indices=(3,)), "sampling_clock_uncertain"),
            (_result((0.0, 0.5, 0.5, 1.5)), "sampling_clock_uncertain"),
            (_result((0.0, 0.1, 0.2, 1.5)), "sampling_time_gap"),
        )
        for result, reason in cases:
            with (
                self.subTest(reason=reason, samples=result.sampling.samples),
                tempfile.TemporaryDirectory() as directory,
            ):
                batch, day = self._run(Path(directory), result)
                self._assert_only_gaps(batch, day, reason)

    def test_complete_zero_box_sampling_clears_at_actual_last_calls_and_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            batch, day = self._run(Path(directory), _result())
            self.assertEqual(3, day.clear_count)
            self.assertEqual(0, day.coverage_gap_count)
            self.assertEqual(0, day.person_detected_count)
            self.assertEqual(
                (StreamPresenceSampleState.CLEAR,) * 3,
                tuple(sample.state for sample in batch.samples),
            )
            self.assertEqual(
                tuple(self.BASE + timedelta(seconds=value) for value in (1.5, 3.5, 5.5)),
                tuple(sample.observed_at for sample in batch.samples),
            )
            self.assertEqual(RoomPresenceStatus.TEMPORARILY_CLEAR, self._room(batch).status)

    def test_early_person_is_not_refreshed_by_later_empty_calls_or_segment_end(self):
        with tempfile.TemporaryDirectory() as directory:
            batch, day = self._run(Path(directory), _result(person_indices=(0,)), count=1)
            self.assertEqual(1, day.person_detected_count)
            self.assertEqual(0, day.clear_count)
            self.assertEqual(0, day.coverage_gap_count)
            self.assertEqual(self.BASE, batch.samples[0].observed_at)
            policy = RoomPresencePolicy(
                present_confirmations=1, freshness_seconds=1, present_ttl_seconds=1
            )
            self.assertEqual(
                RoomPresenceStatus.PERSON_PRESENT,
                self._room(batch, checked_seconds=0.5, policy=policy).status,
            )
            self.assertEqual(
                RoomPresenceStatus.COVERAGE_GAP,
                self._room(batch, checked_seconds=2, policy=policy).status,
            )

    def test_many_person_calls_in_one_segment_are_only_one_confirmation_vote(self):
        with tempfile.TemporaryDirectory() as directory:
            batch, day = self._run(Path(directory), _result(person_indices=(0, 1, 2, 3)), count=1)
            self.assertEqual(1, day.sample_count)
            self.assertEqual(1, day.person_detected_count)
            self.assertEqual(self.BASE + timedelta(seconds=1.5), batch.samples[0].observed_at)
            self.assertEqual(
                RoomPresenceStatus.UNKNOWN, self._room(batch, checked_seconds=2).status
            )

    def test_known_person_before_truncation_is_preserved_but_gap_is_not_a_fresh_vote(self):
        with tempfile.TemporaryDirectory() as directory:
            batch, day = self._run(Path(directory), _result((0.0,), person_indices=(0,)), count=1)
            self.assertEqual(2, day.sample_count)
            self.assertEqual(1, day.person_detected_count)
            self.assertEqual(1, day.coverage_gap_count)
            self.assertEqual(0, day.clear_count)
            self.assertEqual(
                (StreamPresenceSampleState.PERSON_DETECTED, StreamPresenceSampleState.COVERAGE_GAP),
                tuple(sample.state for sample in batch.samples),
            )
            self.assertEqual(
                (self.BASE, self.BASE + timedelta(seconds=2)),
                tuple(sample.observed_at for sample in batch.samples),
            )
            self.assertEqual("sampling_tail_gap", batch.samples[-1].reason_code)
            self.assertIsNone(batch.samples[-1].detector_id)
            self.assertEqual(
                RoomPresenceStatus.COVERAGE_GAP, self._room(batch, checked_seconds=2).status
            )

    def test_person_and_gap_at_same_pts_leave_gap_as_latest_in_both_stores(self):
        with tempfile.TemporaryDirectory() as directory:
            batch, day = self._run(
                Path(directory),
                _result((0.0, 0.5, 1.0, 2.0), person_indices=(3,), fallback_indices=(0,)),
                count=1,
            )
            self.assertEqual(1, day.person_detected_count)
            self.assertEqual(1, day.coverage_gap_count)
            self.assertEqual(
                (self.BASE + timedelta(seconds=2),) * 2,
                tuple(sample.observed_at for sample in batch.samples),
            )
            self.assertEqual(StreamPresenceSampleState.PERSON_DETECTED, batch.samples[0].state)
            self.assertEqual(StreamPresenceSampleState.COVERAGE_GAP, batch.samples[-1].state)
            self.assertEqual(
                RoomPresenceStatus.COVERAGE_GAP,
                self._room(
                    batch, checked_seconds=2, policy=RoomPresencePolicy(present_confirmations=1)
                ).status,
            )

    def test_scan_failure_interrupts_previous_clear_room_and_is_replay_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, store = self._prepare(Path(directory), _result(), count=4)
            self.assertEqual(
                3, CandidateWorker(settings).scan_once(segment_limit=3).scanned_segments
            )
            before = StreamPresenceBatch.read_private(settings.presence_path)
            self.assertEqual(RoomPresenceStatus.TEMPORARILY_CLEAR, self._room(before).status)
            failure = RuntimeError("synthetic detector failure")
            with patch.object(settings.detector, "scan_result", side_effect=failure):
                for _ in range(2):
                    with self.assertRaises(RuntimeError) as raised:
                        CandidateWorker(settings).run()
                    self.assertIs(failure, raised.exception)
            batch = StreamPresenceBatch.read_private(settings.presence_path)
            self.assertEqual(4, len(batch.samples))
            self.assertEqual(before.samples, batch.samples[:-1])
            self.assertEqual("candidate_scan_failed", batch.samples[-1].reason_code)
            self.assertEqual(self.BASE + timedelta(seconds=8), batch.samples[-1].observed_at)
            self.assertEqual(
                batch.samples, store.fetch_between(self.BASE, self.BASE + timedelta(days=1))
            )
            day = read_presence_day_summary(
                store.path, local_date=self.BASE.date(), timezone_name="UTC"
            )
            self.assertEqual(3, day.clear_count)
            self.assertEqual(1, day.coverage_gap_count)
            self.assertEqual(0, day.person_detected_count)
            self.assertEqual(
                RoomPresenceStatus.COVERAGE_GAP, self._room(batch, checked_seconds=8).status
            )
            health = CandidateWorkerHealth.read_private(settings.health_path)
            self.assertEqual(CandidateHealthState.FAILED, health.state)
            self.assertEqual("candidate_scan_failed", health.last_error_code)
            self.assertIsNone(health.last_sampling)

    def test_legacy_same_source_success_id_time_conflict_never_creates_a_second_vote(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, store = self._prepare(Path(directory), _result(), count=1)
            worker = CandidateWorker(settings)
            manifest = SegmentManifest.read(settings.manifest_path)
            # Seed exactly the old segment-end CLEAR, without a processed-scan state.
            worker._record_presence_sample(
                manifest=manifest,
                segment=manifest.segments[0],
                observations=(PresenceSamplingObservation(StreamPresenceSampleState.CLEAR, 2.0),),
            )
            original = StreamPresenceBatch.read_private(settings.presence_path).samples[0]
            self.assertEqual(self.BASE + timedelta(seconds=2), original.observed_at)
            for _ in range(2):
                with self.assertRaisesRegex(RoomPresenceError, "ID conflicts"):
                    CandidateWorker(settings).run()
            batch = StreamPresenceBatch.read_private(settings.presence_path)
            self.assertEqual(2, len(batch.samples))
            self.assertEqual(original, batch.samples[0])
            self.assertEqual("candidate_scan_failed", batch.samples[-1].reason_code)
            self.assertEqual(
                batch.samples, store.fetch_between(self.BASE, self.BASE + timedelta(days=1))
            )
            day = read_presence_day_summary(
                store.path, local_date=self.BASE.date(), timezone_name="UTC"
            )
            self.assertEqual(1, day.clear_count)
            self.assertEqual(1, day.coverage_gap_count)
            self.assertEqual(
                RoomPresenceStatus.COVERAGE_GAP, self._room(batch, checked_seconds=2).status
            )

    def test_inconsistent_segment_duration_writes_only_gap_without_calling_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, store = self._prepare(Path(directory), _result(), count=1)
            manifest = SegmentManifest.read(settings.manifest_path)
            segment = replace(manifest.segments[0], ended_at=self.BASE + timedelta(seconds=1))
            replace(manifest, segments=(segment,)).write_atomic(settings.manifest_path)
            with patch.object(settings.detector, "scan_result") as scan:
                with self.assertRaisesRegex(CandidateWorkerError, "duration does not match"):
                    CandidateWorker(settings).run()
                scan.assert_not_called()
            batch = StreamPresenceBatch.read_private(settings.presence_path)
            self.assertEqual(1, len(batch.samples))
            self.assertEqual(StreamPresenceSampleState.COVERAGE_GAP, batch.samples[0].state)
            self.assertEqual("candidate_scan_failed", batch.samples[0].reason_code)
            self.assertEqual(segment.ended_at, batch.samples[0].observed_at)
            self.assertEqual(
                batch.samples, store.fetch_between(self.BASE, self.BASE + timedelta(days=1))
            )
            day = read_presence_day_summary(
                store.path, local_date=self.BASE.date(), timezone_name="UTC"
            )
            self.assertEqual(1, day.coverage_gap_count)
            self.assertEqual(0, day.clear_count)
            self.assertEqual(0, day.person_detected_count)
            self.assertEqual(
                RoomPresenceStatus.COVERAGE_GAP, self._room(batch, checked_seconds=1).status
            )

    def test_health_failure_does_not_mask_detector_exception_or_prevent_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, store = self._prepare(Path(directory), _result(), count=1)
            failure = RuntimeError("synthetic detector failure")
            with (
                patch.object(settings.detector, "scan_result", side_effect=failure),
                patch.object(
                    CandidateWorker,
                    "_health_scan_failed",
                    side_effect=OSError("health unavailable"),
                ) as health_failure,
                self.assertRaises(RuntimeError) as raised,
            ):
                CandidateWorker(settings).run()
            self.assertIs(failure, raised.exception)
            health_failure.assert_called_once()
            batch = StreamPresenceBatch.read_private(settings.presence_path)
            self.assertEqual(1, len(batch.samples))
            self.assertEqual(StreamPresenceSampleState.COVERAGE_GAP, batch.samples[0].state)
            self.assertEqual("candidate_scan_failed", batch.samples[0].reason_code)
            self.assertEqual(
                batch.samples, store.fetch_between(self.BASE, self.BASE + timedelta(days=1))
            )
            self.assertEqual(
                RoomPresenceStatus.COVERAGE_GAP, self._room(batch, checked_seconds=2).status
            )


if __name__ == "__main__":
    unittest.main()
