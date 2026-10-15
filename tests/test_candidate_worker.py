from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor.adapters.camera import CameraSource
from baby_monitor.adapters.detector import Detector
from baby_monitor.adapters.pose import PoseEstimator
from baby_monitor.analysis_queue import AnalysisQueue
from baby_monitor.candidate_health import CandidateHealthState, CandidateWorkerHealth
from baby_monitor.candidate_worker import (
    CandidateScanState,
    CandidateWorker,
    CandidateWorkerError,
    CandidateWorkerSettings,
    CandidateWorkerStopReason,
    SegmentFingerprint,
    TimedDetection,
    _shadow_sample_due,
)
from baby_monitor.config import StorageSettings
from baby_monitor.database import EventDatabase
from baby_monitor.detector_protocol import DetectorScanResult
from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from baby_monitor.domain import Detection, EventMedia
from baby_monitor.home_daily_report import generate_home_daily_report
from baby_monitor.media_health import MediaSourceHealth, MediaSourceState
from baby_monitor.pose_presence_shadow import (
    PosePresenceShadowCameraRule,
    PosePresenceShadowClassifier,
    PosePresenceShadowPolicy,
)
from baby_monitor.pose_protocol import PoseFrameBatch
from baby_monitor.presence_history import PresenceFactMode, PresenceFactStore
from baby_monitor.ring_buffer import (
    BufferedEvidenceResult,
    BufferSegment,
    SegmentManifest,
)
from baby_monitor.room_presence import (
    StreamPresenceBatch,
    StreamPresenceSample,
    StreamPresenceSampleState,
)
from baby_monitor.sampling_history import SamplingHistoryError, read_sampling_day_summary
from baby_monitor.storage import StorageResolver


class MappingDetector(Detector):
    def __init__(self, values: dict[str, tuple[Detection, ...]]):
        self.values = values
        self.calls: list[str] = []

    def scan(self, source: CameraSource) -> tuple[Detection, ...]:
        filename = Path(source.read_uri).name
        self.calls.append(filename)
        return self.values.get(filename, ())


class NeverPoseEstimator(PoseEstimator):
    runner_adapter = "ultralytics_yolo_pose_subprocess_v1"

    def __init__(self) -> None:
        self.manifest = SimpleNamespace(model_id="pose_fixture", revision="v1")
        self.model_sha256 = "e" * 64
        self.confidence = 0.1
        self.sample_fps = 0.5
        self.max_frames = 128
        self.image_size = 640
        self.rotation_degrees = 0

    def scan(self, source: CameraSource) -> PoseFrameBatch:
        del source
        raise AssertionError("pose-disabled shadow camera loaded pose")


class SampledMappingDetector(MappingDetector):
    """Synthetic actual checks for the two-second segment fixtures only."""

    def scan_result(self, source: CameraSource) -> DetectorScanResult:
        detections = self.scan(source)
        samples = []
        offset = 0
        for index in range(4):
            timestamp = index / 2
            count = sum(item.timestamp_seconds == timestamp for item in detections)
            samples.append(DetectorSampleFrame(index, timestamp, "container_pts", offset, count))
            offset += count
        return DetectorScanResult(
            camera_id=source.camera_id,
            detector_adapter="fixture",
            model_id="fixture",
            model_revision="v1",
            model_sha256="a" * 64,
            detections=detections,
            version=2,
            sampling=DetectorSamplingLedger(
                native_fps=2,
                native_fps_source="reported",
                requested_sample_fps=2,
                frame_step=1,
                decoded_frame_count=4,
                samples=tuple(samples),
                timestamp_fallback_frame_count=0,
                max_adjacent_container_pts_gap_seconds=0.5,
                container_pts_regression_count=0,
            ),
        )


class CandidateWorkerTests(unittest.TestCase):
    BASE = datetime(2026, 8, 3, tzinfo=UTC)
    SESSION = "a1b2c3d4e5f6"

    @classmethod
    def _segments(
        cls,
        root: Path,
        count: int,
    ) -> tuple[BufferSegment, ...]:
        root.mkdir(parents=True, exist_ok=True)
        segments: list[BufferSegment] = []
        for index in range(count):
            filename = f"segment_20260803T000000Z_{cls.SESSION}_{index:06d}.mkv"
            payload = f"segment-{index}".encode()
            (root / filename).write_bytes(payload)
            started = cls.BASE + timedelta(seconds=index * 2)
            segments.append(
                BufferSegment(
                    filename=filename,
                    started_at=started,
                    ended_at=started + timedelta(seconds=2),
                    duration_seconds=2.0,
                    size_bytes=len(payload),
                    session_id=cls.SESSION,
                )
            )
        return tuple(segments)

    @classmethod
    def _manifest(cls, root: Path, count: int) -> tuple[Path, tuple[BufferSegment, ...]]:
        segments = cls._segments(root, count)
        manifest = SegmentManifest(
            camera_id="cam_01",
            source_ref="rtsp://127.0.0.1:8554/test/camera",
            created_at=cls.BASE,
            updated_at=cls.BASE + timedelta(seconds=count * 2),
            segments=segments,
        )
        path = root / "manifest.json"
        manifest.write_atomic(path)
        return path, segments

    @staticmethod
    def _fake_extract(**kwargs: object) -> BufferedEvidenceResult:
        storage = kwargs["storage"]
        if not isinstance(storage, StorageResolver):
            raise AssertionError("expected StorageResolver")
        occurred_at = kwargs["event_time"]
        event_id = kwargs["event_id"]
        keyframes = kwargs["keyframe_count"]
        if not isinstance(occurred_at, datetime):
            raise AssertionError("expected datetime")
        if not isinstance(event_id, str) or not isinstance(keyframes, int):
            raise AssertionError("invalid fake extraction inputs")
        allocated = storage.allocate_media(
            occurred_at=occurred_at,
            camera_id="cam_01",
            event_id=event_id,
            frame_count=keyframes,
        )
        reused = allocated.clip_path.exists()
        if not reused:
            allocated.clip_path.write_bytes(b"clip")
            for path in allocated.frame_paths:
                path.write_bytes(b"frame")
        media = EventMedia(
            clip_path=str(allocated.clip_path),
            frame_paths=tuple(str(path) for path in allocated.frame_paths),
            selected_segments=1,
        )
        timestamp = occurred_at.isoformat()
        return BufferedEvidenceResult(
            media=media,
            requested_start=timestamp,
            requested_end=timestamp,
            covered_start=timestamp,
            covered_end=timestamp,
            complete_coverage=True,
            coverage_gaps=(),
            selected_segments=1,
            reused_existing=reused,
        )

    @staticmethod
    def _settings(
        root: Path,
        manifest: Path,
        detector: Detector,
        *,
        max_segments: int,
        max_active: int = 32,
    ) -> CandidateWorkerSettings:
        storage = StorageResolver(
            StorageSettings(
                local_root=root / "media",
                database=root / "events.sqlite3",
            )
        )
        return CandidateWorkerSettings(
            manifest_path=manifest,
            state_path=root / "candidate-state.json",
            database_path=root / "events.sqlite3",
            storage=storage,
            detector=detector,
            detector_identity="scene_v1_fixture_detector",
            scanner_identity="fixture_scene_policy_v1",
            pre_event_seconds=0.5,
            post_event_seconds=0.5,
            candidate_gap_seconds=1.0,
            max_event_span_seconds=10.0,
            keyframe_count=3,
            max_pending_detections=20,
            max_segments_per_run=max_segments,
            idle_exit_seconds=0,
            max_runtime_seconds=5,
            poll_seconds=0.05,
            max_active_jobs=max_active,
        )

    def test_state_round_trip_is_strict_redacted_and_atomic(self) -> None:
        segment = SegmentFingerprint(
            filename=f"segment_20260803T000000Z_{self.SESSION}_000000.mkv",
            started_at=self.BASE,
            ended_at=self.BASE + timedelta(seconds=2),
            size_bytes=10,
        )
        detection = TimedDetection(
            occurred_at=self.BASE + timedelta(seconds=1),
            segment_filename=segment.filename,
            label="scene_change",
            confidence=0.7,
            bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        )
        state = CandidateScanState(
            camera_id="cam_01",
            source_ref="rtsp://127.0.0.1:8554/test/camera",
            scanner_identity="scene_policy_v1",
            updated_at=self.BASE + timedelta(seconds=2),
            watermark_at=self.BASE + timedelta(seconds=2),
            processed_segments=(segment,),
            pending_detections=(detection,),
        )
        self.assertEqual(state, CandidateScanState.from_dict(state.to_dict()))
        schema = json.loads(
            Path("schemas/candidate-scan-state-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(1, schema["properties"]["version"]["const"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state.write_atomic(path)
            self.assertEqual(state, CandidateScanState.read(path))
            self.assertEqual([], list(path.parent.glob(".state.json.*.tmp")))

        value = state.to_dict()
        # release-audit: allow=fixture-credential
        value["source_ref"] = "rtsp://user:password@127.0.0.1/live?token=x"
        with self.assertRaisesRegex(CandidateWorkerError, "redacted"):
            CandidateScanState.from_dict(value)

        value = state.to_dict()
        value["secret"] = "must-not-be-stored"
        with self.assertRaisesRegex(CandidateWorkerError, "fields"):
            CandidateScanState.from_dict(value)

        value = state.to_dict()
        value["pending_detections"][0]["bbox_xyxy"][0] = float("nan")
        with self.assertRaisesRegex(CandidateWorkerError, "bbox"):
            CandidateScanState.from_dict(value)

    def test_cross_segment_group_enqueues_once_and_restart_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, segments = self._manifest(root / "buffer", 3)
            detector = MappingDetector(
                {
                    segments[0].filename: (
                        Detection(1.8, "scene_change", 0.8, (0.0, 0.0, 1.0, 1.0)),
                    ),
                    segments[1].filename: (
                        Detection(0.2, "scene_change", 0.7, (0.0, 0.0, 1.0, 1.0)),
                    ),
                }
            )
            settings = self._settings(
                root,
                manifest,
                detector,
                max_segments=3,
            )
            with patch(
                "baby_monitor.candidate_worker.extract_buffered_evidence",
                side_effect=self._fake_extract,
            ):
                summary = CandidateWorker(settings).run()
            self.assertEqual(CandidateWorkerStopReason.SEGMENT_LIMIT, summary.stop_reason)
            self.assertEqual(3, summary.scanned_segments)
            self.assertEqual(2, summary.detections)
            self.assertEqual(1, summary.emitted_events)
            self.assertEqual(1, summary.queued_events)
            self.assertEqual(0, summary.pending_detections)

            state = CandidateScanState.read(settings.state_path)
            self.assertEqual(3, len(state.processed_segments))
            self.assertEqual((), state.pending_detections)

            queue = AnalysisQueue(settings.database_path)
            job = queue.claim_next(lease_seconds=30, max_attempts=3)
            self.assertIsNotNone(job)
            assert job is not None
            self.assertTrue(job.candidate.event_id.startswith("candidate_"))
            self.assertEqual(2, len(job.candidate.detections))
            self.assertAlmostEqual(0.5, job.candidate.detections[0].timestamp_seconds)
            self.assertAlmostEqual(0.9, job.candidate.detections[1].timestamp_seconds)
            self.assertEqual("scene_v1_fixture_detector", job.candidate.detector_id)

            restarted = CandidateWorker(settings)
            with patch(
                "baby_monitor.candidate_worker.extract_buffered_evidence",
                side_effect=self._fake_extract,
            ):
                progress = restarted.scan_once(segment_limit=3)
            self.assertEqual(0, progress.scanned_segments)
            self.assertEqual(0, progress.emitted_events)
            self.assertEqual(3, len(detector.calls))

    def test_low_confidence_group_needs_a_strong_hit_to_start_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, segments = self._manifest(root / "buffer", 3)
            detector = MappingDetector(
                {
                    segments[0].filename: (Detection(1.8, "person", 0.4, (0.1, 0.1, 0.5, 0.9)),),
                    segments[1].filename: (Detection(0.2, "person", 0.8, (0.1, 0.1, 0.5, 0.9)),),
                    segments[2].filename: (Detection(0.5, "person", 0.4, (0.1, 0.1, 0.5, 0.9)),),
                }
            )
            health_path = root / "candidate-health.json"
            settings = replace(
                self._settings(root, manifest, detector, max_segments=3),
                candidate_start_confidence=0.65,
                health_path=health_path,
            )
            with patch(
                "baby_monitor.candidate_worker.extract_buffered_evidence",
                side_effect=self._fake_extract,
            ) as extractor:
                summary = CandidateWorker(settings).run()

            self.assertEqual(3, summary.detections)
            self.assertEqual(1, summary.emitted_events)
            self.assertEqual(1, summary.suppressed_candidate_groups)
            self.assertEqual(1, summary.queued_events)
            self.assertEqual(0, summary.pending_detections)
            self.assertEqual(1, extractor.call_count)

            health = CandidateWorkerHealth.read_private(health_path)
            self.assertEqual(4, health.version)
            self.assertEqual(1, health.emitted_events)
            self.assertEqual(1, health.suppressed_candidate_groups)
            assert health.lifetime is not None
            self.assertEqual(1, health.lifetime.candidate_groups_emitted)
            self.assertEqual(1, health.lifetime.candidate_groups_suppressed)

            job = AnalysisQueue(settings.database_path).claim_next(
                lease_seconds=30,
                max_attempts=3,
            )
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(
                (0.4, 0.8), tuple(detection.confidence for detection in job.candidate.detections)
            )

    def test_optional_presence_output_records_person_and_clear_idempotently(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-presence-") as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            manifest, segments = self._manifest(root / "buffer", 2)
            detector = SampledMappingDetector(
                {
                    segments[0].filename: (Detection(0.5, "person", 0.8, (0.1, 0.1, 0.5, 0.9)),),
                    segments[1].filename: (),
                }
            )
            presence_path = root / "presence.json"
            EventDatabase(root / "events.sqlite3").initialize()
            presence_store = PresenceFactStore(root / "events.sqlite3")
            shadow_detector = MappingDetector(
                {
                    segments[0].filename: (Detection(0.5, "person", 0.8, (0.1, 0.1, 0.5, 0.9)),),
                    segments[1].filename: (),
                }
            )
            shadow_detector.runner_adapter = "ultralytics_yolo_subprocess_v1"  # type: ignore[attr-defined]
            shadow_detector.manifest = SimpleNamespace(  # type: ignore[attr-defined]
                model_id="detector_fixture",
                revision="v1",
            )
            shadow_detector.model_sha256 = "d" * 64  # type: ignore[attr-defined]
            shadow_detector.confidence = 0.1  # type: ignore[attr-defined]
            shadow_detector.sample_fps = 0.5  # type: ignore[attr-defined]
            shadow_detector.image_size = 640  # type: ignore[attr-defined]
            shadow_detector.rotation_degrees = 0  # type: ignore[attr-defined]
            shadow_policy = PosePresenceShadowPolicy(
                profile_sha256="a" * 64,
                temporal_holdout_attestation_sha256="b" * 64,
                human_audit_report_sha256="c" * 64,
                detector_adapter="ultralytics_yolo_subprocess_v1",
                detector_model_id="detector_fixture",
                detector_model_revision="v1",
                detector_model_sha256="d" * 64,
                detector_source_confidence_threshold=0.1,
                detector_requested_sample_fps=0.5,
                pose_adapter="ultralytics_yolo_pose_subprocess_v1",
                pose_model_id="pose_fixture",
                pose_model_revision="v1",
                pose_model_sha256="e" * 64,
                pose_source_confidence_threshold=0.1,
                pose_requested_sample_fps=0.5,
                pose_requested_max_frames=128,
                image_size=640,
                rotation_degrees=0,
                bin_seconds=5.0,
                cameras=(
                    PosePresenceShadowCameraRule(
                        camera_id="cam_01",
                        detector_confidence_threshold=0.5,
                        detector_hold_seconds=0.0,
                        pose_enabled=False,
                        pose_confidence_threshold=0.5,
                        pose_hold_seconds=0.0,
                        minimum_pose_bbox_area=0.1,
                        minimum_pose_bbox_height=0.5,
                        keypoint_confidence_threshold=0.5,
                        minimum_keypoint_count=8,
                    ),
                ),
            )
            shadow_classifier = PosePresenceShadowClassifier(
                shadow_policy,
                shadow_detector,
                NeverPoseEstimator(),
            )
            settings = replace(
                self._settings(root, manifest, detector, max_segments=2),
                presence_path=presence_path,
                presence_store=presence_store,
                shadow_presence_path=root / "shadow-presence.json",
                shadow_presence_store=presence_store,
                shadow_presence_classifier=shadow_classifier,
            )
            worker = CandidateWorker(settings)
            with patch(
                "baby_monitor.candidate_worker.extract_buffered_evidence",
                side_effect=self._fake_extract,
            ):
                first = worker.scan_once(segment_limit=2)
                second = worker.scan_once(segment_limit=2)

            self.assertEqual(2, first.scanned_segments)
            self.assertEqual(0, second.scanned_segments)
            batch = StreamPresenceBatch.read_private(presence_path)
            self.assertEqual(2, len(batch.samples))
            self.assertEqual(
                (
                    StreamPresenceSampleState.PERSON_DETECTED,
                    StreamPresenceSampleState.CLEAR,
                ),
                tuple(item.state for item in batch.samples),
            )
            self.assertEqual(
                {settings.scanner_identity}, {item.scanner_id for item in batch.samples}
            )
            self.assertEqual(
                {settings.detector_identity},
                {item.detector_id for item in batch.samples},
            )
            self.assertTrue(all(len(item.source_fingerprint) == 64 for item in batch.samples))
            self.assertEqual(0o600, presence_path.stat().st_mode & 0o777)
            self.assertEqual(
                batch.samples,
                presence_store.fetch_between(
                    self.BASE,
                    self.BASE + timedelta(seconds=5),
                ),
            )
            shadow_batch = StreamPresenceBatch.read_private(settings.shadow_presence_path)
            self.assertEqual(
                (
                    StreamPresenceSampleState.PERSON_DETECTED,
                    StreamPresenceSampleState.CLEAR,
                ),
                tuple(item.state for item in shadow_batch.samples),
            )
            self.assertEqual(
                shadow_batch.samples,
                presence_store.fetch_between(
                    self.BASE,
                    self.BASE + timedelta(seconds=5),
                    mode=PresenceFactMode.SHADOW,
                ),
            )

    def test_shadow_cadence_is_persisted_per_camera_across_restart(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-shadow-cadence-") as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            path = root / "shadow-presence.json"
            StreamPresenceBatch.append_private(
                path,
                StreamPresenceSample(
                    sample_id="shadow_sample_fixture",
                    camera_id="cam_01",
                    observed_at=self.BASE,
                    state=StreamPresenceSampleState.CLEAR,
                    scanner_id="shadow_scanner_fixture",
                    detector_id="shadow_detector_fixture",
                    source_fingerprint="a" * 64,
                ),
            )

            self.assertFalse(
                _shadow_sample_due(
                    path,
                    camera_id="cam_01",
                    observed_at=self.BASE + timedelta(seconds=299),
                    minimum_interval_seconds=300,
                )
            )
            self.assertTrue(
                _shadow_sample_due(
                    path,
                    camera_id="cam_01",
                    observed_at=self.BASE + timedelta(seconds=300),
                    minimum_interval_seconds=300,
                )
            )
            self.assertTrue(
                _shadow_sample_due(
                    path,
                    camera_id="cam_02",
                    observed_at=self.BASE + timedelta(seconds=1),
                    minimum_interval_seconds=300,
                )
            )
            self.assertTrue(
                _shadow_sample_due(
                    root / "new-shadow-presence.json",
                    camera_id="cam_01",
                    observed_at=self.BASE,
                    minimum_interval_seconds=300,
                )
            )

            settings = self._settings(
                root,
                self._manifest(root / "buffer", 1)[0],
                MappingDetector({}),
                max_segments=1,
            )
            with self.assertRaisesRegex(ValueError, "shadow_min_interval_seconds"):
                replace(settings, shadow_min_interval_seconds=-1)

    def test_presence_only_never_extracts_evidence_or_touches_analysis_queue(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-presence-only-") as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            manifest, segments = self._manifest(root / "buffer", 2)
            detector = MappingDetector(
                {
                    segments[0].filename: (Detection(0.5, "person", 0.8, (0.1, 0.1, 0.5, 0.9)),),
                    segments[1].filename: (),
                }
            )
            settings = replace(
                self._settings(root, manifest, detector, max_segments=2),
                presence_path=root / "presence.json",
                emit_candidate_events=False,
            )

            with patch(
                "baby_monitor.candidate_worker.extract_buffered_evidence",
                side_effect=AssertionError("presence-only worker extracted evidence"),
            ):
                progress = CandidateWorker(settings).scan_once(segment_limit=2)

            self.assertEqual(2, progress.scanned_segments)
            self.assertEqual(1, progress.detections)
            self.assertEqual(0, progress.emitted_events)
            self.assertEqual(0, progress.queued_events)
            self.assertEqual(0, progress.queue_full_events)
            self.assertEqual(0, progress.pending_detections)
            self.assertFalse(settings.database_path.exists())
            self.assertFalse((root / "media").exists())
            self.assertEqual(
                (),
                CandidateScanState.read(settings.state_path).pending_detections,
            )
            self.assertEqual(
                (
                    StreamPresenceSampleState.COVERAGE_GAP,
                    StreamPresenceSampleState.COVERAGE_GAP,
                ),
                tuple(
                    item.state
                    for item in StreamPresenceBatch.read_private(settings.presence_path).samples
                ),
            )

    def test_presence_only_seeks_to_three_segment_live_edge_without_backfill(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-presence-live-") as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            manifest, segments = self._manifest(root / "buffer", 8)
            detector = MappingDetector(
                {
                    item.filename: (Detection(0.5, "person", 0.8, (0.1, 0.1, 0.5, 0.9)),)
                    for item in segments
                }
            )
            settings = replace(
                self._settings(root, manifest, detector, max_segments=8),
                presence_path=root / "presence.json",
                emit_candidate_events=False,
            )

            first = CandidateWorker(settings).scan_once(segment_limit=8)
            second = CandidateWorker(settings).scan_once(segment_limit=8)

            self.assertEqual(3, first.scanned_segments)
            self.assertEqual(0, second.scanned_segments)
            self.assertEqual(
                [item.filename for item in segments[-3:]],
                detector.calls,
            )
            state = CandidateScanState.read(settings.state_path)
            self.assertEqual(
                tuple(item.filename for item in segments[-3:]),
                tuple(item.filename for item in state.processed_segments),
            )
            self.assertEqual(
                3,
                len(StreamPresenceBatch.read_private(settings.presence_path).samples),
            )

    def test_queue_full_event_is_tracked_and_removed_from_pending_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, segments = self._manifest(root / "buffer", 4)
            detector = MappingDetector(
                {
                    segments[0].filename: (
                        Detection(0.5, "scene_change", 0.8, (0.0, 0.0, 1.0, 1.0)),
                    ),
                    segments[2].filename: (
                        Detection(0.5, "scene_change", 0.9, (0.0, 0.0, 1.0, 1.0)),
                    ),
                }
            )
            settings = self._settings(
                root,
                manifest,
                detector,
                max_segments=4,
                max_active=1,
            )
            with patch(
                "baby_monitor.candidate_worker.extract_buffered_evidence",
                side_effect=self._fake_extract,
            ):
                summary = CandidateWorker(settings).run()
            self.assertEqual(2, summary.emitted_events)
            self.assertEqual(1, summary.queued_events)
            self.assertEqual(1, summary.queue_full_events)
            self.assertEqual(0, summary.pending_detections)
            counts = AnalysisQueue(settings.database_path).counts()
            self.assertEqual(1, counts.pending)
            self.assertEqual(1, counts.failed)
            self.assertEqual(
                (),
                CandidateScanState.read(settings.state_path).pending_detections,
            )

    def test_evicted_fingerprints_are_pruned_without_replaying_retained_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer_root = root / "buffer"
            manifest, _segments = self._manifest(buffer_root, 3)
            detector = MappingDetector({})
            settings = self._settings(root, manifest, detector, max_segments=3)
            first = CandidateWorker(settings).scan_once(segment_limit=3)
            self.assertEqual(3, first.scanned_segments)

            all_segments = self._segments(buffer_root, 4)
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test/camera",
                created_at=self.BASE,
                updated_at=self.BASE + timedelta(seconds=8),
                segments=(all_segments[2], all_segments[3]),
                evicted_segments=2,
            ).write_atomic(manifest)
            second = CandidateWorker(settings).scan_once(segment_limit=2)
            self.assertEqual(1, second.scanned_segments)
            self.assertEqual(4, len(detector.calls))
            state = CandidateScanState.read(settings.state_path)
            self.assertEqual(
                (all_segments[2].filename, all_segments[3].filename),
                tuple(item.filename for item in state.processed_segments),
            )

    def test_terminal_source_flushes_last_group_for_evidence_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, segments = self._manifest(root / "buffer", 1)
            detector = MappingDetector(
                {segments[0].filename: (Detection(1.8, "scene_change", 0.8, (0.0, 0.0, 1.0, 1.0)),)}
            )
            MediaSourceHealth(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test/camera",
                state=MediaSourceState.STOPPED,
                observed_at=self.BASE + timedelta(seconds=2),
                session_started_at=self.BASE,
                session_ended_at=self.BASE + timedelta(seconds=2),
                segments=1,
                coverage_seconds=2.0,
                gap_count=0,
                reconnect_count=0,
                capacity_reached=False,
                ffmpeg_exit_code=0,
                last_error_code=None,
            ).write_atomic(manifest.with_name("health.json"))
            settings = self._settings(root, manifest, detector, max_segments=1)
            with patch(
                "baby_monitor.candidate_worker.extract_buffered_evidence",
                side_effect=self._fake_extract,
            ):
                summary = CandidateWorker(settings).run()
            self.assertEqual(1, summary.emitted_events)
            self.assertEqual(1, summary.queued_events)
            self.assertEqual(0, summary.pending_detections)

    def test_changed_or_out_of_range_segment_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, segments = self._manifest(root / "buffer", 1)
            detector = MappingDetector(
                {segments[0].filename: (Detection(2.5, "scene_change", 0.8, (0.0, 0.0, 1.0, 1.0)),)}
            )
            settings = self._settings(root, manifest, detector, max_segments=1)
            worker = CandidateWorker(settings)
            with self.assertRaisesRegex(CandidateWorkerError, "exceeds"):
                worker.scan_once(segment_limit=1)
            self.assertFalse(settings.state_path.exists())

            detector.values[segments[0].filename] = ()
            worker.scan_once(segment_limit=1)
            changed = SegmentManifest.read(manifest)
            modified = BufferSegment(
                filename=segments[0].filename,
                started_at=segments[0].started_at,
                ended_at=segments[0].ended_at,
                duration_seconds=segments[0].duration_seconds,
                size_bytes=segments[0].size_bytes + 1,
                session_id=segments[0].session_id,
            )
            SegmentManifest(
                camera_id=changed.camera_id,
                source_ref=changed.source_ref,
                created_at=changed.created_at,
                updated_at=changed.updated_at,
                segments=(modified,),
            ).write_atomic(manifest)
            with self.assertRaisesRegex(CandidateWorkerError, "fingerprint changed"):
                CandidateWorker(settings).scan_once(segment_limit=1)

    def test_keyboard_interrupt_returns_a_bounded_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(
                root,
                root / "missing-manifest.json",
                MappingDetector({}),
                max_segments=1,
            )
            worker = CandidateWorker(settings)
            with patch.object(worker, "scan_once", side_effect=KeyboardInterrupt):
                summary = worker.run()
            self.assertEqual(CandidateWorkerStopReason.INTERRUPTED, summary.stop_reason)
            self.assertEqual(0, summary.scanned_segments)
            self.assertEqual(0, summary.emitted_events)

    def test_run_persists_path_free_detector_wait_and_stops_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _segments = self._manifest(root / "buffer", 1)
            manifest.parent.chmod(0o700)
            detector = MappingDetector({})
            detector.last_execution_lock_wait_seconds = 0.123456
            health_path = manifest.with_name("candidate-health.json")
            settings = replace(
                self._settings(root, manifest, detector, max_segments=1),
                health_path=health_path,
            )

            summary = CandidateWorker(settings).run()
            health = CandidateWorkerHealth.read_private(health_path)

            self.assertEqual(CandidateWorkerStopReason.SEGMENT_LIMIT, summary.stop_reason)
            self.assertEqual(CandidateHealthState.STOPPED, health.state)
            self.assertFalse(health.worker_running)
            self.assertEqual(1, health.scans_started)
            self.assertEqual(1, health.scans_succeeded)
            self.assertEqual(1, health.segments_scanned)
            self.assertEqual(0.123456, health.last_execution_lock_wait_seconds)
            self.assertEqual(4, health.version)
            self.assertIsNotNone(health.lifetime)
            assert health.lifetime is not None
            self.assertEqual(1, health.lifetime.sessions_started)
            self.assertEqual(1, health.lifetime.sessions_stopped)
            self.assertEqual(1, health.lifetime.scans_succeeded)
            self.assertFalse(health.to_dict()["paths_disclosed"])

    def test_clean_restart_preserves_bounded_lifetime_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _segments = self._manifest(root / "buffer", 2)
            manifest.parent.chmod(0o700)
            detector = MappingDetector({})
            detector.last_execution_lock_wait_seconds = 0.25
            health_path = manifest.with_name("candidate-health.json")
            settings = replace(
                self._settings(root, manifest, detector, max_segments=1),
                health_path=health_path,
            )

            CandidateWorker(settings).run()
            CandidateWorker(settings).run()
            health = CandidateWorkerHealth.read_private(health_path)

            assert health.lifetime is not None
            summary = health.lifetime.summary()
            self.assertEqual(2, summary["sessions_started"])
            self.assertEqual(2, summary["sessions_stopped"])
            self.assertEqual(0, summary["unclean_restarts"])
            self.assertEqual(2, summary["scans_succeeded"])
            self.assertEqual(2, summary["scan_duration_sample_count"])
            self.assertEqual(2, summary["execution_lock_wait_sample_count"])

    def test_failed_scan_finishes_one_failed_session_without_double_counting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _segments = self._manifest(root / "buffer", 1)
            manifest.parent.chmod(0o700)
            detector = MappingDetector({})
            health_path = manifest.with_name("candidate-health.json")
            settings = replace(
                self._settings(root, manifest, detector, max_segments=1),
                health_path=health_path,
            )

            with (
                patch.object(detector, "scan", side_effect=RuntimeError("private")),
                self.assertRaises(RuntimeError),
            ):
                CandidateWorker(settings).run()
            health = CandidateWorkerHealth.read_private(health_path)

            self.assertEqual(CandidateHealthState.FAILED, health.state)
            assert health.lifetime is not None
            self.assertEqual(1, health.lifetime.sessions_started)
            self.assertEqual(1, health.lifetime.sessions_failed)
            self.assertEqual(1, health.lifetime.scans_failed)
            self.assertEqual(1, health.lifetime.scan_duration_sample_count)

    @staticmethod
    def _empty_sampling_result(*, camera="cam_01", second_timestamp=1.0):
        return DetectorScanResult(
            camera_id=camera,
            detector_adapter="fixture",
            model_id="fixture",
            model_revision="v1",
            model_sha256="a" * 64,
            detections=(),
            version=2,
            sampling=DetectorSamplingLedger(
                native_fps=1,
                native_fps_source="reported",
                requested_sample_fps=1,
                frame_step=1,
                decoded_frame_count=2,
                samples=(
                    DetectorSampleFrame(0, 0, "container_pts", 0, 0),
                    DetectorSampleFrame(1, second_timestamp, "container_pts", 0, 0),
                ),
                timestamp_fallback_frame_count=0,
                max_adjacent_container_pts_gap_seconds=second_timestamp,
                container_pts_regression_count=0,
            ),
        )

    def test_normal_worker_persists_actual_empty_calls_and_health_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._manifest(root / "buffer", 1)
            manifest.parent.chmod(0o700)
            store = PresenceFactStore(root / "events.sqlite3")
            EventDatabase(store.path).initialize()
            root.chmod(0o700)
            store.path.chmod(0o600)
            detector = MappingDetector({})
            detector.scan_result = lambda source: self._empty_sampling_result()  # type: ignore[attr-defined]
            settings = replace(
                self._settings(root, manifest, detector, max_segments=1),
                presence_path=root / "presence.json",
                presence_store=store,
                health_path=manifest.parent / "candidate-health.json",
            )
            CandidateWorker(settings).run()
            self.assertEqual(detector.calls, [])
            summary = read_sampling_day_summary(
                store.path, local_date=self.BASE.date(), timezone_name="UTC"
            )
            self.assertEqual(summary.sampled_frame_count, 2)
            self.assertEqual(summary.zero_detection_sample_count, 2)
            self.assertEqual(summary.unknown_scan_count, 0)
            health = CandidateWorkerHealth.read_private(settings.health_path)
            self.assertEqual(health.last_sampling.sampled_frame_count, 2)
            self.assertEqual(health.last_sampling.zero_detection_sample_count, 2)
            self.assertFalse(health.last_sampling.source_exhaustion_verified)
            CandidateWorker(settings).run()
            again = read_sampling_day_summary(
                store.path, local_date=self.BASE.date(), timezone_name="UTC"
            )
            self.assertEqual(again, summary)
            before_report = store.path.read_bytes(), store.path.stat().st_mtime_ns
            receipt = generate_home_daily_report(
                database_path=store.path,
                output_root=root / "daily-reports",
                timezone_name="UTC",
                now=self.BASE + timedelta(minutes=1),
            )
            self.assertEqual(receipt.event_count, 0)
            self.assertEqual(
                before_report, (store.path.read_bytes(), store.path.stat().st_mtime_ns)
            )
            for extension in ("md", "html"):
                report = (
                    root / "daily-reports" / f"2026-08-03_daily_report.{extension}"
                ).read_text(encoding="utf-8")
                self.assertIn("实际模型调用（已记录）：2", report)
                self.assertIn("其中零框调用：2", report)
                self.assertIn("并非连续监护", report)
                self.assertNotIn("cam_01", report)

    def test_legacy_scan_persists_unknown_without_inventing_empty_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._manifest(root / "buffer", 1)
            manifest.parent.chmod(0o700)
            store = PresenceFactStore(root / "events.sqlite3")
            store.initialize()
            settings = replace(
                self._settings(root, manifest, MappingDetector({}), max_segments=1),
                presence_path=root / "presence.json",
                presence_store=store,
                health_path=manifest.parent / "candidate-health.json",
            )
            CandidateWorker(settings).run()
            summary = read_sampling_day_summary(
                store.path, local_date=self.BASE.date(), timezone_name="UTC"
            )
            self.assertEqual(summary.sampled_frame_count, 0)
            self.assertEqual(summary.zero_detection_sample_count, 0)
            self.assertEqual(summary.unknown_scan_count, 1)
            self.assertIsNone(
                CandidateWorkerHealth.read_private(settings.health_path).last_sampling
            )

    def test_bad_sampling_stream_or_empty_frame_time_fails_before_persistence(self):
        for bad_result in (
            self._empty_sampling_result(camera="wrong"),
            self._empty_sampling_result(second_timestamp=9),
        ):
            with self.subTest(result=bad_result), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest, _ = self._manifest(root / "buffer", 1)
                manifest.parent.chmod(0o700)
                store = PresenceFactStore(root / "events.sqlite3")
                store.initialize()
                detector = MappingDetector({})
                detector.scan_result = lambda source, result=bad_result: result  # type: ignore[attr-defined]
                settings = replace(
                    self._settings(root, manifest, detector, max_segments=1),
                    presence_path=root / "presence.json",
                    presence_store=store,
                    health_path=manifest.parent / "candidate-health.json",
                )
                with self.assertRaises(CandidateWorkerError):
                    CandidateWorker(settings).run()
                health = CandidateWorkerHealth.read_private(settings.health_path)
                self.assertEqual(health.state, CandidateHealthState.FAILED)
                self.assertIsNone(health.last_sampling)
                summary = read_sampling_day_summary(
                    store.path, local_date=self.BASE.date(), timezone_name="UTC"
                )
                self.assertEqual(summary.sampled_frame_count, 0)
                self.assertEqual(summary.recorded_scan_count + summary.unknown_scan_count, 0)

    def test_sampling_persistence_failure_is_not_a_successful_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._manifest(root / "buffer", 1)
            manifest.parent.chmod(0o700)
            store = PresenceFactStore(root / "events.sqlite3")
            store.initialize()
            settings = replace(
                self._settings(root, manifest, MappingDetector({}), max_segments=1),
                presence_path=root / "presence.json",
                presence_store=store,
                health_path=manifest.parent / "candidate-health.json",
            )
            with (
                patch(
                    "baby_monitor.candidate_worker.SamplingEvidenceStore.record",
                    side_effect=SamplingHistoryError("unavailable"),
                ),
                self.assertRaises(SamplingHistoryError),
            ):
                CandidateWorker(settings).run()
            health = CandidateWorkerHealth.read_private(settings.health_path)
            self.assertEqual(health.scans_succeeded, 0)
            self.assertEqual(health.scans_failed, 1)
            self.assertIsNone(health.last_sampling)

    def test_soak_path_does_not_open_sampling_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _ = self._manifest(root / "buffer", 1)
            detector = MappingDetector({})
            detector.scan_result = lambda source: self._empty_sampling_result()  # type: ignore[attr-defined]
            settings = replace(
                self._settings(root, manifest, detector, max_segments=1),
                presence_path=root / "presence.json",
                emit_candidate_events=False,
            )
            with patch(
                "baby_monitor.candidate_worker.SamplingEvidenceStore",
                side_effect=AssertionError("SOAK opened database"),
            ):
                CandidateWorker(settings).run()
            self.assertFalse((root / "events.sqlite3").exists())


if __name__ == "__main__":
    unittest.main()
