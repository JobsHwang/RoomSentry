from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

from baby_monitor.candidate_trigger_evaluation import (
    CandidateStartGatePolicy,
    CandidateStartGateSweepReport,
    CandidateTriggerEvaluationError,
    evaluate_candidate_start_gate_sweep,
    select_candidate_start_gate_operating_point,
)
from baby_monitor.cli import main
from baby_monitor.detector_evaluation import (
    DetectorBoxQualityReport,
    DetectorEvaluationError,
    build_detector_sampled_tracking_batch,
    build_detector_tracking_batch,
    build_detector_tracking_evaluation_set,
    evaluate_detector_box_quality,
)
from baby_monitor.detector_protocol import AdmittedDetectorScan, DetectorScanResult
from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from baby_monitor.detector_threshold_sweep import (
    DetectorOperatingPointPolicy,
    DetectorThresholdSweepError,
    DetectorThresholdSweepReport,
    evaluate_detector_threshold_sweep,
    select_detector_operating_point,
)
from baby_monitor.domain import Detection
from baby_monitor.motion_tracker import MotionObjectTracker, MotionTrackerConfig
from baby_monitor.object_tracker import (
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
)
from baby_monitor.source_admission import ExternalSourceAdmission
from baby_monitor.tracker_evaluation import (
    TrackerAssociationReview,
    TrackerReviewAssignment,
)


def _with_sampling_for_nominal_test(
    scan: AdmittedDetectorScan,
    *,
    timestamps: tuple[float, ...] = (0.0, 1.0, 2.0),
    decoded_frame_count: int = 5,
    native_fps: float = 2.0,
    native_fps_source: str = "reported",
    fallback_indices: tuple[int, ...] = (),
) -> AdmittedDetectorScan:
    """Attach explicit synthetic invocation evidence, including empty calls."""
    frame_step = max(1, round(native_fps / scan.requested_sample_fps))
    samples = []
    offset = 0
    for index, timestamp in enumerate(timestamps):
        count = sum(item.timestamp_seconds == timestamp for item in scan.result.detections)
        samples.append(
            DetectorSampleFrame(
                decoded_frame_index=index * frame_step,
                timestamp_seconds=timestamp,
                timestamp_source="nominal_fps" if index in fallback_indices else "container_pts",
                detection_start=offset,
                detection_count=count,
            )
        )
        offset += count
    ledger = DetectorSamplingLedger(
        native_fps=native_fps,
        native_fps_source=native_fps_source,
        requested_sample_fps=scan.requested_sample_fps,
        frame_step=frame_step,
        decoded_frame_count=decoded_frame_count,
        samples=tuple(samples),
        timestamp_fallback_frame_count=len(fallback_indices),
        max_adjacent_container_pts_gap_seconds=None,
        container_pts_regression_count=0,
    )
    return replace(scan, version=2, result=replace(scan.result, version=2, sampling=ledger))


class DetectorEvaluationTests(unittest.TestCase):
    BASE = datetime(2026, 8, 10, tzinfo=UTC)

    @staticmethod
    def _admission(
        source_id: str,
        asset_kind: str,
        content: bytes,
    ) -> ExternalSourceAdmission:
        is_media = asset_kind == "media_asset"
        return ExternalSourceAdmission.from_dict(
            {
                "version": 1,
                "source_id": source_id,
                "asset_kind": asset_kind,
                "canonical_url": f"https://example.com/{source_id}",
                "accessed_on": "2026-08-10",
                "source_revision": "fixture_v1",
                "upstream_name": f"{source_id} fixture",
                "license": {
                    "name": "Creative Commons BY-SA",
                    "spdx_id": None,
                    "terms_url": "https://example.com/terms",
                    "commercial_use": "allowed",
                    "modification": "allowed",
                    "redistribution": "allowed",
                },
                "access": {
                    "gated": "no",
                    "authentication_required": "no",
                    "expected_bytes": len(content),
                },
                "content": {
                    "observed_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
                "privacy": {
                    "real_minors": "no",
                    "faces": "yes" if is_media else "not_applicable",
                    "audio": "no" if is_media else "not_applicable",
                    "device_or_account_ids": "no",
                    "precise_timestamps": "yes" if is_media else "no",
                    "private_household_data": "no",
                },
                "labels": {
                    "origin": "none" if is_media else "human_nonexpert",
                    "human_verified": not is_media,
                    "ground_truth": not is_media,
                },
                "admission": {
                    "tier": "B",
                    "status": "verified_local",
                    "purpose": "model_evaluation",
                    "requires_user_approval": False,
                    "user_approval_recorded": False,
                    "git_allowed": False,
                    "ci_allowed": False,
                    "redistribution_allowed": False,
                    "training_allowed": False,
                    "reviewed_on": "2026-08-10",
                    "blocker_codes": [],
                    "restriction_codes": ["local_only"],
                },
            }
        )

    @staticmethod
    def _box(
        detection_id: str,
        bbox: tuple[float, float, float, float],
    ) -> FrameDetection:
        return FrameDetection(detection_id, "person", 1.0, bbox)

    @classmethod
    def _inputs(
        cls,
    ) -> tuple[
        DetectionFrameBatch,
        TrackerAssociationReview,
        ExternalSourceAdmission,
        ExternalSourceAdmission,
        AdmittedDetectorScan,
    ]:
        frames = (
            DetectionFrame(
                "frame_0000",
                cls.BASE,
                (
                    cls._box("a0", (0.10, 0.10, 0.30, 0.80)),
                    cls._box("b0", (0.70, 0.10, 0.90, 0.80)),
                ),
            ),
            DetectionFrame(
                "frame_0001",
                cls.BASE + timedelta(seconds=0.5),
                (cls._box("a1", (0.11, 0.10, 0.31, 0.80)),),
            ),
            DetectionFrame(
                "frame_0002",
                cls.BASE + timedelta(seconds=1.0),
                (
                    cls._box("a2", (0.12, 0.10, 0.32, 0.80)),
                    cls._box("b2", (0.68, 0.10, 0.88, 0.80)),
                ),
            ),
            DetectionFrame(
                "frame_0003",
                cls.BASE + timedelta(seconds=1.5),
                (cls._box("b3", (0.67, 0.10, 0.87, 0.80)),),
            ),
        )
        batch = DetectionFrameBatch(
            camera_id="fixture_cam",
            detector_adapter="publisher_boxes",
            frames=frames,
        )
        ground_truth_admission = cls._admission(
            "fixture_ground_truth",
            "dataset_archive",
            b"ground truth",
        )
        assignments = tuple(
            TrackerReviewAssignment(
                frame.frame_id,
                detection.detection_id,
                "assigned",
                "subject_001" if detection.detection_id.startswith("a") else "subject_002",
            )
            for frame in batch.frames
            for detection in frame.detections
        )
        review = TrackerAssociationReview(
            camera_id=batch.camera_id,
            detector_adapter=batch.detector_adapter,
            detection_batch_sha256=batch.fingerprint,
            source_admission_sha256=ground_truth_admission.fingerprint,
            review_status="approved",
            assignments=assignments,
            reviews=(),
            human_labels=True,
            ground_truth_provenance="admitted_publisher_human",
        )
        media_admission = cls._admission(
            "fixture_media",
            "media_asset",
            b"fixture video",
        )
        result = DetectorScanResult(
            camera_id=batch.camera_id,
            detector_adapter="fixture_yolo_runner",
            model_id="fixture_yolo",
            model_revision="fixture_v1",
            model_sha256="a" * 64,
            detections=(
                Detection(0.0, "person", 0.9, (0.10, 0.10, 0.30, 0.80)),
                Detection(0.0, "person", 0.8, (0.40, 0.10, 0.55, 0.75)),
                Detection(1.0, "person", 0.9, (0.68, 0.10, 0.88, 0.80)),
                Detection(2.0, "person", 0.9, (0.20, 0.10, 0.40, 0.80)),
            ),
        )
        scan = AdmittedDetectorScan(
            source_admission_sha256=media_admission.fingerprint,
            labels=("person",),
            confidence_threshold=0.35,
            requested_sample_fps=1.0,
            result=result,
        )
        return batch, review, ground_truth_admission, media_admission, scan

    def test_counts_empty_samples_and_round_trips_strict_report(self) -> None:
        batch, review, ground_truth_admission, media_admission, scan = self._inputs()

        report = evaluate_detector_box_quality(
            batch,
            review,
            scan,
            ground_truth_admission=ground_truth_admission,
            media_admission=media_admission,
            source_fps=2.0,
            source_frame_count=5,
            annotation_frame_offset=0,
            iou_threshold=0.5,
            frame_alignment_confirmed=True,
        )

        self.assertEqual(2, report.evaluated_frame_count)
        self.assertEqual(4, report.ground_truth_box_count)
        self.assertEqual(3, report.detector_box_count)
        self.assertEqual(2, report.matched_box_count)
        self.assertEqual(2, report.missed_ground_truth_box_count)
        self.assertEqual(1, report.false_positive_box_count)
        self.assertEqual(1, report.ignored_trailing_detector_box_count)
        self.assertAlmostEqual(2 / 3, report.precision or 0.0)
        self.assertEqual(0.5, report.recall)
        self.assertAlmostEqual(4 / 7, report.f1 or 0.0)
        self.assertEqual(2, report.subjects_with_missed_boxes)
        self.assertEqual(1, report.longest_consecutive_subject_miss_samples)
        self.assertEqual(1.0, report.longest_consecutive_subject_miss_seconds)
        self.assertTrue(report.detection_accuracy_evaluated)
        self.assertTrue(report.missed_detections_evaluated)
        self.assertFalse(report.paths_disclosed)

        schema = json.loads(
            Path("schemas/detector-box-quality-report-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))
        self.assertEqual(
            report,
            DetectorBoxQualityReport.from_json(report.to_json()),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            report.write_new(output)
            self.assertEqual(report, DetectorBoxQualityReport.read(output))
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                report.write_new(output)

    def test_complete_nominal_v2_preserves_v1_quality_and_tracking_output(self) -> None:
        batch, review, ground_truth_admission, media_admission, legacy = self._inputs()
        current = _with_sampling_for_nominal_test(legacy)
        arguments = {
            "ground_truth_admission": ground_truth_admission,
            "media_admission": media_admission,
            "source_fps": 2.0,
            "source_frame_count": 5,
            "annotation_frame_offset": 0,
            "frame_alignment_confirmed": True,
        }
        previous_quality = evaluate_detector_box_quality(batch, review, legacy, **arguments)
        current_quality = evaluate_detector_box_quality(batch, review, current, **arguments)
        self.assertEqual(
            replace(previous_quality, detector_scan_sha256=current.fingerprint), current_quality
        )
        tracker_arguments = {
            "media_admission": media_admission,
            "source_fps": 2.0,
            "source_frame_count": 5,
            "timeline_start": self.BASE,
        }
        self.assertEqual(
            build_detector_tracking_batch(legacy, **tracker_arguments),
            build_detector_tracking_batch(current, **tracker_arguments),
        )

    def test_sampled_tracking_cfr_matches_existing_nominal_conversion(self) -> None:
        _, _, _, admission, original = self._inputs()
        current = _with_sampling_for_nominal_test(original)
        actual = build_detector_sampled_tracking_batch(
            current,
            media_admission=admission,
            timeline_start=self.BASE,
        )
        expected = build_detector_tracking_batch(
            current,
            media_admission=admission,
            source_fps=2.0,
            source_frame_count=5,
            timeline_start=self.BASE,
        )
        self.assertEqual(expected, actual)

    def test_sampled_tracking_preserves_vfr_origin_empty_calls_and_box_slices(self) -> None:
        _, _, _, admission, original = self._inputs()
        # Keep two first-call boxes in their supplied order, an empty middle
        # call, and one last-call box. PTS is deliberately not zero based/CFR.
        detections = tuple(
            replace(item, timestamp_seconds=2.5 if index < 2 else 6.7)
            for index, item in enumerate(
                (*original.result.detections[:2], original.result.detections[-1])
            )
        )
        sparse = replace(original, result=replace(original.result, detections=detections))
        current = _with_sampling_for_nominal_test(sparse, timestamps=(2.5, 3.25, 6.7))
        anchor = self.BASE.astimezone(timezone(timedelta(hours=8)))
        actual = build_detector_sampled_tracking_batch(
            current,
            media_admission=admission,
            timeline_start=anchor,
        )
        self.assertEqual(
            ("sample_00000000", "sample_00000002", "sample_00000004"),
            tuple(frame.frame_id for frame in actual.frames),
        )
        self.assertEqual((2, 0, 1), tuple(len(frame.detections) for frame in actual.frames))
        self.assertEqual(
            (2.5, 3.25, 6.7),
            tuple((frame.occurred_at - self.BASE).total_seconds() for frame in actual.frames),
        )
        self.assertTrue(all(frame.occurred_at.tzinfo is UTC for frame in actual.frames))
        self.assertEqual(
            ("det_0000", "det_0001"), tuple(box.detection_id for box in actual.frames[0].detections)
        )
        self.assertEqual("det_0000", actual.frames[-1].detections[0].detection_id)
        self.assertEqual(
            tuple(box.bbox_xyxy for box in detections[:2]),
            tuple(box.bbox_xyxy for box in actual.frames[0].detections),
        )
        self.assertEqual(
            tuple(box.confidence for box in detections[:2]),
            tuple(box.confidence for box in actual.frames[0].detections),
        )
        self.assertEqual(original.result.camera_id, actual.camera_id)
        self.assertEqual(original.result.detector_adapter, actual.detector_adapter)

    def test_sampled_tracking_partial_decode_does_not_add_unobserved_frames(self) -> None:
        _, _, _, admission, original = self._inputs()
        partial = replace(
            original, result=replace(original.result, detections=original.result.detections[:2])
        )
        current = _with_sampling_for_nominal_test(
            partial,
            timestamps=(0.0, 1.35),
            decoded_frame_count=3,
        )
        actual = build_detector_sampled_tracking_batch(
            current,
            media_admission=admission,
            timeline_start=self.BASE,
        )
        self.assertEqual(2, len(actual.frames))
        self.assertEqual(
            ("sample_00000000", "sample_00000002"), tuple(frame.frame_id for frame in actual.frames)
        )
        self.assertEqual((2, 0), tuple(len(frame.detections) for frame in actual.frames))
        self.assertEqual(self.BASE + timedelta(seconds=1.35), actual.frames[-1].occurred_at)

    def test_sampled_tracking_actual_gap_expires_motion_track(self) -> None:
        _, _, _, admission, original = self._inputs()
        same_box = original.result.detections[0]
        two_calls = replace(
            original,
            result=replace(
                original.result,
                detections=(
                    same_box,
                    replace(same_box, timestamp_seconds=3.1),
                ),
            ),
        )
        current = _with_sampling_for_nominal_test(
            two_calls,
            timestamps=(0.0, 3.1),
            decoded_frame_count=3,
        )
        batch = build_detector_sampled_tracking_batch(
            current,
            media_admission=admission,
            timeline_start=self.BASE,
        )
        config = MotionTrackerConfig()
        self.assertGreater(3.1, config.max_track_age_seconds)
        tracker = MotionObjectTracker(
            config=config,
            camera_id=batch.camera_id,
            detector_adapter=batch.detector_adapter,
        )
        first = tracker.process(batch.frames[0])
        second = tracker.process(batch.frames[1])
        self.assertEqual((1, 1), (len(first), len(second)))
        self.assertNotEqual(first[0].tracking_id, second[0].tracking_id)
        # This proves expiry follows retained PTS, not person identity accuracy.
        self.assertEqual(
            timedelta(seconds=3.1), batch.frames[1].occurred_at - batch.frames[0].occurred_at
        )

    def test_sampled_tracking_accepts_known_pts_with_fallback_native_fps(self) -> None:
        _, _, _, admission, original = self._inputs()
        empty = replace(original, result=replace(original.result, detections=()))
        current = _with_sampling_for_nominal_test(
            empty,
            timestamps=(0.15, 1.7, 4.2),
            decoded_frame_count=51,
            native_fps=25.0,
            native_fps_source="fallback_25",
        )
        # The extra fallback belongs to an unsampled decoded frame; all three
        # invocation times remain known container PTS and therefore usable.
        current = replace(
            current,
            result=replace(
                current.result,
                sampling=replace(
                    current.result.sampling,
                    timestamp_fallback_frame_count=1,
                ),
            ),
        )
        actual = build_detector_sampled_tracking_batch(
            current,
            media_admission=admission,
            timeline_start=self.BASE,
        )
        self.assertEqual(
            ("sample_00000000", "sample_00000025", "sample_00000050"),
            tuple(frame.frame_id for frame in actual.frames),
        )
        self.assertEqual(
            (0.15, 1.7, 4.2),
            tuple((frame.occurred_at - self.BASE).total_seconds() for frame in actual.frames),
        )

    def test_sampled_tracking_rejects_unknown_or_colliding_invocation_times(self) -> None:
        _, _, _, admission, original = self._inputs()
        empty = replace(original, result=replace(original.result, detections=()))
        cases = {
            "actual_sample_fallback": {"fallback_indices": (1,)},
            "duplicate_pts": {"timestamps": (0.0, 1.0, 1.0)},
            "microsecond_collision": {"timestamps": (0.0, 0.0000004, 1.0)},
            "timedelta_overflow": {"timestamps": (0.0, 1.0, 1e308)},
        }
        for name, changes in cases.items():
            current = _with_sampling_for_nominal_test(empty, **changes)
            with self.subTest(case=name), self.assertRaises(DetectorEvaluationError):
                build_detector_sampled_tracking_batch(
                    current,
                    media_admission=admission,
                    timeline_start=self.BASE,
                )
        with self.assertRaises(DetectorEvaluationError):
            build_detector_sampled_tracking_batch(
                _with_sampling_for_nominal_test(empty),
                media_admission=admission,
                timeline_start=datetime.max.replace(tzinfo=UTC),
            )

    def test_sampled_tracking_requires_v2_admitted_media_and_aware_anchor(self) -> None:
        _, _, ground_truth_admission, admission, original = self._inputs()
        current = _with_sampling_for_nominal_test(original)
        cases = (
            (original, admission, self.BASE),
            ({}, admission, self.BASE),
            (current, None, self.BASE),
            (current, ground_truth_admission, self.BASE),
            (replace(current, source_admission_sha256="f" * 64), admission, self.BASE),
            (current, admission, self.BASE.replace(tzinfo=None)),
            (current, admission, datetime.min.replace(tzinfo=timezone(timedelta(hours=14)))),
        )
        for index, (candidate, source, anchor) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(DetectorEvaluationError):
                build_detector_sampled_tracking_batch(
                    candidate,
                    media_admission=source,
                    timeline_start=anchor,
                )

    def test_sampled_tracking_rejects_tracker_size_limits_with_domain_error(self) -> None:
        _, _, _, admission, original = self._inputs()
        box = original.result.detections[0]
        empty = replace(original, result=replace(original.result, detections=()))
        too_many_frames = _with_sampling_for_nominal_test(
            empty,
            timestamps=tuple(float(index) for index in range(10001)),
            decoded_frame_count=20001,
        )
        crowded = replace(original, result=replace(original.result, detections=(box,) * 1001))
        too_many_per_frame = _with_sampling_for_nominal_test(
            crowded,
            timestamps=(0.0,),
            decoded_frame_count=1,
        )
        total_boxes = tuple(
            replace(box, timestamp_seconds=float(index // 1000)) for index in range(10001)
        )
        total = replace(original, result=replace(original.result, detections=total_boxes))
        too_many_total = _with_sampling_for_nominal_test(
            total,
            timestamps=tuple(float(index) for index in range(11)),
            decoded_frame_count=21,
        )
        for name, candidate in (
            ("frames", too_many_frames),
            ("per_frame_boxes", too_many_per_frame),
            ("total_boxes", too_many_total),
        ):
            with self.subTest(limit=name), self.assertRaises(DetectorEvaluationError):
                build_detector_sampled_tracking_batch(
                    candidate,
                    media_admission=admission,
                    timeline_start=self.BASE,
                )

    def test_v2_invalid_empty_sample_ledgers_cannot_manufacture_nominal_frames(self) -> None:
        batch, review, ground_truth_admission, media_admission, original = self._inputs()
        empty = replace(original, result=replace(original.result, detections=()))
        cases = {
            "truncated_decode": {"timestamps": (0.0, 1.0), "decoded_frame_count": 4},
            "extra_decoded_frame": {"decoded_frame_count": 6},
            "zero_box_off_grid": {"timestamps": (0.0, 1.1, 2.0)},
            "duplicate_pts": {"timestamps": (0.0, 1.0, 1.0)},
            "timestamp_fallback": {"fallback_indices": (1,)},
            "shifted_origin": {"timestamps": (0.25, 1.25, 2.25)},
            "native_fps_mismatch": {"native_fps": 2.1},
        }
        for name, changes in cases.items():
            candidate = _with_sampling_for_nominal_test(empty, **changes)
            with (
                self.subTest(case=name, consumer="tracking"),
                self.assertRaises(DetectorEvaluationError),
            ):
                build_detector_tracking_batch(
                    candidate,
                    media_admission=media_admission,
                    source_fps=2.0,
                    source_frame_count=5,
                    timeline_start=self.BASE,
                )
            with (
                self.subTest(case=name, consumer="box_quality"),
                self.assertRaises(DetectorEvaluationError),
            ):
                evaluate_detector_box_quality(
                    batch,
                    review,
                    candidate,
                    ground_truth_admission=ground_truth_admission,
                    media_admission=media_admission,
                    source_fps=2.0,
                    source_frame_count=5,
                    annotation_frame_offset=0,
                    frame_alignment_confirmed=True,
                )

    def test_v2_requires_reported_native_fps_even_when_numeric_grid_matches(self) -> None:
        _, _, _, media_admission, original = self._inputs()
        empty = replace(original, result=replace(original.result, detections=()))
        fallback = _with_sampling_for_nominal_test(
            empty,
            native_fps=25.0,
            native_fps_source="fallback_25",
            decoded_frame_count=51,
        )
        with self.assertRaises(DetectorEvaluationError):
            build_detector_tracking_batch(
                fallback,
                media_admission=media_admission,
                source_fps=25.0,
                source_frame_count=51,
                timeline_start=self.BASE,
            )

    def test_v2_unsampled_decoder_fallback_does_not_invent_sample_uncertainty(self) -> None:
        batch, review, ground_truth_admission, media_admission, original = self._inputs()
        nominal = _with_sampling_for_nominal_test(original)
        changed = replace(
            nominal,
            result=replace(
                nominal.result,
                sampling=replace(nominal.result.sampling, timestamp_fallback_frame_count=1),
            ),
        )
        # Decoded frames 1 and 3 were not sampled. One fallback there does not
        # change the known container timestamps of invocations at frames 0/2/4.
        arguments = {
            "ground_truth_admission": ground_truth_admission,
            "media_admission": media_admission,
            "source_fps": 2.0,
            "source_frame_count": 5,
            "annotation_frame_offset": 0,
            "frame_alignment_confirmed": True,
        }
        expected = evaluate_detector_box_quality(batch, review, nominal, **arguments)
        actual = evaluate_detector_box_quality(batch, review, changed, **arguments)
        self.assertEqual(replace(expected, detector_scan_sha256=changed.fingerprint), actual)

    def test_v2_box_quality_checks_empty_samples_outside_annotation_window(self) -> None:
        batch, review, ground_truth_admission, media_admission, original = self._inputs()
        # Frame index 4 is outside this four-frame annotation window. Its actual
        # invocation is empty, so detection-only checks cannot expose wrong PTS.
        without_trailing_box = replace(
            original, result=replace(original.result, detections=original.result.detections[:-1])
        )
        malformed = _with_sampling_for_nominal_test(
            without_trailing_box,
            timestamps=(0.0, 1.0, 2.2),
        )
        with self.assertRaises(DetectorEvaluationError):
            evaluate_detector_box_quality(
                batch,
                review,
                malformed,
                ground_truth_admission=ground_truth_admission,
                media_admission=media_admission,
                source_fps=2.0,
                source_frame_count=5,
                annotation_frame_offset=0,
                frame_alignment_confirmed=True,
            )

    def test_v2_invalid_ledger_rejection_reaches_threshold_and_candidate_sweeps(self) -> None:
        batch, review, ground_truth_admission, media_admission, original = self._inputs()
        empty = replace(original, result=replace(original.result, detections=()))
        malformed = _with_sampling_for_nominal_test(empty, timestamps=(0.0, 1.0, 2.2))
        arguments = {
            "ground_truth_admission": ground_truth_admission,
            "media_admission": media_admission,
            "source_fps": 2.0,
            "source_frame_count": 5,
            "annotation_frame_offset": 0,
            "thresholds": (0.35, 0.85),
            "frame_alignment_confirmed": True,
        }
        with self.assertRaises(DetectorThresholdSweepError):
            evaluate_detector_threshold_sweep(batch, review, malformed, **arguments)
        with self.assertRaises(CandidateTriggerEvaluationError):
            evaluate_candidate_start_gate_sweep(
                batch,
                review,
                malformed,
                candidate_gap_seconds=0.5,
                max_event_span_seconds=30.0,
                **arguments,
            )

    def test_tracking_evaluation_set_rechecks_v2_ledger_not_only_report_fingerprint(self) -> None:
        batch, review, ground_truth_admission, media_admission, original = self._inputs()
        empty = replace(original, result=replace(original.result, detections=()))
        quality = evaluate_detector_box_quality(
            batch,
            review,
            empty,
            ground_truth_admission=ground_truth_admission,
            media_admission=media_admission,
            source_fps=2.0,
            source_frame_count=5,
            annotation_frame_offset=0,
            frame_alignment_confirmed=True,
        )
        malformed = _with_sampling_for_nominal_test(empty, timestamps=(0.0, 1.0, 2.2))
        falsely_rebound_quality = replace(quality, detector_scan_sha256=malformed.fingerprint)
        with self.assertRaises(DetectorEvaluationError):
            build_detector_tracking_evaluation_set(
                batch,
                review,
                malformed,
                falsely_rebound_quality,
                ground_truth_admission=ground_truth_admission,
                media_admission=media_admission,
            )

    def test_requires_alignment_matching_admission_and_exact_timestamps(self) -> None:
        batch, review, ground_truth_admission, media_admission, scan = self._inputs()
        arguments = {
            "ground_truth_admission": ground_truth_admission,
            "media_admission": media_admission,
            "source_fps": 2.0,
            "source_frame_count": 5,
            "annotation_frame_offset": 0,
        }
        with self.assertRaisesRegex(DetectorEvaluationError, "alignment"):
            evaluate_detector_box_quality(batch, review, scan, **arguments)

        wrong_scan = AdmittedDetectorScan(
            source_admission_sha256="f" * 64,
            labels=scan.labels,
            confidence_threshold=scan.confidence_threshold,
            requested_sample_fps=scan.requested_sample_fps,
            result=scan.result,
        )
        with self.assertRaisesRegex(DetectorEvaluationError, "different media"):
            evaluate_detector_box_quality(
                batch,
                review,
                wrong_scan,
                **arguments,
                frame_alignment_confirmed=True,
            )

        non_exact_result = DetectorScanResult(
            camera_id=scan.result.camera_id,
            detector_adapter=scan.result.detector_adapter,
            model_id=scan.result.model_id,
            model_revision=scan.result.model_revision,
            model_sha256=scan.result.model_sha256,
            detections=(Detection(0.1, "person", 0.9, (0.1, 0.1, 0.3, 0.8)),),
        )
        non_exact_scan = AdmittedDetectorScan(
            source_admission_sha256=media_admission.fingerprint,
            labels=("person",),
            confidence_threshold=0.35,
            requested_sample_fps=1.0,
            result=non_exact_result,
        )
        with self.assertRaisesRegex(DetectorEvaluationError, "exact source frame"):
            evaluate_detector_box_quality(
                batch,
                review,
                non_exact_scan,
                **arguments,
                frame_alignment_confirmed=True,
            )

    def test_rejects_pending_publisher_ground_truth_even_without_boxes(self) -> None:
        batch, review, ground_truth_admission, media_admission, scan = self._inputs()
        for empty in (False, True):
            with self.subTest(empty=empty):
                candidate_batch = (
                    replace(
                        batch,
                        frames=tuple(replace(frame, detections=()) for frame in batch.frames),
                    )
                    if empty
                    else batch
                )
                pending = replace(
                    review,
                    detection_batch_sha256=candidate_batch.fingerprint,
                    review_status="pending",
                    assignments=tuple(
                        replace(item, review_state="pending", subject_id=None)
                        for item in review.assignments
                    )
                    if not empty
                    else (),
                    human_labels=False,
                )
                pending.bind(candidate_batch)
                pending.bind_source_admission(ground_truth_admission)
                with self.assertRaisesRegex(DetectorEvaluationError, "approved"):
                    evaluate_detector_box_quality(
                        candidate_batch,
                        pending,
                        scan,
                        ground_truth_admission=ground_truth_admission,
                        media_admission=media_admission,
                        source_fps=2.0,
                        source_frame_count=5,
                        annotation_frame_offset=0,
                        frame_alignment_confirmed=True,
                    )

    def test_rejects_non_assigned_publisher_ground_truth_boxes(self) -> None:
        batch, review, ground_truth_admission, media_admission, scan = self._inputs()
        for state in ("duplicate", "false_positive", "unreviewable"):
            with self.subTest(state=state):
                assignments = list(review.assignments)
                assignments[1] = replace(
                    assignments[1],
                    review_state=state,
                    subject_id=assignments[0].subject_id if state == "duplicate" else None,
                )
                non_assigned = replace(review, assignments=tuple(assignments))
                non_assigned.bind(batch)
                non_assigned.bind_source_admission(ground_truth_admission)
                with self.assertRaisesRegex(DetectorEvaluationError, "every.*assigned"):
                    evaluate_detector_box_quality(
                        batch,
                        non_assigned,
                        scan,
                        ground_truth_admission=ground_truth_admission,
                        media_admission=media_admission,
                        source_fps=2.0,
                        source_frame_count=5,
                        annotation_frame_offset=0,
                        frame_alignment_confirmed=True,
                    )

    def test_rejects_non_assigned_ground_truth_outside_sampling_grid(self) -> None:
        batch, review, ground_truth_admission, media_admission, scan = self._inputs()
        assignments = tuple(
            replace(item, review_state="unreviewable", subject_id=None)
            if item.frame_id == "frame_0001"
            else item
            for item in review.assignments
        )
        non_assigned = replace(review, assignments=assignments)
        non_assigned.bind(batch)
        with self.assertRaisesRegex(DetectorEvaluationError, "every.*assigned"):
            evaluate_detector_box_quality(
                batch,
                non_assigned,
                scan,
                ground_truth_admission=ground_truth_admission,
                media_admission=media_admission,
                source_fps=2.0,
                source_frame_count=5,
                annotation_frame_offset=0,
                frame_alignment_confirmed=True,
            )

    def test_builds_tracker_batch_from_exact_quality_run(self) -> None:
        batch, review, ground_truth_admission, media_admission, scan = self._inputs()
        quality = evaluate_detector_box_quality(
            batch,
            review,
            scan,
            ground_truth_admission=ground_truth_admission,
            media_admission=media_admission,
            source_fps=2.0,
            source_frame_count=5,
            annotation_frame_offset=0,
            frame_alignment_confirmed=True,
        )

        detector_batch, detector_review = build_detector_tracking_evaluation_set(
            batch,
            review,
            scan,
            quality,
            ground_truth_admission=ground_truth_admission,
            media_admission=media_admission,
        )

        self.assertEqual(2, len(detector_batch.frames))
        self.assertEqual(
            ("frame_0000", "frame_0002"), tuple(frame.frame_id for frame in detector_batch.frames)
        )
        self.assertEqual(3, len(detector_review.assignments))
        self.assertEqual(
            2,
            sum(
                assignment.review_state == "assigned" for assignment in detector_review.assignments
            ),
        )
        self.assertEqual(
            1,
            sum(
                assignment.review_state == "false_positive"
                for assignment in detector_review.assignments
            ),
        )
        detector_review.bind(detector_batch)
        detector_review.bind_source_admission(ground_truth_admission)

        changed_quality = replace(
            quality,
            mean_matched_iou=0.9,
        )
        with self.assertRaisesRegex(DetectorEvaluationError, "exact evaluation"):
            build_detector_tracking_evaluation_set(
                batch,
                review,
                scan,
                changed_quality,
                ground_truth_admission=ground_truth_admission,
                media_admission=media_admission,
            )

    def test_builds_unlabelled_tracker_batch_with_empty_sampled_frames(self) -> None:
        _, _, _, media_admission, scan = self._inputs()
        sparse_scan = replace(
            scan,
            result=replace(
                scan.result,
                detections=scan.result.detections[:2],
            ),
        )

        batch = build_detector_tracking_batch(
            sparse_scan,
            media_admission=media_admission,
            source_fps=2.0,
            source_frame_count=5,
            timeline_start=self.BASE,
        )

        self.assertEqual(
            ("sample_00000000", "sample_00000002", "sample_00000004"),
            tuple(frame.frame_id for frame in batch.frames),
        )
        self.assertEqual(
            (2, 0, 0),
            tuple(len(frame.detections) for frame in batch.frames),
        )
        self.assertEqual(
            (0.0, 1.0, 2.0),
            tuple((frame.occurred_at - self.BASE).total_seconds() for frame in batch.frames),
        )
        self.assertEqual(scan.result.camera_id, batch.camera_id)

        wrong_admission_scan = replace(
            sparse_scan,
            source_admission_sha256="f" * 64,
        )
        with self.assertRaisesRegex(DetectorEvaluationError, "different admitted"):
            build_detector_tracking_batch(
                wrong_admission_scan,
                media_admission=media_admission,
                source_fps=2.0,
                source_frame_count=5,
                timeline_start=self.BASE,
            )

        off_grid_scan = replace(
            sparse_scan,
            result=replace(
                sparse_scan.result,
                detections=(Detection(0.5, "person", 0.9, (0.1, 0.1, 0.3, 0.8)),),
            ),
        )
        with self.assertRaisesRegex(DetectorEvaluationError, "sampling step"):
            build_detector_tracking_batch(
                off_grid_scan,
                media_admission=media_admission,
                source_fps=2.0,
                source_frame_count=5,
                timeline_start=self.BASE,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_path = root / "fixture.mpg"
            admission_path = root / "media-admission.json"
            scan_path = root / "scan.json"
            output_path = root / "tracker-frames.json"
            media_path.write_bytes(b"fixture video")
            admission_path.write_text(
                media_admission.to_json() + "\n",
                encoding="utf-8",
            )
            sparse_scan.write_new(scan_path)
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "prepare-detector-tracking-batch",
                        "--media-admission",
                        str(admission_path),
                        "--media-asset",
                        str(media_path),
                        "--detector-scan",
                        str(scan_path),
                        "--source-fps",
                        "2",
                        "--source-frame-count",
                        "5",
                        "--timeline-start",
                        self.BASE.isoformat(),
                        "--output",
                        str(output_path),
                    ]
                )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(0, code)
            self.assertEqual(3, summary["frame_count"])
            self.assertEqual(2, summary["empty_frame_count"])
            self.assertTrue(summary["timeline_is_ordering_anchor"])
            self.assertFalse(summary["network_used"])
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertEqual(batch, DetectionFrameBatch.read(output_path))

    def test_threshold_sweep_filters_one_scan_and_selects_best_f1(self) -> None:
        batch, review, ground_truth_admission, media_admission, scan = self._inputs()
        report = evaluate_detector_threshold_sweep(
            batch,
            review,
            scan,
            ground_truth_admission=ground_truth_admission,
            media_admission=media_admission,
            source_fps=2.0,
            source_frame_count=5,
            annotation_frame_offset=0,
            thresholds=(0.35, 0.85, 0.95),
            frame_alignment_confirmed=True,
        )

        self.assertEqual(
            (0.35, 0.85, 0.95),
            tuple(point.confidence_threshold for point in report.points),
        )
        self.assertEqual(
            (3, 2, 0),
            tuple(point.detector_box_count for point in report.points),
        )
        self.assertEqual(
            (2, 2, 0),
            tuple(point.matched_box_count for point in report.points),
        )
        self.assertEqual(0.85, report.best_f1_threshold)
        self.assertEqual(report.points[1].f1, report.best_f1)
        self.assertTrue(report.derived_without_model_rerun)
        self.assertFalse(report.subject_identity_inference)
        self.assertFalse(report.paths_disclosed)
        selected, feasible_count = select_detector_operating_point(
            report,
            DetectorOperatingPointPolicy(
                objective="recall_first",
                minimum_precision=0.9,
                minimum_recall=0.5,
                maximum_longest_miss_seconds=10.0,
            ),
        )
        self.assertEqual(1, feasible_count)
        self.assertIsNotNone(selected)
        self.assertEqual(0.85, selected.confidence_threshold)
        with self.assertRaisesRegex(
            DetectorThresholdSweepError,
            "objective",
        ):
            DetectorOperatingPointPolicy(
                objective="accuracy",
                minimum_precision=0.0,
                minimum_recall=0.0,
                maximum_longest_miss_seconds=1.0,
            )

        schema = json.loads(
            Path("schemas/detector-threshold-sweep-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))
        self.assertEqual(
            report,
            DetectorThresholdSweepReport.from_json(report.to_json()),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sweep.json"
            report.write_new(output)
            self.assertEqual(report, DetectorThresholdSweepReport.read(output))
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                report.write_new(output)

            oversized = Path(directory) / "oversized.json"
            oversized.write_bytes(b" " * (1024 * 1024 + 1))
            with self.assertRaisesRegex(DetectorThresholdSweepError, "1 MiB"):
                DetectorThresholdSweepReport.read(oversized)

    def test_candidate_start_gate_sweep_measures_trigger_coverage(self) -> None:
        (
            batch,
            review,
            ground_truth_admission,
            media_admission,
            scan,
        ) = self._inputs()
        batch = replace(
            batch,
            frames=(
                *batch.frames,
                DetectionFrame(
                    "frame_0004",
                    self.BASE + timedelta(seconds=2.0),
                    (),
                ),
            ),
        )
        review = replace(
            review,
            detection_batch_sha256=batch.fingerprint,
        )
        scan = replace(
            scan,
            result=replace(
                scan.result,
                detections=(
                    *scan.result.detections[:-1],
                    replace(scan.result.detections[-1], confidence=0.4),
                ),
            ),
        )
        report = evaluate_candidate_start_gate_sweep(
            batch,
            review,
            scan,
            ground_truth_admission=ground_truth_admission,
            media_admission=media_admission,
            source_fps=2.0,
            source_frame_count=5,
            annotation_frame_offset=0,
            thresholds=(0.35, 0.85, 0.95),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
            frame_alignment_confirmed=True,
        )

        self.assertEqual(3, report.evaluated_sample_count)
        self.assertEqual(2, report.positive_sample_count)
        self.assertEqual(1, report.negative_sample_count)
        self.assertEqual(1, report.ground_truth_episode_count)
        self.assertEqual(
            (3, 2, 0),
            tuple(point.emitted_group_count for point in report.points),
        )
        self.assertEqual(
            (0, 1, 3),
            tuple(point.suppressed_group_count for point in report.points),
        )
        self.assertEqual(
            (1, 0, 0),
            tuple(point.activated_negative_sample_count for point in report.points),
        )
        self.assertEqual(
            (1.0, 1.0, 0.0),
            tuple(point.positive_sample_recall for point in report.points),
        )
        self.assertEqual(
            (1.0, 1.0, 0.0),
            tuple(point.ground_truth_episode_recall for point in report.points),
        )
        self.assertEqual(0.0, report.points[1].maximum_detection_delay_seconds)
        self.assertIsNone(report.points[2].maximum_detection_delay_seconds)
        self.assertEqual(2, report.points[2].longest_uncovered_positive_samples)
        self.assertEqual(2.0, report.points[2].longest_uncovered_positive_seconds)
        self.assertTrue(report.candidate_trigger_coverage_evaluated)
        self.assertFalse(report.box_localization_accuracy_evaluated)
        self.assertFalse(report.risk_accuracy_evaluated)
        self.assertFalse(report.deployment_approved)

        policy = CandidateStartGatePolicy(
            minimum_ground_truth_episode_recall=1.0,
            minimum_positive_sample_recall=1.0,
            maximum_detection_delay_seconds=0.0,
            maximum_longest_uncovered_positive_seconds=0.0,
            maximum_negative_false_activation_rate=0.0,
            require_negative_samples=True,
        )
        selected, feasible_count = select_candidate_start_gate_operating_point(report, policy)
        self.assertEqual(1, feasible_count)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(0.85, selected.candidate_start_confidence)
        with self.assertRaisesRegex(
            CandidateTriggerEvaluationError,
            "requires negative samples",
        ):
            CandidateStartGatePolicy(
                minimum_ground_truth_episode_recall=1.0,
                minimum_positive_sample_recall=1.0,
                maximum_detection_delay_seconds=0.0,
                maximum_longest_uncovered_positive_seconds=0.0,
                maximum_negative_false_activation_rate=0.0,
            )

        (
            no_negative_batch,
            no_negative_review,
            no_negative_ground_truth_admission,
            no_negative_media_admission,
            no_negative_scan,
        ) = self._inputs()
        no_negative_report = evaluate_candidate_start_gate_sweep(
            no_negative_batch,
            no_negative_review,
            no_negative_scan,
            ground_truth_admission=no_negative_ground_truth_admission,
            media_admission=no_negative_media_admission,
            source_fps=2.0,
            source_frame_count=5,
            annotation_frame_offset=0,
            thresholds=(0.35, 0.85, 0.95),
            candidate_gap_seconds=3.0,
            max_event_span_seconds=30.0,
            frame_alignment_confirmed=True,
        )
        no_negative_selected, no_negative_feasible = select_candidate_start_gate_operating_point(
            no_negative_report,
            policy,
        )
        self.assertIsNone(no_negative_selected)
        self.assertEqual(0, no_negative_feasible)

        schema = json.loads(
            Path("schemas/candidate-start-gate-sweep-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))
        self.assertEqual(
            report,
            CandidateStartGateSweepReport.from_json(report.to_json()),
        )
        tampered = json.loads(report.to_json())
        tampered["points"][1]["suppressed_group_count"] = 2
        with self.assertRaises(CandidateTriggerEvaluationError):
            CandidateStartGateSweepReport.from_dict(tampered)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_path = root / "frames.json"
            review_path = root / "review.json"
            ground_truth_admission_path = root / "ground-truth-admission.json"
            media_admission_path = root / "media-admission.json"
            media_path = root / "fixture.mpg"
            scan_path = root / "scan.json"
            report_path = root / "candidate-gate-sweep.json"
            batch.write_new(frames_path)
            review.write_new(review_path)
            ground_truth_admission_path.write_text(
                ground_truth_admission.to_json() + "\n",
                encoding="utf-8",
            )
            media_admission_path.write_text(
                media_admission.to_json() + "\n",
                encoding="utf-8",
            )
            media_path.write_bytes(b"fixture video")
            scan.write_new(scan_path)
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "evaluate-candidate-start-gate-sweep",
                        "--ground-truth-frames",
                        str(frames_path),
                        "--ground-truth-review",
                        str(review_path),
                        "--ground-truth-admission",
                        str(ground_truth_admission_path),
                        "--media-admission",
                        str(media_admission_path),
                        "--media-asset",
                        str(media_path),
                        "--detector-scan",
                        str(scan_path),
                        "--source-fps",
                        "2",
                        "--source-frame-count",
                        "5",
                        "--start-threshold",
                        "0.35",
                        "--start-threshold",
                        "0.85",
                        "--start-threshold",
                        "0.95",
                        "--candidate-gap-seconds",
                        "0.5",
                        "--max-event-span-seconds",
                        "30",
                        "--confirm-frame-alignment",
                        "--output",
                        str(report_path),
                    ]
                )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(0, code)
            self.assertEqual(
                report.fingerprint,
                summary["report_sha256"],
            )
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertEqual(
                report,
                CandidateStartGateSweepReport.read(report_path),
            )

            validation = StringIO()
            with redirect_stdout(validation):
                validate_code = main(["validate-candidate-start-gate-sweep", str(report_path)])
            self.assertEqual(0, validate_code)
            self.assertEqual(summary, json.loads(validation.getvalue()))

            selection = StringIO()
            with redirect_stdout(selection):
                selection_code = main(
                    [
                        "select-candidate-start-gate-operating-point",
                        str(report_path),
                        "--minimum-ground-truth-episode-recall",
                        "1",
                        "--minimum-positive-sample-recall",
                        "1",
                        "--maximum-detection-delay-seconds",
                        "0",
                        "--maximum-longest-uncovered-positive-seconds",
                        "0",
                        "--maximum-negative-false-activation-rate",
                        "0",
                        "--require-negative-samples",
                    ]
                )
            selection_summary = json.loads(selection.getvalue())
            self.assertEqual(0, selection_code)
            self.assertEqual(1, selection_summary["feasible_point_count"])
            self.assertEqual(
                0.85,
                selection_summary["selected_point"]["candidate_start_confidence"],
            )
            self.assertTrue(selection_summary["dataset_specific_only"])
            self.assertFalse(selection_summary["deployment_approved"])
            self.assertFalse(selection_summary["household_configuration_changed"])
            self.assertFalse(selection_summary["end_to_end_latency_evaluated"])
            self.assertNotIn(str(root), selection.getvalue())

    def test_threshold_sweep_rejects_unusable_threshold_sequences(self) -> None:
        batch, review, ground_truth_admission, media_admission, scan = self._inputs()
        arguments = {
            "ground_truth_admission": ground_truth_admission,
            "media_admission": media_admission,
            "source_fps": 2.0,
            "source_frame_count": 5,
            "annotation_frame_offset": 0,
            "frame_alignment_confirmed": True,
        }
        for thresholds in (
            (0.4,),
            (0.4, 0.5),
            (0.35, 0.5, 0.5),
            (0.35, 0.8, 0.7),
            (0.35, 1.1),
        ):
            with (
                self.subTest(thresholds=thresholds),
                self.assertRaises(DetectorThresholdSweepError),
            ):
                evaluate_detector_threshold_sweep(
                    batch,
                    review,
                    scan,
                    thresholds=thresholds,
                    **arguments,
                )

    def test_threshold_sweep_cli_is_path_free_and_validates_output(self) -> None:
        batch, review, ground_truth_admission, media_admission, scan = self._inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_path = root / "frames.json"
            review_path = root / "review.json"
            ground_truth_admission_path = root / "ground-truth-admission.json"
            media_admission_path = root / "media-admission.json"
            media_path = root / "fixture.mpg"
            scan_path = root / "scan.json"
            report_path = root / "sweep.json"
            batch.write_new(frames_path)
            review.write_new(review_path)
            ground_truth_admission_path.write_text(
                ground_truth_admission.to_json() + "\n",
                encoding="utf-8",
            )
            media_admission_path.write_text(
                media_admission.to_json() + "\n",
                encoding="utf-8",
            )
            media_path.write_bytes(b"fixture video")
            scan.write_new(scan_path)
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "evaluate-detector-threshold-sweep",
                        "--ground-truth-frames",
                        str(frames_path),
                        "--ground-truth-review",
                        str(review_path),
                        "--ground-truth-admission",
                        str(ground_truth_admission_path),
                        "--media-admission",
                        str(media_admission_path),
                        "--media-asset",
                        str(media_path),
                        "--detector-scan",
                        str(scan_path),
                        "--source-fps",
                        "2",
                        "--source-frame-count",
                        "5",
                        "--threshold",
                        "0.35",
                        "--threshold",
                        "0.85",
                        "--threshold",
                        "0.95",
                        "--confirm-frame-alignment",
                        "--output",
                        str(report_path),
                    ]
                )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(0, code)
            self.assertEqual(0.85, summary["best_f1_threshold"])
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertEqual(
                summary["report_sha256"],
                DetectorThresholdSweepReport.read(report_path).fingerprint,
            )

            validation = StringIO()
            with redirect_stdout(validation):
                validate_code = main(["validate-detector-threshold-sweep", str(report_path)])
            self.assertEqual(0, validate_code)
            self.assertEqual(summary, json.loads(validation.getvalue()))

            selection = StringIO()
            with redirect_stdout(selection):
                selection_code = main(
                    [
                        "select-detector-operating-point",
                        str(report_path),
                        "--objective",
                        "recall_first",
                        "--minimum-precision",
                        "0.9",
                        "--minimum-recall",
                        "0.5",
                        "--maximum-longest-miss-seconds",
                        "10.0",
                    ]
                )
            selection_summary = json.loads(selection.getvalue())
            self.assertEqual(0, selection_code)
            self.assertEqual(1, selection_summary["feasible_point_count"])
            self.assertEqual(
                0.85,
                selection_summary["selected_point"]["confidence_threshold"],
            )
            self.assertTrue(selection_summary["dataset_specific_only"])
            self.assertFalse(selection_summary["deployment_approved"])
            self.assertFalse(selection_summary["paths_disclosed"])
            self.assertNotIn(str(root), selection.getvalue())
            self.assertEqual(
                summary["report_sha256"],
                selection_summary["sweep_report_sha256"],
            )

    def test_cli_verifies_media_and_writes_path_free_report(self) -> None:
        batch, review, ground_truth_admission, media_admission, scan = self._inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_path = root / "frames.json"
            review_path = root / "review.json"
            ground_truth_admission_path = root / "ground-truth-admission.json"
            media_admission_path = root / "media-admission.json"
            media_path = root / "fixture.mpg"
            scan_path = root / "scan.json"
            report_path = root / "report.json"
            batch.write_new(frames_path)
            review.write_new(review_path)
            ground_truth_admission_path.write_text(
                ground_truth_admission.to_json() + "\n",
                encoding="utf-8",
            )
            media_admission_path.write_text(
                media_admission.to_json() + "\n",
                encoding="utf-8",
            )
            media_path.write_bytes(b"fixture video")
            scan.write_new(scan_path)
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "evaluate-detector-box-quality",
                        "--ground-truth-frames",
                        str(frames_path),
                        "--ground-truth-review",
                        str(review_path),
                        "--ground-truth-admission",
                        str(ground_truth_admission_path),
                        "--media-admission",
                        str(media_admission_path),
                        "--media-asset",
                        str(media_path),
                        "--detector-scan",
                        str(scan_path),
                        "--source-fps",
                        "2",
                        "--source-frame-count",
                        "5",
                        "--confirm-frame-alignment",
                        "--output",
                        str(report_path),
                    ]
                )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(0, code)
            self.assertEqual(2, summary["matched_box_count"])
            self.assertFalse(summary["paths_disclosed"])
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertEqual(
                summary["report_sha256"],
                DetectorBoxQualityReport.read(report_path).fingerprint,
            )

            stderr = StringIO()
            with redirect_stderr(stderr):
                no_confirmation = main(
                    [
                        "evaluate-detector-box-quality",
                        "--ground-truth-frames",
                        str(frames_path),
                        "--ground-truth-review",
                        str(review_path),
                        "--ground-truth-admission",
                        str(ground_truth_admission_path),
                        "--media-admission",
                        str(media_admission_path),
                        "--media-asset",
                        str(media_path),
                        "--detector-scan",
                        str(scan_path),
                        "--source-fps",
                        "2",
                        "--source-frame-count",
                        "5",
                        "--output",
                        str(root / "unconfirmed.json"),
                    ]
                )
            self.assertEqual(2, no_confirmation)
            self.assertIn("alignment", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
