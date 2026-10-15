from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.detector_protocol import AdmittedDetectorScan, DetectorScanResult
from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from baby_monitor.domain import Detection
from baby_monitor.household_ground_truth import (
    HOUSEHOLD_TRIGGER_SELECTION_RULE,
    HouseholdEpisodeAnnotation,
    HouseholdEpisodeRange,
    HouseholdGroundTruthError,
    HouseholdTriggerCoverageReport,
    HouseholdTriggerPolicy,
    compare_household_trigger_coverage,
    evaluate_household_trigger_coverage,
    select_household_trigger_operating_point,
)
from baby_monitor.source_admission import ExternalSourceAdmission


def _media_admission(variant: str = "") -> ExternalSourceAdmission:
    content = f"private household fixture clip {variant}".encode()
    return ExternalSourceAdmission.from_dict(
        {
            "version": 1,
            "source_id": f"household_clip_fixture_{variant}",
            "asset_kind": "media_asset",
            "canonical_url": "https://example.invalid/household-clip",
            "accessed_on": "2026-08-28",
            "source_revision": "fixture_v1",
            "upstream_name": "household clip fixture",
            "license": {
                "name": "Private household media",
                "spdx_id": None,
                "terms_url": "https://example.invalid/terms",
                "commercial_use": "prohibited",
                "modification": "prohibited",
                "redistribution": "prohibited",
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
                "real_minors": "yes",
                "faces": "yes",
                "audio": "no",
                "device_or_account_ids": "no",
                "precise_timestamps": "no",
                "private_household_data": "yes",
            },
            "labels": {
                "origin": "none",
                "human_verified": False,
                "ground_truth": False,
            },
            "admission": {
                "tier": "C",
                "status": "verified_local",
                "purpose": "model_evaluation",
                "requires_user_approval": True,
                "user_approval_recorded": True,
                "git_allowed": False,
                "ci_allowed": False,
                "redistribution_allowed": False,
                "training_allowed": False,
                "reviewed_on": "2026-08-28",
                "blocker_codes": [],
                "restriction_codes": ["local_only", "private_household"],
            },
        }
    )


def _annotation(
    media_admission: ExternalSourceAdmission,
    *,
    review_state: str = "approved",
) -> HouseholdEpisodeAnnotation:
    annotation = HouseholdEpisodeAnnotation(
        annotation_id="household_annotation_0001",
        media_admission_sha256=media_admission.fingerprint,
        camera_id="camera_01",
        source_fps=25.0,
        source_frame_count=100,
        source_duration_seconds=4.0,
        scenario_tags=("night_vision", "partial_occlusion"),
        ranges=(
            HouseholdEpisodeRange(0.0, 2.0, "empty_room"),
            HouseholdEpisodeRange(2.0, 3.0, "person_present"),
            HouseholdEpisodeRange(3.0, 4.0, "danger_event"),
        ),
        review_state=review_state,
        reviewers=("reviewer_01",) if review_state == "approved" else (),
        human_verified=review_state == "approved",
    )
    assert annotation.source_duration_seconds == 4.0
    return annotation


def _scan(
    media_admission: ExternalSourceAdmission,
    *,
    camera_id: str = "camera_01",
) -> AdmittedDetectorScan:
    result = DetectorScanResult(
        camera_id=camera_id,
        detector_adapter="yolo_command",
        model_id="yolo11n",
        model_revision="fixture",
        model_sha256=hashlib.sha256(b"weights").hexdigest(),
        detections=(
            Detection(0.0, "person", 0.40, (0.10, 0.10, 0.30, 0.80)),
            Detection(2.0, "person", 0.90, (0.12, 0.12, 0.34, 0.82)),
            Detection(2.2, "person", 0.90, (0.14, 0.12, 0.36, 0.82)),
            Detection(3.0, "person", 0.50, (0.20, 0.10, 0.40, 0.80)),
            Detection(3.2, "person", 0.50, (0.22, 0.10, 0.42, 0.80)),
            Detection(3.6, "person", 0.90, (0.24, 0.10, 0.44, 0.80)),
        ),
    )
    return AdmittedDetectorScan(
        source_admission_sha256=media_admission.fingerprint,
        labels=("person",),
        confidence_threshold=0.35,
        requested_sample_fps=5.0,
        result=result,
    )


class HouseholdEpisodeAnnotationTests(unittest.TestCase):
    def test_valid_annotation_round_trip(self) -> None:
        admission = _media_admission()
        annotation = _annotation(admission)
        restored = HouseholdEpisodeAnnotation.from_json(annotation.to_json())
        self.assertEqual(annotation, restored)
        schema = json.loads(
            Path("schemas/household-episode-annotation-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(annotation.to_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.json"
            annotation.write_new(path)
            self.assertEqual(annotation, HouseholdEpisodeAnnotation.read(path))
            with self.assertRaises(FileExistsError):
                annotation.write_new(path)

    def test_rejects_overlapping_unsorted_and_overflowing_ranges(self) -> None:
        admission = _media_admission()
        base = _annotation(admission)
        with self.assertRaises(HouseholdGroundTruthError):
            replace(
                base,
                ranges=(
                    HouseholdEpisodeRange(0.0, 2.5, "empty_room"),
                    HouseholdEpisodeRange(2.0, 3.0, "person_present"),
                ),
            )
        with self.assertRaises(HouseholdGroundTruthError):
            replace(
                base,
                ranges=(
                    HouseholdEpisodeRange(2.0, 3.0, "person_present"),
                    HouseholdEpisodeRange(0.0, 2.0, "empty_room"),
                ),
            )
        with self.assertRaises(HouseholdGroundTruthError):
            replace(
                base,
                ranges=(HouseholdEpisodeRange(0.0, 4.5, "empty_room"),),
            )

    def test_rejects_duration_frame_grid_conflict_and_bad_tags(self) -> None:
        admission = _media_admission()
        base = _annotation(admission)
        with self.assertRaises(HouseholdGroundTruthError):
            replace(base, source_duration_seconds=3.5)
        with self.assertRaises(HouseholdGroundTruthError):
            replace(base, scenario_tags=("night_vision", "bogus_tag"))
        with self.assertRaises(HouseholdGroundTruthError):
            replace(base, scenario_tags=("night_vision", "night_vision"))

    def test_review_state_requires_anonymous_reviewers(self) -> None:
        admission = _media_admission()
        approved = _annotation(admission)
        with self.assertRaises(HouseholdGroundTruthError):
            replace(approved, reviewers=())
        with self.assertRaises(HouseholdGroundTruthError):
            replace(approved, human_verified=False)
        pending = _annotation(admission, review_state="pending")
        with self.assertRaises(HouseholdGroundTruthError):
            replace(pending, reviewers=("reviewer_02",))
        with self.assertRaises(HouseholdGroundTruthError):
            replace(pending, human_verified=True)

    def test_rejects_duplicate_keys_and_tampered_fingerprint(self) -> None:
        admission = _media_admission()
        annotation = _annotation(admission)
        with self.assertRaises(HouseholdGroundTruthError):
            HouseholdEpisodeAnnotation.from_json('{"version": 1, "version": 1}')
        tampered = json.loads(annotation.to_json())
        tampered["ranges"][0]["kind"] = "person_present"
        with self.assertRaises(HouseholdGroundTruthError):
            HouseholdEpisodeAnnotation.from_dict(tampered)


class HouseholdTriggerCoverageTests(unittest.TestCase):
    @staticmethod
    def _v2_scan(
        admission: ExternalSourceAdmission,
        *,
        empty: bool = False,
        decoded_frame_count: int = 100,
    ) -> AdmittedDetectorScan:
        scan = _scan(admission)
        detections = () if empty else scan.result.detections
        samples: list[DetectorSampleFrame] = []
        offset = 0
        for index in range(0, decoded_frame_count, 5):
            seconds = index / 25.0
            count = sum(detection.timestamp_seconds == seconds for detection in detections)
            samples.append(DetectorSampleFrame(index, seconds, "container_pts", offset, count))
            offset += count
        ledger = DetectorSamplingLedger(
            native_fps=25.0,
            native_fps_source="reported",
            requested_sample_fps=5.0,
            frame_step=5,
            decoded_frame_count=decoded_frame_count,
            samples=tuple(samples),
            timestamp_fallback_frame_count=0,
            max_adjacent_container_pts_gap_seconds=0.04 if decoded_frame_count > 1 else None,
            container_pts_regression_count=0,
        )
        return replace(
            scan,
            version=2,
            result=replace(scan.result, version=2, detections=detections, sampling=ledger),
        )

    @staticmethod
    def _evaluate_v2(
        admission: ExternalSourceAdmission,
        scan: AdmittedDetectorScan,
        *,
        annotation: HouseholdEpisodeAnnotation | None = None,
    ) -> HouseholdTriggerCoverageReport:
        return evaluate_household_trigger_coverage(
            annotation if annotation is not None else _annotation(admission),
            scan,
            media_admission=admission,
            thresholds=(0.35, 0.55),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )

    def test_complete_on_grid_v2_preserves_v1_metrics_including_empty_samples(self) -> None:
        admission = _media_admission()
        for empty in (False, True):
            with self.subTest(empty=empty):
                v2 = self._v2_scan(admission, empty=empty)
                v1 = replace(v2, version=1, result=replace(v2.result, version=1, sampling=None))
                old_report = self._evaluate_v2(admission, v1)
                report = self._evaluate_v2(admission, v2)
                self.assertEqual(20, report.evaluated_sample_count)
                old_payload, payload = old_report.to_dict(), report.to_dict()
                for field in ("detector_scan_sha256", "report_sha256"):
                    old_payload.pop(field)
                    payload.pop(field)
                self.assertEqual(old_payload, payload)
                self.assertEqual(v2.fingerprint, report.detector_scan_sha256)
                self.assertFalse(v2.result.sampling.source_exhaustion_verified)

    def test_v2_rejects_short_or_extra_decode_even_with_identical_sample_count(self) -> None:
        admission = _media_admission()
        # 99 decoded frames still produce the same 20 calls as 100 frames.
        # Sample count alone therefore cannot bind the declared source length.
        for count in (1, 99, 101):
            with (
                self.subTest(decoded_frame_count=count),
                self.assertRaises(HouseholdGroundTruthError),
            ):
                self._evaluate_v2(
                    admission, self._v2_scan(admission, empty=True, decoded_frame_count=count)
                )

    def test_v2_rejects_off_grid_duplicate_and_shifted_pts_without_any_boxes(self) -> None:
        admission = _media_admission()
        scan = self._v2_scan(admission, empty=True)
        ledger = scan.result.sampling
        assert ledger is not None
        changed_samples = (
            (
                ledger.samples[0],
                replace(ledger.samples[1], timestamp_seconds=0.201),
                *ledger.samples[2:],
            ),
            (
                ledger.samples[0],
                replace(ledger.samples[1], timestamp_seconds=0.0),
                *ledger.samples[2:],
            ),
            tuple(
                replace(sample, timestamp_seconds=sample.timestamp_seconds + 0.001)
                for sample in ledger.samples
            ),
        )
        for samples in changed_samples:
            bad = replace(
                scan, result=replace(scan.result, sampling=replace(ledger, samples=samples))
            )
            with self.subTest(samples=samples), self.assertRaises(HouseholdGroundTruthError):
                self._evaluate_v2(admission, bad)

    def test_v2_rejects_nominal_timestamp_fallback_even_when_numerically_on_grid(self) -> None:
        admission = _media_admission()
        scan = self._v2_scan(admission, empty=True)
        ledger = scan.result.sampling
        assert ledger is not None
        samples = (
            ledger.samples[0],
            replace(ledger.samples[1], timestamp_source="nominal_fps"),
            *ledger.samples[2:],
        )
        bad = replace(
            scan,
            result=replace(
                scan.result,
                sampling=replace(ledger, samples=samples, timestamp_fallback_frame_count=1),
            ),
        )
        with self.assertRaises(HouseholdGroundTruthError):
            self._evaluate_v2(admission, bad)

    def test_v2_rejects_native_fps_mismatch_even_when_sample_step_and_pts_match(self) -> None:
        admission = _media_admission()
        scan = self._v2_scan(admission, empty=True)
        # Both 25 and 25.1 FPS round to a five-frame sampling step at 5 FPS.
        for sampling in (
            replace(scan.result.sampling, native_fps=25.1),
            replace(scan.result.sampling, native_fps_source="fallback_25"),
        ):
            bad = replace(scan, result=replace(scan.result, sampling=sampling))
            with self.subTest(sampling=sampling), self.assertRaises(HouseholdGroundTruthError):
                self._evaluate_v2(admission, bad)

    def test_v2_allows_unsampled_timestamp_fallback_without_claiming_exhaustion(self) -> None:
        admission = _media_admission()
        scan = self._v2_scan(admission, empty=True)
        # The nominal gate binds actual sampled frames, not timestamps of
        # frames at which no inference occurred. Those fallback counts remain
        # retained in the ledger without changing the evaluation denominator.
        changed = replace(
            scan,
            result=replace(
                scan.result,
                sampling=replace(scan.result.sampling, timestamp_fallback_frame_count=1),
            ),
        )
        original = self._evaluate_v2(admission, scan)
        report = self._evaluate_v2(admission, changed)
        self.assertEqual(original.points, report.points)
        self.assertEqual(20, report.evaluated_sample_count)
        self.assertEqual(1, changed.result.sampling.timestamp_fallback_frame_count)
        self.assertFalse(changed.result.sampling.source_exhaustion_verified)

    def test_v2_checks_zero_box_pts_outside_annotated_ranges_and_at_source_end(self) -> None:
        admission = _media_admission()
        annotation = replace(
            _annotation(admission), ranges=(HouseholdEpisodeRange(1.0, 2.0, "person_present"),)
        )
        scan = self._v2_scan(admission, empty=True)
        ledger = scan.result.sampling
        assert ledger is not None
        for position, timestamp in ((0, 0.001), (-1, 3.801), (-1, 4.0)):
            samples = list(ledger.samples)
            samples[position] = replace(samples[position], timestamp_seconds=timestamp)
            bad = replace(
                scan, result=replace(scan.result, sampling=replace(ledger, samples=tuple(samples)))
            )
            with (
                self.subTest(position=position, timestamp=timestamp),
                self.assertRaises(HouseholdGroundTruthError),
            ):
                self._evaluate_v2(admission, bad, annotation=annotation)

    def test_v2_checks_ledger_before_filtering_nonperson_detections(self) -> None:
        admission = _media_admission()
        scan = self._v2_scan(admission, empty=True)
        ledger = scan.result.sampling
        assert ledger is not None
        samples = (
            replace(ledger.samples[0], timestamp_seconds=0.001, detection_count=1),
            *(replace(sample, detection_start=1) for sample in ledger.samples[1:]),
        )
        bad = replace(
            scan,
            labels=("cat", "person"),
            result=replace(
                scan.result,
                detections=(Detection(0.001, "cat", 0.9, (0.1, 0.1, 0.3, 0.8)),),
                sampling=replace(ledger, samples=samples),
            ),
        )
        with self.assertRaises(HouseholdGroundTruthError):
            self._evaluate_v2(admission, bad)

    def test_rejects_pts_between_nominal_frames_instead_of_rounding_or_dropping(self) -> None:
        admission = _media_admission()
        scan = _scan(admission)
        # 2.001 would previously round onto the 2.0 s grid point; 2.03
        # would disappear because its rounded frame index is not sampled.
        for timestamp in (2.001, 2.03):
            with self.subTest(timestamp=timestamp):
                shifted = replace(
                    scan,
                    result=replace(
                        scan.result,
                        detections=(Detection(timestamp, "person", 0.9, (0.1, 0.1, 0.3, 0.8)),),
                    ),
                )
                with self.assertRaisesRegex(
                    HouseholdGroundTruthError, "exact nominal source frame"
                ):
                    evaluate_household_trigger_coverage(
                        _annotation(admission),
                        shifted,
                        media_admission=admission,
                        thresholds=(0.35, 0.55),
                        candidate_gap_seconds=0.5,
                        max_event_span_seconds=30.0,
                    )

    def test_rejects_exact_but_unsampled_frame_instead_of_dropping_candidate(self) -> None:
        admission = _media_admission()
        scan = _scan(admission)
        shifted = replace(
            scan,
            result=replace(
                scan.result,
                detections=(Detection(2.04, "person", 0.9, (0.1, 0.1, 0.3, 0.8)),),
            ),
        )
        with self.assertRaisesRegex(HouseholdGroundTruthError, "declared sampling step"):
            evaluate_household_trigger_coverage(
                _annotation(admission),
                shifted,
                media_admission=admission,
                thresholds=(0.35, 0.55),
                candidate_gap_seconds=0.5,
                max_event_span_seconds=30.0,
            )

    def test_rejects_candidate_at_or_beyond_source_end(self) -> None:
        admission = _media_admission()
        scan = _scan(admission)
        for timestamp in (4.0, 4.2):
            with self.subTest(timestamp=timestamp):
                shifted = replace(
                    scan,
                    result=replace(
                        scan.result,
                        detections=(Detection(timestamp, "person", 0.9, (0.1, 0.1, 0.3, 0.8)),),
                    ),
                )
                with self.assertRaisesRegex(
                    HouseholdGroundTruthError, "outside the admitted source"
                ):
                    evaluate_household_trigger_coverage(
                        _annotation(admission),
                        shifted,
                        media_admission=admission,
                        thresholds=(0.35, 0.55),
                        candidate_gap_seconds=0.5,
                        max_event_span_seconds=30.0,
                    )

    def test_alignment_check_includes_unannotated_and_nonperson_evidence(self) -> None:
        admission = _media_admission()
        annotation = replace(
            _annotation(admission),
            ranges=(HouseholdEpisodeRange(2.0, 4.0, "person_present"),),
        )
        scan = _scan(admission)
        for label in ("person", "cat"):
            with self.subTest(label=label):
                shifted = replace(
                    scan,
                    labels=("cat", "person"),
                    result=replace(
                        scan.result,
                        detections=(Detection(0.001, label, 0.9, (0.1, 0.1, 0.3, 0.8)),),
                    ),
                )
                with self.assertRaisesRegex(
                    HouseholdGroundTruthError, "exact nominal source frame"
                ):
                    evaluate_household_trigger_coverage(
                        annotation,
                        shifted,
                        media_admission=admission,
                        thresholds=(0.35, 0.55),
                        candidate_gap_seconds=0.5,
                        max_event_span_seconds=30.0,
                    )

    def test_aligned_float_noise_preserves_coverage_without_filling_neighbor_points(self) -> None:
        admission = _media_admission()
        scan = _scan(admission)
        noisy = replace(
            scan,
            result=replace(
                scan.result,
                detections=(Detection(2.0 + 1e-9, "person", 0.9, (0.1, 0.1, 0.3, 0.8)),),
            ),
        )
        report = evaluate_household_trigger_coverage(
            _annotation(admission),
            noisy,
            media_admission=admission,
            thresholds=(0.35, 0.55),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        self.assertEqual(20, report.evaluated_sample_count)
        for point in report.points:
            self.assertEqual(1, point.covered_positive_sample_count)
            self.assertEqual(9, point.uncovered_positive_sample_count)
            self.assertEqual(1, point.covered_person_episode_count)
            self.assertEqual(0, point.covered_danger_episode_count)

    def test_empty_scan_remains_no_evidence_on_annotation_derived_grid(self) -> None:
        admission = _media_admission()
        scan = _scan(admission)
        empty = replace(scan, result=replace(scan.result, detections=()))
        report = evaluate_household_trigger_coverage(
            _annotation(admission),
            empty,
            media_admission=admission,
            thresholds=(0.35, 0.55),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        self.assertEqual(20, report.evaluated_sample_count)
        for point in report.points:
            self.assertEqual(0, point.candidate_group_count)
            self.assertEqual(0, point.covered_positive_sample_count)
            self.assertEqual(10, point.uncovered_positive_sample_count)
            self.assertIsNone(point.maximum_detection_delay_seconds)

    def test_coverage_points_and_episode_accounting(self) -> None:
        admission = _media_admission()
        annotation = _annotation(admission)
        scan = _scan(admission)
        report = evaluate_household_trigger_coverage(
            annotation,
            scan,
            media_admission=admission,
            thresholds=(0.35, 0.55, 0.95),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        self.assertEqual(20, report.evaluated_sample_count)
        self.assertEqual(10, report.positive_sample_count)
        self.assertEqual(10, report.negative_sample_count)
        self.assertEqual(0, report.unannotated_sample_count)
        self.assertEqual(1, report.person_episode_count)
        self.assertEqual(1, report.danger_episode_count)
        self.assertEqual(5.0, report.effective_sample_fps)
        self.assertEqual(("night_vision", "partial_occlusion"), report.scenario_tags)

        low, medium, high = report.points
        self.assertEqual(0.35, low.candidate_start_confidence)
        self.assertEqual(
            (3, 3, 0),
            (
                low.candidate_group_count,
                low.emitted_group_count,
                low.suppressed_group_count,
            ),
        )
        self.assertEqual(5, low.covered_positive_sample_count)
        self.assertEqual(0.5, low.positive_sample_recall)
        self.assertEqual(1, low.activated_negative_sample_count)
        self.assertEqual(0.1, low.negative_sample_false_activation_rate)
        self.assertEqual(1, low.covered_person_episode_count)
        self.assertEqual(1.0, low.person_episode_recall)
        self.assertEqual(1, low.covered_danger_episode_count)
        self.assertEqual(1.0, low.danger_episode_recall)
        self.assertEqual(0.0, low.maximum_detection_delay_seconds)
        self.assertEqual(3, low.longest_uncovered_positive_samples)
        self.assertEqual(0.6, low.longest_uncovered_positive_seconds)

        self.assertEqual(
            (3, 2, 1),
            (
                medium.candidate_group_count,
                medium.emitted_group_count,
                medium.suppressed_group_count,
            ),
        )
        self.assertEqual(1.0, medium.person_episode_recall)
        self.assertEqual(1.0, medium.danger_episode_recall)
        self.assertEqual(0, medium.activated_negative_sample_count)
        self.assertEqual(0.0, medium.negative_sample_false_activation_rate)

        self.assertEqual(0, high.emitted_group_count)
        self.assertEqual(0.0, high.person_episode_recall)
        self.assertEqual(0.0, high.danger_episode_recall)
        self.assertIsNone(high.maximum_detection_delay_seconds)
        self.assertEqual(10, high.longest_uncovered_positive_samples)
        self.assertEqual(2.0, high.longest_uncovered_positive_seconds)

        schema = json.loads(
            Path("schemas/household-trigger-coverage-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))
        self.assertEqual(
            report,
            HouseholdTriggerCoverageReport.from_json(report.to_json()),
        )
        tampered = json.loads(report.to_json())
        tampered["points"][1]["covered_danger_episode_count"] = 0
        with self.assertRaises(HouseholdGroundTruthError):
            HouseholdTriggerCoverageReport.from_dict(tampered)

    def test_requires_approved_annotation_and_exact_binding(self) -> None:
        admission = _media_admission()
        pending = _annotation(admission, review_state="pending")
        scan = _scan(admission)
        with self.assertRaises(HouseholdGroundTruthError):
            evaluate_household_trigger_coverage(
                pending,
                scan,
                media_admission=admission,
                thresholds=(0.35, 0.55),
                candidate_gap_seconds=0.5,
                max_event_span_seconds=30.0,
            )
        wrong_camera_scan = _scan(admission, camera_id="camera_02")
        annotation = _annotation(admission)
        with self.assertRaises(HouseholdGroundTruthError):
            evaluate_household_trigger_coverage(
                annotation,
                wrong_camera_scan,
                media_admission=admission,
                thresholds=(0.35, 0.55),
                candidate_gap_seconds=0.5,
                max_event_span_seconds=30.0,
            )
        other_admission = _media_admission(variant="second")
        unbound_scan = AdmittedDetectorScan(
            source_admission_sha256=other_admission.fingerprint,
            labels=("person",),
            confidence_threshold=0.35,
            requested_sample_fps=5.0,
            result=scan.result,
        )
        with self.assertRaises(HouseholdGroundTruthError):
            evaluate_household_trigger_coverage(
                annotation,
                unbound_scan,
                media_admission=admission,
                thresholds=(0.35, 0.55),
                candidate_gap_seconds=0.5,
                max_event_span_seconds=30.0,
            )
        with self.assertRaises(HouseholdGroundTruthError):
            evaluate_household_trigger_coverage(
                annotation,
                scan,
                media_admission=admission,
                thresholds=(0.40, 0.55),
                candidate_gap_seconds=0.5,
                max_event_span_seconds=30.0,
            )

    def test_positive_range_without_grid_points_fails_closed(self) -> None:
        admission = _media_admission()
        annotation = _annotation(admission)
        sparse = replace(
            annotation,
            ranges=(
                HouseholdEpisodeRange(0.0, 1.0, "empty_room"),
                HouseholdEpisodeRange(1.02, 1.04, "person_present"),
                HouseholdEpisodeRange(1.04, 4.0, "empty_room"),
            ),
        )
        scan = _scan(admission)
        with self.assertRaises(HouseholdGroundTruthError):
            evaluate_household_trigger_coverage(
                sparse,
                scan,
                media_admission=admission,
                thresholds=(0.35, 0.55),
                candidate_gap_seconds=0.5,
                max_event_span_seconds=30.0,
            )

    def test_pure_negative_annotation_evaluates_without_positives(self) -> None:
        admission = _media_admission()
        annotation = replace(
            _annotation(admission),
            ranges=(HouseholdEpisodeRange(0.0, 4.0, "empty_room"),),
        )
        report = evaluate_household_trigger_coverage(
            annotation,
            _scan(admission),
            media_admission=admission,
            thresholds=(0.35, 0.55),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        self.assertEqual(0, report.positive_sample_count)
        self.assertEqual(0, report.person_episode_count)
        self.assertEqual(0, report.danger_episode_count)
        for point in report.points:
            self.assertIsNone(point.positive_sample_recall)
            self.assertIsNone(point.person_episode_recall)
            self.assertIsNone(point.danger_episode_recall)
            self.assertIsNone(point.maximum_detection_delay_seconds)


class HouseholdTriggerSelectionTests(unittest.TestCase):
    def test_policy_requires_full_danger_recall(self) -> None:
        with self.assertRaises(HouseholdGroundTruthError):
            HouseholdTriggerPolicy(
                minimum_danger_episode_recall=0.9,
                minimum_person_episode_recall=1.0,
                minimum_positive_sample_recall=0.4,
                maximum_detection_delay_seconds=0.0,
                maximum_longest_uncovered_positive_seconds=0.7,
            )
        with self.assertRaises(HouseholdGroundTruthError):
            HouseholdTriggerPolicy(
                minimum_danger_episode_recall=1.0,
                minimum_person_episode_recall=1.0,
                minimum_positive_sample_recall=0.4,
                maximum_detection_delay_seconds=0.0,
                maximum_longest_uncovered_positive_seconds=0.7,
                maximum_negative_false_activation_rate=0.0,
            )

    def test_selects_threshold_that_keeps_episodes_and_suppresses_negatives(
        self,
    ) -> None:
        admission = _media_admission()
        annotation = _annotation(admission)
        report = evaluate_household_trigger_coverage(
            annotation,
            _scan(admission),
            media_admission=admission,
            thresholds=(0.35, 0.55, 0.95),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        policy = HouseholdTriggerPolicy(
            minimum_danger_episode_recall=1.0,
            minimum_person_episode_recall=1.0,
            minimum_positive_sample_recall=0.4,
            maximum_detection_delay_seconds=0.0,
            maximum_longest_uncovered_positive_seconds=0.7,
            maximum_negative_false_activation_rate=0.0,
            require_negative_samples=True,
        )
        selected, feasible = select_household_trigger_operating_point(
            report,
            policy,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(1, feasible)
        assert selected is not None
        self.assertEqual(0.55, selected.candidate_start_confidence)

        recall_first_policy = replace(
            policy,
            maximum_negative_false_activation_rate=None,
            require_negative_samples=False,
        )
        recall_selected, recall_feasible = select_household_trigger_operating_point(
            report,
            recall_first_policy,
        )
        self.assertIsNotNone(recall_selected)
        self.assertEqual(2, recall_feasible)
        assert recall_selected is not None
        # Equal recall keeps accuracy first, then fewer negative activations.
        self.assertEqual(0.55, recall_selected.candidate_start_confidence)

    def test_no_feasible_point_when_danger_episode_is_missed(self) -> None:
        admission = _media_admission()
        annotation = _annotation(admission)
        missing_danger_scan = AdmittedDetectorScan(
            source_admission_sha256=admission.fingerprint,
            labels=("person",),
            confidence_threshold=0.35,
            requested_sample_fps=5.0,
            result=DetectorScanResult(
                camera_id="camera_01",
                detector_adapter="yolo_command",
                model_id="yolo11n",
                model_revision="fixture",
                model_sha256=hashlib.sha256(b"weights").hexdigest(),
                detections=(
                    Detection(0.0, "person", 0.40, (0.10, 0.10, 0.30, 0.80)),
                    Detection(2.0, "person", 0.90, (0.12, 0.12, 0.34, 0.82)),
                    Detection(2.2, "person", 0.90, (0.14, 0.12, 0.36, 0.82)),
                ),
            ),
        )
        report = evaluate_household_trigger_coverage(
            annotation,
            missing_danger_scan,
            media_admission=admission,
            thresholds=(0.35, 0.55),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        self.assertEqual(0.0, report.points[0].danger_episode_recall)
        policy = HouseholdTriggerPolicy(
            minimum_danger_episode_recall=1.0,
            minimum_person_episode_recall=1.0,
            minimum_positive_sample_recall=0.4,
            maximum_detection_delay_seconds=0.0,
            maximum_longest_uncovered_positive_seconds=0.7,
            maximum_negative_false_activation_rate=0.0,
            require_negative_samples=True,
        )
        selected, feasible = select_household_trigger_operating_point(
            report,
            policy,
        )
        self.assertIsNone(selected)
        self.assertEqual(0, feasible)


class HouseholdTriggerComparisonTests(unittest.TestCase):
    def test_compares_same_grid_models_without_selecting_a_winner(self) -> None:
        admission = _media_admission()
        annotation = _annotation(admission)
        first_scan = _scan(admission)
        second_scan = AdmittedDetectorScan(
            source_admission_sha256=admission.fingerprint,
            labels=("person",),
            confidence_threshold=0.35,
            requested_sample_fps=5.0,
            result=replace(
                first_scan.result,
                model_id="yolo26m",
                model_sha256=hashlib.sha256(b"other-weights").hexdigest(),
            ),
        )
        thresholds = (0.35, 0.55)
        first_report = evaluate_household_trigger_coverage(
            annotation,
            first_scan,
            media_admission=admission,
            thresholds=thresholds,
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        second_report = evaluate_household_trigger_coverage(
            annotation,
            second_scan,
            media_admission=admission,
            thresholds=thresholds,
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        comparison = compare_household_trigger_coverage((first_report, second_report))
        self.assertTrue(comparison["same_grid_and_media_verified"])
        self.assertFalse(comparison["winner_selected"])
        self.assertFalse(comparison["deployment_approved"])
        self.assertTrue(comparison["dataset_specific_only"])
        self.assertEqual(
            ["yolo11n", "yolo26m"],
            [model["model_id"] for model in comparison["models"]],
        )
        self.assertEqual(
            [0.35, 0.55],
            [point["candidate_start_confidence"] for point in comparison["points_by_threshold"]],
        )
        self.assertEqual(2, len(comparison["points_by_threshold"][0]["models"]))

    def test_rejects_grid_threshold_and_model_identity_mismatches(self) -> None:
        admission = _media_admission()
        annotation = _annotation(admission)
        first_report = evaluate_household_trigger_coverage(
            annotation,
            _scan(admission),
            media_admission=admission,
            thresholds=(0.35, 0.55),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        with self.assertRaises(HouseholdGroundTruthError):
            compare_household_trigger_coverage((first_report,))
        with self.assertRaises(HouseholdGroundTruthError):
            compare_household_trigger_coverage((first_report, first_report))
        different_thresholds = evaluate_household_trigger_coverage(
            annotation,
            _scan(admission),
            media_admission=admission,
            thresholds=(0.35, 0.65),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        with self.assertRaises(HouseholdGroundTruthError):
            compare_household_trigger_coverage((first_report, different_thresholds))
        other_annotation = replace(
            annotation,
            annotation_id="household_annotation_0002",
            ranges=(HouseholdEpisodeRange(0.0, 4.0, "empty_room"),),
        )
        other_report = evaluate_household_trigger_coverage(
            other_annotation,
            _scan(admission),
            media_admission=admission,
            thresholds=(0.35, 0.55),
            candidate_gap_seconds=0.5,
            max_event_span_seconds=30.0,
        )
        with self.assertRaises(HouseholdGroundTruthError):
            compare_household_trigger_coverage((first_report, other_report))


class HouseholdGroundTruthCLITests(unittest.TestCase):
    def test_cli_round_trip(self) -> None:
        admission = _media_admission()
        annotation = _annotation(admission)
        scan = _scan(admission)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_path = root / "annotation.json"
            admission_path = root / "media-admission.json"
            scan_path = root / "scan.json"
            coverage_path = root / "coverage.json"
            annotation.write_new(annotation_path)
            admission_path.write_text(
                admission.to_json() + "\n",
                encoding="utf-8",
            )
            scan.write_new(scan_path)

            validate_out = StringIO()
            with redirect_stdout(validate_out):
                validate_code = main(
                    [
                        "validate-household-episode-annotation",
                        str(annotation_path),
                    ]
                )
            self.assertEqual(0, validate_code)
            self.assertEqual(
                annotation.fingerprint,
                json.loads(validate_out.getvalue())["annotation_sha256"],
            )

            evaluate_out = StringIO()
            with redirect_stdout(evaluate_out):
                code = main(
                    [
                        "evaluate-household-trigger-coverage",
                        "--annotation",
                        str(annotation_path),
                        "--media-admission",
                        str(admission_path),
                        "--detector-scan",
                        str(scan_path),
                        "--start-threshold",
                        "0.35",
                        "--start-threshold",
                        "0.55",
                        "--start-threshold",
                        "0.95",
                        "--candidate-gap-seconds",
                        "0.5",
                        "--max-event-span-seconds",
                        "30",
                        "--output",
                        str(coverage_path),
                    ]
                )
            summary = json.loads(evaluate_out.getvalue())
            self.assertEqual(0, code)
            self.assertNotIn(str(root), evaluate_out.getvalue())
            report = HouseholdTriggerCoverageReport.read(coverage_path)
            self.assertEqual(report.fingerprint, summary["report_sha256"])

            coverage_out = StringIO()
            with redirect_stdout(coverage_out):
                coverage_code = main(["validate-household-trigger-coverage", str(coverage_path)])
            self.assertEqual(0, coverage_code)
            self.assertEqual(summary, json.loads(coverage_out.getvalue()))

            select_out = StringIO()
            with redirect_stdout(select_out):
                select_code = main(
                    [
                        "select-household-trigger-operating-point",
                        str(coverage_path),
                        "--minimum-danger-episode-recall",
                        "1",
                        "--minimum-person-episode-recall",
                        "1",
                        "--minimum-positive-sample-recall",
                        "0.4",
                        "--maximum-detection-delay-seconds",
                        "0",
                        "--maximum-longest-uncovered-positive-seconds",
                        "0.7",
                        "--maximum-negative-false-activation-rate",
                        "0",
                        "--require-negative-samples",
                    ]
                )
            selection = json.loads(select_out.getvalue())
            self.assertEqual(0, select_code)
            self.assertEqual(1, selection["feasible_point_count"])
            self.assertEqual(
                0.55,
                selection["selected_point"]["candidate_start_confidence"],
            )
            self.assertEqual(
                HOUSEHOLD_TRIGGER_SELECTION_RULE,
                selection["selection_rule"],
            )
            self.assertTrue(selection["dataset_specific_only"])
            self.assertFalse(selection["deployment_approved"])
            self.assertFalse(selection["household_configuration_changed"])
            self.assertFalse(selection["danger_recognition_accuracy_evaluated"])


if __name__ == "__main__":
    unittest.main()
