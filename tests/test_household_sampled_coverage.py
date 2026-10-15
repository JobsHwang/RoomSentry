from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace

from baby_monitor.detector_protocol import AdmittedDetectorScan, DetectorScanResult
from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from baby_monitor.domain import Detection
from baby_monitor.household_ground_truth import HouseholdEpisodeRange
from baby_monitor.household_sampled_coverage import evaluate_household_sampled_coverage
from tests.test_household_ground_truth import _annotation, _media_admission


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _sampled_scan(admission, records, *, native_fps_source="reported", labels=("person",)):
    """Synthetic invocation slices; duplicate timestamps never coalesce boxes."""
    detections = []
    samples = []
    for index, (seconds, boxes, clock) in enumerate(records):
        samples.append(
            DetectorSampleFrame(
                decoded_frame_index=index * 25,
                timestamp_seconds=seconds,
                timestamp_source=clock,
                detection_start=len(detections),
                detection_count=len(boxes),
            )
        )
        detections.extend(
            Detection(seconds, label, confidence, (0.1, 0.1, 0.3, 0.8))
            for label, confidence in boxes
        )
    ledger = DetectorSamplingLedger(
        native_fps=25.0,
        native_fps_source=native_fps_source,
        requested_sample_fps=1.0,
        frame_step=25,
        decoded_frame_count=(len(samples) - 1) * 25 + 1,
        samples=tuple(samples),
        timestamp_fallback_frame_count=sum(s.timestamp_source == "nominal_fps" for s in samples),
        max_adjacent_container_pts_gap_seconds=None,
        container_pts_regression_count=0,
    )
    return AdmittedDetectorScan(
        source_admission_sha256=admission.fingerprint,
        labels=labels,
        confidence_threshold=0.2,
        requested_sample_fps=1.0,
        version=2,
        result=DetectorScanResult(
            camera_id="camera_01",
            detector_adapter="synthetic_detector",
            model_id="fixture",
            model_revision="fixture",
            model_sha256="b" * 64,
            detections=tuple(detections),
            version=2,
            sampling=ledger,
        ),
    )


class HouseholdSampledCoverageTests(unittest.TestCase):
    REFERENCE_SHA256 = "c" * 64

    def setUp(self) -> None:
        self.admission = _media_admission()
        self.annotation = _annotation(self.admission)

    def binding(self, *, annotation=None, offset=0.0):
        annotation = annotation or self.annotation
        return {
            "version": 1,
            "annotation_sha256": annotation.fingerprint,
            "media_admission_sha256": self.admission.fingerprint,
            "reference_sha256": self.REFERENCE_SHA256,
            "source_pts_at_annotation_zero_seconds": offset,
            "alignment_confirmed": True,
        }

    def evaluate(self, records, *, annotation=None, offset=0.0, threshold=0.5, **scan_options):
        annotation = annotation or self.annotation
        scan = _sampled_scan(self.admission, records, **scan_options)
        return evaluate_household_sampled_coverage(
            annotation,
            scan,
            media_admission=self.admission,
            timeline_binding=self.binding(annotation=annotation, offset=offset),
            reference_sha256=self.REFERENCE_SHA256,
            confidence_threshold=threshold,
        )

    def test_unobserved_positive_ranges_remain_in_fixed_interval_denominator(self):
        annotation = replace(
            self.annotation,
            ranges=(
                HouseholdEpisodeRange(0, 1, "empty_room"),
                HouseholdEpisodeRange(1, 1.1, "person_present"),
                HouseholdEpisodeRange(1.1, 2, "person_present"),
                HouseholdEpisodeRange(3, 4, "danger_event"),
            ),
        )
        result = self.evaluate(
            (
                (0.25, (("person", 0.9),), "container_pts"),
                (1.2, (), "container_pts"),
                (2.5, (("person", 0.9),), "container_pts"),
                (3.5, (("person", 0.9),), "nominal_fps"),
            ),
            annotation=annotation,
        )
        self.assertEqual(3, result["positive_interval_count"])
        self.assertEqual(0, result["detected_positive_interval_count"])
        self.assertEqual(1, result["observed_positive_without_detection_count"])
        self.assertEqual(2, result["unobserved_positive_interval_count"])
        self.assertEqual(0.0, result["positive_interval_detection_rate"])
        self.assertEqual(
            ["person_detected", "unobserved", "observed_without_person_detection", "unobserved"],
            [item["state"] for item in result["intervals"]],
        )
        self.assertEqual(
            [1, 0, 1, 0], [item["actual_sample_count"] for item in result["intervals"]]
        )
        self.assertEqual(1, result["unknown_clock_sample_count"])
        self.assertEqual(1, result["unannotated_sample_count"])
        self.assertEqual(1, result["positive_sample_count"])
        self.assertEqual(1, result["negative_sample_count"])
        self.assertEqual(1, result["activated_negative_sample_count"])
        self.assertEqual(1, result["zero_box_sample_count"])
        self.assertEqual(4, result["actual_sample_count"])

    def test_interval_detection_rate_does_not_drop_short_unsampled_positive(self):
        annotation = replace(
            self.annotation,
            ranges=(
                HouseholdEpisodeRange(1.05, 1.1, "person_present"),
                HouseholdEpisodeRange(2, 3, "person_present"),
            ),
        )
        result = self.evaluate(
            (
                (1.0, (), "container_pts"),
                (1.2, (), "container_pts"),
                (2.4, (("person", 0.8),), "container_pts"),
            ),
            annotation=annotation,
        )
        self.assertEqual(2, result["positive_interval_count"])
        self.assertEqual(1, result["detected_positive_interval_count"])
        self.assertEqual(1, result["unobserved_positive_interval_count"])
        self.assertEqual(0.5, result["positive_interval_detection_rate"])
        self.assertAlmostEqual(0.4, result["intervals"][1]["first_person_detection_delay_seconds"])

    def test_synthetic_sparse_and_dense_calls_share_fixed_interval_denominator(self):
        annotation = replace(
            self.annotation,
            ranges=(
                HouseholdEpisodeRange(1.05, 1.1, "person_present"),
                HouseholdEpisodeRange(2, 3, "person_present"),
            ),
        )
        sparse = self.evaluate(
            (
                (1.0, (), "container_pts"),
                (1.2, (), "container_pts"),
                (2.4, (("person", 0.8),), "container_pts"),
            ),
            annotation=annotation,
        )
        dense = self.evaluate(
            (
                (1.0, (), "container_pts"),
                (1.075, (), "container_pts"),
                (1.2, (), "container_pts"),
                (2.4, (("person", 0.8),), "container_pts"),
            ),
            annotation=annotation,
        )
        self.assertEqual(sparse["annotation_sha256"], dense["annotation_sha256"])
        self.assertEqual(sparse["timeline_binding_sha256"], dense["timeline_binding_sha256"])
        self.assertEqual(
            (2, 2), (sparse["positive_interval_count"], dense["positive_interval_count"])
        )
        self.assertEqual((1, 2), (sparse["positive_sample_count"], dense["positive_sample_count"]))
        self.assertEqual("unobserved", sparse["intervals"][0]["state"])
        self.assertEqual("observed_without_person_detection", dense["intervals"][0]["state"])
        self.assertEqual(
            (1, 0),
            (
                sparse["unobserved_positive_interval_count"],
                dense["unobserved_positive_interval_count"],
            ),
        )
        self.assertEqual(
            (0.5, 0.5),
            (sparse["positive_interval_detection_rate"], dense["positive_interval_detection_rate"]),
        )
        # Pure synthetic calls test denominator semantics. They are not real
        # denser model runs, an operating-point winner, or an accuracy gain.

    def test_actual_pts_uses_explicit_offset_not_first_sample_or_nominal_grid(self):
        result = self.evaluate(
            (
                (10.5, (), "container_pts"),
                (12.25, (("person", 0.9),), "container_pts"),
                (13.2, (("person", 0.8),), "container_pts"),
            ),
            offset=10.0,
            native_fps_source="fallback_25",
        )
        self.assertEqual(3, result["actual_sample_count"])
        self.assertEqual(2, result["detected_positive_interval_count"])
        self.assertEqual(0, result["outside_source_sample_count"])
        self.assertAlmostEqual(0.25, result["intervals"][1]["first_person_detection_delay_seconds"])
        self.assertAlmostEqual(0.2, result["intervals"][2]["first_person_detection_delay_seconds"])
        # 51 decoded frames versus annotation's 100 is intentionally accepted:
        # these are actual point observations, not a completeness assertion.
        self.assertEqual(1.0, result["positive_interval_detection_rate"])

    def test_half_open_boundaries_outside_source_and_fallback_partition(self):
        result = self.evaluate(
            (
                (0.0, (("person", 0.9),), "container_pts"),
                (2.0, (("person", 0.9),), "container_pts"),
                (3.0, (("person", 0.9),), "container_pts"),
                (4.0, (("person", 0.9),), "container_pts"),
                (4.1, (), "nominal_fps"),
            )
        )
        self.assertEqual(1, result["negative_sample_count"])
        self.assertEqual(2, result["positive_sample_count"])
        self.assertEqual(1, result["outside_source_sample_count"])
        self.assertEqual(1, result["unknown_clock_sample_count"])
        self.assertEqual(
            5,
            sum(
                result[name]
                for name in (
                    "positive_sample_count",
                    "negative_sample_count",
                    "outside_source_sample_count",
                    "unannotated_sample_count",
                    "unknown_clock_sample_count",
                )
            ),
        )
        self.assertEqual(1, result["activated_negative_interval_count"])
        self.assertEqual(0.0, result["intervals"][1]["first_person_detection_delay_seconds"])

    def test_duplicate_pts_are_distinct_invocations_not_a_timestamp_dictionary(self):
        result = self.evaluate(
            (
                (2.25, (), "container_pts"),
                (2.25, (("person", 0.9),), "container_pts"),
            )
        )
        self.assertEqual(2, result["actual_sample_count"])
        self.assertEqual(2, result["positive_sample_count"])
        self.assertEqual(1, result["detected_positive_sample_count"])
        self.assertEqual(1, result["zero_box_sample_count"])
        self.assertEqual(2, result["intervals"][1]["actual_sample_count"])
        self.assertEqual(1, result["intervals"][1]["person_detection_sample_count"])

    def test_offset_source_lower_and_upper_bounds_remain_half_open(self):
        result = self.evaluate(
            (
                (9.9, (("person", 0.9),), "container_pts"),
                (10.0, (("person", 0.9),), "container_pts"),
                (13.9, (), "container_pts"),
                (14.0, (("person", 0.9),), "container_pts"),
            ),
            offset=10.0,
        )
        self.assertEqual(2, result["outside_source_sample_count"])
        self.assertEqual(1, result["negative_sample_count"])
        self.assertEqual(1, result["positive_sample_count"])
        self.assertEqual(0, result["detected_positive_sample_count"])

    def test_offset_cannot_collapse_tiny_annotation_interval(self):
        annotation = replace(
            self.annotation, ranges=(HouseholdEpisodeRange(0.0, 1e-12, "person_present"),)
        )
        with self.assertRaises(ValueError):
            self.evaluate(((86400.0, (), "container_pts"),), annotation=annotation, offset=86400.0)

    def test_nonperson_and_below_threshold_boxes_are_not_raw_zero_box_samples(self):
        result = self.evaluate(
            (
                (2.1, (("cat", 0.9),), "container_pts"),
                (2.2, (("person", 0.4),), "container_pts"),
                (2.3, (), "container_pts"),
            ),
            labels=("cat", "person"),
        )
        self.assertEqual(3, result["positive_sample_count"])
        self.assertEqual(0, result["detected_positive_sample_count"])
        self.assertEqual(1, result["zero_box_sample_count"])
        self.assertEqual("observed_without_person_detection", result["intervals"][1]["state"])

    def test_all_unknown_clocks_do_not_become_zero_observations(self):
        result = self.evaluate(((2.5, (), "nominal_fps"),))
        self.assertEqual(1, result["actual_sample_count"])
        self.assertEqual(1, result["unknown_clock_sample_count"])
        self.assertEqual(0, result["positive_sample_count"])
        self.assertEqual(2, result["unobserved_positive_interval_count"])
        self.assertEqual(1, result["unobserved_negative_interval_count"])
        self.assertTrue(all(item["state"] == "unobserved" for item in result["intervals"]))

    def test_no_positive_annotations_has_null_positive_rate(self):
        annotation = replace(self.annotation, ranges=(HouseholdEpisodeRange(0, 4, "empty_room"),))
        result = self.evaluate(((0.1, (), "container_pts"),), annotation=annotation)
        self.assertEqual(0, result["positive_interval_count"])
        self.assertIsNone(result["positive_interval_detection_rate"])
        self.assertEqual(1, result["negative_interval_count"])

    def test_hashes_bind_original_inputs_and_authority_remains_false(self):
        scan = _sampled_scan(self.admission, ((2.1, (("person", 0.8),), "container_pts"),))
        binding = self.binding()
        result = evaluate_household_sampled_coverage(
            self.annotation,
            scan,
            media_admission=self.admission,
            timeline_binding=binding,
            reference_sha256=self.REFERENCE_SHA256,
            confidence_threshold=0.5,
        )
        for field, expected in (
            ("detector_scan_sha256", scan.fingerprint),
            ("annotation_sha256", self.annotation.fingerprint),
            ("media_admission_sha256", self.admission.fingerprint),
            ("timeline_binding_sha256", _sha(binding)),
            ("reference_sha256", self.REFERENCE_SHA256),
        ):
            self.assertEqual(expected, result[field])
        self.assertEqual(
            _sha({k: v for k, v in result.items() if k != "report_sha256"}), result["report_sha256"]
        )
        for field in (
            "source_exhaustion_verified",
            "baby_identity_evaluated",
            "box_localization_evaluated",
            "danger_recognition_evaluated",
            "deployment_approved",
        ):
            self.assertIs(False, result[field])
        self.assertIs(False, result["reference_semantics_verified"])
        self.assertIs(True, result["alignment_asserted_by_input"])
        self.assertIs(True, result["human_ground_truth"])
        json.dumps(result, allow_nan=False)

    def test_binding_must_be_exact_confirmed_and_match_all_three_sources(self):
        scan = _sampled_scan(self.admission, ((2.1, (), "container_pts"),))
        cases = (
            {"alignment_confirmed": False},
            {"alignment_confirmed": 1},
            {"version": True},
            {"annotation_sha256": "f" * 64},
            {"media_admission_sha256": "f" * 64},
            {"reference_sha256": "f" * 64},
            {"unexpected": "private/path"},
            {"source_pts_at_annotation_zero_seconds": True},
            {"source_pts_at_annotation_zero_seconds": float("nan")},
            {"source_pts_at_annotation_zero_seconds": -1},
            {"source_pts_at_annotation_zero_seconds": 86401},
        )
        for changes in cases:
            with self.subTest(fields=tuple(changes)), self.assertRaises(ValueError) as error:
                evaluate_household_sampled_coverage(
                    self.annotation,
                    scan,
                    media_admission=self.admission,
                    timeline_binding=self.binding() | changes,
                    reference_sha256=self.REFERENCE_SHA256,
                    confidence_threshold=0.5,
                )
            self.assertNotIn("private/path", str(error.exception))
        with self.assertRaises(ValueError):
            evaluate_household_sampled_coverage(
                self.annotation,
                scan,
                media_admission=self.admission,
                timeline_binding=self.binding(),
                reference_sha256="f" * 64,
                confidence_threshold=0.5,
            )

    def test_pending_annotations_v1_wrong_camera_and_nonperson_scan_are_rejected(self):
        scan = _sampled_scan(self.admission, ((2.1, (), "container_pts"),))
        pending = _annotation(self.admission, review_state="pending")
        cases = (
            (pending, scan),
            (
                self.annotation,
                replace(scan, version=1, result=replace(scan.result, version=1, sampling=None)),
            ),
            (self.annotation, replace(scan, result=replace(scan.result, camera_id="camera_other"))),
            (self.annotation, replace(scan, source_admission_sha256="f" * 64)),
            (self.annotation, replace(scan, labels=("cat",))),
        )
        for index, (annotation, candidate) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                evaluate_household_sampled_coverage(
                    annotation,
                    candidate,
                    media_admission=self.admission,
                    timeline_binding=self.binding(annotation=annotation),
                    reference_sha256=self.REFERENCE_SHA256,
                    confidence_threshold=0.5,
                )

    def test_threshold_cannot_invent_detections_below_the_stored_floor(self):
        records = ((2.1, (("person", 0.5),), "container_pts"),)
        for invalid in (0.19, 1.01, True, float("nan"), float("inf")):
            with self.subTest(threshold=invalid), self.assertRaises(ValueError):
                self.evaluate(records, threshold=invalid)
        self.assertEqual(1, self.evaluate(records, threshold=0.5)["detected_positive_sample_count"])
        self.assertEqual(
            0, self.evaluate(records, threshold=0.5001)["detected_positive_sample_count"]
        )


if __name__ == "__main__":
    unittest.main()
