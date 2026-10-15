from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace

from baby_monitor.detector_protocol import AdmittedDetectorScan, DetectorScanResult
from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from scripts.detector_timeline_diagnostic import (
    MAX_REFERENCE_FRAMES,
    diagnose_detector_timeline,
    diagnose_packet_timestamps,
)


def scan(timestamps=(0.0, 1.0, 2.0), *, fallback_indices=(), frame_step=1):
    ledger = DetectorSamplingLedger(
        native_fps=frame_step,
        native_fps_source="reported",
        requested_sample_fps=1,
        frame_step=frame_step,
        decoded_frame_count=(len(timestamps) - 1) * frame_step + 1,
        samples=tuple(
            DetectorSampleFrame(
                index * frame_step,
                timestamp,
                "nominal_fps" if index in fallback_indices else "container_pts",
                0,
                0,
            )
            for index, timestamp in enumerate(timestamps)
        ),
        timestamp_fallback_frame_count=len(fallback_indices),
        max_adjacent_container_pts_gap_seconds=None,
        container_pts_regression_count=0,
    )
    return AdmittedDetectorScan(
        source_admission_sha256="a" * 64,
        labels=("person",),
        confidence_threshold=0.35,
        requested_sample_fps=1,
        result=DetectorScanResult(
            camera_id="camera_01",
            detector_adapter="synthetic_detector",
            model_id="fixture",
            model_revision="test",
            model_sha256="b" * 64,
            detections=(),
            version=2,
            sampling=ledger,
        ),
        version=2,
    )


def diagnose(value, reference):
    return diagnose_detector_timeline(value, reference, reference_source_admission_sha256="a" * 64)


class DetectorTimelineDiagnosticTests(unittest.TestCase):
    def test_duplicate_timestamps_keep_distinct_frame_indices(self):
        result = diagnose(scan((0, 0, 1)), (0, 0, 1))
        self.assertEqual(3, result["matched_sample_count"])
        self.assertEqual(1, result["reference_adjacent_duplicate_pts_count"])
        self.assertTrue(result["all_sample_timestamps_match"])

    def test_off_nominal_samples_match_by_frame_index(self):
        result = diagnose(scan((5, 5.7, 7.9), frame_step=2), (5, 5.1, 5.7, 7.2, 7.9))
        self.assertEqual(3, result["matched_sample_count"])
        self.assertTrue(result["decoded_frame_count_matches_reference"])
        self.assertAlmostEqual(2.2, result["sampled_max_adjacent_timestamp_gap_seconds"])
        self.assertEqual([2, 3], result["reference_max_gap_first_frame_indices"])

    def test_count_match_is_independent_of_matching_sample_timestamps(self):
        result = diagnose(scan(), (0, 1, 2, 3))
        self.assertFalse(result["decoded_frame_count_matches_reference"])
        self.assertTrue(result["all_sample_timestamps_match"])
        short = diagnose(scan(), (0, 1))
        self.assertEqual(1, short["missing_reference_sample_count"])
        self.assertFalse(short["all_sample_timestamps_match"])

    def test_fallback_and_missing_reference_are_unknown_not_matches(self):
        result = diagnose(scan(fallback_indices=(1,)), (0, None, None))
        self.assertEqual(1, result["matched_sample_count"])
        self.assertEqual(1, result["fallback_sample_count"])
        self.assertEqual(1, result["missing_reference_sample_count"])
        self.assertEqual(0, result["mismatched_sample_count"])
        self.assertFalse(result["all_sample_timestamps_match"])
        self.assertEqual(2, result["reference_missing_pts_count"])
        self.assertIsNone(result["reference_max_adjacent_nonnegative_pts_gap_seconds"])
        self.assertEqual(
            3,
            sum(
                result[key]
                for key in (
                    "matched_sample_count",
                    "mismatched_sample_count",
                    "fallback_sample_count",
                    "missing_reference_sample_count",
                )
            ),
        )

    def test_zero_point_offsets_are_not_automatically_corrected(self):
        result = diagnose(scan(), (10, 11, 12))
        self.assertEqual(3, result["mismatched_sample_count"])
        self.assertEqual(0, result["matched_sample_count"])
        self.assertEqual(10, result["reference_first_valid_pts_seconds"])

    def test_absolute_tolerance_has_no_relative_component(self):
        result = diagnose(scan((1e6,)), (1e6 + 0.001,))
        self.assertEqual(1, result["mismatched_sample_count"])
        self.assertEqual(1, diagnose(scan((0,)), (0.000009,))["matched_sample_count"])

    def test_reference_regressions_and_missing_break_adjacency(self):
        result = diagnose(scan((0,)), (0, 2, 1, None, 30, 32))
        self.assertEqual(1, result["reference_pts_regression_count"])
        self.assertEqual(3, result["reference_adjacent_valid_pair_count"])
        self.assertEqual(2, result["reference_max_adjacent_nonnegative_pts_gap_seconds"])
        self.assertEqual([0, 1], result["reference_max_gap_first_frame_indices"])

    def test_details_are_bounded_but_denominator_is_complete(self):
        result = diagnose(scan(tuple(range(30))), tuple(value + 0.2 for value in range(30)))
        self.assertEqual(30, result["mismatched_sample_count"])
        self.assertEqual(16, len(result["mismatch_details"]))
        self.assertTrue(result["mismatch_details_truncated"])
        self.assertEqual(15, result["mismatch_details"][-1]["decoded_frame_index"])

    def test_reference_binding_and_v2_are_required(self):
        value = scan()
        for fingerprint in ("b" * 64, "bad", None):
            with self.subTest(fingerprint=fingerprint), self.assertRaises(ValueError):
                diagnose_detector_timeline(
                    value, (0, 1, 2), reference_source_admission_sha256=fingerprint
                )
        legacy = replace(value, version=1, result=replace(value.result, version=1, sampling=None))
        with self.assertRaisesRegex(ValueError, "v2"):
            diagnose(legacy, (0, 1, 2))

    def test_invalid_timelines_are_rejected_without_echoing_values(self):
        for reference in (
            (),
            [],
            (True,),
            (float("nan"),),
            (float("inf"),),
            (-1,),
            (10**1000,),
            ("private/path",),
            (None,) * (MAX_REFERENCE_FRAMES + 1),
        ):
            with self.subTest(kind=type(reference)), self.assertRaises(ValueError) as error:
                diagnose(scan(), reference)
            self.assertNotIn("private/path", str(error.exception))

    def test_hash_bindings_and_fixed_limits(self):
        value = scan()
        result = diagnose(value, (0.0, 1.0, 2.0))
        self.assertEqual(value.fingerprint, result["admitted_scan_sha256"])
        self.assertEqual(
            hashlib.sha256(b"[0.0,1.0,2.0]").hexdigest(), result["reference_timeline_sha256"]
        )
        for key in (
            "source_exhaustion_verified",
            "accuracy_verified",
            "deployment_authorized",
            "subject_identity_inference",
            "paths_disclosed",
            "media_bytes_verified_by_helper",
            "reference_provenance_verified_by_helper",
        ):
            self.assertIs(False, result[key])
        self.assertIsNone(result["continuous_observation_seconds"])
        json.dumps(result, allow_nan=False)


class PacketTimelineDiagnosticTests(unittest.TestCase):
    def test_b_frame_decode_reordering_uses_presentation_order(self):
        result = diagnose_packet_timestamps((0, 1, 2, 3), (0, 3, 1, 2))
        self.assertTrue(result["valid_pts_multisets_match"])
        self.assertEqual(1, result["max_presentation_pts_gap_seconds"])
        self.assertFalse(result["one_packet_per_frame_assumed"])

    def test_multiset_preserves_duplicate_multiplicity(self):
        result = diagnose_packet_timestamps((0, 1, 1), (0, 0, 1))
        self.assertFalse(result["valid_pts_multisets_match"])
        self.assertEqual(1, result["duplicate_present_pts_count"])
        self.assertTrue(
            diagnose_packet_timestamps((0, 1, 1), (1, 0, 1))["valid_pts_multisets_match"]
        )

    def test_missing_pts_are_reported_without_synthetic_values(self):
        result = diagnose_packet_timestamps((None, 1), (1, None, None))
        self.assertEqual(3, result["packet_count"])
        self.assertEqual(2, result["missing_pts_count"])
        self.assertTrue(result["valid_pts_multisets_match"])
        self.assertIsNone(result["max_presentation_pts_gap_seconds"])
        self.assertFalse(result["source_exhaustion_verified"])

    def test_container_pts_gap_is_not_missing_content_duration(self):
        result = diagnose_packet_timestamps((0, 0.1, 3.2), (3.2, 0, 0.1))
        self.assertAlmostEqual(3.1, result["max_presentation_pts_gap_seconds"])
        self.assertIsNone(result["missing_content_duration_seconds"])
        for invalid in ((True,), (float("nan"),), (), (None,) * (MAX_REFERENCE_FRAMES + 1)):
            with self.subTest(invalid_type=type(invalid)), self.assertRaises(ValueError):
                diagnose_packet_timestamps((0,), invalid)


if __name__ == "__main__":
    unittest.main()
