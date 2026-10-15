from __future__ import annotations

import hashlib
import json
import math
import unittest

from scripts.pts_sampling_diagnostic import MAX_REFERENCE_FRAMES, plan_pts_sampling


class PtsSamplingDiagnosticTests(unittest.TestCase):
    def test_cfr_stride_and_pts_deadlines_choose_same_available_frames(self):
        reference = tuple(index / 25 for index in range(76))
        result = plan_pts_sampling(reference, native_fps=25, sample_fps=1)
        for key in ("decoded_stride", "source_pts_deadline"):
            self.assertEqual([0, 25, 50, 75], result[key]["decoded_frame_indices"])
            self.assertEqual([0.0, 1.0, 2.0, 3.0], result[key]["pts_seconds"])
            self.assertEqual(4, result[key]["actual_sample_count"])
            self.assertEqual(1.0, result[key]["max_adjacent_sample_pts_gap_seconds"])
        self.assertEqual(0, result["source_pts_deadline"]["missed_deadline_count"])

    def test_vfr_causal_selection_follows_pts_not_nominal_frame_count(self):
        result = plan_pts_sampling((0.0, 0.1, 0.2, 1.2, 1.3, 2.1), native_fps=2, sample_fps=1)
        self.assertEqual([0, 2, 4], result["decoded_stride"]["decoded_frame_indices"])
        self.assertEqual([0, 3, 5], result["source_pts_deadline"]["decoded_frame_indices"])
        self.assertEqual([0.0, 1.2, 2.1], result["source_pts_deadline"]["pts_seconds"])
        self.assertEqual(0, result["source_pts_deadline"]["missed_deadline_count"])

    def test_long_source_gap_skips_deadlines_without_duplicate_catchup_frames(self):
        result = plan_pts_sampling((0.0, 0.1, 3.1, 3.2, 4.0), native_fps=25, sample_fps=1)
        selected = result["source_pts_deadline"]
        self.assertEqual([0, 2, 4], selected["decoded_frame_indices"])
        self.assertEqual(2, selected["missed_deadline_count"])
        self.assertEqual(3.1, selected["max_adjacent_sample_pts_gap_seconds"])
        self.assertEqual(3.0, result["source_max_adjacent_pts_gap_seconds"])
        self.assertEqual(5.0, selected["next_deadline_seconds"])
        self.assertEqual(3, len(set(selected["decoded_frame_indices"])))

    def test_nonzero_source_origin_is_not_rebased(self):
        result = plan_pts_sampling((2.5, 2.7, 3.0), native_fps=25, sample_fps=1)
        selected = result["source_pts_deadline"]
        self.assertEqual([0, 2], selected["decoded_frame_indices"])
        self.assertEqual([2.5, 3.0], selected["pts_seconds"])
        self.assertEqual(2, selected["missed_deadline_count"])
        self.assertEqual(0.0, selected["anchor_pts_seconds"])
        self.assertEqual(2.5, result["source_first_pts_seconds"])

    def test_single_source_frame_has_no_measured_gap_or_future_catchup(self):
        result = plan_pts_sampling((0.35,), native_fps=0.01, sample_fps=10)
        self.assertEqual(1, result["decoded_stride"]["frame_step"])
        self.assertIsNone(result["source_max_adjacent_pts_gap_seconds"])
        self.assertIsNone(result["source_pts_deadline"]["max_adjacent_sample_pts_gap_seconds"])
        self.assertEqual(1, result["source_pts_deadline"]["actual_sample_count"])
        self.assertEqual(3, result["source_pts_deadline"]["missed_deadline_count"])
        self.assertEqual(0.4, result["source_pts_deadline"]["next_deadline_seconds"])

    def test_deadline_boundary_uses_causal_comparison_without_epsilon(self):
        below = math.nextafter(0.3, 0.0)
        result = plan_pts_sampling((0.0, below, 0.3, 0.31), native_fps=25, sample_fps=10)
        selected = result["source_pts_deadline"]
        # The preceding frame consumes the overdue 0.1 deadline and skips 0.2;
        # exactly 0.3 is a distinct available frame, so it must still be selected.
        self.assertEqual([0, 1, 2], selected["decoded_frame_indices"])
        self.assertEqual(1, selected["missed_deadline_count"])
        self.assertGreater(selected["next_deadline_seconds"], 0.31)

    def test_stride_rounding_uses_executed_integer_stride(self):
        result = plan_pts_sampling(
            tuple(index / 25 for index in range(26)), native_fps=25, sample_fps=2
        )
        self.assertEqual(12, result["decoded_stride"]["frame_step"])
        self.assertEqual([0, 12, 24], result["decoded_stride"]["decoded_frame_indices"])
        self.assertEqual([0, 13, 25], result["source_pts_deadline"]["decoded_frame_indices"])

    def test_pts_selection_is_causal_and_deadline_counts_partition_due_schedule(self):
        reference = (0.3, 0.4, 1.1, 1.9, 3.1, 3.2)
        full = plan_pts_sampling(reference, native_fps=25, sample_fps=2.5)["source_pts_deadline"]
        for length in range(1, len(reference) + 1):
            prefix = plan_pts_sampling(reference[:length], native_fps=25, sample_fps=2.5)[
                "source_pts_deadline"
            ]
            self.assertEqual(
                [index for index in full["decoded_frame_indices"] if index < length],
                prefix["decoded_frame_indices"],
            )
            due = sum(index / 2.5 <= reference[length - 1] for index in range(20))
            self.assertEqual(due, prefix["actual_sample_count"] + prefix["missed_deadline_count"])
            self.assertGreater(prefix["next_deadline_seconds"], reference[length - 1])

    def test_maximum_reference_frame_count_is_accepted(self):
        reference = tuple(index / 1000 for index in range(MAX_REFERENCE_FRAMES))
        result = plan_pts_sampling(reference, native_fps=1000, sample_fps=0.01)
        self.assertEqual(MAX_REFERENCE_FRAMES, result["source_frame_count"])
        self.assertEqual([0, 100000], result["decoded_stride"]["decoded_frame_indices"])
        self.assertEqual([0, 100000], result["source_pts_deadline"]["decoded_frame_indices"])

    def test_invalid_unknown_repeated_or_regressing_pts_are_rejected(self):
        for reference in (
            (),
            [],
            (None,),
            (True,),
            (float("nan"),),
            (float("inf"),),
            (-1,),
            (86400.1,),
            (10**1000,),
            (0, 0),
            (1, 0),
            ("private/path",),
        ):
            with self.subTest(kind=type(reference)), self.assertRaises(ValueError) as error:
                plan_pts_sampling(reference, native_fps=25, sample_fps=1)
            self.assertNotIn("private/path", str(error.exception))

    def test_fps_bounds_and_reference_size_are_strict(self):
        for key, invalid in (
            ("native_fps", True),
            ("native_fps", 0.009),
            ("native_fps", 1000.1),
            ("sample_fps", 0.009),
            ("sample_fps", 10.1),
            ("sample_fps", float("nan")),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan_pts_sampling((0.0,), **({"native_fps": 25, "sample_fps": 1} | {key: invalid}))
        with self.assertRaises(ValueError):
            plan_pts_sampling((0.0,) * (MAX_REFERENCE_FRAMES + 1), native_fps=25, sample_fps=1)
        result = plan_pts_sampling((86400.0,), native_fps=1000, sample_fps=0.01)
        self.assertEqual(864, result["source_pts_deadline"]["missed_deadline_count"])
        self.assertEqual(86500.0, result["source_pts_deadline"]["next_deadline_seconds"])

    def test_reference_and_report_hashes_bind_values_without_authority_claims(self):
        reference = (0.0, 1.0, 2.0)
        result = plan_pts_sampling(reference, native_fps=25, sample_fps=1)
        self.assertEqual(
            hashlib.sha256(b"[0.0,1.0,2.0]").hexdigest(), result["reference_pts_sha256"]
        )
        payload = {key: value for key, value in result.items() if key != "report_sha256"}
        expected = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        self.assertEqual(expected, result["report_sha256"])
        for key in (
            "accuracy_evaluated",
            "ground_truth",
            "source_exhaustion_verified",
            "deployment_approved",
            "equal_stride_ledger_claimed",
        ):
            self.assertIs(False, result[key])
        self.assertTrue(result["offline_diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
