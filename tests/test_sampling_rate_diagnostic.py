from __future__ import annotations

import hashlib
import json
import unittest

from scripts.pts_sampling_diagnostic import plan_pts_sampling
from scripts.sampling_opportunity_diagnostic import evaluate_sampling_opportunity
from scripts.sampling_rate_diagnostic import plan_sampling_rates


def _sha(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class SamplingRateDiagnosticTests(unittest.TestCase):
    def test_fixed_cfr_rates_and_deduplicated_union_have_independent_exact_counts(self):
        reference = tuple(index / 20 for index in range(120))
        result = plan_sampling_rates(reference, native_fps=20, source_duration_seconds=6.0)
        expected_steps = {"rate_0_5": 40, "rate_1": 20, "rate_2": 10, "rate_4": 5}
        self.assertEqual([0.5, 1.0, 2.0, 4.0], result["rates_fps"])
        self.assertEqual([0.25, 0.5, 1.0, 2.0, 5.0], result["hypothetical_event_durations_seconds"])
        for name, step in expected_steps.items():
            with self.subTest(branch=name):
                expected = list(range(0, 120, step))
                branch = result["branches"][name]
                self.assertEqual(expected, branch["selection"]["decoded_frame_indices"])
                self.assertEqual(len(expected), branch["planned_slot_count"])
        self.assertEqual(
            [3, 6, 12, 24], [branch["planned_slot_count"] for branch in result["branches"].values()]
        )
        self.assertEqual(list(range(0, 120, 5)), result["union_decoded_frame_indices"])
        self.assertEqual(24, result["unique_union_planned_slot_count"])
        self.assertEqual(
            [reference[index] for index in range(0, 120, 5)], result["unique_union_pts_seconds"]
        )

    def test_retained_plans_and_opportunity_report_bind_unchanged_helpers(self):
        reference = (0.0, 0.15, 0.4, 1.2, 1.6, 2.1, 4.0, 5.9)
        result = plan_sampling_rates(reference, native_fps=25, source_duration_seconds=6.0)
        selections = {}
        for name, branch in result["branches"].items():
            original = plan_pts_sampling(reference, native_fps=25, sample_fps=branch["sample_fps"])
            self.assertEqual(original["source_pts_deadline"], branch["selection"])
            self.assertEqual(original["report_sha256"], branch["pts_plan_sha256"])
            selections[name] = tuple(branch["selection"]["decoded_frame_indices"])
        self.assertEqual(
            evaluate_sampling_opportunity(
                reference,
                selections,
                source_duration_seconds=6.0,
                event_durations_seconds=(0.25, 0.5, 1.0, 2.0, 5.0),
            ),
            result["opportunity"],
        )
        self.assertEqual(
            sorted(set().union(*selections.values())), result["union_decoded_frame_indices"]
        )

    def test_higher_rate_cannot_fill_a_reference_source_gap(self):
        reference = (0.0, 0.25, 0.5, 4.0, 4.25, 4.5, 5.75)
        result = plan_sampling_rates(reference, native_fps=25, source_duration_seconds=6.0)
        high = result["branches"]["rate_4"]["selection"]
        self.assertEqual(list(range(len(reference))), high["decoded_frame_indices"])
        self.assertEqual(3.5, high["max_adjacent_sample_pts_gap_seconds"])
        self.assertGreater(high["missed_deadline_count"], 0)
        quarter = result["opportunity"]["rows"][0]
        self.assertEqual(0.25, quarter["hypothetical_event_duration_seconds"])
        self.assertGreater(quarter["no_reference_frame_start_position_measure_seconds"], 0)
        self.assertEqual(
            quarter["reference_frame_opportunity_measure_seconds"],
            quarter["branches"]["rate_4"]["covered_start_position_measure_seconds"],
        )
        self.assertLess(
            quarter["branches"]["rate_4"]["uniform_start_position_opportunity_fraction"], 1.0
        )

    def test_all_rates_compare_identical_window_start_denominators(self):
        reference = tuple(index / 20 for index in range(120))
        result = plan_sampling_rates(reference, native_fps=20, source_duration_seconds=6.0)
        for row in result["opportunity"]["rows"]:
            denominator = 6.0 - row["hypothetical_event_duration_seconds"]
            self.assertEqual(denominator, row["possible_start_position_measure_seconds"])
            for branch in row["branches"].values():
                self.assertAlmostEqual(
                    denominator,
                    branch["covered_start_position_measure_seconds"]
                    + branch["uncovered_start_position_measure_seconds"],
                )

    def test_exact_256_union_slots_allowed_but_257_rejected(self):
        reference = tuple(index / 4 for index in range(256))
        result = plan_sampling_rates(reference, native_fps=4, source_duration_seconds=64.0)
        self.assertEqual(256, result["unique_union_planned_slot_count"])
        with self.assertRaisesRegex(ValueError, "256"):
            plan_sampling_rates(
                tuple(index / 4 for index in range(257)), native_fps=4, source_duration_seconds=64.1
            )

    def test_nonzero_origin_and_single_available_frame_remain_unshifted(self):
        result = plan_sampling_rates((5.1,), native_fps=25, source_duration_seconds=6.0)
        self.assertEqual([0], result["union_decoded_frame_indices"])
        self.assertEqual([5.1], result["unique_union_pts_seconds"])
        for branch in result["branches"].values():
            self.assertEqual(1, branch["planned_slot_count"])
            self.assertEqual([5.1], branch["selection"]["pts_seconds"])
            self.assertGreater(branch["selection"]["missed_deadline_count"], 0)

    def test_duration_must_fit_longest_fixed_window_and_source_pts(self):
        for duration in (5.0, 4.9, True, float("nan"), float("inf"), 86401, 10**1000, "6"):
            with self.subTest(kind=type(duration)), self.assertRaises(ValueError):
                plan_sampling_rates((0.0,), native_fps=25, source_duration_seconds=duration)
        for reference in ((0.0, 6.0), (0.0, 6.1)):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                plan_sampling_rates(reference, native_fps=25, source_duration_seconds=6.0)
        self.assertEqual(
            1,
            plan_sampling_rates((0.0,), native_fps=25, source_duration_seconds=5.001)[
                "unique_union_planned_slot_count"
            ],
        )

    def test_unknown_bad_or_repeated_pts_and_invalid_native_fps_are_rejected(self):
        for reference in ((), [], (None,), (True,), (float("nan"),), (0.0, 0.0), (1.0, 0.0)):
            with self.subTest(kind=type(reference)), self.assertRaises(ValueError):
                plan_sampling_rates(reference, native_fps=25, source_duration_seconds=6.0)
        for fps in (True, 0.009, 1000.1, float("nan")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                plan_sampling_rates((0.0,), native_fps=fps, source_duration_seconds=6.0)

    def test_input_and_report_hashes_and_planned_only_flags(self):
        reference = (0.0, 1.0, 2.0, 5.5)
        result = plan_sampling_rates(reference, native_fps=25, source_duration_seconds=6.0)
        self.assertEqual(_sha(reference), result["reference_pts_sha256"])
        self.assertEqual(
            _sha(
                {
                    "reference_pts_seconds": reference,
                    "native_fps": 25.0,
                    "source_duration_seconds": 6.0,
                    "rates_fps": [0.5, 1.0, 2.0, 4.0],
                    "hypothetical_event_durations_seconds": [0.25, 0.5, 1.0, 2.0, 5.0],
                }
            ),
            result["input_sha256"],
        )
        self.assertEqual(
            _sha({key: value for key, value in result.items() if key != "report_sha256"}),
            result["report_sha256"],
        )
        self.assertTrue(result["planned_slots_only"])
        self.assertTrue(result["offline_diagnostic_only"])
        for field in (
            "actual_model_calls_verified",
            "empirical_event_probability_evaluated",
            "accuracy_evaluated",
            "event_recall_evaluated",
            "ground_truth",
            "source_exhaustion_verified",
            "deployment_approved",
            "equal_stride_ledger_claimed",
        ):
            self.assertIs(False, result[field])


if __name__ == "__main__":
    unittest.main()
