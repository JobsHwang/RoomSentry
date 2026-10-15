from __future__ import annotations

import hashlib
import json
import unittest
from fractions import Fraction

from scripts.pts_sampling_diagnostic import MAX_REFERENCE_FRAMES
from scripts.sampling_opportunity_diagnostic import evaluate_sampling_opportunity


def _oracle(reference, indices, duration, window):
    """Independent rational endpoint partition; test the event-window predicate.

    Between all t and t-d boundaries, s<=t<s+d is constant. Midpoint testing
    therefore integrates exactly, without merging the helper's intervals or
    imposing a discrete sampling grid on event start positions.
    """
    points = tuple(Fraction(str(reference[index])) for index in indices)
    length = Fraction(str(window))
    end = Fraction(str(duration)) - length
    boundaries = {Fraction(0), end}
    for point in points:
        for boundary in (point, point - length):
            if 0 < boundary < end:
                boundaries.add(boundary)
    ordered = sorted(boundaries)
    measure = Fraction(0)
    for start, stop in zip(ordered, ordered[1:], strict=False):
        middle = (start + stop) / 2
        if any(middle <= point < middle + length for point in points):
            measure += stop - start
    return measure


def _sha(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class SamplingOpportunityDiagnosticTests(unittest.TestCase):
    def test_matches_independent_rational_midpoint_oracle(self):
        cases = (
            ((0.0, 0.1, 0.35, 0.9), 1.0, (0.001, 0.1, 0.3, 0.9)),
            ((1.2, 2.0, 2.1, 5.7), 6.0, (0.05, 0.5, 2.0, 5.5)),
            ((0.0, 1.0, 4.0, 9.0), 10.0, (0.25, 1.0, 4.0, 8.0)),
        )
        for reference, duration, windows in cases:
            schedules = {
                "empty": (),
                "first": (0,),
                "alternating": tuple(range(0, len(reference), 2)),
                "all": tuple(range(len(reference))),
            }
            result = evaluate_sampling_opportunity(
                reference,
                schedules,
                source_duration_seconds=duration,
                event_durations_seconds=windows,
            )
            for row in result["rows"]:
                window = row["hypothetical_event_duration_seconds"]
                expected_reference = _oracle(reference, range(len(reference)), duration, window)
                self.assertAlmostEqual(
                    float(expected_reference), row["reference_frame_opportunity_measure_seconds"]
                )
                for name, indices in schedules.items():
                    with self.subTest(reference=reference, window=window, schedule=name):
                        expected = _oracle(reference, indices, duration, window)
                        branch = row["branches"][name]
                        self.assertAlmostEqual(
                            float(expected), branch["covered_start_position_measure_seconds"]
                        )
                        self.assertAlmostEqual(
                            float(expected) / (duration - window),
                            branch["uniform_start_position_opportunity_fraction"],
                        )

    def test_nonzero_source_pts_is_not_shifted_to_first_frame(self):
        result = evaluate_sampling_opportunity(
            (4.0,), {"selected": (0,)}, source_duration_seconds=10.0, event_durations_seconds=(1.0,)
        )
        row = result["rows"][0]
        self.assertEqual(9.0, row["possible_start_position_measure_seconds"])
        self.assertEqual(1.0, row["branches"]["selected"]["covered_start_position_measure_seconds"])

    def test_half_open_boundary_only_opportunities_have_zero_measure(self):
        result = evaluate_sampling_opportunity(
            (0.0, 9.0),
            {"first_only": (0,), "tail": (1,), "all": (0, 1)},
            source_duration_seconds=10.0,
            event_durations_seconds=(1.0,),
        )
        branches = result["rows"][0]["branches"]
        self.assertEqual(0.0, branches["first_only"]["covered_start_position_measure_seconds"])
        self.assertEqual(1.0, branches["tail"]["covered_start_position_measure_seconds"])
        self.assertEqual(1.0, branches["all"]["covered_start_position_measure_seconds"])
        self.assertTrue(result["endpoint_only_opportunities_have_zero_measure"])
        # The exact source-end PTS is outside [0,D), not an extra measurable opportunity.
        with self.assertRaises(ValueError):
            evaluate_sampling_opportunity(
                (0.0, 10.0),
                {"all": (0, 1)},
                source_duration_seconds=10.0,
                event_durations_seconds=(1.0,),
            )

    def test_overlapping_windows_union_is_not_summed_twice(self):
        result = evaluate_sampling_opportunity(
            (2.0, 2.5, 3.0),
            {"all": (0, 1, 2)},
            source_duration_seconds=10.0,
            event_durations_seconds=(2.0,),
        )
        self.assertEqual(
            3.0, result["rows"][0]["branches"]["all"]["covered_start_position_measure_seconds"]
        )
        self.assertEqual(
            3 / 8,
            result["rows"][0]["branches"]["all"]["uniform_start_position_opportunity_fraction"],
        )

    def test_empty_schedule_and_reference_upper_bound_keep_same_denominator(self):
        result = evaluate_sampling_opportunity(
            (0.0, 1.0, 4.0, 9.0),
            {"empty": (), "one": (2,), "all": (0, 1, 2, 3)},
            source_duration_seconds=10.0,
            event_durations_seconds=(1.0,),
        )
        row = result["rows"][0]
        self.assertEqual(9.0, row["possible_start_position_measure_seconds"])
        self.assertEqual(3.0, row["reference_frame_opportunity_measure_seconds"])
        self.assertEqual(6.0, row["no_reference_frame_start_position_measure_seconds"])
        self.assertEqual(
            0.0, row["branches"]["empty"]["uniform_start_position_opportunity_fraction"]
        )
        self.assertEqual(
            2.0, row["branches"]["one"]["extra_unsampled_start_position_measure_seconds"]
        )
        self.assertEqual(
            0.0, row["branches"]["all"]["extra_unsampled_start_position_measure_seconds"]
        )
        self.assertEqual(
            row["reference_frame_opportunity_measure_seconds"],
            row["branches"]["all"]["covered_start_position_measure_seconds"],
        )
        for branch in row["branches"].values():
            self.assertLessEqual(
                branch["covered_start_position_measure_seconds"],
                row["reference_frame_opportunity_measure_seconds"],
            )
            self.assertEqual(
                9.0,
                branch["covered_start_position_measure_seconds"]
                + branch["uncovered_start_position_measure_seconds"],
            )

    def test_different_sample_counts_do_not_change_fixed_window_denominator(self):
        result = evaluate_sampling_opportunity(
            (0.0, 4.0),
            {"one": (1,), "two": (0, 1)},
            source_duration_seconds=10.0,
            event_durations_seconds=(1.0,),
        )
        branches = result["rows"][0]["branches"]
        self.assertEqual((1, 2), (branches["one"]["sample_count"], branches["two"]["sample_count"]))
        self.assertEqual(1 / 9, branches["one"]["uniform_start_position_opportunity_fraction"])
        self.assertEqual(
            branches["one"]["uniform_start_position_opportunity_fraction"],
            branches["two"]["uniform_start_position_opportunity_fraction"],
        )

    def test_longer_windows_change_start_domain_not_observed_duration(self):
        result = evaluate_sampling_opportunity(
            (5.0,),
            {"selected": (0,)},
            source_duration_seconds=10.0,
            event_durations_seconds=(1.0, 4.0, 8.0),
        )
        self.assertEqual(
            [9.0, 6.0, 2.0],
            [row["possible_start_position_measure_seconds"] for row in result["rows"]],
        )
        self.assertEqual(
            [1.0, 4.0, 2.0],
            [
                row["branches"]["selected"]["covered_start_position_measure_seconds"]
                for row in result["rows"]
            ],
        )
        self.assertEqual(
            [1 / 9, 4 / 6, 1.0],
            [
                row["branches"]["selected"]["uniform_start_position_opportunity_fraction"]
                for row in result["rows"]
            ],
        )

    def test_minimum_duration_is_supported_but_d_must_be_strictly_below_source(self):
        result = evaluate_sampling_opportunity(
            (0.0, 0.001),
            {"all": (0, 1)},
            source_duration_seconds=0.002,
            event_durations_seconds=(0.001,),
        )
        self.assertEqual(
            1.0, result["rows"][0]["branches"]["all"]["uniform_start_position_opportunity_fraction"]
        )
        for length in (0.000999, 0.002, 0.003):
            with self.subTest(length=length), self.assertRaises(ValueError):
                evaluate_sampling_opportunity(
                    (0.0,),
                    {"all": (0,)},
                    source_duration_seconds=0.002,
                    event_durations_seconds=(length,),
                )

    def test_invalid_reference_types_values_duplicates_and_limits(self):
        for reference in (
            (),
            [],
            (True,),
            (None,),
            (float("nan"),),
            (float("inf"),),
            (-1.0,),
            (0.0, 0.0),
            (2.0, 1.0),
            (10**1000,),
            (0.0,) * (MAX_REFERENCE_FRAMES + 1),
        ):
            with self.subTest(kind=type(reference)), self.assertRaises(ValueError):
                evaluate_sampling_opportunity(
                    reference,
                    {"all": ()},
                    source_duration_seconds=10.0,
                    event_durations_seconds=(1.0,),
                )

    def test_invalid_schedule_indices_names_order_and_count(self):
        for schedules in (
            {},
            None,
            {"private/path": (0,)},
            {"bad": [0]},
            {"bad": (True,)},
            {"bad": (-1,)},
            {"bad": (2,)},
            {"bad": (1, 0)},
            {"bad": (0, 0)},
            {f"schedule_{index}": () for index in range(9)},
        ):
            with self.subTest(kind=type(schedules)), self.assertRaises(ValueError) as error:
                evaluate_sampling_opportunity(
                    (0.0, 1.0),
                    schedules,
                    source_duration_seconds=10.0,
                    event_durations_seconds=(1.0,),
                )
            self.assertNotIn("private/path", str(error.exception))

    def test_invalid_durations_and_source_values(self):
        for lengths in (
            (),
            [],
            (True,),
            (float("nan"),),
            (float("inf"),),
            (1.0, 1.0),
            (2.0, 1.0),
            tuple(index / 100 for index in range(1, 18)),
        ):
            with self.subTest(kind=type(lengths)), self.assertRaises(ValueError):
                evaluate_sampling_opportunity(
                    (0.0,),
                    {"all": (0,)},
                    source_duration_seconds=10.0,
                    event_durations_seconds=lengths,
                )
        for duration in (True, 0, 0.0001, float("nan"), float("inf"), 86401, 10**1000):
            with self.subTest(kind=type(duration)), self.assertRaises(ValueError):
                evaluate_sampling_opportunity(
                    (0.0,),
                    {"all": (0,)},
                    source_duration_seconds=duration,
                    event_durations_seconds=(0.001,),
                )

    def test_hashes_and_flags_do_not_claim_event_probability_recall_or_completeness(self):
        reference = (0.0, 1.0, 4.0)
        schedules = {"full": (0, 1, 2), "partial": (1,)}
        result = evaluate_sampling_opportunity(
            reference, schedules, source_duration_seconds=10.0, event_durations_seconds=(1.0, 2.0)
        )
        self.assertEqual(_sha(reference), result["reference_pts_sha256"])
        self.assertEqual(
            _sha(
                {
                    "reference_pts_seconds": reference,
                    "selections": schedules,
                    "source_duration_seconds": 10.0,
                    "event_durations_seconds": (1.0, 2.0),
                }
            ),
            result["input_sha256"],
        )
        self.assertEqual(
            _sha({k: v for k, v in result.items() if k != "report_sha256"}), result["report_sha256"]
        )
        self.assertEqual(
            "seconds_of_possible_start_positions_not_observation_duration", result["measure_unit"]
        )
        self.assertTrue(
            result["uniform_start_position_assumption_is_not_empirical_event_distribution"]
        )
        self.assertTrue(result["reference_is_decoded_frames_not_complete_source"])
        for key in (
            "model_calls_verified_by_helper",
            "media_bytes_verified_by_helper",
            "actual_event_labels_used",
            "ground_truth",
            "accuracy_evaluated",
            "event_recall_evaluated",
            "source_exhaustion_verified",
            "deployment_approved",
        ):
            self.assertIs(False, result[key])


if __name__ == "__main__":
    unittest.main()
