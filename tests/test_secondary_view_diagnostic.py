from __future__ import annotations

import copy
import math
import unittest

from scripts.secondary_view_diagnostic import MAX_BOXES_PER_VIEW, diagnose_views


def box(bbox=(0.1, 0.1, 0.5, 0.5), confidence=0.5):
    return {"bbox_xyxy": bbox, "confidence": confidence}


class SecondaryViewDiagnosticTests(unittest.TestCase):
    def diagnose(self, primary, secondary, **kwargs):
        return diagnose_views(
            primary,
            secondary,
            minimum_confidence=kwargs.get("minimum_confidence", 0.1),
            iou_threshold=kwargs.get("iou_threshold", 0.5),
        )

    def test_empty_inputs_are_unknown_not_absent(self):
        result = self.diagnose([], [])
        self.assertEqual(result["matched_pairs"], [])
        self.assertEqual(result["within_view_overlap_candidates"], [])
        self.assertEqual(result["presence"], "unknown")
        self.assertIsNone(result["true_person_count"])
        self.assertFalse(result["deployment_approved"])
        self.assertFalse(result["coordinate_system_verified_by_helper"])

    def test_duplicate_boxes_are_preserved_and_matched_once(self):
        result = self.diagnose([box(), box()], [box()])
        self.assertEqual(
            result["matched_pairs"], [{"primary_index": 0, "secondary_index": 0, "iou": 1.0}]
        )
        self.assertEqual(result["unmatched_primary_indices"], [1])
        self.assertEqual(len(result["primary_boxes"]), 2)
        self.assertEqual(len(result["within_view_overlap_candidates"]), 1)
        self.assertFalse(result["within_view_overlap_is_duplicate_label"])

    def test_highly_overlapping_people_not_suppressed(self):
        people = [box((0.1, 0.1, 0.7, 0.9)), box((0.2, 0.1, 0.8, 0.9))]
        result = self.diagnose(people, people)
        self.assertEqual(
            [(p["primary_index"], p["secondary_index"]) for p in result["matched_pairs"]],
            [(0, 0), (1, 1)],
        )
        self.assertEqual(len(result["within_view_overlap_candidates"]), 2)
        self.assertEqual(len(result["primary_boxes"]), 2)

    def test_asymmetric_secondary_boxes_keep_original_indices(self):
        result = self.diagnose([box()], [box(confidence=0.01), box(), box()])
        self.assertEqual(result["matched_pairs"][0]["secondary_index"], 1)
        self.assertEqual(result["unmatched_secondary_indices"], [2])
        self.assertEqual(result["below_threshold_secondary_indices"], [0])

    def test_highest_iou_wins_before_index(self):
        result = self.diagnose([box((0.1, 0.1, 0.55, 0.5)), box()], [box()])
        self.assertEqual(result["matched_pairs"][0]["primary_index"], 1)
        self.assertEqual(result["unmatched_primary_indices"], [0])

    def test_confidence_threshold_is_inclusive(self):
        result = self.diagnose(
            [box(confidence=0.1), box(confidence=math.nextafter(0.1, 0))], [box()]
        )
        self.assertEqual(len(result["matched_pairs"]), 1)
        self.assertEqual(result["below_threshold_primary_indices"], [1])
        self.assertEqual(len(result["primary_boxes"]), 2)

    def test_iou_threshold_is_inclusive(self):
        primary = [box((0, 0, 1, 1))]
        secondary = [box((0, 0, 0.5, 1))]
        self.assertEqual(self.diagnose(primary, secondary)["matched_pairs"][0]["iou"], 0.5)
        self.assertEqual(
            self.diagnose(primary, secondary, iou_threshold=math.nextafter(0.5, 1))[
                "matched_pairs"
            ],
            [],
        )
        self.assertEqual(len(self.diagnose(primary, primary, iou_threshold=1)["matched_pairs"]), 1)

    def test_touching_boxes_are_unmatched(self):
        result = self.diagnose([box((0, 0, 0.5, 1))], [box((0.5, 0, 1, 1))], iou_threshold=0.001)
        self.assertEqual(result["unmatched_primary_indices"], [0])
        self.assertEqual(result["unmatched_secondary_indices"], [0])

    def test_below_threshold_overlaps_remain_review_evidence(self):
        result = self.diagnose([box(confidence=0), box()], [])
        self.assertEqual(result["below_threshold_primary_indices"], [0])
        self.assertFalse(
            result["within_view_overlap_candidates"][0]["both_at_or_above_minimum_confidence"]
        )

    def test_zero_and_one_confidence_supported(self):
        self.assertEqual(
            len(
                self.diagnose([box(confidence=0)], [box(confidence=1)], minimum_confidence=0)[
                    "matched_pairs"
                ]
            ),
            1,
        )
        self.assertEqual(
            len(
                self.diagnose([box(confidence=1)], [box(confidence=1)], minimum_confidence=1)[
                    "matched_pairs"
                ]
            ),
            1,
        )

    def test_no_input_mutation_or_confidence_increase(self):
        primary = [box([0, 0, 1, 1], 0.17)]
        original = copy.deepcopy(primary)
        result = self.diagnose(primary, primary)
        self.assertEqual(primary, original)
        self.assertEqual(result["primary_boxes"][0]["confidence"], 0.17)
        result["primary_boxes"][0]["bbox_xyxy"][0] = 0.5
        self.assertEqual(primary, original)

    def test_deterministic_ties_one_to_one(self):
        result = self.diagnose([box()] * 3, [box()] * 3)
        self.assertEqual(
            [(p["primary_index"], p["secondary_index"]) for p in result["matched_pairs"]],
            [(0, 0), (1, 1), (2, 2)],
        )
        self.assertEqual(result, self.diagnose([box()] * 3, [box()] * 3))

    def test_rejects_invalid_thresholds(self):
        for value in (True, False, None, "0.1", float("nan"), float("inf"), -0.1, 1.1, 10**1000):
            for name in ("minimum_confidence", "iou_threshold"):
                with self.subTest(name=name, value=str(value)), self.assertRaises(ValueError):
                    self.diagnose([], [], **{name: value})
        with self.assertRaises(ValueError):
            self.diagnose([], [], iou_threshold=0)

    def test_rejects_invalid_confidence_even_below_threshold(self):
        for value in (True, None, "0.1", float("nan"), float("inf"), -0.1, 1.1, 10**1000):
            with self.subTest(value=str(value)), self.assertRaises(ValueError):
                self.diagnose([box(confidence=value)], [])

    def test_rejects_invalid_boxes(self):
        for bbox in (
            (0, 0, 1),
            (0, 0, 1, 1, 1),
            "0001",
            (False, 0, 1, 1),
            (0, 0, float("nan"), 1),
            (0, 0, float("inf"), 1),
            (-0.1, 0, 1, 1),
            (0, 0, 1.1, 1),
            (0.5, 0, 0.5, 1),
            (1, 0, 0, 1),
            (0, 0, 1, 0),
            (0, 0, 5e-324, 5e-324),
        ):
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                self.diagnose([box(bbox)], [])

    def test_rejects_unbounded_or_malformed_inputs(self):
        for value in (
            None,
            {},
            "",
            iter([]),
            [box()] * (MAX_BOXES_PER_VIEW + 1),
            [None],
            [{"bbox_xyxy": (0, 0, 1, 1)}],
            [{**box(), "label": "person"}],
        ):
            for primary, secondary in ((value, []), ([], value)):
                with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                    self.diagnose(primary, secondary)

    def test_limit_is_inclusive(self):
        result = self.diagnose([box(confidence=0)] * MAX_BOXES_PER_VIEW, [])
        self.assertEqual(len(result["primary_boxes"]), MAX_BOXES_PER_VIEW)
        self.assertEqual(len(result["below_threshold_primary_indices"]), MAX_BOXES_PER_VIEW)


if __name__ == "__main__":
    unittest.main()
