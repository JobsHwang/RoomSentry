from __future__ import annotations

import copy
import unittest

from scripts.dual_view_candidate_diagnostic import diagnose_candidates


def box(confidence=0.2, xy=None):
    return {"confidence": confidence, "bbox_xyxy": xy if xy is not None else [0.1, 0.2, 0.3, 0.6]}


def diagnose(primary, rotated, **kwargs):
    settings = dict(support_confidence=0.1, candidate_confidence=0.35, iou_threshold=0.5)
    settings.update(kwargs)
    return diagnose_candidates(primary, rotated, **settings)


class DualViewCandidateDiagnosticTests(unittest.TestCase):
    def test_weak_original_supports_rotated_after_coordinate_mapping(self):
        result = diagnose([box()], [box(0.8, [0.7, 0.4, 0.9, 0.8])])
        self.assertEqual([0], result["rotated_strong_supported_indices"])
        self.assertEqual([], result["primary_strong_indices"])
        pair = result["geometric_support_pairs"][0]
        self.assertEqual("weak", pair["primary_support_kind"])
        self.assertAlmostEqual(1, pair["iou"])
        self.assertEqual(0.2, result["primary_boxes"][0]["confidence"])
        self.assertEqual(0.8, result["rotated_boxes_source_coordinates"][0]["confidence"])

    def test_empty_original_keeps_unilateral_and_unknown(self):
        result = diagnose([], [box(0.8)])
        self.assertEqual([0], result["rotated_strong_unilateral_indices"])
        self.assertEqual(1, len(result["rotated_boxes_source_coordinates"]))
        for key in ("presence", "baby_identity", "baby_state"):
            self.assertEqual("unknown", result[key])
        self.assertIsNone(result["true_person_count"])
        for key in (
            "deployment_approved",
            "unilateral_candidates_suppressed",
            "cross_view_scores_are_independent",
        ):
            self.assertIs(result[key], False)

    def test_both_zero_are_explicit_without_absence_claim(self):
        result = diagnose([], [])
        self.assertEqual([], result["primary_boxes"])
        self.assertEqual([], result["geometric_support_pairs"])
        self.assertEqual("unknown", result["presence"])

    def test_weak_rotated_cannot_steal_support_and_original_indices_remain(self):
        result = diagnose([box()], [box(0.2, [0.7, 0.4, 0.9, 0.8]), box(0.8, [0.7, 0.4, 0.9, 0.8])])
        self.assertEqual([1], result["rotated_strong_supported_indices"])
        self.assertEqual([0], result["rotated_below_candidate_indices"])
        self.assertEqual(1, result["geometric_support_pairs"][0]["rotated_index"])

    def test_strong_original_and_many_to_one_do_not_prove_identity(self):
        result = diagnose([box(0.6), box(0.7)], [box(0.8, [0.7, 0.4, 0.9, 0.8])])
        self.assertEqual([0, 1], result["primary_strong_indices"])
        self.assertEqual(1, len(result["geometric_support_pairs"]))
        self.assertEqual("strong", result["geometric_support_pairs"][0]["primary_support_kind"])
        self.assertIs(result["geometric_pairs_are_unique_person_matches"], False)
        self.assertTrue(result["within_view_overlap_candidates"])

    def test_equal_threshold_is_included_and_too_weak_does_not_support(self):
        result = diagnose([box(0.1)], [box(0.35, [0.7, 0.4, 0.9, 0.8])])
        self.assertEqual([0], result["rotated_strong_supported_indices"])
        result = diagnose([box(0.09)], [box(0.35, [0.7, 0.4, 0.9, 0.8])])
        self.assertEqual([0], result["rotated_strong_unilateral_indices"])

    def test_wrong_direction_is_not_silently_treated_as_source_coordinates(self):
        result = diagnose([box()], [box(0.8)])
        self.assertEqual([], result["geometric_support_pairs"])
        self.assertIs(result["source_frame_and_rotation_verified_by_helper"], False)

    def test_invalid_thresholds_and_geometry_fail(self):
        for setting in (
            {"support_confidence": True},
            {"support_confidence": float("nan")},
            {"candidate_confidence": -0.1},
            {"candidate_confidence": 0.05},
            {"iou_threshold": 0},
            {"iou_threshold": float("inf")},
        ):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                diagnose([], [], **setting)
        for invalid in (
            None,
            (),
            [box(True)],
            [box(0.2, [0, 0, 0, 1])],
            [box(0.2, [-1, 0, 1, 1])],
            [box()] * 101,
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                diagnose([], invalid)

    def test_all_boxes_are_preserved_and_inputs_not_mutated(self):
        primary = [box(), box(0.09)]
        rotated = [box(0.8), box(0.2)]
        before = copy.deepcopy((primary, rotated))
        result = diagnose(primary, rotated)
        self.assertEqual(before, (primary, rotated))
        self.assertEqual(len(primary), len(result["primary_boxes"]))
        self.assertEqual(len(rotated), len(result["rotated_boxes_source_coordinates"]))


if __name__ == "__main__":
    unittest.main()
