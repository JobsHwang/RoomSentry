from __future__ import annotations

import copy
import itertools
import unittest

from scripts.subject_assignment_diagnostic import _cover, diagnose_global_assignment

HASH = "a" * 64
LEFT = [0.1, 0.1, 0.4, 0.7]
RIGHT = [0.6, 0.1, 0.9, 0.7]
WHOLE = [0.05, 0.05, 0.95, 0.95]


class SubjectAssignmentDiagnosticTests(unittest.TestCase):
    def diagnose(self, centers, boxes):
        return diagnose_global_assignment(
            probe_input_sha256=HASH,
            detector_input_sha256=HASH,
            centers=centers,
            boxes=boxes,
        )

    def test_forced_global_assignment_resolves_nested_containment_as_geometry_only(self):
        centers, boxes = [[250, 300], [750, 300]], [WHOLE, RIGHT]
        prior = copy.deepcopy((centers, boxes))
        result = self.diagnose(centers, boxes)
        self.assertEqual("unique_complete_matching", result["status"])
        self.assertEqual([0, 1], result["candidate_box_indices"])
        self.assertEqual(
            ["ambiguous_head_match"] * 2, [r["status"] for r in result["legacy"]["subjects"]]
        )
        self.assertEqual(prior, (centers, boxes))
        for key in (
            "roles_propagated",
            "model_claims_are_ground_truth",
            "accuracy_measured",
            "production_changed",
            "deployment_approved",
        ):
            self.assertFalse(result[key])

    def test_legacy_unique_edges_are_preserved_in_complete_assignment(self):
        result = self.diagnose([[250, 300], [750, 300]], [RIGHT, LEFT])
        self.assertEqual([1, 0], result["candidate_box_indices"])
        self.assertEqual(
            ["unique_spatial_match"] * 2, [r["status"] for r in result["legacy"]["subjects"]]
        )

    def test_one_box_for_two_heads_does_not_fabricate_a_second_person_box(self):
        result = self.diagnose([[250, 300], [750, 300]], [WHOLE])
        self.assertEqual("no_complete_matching", result["status"])
        self.assertIsNone(result["candidate_box_indices"])

    def test_two_overlapping_choices_do_not_force_a_permutation(self):
        result = self.diagnose([[250, 300], [750, 300]], [WHOLE, WHOLE])
        self.assertEqual("multiple_complete_matchings", result["status"])
        self.assertIsNone(result["candidate_box_indices"])
        self.assertEqual([0, 1], result["alternative_edge_subject_indices"])

    def test_surplus_containing_box_is_an_alternative_not_a_unique_assignment(self):
        result = self.diagnose([[250, 300]], [LEFT, WHOLE])
        self.assertEqual("multiple_complete_matchings", result["status"])
        self.assertIsNone(result["candidate_box_indices"])

    def test_duplicate_model_points_remain_ambiguous(self):
        result = self.diagnose([[250, 300], [250, 300]], [LEFT, WHOLE])
        self.assertEqual("multiple_complete_matchings", result["status"])
        self.assertIsNone(result["candidate_box_indices"])

    def test_missing_unlocated_or_empty_subjects_are_not_fixed(self):
        for centers, boxes, expected in (
            ([[250, 300]], [], "no_complete_matching"),
            ([[250, 300], None], [LEFT, RIGHT], "unlocated_model_head"),
            ([], [WHOLE], "no_model_subjects"),
        ):
            result = self.diagnose(centers, boxes)
            self.assertEqual(expected, result["status"])
            self.assertIsNone(result["candidate_box_indices"])

    def test_ambiguous_component_does_not_hide_legacy_unique_other_component(self):
        result = self.diagnose([[250, 300], [750, 300]], [LEFT, RIGHT, RIGHT])
        self.assertEqual("multiple_complete_matchings", result["status"])
        self.assertEqual("unique_spatial_match", result["legacy"]["subjects"][0]["status"])

    def test_permutations_of_subjects_and_boxes_preserve_unique_geometry(self):
        centers, boxes = [[250, 300], [750, 300]], [WHOLE, RIGHT]
        for p, q in itertools.product(itertools.permutations(range(2)), repeat=2):
            result = self.diagnose([centers[i] for i in p], [boxes[i] for i in q])
            self.assertEqual([q.index(subject) for subject in p], result["candidate_box_indices"])

    def test_binding_bounds_types_and_coordinates_fail_closed(self):
        for centers, boxes in (
            ([[True, 300]], [LEFT]),
            ([[250, 300]], [LEFT] * 257),
            ([[250, 300]], [[0, 0, 1, float("nan")]]),
            ([[250, 300]], [[0, 0, 0, 1]]),
            ([], None),
        ):
            with self.assertRaises(ValueError):
                self.diagnose(centers, boxes)
        with self.assertRaises(ValueError):
            diagnose_global_assignment(
                probe_input_sha256=HASH,
                detector_input_sha256="b" * 64,
                centers=[[250, 300]],
                boxes=[LEFT],
            )

    def test_cover_and_alternative_detection_match_exhaustive_small_graph_oracle(self):
        for flags in itertools.product((False, True), repeat=9):
            options = [[j for j in range(3) if flags[i * 3 + j]] for i in range(3)]
            solutions = [
                p
                for p in itertools.permutations(range(3))
                if all(p[i] in options[i] for i in range(3))
            ]
            assignment = _cover(options)
            self.assertEqual(bool(solutions), assignment is not None)
            if assignment is not None:
                self.assertIn(tuple(assignment), solutions)
                any_alternative = any(
                    _cover(options, forbidden=(i, box)) is not None
                    for i, box in enumerate(assignment)
                )
                self.assertEqual(len(solutions) > 1, any_alternative)


if __name__ == "__main__":
    unittest.main()
