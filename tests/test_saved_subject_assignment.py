from __future__ import annotations

import copy
import json
import unittest

from scripts.evaluate_subject_assignment import evaluate_records
from tests.test_live_people import BOX, OTHER, claims
from tests.test_subject_layer_diagnostic import record


class SavedSubjectAssignmentTests(unittest.TestCase):
    def test_added_graph_edges_are_separate_from_original_observed_gap(self):
        value = claims(head=(250, 300))
        value["people"] += claims("adult", head=(750, 300))["people"]
        mixed = (0.05, 0.05, 0.95, 0.95)
        source = record(boxes=((mixed, OTHER), (), (mixed, OTHER)), value=value)
        prior = copy.deepcopy(source)
        result = evaluate_records([source])
        self.assertEqual(2, result["counts"]["graph_added_candidate_edges"])
        self.assertEqual(2, result["counts"]["added_candidates_with_observed_gap"])
        self.assertEqual(1, result["counts"]["baby_model_added_candidate_edges"])
        self.assertEqual(1, result["counts"]["adult_model_added_candidate_edges"])
        self.assertEqual(source, prior)
        self.assertTrue(result["rows"][0]["actual_assignment_and_audit_reproduced"])
        self.assertFalse(result["roles_propagated"])
        self.assertFalse(result["accuracy_measured"])

    def test_missing_other_head_does_not_suppress_legacy_unique_edge(self):
        value = claims()
        value["people"] += claims("adult", head=(750, 300))["people"]
        result = evaluate_records([record(boxes=((BOX,),), value=value)])
        self.assertEqual("no_complete_matching", result["rows"][0]["graph"]["status"])
        self.assertEqual(1, result["counts"]["legacy_unique_preserved"])
        self.assertEqual(1, result["counts"]["effective_origin_candidates"])
        self.assertEqual(0, result["counts"]["graph_added_candidate_edges"])

    def test_valid_single_head_legacy_match_is_unchanged(self):
        result = evaluate_records([record()])
        self.assertEqual(1, result["counts"]["legacy_unique_claims"])
        self.assertEqual(1, result["counts"]["legacy_unique_preserved"])
        self.assertFalse(result["rows"][0]["claims"][0]["candidate_added"])

    def test_known_invalid_image_replayed_but_not_evaluated_as_negative(self):
        source = record(value={"people": [], "uncertainties": []})
        result = evaluate_records(
            [source], unusable_png_sha256=[source["subject_result"]["input_sha256"]]
        )
        self.assertEqual(1, result["counts"]["matching_replays_passed"])
        self.assertEqual(1, result["counts"]["known_unusable_visual_input_excluded"])
        self.assertNotIn("graph_evaluated_records", result["counts"])
        self.assertNotIn("graph", result["rows"][0])

    def test_no_model_subjects_are_not_human_negative_or_accuracy(self):
        result = evaluate_records([record(value={"people": [], "uncertainties": []})])
        self.assertEqual(1, result["counts"]["graph_status_no_model_subjects"])
        self.assertEqual(0, result["counts"].get("model_claims", 0))
        self.assertFalse(result["ground_truth"])
        self.assertFalse(result["human_verified"])

    def test_unreproducible_trace_kept_as_rejection_not_success(self):
        for kind in ("partial", "audit"):
            source = record()
            if kind == "partial":
                source["people_trace"]["omitted_middle_frames"] = 1
            else:
                source["people_trace"]["frames"][0]["association_audit"]["extra"] = 1
            with self.subTest(kind=kind):
                result = evaluate_records([source])
                self.assertEqual(1, result["counts"]["native_replay_rejected"])
                self.assertNotIn("graph", result["rows"][0])
                self.assertNotIn("matching_replays_passed", result["counts"])

    def test_auxiliary_lane_does_not_borrow_original_coordinates(self):
        source = record()
        source["detector_view"] = "rotation_180"
        result = evaluate_records([source])
        self.assertEqual(1, result["counts"]["non_native_lane_excluded"])
        self.assertNotIn("graph", result["rows"][0])

    def test_malformed_duplicate_or_unbounded_batches_reject(self):
        source = record()
        for rows in ([], [source] * 129, [source, copy.deepcopy(source)], [None], [{}], None):
            with self.assertRaises(ValueError):
                evaluate_records(rows)
        for excludes in (["bad"], ["a" * 64] * 2, ["a" * 64] * 9, None):
            with self.assertRaises(ValueError):
                evaluate_records([source], unusable_png_sha256=excludes)

    def test_model_prose_and_paths_not_copied_to_report(self):
        source = record()
        source["subject_result"]["claims"]["people"][0]["visual_evidence"] = "private prose"
        encoded = json.dumps(evaluate_records([source]))
        self.assertNotIn("private prose", encoded)
        self.assertNotIn("synthetic.png", encoded)


if __name__ == "__main__":
    unittest.main()
