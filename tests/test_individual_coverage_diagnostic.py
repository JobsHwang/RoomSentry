from __future__ import annotations

import copy
import math
import unittest

from scripts.individual_coverage_diagnostic import (
    candidate_fingerprint,
    diagnose_individual_coverage,
)

FRAME = "a" * 64


def candidate(identifier, confidence=0.5):
    return {"detection_id": identifier, "bbox_xyxy": [0.1, 0.1, 0.9, 0.9], "confidence": confidence}


def assignment(identifier, state="single_subject", refs=("p1",)):
    return {"detection_id": identifier, "state": state, "reference_ids": list(refs)}


def review_for(boxes, assignments=(), refs=("p1", "p2"), complete=True):
    return {
        "frame_sha256": FRAME,
        "candidates_sha256": candidate_fingerprint(FRAME, boxes),
        "provenance": "synthetic",
        "reference_ids": list(refs),
        "reference_inventory_complete": complete,
        "assignments": list(assignments),
    }


class IndividualCoverageDiagnosticTests(unittest.TestCase):
    def run_review(self, boxes, assignments=(), *, refs=("p1", "p2"), complete=True, threshold=0.2):
        return diagnose_individual_coverage(
            FRAME,
            boxes,
            review_for(boxes, assignments, refs, complete),
            minimum_confidence=threshold,
        )

    def test_equal_box_count_does_not_prove_two_people_covered(self):
        result = self.run_review(
            [candidate("b1"), candidate("b2")], [assignment("b1"), assignment("b2")]
        )
        self.assertEqual(result["separately_localized_reference_count"], 1)
        self.assertEqual(result["reference_coverage"][1]["status"], "not_separately_localized")
        self.assertEqual(
            result["duplicate_assignment_groups"],
            [{"reference_id": "p1", "candidate_ids": ["b1", "b2"]}],
        )

    def test_multi_subject_box_gives_no_separate_coverage_credit(self):
        result = self.run_review(
            [candidate("b1")], [assignment("b1", "multi_subject", ("p1", "p2"))]
        )
        self.assertEqual(result["separately_localized_reference_count"], 0)
        self.assertEqual(result["multi_subject_candidate_ids"], ["b1"])
        self.assertEqual(
            result["whole_frame_review_status"], "not_all_references_separately_localized"
        )

    def test_geometry_does_not_override_distinct_reviewed_targets(self):
        result = self.run_review(
            [candidate("b1"), candidate("b2")], [assignment("b1"), assignment("b2", refs=("p2",))]
        )
        self.assertEqual(result["separately_localized_reference_count"], 2)
        self.assertEqual(result["duplicate_assignment_groups"], [])
        self.assertEqual(result["whole_frame_review_status"], "all_references_separately_localized")
        self.assertFalse(result["separate_individual_coverage_verified"])

    def test_high_confidence_second_same_person_box_still_counts(self):
        result = self.run_review(
            [candidate("b1", 0.1), candidate("b2", 0.2)], [assignment("b1"), assignment("b2")]
        )
        self.assertEqual(result["separately_localized_reference_count"], 1)
        self.assertEqual(result["duplicate_assignment_groups"], [])
        self.assertEqual(len(result["candidates"]), 2)
        self.assertFalse(result["candidates"][0]["eligible"])

    def test_omitted_assignment_becomes_unknown_not_background(self):
        result = self.run_review([candidate("b1"), candidate("b2")], [assignment("b1")])
        self.assertEqual(result["unknown_candidate_ids"], ["b2"])
        self.assertEqual(result["reference_coverage"][1]["status"], "unknown")
        self.assertFalse(result["candidates"][1]["explicitly_reviewed"])

    def test_explicit_unknown_has_no_reference_credit(self):
        result = self.run_review([candidate("b1")], [assignment("b1", "unknown", ())])
        self.assertEqual(result["whole_frame_review_status"], "unknown_assignment")
        self.assertTrue(result["candidates"][0]["explicitly_reviewed"])

    def test_low_confidence_unknown_does_not_block_threshold_review(self):
        result = self.run_review([candidate("b1", 0.19)])
        self.assertEqual(result["unknown_candidate_ids"], [])
        self.assertEqual(
            result["whole_frame_review_status"], "not_all_references_separately_localized"
        )

    def test_empty_unknown_inventory_is_not_empty_scene(self):
        result = self.run_review([], refs=(), complete=False)
        self.assertEqual(result["whole_frame_review_status"], "unknown_inventory")
        self.assertEqual(result["presence"], "unknown")
        self.assertIsNone(result["true_person_count"])

    def test_empty_complete_inventory_is_not_vacuous_full_coverage(self):
        result = self.run_review([], refs=())
        self.assertEqual(result["whole_frame_review_status"], "empty_reference_inventory")
        self.assertFalse(result["separate_individual_coverage_verified"])

    def test_incomplete_inventory_cannot_claim_whole_frame_coverage(self):
        result = self.run_review(
            [candidate("b1")], [assignment("b1")], refs=("p1",), complete=False
        )
        self.assertEqual(result["separately_localized_reference_count"], 1)
        self.assertEqual(result["whole_frame_review_status"], "unknown_inventory")

    def test_background_is_distinct_from_unknown(self):
        result = self.run_review([candidate("b1")], [assignment("b1", "background", ())])
        self.assertEqual(result["unknown_candidate_ids"], [])
        self.assertEqual(result["candidates"][0]["state"], "background")

    def test_weak_review_never_becomes_human_or_deployment_evidence(self):
        boxes = [candidate("b1")]
        review = review_for(boxes, [assignment("b1")], ("p1",))
        review["provenance"] = "assistant_weak"
        result = diagnose_individual_coverage(FRAME, boxes, review, minimum_confidence=0)
        self.assertTrue(result["model_generated_labels"])
        for key in (
            "human_verified",
            "ground_truth",
            "official_accuracy_claim",
            "deployment_approved",
        ):
            self.assertFalse(result[key])

    def test_foreign_truth_provenance_and_extra_fields_rejected(self):
        for change in (
            {"provenance": "admitted_publisher_human"},
            {"human_verified": True},
            {"reference_inventory_complete": 1},
        ):
            review = review_for([])
            review.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                diagnose_individual_coverage(FRAME, [], review, minimum_confidence=0.2)

    def test_frame_box_score_order_and_id_binding(self):
        boxes = [candidate("b1"), candidate("b2")]
        review = review_for(boxes)
        mutations = [
            list(reversed(boxes)),
            [candidate("changed"), boxes[1]],
            [candidate("b1", 0.6), boxes[1]],
        ]
        mutated = copy.deepcopy(boxes)
        mutated[0]["bbox_xyxy"][0] = 0.2
        mutations.append(mutated)
        for changed in mutations:
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, "binding mismatch"),
            ):
                diagnose_individual_coverage(FRAME, changed, review, minimum_confidence=0.2)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            diagnose_individual_coverage("b" * 64, boxes, review, minimum_confidence=0.2)

    def test_assignment_id_and_cardinality_errors(self):
        boxes = [candidate("b1"), candidate("b2")]
        invalid = [
            [assignment("missing")],
            [assignment("b1"), assignment("b1")],
            [assignment("b1", refs=("missing",))],
            [assignment("b1", refs=())],
            [assignment("b1", refs=("p1", "p2"))],
            [assignment("b1", "multi_subject", ("p1",))],
            [assignment("b1", "multi_subject", ("p1", "p1"))],
            [assignment("b1", "unknown")],
            [assignment("b1", "background")],
            [assignment("b1", "duplicate")],
        ]
        for assignments in invalid:
            with self.subTest(assignments=assignments), self.assertRaises(ValueError):
                self.run_review(boxes, assignments)

    def test_reference_ids_must_be_unique_safe_and_bounded(self):
        for refs in (("p1", "p1"), ("../p1",), (True,), [f"p{i}" for i in range(257)]):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                self.run_review([], refs=refs)

    def test_numeric_and_geometry_validation(self):
        for value in (True, float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_review([candidate("b1", value)])
            with self.subTest(threshold=value), self.assertRaises(ValueError):
                self.run_review([], threshold=value)
        for bbox in ([0, 0, 0, 1], [0.9, 0, 0.1, 1], [0, 0, 1, float("nan")]):
            item = candidate("b1")
            item["bbox_xyxy"] = bbox
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                candidate_fingerprint(FRAME, [item])

    def test_confidence_boundary_inclusive(self):
        boxes = [candidate("b1", 0.2), candidate("b2", math.nextafter(0.2, 0))]
        result = self.run_review(boxes)
        self.assertEqual(result["unknown_candidate_ids"], ["b1"])

    def test_bounded_candidate_shape_and_sha(self):
        for boxes in (
            [candidate("same"), candidate("same")],
            [candidate(f"b{i}") for i in range(257)],
            {},
            [{"detection_id": "b1"}],
        ):
            with self.assertRaises(ValueError):
                candidate_fingerprint(FRAME, boxes)
        for sha in ("A" * 64, "a" * 63, None, True):
            with self.assertRaises(ValueError):
                candidate_fingerprint(sha, [])

    def test_no_mutation_or_output_aliasing(self):
        boxes = [candidate("b1")]
        review = review_for(boxes, [assignment("b1")])
        original = copy.deepcopy((boxes, review))
        result = diagnose_individual_coverage(FRAME, boxes, review, minimum_confidence=0.2)
        result["candidates"][0]["bbox_xyxy"][0] = 0
        result["candidates"][0]["reference_ids"].append("changed")
        self.assertEqual((boxes, review), original)

    def test_reordering_with_rebound_review_preserves_counts(self):
        boxes = [candidate("b1"), candidate("b2")]
        assignments = [assignment("b1"), assignment("b2", refs=("p2",))]
        first = self.run_review(boxes, assignments)
        second = self.run_review(boxes[::-1], assignments[::-1])
        self.assertEqual(first["reference_coverage"], second["reference_coverage"])
        self.assertNotEqual(first["candidates_sha256"], second["candidates_sha256"])


if __name__ == "__main__":
    unittest.main()
