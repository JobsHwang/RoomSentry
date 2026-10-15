from __future__ import annotations

import copy
import unittest

from scripts.rotation_supplement_diagnostic import SETTINGS, diagnose_supplements

SHA = "a" * 64


def box(xy, confidence=0.5):
    return {"bbox_xyxy": xy, "confidence": confidence}


BIG = box([0.05, 0.05, 0.85, 0.95])
MATCH = box([0.05, 0.05, 0.60, 0.95])
SMALL = box([0.62, 0.1, 0.82, 0.4])


def run(native, rotated, **kw):
    return diagnose_supplements(
        native, rotated, source_sha256=SHA, rotated_source_sha256=kw.get("sha", SHA)
    )


class RotationSupplementTests(unittest.TestCase):
    def test_native_reservation_and_distinct_subbox_remain_only_pending(self):
        result = run([BIG], [MATCH, SMALL])
        self.assertEqual([{"native_index": 0, "rotated_index": 0}], result["reserved_pairs"])
        self.assertEqual([1], result["pending_rotated_indices"])
        self.assertEqual([0], result["rows"][1]["native_anchor_indices"])
        self.assertEqual("unknown", result["presence"])
        self.assertEqual("unknown", result["baby_identity"])
        self.assertIsNone(result["new_person_count"])
        self.assertFalse(result["roles_propagated"])
        self.assertFalse(result["accuracy_measured"])
        self.assertFalse(result["deployment_approved"])

    def test_same_person_part_counterexample_is_not_certified_as_extra_person(self):
        # Identical geometry could be one adult and a separately predicted body
        # part. Passing this diagnostic therefore cannot establish person truth.
        result = run([BIG], [MATCH, SMALL])
        self.assertEqual("pending_subject_reference", result["rows"][1]["status"])
        self.assertIsNone(result["new_person_count"])

    def test_zero_native_does_not_create_supplements_or_absence(self):
        result = run([], [MATCH, SMALL])
        self.assertEqual([], result["pending_rotated_indices"])
        self.assertEqual(2, len(result["rotated_boxes_source_coordinates"]))
        self.assertEqual("unknown", result["presence"])
        self.assertEqual([], run([], [])["pending_rotated_indices"])

    def test_empty_rotation_keeps_every_native_box(self):
        result = run([BIG, SMALL], [])
        self.assertEqual(2, len(result["native_boxes"]))
        self.assertEqual([], result["pending_rotated_indices"])
        self.assertFalse(result["native_boxes_modified"])

    def test_duplicate_rotation_cannot_supply_or_reserve_seed(self):
        result = run([BIG], [MATCH, MATCH, SMALL])
        self.assertEqual([], result["reserved_pairs"])
        self.assertEqual([], result["pending_rotated_indices"])
        self.assertEqual("rotated_overlap_unresolved", result["rows"][0]["status"])
        result = run([BIG], [MATCH, SMALL, SMALL])
        self.assertEqual([], result["pending_rotated_indices"])
        self.assertEqual(3, len(result["rotated_boxes_source_coordinates"]))

    def test_many_to_one_or_one_to_many_not_greedily_reserved(self):
        result = run([BIG, BIG], [MATCH, SMALL])
        self.assertEqual([], result["reserved_pairs"])
        self.assertEqual([], result["pending_rotated_indices"])
        self.assertEqual("native_iou_correspondence_unresolved", result["rows"][0]["status"])

    def test_existing_unreserved_native_center_blocks_new_subbox(self):
        covering = box([0.60, 0.05, 0.84, 0.7])
        result = run([BIG, covering], [MATCH, SMALL])
        self.assertEqual([], result["pending_rotated_indices"])
        self.assertEqual("unreserved_native_center_coverage", result["rows"][1]["status"])

    def test_uncontained_or_large_or_unseeded_candidates_do_not_pass(self):
        for candidate in (box([0.7, 0.1, 0.95, 0.4]), box([0.05, 0.05, 0.85, 0.95])):
            self.assertEqual([], run([BIG], [MATCH, candidate])["pending_rotated_indices"])
        self.assertEqual([], run([BIG], [SMALL])["pending_rotated_indices"])

    def test_threshold_and_weak_rows_are_preserved(self):
        self.assertEqual(
            [1], run([BIG], [MATCH, box(SMALL["bbox_xyxy"], 0.35)])["pending_rotated_indices"]
        )
        result = run([BIG], [MATCH, box(SMALL["bbox_xyxy"], 0.349)])
        self.assertEqual("below_confidence", result["rows"][1]["status"])
        self.assertEqual([], result["pending_rotated_indices"])
        self.assertEqual(
            [], run([box(BIG["bbox_xyxy"], 0.349)], [MATCH, SMALL])["pending_rotated_indices"]
        )

    def test_hash_and_bounded_invalid_geometry_fail_closed(self):
        for sha in ("b" * 64, "bad", True):
            with self.subTest(sha=sha), self.assertRaises(ValueError):
                run([], [], sha=sha)
        for invalid in (None, (), [BIG] * 101, [box([0, 0, 0, 1])], [box([0, 0, 1, 1], True)]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                run([], invalid)

    def test_permutation_equivariance_and_no_input_mutation(self):
        native, rotated = [BIG], [MATCH, SMALL]
        before = copy.deepcopy((native, rotated))
        result = run(native, rotated)
        self.assertEqual(before, (native, rotated))
        self.assertEqual([1], result["pending_rotated_indices"])
        self.assertEqual([0], run(native, list(reversed(rotated)))["pending_rotated_indices"])
        result["settings"]["confidence"] = 0
        self.assertEqual(0.35, SETTINGS["confidence"])
        result["native_boxes"][0]["bbox_xyxy"][0] = 0
        self.assertEqual(before, (native, rotated))


if __name__ == "__main__":
    unittest.main()
