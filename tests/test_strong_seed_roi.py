import copy
import math
import unittest

from scripts.strong_seed_roi import evaluate_crops, plan_regions


def box(bounds, confidence=0.8):
    return {"bbox_xyxy": bounds, "confidence": confidence}


class StrongSeedRoiTests(unittest.TestCase):
    def plan(self, previous=None, current=None, **kwargs):
        return plan_regions(
            previous if previous is not None else [box([0.4, 0.4, 0.6, 0.6])],
            current if current is not None else [],
            width=1000,
            height=1000,
            elapsed_seconds=kwargs.pop("elapsed_seconds", 0.1),
            **kwargs,
        )

    def test_missing_seed_maps_to_source_without_role_or_person_fact(self):
        p = self.plan()
        region = p["regions"][0]["pixel_xyxy"]
        x1, y1, x2, y2 = region
        local = [
            (400 - x1) / (x2 - x1),
            (400 - y1) / (y2 - y1),
            (600 - x1) / (x2 - x1),
            (600 - y1) / (y2 - y1),
        ]
        result = evaluate_crops(p, [[box(local)]])
        self.assertEqual(len(result["proposals"]), 1)
        for actual, expected in zip(
            result["proposals"][0]["bbox_xyxy"], [0.4, 0.4, 0.6, 0.6], strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertIsNone(result["person_count"])
        self.assertFalse(result["role_inheritance"])
        self.assertFalse(result["deployment_approved"])

    def test_existing_native_and_enclosing_box_block_duplicate_crop(self):
        for current in ([box([0.4, 0.4, 0.6, 0.6])], [box([0.1, 0.1, 0.9, 0.9])]):
            with self.subTest(current=current):
                p = self.plan(current=current)
                self.assertEqual(p["regions"], [])
                self.assertEqual(evaluate_crops(p, [])["native"], current)

    def test_prior_overlap_blocks_arbitrary_person_selection(self):
        p = self.plan(previous=[box([0.4, 0.4, 0.6, 0.6]), box([0.41, 0.41, 0.61, 0.61])])
        self.assertEqual(p["regions"], [])
        self.assertEqual({r["reason"] for r in p["seed_decisions"]}, {"prior_geometry_ambiguous"})

    def test_disjoint_survivor_is_preserved(self):
        current = [box([0.8, 0.8, 0.9, 0.9])]
        p = self.plan(current=current)
        r = evaluate_crops(p, [[]])
        self.assertEqual(len(p["regions"]), 1)
        self.assertEqual(r["native"], current)
        self.assertEqual(r["proposals"], [])

    def test_weak_seed_and_gap_boundaries(self):
        self.assertEqual(self.plan(previous=[box([0.4, 0.4, 0.6, 0.6], 0.3499)])["regions"], [])
        for gap in (0, -1, 2.00001):
            self.assertEqual(self.plan(elapsed_seconds=gap)["regions"], [])
        self.assertEqual(len(self.plan(elapsed_seconds=2)["regions"]), 1)
        self.assertEqual(self.plan(same_generation=False)["regions"], [])
        for gap in (True, math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.plan(elapsed_seconds=gap)

    def test_no_seed_or_no_native_does_not_invent_seeds(self):
        self.assertEqual(self.plan(previous=[])["regions"], [])
        self.assertEqual(self.plan(previous=[], current=[box([0.4, 0.4, 0.6, 0.6])])["regions"], [])

    def test_seams_wrong_location_low_score_and_multiple_candidates_reject(self):
        p = self.plan()
        for boxes, reason in (
            ([box([0, 0.2, 0.8, 0.8])], "internal_crop_boundary"),
            ([box([0.05, 0.05, 0.1, 0.1])], "not_seed_geometry"),
            ([box([0.25, 0.25, 0.75, 0.75], 0.2)], "crop_not_strong"),
            (
                [box([0.25, 0.25, 0.75, 0.75]), box([0.26, 0.26, 0.74, 0.74])],
                "multiple_candidates_for_seed",
            ),
        ):
            with self.subTest(reason=reason):
                result = evaluate_crops(p, [boxes])
                self.assertEqual(result["proposals"], [])
                self.assertEqual({r["reason"] for r in result["crop_diagnostics"]}, {reason})

    def test_inputs_are_not_mutated(self):
        p, rows = self.plan(), [[box([0.25, 0.25, 0.75, 0.75])]]
        before = copy.deepcopy((p, rows))
        result = evaluate_crops(p, rows)
        result["proposals"][0]["bbox_xyxy"][0] = 0
        self.assertEqual((p, rows), before)

    def test_same_geometry_from_false_seed_cannot_certify_person(self):
        result = evaluate_crops(self.plan(), [[box([0.25, 0.25, 0.75, 0.75])]])
        self.assertEqual(len(result["proposals"]), 1)
        self.assertFalse(result["baby_accuracy_measured"])
        self.assertEqual(result["proposals"][0]["reason"], "pending_visible_person_reference")

    def test_capacity_is_explicit_and_invalid_geometry_rejected(self):
        p = self.plan(previous=[box([i * 0.15, 0.1, i * 0.15 + 0.05, 0.2]) for i in range(5)])
        self.assertEqual(len(p["regions"]), 4)
        self.assertEqual(p["seed_decisions"][-1]["reason"], "region_capacity")
        with self.assertRaises(ValueError):
            self.plan(previous=[box([0.5, 0.1, 0.4, 0.2])])
        with self.assertRaises(ValueError):
            evaluate_crops(self.plan(), [])
