from __future__ import annotations

import copy
import unittest

from scripts.compare_candidate_confirmation import (
    ARMS,
    compare_confirmation,
    compare_synthetic_cases,
    synthetic_confirmation_cases,
)


class CandidateConfirmationComparisonTests(unittest.TestCase):
    def cases(self):
        return {c["case_id"]: c for c in synthetic_confirmation_cases()}

    def test_fixed_synthetic_ablation_is_reproducible_and_explicitly_not_accuracy(self):
        report = compare_synthetic_cases()
        self.assertEqual(report, compare_synthetic_cases())
        self.assertFalse(report["household_accuracy_verified"])
        self.assertEqual(15, len(report["cases"]))
        self.assertEqual(dict(tp=2, fn=2, fp=2, tn=9), report["totals"]["whole_frame"])
        self.assertEqual(
            dict(tp=3, fn=1, fp=2, tn=9), report["totals"]["whole_frame_candidate_gate"]
        )
        self.assertEqual(dict(tp=4, fn=0, fp=1, tn=10), report["totals"]["per_candidate"])

    def test_motion_confirmation_latency_is_not_reported_as_episode_miss(self):
        report = compare_confirmation(self.cases()["smooth_true_grid_crossing"]["frames"])
        self.assertEqual(
            ["smooth_true_grid_crossing_9"], report["metrics"]["whole_frame"]["request_frames"]
        )
        self.assertEqual(
            ["smooth_true_grid_crossing_5"], report["metrics"]["per_candidate"]["request_frames"]
        )

    def test_confidence_control_separates_filtering_from_candidate_confirmation(self):
        cases = self.cases()
        weak = compare_confirmation(cases["true_with_weak_flicker"]["frames"])
        self.assertFalse(weak["metrics"]["whole_frame"]["request_frames"])
        self.assertTrue(weak["metrics"]["whole_frame_candidate_gate"]["request_frames"])
        strong = compare_confirmation(cases["true_with_strong_flicker"]["frames"])
        for name in ARMS[:2]:
            self.assertFalse(strong["metrics"][name]["request_frames"])
        self.assertTrue(strong["metrics"]["per_candidate"]["request_frames"])

    def test_raw_weak_boxes_zero_rows_and_original_input_are_preserved(self):
        frames = self.cases()["explicit_zero_gap"]["frames"]
        frames[0]["boxes"].append({"confidence": 0.1, "bbox_xyxy": [0.7, 0.1, 0.8, 0.2]})
        before = copy.deepcopy(frames)
        report = compare_confirmation(frames)
        self.assertEqual(before, frames)
        self.assertEqual(before, [row["input"] for row in report["rows"]])
        self.assertEqual([], report["rows"][3]["input"]["boxes"])
        for name in ARMS:
            self.assertEqual(len(frames), report["metrics"][name]["checked_frames"])
        self.assertFalse(report["deployment_approved"])
        self.assertEqual("unknown", report["baby_state"])

    def test_malformed_or_incomplete_fields_and_duplicate_binding_rejected(self):
        base = self.cases()["stable_true"]["frames"]
        for edit in (
            lambda rows: rows[0].pop("sequence"),
            lambda rows: rows[0].update(frame_sha256="wrong"),
            lambda rows: rows[1].update(frame_id=rows[0]["frame_id"]),
            lambda rows: rows[0].update(
                boxes=[{"confidence": True, "bbox_xyxy": [0.1, 0.1, 0.4, 0.6]}]
            ),
            lambda rows: rows[0].update(
                boxes=[{"confidence": 0.8, "bbox_xyxy": [0.4, 0.1, 0.1, 0.6]}]
            ),
            lambda rows: rows[0].update(
                boxes=[{"confidence": 0.8, "bbox_xyxy": [0.1, 0.1, 0.4, 0.6], "role": "baby"}]
            ),
        ):
            rows = copy.deepcopy(base)
            edit(rows)
            with self.assertRaises(ValueError):
                compare_confirmation(rows)
        for invalid in (None, [], (), [None], [{}], base * 400):
            with self.assertRaises(ValueError):
                compare_confirmation(invalid)

    def test_frame_order_is_replayed_not_silently_sorted(self):
        frames = self.cases()["stable_true"]["frames"][:3]
        frames[1]["sequence"] = 0
        report = compare_confirmation(frames)
        self.assertEqual(
            "invalid_order", report["rows"][1]["arms"]["per_candidate"]["reset_reason"]
        )
        self.assertFalse(report["metrics"]["per_candidate"]["request_frames"])
