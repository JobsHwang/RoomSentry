from __future__ import annotations

import copy
import unittest

from scripts.candidate_confirmation_cadence import compare_cadence
from scripts.compare_candidate_confirmation import synthetic_confirmation_cases


class CandidateConfirmationCadenceTests(unittest.TestCase):
    def frames(self):
        return synthetic_confirmation_cases()[0]["frames"]

    def test_fixed_interval_uses_actual_receipts_and_keeps_source_ids(self):
        frames = self.frames()
        before = copy.deepcopy(frames)
        result = compare_cadence(frames, phases=(0.0, 0.2))
        self.assertEqual(before, frames)
        self.assertEqual(
            ["stable_true_0", "stable_true_5", "stable_true_10", "stable_true_15"],
            result["phases"][0]["selected_frame_ids"],
        )
        self.assertEqual(
            ["stable_true_1", "stable_true_6", "stable_true_11"],
            result["phases"][1]["selected_frame_ids"],
        )
        self.assertEqual(12, result["phases"][0]["unselected_frames"])
        self.assertFalse(result["independent_trials"])
        self.assertFalse(result["measured_live_auxiliary_cadence"])

    def test_selected_zeros_remain_checks_and_do_not_become_missing(self):
        frames = self.frames()
        frames[5]["boxes"] = []
        result = compare_cadence(frames, phases=(0.0,))
        report = result["phases"][0]["comparison"]
        self.assertEqual([], report["rows"][1]["input"]["boxes"])
        self.assertEqual(["stable_true_15"], report["metrics"]["per_candidate"]["request_frames"])

    def test_invalid_unselected_frames_cannot_be_hidden_by_sampling(self):
        for edit in (
            lambda rows: rows[1].update(sequence=0),
            lambda rows: rows[1].update(received_monotonic=0.0),
            lambda rows: rows[1].update(
                boxes=[{"confidence": True, "bbox_xyxy": [0.1, 0.1, 0.4, 0.6]}]
            ),
        ):
            rows = self.frames()
            edit(rows)
            with self.assertRaises(ValueError):
                compare_cadence(rows, phases=(0.0,))

    def test_generation_origins_are_independent_and_stale_generation_rejected(self):
        frames = self.frames()
        for i, row in enumerate(frames[8:]):
            row["generation"] = 1
            row["sequence"] = i
            row["received_monotonic"] = row["completed_monotonic"] = i / 5
        result = compare_cadence(frames, phases=(0.2,))
        self.assertIn("stable_true_9", result["phases"][0]["selected_frame_ids"])
        frames[10]["generation"] = 0
        with self.assertRaises(ValueError):
            compare_cadence(frames)

    def test_camera_sampling_origins_are_isolated(self):
        frames = self.frames()
        other = copy.deepcopy(frames)
        for row in other:
            row["frame_id"] += "_b"
            row["camera_id"] = "second"
        rows = [item for pair in zip(frames, other, strict=True) for item in pair]
        result = compare_cadence(rows, phases=(0.0,))
        self.assertEqual(8, len(result["phases"][0]["selected_frame_ids"]))

    def test_no_frames_at_phase_is_explicit_not_an_empty_success(self):
        result = compare_cadence(self.frames()[:1], phases=(0.8,))
        self.assertIsNone(result["phases"][0]["comparison"])
        self.assertEqual([], result["phases"][0]["selected_frame_ids"])

    def test_invalid_interval_and_phase_types_or_duplicate_phases_rejected(self):
        for settings in (
            {"interval_seconds": True},
            {"interval_seconds": float("inf")},
            {"interval_seconds": 0},
            {"interval_seconds": 10**1000},
            {"phases": []},
            {"phases": ()},
            {"phases": (0.0, 0.0)},
            {"phases": (True,)},
            {"phases": (float("nan"),)},
            {"phases": (1.0,)},
            {"phases": (10**1000,)},
            {"phases": (-0.1,)},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                compare_cadence(self.frames(), **settings)
