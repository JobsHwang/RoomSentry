from __future__ import annotations

import unittest
from dataclasses import replace

from baby_monitor.candidate_confirmation import (
    CandidateConfirmationPolicy,
    ConfirmationFrame,
    ConfirmationSettings,
)
from baby_monitor.domain import Detection

A = Detection(0, "person", 0.8, (0.1, 0.1, 0.4, 0.6))
B = Detection(0, "person", 0.8, (0.7, 0.1, 0.95, 0.6))


def metadata(
    sequence=0, received=0.0, detections=(A,), *, completed=None, generation=0, camera="camera_01"
):
    return ConfirmationFrame(
        camera,
        generation,
        sequence,
        received,
        received if completed is None else completed,
        detections,
    )


class CandidateConfirmationTests(unittest.TestCase):
    def policy(self):
        return CandidateConfirmationPolicy(("camera_01", "camera_02"))

    def test_measured_chain_confirms_at_one_second_not_before(self):
        policy = self.policy()
        for sequence, time in enumerate((0.0, 0.2, 0.4, 0.6, 0.8, 0.999)):
            result = policy.observe(metadata(sequence, time))
            self.assertEqual((), result.review_requested_indices)
        result = policy.observe(metadata(6, 1.0))
        self.assertEqual((0,), result.review_requested_indices)
        self.assertEqual(7, result.chains[0].measured_frames)
        self.assertTrue(result.chains[0].confirmed)

    def test_additional_flickering_candidate_does_not_reset_stable_chain(self):
        policy = self.policy()
        ids = []
        for sequence in range(7):
            result = policy.observe(
                metadata(sequence, sequence / 5, (A, B) if sequence % 2 else (A,))
            )
            ids.append(result.chains[0].chain_id)
        self.assertEqual(1, len(set(ids)))
        self.assertTrue(result.chains[0].confirmed)

    def test_current_candidate_order_is_not_identity(self):
        policy = self.policy()
        first = policy.observe(metadata(detections=(A, B)))
        second = policy.observe(metadata(1, 1.0, (B, A)))
        self.assertEqual(
            [c.chain_id for c in first.chains], list(reversed([c.chain_id for c in second.chains]))
        )
        self.assertEqual((0, 1), second.review_requested_indices)

    def test_smooth_motion_crossing_signature_cells_can_confirm(self):
        policy = self.policy()
        requests = []
        for i in range(16):
            box = replace(A, bbox_xyxy=(0.1 + i * 0.02, 0.1, 0.4 + i * 0.02, 0.6))
            result = policy.observe(metadata(i, i / 5, (box,)))
            requests.extend(result.review_requested_indices)
        self.assertEqual([0], requests)

    def test_one_or_two_transient_frames_do_not_confirm(self):
        for length in (1, 2):
            policy = self.policy()
            for i in range(8):
                result = policy.observe(metadata(i, i / 5, (A,) if i < length else ()))
                self.assertFalse(result.review_requested_indices)

    def test_alternating_disjoint_boxes_cannot_combine_votes(self):
        policy = self.policy()
        for i in range(16):
            result = policy.observe(metadata(i, i / 5, (A,) if i % 2 else (B,)))
            self.assertFalse(result.chains[0].confirmed)

    def test_persistent_false_box_remains_a_known_limitation(self):
        policy = self.policy()
        policy.observe(metadata())
        # Geometry has no semantics: a stable nonperson proposal would also pass.
        self.assertEqual((0,), policy.observe(metadata(1, 1.0)).review_requested_indices)

    def test_only_candidate_gate_not_other_strong_box_can_confirm_weak_box(self):
        policy = self.policy()
        pair = (A, replace(B, confidence=0.39))
        policy.observe(metadata(detections=pair))
        result = policy.observe(metadata(1, 1.0, pair))
        self.assertEqual((1,), result.below_gate_indices)
        self.assertEqual((0,), result.review_requested_indices)
        self.assertEqual(pair, metadata(1, 1.0, pair).detections)

    def test_confidence_boundary_and_weak_gap(self):
        policy = self.policy()
        boundary = replace(A, confidence=0.4)
        policy.observe(metadata(detections=(boundary,)))
        self.assertEqual(
            (0,), policy.observe(metadata(1, 1.0, (boundary,))).review_requested_indices
        )
        policy = self.policy()
        policy.observe(metadata())
        policy.observe(metadata(1, 0.5, (replace(A, confidence=0.399),)))
        self.assertFalse(policy.observe(metadata(2, 1.0)).chains[0].confirmed)

    def test_explicit_zero_frame_breaks_chain_without_prediction(self):
        policy = self.policy()
        first = policy.observe(metadata()).chains[0].chain_id
        self.assertEqual((), policy.observe(metadata(1, 0.5, ())).chains)
        next_result = policy.observe(metadata(2, 1.0))
        self.assertNotEqual(first, next_result.chains[0].chain_id)
        self.assertFalse(next_result.chains[0].confirmed)

    def test_explicit_unchecked_gap_breaks_chain_without_fake_zero(self):
        policy = self.policy()
        policy.observe(metadata())
        policy.gap("camera_01")
        self.assertFalse(policy.observe(metadata(1, 1.0)).chains[0].confirmed)

    def test_receipt_and_completion_span_are_both_required(self):
        for first, second in (
            (metadata(completed=1.0), metadata(1, 1.0, completed=1.1)),
            (metadata(), metadata(1, 0.2, completed=1.0)),
        ):
            policy = self.policy()
            policy.observe(first)
            self.assertFalse(policy.observe(second).review_requested_indices)

    def test_gap_in_either_clock_breaks_chain(self):
        for second in (metadata(1, 3.01), metadata(1, 0.2, completed=3.01)):
            policy = self.policy()
            policy.observe(metadata())
            result = policy.observe(second)
            self.assertEqual("clock_gap", result.reset_reason)
            self.assertFalse(result.chains[0].confirmed)

    def test_gap_boundary_is_inclusive(self):
        policy = self.policy()
        policy.observe(metadata())
        self.assertEqual((0,), policy.observe(metadata(1, 3.0)).review_requested_indices)

    def test_duplicate_sequence_or_receipt_and_completion_regression_fail_closed(self):
        for invalid in (
            metadata(0, 1.0),
            metadata(1, 0.0, completed=1.0),
            metadata(1, 0.5, completed=0.5),
        ):
            policy = self.policy()
            policy.observe(metadata(completed=1.0))
            result = policy.observe(invalid)
            self.assertEqual("invalid_order", result.reset_reason)
            self.assertFalse(result.chains)
            self.assertFalse(policy.observe(metadata(2, 2.0, completed=2.0)).chains[0].confirmed)

    def test_invalid_order_high_water_does_not_roll_back(self):
        policy = self.policy()
        policy.observe(metadata(10, 10.0))
        policy.observe(metadata(11, 9.0, completed=10.0))
        self.assertEqual("invalid_order", policy.observe(metadata(10, 11.0)).reset_reason)
        self.assertFalse(policy.observe(metadata(12, 12.0)).chains[0].confirmed)

    def test_generation_change_and_old_generation_do_not_join_or_roll_back(self):
        policy = self.policy()
        policy.observe(metadata())
        policy.observe(metadata(1, 0.5))
        new = policy.observe(metadata(0, 1.0, generation=1))
        self.assertFalse(new.chains[0].confirmed)
        self.assertEqual("generation", new.reset_reason)
        stale = policy.observe(metadata(99, 50.0, generation=0))
        self.assertEqual("stale_generation", stale.reset_reason)
        self.assertFalse(policy.observe(metadata(1, 1.5, generation=1)).chains[0].confirmed)
        self.assertTrue(policy.observe(metadata(2, 2.5, generation=1)).review_requested_indices)

    def test_generation_change_preserves_camera_cooldown(self):
        policy = self.policy()
        policy.observe(metadata())
        policy.observe(metadata(1, 1.0))
        policy.observe(metadata(0, 2.0, generation=1))
        result = policy.observe(metadata(1, 3.0, generation=1))
        self.assertEqual((), result.review_requested_indices)
        self.assertEqual((0,), result.cooldown_blocked_indices)

    def test_cameras_have_independent_confirmation_and_cooldown(self):
        policy = self.policy()
        policy.observe(metadata())
        self.assertFalse(
            policy.observe(metadata(1, 1.0, camera="camera_02")).review_requested_indices
        )
        self.assertEqual((0,), policy.observe(metadata(1, 1.0)).review_requested_indices)
        self.assertEqual(
            (0,), policy.observe(metadata(2, 2.0, camera="camera_02")).review_requested_indices
        )

    def test_accepted_chain_requests_once_but_new_chain_pending_can_request_later(self):
        policy = self.policy()
        policy.observe(metadata())
        self.assertTrue(policy.observe(metadata(1, 1.0)).review_requested_indices)
        policy.observe(metadata(2, 2.0, (A, B)))
        result = policy.observe(metadata(3, 3.0, (A, B)))
        self.assertEqual((1,), result.cooldown_blocked_indices)
        for i in range(4, 32):
            result = policy.observe(metadata(i, float(i), (A, B)))
        self.assertEqual((1,), result.review_requested_indices)
        self.assertFalse(policy.observe(metadata(32, 32.0, (A, B))).review_requested_indices)

    def test_identical_and_overlapping_current_boxes_are_ambiguous_even_on_first_frame(self):
        for pair in ((A, A), (A, replace(A, bbox_xyxy=(0.15, 0.1, 0.45, 0.6)))):
            policy = self.policy()
            for i in range(8):
                result = policy.observe(metadata(i, float(i), pair))
                self.assertEqual((0, 1), result.ambiguous_indices)
                self.assertFalse(result.chains)

    def test_many_to_one_and_one_to_many_edges_do_not_greedily_merge(self):
        left = replace(A, bbox_xyxy=(0.1, 0.1, 0.4, 0.6))
        right = replace(A, bbox_xyxy=(0.27, 0.1, 0.57, 0.6))
        middle = replace(A, bbox_xyxy=(0.185, 0.1, 0.485, 0.6))
        for first, second in (((left, right), (middle,)), ((middle,), (left, right))):
            policy = self.policy()
            policy.observe(metadata(detections=first))
            result = policy.observe(metadata(1, 1.0, second))
            self.assertEqual(tuple(range(len(second))), result.ambiguous_indices)
            self.assertFalse(result.chains)
            self.assertFalse(policy.observe(metadata(2, 2.0, (left,))).chains[0].confirmed)

    def test_box_limit_explicit_and_state_bounded(self):
        policy = self.policy()
        result = policy.observe(metadata(detections=(A,) * 100))
        self.assertTrue(result.source_at_box_limit)
        self.assertEqual(100, len(result.ambiguous_indices))
        self.assertFalse(result.chains)
        with self.assertRaises(ValueError):
            metadata(detections=(A,) * 101)

    def test_frame_rejects_invalid_identifiers_clocks_types_and_nonperson(self):
        for changes in (
            {"camera_id": "../secret"},
            {"camera_id": 1},
            {"generation": True},
            {"generation": -1},
            {"sequence": True},
            {"received_monotonic": True},
            {"received_monotonic": float("nan")},
            {"completed_monotonic": float("inf")},
            {"received_monotonic": 1.0, "completed_monotonic": 0.0},
            {"detections": [A]},
            {"detections": (replace(A, label="cat"),)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(metadata(), **changes)

    def test_allowlist_settings_and_unadmitted_frames_are_rejected(self):
        for cameras in ((), ["a"], ("a", "a"), ("../a",), tuple(str(i) for i in range(65))):
            with self.assertRaises(ValueError):
                CandidateConfirmationPolicy(cameras)
        with self.assertRaises(ValueError):
            self.policy().observe(metadata(camera="unconfigured"))
        with self.assertRaises(ValueError):
            self.policy().observe(None)
        with self.assertRaises(ValueError):
            CandidateConfirmationPolicy(("a",), {})
        for name, values in (
            ("minimum_candidate_confidence", (True, -1, 1.1, float("nan"))),
            ("minimum_confirmation_seconds", (0.99, 3.1)),
            ("minimum_iou", (0, 1.1)),
            ("maximum_gap_seconds", (0, 3.1)),
            ("review_cooldown_seconds", (29, 121)),
        ):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    ConfirmationSettings(**{name: value})
