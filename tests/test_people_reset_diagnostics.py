"""Exact continuity-boundary telemetry, without changing tracker decisions."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from datetime import timedelta

from baby_monitor.live_people import LivePeople
from tests.test_live_people import BASE, claims, observation


class PeopleResetDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.people = LivePeople(("a", "b"), auxiliary=True, association_diagnostics=True)

    def trace(self, obs, *, view="original"):
        return self.people.review_trace(obs, view=view)

    def test_initial_input_is_not_a_decoder_failure_or_gap(self):
        obs = observation()
        self.people.observe(obs)
        trace = self.trace(obs)
        self.assertEqual({"initial_input": 1}, trace["history_reset_reasons_total"])
        reset = trace["latest_history_reset"]
        self.assertEqual(["initial_input"], reset["reasons"])
        self.assertIsNone(reset["previous_sequence"])
        self.assertIsNone(reset["elapsed_receipt_gap_seconds"])
        self.assertEqual(0, reset["discarded_history_frames"])
        self.assertFalse(trace["reset_diagnostics"]["complete_reset_event_history"])

    def test_monotonic_and_utc_gaps_are_independent(self):
        for mono, utc, expected in (
            (0.1, 3, ["utc_gap"]),
            (3, 0.1, ["monotonic_gap"]),
            (3, 4, ["monotonic_gap", "utc_gap"]),
        ):
            with self.subTest(mono=mono, utc=utc):
                self.people = LivePeople(("a",), association_diagnostics=True)
                first = observation()
                self.people.observe(first)
                new = observation(1, elapsed=mono)
                new = replace(
                    new, frame=replace(new.frame, received_at=BASE + timedelta(seconds=utc))
                )
                self.people.observe(new)
                reset = self.trace(new)["latest_history_reset"]
                self.assertEqual(expected, reset["reasons"])
                self.assertEqual(mono, reset["elapsed_receipt_gap_seconds"])
                self.assertEqual(utc, reset["utc_receipt_gap_seconds"])
                self.assertEqual(1, reset["discarded_history_frames"])
                self.assertEqual(1, reset["discarded_visible_tracks"])

    def test_non_increasing_clocks_are_not_called_reconnections(self):
        for mono, utc, expected in (
            (0, 1, ["non_increasing_monotonic"]),
            (1, 0, ["non_increasing_utc"]),
            (-0.1, -0.2, ["non_increasing_monotonic", "non_increasing_utc"]),
        ):
            with self.subTest(mono=mono, utc=utc):
                p = LivePeople(("a",), association_diagnostics=True)
                p.observe(observation())
                new = observation(1, elapsed=mono)
                new = replace(
                    new, frame=replace(new.frame, received_at=BASE + timedelta(seconds=utc))
                )
                p.observe(new)
                self.assertEqual(expected, p.review_trace(new)["latest_history_reset"]["reasons"])

    def test_generation_change_can_have_overlapping_clock_causes(self):
        self.people.observe(observation())
        new = observation(1, generation=1, elapsed=3)
        self.people.observe(new)
        trace = self.trace(new)
        self.assertEqual(
            {"initial_input": 1, "generation_changed": 1, "monotonic_gap": 1, "utc_gap": 1},
            trace["history_reset_reasons_total"],
        )
        self.assertEqual(2, trace["history_resets_total"])
        self.assertTrue(trace["reset_diagnostics"]["cause_counts_may_overlap"])
        reset = trace["latest_history_reset"]
        self.assertEqual((0, 1), (reset["previous_generation"], reset["generation"]))

    def test_exact_threshold_does_not_create_a_reset(self):
        self.people.observe(observation())
        new = observation(1, elapsed=2)
        self.people.observe(new)
        trace = self.trace(new)
        self.assertEqual(1, trace["history_resets_total"])
        self.assertEqual({"initial_input": 1}, trace["history_reset_reasons_total"])

    def test_unavailable_old_review_still_carries_exact_latest_boundary(self):
        old = observation()
        self.people.observe(old)
        self.people.observe(observation(1, elapsed=4))
        trace = self.trace(old)
        self.assertEqual("review_frame_unavailable", trace["status"])
        self.assertEqual([], trace["frames"])
        reset = trace["latest_history_reset"]
        self.assertEqual((0, 1), (reset["previous_sequence"], reset["sequence"]))
        self.assertEqual(4, reset["elapsed_receipt_gap_seconds"])

    def test_duplicate_old_sequence_or_generation_does_not_change_diagnostics(self):
        first = observation(5, generation=1, elapsed=0)
        self.people.observe(first)
        before = self.trace(first)
        for sequence, generation in ((5, 1), (4, 1), (10, 0)):
            self.assertFalse(self.people.observe(observation(sequence, generation=generation)))
            self.assertEqual(before, self.trace(first))

    def test_native_auxiliary_and_cameras_remain_independent(self):
        native = observation()
        self.people.observe(native)
        self.people.observe_auxiliary(native, view="rotation_180")
        alternate = observation(1, elapsed=4)
        self.people.observe_auxiliary(alternate, view="rotation_180")
        self.assertEqual({"initial_input": 1}, self.trace(native)["history_reset_reasons_total"])
        self.assertEqual(2, self.trace(alternate, view="rotation_180")["history_resets_total"])
        self.assertEqual({}, self.people.cameras["b"]["history_reset_reasons"])
        self.assertIsNone(self.people.cameras["b"]["latest_history_reset"])

    def test_export_is_copied_not_mutable_runtime_state(self):
        obs = observation()
        self.people.observe(obs)
        trace = self.trace(obs)
        trace["history_reset_reasons_total"].clear()
        trace["latest_history_reset"]["reasons"].clear()
        self.assertEqual({"initial_input": 1}, self.trace(obs)["history_reset_reasons_total"])
        self.assertEqual(["initial_input"], self.trace(obs)["latest_history_reset"]["reasons"])

    def test_telemetry_is_bounded_and_strict_json(self):
        for i in range(100):
            self.people.observe(observation(i, elapsed=i * 3))
        trace = self.trace(observation(99, elapsed=297))
        self.assertEqual(100, trace["history_resets_total"])
        self.assertEqual(
            {"initial_input": 1, "monotonic_gap": 99, "utc_gap": 99},
            trace["history_reset_reasons_total"],
        )
        self.assertEqual(99, trace["latest_history_reset"]["sequence"])
        self.assertEqual(trace, json.loads(json.dumps(trace, allow_nan=False)))
        self.assertLess(len(json.dumps(trace["latest_history_reset"])), 1100)

    def test_default_disabled_trace_keeps_protocol_fields_absent(self):
        p = LivePeople(("a",))
        obs = observation()
        p.observe(obs)
        trace = p.review_trace(obs)
        self.assertNotIn("history_reset_reasons_total", trace)
        self.assertNotIn("latest_history_reset", trace)
        self.assertNotIn("reset_diagnostics", trace)

    def test_diagnostics_do_not_change_assignments_roles_or_projection(self):
        plain = LivePeople(("a",), auxiliary=True)
        for i, elapsed in enumerate((0, 1, 1.2, 4, 4.1, 5.2)):
            obs = observation(i, elapsed=elapsed)
            for p in (plain, self.people):
                p.observe(obs)
                p.apply_review(
                    obs,
                    claims(),
                    evidence_ref="synthetic.png",
                    now=elapsed,
                    now_utc=BASE + timedelta(seconds=elapsed),
                )
            left = plain.cameras["a"]
            right = self.people.cameras["a"]
            # UUID namespaces differ; compare the canonical detector facts,
            # matching hits and independent role evidence, not literal IDs.
            self.assertEqual(
                [(v["bbox"], v["hits"], v["epoch"], v["role"]) for v in left["tracks"].values()],
                [(v["bbox"], v["hits"], v["epoch"], v["role"]) for v in right["tracks"].values()],
            )
            self.assertEqual(left["history_resets"], right["history_resets"])
            self.assertEqual(plain.counters, self.people.counters)


if __name__ == "__main__":
    unittest.main()
