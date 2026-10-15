from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

from baby_monitor.live_detection import LiveFrame
from baby_monitor.live_people import LivePeople
from baby_monitor.tracker_evaluation._shared import TrackerEvaluationError
from baby_monitor.tracker_evaluation.clock_boundary import compare_utc_only_reset_checkpoint
from tests.test_live_people import BASE, BOX, OTHER, claims, observation


class ClockBoundaryCheckpointTests(unittest.TestCase):
    def capture(self, *, boxes=(BOX,), previous_boxes=(BOX,), wall=3, auxiliary=False):
        people = LivePeople(("a",), auxiliary=auxiliary, association_diagnostics=True)
        people.session = "synthetic"
        rows = [
            observation(0, boxes=previous_boxes, elapsed=0),
            observation(1, boxes=previous_boxes, elapsed=0.1),
            observation(2, boxes=boxes, elapsed=0.2),
        ]
        rows[-1] = replace(
            rows[-1], frame=replace(rows[-1].frame, received_at=BASE + timedelta(seconds=wall))
        )
        for obs in rows:
            if auxiliary:
                people.observe_auxiliary(obs, view="rotation_180")
            else:
                people.observe(obs)
        trace = people.review_trace(rows[-1], view="rotation_180" if auxiliary else "original")
        return people, rows, trace["latest_reset_checkpoint"]

    def test_actual_reset_baseline_reproduces_before_geometry_compare(self):
        people, rows, checkpoint = self.capture()
        report = compare_utc_only_reset_checkpoint(checkpoint)
        self.assertTrue(report["actual_reset_assignments_reproduced"])
        self.assertTrue(report["actual_reset_audit_reproduced"])
        self.assertEqual(1, len(report["candidate_adjacent_geometry_links"]))
        self.assertEqual(checkpoint["previous_assignment_ids"], report["candidate_assignment_ids"])
        self.assertNotEqual(report["original_assignment_ids"], report["candidate_assignment_ids"])
        self.assertEqual(checkpoint["original_assignment_ids"], list(people.cameras["a"]["tracks"]))
        self.assertFalse(report["accuracy_measured"])
        self.assertFalse(report["role_policy_evaluated"])
        self.assertFalse(report["old_async_review_admitted"])
        self.assertEqual(0, report["extra_model_calls"])
        self.assertTrue(report["single_boundary_not_continuous_session"])

    def test_backward_and_equal_utc_with_healthy_monotonic_can_be_compared(self):
        for wall in (0.1, -3):
            with self.subTest(wall=wall):
                _, _, checkpoint = self.capture(wall=wall)
                report = compare_utc_only_reset_checkpoint(checkpoint)
                self.assertEqual(["non_increasing_utc"], report["boundary_reasons"])
                self.assertEqual(1, len(report["candidate_adjacent_geometry_links"]))

    def test_zero_boxes_are_preserved_not_fabricated_as_people(self):
        for current, previous in (((), ()), ((), (BOX,)), ((BOX,), ())):
            with self.subTest():
                _, _, checkpoint = self.capture(boxes=current, previous_boxes=previous)
                report = compare_utc_only_reset_checkpoint(checkpoint)
                self.assertEqual(len(current), len(report["candidate_assignment_ids"]))
                self.assertEqual([], report["candidate_adjacent_geometry_links"])

    def test_distant_box_is_not_forced_to_old_id(self):
        _, _, checkpoint = self.capture(boxes=(OTHER,))
        report = compare_utc_only_reset_checkpoint(checkpoint)
        self.assertEqual([], report["candidate_adjacent_geometry_links"])
        self.assertEqual(report["original_assignment_ids"], report["candidate_assignment_ids"])

    def test_two_people_keep_one_to_one_geometry_not_a_merged_claim(self):
        _, _, checkpoint = self.capture(boxes=(BOX, OTHER), previous_boxes=(BOX, OTHER))
        report = compare_utc_only_reset_checkpoint(checkpoint)
        self.assertEqual(2, len(report["candidate_adjacent_geometry_links"]))
        self.assertEqual(2, len(set(report["candidate_assignment_ids"])))

    def test_auxiliary_lane_has_its_own_checkpoint(self):
        people, rows, checkpoint = self.capture(auxiliary=True)
        self.assertTrue(
            compare_utc_only_reset_checkpoint(checkpoint)["actual_reset_assignments_reproduced"]
        )
        self.assertIsNone(people.cameras["a"]["latest_reset_checkpoint"])
        people.observe(rows[-1])
        self.assertIsNone(people.review_trace(rows[-1])["latest_reset_checkpoint"])

    def test_production_roles_and_old_async_binding_still_fail_closed(self):
        people = LivePeople(("a",), association_diagnostics=True)
        old = observation(0, elapsed=0)
        people.observe(old)
        people.apply_review(old, claims(), evidence_ref="synthetic.png", now=0, now_utc=BASE)
        new = observation(1, elapsed=0.1)
        new = replace(new, frame=replace(new.frame, received_at=BASE + timedelta(seconds=3)))
        people.observe(new)
        self.assertTrue(all(t["role"] is None for t in people.cameras["a"]["tracks"].values()))
        before = copy.deepcopy(people.counters)
        checkpoint = people.review_trace(new)["latest_reset_checkpoint"]
        compare_utc_only_reset_checkpoint(checkpoint)
        self.assertEqual(before, people.counters)
        people.apply_review(
            old,
            claims(),
            evidence_ref="synthetic.png",
            now=0.1,
            now_utc=BASE + timedelta(seconds=3),
        )
        self.assertTrue(all(t["role"] is None for t in people.cameras["a"]["tracks"].values()))
        self.assertNotIn('"role":', json.dumps(checkpoint))
        self.assertEqual([], checkpoint["before_state"]["pending_observations"])

    def test_export_is_private_metadata_only_and_detached(self):
        people, rows, checkpoint = self.capture()
        clone = json.loads(json.dumps(checkpoint, allow_nan=False))
        self.assertEqual(checkpoint, clone)
        checkpoint["before_state"]["active_tracks"][0]["current"]["bbox_xyxy"][0] = 0
        again = people.review_trace(rows[-1])["latest_reset_checkpoint"]
        self.assertEqual(
            BOX[0], again["before_state"]["active_tracks"][0]["current"]["bbox_xyxy"][0]
        )
        self.assertTrue(
            compare_utc_only_reset_checkpoint(again)["actual_reset_assignments_reproduced"]
        )

    def test_only_latest_utc_boundary_and_non_utc_reset_clears_checkpoint(self):
        people, _, checkpoint = self.capture()
        new = observation(3, elapsed=0.3)
        new = replace(new, frame=replace(new.frame, received_at=BASE + timedelta(seconds=6)))
        people.observe(new)
        self.assertEqual(
            3, people.review_trace(new)["latest_reset_checkpoint"]["boundary"]["sequence"]
        )
        gen = observation(4, generation=1, elapsed=0.4)
        people.observe(gen)
        self.assertIsNone(people.review_trace(gen)["latest_reset_checkpoint"])
        self.assertEqual(2, checkpoint["boundary"]["sequence"])

    def test_default_off_does_not_store_or_export_checkpoint(self):
        people = LivePeople(("a",))
        people.observe(observation(0, elapsed=0))
        new = observation(1, elapsed=0.1)
        new = replace(new, frame=replace(new.frame, received_at=BASE + timedelta(seconds=3)))
        people.observe(new)
        self.assertIsNone(people.cameras["a"]["latest_reset_checkpoint"])
        self.assertNotIn("latest_reset_checkpoint", people.review_trace(new))

    def test_capture_reuses_normal_digests_without_extra_pixel_hashing(self):
        original = LiveFrame.sha256.fget
        counts = []
        for enabled in (False, True):
            calls = []
            people = LivePeople(("a",), association_diagnostics=enabled)
            first = observation(0, elapsed=0)
            second = observation(1, elapsed=0.1)
            second = replace(
                second, frame=replace(second.frame, received_at=BASE + timedelta(seconds=3))
            )

            def digest(frame, calls=calls):
                calls.append(frame.sequence)
                return original(frame)

            with patch.object(LiveFrame, "sha256", property(digest)):
                people.observe(first)
                people.observe(second)
            counts.append(calls)
        self.assertEqual([[0, 1], [0, 1]], counts)

    def test_full_configured_capacity_is_bounded_without_silent_track_truncation(self):
        boxes = tuple((i / 128, 0.1, (i + 0.8) / 128, 0.8) for i in range(128))
        _, _, checkpoint = self.capture(boxes=boxes, previous_boxes=boxes)
        self.assertEqual(128, len(checkpoint["before_state"]["active_tracks"]))
        self.assertEqual(1, len(checkpoint["before_state"]["processed_frames"]))
        self.assertLess(len(json.dumps(checkpoint, allow_nan=False)), 160 * 1024)
        report = compare_utc_only_reset_checkpoint(checkpoint)
        self.assertTrue(report["actual_reset_audit_reproduced"])
        self.assertEqual(128, len(report["candidate_assignment_ids"]))

    def test_incomplete_wrong_source_cause_input_or_assignment_rejected(self):
        _, _, checkpoint = self.capture()
        for mutate in (
            lambda c: c.pop("before_state"),
            lambda c: c.update(version=True),
            lambda c: c.update(full_active_track_state=False),
            lambda c: c.update(role_evidence_included=True),
            lambda c: c.update(ordinal=True),
            lambda c: c.update(ordinal=10),
            lambda c: c.update(previous_raw_sha256="bad"),
            lambda c: c.update(camera_id="other"),
            lambda c: c.update(previous_detection_view="rotation_180"),
            lambda c: c.update(extra_role="baby"),
            lambda c: c["boundary"].update(reasons=["utc_gap", "monotonic_gap"]),
            lambda c: c["boundary"].update(generation=1),
            lambda c: c["boundary"].update(sequence=1),
            lambda c: c["boundary"].update(elapsed_receipt_gap_seconds=0.3),
            lambda c: c["boundary"].update(received_monotonic_seconds=float("nan")),
            lambda c: c["boundary"].update(track_gap_seconds=True),
            lambda c: c["boundary"].update(received_at="2026-10-12T00:00:03"),
            lambda c: c["before_state"]["active_tracks"].clear(),
            lambda c: c["before_state"]["processed_frames"][0].update(fingerprint="0" * 64),
            lambda c: c["input_frame"].update(frame_id="f_other_2_0_2"),
            lambda c: c["input_frame"]["detections"][0].update(bbox_xyxy=list(OTHER)),
            lambda c: c["original_assignment_ids"].clear(),
            lambda c: c["original_association_audit"].update(tracker_revision="old"),
        ):
            broken = copy.deepcopy(checkpoint)
            mutate(broken)
            with self.subTest(mutation=mutate), self.assertRaises(TrackerEvaluationError):
                compare_utc_only_reset_checkpoint(broken)


if __name__ == "__main__":
    unittest.main()
