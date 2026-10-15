from __future__ import annotations

import copy
import unittest

from baby_monitor.live_people import LivePeople
from baby_monitor.tracker_evaluation._shared import TrackerEvaluationError
from baby_monitor.tracker_evaluation.association_checkpoint import (
    compare_prediction_fallback_checkpoint,
)
from tests.test_live_people import BOX, OTHER, observation


class AssociationCheckpointTests(unittest.TestCase):
    def trace(self, *, auxiliary=False):
        people = LivePeople(("a",), auxiliary=auxiliary, association_diagnostics=True)
        people.session = "synthetic"
        rows = [
            observation(i, boxes=boxes, elapsed=i / 10)
            for i, boxes in enumerate(((BOX, OTHER), (BOX,), (), (BOX, OTHER), (BOX,)))
        ]
        for obs in rows:
            if auxiliary:
                people.observe_auxiliary(obs, view="rotation_180")
            else:
                people.observe(obs)
        return (
            people,
            rows,
            people.review_trace(rows[1], view="rotation_180" if auxiliary else "original"),
        )

    def test_full_checkpoint_reproduces_dormant_and_recent_competition(self):
        _, _, trace = self.trace()
        checkpoint = trace["association_checkpoint"]
        self.assertTrue(checkpoint["full_active_track_state"])
        self.assertEqual(2, len(checkpoint["before_state"]["active_tracks"]))
        report = compare_prediction_fallback_checkpoint(trace)
        self.assertTrue(report["original_assignments_reproduced"])
        self.assertEqual(4, report["frame_count"])
        self.assertEqual(0, report["fallback_input_count"])
        self.assertEqual([], report["first_assignment_changes"])
        self.assertFalse(report["accuracy_measured"])
        self.assertFalse(report["role_inheritance"])

    def test_cold_start_checkpoint_and_auxiliary_lane_both_reproduce(self):
        for auxiliary in (False, True):
            people, rows, _ = self.trace(auxiliary=auxiliary)
            trace = people.review_trace(rows[0], view="rotation_180" if auxiliary else "original")
            self.assertEqual([], trace["association_checkpoint"]["before_state"]["active_tracks"])
            self.assertTrue(
                compare_prediction_fallback_checkpoint(trace)["original_assignments_reproduced"]
            )

    def test_immutable_checkpoint_export_detaches_from_live_state(self):
        people, rows, trace = self.trace()
        trace["association_checkpoint"]["before_state"]["active_tracks"][0]["current"]["bbox_xyxy"][
            0
        ] = 0
        again = people.review_trace(rows[1])
        self.assertNotEqual(
            0,
            again["association_checkpoint"]["before_state"]["active_tracks"][0]["current"][
                "bbox_xyxy"
            ][0],
        )
        self.assertTrue(
            compare_prediction_fallback_checkpoint(again)["original_assignments_reproduced"]
        )

    def test_partial_or_unbound_interval_cannot_be_cold_replayed_as_actual(self):
        _, _, trace = self.trace()
        for mutate in (
            lambda t: t.pop("association_checkpoint"),
            lambda t: t.update(omitted_middle_frames=1),
            lambda t: t.update(status="review_frame_unavailable"),
            lambda t: t["frames"][1].update(ordinal=999),
            lambda t: t["association_checkpoint"].update(full_active_track_state=False),
        ):
            bad = copy.deepcopy(trace)
            mutate(bad)
            with self.subTest(), self.assertRaises(TrackerEvaluationError):
                compare_prediction_fallback_checkpoint(bad)

    def test_changed_checkpoint_assignment_config_or_original_frame_fails(self):
        _, _, trace = self.trace()
        for mutate in (
            lambda t: t["association_checkpoint"].update(input_frame_id="different"),
            lambda t: t["association_checkpoint"]["before_state"].update(camera_id="other"),
            lambda t: t["association_checkpoint"]["tracker_config"].update(
                max_prediction_seconds=0
            ),
            lambda t: t["frames"][0].update(raw_sha256="0" * 64),
            lambda t: t["frames"][0]["detections"][0].update(track_id="wrong"),
            lambda t: t["frames"][0]["detections"][0].update(bbox_xyxy=[0.7, 0.1, 0.9, 0.8]),
        ):
            bad = copy.deepcopy(trace)
            mutate(bad)
            with self.subTest(), self.assertRaises(TrackerEvaluationError):
                compare_prediction_fallback_checkpoint(bad)

    def test_compact_checkpoint_reports_dedup_omission_and_no_delivery_outbox(self):
        people = LivePeople(("a",), association_diagnostics=True)
        for i in range(70):
            people.observe(observation(i, elapsed=i / 10))
        trace = people.review_trace(observation(69, elapsed=6.9))
        checkpoint = trace["association_checkpoint"]
        self.assertEqual(63, checkpoint["omitted_processed_dedup_frames"])
        self.assertEqual(1, len(checkpoint["before_state"]["processed_frames"]))
        self.assertEqual([], checkpoint["before_state"]["pending_observations"])
        self.assertTrue(
            compare_prediction_fallback_checkpoint(trace)["original_assignments_reproduced"]
        )

    def test_bool_versions_malformed_clocks_and_mixed_views_fail_closed(self):
        _, _, trace = self.trace()
        for mutate in (
            lambda t: t.update(version=True),
            lambda t: t.update(omitted_middle_frames=False),
            lambda t: t["association_checkpoint"].update(version=True),
            lambda t: t["frames"][1].update(view="rotation_180"),
            lambda t: t["frames"][1].update(received_monotonic_seconds=True),
            lambda t: t["frames"][1].update(completed_monotonic_seconds=float("nan")),
            lambda t: t["frames"][1].update(raw_sha256="not_sha"),
            lambda t: t["frames"][1].update(generation=False),
        ):
            bad = copy.deepcopy(trace)
            mutate(bad)
            with self.subTest(), self.assertRaises(TrackerEvaluationError):
                compare_prediction_fallback_checkpoint(bad)

    def test_auxiliary_scale_change_preserves_actual_same_lane_replay(self):
        people = LivePeople(("a",), auxiliary=True, association_diagnostics=True)
        rows = [observation(i, elapsed=i / 10) for i in range(4)]
        for i, row in enumerate(rows):
            people.observe_auxiliary(
                row, view="rotation_180" if i < 2 else "rotation_180_scale_fallback"
            )
        trace = people.review_trace(rows[1], view="rotation_180")
        self.assertEqual(
            ["rotation_180", "rotation_180_scale_fallback", "rotation_180_scale_fallback"],
            [r["view"] for r in trace["frames"]],
        )
        self.assertTrue(
            compare_prediction_fallback_checkpoint(trace)["original_assignments_reproduced"]
        )


if __name__ == "__main__":
    unittest.main()
