from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from baby_monitor.live_people import LivePeople
from baby_monitor.live_people_association import describe_association
from baby_monitor.motion_tracker import MotionObjectTracker, MotionTrackerConfig
from baby_monitor.object_tracker import DetectionFrame, FrameDetection
from tests.test_live_people import BOX, OTHER, observation


class PeopleAssociationAuditTests(unittest.TestCase):
    base = datetime(2026, 10, 12, tzinfo=UTC)

    def frame(self, i, boxes=()):
        return DetectionFrame(
            f"f{i}",
            self.base + timedelta(seconds=i / 10),
            tuple(FrameDetection(f"d{j}", "person", 0.9, box) for j, box in enumerate(boxes)),
        )

    def tracker(self):
        return MotionObjectTracker(
            config=MotionTrackerConfig(), camera_id="synthetic", detector_adapter="synthetic"
        )

    def audit(self, tracker, frame):
        before = tracker.state
        rows = tracker.process(frame)
        before_copy = before.to_dict()
        result = describe_association(before, frame, rows, tracker.config)
        self.assertEqual(before_copy, before.to_dict())
        return result, before, rows

    def test_new_recent_dormant_and_expired_are_distinct(self):
        tracker = self.tracker()
        first, _, _ = self.audit(tracker, self.frame(0, (BOX,)))
        self.assertEqual("new_anonymous_id", first["detections"][0]["selected_source"])
        self.assertIsNone(first["detections"][0]["selected_match_score"])
        second, _, _ = self.audit(tracker, self.frame(1, (BOX,)))
        row = second["detections"][0]
        self.assertEqual("previous_input", row["selected_source"])
        self.assertAlmostEqual(1, row["selected_match_score"])
        self.assertEqual(1, row["eligible_previous_input_track_count"])
        empty, _, _ = self.audit(tracker, self.frame(2))
        self.assertEqual([], empty["detections"])
        returned, _, _ = self.audit(tracker, self.frame(3, (BOX,)))
        row = returned["detections"][0]
        self.assertEqual("dormant_retained_track", row["selected_source"])
        self.assertEqual(0, row["eligible_previous_input_track_count"])
        expired, _, _ = self.audit(tracker, self.frame(30, (BOX,)))
        self.assertEqual(1, expired["expired_before_count"])
        self.assertEqual("new_anonymous_id", expired["detections"][0]["selected_source"])
        self.assertFalse(expired["accuracy_measured"])

    def test_dormant_winner_with_eligible_previous_input_is_not_called_wrong(self):
        tracker = self.tracker()
        box_a, box_b = (0.1, 0.2, 0.3, 0.8), (0.12, 0.2, 0.32, 0.8)
        self.audit(tracker, self.frame(0, (box_a, box_b)))
        self.audit(tracker, self.frame(1, (box_a,)))
        result, _, _ = self.audit(tracker, self.frame(2, (box_b,)))
        row = result["detections"][0]
        self.assertEqual("dormant_retained_track", row["selected_source"])
        self.assertEqual(1, row["eligible_previous_input_track_count"])
        self.assertNotEqual(row["track_id"], row["eligible_previous_input_tracks"][0]["track_id"])
        self.assertTrue(result["eligible_edge_does_not_establish_correct_assignment"])

    def test_competitor_export_has_a_hard_bound_and_explicit_omission(self):
        tracker = self.tracker()
        self.audit(tracker, self.frame(0, (BOX,) * 12))
        result, _, _ = self.audit(tracker, self.frame(1, (BOX,)))
        row = result["detections"][0]
        self.assertEqual(12, row["eligible_previous_input_track_count"])
        self.assertEqual(8, len(row["eligible_previous_input_tracks"]))
        self.assertTrue(row["eligible_previous_input_tracks_truncated"])

    def test_bad_config_stale_time_or_changed_detection_fails_closed(self):
        tracker = self.tracker()
        frame = self.frame(0, (BOX,))
        _, before, rows = self.audit(tracker, frame)
        for state, changed_frame, changed_rows, config in (
            (before, frame, rows, MotionTrackerConfig(max_prediction_seconds=0)),
            (tracker.state, frame, rows, tracker.config),
            (
                before,
                replace(frame, detections=(FrameDetection("d0", "person", 0.8, BOX),)),
                rows,
                tracker.config,
            ),
            (before, frame, (), tracker.config),
        ):
            with self.subTest(), self.assertRaises(ValueError):
                describe_association(state, changed_frame, changed_rows, config)

    def test_trace_is_detached_opt_in_and_auxiliary_inherits_flag(self):
        people = LivePeople(("a",), auxiliary=True, association_diagnostics=True)
        first = observation()
        people.observe(first)
        people.observe_auxiliary(first, view="rotation_180")
        people.observe(observation(1))
        trace = people.review_trace(first)
        row = trace["frames"][1]["association_audit"]["detections"][0]
        row["selected_prior_bbox_xyxy"][0] = 0
        row["eligible_previous_input_tracks"][0]["match_score"] = -1
        again = people.review_trace(first)
        row = again["frames"][1]["association_audit"]["detections"][0]
        self.assertEqual(BOX[0], row["selected_prior_bbox_xyxy"][0])
        self.assertGreater(row["eligible_previous_input_tracks"][0]["match_score"], 0)
        self.assertIn(
            "association_audit", people.review_trace(first, view="rotation_180")["frames"][0]
        )
        default = LivePeople(("a",))
        default.observe(first)
        self.assertNotIn("association_audit", default.review_trace(first)["frames"][0])
        for value in (1, "yes", None):
            with self.assertRaisesRegex(ValueError, "diagnostics flag"):
                LivePeople(("a",), association_diagnostics=value)

    def test_opt_in_does_not_change_assignments_binding_roles_or_source_boxes(self):
        plain = LivePeople(("a",))
        audited = LivePeople(("a",), association_diagnostics=True)
        plain.session = audited.session = "synthetic"
        frames = [
            observation(i, boxes=boxes, elapsed=i / 10)
            for i, boxes in enumerate(((BOX,), (BOX, OTHER), (OTHER,), (), (BOX,), (BOX,)))
        ]
        from tests.test_live_people import claims

        for obs in frames:
            for people in (plain, audited):
                people.observe(obs)
            args = dict(
                evidence_ref="synthetic.png",
                now=obs.completed_monotonic,
                now_utc=obs.frame.received_at,
            )
            self.assertEqual(
                plain.apply_review(obs, claims(), **args),
                audited.apply_review(obs, claims(), **args),
            )
            self.assertEqual(
                plain.cameras["a"]["tracker"].state, audited.cameras["a"]["tracker"].state
            )
            self.assertEqual(plain.cameras["a"]["tracks"], audited.cameras["a"]["tracks"])
        left, right = plain.review_trace(frames[0]), copy.deepcopy(audited.review_trace(frames[0]))
        for row in right["frames"]:
            row.pop("association_audit")
        right.pop("association_checkpoint")
        self.assertEqual({"initial_input": 1}, right.pop("history_reset_reasons_total"))
        self.assertEqual(["initial_input"], right.pop("latest_history_reset")["reasons"])
        self.assertEqual(
            {
                "version": 1,
                "cause_counts_may_overlap": True,
                "latest_event_only": True,
                "complete_reset_event_history": False,
                "production_rules_changed": False,
            },
            right.pop("reset_diagnostics"),
        )
        self.assertIsNone(right.pop("latest_reset_checkpoint"))
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
