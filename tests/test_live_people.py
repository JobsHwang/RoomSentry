from __future__ import annotations

import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from baby_monitor.domain import Detection
from baby_monitor.live_detection import LiveFrame, LiveObservation
from baby_monitor.live_people import LivePeople, PeopleSettings

BASE = datetime(2026, 10, 12, tzinfo=UTC)
BOX = (0.1, 0.1, 0.4, 0.8)
OTHER = (0.6, 0.1, 0.9, 0.8)


def observation(sequence=0, *, camera="a", boxes=(BOX,), generation=0, elapsed=None):
    elapsed = sequence if elapsed is None else elapsed
    frame = LiveFrame(
        camera,
        sequence,
        4,
        4,
        BASE + timedelta(seconds=elapsed),
        float(elapsed),
        bytes([sequence % 256]) * 48,
        generation,
    )
    return LiveObservation(
        frame, tuple(Detection(0.0, "person", 0.8, b) for b in boxes), float(elapsed)
    )


def claims(role="baby", *, head=(250, 300), posture="lying"):
    return {
        "people": [
            {
                "head_center": None if head is None else list(head),
                "age_group": role,
                "visible_posture": posture,
                "visibility": "clear",
                "visual_evidence": "synthetic",
            }
        ],
        "uncertainties": [],
    }


class LivePeopleTests(unittest.TestCase):
    def setUp(self):
        registry = SimpleNamespace(
            room_for_stream=lambda c: SimpleNamespace(
                to_dict=lambda: {"room_id": "room_" + c, "display_name": "Room " + c}
            )
        )
        self.people = LivePeople(("a", "b"), room_registry=registry)

    def review(self, obs, value=None, *, now=None):
        now = obs.completed_monotonic if now is None else now
        return self.people.apply_review(
            obs,
            claims() if value is None else value,
            evidence_ref="synthetic.png",
            now=now,
            now_utc=BASE + timedelta(seconds=now),
        )

    def project(self, now, **kwargs):
        return self.people.project(
            now=now,
            now_utc=BASE + timedelta(seconds=now),
            sources={
                "a": {"phase": "receiving", "generation": 0},
                "b": {"phase": "receiving", "generation": 0},
            },
            monitor_phase="detecting",
            **kwargs,
        )

    def establish(self, camera="a"):
        for i in (0, 1):
            obs = observation(i, camera=camera)
            self.people.observe(obs)
            self.review(obs)
        return self.project(1)

    def test_two_reviews_bind_one_track_and_room_visible_posture(self):
        first = observation()
        self.people.observe(first)
        self.review(first)
        self.assertEqual("candidate", self.project(0)["status"])
        second = observation(1)
        self.people.observe(second)
        self.review(second)
        result = self.project(1)
        self.assertEqual("current_model_estimate", result["status"])
        self.assertEqual("room_a", result["current"]["room"]["room_id"])
        self.assertEqual("lying", result["current"]["visible_posture"])
        self.assertFalse(result["accuracy_verified"])
        self.assertFalse(result["sleep_established"])

    def test_same_frame_cannot_supply_second_support(self):
        obs = observation()
        self.people.observe(obs)
        self.review(obs)
        self.assertEqual("stale_or_unbound_review", self.review(obs)["status"])
        self.assertEqual("candidate", self.project(0)["status"])

    def test_too_close_reviews_need_one_second_spacing(self):
        for i, t in enumerate((0, 0.2, 0.4)):
            obs = observation(i, elapsed=t)
            self.people.observe(obs)
            self.review(obs)
        self.assertEqual("candidate", self.project(0.4)["status"])

    def test_zero_detections_retains_untracked_baby_candidate(self):
        obs = observation(boxes=())
        self.people.observe(obs)
        result = self.review(obs)
        self.assertEqual(0, result["bound_people"])
        projection = self.project(0)
        self.assertEqual("candidate", projection["status"])
        self.assertEqual("untracked_candidate", projection["candidates"][0]["role_status"])
        self.assertEqual([], projection["cameras"]["a"]["people"])

    def test_old_role_becomes_last_seen_when_detection_disappears(self):
        before = self.establish()
        self.people.observe(observation(2, boxes=()))
        after = self.project(2)
        self.assertEqual("last_seen", after["status"])
        self.assertIsNone(after["current"])
        self.assertEqual(before["current"]["track_id"], after["last_seen"]["track_id"])

    def test_role_expires_even_if_yolo_continues_detecting(self):
        self.establish()
        for i in range(2, 33):
            self.people.observe(observation(i))
        result = self.project(32)
        self.assertIsNone(result["current"])
        self.assertEqual("unknown", result["cameras"]["a"]["people"][0]["age_claim"])

    def test_role_conflict_immediately_removes_baby_support(self):
        self.establish()
        obs = observation(2)
        self.people.observe(obs)
        self.review(obs, claims("adult", posture="standing"))
        result = self.project(2)
        self.assertIsNone(result["current"])
        self.assertEqual("candidate", result["cameras"]["a"]["people"][0]["role_status"])

    def test_posture_change_waits_for_new_support_without_changing_location(self):
        self.establish()
        for sequence in (2, 3):
            obs = observation(sequence)
            self.people.observe(obs)
            self.review(obs, claims(posture="held"))
            result = self.project(sequence)
            self.assertEqual("current_model_estimate", result["status"])
            self.assertEqual(
                "unknown" if sequence == 2 else "held", result["current"]["visible_posture"]
            )

    def test_partial_visibility_does_not_confirm_posture(self):
        for sequence in (0, 1):
            obs = observation(sequence)
            self.people.observe(obs)
            value = claims()
            value["people"][0]["visibility"] = "partial"
            self.review(obs, value)
        self.assertEqual("unknown", self.project(1)["current"]["visible_posture"])

    def test_two_heads_inside_one_body_box_are_not_assigned(self):
        obs = observation()
        self.people.observe(obs)
        value = claims()
        value["people"] += claims("adult", head=(300, 400))["people"]
        self.assertEqual(0, self.review(obs, value)["bound_people"])

    def test_identical_or_nested_boxes_do_not_duplicate_role(self):
        for boxes in ((BOX, BOX), (BOX, (0.05, 0.05, 0.5, 0.9))):
            with self.subTest(boxes=boxes):
                obs = observation(boxes=boxes)
                self.people = LivePeople(("a",))
                self.people.observe(obs)
                self.assertEqual(0, self.review(obs)["bound_people"])

    def test_null_or_outside_head_cannot_assign_body_role(self):
        for head in (None, (950, 950)):
            self.people = LivePeople(("a",))
            obs = observation()
            self.people.observe(obs)
            self.assertEqual(0, self.review(obs, claims(head=head))["bound_people"])

    def test_crossing_rejects_late_pre_crossing_review(self):
        first = observation(boxes=(BOX, OTHER))
        self.people.observe(first)
        crossed = observation(1, boxes=((0.2, 0.1, 0.65, 0.8), (0.35, 0.1, 0.8, 0.8)))
        self.people.observe(crossed)
        self.assertEqual(0, self.review(first, now=1)["bound_people"])
        self.assertIsNone(self.project(1)["current"])

    def test_reconnect_or_input_gap_rejects_previous_frame(self):
        for new in (observation(2, generation=1), observation(2, elapsed=5)):
            self.people = LivePeople(("a",))
            first = observation()
            self.people.observe(first)
            self.people.observe(new)
            self.assertEqual("stale_or_unbound_review", self.review(first, now=5)["status"])

    def test_either_clock_expires_evidence_and_future_clock_is_unavailable(self):
        self.establish()
        for mono, wall in ((1, 33), (33, 1), (0, 1), (1, 0)):
            result = self.people.project(
                now=mono,
                now_utc=BASE + timedelta(seconds=wall),
                sources={"a": {"phase": "receiving", "generation": 0}},
                monitor_phase="detecting",
            )
            self.assertIsNone(result["current"])

    def test_stopped_disconnected_or_suspended_never_current(self):
        self.establish()
        for phase in ("stopped", "detector_failed"):
            result = self.people.project(
                now=1,
                now_utc=BASE + timedelta(seconds=1),
                sources={"a": {"phase": "receiving", "generation": 0}},
                monitor_phase=phase,
            )
            self.assertIsNone(result["current"])
        self.assertIsNone(self.project(1, review_available=False)["current"])

    def test_two_camera_claims_are_ambiguous_without_cross_camera_identity(self):
        self.establish("a")
        result = self.establish("b")
        self.assertEqual("ambiguous", result["status"])
        self.assertIsNone(result["current"])
        self.assertNotEqual(
            result["cameras"]["a"]["people"][0]["track_id"],
            result["cameras"]["b"]["people"][0]["track_id"],
        )

    def test_history_and_tracker_outbox_are_bounded(self):
        self.people = LivePeople(("a",), settings=PeopleSettings(frame_history_per_camera=16))
        for i in range(100):
            self.people.observe(observation(i))
        state = self.people.cameras["a"]
        self.assertEqual(16, len(state["history"]))
        self.assertEqual(64, len(state["tracker"].state.processed_frames))
        self.assertEqual((), state["tracker"].state.pending_observations)

    def test_review_trace_preserves_every_actual_input_including_empty(self):
        first = observation(0)
        for row in (first, observation(4, boxes=(), elapsed=0.2), observation(8, elapsed=0.4)):
            self.people.observe(row)
        trace = self.people.review_trace(first)
        self.assertEqual("bound_interval", trace["status"])
        self.assertEqual(3, trace["interval_frames"])
        self.assertEqual(0, trace["omitted_middle_frames"])
        self.assertEqual([0, 4, 8], [r["sequence"] for r in trace["frames"]])
        self.assertEqual([1, 2, 3], [r["ordinal"] for r in trace["frames"]])
        self.assertEqual([1, 0, 1], [len(r["detections"]) for r in trace["frames"]])
        self.assertEqual(first.frame.sha256, trace["frames"][0]["raw_sha256"])
        self.assertEqual([list(BOX)], [r["bbox_xyxy"] for r in trace["frames"][0]["detections"]])
        self.assertEqual(0.8, trace["frames"][0]["detections"][0]["confidence"])
        self.assertEqual({"detection_gap": 1}, trace["frames"][2]["detections"][0]["epoch_causes"])
        self.assertFalse(trace["accuracy_measured"])
        serialized = json.dumps(trace)
        for token in ("pixels", "visual_evidence", "evidence_ref", "Room ", "age_claim"):
            self.assertNotIn(token, serialized)

    def test_trace_truncation_retains_bound_input_and_explicit_middle_gap(self):
        first = observation()
        for i in range(10):
            self.people.observe(observation(i, elapsed=i / 10))
        trace = self.people.review_trace(first, maximum_frames=4)
        self.assertEqual(10, trace["interval_frames"])
        self.assertEqual(6, trace["omitted_middle_frames"])
        self.assertEqual([0, 7, 8, 9], [r["sequence"] for r in trace["frames"]])
        self.assertEqual(4, trace["exported_frames"])

    def test_evicted_or_reset_trace_is_unavailable_not_empty_interval(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                self.people = LivePeople(
                    ("a",), settings=PeopleSettings(frame_history_per_camera=16)
                )
                first = observation()
                for i in range(20):
                    self.people.observe(observation(i, elapsed=i / 10))
                if reset:
                    self.people.observe(observation(20, generation=1, elapsed=2))
                trace = self.people.review_trace(first)
                self.assertEqual("review_frame_unavailable", trace["status"])
                self.assertIsNone(trace["interval_frames"])
                self.assertEqual([], trace["frames"])
                self.assertEqual(4, trace["history_evicted_frames_total"])
                self.assertEqual(16 if reset else 0, trace["history_reset_discarded_frames_total"])

    def test_trace_rejects_changed_bytes_or_detector_input(self):
        first = observation()
        self.people.observe(first)
        changed = replace(first, frame=replace(first.frame, pixels=b"x" * 48))
        self.assertEqual("frame_bytes_mismatch", self.people.review_trace(changed)["status"])
        changed = replace(first, detections=())
        self.assertEqual("detector_input_mismatch", self.people.review_trace(changed)["status"])

    def test_trace_is_detached_and_does_not_change_cadence_roles_or_binding(self):
        first = observation()
        self.people.observe(first)
        trace = self.people.review_trace(first)
        trace["frames"][0]["detections"][0]["epoch_causes"]["fake"] = 9
        trace["frames"][0]["detections"][0]["bbox_xyxy"][0] = 0
        again = self.people.review_trace(first)
        self.assertEqual({}, again["frames"][0]["detections"][0]["epoch_causes"])
        self.assertEqual(list(BOX), again["frames"][0]["detections"][0]["bbox_xyxy"])
        self.assertTrue(self.people.review_due("a", now=0, now_utc=BASE))
        self.assertEqual(1, self.review(first)["bound_people"])
        self.assertEqual(
            1,
            self.people.cameras["a"]["tracks"][again["frames"][0]["detections"][0]["track_id"]][
                "role"
            ]["support_count"],
        )

    def test_trace_auxiliary_keeps_independent_lane_and_actual_per_frame_view(self):
        self.people = LivePeople(("a",), auxiliary=True)
        native = observation(boxes=())
        first = observation()
        self.people.observe(native)
        self.people.observe_auxiliary(first, view="rotation_180")
        self.people.observe_auxiliary(observation(1), view="rotation_180_scale_fallback")
        trace = self.people.review_trace(first, view="rotation_180")
        self.assertEqual(
            ["rotation_180", "rotation_180_scale_fallback"], [r["view"] for r in trace["frames"]]
        )
        self.assertEqual(2, trace["interval_frames"])
        self.assertEqual(1, self.people.review_trace(native)["interval_frames"])
        self.assertEqual(
            "input_view_mismatch",
            self.people.review_trace(first, view="rotation_180_scale_fallback")["status"],
        )

    def test_trace_limit_and_view_are_strict(self):
        first = observation()
        self.people.observe(first)
        for limit in (True, 1, 257, 3.0):
            with self.assertRaises(ValueError):
                self.people.review_trace(first, maximum_frames=limit)
        for view in ("unknown", "rotation_180"):
            with self.assertRaises(ValueError):
                self.people.review_trace(first, view=view)
        with self.assertRaises(ValueError):
            self.people.observe(observation(1), view="unknown")

    def test_conflicting_frame_bytes_and_old_sequence_cannot_bind(self):
        obs = observation()
        self.people.observe(obs)
        changed = replace(obs, frame=replace(obs.frame, pixels=b"x" * 48))
        self.assertEqual("stale_or_unbound_review", self.review(changed)["status"])
        self.people.observe(observation(1))
        self.people.observe(obs)
        self.assertEqual(1, self.people.cameras["a"]["previous"].sequence)

    def test_clear_roles_invalidates_inflight_review(self):
        self.establish()
        self.people.clear_roles()
        self.assertIsNone(self.project(1)["current"])

    def test_reused_motion_id_after_empty_frame_rejects_pre_gap_role(self):
        first = observation()
        self.people.observe(first)
        track_id = next(iter(self.people.cameras["a"]["tracks"]))
        self.people.observe(observation(1, boxes=()))
        self.people.observe(observation(2))
        self.assertIn(track_id, self.people.cameras["a"]["tracks"])
        result = self.review(first, now=2)
        self.assertEqual(0, result["bound_people"])
        self.assertEqual({"track_continuity_lost": 1}, result["unbound_reasons"])

        details = result["unbound_diagnostics"][0]
        self.assertTrue(details["current_track_present"])
        self.assertEqual({"detection_gap": 1}, details["epoch_changes_since_input"])

    def test_head_or_other_label_cannot_become_body_person(self):
        first = observation()
        changed = replace(first, detections=(Detection(0, "head", 0.8, BOX),))
        with self.assertRaises(ValueError):
            self.people.observe(changed)

    def test_multiple_role_clears_and_gap_never_reuse_role_epoch(self):
        self.establish()
        self.people.clear_roles()
        submitted = observation(2)
        self.people.observe(submitted)
        self.people.clear_roles()
        self.people.clear_roles()
        self.people.observe(observation(3, boxes=()))
        self.people.observe(observation(4))
        result = self.review(submitted, now=4)
        self.assertEqual(0, result["bound_people"])
        self.assertEqual({"review_scope_invalidated": 1}, result["unbound_reasons"])

    def test_role_clear_while_track_invisible_invalidates_old_frame(self):
        submitted = observation()
        self.people.observe(submitted)
        self.people.observe(observation(1, boxes=()))
        self.people.clear_roles()
        self.people.observe(observation(2))
        result = self.review(submitted, now=2)
        self.assertEqual(0, result["bound_people"])
        self.assertEqual({"review_scope_invalidated": 1}, result["unbound_reasons"])

    def test_stable_roles_are_refreshed_less_often_and_empty_scene_patrol_remains(self):
        self.establish()
        self.assertFalse(self.people.review_due("a", now=2, now_utc=BASE + timedelta(seconds=2)))
        self.assertTrue(self.people.review_due("a", now=11, now_utc=BASE + timedelta(seconds=11)))
        empty = observation(2, boxes=())
        self.people.observe(empty)
        self.review(empty, {"people": [], "uncertainties": []})
        self.assertFalse(self.people.review_due("a", now=3, now_utc=BASE + timedelta(seconds=3)))
        self.assertTrue(self.people.review_due("a", now=17, now_utc=BASE + timedelta(seconds=17)))

    def test_auxiliary_lane_can_bind_exact_rotated_frame_without_changing_native(self):
        self.people = LivePeople(("a",), auxiliary=True)
        for i in (0, 1):
            native = observation(i, boxes=())
            self.people.observe(native)
            auxiliary = replace(native, detections=observation(i).detections)
            self.people.observe_auxiliary(auxiliary, view="rotation_180")
            selected, view = self.people.review_observation(
                native, now=i, now_utc=BASE + timedelta(seconds=i)
            )
            self.assertEqual(auxiliary, selected)
            self.assertEqual("rotation_180", view)
            self.people.apply_review(
                selected,
                claims(),
                evidence_ref="synthetic.png",
                now=i,
                now_utc=BASE + timedelta(seconds=i),
                view=view,
            )
        result = self.people.snapshot(
            now=1,
            now_utc=BASE + timedelta(seconds=1),
            sources={"a": {"phase": "receiving", "generation": 0}},
            monitor_phase="detecting",
        )
        self.assertEqual("current_model_estimate", result["status"])
        self.assertEqual("rotation_180", result["current"]["detection_view"])
        self.assertEqual({}, self.people.cameras["a"]["tracks"])
        self.assertEqual((), native.detections)

    def test_native_return_never_inherits_auxiliary_role(self):
        self.test_auxiliary_lane_can_bind_exact_rotated_frame_without_changing_native()
        self.people.observe(observation(2))
        result = self.people.snapshot(
            now=2,
            now_utc=BASE + timedelta(seconds=2),
            sources={"a": {"phase": "receiving", "generation": 0}},
            monitor_phase="detecting",
        )
        self.assertIsNone(result["current"])
        self.assertEqual("unknown", result["cameras"]["a"]["people"][0]["age_claim"])

    def test_native_lane_takeover_invalidates_auxiliary_once_not_every_frame(self):
        self.test_auxiliary_lane_can_bind_exact_rotated_frame_without_changing_native()
        child = self.people.auxiliary.cameras["a"]
        track_id = next(iter(child["tracks"]))
        self.people.observe(observation(2))
        first_epoch = child["tracks"][track_id]["epoch"]
        self.people.observe(observation(3))
        self.assertEqual(first_epoch, child["tracks"][track_id]["epoch"])
        result = self.people.apply_review(
            observation(1),
            claims(),
            evidence_ref="synthetic.png",
            now=3,
            now_utc=BASE + timedelta(seconds=3),
            view="rotation_180",
        )
        # The old reviewed frame was already consumed; the scope remains closed.
        self.assertEqual("stale_or_unbound_review", result["status"])
        self.assertEqual({"native_lane_took_over": 1}, child["epoch_causes"][track_id])

    def test_late_auxiliary_review_is_rejected_after_native_takeover(self):
        self.people = LivePeople(("a",), auxiliary=True)
        native = observation(0, boxes=())
        auxiliary = replace(native, detections=observation().detections)
        self.people.observe(native)
        self.people.observe_auxiliary(auxiliary, view="rotation_180")
        self.people.observe(observation(1))
        self.people.observe(observation(2))
        result = self.people.apply_review(
            auxiliary,
            claims(),
            evidence_ref="synthetic.png",
            now=2,
            now_utc=BASE + timedelta(seconds=2),
            view="rotation_180",
        )
        self.assertEqual(0, result["bound_people"])
        self.assertEqual({"review_scope_invalidated": 1}, result["unbound_reasons"])
        self.assertEqual(
            {"native_lane_took_over": 1},
            result["unbound_diagnostics"][0]["epoch_changes_since_input"],
        )

    def _establish_generation_one_auxiliary(self):
        self.people = LivePeople(("a",), auxiliary=True)
        for sequence, elapsed in ((5, 0), (6, 1)):
            native = observation(sequence, boxes=(), generation=1, elapsed=elapsed)
            auxiliary = replace(native, detections=observation().detections)
            self.people.observe(native)
            self.people.observe_auxiliary(auxiliary, view="rotation_180")
            self.people.apply_review(
                auxiliary,
                claims(),
                evidence_ref="synthetic.png",
                now=elapsed,
                now_utc=BASE + timedelta(seconds=elapsed),
                view="rotation_180",
            )
        return native, auxiliary

    def test_rejected_native_inputs_cannot_revoke_newer_auxiliary_roles(self):
        for sequence, generation in ((6, 1), (5, 1), (100, 0)):
            with self.subTest(sequence=sequence, generation=generation):
                self._establish_generation_one_auxiliary()
                child = self.people.auxiliary.cameras["a"]
                before = json.dumps(child["epoch_causes"], sort_keys=True)
                track_id = next(iter(child["tracks"]))
                epoch = child["tracks"][track_id]["epoch"]
                self.people.observe(observation(sequence, generation=generation, elapsed=1))
                self.assertEqual(epoch, child["tracks"][track_id]["epoch"])
                self.assertEqual(before, json.dumps(child["epoch_causes"], sort_keys=True))
                self.assertEqual(2, child["tracks"][track_id]["role"]["support_count"])
                self.assertEqual({}, self.people.cameras["a"]["tracks"])

    def test_rejected_auxiliary_inputs_cannot_replace_selected_input_or_view(self):
        for sequence, generation in ((6, 1), (5, 1), (100, 0)):
            with self.subTest(sequence=sequence, generation=generation):
                native, accepted = self._establish_generation_one_auxiliary()
                child = self.people.auxiliary.cameras["a"]
                before = self.people.review_trace(accepted, view="rotation_180")
                rejected = observation(sequence, generation=generation, boxes=(OTHER,), elapsed=1)
                self.people.observe_auxiliary(rejected, view="rotation_180_scale_fallback")
                selected, view = self.people.review_observation(
                    native, now=1, now_utc=BASE + timedelta(seconds=1)
                )
                self.assertIs(accepted, selected)
                self.assertEqual("rotation_180", view)
                self.assertEqual(before, self.people.review_trace(accepted, view="rotation_180"))
                self.assertIs(accepted, child["observation"])

    def test_auxiliary_expiration_preserves_last_seen_without_current_claim(self):
        self.test_auxiliary_lane_can_bind_exact_rotated_frame_without_changing_native()
        self.people.observe(observation(4, boxes=()))
        result = self.people.snapshot(
            now=4,
            now_utc=BASE + timedelta(seconds=4),
            sources={"a": {"phase": "receiving", "generation": 0}},
            monitor_phase="detecting",
        )
        self.assertEqual("last_seen", result["status"])
        self.assertIsNotNone(result["cameras"]["a"]["last_seen_baby"])

    def test_auxiliary_admission_expires_on_either_clock(self):
        self.people = LivePeople(("a",), auxiliary=True)
        native = observation(boxes=())
        self.people.observe(native)
        auxiliary = replace(native, detections=observation().detections)
        self.people.observe_auxiliary(auxiliary, view="rotation_180")
        for elapsed, wall in ((0, 2), (2, 0)):
            selected, view = self.people.review_observation(
                native, now=elapsed, now_utc=BASE + timedelta(seconds=wall)
            )
            self.assertEqual(native, selected)
            self.assertEqual("original", view)

    def test_empty_auxiliary_does_not_bypass_native_empty_scene_patrol(self):
        self.people = LivePeople(("a",), auxiliary=True)
        first = observation(boxes=())
        self.people.observe(first)
        self.people.observe_auxiliary(first, view="rotation_180")
        self.review(first, {"people": [], "uncertainties": []})
        for sequence in range(1, 31):
            elapsed = sequence / 2
            native = observation(sequence, boxes=(), elapsed=elapsed)
            self.people.observe(native)
            self.people.observe_auxiliary(native, view="rotation_180")
            now_utc = BASE + timedelta(seconds=elapsed)
            selected, view = self.people.review_observation(native, now=elapsed, now_utc=now_utc)
            self.assertEqual(native, selected)
            self.assertEqual("original", view)
            self.assertEqual(
                elapsed >= 15, self.people.review_due("a", now=elapsed, now_utc=now_utc)
            )

    def test_previous_generation_auxiliary_cannot_override_native_review_schedule(self):
        self.people = LivePeople(("a",), auxiliary=True)
        first = observation(boxes=())
        self.people.observe(first)
        self.people.observe_auxiliary(observation(), view="rotation_180")
        native = observation(1, boxes=(), generation=1, elapsed=0.5)
        self.people.observe(native)
        self.review(native, {"people": [], "uncertainties": []})
        selected, view = self.people.review_observation(
            native, now=0.6, now_utc=BASE + timedelta(seconds=0.6)
        )
        self.assertEqual(native, selected)
        self.assertEqual("original", view)
        self.assertFalse(
            self.people.review_due("a", now=0.6, now_utc=BASE + timedelta(seconds=0.6))
        )

    def test_fresh_positive_auxiliary_still_uses_its_own_role_schedule(self):
        self.test_auxiliary_lane_can_bind_exact_rotated_frame_without_changing_native()
        self.assertFalse(
            self.people.review_due("a", now=1.5, now_utc=BASE + timedelta(seconds=1.5))
        )
        native = observation(2, boxes=())
        self.people.observe(native)
        # The native lane has never been reviewed. Do not borrow the old
        # auxiliary schedule when its input is no longer eligible for selection.
        self.assertTrue(self.people.review_due("a", now=2.5, now_utc=BASE + timedelta(seconds=2.5)))


if __name__ == "__main__":
    unittest.main()
