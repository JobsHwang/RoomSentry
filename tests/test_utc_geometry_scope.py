"""Anonymous continuity must not preserve a role or an old review scope."""

from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.live_monitor import run_live_monitor
from baby_monitor.live_people import LivePeople
from baby_monitor.motion_tracker import MotionObjectTracker, MotionTrackerError
from baby_monitor.tracker_evaluation.association_checkpoint import _inputs
from tests.test_live_people import BASE, BOX, OTHER, claims, observation


def frame(seq, mono, wall, *, boxes=(BOX,), generation=0):
    obs = observation(seq, elapsed=mono, boxes=boxes, generation=generation)
    return replace(obs, frame=replace(obs.frame, received_at=BASE + timedelta(seconds=wall)))


class UtcGeometryScopeTests(unittest.TestCase):
    def people(self, **kwargs):
        return LivePeople(
            ("a", "b"), association_diagnostics=True, utc_geometry_continuity=True, **kwargs
        )

    def review(self, people, obs, *, view="original"):
        return people.apply_review(
            obs,
            claims(),
            evidence_ref="synthetic.png",
            now=obs.frame.received_monotonic,
            now_utc=obs.frame.received_at,
            view=view,
        )

    def test_only_utc_anomaly_preserves_geometry_and_clears_all_role_support(self):
        for wall in (4, 1, -1):
            with self.subTest(wall=wall):
                people = self.people()
                for seq in (0, 1):
                    old = frame(seq, seq, seq)
                    people.observe(old)
                    self.review(people, old)
                camera = people.cameras["a"]
                prior = copy.deepcopy(camera["tracks"])
                new = frame(2, 1.2, wall)
                people.observe(new)
                self.assertEqual(list(prior), list(camera["tracks"]))
                for key, track in camera["tracks"].items():
                    self.assertIsNone(track["role"])
                    self.assertGreater(track["epoch"], prior[key]["epoch"])
                    self.assertEqual(1, track["hits"])
                self.assertEqual([], camera["untracked"])
                self.assertIsNone(camera["review_at"])
                self.assertEqual([people._key(new.frame)], list(camera["history"]))
                self.assertEqual("stale_or_unbound_review", self.review(people, old)["status"])
                self.assertIsNone(camera["latest_reset_checkpoint"])
                trace = people.review_trace(new)
                self.assertEqual(
                    "retained", trace["geometry_continuity"]["latest_boundary"]["action"]
                )
                self.assertEqual(1, trace["geometry_continuity"]["utc_geometry_retained_total"])
                self.assertEqual(1, trace["geometry_continuity"]["geometry_resets_total"])
                self.assertEqual(2, trace["history_resets_total"])

    def test_new_scope_needs_two_new_spaced_reviews_not_one_carried_vote(self):
        people = self.people()
        for seq in (0, 1):
            obs = frame(seq, seq, seq)
            people.observe(obs)
            self.review(people, obs)
        new = frame(2, 1.2, 4)
        people.observe(new)
        self.review(people, new)
        track = next(iter(people.cameras["a"]["tracks"].values()))
        self.assertEqual(1, track["role"]["support_count"])
        self.assertEqual("unknown", track["role"]["visible_posture"])
        newer = frame(3, 2.3, 5.1)
        people.observe(newer)
        self.review(people, newer)
        track = next(iter(people.cameras["a"]["tracks"].values()))
        self.assertEqual(2, track["role"]["support_count"])
        self.assertEqual("lying", track["role"]["visible_posture"])

    def test_geometry_does_not_relax_dual_clock_current_or_role_ttl(self):
        people = self.people()
        obs = frame(0, 0, 0)
        people.observe(obs)
        self.review(people, obs)
        projection = people.project(
            now=0.1,
            now_utc=BASE + timedelta(seconds=3),
            sources={"a": {"phase": "receiving", "generation": 0}},
            monitor_phase="detecting",
        )
        self.assertEqual("unavailable", projection["cameras"]["a"]["status"])
        self.assertIsNone(projection["current"])
        result = people.apply_review(
            obs,
            claims(),
            evidence_ref="synthetic.png",
            now=0.2,
            now_utc=BASE + timedelta(seconds=31),
        )
        self.assertEqual("stale_or_unbound_review", result["status"])

    def test_generation_monotonic_gap_and_regression_still_discard_geometry(self):
        for mono, wall, generation in (
            (3, 3, 0),
            (3, 0.1, 0),
            (0, 3, 0),
            (-0.1, 3, 0),
            (0.1, 3, 1),
        ):
            with self.subTest(mono=mono, wall=wall, generation=generation):
                people = self.people()
                people.observe(frame(0, 0, 0))
                old_id = list(people.cameras["a"]["tracks"])
                obs = frame(1, mono, wall, generation=generation)
                people.observe(obs)
                self.assertNotEqual(old_id, list(people.cameras["a"]["tracks"]))
                self.assertEqual(
                    0,
                    people.review_trace(obs)["geometry_continuity"]["utc_geometry_retained_total"],
                )

    def test_zero_before_or_after_boundary_never_inherits_baby_role(self):
        for before_zero in (False, True):
            people = self.people()
            obs = frame(0, 0, 0)
            people.observe(obs)
            self.review(people, obs)
            if before_zero:
                people.observe(frame(1, 0.1, 0.1, boxes=()))
            boundary = frame(2, 0.2, 3, boxes=(BOX,) if before_zero else ())
            people.observe(boundary)
            resumed = frame(3, 0.3, 3.1)
            people.observe(resumed)
            self.assertTrue(all(t["role"] is None for t in people.cameras["a"]["tracks"].values()))
            self.assertEqual(
                1 if before_zero else 0,
                len(people.cameras["a"]["history"][people._key(boundary.frame)]["tracks"]),
            )
            self.assertEqual("stale_or_unbound_review", self.review(people, obs)["status"])

    def test_two_people_merged_and_duplicate_geometry_cannot_carry_roles(self):
        for boxes in ((BOX, OTHER), ((0.1, 0.1, 0.9, 0.8),), (BOX, BOX)):
            people = self.people()
            people.observe(frame(0, 0, 0, boxes=(BOX, OTHER)))
            obs = frame(1, 0.2, 3, boxes=boxes)
            people.observe(obs)
            self.assertEqual(len(boxes), len(people.cameras["a"]["tracks"]))
            self.assertTrue(all(t["role"] is None for t in people.cameras["a"]["tracks"].values()))
            self.assertEqual(len(boxes), len(set(people.cameras["a"]["tracks"])))

    def test_lanes_and_cameras_have_independent_geometry_and_review_scopes(self):
        people = self.people(auxiliary=True)
        old = frame(0, 0, 0)
        people.observe(old)
        people.observe_auxiliary(old, view="rotation_180")
        self.review(people, old, view="rotation_180")
        alternate_id = list(people.auxiliary.cameras["a"]["tracks"])
        obs = frame(1, 0.2, 3)
        people.observe_auxiliary(obs, view="rotation_180")
        self.assertEqual(alternate_id, list(people.auxiliary.cameras["a"]["tracks"]))
        self.assertEqual(1, people.cameras["a"]["history_resets"])
        self.assertEqual(0, people.cameras["b"]["history_resets"])
        self.assertEqual(
            "stale_or_unbound_review", self.review(people, old, view="rotation_180")["status"]
        )
        self.assertEqual(
            1,
            people.review_trace(obs, view="rotation_180")["geometry_continuity"][
                "utc_geometry_retained_total"
            ],
        )

    def test_rejected_order_has_no_counter_or_lane_side_effect(self):
        people = self.people(auxiliary=True)
        obs = frame(5, 0, 0, generation=1)
        people.observe(obs)
        before = people.review_trace(obs)
        for seq, gen in ((5, 1), (4, 1), (9, 0)):
            self.assertFalse(people.observe(frame(seq, 0.1, 3, generation=gen)))
            self.assertEqual(before, people.review_trace(obs))

    def test_full_capacity_failure_falls_back_without_expanding_track_limit(self):
        people = self.people()
        boxes = tuple((i / 128, 0.0, (i + 0.5) / 128, 0.3) for i in range(128))
        people.observe(frame(0, 0, 0, boxes=boxes))
        obs = frame(1, 0.1, 3, boxes=((0.4, 0.6, 0.6, 0.9),))
        people.observe(obs)
        self.assertEqual(1, len(people.cameras["a"]["tracker"].state.active_tracks))
        diagnostic = people.review_trace(obs)["geometry_continuity"]
        self.assertEqual(1, diagnostic["candidate_failures_total"])
        self.assertEqual("matcher_rejected", diagnostic["latest_boundary"]["fallback_code"])
        self.assertEqual("reset", diagnostic["latest_boundary"]["action"])

    def test_unexpected_error_propagates_and_invalidates_old_roles_and_reviews(self):
        people = self.people()
        old = frame(0, 0, 0)
        people.observe(old)
        self.review(people, old)
        with (
            patch.object(MotionObjectTracker, "process", side_effect=RuntimeError("synthetic")),
            self.assertRaises(RuntimeError),
        ):
            people.observe(frame(1, 0.1, 3))
        self.assertIsNone(people.cameras["a"]["previous"])
        self.assertEqual({}, people.cameras["a"]["tracks"])
        self.assertEqual("stale_or_unbound_review", self.review(people, old)["status"])
        self.assertTrue(people.observe(frame(2, 0.2, 3.1)))

    def test_controlled_rejection_does_not_hide_empty_baseline_failure(self):
        people = self.people()
        people.observe(frame(0, 0, 0))
        with (
            patch.object(
                MotionObjectTracker, "process", side_effect=MotionTrackerError("synthetic")
            ),
            self.assertRaises(MotionTrackerError),
        ):
            people.observe(frame(1, 0.1, 3))
        self.assertIsNone(people.cameras["a"]["previous"])

    def test_audit_failure_cannot_publish_a_retained_success(self):
        people = self.people()
        people.observe(frame(0, 0, 0))
        with (
            patch(
                "baby_monitor.live_people.describe_association",
                side_effect=RuntimeError("synthetic"),
            ),
            self.assertRaises(RuntimeError),
        ):
            people.observe(frame(1, 0.1, 3))
        camera = people.cameras["a"]
        self.assertIsNone(camera["previous"])
        self.assertEqual(0, camera["utc_geometry_retained"])
        self.assertEqual("failed", camera["latest_geometry_boundary"]["action"])

    def test_view_family_change_does_not_preserve_geometry(self):
        people = self.people()
        people.observe(frame(0, 0, 0))
        old_ids = list(people.cameras["a"]["tracks"])
        people.observe(frame(1, 0.1, 3), view="rotation_180")
        self.assertNotEqual(old_ids, list(people.cameras["a"]["tracks"]))
        self.assertEqual(0, people.cameras["a"]["utc_geometry_retained"])

    def test_actual_carried_checkpoint_replays_and_export_cannot_mutate_state(self):
        people = self.people()
        people.observe(frame(0, 0, 0))
        obs = frame(1, 0.1, 3)
        people.observe(obs)
        newer = frame(2, 0.2, 3.1)
        people.observe(newer)
        trace = people.review_trace(obs)
        config, before, batch, expected = _inputs(trace)
        replay = MotionObjectTracker(
            config=config, camera_id="a", detector_adapter="live_yolo", state=before
        )
        actual = []
        for row in batch.frames:
            actual.extend(
                [row.frame_id, d.detection_id, o.tracking_id]
                for d, o in zip(row.detections, replay.process(row), strict=True)
            )
        self.assertEqual(expected, actual)
        trace["geometry_continuity"]["latest_boundary"].clear()
        self.assertTrue(people.review_trace(newer)["geometry_continuity"]["latest_boundary"])
        self.assertEqual(trace, json.loads(json.dumps(trace, allow_nan=False)))

    def test_default_off_preserves_old_reset_and_trace_protocol(self):
        people = LivePeople(("a",), association_diagnostics=True)
        people.observe(frame(0, 0, 0))
        old_id = list(people.cameras["a"]["tracks"])
        obs = frame(1, 0.1, 3)
        people.observe(obs)
        self.assertNotEqual(old_id, list(people.cameras["a"]["tracks"]))
        self.assertNotIn("geometry_continuity", people.review_trace(obs))
        self.assertEqual(1, people.review_trace(obs)["latest_reset_checkpoint"]["version"])

    def test_flags_are_strict_and_monitor_gate_precedes_any_io(self):
        for value in (1, "yes", None):
            with self.assertRaises(ValueError):
                LivePeople(("a",), utc_geometry_continuity=value)
        for value, trace, tracking in ((True, False, True), (True, True, False), (1, True, True)):
            with self.assertRaisesRegex(ValueError, "UTC geometry"):
                run_live_monitor(
                    streams=(),
                    detector=None,
                    local_vlm=None,
                    root=Path("/missing"),
                    review_admission=lambda: True,
                    people_tracking=tracking,
                    people_review_trace=trace,
                    people_utc_geometry_continuity=value,
                )
