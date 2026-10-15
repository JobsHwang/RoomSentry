from __future__ import annotations

import copy
import json
import random
import unittest
from datetime import UTC, datetime, timedelta

from baby_monitor.motion_tracker import (
    MotionActiveTrack,
    MotionTrackerConfig,
    MotionTrackSample,
    _match,
)
from baby_monitor.object_tracker import FrameDetection
from baby_monitor.tracker_evaluation.weak_continuation import compare_weak_continuation
from baby_monitor.tracker_evaluation.weak_geometry_guard import (
    _primary_pairs,
    compare_guarded_weak_continuation,
)
from tests.test_live_weak_trace import replay_row


def box(x=0.1, *, confidence=0.8):
    return {"bbox_xyxy": [x, 0.2, x + 0.05, 0.3], "confidence": confidence}


def row(sequence, seconds, native=(), weak=(), **kwargs):
    return replay_row(sequence, seconds, native=list(native), low=[*native, *weak], **kwargs)


def links(report, baseline=False):
    return {
        (frame["camera_id"], tuple(left), tuple(right))
        for frame in (report["baseline"]["frames"] if baseline else report["frames"])
        for left, right in (
            frame["arms"]["strong"]["strong_links"] if baseline else frame["strong_links"]
        )
    }


class WeakGeometryGuardTests(unittest.TestCase):
    def test_original_three_arm_baseline_preserved_verbatim(self):
        rows = [
            row(0, 1, [box()]),
            row(1, 1.2, weak=[box(0.13, confidence=0.25)]),
            row(2, 1.4, [box(0.16)]),
        ]
        before = copy.deepcopy(rows)
        report = compare_guarded_weak_continuation(rows)
        self.assertEqual(compare_weak_continuation(rows), report["baseline"])
        self.assertEqual(before, rows)
        for key in (
            "production_modified",
            "live_assignments_reproduced",
            "accuracy_evaluated",
            "role_propagation",
        ):
            self.assertFalse(report[key])
        self.assertEqual(0, report["new_model_calls"])

    def test_strong_only_matches_full_baseline_for_seeded_synthetic_sequences(self):
        randomizer = random.Random(529)
        rows = []
        for i in range(140):
            native = [box(0.1 + randomizer.random() * 0.08), box(0.7 + randomizer.random() * 0.08)]
            if i % 11 == 0:
                native = []
            if i % 7 == 0:
                native.reverse()
            rows.append(row(i, 1 + i * 0.1, native))
        report = compare_guarded_weak_continuation(rows)
        self.assertEqual(links(report, True), links(report))
        self.assertTrue(all(frame["weak_recovery_matches"] == 0 for frame in report["frames"]))

    def test_weak_recovery_only_when_strong_reference_cannot_qualify(self):
        rows = [row(0, 1, [box()])]
        rows += [
            row(i, 1 + i * 0.1, weak=[box(0.1 + i * 0.03, confidence=0.25)]) for i in range(1, 4)
        ]
        rows.append(row(4, 1.4, [box(0.22)]))
        report = compare_guarded_weak_continuation(rows)
        self.assertEqual(1, report["frames"][-1]["weak_recovery_matches"])
        self.assertEqual([], report["baseline"]["frames"][-1]["arms"]["strong"]["strong_links"])
        self.assertEqual([[[0, 0, 0], [0, 4, 0]]], report["frames"][-1]["strong_links"])

    def test_reference_match_reserved_despite_changed_weak_kinematics(self):
        clock = datetime(2000, 1, 1, tzinfo=UTC)

        def make(identifier, x):
            return MotionActiveTrack(
                identifier, "person", MotionTrackSample(clock, 0.8, tuple(box(x)["bbox_xyxy"]))
            )

        references = {"track_a": make("track_a", 0.1), "track_b": make("track_b", 0.14)}
        mixed = (make("track_a", 0.3), make("track_b", 0.1))
        detection = (FrameDetection("s_0", "person", 0.8, tuple(box()["bbox_xyxy"])),)
        config = MotionTrackerConfig(high_confidence_threshold=0.35, low_confidence_threshold=0.20)
        kwargs = dict(occurred_at=clock + timedelta(seconds=0.1), minimum_score=0.30, config=config)
        self.assertEqual(((1, 0),), _match(mixed, detection, **kwargs))
        pairs, reserved, recovered, protected = _primary_pairs(
            mixed, references, detection, kwargs["occurred_at"], config
        )
        self.assertEqual(((0, 0),), pairs)
        self.assertEqual((1, 0, 1), (reserved, recovered, protected))

    def test_recovery_cannot_steal_track_from_qualified_reference(self):
        clock = datetime(2000, 1, 1, tzinfo=UTC)
        reference = MotionActiveTrack(
            "track_a", "person", MotionTrackSample(clock, 0.8, tuple(box()["bbox_xyxy"]))
        )
        mixed = MotionActiveTrack(
            "track_a", "person", MotionTrackSample(clock, 0.25, tuple(box(0.6)["bbox_xyxy"]))
        )
        strong = tuple(
            FrameDetection(f"s_{i}", "person", 0.8, tuple(box(x)["bbox_xyxy"]))
            for i, x in enumerate((0.1, 0.6))
        )
        config = MotionTrackerConfig(high_confidence_threshold=0.35, low_confidence_threshold=0.20)
        pairs, reserved, recovered, protected = _primary_pairs(
            (mixed,), {"track_a": reference}, strong, clock + timedelta(seconds=0.1), config
        )
        self.assertEqual(((0, 0),), pairs)
        self.assertEqual((1, 0, 1), (reserved, recovered, protected))

    def test_weak_without_strong_seed_cannot_create_person(self):
        report = compare_guarded_weak_continuation(
            [row(0, 1, weak=[box(confidence=0.25)]), row(1, 1.1, [box()])]
        )
        self.assertTrue(
            all(
                not frame["strong_links"] and not frame["weak_continuations"]
                for frame in report["frames"]
            )
        )

    def test_weak_does_not_renew_two_second_strong_seed(self):
        rows = [row(0, 1, [box()])]
        rows.extend(row(i, 1 + i * 0.5, weak=[box(confidence=0.25)]) for i in range(1, 6))
        rows.append(row(6, 4, [box()]))
        report = compare_guarded_weak_continuation(rows)
        self.assertEqual(1, report["frames"][4]["weak_continuations"])
        self.assertEqual(0, report["frames"][5]["weak_continuations"])
        self.assertEqual([], report["frames"][-1]["strong_links"])

    def test_duplicate_and_ambiguous_weak_rejected(self):
        duplicate = box(confidence=0.25)
        report = compare_guarded_weak_continuation(
            [row(0, 1, [box()]), row(1, 1.1, weak=[duplicate, duplicate])]
        )
        self.assertEqual(2, report["frames"][-1]["duplicate_weak_rejected"])
        self.assertEqual(0, report["frames"][-1]["weak_continuations"])
        report = compare_guarded_weak_continuation(
            [row(0, 1, [box(), box(0.12)]), row(1, 1.1, weak=[box(0.11, confidence=0.25)])]
        )
        self.assertEqual(1, report["frames"][-1]["ambiguous_weak_rejected"])
        self.assertEqual(0, report["frames"][-1]["weak_continuations"])

    def test_generation_gap_and_mismatch_are_causal_breaks(self):
        for bad in (
            row(1, 1.1, [box()], generation=1),
            row(1, 4, [box()]),
            row(1, 1.1, [box()], [box(confidence=0.4)]),
        ):
            report = compare_guarded_weak_continuation(
                [
                    row(0, 1, [box()]),
                    bad,
                    row(
                        2,
                        4.2 if bad["received_monotonic"] == 4 else 1.2,
                        [box()],
                        generation=bad["generation"],
                    ),
                ]
            )
            self.assertEqual([], report["frames"][1]["strong_links"])
            if report["frames"][1]["rejected"]:
                self.assertEqual([], report["frames"][2]["strong_links"])
                self.assertEqual(1, report["frames"][1]["native_count"])

    def test_failed_capture_preserves_denominator_and_resets_candidate(self):
        bad = row(1, 1.1, [box()])
        bad.update(low=None, strong_output_equal=None)
        report = compare_guarded_weak_continuation([row(0, 1, [box()]), bad, row(2, 1.2, [box()])])
        self.assertTrue(report["frames"][1]["rejected"])
        self.assertEqual(1, report["frames"][1]["native_count"])
        self.assertEqual([], report["frames"][2]["strong_links"])

    def test_prefix_determinism_json_and_camera_isolation(self):
        rows = [row(0, 1, [box()]), row(1, 1.1, weak=[box(confidence=0.25)]), row(2, 1.2, [box()])]
        report = compare_guarded_weak_continuation(rows)
        self.assertEqual(
            report["frames"][:2], compare_guarded_weak_continuation(rows[:2])["frames"]
        )
        self.assertEqual(report, compare_guarded_weak_continuation(rows))
        self.assertEqual(report, json.loads(json.dumps(report)))
        second = [{**item, "camera_id": "camera_02"} for item in rows]
        interleaved = [item for pair in zip(rows, second, strict=True) for item in pair]
        both = compare_guarded_weak_continuation(interleaved)
        self.assertEqual(
            report["frames"], [f for f in both["frames"] if f["camera_id"] != "camera_02"]
        )

    def test_invalid_chronology_and_forged_equality_fail_closed(self):
        with self.assertRaises(ValueError):
            compare_guarded_weak_continuation([row(1, 1), row(0, 1.1)])
        bad = row(0, 1, [box()])
        bad["strong_output_equal"] = False
        with self.assertRaises(ValueError):
            compare_guarded_weak_continuation([bad])
