from __future__ import annotations

import unittest
from dataclasses import replace

from baby_monitor.adapters.live_detector import PersistentYoloDetector
from baby_monitor.live_detection import AlternateViewReviewPolicy
from baby_monitor.live_review_cadence import (
    AlternateReviewCadence,
    validate_empty_review_intervals,
)
from tests.test_live_detection import frame, observation
from tests.test_live_shadow_review import run_runner, start


def client(intervals=None, *, enabled=True):
    return PersistentYoloDetector(
        command=("fixture",),
        model_path="fixture",
        manifest_path="fixture",
        review_rotation_180=enabled,
        review_empty_interval_seconds_by_camera=intervals,
    )


class ReviewCadenceTests(unittest.TestCase):
    def test_default_and_strict_option_admission(self):
        self.assertEqual({}, client().settings["review_empty_interval_seconds_by_camera"])
        self.assertEqual(
            {"cam1": 0.2}, validate_empty_review_intervals({"cam1": 0.2}, enabled=True)
        )
        for value in (
            False,
            [],
            "0.2",
            {"cam1": True},
            {"cam1": "0.2"},
            {"cam1": 0.1},
            {"cam1": 1.01},
            {"cam1": float("nan")},
            {"cam1": float("inf")},
            {"bad/id": 0.2},
            {"": 0.2},
            {1: 0.2},
            {"x" * 81: 0.2},
            {str(i): 0.2 for i in range(5)},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                client(value)
        with self.assertRaises(ValueError):
            client({"cam1": 0.2}, enabled=False)
        self.assertEqual(
            {}, client({}, enabled=False).settings["review_empty_interval_seconds_by_camera"]
        )

    def test_default_rotation_is_shared_across_empty_and_shadow(self):
        cadence = AlternateReviewCadence({})
        self.assertTrue(cadence.rotation_due("cam1", native_present=False, now=10))
        cadence.mark_rotation("cam1", 10)
        self.assertFalse(cadence.rotation_due("cam1", native_present=True, now=10.2))
        self.assertFalse(cadence.rotation_due("cam1", native_present=False, now=10.2))
        self.assertTrue(cadence.rotation_due("cam1", native_present=True, now=11))

    def test_opt_in_only_accelerates_named_empty_view_and_not_fallback(self):
        cadence = AlternateReviewCadence({"cam1": 0.2})
        cadence.mark_rotation("cam1", 10)
        cadence.mark_rotation("cam2", 10)
        cadence.mark_fallback("cam1", 10)
        self.assertTrue(cadence.rotation_due("cam1", native_present=False, now=10.25))
        self.assertFalse(cadence.rotation_due("cam1", native_present=True, now=10.25))
        self.assertFalse(cadence.rotation_due("cam2", native_present=False, now=10.25))
        self.assertFalse(cadence.fallback_due("cam1", 10.25))
        self.assertTrue(cadence.fallback_due("cam2", 10.25))
        self.assertTrue(cadence.fallback_due("cam1", 11))

    def test_runner_checks_named_empty_camera_more_often_without_changing_native(self):
        batches = [[frame(sequence=i), frame("cam2", sequence=i)] for i in range(3)]
        messages, calls, loads = run_runner(
            batches,
            [(False, False), (True, True), (False, False), (True,), (False, False), (True, True)],
            [10, 10.25, 11.25],
            settings={"review_empty_interval_seconds_by_camera": {"cam1": 0.2}},
        )
        self.assertEqual(1, loads)
        self.assertEqual({"cam1": 0.2}, messages[0]["review_empty_interval_seconds_by_camera"])
        self.assertEqual([2, 1, 2], [len(c["rotations"]) for c in calls[5::2]])
        self.assertTrue(all(not row["detections"] for m in messages[1:] for row in m["frames"]))
        self.assertEqual(
            [True, True, True], [m["frames"][0]["review_checked"] for m in messages[1:]]
        )
        self.assertEqual(
            [True, False, True], [m["frames"][1]["review_checked"] for m in messages[1:]]
        )

    def test_runner_keeps_shadow_at_one_hz_under_fast_override(self):
        messages, _, _ = run_runner(
            [[frame(sequence=i)] for i in range(3)],
            [(True,), (True,), (True,), (True,), (True,)],
            [10, 10.25, 11.25],
            settings={
                "review_nonempty_shadow": True,
                "review_empty_interval_seconds_by_camera": {"cam1": 0.2},
            },
        )
        self.assertEqual(
            [True, False, True], [m["frames"][0]["shadow_checked"] for m in messages[1:]]
        )
        self.assertTrue(all(len(m["frames"][0]["detections"]) == 1 for m in messages[1:]))

    def test_runner_keeps_double_empty_fallback_at_one_hz(self):
        messages, calls, _ = run_runner(
            [[frame(sequence=i)] for i in range(3)],
            [(False,), (False,), (True,), (False,), (False,), (False,), (False,), (True,)],
            [10, 10.25, 11.25],
            settings={
                "review_empty_scale_fallback": True,
                "review_empty_interval_seconds_by_camera": {"cam1": 0.2},
            },
        )
        rows = [m["frames"][0] for m in messages[1:]]
        self.assertEqual(
            [True, False, True], [r.get("review_fallback_checked", False) for r in rows]
        )
        self.assertEqual([1, 0, 1], [len(r["review_detections"]) for r in rows])
        self.assertEqual(2, sum(c["imgsz"] == (576, 960) for c in calls[8:]))

    def test_slow_rotation_cannot_backdate_fallback_start(self):
        messages, calls, _ = run_runner(
            [[frame(sequence=i)] for i in range(3)],
            [(False,), (False,), (True,), (False,), (False,), (False,), (False,), (True,)],
            [10, 11.4, 12.4],
            settings={"review_empty_scale_fallback": True},
            # First actual fallback starts 11.2, second alternate ends 11.45.
            # The old rotation time (10) must not admit another fallback at 11.45.
            prediction_delays=[0, 1.2, 0, 0, 0.05, 0, 0, 0],
        )
        rows = [m["frames"][0] for m in messages[1:]]
        self.assertEqual(
            [True, False, True], [r.get("review_fallback_checked", False) for r in rows]
        )
        self.assertEqual(2, sum(c["imgsz"] == (576, 960) for c in calls[8:]))

    def test_invalid_runner_cadence_is_rejected_before_model_load(self):
        for settings in (
            {"review_empty_interval_seconds_by_camera": {"cam1": 0.1}},
            {
                "review_empty_interval_seconds_by_camera": {"cam1": 0.2},
                "review_rotation_180": False,
            },
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_runner([], [], [], settings=settings)

    def test_requested_cadence_requires_exact_ready_ack_and_expensive_ceiling(self):
        ready = {
            "type": "ready",
            "review_rotation_180": True,
            "review_empty_interval_seconds_by_camera": {"cam1": 0.2},
            "review_shadow_interval_seconds": 1.0,
            "review_fallback_interval_seconds": 1.0,
        }
        start(client({"cam1": 0.2}), ready)
        for key, value in (
            ("review_empty_interval_seconds_by_camera", None),
            ("review_empty_interval_seconds_by_camera", {"cam1": 0.3}),
            ("review_shadow_interval_seconds", 0.2),
            ("review_fallback_interval_seconds", True),
            ("review_rotation_180", False),
        ):
            with self.subTest(key=key), self.assertRaises((RuntimeError, ValueError)):
                start(client({"cam1": 0.2}), {**ready, key: value})
        with self.assertRaises(RuntimeError):
            start(client(), ready)


class AlternateConfirmationSpacingTests(unittest.TestCase):
    def test_many_fast_matches_cannot_trigger_before_one_second(self):
        policy = AlternateViewReviewPolicy()
        for i, now in enumerate((10, 10.25, 10.5, 10.75)):
            self.assertIsNone(policy.observe(observation(sequence=i, now=now, count=1)))
        self.assertIsNotNone(policy.observe(observation(sequence=4, now=11, count=1)))

    def test_duplicate_and_out_of_order_frames_reset_confirmation(self):
        policy = AlternateViewReviewPolicy()
        self.assertIsNone(policy.observe(observation(sequence=2, now=10, count=1)))
        self.assertIsNone(policy.observe(observation(sequence=2, now=11, count=1)))
        self.assertIsNone(policy.observe(observation(sequence=1, now=11, count=1)))
        self.assertIsNone(policy.observe(observation(sequence=3, now=11.25, count=1)))
        self.assertIsNotNone(policy.observe(observation(sequence=4, now=12.25, count=1)))

    def test_duplicate_empty_and_out_of_order_weak_frames_are_continuity_gaps(self):
        for middle in (
            observation(sequence=2, now=10.5, count=0),
            observation(sequence=1, now=10.5, count=1),
        ):
            policy = AlternateViewReviewPolicy(minimum_frame_confidence=0.9)
            strong = observation(sequence=2, now=10, count=1)
            strong = replace(strong, detections=(replace(strong.detections[0], confidence=1.0),))
            policy.observe(strong)
            self.assertIsNone(policy.observe(middle))
            self.assertIsNone(
                policy.observe(
                    replace(strong, frame=frame(sequence=3, now=11), completed_monotonic=11)
                )
            )
            self.assertIsNotNone(
                policy.observe(
                    replace(strong, frame=frame(sequence=4, now=12), completed_monotonic=12)
                )
            )

    def test_receipt_and_completion_reversals_reset_chain_but_keep_high_water_mark(self):
        for middle in (
            replace(observation(sequence=1, now=9.9, count=1), completed_monotonic=10.5),
            replace(observation(sequence=1, now=10.5, count=1), completed_monotonic=9.9),
        ):
            policy = AlternateViewReviewPolicy()
            policy.observe(observation(sequence=0, now=10, count=1))
            self.assertIsNone(policy.observe(middle))
            self.assertIsNone(policy.observe(observation(sequence=2, now=11, count=1)))
            self.assertIsNotNone(policy.observe(observation(sequence=3, now=12, count=1)))

    def test_signature_change_restarts_one_second_confirmation(self):
        policy = AlternateViewReviewPolicy()
        policy.observe(observation(sequence=0, now=10, count=1))
        self.assertIsNone(policy.observe(observation(sequence=1, now=10.5, count=2)))
        self.assertIsNone(policy.observe(observation(sequence=2, now=11, count=2)))
        self.assertIsNotNone(policy.observe(observation(sequence=3, now=11.5, count=2)))

    def test_order_gap_does_not_reset_thirty_second_trigger_cooldown(self):
        policy = AlternateViewReviewPolicy()
        policy.observe(observation(sequence=0, now=10, count=1))
        self.assertIsNotNone(policy.observe(observation(sequence=1, now=11, count=1)))
        policy.observe(observation(sequence=1, now=11.5, count=0))
        policy.observe(observation(sequence=2, now=12, count=1))
        self.assertIsNone(policy.observe(observation(sequence=3, now=13, count=1)))

    def test_long_gap_frame_becomes_new_first_vote(self):
        policy = AlternateViewReviewPolicy()
        policy.observe(observation(sequence=0, now=10, count=1))
        self.assertIsNone(policy.observe(observation(sequence=1, now=14, count=1)))
        self.assertIsNotNone(policy.observe(observation(sequence=2, now=15, count=1)))

    def test_receipt_gap_is_not_hidden_by_short_completion_gap(self):
        policy = AlternateViewReviewPolicy()
        policy.observe(replace(observation(sequence=0, now=10, count=1), completed_monotonic=100))
        self.assertIsNone(
            policy.observe(
                replace(observation(sequence=1, now=14, count=1), completed_monotonic=101)
            )
        )
        self.assertIsNotNone(
            policy.observe(
                replace(observation(sequence=2, now=15, count=1), completed_monotonic=102)
            )
        )

    def test_generation_change_gets_new_high_water_mark_and_first_vote(self):
        policy = AlternateViewReviewPolicy()
        policy.observe(observation(sequence=99, now=10, count=1))
        for sequence, now in ((0, 10.25), (1, 10.75)):
            fresh = observation(sequence=sequence, now=now, count=1)
            fresh = replace(fresh, frame=replace(fresh.frame, generation=1))
            self.assertIsNone(policy.observe(fresh))
        later = observation(sequence=2, now=11.25, count=1)
        later = replace(later, frame=replace(later.frame, generation=1))
        self.assertIsNotNone(policy.observe(later))

    def test_empty_or_weak_middle_check_resets_confirmation_age(self):
        for middle_count in (0, 1):
            policy = AlternateViewReviewPolicy(minimum_frame_confidence=0.99 if middle_count else 0)
            if middle_count:
                strong = observation(sequence=0, now=10, count=1)
                strong = replace(
                    strong, detections=(replace(strong.detections[0], confidence=1.0),)
                )
            else:
                strong = observation(sequence=0, now=10, count=1)
            policy.observe(strong)
            policy.observe(observation(sequence=1, now=10.5, count=middle_count))
            policy.observe(
                replace(strong, frame=frame(sequence=2, now=10.75), completed_monotonic=10.75)
            )
            self.assertIsNone(
                policy.observe(
                    replace(strong, frame=frame(sequence=3, now=11.25), completed_monotonic=11.25)
                )
            )
            self.assertIsNotNone(
                policy.observe(
                    replace(strong, frame=frame(sequence=4, now=11.75), completed_monotonic=11.75)
                )
            )

    def test_slow_inference_does_not_confirm_two_rapidly_received_frames(self):
        policy = AlternateViewReviewPolicy()
        policy.observe(observation(sequence=0, now=10, count=1))
        delayed = replace(observation(sequence=1, now=10.25, count=1), completed_monotonic=12)
        self.assertIsNone(policy.observe(delayed))
        later = replace(observation(sequence=2, now=11, count=1), completed_monotonic=12.25)
        self.assertIsNotNone(policy.observe(later))

    def test_confirmation_interval_admission(self):
        for value in (True, 0.2, 3.1, float("nan"), "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AlternateViewReviewPolicy(minimum_confirmation_seconds=value)


if __name__ == "__main__":
    unittest.main()
