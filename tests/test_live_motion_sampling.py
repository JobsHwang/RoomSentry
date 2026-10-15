from __future__ import annotations

import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

from baby_monitor.live_detection import FrameMailbox, MotionBurstSettings
from scripts.compare_live_sampling import phase_sweep
from tests.test_live_detection import frame


def changed_frame(*, sequence=1, now=10.05, camera="cam1", value=230):
    pixels = bytearray(bytes([30]) * (32 * 24 * 3))
    for y in range(8, 16):
        for x in range(12, 20):
            offset = (y * 32 + x) * 3
            pixels[offset : offset + 3] = bytes([value]) * 3
    return replace(frame(camera, sequence, now), width=32, height=24, pixels=bytes(pixels))


def blank_frame(*, sequence=0, now=10.0, camera="cam1", value=30):
    return replace(
        frame(camera, sequence, now), width=32, height=24, pixels=bytes([value]) * (32 * 24 * 3)
    )


class LiveMotionSamplingTests(unittest.TestCase):
    def mailbox(self, cameras=("cam1",), **settings):
        return FrameMailbox(cameras, sample_fps=5, motion_burst=MotionBurstSettings(**settings))

    def test_change_on_skipped_frame_is_admitted_before_baseline_deadline(self):
        mailbox = self.mailbox()
        mailbox.put(blank_frame())
        mailbox.take(10)
        mailbox.put(changed_frame())
        self.assertEqual([1], [item.sequence for item in mailbox.take(10.05)])
        status = mailbox.snapshot(10.05)["cam1"]
        self.assertTrue(status["sampling_burst_active"])
        self.assertEqual(20, status["sampling_target_fps"])
        self.assertEqual(1, status["motion_burst_episodes"])
        self.assertEqual(1, status["motion_trigger_frames"])

    def test_other_camera_stays_at_baseline(self):
        mailbox = self.mailbox(("cam1", "cam2"))
        for camera in ("cam1", "cam2"):
            mailbox.put(blank_frame(camera=camera))
        mailbox.take(10)
        mailbox.put(changed_frame())
        mailbox.put(blank_frame(camera="cam2", sequence=1, now=10.05))
        self.assertEqual(["cam1"], [item.camera_id for item in mailbox.take(10.05)])
        self.assertFalse(mailbox.snapshot(10.05)["cam2"]["sampling_burst_active"])

    def test_hold_expires_and_motion_renews_it_without_unbounded_rate(self):
        mailbox = self.mailbox(hold_seconds=0.2)
        mailbox.put(blank_frame())
        mailbox.take(10)
        mailbox.put(changed_frame())
        mailbox.take(10.05)
        mailbox.put(blank_frame(sequence=2, now=10.1))
        mailbox.take(10.1)
        self.assertTrue(mailbox.snapshot(10.29)["cam1"]["sampling_burst_active"])
        mailbox.put(blank_frame(sequence=3, now=10.3))
        self.assertEqual([3], [item.sequence for item in mailbox.take(10.3)])
        self.assertFalse(mailbox.snapshot(10.3)["cam1"]["sampling_burst_active"])
        mailbox.put(blank_frame(sequence=4, now=10.35))
        self.assertEqual((), mailbox.take(10.35))
        self.assertEqual(1, mailbox.snapshot()["cam1"]["motion_burst_episodes"])

    def test_uniform_brightness_change_does_not_trigger_motion(self):
        mailbox = self.mailbox()
        mailbox.put(blank_frame())
        mailbox.take(10)
        mailbox.put(blank_frame(sequence=1, now=10.05, value=130))
        self.assertEqual((), mailbox.take(10.05))
        self.assertEqual(0, mailbox.snapshot()["cam1"]["motion_trigger_frames"])

    def test_small_noise_does_not_trigger_motion(self):
        mailbox = self.mailbox()
        mailbox.put(blank_frame())
        mailbox.take(10)
        mailbox.put(changed_frame(value=35))
        self.assertEqual((), mailbox.take(10.05))

    def test_tiny_change_can_be_missed_and_does_not_become_a_person_fact(self):
        mailbox = self.mailbox()
        mailbox.put(blank_frame())
        mailbox.take(10)
        altered = bytearray(blank_frame().pixels)
        altered[0:3] = b"\xff\xff\xff"
        mailbox.put(replace(blank_frame(sequence=1, now=10.05), pixels=bytes(altered)))
        self.assertEqual((), mailbox.take(10.05))
        self.assertFalse(mailbox.snapshot()["cam1"]["sampling_burst_active"])
        self.assertNotIn("person_count", mailbox.snapshot()["cam1"])

    def test_continuous_changes_keep_burst_bounded_and_quiet_returns_to_baseline(self):
        mailbox = self.mailbox(hold_seconds=0.2)
        admitted = []
        for sequence in range(201):
            now = 10 + sequence / 100
            moving = sequence < 100 and sequence % 2 == 1
            current = (
                changed_frame(sequence=sequence, now=now)
                if moving
                else blank_frame(sequence=sequence, now=now)
            )
            mailbox.put(current)
            admitted.extend(item.received_monotonic for item in mailbox.take(now))
        self.assertTrue(
            all(b - a >= 0.05 - 1e-9 for a, b in zip(admitted, admitted[1:], strict=False))
        )
        self.assertLessEqual(len(admitted), 30)
        self.assertFalse(mailbox.snapshot(12)["cam1"]["sampling_burst_active"])

    def test_reconnect_and_long_gap_reset_reference_and_burst(self):
        for reconnect in (True, False):
            with self.subTest(reconnect=reconnect):
                mailbox = self.mailbox()
                mailbox.put(blank_frame())
                mailbox.take(10)
                mailbox.put(changed_frame())
                mailbox.take(10.05)
                now = 10.1 if reconnect else 12.0
                resumed = blank_frame(sequence=2, now=now)
                if reconnect:
                    resumed = replace(resumed, generation=1)
                mailbox.put(resumed)
                mailbox.take(now)
                self.assertFalse(mailbox.snapshot(now)["cam1"]["sampling_burst_active"])
                self.assertEqual(1, mailbox.snapshot(now)["cam1"]["motion_trigger_frames"])

    def test_adaptive_queue_overload_is_not_hidden_as_sampling_skip(self):
        mailbox = self.mailbox()
        mailbox.put(blank_frame())
        mailbox.take(10)
        for sequence in range(1, 5):
            mailbox.put(changed_frame(sequence=sequence, now=10 + sequence / 20))
        self.assertEqual([4], [item.sequence for item in mailbox.take(10.2)])
        row = mailbox.snapshot()["cam1"]
        self.assertEqual(3, row["overload_discarded_frames"])
        self.assertEqual(0, row["sampling_skipped_frames"])
        self.assertEqual(
            row["decoded_frames"], row["eligible_frames"] + row["sampling_skipped_frames"]
        )
        self.assertEqual(
            row["eligible_frames"],
            row["dispatched_frames"] + row["discarded_frames"] + row["queued_frames"],
        )

    def test_rate_transition_does_not_accept_frames_faster_than_burst_interval(self):
        mailbox = self.mailbox()
        mailbox.put(blank_frame())
        mailbox.take(10)
        mailbox.put(changed_frame(now=10.01))
        self.assertEqual((), mailbox.take(10.01))
        mailbox.put(changed_frame(sequence=2, now=10.05))
        self.assertEqual([2], [item.sequence for item in mailbox.take(10.05)])

    def test_two_slot_burst_preserves_first_short_change_and_latest_evidence(self):
        mailbox = self.mailbox(pending_capacity=2)
        mailbox.put(blank_frame())
        mailbox.take(10)
        mailbox.put(changed_frame(sequence=1, now=10.05))
        for sequence in range(2, 5):
            mailbox.put(blank_frame(sequence=sequence, now=10 + sequence / 20))
        self.assertEqual([1], [item.sequence for item in mailbox.take(10.2)])
        self.assertEqual([4], [item.sequence for item in mailbox.take(10.2)])
        row = mailbox.snapshot()["cam1"]
        self.assertEqual(2, row["overload_discarded_frames"])
        self.assertEqual(2, row["sampling_pending_capacity"])
        self.assertEqual(
            row["eligible_frames"],
            row["dispatched_frames"] + row["discarded_frames"] + row["queued_frames"],
        )

    def test_two_slot_burst_still_expires_stale_evidence(self):
        mailbox = self.mailbox(pending_capacity=2)
        mailbox.put(blank_frame())
        mailbox.take(10)
        mailbox.put(changed_frame(sequence=1, now=10.05))
        mailbox.put(blank_frame(sequence=2, now=10.1))
        self.assertEqual((), mailbox.take(12))
        self.assertEqual(2, mailbox.snapshot()["cam1"]["stale_discarded_frames"])

    def test_pending_prechange_baseline_cannot_displace_first_burst_frame(self):
        mailbox = self.mailbox(pending_capacity=2)
        mailbox.put(blank_frame())
        mailbox.put(changed_frame(sequence=1, now=10.05))
        mailbox.put(blank_frame(sequence=2, now=10.1))
        mailbox.put(blank_frame(sequence=3, now=10.15))
        self.assertEqual([1], [item.sequence for item in mailbox.take(10.2)])
        self.assertEqual([3], [item.sequence for item in mailbox.take(10.2)])
        self.assertEqual(2, mailbox.snapshot()["cam1"]["overload_discarded_frames"])

    def test_burst_to_baseline_reduces_queue_without_exceeding_capacity(self):
        mailbox = self.mailbox(pending_capacity=2, hold_seconds=0.2)
        mailbox.put(blank_frame())
        mailbox.take(10)
        mailbox.put(changed_frame(sequence=1, now=10.05))
        mailbox.put(changed_frame(sequence=2, now=10.1))
        mailbox.put(changed_frame(sequence=3, now=10.3))
        row = mailbox.snapshot(10.3)["cam1"]
        self.assertFalse(row["sampling_burst_active"])
        self.assertEqual(1, row["queued_frames"])
        self.assertEqual(2, row["overload_discarded_frames"])
        self.assertEqual([3], [item.sequence for item in mailbox.take(10.3)])

    def test_burst_configuration_is_explicit_finite_and_bounded(self):
        for settings in (
            {"fps": True},
            {"fps": 31},
            {"fps": float("nan")},
            {"hold_seconds": 11},
            {"hold_seconds": float("inf")},
            {"pixel_delta": 0},
            {"changed_fraction": 0},
            {"changed_fraction": "0.1"},
            {"pending_capacity": True},
            {"pending_capacity": 0},
            {"pending_capacity": 5},
            {"pending_capacity": 2.0},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                MotionBurstSettings(**settings)
        for baseline in (None, 20, 25):
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                FrameMailbox(("cam1",), sample_fps=baseline, motion_burst=MotionBurstSettings())
        with self.assertRaises(ValueError):
            FrameMailbox(("cam1",), sample_fps=5, motion_burst=True)

    def test_empty_and_complete_batches_do_not_wait(self):
        mailbox = FrameMailbox(("cam1", "cam2"), batch_coalesce_seconds=0.01)
        with patch.object(mailbox.condition, "wait") as waiting:
            mailbox.wait_for_batch()
            for camera in ("cam1", "cam2"):
                mailbox.put(blank_frame(camera=camera))
            mailbox.wait_for_batch()
            waiting.assert_not_called()

    def test_batch_wait_releases_lock_and_groups_staggered_camera(self):
        mailbox = FrameMailbox(("cam1", "cam2"), batch_coalesce_seconds=0.02)
        mailbox.put(blank_frame())

        def feed():
            time.sleep(0.005)
            mailbox.put(blank_frame(camera="cam2"))

        source = threading.Thread(target=feed)
        source.start()
        mailbox.wait_for_batch()
        selected = mailbox.take(10)
        source.join(timeout=1)
        self.assertFalse(source.is_alive())
        self.assertEqual(["cam1", "cam2"], [item.camera_id for item in selected])

    def test_missing_camera_wait_is_bounded(self):
        mailbox = FrameMailbox(("cam1", "cam2"), batch_coalesce_seconds=0.01)
        mailbox.put(blank_frame())
        started = time.monotonic()
        mailbox.wait_for_batch()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(["cam1"], [item.camera_id for item in mailbox.take(10)])

    def test_batch_window_rejects_nonfinite_and_unbounded_values(self):
        for window in (True, "0.01", -0.1, 0.03, float("nan"), float("inf")):
            with self.subTest(window=window), self.assertRaises(ValueError):
                FrameMailbox(("cam1",), batch_coalesce_seconds=window)

    def test_phase_sweep_is_opportunity_not_model_accuracy(self):
        report = phase_sweep()
        self.assertEqual("synthetic_sampling_opportunity_not_detection_accuracy", report["scope"])
        rows = {(row["duration_ms"], row["mode"]): row for row in report["rows"]}
        for duration in (50, 100, 200, 500):
            self.assertEqual(40, rows[duration, "full_20fps"]["at_least_one_admitted_frame"])
            self.assertEqual(40, rows[duration, "motion_5_to_20fps"]["at_least_one_admitted_frame"])
        self.assertEqual(10, rows[50, "fixed_5fps"]["at_least_one_admitted_frame"])
        self.assertEqual(20, rows[100, "fixed_5fps"]["at_least_one_admitted_frame"])
        self.assertEqual(0, rows[50, "motion_5_to_20fps"]["at_least_two_admitted_frames"])
        self.assertEqual(40, rows[100, "motion_5_to_20fps"]["at_least_two_admitted_frames"])


if __name__ == "__main__":
    unittest.main()
