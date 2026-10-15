"""Synthetic receipt/clock regressions; no devices, models or household media."""

import math
import threading
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from baby_monitor import live_detection
from baby_monitor.live_detection import FrameMailbox, ReviewCandidate, ReviewMailbox
from baby_monitor.live_monitor import ReviewTaskCycle
from tests.test_live_detection import frame, observation


class QueueClockFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.base = datetime(2026, 1, 1, tzinfo=UTC)

    def source(self, camera="cam1", sequence=0, *, elapsed=100, utc_age=0):
        return replace(
            frame(camera, sequence, now=elapsed), received_at=self.base - timedelta(seconds=utc_age)
        )

    def candidate(self, camera="cam1", sequence=0, *, elapsed=100, utc_age=0, priority=2):
        source = self.source(camera, sequence, elapsed=elapsed, utc_age=utc_age)
        obs = replace(observation(camera, sequence), frame=source, completed_monotonic=elapsed)
        return ReviewCandidate(obs, "synthetic_change", priority, elapsed)

    def test_frame_queue_expires_utc_old_frame_despite_young_elapsed_age(self):
        queue = FrameMailbox(("cam1",))
        queue.put(self.source(utc_age=2))
        self.assertEqual((), queue.take(100.5, now_utc=self.base))
        status = queue.snapshot()["cam1"]
        self.assertEqual(1, status["stale_discarded_frames"])
        self.assertEqual(0, status["receipt_clock_discarded_frames"])
        self.assertEqual(0, status["dispatched_frames"])

    def test_latest_subject_queue_expires_utc_old_frame_without_consuming_event(self):
        event, subjects = (
            ReviewMailbox(("cam1",)),
            ReviewMailbox(("cam1",), max_evidence_age_seconds=10),
        )
        event.put(self.candidate(sequence=1, utc_age=11))
        subjects.put(self.candidate(sequence=2, utc_age=11, priority=0))
        task, selected = ReviewTaskCycle(4).take(event, subjects, 102, now_utc=self.base)
        self.assertEqual("event", task)
        self.assertEqual(1, selected.observation.frame.sequence)
        self.assertEqual(1, subjects.snapshot()["expired_candidates"])
        self.assertEqual(0, event.snapshot()["expired_candidates"])

    def test_historical_event_queue_keeps_120_second_limit_not_subject_window(self):
        for utc_age, allowed in ((30, True), (120, True), (120.000001, False)):
            queue = ReviewMailbox(("cam1",))
            candidate = self.candidate(utc_age=utc_age)
            queue.put(candidate)
            self.assertEqual(allowed, queue.take(101, now_utc=self.base) is candidate)

    def test_elapsed_clock_still_expires_after_wall_rollback(self):
        queue = ReviewMailbox(("cam1",), max_evidence_age_seconds=10)
        queue.put(self.candidate(utc_age=2))
        self.assertIsNone(queue.take(111, now_utc=self.base))
        self.assertEqual(1, queue.snapshot()["expired_candidates"])

    def test_future_receipt_and_invalid_clocks_fail_closed_and_are_counted(self):
        for elapsed, utc_age, now, now_utc in (
            (101, 0, 100, self.base),
            (100, -1, 100, self.base),
            (100, 0, True, self.base),
            (100, 0, math.nan, self.base),
            (100, 0, math.inf, self.base),
            (100, 0, 100, datetime(2026, 1, 1)),
            (100, 0, 100, "invalid"),
            (100, 0, 10**400, self.base),
        ):
            with self.subTest(elapsed=elapsed, utc_age=utc_age, now_type=type(now).__name__):
                frames, reviews = FrameMailbox(("cam1",)), ReviewMailbox(("cam1",))
                frames.put(self.source(elapsed=elapsed, utc_age=utc_age))
                reviews.put(self.candidate(elapsed=elapsed, utc_age=utc_age))
                self.assertEqual((), frames.take(now, now_utc=now_utc))
                self.assertIsNone(reviews.take(now, now_utc=now_utc))
                self.assertEqual(1, frames.snapshot()["cam1"]["receipt_clock_discarded_frames"])
                self.assertEqual(1, reviews.snapshot()["receipt_clock_discarded_candidates"])

    def test_frame_drop_preserves_fresh_successor_and_accounting(self):
        queue = FrameMailbox(("cam1",), capacity=2)
        queue.put(self.source(sequence=1, utc_age=2))
        current = self.source(sequence=2)
        queue.put(current)
        self.assertEqual((current,), queue.take(100.5, now_utc=self.base))
        status = queue.snapshot()["cam1"]
        self.assertEqual(2, status["decoded_frames"])
        self.assertEqual(2, status["eligible_frames"])
        self.assertEqual(1, status["discarded_frames"])
        self.assertEqual(1, status["dispatched_frames"])
        self.assertEqual(0, status["queued_frames"])

    def test_nonfinite_or_negative_elapsed_reader_rejects_without_utc(self):
        for now in (-1, math.nan, math.inf):
            frames, reviews = FrameMailbox(("cam1",)), ReviewMailbox(("cam1",))
            frames.put(self.source())
            reviews.put(self.candidate())
            self.assertEqual((), frames.take(now))
            self.assertIsNone(reviews.take(now))

    def test_fresh_four_cameras_keep_fair_order(self):
        cameras = ("a", "b", "c", "d")
        queue = ReviewMailbox(cameras, max_evidence_age_seconds=10)
        for camera in cameras:
            queue.put(self.candidate(camera, priority=0))
        self.assertEqual(
            list(cameras),
            [queue.take(101, now_utc=self.base).observation.frame.camera_id for _ in cameras],
        )

    def test_live_queue_clock_sampling_holds_admission_lock(self):
        for queue in (FrameMailbox(("cam1",)), ReviewMailbox(("cam1",))):
            acquired = []

            def clock(queue=queue, acquired=acquired):
                def probe():
                    got = queue.condition.acquire(blocking=False)
                    acquired.append(got)
                    if got:
                        queue.condition.release()

                thread = threading.Thread(target=probe)
                thread.start()
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive())
                self.assertEqual([False], acquired)
                return 100.5

            if isinstance(queue, FrameMailbox):
                queue.put(self.source())
            else:
                queue.put(self.candidate())
            with (
                patch.object(live_detection.time, "monotonic", side_effect=clock),
                patch.object(live_detection, "datetime") as date,
            ):
                date.now.return_value = self.base
                self.assertTrue(queue.take_live())
                date.now.assert_called_once_with(UTC)

    def test_real_task_cycle_uses_live_guard_and_falls_back_to_current_subjects(self):
        event, subjects = (
            ReviewMailbox(("cam1",)),
            ReviewMailbox(("cam1",), max_evidence_age_seconds=10),
        )
        event.put(self.candidate(sequence=1, utc_age=121))
        subjects.put(self.candidate(sequence=2, utc_age=1, priority=0))
        cycle = ReviewTaskCycle(4)
        cycle.remaining = 0
        with (
            patch.object(live_detection.time, "monotonic", return_value=100.5),
            patch.object(live_detection, "datetime") as date,
        ):
            date.now.return_value = self.base
            task, selected = cycle.take(event, subjects)
        self.assertEqual("subjects", task)
        self.assertEqual(2, selected.observation.frame.sequence)
        self.assertEqual(1, event.snapshot()["expired_candidates"])
        self.assertEqual(0, subjects.snapshot()["expired_candidates"])


if __name__ == "__main__":
    unittest.main()
