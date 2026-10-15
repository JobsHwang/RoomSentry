from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from baby_monitor.live_detection import ReviewCandidate, ReviewMailbox, ReviewTriggerPolicy
from baby_monitor.live_monitor import _review_storage_available
from tests.test_live_detection import observation


class LiveReviewEvidenceTests(unittest.TestCase):
    def test_person_appears_then_leaves_while_bonsai_busy_preserves_appearance(self):
        policy = ReviewTriggerPolicy()
        queue = ReviewMailbox(("cam1",))
        queue.put(policy.observe(observation(now=10)))
        queue.take(10)  # Bonsai starts baseline; subsequent frames arrive while it is busy.
        for sequence, now, count in (
            (1, 10.2, 1),
            (2, 10.4, 1),
            (3, 10.6, 0),
            (4, 10.8, 0),
            (5, 11, 0),
        ):
            current = observation(sequence=sequence, now=now, count=count)
            trigger = policy.observe(current)
            if trigger:
                queue.put(trigger)
            else:
                queue.refresh(current)
        appeared = queue.take(30)
        departed = queue.take(31)
        self.assertEqual(2, appeared.observation.frame.sequence)
        self.assertEqual(1, len(appeared.observation.detections))
        self.assertEqual("trigger_frame", appeared.evidence_role)
        self.assertEqual(4, departed.observation.frame.sequence)
        self.assertEqual(0, len(departed.observation.detections))
        self.assertIsNone(queue.take(32))

    def test_first_and_latest_trigger_are_bounded_and_frame_bound(self):
        queue = ReviewMailbox(("cam1",))
        original = observation(sequence=1, count=1)
        queue.put(ReviewCandidate(original, "person_count_changed", 2, 10))
        for sequence in range(2, 100):
            queue.put(
                ReviewCandidate(
                    observation(sequence=sequence, now=10 + sequence / 10),
                    "person_position_changed",
                    2,
                    10 + sequence / 10,
                )
            )
        queue.refresh(observation(sequence=100, now=20, count=0))
        self.assertEqual(2, queue.snapshot()["pending_candidates"])
        self.assertEqual(97, queue.snapshot()["coalesced_candidates"])
        first = queue.take(21)
        self.assertIs(original, first.observation)
        self.assertEqual("person_count_changed", first.reason)
        self.assertEqual(99, queue.take(22).observation.frame.sequence)

    def test_patrol_cannot_overwrite_pending_trigger_evidence(self):
        queue = ReviewMailbox(("cam1",))
        for sequence in (1, 2):
            queue.put(ReviewCandidate(observation(sequence=sequence), "change", 2, 10))
        queue.put(ReviewCandidate(observation(sequence=3), "patrol", 0, 11))
        queue.refresh(observation(sequence=4))
        self.assertEqual([1, 2], [queue.take().observation.frame.sequence for _ in range(2)])

    def test_old_frame_expires_by_capture_age_even_if_recently_queued(self):
        queue = ReviewMailbox(("cam1",), max_evidence_age_seconds=10)
        queue.put(ReviewCandidate(observation(now=0), "change", 2, 9))
        queue.put(ReviewCandidate(observation(sequence=1, now=9), "change", 2, 9))
        self.assertEqual(1, queue.take(11).observation.frame.sequence)
        self.assertEqual(1, queue.snapshot()["expired_candidates"])

    def test_reconnected_generation_invalidates_old_pending_evidence(self):
        queue = ReviewMailbox(("cam1",))
        old = observation()
        queue.put(ReviewCandidate(old, "change", 2, 10))
        fresh = replace(old, frame=replace(old.frame, sequence=1, generation=1))
        queue.put(ReviewCandidate(fresh, "initial_or_reconnected_scene", 0, 11))
        queue.refresh(old)
        queue.put(ReviewCandidate(old, "change", 2, 12))
        self.assertEqual(1, queue.take().observation.frame.generation)
        self.assertIsNone(queue.take())
        self.assertEqual(1, queue.snapshot()["generation_discarded_candidates"])

    def test_busy_four_cameras_receive_fair_first_review_with_two_slots_each(self):
        queue = ReviewMailbox(("a", "b", "c", "d"))
        for camera in ("a", "b", "c", "d"):
            for sequence in (1, 2):
                queue.put(ReviewCandidate(observation(camera, sequence=sequence), "change", 2, 10))
        selected = [queue.take().observation.frame.camera_id for _ in range(4)]
        self.assertEqual({"a", "b", "c", "d"}, set(selected))

    def test_invalid_evidence_age_rejected(self):
        for value in (True, "120", 0, 601, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReviewMailbox(("cam1",), max_evidence_age_seconds=value)

    def test_storage_scan_tolerates_disappearing_atomic_temporary_file(self):
        disappearing = Mock()
        disappearing.stat.side_effect = FileNotFoundError
        remaining = Mock()
        remaining.stat.return_value = SimpleNamespace(st_mode=0o100600, st_size=100)
        iterator = MagicMock()
        iterator.__enter__.return_value = iter((disappearing, remaining))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("baby_monitor.live_monitor.os.scandir", return_value=iterator),
            patch(
                "baby_monitor.live_monitor.shutil.disk_usage",
                return_value=SimpleNamespace(free=10 * 1024**3),
            ),
        ):
            self.assertTrue(_review_storage_available(Path(directory), 64 * 1024**2))

    def test_storage_scan_defers_on_permission_errors_or_symlink(self):
        for metadata in (PermissionError(), SimpleNamespace(st_mode=0o120777, st_size=0)):
            entry = Mock()
            if isinstance(metadata, Exception):
                entry.stat.side_effect = metadata
            else:
                entry.stat.return_value = metadata
            iterator = MagicMock()
            iterator.__enter__.return_value = iter((entry,))
            with (
                tempfile.TemporaryDirectory() as directory,
                patch("baby_monitor.live_monitor.os.scandir", return_value=iterator),
            ):
                self.assertFalse(_review_storage_available(Path(directory), 64 * 1024**2))


if __name__ == "__main__":
    unittest.main()
