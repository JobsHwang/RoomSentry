import unittest
from dataclasses import replace

from baby_monitor.live_detection import ReviewCandidate, ReviewMailbox
from baby_monitor.live_monitor import ReviewTaskCycle, _enqueue_review, _take_next_review
from tests.test_live_detection import observation


class SubjectSchedulingTests(unittest.TestCase):
    def test_four_subject_slots_preserve_event_service_under_saturation(self):
        cycle = ReviewTaskCycle(4)
        event, subjects = ReviewMailbox(("cam1",)), ReviewMailbox(("cam1",))
        tasks = []
        for i in range(15):
            obs = observation(sequence=i)
            event.put(ReviewCandidate(obs, "change", 2, 10))
            _enqueue_review(subjects, obs, None, review_task="subjects")
            tasks.append(cycle.take(event, subjects, 11)[0])
        self.assertEqual((["subjects"] * 4 + ["event"]) * 3, tasks)

    def test_two_subject_slots_then_event_with_both_queues_busy(self):
        cycle = ReviewTaskCycle(2)
        event, subjects = ReviewMailbox(("cam1",)), ReviewMailbox(("cam1",))
        tasks = []
        for i in range(9):
            obs = observation(sequence=i)
            event.put(ReviewCandidate(obs, "change", 2, 10))
            _enqueue_review(subjects, obs, None, review_task="subjects")
            tasks.append(cycle.take(event, subjects, 11)[0])
        self.assertEqual(["subjects", "subjects", "event"] * 3, tasks)

    def test_empty_event_queue_does_not_accumulate_subject_credit(self):
        cycle = ReviewTaskCycle(2)
        event, subjects = ReviewMailbox(("cam1",)), ReviewMailbox(("cam1",))
        for i in range(6):
            _enqueue_review(subjects, observation(sequence=i), None, review_task="subjects")
            self.assertEqual("subjects", cycle.take(event, subjects, 11)[0])
        event.put(ReviewCandidate(observation(), "change", 2, 10))
        _enqueue_review(subjects, observation(), None, review_task="subjects")
        self.assertEqual("event", cycle.take(event, subjects, 11)[0])

    def test_invalid_cycle_ratio_is_rejected(self):
        for ratio in (0, 5, True, 2.0, "2"):
            with self.assertRaises(ValueError):
                ReviewTaskCycle(ratio)

    def test_hybrid_alternates_without_consuming_other_task_evidence(self):
        event = ReviewMailbox(("cam1",))
        subjects = ReviewMailbox(("cam1",), max_evidence_age_seconds=10)
        pinned = ReviewCandidate(observation(sequence=1), "change", 2, 10)
        event.put(pinned)
        _enqueue_review(subjects, observation(sequence=99), None, review_task="subjects")
        task, selected, preferred = _take_next_review(event, subjects, "subjects", 11)
        self.assertEqual(
            ("subjects", 99, "event"), (task, selected.observation.frame.sequence, preferred)
        )
        self.assertEqual(1, event.snapshot()["pending_candidates"])
        _enqueue_review(subjects, observation(sequence=100), None, review_task="subjects")
        task, selected, preferred = _take_next_review(event, subjects, preferred, 11)
        self.assertEqual("event", task)
        self.assertIs(pinned, selected)
        self.assertEqual("subjects", preferred)

    def test_hybrid_empty_or_expired_task_does_not_block_other_task(self):
        event = ReviewMailbox(("cam1",))
        subjects = ReviewMailbox(("cam1",), max_evidence_age_seconds=10)
        event.put(ReviewCandidate(observation(now=10), "change", 2, 10))
        _enqueue_review(subjects, observation(now=0), None, review_task="subjects")
        task, selected, _ = _take_next_review(event, subjects, "subjects", 11)
        self.assertEqual("event", task)
        self.assertIsNotNone(selected)
        self.assertEqual(
            ("subjects", None, "subjects"), _take_next_review(event, subjects, "subjects", 11)
        )

    def test_latest_scene_coalesces_without_starving_quiet_camera(self):
        queue = ReviewMailbox(("a", "b", "c", "d"), max_evidence_age_seconds=10)
        for i in range(100):
            for camera in ("a", "b", "c", "d"):
                obs = observation(camera, sequence=i, now=10 + i / 100, count=camera == "a")
                trigger = ReviewCandidate(obs, "change", 2, obs.completed_monotonic)
                _enqueue_review(queue, obs, trigger, review_task="subjects")
        self.assertEqual(4, queue.snapshot()["pending_candidates"])
        self.assertEqual(0, queue.snapshot()["pinned_trigger_frames"])
        taken = []
        for _ in range(4):
            result = queue.take(11)
            taken.append(result.observation.frame.camera_id)
            self.assertEqual(99, result.observation.frame.sequence)
            self.assertEqual("subject_current_scene", result.reason)
            # The busy camera continuously offers fresh work after each service slot.
            obs = observation("a", sequence=100, now=11)
            _enqueue_review(queue, obs, None, review_task="subjects")
        self.assertEqual(["a", "b", "c", "d"], taken)

    def test_subject_mode_refreshes_without_any_detection_trigger(self):
        queue = ReviewMailbox(("cam1",), max_evidence_age_seconds=10)
        for i in range(2):
            obs = observation(sequence=i, now=10 + i)
            _enqueue_review(queue, obs, None, review_task="subjects")
            self.assertEqual(i, queue.take(11).observation.frame.sequence)

    def test_generation_change_and_disconnect_expire_pending_scene(self):
        queue = ReviewMailbox(("cam1",), max_evidence_age_seconds=10)
        old = observation(now=10)
        new = replace(old, frame=replace(old.frame, generation=1, sequence=1))
        for obs in (old, new, old):
            _enqueue_review(queue, obs, None, review_task="subjects")
        self.assertEqual(1, queue.take(11).observation.frame.generation)
        _enqueue_review(queue, new, None, review_task="subjects")
        self.assertIsNone(queue.take(20.001))
        self.assertEqual(1, queue.snapshot()["expired_candidates"])

    def test_event_mode_keeps_original_trigger_and_refresh_semantics(self):
        queue = ReviewMailbox(("cam1",))
        old = observation(sequence=1, now=10, count=1)
        trigger = ReviewCandidate(old, "change", 2, 10)
        _enqueue_review(queue, old, trigger, review_task="event")
        _enqueue_review(queue, observation(sequence=2, now=11), None, review_task="event")
        self.assertIs(trigger, queue.take(11))


if __name__ == "__main__":
    unittest.main()
