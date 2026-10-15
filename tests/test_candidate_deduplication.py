import unittest
from dataclasses import replace

from baby_monitor.candidate_deduplication import deduplicate_frame_candidates
from baby_monitor.domain import Detection


class CandidateDeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.a = Detection(0, "person", 0.8, (0.1, 0.1, 0.6, 0.9))

    def test_exact_duplicate_keeps_stronger_original_and_trace(self):
        weak = replace(self.a, confidence=0.5)
        retained, trace = deduplicate_frame_candidates((weak, self.a))
        self.assertEqual(((self.a,), ((0, 1),)), (retained, trace))
        self.assertIs(self.a, retained[0])

    def test_equal_scores_stably_keep_first(self):
        self.assertEqual(((self.a,), ((1, 0),)), deduplicate_frame_candidates((self.a, self.a)))

    def test_adjacent_or_moderately_overlapping_people_are_retained(self):
        other = replace(self.a, bbox_xyxy=(0.15, 0.1, 0.65, 0.9))
        self.assertEqual(((self.a, other), ()), deduplicate_frame_candidates((self.a, other)))

    def test_different_labels_are_not_merged(self):
        head = replace(self.a, label="head")
        self.assertEqual(((self.a, head), ()), deduplicate_frame_candidates((self.a, head)))

    def test_suppressed_box_does_not_bridge_distinct_retained_boxes(self):
        a = Detection(0, "person", 0.9, (0, 0, 0.5, 1))
        b = replace(a, confidence=0.8, bbox_xyxy=(0.01, 0, 0.51, 1))
        c = replace(a, confidence=0.7, bbox_xyxy=(0.02, 0, 0.52, 1))
        self.assertEqual(((a, c), ((1, 0),)), deduplicate_frame_candidates((a, b, c)))

    def test_bounds_and_mixed_timestamps_rejected(self):
        for value in (0.89, 1.01, True, float("nan"), float("inf"), "0.95"):
            with self.assertRaises(ValueError):
                deduplicate_frame_candidates((self.a,), minimum_iou=value)
        for items in (
            [self.a],
            (object(),),
            (self.a,) * 101,
            (self.a, replace(self.a, timestamp_seconds=1)),
        ):
            with self.assertRaises(ValueError):
                deduplicate_frame_candidates(items)

    def test_empty_and_singleton_do_not_change(self):
        self.assertEqual(((), ()), deduplicate_frame_candidates(()))
        self.assertEqual(((self.a,), ()), deduplicate_frame_candidates((self.a,)))


if __name__ == "__main__":
    unittest.main()
