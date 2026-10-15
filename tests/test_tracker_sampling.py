from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from baby_monitor.object_tracker import DetectionFrame, DetectionFrameBatch, FrameDetection
from baby_monitor.tracker_evaluation.review import TrackerAssociationReview
from baby_monitor.tracker_evaluation.sampling import (
    SCENARIO_NAMES,
    build_tracker_sampling_scenarios,
)


class TrackerSamplingTests(unittest.TestCase):
    def inputs(self):
        base = datetime(2026, 10, 10, tzinfo=UTC)
        batch = DetectionFrameBatch(
            "synthetic_cam",
            "synthetic_detector",
            tuple(
                DetectionFrame(
                    "f_" + str(i),
                    base + timedelta(microseconds=i * 40000),
                    ()
                    if i == 10
                    else (FrameDetection("d_0", "person", 0.9, (0.1, 0.1, 0.2, 0.7)),),
                )
                for i in range(126)
            ),
        )
        return batch, TrackerAssociationReview.pending(batch)

    def test_six_scenarios_preserve_source_and_pending_review_status(self):
        batch, review = self.inputs()
        before = (batch.fingerprint, review.fingerprint)
        scenarios = build_tracker_sampling_scenarios(batch, review)
        self.assertEqual(SCENARIO_NAMES, tuple(s.name for s in scenarios))
        for scenario in scenarios:
            scenario.bind_source(batch, review)
            self.assertFalse(scenario.review.human_labels)
            self.assertEqual("pending", scenario.review.review_status)
            self.assertFalse(scenario.to_dict()["detection_accuracy_evaluated"])
            self.assertFalse(scenario.to_dict()["actual_camera_fault_injected"])
        self.assertEqual(before, (batch.fingerprint, review.fingerprint))

    def test_original_all_frames_and_empty_calls_preserved(self):
        batch, review = self.inputs()
        original = build_tracker_sampling_scenarios(batch, review)[0]
        self.assertEqual(batch, original.batch)
        self.assertEqual(review, original.review)
        self.assertEqual(tuple(range(126)), original.selected_source_indices)
        self.assertEqual((), original.batch.frames[10].detections)

    def test_fixed_deadlines_use_existing_frames_without_retiming(self):
        batch, review = self.inputs()
        five = build_tracker_sampling_scenarios(batch, review)[1]
        self.assertEqual(tuple(range(0, 126, 5)), five.selected_source_indices)
        self.assertEqual(0.2, five.to_dict()["minimum_selected_gap_seconds"])
        self.assertEqual(0.2, five.to_dict()["maximum_selected_gap_seconds"])
        self.assertEqual((), five.batch.frames[2].detections)
        for index, frame in zip(five.selected_source_indices, five.batch.frames, strict=True):
            self.assertIs(batch.frames[index], frame)

    def test_20fps_deadline_quantization_is_not_claimed_uniform(self):
        batch, review = self.inputs()
        twenty = build_tracker_sampling_scenarios(batch, review)[2]
        self.assertEqual((0, 2, 3, 4, 5, 7, 8), twenty.selected_source_indices[:7])
        self.assertEqual(0.04, twenty.to_dict()["minimum_selected_gap_seconds"])
        self.assertEqual(0.08, twenty.to_dict()["maximum_selected_gap_seconds"])

    def test_adaptive_middle_has_denser_updates_without_duplicates(self):
        batch, review = self.inputs()
        scenario = build_tracker_sampling_scenarios(batch, review)[3]
        indices = scenario.selected_source_indices
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(tuple(sorted(indices)), indices)
        counts = [
            sum(lower <= i * 0.04 < upper for i in indices)
            for lower, upper in ((0, 5 / 3), (5 / 3, 10 / 3), (10 / 3, 6))
        ]
        self.assertGreater(counts[1], counts[0])
        self.assertGreater(counts[1], counts[2])

    def test_sampling_loss_is_explicit_and_does_not_create_empty_frames(self):
        batch, review = self.inputs()
        scenarios = build_tracker_sampling_scenarios(batch, review)
        five, drop = scenarios[1], scenarios[4]
        self.assertEqual(
            tuple(i for n, i in enumerate(five.selected_source_indices) if (n + 1) % 7),
            drop.selected_source_indices,
        )
        self.assertNotIn(30, drop.selected_source_indices)
        self.assertFalse(drop.to_dict()["omitted_inputs_counted_as_missed_detections"])

    def test_gap_preserves_actual_timestamp_difference(self):
        batch, review = self.inputs()
        scenario = build_tracker_sampling_scenarios(batch, review)[5]
        self.assertGreater(scenario.to_dict()["maximum_selected_gap_seconds"], 2)
        self.assertEqual(0, scenario.selected_source_indices[0])
        self.assertEqual(125, scenario.selected_source_indices[-1])
        self.assertTrue(scenario.to_dict()["input_timestamps_preserved"])

    def test_short_batch_and_bad_review_fail_closed(self):
        batch, review = self.inputs()
        short = replace(batch, frames=batch.frames[:20])
        with self.assertRaises(ValueError):
            build_tracker_sampling_scenarios(short, TrackerAssociationReview.pending(short))
        with self.assertRaises(ValueError):
            build_tracker_sampling_scenarios(batch, replace(review, camera_id="other_cam"))

    def test_changed_source_or_selection_cannot_bind(self):
        batch, review = self.inputs()
        scenario = build_tracker_sampling_scenarios(batch, review)[1]
        changed = replace(batch, camera_id="other_cam")
        with self.assertRaises(ValueError):
            scenario.bind_source(changed, TrackerAssociationReview.pending(changed))
        with self.assertRaises(ValueError):
            replace(
                scenario, selected_source_indices=(1, *scenario.selected_source_indices[1:])
            ).bind_source(batch, review)

    def test_invalid_metadata_and_duplicate_indices_fail_closed(self):
        batch, review = self.inputs()
        scenario = build_tracker_sampling_scenarios(batch, review)[1]
        for change in (
            {"name": "secret"},
            {"source_batch_sha256": "bad"},
            {"source_frame_count": True},
            {"source_detection_count": 0},
            {"selected_source_indices": list(scenario.selected_source_indices)},
            {"selected_source_indices": (0, 0, *scenario.selected_source_indices[2:])},
        ):
            with self.subTest(fields=list(change)), self.assertRaises(ValueError):
                replace(scenario, **change)

    def test_repeat_fingerprints_are_stable(self):
        batch, review = self.inputs()
        first = build_tracker_sampling_scenarios(batch, review)
        second = build_tracker_sampling_scenarios(batch, review)
        self.assertEqual([s.fingerprint for s in first], [s.fingerprint for s in second])

    def test_valid_subset_cannot_be_relabelled_as_another_schedule(self):
        batch, review = self.inputs()
        five = build_tracker_sampling_scenarios(batch, review)[1]
        with self.assertRaisesRegex(ValueError, "schedule mismatch"):
            replace(five, name="publisher_25fps").bind_source(batch, review)

    def test_full_span_adaptive_schedule_is_not_prefix_stable_online(self):
        batch, review = self.inputs()
        short = replace(batch, frames=batch.frames[:101])
        short_review = TrackerAssociationReview.pending(short)
        full_indices = build_tracker_sampling_scenarios(batch, review)[3].selected_source_indices
        short_indices = build_tracker_sampling_scenarios(short, short_review)[
            3
        ].selected_source_indices
        self.assertNotEqual(tuple(i for i in full_indices if i < 101), short_indices)

    def test_non_25fps_source_cannot_claim_publisher_25fps(self):
        batch, review = self.inputs()
        different = replace(
            batch,
            frames=tuple(
                replace(f, occurred_at=batch.frames[0].occurred_at + timedelta(seconds=i * 0.05))
                for i, f in enumerate(batch.frames)
            ),
        )
        with self.assertRaisesRegex(ValueError, "original 25fps"):
            build_tracker_sampling_scenarios(different, TrackerAssociationReview.pending(different))


if __name__ == "__main__":
    unittest.main()
