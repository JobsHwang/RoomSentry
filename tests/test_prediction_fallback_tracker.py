from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from baby_monitor.motion_tracker import (
    MotionActiveTrack,
    MotionObjectTracker,
    MotionTrackerConfig,
    MotionTrackSample,
    _predict_bbox,
)
from baby_monitor.object_tracker import DetectionFrame, DetectionFrameBatch, FrameDetection
from baby_monitor.tracker_evaluation._shared import TrackerEvaluationError
from baby_monitor.tracker_evaluation.comparison import compare_tracker_predictions
from baby_monitor.tracker_evaluation.prediction import build_motion_tracker_predictions
from baby_monitor.tracker_evaluation.prediction_fallback import (
    FALLBACK_ADAPTER,
    PredictionFallbackTracker,
    _legacy_predict_bbox,
    build_prediction_fallback_predictions,
    predict_with_fallback,
    prediction_fallback_required,
)


class PredictionFallbackTests(unittest.TestCase):
    base = datetime(2026, 10, 12, tzinfo=UTC)
    big = (0.1, 0.1, 0.9, 0.9)
    small = (0.32, 0.32, 0.68, 0.68)

    def frame(self, i, boxes=(), scores=None, labels=None):
        return DetectionFrame(
            f"f{i}",
            self.base + timedelta(seconds=i),
            tuple(
                FrameDetection(
                    f"d{j}",
                    (labels or ["person"] * len(boxes))[j],
                    (scores or [0.9] * len(boxes))[j],
                    box,
                )
                for j, box in enumerate(boxes)
            ),
        )

    def batch(self, frames):
        return DetectionFrameBatch("synthetic", "synthetic_detector", tuple(frames))

    def sample_track(self, before, current):
        return MotionActiveTrack(
            "synthetic_track",
            "person",
            MotionTrackSample(self.base + timedelta(seconds=1), 0.9, current),
            MotionTrackSample(self.base, 0.9, before),
        )

    def test_invalid_extent_uses_whole_latest_box_without_mutating_history(self):
        track = self.sample_track(self.big, self.small)
        before = track.to_dict()
        when = self.base + timedelta(seconds=2)
        self.assertTrue(prediction_fallback_required(track, when, 1))
        self.assertEqual(self.small, predict_with_fallback(track, when, 1))
        self.assertEqual(before, track.to_dict())
        legacy = _legacy_predict_bbox(track, when, 1)
        self.assertLess(legacy[2] - legacy[0], 1e-8)
        self.assertEqual(self.small, _predict_bbox(track, when, 1))

    def test_valid_velocity_zero_horizon_and_single_sample_preserve_original(self):
        track = self.sample_track((0.1, 0.1, 0.3, 0.6), (0.15, 0.1, 0.35, 0.6))
        for when, maximum in (
            (self.base + timedelta(seconds=2), 1),
            (self.base + timedelta(seconds=30), 1),
            (self.base + timedelta(seconds=2), 0),
        ):
            self.assertFalse(prediction_fallback_required(track, when, maximum))
            self.assertEqual(
                _predict_bbox(track, when, maximum), predict_with_fallback(track, when, maximum)
            )
        single = MotionActiveTrack("synthetic_single", "person", track.current)
        self.assertEqual(
            track.current.bbox_xyxy,
            predict_with_fallback(single, self.base + timedelta(seconds=2), 1),
        )

    def test_fixed_core_and_distinct_offline_adapter_recover_shrinking_box_link(self):
        batch = self.batch(
            [self.frame(0, (self.big,)), self.frame(1, (self.small,)), self.frame(2, (self.big,))]
        )
        config = MotionTrackerConfig()
        original = build_motion_tracker_predictions(batch, config)
        candidate = build_prediction_fallback_predictions(batch, config)
        report = compare_tracker_predictions(batch, original, candidate)
        self.assertEqual(0, report.left_only_link_count)
        self.assertEqual(0, report.right_only_link_count)
        self.assertEqual(1, len({a.tracking_id for a in original.assignments}))
        self.assertEqual(1, len({a.tracking_id for a in candidate.assignments}))
        self.assertEqual(FALLBACK_ADAPTER, candidate.tracker_adapter)
        self.assertNotEqual(original.tracker_config_sha256, candidate.tracker_config_sha256)
        self.assertNotEqual(original.tracker_revision, candidate.tracker_revision)

    def test_ordinary_motion_links_match_original_and_global_function_unchanged(self):
        config = MotionTrackerConfig()
        batch = self.batch(
            [self.frame(i, ((0.1 + i * 0.02, 0.1, 0.3 + i * 0.02, 0.8),)) for i in range(4)]
        )
        original = build_motion_tracker_predictions(batch, config)
        candidate = build_prediction_fallback_predictions(batch, config)
        report = compare_tracker_predictions(batch, original, candidate)
        self.assertEqual(0, report.left_only_link_count)
        self.assertEqual(0, report.right_only_link_count)
        self.assertEqual(original, build_motion_tracker_predictions(batch, config))

    def test_streaming_and_causal_prefixes_equal_without_future_id_seed(self):
        config = MotionTrackerConfig()
        batch = self.batch(
            [self.frame(0, (self.big,)), self.frame(1, (self.small,)), self.frame(2, (self.big,))]
        )
        prediction = build_prediction_fallback_predictions(batch, config)
        tracker = PredictionFallbackTracker(
            config=config, camera_id=batch.camera_id, detector_adapter=batch.detector_adapter
        )
        streamed = tuple(a for frame in batch.frames for a in tracker.process(frame))
        self.assertEqual(prediction.assignments, streamed)
        for size in range(1, 4):
            prefix = self.batch(batch.frames[:size])
            current = build_prediction_fallback_predictions(prefix, config)
            self.assertEqual(
                tuple(
                    a
                    for a in prediction.assignments
                    if a.frame_id in {f.frame_id for f in prefix.frames}
                ),
                current.assignments,
            )

    def test_low_confidence_only_recovers_and_never_seeds(self):
        config = MotionTrackerConfig(high_confidence_threshold=0.7, low_confidence_threshold=0.2)
        tracker = PredictionFallbackTracker(
            config=config, camera_id="synthetic", detector_adapter="synthetic"
        )
        initial = tracker.process(self.frame(0, (self.big,), [0.3]))
        self.assertEqual((), tracker.active)
        strong = tracker.process(self.frame(1, (self.big,), [0.9]))
        low = tracker.process(self.frame(2, (self.big,), [0.3]))
        self.assertEqual(strong[0].tracking_id, low[0].tracking_id)
        self.assertNotEqual(initial[0].tracking_id, low[0].tracking_id)

    def test_adapter_id_names_do_not_change_equal_score_assignment_order(self):
        config = MotionTrackerConfig()
        batch = self.batch(
            [
                self.frame(0, (self.big, self.big, self.big)),
                self.frame(1, (self.big, self.big)),
                self.frame(2, (self.big, self.big, self.big)),
            ]
        )
        original = build_motion_tracker_predictions(batch, config)
        candidate = build_prediction_fallback_predictions(batch, config)
        comparison = compare_tracker_predictions(batch, original, candidate)
        self.assertEqual(0, comparison.left_only_link_count)
        self.assertEqual(0, comparison.right_only_link_count)

    def test_invalid_dormant_track_cannot_steal_qualified_recent_edge(self):
        old = self.sample_track(self.big, self.small)
        current = MotionActiveTrack(
            "recent_track", "person", replace(old.current, bbox_xyxy=(0.33, 0.32, 0.69, 0.68))
        )
        for score in (0.9, 0.4):
            config = MotionTrackerConfig(high_confidence_threshold=0.7)
            original = MotionObjectTracker(
                config=config, camera_id="synthetic", detector_adapter="synthetic"
            )
            original.process(self.frame(1, (self.small,)))
            checkpoint = replace(
                original.state,
                active_tracks=tuple(sorted((old, current), key=lambda t: t.tracking_id)),
            )
            tracker = PredictionFallbackTracker(
                config=config,
                camera_id="synthetic",
                detector_adapter="synthetic",
                checkpoint=checkpoint,
            )
            assigned = tracker.process(self.frame(2, (self.small,), [score]))
            self.assertEqual("recent_track", assigned[0].tracking_id)
            self.assertIn("synthetic_track", tracker.last_fallback_ids)

    def test_empty_expiry_labels_and_duplicate_boxes_preserve_constraints(self):
        config = MotionTrackerConfig()
        tracker = PredictionFallbackTracker(
            config=config, camera_id="synthetic", detector_adapter="synthetic"
        )
        first = tracker.process(self.frame(0, (self.big, self.big)))
        self.assertEqual(2, len({a.tracking_id for a in first}))
        self.assertEqual((), tracker.process(self.frame(1)))
        changed = tracker.process(self.frame(2, (self.big,), labels=["other"]))
        self.assertNotIn(changed[0].tracking_id, {a.tracking_id for a in first})
        expired = tracker.process(self.frame(6, (self.big,)))
        self.assertNotIn(expired[0].tracking_id, {a.tracking_id for a in first})

    def test_failure_does_not_commit_candidate_state(self):
        tracker = PredictionFallbackTracker(
            config=MotionTrackerConfig(max_active_tracks=1),
            camera_id="synthetic",
            detector_adapter="synthetic",
        )
        tracker.process(self.frame(0, (self.big,)))
        before = (tracker.active, tracker.last_frame_at, tracker.last_fallback_ids)
        for frame in (self.frame(0, (self.big,)), self.frame(1, (self.big, self.big))):
            with self.assertRaises(TrackerEvaluationError):
                tracker.process(frame)
            self.assertEqual(
                before, (tracker.active, tracker.last_frame_at, tracker.last_fallback_ids)
            )


if __name__ == "__main__":
    unittest.main()
