from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from baby_monitor.motion_tracker import (
    MotionActiveTrack,
    MotionTrackerConfig,
    MotionTrackSample,
    _match,
)
from baby_monitor.object_tracker import DetectionFrame, DetectionFrameBatch, FrameDetection
from baby_monitor.tracker_evaluation import (
    TrackerEvaluationError,
    build_motion_tracker_predictions,
    compare_tracker_predictions,
)
from baby_monitor.tracker_evaluation.recency import (
    _cascade,
    build_recency_tracker_predictions,
)


class RecencyTrackerTests(unittest.TestCase):
    base = datetime(2026, 10, 12, tzinfo=UTC)

    def detection(self, identifier="d0", x=0.1, *, confidence=0.9, label="person"):
        return FrameDetection(identifier, label, confidence, (x, 0.2, x + 0.2, 0.8))

    def frame(self, index, seconds, *detections):
        return DetectionFrame(
            f"f{index}", self.base + timedelta(seconds=seconds), tuple(detections)
        )

    def batch(self, *frames):
        return DetectionFrameBatch("synthetic_camera", "synthetic_detector", tuple(frames))

    def test_recent_track_has_priority_over_higher_scoring_dormant_track(self):
        recent = MotionActiveTrack(
            "track_recent", "person", MotionTrackSample(self.base, 0.9, (0.1, 0.2, 0.3, 0.8))
        )
        dormant = MotionActiveTrack(
            "track_dormant",
            "person",
            MotionTrackSample(self.base - timedelta(seconds=0.1), 0.9, (0.12, 0.2, 0.32, 0.8)),
        )
        kwargs = dict(
            occurred_at=self.base + timedelta(seconds=0.1),
            minimum_score=0.3,
            config=MotionTrackerConfig(),
        )
        tracks, detections = (recent, dormant), (self.detection(x=0.12),)
        self.assertEqual(((1, 0),), _match(tracks, detections, **kwargs))
        self.assertEqual(
            ((0, 0),), _cascade(tracks, detections, previous_input_at=self.base, **kwargs)
        )
        # This can ALSO be a genuinely reappearing different person. Neither
        # objective establishes identity, so this is never a winner assertion.

    def test_priority_does_not_bypass_score_or_label_gates(self):
        tracks = tuple(
            MotionActiveTrack(identifier, label, MotionTrackSample(stamp, 0.9, box))
            for identifier, label, stamp, box in (
                ("track_recent", "person", self.base, (0.7, 0.2, 0.9, 0.8)),
                (
                    "track_dormant",
                    "person",
                    self.base - timedelta(seconds=0.1),
                    (0.1, 0.2, 0.3, 0.8),
                ),
                ("track_other_label", "car", self.base, (0.1, 0.2, 0.3, 0.8)),
            )
        )
        self.assertEqual(
            ((1, 0),),
            _cascade(
                tracks,
                (self.detection(),),
                previous_input_at=self.base,
                occurred_at=self.base + timedelta(seconds=0.1),
                minimum_score=0.3,
                config=MotionTrackerConfig(),
            ),
        )

    def test_empty_input_advances_recency_without_emitting_predicted_people(self):
        batch = self.batch(
            self.frame(0, 0, self.detection()),
            self.frame(1, 0.1),
            self.frame(2, 0.2, self.detection()),
            self.frame(3, 3, self.detection()),
        )
        result = build_recency_tracker_predictions(batch, MotionTrackerConfig())
        self.assertEqual(["f0", "f2", "f3"], [a.frame_id for a in result.assignments])
        self.assertEqual(result.assignments[0].tracking_id, result.assignments[1].tracking_id)
        self.assertNotEqual(result.assignments[1].tracking_id, result.assignments[2].tracking_id)
        self.assertFalse(result.subject_identity_inference)
        result.bind(batch)

    def test_confidence_stages_and_nonretained_low_or_ignored_inputs(self):
        batch = self.batch(
            self.frame(0, 0, self.detection()),
            self.frame(1, 0.1, self.detection(confidence=0.4)),
            self.frame(2, 0.2, self.detection("weak", 0.7, confidence=0.4)),
            self.frame(3, 0.3, self.detection("weak", 0.7, confidence=0.4)),
            self.frame(4, 0.4, self.detection("ignored", confidence=0.2)),
            self.frame(5, 0.5, self.detection("ignored", confidence=0.2)),
        )
        result = build_recency_tracker_predictions(batch, MotionTrackerConfig())
        ids = [a.tracking_id for a in result.assignments]
        self.assertEqual(ids[0], ids[1])
        self.assertNotEqual(ids[2], ids[3])
        self.assertNotEqual(ids[4], ids[5])
        self.assertEqual(6, len(ids))

    def test_causal_prefix_determinism_and_input_binding(self):
        frames = tuple(self.frame(i, i / 10, self.detection(x=0.1 + i / 100)) for i in range(4))
        config = MotionTrackerConfig()
        prefix = build_recency_tracker_predictions(self.batch(*frames[:2]), config)
        full = build_recency_tracker_predictions(self.batch(*frames), config)
        self.assertEqual(prefix.assignments, full.assignments[:2])
        self.assertEqual(full, build_recency_tracker_predictions(self.batch(*frames), config))
        self.assertNotEqual(prefix.detection_batch_sha256, full.detection_batch_sha256)
        self.assertNotEqual(config.fingerprint, full.tracker_config_sha256)
        with self.assertRaisesRegex(TrackerEvaluationError, "different detection batch"):
            full.bind(self.batch(*frames[:2]))

    def test_output_is_ordered_one_to_one_and_cannot_claim_a_winner(self):
        batch = self.batch(
            *(
                self.frame(i, i / 10, self.detection("a", 0.1), self.detection("b", 0.7))
                for i in range(4)
            )
        )
        config = MotionTrackerConfig()
        result = build_recency_tracker_predictions(batch, config)
        self.assertEqual(
            [(f.frame_id, d.detection_id) for f in batch.frames for d in f.detections],
            [(a.frame_id, a.detection_id) for a in result.assignments],
        )
        comparison = compare_tracker_predictions(
            batch, build_motion_tracker_predictions(batch, config), result
        )
        self.assertEqual(6, comparison.shared_link_count)
        self.assertFalse(comparison.better_adapter_selected)
        self.assertFalse(comparison.accuracy_evaluated)
        self.assertFalse(comparison.human_ground_truth_used)

    def test_configured_limits_and_invalid_arguments_fail_closed(self):
        frame = self.frame(0, 0, self.detection("a", 0.1), self.detection("b", 0.7))
        for config, message in (
            (MotionTrackerConfig(max_frame_detections=1), "detection limit"),
            (MotionTrackerConfig(max_active_tracks=1), "active track limit"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(TrackerEvaluationError, message),
            ):
                build_recency_tracker_predictions(self.batch(frame), config)
        for batch, config in ((None, MotionTrackerConfig()), (self.batch(frame), None)):
            with self.assertRaisesRegex(TrackerEvaluationError, "invalid"):
                build_recency_tracker_predictions(batch, config)


if __name__ == "__main__":
    unittest.main()
