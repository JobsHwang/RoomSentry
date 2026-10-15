from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from baby_monitor.domain import CandidateEvent, Detection, SignalEvent, SignalType
from baby_monitor.local_perception import (
    build_detection_frame_batch,
    run_local_perception,
)
from baby_monitor.object_tracker import IoUTrackerConfig
from baby_monitor.pipeline import bind_signals_to_candidates
from baby_monitor.temporal_zone import TemporalZoneConfig, ZoneDefinition


class LocalPerceptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.started_at = datetime(2026, 8, 3, 0, 0, tzinfo=UTC)
        self.detections = (
            Detection(1.0, "person", 0.8, (0.2, 0.2, 0.4, 0.8)),
            Detection(2.0, "person", 0.82, (0.22, 0.2, 0.42, 0.8)),
        )
        self.tracker_config = IoUTrackerConfig(
            iou_threshold=0.3,
            max_track_age_seconds=2.0,
            max_active_tracks=16,
            max_processed_frames=100,
            max_pending_observations=100,
        )
        self.zone_config = TemporalZoneConfig(
            track_expiry_seconds=30.0,
            max_processed_observations=100,
            zones=(
                ZoneDefinition(
                    zone_id="synthetic_zone",
                    labels=("person",),
                    bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
                    dwell_seconds=0.5,
                    exit_margin=0.0,
                ),
            ),
        )

    def test_detection_facts_form_deterministic_path_free_frames(self) -> None:
        first = build_detection_frame_batch(
            self.detections,
            camera_id="cam_01",
            detector_adapter="synthetic_detector_v1",
            source_started_at=self.started_at,
        )
        second = build_detection_frame_batch(
            self.detections,
            camera_id="cam_01",
            detector_adapter="synthetic_detector_v1",
            source_started_at=self.started_at,
        )

        self.assertEqual(first, second)
        self.assertEqual(2, len(first.frames))
        self.assertEqual(
            [self.started_at + timedelta(seconds=1), self.started_at + timedelta(seconds=2)],
            [frame.occurred_at for frame in first.frames],
        )
        self.assertTrue(all(frame.frame_id.startswith("frame_") for frame in first.frames))
        self.assertTrue(
            all(
                detection.detection_id.startswith("detection_")
                for frame in first.frames
                for detection in frame.detections
            )
        )
        self.assertNotIn("/", str(first.to_dict()))

    def test_tracker_and_zone_emit_anonymous_system_facts(self) -> None:
        result = run_local_perception(
            self.detections,
            camera_id="cam_01",
            detector_adapter="synthetic_detector_v1",
            source_started_at=self.started_at,
            tracker_config=self.tracker_config,
            zone_config=self.zone_config,
        )

        self.assertEqual(2, len(result.observations))
        self.assertEqual(1, result.active_track_count)
        self.assertEqual(
            [SignalType.ZONE_ENTER, SignalType.ZONE_DWELL],
            [signal.signal_type for signal in result.signals],
        )
        tracking_ids = {observation.tracking_id for observation in result.observations}
        self.assertEqual(1, len(tracking_ids))
        self.assertTrue(next(iter(tracking_ids)).startswith("track_"))
        self.assertTrue(
            all(signal.source_adapter == "temporal_zone_v1" for signal in result.signals)
        )
        self.assertTrue(
            all(dict(signal.attributes)["object_label"] == "person" for signal in result.signals)
        )

    def test_signals_bind_once_to_nearest_candidate_and_unmatched_fails(self) -> None:
        candidates = (
            CandidateEvent("event_a", "cam_01", 0.0, 3.0, self.detections[:1]),
            CandidateEvent("event_b", "cam_01", 2.0, 6.0, self.detections[1:]),
        )
        signal = SignalEvent(
            signal_id="signal_01",
            camera_id="cam_01",
            occurred_at=self.started_at + timedelta(seconds=2.75),
            signal_type=SignalType.ZONE_ENTER,
            source_adapter="temporal_zone_v1",
            confidence=0.8,
            attributes=(
                ("object_label", "person"),
                ("tracking_id", "track_fixture"),
                ("zone_id", "synthetic_zone"),
            ),
        )

        bound = bind_signals_to_candidates(
            candidates,
            (signal,),
            source_started_at=self.started_at,
        )
        self.assertEqual(1, sum(len(candidate.signals) for candidate in bound))
        self.assertEqual((signal,), bound[0].signals)
        self.assertEqual((), bound[1].signals)

        unmatched = SignalEvent(
            signal_id="signal_02",
            camera_id="cam_01",
            occurred_at=self.started_at + timedelta(seconds=10),
            signal_type=SignalType.ZONE_EXIT,
            source_adapter="temporal_zone_v1",
            confidence=0.8,
            attributes=(
                ("object_label", "person"),
                ("tracking_id", "track_fixture"),
                ("zone_id", "synthetic_zone"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "outside every candidate"):
            bind_signals_to_candidates(
                candidates,
                (unmatched,),
                source_started_at=self.started_at,
            )


if __name__ == "__main__":
    unittest.main()
