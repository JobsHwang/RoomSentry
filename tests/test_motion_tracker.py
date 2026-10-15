from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.cli import main
from baby_monitor.motion_tracker import (
    MOTION_TRACKER_ADAPTER_ID,
    MotionActiveTrack,
    MotionObjectTracker,
    MotionTrackerConfig,
    MotionTrackerError,
    MotionTrackerState,
    MotionTrackSample,
    _maximum_weight_assignment,
    build_motion_track_assignments,
)
from baby_monitor.object_tracker import (
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
    IoUTrackerConfig,
)
from baby_monitor.tracker_evaluation import (
    TrackerPredictionBatch,
    build_iou_tracker_predictions,
    build_motion_tracker_predictions,
)


class MotionTrackerTests(unittest.TestCase):
    BASE = datetime(2026, 8, 8, 2, 3, 4, tzinfo=UTC)

    @staticmethod
    def _detection(
        detection_id: str,
        x1: float,
        x2: float,
        *,
        confidence: float = 0.9,
        label: str = "person",
    ) -> FrameDetection:
        return FrameDetection(
            detection_id=detection_id,
            label=label,
            confidence=confidence,
            bbox_xyxy=(x1, 0.2, x2, 0.8),
        )

    @classmethod
    def _frame(
        cls,
        index: int,
        *detections: FrameDetection,
    ) -> DetectionFrame:
        return DetectionFrame(
            frame_id=f"frame_{index:04d}",
            occurred_at=cls.BASE + timedelta(seconds=index),
            detections=tuple(detections),
        )

    @classmethod
    def _batch(cls, *frames: DetectionFrame) -> DetectionFrameBatch:
        return DetectionFrameBatch(
            camera_id="motion_test_cam",
            detector_adapter="synthetic_detector",
            frames=tuple(frames),
        )

    def test_example_schema_and_config_are_strict_and_path_safe(self) -> None:
        config = MotionTrackerConfig.read("config/motion-two-stage-tracker.example.json")
        schema = json.loads(
            Path("schemas/motion-two-stage-tracker-config-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(config.to_dict()))
        self.assertEqual(config, MotionTrackerConfig.from_dict(config.to_dict()))

        with self.assertRaisesRegex(MotionTrackerError, "below"):
            MotionTrackerConfig(
                high_confidence_threshold=0.5,
                low_confidence_threshold=0.5,
            )
        with self.assertRaisesRegex(MotionTrackerError, "cannot exceed"):
            MotionTrackerConfig(
                primary_match_threshold=0.2,
                secondary_match_threshold=0.3,
            )
        with self.assertRaisesRegex(MotionTrackerError, "cannot exceed"):
            MotionTrackerConfig(
                max_track_age_seconds=1.0,
                max_prediction_seconds=2.0,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text(
                '{"version":1,"version":1}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MotionTrackerError, "duplicate JSON"):
                MotionTrackerConfig.read(duplicate)
            if os.name == "posix":
                link = root / "config-link.json"
                link.symlink_to(Path("config/motion-two-stage-tracker.example.json").resolve())
                with self.assertRaisesRegex(MotionTrackerError, "missing or unsafe"):
                    MotionTrackerConfig.read(link)

    def test_persistent_runtime_matches_offline_assignments_and_round_trips(self) -> None:
        batch = self._batch(
            self._frame(0, self._detection("high0", 0.10, 0.30)),
            self._frame(
                1,
                self._detection("low1", 0.20, 0.40, confidence=0.4),
                self._detection("ignored1", 0.70, 0.90, confidence=0.2),
            ),
            self._frame(2, self._detection("high2", 0.35, 0.55)),
        )
        config = MotionTrackerConfig()
        expected = {
            (item.frame_id, item.detection_id): item.tracking_id
            for item in build_motion_track_assignments(batch, config)
        }
        tracker = MotionObjectTracker(
            config=config,
            camera_id=batch.camera_id,
            detector_adapter=batch.detector_adapter,
        )
        actual: dict[tuple[str, str], str] = {}
        for frame in batch.frames:
            for observation, detection in zip(
                tracker.process(frame),
                frame.detections,
                strict=True,
            ):
                actual[(frame.frame_id, detection.detection_id)] = observation.tracking_id
                self.assertEqual(
                    MOTION_TRACKER_ADAPTER_ID,
                    observation.source_adapter,
                )
        self.assertEqual(expected, actual)
        self.assertEqual((), tracker.process(batch.frames[-1]))

        schema = json.loads(
            Path("schemas/motion-tracker-state-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(tracker.state.to_dict()))
        self.assertEqual(
            tracker.state,
            MotionTrackerState.from_dict(tracker.state.to_dict()),
        )
        conflicting = self._frame(
            2,
            self._detection("high2", 0.45, 0.65),
        )
        with self.assertRaisesRegex(MotionTrackerError, "conflicting"):
            tracker.process(conflicting)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "motion-state.json"
            tracker.state.write_atomic(state_path)
            loaded = MotionTrackerState.read(state_path)
            self.assertEqual(tracker.state, loaded)
            self.assertEqual(0o600, state_path.stat().st_mode & 0o777)
            self.assertEqual([], list(root.glob(".motion-state.json.*.tmp")))
            first_id = loaded.pending_observations[0].observation_id
            acknowledged = loaded.acknowledge((first_id,))
            self.assertEqual(
                len(loaded.pending_observations) - 1,
                len(acknowledged.pending_observations),
            )
            with self.assertRaisesRegex(MotionTrackerError, "unknown"):
                acknowledged.acknowledge((first_id,))

    def test_persistent_runtime_rejects_wrong_source_config_and_limits(self) -> None:
        config = MotionTrackerConfig(max_active_tracks=1)
        tracker = MotionObjectTracker(
            config=config,
            camera_id="motion_test_cam",
            detector_adapter="synthetic_detector",
        )
        before = tracker.state
        crowded = self._frame(
            0,
            self._detection("left", 0.05, 0.25),
            self._detection("right", 0.70, 0.90),
        )
        with self.assertRaisesRegex(MotionTrackerError, "active track limit"):
            tracker.process(crowded)
        self.assertEqual(before, tracker.state)

        wrong_config_state = MotionTrackerState.empty(
            camera_id="motion_test_cam",
            detector_adapter="synthetic_detector",
            config_fingerprint=MotionTrackerConfig().fingerprint,
        )
        with self.assertRaisesRegex(MotionTrackerError, "different config"):
            MotionObjectTracker(
                config=MotionTrackerConfig(iou_weight=0.5),
                camera_id="motion_test_cam",
                detector_adapter="synthetic_detector",
                state=wrong_config_state,
            )
        with self.assertRaisesRegex(MotionTrackerError, "different input source"):
            MotionObjectTracker(
                config=MotionTrackerConfig(),
                camera_id="another_cam",
                detector_adapter="synthetic_detector",
                state=wrong_config_state,
            )

    def test_global_assignment_maximizes_total_score_deterministically(self) -> None:
        scores = [[0.9, 0.8], [0.85, 0.1]]
        self.assertEqual(
            ((0, 1), (1, 0)),
            _maximum_weight_assignment(scores),
        )
        self.assertEqual(
            ((0, 1), (1, 0)),
            _maximum_weight_assignment(scores),
        )
        self.assertEqual(
            ((0, 0),),
            _maximum_weight_assignment(
                [[0.3, 0.29], [0.29, 0.0]],
                minimum_score=0.3,
            ),
        )
        self.assertEqual((), _maximum_weight_assignment([]))

    def test_motion_prediction_preserves_a_fast_track_that_plain_iou_splits(self) -> None:
        batch = self._batch(
            self._frame(0, self._detection("d0", 0.10, 0.30)),
            self._frame(1, self._detection("d1", 0.20, 0.40)),
            self._frame(2, self._detection("d2", 0.35, 0.55)),
        )
        plain = build_iou_tracker_predictions(batch, IoUTrackerConfig())
        motion = build_motion_tracker_predictions(batch, MotionTrackerConfig())
        self.assertEqual(2, len({item.tracking_id for item in plain.assignments}))
        self.assertEqual(1, len({item.tracking_id for item in motion.assignments}))
        self.assertEqual(
            MOTION_TRACKER_ADAPTER_ID,
            motion.tracker_adapter,
        )
        self.assertFalse(motion.subject_identity_inference)
        self.assertFalse(motion.paths_disclosed)

    def test_invalid_predicted_extent_falls_back_to_last_observed_box(self) -> None:
        cases = (
            ("width", (0.1, 0.1, 0.9, 0.9), (0.4, 0.2, 0.6, 0.8)),
            ("height", (0.1, 0.1, 0.9, 0.9), (0.2, 0.4, 0.8, 0.6)),
        )
        for label, first, second in cases:
            with self.subTest(extent=label):
                batch = self._batch(
                    DetectionFrame(
                        frame_id="frame_0000",
                        occurred_at=self.BASE,
                        detections=(FrameDetection("d0", "person", 0.9, first),),
                    ),
                    DetectionFrame(
                        frame_id="frame_0001",
                        occurred_at=self.BASE + timedelta(seconds=1),
                        detections=(FrameDetection("d1", "person", 0.9, second),),
                    ),
                    DetectionFrame(
                        frame_id="frame_0002",
                        occurred_at=self.BASE + timedelta(seconds=2),
                        detections=(FrameDetection("d2", "person", 0.9, second),),
                    ),
                )
                predictions = build_motion_tracker_predictions(batch, MotionTrackerConfig())
                self.assertEqual(1, len({row.tracking_id for row in predictions.assignments}))

    def test_invalid_dormant_prediction_cannot_steal_qualified_recent_detection(self) -> None:
        big = (0.1, 0.1, 0.9, 0.9)
        small = (0.32, 0.32, 0.68, 0.68)
        for confidence in (0.9, 0.4):
            with self.subTest(confidence=confidence):
                config = MotionTrackerConfig(high_confidence_threshold=0.7)
                tracker = MotionObjectTracker(
                    config=config,
                    camera_id="motion_test_cam",
                    detector_adapter="synthetic_detector",
                )
                tracker.process(self._frame(1, FrameDetection("seed", "person", 0.9, small)))
                old = MotionActiveTrack(
                    "old_track",
                    "person",
                    MotionTrackSample(self.BASE + timedelta(seconds=1), 0.9, small),
                    MotionTrackSample(self.BASE, 0.9, big),
                )
                recent = MotionActiveTrack(
                    "recent_track",
                    "person",
                    MotionTrackSample(
                        self.BASE + timedelta(seconds=1), 0.9, (0.33, 0.32, 0.69, 0.68)
                    ),
                )
                tracker.state = replace(tracker.state, active_tracks=(old, recent))
                rows = tracker.process(
                    self._frame(2, FrameDetection("next", "person", confidence, small))
                )
                self.assertEqual("recent_track", rows[0].tracking_id)

    def test_low_confidence_stage_recovers_but_does_not_seed_or_pollute(self) -> None:
        batch = self._batch(
            self._frame(0, self._detection("main0", 0.10, 0.30)),
            self._frame(
                1,
                self._detection("main1", 0.11, 0.31),
                self._detection(
                    "duplicate1",
                    0.12,
                    0.32,
                    confidence=0.4,
                ),
            ),
            self._frame(2, self._detection("main2", 0.12, 0.32)),
        )
        assignments = {
            (item.frame_id, item.detection_id): item.tracking_id
            for item in build_motion_track_assignments(batch, MotionTrackerConfig())
        }
        main_id = assignments[("frame_0000", "main0")]
        self.assertEqual(main_id, assignments[("frame_0001", "main1")])
        self.assertEqual(main_id, assignments[("frame_0002", "main2")])
        self.assertNotEqual(
            main_id,
            assignments[("frame_0001", "duplicate1")],
        )

        recovered = self._batch(
            self._frame(0, self._detection("high0", 0.10, 0.30)),
            self._frame(
                1,
                self._detection("low1", 0.11, 0.31, confidence=0.4),
            ),
            self._frame(2, self._detection("high2", 0.12, 0.32)),
        )
        recovered_ids = [
            item.tracking_id
            for item in build_motion_track_assignments(
                recovered,
                MotionTrackerConfig(),
            )
        ]
        self.assertEqual(1, len(set(recovered_ids)))

        low_first = self._batch(
            self._frame(
                0,
                self._detection("low0", 0.10, 0.30, confidence=0.4),
            ),
            self._frame(1, self._detection("high1", 0.11, 0.31)),
        )
        low_first_ids = [
            item.tracking_id
            for item in build_motion_track_assignments(
                low_first,
                MotionTrackerConfig(),
            )
        ]
        self.assertEqual(2, len(set(low_first_ids)))

    def test_label_capacity_and_per_frame_bounds_fail_closed(self) -> None:
        labels = self._batch(
            self._frame(
                0,
                self._detection("person0", 0.10, 0.30, label="person"),
            ),
            self._frame(
                1,
                self._detection("dog1", 0.11, 0.31, label="dog"),
            ),
        )
        label_ids = {
            item.tracking_id
            for item in build_motion_track_assignments(
                labels,
                MotionTrackerConfig(),
            )
        }
        self.assertEqual(2, len(label_ids))

        crowded = self._batch(
            self._frame(
                0,
                self._detection("left", 0.05, 0.25),
                self._detection("right", 0.70, 0.90),
            )
        )
        with self.assertRaisesRegex(MotionTrackerError, "active track limit"):
            build_motion_track_assignments(
                crowded,
                MotionTrackerConfig(max_active_tracks=1),
            )
        with self.assertRaisesRegex(MotionTrackerError, "max_frame_detections"):
            build_motion_track_assignments(
                crowded,
                MotionTrackerConfig(max_frame_detections=1),
            )

    def test_cli_prediction_is_path_free_private_exclusive_and_bound(self) -> None:
        batch = self._batch(
            self._frame(0, self._detection("d0", 0.10, 0.30)),
            self._frame(1, self._detection("d1", 0.12, 0.32)),
        )
        with tempfile.TemporaryDirectory(prefix="PRIVATE_MOTION_TRACKER_") as directory:
            root = Path(directory)
            frames = root / "frames.json"
            config = root / "config.json"
            output = root / "predictions.json"
            frames.write_text(json.dumps(batch.to_dict()), encoding="utf-8")
            config.write_text(
                json.dumps(MotionTrackerConfig().to_dict()),
                encoding="utf-8",
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "track-motion-predict",
                            "--tracker-config",
                            str(config),
                            "--frames",
                            str(frames),
                            "--output",
                            str(output),
                        ]
                    ),
                )
            public = json.loads(stdout.getvalue())
            self.assertEqual(MOTION_TRACKER_ADAPTER_ID, public["tracker_adapter"])
            self.assertNotIn(str(root), stdout.getvalue())
            predictions = TrackerPredictionBatch.read(output)
            predictions.bind(batch)
            self.assertEqual(0o600, output.stat().st_mode & 0o777)

            validate_stdout = io.StringIO()
            with redirect_stdout(validate_stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "validate-tracker-predictions",
                            "--frames",
                            str(frames),
                            "--predictions",
                            str(output),
                        ]
                    ),
                )
            self.assertNotIn(str(root), validate_stdout.getvalue())

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(
                    2,
                    main(
                        [
                            "track-motion-predict",
                            "--tracker-config",
                            str(config),
                            "--frames",
                            str(frames),
                            "--output",
                            str(output),
                        ]
                    ),
                )
            self.assertNotIn(str(root), stderr.getvalue())

    def test_cli_persistent_run_replay_pending_and_ack_are_path_free(self) -> None:
        with tempfile.TemporaryDirectory(prefix="PRIVATE_MOTION_RUNTIME_") as directory:
            root = Path(directory)
            state = root / "state.json"
            arguments = [
                "track-motion-run",
                "--tracker-config",
                "config/motion-two-stage-tracker.example.json",
                "--frames",
                "tests/fixtures/valid_detection_frame_batch.json",
                "--state",
                str(state),
            ]
            first_stdout = io.StringIO()
            with redirect_stdout(first_stdout):
                self.assertEqual(0, main(arguments))
            first = json.loads(first_stdout.getvalue())
            self.assertEqual(MOTION_TRACKER_ADAPTER_ID, first["tracker_adapter"])
            self.assertEqual(2, first["new_frames"])
            self.assertEqual(4, first["observations_added"])
            self.assertEqual(4, first["pending_observation_count"])
            self.assertNotIn(str(root), first_stdout.getvalue())

            replay_stdout = io.StringIO()
            with redirect_stdout(replay_stdout):
                self.assertEqual(0, main(arguments))
            replay = json.loads(replay_stdout.getvalue())
            self.assertEqual(0, replay["new_frames"])
            self.assertEqual(2, replay["replayed_frames"])
            self.assertFalse(replay["state_written"])

            pending_stdout = io.StringIO()
            with redirect_stdout(pending_stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "track-motion-pending",
                            "--state",
                            str(state),
                        ]
                    ),
                )
            pending = json.loads(pending_stdout.getvalue())
            ack = ["track-motion-ack", "--state", str(state)]
            for observation in pending["pending_observations"]:
                ack.extend(("--observation-id", observation["observation_id"]))
            ack_stdout = io.StringIO()
            with redirect_stdout(ack_stdout):
                self.assertEqual(0, main(ack))
            acknowledged = json.loads(ack_stdout.getvalue())
            self.assertEqual(4, acknowledged["acknowledged"])
            self.assertEqual(0, acknowledged["pending_observation_count"])
            self.assertNotIn(str(root), ack_stdout.getvalue())

    def test_cli_zone_bridge_recovers_after_motion_ack_write_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker_state = root / "motion.json"
            zone_state = root / "zone.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    0,
                    main(
                        [
                            "track-motion-run",
                            "--tracker-config",
                            "config/motion-two-stage-tracker.example.json",
                            "--frames",
                            "tests/fixtures/valid_detection_frame_batch.json",
                            "--state",
                            str(tracker_state),
                        ]
                    ),
                )
            arguments = [
                "track-motion-zone-run",
                "--tracker-state",
                str(tracker_state),
                "--zone-config",
                "config/temporal-zones.example.json",
                "--zone-state",
                str(zone_state),
            ]
            with (
                patch(
                    "baby_monitor.cli_commands.tracking.MotionTrackerState.write_atomic",
                    side_effect=MotionTrackerError("fixture write failure"),
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(2, main(arguments))
            self.assertTrue(zone_state.is_file())
            self.assertEqual(
                4,
                len(MotionTrackerState.read(tracker_state).pending_observations),
            )

            recovered_stdout = io.StringIO()
            with redirect_stdout(recovered_stdout):
                self.assertEqual(0, main(arguments))
            recovered = json.loads(recovered_stdout.getvalue())
            self.assertEqual(4, recovered["processed_observations"])
            self.assertEqual([], recovered["emitted_signals"])
            self.assertEqual(0, recovered["tracker_pending_observation_count"])


if __name__ == "__main__":
    unittest.main()
