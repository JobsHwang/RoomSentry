from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.cli import main
from baby_monitor.domain import SignalType
from baby_monitor.object_tracker import (
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
    IoUObjectTracker,
    IoUTrackerConfig,
    IoUTrackerState,
    ObjectTrackerError,
)
from baby_monitor.temporal_zone import (
    TemporalZoneAssembler,
    TemporalZoneConfig,
    ZoneDefinition,
)


class ObjectTrackerTests(unittest.TestCase):
    BASE = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)

    @staticmethod
    def _detection(
        detection_id: str,
        label: str,
        bbox: tuple[float, float, float, float],
        confidence: float = 0.9,
    ) -> FrameDetection:
        return FrameDetection(detection_id, label, confidence, bbox)

    @classmethod
    def _frame(
        cls,
        index: int,
        *detections: FrameDetection,
        seconds: float | None = None,
        frame_id: str | None = None,
    ) -> DetectionFrame:
        return DetectionFrame(
            frame_id=frame_id or f"frame_{index:04d}",
            occurred_at=cls.BASE + timedelta(seconds=index if seconds is None else seconds),
            detections=tuple(detections),
        )

    @staticmethod
    def _tracker(
        *,
        config: IoUTrackerConfig | None = None,
        state: IoUTrackerState | None = None,
    ) -> IoUObjectTracker:
        return IoUObjectTracker(
            config=config or IoUTrackerConfig(),
            camera_id="cam_01",
            detector_adapter="synthetic_detector",
            state=state,
        )

    def test_public_contracts_example_and_fixture_match_runtime(self) -> None:
        config_data = json.loads(
            Path("config/iou-tracker.example.json").read_text(encoding="utf-8")
        )
        batch_data = json.loads(
            Path("tests/fixtures/valid_detection_frame_batch.json").read_text(encoding="utf-8")
        )
        config = IoUTrackerConfig.from_dict(config_data)
        batch = DetectionFrameBatch.from_dict(batch_data)
        config_schema = json.loads(
            Path("schemas/iou-tracker-config-v1.schema.json").read_text(encoding="utf-8")
        )
        batch_schema = json.loads(
            Path("schemas/detection-frame-batch-v1.schema.json").read_text(encoding="utf-8")
        )
        state_schema = json.loads(
            Path("schemas/iou-tracker-state-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(config_schema["required"]), set(config.to_dict()))
        self.assertEqual(set(batch_schema["required"]), set(batch.to_dict()))
        self.assertEqual("sports ball", batch.frames[0].detections[1].label)

        tracker = self._tracker(config=config)
        for frame in batch.frames:
            tracker.process(frame)
        self.assertEqual(set(state_schema["required"]), set(tracker.state.to_dict()))
        self.assertEqual(tracker.state, IoUTrackerState.from_dict(tracker.state.to_dict()))
        tampered = tracker.state.to_dict()
        tampered["pending_observations"] = list(reversed(tampered["pending_observations"]))
        with self.assertRaisesRegex(ObjectTrackerError, "not chronological"):
            IoUTrackerState.from_dict(tampered)

    def test_detection_batch_write_is_private_exclusive_and_round_trips(self) -> None:
        batch = DetectionFrameBatch.read("tests/fixtures/valid_detection_frame_batch.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "detection-frames.json"
            batch.write_new(output)
            self.assertEqual(batch, DetectionFrameBatch.read(output))
            self.assertEqual(0o600, output.stat().st_mode & 0o777)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                batch.write_new(output)

            if os.name == "posix":
                link = root / "detection-link.json"
                link.symlink_to(output)
                with self.assertRaisesRegex(ObjectTrackerError, "unsafe"):
                    batch.write_new(link)

    def test_iou_association_is_deterministic_and_label_preserving(self) -> None:
        first = self._frame(
            0,
            self._detection("det_a", "person", (0.1, 0.1, 0.3, 0.8)),
            self._detection("det_b", "dog", (0.7, 0.4, 0.9, 0.8)),
        )
        second = self._frame(
            1,
            self._detection("det_c", "person", (0.12, 0.1, 0.32, 0.8)),
            self._detection("det_d", "dog", (0.68, 0.4, 0.88, 0.8)),
        )
        tracker = self._tracker()
        initial = tracker.process(first)
        continued = tracker.process(second)
        initial_by_label = {item.label: item.tracking_id for item in initial}
        continued_by_label = {item.label: item.tracking_id for item in continued}
        self.assertEqual(initial_by_label, continued_by_label)
        self.assertEqual({"person", "dog"}, set(initial_by_label))
        self.assertFalse(any("baby" in item.tracking_id for item in continued))

        replay_from_empty = self._tracker()
        self.assertEqual(initial, replay_from_empty.process(first))
        self.assertEqual(continued, replay_from_empty.process(second))

    def test_label_change_or_expiry_creates_a_new_anonymous_track(self) -> None:
        tracker = self._tracker()
        first = tracker.process(
            self._frame(
                0,
                self._detection("det_a", "person", (0.1, 0.1, 0.4, 0.8)),
            )
        )[0]
        changed_label = tracker.process(
            self._frame(
                1,
                self._detection("det_b", "adult", (0.1, 0.1, 0.4, 0.8)),
            )
        )[0]
        self.assertNotEqual(first.tracking_id, changed_label.tracking_id)
        self.assertEqual("adult", changed_label.label)

        expired = tracker.process(
            self._frame(
                2,
                self._detection("det_c", "person", (0.1, 0.1, 0.4, 0.8)),
                seconds=4,
            )
        )[0]
        self.assertNotEqual(first.tracking_id, expired.tracking_id)

    def test_replay_is_idempotent_conflict_and_stale_input_fail_closed(self) -> None:
        tracker = self._tracker()
        frame = self._frame(
            0,
            self._detection("det_a", "person", (0.1, 0.1, 0.4, 0.8)),
        )
        original = tracker.process(frame)
        self.assertEqual((), tracker.process(frame))
        self.assertEqual(original, tracker.state.pending_observations)

        conflicting = self._frame(
            0,
            self._detection("det_a", "person", (0.2, 0.1, 0.5, 0.8)),
        )
        with self.assertRaisesRegex(ObjectTrackerError, "conflicting"):
            tracker.process(conflicting)
        stale = self._frame(
            9,
            self._detection("det_z", "person", (0.1, 0.1, 0.4, 0.8)),
            seconds=-1,
        )
        with self.assertRaisesRegex(ObjectTrackerError, "monotonic"):
            tracker.process(stale)

    def test_active_pending_and_processed_limits_preserve_previous_state(self) -> None:
        active_config = IoUTrackerConfig(max_active_tracks=1)
        active_tracker = self._tracker(config=active_config)
        with self.assertRaisesRegex(ObjectTrackerError, "active track limit"):
            active_tracker.process(
                self._frame(
                    0,
                    self._detection("a", "person", (0.1, 0.1, 0.3, 0.5)),
                    self._detection("b", "person", (0.7, 0.1, 0.9, 0.5)),
                )
            )
        self.assertIsNone(active_tracker.state.last_frame_at)

        pending_config = IoUTrackerConfig(max_pending_observations=1)
        pending_tracker = self._tracker(config=pending_config)
        pending_tracker.process(
            self._frame(
                0,
                self._detection("a", "person", (0.1, 0.1, 0.3, 0.5)),
            )
        )
        before = pending_tracker.state
        with self.assertRaisesRegex(ObjectTrackerError, "outbox is full"):
            pending_tracker.process(
                self._frame(
                    1,
                    self._detection("b", "person", (0.11, 0.1, 0.31, 0.5)),
                )
            )
        self.assertEqual(before, pending_tracker.state)

        processed_config = IoUTrackerConfig(max_processed_frames=1)
        processed_tracker = self._tracker(config=processed_config)
        old = self._frame(0)
        processed_tracker.process(old)
        processed_tracker.process(self._frame(1))
        self.assertEqual(1, len(processed_tracker.state.processed_frames))
        with self.assertRaisesRegex(ObjectTrackerError, "monotonic"):
            processed_tracker.process(old)

    def test_state_outbox_round_trip_ack_and_symlink_policy(self) -> None:
        tracker = self._tracker()
        observations = tracker.process(
            self._frame(
                0,
                self._detection("a", "person", (0.1, 0.1, 0.3, 0.5)),
                self._detection("b", "sports ball", (0.7, 0.6, 0.8, 0.8)),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_path = root / "tracker-state.json"
            tracker.state.write_atomic(state_path)
            loaded = IoUTrackerState.read(state_path)
            self.assertEqual(tracker.state, loaded)
            self.assertEqual([], list(root.glob(".tracker-state.json.*.tmp")))
            updated = loaded.acknowledge((observations[0].observation_id,))
            self.assertEqual(1, len(updated.pending_observations))
            with self.assertRaisesRegex(ObjectTrackerError, "unknown"):
                updated.acknowledge((observations[0].observation_id,))

            if os.name == "posix":
                link = root / "state-link.json"
                link.symlink_to(state_path)
                with self.assertRaisesRegex(ObjectTrackerError, "missing or unsafe"):
                    IoUTrackerState.read(link)

    def test_tracker_output_can_drive_multiword_temporal_zone_without_identity_claim(self) -> None:
        tracker = self._tracker()
        observation = tracker.process(
            self._frame(
                0,
                self._detection("ball", "sports ball", (0.4, 0.4, 0.6, 0.6)),
            )
        )[0]
        assembler = TemporalZoneAssembler(
            config=TemporalZoneConfig(
                zones=(
                    ZoneDefinition(
                        zone_id="floor_zone",
                        labels=("sports ball",),
                        bbox_xyxy=(0.2, 0.2, 0.8, 0.8),
                        dwell_seconds=1,
                        exit_margin=0.05,
                    ),
                ),
                track_expiry_seconds=10,
                max_processed_observations=100,
            )
        )
        signals = assembler.process(observation)
        self.assertEqual((SignalType.ZONE_ENTER,), tuple(item.signal_type for item in signals))
        self.assertEqual("sports ball", dict(signals[0].attributes)["object_label"])

    def test_cli_validate_run_replay_pending_and_ack_are_path_free(self) -> None:
        with tempfile.TemporaryDirectory(prefix="PRIVATE_TRACKER_ROOT_") as directory:
            root = Path(directory)
            state_path = root / "state.json"
            validate_output = io.StringIO()
            with redirect_stdout(validate_output):
                validate_code = main(
                    [
                        "validate-detection-frames",
                        "tests/fixtures/valid_detection_frame_batch.json",
                    ]
                )
            self.assertEqual(0, validate_code)
            self.assertNotIn(str(root), validate_output.getvalue())

            arguments = [
                "track-iou-run",
                "--tracker-config",
                "config/iou-tracker.example.json",
                "--frames",
                "tests/fixtures/valid_detection_frame_batch.json",
                "--state",
                str(state_path),
            ]
            first_output = io.StringIO()
            with redirect_stdout(first_output):
                first_code = main(arguments)
            first = json.loads(first_output.getvalue())
            self.assertEqual(0, first_code)
            self.assertEqual(2, first["new_frames"])
            self.assertEqual(4, first["observations_added"])
            self.assertEqual(4, first["pending_observation_count"])
            self.assertNotIn(str(root), first_output.getvalue())

            replay_output = io.StringIO()
            with redirect_stdout(replay_output):
                replay_code = main(arguments)
            replay = json.loads(replay_output.getvalue())
            self.assertEqual(0, replay_code)
            self.assertEqual(0, replay["new_frames"])
            self.assertEqual(2, replay["replayed_frames"])
            self.assertEqual(4, replay["pending_observation_count"])

            pending_output = io.StringIO()
            with redirect_stdout(pending_output):
                pending_code = main(["track-iou-pending", "--state", str(state_path)])
            pending = json.loads(pending_output.getvalue())
            self.assertEqual(0, pending_code)
            observation_ids = [item["observation_id"] for item in pending["pending_observations"]]
            ack_args = ["track-iou-ack", "--state", str(state_path)]
            for observation_id in observation_ids:
                ack_args.extend(("--observation-id", observation_id))
            ack_output = io.StringIO()
            with redirect_stdout(ack_output):
                ack_code = main(ack_args)
            acknowledged = json.loads(ack_output.getvalue())
            self.assertEqual(0, ack_code)
            self.assertEqual(4, acknowledged["acknowledged"])
            self.assertEqual(0, acknowledged["pending_observation_count"])
            self.assertNotIn(str(root), ack_output.getvalue())

    def test_cli_zone_bridge_persists_before_ack_and_replay_is_empty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="PRIVATE_TRACK_ZONE_ROOT_") as directory:
            root = Path(directory)
            tracker_state = root / "tracker.json"
            zone_state = root / "zone.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    0,
                    main(
                        [
                            "track-iou-run",
                            "--tracker-config",
                            "config/iou-tracker.example.json",
                            "--frames",
                            "tests/fixtures/valid_detection_frame_batch.json",
                            "--state",
                            str(tracker_state),
                        ]
                    ),
                )
            bridge_arguments = [
                "track-iou-zone-run",
                "--tracker-state",
                str(tracker_state),
                "--zone-config",
                "config/temporal-zones.example.json",
                "--zone-state",
                str(zone_state),
            ]
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(bridge_arguments)
            payload = json.loads(output.getvalue())
            self.assertEqual(0, exit_code)
            self.assertEqual(4, payload["processed_observations"])
            self.assertEqual(0, payload["tracker_pending_observation_count"])
            self.assertGreaterEqual(payload["zone_pending_signal_count"], 1)
            self.assertTrue(payload["states_written"])
            self.assertNotIn(str(root), output.getvalue())
            self.assertEqual((), IoUTrackerState.read(tracker_state).pending_observations)

            replay_output = io.StringIO()
            with redirect_stdout(replay_output):
                replay_code = main(bridge_arguments)
            replay = json.loads(replay_output.getvalue())
            self.assertEqual(0, replay_code)
            self.assertEqual(0, replay["processed_observations"])
            self.assertFalse(replay["states_written"])
            self.assertEqual(
                payload["zone_pending_signal_count"],
                replay["zone_pending_signal_count"],
            )

    def test_zone_bridge_recovers_if_tracker_ack_persistence_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker_state = root / "tracker.json"
            zone_state = root / "zone.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    0,
                    main(
                        [
                            "track-iou-run",
                            "--tracker-config",
                            "config/iou-tracker.example.json",
                            "--frames",
                            "tests/fixtures/valid_detection_frame_batch.json",
                            "--state",
                            str(tracker_state),
                        ]
                    ),
                )
            arguments = [
                "track-iou-zone-run",
                "--tracker-state",
                str(tracker_state),
                "--zone-config",
                "config/temporal-zones.example.json",
                "--zone-state",
                str(zone_state),
            ]
            with (
                patch(
                    "baby_monitor.cli_commands.tracking.IoUTrackerState.write_atomic",
                    side_effect=ObjectTrackerError("fixture write failure"),
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(2, main(arguments))
            self.assertTrue(zone_state.is_file())
            self.assertEqual(4, len(IoUTrackerState.read(tracker_state).pending_observations))

            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(arguments))
            recovered = json.loads(output.getvalue())
            self.assertEqual(4, recovered["processed_observations"])
            self.assertEqual([], recovered["emitted_signals"])
            self.assertEqual(0, recovered["tracker_pending_observation_count"])


if __name__ == "__main__":
    unittest.main()
