from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.adapters.vlm import MockVLMAdapter
from baby_monitor.cli import main
from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    EventMedia,
    EventType,
    RiskLevel,
    SignalEvent,
    SignalType,
)
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy
from baby_monitor.temporal_zone import (
    TemporalZoneAssembler,
    TemporalZoneConfig,
    TemporalZoneError,
    TemporalZoneState,
    TrackedObjectObservation,
    ZoneDefinition,
    load_tracked_observations,
)


class TemporalZoneTests(unittest.TestCase):
    BASE = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)

    @staticmethod
    def _config(
        *,
        expiry: float = 10.0,
        minimum_confidence: float = 0.0,
    ) -> TemporalZoneConfig:
        return TemporalZoneConfig(
            zones=(
                ZoneDefinition(
                    zone_id="crib_zone",
                    labels=("person",),
                    bbox_xyxy=(0.2, 0.2, 0.8, 0.8),
                    dwell_seconds=2.0,
                    exit_margin=0.05,
                    minimum_confidence=minimum_confidence,
                ),
            ),
            track_expiry_seconds=expiry,
            max_processed_observations=100,
        )

    @classmethod
    def _observation(
        cls,
        index: int,
        center_x: float,
        *,
        seconds: float | None = None,
        observation_id: str | None = None,
        label: str = "person",
        confidence: float = 0.9,
    ) -> TrackedObjectObservation:
        moment = cls.BASE + timedelta(seconds=index if seconds is None else seconds)
        return TrackedObjectObservation(
            observation_id=observation_id or f"obs_{index}",
            camera_id="cam_01",
            occurred_at=moment,
            source_adapter="fixture_tracker",
            tracking_id="track_01",
            label=label,
            confidence=confidence,
            bbox_xyxy=(center_x - 0.05, 0.4, center_x + 0.05, 0.6),
        )

    def test_enter_dwell_hysteresis_and_conservative_exit_duration(self) -> None:
        assembler = TemporalZoneAssembler(config=self._config())
        self.assertEqual((), assembler.process(self._observation(0, 0.1)))
        entered = assembler.process(self._observation(1, 0.3))
        self.assertEqual((SignalType.ZONE_ENTER,), tuple(s.signal_type for s in entered))

        hysteresis = assembler.process(self._observation(2, 0.82))
        self.assertEqual((), hysteresis)
        dwell = assembler.process(self._observation(4, 0.7))
        self.assertEqual((SignalType.ZONE_DWELL,), tuple(s.signal_type for s in dwell))
        self.assertEqual(3000, dict(dwell[0].attributes)["duration_ms"])

        exited = assembler.process(self._observation(5, 0.9))
        self.assertEqual((SignalType.ZONE_EXIT,), tuple(s.signal_type for s in exited))
        self.assertEqual(3000, dict(exited[0].attributes)["duration_ms"])
        self.assertEqual("person", dict(exited[0].attributes)["object_label"])
        self.assertEqual(3, len(assembler.state.pending_signals))

    def test_duplicate_conflict_order_and_label_change_fail_closed(self) -> None:
        assembler = TemporalZoneAssembler(config=self._config())
        original = self._observation(1, 0.3)
        self.assertEqual(1, len(assembler.process(original)))
        self.assertEqual((), assembler.process(original))
        conflicting = self._observation(
            1,
            0.4,
            observation_id=original.observation_id,
        )
        with self.assertRaisesRegex(TemporalZoneError, "conflicting"):
            assembler.process(conflicting)
        with self.assertRaisesRegex(TemporalZoneError, "monotonic"):
            assembler.process(self._observation(0, 0.4, observation_id="obs_old"))
        with self.assertRaisesRegex(TemporalZoneError, "changed object label"):
            assembler.process(self._observation(2, 0.4, label="pet"))

    def test_expired_track_does_not_invent_exit(self) -> None:
        assembler = TemporalZoneAssembler(config=self._config(expiry=2.0))
        entered = assembler.process(self._observation(0, 0.3))
        self.assertEqual(1, len(entered))
        later = TrackedObjectObservation(
            observation_id="other_track",
            camera_id="cam_01",
            occurred_at=self.BASE + timedelta(seconds=5),
            source_adapter="fixture_tracker",
            tracking_id="track_02",
            label="person",
            confidence=0.8,
            bbox_xyxy=(0.0, 0.4, 0.1, 0.6),
        )
        self.assertEqual((), assembler.process(later))
        self.assertFalse(
            any(
                signal.signal_type is SignalType.ZONE_EXIT
                for signal in assembler.state.pending_signals
            )
        )
        self.assertFalse(any(track.tracking_id == "track_01" for track in assembler.state.tracks))

    def test_zone_confidence_gate_ignores_low_boxes_without_forcing_exit(self) -> None:
        assembler = TemporalZoneAssembler(config=self._config(minimum_confidence=0.65))
        low_inside = self._observation(0, 0.3, confidence=0.5)
        self.assertEqual((), assembler.process(low_inside))
        self.assertEqual((), assembler.state.tracks)
        self.assertEqual(1, len(assembler.state.processed_observations))

        entered = assembler.process(self._observation(1, 0.3))
        self.assertEqual((SignalType.ZONE_ENTER,), tuple(signal.signal_type for signal in entered))
        last_seen = assembler.state.tracks[0].last_seen_at
        self.assertEqual(
            (),
            assembler.process(self._observation(2, 0.95, confidence=0.5)),
        )
        self.assertEqual(last_seen, assembler.state.tracks[0].last_seen_at)

        exited = assembler.process(self._observation(3, 0.95))
        self.assertEqual((SignalType.ZONE_EXIT,), tuple(signal.signal_type for signal in exited))

        legacy = assembler.config.zones[0].to_dict()
        legacy.pop("minimum_confidence")
        self.assertEqual(0.0, ZoneDefinition.from_dict(legacy).minimum_confidence)

    def test_state_round_trip_config_identity_outbox_and_ack(self) -> None:
        config = self._config()
        assembler = TemporalZoneAssembler(config=config)
        signal = assembler.process(self._observation(0, 0.3))[0]
        with tempfile.TemporaryDirectory(prefix="baby-monitor-temporal-") as directory:
            state_path = Path(directory).resolve() / "state.json"
            assembler.state.write_atomic(state_path)
            loaded = TemporalZoneState.read(state_path)
            self.assertEqual(assembler.state, loaded)
            self.assertEqual([], list(state_path.parent.glob(".state.json.*.tmp")))
            self.assertEqual(
                (),
                TemporalZoneAssembler(config=config, state=loaded).process(
                    self._observation(0, 0.3)
                ),
            )
            acknowledged = loaded.acknowledge((signal.signal_id,))
            self.assertEqual((), acknowledged.pending_signals)
            changed = TemporalZoneConfig(
                zones=config.zones,
                track_expiry_seconds=11,
                max_processed_observations=100,
            )
            with self.assertRaisesRegex(TemporalZoneError, "different config"):
                TemporalZoneAssembler(config=changed, state=loaded)

            tampered = loaded.to_dict()
            tampered["pending_signals"][0]["signal_type"] = "manual_trigger"
            with self.assertRaisesRegex(TemporalZoneError, "outbox signal type"):
                TemporalZoneState.from_dict(tampered)

            tampered = loaded.to_dict()
            tampered["pending_signals"][0]["attributes"]["zone_id"] = "unknown_zone"
            valid_but_wrong_zone = TemporalZoneState.from_dict(tampered)
            with self.assertRaisesRegex(TemporalZoneError, "outbox is incompatible"):
                TemporalZoneAssembler(config=config, state=valid_but_wrong_zone)

            oversized_path = state_path.with_name("oversized.json")
            with (
                patch("baby_monitor.temporal_zone.MAX_TEMPORAL_FILE_BYTES", 10),
                self.assertRaisesRegex(TemporalZoneError, "persistence limit"),
            ):
                loaded.write_atomic(oversized_path)
            self.assertFalse(oversized_path.exists())

    def test_contract_schema_and_signal_schema_are_exact(self) -> None:
        observation = self._observation(0, 0.3)
        self.assertEqual(observation, TrackedObjectObservation.from_dict(observation.to_dict()))
        config = self._config()
        self.assertEqual(config, TemporalZoneConfig.from_dict(config.to_dict()))
        observation_schema = json.loads(
            Path("schemas/tracked-object-observation-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(observation_schema["required"]), set(observation.to_dict()))
        config_schema = json.loads(
            Path("schemas/temporal-zone-config-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(config_schema["required"]), set(config.to_dict()))
        state_schema = json.loads(
            Path("schemas/temporal-zone-state-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(state_schema["required"]),
            set(TemporalZoneState.empty(config).to_dict()),
        )
        signal_schema = json.loads(
            Path("schemas/signal-event-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {item.value for item in SignalType},
            set(signal_schema["properties"]["signal_type"]["enum"]),
        )
        signal = TemporalZoneAssembler(config=config).process(observation)[0]
        self.assertEqual(signal, SignalEvent.from_dict(signal.to_dict()))
        fixture = load_tracked_observations("tests/fixtures/valid_tracked_object_observations.json")
        self.assertEqual(4, len(fixture))
        self.assertEqual(
            "example_zone",
            TemporalZoneConfig.read("config/temporal-zones.example.json").zones[0].zone_id,
        )

    def test_zone_signal_stays_a_system_fact_and_does_not_invent_risk(self) -> None:
        signal = TemporalZoneAssembler(config=self._config()).process(self._observation(0, 0.3))[0]
        candidate = CandidateEvent(
            event_id="temporal_event_01",
            camera_id="cam_01",
            start_seconds=0.0,
            end_seconds=1.0,
            detections=(),
            signals=(signal,),
        )
        with tempfile.TemporaryDirectory(prefix="baby-monitor-temporal-") as directory:
            root = Path(directory).resolve()
            database = EventDatabase(root / "events.sqlite3")
            process_prepared_event(
                candidate=candidate,
                occurred_at=signal.occurred_at,
                source_kind="temporal_signal",
                source_ref="temporal_zone_v1",
                media=EventMedia(
                    clip_path=str(root / "clip.mp4"),
                    frame_paths=(str(root / "frame.jpg"),),
                ),
                local_vlm=MockVLMAdapter(),
                database=database,
                policy=ReviewPolicy(confidence_threshold=0.0),
            )
            stored = database.get_event(candidate.event_id)
            assert stored is not None
            self.assertEqual(EventType.UNKNOWN, stored.result.event_type)
            self.assertEqual(RiskLevel.NONE, stored.result.risk_level)
            self.assertEqual((), stored.result.inferences)
            self.assertTrue(any("zone_enter" in fact for fact in stored.result.observations))

    def test_cli_persists_replays_and_acknowledges_path_free_outbox(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-temporal-") as directory:
            root = Path(directory).resolve()
            config_path = root / "zones.json"
            observations_path = root / "observations.json"
            state_path = root / "state.json"
            config_path.write_text(
                json.dumps(self._config().to_dict()),
                encoding="utf-8",
            )
            observations_path.write_text(
                json.dumps([self._observation(0, 0.3).to_dict()]),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "temporal-zone-run",
                        "--zone-config",
                        str(config_path),
                        "--observations",
                        str(observations_path),
                        "--state",
                        str(state_path),
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(0, code)
            self.assertEqual(1, len(payload["emitted_signals"]))
            self.assertNotIn(str(root), output.getvalue())
            signal_id = payload["emitted_signals"][0]["signal_id"]

            pending_output = io.StringIO()
            with redirect_stdout(pending_output):
                pending_code = main(["temporal-zone-pending", "--state", str(state_path)])
            self.assertEqual(0, pending_code)
            self.assertEqual(1, json.loads(pending_output.getvalue())["pending_signal_count"])
            self.assertNotIn(str(root), pending_output.getvalue())

            ack_output = io.StringIO()
            with redirect_stdout(ack_output):
                ack_code = main(
                    [
                        "temporal-zone-ack",
                        "--state",
                        str(state_path),
                        "--signal-id",
                        signal_id,
                    ]
                )
            self.assertEqual(0, ack_code)
            self.assertEqual(0, json.loads(ack_output.getvalue())["pending_signal_count"])
            self.assertNotIn(str(root), ack_output.getvalue())


if __name__ == "__main__":
    unittest.main()
