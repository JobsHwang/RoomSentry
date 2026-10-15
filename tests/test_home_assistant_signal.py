from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from baby_monitor.adapters.home_assistant import (
    HomeAssistantJSONSignalSource,
    HomeAssistantSignalError,
    HomeAssistantSignalMap,
    normalize_home_assistant_events,
)
from baby_monitor.cli import main
from baby_monitor.domain import SignalEvent, SignalType


def _mapping_value() -> dict[str, object]:
    return json.loads(
        Path("config/home-assistant-signals.example.json").read_text(encoding="utf-8")
    )


def _event(
    entity_id: str,
    new_state: str,
    *,
    old_state: str | None = "off",
    time_fired: str = "2026-08-03T01:02:03.123456+00:00",
    context_id: str = "private_context_identifier",
    event_type: str | None = None,
) -> dict[str, object]:
    old = None
    if old_state is not None:
        old = {
            "entity_id": entity_id,
            "state": old_state,
            "attributes": {"friendly_name": "Private nursery label"},
        }
    new_attributes = {"friendly_name": "Private nursery label"}
    if event_type is not None:
        new_attributes["event_type"] = event_type
    return {
        "id": 18,
        "type": "event",
        "event": {
            "event_type": "state_changed",
            "data": {
                "entity_id": entity_id,
                "old_state": old,
                "new_state": {
                    "entity_id": entity_id,
                    "state": new_state,
                    "attributes": new_attributes,
                },
            },
            "origin": "LOCAL",
            "time_fired": time_fired,
            "context": {
                "id": context_id,
                "parent_id": None,
                "user_id": "private_user_identifier",
            },
        },
    }


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


class HomeAssistantSignalTests(unittest.TestCase):
    def test_rising_edge_normalizes_to_path_free_signal_and_deduplicates(self) -> None:
        signal_map = HomeAssistantSignalMap.from_dict(_mapping_value())
        event = _event("binary_sensor.camera_01_motion", "on")
        equivalent_time = _event(
            "binary_sensor.camera_01_motion",
            "on",
            time_fired="2026-08-03T01:02:03.123456Z",
        )
        signals = normalize_home_assistant_events(
            [event, event, equivalent_time],
            signal_map=signal_map,
        )

        self.assertEqual(1, len(signals))
        signal = signals[0]
        self.assertEqual(SignalType.MOTION, signal.signal_type)
        self.assertEqual("cam_01", signal.camera_id)
        self.assertEqual("home_assistant", signal.source_adapter)
        self.assertRegex(signal.signal_id, r"^ha_[0-9a-f]{32}$")
        payload = json.dumps(signal.to_dict(), ensure_ascii=False)
        for private_value in (
            "binary_sensor.camera_01_motion",
            "private_context_identifier",
            "Private nursery label",
            "private_user_identifier",
        ):
            self.assertNotIn(private_value, payload)
        self.assertEqual(signal, SignalEvent.from_dict(signal.to_dict()))

    def test_state_change_supports_event_entity_timestamp(self) -> None:
        signal_map = HomeAssistantSignalMap.from_dict(_mapping_value())
        entity_id = "event.camera_01_cry"
        first = _event(
            entity_id,
            "2026-08-03T01:02:03.100000+00:00",
            old_state="2026-08-03T00:59:00.000000+00:00",
            event_type="cry",
        )
        unchanged = _event(
            entity_id,
            "2026-08-03T01:02:03.100000+00:00",
            old_state="2026-08-03T01:02:03.100000+00:00",
            event_type="cry",
        )
        unavailable = _event(entity_id, "unavailable", event_type="cry")
        other_event_type = _event(
            entity_id,
            "2026-08-03T01:03:03.100000+00:00",
            old_state="2026-08-03T01:02:03.100000+00:00",
            event_type="button_press",
        )
        signals = normalize_home_assistant_events(
            [first, unchanged, unavailable, other_event_type],
            signal_map=signal_map,
        )

        self.assertEqual(1, len(signals))
        self.assertEqual(SignalType.HARDWARE_CRY, signals[0].signal_type)
        self.assertEqual(
            {"channel": "main", "severity_hint": "medium"},
            dict(signals[0].attributes),
        )

    def test_unmapped_non_event_and_inactive_messages_are_ignored(self) -> None:
        signal_map = HomeAssistantSignalMap.from_dict(_mapping_value())
        values = [
            {"type": "auth_required", "ha_version": "example"},
            {"type": "event", "event": {"event_type": "other", "data": {}}},
            _event("binary_sensor.unmapped", "on"),
            _event("binary_sensor.camera_01_motion", "off", old_state="on"),
            _event("binary_sensor.camera_01_motion", "on", old_state="on"),
        ]
        self.assertEqual((), normalize_home_assistant_events(values, signal_map=signal_map))

    def test_mapped_malformed_event_fails_without_echoing_private_values(self) -> None:
        signal_map = HomeAssistantSignalMap.from_dict(_mapping_value())
        event = _event("binary_sensor.camera_01_motion", "on")
        event["event"]["time_fired"] = "private invalid timestamp"  # type: ignore[index]
        with self.assertRaises(HomeAssistantSignalError) as raised:
            normalize_home_assistant_events([event], signal_map=signal_map)
        message = str(raised.exception)
        self.assertNotIn("private invalid timestamp", message)
        self.assertNotIn("binary_sensor.camera_01_motion", message)

        event = _event("binary_sensor.camera_01_motion", "on")
        event["event"]["data"]["old_state"]["entity_id"] = "private.mismatch"  # type: ignore[index]
        with self.assertRaisesRegex(HomeAssistantSignalError, "identity"):
            normalize_home_assistant_events([event], signal_map=signal_map)

    def test_mapping_rejects_extra_secret_duplicate_and_trigger_mismatch(self) -> None:
        value = _mapping_value()
        value["token"] = "must-not-be-accepted"
        with self.assertRaisesRegex(HomeAssistantSignalError, "fields"):
            HomeAssistantSignalMap.from_dict(value)

        value = _mapping_value()
        value["mappings"].append(dict(value["mappings"][0]))  # type: ignore[union-attr]
        with self.assertRaisesRegex(HomeAssistantSignalError, "unique"):
            HomeAssistantSignalMap.from_dict(value)

        value = _mapping_value()
        value["mappings"][1]["active_states"] = ["on"]  # type: ignore[index]
        with self.assertRaisesRegex(HomeAssistantSignalError, "cannot define"):
            HomeAssistantSignalMap.from_dict(value)

        value = _mapping_value()
        value["mappings"][0]["attributes"]["token"] = "secret"  # type: ignore[index]
        with self.assertRaisesRegex(HomeAssistantSignalError, "output fields"):
            HomeAssistantSignalMap.from_dict(value)

    def test_file_source_is_bounded_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = _write_json(root / "mapping.json", _mapping_value())
            events = _write_json(
                root / "events.json",
                [_event("binary_sensor.camera_01_motion", "on")],
            )
            self.assertEqual(
                1,
                len(HomeAssistantJSONSignalSource(mapping, events).events()),
            )

            oversized = root / "oversized.json"
            oversized.write_bytes(b" " * (2 * 1024 * 1024 + 1))
            with self.assertRaisesRegex(HomeAssistantSignalError, "size"):
                HomeAssistantJSONSignalSource(mapping, oversized).events()

            link = root / "events-link.json"
            try:
                link.symlink_to(events)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {type(exc).__name__}")
            with self.assertRaisesRegex(HomeAssistantSignalError, "unsafe"):
                HomeAssistantJSONSignalSource(mapping, link).events()

    def test_cli_outputs_only_normalized_signals_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = _write_json(root / "mapping.json", _mapping_value())
            events = _write_json(
                root / "events.json",
                [_event("binary_sensor.camera_01_motion", "on")],
            )
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "normalize-ha-signals",
                        "--mapping",
                        str(mapping),
                        "--events",
                        str(events),
                    ]
                )
            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            result = json.loads(stdout.getvalue())
            self.assertEqual(1, result["emitted_count"])
            self.assertFalse(result["paths_disclosed"])
            self.assertEqual(
                SignalType.MOTION,
                SignalEvent.from_dict(result["signals"][0]).signal_type,
            )
            self.assertNotIn("binary_sensor", stdout.getvalue())
            self.assertNotIn("private_", stdout.getvalue())

    def test_public_schema_and_example_match_runtime_contract(self) -> None:
        schema = json.loads(
            Path("schemas/home-assistant-signal-map-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual("home_assistant", schema["properties"]["source_adapter"]["const"])
        mapping_schema = schema["properties"]["mappings"]["items"]
        self.assertFalse(mapping_schema["additionalProperties"])
        self.assertEqual(
            {item.value for item in SignalType},
            set(mapping_schema["properties"]["signal_type"]["enum"]),
        )
        signal_map = HomeAssistantSignalMap.from_dict(_mapping_value())
        self.assertEqual(2, len(signal_map.mappings))

    def test_adapter_has_no_network_or_credential_handling(self) -> None:
        source = (
            Path("src/baby_monitor/adapters/home_assistant.py").read_text(encoding="utf-8").lower()
        )
        for forbidden in (
            "access_token",
            "authorization",
            "requests",
            "urllib",
            "websocket-client",
            "socket.",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
