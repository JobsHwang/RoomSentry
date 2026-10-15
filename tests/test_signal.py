from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.adapters.signal import (
    JSONSignalSource,
    SignalSourceError,
    deduplicate_signals,
    signal_event_id,
)
from baby_monitor.domain import SchemaError, SignalEvent, SignalType


class SignalContractTests(unittest.TestCase):
    def test_valid_fixture_round_trips_and_matches_public_schema_enum(self) -> None:
        signal = JSONSignalSource(
            "tests/fixtures/valid_signal_event.json",
            max_events=1,
        ).events()[0]
        self.assertEqual(SignalType.HARDWARE_CRY, signal.signal_type)
        self.assertEqual(UTC, signal.occurred_at.tzinfo)
        self.assertEqual(signal, SignalEvent.from_dict(signal.to_dict()))

        schema = json.loads(Path("schemas/signal-event-v1.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {item.value for item in SignalType},
            set(schema["properties"]["signal_type"]["enum"]),
        )
        self.assertFalse(schema["additionalProperties"])

    def test_unsupported_attribute_and_naive_time_are_rejected(self) -> None:
        value = json.loads(
            Path("tests/fixtures/valid_signal_event.json").read_text(encoding="utf-8")
        )
        value["attributes"]["token"] = "must_not_be_stored"
        with self.assertRaisesRegex(SchemaError, "unsupported"):
            SignalEvent.from_dict(value)
        value["attributes"].pop("token")
        value["occurred_at"] = "2026-08-03T01:02:03"
        with self.assertRaisesRegex(SchemaError, "timezone"):
            SignalEvent.from_dict(value)

    def test_direct_construction_rejects_wrong_domain_types(self) -> None:
        with self.assertRaisesRegex(SchemaError, "datetime"):
            SignalEvent(
                signal_id="source_event_01",
                camera_id="cam_01",
                occurred_at="2026-08-03T01:02:03Z",  # type: ignore[arg-type]
                signal_type=SignalType.MOTION,
                source_adapter="fixture",
                confidence=0.8,
            )
        with self.assertRaisesRegex(SchemaError, "signal_type"):
            SignalEvent(
                signal_id="source_event_01",
                camera_id="cam_01",
                occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
                signal_type="motion",  # type: ignore[arg-type]
                source_adapter="fixture",
                confidence=0.8,
            )

    def test_identity_is_stable_and_conflicting_duplicate_fails(self) -> None:
        original = SignalEvent(
            signal_id="source_event_01",
            camera_id="cam_01",
            occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
            signal_type=SignalType.MOTION,
            source_adapter="fixture",
            confidence=0.8,
        )
        changed = replace(
            original,
            occurred_at=datetime(2026, 8, 3, 0, 0, 1, tzinfo=UTC),
            confidence=0.7,
        )
        self.assertEqual(signal_event_id(original), signal_event_id(changed))
        self.assertEqual((original,), deduplicate_signals([original, original]))
        with self.assertRaisesRegex(SignalSourceError, "conflicting"):
            deduplicate_signals([original, changed])

    def test_json_source_is_bounded_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oversized = root / "oversized.json"
            oversized.write_bytes(b" " * 1_048_577)
            with self.assertRaisesRegex(SignalSourceError, "1 MiB"):
                JSONSignalSource(oversized).events()

            target = root / "target.json"
            target.write_text(
                Path("tests/fixtures/valid_signal_event.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            link = root / "link.json"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {type(exc).__name__}")
            with self.assertRaisesRegex(SignalSourceError, "unsafe"):
                JSONSignalSource(link).events()


if __name__ == "__main__":
    unittest.main()
