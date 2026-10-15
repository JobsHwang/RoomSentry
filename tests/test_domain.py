from __future__ import annotations

import json
import unittest
from pathlib import Path

from baby_monitor.domain import EventType, SchemaError, VLMEvent

FIXTURES = Path(__file__).parent / "fixtures"


class VLMEventSchemaTests(unittest.TestCase):
    def test_valid_fixture_round_trips(self) -> None:
        raw = (FIXTURES / "valid_vlm_event.json").read_text(encoding="utf-8")
        event = VLMEvent.from_json(raw)
        self.assertEqual(EventType.SLEEPING, event.event_type)
        self.assertEqual(2, len(event.observations))
        self.assertEqual(event, VLMEvent.from_json(event.to_json()))

    def test_extra_key_is_rejected(self) -> None:
        data = json.loads((FIXTURES / "valid_vlm_event.json").read_text(encoding="utf-8"))
        data["untrusted_extra"] = "must fail"
        with self.assertRaisesRegex(SchemaError, "extra"):
            VLMEvent.from_dict(data)

    def test_missing_key_is_rejected(self) -> None:
        data = json.loads((FIXTURES / "valid_vlm_event.json").read_text(encoding="utf-8"))
        del data["observations"]
        with self.assertRaisesRegex(SchemaError, "missing"):
            VLMEvent.from_dict(data)

    def test_non_standard_nan_is_rejected(self) -> None:
        raw = (FIXTURES / "valid_vlm_event.json").read_text(encoding="utf-8")
        raw = raw.replace("0.78", "NaN")
        with self.assertRaises(SchemaError):
            VLMEvent.from_json(raw)

    def test_boolean_confidence_is_rejected(self) -> None:
        data = json.loads((FIXTURES / "valid_vlm_event.json").read_text(encoding="utf-8"))
        data["confidence"] = True
        with self.assertRaisesRegex(SchemaError, "number"):
            VLMEvent.from_dict(data)


if __name__ == "__main__":
    unittest.main()
