"""SignalEvent v1: discrete, vendor-neutral signal facts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from baby_monitor.domain._shared import (
    SchemaError,
    _confidence,
)
from baby_monitor.domain.enums import (
    SignalType,
)

_SAFE_SIGNAL_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,80}")


_SIGNAL_ATTRIBUTE_KEYS = {
    "channel",
    "direction",
    "duration_ms",
    "object_label",
    "severity_hint",
    "tracking_id",
    "zone_id",
}


@dataclass(frozen=True, slots=True)
class SignalEvent:
    signal_id: str
    camera_id: str
    occurred_at: datetime
    signal_type: SignalType
    source_adapter: str
    confidence: float
    attributes: tuple[tuple[str, str | int], ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise SchemaError("unsupported signal event version")
        for value, field in (
            (self.signal_id, "signal_id"),
            (self.camera_id, "camera_id"),
            (self.source_adapter, "source_adapter"),
        ):
            if not isinstance(value, str) or _SAFE_SIGNAL_IDENTIFIER.fullmatch(value) is None:
                raise SchemaError(f"signal {field} is invalid")
        if not isinstance(self.occurred_at, datetime):
            raise SchemaError("signal occurred_at must be a datetime")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise SchemaError("signal occurred_at must include a timezone offset")
        if not isinstance(self.signal_type, SignalType):
            raise SchemaError("signal signal_type is invalid")
        _confidence(self.confidence, "signal.confidence")
        if not isinstance(self.attributes, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in self.attributes
        ):
            raise SchemaError("signal attributes must be immutable key/value pairs")
        keys = [key for key, _value in self.attributes]
        if len(set(keys)) != len(keys):
            raise SchemaError("signal attributes must have unique keys")
        if any(key not in _SIGNAL_ATTRIBUTE_KEYS for key in keys):
            raise SchemaError("signal attributes contain an unsupported key")
        for key, value in self.attributes:
            if key == "duration_ms":
                if type(value) is not int or not 0 <= value <= 86_400_000:
                    raise SchemaError("signal duration_ms is invalid")
            elif key == "object_label":
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or value != value.strip()
                    or len(value) > 80
                    or any(character in value for character in ("\x00", "\r", "\n"))
                ):
                    raise SchemaError("signal object_label is invalid")
            elif not isinstance(value, str) or _SAFE_SIGNAL_IDENTIFIER.fullmatch(value) is None:
                raise SchemaError("signal string attribute is invalid")
        attributes = dict(self.attributes)
        if "direction" in attributes and attributes["direction"] not in {
            "enter",
            "exit",
            "unknown",
        }:
            raise SchemaError("signal direction is invalid")
        if "severity_hint" in attributes and attributes["severity_hint"] not in {
            "none",
            "low",
            "medium",
            "high",
        }:
            raise SchemaError("signal severity_hint is invalid")

    @classmethod
    def from_dict(cls, data: Any) -> SignalEvent:
        expected = {
            "version",
            "signal_id",
            "camera_id",
            "occurred_at",
            "signal_type",
            "source_adapter",
            "confidence",
            "attributes",
        }
        if not isinstance(data, dict) or set(data) != expected:
            raise SchemaError("signal event must contain exactly the v1 fields")
        if type(data["version"]) is not int or data["version"] != 1:
            raise SchemaError("unsupported signal event version")
        if not all(
            isinstance(data[field], str)
            for field in ("signal_id", "camera_id", "occurred_at", "source_adapter")
        ):
            raise SchemaError("signal identity fields must be strings")
        try:
            occurred_at = datetime.fromisoformat(data["occurred_at"])
            signal_type = SignalType(data["signal_type"])
        except (TypeError, ValueError) as exc:
            raise SchemaError("signal event contains an invalid timestamp or type") from exc
        if occurred_at.tzinfo is None:
            raise SchemaError("signal occurred_at must include a timezone offset")
        attributes = data["attributes"]
        if not isinstance(attributes, dict):
            raise SchemaError("signal attributes must be an object")
        if type(data["confidence"]) not in {int, float}:
            raise SchemaError("signal confidence must be a number")
        return cls(
            version=data["version"],
            signal_id=data["signal_id"],
            camera_id=data["camera_id"],
            occurred_at=occurred_at.astimezone(UTC),
            signal_type=signal_type,
            source_adapter=data["source_adapter"],
            confidence=float(data["confidence"]),
            attributes=tuple(sorted(attributes.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "signal_id": self.signal_id,
            "camera_id": self.camera_id,
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat(),
            "signal_type": self.signal_type.value,
            "source_adapter": self.source_adapter,
            "confidence": self.confidence,
            "attributes": dict(self.attributes),
        }
