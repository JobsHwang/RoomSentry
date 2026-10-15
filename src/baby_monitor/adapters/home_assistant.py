from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from baby_monitor.adapters.signal import SignalSource, SignalSourceError, deduplicate_signals
from baby_monitor.domain import SchemaError, SignalEvent, SignalType


class HomeAssistantSignalError(SignalSourceError):
    pass


_SOURCE_ADAPTER = "home_assistant"
_MAX_MAPPING_BYTES = 256 * 1024
_MAX_EVENT_BYTES = 2 * 1024 * 1024
_MAX_MAPPINGS = 256
_IGNORED_STATES = {"unavailable", "unknown"}
_TRIGGERS = {"rising_edge", "state_change"}


def _exact_object(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise HomeAssistantSignalError(f"{label} fields are invalid")
    return value


def _bounded_json(path: str | Path, *, max_bytes: int, label: str) -> object:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise HomeAssistantSignalError(f"{label} file is missing or unsafe")
    try:
        size = source.stat().st_size
    except OSError:
        raise HomeAssistantSignalError(f"{label} file cannot be inspected") from None
    if not 0 < size <= max_bytes:
        raise HomeAssistantSignalError(f"{label} file size is outside the allowed limit")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise HomeAssistantSignalError(f"{label} file is not valid UTF-8 JSON") from None


@dataclass(frozen=True, slots=True)
class HomeAssistantEntityMapping:
    entity_id: str
    camera_id: str
    signal_type: SignalType
    confidence: float
    trigger: str
    active_states: tuple[str, ...]
    required_event_type: str | None
    attributes: tuple[tuple[str, str | int], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.entity_id, str)
            or not 3 <= len(self.entity_id) <= 255
            or self.entity_id.count(".") != 1
            or any(
                not part
                or any(
                    character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in part
                )
                for part in self.entity_id.split(".")
            )
        ):
            raise HomeAssistantSignalError("HA entity_id is invalid")
        if not isinstance(self.trigger, str) or self.trigger not in _TRIGGERS:
            raise HomeAssistantSignalError("HA mapping trigger is invalid")
        if not isinstance(self.active_states, tuple) or any(
            not isinstance(value, str)
            or not value
            or len(value) > 80
            or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
            for value in self.active_states
        ):
            raise HomeAssistantSignalError("HA mapping active states are invalid")
        if len(set(self.active_states)) != len(self.active_states):
            raise HomeAssistantSignalError("HA mapping active states must be unique")
        if self.trigger == "rising_edge" and not 1 <= len(self.active_states) <= 8:
            raise HomeAssistantSignalError("HA rising edge mapping requires 1 to 8 active states")
        if self.trigger == "state_change" and self.active_states:
            raise HomeAssistantSignalError("HA state change mapping cannot define active states")
        if self.required_event_type is not None and (
            not isinstance(self.required_event_type, str)
            or not self.required_event_type
            or len(self.required_event_type) > 80
            or any(
                character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
                for character in self.required_event_type
            )
        ):
            raise HomeAssistantSignalError("HA required event type is invalid")
        try:
            SignalEvent(
                signal_id="mapping_validation",
                camera_id=self.camera_id,
                occurred_at=datetime(1970, 1, 1, tzinfo=UTC),
                signal_type=self.signal_type,
                source_adapter=_SOURCE_ADAPTER,
                confidence=self.confidence,
                attributes=self.attributes,
            )
        except SchemaError:
            raise HomeAssistantSignalError("HA mapping output fields are invalid") from None

    @classmethod
    def from_dict(cls, value: object) -> HomeAssistantEntityMapping:
        data = _exact_object(
            value,
            {
                "entity_id",
                "camera_id",
                "signal_type",
                "confidence",
                "trigger",
                "active_states",
                "required_event_type",
                "attributes",
            },
            "HA entity mapping",
        )
        if not isinstance(data["active_states"], list):
            raise HomeAssistantSignalError("HA mapping active_states must be an array")
        if not isinstance(data["attributes"], dict):
            raise HomeAssistantSignalError("HA mapping attributes must be an object")
        if type(data["confidence"]) not in {int, float}:
            raise HomeAssistantSignalError("HA mapping confidence must be a number")
        try:
            signal_type = SignalType(data["signal_type"])
        except (TypeError, ValueError):
            raise HomeAssistantSignalError("HA mapping signal type is invalid") from None
        return cls(
            entity_id=data["entity_id"],
            camera_id=data["camera_id"],
            signal_type=signal_type,
            confidence=float(data["confidence"]),
            trigger=data["trigger"],
            active_states=tuple(data["active_states"]),
            required_event_type=data["required_event_type"],
            attributes=tuple(sorted(data["attributes"].items())),
        )


@dataclass(frozen=True, slots=True)
class HomeAssistantSignalMap:
    mappings: tuple[HomeAssistantEntityMapping, ...]
    source_adapter: str = _SOURCE_ADAPTER
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HomeAssistantSignalError("unsupported HA signal map version")
        if self.source_adapter != _SOURCE_ADAPTER:
            raise HomeAssistantSignalError("HA signal map adapter identity is invalid")
        if not isinstance(self.mappings, tuple) or not 1 <= len(self.mappings) <= _MAX_MAPPINGS:
            raise HomeAssistantSignalError("HA signal map must contain 1 to 256 mappings")
        if any(not isinstance(item, HomeAssistantEntityMapping) for item in self.mappings):
            raise HomeAssistantSignalError("HA signal map contains an invalid mapping")
        entity_ids = tuple(item.entity_id for item in self.mappings)
        if len(set(entity_ids)) != len(entity_ids):
            raise HomeAssistantSignalError("HA entity mappings must be unique")

    @classmethod
    def from_dict(cls, value: object) -> HomeAssistantSignalMap:
        data = _exact_object(
            value,
            {"version", "source_adapter", "mappings"},
            "HA signal map",
        )
        if not isinstance(data["mappings"], list):
            raise HomeAssistantSignalError("HA signal mappings must be an array")
        return cls(
            version=data["version"],
            source_adapter=data["source_adapter"],
            mappings=tuple(HomeAssistantEntityMapping.from_dict(item) for item in data["mappings"]),
        )

    @classmethod
    def read(cls, path: str | Path) -> HomeAssistantSignalMap:
        return cls.from_dict(
            _bounded_json(
                path,
                max_bytes=_MAX_MAPPING_BYTES,
                label="HA signal map",
            )
        )

    def by_entity_id(self) -> dict[str, HomeAssistantEntityMapping]:
        return {item.entity_id: item for item in self.mappings}


def _state_value(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise HomeAssistantSignalError(f"mapped HA {label} must be an object or null")
    state = value.get("state")
    if not isinstance(state, str) or not state or len(state) > 256:
        raise HomeAssistantSignalError(f"mapped HA {label} state is invalid")
    if any(ord(character) < 0x20 for character in state):
        raise HomeAssistantSignalError(f"mapped HA {label} state is invalid")
    return state


def _occurred_at(value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 80:
        raise HomeAssistantSignalError("mapped HA event time is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HomeAssistantSignalError("mapped HA event time is invalid") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HomeAssistantSignalError("mapped HA event time must include a timezone")
    return parsed.astimezone(UTC)


def _signal_id(
    mapping: HomeAssistantEntityMapping,
    *,
    event_time: str,
    state: str,
    context_id: str,
) -> str:
    identity = "\x00".join(
        [mapping.entity_id, mapping.camera_id, event_time, state, context_id]
    ).encode("utf-8")
    return "ha_" + hashlib.sha256(identity).hexdigest()[:32]


def normalize_home_assistant_event(
    value: object,
    *,
    mappings: dict[str, HomeAssistantEntityMapping],
) -> SignalEvent | None:
    if not isinstance(value, dict) or value.get("type") != "event":
        return None
    event = value.get("event")
    if not isinstance(event, dict) or event.get("event_type") != "state_changed":
        return None
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    entity_id = data.get("entity_id")
    if not isinstance(entity_id, str) or entity_id not in mappings:
        return None
    mapping = mappings[entity_id]
    new_state_value = data.get("new_state")
    if new_state_value is None:
        return None
    new_state = _state_value(new_state_value, label="new_state")
    old_state = _state_value(data.get("old_state"), label="old_state")
    if new_state is None or new_state.lower() in _IGNORED_STATES:
        return None
    for state_value in (new_state_value, data.get("old_state")):
        if isinstance(state_value, dict):
            nested_entity = state_value.get("entity_id")
            if nested_entity is not None and nested_entity != entity_id:
                raise HomeAssistantSignalError("mapped HA event entity identity is inconsistent")

    if mapping.required_event_type is not None:
        state_attributes = new_state_value.get("attributes")
        if not isinstance(state_attributes, dict):
            raise HomeAssistantSignalError("mapped HA event attributes are invalid")
        observed_event_type = state_attributes.get("event_type")
        if observed_event_type != mapping.required_event_type:
            return None

    if mapping.trigger == "state_change":
        if old_state == new_state:
            return None
    else:
        active = set(mapping.active_states)
        if new_state not in active or old_state in active:
            return None

    event_time = event.get("time_fired")
    occurred_at = _occurred_at(event_time)
    context = event.get("context", {})
    if context is None:
        context = {}
    if not isinstance(context, dict):
        raise HomeAssistantSignalError("mapped HA event context is invalid")
    raw_context_id = context.get("id", "")
    if raw_context_id is None:
        raw_context_id = ""
    if not isinstance(raw_context_id, str) or len(raw_context_id) > 256:
        raise HomeAssistantSignalError("mapped HA event context is invalid")
    signal_id = _signal_id(
        mapping,
        event_time=occurred_at.isoformat(),
        state=new_state,
        context_id=raw_context_id,
    )
    return SignalEvent(
        signal_id=signal_id,
        camera_id=mapping.camera_id,
        occurred_at=occurred_at,
        signal_type=mapping.signal_type,
        source_adapter=_SOURCE_ADAPTER,
        confidence=mapping.confidence,
        attributes=mapping.attributes,
    )


def normalize_home_assistant_events(
    values: Iterable[object],
    *,
    signal_map: HomeAssistantSignalMap,
) -> tuple[SignalEvent, ...]:
    mappings = signal_map.by_entity_id()
    normalized = (
        signal
        for value in values
        if (signal := normalize_home_assistant_event(value, mappings=mappings)) is not None
    )
    return deduplicate_signals(normalized)


class HomeAssistantJSONSignalSource(SignalSource):
    """Normalize a bounded offline capture of official HA WebSocket event envelopes."""

    def __init__(
        self,
        mapping_path: str | Path,
        events_path: str | Path,
        *,
        max_events: int = 1_000,
    ):
        if type(max_events) is not int or not 1 <= max_events <= 10_000:
            raise ValueError("max_events must be between 1 and 10000")
        self.mapping_path = Path(mapping_path).expanduser()
        self.events_path = Path(events_path).expanduser()
        self.max_events = max_events

    def events(self) -> Sequence[SignalEvent]:
        signal_map = HomeAssistantSignalMap.read(self.mapping_path)
        value = _bounded_json(
            self.events_path,
            max_bytes=_MAX_EVENT_BYTES,
            label="HA event capture",
        )
        values = value if isinstance(value, list) else [value]
        if not 1 <= len(values) <= self.max_events:
            raise HomeAssistantSignalError("HA event capture count is outside the configured limit")
        return normalize_home_assistant_events(values, signal_map=signal_map)
