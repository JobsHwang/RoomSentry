from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path

from baby_monitor.domain import SchemaError, SignalEvent


class SignalSourceError(RuntimeError):
    pass


class SignalSource(ABC):
    @abstractmethod
    def events(self) -> Sequence[SignalEvent]:
        raise NotImplementedError


class StaticSignalSource(SignalSource):
    def __init__(self, events: Iterable[SignalEvent]):
        self._events = tuple(events)

    def events(self) -> Sequence[SignalEvent]:
        return self._events


class JSONSignalSource(SignalSource):
    """Read one strict v1 object or a bounded array from an explicit local file."""

    def __init__(self, path: str | Path, *, max_events: int = 1_000):
        self.path = Path(path).expanduser()
        if not 1 <= max_events <= 10_000:
            raise ValueError("max_events must be between 1 and 10000")
        self.max_events = max_events

    def events(self) -> Sequence[SignalEvent]:
        if self.path.is_symlink() or not self.path.is_file():
            raise SignalSourceError("signal fixture is missing or unsafe")
        try:
            size = self.path.stat().st_size
        except OSError as exc:
            raise SignalSourceError("signal fixture cannot be inspected") from exc
        if size > 1_048_576:
            raise SignalSourceError("signal fixture exceeds the 1 MiB limit")
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SignalSourceError("signal fixture is not valid JSON") from exc
        values = value if isinstance(value, list) else [value]
        if not 1 <= len(values) <= self.max_events:
            raise SignalSourceError("signal fixture event count is outside the configured limit")
        try:
            return tuple(SignalEvent.from_dict(item) for item in values)
        except SchemaError as exc:
            raise SignalSourceError("signal fixture does not match the v1 contract") from exc


def signal_event_id(event: SignalEvent) -> str:
    identity = "\x00".join(
        [str(event.version), event.source_adapter, event.camera_id, event.signal_id]
    ).encode("utf-8")
    return "sig_" + hashlib.sha256(identity).hexdigest()[:32]


def deduplicate_signals(events: Iterable[SignalEvent]) -> tuple[SignalEvent, ...]:
    unique: dict[str, SignalEvent] = {}
    for event in events:
        event_id = signal_event_id(event)
        existing = unique.get(event_id)
        if existing is not None and existing.to_dict() != event.to_dict():
            raise SignalSourceError("one signal identity has conflicting immutable data")
        unique[event_id] = event
    return tuple(sorted(unique.values(), key=lambda item: (item.occurred_at, item.signal_id)))
