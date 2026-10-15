from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .domain import SchemaError, SignalEvent, SignalType

MAX_TEMPORAL_FILE_BYTES = 4 * 1024 * 1024
MAX_TEMPORAL_OBSERVATIONS = 10_000
MAX_TEMPORAL_TRACKS = 10_000
MAX_PENDING_TEMPORAL_SIGNALS = 10_000
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class TemporalZoneError(RuntimeError):
    pass


def _aware(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TemporalZoneError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise TemporalZoneError(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TemporalZoneError(f"{field} must be an ISO 8601 timestamp") from exc
    return _aware(parsed, field)


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise TemporalZoneError(f"{field} is invalid")
    return value


def _safe_label(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 80
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise TemporalZoneError(f"{field} is invalid")
    return value


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemporalZoneError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise TemporalZoneError(f"{field} is outside the allowed range")
    return parsed


def _bbox(value: Any, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise TemporalZoneError(f"{field} must contain four normalized numbers")
    parsed = tuple(_number(item, field, minimum=0.0, maximum=1.0) for item in value)
    x1, y1, x2, y2 = parsed
    if x1 >= x2 or y1 >= y2:
        raise TemporalZoneError(f"{field} must have positive normalized area")
    return parsed


def _strict_json(raw: str, context: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TemporalZoneError(f"{context} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                TemporalZoneError(f"{context} contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise TemporalZoneError(f"{context} is invalid JSON") from exc


def _read_json(path: str | Path, *, context: str) -> Any:
    source = Path(path).expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        if source.stat().st_size > MAX_TEMPORAL_FILE_BYTES:
            raise TemporalZoneError(f"{context} exceeds the 4 MiB limit")
        raw = source.read_text(encoding="utf-8")
    except TemporalZoneError:
        raise
    except (OSError, UnicodeError) as exc:
        raise TemporalZoneError(f"{context} is missing or unsafe") from exc
    return _strict_json(raw, context)


@dataclass(frozen=True, slots=True)
class TrackedObjectObservation:
    observation_id: str
    camera_id: str
    occurred_at: datetime
    source_adapter: str
    tracking_id: str
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TemporalZoneError("unsupported tracked observation version")
        for value, field in (
            (self.observation_id, "observation_id"),
            (self.camera_id, "camera_id"),
            (self.source_adapter, "source_adapter"),
            (self.tracking_id, "tracking_id"),
        ):
            _safe_id(value, field)
        _safe_label(self.label, "label")
        object.__setattr__(self, "occurred_at", _aware(self.occurred_at, "occurred_at"))
        object.__setattr__(
            self,
            "confidence",
            _number(self.confidence, "confidence", minimum=0.0, maximum=1.0),
        )
        if not isinstance(self.bbox_xyxy, tuple):
            raise TemporalZoneError("bbox_xyxy must be an immutable tuple")
        object.__setattr__(self, "bbox_xyxy", _bbox(list(self.bbox_xyxy), "bbox_xyxy"))

    @classmethod
    def from_dict(cls, value: Any) -> TrackedObjectObservation:
        expected = {
            "version",
            "observation_id",
            "camera_id",
            "occurred_at",
            "source_adapter",
            "tracking_id",
            "label",
            "confidence",
            "bbox_xyxy",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise TemporalZoneError("tracked observation fields do not match v1")
        return cls(
            version=value["version"],
            observation_id=value["observation_id"],
            camera_id=value["camera_id"],
            occurred_at=_timestamp(value["occurred_at"], "occurred_at"),
            source_adapter=value["source_adapter"],
            tracking_id=value["tracking_id"],
            label=value["label"],
            confidence=value["confidence"],
            bbox_xyxy=_bbox(value["bbox_xyxy"], "bbox_xyxy"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "observation_id": self.observation_id,
            "camera_id": self.camera_id,
            "occurred_at": self.occurred_at.isoformat(),
            "source_adapter": self.source_adapter,
            "tracking_id": self.tracking_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ZoneDefinition:
    zone_id: str
    labels: tuple[str, ...]
    bbox_xyxy: tuple[float, float, float, float]
    dwell_seconds: float
    exit_margin: float
    minimum_confidence: float = 0.0

    def __post_init__(self) -> None:
        _safe_id(self.zone_id, "zone_id")
        if (
            not isinstance(self.labels, tuple)
            or not 1 <= len(self.labels) <= 32
            or tuple(sorted(set(self.labels))) != self.labels
        ):
            raise TemporalZoneError("zone labels must be sorted and unique")
        for label in self.labels:
            _safe_label(label, "zone label")
        if not isinstance(self.bbox_xyxy, tuple):
            raise TemporalZoneError("zone bbox_xyxy must be an immutable tuple")
        object.__setattr__(self, "bbox_xyxy", _bbox(list(self.bbox_xyxy), "zone bbox_xyxy"))
        object.__setattr__(
            self,
            "dwell_seconds",
            _number(
                self.dwell_seconds,
                "dwell_seconds",
                minimum=0.1,
                maximum=86_400.0,
            ),
        )
        object.__setattr__(
            self,
            "exit_margin",
            _number(self.exit_margin, "exit_margin", minimum=0.0, maximum=0.25),
        )
        object.__setattr__(
            self,
            "minimum_confidence",
            _number(
                self.minimum_confidence,
                "minimum_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )

    @classmethod
    def from_dict(cls, value: Any) -> ZoneDefinition:
        expected = {
            "zone_id",
            "labels",
            "bbox_xyxy",
            "dwell_seconds",
            "exit_margin",
        }
        if not isinstance(value, dict) or set(value) not in {
            frozenset(expected),
            frozenset((*expected, "minimum_confidence")),
        }:
            raise TemporalZoneError("zone definition fields are invalid")
        labels = value["labels"]
        if not isinstance(labels, list):
            raise TemporalZoneError("zone labels must be an array")
        return cls(
            zone_id=value["zone_id"],
            labels=tuple(labels),
            bbox_xyxy=_bbox(value["bbox_xyxy"], "zone bbox_xyxy"),
            dwell_seconds=value["dwell_seconds"],
            exit_margin=value["exit_margin"],
            minimum_confidence=value.get("minimum_confidence", 0.0),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "zone_id": self.zone_id,
            "labels": list(self.labels),
            "bbox_xyxy": list(self.bbox_xyxy),
            "dwell_seconds": self.dwell_seconds,
            "exit_margin": self.exit_margin,
            "minimum_confidence": self.minimum_confidence,
        }

    def contains(self, bbox: tuple[float, float, float, float], *, expanded: bool) -> bool:
        x1, y1, x2, y2 = self.bbox_xyxy
        margin = self.exit_margin if expanded else 0.0
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        return max(0.0, x1 - margin) <= center_x <= min(1.0, x2 + margin) and max(
            0.0, y1 - margin
        ) <= center_y <= min(1.0, y2 + margin)


@dataclass(frozen=True, slots=True)
class TemporalZoneConfig:
    zones: tuple[ZoneDefinition, ...]
    track_expiry_seconds: float = 300.0
    max_processed_observations: int = 10_000
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TemporalZoneError("unsupported temporal zone config version")
        if (
            not isinstance(self.zones, tuple)
            or not 1 <= len(self.zones) <= 64
            or any(not isinstance(zone, ZoneDefinition) for zone in self.zones)
            or tuple(sorted(zone.zone_id for zone in self.zones))
            != tuple(zone.zone_id for zone in self.zones)
            or len({zone.zone_id for zone in self.zones}) != len(self.zones)
        ):
            raise TemporalZoneError("zones must be sorted, unique, and bounded")
        object.__setattr__(
            self,
            "track_expiry_seconds",
            _number(
                self.track_expiry_seconds,
                "track_expiry_seconds",
                minimum=1.0,
                maximum=86_400.0,
            ),
        )
        if (
            type(self.max_processed_observations) is not int
            or not 1 <= self.max_processed_observations <= MAX_TEMPORAL_OBSERVATIONS
        ):
            raise TemporalZoneError("max_processed_observations is invalid")

    @classmethod
    def from_dict(cls, value: Any) -> TemporalZoneConfig:
        expected = {
            "version",
            "track_expiry_seconds",
            "max_processed_observations",
            "zones",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise TemporalZoneError("temporal zone config fields do not match v1")
        zones = value["zones"]
        if not isinstance(zones, list):
            raise TemporalZoneError("zones must be an array")
        return cls(
            version=value["version"],
            track_expiry_seconds=value["track_expiry_seconds"],
            max_processed_observations=value["max_processed_observations"],
            zones=tuple(ZoneDefinition.from_dict(item) for item in zones),
        )

    @classmethod
    def read(cls, path: str | Path) -> TemporalZoneConfig:
        return cls.from_dict(_read_json(path, context="temporal zone config"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "track_expiry_seconds": self.track_expiry_seconds,
            "max_processed_observations": self.max_processed_observations,
            "zones": [zone.to_dict() for zone in self.zones],
        }

    @property
    def identity(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ProcessedObservation:
    observation_id: str
    fingerprint: str

    def __post_init__(self) -> None:
        _safe_id(self.observation_id, "processed observation_id")
        if not isinstance(self.fingerprint, str) or _SHA256.fullmatch(self.fingerprint) is None:
            raise TemporalZoneError("processed observation fingerprint is invalid")

    @classmethod
    def from_dict(cls, value: Any) -> ProcessedObservation:
        if not isinstance(value, dict) or set(value) != {"observation_id", "fingerprint"}:
            raise TemporalZoneError("processed observation fields are invalid")
        return cls(value["observation_id"], value["fingerprint"])

    def to_dict(self) -> dict[str, str]:
        return {
            "observation_id": self.observation_id,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ZoneTrackState:
    camera_id: str
    tracking_id: str
    zone_id: str
    label: str
    inside: bool
    entered_at: datetime | None
    last_inside_at: datetime | None
    last_seen_at: datetime
    dwell_emitted: bool

    @property
    def key(self) -> tuple[str, str, str]:
        return self.camera_id, self.tracking_id, self.zone_id

    def __post_init__(self) -> None:
        for value, field in (
            (self.camera_id, "track camera_id"),
            (self.tracking_id, "track tracking_id"),
            (self.zone_id, "track zone_id"),
        ):
            _safe_id(value, field)
        _safe_label(self.label, "track label")
        if type(self.inside) is not bool or type(self.dwell_emitted) is not bool:
            raise TemporalZoneError("track boolean fields are invalid")
        object.__setattr__(self, "last_seen_at", _aware(self.last_seen_at, "last_seen_at"))
        entered = _aware(self.entered_at, "entered_at") if self.entered_at is not None else None
        last_inside = (
            _aware(self.last_inside_at, "last_inside_at")
            if self.last_inside_at is not None
            else None
        )
        object.__setattr__(self, "entered_at", entered)
        object.__setattr__(self, "last_inside_at", last_inside)
        if self.inside and (entered is None or last_inside is None):
            raise TemporalZoneError("inside track requires entry and inside timestamps")
        if not self.inside and (entered is not None or last_inside is not None):
            raise TemporalZoneError("outside track cannot retain entry timestamps")
        if last_inside is not None and (last_inside < entered or last_inside > self.last_seen_at):
            raise TemporalZoneError("track timestamps are inconsistent")

    @classmethod
    def from_dict(cls, value: Any) -> ZoneTrackState:
        expected = {
            "camera_id",
            "tracking_id",
            "zone_id",
            "label",
            "inside",
            "entered_at",
            "last_inside_at",
            "last_seen_at",
            "dwell_emitted",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise TemporalZoneError("zone track state fields are invalid")
        return cls(
            camera_id=value["camera_id"],
            tracking_id=value["tracking_id"],
            zone_id=value["zone_id"],
            label=value["label"],
            inside=value["inside"],
            entered_at=(
                _timestamp(value["entered_at"], "entered_at")
                if value["entered_at"] is not None
                else None
            ),
            last_inside_at=(
                _timestamp(value["last_inside_at"], "last_inside_at")
                if value["last_inside_at"] is not None
                else None
            ),
            last_seen_at=_timestamp(value["last_seen_at"], "last_seen_at"),
            dwell_emitted=value["dwell_emitted"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "tracking_id": self.tracking_id,
            "zone_id": self.zone_id,
            "label": self.label,
            "inside": self.inside,
            "entered_at": self.entered_at.isoformat() if self.entered_at else None,
            "last_inside_at": (self.last_inside_at.isoformat() if self.last_inside_at else None),
            "last_seen_at": self.last_seen_at.isoformat(),
            "dwell_emitted": self.dwell_emitted,
        }


@dataclass(frozen=True, slots=True)
class TemporalZoneState:
    config_identity: str
    updated_at: datetime | None
    tracks: tuple[ZoneTrackState, ...]
    processed_observations: tuple[ProcessedObservation, ...]
    pending_signals: tuple[SignalEvent, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TemporalZoneError("unsupported temporal zone state version")
        if (
            not isinstance(self.config_identity, str)
            or _SHA256.fullmatch(self.config_identity) is None
        ):
            raise TemporalZoneError("temporal zone config identity is invalid")
        updated = _aware(self.updated_at, "updated_at") if self.updated_at is not None else None
        object.__setattr__(self, "updated_at", updated)
        if (
            not isinstance(self.tracks, tuple)
            or len(self.tracks) > MAX_TEMPORAL_TRACKS
            or any(not isinstance(track, ZoneTrackState) for track in self.tracks)
        ):
            raise TemporalZoneError("temporal zone tracks are invalid or unbounded")
        if len({track.key for track in self.tracks}) != len(self.tracks):
            raise TemporalZoneError("temporal zone track keys must be unique")
        if (
            not isinstance(self.processed_observations, tuple)
            or len(self.processed_observations) > MAX_TEMPORAL_OBSERVATIONS
            or any(
                not isinstance(item, ProcessedObservation) for item in self.processed_observations
            )
            or len({item.observation_id for item in self.processed_observations})
            != len(self.processed_observations)
        ):
            raise TemporalZoneError("processed observations are invalid or unbounded")
        if (
            not isinstance(self.pending_signals, tuple)
            or len(self.pending_signals) > MAX_PENDING_TEMPORAL_SIGNALS
            or any(not isinstance(signal, SignalEvent) for signal in self.pending_signals)
            or len({signal.signal_id for signal in self.pending_signals})
            != len(self.pending_signals)
        ):
            raise TemporalZoneError("pending temporal signals are invalid or unbounded")
        for signal in self.pending_signals:
            attributes = dict(signal.attributes)
            expected_keys: set[str]
            if signal.signal_type is SignalType.ZONE_ENTER:
                expected_keys = {
                    "direction",
                    "object_label",
                    "tracking_id",
                    "zone_id",
                }
                valid_direction = attributes.get("direction") == "enter"
            elif signal.signal_type is SignalType.ZONE_DWELL:
                expected_keys = {
                    "duration_ms",
                    "object_label",
                    "tracking_id",
                    "zone_id",
                }
                valid_direction = True
            elif signal.signal_type is SignalType.ZONE_EXIT:
                expected_keys = {
                    "direction",
                    "duration_ms",
                    "object_label",
                    "tracking_id",
                    "zone_id",
                }
                valid_direction = attributes.get("direction") == "exit"
            else:
                raise TemporalZoneError("temporal outbox signal type is invalid")
            if (
                signal.source_adapter != "temporal_zone_v1"
                or set(attributes) != expected_keys
                or not valid_direction
            ):
                raise TemporalZoneError("temporal outbox signal provenance is invalid")
        if updated is None and (self.tracks or self.processed_observations or self.pending_signals):
            raise TemporalZoneError("non-empty temporal state requires updated_at")
        if updated is not None and any(track.last_seen_at > updated for track in self.tracks):
            raise TemporalZoneError("temporal state updated_at precedes a track")

    @classmethod
    def empty(cls, config: TemporalZoneConfig) -> TemporalZoneState:
        return cls(config.identity, None, (), (), ())

    @classmethod
    def from_dict(cls, value: Any) -> TemporalZoneState:
        expected = {
            "version",
            "config_identity",
            "updated_at",
            "tracks",
            "processed_observations",
            "pending_signals",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise TemporalZoneError("temporal zone state fields do not match v1")
        if not all(
            isinstance(value[field], list)
            for field in ("tracks", "processed_observations", "pending_signals")
        ):
            raise TemporalZoneError("temporal zone state arrays are invalid")
        try:
            pending = tuple(SignalEvent.from_dict(item) for item in value["pending_signals"])
        except SchemaError as exc:
            raise TemporalZoneError("pending temporal signal is invalid") from exc
        return cls(
            version=value["version"],
            config_identity=value["config_identity"],
            updated_at=(
                _timestamp(value["updated_at"], "updated_at")
                if value["updated_at"] is not None
                else None
            ),
            tracks=tuple(ZoneTrackState.from_dict(item) for item in value["tracks"]),
            processed_observations=tuple(
                ProcessedObservation.from_dict(item) for item in value["processed_observations"]
            ),
            pending_signals=pending,
        )

    @classmethod
    def read(cls, path: str | Path) -> TemporalZoneState:
        return cls.from_dict(_read_json(path, context="temporal zone state"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "config_identity": self.config_identity,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tracks": [track.to_dict() for track in self.tracks],
            "processed_observations": [item.to_dict() for item in self.processed_observations],
            "pending_signals": [signal.to_dict() for signal in self.pending_signals],
        }

    def write_atomic(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        payload = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if len(payload) > MAX_TEMPORAL_FILE_BYTES:
            raise TemporalZoneError("temporal zone state exceeds the 4 MiB persistence limit")
        try:
            parent = target.parent.resolve(strict=True)
            if parent != target.parent.absolute() or not parent.is_dir() or target.is_symlink():
                raise OSError
            if target.exists() and not target.is_file():
                raise OSError
        except OSError as exc:
            raise TemporalZoneError("temporal zone state path is unsafe") from exc
        temporary = parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            raise TemporalZoneError("temporal zone state could not be written") from exc
        finally:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    def acknowledge(self, signal_ids: tuple[str, ...]) -> TemporalZoneState:
        if not signal_ids or len(set(signal_ids)) != len(signal_ids):
            raise TemporalZoneError("signal acknowledgements must be non-empty and unique")
        for signal_id in signal_ids:
            _safe_id(signal_id, "signal_id")
        available = {signal.signal_id for signal in self.pending_signals}
        if not set(signal_ids).issubset(available):
            raise TemporalZoneError("cannot acknowledge an unknown pending signal")
        acknowledged = set(signal_ids)
        return replace(
            self,
            pending_signals=tuple(
                signal for signal in self.pending_signals if signal.signal_id not in acknowledged
            ),
        )


def load_tracked_observations(path: str | Path) -> tuple[TrackedObjectObservation, ...]:
    value = _read_json(path, context="tracked observation fixture")
    values = value if isinstance(value, list) else [value]
    if not 1 <= len(values) <= MAX_TEMPORAL_OBSERVATIONS:
        raise TemporalZoneError("tracked observation count is outside the allowed range")
    observations = tuple(TrackedObjectObservation.from_dict(item) for item in values)
    ids = [item.observation_id for item in observations]
    if len(set(ids)) != len(ids):
        raise TemporalZoneError("tracked observation fixture contains duplicate identities")
    return observations


class TemporalZoneAssembler:
    def __init__(
        self,
        *,
        config: TemporalZoneConfig,
        state: TemporalZoneState | None = None,
    ):
        self.config = config
        self._state = state or TemporalZoneState.empty(config)
        if self._state.config_identity != config.identity:
            raise TemporalZoneError("temporal zone state belongs to a different config")
        zone_by_id = {zone.zone_id: zone for zone in config.zones}
        for track in self._state.tracks:
            zone = zone_by_id.get(track.zone_id)
            if zone is None or track.label not in zone.labels:
                raise TemporalZoneError("temporal zone state is incompatible with the config")
        for signal in self._state.pending_signals:
            attributes = dict(signal.attributes)
            zone = zone_by_id.get(str(attributes["zone_id"]))
            if zone is None or str(attributes["object_label"]) not in zone.labels:
                raise TemporalZoneError("temporal outbox is incompatible with the config")

    @property
    def state(self) -> TemporalZoneState:
        return self._state

    @staticmethod
    def _signal(
        observation: TrackedObjectObservation,
        zone: ZoneDefinition,
        signal_type: SignalType,
        *,
        duration_ms: int | None = None,
    ) -> SignalEvent:
        identity = "\0".join([observation.observation_id, zone.zone_id, signal_type.value]).encode(
            "utf-8"
        )
        attributes: dict[str, str | int] = {
            "object_label": observation.label,
            "tracking_id": observation.tracking_id,
            "zone_id": zone.zone_id,
        }
        if signal_type is SignalType.ZONE_ENTER:
            attributes["direction"] = "enter"
        elif signal_type is SignalType.ZONE_EXIT:
            attributes["direction"] = "exit"
        if duration_ms is not None:
            attributes["duration_ms"] = min(86_400_000, max(0, duration_ms))
        return SignalEvent(
            signal_id="tz_" + hashlib.sha256(identity).hexdigest()[:32],
            camera_id=observation.camera_id,
            occurred_at=observation.occurred_at,
            signal_type=signal_type,
            source_adapter="temporal_zone_v1",
            confidence=observation.confidence,
            attributes=tuple(sorted(attributes.items())),
        )

    def process(self, observation: TrackedObjectObservation) -> tuple[SignalEvent, ...]:
        processed = {
            item.observation_id: item.fingerprint for item in self._state.processed_observations
        }
        existing_fingerprint = processed.get(observation.observation_id)
        if existing_fingerprint is not None:
            if existing_fingerprint != observation.fingerprint:
                raise TemporalZoneError("one observation identity has conflicting immutable data")
            return ()

        expiry_before = observation.occurred_at - timedelta(
            seconds=self.config.track_expiry_seconds
        )
        tracks = {
            track.key: track for track in self._state.tracks if track.last_seen_at >= expiry_before
        }
        matching_track_states = [
            track
            for track in tracks.values()
            if track.camera_id == observation.camera_id
            and track.tracking_id == observation.tracking_id
        ]
        if any(track.label != observation.label for track in matching_track_states):
            raise TemporalZoneError("one tracking_id changed object label before expiry")
        if any(track.last_seen_at > observation.occurred_at for track in matching_track_states):
            raise TemporalZoneError("tracked observations must be monotonic per track")

        emitted: list[SignalEvent] = []
        for zone in self.config.zones:
            if (
                observation.label not in zone.labels
                or observation.confidence < zone.minimum_confidence
            ):
                continue
            key = (observation.camera_id, observation.tracking_id, zone.zone_id)
            previous = tracks.get(key)
            inside = zone.contains(
                observation.bbox_xyxy,
                expanded=previous.inside if previous is not None else False,
            )
            if previous is None:
                current = ZoneTrackState(
                    camera_id=observation.camera_id,
                    tracking_id=observation.tracking_id,
                    zone_id=zone.zone_id,
                    label=observation.label,
                    inside=inside,
                    entered_at=observation.occurred_at if inside else None,
                    last_inside_at=observation.occurred_at if inside else None,
                    last_seen_at=observation.occurred_at,
                    dwell_emitted=False,
                )
                if inside:
                    emitted.append(self._signal(observation, zone, SignalType.ZONE_ENTER))
            elif previous.inside and not inside:
                assert previous.entered_at is not None
                assert previous.last_inside_at is not None
                confirmed_duration = previous.last_inside_at - previous.entered_at
                emitted.append(
                    self._signal(
                        observation,
                        zone,
                        SignalType.ZONE_EXIT,
                        duration_ms=round(confirmed_duration.total_seconds() * 1000),
                    )
                )
                current = replace(
                    previous,
                    inside=False,
                    entered_at=None,
                    last_inside_at=None,
                    last_seen_at=observation.occurred_at,
                    dwell_emitted=False,
                )
            elif not previous.inside and inside:
                emitted.append(self._signal(observation, zone, SignalType.ZONE_ENTER))
                current = replace(
                    previous,
                    inside=True,
                    entered_at=observation.occurred_at,
                    last_inside_at=observation.occurred_at,
                    last_seen_at=observation.occurred_at,
                    dwell_emitted=False,
                )
            elif inside:
                assert previous.entered_at is not None
                elapsed = observation.occurred_at - previous.entered_at
                dwell_emitted = previous.dwell_emitted
                if not dwell_emitted and elapsed.total_seconds() >= zone.dwell_seconds:
                    emitted.append(
                        self._signal(
                            observation,
                            zone,
                            SignalType.ZONE_DWELL,
                            duration_ms=round(elapsed.total_seconds() * 1000),
                        )
                    )
                    dwell_emitted = True
                current = replace(
                    previous,
                    last_inside_at=observation.occurred_at,
                    last_seen_at=observation.occurred_at,
                    dwell_emitted=dwell_emitted,
                )
            else:
                current = replace(previous, last_seen_at=observation.occurred_at)
            tracks[key] = current

        processed_items = list(self._state.processed_observations)
        processed_items.append(
            ProcessedObservation(observation.observation_id, observation.fingerprint)
        )
        processed_items = processed_items[-self.config.max_processed_observations :]
        pending = list(self._state.pending_signals)
        pending_ids = {signal.signal_id for signal in pending}
        for signal in emitted:
            if signal.signal_id not in pending_ids:
                pending.append(signal)
                pending_ids.add(signal.signal_id)
        if len(pending) > MAX_PENDING_TEMPORAL_SIGNALS:
            raise TemporalZoneError(
                "temporal signal outbox is full; acknowledge signals before processing more"
            )
        updated_at = (
            max(self._state.updated_at, observation.occurred_at)
            if self._state.updated_at is not None
            else observation.occurred_at
        )
        self._state = TemporalZoneState(
            config_identity=self.config.identity,
            updated_at=updated_at,
            tracks=tuple(sorted(tracks.values(), key=lambda item: item.key)),
            processed_observations=tuple(processed_items),
            pending_signals=tuple(pending),
        )
        return tuple(emitted)

    def process_all(
        self, observations: tuple[TrackedObjectObservation, ...]
    ) -> tuple[SignalEvent, ...]:
        emitted: list[SignalEvent] = []
        for observation in observations:
            emitted.extend(self.process(observation))
        return tuple(emitted)
