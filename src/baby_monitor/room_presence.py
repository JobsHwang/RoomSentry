from __future__ import annotations

import json
import math
import os
import re
import stat
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from .room_registry import (
    RoomDefinition,
    RoomRegistry,
    RoomRegistryError,
    _read_private_json,
    _require_private_linux_path,
)

MAX_PRESENCE_BATCH_BYTES = 4 * 1024 * 1024
MAX_PRESENCE_SAMPLES = 4096

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


class RoomPresenceError(RuntimeError):
    pass


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise RoomPresenceError(f"{field} is invalid")
    return value


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RoomPresenceError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise RoomPresenceError(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RoomPresenceError(f"{field} must be an ISO 8601 timestamp") from exc
    return _utc(parsed, field)


def _bounded_number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoomPresenceError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise RoomPresenceError(f"{field} is outside the allowed range")
    return result


class StreamPresenceSampleState(StrEnum):
    PERSON_DETECTED = "person_detected"
    CLEAR = "clear"
    COVERAGE_GAP = "coverage_gap"


class RoomPresenceStatus(StrEnum):
    PERSON_PRESENT = "person_present"
    TEMPORARILY_CLEAR = "temporarily_clear"
    UNKNOWN = "unknown"
    COVERAGE_GAP = "coverage_gap"


ROOM_PRESENCE_REASON_CODES = frozenset(
    {
        "all_streams_clear",
        "confirmation_pending",
        "no_samples",
        "person_confirmed",
        "stream_coverage_gap",
        "stream_stale",
    }
)


@dataclass(frozen=True, slots=True)
class StreamPresenceSample:
    sample_id: str
    camera_id: str
    observed_at: datetime
    state: StreamPresenceSampleState
    scanner_id: str
    detector_id: str | None
    source_fingerprint: str
    reason_code: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise RoomPresenceError("unsupported stream presence sample version")
        _safe_id(self.sample_id, "sample_id")
        _safe_id(self.camera_id, "camera_id")
        _safe_id(self.scanner_id, "scanner_id")
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        if not isinstance(self.state, StreamPresenceSampleState):
            raise RoomPresenceError("stream presence sample state is invalid")
        if (
            not isinstance(self.source_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.source_fingerprint) is None
        ):
            raise RoomPresenceError("source_fingerprint is invalid")
        if self.state is StreamPresenceSampleState.COVERAGE_GAP:
            if self.detector_id is not None:
                raise RoomPresenceError("coverage gap cannot claim detector execution")
            _safe_id(self.reason_code, "coverage gap reason_code")
        elif self.reason_code is not None:
            raise RoomPresenceError("successful stream sample cannot carry a failure reason")
        else:
            _safe_id(self.detector_id, "detector_id")

    @classmethod
    def from_dict(cls, value: Any) -> StreamPresenceSample:
        expected = {
            "version",
            "sample_id",
            "camera_id",
            "observed_at",
            "state",
            "scanner_id",
            "detector_id",
            "source_fingerprint",
            "reason_code",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RoomPresenceError("stream presence sample fields do not match v1")
        try:
            state = StreamPresenceSampleState(value["state"])
        except (TypeError, ValueError) as exc:
            raise RoomPresenceError("stream presence sample state is invalid") from exc
        return cls(
            version=value["version"],
            sample_id=value["sample_id"],
            camera_id=value["camera_id"],
            observed_at=_timestamp(value["observed_at"], "observed_at"),
            state=state,
            scanner_id=value["scanner_id"],
            detector_id=value["detector_id"],
            source_fingerprint=value["source_fingerprint"],
            reason_code=value["reason_code"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "sample_id": self.sample_id,
            "camera_id": self.camera_id,
            "observed_at": self.observed_at.isoformat(),
            "state": self.state.value,
            "scanner_id": self.scanner_id,
            "detector_id": self.detector_id,
            "source_fingerprint": self.source_fingerprint,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class StreamPresenceBatch:
    samples: tuple[StreamPresenceSample, ...]
    version: int = 1
    media_read: bool = False
    identity_inference: bool = False
    risk_assessment: bool = False
    notification_authority: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise RoomPresenceError("unsupported stream presence batch version")
        if (
            not isinstance(self.samples, tuple)
            or len(self.samples) > MAX_PRESENCE_SAMPLES
            or any(not isinstance(item, StreamPresenceSample) for item in self.samples)
        ):
            raise RoomPresenceError("stream presence samples are invalid")
        for field in (
            "media_read",
            "identity_inference",
            "risk_assessment",
            "notification_authority",
        ):
            if getattr(self, field) is not False:
                raise RoomPresenceError(f"stream presence batch {field} must remain false")
        sample_ids = tuple(item.sample_id for item in self.samples)
        if len(sample_ids) != len(set(sample_ids)):
            raise RoomPresenceError("stream presence sample IDs must be unique")
        canonical = tuple(
            sorted(
                self.samples,
                key=lambda item: (item.observed_at, item.camera_id, item.sample_id),
            )
        )
        if canonical != self.samples:
            raise RoomPresenceError("stream presence samples must use canonical order")

    @classmethod
    def from_dict(cls, value: Any) -> StreamPresenceBatch:
        expected = {
            "version",
            "samples",
            "media_read",
            "identity_inference",
            "risk_assessment",
            "notification_authority",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RoomPresenceError("stream presence batch fields do not match v1")
        if not isinstance(value["samples"], list):
            raise RoomPresenceError("stream presence samples are invalid")
        return cls(
            version=value["version"],
            samples=tuple(StreamPresenceSample.from_dict(item) for item in value["samples"]),
            media_read=value["media_read"],
            identity_inference=value["identity_inference"],
            risk_assessment=value["risk_assessment"],
            notification_authority=value["notification_authority"],
        )

    @classmethod
    def read_private(cls, path: str | Path) -> StreamPresenceBatch:
        try:
            value = _read_private_json(
                path,
                label="stream presence batch",
                max_bytes=MAX_PRESENCE_BATCH_BYTES,
                require_git_external=False,
            )
        except RoomRegistryError as exc:
            raise RoomPresenceError("stream presence batch is missing or unsafe") from exc
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "samples": [item.to_dict() for item in self.samples],
            "media_read": self.media_read,
            "identity_inference": self.identity_inference,
            "risk_assessment": self.risk_assessment,
            "notification_authority": self.notification_authority,
        }

    def append(
        self,
        sample: StreamPresenceSample,
        *,
        max_samples_per_stream: int = 64,
    ) -> StreamPresenceBatch:
        if not isinstance(sample, StreamPresenceSample):
            raise RoomPresenceError("stream presence sample is invalid")
        if type(max_samples_per_stream) is not int or not 1 <= max_samples_per_stream <= 128:
            raise RoomPresenceError("max_samples_per_stream is outside the allowed range")
        existing = next((item for item in self.samples if item.sample_id == sample.sample_id), None)
        if existing is not None:
            if existing != sample:
                raise RoomPresenceError("stream presence sample ID conflicts with existing facts")
            return self
        combined = self.samples + (sample,)
        retained: list[StreamPresenceSample] = []
        for camera_id in sorted({item.camera_id for item in combined}):
            camera_samples = sorted(
                (item for item in combined if item.camera_id == camera_id),
                key=lambda item: (item.observed_at, item.sample_id),
            )
            retained.extend(camera_samples[-max_samples_per_stream:])
        canonical = tuple(
            sorted(
                retained,
                key=lambda item: (item.observed_at, item.camera_id, item.sample_id),
            )
        )
        if len(canonical) > MAX_PRESENCE_SAMPLES:
            raise RoomPresenceError("stream presence batch exceeds the global sample bound")
        return StreamPresenceBatch(samples=canonical)

    @classmethod
    def append_private(
        cls,
        path: str | Path,
        sample: StreamPresenceSample,
        *,
        max_samples_per_stream: int = 64,
    ) -> StreamPresenceBatch:
        target = Path(path).expanduser()
        _admit_private_output(target, label="stream presence batch")
        if target.exists() or target.is_symlink():
            current = cls.read_private(target)
        else:
            current = cls(samples=())
        updated = current.append(
            sample,
            max_samples_per_stream=max_samples_per_stream,
        )
        if updated is not current:
            _write_private_json(target, updated.to_dict(), label="stream presence batch")
        return updated


def _admit_private_output(path: Path, *, label: str) -> None:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise RoomPresenceError(f"{label} path must be absolute")
    try:
        _require_private_linux_path(path, label=label)
        parent = path.parent
        metadata = parent.lstat()
    except (OSError, RoomRegistryError) as exc:
        raise RoomPresenceError(f"{label} parent is missing or unsafe") from exc
    if parent.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise RoomPresenceError(f"{label} parent is missing or unsafe")
    if os.name == "posix" and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RoomPresenceError(f"{label} parent permissions are unsafe")
    if path.is_symlink():
        raise RoomPresenceError(f"{label} is missing or unsafe")


def _write_private_json(path: Path, value: object, *, label: str) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if not 1 <= len(encoded) <= MAX_PRESENCE_BATCH_BYTES:
        raise RoomPresenceError(f"{label} exceeds the persistence limit")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError as exc:
        raise RoomPresenceError(f"{label} could not be persisted") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class RoomPresencePolicy:
    present_confirmations: int = 2
    clear_confirmations: int = 3
    confirmation_window_seconds: float = 60.0
    present_ttl_seconds: float = 90.0
    freshness_seconds: float = 45.0
    max_samples_per_stream: int = 64
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise RoomPresenceError("unsupported room presence policy version")
        for value, field, minimum, maximum in (
            (self.present_confirmations, "present_confirmations", 1, 10),
            (self.clear_confirmations, "clear_confirmations", 1, 20),
            (self.max_samples_per_stream, "max_samples_per_stream", 1, 256),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise RoomPresenceError(f"{field} is outside the allowed range")
        for value, field, minimum, maximum in (
            (
                self.confirmation_window_seconds,
                "confirmation_window_seconds",
                1.0,
                600.0,
            ),
            (self.present_ttl_seconds, "present_ttl_seconds", 1.0, 1800.0),
            (self.freshness_seconds, "freshness_seconds", 1.0, 600.0),
        ):
            object.__setattr__(
                self,
                field,
                _bounded_number(value, field, minimum=minimum, maximum=maximum),
            )
        if self.present_ttl_seconds < self.freshness_seconds:
            raise RoomPresenceError("present_ttl_seconds cannot be shorter than freshness_seconds")

    @classmethod
    def from_dict(cls, value: Any) -> RoomPresencePolicy:
        expected = {
            "version",
            "present_confirmations",
            "clear_confirmations",
            "confirmation_window_seconds",
            "present_ttl_seconds",
            "freshness_seconds",
            "max_samples_per_stream",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RoomPresenceError("room presence policy fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "present_confirmations": self.present_confirmations,
            "clear_confirmations": self.clear_confirmations,
            "confirmation_window_seconds": self.confirmation_window_seconds,
            "present_ttl_seconds": self.present_ttl_seconds,
            "freshness_seconds": self.freshness_seconds,
            "max_samples_per_stream": self.max_samples_per_stream,
        }


@dataclass(frozen=True, slots=True)
class RoomPresenceAssessment:
    room_id: str
    display_name: str
    status: RoomPresenceStatus
    configured_stream_count: int
    fresh_stream_count: int
    confirmed_person_stream_count: int
    coverage_gap_stream_count: int
    last_confirmed_person_at: datetime | None
    reason_codes: tuple[str, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise RoomPresenceError("unsupported room presence assessment version")
        _safe_id(self.room_id, "assessment room_id")
        if not isinstance(self.display_name, str) or not self.display_name:
            raise RoomPresenceError("assessment display_name is invalid")
        if not isinstance(self.status, RoomPresenceStatus):
            raise RoomPresenceError("room presence status is invalid")
        for value, field in (
            (self.configured_stream_count, "configured_stream_count"),
            (self.fresh_stream_count, "fresh_stream_count"),
            (self.confirmed_person_stream_count, "confirmed_person_stream_count"),
            (self.coverage_gap_stream_count, "coverage_gap_stream_count"),
        ):
            if type(value) is not int or value < 0:
                raise RoomPresenceError(f"{field} is invalid")
        if not 1 <= self.configured_stream_count <= MAX_PRESENCE_SAMPLES:
            raise RoomPresenceError("configured_stream_count is invalid")
        if any(
            value > self.configured_stream_count
            for value in (
                self.fresh_stream_count,
                self.confirmed_person_stream_count,
                self.coverage_gap_stream_count,
            )
        ):
            raise RoomPresenceError("room presence counts are inconsistent")
        if self.last_confirmed_person_at is not None:
            object.__setattr__(
                self,
                "last_confirmed_person_at",
                _utc(self.last_confirmed_person_at, "last_confirmed_person_at"),
            )
        if (
            not isinstance(self.reason_codes, tuple)
            or tuple(sorted(set(self.reason_codes))) != self.reason_codes
            or not self.reason_codes
            or any(item not in ROOM_PRESENCE_REASON_CODES for item in self.reason_codes)
        ):
            raise RoomPresenceError("room presence reason_codes are invalid")
        if self.status is RoomPresenceStatus.PERSON_PRESENT:
            if self.confirmed_person_stream_count < 1 or self.last_confirmed_person_at is None:
                raise RoomPresenceError("person_present assessment lacks confirmed evidence")
        elif self.confirmed_person_stream_count != 0 or self.last_confirmed_person_at is not None:
            raise RoomPresenceError(
                "non-person assessment cannot retain confirmed identity-free presence"
            )
        if (
            self.status is RoomPresenceStatus.TEMPORARILY_CLEAR
            and self.fresh_stream_count != self.configured_stream_count
        ):
            raise RoomPresenceError("temporarily_clear requires every stream to be fresh")
        if self.status is RoomPresenceStatus.COVERAGE_GAP and self.coverage_gap_stream_count < 1:
            raise RoomPresenceError("coverage_gap assessment lacks a coverage gap")

    @classmethod
    def from_dict(cls, value: Any) -> RoomPresenceAssessment:
        expected = {
            "version",
            "room_id",
            "display_name",
            "status",
            "configured_stream_count",
            "fresh_stream_count",
            "confirmed_person_stream_count",
            "coverage_gap_stream_count",
            "last_confirmed_person_at",
            "reason_codes",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RoomPresenceError("room presence assessment fields do not match v1")
        if not isinstance(value["reason_codes"], list):
            raise RoomPresenceError("room presence reason_codes are invalid")
        try:
            status = RoomPresenceStatus(value["status"])
        except (TypeError, ValueError) as exc:
            raise RoomPresenceError("room presence status is invalid") from exc
        last_seen = value["last_confirmed_person_at"]
        return cls(
            version=value["version"],
            room_id=value["room_id"],
            display_name=value["display_name"],
            status=status,
            configured_stream_count=value["configured_stream_count"],
            fresh_stream_count=value["fresh_stream_count"],
            confirmed_person_stream_count=value["confirmed_person_stream_count"],
            coverage_gap_stream_count=value["coverage_gap_stream_count"],
            last_confirmed_person_at=(
                _timestamp(last_seen, "last_confirmed_person_at") if last_seen is not None else None
            ),
            reason_codes=tuple(value["reason_codes"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "room_id": self.room_id,
            "display_name": self.display_name,
            "status": self.status.value,
            "configured_stream_count": self.configured_stream_count,
            "fresh_stream_count": self.fresh_stream_count,
            "confirmed_person_stream_count": self.confirmed_person_stream_count,
            "coverage_gap_stream_count": self.coverage_gap_stream_count,
            "last_confirmed_person_at": (
                self.last_confirmed_person_at.isoformat()
                if self.last_confirmed_person_at is not None
                else None
            ),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class RoomPresenceReport:
    registry_fingerprint: str
    policy: RoomPresencePolicy
    checked_at: datetime
    rooms: tuple[RoomPresenceAssessment, ...]
    version: int = 1
    media_read: bool = False
    paths_disclosed: bool = False
    identity_inference: bool = False
    risk_assessment: bool = False
    notification_authority: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise RoomPresenceError("unsupported room presence report version")
        if (
            not isinstance(self.registry_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.registry_fingerprint) is None
        ):
            raise RoomPresenceError("registry_fingerprint is invalid")
        if not isinstance(self.policy, RoomPresencePolicy):
            raise RoomPresenceError("room presence report policy is invalid")
        object.__setattr__(self, "checked_at", _utc(self.checked_at, "checked_at"))
        if (
            not isinstance(self.rooms, tuple)
            or not self.rooms
            or any(not isinstance(item, RoomPresenceAssessment) for item in self.rooms)
        ):
            raise RoomPresenceError("room presence report rooms are invalid")
        room_ids = tuple(item.room_id for item in self.rooms)
        if len(room_ids) != len(set(room_ids)):
            raise RoomPresenceError("room presence report room IDs must be unique")
        for field in (
            "media_read",
            "paths_disclosed",
            "identity_inference",
            "risk_assessment",
            "notification_authority",
        ):
            if getattr(self, field) is not False:
                raise RoomPresenceError(f"room presence report {field} must remain false")

    @classmethod
    def from_dict(cls, value: Any) -> RoomPresenceReport:
        expected = {
            "version",
            "registry_fingerprint",
            "policy",
            "checked_at",
            "rooms",
            "media_read",
            "paths_disclosed",
            "identity_inference",
            "risk_assessment",
            "notification_authority",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RoomPresenceError("room presence report fields do not match v1")
        if not isinstance(value["rooms"], list):
            raise RoomPresenceError("room presence report rooms are invalid")
        return cls(
            version=value["version"],
            registry_fingerprint=value["registry_fingerprint"],
            policy=RoomPresencePolicy.from_dict(value["policy"]),
            checked_at=_timestamp(value["checked_at"], "checked_at"),
            rooms=tuple(RoomPresenceAssessment.from_dict(item) for item in value["rooms"]),
            media_read=value["media_read"],
            paths_disclosed=value["paths_disclosed"],
            identity_inference=value["identity_inference"],
            risk_assessment=value["risk_assessment"],
            notification_authority=value["notification_authority"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "registry_fingerprint": self.registry_fingerprint,
            "policy": self.policy.to_dict(),
            "checked_at": self.checked_at.isoformat(),
            "rooms": [item.to_dict() for item in self.rooms],
            "media_read": self.media_read,
            "paths_disclosed": self.paths_disclosed,
            "identity_inference": self.identity_inference,
            "risk_assessment": self.risk_assessment,
            "notification_authority": self.notification_authority,
        }


@dataclass(frozen=True, slots=True)
class _StreamAssessment:
    state: str
    last_confirmed_person_at: datetime | None = None
    reason_code: str | None = None


def _tail_run(
    samples: tuple[StreamPresenceSample, ...],
    state: StreamPresenceSampleState,
) -> tuple[StreamPresenceSample, ...]:
    result: list[StreamPresenceSample] = []
    for sample in reversed(samples):
        if sample.state is not state:
            break
        result.append(sample)
    return tuple(reversed(result))


def _run_is_confirmed(
    run: tuple[StreamPresenceSample, ...],
    *,
    confirmations: int,
    confirmation_window_seconds: float,
) -> bool:
    if len(run) < confirmations:
        return False
    selected = run[-confirmations:]
    return (
        selected[-1].observed_at - selected[0].observed_at
    ).total_seconds() <= confirmation_window_seconds


def _assess_stream(
    samples: tuple[StreamPresenceSample, ...],
    *,
    checked_at: datetime,
    policy: RoomPresencePolicy,
) -> _StreamAssessment:
    if not samples:
        return _StreamAssessment(state="gap", reason_code="no_samples")
    latest = samples[-1]
    if checked_at - latest.observed_at > timedelta(seconds=policy.freshness_seconds):
        return _StreamAssessment(state="gap", reason_code="stream_stale")
    if latest.state is StreamPresenceSampleState.COVERAGE_GAP:
        return _StreamAssessment(state="gap", reason_code="stream_coverage_gap")

    since_last_gap: tuple[StreamPresenceSample, ...] = samples
    for index in range(len(samples) - 1, -1, -1):
        if samples[index].state is StreamPresenceSampleState.COVERAGE_GAP:
            since_last_gap = samples[index + 1 :]
            break

    latest_confirmation: datetime | None = None
    person_run: list[StreamPresenceSample] = []
    for sample in since_last_gap:
        if sample.state is StreamPresenceSampleState.PERSON_DETECTED:
            person_run.append(sample)
            if _run_is_confirmed(
                tuple(person_run),
                confirmations=policy.present_confirmations,
                confirmation_window_seconds=policy.confirmation_window_seconds,
            ):
                latest_confirmation = sample.observed_at
        else:
            person_run.clear()

    clear_run = _tail_run(since_last_gap, StreamPresenceSampleState.CLEAR)
    clear_confirmed = _run_is_confirmed(
        clear_run,
        confirmations=policy.clear_confirmations,
        confirmation_window_seconds=policy.confirmation_window_seconds,
    )
    if (
        latest_confirmation is not None
        and not clear_confirmed
        and checked_at - latest_confirmation <= timedelta(seconds=policy.present_ttl_seconds)
    ):
        return _StreamAssessment(
            state="present",
            last_confirmed_person_at=latest_confirmation,
        )
    if clear_confirmed:
        return _StreamAssessment(state="clear")
    return _StreamAssessment(state="unknown", reason_code="confirmation_pending")


def evaluate_room_presence(
    *,
    registry: RoomRegistry,
    batch: StreamPresenceBatch,
    checked_at: datetime,
    policy: RoomPresencePolicy | None = None,
) -> RoomPresenceReport:
    if not isinstance(registry, RoomRegistry):
        raise RoomPresenceError("room registry is invalid")
    if not isinstance(batch, StreamPresenceBatch):
        raise RoomPresenceError("stream presence batch is invalid")
    current = _utc(checked_at, "checked_at")
    active_policy = policy or RoomPresencePolicy()
    if not isinstance(active_policy, RoomPresencePolicy):
        raise RoomPresenceError("room presence policy is invalid")

    known_streams = {item.camera_id for item in registry.logical_streams}
    if any(item.camera_id not in known_streams for item in batch.samples):
        raise RoomPresenceError("stream presence batch references an unregistered camera")
    if any(item.observed_at > current for item in batch.samples):
        raise RoomPresenceError("stream presence sample is in the future")

    by_stream: dict[str, tuple[StreamPresenceSample, ...]] = {}
    for camera_id in known_streams:
        samples = tuple(item for item in batch.samples if item.camera_id == camera_id)
        if len(samples) > active_policy.max_samples_per_stream:
            raise RoomPresenceError("stream presence history exceeds the configured bound")
        by_stream[camera_id] = samples

    assessments: list[RoomPresenceAssessment] = []
    for room in registry.rooms:
        assessments.append(
            _assess_room(
                room,
                registry=registry,
                by_stream=by_stream,
                checked_at=current,
                policy=active_policy,
            )
        )
    return RoomPresenceReport(
        registry_fingerprint=registry.fingerprint,
        policy=active_policy,
        checked_at=current,
        rooms=tuple(assessments),
    )


def _assess_room(
    room: RoomDefinition,
    *,
    registry: RoomRegistry,
    by_stream: dict[str, tuple[StreamPresenceSample, ...]],
    checked_at: datetime,
    policy: RoomPresencePolicy,
) -> RoomPresenceAssessment:
    streams = registry.streams_for_room(room.room_id)
    stream_results = tuple(
        _assess_stream(
            by_stream[item.camera_id],
            checked_at=checked_at,
            policy=policy,
        )
        for item in streams
    )
    present = tuple(item for item in stream_results if item.state == "present")
    gaps = tuple(item for item in stream_results if item.state == "gap")
    clear = tuple(item for item in stream_results if item.state == "clear")
    fresh_count = len(stream_results) - len(gaps)

    if present:
        status = RoomPresenceStatus.PERSON_PRESENT
        last_confirmed = max(
            item.last_confirmed_person_at
            for item in present
            if item.last_confirmed_person_at is not None
        )
        reason_codes = ("person_confirmed",)
    elif gaps:
        status = RoomPresenceStatus.COVERAGE_GAP
        last_confirmed = None
        reason_codes = tuple(sorted({item.reason_code or "stream_coverage_gap" for item in gaps}))
    elif len(clear) == len(stream_results):
        status = RoomPresenceStatus.TEMPORARILY_CLEAR
        last_confirmed = None
        reason_codes = ("all_streams_clear",)
    else:
        status = RoomPresenceStatus.UNKNOWN
        last_confirmed = None
        reason_codes = ("confirmation_pending",)

    return RoomPresenceAssessment(
        room_id=room.room_id,
        display_name=room.display_name,
        status=status,
        configured_stream_count=len(streams),
        fresh_stream_count=fresh_count,
        confirmed_person_stream_count=len(present),
        coverage_gap_stream_count=len(gaps),
        last_confirmed_person_at=last_confirmed,
        reason_codes=tuple(sorted(reason_codes)),
    )
