from __future__ import annotations

import json
import math
import os
import re
import stat
import uuid
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .candidate_health import (
    CandidateHealthError,
    CandidateHealthLifetime,
    CandidateWorkerHealth,
)


class CandidateSoakError(ValueError):
    """Path-free failure for the manual candidate soak ledger."""


_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_SESSION_ID = re.compile(r"[a-f0-9]{32}")
_MAX_LEDGER_BYTES = 256 * 1024
_MAX_SESSIONS = 32
_DELTA_FIELDS = {
    "sessions_started",
    "sessions_stopped",
    "sessions_failed",
    "unclean_restarts",
    "scans_succeeded",
    "scans_failed",
    "candidate_groups_emitted",
    "candidate_groups_suppressed",
    "scan_duration_sample_count",
    "execution_lock_wait_sample_count",
    "scan_duration_p95_upper_bound_seconds",
    "scan_duration_max_upper_bound_seconds",
    "execution_lock_wait_p95_upper_bound_seconds",
    "execution_lock_wait_max_upper_bound_seconds",
    "approximate_percentile",
    "media_read",
    "paths_disclosed",
}


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CandidateSoakError(f"candidate soak {field} is invalid")
    return value.astimezone(UTC)


def _parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise CandidateSoakError(f"candidate soak {field} is invalid")
    try:
        return _utc(datetime.fromisoformat(value), field)
    except ValueError:
        raise CandidateSoakError(f"candidate soak {field} is invalid") from None


def _counter(value: Any, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 1_000_000_000:
        raise CandidateSoakError(f"candidate soak {field} is invalid")
    return value


def _optional_duration(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 604_800.0
    ):
        raise CandidateSoakError(f"candidate soak {field} is invalid")
    return float(value)


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise CandidateSoakError("candidate soak JSON is invalid")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                CandidateSoakError("candidate soak JSON is invalid")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise CandidateSoakError("candidate soak JSON is invalid") from None


def _private_parent(path: Path) -> Path:
    if os.name != "posix" or not path.is_absolute() or path.name in {"", ".", ".."}:
        raise CandidateSoakError("candidate soak ledger path is unsafe")
    parent = path.parent
    try:
        metadata = parent.lstat()
        if (
            parent.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o777 != 0o700
            or parent.resolve(strict=True) != parent
        ):
            raise CandidateSoakError("candidate soak ledger path is unsafe")
    except OSError:
        raise CandidateSoakError("candidate soak ledger path is unsafe") from None
    return parent


def _read_private_payload(path: Path) -> Any:
    _private_parent(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise CandidateSoakError("candidate soak ledger is unavailable") from None
    except OSError:
        raise CandidateSoakError("candidate soak ledger is unsafe") from None
    try:
        metadata = os.fstat(descriptor)
        path_metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o777 != 0o600
            or metadata.st_size > _MAX_LEDGER_BYTES
            or (metadata.st_dev, metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise CandidateSoakError("candidate soak ledger is unsafe")
        raw = bytearray()
        while len(raw) <= _MAX_LEDGER_BYTES:
            chunk = os.read(descriptor, min(65_536, _MAX_LEDGER_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) > _MAX_LEDGER_BYTES:
            raise CandidateSoakError("candidate soak ledger is unsafe")
    except (OSError, CandidateSoakError):
        raise CandidateSoakError("candidate soak ledger is unsafe") from None
    finally:
        os.close(descriptor)
    try:
        return _strict_json(raw.decode("utf-8"))
    except (CandidateSoakError, UnicodeDecodeError):
        raise CandidateSoakError("candidate soak ledger is invalid") from None


def _write_private_payload(path: Path, payload: dict[str, object]) -> None:
    parent = _private_parent(path)
    if path.exists() or path.is_symlink():
        try:
            metadata = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o777 != 0o600
            ):
                raise CandidateSoakError("candidate soak ledger path is unsafe")
        except OSError:
            raise CandidateSoakError("candidate soak ledger path is unsafe") from None
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_LEDGER_BYTES:
        raise CandidateSoakError("candidate soak ledger exceeded the size limit")
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except (OSError, CandidateSoakError):
        raise CandidateSoakError("candidate soak ledger persistence failed") from None
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _validated_delta(value: Any) -> dict[str, object]:
    candidate_fields = {
        "candidate_groups_emitted",
        "candidate_groups_suppressed",
    }
    if not isinstance(value, dict) or frozenset(value) not in {
        frozenset(_DELTA_FIELDS),
        frozenset(_DELTA_FIELDS - candidate_fields),
    }:
        raise CandidateSoakError("candidate soak delta fields are invalid")
    result = dict(value)
    for field in candidate_fields:
        result.setdefault(field, 0)
    for field in (
        "sessions_started",
        "sessions_stopped",
        "sessions_failed",
        "unclean_restarts",
        "scans_succeeded",
        "scans_failed",
        "candidate_groups_emitted",
        "candidate_groups_suppressed",
        "scan_duration_sample_count",
        "execution_lock_wait_sample_count",
    ):
        result[field] = _counter(result[field], f"delta.{field}")
    for field in (
        "scan_duration_p95_upper_bound_seconds",
        "scan_duration_max_upper_bound_seconds",
        "execution_lock_wait_p95_upper_bound_seconds",
        "execution_lock_wait_max_upper_bound_seconds",
    ):
        result[field] = _optional_duration(result[field], f"delta.{field}")
    if any(result[field] is not True for field in ("approximate_percentile",)):
        raise CandidateSoakError("candidate soak delta flags are invalid")
    if any(result[field] is not False for field in ("media_read", "paths_disclosed")):
        raise CandidateSoakError("candidate soak delta privacy flags are invalid")
    return result


@dataclass(frozen=True, slots=True)
class CandidateSoakCamera:
    camera_id: str
    scanner_identity: str
    detector_identity: str
    baseline: CandidateHealthLifetime
    result: dict[str, object] | None = None

    def __post_init__(self) -> None:
        for value, field in (
            (self.camera_id, "camera_id"),
            (self.scanner_identity, "scanner_identity"),
            (self.detector_identity, "detector_identity"),
        ):
            if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
                raise CandidateSoakError(f"candidate soak {field} is invalid")
        if not isinstance(self.baseline, CandidateHealthLifetime):
            raise CandidateSoakError("candidate soak baseline is invalid")
        if self.result is not None:
            object.__setattr__(self, "result", _validated_delta(self.result))

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "scanner_identity": self.scanner_identity,
            "detector_identity": self.detector_identity,
            "baseline": self.baseline.to_dict(),
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CandidateSoakCamera:
        expected = {
            "camera_id",
            "scanner_identity",
            "detector_identity",
            "baseline",
            "result",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise CandidateSoakError("candidate soak camera fields are invalid")
        try:
            baseline = CandidateHealthLifetime.from_dict(value["baseline"])
        except CandidateHealthError:
            raise CandidateSoakError("candidate soak baseline is invalid") from None
        return cls(
            camera_id=value["camera_id"],
            scanner_identity=value["scanner_identity"],
            detector_identity=value["detector_identity"],
            baseline=baseline,
            result=value["result"],
        )


@dataclass(frozen=True, slots=True)
class CandidateSoakSession:
    session_id: str
    started_at: datetime
    expected_duration_seconds: int
    state: str
    cameras: tuple[CandidateSoakCamera, ...]
    ended_at: datetime | None = None
    attention_required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or _SESSION_ID.fullmatch(self.session_id) is None:
            raise CandidateSoakError("candidate soak session_id is invalid")
        started = _utc(self.started_at, "started_at")
        if (
            type(self.expected_duration_seconds) is not int
            or not 60 <= self.expected_duration_seconds <= 604_800
        ):
            raise CandidateSoakError("candidate soak expected duration is invalid")
        if self.state not in {"running", "completed", "stopped_early", "interrupted"}:
            raise CandidateSoakError("candidate soak state is invalid")
        if not 1 <= len(self.cameras) <= 16 or not all(
            isinstance(item, CandidateSoakCamera) for item in self.cameras
        ):
            raise CandidateSoakError("candidate soak cameras are invalid")
        camera_ids = [item.camera_id for item in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise CandidateSoakError("candidate soak camera IDs must be unique")
        if type(self.attention_required) is not bool:
            raise CandidateSoakError("candidate soak attention flag is invalid")
        ended = _utc(self.ended_at, "ended_at") if self.ended_at is not None else None
        if ended is not None and ended < started:
            raise CandidateSoakError("candidate soak timestamps are inconsistent")
        has_results = all(item.result is not None for item in self.cameras)
        if self.state == "running":
            if ended is not None or has_results or self.attention_required:
                raise CandidateSoakError("candidate soak running state is inconsistent")
        elif self.state == "interrupted":
            if ended is None or has_results or not self.attention_required:
                raise CandidateSoakError("candidate soak interrupted state is inconsistent")
        elif ended is None or not has_results:
            raise CandidateSoakError("candidate soak completed state is inconsistent")
        else:
            derived_attention = self.state != "completed" or any(
                item.result is not None
                and bool(
                    item.result["scans_succeeded"] == 0
                    or item.result["scans_failed"]
                    or item.result["sessions_failed"]
                    or item.result["unclean_restarts"]
                )
                for item in self.cameras
            )
            if self.attention_required is not derived_attention:
                raise CandidateSoakError("candidate soak attention state is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "ended_at": (
                self.ended_at.astimezone(UTC).isoformat() if self.ended_at is not None else None
            ),
            "expected_duration_seconds": self.expected_duration_seconds,
            "state": self.state,
            "attention_required": self.attention_required,
            "cameras": [item.to_dict() for item in self.cameras],
        }

    @classmethod
    def from_dict(cls, value: Any) -> CandidateSoakSession:
        expected = {
            "session_id",
            "started_at",
            "ended_at",
            "expected_duration_seconds",
            "state",
            "attention_required",
            "cameras",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise CandidateSoakError("candidate soak session fields are invalid")
        cameras = value["cameras"]
        if not isinstance(cameras, list):
            raise CandidateSoakError("candidate soak cameras are invalid")
        return cls(
            session_id=value["session_id"],
            started_at=_parse_time(value["started_at"], "started_at"),
            ended_at=(
                _parse_time(value["ended_at"], "ended_at")
                if value["ended_at"] is not None
                else None
            ),
            expected_duration_seconds=value["expected_duration_seconds"],
            state=value["state"],
            attention_required=value["attention_required"],
            cameras=tuple(CandidateSoakCamera.from_dict(item) for item in cameras),
        )


@dataclass(frozen=True, slots=True)
class CandidateSoakLedger:
    updated_at: datetime
    sessions: tuple[CandidateSoakSession, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise CandidateSoakError("unsupported candidate soak ledger version")
        updated = _utc(self.updated_at, "updated_at")
        if len(self.sessions) > _MAX_SESSIONS or not all(
            isinstance(item, CandidateSoakSession) for item in self.sessions
        ):
            raise CandidateSoakError("candidate soak sessions are invalid")
        ids = [item.session_id for item in self.sessions]
        if len(ids) != len(set(ids)):
            raise CandidateSoakError("candidate soak session IDs must be unique")
        if any(
            left.started_at >= right.started_at
            for left, right in zip(self.sessions, self.sessions[1:], strict=False)
        ):
            raise CandidateSoakError("candidate soak sessions are out of order")
        if any(
            left.ended_at is None or left.ended_at > right.started_at
            for left, right in zip(self.sessions, self.sessions[1:], strict=False)
        ):
            raise CandidateSoakError("candidate soak sessions overlap")
        running = [index for index, item in enumerate(self.sessions) if item.state == "running"]
        if running and running != [len(self.sessions) - 1]:
            raise CandidateSoakError("candidate soak running session is inconsistent")
        if self.sessions:
            latest_time = self.sessions[-1].ended_at or self.sessions[-1].started_at
            if latest_time > updated:
                raise CandidateSoakError("candidate soak ledger timestamp is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "updated_at": self.updated_at.astimezone(UTC).isoformat(),
            "sessions": [item.to_dict() for item in self.sessions],
            "media_read": False,
            "paths_disclosed": False,
            "identity_inference": False,
            "risk_assessment": False,
            "notification_authority": False,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CandidateSoakLedger:
        expected = {
            "version",
            "updated_at",
            "sessions",
            "media_read",
            "paths_disclosed",
            "identity_inference",
            "risk_assessment",
            "notification_authority",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise CandidateSoakError("candidate soak ledger fields are invalid")
        if any(
            value[field] is not False
            for field in (
                "media_read",
                "paths_disclosed",
                "identity_inference",
                "risk_assessment",
                "notification_authority",
            )
        ):
            raise CandidateSoakError("candidate soak privacy flags are invalid")
        sessions = value["sessions"]
        if not isinstance(sessions, list):
            raise CandidateSoakError("candidate soak sessions are invalid")
        return cls(
            version=value["version"],
            updated_at=_parse_time(value["updated_at"], "updated_at"),
            sessions=tuple(CandidateSoakSession.from_dict(item) for item in sessions),
        )

    @classmethod
    def read_private(cls, path: str | Path) -> CandidateSoakLedger:
        return cls.from_dict(_read_private_payload(Path(path).expanduser()))

    def write_atomic_private(self, path: str | Path) -> None:
        _write_private_payload(Path(path).expanduser(), self.to_dict())


def _read_health(paths: tuple[str | Path, ...]) -> tuple[CandidateWorkerHealth, ...]:
    if not 1 <= len(paths) <= 16:
        raise CandidateSoakError("candidate soak health inputs are invalid")
    try:
        health = tuple(CandidateWorkerHealth.read_private(path) for path in paths)
    except (CandidateHealthError, OSError):
        raise CandidateSoakError("candidate soak health is unavailable") from None
    ids = [item.camera_id for item in health]
    if len(ids) != len(set(ids)):
        raise CandidateSoakError("candidate soak camera IDs must be unique")
    if any(item.version not in {2, 3} or item.lifetime is None for item in health):
        raise CandidateSoakError("candidate soak requires candidate health v2 or v3")
    return health


def start_candidate_soak(
    *,
    ledger_path: str | Path,
    health_paths: tuple[str | Path, ...],
    started_at: datetime,
    expected_duration_seconds: int = 86_400,
) -> dict[str, object]:
    now = _utc(started_at, "started_at")
    health = _read_health(health_paths)
    if any(item.worker_running for item in health):
        raise CandidateSoakError("candidate soak workers must be stopped before baseline")
    path = Path(ledger_path).expanduser()
    if not path.exists() and not path.is_symlink():
        _private_parent(path)
        ledger = CandidateSoakLedger(updated_at=now)
    else:
        ledger = CandidateSoakLedger.read_private(path)
    sessions = list(ledger.sessions)
    if sessions and sessions[-1].state == "running":
        sessions[-1] = replace(
            sessions[-1],
            ended_at=now,
            state="interrupted",
            attention_required=True,
        )
    cameras = tuple(
        CandidateSoakCamera(
            camera_id=item.camera_id,
            scanner_identity=item.scanner_identity,
            detector_identity=item.detector_identity,
            baseline=item.lifetime,
        )
        for item in health
        if item.lifetime is not None
    )
    session = CandidateSoakSession(
        session_id=uuid.uuid4().hex,
        started_at=now,
        expected_duration_seconds=expected_duration_seconds,
        state="running",
        cameras=cameras,
    )
    sessions.append(session)
    updated = CandidateSoakLedger(updated_at=now, sessions=tuple(sessions[-_MAX_SESSIONS:]))
    updated.write_atomic_private(path)
    return inspect_candidate_soak(ledger_path=path, checked_at=now)


def finish_candidate_soak(
    *,
    ledger_path: str | Path,
    health_paths: tuple[str | Path, ...],
    ended_at: datetime,
) -> dict[str, object]:
    now = _utc(ended_at, "ended_at")
    path = Path(ledger_path).expanduser()
    if not path.exists() and not path.is_symlink():
        return inspect_candidate_soak(ledger_path=path, checked_at=now)
    ledger = CandidateSoakLedger.read_private(path)
    if not ledger.sessions or ledger.sessions[-1].state != "running":
        return inspect_candidate_soak(ledger_path=path, checked_at=now)
    health = _read_health(health_paths)
    if any(item.worker_running for item in health):
        raise CandidateSoakError("candidate soak workers must stop before report")
    by_camera = {item.camera_id: item for item in health}
    session = ledger.sessions[-1]
    if set(by_camera) != {item.camera_id for item in session.cameras}:
        raise CandidateSoakError("candidate soak report cameras do not match baseline")
    cameras: list[CandidateSoakCamera] = []
    attention = False
    for baseline in session.cameras:
        current = by_camera[baseline.camera_id]
        if (
            current.scanner_identity != baseline.scanner_identity
            or current.detector_identity != baseline.detector_identity
            or current.lifetime is None
        ):
            raise CandidateSoakError("candidate soak report identity changed")
        try:
            result = current.lifetime.delta_summary(baseline.baseline)
        except CandidateHealthError:
            raise CandidateSoakError("candidate soak metrics moved backwards") from None
        attention = attention or bool(
            result["scans_succeeded"] == 0
            or result["scans_failed"]
            or result["sessions_failed"]
            or result["unclean_restarts"]
        )
        cameras.append(replace(baseline, result=result))
    elapsed = (now - session.started_at).total_seconds()
    state = "completed" if elapsed >= session.expected_duration_seconds - 5 else "stopped_early"
    attention = attention or state != "completed"
    finished = replace(
        session,
        ended_at=now,
        state=state,
        cameras=tuple(cameras),
        attention_required=attention,
    )
    updated = CandidateSoakLedger(
        updated_at=now,
        sessions=(*ledger.sessions[:-1], finished),
    )
    updated.write_atomic_private(path)
    return inspect_candidate_soak(ledger_path=path, checked_at=now)


def inspect_candidate_soak(
    *,
    ledger_path: str | Path,
    checked_at: datetime,
) -> dict[str, object]:
    now = _utc(checked_at, "checked_at")
    path = Path(ledger_path).expanduser()
    if not path.exists() and not path.is_symlink():
        _private_parent(path)
        return {
            "version": 1,
            "status": "not_initialized",
            "session_count": 0,
            "latest": None,
            "attention_required": False,
            "media_read": False,
            "paths_disclosed": False,
            "identity_inference": False,
            "risk_assessment": False,
            "notification_authority": False,
        }
    try:
        ledger = CandidateSoakLedger.read_private(path)
    except CandidateSoakError:
        return {
            "version": 1,
            "status": "unavailable",
            "session_count": 0,
            "latest": None,
            "attention_required": True,
            "media_read": False,
            "paths_disclosed": False,
            "identity_inference": False,
            "risk_assessment": False,
            "notification_authority": False,
        }
    latest = ledger.sessions[-1] if ledger.sessions else None
    public_latest: dict[str, object] | None = None
    if latest is not None:
        end = latest.ended_at or now
        elapsed = max(0.0, (end - latest.started_at).total_seconds())
        results = [item.result for item in latest.cameras if item.result is not None]
        public_latest = {
            "state": latest.state,
            "started_at": latest.started_at.astimezone(UTC).isoformat(),
            "ended_at": (
                latest.ended_at.astimezone(UTC).isoformat() if latest.ended_at is not None else None
            ),
            "elapsed_seconds": round(elapsed, 3),
            "expected_duration_seconds": latest.expected_duration_seconds,
            "camera_count": len(latest.cameras),
            "attention_required": latest.attention_required,
            "cameras": [
                {
                    "camera_id": item.camera_id,
                    "result": item.result,
                }
                for item in latest.cameras
            ],
            "scans_succeeded": sum(int(item["scans_succeeded"]) for item in results),
            "scans_failed": sum(int(item["scans_failed"]) for item in results),
            "candidate_groups_emitted": sum(
                int(item["candidate_groups_emitted"]) for item in results
            ),
            "candidate_groups_suppressed": sum(
                int(item["candidate_groups_suppressed"]) for item in results
            ),
            "sessions_failed": sum(int(item["sessions_failed"]) for item in results),
            "unclean_restarts": sum(int(item["unclean_restarts"]) for item in results),
        }
    return {
        "version": 1,
        "status": "available",
        "session_count": len(ledger.sessions),
        "latest": public_latest,
        "attention_required": bool(
            public_latest is not None and public_latest["attention_required"]
        ),
        "media_read": False,
        "paths_disclosed": False,
        "identity_inference": False,
        "risk_assessment": False,
        "notification_authority": False,
    }
