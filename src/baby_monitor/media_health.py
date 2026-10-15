from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class MediaHealthError(ValueError):
    pass


class MediaSourceState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    CAPACITY_REACHED = "capacity_reached"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            MediaSourceState.STOPPED,
            MediaSourceState.CAPACITY_REACHED,
            MediaSourceState.FAILED,
        }


class MediaHealthFreshness(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    TERMINAL = "terminal"
    CLOCK_SKEW = "clock_skew"


def _aware(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise MediaHealthError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise MediaHealthError(f"{field} must be an ISO timestamp")
    try:
        return _aware(datetime.fromisoformat(value), field)
    except ValueError as exc:
        raise MediaHealthError(f"{field} must be an ISO timestamp") from exc


@dataclass(frozen=True, slots=True)
class MediaSourceHealth:
    camera_id: str
    source_ref: str
    state: MediaSourceState
    observed_at: datetime
    session_started_at: datetime
    session_ended_at: datetime | None
    segments: int
    coverage_seconds: float
    gap_count: int
    reconnect_count: int
    capacity_reached: bool
    ffmpeg_exit_code: int | None
    last_error_code: str | None
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise MediaHealthError("unsupported media health version")
        if (
            not isinstance(self.camera_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.camera_id) is None
        ):
            raise MediaHealthError("media health camera_id is invalid")
        if not isinstance(self.source_ref, str):
            raise MediaHealthError("media health source_ref must be a redacted RTSP URI")
        parsed = urlsplit(self.source_ref)
        if (
            parsed.scheme.lower() != "rtsp"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise MediaHealthError("media health source_ref must be a redacted RTSP URI")
        if not isinstance(self.state, MediaSourceState):
            raise MediaHealthError("media health state is invalid")
        observed = _aware(self.observed_at, "observed_at")
        started = _aware(self.session_started_at, "session_started_at")
        ended = (
            _aware(self.session_ended_at, "session_ended_at")
            if self.session_ended_at is not None
            else None
        )
        if observed < started or (ended is not None and ended < started):
            raise MediaHealthError("media health timestamps are inconsistent")
        if self.state.terminal != (ended is not None):
            raise MediaHealthError("terminal media health state requires session_ended_at")
        for value, field, maximum in (
            (self.segments, "segments", 10_000_000),
            (self.gap_count, "gap_count", 10_000_000),
            (self.reconnect_count, "reconnect_count", 10_000_000),
        ):
            if type(value) is not int or not 0 <= value <= maximum:
                raise MediaHealthError(f"media health {field} is invalid")
        if (
            type(self.coverage_seconds) not in {int, float}
            or not math.isfinite(self.coverage_seconds)
            or self.coverage_seconds < 0
        ):
            raise MediaHealthError("media health coverage_seconds is invalid")
        if type(self.capacity_reached) is not bool:
            raise MediaHealthError("media health capacity_reached must be a boolean")
        if self.capacity_reached != (self.state is MediaSourceState.CAPACITY_REACHED):
            raise MediaHealthError("media health capacity state is inconsistent")
        if self.ffmpeg_exit_code is not None and type(self.ffmpeg_exit_code) is not int:
            raise MediaHealthError("media health ffmpeg_exit_code is invalid")
        if self.last_error_code is not None and (
            not isinstance(self.last_error_code, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.last_error_code) is None
        ):
            raise MediaHealthError("media health last_error_code is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "source_ref": self.source_ref,
            "state": self.state.value,
            "observed_at": self.observed_at.astimezone(UTC).isoformat(),
            "session_started_at": self.session_started_at.astimezone(UTC).isoformat(),
            "session_ended_at": (
                self.session_ended_at.astimezone(UTC).isoformat()
                if self.session_ended_at is not None
                else None
            ),
            "segments": self.segments,
            "coverage_seconds": self.coverage_seconds,
            "gap_count": self.gap_count,
            "reconnect_count": self.reconnect_count,
            "capacity_reached": self.capacity_reached,
            "ffmpeg_exit_code": self.ffmpeg_exit_code,
            "last_error_code": self.last_error_code,
        }

    @classmethod
    def from_dict(cls, value: Any) -> MediaSourceHealth:
        if not isinstance(value, dict):
            raise MediaHealthError("media health must be an object")
        expected = {
            "version",
            "camera_id",
            "source_ref",
            "state",
            "observed_at",
            "session_started_at",
            "session_ended_at",
            "segments",
            "coverage_seconds",
            "gap_count",
            "reconnect_count",
            "capacity_reached",
            "ffmpeg_exit_code",
            "last_error_code",
        }
        if set(value) != expected:
            raise MediaHealthError("media health fields do not match v1")
        try:
            state = MediaSourceState(value["state"])
        except (TypeError, ValueError) as exc:
            raise MediaHealthError("media health state is invalid") from exc
        ended_value = value["session_ended_at"]
        return cls(
            version=value["version"],
            camera_id=value["camera_id"],
            source_ref=value["source_ref"],
            state=state,
            observed_at=_timestamp(value["observed_at"], "observed_at"),
            session_started_at=_timestamp(value["session_started_at"], "session_started_at"),
            session_ended_at=(
                _timestamp(ended_value, "session_ended_at") if ended_value is not None else None
            ),
            segments=value["segments"],
            coverage_seconds=value["coverage_seconds"],
            gap_count=value["gap_count"],
            reconnect_count=value["reconnect_count"],
            capacity_reached=value["capacity_reached"],
            ffmpeg_exit_code=value["ffmpeg_exit_code"],
            last_error_code=value["last_error_code"],
        )

    @classmethod
    def read(cls, path: str | Path) -> MediaSourceHealth:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MediaHealthError("media health file is missing or invalid") from exc
        return cls.from_dict(value)

    def write_atomic(self, path: str | Path) -> None:
        health_path = Path(path).expanduser()
        health_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = health_path.with_name(f".{health_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, health_path)
        finally:
            if temporary.exists():
                temporary.unlink()


@dataclass(frozen=True, slots=True)
class MediaHealthAssessment:
    camera_id: str
    reported_state: MediaSourceState
    freshness: MediaHealthFreshness
    checked_at: datetime
    observed_at: datetime
    age_seconds: float
    stale_after_seconds: float
    future_tolerance_seconds: float
    operational: bool
    reason_code: str
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "reported_state": self.reported_state.value,
            "freshness": self.freshness.value,
            "checked_at": self.checked_at.astimezone(UTC).isoformat(),
            "observed_at": self.observed_at.astimezone(UTC).isoformat(),
            "age_seconds": self.age_seconds,
            "stale_after_seconds": self.stale_after_seconds,
            "future_tolerance_seconds": self.future_tolerance_seconds,
            "operational": self.operational,
            "reason_code": self.reason_code,
        }


def assess_media_health(
    health: MediaSourceHealth,
    *,
    checked_at: datetime | None = None,
    stale_after_seconds: float = 10.0,
    future_tolerance_seconds: float = 5.0,
) -> MediaHealthAssessment:
    if type(stale_after_seconds) not in {int, float} or not math.isfinite(stale_after_seconds):
        raise MediaHealthError("stale_after_seconds must be finite")
    if not 0 < stale_after_seconds <= 604_800:
        raise MediaHealthError("stale_after_seconds must be between 0 and 604800")
    if type(future_tolerance_seconds) not in {int, float} or not math.isfinite(
        future_tolerance_seconds
    ):
        raise MediaHealthError("future_tolerance_seconds must be finite")
    if not 0 <= future_tolerance_seconds <= 3_600:
        raise MediaHealthError("future_tolerance_seconds must be between 0 and 3600")

    checked = _aware(checked_at or datetime.now(UTC), "checked_at")
    observed = health.observed_at.astimezone(UTC)
    age = (checked - observed).total_seconds()
    if age < -future_tolerance_seconds:
        freshness = MediaHealthFreshness.CLOCK_SKEW
        reason = "observed_at_in_future"
    elif health.state.terminal:
        freshness = MediaHealthFreshness.TERMINAL
        reason = f"terminal_{health.state.value}"
    elif age > stale_after_seconds:
        freshness = MediaHealthFreshness.STALE
        reason = "active_snapshot_stale"
    else:
        freshness = MediaHealthFreshness.FRESH
        reason = f"fresh_{health.state.value}"

    operational = (
        freshness is MediaHealthFreshness.FRESH and health.state is MediaSourceState.RUNNING
    )
    return MediaHealthAssessment(
        camera_id=health.camera_id,
        reported_state=health.state,
        freshness=freshness,
        checked_at=checked,
        observed_at=observed,
        age_seconds=round(age, 3),
        stale_after_seconds=float(stale_after_seconds),
        future_tolerance_seconds=float(future_tolerance_seconds),
        operational=operational,
        reason_code=reason,
    )
