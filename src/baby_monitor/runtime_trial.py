from __future__ import annotations

import os
import re
import sqlite3
import stat
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .candidate_health import CandidateHealthError
from .candidate_soak import (
    CandidateSoakCamera,
    CandidateSoakError,
    _read_health,
    _read_private_payload,
    _write_private_payload,
)


class RuntimeTrialError(ValueError):
    """Path-free failure for one bounded home-runtime trial ledger."""


_SESSION_ID = re.compile(r"[a-f0-9]{32}")
_SAFE_ERROR = re.compile(r"[a-zA-Z0-9_.-]{1,80}")
_MAX_SESSIONS = 32
_RESULT_FIELDS = {
    "analysis_completed",
    "analysis_failed",
    "analysis_failed_error_counts",
    "notification_delivered",
    "notification_failed",
    "notification_failed_error_counts",
    "primary_sample_count",
    "primary_person_count",
    "primary_clear_count",
    "primary_coverage_gap_count",
    "primary_camera_count",
    "shadow_sample_count",
    "shadow_person_count",
    "shadow_clear_count",
    "shadow_coverage_gap_count",
    "shadow_camera_count",
    "media_read",
    "paths_disclosed",
    "identity_inference",
    "risk_assessment",
    "notification_authority",
}


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RuntimeTrialError(f"runtime trial {field} is invalid")
    return value.astimezone(UTC)


def _parse_time(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeTrialError(f"runtime trial {field} is invalid")
    try:
        return _utc(datetime.fromisoformat(value), field)
    except ValueError:
        raise RuntimeTrialError(f"runtime trial {field} is invalid") from None


def _counter(value: object, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 1_000_000_000:
        raise RuntimeTrialError(f"runtime trial {field} is invalid")
    return value


def _error_counts(value: object, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or len(value) > 64:
        raise RuntimeTrialError(f"runtime trial {field} is invalid")
    result: dict[str, int] = {}
    for code, count in value.items():
        if not isinstance(code, str) or _SAFE_ERROR.fullmatch(code) is None:
            raise RuntimeTrialError(f"runtime trial {field} is invalid")
        parsed = _counter(count, field)
        if parsed < 1:
            raise RuntimeTrialError(f"runtime trial {field} is invalid")
        result[code] = parsed
    return dict(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class RuntimeTrialDatabaseBaseline:
    analysis_completed: int
    analysis_failed: int
    notification_delivered: int
    notification_failed: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _counter(getattr(self, field), f"baseline.{field}")

    def to_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: object) -> RuntimeTrialDatabaseBaseline:
        expected = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != expected:
            raise RuntimeTrialError("runtime trial database baseline is invalid")
        return cls(**{field: value[field] for field in expected})


def _validated_result(value: object, *, camera_count: int) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _RESULT_FIELDS:
        raise RuntimeTrialError("runtime trial result fields are invalid")
    result = dict(value)
    for field in (
        "analysis_completed",
        "analysis_failed",
        "notification_delivered",
        "notification_failed",
        "primary_sample_count",
        "primary_person_count",
        "primary_clear_count",
        "primary_coverage_gap_count",
        "primary_camera_count",
        "shadow_sample_count",
        "shadow_person_count",
        "shadow_clear_count",
        "shadow_coverage_gap_count",
        "shadow_camera_count",
    ):
        result[field] = _counter(result[field], f"result.{field}")
    for prefix in ("primary", "shadow"):
        if result[f"{prefix}_sample_count"] != sum(
            int(result[f"{prefix}_{suffix}_count"])
            for suffix in ("person", "clear", "coverage_gap")
        ):
            raise RuntimeTrialError("runtime trial presence counts are inconsistent")
        if int(result[f"{prefix}_camera_count"]) > camera_count:
            raise RuntimeTrialError("runtime trial presence camera count is invalid")
    for prefix in ("analysis", "notification"):
        errors = _error_counts(
            result[f"{prefix}_failed_error_counts"],
            f"result.{prefix}_failed_error_counts",
        )
        if sum(errors.values()) != result[f"{prefix}_failed"]:
            raise RuntimeTrialError("runtime trial failure counts are inconsistent")
        result[f"{prefix}_failed_error_counts"] = errors
    if any(
        result[field] is not False
        for field in (
            "media_read",
            "paths_disclosed",
            "identity_inference",
            "risk_assessment",
            "notification_authority",
        )
    ):
        raise RuntimeTrialError("runtime trial authority flags are invalid")
    return result


def _derived_codes(
    *,
    state: str,
    shadow_expected: bool,
    cameras: tuple[CandidateSoakCamera, ...],
    result: dict[str, object] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if state == "running":
        return (), ()
    if state == "interrupted":
        return ("trial_interrupted",), ()
    assert result is not None
    attention: list[str] = []
    advisory: list[str] = []
    if state == "stopped_early":
        attention.append("trial_stopped_early")
    if any(
        item.result is not None
        and (
            item.result["scans_succeeded"] == 0
            or item.result["scans_failed"]
            or item.result["sessions_failed"]
            or item.result["unclean_restarts"]
        )
        for item in cameras
    ):
        attention.append("candidate_runtime_failure")
    if result["analysis_failed"]:
        attention.append("analysis_failure")
    if result["notification_failed"]:
        attention.append("notification_failure")
    if result["primary_coverage_gap_count"]:
        attention.append("primary_coverage_gap")
    if result["primary_camera_count"] < len(cameras):
        attention.append("primary_camera_missing")
    if shadow_expected:
        if result["shadow_coverage_gap_count"]:
            advisory.append("shadow_coverage_gap")
        if result["shadow_camera_count"] < len(cameras):
            advisory.append("shadow_camera_missing")
    return tuple(attention), tuple(advisory)


@dataclass(frozen=True, slots=True)
class RuntimeTrialSession:
    session_id: str
    started_at: datetime
    expected_duration_seconds: int
    state: str
    database_baseline: RuntimeTrialDatabaseBaseline
    cameras: tuple[CandidateSoakCamera, ...]
    shadow_expected: bool = False
    ended_at: datetime | None = None
    result: dict[str, object] | None = None
    attention_codes: tuple[str, ...] = ()
    advisory_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or _SESSION_ID.fullmatch(self.session_id) is None:
            raise RuntimeTrialError("runtime trial session_id is invalid")
        started = _utc(self.started_at, "started_at")
        object.__setattr__(self, "started_at", started)
        if (
            type(self.expected_duration_seconds) is not int
            or not 3_600 <= self.expected_duration_seconds <= 604_800
        ):
            raise RuntimeTrialError("runtime trial expected duration is invalid")
        if self.state not in {"running", "completed", "stopped_early", "interrupted"}:
            raise RuntimeTrialError("runtime trial state is invalid")
        if not isinstance(self.database_baseline, RuntimeTrialDatabaseBaseline):
            raise RuntimeTrialError("runtime trial database baseline is invalid")
        if type(self.shadow_expected) is not bool:
            raise RuntimeTrialError("runtime trial shadow expectation is invalid")
        if not 1 <= len(self.cameras) <= 16 or not all(
            isinstance(item, CandidateSoakCamera) for item in self.cameras
        ):
            raise RuntimeTrialError("runtime trial cameras are invalid")
        camera_ids = [item.camera_id for item in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            raise RuntimeTrialError("runtime trial camera IDs must be unique")
        ended = _utc(self.ended_at, "ended_at") if self.ended_at is not None else None
        object.__setattr__(self, "ended_at", ended)
        if ended is not None and ended < started:
            raise RuntimeTrialError("runtime trial timestamps are inconsistent")
        parsed_result = (
            _validated_result(self.result, camera_count=len(self.cameras))
            if self.result is not None
            else None
        )
        object.__setattr__(self, "result", parsed_result)
        any_candidate_results = any(item.result is not None for item in self.cameras)
        all_candidate_results = all(item.result is not None for item in self.cameras)
        if self.state == "running":
            if ended is not None or parsed_result is not None or any_candidate_results:
                raise RuntimeTrialError("runtime trial running state is inconsistent")
        elif self.state == "interrupted":
            if ended is None or parsed_result is not None or any_candidate_results:
                raise RuntimeTrialError("runtime trial interrupted state is inconsistent")
        elif ended is None or parsed_result is None or not all_candidate_results:
            raise RuntimeTrialError("runtime trial completed state is inconsistent")
        expected_attention, expected_advisory = _derived_codes(
            state=self.state,
            shadow_expected=self.shadow_expected,
            cameras=self.cameras,
            result=parsed_result,
        )
        if self.attention_codes != expected_attention or self.advisory_codes != expected_advisory:
            raise RuntimeTrialError("runtime trial result codes are inconsistent")

    @property
    def attention_required(self) -> bool:
        return bool(self.attention_codes)

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at is not None else None,
            "expected_duration_seconds": self.expected_duration_seconds,
            "state": self.state,
            "database_baseline": self.database_baseline.to_dict(),
            "cameras": [item.to_dict() for item in self.cameras],
            "shadow_expected": self.shadow_expected,
            "result": self.result,
            "attention_codes": list(self.attention_codes),
            "advisory_codes": list(self.advisory_codes),
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimeTrialSession:
        expected = {
            "session_id",
            "started_at",
            "ended_at",
            "expected_duration_seconds",
            "state",
            "database_baseline",
            "cameras",
            "shadow_expected",
            "result",
            "attention_codes",
            "advisory_codes",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RuntimeTrialError("runtime trial session fields are invalid")
        cameras = value["cameras"]
        if not isinstance(cameras, list):
            raise RuntimeTrialError("runtime trial cameras are invalid")
        try:
            parsed_cameras = tuple(CandidateSoakCamera.from_dict(item) for item in cameras)
        except (CandidateSoakError, CandidateHealthError):
            raise RuntimeTrialError("runtime trial cameras are invalid") from None
        for field in ("attention_codes", "advisory_codes"):
            codes = value[field]
            if not isinstance(codes, list) or any(
                not isinstance(code, str) or _SAFE_ERROR.fullmatch(code) is None for code in codes
            ):
                raise RuntimeTrialError("runtime trial result codes are invalid")
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
            database_baseline=RuntimeTrialDatabaseBaseline.from_dict(value["database_baseline"]),
            cameras=parsed_cameras,
            shadow_expected=value["shadow_expected"],
            result=value["result"],
            attention_codes=tuple(value["attention_codes"]),
            advisory_codes=tuple(value["advisory_codes"]),
        )


@dataclass(frozen=True, slots=True)
class RuntimeTrialLedger:
    updated_at: datetime
    sessions: tuple[RuntimeTrialSession, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise RuntimeTrialError("unsupported runtime trial ledger version")
        updated = _utc(self.updated_at, "updated_at")
        object.__setattr__(self, "updated_at", updated)
        if len(self.sessions) > _MAX_SESSIONS or not all(
            isinstance(item, RuntimeTrialSession) for item in self.sessions
        ):
            raise RuntimeTrialError("runtime trial sessions are invalid")
        ids = [item.session_id for item in self.sessions]
        if len(ids) != len(set(ids)):
            raise RuntimeTrialError("runtime trial session IDs must be unique")
        if any(
            left.started_at >= right.started_at
            for left, right in zip(self.sessions, self.sessions[1:], strict=False)
        ):
            raise RuntimeTrialError("runtime trial sessions are out of order")
        if any(
            left.ended_at is None or left.ended_at > right.started_at
            for left, right in zip(self.sessions, self.sessions[1:], strict=False)
        ):
            raise RuntimeTrialError("runtime trial sessions overlap")
        running = [index for index, item in enumerate(self.sessions) if item.state == "running"]
        if running and running != [len(self.sessions) - 1]:
            raise RuntimeTrialError("runtime trial running session is inconsistent")
        if self.sessions:
            latest_time = self.sessions[-1].ended_at or self.sessions[-1].started_at
            if latest_time > updated:
                raise RuntimeTrialError("runtime trial ledger timestamp is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "updated_at": self.updated_at.isoformat(),
            "sessions": [item.to_dict() for item in self.sessions],
            "media_read": False,
            "paths_disclosed": False,
            "identity_inference": False,
            "risk_assessment": False,
            "notification_authority": False,
        }

    @classmethod
    def from_dict(cls, value: object) -> RuntimeTrialLedger:
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
            raise RuntimeTrialError("runtime trial ledger fields are invalid")
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
            raise RuntimeTrialError("runtime trial authority flags are invalid")
        sessions = value["sessions"]
        if not isinstance(sessions, list):
            raise RuntimeTrialError("runtime trial sessions are invalid")
        return cls(
            version=value["version"],
            updated_at=_parse_time(value["updated_at"], "updated_at"),
            sessions=tuple(RuntimeTrialSession.from_dict(item) for item in sessions),
        )

    @classmethod
    def read_private(cls, path: str | Path) -> RuntimeTrialLedger:
        try:
            return cls.from_dict(_read_private_payload(Path(path).expanduser()))
        except CandidateSoakError:
            raise RuntimeTrialError("runtime trial ledger is unavailable") from None

    def write_atomic_private(self, path: str | Path) -> None:
        try:
            _write_private_payload(Path(path).expanduser(), self.to_dict())
        except CandidateSoakError:
            raise RuntimeTrialError("runtime trial ledger persistence failed") from None


def _admit_database(path: str | Path) -> Path:
    database = Path(path).expanduser()
    if os.name != "posix" or not database.is_absolute():
        raise RuntimeTrialError("runtime trial database is unavailable")
    try:
        metadata = database.lstat()
        resolved = database.resolve(strict=True)
    except OSError:
        raise RuntimeTrialError("runtime trial database is unavailable") from None
    if (
        database.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or resolved != database
    ):
        raise RuntimeTrialError("runtime trial database is unavailable")
    return database


def _database_baseline(path: Path) -> RuntimeTrialDatabaseBaseline:
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("BEGIN")
        analysis_rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM analysis_jobs GROUP BY status"
        ).fetchall()
        notification_rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM notification_jobs GROUP BY status"
        ).fetchall()
        connection.rollback()
    except (OSError, sqlite3.Error, ValueError):
        raise RuntimeTrialError("runtime trial database is unavailable") from None
    finally:
        if "connection" in locals():
            connection.close()
    analysis = {str(row["status"]): int(row["count"]) for row in analysis_rows}
    notification = {str(row["status"]): int(row["count"]) for row in notification_rows}
    if set(analysis) - {"pending", "running", "completed", "failed"} or set(notification) - {
        "pending",
        "running",
        "delivered",
        "failed",
    }:
        raise RuntimeTrialError("runtime trial database is invalid")
    return RuntimeTrialDatabaseBaseline(
        analysis_completed=analysis.get("completed", 0),
        analysis_failed=analysis.get("failed", 0),
        notification_delivered=notification.get("delivered", 0),
        notification_failed=notification.get("failed", 0),
    )


def _interval_result(
    path: Path,
    *,
    started_at: datetime,
    ended_at: datetime,
    camera_ids: tuple[str, ...],
) -> dict[str, object]:
    placeholders = ",".join("?" for _ in camera_ids)
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("BEGIN")
        required_tables = {"analysis_jobs", "notification_jobs", "presence_facts"}
        present_tables = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not required_tables <= present_tables:
            raise ValueError
        analysis_rows = connection.execute(
            """
            SELECT status, last_error_code, COUNT(*) AS count
            FROM analysis_jobs
            WHERE updated_at >= ? AND updated_at <= ?
              AND status IN ('completed', 'failed')
            GROUP BY status, last_error_code
            """,
            (started_at.isoformat(), ended_at.isoformat()),
        ).fetchall()
        notification_rows = connection.execute(
            """
            SELECT status, last_error_code, COUNT(*) AS count
            FROM notification_jobs
            WHERE updated_at >= ? AND updated_at <= ?
              AND status IN ('delivered', 'failed')
            GROUP BY status, last_error_code
            """,
            (started_at.isoformat(), ended_at.isoformat()),
        ).fetchall()
        presence_rows = connection.execute(
            f"""
            SELECT mode, state, COUNT(*) AS count, COUNT(DISTINCT camera_id) AS camera_count
            FROM presence_facts
            WHERE recorded_at >= ? AND recorded_at <= ?
              AND camera_id IN ({placeholders})
            GROUP BY mode, state
            """,
            (started_at.isoformat(), ended_at.isoformat(), *camera_ids),
        ).fetchall()
        camera_rows = connection.execute(
            f"""
            SELECT mode, COUNT(DISTINCT camera_id) AS camera_count
            FROM presence_facts
            WHERE recorded_at >= ? AND recorded_at <= ?
              AND camera_id IN ({placeholders})
            GROUP BY mode
            """,
            (started_at.isoformat(), ended_at.isoformat(), *camera_ids),
        ).fetchall()
        connection.rollback()
    except (OSError, sqlite3.Error, ValueError):
        raise RuntimeTrialError("runtime trial database is unavailable") from None
    finally:
        if "connection" in locals():
            connection.close()

    analysis_counts = {"completed": 0, "failed": 0}
    analysis_errors: dict[str, int] = {}
    for row in analysis_rows:
        status = str(row["status"])
        count = int(row["count"])
        analysis_counts[status] += count
        if status == "failed":
            code = row["last_error_code"]
            if not isinstance(code, str):
                raise RuntimeTrialError("runtime trial database is invalid")
            analysis_errors[code] = analysis_errors.get(code, 0) + count
    notification_counts = {"delivered": 0, "failed": 0}
    notification_errors: dict[str, int] = {}
    for row in notification_rows:
        status = str(row["status"])
        count = int(row["count"])
        notification_counts[status] += count
        if status == "failed":
            code = row["last_error_code"]
            if not isinstance(code, str):
                raise RuntimeTrialError("runtime trial database is invalid")
            notification_errors[code] = notification_errors.get(code, 0) + count
    presence = {
        (mode, state): 0
        for mode in ("primary", "shadow")
        for state in ("person_detected", "clear", "coverage_gap")
    }
    for row in presence_rows:
        key = (str(row["mode"]), str(row["state"]))
        if key not in presence:
            raise RuntimeTrialError("runtime trial database is invalid")
        presence[key] = int(row["count"])
    presence_cameras = {"primary": 0, "shadow": 0}
    for row in camera_rows:
        mode = str(row["mode"])
        if mode not in presence_cameras:
            raise RuntimeTrialError("runtime trial database is invalid")
        presence_cameras[mode] = int(row["camera_count"])
    result: dict[str, object] = {
        "analysis_completed": analysis_counts["completed"],
        "analysis_failed": analysis_counts["failed"],
        "analysis_failed_error_counts": analysis_errors,
        "notification_delivered": notification_counts["delivered"],
        "notification_failed": notification_counts["failed"],
        "notification_failed_error_counts": notification_errors,
        "media_read": False,
        "paths_disclosed": False,
        "identity_inference": False,
        "risk_assessment": False,
        "notification_authority": False,
    }
    for mode in ("primary", "shadow"):
        prefix = mode
        result[f"{prefix}_person_count"] = presence[(mode, "person_detected")]
        result[f"{prefix}_clear_count"] = presence[(mode, "clear")]
        result[f"{prefix}_coverage_gap_count"] = presence[(mode, "coverage_gap")]
        result[f"{prefix}_sample_count"] = sum(
            int(result[f"{prefix}_{suffix}_count"])
            for suffix in ("person", "clear", "coverage_gap")
        )
        result[f"{prefix}_camera_count"] = presence_cameras[mode]
    return _validated_result(result, camera_count=len(camera_ids))


def start_runtime_trial(
    *,
    ledger_path: str | Path,
    health_paths: tuple[str | Path, ...],
    database_path: str | Path,
    started_at: datetime,
    expected_duration_seconds: int,
    maximum_health_age_seconds: float = 120.0,
    health_wait_seconds: float = 0.0,
    shadow_expected: bool = False,
) -> dict[str, object]:
    now = _utc(started_at, "started_at")
    if (
        isinstance(maximum_health_age_seconds, bool)
        or not isinstance(maximum_health_age_seconds, (int, float))
        or not 1.0 <= float(maximum_health_age_seconds) <= 600.0
    ):
        raise RuntimeTrialError("runtime trial health age is invalid")
    if (
        isinstance(health_wait_seconds, bool)
        or not isinstance(health_wait_seconds, (int, float))
        or not 0.0 <= float(health_wait_seconds) <= 60.0
    ):
        raise RuntimeTrialError("runtime trial health wait is invalid")
    if type(shadow_expected) is not bool:
        raise RuntimeTrialError("runtime trial shadow expectation is invalid")
    deadline = time.monotonic() + float(health_wait_seconds)
    while True:
        health_checked_at = datetime.now(UTC) if health_wait_seconds else now
        try:
            health = _read_health(health_paths)
            health_ready = all(
                item.worker_running
                and item.lifetime is not None
                and 0.0
                <= (health_checked_at - item.checked_at.astimezone(UTC)).total_seconds()
                <= float(maximum_health_age_seconds)
                for item in health
            )
        except CandidateSoakError:
            health_ready = False
        if health_ready:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise RuntimeTrialError("runtime trial candidates must be fresh and running")
        time.sleep(min(0.25, remaining))
    now = health_checked_at
    database = _admit_database(database_path)
    baseline = _database_baseline(database)
    ledger_file = Path(ledger_path).expanduser()
    if ledger_file.exists() or ledger_file.is_symlink():
        ledger = RuntimeTrialLedger.read_private(ledger_file)
    else:
        ledger = RuntimeTrialLedger(updated_at=now)
    sessions = list(ledger.sessions)
    if sessions and sessions[-1].state == "running":
        sessions[-1] = replace(
            sessions[-1],
            ended_at=now,
            state="interrupted",
            attention_codes=("trial_interrupted",),
        )
    session = RuntimeTrialSession(
        session_id=uuid.uuid4().hex,
        started_at=now,
        expected_duration_seconds=expected_duration_seconds,
        state="running",
        database_baseline=baseline,
        shadow_expected=shadow_expected,
        cameras=tuple(
            CandidateSoakCamera(
                camera_id=item.camera_id,
                scanner_identity=item.scanner_identity,
                detector_identity=item.detector_identity,
                baseline=item.lifetime,
            )
            for item in health
            if item.lifetime is not None
        ),
    )
    sessions.append(session)
    updated = RuntimeTrialLedger(updated_at=now, sessions=tuple(sessions[-_MAX_SESSIONS:]))
    updated.write_atomic_private(ledger_file)
    return inspect_runtime_trial(ledger_path=ledger_file, checked_at=now)


def finish_runtime_trial(
    *,
    ledger_path: str | Path,
    health_paths: tuple[str | Path, ...],
    database_path: str | Path,
    ended_at: datetime,
) -> dict[str, object]:
    now = _utc(ended_at, "ended_at")
    ledger_file = Path(ledger_path).expanduser()
    ledger = RuntimeTrialLedger.read_private(ledger_file)
    if not ledger.sessions or ledger.sessions[-1].state != "running":
        return inspect_runtime_trial(ledger_path=ledger_file, checked_at=now)
    try:
        health = _read_health(health_paths)
    except CandidateSoakError:
        raise RuntimeTrialError("runtime trial candidate health is unavailable") from None
    if any(item.worker_running for item in health):
        raise RuntimeTrialError("runtime trial candidates must stop before settlement")
    session = ledger.sessions[-1]
    by_camera = {item.camera_id: item for item in health}
    if set(by_camera) != {item.camera_id for item in session.cameras}:
        raise RuntimeTrialError("runtime trial cameras changed")
    cameras: list[CandidateSoakCamera] = []
    for baseline in session.cameras:
        current = by_camera[baseline.camera_id]
        if (
            current.scanner_identity != baseline.scanner_identity
            or current.detector_identity != baseline.detector_identity
            or current.lifetime is None
        ):
            raise RuntimeTrialError("runtime trial candidate identity changed")
        try:
            delta = current.lifetime.delta_summary(baseline.baseline)
        except CandidateHealthError:
            raise RuntimeTrialError("runtime trial candidate metrics moved backwards") from None
        cameras.append(replace(baseline, result=delta))
    result = _interval_result(
        _admit_database(database_path),
        started_at=session.started_at,
        ended_at=now,
        camera_ids=tuple(item.camera_id for item in session.cameras),
    )
    elapsed = (now - session.started_at).total_seconds()
    state = "completed" if elapsed >= session.expected_duration_seconds - 5 else "stopped_early"
    attention_codes, advisory_codes = _derived_codes(
        state=state,
        shadow_expected=session.shadow_expected,
        cameras=tuple(cameras),
        result=result,
    )
    finished = replace(
        session,
        ended_at=now,
        state=state,
        cameras=tuple(cameras),
        result=result,
        attention_codes=attention_codes,
        advisory_codes=advisory_codes,
    )
    updated = RuntimeTrialLedger(
        updated_at=now,
        sessions=(*ledger.sessions[:-1], finished),
    )
    updated.write_atomic_private(ledger_file)
    return inspect_runtime_trial(ledger_path=ledger_file, checked_at=now)


def inspect_runtime_trial(
    *,
    ledger_path: str | Path,
    checked_at: datetime,
) -> dict[str, object]:
    now = _utc(checked_at, "checked_at")
    path = Path(ledger_path).expanduser()
    common = {
        "version": 1,
        "media_read": False,
        "paths_disclosed": False,
        "identity_inference": False,
        "risk_assessment": False,
        "notification_authority": False,
    }
    if not path.exists() and not path.is_symlink():
        return {
            **common,
            "status": "not_initialized",
            "session_count": 0,
            "latest": None,
            "attention_required": False,
        }
    try:
        ledger = RuntimeTrialLedger.read_private(path)
    except RuntimeTrialError:
        return {
            **common,
            "status": "unavailable",
            "session_count": 0,
            "latest": None,
            "attention_required": True,
        }
    latest = ledger.sessions[-1] if ledger.sessions else None
    public_latest: dict[str, object] | None = None
    if latest is not None:
        end = latest.ended_at or now
        public_latest = {
            "state": latest.state,
            "started_at": latest.started_at.isoformat(),
            "ended_at": latest.ended_at.isoformat() if latest.ended_at is not None else None,
            "elapsed_seconds": round(max(0.0, (end - latest.started_at).total_seconds()), 3),
            "expected_duration_seconds": latest.expected_duration_seconds,
            "camera_count": len(latest.cameras),
            "database_baseline": latest.database_baseline.to_dict(),
            "shadow_expected": latest.shadow_expected,
            "cameras": [
                {"camera_id": item.camera_id, "result": item.result} for item in latest.cameras
            ],
            "result": latest.result,
            "attention_codes": list(latest.attention_codes),
            "advisory_codes": list(latest.advisory_codes),
            "attention_required": latest.attention_required,
        }
    return {
        **common,
        "status": "available",
        "session_count": len(ledger.sessions),
        "latest": public_latest,
        "attention_required": bool(
            public_latest is not None and public_latest["attention_required"]
        ),
    }
