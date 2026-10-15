from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sqlite3
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .daily_digest import DailyDigestCandidate
from .domain import (
    CandidateEvent,
    DecisionRecord,
    Detection,
    EventMedia,
    RiskLevel,
    SignalEvent,
    SignalType,
)

_MAX_DATABASE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_MEDIA_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_MEDIA_BYTES = 512 * 1024 * 1024
_MAX_EVENTS = 1_000
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_PROBE_OUTPUT_BYTES = 64 * 1024
_SAFE_ERROR = "daily digest source admission failed"
_MEDIA_SUFFIXES = frozenset({".mp4", ".mkv"})
_REQUIRED_COLUMNS = frozenset(
    {
        "event_id",
        "camera_id",
        "occurred_at",
        "risk_level",
        "candidate_json",
        "decision_json",
        "evidence_json",
        "clip_path",
    }
)
_ZONE_SIGNAL_TYPES = frozenset({SignalType.ZONE_ENTER, SignalType.ZONE_DWELL, SignalType.ZONE_EXIT})
_HARDWARE_SIGNAL_TYPES = frozenset(
    {
        SignalType.MOTION,
        SignalType.PERSON,
        SignalType.FACE,
        SignalType.SOUND_PEAK,
        SignalType.HARDWARE_CRY,
        SignalType.FALL_HINT,
        SignalType.MANUAL_TRIGGER,
    }
)


class DailyDigestSourceError(RuntimeError):
    """A fixed, path-free source-admission failure."""


DurationProbe = Callable[[int], float]


@dataclass(frozen=True, slots=True)
class DailyDigestMediaSource:
    event_id: str
    camera_id: str
    clip_path: Path = field(repr=False)
    clip_sha256: str
    clip_bytes: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class DailyDigestAdmission:
    candidates: tuple[DailyDigestCandidate, ...] = field(repr=False)
    local_date: str
    source_event_count: int
    media_files_hashed: int
    media_bytes_hashed: int
    coverage_gap_count: int
    media_sources: tuple[DailyDigestMediaSource, ...] = field(default=(), repr=False)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": "admitted",
            "local_date": self.local_date,
            "source_event_count": self.source_event_count,
            "media_files_hashed": self.media_files_hashed,
            "media_bytes_hashed": self.media_bytes_hashed,
            "coverage_gap_count": self.coverage_gap_count,
            "database_read_only": True,
            "source_hashes_verified": True,
            "durations_probed": True,
            "media_read": self.media_files_hashed > 0,
            "media_content_decoded": False,
            "fact_offsets_admitted": False,
            "paths_disclosed": False,
            "event_ids_disclosed": False,
            "camera_ids_disclosed": False,
            "identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
            "video_generated": False,
        }


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _strict_json(value: object) -> object:
    if not isinstance(value, str):
        raise DailyDigestSourceError(_SAFE_ERROR)
    try:
        encoded_size = len(value.encode("utf-8"))
    except UnicodeError:
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    if not 1 <= encoded_size <= _MAX_JSON_BYTES:
        raise DailyDigestSourceError(_SAFE_ERROR)
    try:
        return json.loads(
            value,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise DailyDigestSourceError(_SAFE_ERROR) from None


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    return hashlib.sha256(payload).hexdigest()


def _admit_database(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    parent = candidate.parent
    try:
        metadata = candidate.lstat()
        parent_metadata = parent.lstat()
        resolved = candidate.resolve(strict=True)
        resolved_parent = parent.resolve(strict=True)
    except OSError:
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not 0 < metadata.st_size <= _MAX_DATABASE_BYTES
        or resolved != candidate
        or parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        or resolved_parent != parent
    ):
        raise DailyDigestSourceError(_SAFE_ERROR)
    return resolved


def _admit_storage_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or resolved != candidate
    ):
        raise DailyDigestSourceError(_SAFE_ERROR)
    return resolved


def _private_media_path(value: object, *, storage_root: Path) -> tuple[Path, os.stat_result]:
    if not isinstance(value, str) or not value or len(value) > 4_096:
        raise DailyDigestSourceError(_SAFE_ERROR)
    candidate = Path(value)
    if not candidate.is_absolute() or candidate.suffix.lower() not in _MEDIA_SUFFIXES:
        raise DailyDigestSourceError(_SAFE_ERROR)
    try:
        relative = candidate.relative_to(storage_root)
    except ValueError:
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    current = storage_root
    for component in relative.parts[:-1]:
        current = current / component
        try:
            directory_metadata = current.lstat()
        except OSError:
            raise DailyDigestSourceError(_SAFE_ERROR) from None
        if (
            current.is_symlink()
            or not stat.S_ISDIR(directory_metadata.st_mode)
            or directory_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(directory_metadata.st_mode) & 0o077
        ):
            raise DailyDigestSourceError(_SAFE_ERROR)
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(storage_root)
    except (OSError, ValueError):
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    if (
        candidate.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not 0 < metadata.st_size <= _MAX_MEDIA_BYTES
        or resolved != candidate
    ):
        raise DailyDigestSourceError(_SAFE_ERROR)
    return resolved, metadata


def _trusted_executable(path: str | Path) -> Path:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not os.access(candidate, os.X_OK)
        or resolved != candidate
    ):
        raise DailyDigestSourceError(_SAFE_ERROR)
    return resolved


def _probe_duration_fd(
    descriptor: int,
    *,
    executable: str | Path = "/usr/bin/ffprobe",
    timeout_seconds: float = 15.0,
) -> float:
    trusted = _trusted_executable(executable)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not 1.0 <= float(timeout_seconds) <= 60.0
    ):
        raise DailyDigestSourceError(_SAFE_ERROR)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        process = subprocess.Popen(
            (
                str(trusted),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                f"/proc/self/fd/{descriptor}",
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            pass_fds=(descriptor,),
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise DailyDigestSourceError(_SAFE_ERROR) from None
    except (OSError, subprocess.SubprocessError):
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    if (
        process.returncode != 0
        or len(stdout) > _MAX_PROBE_OUTPUT_BYTES
        or len(stderr) > _MAX_PROBE_OUTPUT_BYTES
    ):
        raise DailyDigestSourceError(_SAFE_ERROR)
    try:
        payload = json.loads(
            stdout.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(payload, dict) or set(payload) != {"format"}:
            raise ValueError
        format_payload = payload["format"]
        if not isinstance(format_payload, dict) or set(format_payload) != {"duration"}:
            raise ValueError
        duration = float(format_payload["duration"])
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    if not math.isfinite(duration) or not 0.1 <= duration <= 3_600.0:
        raise DailyDigestSourceError(_SAFE_ERROR)
    return duration


def _hash_and_probe(
    path: Path,
    expected: os.stat_result,
    *,
    duration_probe: DurationProbe,
) -> tuple[str, float]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (
                expected.st_dev,
                expected.st_ino,
                expected.st_size,
            ):
                raise DailyDigestSourceError(_SAFE_ERROR)
            digest = hashlib.sha256()
            os.lseek(descriptor, 0, os.SEEK_SET)
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            duration = duration_probe(descriptor)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (expected.st_dev, expected.st_ino, expected.st_size, expected.st_mtime_ns)
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or not 0.1 <= float(duration) <= 3_600.0
    ):
        raise DailyDigestSourceError(_SAFE_ERROR)
    return digest.hexdigest(), float(duration)


def _candidate_event(
    payload: object,
    *,
    event_id: str,
    camera_id: str,
) -> CandidateEvent:
    if not isinstance(payload, dict):
        raise DailyDigestSourceError(_SAFE_ERROR)
    required = {"event_id", "camera_id", "start_seconds", "end_seconds", "detections"}
    optional = {"signals", "detector_id"}
    if not required.issubset(payload) or set(payload) - required - optional:
        raise DailyDigestSourceError(_SAFE_ERROR)
    if payload["event_id"] != event_id or payload["camera_id"] != camera_id:
        raise DailyDigestSourceError(_SAFE_ERROR)
    for key in ("start_seconds", "end_seconds"):
        value = payload[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 86_400.0
        ):
            raise DailyDigestSourceError(_SAFE_ERROR)
    if float(payload["end_seconds"]) < float(payload["start_seconds"]):
        raise DailyDigestSourceError(_SAFE_ERROR)
    raw_detections = payload["detections"]
    if not isinstance(raw_detections, list) or len(raw_detections) > 10_000:
        raise DailyDigestSourceError(_SAFE_ERROR)
    detections: list[Detection] = []
    try:
        for item in raw_detections:
            if not isinstance(item, dict) or set(item) != {
                "timestamp_seconds",
                "label",
                "confidence",
                "bbox_xyxy",
            }:
                raise ValueError
            bbox = item["bbox_xyxy"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError
            detections.append(
                Detection(
                    timestamp_seconds=item["timestamp_seconds"],
                    label=item["label"],
                    confidence=item["confidence"],
                    bbox_xyxy=tuple(bbox),
                )
            )
        raw_signals = payload.get("signals", [])
        if not isinstance(raw_signals, list) or len(raw_signals) > 1_000:
            raise ValueError
        signals = tuple(SignalEvent.from_dict(item) for item in raw_signals)
        if any(signal.camera_id != camera_id for signal in signals):
            raise ValueError
        return CandidateEvent(
            event_id=event_id,
            camera_id=camera_id,
            start_seconds=float(payload["start_seconds"]),
            end_seconds=float(payload["end_seconds"]),
            detections=tuple(detections),
            signals=signals,
            detector_id=payload.get("detector_id"),
        )
    except (TypeError, ValueError):
        raise DailyDigestSourceError(_SAFE_ERROR) from None


def _read_rows(
    database: Path,
    *,
    start: datetime,
    end: datetime,
) -> tuple[sqlite3.Row, ...]:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
        columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(events)").fetchall()
        }
        if not _REQUIRED_COLUMNS.issubset(columns):
            raise DailyDigestSourceError(_SAFE_ERROR)
        rows = connection.execute(
            "SELECT event_id, camera_id, occurred_at, risk_level, candidate_json, "
            "decision_json, evidence_json, clip_path FROM events "
            "WHERE occurred_at >= ? AND occurred_at < ? "
            "ORDER BY occurred_at ASC, event_id ASC LIMIT ?",
            (start.isoformat(), end.isoformat(), _MAX_EVENTS + 1),
        ).fetchall()
    except (sqlite3.Error, DailyDigestSourceError):
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    finally:
        if connection is not None:
            connection.close()
    if len(rows) > _MAX_EVENTS:
        raise DailyDigestSourceError(_SAFE_ERROR)
    return tuple(rows)


def latest_daily_digest_local_date(*, database_path: str | Path, timezone_name: str) -> date:
    database = _admit_database(database_path)
    try:
        timezone = ZoneInfo(timezone_name)
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA query_only = ON")
            row = connection.execute("SELECT MAX(occurred_at) FROM events").fetchone()
        finally:
            connection.close()
        if row is None or not isinstance(row[0], str):
            raise ValueError
        occurred_at = datetime.fromisoformat(row[0])
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise ValueError
    except (sqlite3.Error, TypeError, ValueError, ZoneInfoNotFoundError):
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    return occurred_at.astimezone(timezone).date()


def load_daily_digest_candidates(
    *,
    database_path: str | Path,
    storage_root: str | Path,
    local_date: date,
    timezone_name: str,
    duration_probe: DurationProbe | None = None,
) -> DailyDigestAdmission:
    if not isinstance(local_date, date) or isinstance(local_date, datetime):
        raise DailyDigestSourceError(_SAFE_ERROR)
    try:
        timezone = ZoneInfo(timezone_name)
    except (TypeError, ValueError, ZoneInfoNotFoundError):
        raise DailyDigestSourceError(_SAFE_ERROR) from None
    database = _admit_database(database_path)
    root = _admit_storage_root(storage_root)
    local_start = datetime.combine(local_date, time.min, tzinfo=timezone)
    start = local_start.astimezone(UTC)
    end = (local_start + timedelta(days=1)).astimezone(UTC)
    rows = _read_rows(database, start=start, end=end)
    probe = duration_probe or _probe_duration_fd
    candidates: list[DailyDigestCandidate] = []
    media_sources: list[DailyDigestMediaSource] = []
    total_bytes = 0
    gap_count = 0
    for row in rows:
        event_id = row["event_id"]
        camera_id = row["camera_id"]
        occurred_at_raw = row["occurred_at"]
        if not all(isinstance(value, str) for value in (event_id, camera_id, occurred_at_raw)):
            raise DailyDigestSourceError(_SAFE_ERROR)
        try:
            occurred_at = datetime.fromisoformat(occurred_at_raw)
            risk_level = RiskLevel(row["risk_level"])
        except (TypeError, ValueError):
            raise DailyDigestSourceError(_SAFE_ERROR) from None
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise DailyDigestSourceError(_SAFE_ERROR)
        occurred_at = occurred_at.astimezone(UTC)
        candidate_payload = _strict_json(row["candidate_json"])
        evidence_payload = _strict_json(row["evidence_json"])
        decision_payload = _strict_json(row["decision_json"])
        try:
            candidate = _candidate_event(
                candidate_payload,
                event_id=event_id,
                camera_id=camera_id,
            )
            evidence = EventMedia.from_dict(evidence_payload)
            decision = DecisionRecord.from_dict(decision_payload)
        except (TypeError, ValueError):
            raise DailyDigestSourceError(_SAFE_ERROR) from None
        if (
            evidence.clip_path != row["clip_path"]
            or decision.risk_level is not risk_level
            or not evidence.clip_path
        ):
            raise DailyDigestSourceError(_SAFE_ERROR)
        clip, metadata = _private_media_path(evidence.clip_path, storage_root=root)
        total_bytes += metadata.st_size
        if total_bytes > _MAX_TOTAL_MEDIA_BYTES:
            raise DailyDigestSourceError(_SAFE_ERROR)
        clip_sha256, media_duration = _hash_and_probe(
            clip,
            metadata,
            duration_probe=probe,
        )
        source_binding = _canonical_sha256(
            {
                "event_id": event_id,
                "camera_id": camera_id,
                "occurred_at": occurred_at.isoformat(),
                "candidate_sha256": _canonical_sha256(candidate_payload),
                "evidence_sha256": _canonical_sha256(evidence_payload),
                "decision_sha256": _canonical_sha256(decision_payload),
                "clip_sha256": clip_sha256,
                "clip_bytes": metadata.st_size,
                "duration_seconds": round(media_duration, 6),
            }
        )
        person_confidence = max(
            (
                float(detection.confidence)
                for detection in candidate.detections
                if detection.label == "person"
            ),
            default=0.0,
        )
        motion_score = max(
            (
                float(signal_event.confidence)
                for signal_event in candidate.signals
                if signal_event.signal_type is SignalType.MOTION
            ),
            default=0.0,
        )
        gap_reasons = tuple(sorted({gap.reason for gap in evidence.coverage_gaps}))
        gap_count += len(evidence.coverage_gaps)
        candidates.append(
            DailyDigestCandidate(
                event_id=event_id,
                camera_id=camera_id,
                occurred_at=occurred_at,
                source_binding_sha256=source_binding,
                media_duration_seconds=media_duration,
                fact_offsets_seconds=(),
                person_confidence=person_confidence,
                motion_score=motion_score,
                hardware_signal=any(
                    item.signal_type in _HARDWARE_SIGNAL_TYPES for item in candidate.signals
                ),
                zone_signal=any(
                    item.signal_type in _ZONE_SIGNAL_TYPES for item in candidate.signals
                ),
                risk_level=risk_level,
                human_keep=False,
                anonymous_track_key=None,
                coverage_gap_reasons=gap_reasons,
            )
        )
        media_sources.append(
            DailyDigestMediaSource(
                event_id=event_id,
                camera_id=camera_id,
                clip_path=clip,
                clip_sha256=clip_sha256,
                clip_bytes=metadata.st_size,
                duration_seconds=media_duration,
            )
        )
    return DailyDigestAdmission(
        candidates=tuple(candidates),
        local_date=local_date.isoformat(),
        source_event_count=len(candidates),
        media_files_hashed=len(candidates),
        media_bytes_hashed=total_bytes,
        coverage_gap_count=gap_count,
        media_sources=tuple(media_sources),
    )
