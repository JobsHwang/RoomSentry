"""Media-free persistence of actual detector invocations, never coverage duration.

Source fingerprints identify SegmentFingerprint metadata, not media byte hashes.
Day sample totals follow actual observation timestamps. Scan totals and decoder
diagnostics follow overlapping scan intervals: a cross-midnight scan can appear
on both days. Neither total establishes continuous observation or infant safety.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .detector_sampling import DetectorSamplingError, DetectorSamplingLedger

MAX_LEDGER_BYTES = 8 * 1024 * 1024
_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")

SAMPLING_HISTORY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sampling_scans (
    record_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    scanner_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    evidence_state TEXT NOT NULL CHECK (evidence_state IN ('recorded', 'unknown')),
    ledger_json TEXT NOT NULL,
    ledger_fingerprint TEXT NOT NULL,
    sampled_frame_count INTEGER NOT NULL CHECK (sampled_frame_count >= 0),
    max_adjacent_container_pts_gap_seconds REAL,
    container_pts_regression_count INTEGER NOT NULL CHECK (container_pts_regression_count >= 0)
);
CREATE INDEX IF NOT EXISTS idx_sampling_scans_interval
    ON sampling_scans (started_at, ended_at);
CREATE TABLE IF NOT EXISTS sampling_frames (
    record_id TEXT NOT NULL REFERENCES sampling_scans(record_id),
    decoded_frame_index INTEGER NOT NULL CHECK (decoded_frame_index >= 0),
    observed_at TEXT NOT NULL,
    detection_count INTEGER NOT NULL CHECK (detection_count >= 0),
    timestamp_source TEXT NOT NULL CHECK (timestamp_source IN ('container_pts', 'nominal_fps')),
    PRIMARY KEY (record_id, decoded_frame_index)
);
CREATE INDEX IF NOT EXISTS idx_sampling_frames_observed_at
    ON sampling_frames (observed_at);
"""

_COLUMNS = {
    "sampling_scans": {
        "record_id",
        "camera_id",
        "scanner_id",
        "source_fingerprint",
        "started_at",
        "ended_at",
        "evidence_state",
        "ledger_json",
        "ledger_fingerprint",
        "sampled_frame_count",
        "max_adjacent_container_pts_gap_seconds",
        "container_pts_regression_count",
    },
    "sampling_frames": {
        "record_id",
        "decoded_frame_index",
        "observed_at",
        "detection_count",
        "timestamp_source",
    },
}


class SamplingHistoryError(RuntimeError):
    """A path-free sampling history failure."""


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SamplingHistoryError("sampling history timestamp must be timezone-aware")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError):
        raise SamplingHistoryError("sampling history timestamp is invalid") from None


def _stamp(value: datetime) -> str:
    return _utc(value).isoformat(timespec="microseconds")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _path(value: str | Path) -> Path:
    try:
        result = Path(value).expanduser()
        if "\x00" in str(result):
            raise ValueError
        return result
    except (OSError, RuntimeError, TypeError, ValueError):
        raise SamplingHistoryError("sampling history database path is invalid") from None


def _connect(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    connection = None
    try:
        connection = (
            sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
            if read_only
            else sqlite3.connect(path)
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        if read_only:
            connection.execute("PRAGMA query_only = ON")
        return connection
    except (OSError, RuntimeError, ValueError, sqlite3.Error):
        if connection is not None:
            connection.close()
        raise SamplingHistoryError("sampling history database is unavailable") from None


def _schema(connection: sqlite3.Connection) -> bool:
    tables = connection.execute(
        "SELECT name, type FROM sqlite_master WHERE name IN ('sampling_scans','sampling_frames')"
    ).fetchall()
    if not tables:
        return False
    if len(tables) != 2 or any(row["type"] != "table" for row in tables):
        raise SamplingHistoryError("sampling history schema is invalid")
    for table, columns in _COLUMNS.items():
        info = connection.execute(f"PRAGMA table_info({table})").fetchall()
        if {row["name"] for row in info} != columns:
            raise SamplingHistoryError("sampling history schema is invalid")
        integers = {
            "sampled_frame_count",
            "container_pts_regression_count",
            "decoded_frame_index",
            "detection_count",
        }
        for row in info:
            name = row["name"]
            expected_type = (
                "INTEGER"
                if name in integers
                else "REAL"
                if name == "max_adjacent_container_pts_gap_seconds"
                else "TEXT"
            )
            expected_required = int(
                name != "max_adjacent_container_pts_gap_seconds"
                and not (table == "sampling_scans" and name == "record_id")
            )
            if row["type"].upper() != expected_type or row["notnull"] != expected_required:
                raise SamplingHistoryError("sampling history schema is invalid")
        primary = {row["name"]: row["pk"] for row in info if row["pk"]}
        expected = {"record_id": 1}
        if table == "sampling_frames":
            expected["decoded_frame_index"] = 2
        if primary != expected:
            raise SamplingHistoryError("sampling history schema is invalid")
    for index, expected_columns in (
        ("idx_sampling_scans_interval", ["started_at", "ended_at"]),
        ("idx_sampling_frames_observed_at", ["observed_at"]),
    ):
        actual_columns = [row["name"] for row in connection.execute(f"PRAGMA index_info({index})")]
        if actual_columns != expected_columns:
            raise SamplingHistoryError("sampling history schema is invalid")
    return True


@dataclass(frozen=True, slots=True)
class SamplingEvidenceWriteReceipt:
    record_id: str
    reused_existing: bool


class SamplingEvidenceStore:
    def __init__(self, path: str | Path):
        self.path = _path(path)

    def initialize(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError):
            raise SamplingHistoryError("sampling history database is unavailable") from None
        connection = _connect(self.path)
        try:
            connection.executescript(SAMPLING_HISTORY_SCHEMA_SQL)
            _schema(connection)
        except sqlite3.Error:
            raise SamplingHistoryError("sampling history schema is invalid") from None
        finally:
            connection.close()

    def record(
        self,
        *,
        camera_id: str,
        scanner_id: str,
        source_fingerprint: str,
        started_at: datetime,
        ended_at: datetime,
        ledger: DetectorSamplingLedger | None,
    ) -> SamplingEvidenceWriteReceipt:
        for value in (camera_id, scanner_id):
            if not isinstance(value, str) or _ID.fullmatch(value) is None:
                raise SamplingHistoryError("sampling history identifier is invalid")
        if not isinstance(source_fingerprint, str) or _SHA.fullmatch(source_fingerprint) is None:
            raise SamplingHistoryError("sampling history source fingerprint is invalid")
        start, end = _utc(started_at), _utc(ended_at)
        duration = (end - start).total_seconds()
        if not math.isfinite(duration) or duration <= 0:
            raise SamplingHistoryError("sampling history scan duration is invalid")
        if ledger is not None:
            if not isinstance(ledger, DetectorSamplingLedger):
                raise SamplingHistoryError("sampling history ledger is invalid")
            try:
                ledger = DetectorSamplingLedger.from_dict(ledger.to_dict())
            except (DetectorSamplingError, TypeError, ValueError, OverflowError):
                raise SamplingHistoryError("sampling history ledger is invalid") from None
            if any(sample.timestamp_seconds > duration + 0.25 for sample in ledger.samples):
                raise SamplingHistoryError("sampling history sample is outside scan interval")
        ledger_json = _canonical(None if ledger is None else ledger.to_dict())
        if len(ledger_json.encode("utf-8")) > MAX_LEDGER_BYTES:
            raise SamplingHistoryError("sampling history ledger exceeds size limit")
        identity = (camera_id, scanner_id, source_fingerprint, _stamp(start), _stamp(end))
        record_id = _hash(_canonical(identity))
        values = (
            record_id,
            *identity,
            "unknown" if ledger is None else "recorded",
            ledger_json,
            _hash(ledger_json),
            0 if ledger is None else len(ledger.samples),
            None if ledger is None else ledger.max_adjacent_container_pts_gap_seconds,
            0 if ledger is None else ledger.container_pts_regression_count,
        )
        try:
            frames = (
                ()
                if ledger is None
                else tuple(
                    (
                        record_id,
                        sample.decoded_frame_index,
                        _stamp(start + timedelta(seconds=sample.timestamp_seconds)),
                        sample.detection_count,
                        sample.timestamp_source,
                    )
                    for sample in ledger.samples
                )
            )
        except (OverflowError, ValueError):
            raise SamplingHistoryError("sampling history observation time is invalid") from None
        connection = _connect(self.path)
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                if not _schema(connection):
                    raise SamplingHistoryError("sampling history schema is unavailable")
                row = connection.execute(
                    "SELECT * FROM sampling_scans WHERE record_id = ?", (record_id,)
                ).fetchone()
                if row is not None:
                    saved_frames = connection.execute(
                        "SELECT * FROM sampling_frames WHERE record_id = ? "
                        "ORDER BY decoded_frame_index",
                        (record_id,),
                    ).fetchall()
                    if tuple(row) != values or tuple(map(tuple, saved_frames)) != frames:
                        raise SamplingHistoryError(
                            "sampling history record conflicts with stored evidence"
                        )
                    return SamplingEvidenceWriteReceipt(record_id, True)
                connection.execute(
                    "INSERT INTO sampling_scans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
                connection.executemany(
                    "INSERT INTO sampling_frames VALUES (?, ?, ?, ?, ?)",
                    frames,
                )
        except sqlite3.Error:
            raise SamplingHistoryError("sampling history evidence could not be persisted") from None
        finally:
            connection.close()
        return SamplingEvidenceWriteReceipt(record_id, False)


@dataclass(frozen=True, slots=True)
class SamplingDaySummary:
    local_date: str
    timezone_name: str
    sampled_frame_count: int
    zero_detection_sample_count: int
    unknown_scan_count: int
    recorded_scan_count: int
    max_adjacent_container_pts_gap_seconds: float | None
    timestamp_fallback_sample_count: int
    container_pts_regression_count: int
    source_exhaustion_verified: bool = field(default=False, init=False)
    continuous_observation_seconds: None = field(default=None, init=False)

    def __post_init__(self) -> None:
        try:
            if date.fromisoformat(self.local_date).isoformat() != self.local_date:
                raise ValueError
            ZoneInfo(self.timezone_name)
        except (ValueError, TypeError, ZoneInfoNotFoundError):
            raise SamplingHistoryError("sampling day summary date or timezone is invalid") from None
        for name in (
            "sampled_frame_count",
            "zero_detection_sample_count",
            "unknown_scan_count",
            "recorded_scan_count",
            "timestamp_fallback_sample_count",
            "container_pts_regression_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SamplingHistoryError("sampling day summary count is invalid")
        if (
            max(self.zero_detection_sample_count, self.timestamp_fallback_sample_count)
            > self.sampled_frame_count
        ):
            raise SamplingHistoryError("sampling day summary counts are inconsistent")
        gap = self.max_adjacent_container_pts_gap_seconds
        if gap is not None and (
            isinstance(gap, bool)
            or not isinstance(gap, (int, float))
            or not math.isfinite(gap)
            or gap < 0
        ):
            raise SamplingHistoryError("sampling day summary gap is invalid")


def _valid_stamp(value: object) -> bool:
    try:
        return isinstance(value, str) and _stamp(datetime.fromisoformat(value)) == value
    except (ValueError, SamplingHistoryError, OverflowError):
        return False


def _valid_header(*row: object) -> bool:
    (
        record_id,
        camera,
        scanner,
        source,
        start,
        end,
        state,
        ledger_size,
        ledger_is_null,
        fingerprint,
        count,
        gap,
        regressions,
    ) = row
    if not all(isinstance(v, str) and _SHA.fullmatch(v) for v in (record_id, source, fingerprint)):
        return False
    if not all(isinstance(v, str) and _ID.fullmatch(v) for v in (camera, scanner)):
        return False
    if not _valid_stamp(start) or not _valid_stamp(end):
        return False
    duration = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    if (
        not math.isfinite(duration)
        or duration <= 0
        or _hash(_canonical((camera, scanner, source, start, end))) != record_id
    ):
        return False
    if type(count) is not int or not 0 <= count <= 100_000:
        return False
    if type(regressions) is not int or not 0 <= regressions < 10_000_000:
        return False
    if gap is not None and (not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap < 0):
        return False
    # The reader does not parse or transfer ledger JSON. Scalar headers drive the
    # indexed day aggregation; exact JSON replay binding is checked by record().
    if type(ledger_size) is not int or not 1 <= ledger_size <= MAX_LEDGER_BYTES:
        return False
    if state == "unknown":
        return (
            count == 0
            and gap is None
            and regressions == 0
            and ledger_is_null == 1
            and fingerprint == _hash("null")
        )
    return state == "recorded" and count > 0 and ledger_is_null == 0


def _valid_sample(
    index: object,
    observed: object,
    count: object,
    source: object,
    start: object,
    end: object,
    state: object,
) -> bool:
    if type(index) is not int or not 0 <= index < 10_000_000:
        return False
    if type(count) is not int or not 0 <= count <= 100_000:
        return False
    if source not in ("container_pts", "nominal_fps") or state != "recorded":
        return False
    if not all(_valid_stamp(v) for v in (observed, start, end)):
        return False
    point = datetime.fromisoformat(observed)
    beginning, ending = datetime.fromisoformat(start), datetime.fromisoformat(end)
    return 0 <= (point - beginning).total_seconds() <= (ending - beginning).total_seconds() + 0.25


def read_sampling_day_summary(
    path: str | Path,
    *,
    local_date: date,
    timezone_name: str,
) -> SamplingDaySummary | None:
    """Read indexed point evidence and overlapping scan diagnostics without writes.

    Missing stores/tables return None. An initialized day with no rows returns
    zero counts, not proof of an empty scene. No media or complete-ledger replay
    is needed for this aggregation. Scan-level diagnostics can repeat on both
    sides of midnight and must not be added as observation durations. Compatible
    BufferSegment intervals have no hard maximum duration, so overlap selection
    cannot safely use a fixed historical lookback; older headers may be visited.
    """
    if not isinstance(local_date, date) or isinstance(local_date, datetime):
        raise SamplingHistoryError("sampling history date is invalid")
    try:
        if not isinstance(timezone_name, str) or not 1 <= len(timezone_name) <= 80:
            raise ValueError
        zone = ZoneInfo(timezone_name)
        local_start = datetime.combine(local_date, time.min, tzinfo=zone)
        start = _stamp(local_start)
        end = _stamp(local_start + timedelta(days=1))
    except (ValueError, OverflowError, ZoneInfoNotFoundError):
        raise SamplingHistoryError("sampling history timezone or day is invalid") from None
    database = _path(path)
    try:
        if not database.exists():
            return None
    except OSError:
        raise SamplingHistoryError("sampling history database is unavailable") from None
    connection = _connect(database, read_only=True)
    try:
        # Hold one read snapshot across schema checks and the separate aggregates.
        # Otherwise a concurrent writer could make valid rows look inconsistent.
        connection.execute("BEGIN")
        if not _schema(connection):
            return None
        connection.create_function("valid_sampling_header", 13, _valid_header, deterministic=True)
        connection.create_function("valid_sampling_frame", 7, _valid_sample, deterministic=True)
        # Scalar validity is checked in SQL before totals can become user-facing.
        scans = connection.execute(
            """SELECT COUNT(*) AS scans,
                COALESCE(SUM(evidence_state = 'unknown'), 0) AS unknowns,
                MAX(max_adjacent_container_pts_gap_seconds) AS max_gap,
                COALESCE(SUM(container_pts_regression_count), 0) AS regressions,
                COALESCE(SUM(NOT valid_sampling_header(record_id, camera_id, scanner_id,
                    source_fingerprint, started_at, ended_at, evidence_state,
                    length(CAST(ledger_json AS BLOB)), ledger_json = 'null',
                    ledger_fingerprint, sampled_frame_count,
                    max_adjacent_container_pts_gap_seconds,
                    container_pts_regression_count)), 0) AS invalid
            FROM sampling_scans WHERE started_at < ? AND ended_at > ?""",
            (end, start),
        ).fetchone()
        frames = connection.execute(
            """SELECT COUNT(*) AS samples,
                COALESCE(SUM(f.detection_count = 0), 0) AS zeros,
                COALESCE(SUM(f.timestamp_source = 'nominal_fps'), 0) AS fallbacks,
                COALESCE(SUM(NOT valid_sampling_frame(f.decoded_frame_index, f.observed_at,
                    f.detection_count, f.timestamp_source, s.started_at, s.ended_at,
                    s.evidence_state)), 0) AS invalid
            FROM sampling_frames f LEFT JOIN sampling_scans s ON s.record_id = f.record_id
            WHERE f.observed_at >= ? AND f.observed_at < ?""",
            (start, end),
        ).fetchone()
        inconsistent = connection.execute(
            """SELECT 1 FROM sampling_scans s
            WHERE s.started_at < ? AND s.ended_at > ? AND s.sampled_frame_count !=
                (SELECT COUNT(*) FROM sampling_frames f WHERE f.record_id = s.record_id)
            LIMIT 1""",
            (end, start),
        ).fetchone()
        if scans["invalid"] or frames["invalid"] or inconsistent is not None:
            raise SamplingHistoryError("sampling history row is invalid")
        return SamplingDaySummary(
            local_date=local_date.isoformat(),
            timezone_name=timezone_name,
            sampled_frame_count=frames["samples"],
            zero_detection_sample_count=frames["zeros"],
            unknown_scan_count=scans["unknowns"],
            recorded_scan_count=scans["scans"] - scans["unknowns"],
            max_adjacent_container_pts_gap_seconds=scans["max_gap"],
            timestamp_fallback_sample_count=frames["fallbacks"],
            container_pts_regression_count=scans["regressions"],
        )
    except (sqlite3.Error, ValueError, TypeError, OverflowError):
        raise SamplingHistoryError("sampling history database is invalid") from None
    finally:
        connection.close()
