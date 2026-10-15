from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .room_presence import (
    RoomPresenceError,
    StreamPresenceSample,
    StreamPresenceSampleState,
)
from .sampling_history import SAMPLING_HISTORY_SCHEMA_SQL

PRESENCE_FACT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS presence_facts (
    sample_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL CHECK (mode IN ('primary', 'shadow')),
    camera_id TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('person_detected', 'clear', 'coverage_gap')),
    scanner_id TEXT NOT NULL,
    detector_id TEXT,
    source_fingerprint TEXT NOT NULL,
    reason_code TEXT,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_presence_facts_observed_at
    ON presence_facts (observed_at);
CREATE INDEX IF NOT EXISTS idx_presence_facts_camera_time
    ON presence_facts (camera_id, observed_at);
"""

_REQUIRED_COLUMNS = frozenset(
    {
        "sample_id",
        "mode",
        "camera_id",
        "observed_at",
        "state",
        "scanner_id",
        "detector_id",
        "source_fingerprint",
        "reason_code",
        "recorded_at",
    }
)


class PresenceHistoryError(RuntimeError):
    """A path-free presence fact persistence or read failure."""


class PresenceFactMode(StrEnum):
    PRIMARY = "primary"
    SHADOW = "shadow"


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise PresenceHistoryError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise PresenceHistoryError(f"presence history {field} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise PresenceHistoryError(f"presence history {field} is invalid") from exc
    return _aware_utc(parsed, field=f"presence history {field}")


def _timezone(name: str) -> ZoneInfo:
    if not isinstance(name, str) or not 1 <= len(name) <= 80:
        raise PresenceHistoryError("presence history timezone is invalid")
    try:
        return ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise PresenceHistoryError("presence history timezone is invalid") from exc


def _sample_from_row(row: sqlite3.Row) -> StreamPresenceSample:
    try:
        return StreamPresenceSample.from_dict(
            {
                "version": 1,
                "sample_id": row["sample_id"],
                "camera_id": row["camera_id"],
                "observed_at": row["observed_at"],
                "state": row["state"],
                "scanner_id": row["scanner_id"],
                "detector_id": row["detector_id"],
                "source_fingerprint": row["source_fingerprint"],
                "reason_code": row["reason_code"],
            }
        )
    except (KeyError, RoomPresenceError, TypeError, ValueError) as exc:
        raise PresenceHistoryError("presence history row is invalid") from exc


@dataclass(frozen=True, slots=True)
class PresenceFactWriteReceipt:
    sample_id: str
    mode: PresenceFactMode
    reused_existing: bool


class PresenceFactStore:
    """Idempotent media-free storage for normalized stream presence facts."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        try:
            connection = (
                sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
                if read_only
                else sqlite3.connect(self.path)
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            if read_only:
                connection.execute("PRAGMA query_only = ON")
            return connection
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            raise PresenceHistoryError("presence history database is unavailable") from exc

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        except sqlite3.Error as exc:
            raise PresenceHistoryError("presence fact could not be persisted") from exc
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as connection:
            connection.executescript(PRESENCE_FACT_SCHEMA_SQL)
            connection.executescript(SAMPLING_HISTORY_SCHEMA_SQL)

    def record(
        self,
        sample: StreamPresenceSample,
        *,
        mode: PresenceFactMode = PresenceFactMode.PRIMARY,
        recorded_at: datetime | None = None,
    ) -> PresenceFactWriteReceipt:
        if not isinstance(sample, StreamPresenceSample):
            raise PresenceHistoryError("presence fact sample is invalid")
        if not isinstance(mode, PresenceFactMode):
            raise PresenceHistoryError("presence fact mode is invalid")
        stored_at = _aware_utc(recorded_at or datetime.now(UTC), field="recorded_at")
        values = (
            sample.sample_id,
            mode.value,
            sample.camera_id,
            sample.observed_at.isoformat(),
            sample.state.value,
            sample.scanner_id,
            sample.detector_id,
            sample.source_fingerprint,
            sample.reason_code,
            stored_at.isoformat(),
        )
        with self._session() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO presence_facts (
                    sample_id, mode, camera_id, observed_at, state, scanner_id,
                    detector_id, source_fingerprint, reason_code, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            reused = cursor.rowcount == 0
            row = connection.execute(
                """
                SELECT sample_id, mode, camera_id, observed_at, state, scanner_id,
                       detector_id, source_fingerprint, reason_code
                FROM presence_facts WHERE sample_id = ?
                """,
                (sample.sample_id,),
            ).fetchone()
        if row is None or row["mode"] != mode.value or _sample_from_row(row) != sample:
            raise PresenceHistoryError("presence fact ID conflicts with stored facts")
        return PresenceFactWriteReceipt(
            sample_id=sample.sample_id,
            mode=mode,
            reused_existing=reused,
        )

    def fetch_between(
        self,
        start: datetime,
        end: datetime,
        *,
        mode: PresenceFactMode = PresenceFactMode.PRIMARY,
    ) -> tuple[StreamPresenceSample, ...]:
        beginning = _aware_utc(start, field="start")
        ending = _aware_utc(end, field="end")
        if ending <= beginning:
            raise PresenceHistoryError("presence history range is invalid")
        if not isinstance(mode, PresenceFactMode):
            raise PresenceHistoryError("presence fact mode is invalid")
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT sample_id, camera_id, observed_at, state, scanner_id,
                       detector_id, source_fingerprint, reason_code
                FROM presence_facts
                WHERE mode = ? AND observed_at >= ? AND observed_at < ?
                ORDER BY observed_at ASC, camera_id ASC, sample_id ASC
                """,
                (mode.value, beginning.isoformat(), ending.isoformat()),
            ).fetchall()
        except sqlite3.Error as exc:
            raise PresenceHistoryError("presence history database is invalid") from exc
        finally:
            connection.close()
        return tuple(_sample_from_row(row) for row in rows)


@dataclass(frozen=True, slots=True)
class PresenceStreamDaySummary:
    camera_id: str
    sample_count: int
    person_detected_count: int
    clear_count: int
    coverage_gap_count: int
    first_observed_at: datetime
    last_observed_at: datetime
    maximum_observation_gap_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise PresenceHistoryError("presence stream summary camera is invalid")
        for field in (
            "sample_count",
            "person_detected_count",
            "clear_count",
            "coverage_gap_count",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise PresenceHistoryError("presence stream summary count is invalid")
        if self.sample_count < 1 or self.sample_count != (
            self.person_detected_count + self.clear_count + self.coverage_gap_count
        ):
            raise PresenceHistoryError("presence stream summary counts are inconsistent")
        first = _aware_utc(self.first_observed_at, field="first_observed_at")
        last = _aware_utc(self.last_observed_at, field="last_observed_at")
        object.__setattr__(self, "first_observed_at", first)
        object.__setattr__(self, "last_observed_at", last)
        if last < first:
            raise PresenceHistoryError("presence stream summary range is invalid")
        gap = self.maximum_observation_gap_seconds
        if (
            isinstance(gap, bool)
            or not isinstance(gap, (int, float))
            or not math.isfinite(float(gap))
            or float(gap) < 0.0
        ):
            raise PresenceHistoryError("presence stream summary gap is invalid")
        object.__setattr__(self, "maximum_observation_gap_seconds", round(float(gap), 3))


@dataclass(frozen=True, slots=True)
class PresenceDaySummary:
    mode: PresenceFactMode
    local_date: date
    timezone_name: str
    start_at: datetime
    end_at: datetime
    streams: tuple[PresenceStreamDaySummary, ...]
    table_available: bool = True
    media_read: bool = False
    subject_identity_inference: bool = False
    risk_assessment: bool = False
    notification_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.local_date, date) or isinstance(self.local_date, datetime):
            raise PresenceHistoryError("presence day summary date is invalid")
        if not isinstance(self.mode, PresenceFactMode):
            raise PresenceHistoryError("presence day summary mode is invalid")
        _timezone(self.timezone_name)
        start = _aware_utc(self.start_at, field="start_at")
        end = _aware_utc(self.end_at, field="end_at")
        object.__setattr__(self, "start_at", start)
        object.__setattr__(self, "end_at", end)
        if end <= start:
            raise PresenceHistoryError("presence day summary range is invalid")
        if (
            not isinstance(self.streams, tuple)
            or any(not isinstance(item, PresenceStreamDaySummary) for item in self.streams)
            or tuple(item.camera_id for item in self.streams)
            != tuple(sorted(item.camera_id for item in self.streams))
        ):
            raise PresenceHistoryError("presence day summary streams are invalid")
        for field, expected in (
            ("table_available", True),
            ("media_read", False),
            ("subject_identity_inference", False),
            ("risk_assessment", False),
            ("notification_authority", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise PresenceHistoryError(f"presence day summary {field} must be {expected}")

    @property
    def sample_count(self) -> int:
        return sum(item.sample_count for item in self.streams)

    @property
    def person_detected_count(self) -> int:
        return sum(item.person_detected_count for item in self.streams)

    @property
    def clear_count(self) -> int:
        return sum(item.clear_count for item in self.streams)

    @property
    def coverage_gap_count(self) -> int:
        return sum(item.coverage_gap_count for item in self.streams)

    @property
    def maximum_observation_gap_seconds(self) -> float | None:
        if not self.streams:
            return None
        return max(item.maximum_observation_gap_seconds for item in self.streams)


@dataclass(slots=True)
class _StreamDayAccumulator:
    """Keep only counts and adjacent timestamps for one chronological stream."""

    first_observed_at: datetime
    last_observed_at: datetime
    maximum_gap_seconds: float = 0.0
    person_detected_count: int = 0
    clear_count: int = 0
    coverage_gap_count: int = 0

    def add(self, sample: StreamPresenceSample) -> None:
        self.maximum_gap_seconds = max(
            self.maximum_gap_seconds,
            (sample.observed_at - self.last_observed_at).total_seconds(),
        )
        self.last_observed_at = sample.observed_at
        if sample.state is StreamPresenceSampleState.PERSON_DETECTED:
            self.person_detected_count += 1
        elif sample.state is StreamPresenceSampleState.CLEAR:
            self.clear_count += 1
        else:
            self.coverage_gap_count += 1

    def finish(self, camera_id: str, *, end: datetime) -> PresenceStreamDaySummary:
        return PresenceStreamDaySummary(
            camera_id=camera_id,
            sample_count=(self.person_detected_count + self.clear_count + self.coverage_gap_count),
            person_detected_count=self.person_detected_count,
            clear_count=self.clear_count,
            coverage_gap_count=self.coverage_gap_count,
            first_observed_at=self.first_observed_at,
            last_observed_at=self.last_observed_at,
            maximum_observation_gap_seconds=max(
                self.maximum_gap_seconds,
                (end - self.last_observed_at).total_seconds(),
            ),
        )


def read_presence_day_summary(
    database_path: str | Path,
    *,
    local_date: date,
    timezone_name: str,
    mode: PresenceFactMode = PresenceFactMode.PRIMARY,
) -> PresenceDaySummary | None:
    """Read one local day without creating or migrating the database.

    ``None`` means the database predates the presence history table.  An empty
    ``streams`` tuple means the table exists but no facts were recorded that day.
    Neither case is treated as evidence that the monitored space was clear.
    Rows are validated one at a time; Python aggregation memory grows with the
    number of cameras, not the number of daily observations.
    """

    timezone = _timezone(timezone_name)
    if not isinstance(local_date, date) or isinstance(local_date, datetime):
        raise PresenceHistoryError("presence history local_date is invalid")
    if not isinstance(mode, PresenceFactMode):
        raise PresenceHistoryError("presence fact mode is invalid")
    local_start = datetime.combine(local_date, time.min, tzinfo=timezone)
    start = local_start.astimezone(UTC)
    end = (local_start + timedelta(days=1)).astimezone(UTC)
    store = PresenceFactStore(database_path)
    connection = store._connect(read_only=True)
    streams: dict[str, _StreamDayAccumulator] = {}
    try:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'presence_facts'"
        ).fetchone()
        if table is None:
            return None
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(presence_facts)").fetchall()
        }
        if not _REQUIRED_COLUMNS.issubset(columns):
            raise PresenceHistoryError("presence history database is invalid")
        rows = connection.execute(
            """
            SELECT sample_id, camera_id, observed_at, state, scanner_id,
                   detector_id, source_fingerprint, reason_code
            FROM presence_facts
            WHERE mode = ? AND observed_at >= ? AND observed_at < ?
            ORDER BY observed_at ASC, camera_id ASC, sample_id ASC
            """,
            (mode.value, start.isoformat(), end.isoformat()),
        )
        for row in rows:
            sample = _sample_from_row(row)
            accumulator = streams.get(sample.camera_id)
            if accumulator is None:
                accumulator = _StreamDayAccumulator(
                    first_observed_at=sample.observed_at,
                    last_observed_at=start,
                )
                streams[sample.camera_id] = accumulator
            accumulator.add(sample)
    except sqlite3.Error as exc:
        raise PresenceHistoryError("presence history database is invalid") from exc
    finally:
        connection.close()
    return PresenceDaySummary(
        mode=mode,
        local_date=local_date,
        timezone_name=timezone_name,
        start_at=start,
        end_at=end,
        streams=tuple(
            streams[camera_id].finish(camera_id, end=end) for camera_id in sorted(streams)
        ),
    )
