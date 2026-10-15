from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .domain import DecisionRecord, EventType, RiskLevel

_REQUIRED_COLUMNS = {
    "occurred_at",
    "event_type",
    "risk_level",
    "confidence",
    "needs_remote_review",
}
_EVENT_TYPES = frozenset(item.value for item in EventType)
_RISK_LEVELS = frozenset(item.value for item in RiskLevel)
_COUNT_COLUMNS = frozenset({"event_type", "risk_level"})
_MAX_RECENT = 16
_MAX_DECISION_BYTES = 64 * 1024


class HomeEventDigestError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HomeEventDigestEntry:
    local_time: str
    event_type: str
    risk_level: str
    confidence: float
    needs_review: bool

    def to_public_dict(self) -> dict[str, object]:
        return {
            "local_time": self.local_time,
            "event_type": self.event_type,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
        }


@dataclass(frozen=True, slots=True)
class HomeEventDigest:
    local_date: str
    today_total: int
    event_type_counts: tuple[tuple[str, int], ...]
    risk_level_counts: tuple[tuple[str, int], ...]
    recent: tuple[HomeEventDigestEntry, ...]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": "available",
            "local_date": self.local_date,
            "today_total": self.today_total,
            "event_type_counts": dict(self.event_type_counts),
            "risk_level_counts": dict(self.risk_level_counts),
            "recent": [entry.to_public_dict() for entry in self.recent],
            "media_read": False,
            "paths_disclosed": False,
            "event_ids_disclosed": False,
            "camera_ids_disclosed": False,
            "summaries_disclosed": False,
            "risk_authority": False,
            "notification_decision": False,
        }


def _private_existing_database(path: str | Path) -> Path:
    database = Path(path).expanduser()
    if not database.is_absolute():
        raise HomeEventDigestError("home event digest database is unavailable")
    try:
        metadata = database.lstat()
    except OSError as exc:
        raise HomeEventDigestError("home event digest database is unavailable") from exc
    if database.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise HomeEventDigestError("home event digest database is unavailable")
    if os.name == "posix" and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise HomeEventDigestError("home event digest database is unavailable")
    return database.resolve()


def _timezone(name: str) -> ZoneInfo:
    if not isinstance(name, str) or not 1 <= len(name) <= 80:
        raise HomeEventDigestError("home event digest timezone is invalid")
    try:
        return ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise HomeEventDigestError("home event digest timezone is invalid") from exc


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise HomeEventDigestError("home event digest timestamp is invalid")
    return value.astimezone(UTC)


def _counts(
    connection: sqlite3.Connection,
    *,
    column: str,
    start: datetime,
    end: datetime,
    allowed: set[str],
) -> tuple[tuple[str, int], ...]:
    if column not in _COUNT_COLUMNS:
        raise HomeEventDigestError("home event digest query is invalid")
    rows = connection.execute(
        f"SELECT {column}, COUNT(*) AS count FROM events "
        "WHERE occurred_at >= ? AND occurred_at < ? GROUP BY " + column,
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    values: list[tuple[str, int]] = []
    for row in rows:
        key = row[column]
        count = row["count"]
        if not isinstance(key, str) or key not in allowed or type(count) is not int or count < 0:
            raise HomeEventDigestError("home event digest database is invalid")
        values.append((key, count))
    return tuple(sorted(values))


def _decision_needs_review(row: sqlite3.Row) -> bool:
    """Read only bounded recent decision metadata; never return its contents."""
    state = row["decision_json_state"]
    if state == 0:
        return False  # Legacy schema or NULL decision: retain the legacy flag.
    if state != 1 or not isinstance(row["decision_payload"], str):
        raise HomeEventDigestError("home event digest decision is invalid")
    payload = row["decision_payload"]

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate decision key")
            result[key] = value
        return result

    try:
        # Keep the existing global DecisionRecord wire protocol unchanged while
        # enforcing duplicate-key rejection at this read-only public boundary.
        json.loads(payload, object_pairs_hook=unique_object)
        return DecisionRecord.from_json(payload).needs_review
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise HomeEventDigestError("home event digest decision is invalid") from None


def read_home_event_digest(
    database_path: str | Path,
    *,
    timezone_name: str = "Asia/Shanghai",
    recent_limit: int = 8,
    recent_day_only: bool = False,
    now: datetime | None = None,
) -> HomeEventDigest:
    if type(recent_limit) is not int or not 1 <= recent_limit <= _MAX_RECENT:
        raise HomeEventDigestError("home event digest recent limit is invalid")
    if type(recent_day_only) is not bool:
        raise HomeEventDigestError("home event digest recent scope is invalid")
    observed_at = _aware(now or datetime.now(UTC))
    timezone = _timezone(timezone_name)
    local_date = observed_at.astimezone(timezone).date()
    local_start = datetime.combine(local_date, time.min, tzinfo=timezone)
    start = local_start.astimezone(UTC)
    end = (local_start + timedelta(days=1)).astimezone(UTC)
    database = _private_existing_database(database_path)

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA query_only = ON")
    except sqlite3.Error as exc:
        if connection is not None:
            connection.close()
        raise HomeEventDigestError("home event digest database is unavailable") from exc
    try:
        connection.execute("BEGIN")
        columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(events)").fetchall()
        }
        if not _REQUIRED_COLUMNS.issubset(columns):
            raise HomeEventDigestError("home event digest database is invalid")
        event_type_counts = _counts(
            connection,
            column="event_type",
            start=start,
            end=end,
            allowed=set(_EVENT_TYPES),
        )
        risk_level_counts = _counts(
            connection,
            column="risk_level",
            start=start,
            end=end,
            allowed=set(_RISK_LEVELS),
        )
        # Do not transfer oversized or non-text decisions out of SQLite. Byte
        # length (rather than character count) bounds UTF-8 payloads, and substr
        # provides an additional extraction bound. NULL is a legacy absence,
        # whereas malformed/oversized records must fail closed, not hide review.
        decision_columns = "0 AS decision_json_state, NULL AS decision_payload"
        if "decision_json" in columns:
            valid_decision = (
                "typeof(decision_json) = 'text' AND "
                f"length(CAST(decision_json AS BLOB)) BETWEEN 1 AND {_MAX_DECISION_BYTES}"
            )
            decision_columns = (
                "CASE WHEN decision_json IS NULL THEN 0 "
                f"WHEN {valid_decision} THEN 1 ELSE 2 END AS decision_json_state, "
                # BLOB slicing preserves embedded NUL bytes. TEXT substr can
                # hide invalid trailing data after a valid JSON prefix.
                f"CASE WHEN {valid_decision} THEN CAST(substr(CAST(decision_json AS BLOB), 1, "
                f"{_MAX_DECISION_BYTES}) AS TEXT) ELSE NULL END AS decision_payload"
            )
        if recent_day_only:
            recent_rows = connection.execute(
                "SELECT occurred_at, event_type, risk_level, confidence, "
                f"needs_remote_review, {decision_columns} FROM events "
                "WHERE occurred_at >= ? AND occurred_at < ? "
                "ORDER BY occurred_at DESC LIMIT ?",
                (start.isoformat(), end.isoformat(), recent_limit),
            ).fetchall()
        else:
            recent_rows = connection.execute(
                "SELECT occurred_at, event_type, risk_level, confidence, "
                f"needs_remote_review, {decision_columns} FROM events "
                "ORDER BY occurred_at DESC LIMIT ?",
                (recent_limit,),
            ).fetchall()
    except sqlite3.Error:
        raise HomeEventDigestError("home event digest database is invalid") from None
    finally:
        connection.close()

    recent: list[HomeEventDigestEntry] = []
    for row in recent_rows:
        try:
            occurred_at = datetime.fromisoformat(row["occurred_at"])
        except (TypeError, ValueError) as exc:
            raise HomeEventDigestError("home event digest database is invalid") from exc
        occurred_at = _aware(occurred_at)
        event_type = row["event_type"]
        risk_level = row["risk_level"]
        confidence = row["confidence"]
        needs_review = row["needs_remote_review"]
        if (
            not isinstance(event_type, str)
            or event_type not in _EVENT_TYPES
            or not isinstance(risk_level, str)
            or risk_level not in _RISK_LEVELS
            or type(confidence) not in {int, float}
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
            or type(needs_review) is not int
            or needs_review not in {0, 1}
        ):
            raise HomeEventDigestError("home event digest database is invalid")
        decision_needs_review = _decision_needs_review(row)
        recent.append(
            HomeEventDigestEntry(
                local_time=occurred_at.astimezone(timezone).strftime("%m-%d %H:%M:%S"),
                event_type=event_type,
                risk_level=risk_level,
                confidence=round(float(confidence), 3),
                needs_review=bool(needs_review) or decision_needs_review,
            )
        )

    today_total = sum(count for _key, count in event_type_counts)
    if today_total != sum(count for _key, count in risk_level_counts):
        raise HomeEventDigestError("home event digest database is invalid")
    return HomeEventDigest(
        local_date=local_date.isoformat(),
        today_total=today_total,
        event_type_counts=event_type_counts,
        risk_level_counts=risk_level_counts,
        recent=tuple(recent),
    )
