from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .remote_review_port import RemoteAttemptStatus, RemoteBudgetError

REMOTE_REVIEW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS remote_review_attempts (
    attempt_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    completed_at TEXT,
    target_origin TEXT NOT NULL,
    model TEXT NOT NULL,
    frame_count INTEGER NOT NULL CHECK (frame_count >= 1 AND frame_count <= 3),
    frame_bytes INTEGER NOT NULL CHECK (frame_bytes >= 0),
    status TEXT NOT NULL CHECK (status IN ('reserved', 'succeeded', 'failed')),
    duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0),
    error_code TEXT
);

CREATE INDEX IF NOT EXISTS idx_remote_review_requested_at
    ON remote_review_attempts (requested_at);
CREATE INDEX IF NOT EXISTS idx_remote_review_event_id
    ON remote_review_attempts (event_id, requested_at);
"""


@dataclass(frozen=True, slots=True)
class RemoteReviewAttempt:
    attempt_id: str
    event_id: str
    requested_at: datetime
    completed_at: datetime | None
    target_origin: str
    model: str
    frame_count: int
    frame_bytes: int
    status: RemoteAttemptStatus
    duration_ms: int | None
    error_code: str | None

    def to_public_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "event_id": self.event_id,
            "requested_at": self.requested_at.astimezone(UTC).isoformat(),
            "completed_at": (
                self.completed_at.astimezone(UTC).isoformat()
                if self.completed_at is not None
                else None
            ),
            "target_origin": self.target_origin,
            "model": self.model,
            "frame_count": self.frame_count,
            "frame_bytes": self.frame_bytes,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error_code": self.error_code,
        }


class RemoteReviewAuditStore:
    """A media-free SQLite ledger and atomic per-process/provider call gate."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _connect_read_only(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise FileNotFoundError("remote review audit database does not exist")
        uri = self.path.resolve().as_uri() + "?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True)
        except sqlite3.Error:
            raise RuntimeError("remote review audit database is unreadable") from None
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
        except sqlite3.Error:
            connection.close()
            raise RuntimeError("remote review audit database is unreadable") from None
        return connection

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as connection:
            connection.executescript(REMOTE_REVIEW_SCHEMA_SQL)

    def reserve(
        self,
        *,
        event_id: str,
        requested_at: datetime,
        target_origin: str,
        model: str,
        frame_count: int,
        frame_bytes: int,
        max_requests_per_day: int,
        min_interval_seconds: int,
    ) -> str:
        if requested_at.tzinfo is None or requested_at.utcoffset() is None:
            raise ValueError("remote review requested_at must be timezone-aware")
        if not event_id or len(event_id) > 256 or any(ord(char) < 32 for char in event_id):
            raise ValueError("remote review event_id is invalid")
        if not target_origin or len(target_origin) > 512:
            raise ValueError("remote review target origin is invalid")
        if not model or len(model) > 256 or any(ord(char) < 32 for char in model):
            raise ValueError("remote review model is invalid")
        if type(frame_count) is not int or not 1 <= frame_count <= 3:
            raise ValueError("remote review frame_count must be between 1 and 3")
        if type(frame_bytes) is not int or frame_bytes < 0:
            raise ValueError("remote review frame_bytes must be non-negative")
        if type(max_requests_per_day) is not int or not 1 <= max_requests_per_day <= 10_000:
            raise ValueError("remote review daily budget must be between 1 and 10000")
        if type(min_interval_seconds) is not int or not 0 <= min_interval_seconds <= 86_400:
            raise ValueError("remote review interval must be between 0 and 86400 seconds")

        requested_utc = requested_at.astimezone(UTC)
        day_start = requested_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        attempt_id = uuid.uuid4().hex
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM remote_review_attempts "
                    "WHERE requested_at >= ? AND requested_at <= ?",
                    (day_start.isoformat(), requested_utc.isoformat()),
                ).fetchone()[0]
            )
            if count >= max_requests_per_day:
                raise RemoteBudgetError("daily_budget_exhausted")
            latest = connection.execute(
                "SELECT requested_at FROM remote_review_attempts "
                "WHERE requested_at <= ? ORDER BY requested_at DESC LIMIT 1",
                (requested_utc.isoformat(),),
            ).fetchone()
            if latest is not None:
                latest_at = datetime.fromisoformat(latest["requested_at"])
                elapsed = (requested_utc - latest_at.astimezone(UTC)).total_seconds()
                if elapsed < min_interval_seconds:
                    raise RemoteBudgetError("rate_limited")
            connection.execute(
                """
                INSERT INTO remote_review_attempts (
                    attempt_id, event_id, requested_at, completed_at,
                    target_origin, model, frame_count, frame_bytes,
                    status, duration_ms, error_code
                ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, 'reserved', NULL, NULL)
                """,
                (
                    attempt_id,
                    event_id,
                    requested_utc.isoformat(),
                    target_origin,
                    model,
                    frame_count,
                    frame_bytes,
                ),
            )
        return attempt_id

    def complete(
        self,
        attempt_id: str,
        *,
        status: RemoteAttemptStatus,
        completed_at: datetime,
        duration_ms: int,
        error_code: str | None = None,
    ) -> None:
        if status not in {RemoteAttemptStatus.SUCCEEDED, RemoteAttemptStatus.FAILED}:
            raise ValueError("remote review completion status must be terminal")
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise ValueError("remote review completed_at must be timezone-aware")
        if type(duration_ms) is not int or duration_ms < 0:
            raise ValueError("remote review duration_ms must be non-negative")
        if status is RemoteAttemptStatus.SUCCEEDED and error_code is not None:
            raise ValueError("successful remote review cannot have an error code")
        if status is RemoteAttemptStatus.FAILED and (
            not isinstance(error_code, str)
            or not error_code
            or len(error_code) > 64
            or not all(char.islower() or char.isdigit() or char == "_" for char in error_code)
        ):
            raise ValueError("failed remote review requires a safe error code")
        with self._session() as connection:
            cursor = connection.execute(
                """
                UPDATE remote_review_attempts
                SET status = ?, completed_at = ?, duration_ms = ?, error_code = ?
                WHERE attempt_id = ? AND status = 'reserved'
                """,
                (
                    status.value,
                    completed_at.astimezone(UTC).isoformat(),
                    duration_ms,
                    error_code,
                    attempt_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("remote review attempt is missing or already completed")

    def recent(self, *, limit: int = 20) -> tuple[RemoteReviewAttempt, ...]:
        if type(limit) is not int or not 1 <= limit <= 1_000:
            raise ValueError("remote review audit limit must be between 1 and 1000")
        connection = self._connect_read_only()
        try:
            try:
                rows = connection.execute(
                    "SELECT * FROM remote_review_attempts "
                    "ORDER BY requested_at DESC, attempt_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            except sqlite3.Error:
                raise RuntimeError(
                    "remote review audit schema is missing or invalid; run init-db"
                ) from None
        finally:
            connection.close()
        return tuple(self._row_to_attempt(row) for row in rows)

    @staticmethod
    def _row_to_attempt(row: sqlite3.Row) -> RemoteReviewAttempt:
        try:
            return RemoteReviewAttempt(
                attempt_id=row["attempt_id"],
                event_id=row["event_id"],
                requested_at=datetime.fromisoformat(row["requested_at"]),
                completed_at=(
                    datetime.fromisoformat(row["completed_at"])
                    if row["completed_at"] is not None
                    else None
                ),
                target_origin=row["target_origin"],
                model=row["model"],
                frame_count=int(row["frame_count"]),
                frame_bytes=int(row["frame_bytes"]),
                status=RemoteAttemptStatus(row["status"]),
                duration_ms=(int(row["duration_ms"]) if row["duration_ms"] is not None else None),
                error_code=row["error_code"],
            )
        except (IndexError, TypeError, ValueError):
            raise RuntimeError("remote review audit row is invalid") from None
