from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import stat
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from .adapters.signal import signal_event_id
from .domain import SchemaError, SignalEvent

SIGNAL_OUTBOX_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signal_outbox (
    outbox_id TEXT PRIMARY KEY,
    signal_json TEXT,
    content_sha256 TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
    available_at TEXT NOT NULL,
    lease_until TEXT,
    lease_token TEXT,
    last_error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK (
        (status = 'running' AND lease_until IS NOT NULL AND lease_token IS NOT NULL)
        OR (status != 'running' AND lease_until IS NULL AND lease_token IS NULL)
    ),
    CHECK (
        (status = 'completed' AND signal_json IS NULL AND completed_at IS NOT NULL)
        OR (status != 'completed' AND signal_json IS NOT NULL AND completed_at IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_signal_outbox_status_available
    ON signal_outbox (status, available_at, created_at);
"""


class SignalOutboxError(RuntimeError):
    pass


class SignalOutboxFullError(SignalOutboxError):
    pass


class SignalOutboxStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SignalOutboxEnqueueResult:
    outbox_id: str
    status: SignalOutboxStatus
    reused_existing: bool

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "outbox_id": self.outbox_id,
            "status": self.status.value,
            "reused_existing": self.reused_existing,
            "paths_disclosed": False,
            "entity_ids_disclosed": False,
        }


@dataclass(frozen=True, slots=True)
class SignalOutboxClaim:
    outbox_id: str
    signal: SignalEvent
    lease_token: str
    attempt_count: int
    lease_until: datetime

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": "claimed",
            "outbox_id": self.outbox_id,
            "signal": self.signal.to_dict(),
            "lease_token": self.lease_token,
            "attempt_count": self.attempt_count,
            "lease_until": self.lease_until.isoformat(),
            "paths_disclosed": False,
            "entity_ids_disclosed": False,
        }


@dataclass(frozen=True, slots=True)
class SignalOutboxCounts:
    pending: int
    running: int
    completed: int
    failed: int

    @property
    def active(self) -> int:
        return self.pending + self.running

    def to_dict(self) -> dict[str, int]:
        return {
            "pending": self.pending,
            "running": self.running,
            "completed": self.completed,
            "failed": self.failed,
            "active": self.active,
        }


@dataclass(frozen=True, slots=True)
class SignalOutboxSnapshot:
    counts: SignalOutboxCounts
    ready_pending: int
    oldest_pending_age_seconds: float | None
    failed_error_counts: tuple[tuple[str, int], ...]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": "available",
            **self.counts.to_dict(),
            "ready_pending": self.ready_pending,
            "oldest_pending_age_seconds": self.oldest_pending_age_seconds,
            "failed_error_counts": dict(self.failed_error_counts),
            "payloads_disclosed": False,
            "paths_disclosed": False,
            "entity_ids_disclosed": False,
        }


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO 8601 timestamp") from exc
    return _utc(parsed, field=field)


def _signal_json(signal: SignalEvent) -> str:
    return json.dumps(
        signal.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _content_sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _error_code(value: str) -> str:
    if not value or len(value) > 80 or not value.replace("_", "").isalnum():
        raise ValueError("error_code must be a short identifier")
    return value


class SignalOutbox:
    """Durable, bounded hand-off for normalized path-free SignalEvent values."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path).expanduser()
        if not self.path.is_absolute():
            raise SignalOutboxError("signal outbox path must be absolute")

    def _validate_path(self) -> None:
        if not self.path.exists():
            return
        try:
            metadata = self.path.lstat()
        except OSError as exc:
            raise SignalOutboxError("signal outbox database is missing or unsafe") from exc
        if self.path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SignalOutboxError("signal outbox database is missing or unsafe")
        if os.name == "posix" and (
            metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise SignalOutboxError("signal outbox database permissions are unsafe")

    def _connect(self) -> sqlite3.Connection:
        self._validate_path()
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _connect_read_only(self) -> sqlite3.Connection:
        self._validate_path()
        path = self.path.resolve(strict=True)
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _connect_existing(self) -> sqlite3.Connection:
        self._validate_path()
        try:
            path = self.path.resolve(strict=True)
            connection = sqlite3.connect(f"{path.as_uri()}?mode=rw", uri=True)
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            raise SignalOutboxError("signal outbox database is missing or invalid") from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self._validate_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(self.path, flags, 0o600)
            except FileExistsError:
                pass
            except OSError as exc:
                raise SignalOutboxError("signal outbox database cannot be initialized") from exc
            else:
                os.close(descriptor)
        self._validate_path()
        connection = self._connect_existing()
        try:
            with connection:
                connection.executescript(SIGNAL_OUTBOX_SCHEMA_SQL)
        finally:
            connection.close()
        if os.name == "posix":
            with suppress(OSError):
                self.path.chmod(0o600)

    def enqueue(
        self,
        signal: SignalEvent,
        *,
        max_active: int = 10_000,
        max_completed: int = 100_000,
        now: datetime | None = None,
    ) -> SignalOutboxEnqueueResult:
        if not isinstance(signal, SignalEvent):
            raise TypeError("signal must be a SignalEvent")
        if not 1 <= max_active <= 100_000:
            raise ValueError("max_active must be between 1 and 100000")
        if not 0 <= max_completed <= 1_000_000:
            raise ValueError("max_completed must be between 0 and 1000000")
        queued_at = _utc(now or datetime.now(UTC), field="now")
        outbox_id = signal_event_id(signal)
        payload = _signal_json(signal)
        content_sha256 = _content_sha256(payload)
        self.initialize()
        connection = self._connect_existing()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT content_sha256, status FROM signal_outbox WHERE outbox_id = ?",
                (outbox_id,),
            ).fetchone()
            if existing is not None:
                if existing["content_sha256"] != content_sha256:
                    raise SignalOutboxError(
                        "signal identity already exists with different immutable data"
                    )
                connection.commit()
                return SignalOutboxEnqueueResult(
                    outbox_id=outbox_id,
                    status=SignalOutboxStatus(existing["status"]),
                    reused_existing=True,
                )
            active = int(
                connection.execute(
                    "SELECT COUNT(*) FROM signal_outbox WHERE status IN ('pending', 'running')"
                ).fetchone()[0]
            )
            if active >= max_active:
                raise SignalOutboxFullError(
                    "signal outbox is full; the new signal was not accepted"
                )
            connection.execute(
                """
                INSERT INTO signal_outbox (
                    outbox_id, signal_json, content_sha256, status, attempt_count,
                    available_at, lease_until, lease_token, last_error_code,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, 'pending', 0, ?, NULL, NULL, NULL, ?, ?, NULL)
                """,
                (
                    outbox_id,
                    payload,
                    content_sha256,
                    queued_at.isoformat(),
                    queued_at.isoformat(),
                    queued_at.isoformat(),
                ),
            )
            completed_ids = connection.execute(
                """
                SELECT outbox_id FROM signal_outbox
                WHERE status = 'completed'
                ORDER BY completed_at DESC, outbox_id DESC
                LIMIT -1 OFFSET ?
                """,
                (max_completed,),
            ).fetchall()
            if completed_ids:
                connection.executemany(
                    "DELETE FROM signal_outbox WHERE outbox_id = ? AND status = 'completed'",
                    ((row["outbox_id"],) for row in completed_ids),
                )
            connection.commit()
        finally:
            connection.close()
        return SignalOutboxEnqueueResult(
            outbox_id=outbox_id,
            status=SignalOutboxStatus.PENDING,
            reused_existing=False,
        )

    def claim_next(
        self,
        *,
        lease_seconds: int = 60,
        max_attempts: int = 5,
        now: datetime | None = None,
    ) -> SignalOutboxClaim | None:
        if not 10 <= lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be between 10 and 86400")
        if not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts must be between 1 and 100")
        claimed_at = _utc(now or datetime.now(UTC), field="now")
        lease_until = claimed_at + timedelta(seconds=lease_seconds)
        lease_token = secrets.token_hex(16)
        self.initialize()
        connection = self._connect_existing()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE signal_outbox
                SET status = CASE WHEN attempt_count >= ? THEN 'failed' ELSE 'pending' END,
                    available_at = ?, lease_until = NULL, lease_token = NULL,
                    last_error_code = 'lease_expired', updated_at = ?
                WHERE status = 'running' AND lease_until <= ?
                """,
                (
                    max_attempts,
                    claimed_at.isoformat(),
                    claimed_at.isoformat(),
                    claimed_at.isoformat(),
                ),
            )
            row = connection.execute(
                """
                SELECT outbox_id FROM signal_outbox
                WHERE status = 'pending' AND available_at <= ?
                ORDER BY created_at ASC, outbox_id ASC
                LIMIT 1
                """,
                (claimed_at.isoformat(),),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            updated = connection.execute(
                """
                UPDATE signal_outbox
                SET status = 'running', attempt_count = attempt_count + 1,
                    lease_until = ?, lease_token = ?, last_error_code = NULL,
                    updated_at = ?
                WHERE outbox_id = ? AND status = 'pending'
                """,
                (
                    lease_until.isoformat(),
                    lease_token,
                    claimed_at.isoformat(),
                    row["outbox_id"],
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise SignalOutboxError("signal outbox claim lost a concurrency race")
            claimed = connection.execute(
                "SELECT * FROM signal_outbox WHERE outbox_id = ?",
                (row["outbox_id"],),
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        if claimed is None:
            raise SignalOutboxError("claimed signal disappeared from SQLite")
        return self._row_to_claim(claimed)

    def acknowledge(
        self,
        outbox_id: str,
        lease_token: str,
        *,
        now: datetime | None = None,
    ) -> None:
        completed_at = _utc(now or datetime.now(UTC), field="now")
        connection = self._connect_existing()
        try:
            with connection:
                updated = connection.execute(
                    """
                    UPDATE signal_outbox
                    SET status = 'completed', signal_json = NULL, lease_until = NULL,
                        lease_token = NULL, last_error_code = NULL,
                        completed_at = ?, updated_at = ?
                    WHERE outbox_id = ? AND status = 'running' AND lease_token = ?
                    """,
                    (
                        completed_at.isoformat(),
                        completed_at.isoformat(),
                        outbox_id,
                        lease_token,
                    ),
                )
                if updated.rowcount != 1:
                    raise SignalOutboxError("signal acknowledgement is stale or invalid")
        finally:
            connection.close()

    def record_failure(
        self,
        outbox_id: str,
        lease_token: str,
        *,
        error_code: str,
        retry_delay_seconds: int = 5,
        max_attempts: int = 5,
        now: datetime | None = None,
    ) -> SignalOutboxStatus:
        code = _error_code(error_code)
        if not 0 <= retry_delay_seconds <= 86_400:
            raise ValueError("retry_delay_seconds must be between 0 and 86400")
        if not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts must be between 1 and 100")
        failed_at = _utc(now or datetime.now(UTC), field="now")
        available_at = failed_at + timedelta(seconds=retry_delay_seconds)
        connection = self._connect_existing()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT attempt_count FROM signal_outbox
                WHERE outbox_id = ? AND status = 'running' AND lease_token = ?
                """,
                (outbox_id, lease_token),
            ).fetchone()
            if row is None:
                raise SignalOutboxError("signal failure receipt is stale or invalid")
            status = (
                SignalOutboxStatus.FAILED
                if int(row["attempt_count"]) >= max_attempts
                else SignalOutboxStatus.PENDING
            )
            connection.execute(
                """
                UPDATE signal_outbox
                SET status = ?, available_at = ?, lease_until = NULL,
                    lease_token = NULL, last_error_code = ?, updated_at = ?
                WHERE outbox_id = ?
                """,
                (
                    status.value,
                    available_at.isoformat(),
                    code,
                    failed_at.isoformat(),
                    outbox_id,
                ),
            )
            connection.commit()
        finally:
            connection.close()
        return status

    def retry_failed(
        self,
        outbox_id: str,
        *,
        max_active: int = 10_000,
        now: datetime | None = None,
    ) -> None:
        if not 1 <= max_active <= 100_000:
            raise ValueError("max_active must be between 1 and 100000")
        retried_at = _utc(now or datetime.now(UTC), field="now")
        connection = self._connect_existing()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = int(
                connection.execute(
                    "SELECT COUNT(*) FROM signal_outbox WHERE status IN ('pending', 'running')"
                ).fetchone()[0]
            )
            if active >= max_active:
                raise SignalOutboxFullError("signal outbox is full; failed signal remains tracked")
            updated = connection.execute(
                """
                UPDATE signal_outbox
                SET status = 'pending', attempt_count = 0, available_at = ?,
                    lease_until = NULL, lease_token = NULL, last_error_code = NULL,
                    updated_at = ?
                WHERE outbox_id = ? AND status = 'failed'
                """,
                (retried_at.isoformat(), retried_at.isoformat(), outbox_id),
            )
            if updated.rowcount != 1:
                raise SignalOutboxError("only a failed signal can be retried")
            connection.commit()
        finally:
            connection.close()

    def snapshot_read_only(
        self,
        *,
        now: datetime | None = None,
    ) -> SignalOutboxSnapshot:
        observed_at = _utc(now or datetime.now(UTC), field="now")
        try:
            connection = self._connect_read_only()
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            raise SignalOutboxError("signal outbox database is missing or invalid") from exc
        try:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM signal_outbox GROUP BY status"
            ).fetchall()
            values = {str(row["status"]): int(row["count"]) for row in rows}
            pending = connection.execute(
                """
                SELECT COALESCE(SUM(CASE WHEN available_at <= ? THEN 1 ELSE 0 END), 0)
                           AS ready_pending,
                       MIN(created_at) AS oldest_created
                FROM signal_outbox WHERE status = 'pending'
                """,
                (observed_at.isoformat(),),
            ).fetchone()
            failed = connection.execute(
                """
                SELECT last_error_code, COUNT(*) AS count FROM signal_outbox
                WHERE status = 'failed' AND last_error_code IS NOT NULL
                GROUP BY last_error_code
                """
            ).fetchall()
            oldest = (
                _parse_utc(pending["oldest_created"], field="oldest_created")
                if pending["oldest_created"] is not None
                else None
            )
            return SignalOutboxSnapshot(
                counts=SignalOutboxCounts(
                    pending=values.get("pending", 0),
                    running=values.get("running", 0),
                    completed=values.get("completed", 0),
                    failed=values.get("failed", 0),
                ),
                ready_pending=int(pending["ready_pending"]),
                oldest_pending_age_seconds=(
                    max(0.0, (observed_at - oldest).total_seconds()) if oldest is not None else None
                ),
                failed_error_counts=tuple(
                    sorted((str(row["last_error_code"]), int(row["count"])) for row in failed)
                ),
            )
        except (KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            raise SignalOutboxError("signal outbox database is missing or invalid") from exc
        finally:
            connection.close()

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> SignalOutboxClaim:
        try:
            payload = row["signal_json"]
            if not isinstance(payload, str):
                raise ValueError
            signal = SignalEvent.from_dict(json.loads(payload))
            if (
                signal_event_id(signal) != row["outbox_id"]
                or _content_sha256(payload) != row["content_sha256"]
                or row["status"] != "running"
                or not isinstance(row["lease_token"], str)
            ):
                raise ValueError
            return SignalOutboxClaim(
                outbox_id=str(row["outbox_id"]),
                signal=signal,
                lease_token=str(row["lease_token"]),
                attempt_count=int(row["attempt_count"]),
                lease_until=_parse_utc(row["lease_until"], field="lease_until"),
            )
        except (SchemaError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SignalOutboxError("signal outbox row is invalid") from exc
