from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from baby_monitor.archive import ArchiveManifest, validate_archive_sources

ARCHIVE_QUEUE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS archive_jobs (
    archive_id TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    source_root TEXT NOT NULL,
    item_count INTEGER NOT NULL CHECK (item_count >= 1),
    total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0),
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
    available_at TEXT NOT NULL,
    lease_until TEXT,
    last_error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_archive_jobs_status_available
    ON archive_jobs (status, available_at, created_at);
"""


class ArchiveQueueError(RuntimeError):
    pass


class ArchiveQueueFullError(ArchiveQueueError):
    pass


class ArchiveJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ArchiveJob:
    manifest: ArchiveManifest
    source_root: str
    status: ArchiveJobStatus
    attempt_count: int
    available_at: datetime
    lease_until: datetime | None
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    @property
    def archive_id(self) -> str:
        return self.manifest.archive_id


@dataclass(frozen=True, slots=True)
class ArchiveEnqueueResult:
    job: ArchiveJob
    reused_existing: bool


@dataclass(frozen=True, slots=True)
class ArchiveQueueCounts:
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
class ArchiveQueueSnapshot:
    counts: ArchiveQueueCounts
    ready_pending: int
    oldest_pending_age_seconds: float | None
    pending_items: int
    pending_bytes: int
    failed_error_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **self.counts.to_dict(),
            "ready_pending": self.ready_pending,
            "oldest_pending_age_seconds": self.oldest_pending_age_seconds,
            "pending_items": self.pending_items,
            "pending_bytes": self.pending_bytes,
            "failed_error_counts": dict(self.failed_error_counts),
        }


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_utc_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO 8601 timestamp") from exc
    return _utc(parsed, field=field)


def _manifest_json(manifest: ArchiveManifest) -> str:
    return json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class ArchiveQueue:
    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _connect_read_only(self) -> sqlite3.Connection:
        path = self.path.expanduser().resolve()
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA query_only = ON")
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
            connection.executescript(ARCHIVE_QUEUE_SCHEMA_SQL)

    def enqueue(
        self,
        *,
        manifest: ArchiveManifest,
        source_root: str | Path,
        max_active: int,
        now: datetime | None = None,
    ) -> ArchiveEnqueueResult:
        if not 1 <= max_active <= 10_000:
            raise ValueError("max_active must be between 1 and 10000")
        root = validate_archive_sources(source_root, manifest)
        queued_at = _utc(now or datetime.now(UTC), field="now")
        manifest_json = _manifest_json(manifest)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM archive_jobs WHERE archive_id = ?",
                (manifest.archive_id,),
            ).fetchone()
            if existing is not None:
                immutable = (
                    existing["manifest_json"] == manifest_json
                    and existing["source_root"] == str(root)
                    and int(existing["item_count"]) == len(manifest.items)
                    and int(existing["total_bytes"]) == manifest.total_bytes
                )
                if not immutable:
                    raise ArchiveQueueError(
                        "archive_id already exists with different immutable data"
                    )
                connection.commit()
                return ArchiveEnqueueResult(
                    self._row_to_job(existing),
                    reused_existing=True,
                )
            active = int(
                connection.execute(
                    "SELECT COUNT(*) FROM archive_jobs WHERE status IN ('pending', 'running')"
                ).fetchone()[0]
            )
            queue_full = active >= max_active
            status = ArchiveJobStatus.FAILED if queue_full else ArchiveJobStatus.PENDING
            error_code = "queue_full" if queue_full else None
            connection.execute(
                """
                INSERT INTO archive_jobs (
                    archive_id, manifest_json, source_root, item_count, total_bytes,
                    status, attempt_count, available_at, lease_until, last_error_code,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.archive_id,
                    manifest_json,
                    str(root),
                    len(manifest.items),
                    manifest.total_bytes,
                    status.value,
                    0,
                    queued_at.isoformat(),
                    None,
                    error_code,
                    queued_at.isoformat(),
                    queued_at.isoformat(),
                    None,
                ),
            )
            row = connection.execute(
                "SELECT * FROM archive_jobs WHERE archive_id = ?",
                (manifest.archive_id,),
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        if row is None:
            raise ArchiveQueueError("archive job disappeared from SQLite")
        if queue_full:
            raise ArchiveQueueFullError(
                "archive queue is full; source files remain unchanged and the job is tracked"
            )
        return ArchiveEnqueueResult(self._row_to_job(row), reused_existing=False)

    def claim_next(
        self,
        *,
        lease_seconds: int,
        max_attempts: int,
        now: datetime | None = None,
    ) -> ArchiveJob | None:
        if not 10 <= lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be between 10 and 86400")
        if not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts must be between 1 and 100")
        claimed_at = _utc(now or datetime.now(UTC), field="now")
        lease_until = claimed_at + timedelta(seconds=lease_seconds)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE archive_jobs
                SET status = CASE WHEN attempt_count >= ? THEN 'failed' ELSE 'pending' END,
                    available_at = ?, lease_until = NULL,
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
                SELECT * FROM archive_jobs
                WHERE status = 'pending' AND available_at <= ?
                ORDER BY created_at ASC, archive_id ASC
                LIMIT 1
                """,
                (claimed_at.isoformat(),),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            updated = connection.execute(
                """
                UPDATE archive_jobs
                SET status = 'running', attempt_count = attempt_count + 1,
                    lease_until = ?, updated_at = ?, last_error_code = NULL
                WHERE archive_id = ? AND status = 'pending'
                """,
                (lease_until.isoformat(), claimed_at.isoformat(), row["archive_id"]),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise ArchiveQueueError("archive queue claim lost a concurrency race")
            claimed = connection.execute(
                "SELECT * FROM archive_jobs WHERE archive_id = ?",
                (row["archive_id"],),
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        return self._row_to_job(claimed) if claimed is not None else None

    def mark_completed(
        self,
        archive_id: str,
        *,
        now: datetime | None = None,
    ) -> ArchiveJob:
        completed_at = _utc(now or datetime.now(UTC), field="now")
        with self._session() as connection:
            updated = connection.execute(
                """
                UPDATE archive_jobs
                SET status = 'completed', lease_until = NULL, last_error_code = NULL,
                    completed_at = ?, updated_at = ?
                WHERE archive_id = ? AND status = 'running'
                """,
                (completed_at.isoformat(), completed_at.isoformat(), archive_id),
            )
            if updated.rowcount != 1:
                raise ArchiveQueueError("only a running archive job can complete")
            row = connection.execute(
                "SELECT * FROM archive_jobs WHERE archive_id = ?",
                (archive_id,),
            ).fetchone()
        if row is None:
            raise ArchiveQueueError("completed archive job disappeared from SQLite")
        return self._row_to_job(row)

    def mark_failed_attempt(
        self,
        archive_id: str,
        *,
        error_code: str,
        retry_delay_seconds: int,
        max_attempts: int,
        now: datetime | None = None,
    ) -> ArchiveJob:
        if not error_code or len(error_code) > 80 or not error_code.replace("_", "").isalnum():
            raise ValueError("error_code must be a short identifier")
        if not 0 <= retry_delay_seconds <= 86_400:
            raise ValueError("retry_delay_seconds must be between 0 and 86400")
        if not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts must be between 1 and 100")
        failed_at = _utc(now or datetime.now(UTC), field="now")
        available_at = failed_at + timedelta(seconds=retry_delay_seconds)
        with self._session() as connection:
            row = connection.execute(
                "SELECT attempt_count FROM archive_jobs "
                "WHERE archive_id = ? AND status = 'running'",
                (archive_id,),
            ).fetchone()
            if row is None:
                raise ArchiveQueueError("only a running archive job can record failure")
            status = (
                ArchiveJobStatus.FAILED
                if int(row["attempt_count"]) >= max_attempts
                else ArchiveJobStatus.PENDING
            )
            connection.execute(
                """
                UPDATE archive_jobs
                SET status = ?, available_at = ?, lease_until = NULL,
                    last_error_code = ?, updated_at = ?
                WHERE archive_id = ?
                """,
                (
                    status.value,
                    available_at.isoformat(),
                    error_code,
                    failed_at.isoformat(),
                    archive_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM archive_jobs WHERE archive_id = ?",
                (archive_id,),
            ).fetchone()
        if updated is None:
            raise ArchiveQueueError("failed archive job disappeared from SQLite")
        return self._row_to_job(updated)

    def retry_failed(
        self,
        archive_id: str,
        *,
        max_active: int,
        now: datetime | None = None,
    ) -> ArchiveJob:
        if not 1 <= max_active <= 10_000:
            raise ValueError("max_active must be between 1 and 10000")
        retried_at = _utc(now or datetime.now(UTC), field="now")
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = int(
                connection.execute(
                    "SELECT COUNT(*) FROM archive_jobs WHERE status IN ('pending', 'running')"
                ).fetchone()[0]
            )
            if active >= max_active:
                raise ArchiveQueueFullError("archive queue is full; failed job remains tracked")
            updated = connection.execute(
                """
                UPDATE archive_jobs
                SET status = 'pending', attempt_count = 0, available_at = ?,
                    lease_until = NULL, last_error_code = NULL, completed_at = NULL,
                    updated_at = ?
                WHERE archive_id = ? AND status = 'failed'
                """,
                (retried_at.isoformat(), retried_at.isoformat(), archive_id),
            )
            if updated.rowcount != 1:
                raise ArchiveQueueError("only a failed archive job can be retried")
            row = connection.execute(
                "SELECT * FROM archive_jobs WHERE archive_id = ?",
                (archive_id,),
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        if row is None:
            raise ArchiveQueueError("retried archive job disappeared from SQLite")
        return self._row_to_job(row)

    def snapshot(self, *, now: datetime | None = None) -> ArchiveQueueSnapshot:
        observed_at = _utc(now or datetime.now(UTC), field="now")
        self.initialize()
        with self._session() as connection:
            return self._snapshot_from_connection(connection, observed_at=observed_at)

    def snapshot_read_only(
        self,
        *,
        now: datetime | None = None,
    ) -> ArchiveQueueSnapshot:
        observed_at = _utc(now or datetime.now(UTC), field="now")
        try:
            connection = self._connect_read_only()
        except sqlite3.Error as exc:
            raise ArchiveQueueError("archive queue database is missing or invalid") from exc
        try:
            return self._snapshot_from_connection(connection, observed_at=observed_at)
        except (sqlite3.Error, ValueError) as exc:
            raise ArchiveQueueError("archive queue database is missing or invalid") from exc
        finally:
            connection.close()

    @staticmethod
    def _snapshot_from_connection(
        connection: sqlite3.Connection,
        *,
        observed_at: datetime,
    ) -> ArchiveQueueSnapshot:
        count_rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM archive_jobs GROUP BY status"
        ).fetchall()
        values = {str(row["status"]): int(row["count"]) for row in count_rows}
        counts = ArchiveQueueCounts(
            pending=values.get(ArchiveJobStatus.PENDING.value, 0),
            running=values.get(ArchiveJobStatus.RUNNING.value, 0),
            completed=values.get(ArchiveJobStatus.COMPLETED.value, 0),
            failed=values.get(ArchiveJobStatus.FAILED.value, 0),
        )
        pending_row = connection.execute(
            """
            SELECT COALESCE(SUM(CASE WHEN available_at <= ? THEN 1 ELSE 0 END), 0)
                       AS ready_pending,
                   MIN(created_at) AS oldest_created,
                   COALESCE(SUM(item_count), 0) AS pending_items,
                   COALESCE(SUM(total_bytes), 0) AS pending_bytes
            FROM archive_jobs
            WHERE status = 'pending'
            """,
            (observed_at.isoformat(),),
        ).fetchone()
        failed_rows = connection.execute(
            """
            SELECT last_error_code, COUNT(*) AS count
            FROM archive_jobs
            WHERE status = 'failed' AND last_error_code IS NOT NULL
            GROUP BY last_error_code
            """
        ).fetchall()
        oldest = (
            _parse_utc_timestamp(
                pending_row["oldest_created"],
                field="oldest_created",
            )
            if pending_row["oldest_created"] is not None
            else None
        )
        age = max(0.0, (observed_at - oldest).total_seconds()) if oldest is not None else None
        return ArchiveQueueSnapshot(
            counts=counts,
            ready_pending=int(pending_row["ready_pending"]),
            oldest_pending_age_seconds=age,
            pending_items=int(pending_row["pending_items"]),
            pending_bytes=int(pending_row["pending_bytes"]),
            failed_error_counts=tuple(
                sorted((str(row["last_error_code"]), int(row["count"])) for row in failed_rows)
            ),
        )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> ArchiveJob:
        try:
            manifest_value = json.loads(row["manifest_json"])
            manifest = ArchiveManifest.from_dict(manifest_value)
            if (
                manifest.archive_id != row["archive_id"]
                or len(manifest.items) != int(row["item_count"])
                or manifest.total_bytes != int(row["total_bytes"])
            ):
                raise ValueError
            source_root = Path(row["source_root"])
            if not source_root.is_absolute():
                raise ValueError
            return ArchiveJob(
                manifest=manifest,
                source_root=str(source_root),
                status=ArchiveJobStatus(row["status"]),
                attempt_count=int(row["attempt_count"]),
                available_at=_parse_utc_timestamp(row["available_at"], field="available_at"),
                lease_until=(
                    _parse_utc_timestamp(row["lease_until"], field="lease_until")
                    if row["lease_until"] is not None
                    else None
                ),
                last_error_code=(
                    str(row["last_error_code"]) if row["last_error_code"] is not None else None
                ),
                created_at=_parse_utc_timestamp(row["created_at"], field="created_at"),
                updated_at=_parse_utc_timestamp(row["updated_at"], field="updated_at"),
                completed_at=(
                    _parse_utc_timestamp(row["completed_at"], field="completed_at")
                    if row["completed_at"] is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArchiveQueueError("archive queue row is invalid") from exc
