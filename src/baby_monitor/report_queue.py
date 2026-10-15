"""Coalesced daily-report refreshes with durable requests and fenced leases."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ReportQueueError(RuntimeError):
    """A report refresh request or queue state is invalid."""


class ReportQueueFullError(ReportQueueError):
    """Another destination cannot enter the bounded active queue."""


class ReportClaimLostError(ReportQueueError):
    """The supplied report lease no longer belongs to its worker."""


class ReportRefreshStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


REPORT_REFRESH_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS report_refresh_jobs (
    job_id TEXT PRIMARY KEY,
    local_date TEXT NOT NULL,
    timezone_name TEXT NOT NULL,
    start_at TEXT NOT NULL,
    end_at TEXT NOT NULL,
    storage_root TEXT NOT NULL,
    requested_revision INTEGER NOT NULL CHECK (requested_revision >= 1),
    claimed_revision INTEGER NOT NULL DEFAULT 0 CHECK (claimed_revision >= 0),
    completed_revision INTEGER NOT NULL DEFAULT 0 CHECK (completed_revision >= 0),
    claim_token TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending','running','completed','failed')),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    available_at TEXT NOT NULL,
    lease_until TEXT,
    last_error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_requested_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK (completed_revision <= claimed_revision),
    CHECK (claimed_revision <= requested_revision),
    CHECK ((status = 'running' AND claim_token IS NOT NULL AND lease_until IS NOT NULL)
        OR (status != 'running' AND claim_token IS NULL AND lease_until IS NULL)),
    UNIQUE (storage_root, local_date)
);
CREATE INDEX IF NOT EXISTS idx_report_refresh_ready
    ON report_refresh_jobs (status, available_at, created_at, job_id);
CREATE TABLE IF NOT EXISTS report_refresh_requests (
    request_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES report_refresh_jobs(job_id),
    requested_revision INTEGER NOT NULL CHECK (requested_revision >= 1),
    requested_at TEXT NOT NULL,
    UNIQUE (job_id, requested_revision)
);
"""

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
_JOB_ID = re.compile(r"report_[0-9a-f]{64}")
_TOKEN = re.compile(r"[0-9a-f]{64}")
_ERROR_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_UNRESOLVED_JOB_LIMIT = 20
_JOB_COLUMNS = frozenset(
    {
        "job_id",
        "local_date",
        "timezone_name",
        "start_at",
        "end_at",
        "storage_root",
        "requested_revision",
        "claimed_revision",
        "completed_revision",
        "claim_token",
        "status",
        "attempt_count",
        "available_at",
        "lease_until",
        "last_error_code",
        "created_at",
        "updated_at",
        "last_requested_at",
        "completed_at",
    }
)


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ReportQueueError("report queue timestamp must be timezone-aware")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError):
        raise ReportQueueError("report queue timestamp is invalid") from None


def _integer(value: int, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ReportQueueError("report queue numeric limit is invalid")
    return value


def _seconds(value: float, maximum: float = 86400) -> float:
    if type(value) not in {int, float} or not 0 <= value <= maximum or not math.isfinite(value):
        raise ReportQueueError("report queue duration is invalid")
    return float(value)


def _later(now: datetime, seconds: float) -> str:
    try:
        return (now + timedelta(seconds=seconds)).isoformat()
    except (OverflowError, ValueError):
        raise ReportQueueError("report queue timestamp is outside its range") from None


def _identifier(value: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ReportQueueError("report queue identifier is invalid")
    return value


def _timezone(value: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ReportQueueError("report queue timezone is invalid")
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise ReportQueueError("report queue timezone is invalid") from None
    return value


def _day_bounds(local_date: date, timezone_name: str) -> tuple[datetime, datetime]:
    try:
        timezone = ZoneInfo(timezone_name)
        start = datetime.combine(local_date, time.min, timezone).astimezone(UTC)
        end = datetime.combine(local_date + timedelta(days=1), time.min, timezone).astimezone(UTC)
        if end <= start:
            raise ValueError
        return start, end
    except (OverflowError, ValueError):
        raise ReportQueueError("report queue local day is outside its supported range") from None


def _root(value: Path, *, admit: bool = True) -> Path:
    if not isinstance(value, (str, Path)):
        raise ReportQueueError("report queue storage root is invalid")
    text = str(value)
    candidate = Path(text)
    if (
        not text
        or len(text) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
        or not candidate.is_absolute()
        or candidate == Path(candidate.anchor)
        or ".." in candidate.parts
    ):
        raise ReportQueueError("report queue storage root is invalid")
    if admit:
        try:
            if (
                candidate.resolve(strict=True) != candidate
                or not candidate.is_dir()
                or not hasattr(os, "geteuid")
                or candidate.stat().st_uid != os.geteuid()
            ):
                raise ReportQueueError("report queue storage root is invalid")
        except (OSError, RuntimeError):
            raise ReportQueueError("report queue storage root is invalid") from None
    return candidate


def _job_id(root: Path, local_date: date) -> str:
    binding = json.dumps([str(root), local_date.isoformat()], separators=(",", ":"))
    return "report_" + hashlib.sha256(binding.encode("utf-8")).hexdigest()


def _stored_time(value: str | None, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        normalized = _utc(parsed)
        if normalized.isoformat() != value:
            raise ValueError
        return normalized
    except (TypeError, ValueError):
        raise ReportQueueError("report queue stored timestamp is invalid") from None


@dataclass(frozen=True, slots=True)
class ReportRefreshJob:
    job_id: str
    local_date: date
    timezone_name: str
    start_at: datetime
    end_at: datetime
    storage_root: Path = field(repr=False)
    requested_revision: int
    claimed_revision: int
    completed_revision: int
    claim_token: str | None = field(repr=False)
    status: ReportRefreshStatus
    attempt_count: int
    available_at: datetime
    lease_until: datetime | None
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime
    last_requested_at: datetime
    completed_at: datetime | None

    def to_public_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "local_date": self.local_date.isoformat(),
            "timezone_name": self.timezone_name,
            "start_at": self.start_at.isoformat(),
            "end_at": self.end_at.isoformat(),
            "requested_revision": self.requested_revision,
            "claimed_revision": self.claimed_revision,
            "completed_revision": self.completed_revision,
            "dirty": self.requested_revision > self.completed_revision,
            "status": self.status.value,
            "attempt_count": self.attempt_count,
            "available_at": self.available_at.isoformat(),
            "lease_until": self.lease_until.isoformat() if self.lease_until else None,
            "last_error_code": self.last_error_code,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_requested_at": self.last_requested_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "paths_disclosed": False,
            "claim_tokens_disclosed": False,
            "request_ids_disclosed": False,
        }


class ReportRefreshQueue:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def _connection(
        self, *, write: bool = False, initialize: bool = False
    ) -> Iterator[sqlite3.Connection | None]:
        connection = None
        try:
            if initialize:
                self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                try:
                    descriptor = os.open(
                        self.path,
                        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                except FileExistsError:
                    pass
                else:
                    try:
                        os.fchmod(descriptor, 0o600)
                    finally:
                        os.close(descriptor)
            elif not self.path.is_file():
                yield None
                return
            if self.path.is_symlink():
                raise ReportQueueError("report queue database is unavailable")
            if write and initialize:
                connection = sqlite3.connect(self.path)
            else:
                uri = self.path.resolve().as_uri() + ("?mode=rw" if write else "?mode=ro")
                connection = sqlite3.connect(uri, uri=True)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            if write:
                connection.execute("BEGIN IMMEDIATE")
                if initialize:
                    self._present(connection)
                    for statement in REPORT_REFRESH_SCHEMA_SQL.split(";"):
                        if statement.strip():
                            connection.execute(statement)
            else:
                connection.execute("PRAGMA query_only = ON")
            yield connection
            if write:
                connection.commit()
        except (sqlite3.Error, OSError):
            raise ReportQueueError("report queue database is unavailable or invalid") from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _present(connection: sqlite3.Connection | None) -> bool:
        if connection is None:
            return False
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('report_refresh_jobs', 'report_refresh_requests')"
            )
        }
        if not tables:
            return False
        if len(tables) != 2:
            raise ReportQueueError("report queue schema is incomplete")
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(report_refresh_jobs)")
        }
        if columns != _JOB_COLUMNS:
            raise ReportQueueError("report queue schema is unsupported")
        request_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(report_refresh_requests)")
        }
        if request_columns != {"request_id", "job_id", "requested_revision", "requested_at"}:
            raise ReportQueueError("report queue schema is unsupported")
        return True

    @staticmethod
    def _decode(row: sqlite3.Row) -> ReportRefreshJob:
        try:
            local_date = date.fromisoformat(row["local_date"])
            root = _root(row["storage_root"], admit=False)
            job_id = _identifier(row["job_id"], _JOB_ID)
            timezone_name = _timezone(row["timezone_name"])
            start_at, end_at = _day_bounds(local_date, timezone_name)
            if _stored_time(row["start_at"]) != start_at or _stored_time(row["end_at"]) != end_at:
                raise ReportQueueError("report queue frozen day boundary is inconsistent")
            if job_id != _job_id(root, local_date):
                raise ReportQueueError("report queue destination binding is invalid")
            revisions = tuple(
                _integer(row[name], 0, 2**63 - 1)
                for name in ("completed_revision", "claimed_revision", "requested_revision")
            )
            if revisions != tuple(sorted(revisions)) or revisions[-1] < 1:
                raise ReportQueueError("report queue revision state is invalid")
            status = ReportRefreshStatus(row["status"])
            token = row["claim_token"]
            lease = _stored_time(row["lease_until"], optional=True)
            if status is ReportRefreshStatus.RUNNING:
                _identifier(token, _TOKEN)
                if lease is None or revisions[1] < 1:
                    raise ReportQueueError("report queue lease is invalid")
            elif token is not None or lease is not None:
                raise ReportQueueError("report queue lease is invalid")
            error = row["last_error_code"]
            if error is not None:
                _identifier(error, _ERROR_CODE)
            return ReportRefreshJob(
                job_id=job_id,
                local_date=local_date,
                timezone_name=timezone_name,
                start_at=start_at,
                end_at=end_at,
                storage_root=root,
                requested_revision=revisions[2],
                claimed_revision=revisions[1],
                completed_revision=revisions[0],
                claim_token=token,
                status=status,
                attempt_count=_integer(row["attempt_count"], 0, 1000000),
                available_at=_stored_time(row["available_at"]),
                lease_until=lease,
                last_error_code=error,
                created_at=_stored_time(row["created_at"]),
                updated_at=_stored_time(row["updated_at"]),
                last_requested_at=_stored_time(row["last_requested_at"]),
                completed_at=_stored_time(row["completed_at"], optional=True),
            )
        except (KeyError, IndexError, TypeError, ValueError):
            raise ReportQueueError("report queue stored job is invalid") from None

    @classmethod
    def _get(cls, connection: sqlite3.Connection, job_id: str) -> ReportRefreshJob | None:
        row = connection.execute(
            "SELECT * FROM report_refresh_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return cls._decode(row) if row is not None else None

    @staticmethod
    def _capacity(connection: sqlite3.Connection, maximum: int) -> None:
        active = connection.execute(
            "SELECT COUNT(*) FROM report_refresh_jobs WHERE status IN ('pending','running')"
        ).fetchone()[0]
        if active >= maximum:
            raise ReportQueueFullError("report refresh queue is full")

    def request(
        self,
        request_id: str,
        local_date: date,
        timezone_name: str,
        storage_root: Path,
        *,
        max_active: int = 366,
        debounce_seconds: float = 30,
        now: datetime | None = None,
    ) -> ReportRefreshJob:
        _identifier(request_id, _IDENTIFIER)
        if type(local_date) is not date:
            raise ReportQueueError("report queue local date is invalid")
        timezone_name = _timezone(timezone_name)
        start_at, end_at = _day_bounds(local_date, timezone_name)
        root = _root(storage_root)
        maximum = _integer(max_active, 1, 10000)
        observed = _utc(now)
        available = _later(observed, _seconds(debounce_seconds))
        timestamp = observed.isoformat()
        job_id = _job_id(root, local_date)
        with self._connection(write=True, initialize=True) as connection:
            assert connection is not None
            self._present(connection)
            prior = connection.execute(
                "SELECT job_id FROM report_refresh_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            if prior is not None and prior["job_id"] != job_id:
                raise ReportQueueError("report request already belongs to another destination")
            job = self._get(connection, job_id)
            if job is not None and job.timezone_name != timezone_name:
                raise ReportQueueError("report destination already uses another timezone")
            if prior is not None:
                if job is None:
                    raise ReportQueueError("report request destination is missing")
                return job
            if job is None:
                self._capacity(connection, maximum)
                connection.execute(
                    "INSERT INTO report_refresh_jobs "
                    "(job_id,local_date,timezone_name,start_at,end_at,storage_root,"
                    "requested_revision,status,"
                    "available_at,created_at,updated_at,last_requested_at) "
                    "VALUES (?,?,?,?,?,?,1,'pending',?,?,?,?)",
                    (
                        job_id,
                        local_date.isoformat(),
                        timezone_name,
                        start_at.isoformat(),
                        end_at.isoformat(),
                        str(root),
                        available,
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
                revision = 1
            else:
                revision = _integer(job.requested_revision + 1, 1, 2**63 - 1)
                if job.status is ReportRefreshStatus.COMPLETED:
                    self._capacity(connection, maximum)
                    connection.execute(
                        "UPDATE report_refresh_jobs SET status='pending', attempt_count=0,"
                        "available_at=?,last_error_code=NULL WHERE job_id=?",
                        (available, job_id),
                    )
                connection.execute(
                    "UPDATE report_refresh_jobs SET requested_revision=?,updated_at=?,"
                    "last_requested_at=? WHERE job_id=?",
                    (revision, timestamp, timestamp, job_id),
                )
            connection.execute(
                "INSERT INTO report_refresh_requests "
                "(request_id,job_id,requested_revision,requested_at) VALUES (?,?,?,?)",
                (request_id, job_id, revision, timestamp),
            )
            result = self._get(connection, job_id)
            assert result is not None
            return result

    def claim_next(
        self, *, lease_seconds: int = 300, max_attempts: int = 3, now: datetime | None = None
    ) -> ReportRefreshJob | None:
        lease_seconds = _integer(lease_seconds, 1, 86400)
        maximum = _integer(max_attempts, 1, 100)
        observed = _utc(now)
        timestamp = observed.isoformat()
        lease_until = _later(observed, lease_seconds)
        with self._connection(write=True, initialize=True) as connection:
            assert connection is not None
            self._present(connection)
            connection.execute(
                "UPDATE report_refresh_jobs SET "
                "status=CASE WHEN attempt_count>=? THEN 'failed' ELSE 'pending' END,"
                "claim_token=NULL,lease_until=NULL,available_at=?,updated_at=?,"
                "last_error_code='lease_expired' WHERE status='running' AND lease_until<=?",
                (maximum, timestamp, timestamp, timestamp),
            )
            connection.execute(
                "UPDATE report_refresh_jobs SET status='failed',updated_at=? "
                "WHERE status='pending' AND attempt_count>=?",
                (timestamp, maximum),
            )
            row = connection.execute(
                "SELECT job_id FROM report_refresh_jobs WHERE status='pending' AND available_at<=? "
                "ORDER BY available_at,created_at,job_id LIMIT 1",
                (timestamp,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE report_refresh_jobs SET status='running',"
                "claimed_revision=requested_revision,"
                "claim_token=?,lease_until=?,attempt_count=attempt_count+1,updated_at=?,"
                "last_error_code=NULL WHERE job_id=?",
                (secrets.token_hex(32), lease_until, timestamp, row["job_id"]),
            )
            return self._get(connection, row["job_id"])

    def _current(
        self, connection: sqlite3.Connection | None, job: ReportRefreshJob
    ) -> ReportRefreshJob:
        if (
            not isinstance(job, ReportRefreshJob)
            or job.status is not ReportRefreshStatus.RUNNING
            or not job.claim_token
            or not self._present(connection)
        ):
            raise ReportClaimLostError("report refresh claim is no longer current")
        assert connection is not None
        current = self._get(connection, job.job_id)
        if (
            current is None
            or current.status is not ReportRefreshStatus.RUNNING
            or current.claim_token != job.claim_token
            or current.claimed_revision != job.claimed_revision
            or current.storage_root != job.storage_root
            or current.local_date != job.local_date
            or current.timezone_name != job.timezone_name
            or current.start_at != job.start_at
            or current.end_at != job.end_at
            or current.lease_until is None
        ):
            raise ReportClaimLostError("report refresh claim is no longer current")
        return current

    def assert_claim_current(self, job: ReportRefreshJob) -> None:
        with self._connection() as connection:
            self._current(connection, job)

    def complete(self, job: ReportRefreshJob, now: datetime | None = None) -> ReportRefreshJob:
        observed = _utc(now)
        timestamp = observed.isoformat()
        with self._connection(write=True) as connection:
            current = self._current(connection, job)
            assert connection is not None
            dirty = current.requested_revision > current.claimed_revision
            connection.execute(
                "UPDATE report_refresh_jobs SET status=?,completed_revision=claimed_revision,"
                "claim_token=NULL,lease_until=NULL,attempt_count=?,available_at=?,updated_at=?,"
                "completed_at=?,last_error_code=NULL WHERE job_id=? AND claim_token=? "
                "AND claimed_revision=?",
                (
                    "pending" if dirty else "completed",
                    0 if dirty else current.attempt_count,
                    timestamp,
                    timestamp,
                    timestamp,
                    job.job_id,
                    job.claim_token,
                    job.claimed_revision,
                ),
            )
            result = self._get(connection, job.job_id)
            assert result is not None
            return result

    def fail(
        self,
        job: ReportRefreshJob,
        *,
        error_code: str,
        retry_delay_seconds: float = 60,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> ReportRefreshJob:
        error_code = _identifier(error_code, _ERROR_CODE)
        maximum = _integer(max_attempts, 1, 100)
        observed = _utc(now)
        available = _later(observed, _seconds(retry_delay_seconds))
        with self._connection(write=True) as connection:
            current = self._current(connection, job)
            assert connection is not None
            connection.execute(
                "UPDATE report_refresh_jobs SET status=?,claim_token=NULL,lease_until=NULL,"
                "available_at=?,updated_at=?,last_error_code=? "
                "WHERE job_id=? AND claim_token=? AND claimed_revision=?",
                (
                    "failed" if current.attempt_count >= maximum else "pending",
                    available,
                    observed.isoformat(),
                    error_code,
                    job.job_id,
                    job.claim_token,
                    job.claimed_revision,
                ),
            )
            result = self._get(connection, job.job_id)
            assert result is not None
            return result

    def retry_failed(
        self, job_id: str, *, max_active: int = 366, now: datetime | None = None
    ) -> ReportRefreshJob:
        _identifier(job_id, _JOB_ID)
        maximum = _integer(max_active, 1, 10000)
        timestamp = _utc(now).isoformat()
        with self._connection(write=True) as connection:
            if not self._present(connection):
                raise ReportQueueError("report refresh job is unavailable")
            assert connection is not None
            job = self._get(connection, job_id)
            if job is None or job.status is not ReportRefreshStatus.FAILED:
                raise ReportQueueError("only failed report refreshes can be retried")
            self._capacity(connection, maximum)
            connection.execute(
                "UPDATE report_refresh_jobs SET status='pending',attempt_count=0,available_at=?,"
                "updated_at=?,last_error_code=NULL WHERE job_id=?",
                (timestamp, timestamp, job_id),
            )
            result = self._get(connection, job_id)
            assert result is not None
            return result

    def get_read_only(self, job_id: str) -> ReportRefreshJob | None:
        _identifier(job_id, _JOB_ID)
        with self._connection() as connection:
            if not self._present(connection):
                return None
            assert connection is not None
            return self._get(connection, job_id)

    def snapshot_read_only(self, now: datetime | None = None) -> dict[str, object]:
        observed = _utc(now)
        counts = {state.value: 0 for state in ReportRefreshStatus}
        snapshot: dict[str, object] = {
            "version": 1,
            "table_present": False,
            "counts": counts,
            "ready": 0,
            "expired_leases": 0,
            "dirty_jobs": 0,
            "oldest_pending_age_seconds": None,
            "errors": [],
            "unresolved_jobs": [],
            "unresolved_jobs_limit": _UNRESOLVED_JOB_LIMIT,
            "paths_disclosed": False,
            "claim_tokens_disclosed": False,
            "request_ids_disclosed": False,
        }
        with self._connection() as connection:
            present = self._present(connection)
            snapshot["table_present"] = present
            if present:
                assert connection is not None
                for row in connection.execute(
                    "SELECT status,COUNT(*) AS total FROM report_refresh_jobs GROUP BY status"
                ):
                    if row["status"] not in counts:
                        raise ReportQueueError("report queue stored status is invalid")
                    counts[row["status"]] = row["total"]
                row = connection.execute(
                    "SELECT SUM(status='pending' AND available_at<=?) AS ready,"
                    "SUM(status='running' AND lease_until<=?) AS expired,"
                    "SUM(requested_revision>completed_revision) AS dirty,"
                    "MIN(CASE WHEN status='pending' THEN available_at END) AS oldest "
                    "FROM report_refresh_jobs",
                    (observed.isoformat(), observed.isoformat()),
                ).fetchone()
                snapshot.update(
                    ready=row["ready"] or 0,
                    expired_leases=row["expired"] or 0,
                    dirty_jobs=row["dirty"] or 0,
                )
                if row["oldest"] is not None:
                    oldest = _stored_time(row["oldest"])
                    assert oldest is not None
                    snapshot["oldest_pending_age_seconds"] = max(
                        0.0, (observed - oldest).total_seconds()
                    )
                errors = []
                for row in connection.execute(
                    "SELECT last_error_code,COUNT(*) AS total FROM report_refresh_jobs "
                    "WHERE last_error_code IS NOT NULL GROUP BY last_error_code "
                    "ORDER BY last_error_code LIMIT 64"
                ):
                    code = _identifier(row["last_error_code"], _ERROR_CODE)
                    errors.append({"error_code": code, "count": row["total"]})
                snapshot["errors"] = errors
                snapshot["unresolved_jobs"] = [
                    self._decode(row).to_public_dict()
                    for row in connection.execute(
                        "SELECT * FROM report_refresh_jobs "
                        "WHERE status IN ('failed','running','pending') "
                        "ORDER BY CASE WHEN status='failed' THEN 0 ELSE 1 END,"
                        "available_at,job_id LIMIT ?",
                        (_UNRESOLVED_JOB_LIMIT,),
                    )
                ]
        counts["active"] = counts["pending"] + counts["running"]
        counts["total"] = sum(counts[state.value] for state in ReportRefreshStatus)
        return snapshot
