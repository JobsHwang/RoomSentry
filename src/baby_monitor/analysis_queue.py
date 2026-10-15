from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from baby_monitor.adapters.camera import redact_uri
from baby_monitor.domain import CandidateEvent, Detection, EventMedia, SignalEvent

QUEUE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analysis_jobs (
    event_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    clip_path TEXT NOT NULL,
    frame_paths_json TEXT NOT NULL,
    evidence_json TEXT,
    storage_root TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
    available_at TEXT NOT NULL,
    lease_until TEXT,
    last_error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status_available
    ON analysis_jobs (status, available_at, occurred_at);
"""


class AnalysisQueueError(RuntimeError):
    pass


class QueueFullError(AnalysisQueueError):
    pass


class AnalysisJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AnalysisJob:
    candidate: CandidateEvent
    occurred_at: datetime
    source_kind: str
    source_ref: str
    media: EventMedia
    storage_root: str
    status: AnalysisJobStatus
    attempt_count: int
    available_at: datetime
    lease_until: datetime | None
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    @property
    def event_id(self) -> str:
        return self.candidate.event_id

    def require_local_media(self) -> None:
        clip = Path(self.media.clip_path)
        frames = tuple(Path(path) for path in self.media.frame_paths)
        if not clip.is_absolute() or clip.is_symlink() or not clip.is_file():
            raise AnalysisQueueError("queued event clip is missing or unsafe")
        if not 1 <= len(frames) <= 8:
            raise AnalysisQueueError("queued event must contain between 1 and 8 frames")
        if any(
            not frame.is_absolute() or frame.is_symlink() or not frame.is_file() for frame in frames
        ):
            raise AnalysisQueueError("a queued evidence frame is missing or unsafe")


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    job: AnalysisJob
    reused_existing: bool


@dataclass(frozen=True, slots=True)
class QueueCounts:
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
class QueueSnapshot:
    counts: QueueCounts
    ready_pending: int
    oldest_pending_age_seconds: float | None
    queued_media_files: int
    queued_media_bytes: int
    missing_media_files: int
    failed_error_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **self.counts.to_dict(),
            "ready_pending": self.ready_pending,
            "oldest_pending_age_seconds": self.oldest_pending_age_seconds,
            "queued_media_files": self.queued_media_files,
            "queued_media_bytes": self.queued_media_bytes,
            "missing_media_files": self.missing_media_files,
            "failed_error_counts": dict(self.failed_error_counts),
        }


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _candidate_from_dict(value: Any) -> CandidateEvent:
    if not isinstance(value, dict):
        raise AnalysisQueueError("queued candidate JSON is invalid")
    try:
        detections = tuple(
            Detection(
                timestamp_seconds=float(item["timestamp_seconds"]),
                label=str(item["label"]),
                confidence=float(item["confidence"]),
                bbox_xyxy=tuple(float(part) for part in item["bbox_xyxy"]),
            )
            for item in value["detections"]
        )
        signals = tuple(SignalEvent.from_dict(item) for item in value.get("signals", []))
        return CandidateEvent(
            event_id=str(value["event_id"]),
            camera_id=str(value["camera_id"]),
            start_seconds=float(value["start_seconds"]),
            end_seconds=float(value["end_seconds"]),
            detections=detections,
            signals=signals,
            detector_id=value.get("detector_id"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisQueueError("queued candidate JSON is invalid") from exc


class AnalysisQueue:
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
            connection.executescript(QUEUE_SCHEMA_SQL)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(analysis_jobs)").fetchall()
            }
            if "evidence_json" not in columns:
                connection.execute("ALTER TABLE analysis_jobs ADD COLUMN evidence_json TEXT")

    def enqueue(
        self,
        *,
        candidate: CandidateEvent,
        occurred_at: datetime,
        source_kind: str,
        source_ref: str,
        media: EventMedia,
        storage_root: str | Path,
        max_active: int,
        now: datetime | None = None,
    ) -> EnqueueResult:
        if not 1 <= max_active <= 10_000:
            raise ValueError("max_active must be between 1 and 10000")
        if not source_kind or not source_ref:
            raise ValueError("source identity cannot be empty")
        if source_kind == "rtsp" and redact_uri(source_ref) != source_ref:
            raise AnalysisQueueError("queued RTSP source reference must be redacted")
        occurred = _utc(occurred_at, field="occurred_at")
        queued_at = _utc(now or datetime.now(UTC), field="now")
        root = Path(storage_root).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise AnalysisQueueError("queue storage root must be an existing absolute directory")
        pending_job = AnalysisJob(
            candidate=candidate,
            occurred_at=occurred,
            source_kind=source_kind,
            source_ref=source_ref,
            media=media,
            storage_root=str(root),
            status=AnalysisJobStatus.PENDING,
            attempt_count=0,
            available_at=queued_at,
            lease_until=None,
            last_error_code=None,
            created_at=queued_at,
            updated_at=queued_at,
            completed_at=None,
        )
        pending_job.require_local_media()
        candidate_json = _json(candidate.to_dict())
        frames_json = _json(list(media.frame_paths))
        evidence_json = _json(media.to_dict())
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM analysis_jobs WHERE event_id = ?", (candidate.event_id,)
            ).fetchone()
            if existing is not None:
                immutable = (
                    existing["camera_id"] == candidate.camera_id
                    and existing["occurred_at"] == occurred.isoformat()
                    and existing["source_kind"] == source_kind
                    and existing["source_ref"] == source_ref
                    and existing["candidate_json"] == candidate_json
                    and existing["clip_path"] == media.clip_path
                    and existing["frame_paths_json"] == frames_json
                    and (
                        existing["evidence_json"] is None
                        or existing["evidence_json"] == evidence_json
                    )
                    and existing["storage_root"] == str(root)
                )
                if not immutable:
                    raise AnalysisQueueError(
                        "event_id already exists with different queued event data"
                    )
                connection.commit()
                result = EnqueueResult(self._row_to_job(existing), reused_existing=True)
                return result
            active = connection.execute(
                "SELECT COUNT(*) FROM analysis_jobs WHERE status IN ('pending', 'running')"
            ).fetchone()[0]
            queue_full = int(active) >= max_active
            status = AnalysisJobStatus.FAILED if queue_full else AnalysisJobStatus.PENDING
            last_error_code = "queue_full" if queue_full else None
            connection.execute(
                """
                INSERT INTO analysis_jobs (
                    event_id, camera_id, occurred_at, source_kind, source_ref,
                    candidate_json, clip_path, frame_paths_json, evidence_json, storage_root,
                    status, attempt_count, available_at, lease_until,
                    last_error_code, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.event_id,
                    candidate.camera_id,
                    occurred.isoformat(),
                    source_kind,
                    source_ref,
                    candidate_json,
                    media.clip_path,
                    frames_json,
                    evidence_json,
                    str(root),
                    status.value,
                    0,
                    queued_at.isoformat(),
                    None,
                    last_error_code,
                    queued_at.isoformat(),
                    queued_at.isoformat(),
                    None,
                ),
            )
            row = connection.execute(
                "SELECT * FROM analysis_jobs WHERE event_id = ?", (candidate.event_id,)
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        if row is None:
            raise AnalysisQueueError("queued event disappeared from SQLite")
        if queue_full:
            raise QueueFullError(
                "analysis queue is full; extracted evidence was preserved and tracked"
            )
        return EnqueueResult(self._row_to_job(row), reused_existing=False)

    def claim_next(
        self,
        *,
        lease_seconds: int,
        max_attempts: int,
        now: datetime | None = None,
    ) -> AnalysisJob | None:
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
                UPDATE analysis_jobs
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
                SELECT * FROM analysis_jobs
                WHERE status = 'pending' AND available_at <= ?
                ORDER BY occurred_at ASC, created_at ASC, event_id ASC
                LIMIT 1
                """,
                (claimed_at.isoformat(),),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            updated = connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'running', attempt_count = attempt_count + 1,
                    lease_until = ?, updated_at = ?, last_error_code = NULL
                WHERE event_id = ? AND status = 'pending'
                """,
                (lease_until.isoformat(), claimed_at.isoformat(), row["event_id"]),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise AnalysisQueueError("analysis queue claim lost a concurrency race")
            claimed = connection.execute(
                "SELECT * FROM analysis_jobs WHERE event_id = ?", (row["event_id"],)
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        return self._row_to_job(claimed) if claimed is not None else None

    def mark_completed(self, event_id: str, *, now: datetime | None = None) -> AnalysisJob:
        completed_at = _utc(now or datetime.now(UTC), field="now")
        with self._session() as connection:
            updated = connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'completed', lease_until = NULL, last_error_code = NULL,
                    completed_at = ?, updated_at = ?
                WHERE event_id = ? AND status = 'running'
                """,
                (completed_at.isoformat(), completed_at.isoformat(), event_id),
            )
            if updated.rowcount != 1:
                raise AnalysisQueueError("only a running analysis job can complete")
            row = connection.execute(
                "SELECT * FROM analysis_jobs WHERE event_id = ?", (event_id,)
            ).fetchone()
        if row is None:
            raise AnalysisQueueError("completed analysis job disappeared from SQLite")
        return self._row_to_job(row)

    def mark_failed_attempt(
        self,
        event_id: str,
        *,
        error_code: str,
        retry_delay_seconds: int,
        max_attempts: int,
        now: datetime | None = None,
    ) -> AnalysisJob:
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
                "SELECT attempt_count FROM analysis_jobs WHERE event_id = ? AND status = 'running'",
                (event_id,),
            ).fetchone()
            if row is None:
                raise AnalysisQueueError("only a running analysis job can record failure")
            status = (
                AnalysisJobStatus.FAILED
                if int(row["attempt_count"]) >= max_attempts
                else AnalysisJobStatus.PENDING
            )
            connection.execute(
                """
                UPDATE analysis_jobs
                SET status = ?, available_at = ?, lease_until = NULL,
                    last_error_code = ?, updated_at = ?
                WHERE event_id = ?
                """,
                (
                    status.value,
                    available_at.isoformat(),
                    error_code,
                    failed_at.isoformat(),
                    event_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM analysis_jobs WHERE event_id = ?", (event_id,)
            ).fetchone()
        if updated is None:
            raise AnalysisQueueError("failed analysis job disappeared from SQLite")
        return self._row_to_job(updated)

    def retry_failed(
        self,
        event_id: str,
        *,
        max_active: int,
        now: datetime | None = None,
    ) -> AnalysisJob:
        if not 1 <= max_active <= 10_000:
            raise ValueError("max_active must be between 1 and 10000")
        retried_at = _utc(now or datetime.now(UTC), field="now")
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT COUNT(*) FROM analysis_jobs WHERE status IN ('pending', 'running')"
            ).fetchone()[0]
            if int(active) >= max_active:
                raise QueueFullError("analysis queue is full; failed job remains tracked")
            updated = connection.execute(
                """
                UPDATE analysis_jobs
                SET status = 'pending', attempt_count = 0, available_at = ?,
                    lease_until = NULL, last_error_code = NULL, completed_at = NULL,
                    updated_at = ?
                WHERE event_id = ? AND status = 'failed'
                """,
                (retried_at.isoformat(), retried_at.isoformat(), event_id),
            )
            if updated.rowcount != 1:
                raise AnalysisQueueError("only a failed analysis job can be retried")
            row = connection.execute(
                "SELECT * FROM analysis_jobs WHERE event_id = ?", (event_id,)
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        if row is None:
            raise AnalysisQueueError("retried analysis job disappeared from SQLite")
        return self._row_to_job(row)

    def get_read_only(self, event_id: str) -> AnalysisJob | None:
        if not isinstance(event_id, str) or not event_id or len(event_id) > 128:
            raise ValueError("event_id must be a bounded identifier")
        try:
            connection = self._connect_read_only()
        except sqlite3.Error as exc:
            raise AnalysisQueueError("analysis queue database is missing or invalid") from exc
        try:
            row = connection.execute(
                "SELECT * FROM analysis_jobs WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            return self._row_to_job(row) if row is not None else None
        except sqlite3.Error as exc:
            raise AnalysisQueueError("analysis queue database is missing or invalid") from exc
        finally:
            connection.close()

    def counts(self) -> QueueCounts:
        self.initialize()
        with self._session() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM analysis_jobs GROUP BY status"
            ).fetchall()
        values = {str(row["status"]): int(row["count"]) for row in rows}
        return QueueCounts(
            pending=values.get(AnalysisJobStatus.PENDING.value, 0),
            running=values.get(AnalysisJobStatus.RUNNING.value, 0),
            completed=values.get(AnalysisJobStatus.COMPLETED.value, 0),
            failed=values.get(AnalysisJobStatus.FAILED.value, 0),
        )

    def snapshot(self, *, now: datetime | None = None) -> QueueSnapshot:
        observed_at = _utc(now or datetime.now(UTC), field="now")
        self.initialize()
        with self._session() as connection:
            return self._snapshot_from_connection(connection, observed_at=observed_at)

    def snapshot_read_only(self, *, now: datetime | None = None) -> QueueSnapshot:
        """Inspect an existing queue without creating a database or changing its schema."""
        observed_at = _utc(now or datetime.now(UTC), field="now")
        try:
            connection = self._connect_read_only()
        except sqlite3.Error as exc:
            raise AnalysisQueueError("analysis queue database is missing or invalid") from exc
        try:
            return self._snapshot_from_connection(connection, observed_at=observed_at)
        except sqlite3.Error as exc:
            raise AnalysisQueueError("analysis queue database is missing or invalid") from exc
        finally:
            connection.close()

    @staticmethod
    def _snapshot_from_connection(
        connection: sqlite3.Connection,
        *,
        observed_at: datetime,
    ) -> QueueSnapshot:
        rows = connection.execute(
            """
            SELECT status, available_at, created_at, clip_path,
                   frame_paths_json, last_error_code
            FROM analysis_jobs
            WHERE status != 'completed'
            """
        ).fetchall()
        count_rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM analysis_jobs GROUP BY status"
        ).fetchall()
        count_values = {str(row["status"]): int(row["count"]) for row in count_rows}
        counts = QueueCounts(
            pending=count_values.get(AnalysisJobStatus.PENDING.value, 0),
            running=count_values.get(AnalysisJobStatus.RUNNING.value, 0),
            completed=count_values.get(AnalysisJobStatus.COMPLETED.value, 0),
            failed=count_values.get(AnalysisJobStatus.FAILED.value, 0),
        )
        ready_pending = 0
        oldest_pending: datetime | None = None
        media_paths: set[str] = set()
        failed_errors: dict[str, int] = {}
        for row in rows:
            status = AnalysisJobStatus(row["status"])
            if status is AnalysisJobStatus.PENDING:
                available_at = datetime.fromisoformat(row["available_at"])
                created_at = datetime.fromisoformat(row["created_at"])
                ready_pending += int(available_at <= observed_at)
                if oldest_pending is None or created_at < oldest_pending:
                    oldest_pending = created_at
            if status is AnalysisJobStatus.FAILED and row["last_error_code"] is not None:
                code = str(row["last_error_code"])
                failed_errors[code] = failed_errors.get(code, 0) + 1
            media_paths.add(str(row["clip_path"]))
            try:
                frames = json.loads(row["frame_paths_json"])
            except (TypeError, json.JSONDecodeError):
                frames = []
            if isinstance(frames, list):
                media_paths.update(str(path) for path in frames if isinstance(path, str))

        queued_media_bytes = 0
        missing_media_files = 0
        for raw_path in media_paths:
            path = Path(raw_path)
            try:
                if path.is_symlink() or not path.is_file():
                    missing_media_files += 1
                    continue
                queued_media_bytes += path.stat().st_size
            except OSError:
                missing_media_files += 1
        age = (
            max(0.0, (observed_at - oldest_pending).total_seconds())
            if oldest_pending is not None
            else None
        )
        return QueueSnapshot(
            counts=counts,
            ready_pending=ready_pending,
            oldest_pending_age_seconds=age,
            queued_media_files=len(media_paths),
            queued_media_bytes=queued_media_bytes,
            missing_media_files=missing_media_files,
            failed_error_counts=tuple(sorted(failed_errors.items())),
        )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> AnalysisJob:
        try:
            frames = json.loads(row["frame_paths_json"])
            if not isinstance(frames, list) or not all(isinstance(item, str) for item in frames):
                raise ValueError
            media = (
                EventMedia.from_dict(json.loads(row["evidence_json"]))
                if row["evidence_json"] is not None
                else EventMedia(str(row["clip_path"]), tuple(frames))
            )
            if media.clip_path != row["clip_path"] or list(media.frame_paths) != frames:
                raise ValueError
            candidate = _candidate_from_dict(json.loads(row["candidate_json"]))
            return AnalysisJob(
                candidate=candidate,
                occurred_at=datetime.fromisoformat(row["occurred_at"]),
                source_kind=str(row["source_kind"]),
                source_ref=str(row["source_ref"]),
                media=media,
                storage_root=str(row["storage_root"]),
                status=AnalysisJobStatus(row["status"]),
                attempt_count=int(row["attempt_count"]),
                available_at=datetime.fromisoformat(row["available_at"]),
                lease_until=(
                    datetime.fromisoformat(row["lease_until"])
                    if row["lease_until"] is not None
                    else None
                ),
                last_error_code=(
                    str(row["last_error_code"]) if row["last_error_code"] is not None else None
                ),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                completed_at=(
                    datetime.fromisoformat(row["completed_at"])
                    if row["completed_at"] is not None
                    else None
                ),
            )
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AnalysisQueueError("analysis queue row is invalid") from exc
