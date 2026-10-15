from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from .database import StoredEvent
from .domain import (
    EventType,
    FinalSource,
    ReviewStatus,
    ReviewTier,
    RiskLevel,
    SubjectKind,
)

NOTIFICATION_QUEUE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS notification_jobs (
    notification_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    sink_kind TEXT NOT NULL CHECK (sink_kind IN ('stdout', 'file', 'smtp', 'webhook')),
    sink_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'delivered', 'failed')),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
    available_at TEXT NOT NULL,
    lease_until TEXT,
    last_error_code TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    delivered_at TEXT,
    UNIQUE (event_id, sink_kind, sink_id)
);

CREATE INDEX IF NOT EXISTS idx_notification_jobs_status_available
    ON notification_jobs (sink_kind, sink_id, status, available_at, created_at);
"""

_NOTIFICATION_TABLE_SQL = NOTIFICATION_QUEUE_SCHEMA_SQL.split(
    "CREATE INDEX IF NOT EXISTS", maxsplit=1
)[0]
_NOTIFICATION_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS"
    + NOTIFICATION_QUEUE_SCHEMA_SQL.split("CREATE INDEX IF NOT EXISTS", maxsplit=1)[1]
)
_SUPPORTED_SINK_KINDS = frozenset({"stdout", "file", "smtp", "webhook"})

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SAFE_SINK_ID = re.compile(r"[a-z0-9_.-]{1,64}")
_SAFE_CODE = re.compile(r"[a-z0-9_]{1,80}")
_FORBIDDEN_LOCATION = re.compile(
    r"(?i)(?:[a-z]:[\\/]|[a-z][a-z0-9+.-]*://|(?:^|[^A-Za-z0-9])/(?:home|mnt|users|var|tmp)/|\\\\)"
)


class NotificationJobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DELIVERED = "delivered"
    FAILED = "failed"


class NotificationQueueError(RuntimeError):
    pass


class NotificationQueueFullError(NotificationQueueError):
    pass


class NotificationDeliveryError(RuntimeError):
    def __init__(self, code: str):
        if _SAFE_CODE.fullmatch(code) is None:
            raise ValueError("notification error code is invalid")
        self.code = code
        super().__init__("local notification delivery failed")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _strict_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    if set(data) != expected:
        raise ValueError(f"{context} fields do not match the v1 contract")


def _bounded_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 128:
        raise ValueError(f"{field} must be a bounded array")
    result = tuple(value)
    if any(not isinstance(item, str) or not 1 <= len(item) <= 2_000 for item in result):
        raise ValueError(f"{field} contains an invalid string")
    return result


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("notification confidence must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError("notification confidence must be between 0 and 1")
    return parsed


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_timestamp(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from None
    return _utc(parsed, field=field)


@dataclass(frozen=True, slots=True)
class NotificationSubject:
    kind: SubjectKind
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SubjectKind):
            raise ValueError("notification subject kind is invalid")
        _confidence(self.confidence)

    @classmethod
    def from_dict(cls, data: object) -> NotificationSubject:
        if not isinstance(data, dict):
            raise ValueError("notification subject must be an object")
        _strict_keys(data, {"kind", "confidence"}, "notification subject")
        try:
            kind = SubjectKind(data["kind"])
        except (TypeError, ValueError):
            raise ValueError("notification subject kind is invalid") from None
        return cls(kind=kind, confidence=_confidence(data["confidence"]))

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "confidence": self.confidence}


@dataclass(frozen=True, slots=True)
class NotificationMessage:
    notification_id: str
    event_id: str
    camera_id: str
    occurred_at: datetime
    review_status: ReviewStatus
    event_type: EventType
    risk_level: RiskLevel
    confidence: float
    subjects: tuple[NotificationSubject, ...]
    observations: tuple[str, ...]
    inferences: tuple[str, ...]
    summary_zh: str
    review_tier: ReviewTier
    review_reasons: tuple[str, ...]
    contradictions: tuple[str, ...]
    final_source: FinalSource
    remote_used: bool
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported notification message version")
        if re.fullmatch(r"[0-9a-f]{32}", self.notification_id) is None:
            raise ValueError("notification_id must be a 32-character lowercase hex ID")
        for value, field in ((self.event_id, "event_id"), (self.camera_id, "camera_id")):
            if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
                raise ValueError(f"notification {field} is invalid")
        _utc(self.occurred_at, field="notification occurred_at")
        if not isinstance(self.review_status, ReviewStatus):
            raise ValueError("notification review_status is invalid")
        if not isinstance(self.event_type, EventType):
            raise ValueError("notification event_type is invalid")
        if not isinstance(self.risk_level, RiskLevel):
            raise ValueError("notification risk_level is invalid")
        _confidence(self.confidence)
        if (
            not isinstance(self.subjects, tuple)
            or len(self.subjects) > 16
            or any(not isinstance(subject, NotificationSubject) for subject in self.subjects)
        ):
            raise ValueError("notification subjects are invalid")
        for values, field in (
            (self.observations, "observations"),
            (self.inferences, "inferences"),
        ):
            if (
                not isinstance(values, tuple)
                or len(values) > 128
                or any(
                    not isinstance(value, str) or not 1 <= len(value) <= 2_000 for value in values
                )
            ):
                raise ValueError(f"notification {field} is invalid")
        for values, field in (
            (self.review_reasons, "review_reasons"),
            (self.contradictions, "contradictions"),
        ):
            if (
                not isinstance(values, tuple)
                or len(values) > 128
                or any(
                    not isinstance(value, str) or _SAFE_CODE.fullmatch(value) is None
                    for value in values
                )
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"notification {field} is invalid")
        if not isinstance(self.summary_zh, str) or len(self.summary_zh) > 4_096:
            raise ValueError("notification summary_zh is invalid")
        if any(
            _FORBIDDEN_LOCATION.search(value) is not None
            for value in self.observations + self.inferences + (self.summary_zh,)
        ):
            raise ValueError("notification text contains a path or source URI")
        if not isinstance(self.review_tier, ReviewTier):
            raise ValueError("notification review_tier is invalid")
        if not isinstance(self.final_source, FinalSource):
            raise ValueError("notification final_source is invalid")
        if type(self.remote_used) is not bool:
            raise ValueError("notification remote_used must be a boolean")
        if self.remote_used != (self.final_source is FinalSource.REMOTE):
            raise ValueError("notification remote_used and final_source must agree")

    @classmethod
    def from_event(
        cls,
        event: StoredEvent,
        *,
        sink_kind: str,
        sink_id: str,
    ) -> NotificationMessage:
        _validate_sink(sink_kind, sink_id)
        notification_id = _notification_id(event.event_id, sink_kind, sink_id)
        result = event.result
        pending = result.needs_remote_review or event.decision.needs_review
        message = cls(
            notification_id=notification_id,
            event_id=event.event_id,
            camera_id=event.camera_id,
            occurred_at=event.occurred_at,
            review_status=(ReviewStatus.PENDING_REVIEW if pending else ReviewStatus.FINAL),
            event_type=result.event_type,
            risk_level=result.risk_level,
            confidence=result.confidence,
            subjects=tuple(
                NotificationSubject(subject.kind, subject.confidence) for subject in result.subjects
            ),
            observations=result.observations,
            inferences=result.inferences,
            summary_zh=result.summary_zh,
            review_tier=event.decision.review_tier,
            review_reasons=event.decision.review_reasons,
            contradictions=event.decision.contradictions,
            final_source=event.decision.final_source,
            remote_used=event.remote_used,
        )
        serialized = message.to_json(indent=None)
        media_references = (event.evidence.clip_path,) + event.evidence.frame_paths
        for reference in media_references:
            name = Path(reference).name if reference else ""
            if reference and (reference in serialized or (name and name in serialized)):
                raise ValueError("notification text contains an event media reference")
        return message

    @classmethod
    def local_test(
        cls,
        *,
        test_id: str,
        sink_kind: str,
        sink_id: str,
        occurred_at: datetime | None = None,
    ) -> NotificationMessage:
        if re.fullmatch(r"[0-9a-f]{32}", test_id) is None:
            raise ValueError("notification test_id must be a 32-character lowercase hex ID")
        _validate_sink(sink_kind, sink_id)
        event_id = f"local_test_{test_id}"
        return cls(
            notification_id=_notification_id(event_id, sink_kind, sink_id),
            event_id=event_id,
            camera_id="local_system",
            occurred_at=_utc(
                occurred_at or datetime.now(UTC),
                field="notification occurred_at",
            ),
            review_status=ReviewStatus.FINAL,
            event_type=EventType.UNKNOWN,
            risk_level=RiskLevel.NONE,
            confidence=0.0,
            subjects=(),
            observations=("manual_local_notification_test",),
            inferences=(),
            summary_zh="这是一条手动触发的本地通知通道测试，不代表真实监控事件或风险。",
            review_tier=ReviewTier.NONE,
            review_reasons=(),
            contradictions=(),
            final_source=FinalSource.LOCAL_PRIMARY,
            remote_used=False,
        )

    @classmethod
    def from_dict(cls, data: object) -> NotificationMessage:
        if not isinstance(data, dict):
            raise ValueError("notification message must be an object")
        expected = {
            "version",
            "notification_id",
            "event_id",
            "camera_id",
            "occurred_at",
            "review_status",
            "event_type",
            "risk_level",
            "confidence",
            "subjects",
            "observations",
            "inferences",
            "summary_zh",
            "review_tier",
            "review_reasons",
            "contradictions",
            "final_source",
            "remote_used",
        }
        _strict_keys(data, expected, "notification message")
        try:
            subjects_raw = data["subjects"]
            if not isinstance(subjects_raw, list):
                raise TypeError
            return cls(
                version=data["version"],
                notification_id=data["notification_id"],
                event_id=data["event_id"],
                camera_id=data["camera_id"],
                occurred_at=_parse_timestamp(data["occurred_at"], field="occurred_at"),
                review_status=ReviewStatus(data["review_status"]),
                event_type=EventType(data["event_type"]),
                risk_level=RiskLevel(data["risk_level"]),
                confidence=_confidence(data["confidence"]),
                subjects=tuple(NotificationSubject.from_dict(value) for value in subjects_raw),
                observations=_bounded_strings(data["observations"], field="observations"),
                inferences=_bounded_strings(data["inferences"], field="inferences"),
                summary_zh=data["summary_zh"],
                review_tier=ReviewTier(data["review_tier"]),
                review_reasons=_bounded_strings(data["review_reasons"], field="review_reasons"),
                contradictions=_bounded_strings(data["contradictions"], field="contradictions"),
                final_source=FinalSource(data["final_source"]),
                remote_used=data["remote_used"],
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("notification message is invalid") from None

    @classmethod
    def from_json(cls, raw: str) -> NotificationMessage:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 262_144:
            raise ValueError("notification message JSON is invalid or too large")
        try:
            data = json.loads(raw, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, TypeError, ValueError):
            raise ValueError("notification message is not strict JSON") from None
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "notification_id": self.notification_id,
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat(),
            "review_status": self.review_status.value,
            "event_type": self.event_type.value,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "subjects": [subject.to_dict() for subject in self.subjects],
            "observations": list(self.observations),
            "inferences": list(self.inferences),
            "summary_zh": self.summary_zh,
            "review_tier": self.review_tier.value,
            "review_reasons": list(self.review_reasons),
            "contradictions": list(self.contradictions),
            "final_source": self.final_source.value,
            "remote_used": self.remote_used,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )
        if len(payload.encode("utf-8")) > 262_144:
            raise ValueError("notification message exceeds the size limit")
        return payload


def _validate_sink(sink_kind: str, sink_id: str) -> None:
    if sink_kind not in _SUPPORTED_SINK_KINDS:
        raise ValueError("notification sink_kind is unsupported")
    if not isinstance(sink_id, str) or _SAFE_SINK_ID.fullmatch(sink_id) is None:
        raise ValueError("notification sink_id is invalid")


def _notification_id(event_id: str, sink_kind: str, sink_id: str) -> str:
    _validate_sink(sink_kind, sink_id)
    if not isinstance(event_id, str) or _SAFE_ID.fullmatch(event_id) is None:
        raise ValueError("notification event_id is invalid")
    return hashlib.sha256(
        f"notification-v1\0{event_id}\0{sink_kind}\0{sink_id}".encode()
    ).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class NotificationJob:
    message: NotificationMessage
    sink_kind: str
    sink_id: str
    status: NotificationJobStatus
    attempt_count: int
    available_at: datetime
    lease_until: datetime | None
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime
    delivered_at: datetime | None

    @property
    def notification_id(self) -> str:
        return self.message.notification_id


@dataclass(frozen=True, slots=True)
class NotificationEnqueueResult:
    job: NotificationJob
    reused_existing: bool


@dataclass(frozen=True, slots=True)
class NotificationQueueCounts:
    pending: int
    running: int
    delivered: int
    failed: int

    @property
    def active(self) -> int:
        return self.pending + self.running

    def to_dict(self) -> dict[str, int]:
        return {
            "pending": self.pending,
            "running": self.running,
            "delivered": self.delivered,
            "failed": self.failed,
            "active": self.active,
        }


@dataclass(frozen=True, slots=True)
class NotificationQueueSnapshot:
    counts: NotificationQueueCounts
    ready_pending: int
    oldest_pending_age_seconds: float | None
    failed_error_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **self.counts.to_dict(),
            "ready_pending": self.ready_pending,
            "oldest_pending_age_seconds": self.oldest_pending_age_seconds,
            "failed_error_counts": dict(self.failed_error_counts),
        }


class NotificationOutbox:
    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _connect_read_only(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise NotificationQueueError("notification database is missing or invalid")
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro",
                uri=True,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            return connection
        except sqlite3.Error:
            if connection is not None:
                connection.close()
            raise NotificationQueueError("notification database is missing or invalid") from None

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
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'notification_jobs'"
            ).fetchone()
            if row is None:
                connection.executescript(NOTIFICATION_QUEUE_SCHEMA_SQL)
                return
            schema = str(row["sql"] or "")
            if "'smtp'" in schema and "'webhook'" in schema:
                connection.execute(_NOTIFICATION_INDEX_SQL)
                return
            if "'stdout'" not in schema or "'file'" not in schema:
                raise NotificationQueueError("notification database schema is unsupported")
            connection.execute("DROP INDEX IF EXISTS idx_notification_jobs_status_available")
            connection.execute("ALTER TABLE notification_jobs RENAME TO notification_jobs_local_v1")
            connection.execute(_NOTIFICATION_TABLE_SQL)
            connection.execute(
                """
                INSERT INTO notification_jobs (
                    notification_id, event_id, sink_kind, sink_id, payload_json,
                    status, attempt_count, available_at, lease_until, last_error_code,
                    created_at, updated_at, delivered_at
                )
                SELECT notification_id, event_id, sink_kind, sink_id, payload_json,
                       status, attempt_count, available_at, lease_until, last_error_code,
                       created_at, updated_at, delivered_at
                FROM notification_jobs_local_v1
                """
            )
            connection.execute("DROP TABLE notification_jobs_local_v1")
            connection.execute(_NOTIFICATION_INDEX_SQL)

    def is_initialized_read_only(self) -> bool:
        connection = self._connect_read_only()
        try:
            try:
                row = connection.execute(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'notification_jobs'"
                ).fetchone()
            except sqlite3.Error:
                raise NotificationQueueError(
                    "notification database is missing or invalid"
                ) from None
            if row is None:
                return False
            schema = str(row["sql"] or "")
            if "'smtp'" not in schema or "'webhook'" not in schema:
                raise NotificationQueueError("notification database schema is unsupported")
            return True
        finally:
            connection.close()

    def enqueue(
        self,
        *,
        event: StoredEvent,
        sink_kind: str,
        sink_id: str,
        max_active: int,
        now: datetime | None = None,
    ) -> NotificationEnqueueResult:
        message = NotificationMessage.from_event(
            event,
            sink_kind=sink_kind,
            sink_id=sink_id,
        )
        return self.enqueue_message(
            message=message,
            sink_kind=sink_kind,
            sink_id=sink_id,
            max_active=max_active,
            now=now,
        )

    def enqueue_local_test(
        self,
        *,
        test_id: str,
        sink_kind: str,
        sink_id: str,
        max_active: int,
        now: datetime | None = None,
    ) -> NotificationEnqueueResult:
        message = NotificationMessage.local_test(
            test_id=test_id,
            sink_kind=sink_kind,
            sink_id=sink_id,
            occurred_at=now,
        )
        return self.enqueue_message(
            message=message,
            sink_kind=sink_kind,
            sink_id=sink_id,
            max_active=max_active,
            now=now,
        )

    def enqueue_message(
        self,
        *,
        message: NotificationMessage,
        sink_kind: str,
        sink_id: str,
        max_active: int,
        now: datetime | None = None,
    ) -> NotificationEnqueueResult:
        _validate_sink(sink_kind, sink_id)
        if not isinstance(message, NotificationMessage) or message.notification_id != (
            _notification_id(message.event_id, sink_kind, sink_id)
        ):
            raise ValueError("notification message does not match sink selection")
        if not 1 <= max_active <= 10_000:
            raise ValueError("notification max_active must be between 1 and 10000")
        queued_at = _utc(now or datetime.now(UTC), field="now")
        payload_json = message.to_json(indent=None)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM notification_jobs WHERE notification_id = ?",
                (message.notification_id,),
            ).fetchone()
            if existing is not None:
                immutable = (
                    existing["event_id"] == message.event_id
                    and existing["sink_kind"] == sink_kind
                    and existing["sink_id"] == sink_id
                    and existing["payload_json"] == payload_json
                )
                if not immutable:
                    raise NotificationQueueError(
                        "notification_id already exists with different immutable data"
                    )
                connection.commit()
                return NotificationEnqueueResult(
                    job=self._row_to_job(existing),
                    reused_existing=True,
                )
            active = int(
                connection.execute(
                    "SELECT COUNT(*) FROM notification_jobs WHERE status IN ('pending', 'running')"
                ).fetchone()[0]
            )
            queue_full = active >= max_active
            status = NotificationJobStatus.FAILED if queue_full else NotificationJobStatus.PENDING
            error_code = "queue_full" if queue_full else None
            connection.execute(
                """
                INSERT INTO notification_jobs (
                    notification_id, event_id, sink_kind, sink_id, payload_json,
                    status, attempt_count, available_at, lease_until, last_error_code,
                    created_at, updated_at, delivered_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, NULL, ?, ?, ?, NULL)
                """,
                (
                    message.notification_id,
                    message.event_id,
                    sink_kind,
                    sink_id,
                    payload_json,
                    status.value,
                    queued_at.isoformat(),
                    error_code,
                    queued_at.isoformat(),
                    queued_at.isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM notification_jobs WHERE notification_id = ?",
                (message.notification_id,),
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        if row is None:
            raise NotificationQueueError("notification job disappeared from SQLite")
        if queue_full:
            raise NotificationQueueFullError(
                "notification queue is full; the failed job remains tracked"
            )
        return NotificationEnqueueResult(self._row_to_job(row), reused_existing=False)

    def claim_next(
        self,
        *,
        sink_kind: str,
        sink_id: str,
        lease_seconds: int,
        max_attempts: int,
        now: datetime | None = None,
    ) -> NotificationJob | None:
        _validate_sink(sink_kind, sink_id)
        if not 10 <= lease_seconds <= 86_400:
            raise ValueError("notification lease_seconds must be between 10 and 86400")
        if not 1 <= max_attempts <= 100:
            raise ValueError("notification max_attempts must be between 1 and 100")
        claimed_at = _utc(now or datetime.now(UTC), field="now")
        lease_until = claimed_at + timedelta(seconds=lease_seconds)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE notification_jobs
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
                SELECT * FROM notification_jobs
                WHERE sink_kind = ? AND sink_id = ?
                  AND status = 'pending' AND available_at <= ?
                ORDER BY created_at ASC, notification_id ASC
                LIMIT 1
                """,
                (sink_kind, sink_id, claimed_at.isoformat()),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            updated = connection.execute(
                """
                UPDATE notification_jobs
                SET status = 'running', attempt_count = attempt_count + 1,
                    lease_until = ?, updated_at = ?, last_error_code = NULL
                WHERE notification_id = ? AND status = 'pending'
                """,
                (
                    lease_until.isoformat(),
                    claimed_at.isoformat(),
                    row["notification_id"],
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise NotificationQueueError("notification queue claim lost a concurrency race")
            claimed = connection.execute(
                "SELECT * FROM notification_jobs WHERE notification_id = ?",
                (row["notification_id"],),
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        return self._row_to_job(claimed) if claimed is not None else None

    def claim_notification(
        self,
        notification_id: str,
        *,
        sink_kind: str,
        sink_id: str,
        lease_seconds: int,
        max_attempts: int,
        now: datetime | None = None,
    ) -> NotificationJob | None:
        _validate_sink(sink_kind, sink_id)
        if re.fullmatch(r"[0-9a-f]{32}", notification_id) is None:
            raise ValueError("notification_id must be a 32-character lowercase hex ID")
        if not 10 <= lease_seconds <= 86_400:
            raise ValueError("notification lease_seconds must be between 10 and 86400")
        if not 1 <= max_attempts <= 100:
            raise ValueError("notification max_attempts must be between 1 and 100")
        claimed_at = _utc(now or datetime.now(UTC), field="now")
        lease_until = claimed_at + timedelta(seconds=lease_seconds)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE notification_jobs
                SET status = CASE WHEN attempt_count >= ? THEN 'failed' ELSE 'pending' END,
                    available_at = ?, lease_until = NULL,
                    last_error_code = 'lease_expired', updated_at = ?
                WHERE notification_id = ? AND status = 'running' AND lease_until <= ?
                """,
                (
                    max_attempts,
                    claimed_at.isoformat(),
                    claimed_at.isoformat(),
                    notification_id,
                    claimed_at.isoformat(),
                ),
            )
            row = connection.execute(
                """
                SELECT * FROM notification_jobs
                WHERE notification_id = ? AND sink_kind = ? AND sink_id = ?
                  AND status = 'pending' AND available_at <= ?
                """,
                (
                    notification_id,
                    sink_kind,
                    sink_id,
                    claimed_at.isoformat(),
                ),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            updated = connection.execute(
                """
                UPDATE notification_jobs
                SET status = 'running', attempt_count = attempt_count + 1,
                    lease_until = ?, updated_at = ?, last_error_code = NULL
                WHERE notification_id = ? AND status = 'pending'
                """,
                (
                    lease_until.isoformat(),
                    claimed_at.isoformat(),
                    notification_id,
                ),
            )
            if updated.rowcount != 1:
                connection.rollback()
                raise NotificationQueueError("notification queue claim lost a concurrency race")
            claimed = connection.execute(
                "SELECT * FROM notification_jobs WHERE notification_id = ?",
                (notification_id,),
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        return self._row_to_job(claimed) if claimed is not None else None

    def mark_delivered(
        self,
        notification_id: str,
        *,
        now: datetime | None = None,
    ) -> NotificationJob:
        delivered_at = _utc(now or datetime.now(UTC), field="now")
        with self._session() as connection:
            updated = connection.execute(
                """
                UPDATE notification_jobs
                SET status = 'delivered', lease_until = NULL, last_error_code = NULL,
                    delivered_at = ?, updated_at = ?
                WHERE notification_id = ? AND status = 'running'
                """,
                (
                    delivered_at.isoformat(),
                    delivered_at.isoformat(),
                    notification_id,
                ),
            )
            if updated.rowcount != 1:
                raise NotificationQueueError("only a running notification job can be delivered")
            row = connection.execute(
                "SELECT * FROM notification_jobs WHERE notification_id = ?",
                (notification_id,),
            ).fetchone()
        if row is None:
            raise NotificationQueueError("delivered notification disappeared from SQLite")
        return self._row_to_job(row)

    def mark_failed_attempt(
        self,
        notification_id: str,
        *,
        error_code: str,
        retry_delay_seconds: int,
        max_attempts: int,
        now: datetime | None = None,
    ) -> NotificationJob:
        if _SAFE_CODE.fullmatch(error_code) is None:
            raise ValueError("notification error_code is invalid")
        if not 0 <= retry_delay_seconds <= 86_400:
            raise ValueError("notification retry delay must be between 0 and 86400")
        if not 1 <= max_attempts <= 100:
            raise ValueError("notification max_attempts must be between 1 and 100")
        failed_at = _utc(now or datetime.now(UTC), field="now")
        available_at = failed_at + timedelta(seconds=retry_delay_seconds)
        with self._session() as connection:
            row = connection.execute(
                "SELECT attempt_count FROM notification_jobs "
                "WHERE notification_id = ? AND status = 'running'",
                (notification_id,),
            ).fetchone()
            if row is None:
                raise NotificationQueueError("only a running notification job can record failure")
            status = (
                NotificationJobStatus.FAILED
                if int(row["attempt_count"]) >= max_attempts
                else NotificationJobStatus.PENDING
            )
            connection.execute(
                """
                UPDATE notification_jobs
                SET status = ?, available_at = ?, lease_until = NULL,
                    last_error_code = ?, updated_at = ?
                WHERE notification_id = ?
                """,
                (
                    status.value,
                    available_at.isoformat(),
                    error_code,
                    failed_at.isoformat(),
                    notification_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM notification_jobs WHERE notification_id = ?",
                (notification_id,),
            ).fetchone()
        if updated is None:
            raise NotificationQueueError("failed notification disappeared from SQLite")
        return self._row_to_job(updated)

    def retry_failed(
        self,
        notification_id: str,
        *,
        max_active: int,
        now: datetime | None = None,
    ) -> NotificationJob:
        if not 1 <= max_active <= 10_000:
            raise ValueError("notification max_active must be between 1 and 10000")
        retried_at = _utc(now or datetime.now(UTC), field="now")
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = int(
                connection.execute(
                    "SELECT COUNT(*) FROM notification_jobs WHERE status IN ('pending', 'running')"
                ).fetchone()[0]
            )
            if active >= max_active:
                raise NotificationQueueFullError(
                    "notification queue is full; failed job remains tracked"
                )
            updated = connection.execute(
                """
                UPDATE notification_jobs
                SET status = 'pending', attempt_count = 0, available_at = ?,
                    lease_until = NULL, last_error_code = NULL, delivered_at = NULL,
                    updated_at = ?
                WHERE notification_id = ? AND status = 'failed'
                """,
                (retried_at.isoformat(), retried_at.isoformat(), notification_id),
            )
            if updated.rowcount != 1:
                raise NotificationQueueError("only a failed notification job can be retried")
            row = connection.execute(
                "SELECT * FROM notification_jobs WHERE notification_id = ?",
                (notification_id,),
            ).fetchone()
            connection.commit()
        finally:
            connection.close()
        if row is None:
            raise NotificationQueueError("retried notification disappeared from SQLite")
        return self._row_to_job(row)

    def snapshot_read_only(
        self,
        *,
        now: datetime | None = None,
    ) -> NotificationQueueSnapshot:
        observed_at = _utc(now or datetime.now(UTC), field="now")
        connection = self._connect_read_only()
        try:
            try:
                count_rows = connection.execute(
                    "SELECT status, COUNT(*) AS count FROM notification_jobs GROUP BY status"
                ).fetchall()
                values = {str(row["status"]): int(row["count"]) for row in count_rows}
                if set(values) - {status.value for status in NotificationJobStatus}:
                    raise ValueError
                counts = NotificationQueueCounts(
                    pending=values.get(NotificationJobStatus.PENDING.value, 0),
                    running=values.get(NotificationJobStatus.RUNNING.value, 0),
                    delivered=values.get(NotificationJobStatus.DELIVERED.value, 0),
                    failed=values.get(NotificationJobStatus.FAILED.value, 0),
                )
                pending = connection.execute(
                    """
                    SELECT COALESCE(SUM(CASE WHEN available_at <= ? THEN 1 ELSE 0 END), 0)
                               AS ready_pending,
                           MIN(created_at) AS oldest_created
                    FROM notification_jobs WHERE status = 'pending'
                    """,
                    (observed_at.isoformat(),),
                ).fetchone()
                failed_rows = connection.execute(
                    """
                    SELECT last_error_code, COUNT(*) AS count
                    FROM notification_jobs
                    WHERE status = 'failed' AND last_error_code IS NOT NULL
                    GROUP BY last_error_code
                    """
                ).fetchall()
                oldest = (
                    _parse_timestamp(pending["oldest_created"], field="oldest_created")
                    if pending["oldest_created"] is not None
                    else None
                )
                failed_error_counts = tuple(
                    sorted((str(row["last_error_code"]), int(row["count"])) for row in failed_rows)
                )
                if any(
                    _SAFE_CODE.fullmatch(code) is None or count < 1
                    for code, count in failed_error_counts
                ):
                    raise ValueError
                return NotificationQueueSnapshot(
                    counts=counts,
                    ready_pending=int(pending["ready_pending"]),
                    oldest_pending_age_seconds=(
                        max(0.0, (observed_at - oldest).total_seconds())
                        if oldest is not None
                        else None
                    ),
                    failed_error_counts=failed_error_counts,
                )
            except (sqlite3.Error, TypeError, ValueError):
                raise NotificationQueueError(
                    "notification database is missing or invalid"
                ) from None
        finally:
            connection.close()

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> NotificationJob:
        try:
            message = NotificationMessage.from_json(row["payload_json"])
            sink_kind = str(row["sink_kind"])
            sink_id = str(row["sink_id"])
            _validate_sink(sink_kind, sink_id)
            if (
                message.notification_id != row["notification_id"]
                or message.event_id != row["event_id"]
                or message.notification_id != _notification_id(message.event_id, sink_kind, sink_id)
            ):
                raise ValueError
            last_error = row["last_error_code"]
            if last_error is not None and _SAFE_CODE.fullmatch(str(last_error)) is None:
                raise ValueError
            status = NotificationJobStatus(row["status"])
            attempt_count = int(row["attempt_count"])
            lease_until = (
                _parse_timestamp(row["lease_until"], field="lease_until")
                if row["lease_until"] is not None
                else None
            )
            delivered_at = (
                _parse_timestamp(row["delivered_at"], field="delivered_at")
                if row["delivered_at"] is not None
                else None
            )
            if not 0 <= attempt_count <= 100:
                raise ValueError
            if (status is NotificationJobStatus.RUNNING) != (lease_until is not None):
                raise ValueError
            if (status is NotificationJobStatus.DELIVERED) != (delivered_at is not None):
                raise ValueError
            if status is NotificationJobStatus.FAILED and last_error is None:
                raise ValueError
            return NotificationJob(
                message=message,
                sink_kind=sink_kind,
                sink_id=sink_id,
                status=status,
                attempt_count=attempt_count,
                available_at=_parse_timestamp(row["available_at"], field="available_at"),
                lease_until=lease_until,
                last_error_code=(str(last_error) if last_error is not None else None),
                created_at=_parse_timestamp(row["created_at"], field="created_at"),
                updated_at=_parse_timestamp(row["updated_at"], field="updated_at"),
                delivered_at=delivered_at,
            )
        except (IndexError, TypeError, ValueError):
            raise NotificationQueueError("notification job data is invalid") from None


@dataclass(frozen=True, slots=True)
class NotificationDeliveryReceipt:
    sink_kind: str
    reused_existing: bool


class NotificationSink(Protocol):
    kind: str

    def deliver(self, message: NotificationMessage) -> NotificationDeliveryReceipt: ...


class StdoutNotificationSink:
    kind = "stdout"

    def __init__(self, emit: Callable[[str], None] = print):
        self.emit = emit

    def deliver(self, message: NotificationMessage) -> NotificationDeliveryReceipt:
        try:
            self.emit(message.to_json(indent=None))
        except Exception:
            raise NotificationDeliveryError("stdout_error") from None
        return NotificationDeliveryReceipt(self.kind, reused_existing=False)


class FileNotificationSink:
    kind = "file"

    def __init__(self, target_root: str | Path):
        self.target_root = Path(target_root).expanduser()

    def deliver(self, message: NotificationMessage) -> NotificationDeliveryReceipt:
        root = self.target_root
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise NotificationDeliveryError("file_target_unavailable")
        payload = (message.to_json(indent=2) + "\n").encode("utf-8")
        target = root / f"{message.notification_id}.json"
        if target.is_symlink():
            raise NotificationDeliveryError("file_conflict")
        partial = root / f".{message.notification_id}.{uuid.uuid4().hex}.partial"
        reused = False
        try:
            with partial.open("xb") as handle:
                if hasattr(os, "fchmod"):
                    os.fchmod(handle.fileno(), 0o600)
                else:
                    partial.chmod(0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(partial, target)
            except FileExistsError:
                if target.is_symlink():
                    raise NotificationDeliveryError("file_conflict") from None
                try:
                    reused = target.is_file() and target.read_bytes() == payload
                except OSError:
                    raise NotificationDeliveryError("file_io_error") from None
                if not reused:
                    raise NotificationDeliveryError("file_conflict") from None
            if os.name != "nt":
                try:
                    directory_fd = os.open(root, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                except OSError:
                    raise NotificationDeliveryError("file_io_error") from None
        except NotificationDeliveryError:
            raise
        except OSError:
            raise NotificationDeliveryError("file_io_error") from None
        finally:
            with suppress(OSError):
                partial.unlink(missing_ok=True)
        return NotificationDeliveryReceipt(self.kind, reused_existing=reused)


@dataclass(frozen=True, slots=True)
class NotificationDeliveryAttempt:
    status: str
    notification_id: str | None
    sink_kind: str
    reused_existing: bool
    error_code: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "notification_id": self.notification_id,
            "sink_kind": self.sink_kind,
            "reused_existing": self.reused_existing,
            "error_code": self.error_code,
        }


def deliver_notification_once(
    *,
    outbox: NotificationOutbox,
    sink: NotificationSink,
    sink_id: str,
    lease_seconds: int,
    retry_delay_seconds: int,
    max_attempts: int,
    now: datetime | None = None,
    notification_id: str | None = None,
) -> NotificationDeliveryAttempt:
    _validate_sink(sink.kind, sink_id)
    if notification_id is None:
        job = outbox.claim_next(
            sink_kind=sink.kind,
            sink_id=sink_id,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            now=now,
        )
    else:
        job = outbox.claim_notification(
            notification_id,
            sink_kind=sink.kind,
            sink_id=sink_id,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            now=now,
        )
    if job is None:
        return NotificationDeliveryAttempt(
            status="idle",
            notification_id=None,
            sink_kind=sink.kind,
            reused_existing=False,
            error_code=None,
        )
    try:
        receipt = sink.deliver(job.message)
    except NotificationDeliveryError as exc:
        failed = outbox.mark_failed_attempt(
            job.notification_id,
            error_code=exc.code,
            retry_delay_seconds=retry_delay_seconds,
            max_attempts=max_attempts,
            now=now,
        )
        return NotificationDeliveryAttempt(
            status=failed.status.value,
            notification_id=job.notification_id,
            sink_kind=sink.kind,
            reused_existing=False,
            error_code=exc.code,
        )
    outbox.mark_delivered(job.notification_id, now=now)
    return NotificationDeliveryAttempt(
        status=NotificationJobStatus.DELIVERED.value,
        notification_id=job.notification_id,
        sink_kind=sink.kind,
        reused_existing=receipt.reused_existing,
        error_code=None,
    )
