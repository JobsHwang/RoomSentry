from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from baby_monitor.visual_predicate import VisualPredicate, VisualPredicateResult
from baby_monitor.visual_predicate_shadow import (
    VisualPredicateShadowError,
    VisualPredicateShadowGate,
    VisualPredicateShadowRun,
    VisualPredicateShadowTrigger,
)

_ERROR_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]{0,79}")

SHADOW_ATTEMPT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS visual_predicate_shadow_attempts (
    event_id TEXT PRIMARY KEY,
    gate_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
    available_at TEXT NOT NULL,
    lease_until TEXT,
    last_error_code TEXT,
    result_binding_sha256 TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_visual_predicate_shadow_attempts_ready
    ON visual_predicate_shadow_attempts (status, available_at, created_at);
"""


class VisualPredicateShadowQueueError(VisualPredicateShadowError):
    """Path-free shadow scheduling and attempt-ledger error."""


class VisualPredicateShadowQueueFullError(VisualPredicateShadowQueueError):
    pass


class VisualPredicateShadowAttemptStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowAttempt:
    gate: VisualPredicateShadowGate
    status: VisualPredicateShadowAttemptStatus
    attempt_count: int
    available_at: datetime
    lease_until: datetime | None
    last_error_code: str | None
    result_binding_sha256: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    def __post_init__(self) -> None:
        if not isinstance(self.gate, VisualPredicateShadowGate) or not isinstance(
            self.status, VisualPredicateShadowAttemptStatus
        ):
            raise VisualPredicateShadowQueueError("visual predicate shadow attempt row is invalid")
        if type(self.attempt_count) is not int or self.attempt_count < 0:
            raise VisualPredicateShadowQueueError("visual predicate shadow attempt row is invalid")
        timestamps = (self.available_at, self.created_at, self.updated_at)
        if any(
            not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
            for value in timestamps
        ):
            raise VisualPredicateShadowQueueError("visual predicate shadow attempt row is invalid")
        if self.completed_at is not None and (
            not isinstance(self.completed_at, datetime)
            or self.completed_at.tzinfo is None
            or self.completed_at.utcoffset() is None
        ):
            raise VisualPredicateShadowQueueError("visual predicate shadow attempt row is invalid")
        if self.lease_until is not None and (
            not isinstance(self.lease_until, datetime)
            or self.lease_until.tzinfo is None
            or self.lease_until.utcoffset() is None
        ):
            raise VisualPredicateShadowQueueError("visual predicate shadow attempt row is invalid")
        if self.last_error_code is not None and (
            not isinstance(self.last_error_code, str)
            or _ERROR_CODE.fullmatch(self.last_error_code) is None
        ):
            raise VisualPredicateShadowQueueError("visual predicate shadow attempt row is invalid")
        if self.result_binding_sha256 is not None and (
            not isinstance(self.result_binding_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.result_binding_sha256) is None
        ):
            raise VisualPredicateShadowQueueError("visual predicate shadow attempt row is invalid")
        if self.updated_at < self.created_at or (
            self.completed_at is not None and self.completed_at < self.created_at
        ):
            raise VisualPredicateShadowQueueError("visual predicate shadow attempt row is invalid")
        if self.status is VisualPredicateShadowAttemptStatus.RUNNING:
            valid_state = (
                self.gate.selected
                and self.attempt_count >= 1
                and self.lease_until is not None
                and self.last_error_code is None
                and self.result_binding_sha256 is None
                and self.completed_at is None
            )
        elif self.status is VisualPredicateShadowAttemptStatus.COMPLETED:
            valid_state = (
                self.gate.selected
                and self.attempt_count >= 1
                and self.lease_until is None
                and self.last_error_code is None
                and self.result_binding_sha256 is not None
                and self.completed_at is not None
            )
        elif self.status is VisualPredicateShadowAttemptStatus.FAILED:
            valid_state = (
                self.gate.selected
                and self.lease_until is None
                and self.last_error_code is not None
                and self.result_binding_sha256 is None
                and self.completed_at is None
            )
        elif self.status is VisualPredicateShadowAttemptStatus.SKIPPED:
            valid_state = (
                not self.gate.selected
                and self.attempt_count == 0
                and self.lease_until is None
                and self.last_error_code is None
                and self.result_binding_sha256 is None
                and self.completed_at is not None
            )
        else:
            valid_state = (
                self.gate.selected
                and self.lease_until is None
                and self.result_binding_sha256 is None
                and self.completed_at is None
            )
        if not valid_state:
            raise VisualPredicateShadowQueueError("visual predicate shadow attempt row is invalid")

    @property
    def event_id(self) -> str:
        return self.gate.event_id

    def to_public_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "status": self.status.value,
            "attempt_count": self.attempt_count,
            "available_at": self.available_at.isoformat(),
            "lease_until": self.lease_until.isoformat() if self.lease_until else None,
            "last_error_code": self.last_error_code,
            "result_binding_sha256": self.result_binding_sha256,
            "gate": self.gate.to_dict(),
            "paths_disclosed": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowEnqueueResult:
    attempt: VisualPredicateShadowAttempt
    reused_existing: bool


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowAttemptCounts:
    pending: int
    running: int
    completed: int
    failed: int
    skipped: int

    @property
    def active(self) -> int:
        return self.pending + self.running

    def to_dict(self) -> dict[str, int]:
        return {
            "pending": self.pending,
            "running": self.running,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "active": self.active,
        }


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowAttemptSnapshot:
    counts: VisualPredicateShadowAttemptCounts
    ready_pending: int
    oldest_pending_age_seconds: float | None
    failed_error_counts: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **self.counts.to_dict(),
            "ready_pending": self.ready_pending,
            "oldest_pending_age_seconds": self.oldest_pending_age_seconds,
            "failed_error_counts": dict(self.failed_error_counts),
            "paths_disclosed": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowObservation:
    attempt: VisualPredicateShadowAttempt
    results: tuple[VisualPredicateResult, ...] = ()
    recorded_at: datetime | None = None

    def __post_init__(self) -> None:
        completed = self.attempt.status is VisualPredicateShadowAttemptStatus.COMPLETED
        if completed != bool(
            self.results
            and self.recorded_at is not None
            and self.recorded_at.tzinfo is not None
            and self.recorded_at.utcoffset() is not None
        ) or any(not isinstance(item, VisualPredicateResult) for item in self.results):
            raise VisualPredicateShadowQueueError("visual predicate shadow observation is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt": self.attempt.to_public_dict(),
            "results": [item.to_dict() for item in self.results],
            "recorded_at": (
                self.recorded_at.astimezone(UTC).isoformat()
                if self.recorded_at is not None
                else None
            ),
            "paths_disclosed": False,
            "media_metadata_included": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }


def _utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError
            value[key] = item
        return value

    return json.loads(
        raw,
        object_pairs_hook=pairs_hook,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
    )


def _gate_from_dict(value: Any) -> VisualPredicateShadowGate:
    expected = {
        "version",
        "event_id",
        "selected",
        "triggers",
        "predicates",
        "person_detector_id",
        "person_detection_count",
        "hardware_signal_adapters",
        "hardware_signal_count",
        "subject_identity_inference",
        "risk_inference",
        "notification_decision",
        "paths_disclosed",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError
    if any(
        value[field] is not False
        for field in (
            "subject_identity_inference",
            "risk_inference",
            "notification_decision",
            "paths_disclosed",
        )
    ):
        raise ValueError
    triggers = value["triggers"]
    predicates = value["predicates"]
    adapters = value["hardware_signal_adapters"]
    if (
        not isinstance(triggers, list)
        or not isinstance(predicates, list)
        or not isinstance(adapters, list)
    ):
        raise ValueError
    return VisualPredicateShadowGate(
        version=value["version"],
        event_id=value["event_id"],
        selected=value["selected"],
        triggers=tuple(VisualPredicateShadowTrigger(item) for item in triggers),
        predicates=tuple(VisualPredicate(item) for item in predicates),
        person_detector_id=value["person_detector_id"],
        person_detection_count=value["person_detection_count"],
        hardware_signal_adapters=tuple(adapters),
        hardware_signal_count=value["hardware_signal_count"],
    )


class VisualPredicateShadowAttemptQueue:
    def __init__(self, database_path: str | Path):
        path = Path(database_path).expanduser()
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            )
        try:
            self.path = path.resolve(strict=True)
        except OSError:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        mode = "ro" if read_only else "rw"
        try:
            connection = sqlite3.connect(f"{self.path.as_uri()}?mode={mode}", uri=True)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            if read_only:
                connection.execute("PRAGMA query_only = ON")
            return connection
        except sqlite3.Error:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        except sqlite3.Error:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._session() as connection:
            connection.executescript(SHADOW_ATTEMPT_SCHEMA_SQL)

    def is_initialized_read_only(self) -> bool:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'visual_predicate_shadow_attempts'
                """
            ).fetchone()
            return row is not None
        except sqlite3.Error:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()

    def contains(self, event_id: str) -> bool:
        if not isinstance(event_id, str) or not event_id or len(event_id) > 128:
            raise ValueError("event_id must be a bounded identifier")
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                """
                SELECT 1 FROM visual_predicate_shadow_attempts
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
            return row is not None
        except sqlite3.Error:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()

    def completed_analysis_sync_batch(
        self,
        *,
        limit: int = 1_000,
    ) -> tuple[tuple[str, ...], int]:
        rows, existing_count = self.analysis_sync_batch(
            statuses=("completed",),
            limit=limit,
        )
        return tuple(event_id for event_id, _status in rows), existing_count

    def analysis_sync_batch(
        self,
        *,
        statuses: tuple[str, ...],
        limit: int = 1_000,
    ) -> tuple[tuple[tuple[str, str], ...], int]:
        allowed_statuses = {"pending", "completed"}
        if (
            not isinstance(statuses, tuple)
            or not statuses
            or len(set(statuses)) != len(statuses)
            or any(status not in allowed_statuses for status in statuses)
        ):
            raise ValueError("shadow sync statuses are invalid")
        if type(limit) is not int or not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        placeholders = ",".join("?" for _status in statuses)
        connection = self._connect(read_only=True)
        try:
            existing_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM analysis_jobs AS analysis
                    INNER JOIN visual_predicate_shadow_attempts AS shadow
                        ON shadow.event_id = analysis.event_id
                    WHERE analysis.status IN ({placeholders})
                    """,
                    statuses,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT analysis.event_id, analysis.status
                FROM analysis_jobs AS analysis
                LEFT JOIN visual_predicate_shadow_attempts AS shadow
                    ON shadow.event_id = analysis.event_id
                WHERE analysis.status IN ({placeholders}) AND shadow.event_id IS NULL
                ORDER BY
                    CASE analysis.status WHEN 'pending' THEN 0 ELSE 1 END,
                    analysis.occurred_at ASC,
                    analysis.event_id ASC
                LIMIT ?
                """,
                (*statuses, limit),
            ).fetchall()
            events = tuple((str(row["event_id"]), str(row["status"])) for row in rows)
            return events, existing_count
        except sqlite3.Error:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()

    def record_unselected(
        self,
        gate: VisualPredicateShadowGate,
        *,
        now: datetime | None = None,
    ) -> VisualPredicateShadowEnqueueResult:
        if not isinstance(gate, VisualPredicateShadowGate) or gate.selected:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow skip requires an unselected gate"
            )
        recorded_at = _utc(now or datetime.now(UTC), field="now")
        gate_json = _canonical_json(gate.to_dict())
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM visual_predicate_shadow_attempts WHERE event_id = ?",
                (gate.event_id,),
            ).fetchone()
            if existing is not None:
                if existing["gate_json"] != gate_json:
                    raise VisualPredicateShadowQueueError(
                        "visual predicate shadow attempt conflicts with an existing event"
                    )
                connection.commit()
                return VisualPredicateShadowEnqueueResult(
                    attempt=self._row_to_attempt(existing),
                    reused_existing=True,
                )
            connection.execute(
                """
                INSERT INTO visual_predicate_shadow_attempts (
                    event_id, gate_json, status, attempt_count, available_at,
                    lease_until, last_error_code, result_binding_sha256,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, 'skipped', 0, ?, NULL, NULL, NULL, ?, ?, ?)
                """,
                (
                    gate.event_id,
                    gate_json,
                    recorded_at.isoformat(),
                    recorded_at.isoformat(),
                    recorded_at.isoformat(),
                    recorded_at.isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM visual_predicate_shadow_attempts WHERE event_id = ?",
                (gate.event_id,),
            ).fetchone()
            connection.commit()
        except VisualPredicateShadowQueueError:
            connection.rollback()
            raise
        except sqlite3.Error:
            connection.rollback()
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()
        if row is None:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow skip disappeared from SQLite"
            )
        return VisualPredicateShadowEnqueueResult(
            attempt=self._row_to_attempt(row),
            reused_existing=False,
        )

    def enqueue(
        self,
        gate: VisualPredicateShadowGate,
        *,
        max_active: int,
        now: datetime | None = None,
    ) -> VisualPredicateShadowEnqueueResult:
        if not isinstance(gate, VisualPredicateShadowGate) or not gate.selected:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt requires a selected gate"
            )
        if type(max_active) is not int or not 1 <= max_active <= 10_000:
            raise ValueError("max_active must be between 1 and 10000")
        queued_at = _utc(now or datetime.now(UTC), field="now")
        gate_json = _canonical_json(gate.to_dict())
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM visual_predicate_shadow_attempts WHERE event_id = ?",
                (gate.event_id,),
            ).fetchone()
            if existing is not None:
                if existing["gate_json"] != gate_json:
                    raise VisualPredicateShadowQueueError(
                        "visual predicate shadow attempt conflicts with an existing event"
                    )
                connection.commit()
                return VisualPredicateShadowEnqueueResult(
                    attempt=self._row_to_attempt(existing),
                    reused_existing=True,
                )
            active = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM visual_predicate_shadow_attempts
                    WHERE status IN ('pending', 'running')
                    """
                ).fetchone()[0]
            )
            queue_full = active >= max_active
            status = (
                VisualPredicateShadowAttemptStatus.FAILED
                if queue_full
                else VisualPredicateShadowAttemptStatus.PENDING
            )
            connection.execute(
                """
                INSERT INTO visual_predicate_shadow_attempts (
                    event_id, gate_json, status, attempt_count, available_at,
                    lease_until, last_error_code, result_binding_sha256,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, 0, ?, NULL, ?, NULL, ?, ?, NULL)
                """,
                (
                    gate.event_id,
                    gate_json,
                    status.value,
                    queued_at.isoformat(),
                    "queue_full" if queue_full else None,
                    queued_at.isoformat(),
                    queued_at.isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM visual_predicate_shadow_attempts WHERE event_id = ?",
                (gate.event_id,),
            ).fetchone()
            connection.commit()
        except VisualPredicateShadowQueueError:
            connection.rollback()
            raise
        except sqlite3.Error:
            connection.rollback()
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()
        if row is None:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt disappeared from SQLite"
            )
        attempt = self._row_to_attempt(row)
        if queue_full:
            raise VisualPredicateShadowQueueFullError(
                "visual predicate shadow attempt queue is full; event remains tracked"
            )
        return VisualPredicateShadowEnqueueResult(attempt=attempt, reused_existing=False)

    def claim_next(
        self,
        *,
        lease_seconds: int,
        max_attempts: int,
        now: datetime | None = None,
    ) -> VisualPredicateShadowAttempt | None:
        if type(lease_seconds) is not int or not 10 <= lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be between 10 and 86400")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts must be between 1 and 100")
        claimed_at = _utc(now or datetime.now(UTC), field="now")
        lease_until = claimed_at + timedelta(seconds=lease_seconds)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE visual_predicate_shadow_attempts
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
                SELECT * FROM visual_predicate_shadow_attempts
                WHERE status = 'pending' AND available_at <= ?
                ORDER BY created_at ASC, event_id ASC
                LIMIT 1
                """,
                (claimed_at.isoformat(),),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            updated = connection.execute(
                """
                UPDATE visual_predicate_shadow_attempts
                SET status = 'running', attempt_count = attempt_count + 1,
                    lease_until = ?, last_error_code = NULL, updated_at = ?
                WHERE event_id = ? AND status = 'pending'
                """,
                (lease_until.isoformat(), claimed_at.isoformat(), row["event_id"]),
            )
            if updated.rowcount != 1:
                raise VisualPredicateShadowQueueError(
                    "visual predicate shadow attempt claim lost a concurrency race"
                )
            claimed = connection.execute(
                "SELECT * FROM visual_predicate_shadow_attempts WHERE event_id = ?",
                (row["event_id"],),
            ).fetchone()
            connection.commit()
        except VisualPredicateShadowQueueError:
            connection.rollback()
            raise
        except sqlite3.Error:
            connection.rollback()
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()
        return self._row_to_attempt(claimed) if claimed is not None else None

    def mark_completed(
        self,
        event_id: str,
        *,
        now: datetime | None = None,
    ) -> VisualPredicateShadowAttempt:
        completed_at = _utc(now or datetime.now(UTC), field="now")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            attempt = connection.execute(
                """
                SELECT * FROM visual_predicate_shadow_attempts
                WHERE event_id = ? AND status = 'running'
                """,
                (event_id,),
            ).fetchone()
            if attempt is None:
                raise VisualPredicateShadowQueueError(
                    "only a running visual predicate shadow attempt can complete"
                )
            try:
                result = connection.execute(
                    """
                    SELECT binding_json FROM visual_predicate_shadow_runs
                    WHERE event_id = ?
                    """,
                    (event_id,),
                ).fetchone()
            except sqlite3.Error:
                result = None
            if result is None:
                raise VisualPredicateShadowQueueError(
                    "visual predicate shadow result is unavailable"
                )
            try:
                result_binding = _strict_json(result["binding_json"])
                gate_binding = _strict_json(attempt["gate_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                raise VisualPredicateShadowQueueError(
                    "visual predicate shadow result is invalid"
                ) from None
            if (
                not isinstance(result_binding, dict)
                or result_binding.get("gate") != gate_binding
                or not isinstance(result_binding.get("results"), list)
                or not result_binding["results"]
            ):
                raise VisualPredicateShadowQueueError(
                    "visual predicate shadow result does not match the attempt"
                )
            result_digest = hashlib.sha256(result["binding_json"].encode("utf-8")).hexdigest()
            updated = connection.execute(
                """
                UPDATE visual_predicate_shadow_attempts
                SET status = 'completed', lease_until = NULL, last_error_code = NULL,
                    result_binding_sha256 = ?, completed_at = ?, updated_at = ?
                WHERE event_id = ? AND status = 'running'
                """,
                (
                    result_digest,
                    completed_at.isoformat(),
                    completed_at.isoformat(),
                    event_id,
                ),
            )
            if updated.rowcount != 1:
                raise VisualPredicateShadowQueueError(
                    "only a running visual predicate shadow attempt can complete"
                )
            row = connection.execute(
                "SELECT * FROM visual_predicate_shadow_attempts WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            connection.commit()
        except VisualPredicateShadowQueueError:
            connection.rollback()
            raise
        except sqlite3.Error:
            connection.rollback()
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()
        if row is None:
            raise VisualPredicateShadowQueueError(
                "completed visual predicate shadow attempt disappeared from SQLite"
            )
        return self._row_to_attempt(row)

    def mark_failed_attempt(
        self,
        event_id: str,
        *,
        error_code: str,
        retry_delay_seconds: int,
        max_attempts: int,
        now: datetime | None = None,
    ) -> VisualPredicateShadowAttempt:
        if not isinstance(error_code, str) or _ERROR_CODE.fullmatch(error_code) is None:
            raise ValueError("error_code must be a short identifier")
        if type(retry_delay_seconds) is not int or not 0 <= retry_delay_seconds <= 86_400:
            raise ValueError("retry_delay_seconds must be between 0 and 86400")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts must be between 1 and 100")
        failed_at = _utc(now or datetime.now(UTC), field="now")
        available_at = failed_at + timedelta(seconds=retry_delay_seconds)
        with self._session() as connection:
            row = connection.execute(
                """
                SELECT attempt_count FROM visual_predicate_shadow_attempts
                WHERE event_id = ? AND status = 'running'
                """,
                (event_id,),
            ).fetchone()
            if row is None:
                raise VisualPredicateShadowQueueError(
                    "only a running visual predicate shadow attempt can record failure"
                )
            status = (
                VisualPredicateShadowAttemptStatus.FAILED
                if int(row["attempt_count"]) >= max_attempts
                else VisualPredicateShadowAttemptStatus.PENDING
            )
            connection.execute(
                """
                UPDATE visual_predicate_shadow_attempts
                SET status = ?, available_at = ?, lease_until = NULL,
                    last_error_code = ?, updated_at = ?
                WHERE event_id = ? AND status = 'running'
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
                "SELECT * FROM visual_predicate_shadow_attempts WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        if updated is None:
            raise VisualPredicateShadowQueueError(
                "failed visual predicate shadow attempt disappeared from SQLite"
            )
        return self._row_to_attempt(updated)

    def retry_failed(
        self,
        event_id: str,
        *,
        max_active: int,
        now: datetime | None = None,
    ) -> VisualPredicateShadowAttempt:
        if type(max_active) is not int or not 1 <= max_active <= 10_000:
            raise ValueError("max_active must be between 1 and 10000")
        retried_at = _utc(now or datetime.now(UTC), field="now")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM visual_predicate_shadow_attempts
                    WHERE status IN ('pending', 'running')
                    """
                ).fetchone()[0]
            )
            if active >= max_active:
                raise VisualPredicateShadowQueueFullError(
                    "visual predicate shadow attempt queue is full; failed event remains tracked"
                )
            updated = connection.execute(
                """
                UPDATE visual_predicate_shadow_attempts
                SET status = 'pending', attempt_count = 0, available_at = ?,
                    lease_until = NULL, last_error_code = NULL,
                    result_binding_sha256 = NULL, completed_at = NULL, updated_at = ?
                WHERE event_id = ? AND status = 'failed'
                """,
                (retried_at.isoformat(), retried_at.isoformat(), event_id),
            )
            if updated.rowcount != 1:
                raise VisualPredicateShadowQueueError(
                    "only a failed visual predicate shadow attempt can be retried"
                )
            row = connection.execute(
                "SELECT * FROM visual_predicate_shadow_attempts WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            connection.commit()
        except VisualPredicateShadowQueueError:
            connection.rollback()
            raise
        except sqlite3.Error:
            connection.rollback()
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()
        if row is None:
            raise VisualPredicateShadowQueueError(
                "retried visual predicate shadow attempt disappeared from SQLite"
            )
        return self._row_to_attempt(row)

    def recent_observations_read_only(
        self,
        *,
        limit: int = 100,
    ) -> tuple[VisualPredicateShadowObservation, ...]:
        if type(limit) is not int or not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT * FROM visual_predicate_shadow_attempts
                ORDER BY updated_at DESC, event_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            observations: list[VisualPredicateShadowObservation] = []
            for row in rows:
                attempt = self._row_to_attempt(row)
                if attempt.status is not VisualPredicateShadowAttemptStatus.COMPLETED:
                    observations.append(VisualPredicateShadowObservation(attempt=attempt))
                    continue
                result_row = connection.execute(
                    """
                    SELECT binding_json, recorded_at
                    FROM visual_predicate_shadow_runs
                    WHERE event_id = ?
                    """,
                    (attempt.event_id,),
                ).fetchone()
                if result_row is None:
                    raise VisualPredicateShadowQueueError(
                        "visual predicate shadow observation is invalid"
                    )
                raw_binding = result_row["binding_json"]
                try:
                    if (
                        not isinstance(raw_binding, str)
                        or hashlib.sha256(raw_binding.encode("utf-8")).hexdigest()
                        != attempt.result_binding_sha256
                    ):
                        raise ValueError
                    binding = _strict_json(raw_binding)
                    if (
                        not isinstance(binding, dict)
                        or binding.get("gate") != attempt.gate.to_dict()
                        or not isinstance(binding.get("results"), list)
                    ):
                        raise ValueError
                    results = tuple(
                        VisualPredicateResult.from_dict(item) for item in binding["results"]
                    )
                    recorded_at = datetime.fromisoformat(result_row["recorded_at"])
                    run = VisualPredicateShadowRun(
                        gate=attempt.gate,
                        results=results,
                        recorded_at=recorded_at,
                    )
                    if _canonical_json(run.binding_dict()) != raw_binding:
                        raise ValueError
                except Exception:
                    raise VisualPredicateShadowQueueError(
                        "visual predicate shadow observation is invalid"
                    ) from None
                observations.append(
                    VisualPredicateShadowObservation(
                        attempt=attempt,
                        results=results,
                        recorded_at=recorded_at,
                    )
                )
            return tuple(observations)
        except VisualPredicateShadowQueueError:
            raise
        except sqlite3.Error:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()

    def snapshot(
        self,
        *,
        now: datetime | None = None,
        read_only: bool = False,
    ) -> VisualPredicateShadowAttemptSnapshot:
        observed_at = _utc(now or datetime.now(UTC), field="now")
        if not read_only:
            self.initialize()
        connection = self._connect(read_only=read_only)
        try:
            rows = connection.execute(
                """
                SELECT status, available_at, created_at, last_error_code
                FROM visual_predicate_shadow_attempts
                """
            ).fetchall()
        except sqlite3.Error:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt database is unavailable"
            ) from None
        finally:
            connection.close()
        values = {status.value: 0 for status in VisualPredicateShadowAttemptStatus}
        ready_pending = 0
        oldest_pending: datetime | None = None
        failed_errors: dict[str, int] = {}
        for row in rows:
            status = VisualPredicateShadowAttemptStatus(row["status"])
            values[status.value] += 1
            if status is VisualPredicateShadowAttemptStatus.PENDING:
                available_at = datetime.fromisoformat(row["available_at"])
                created_at = datetime.fromisoformat(row["created_at"])
                ready_pending += int(available_at <= observed_at)
                if oldest_pending is None or created_at < oldest_pending:
                    oldest_pending = created_at
            elif status is VisualPredicateShadowAttemptStatus.FAILED:
                code = row["last_error_code"]
                if code is not None:
                    failed_errors[str(code)] = failed_errors.get(str(code), 0) + 1
        counts = VisualPredicateShadowAttemptCounts(
            pending=values[VisualPredicateShadowAttemptStatus.PENDING.value],
            running=values[VisualPredicateShadowAttemptStatus.RUNNING.value],
            completed=values[VisualPredicateShadowAttemptStatus.COMPLETED.value],
            failed=values[VisualPredicateShadowAttemptStatus.FAILED.value],
            skipped=values[VisualPredicateShadowAttemptStatus.SKIPPED.value],
        )
        return VisualPredicateShadowAttemptSnapshot(
            counts=counts,
            ready_pending=ready_pending,
            oldest_pending_age_seconds=(
                max(0.0, (observed_at - oldest_pending).total_seconds())
                if oldest_pending is not None
                else None
            ),
            failed_error_counts=tuple(sorted(failed_errors.items())),
        )

    @staticmethod
    def _row_to_attempt(row: sqlite3.Row) -> VisualPredicateShadowAttempt:
        try:
            gate = _gate_from_dict(_strict_json(row["gate_json"]))
            status = VisualPredicateShadowAttemptStatus(row["status"])
            result_digest = row["result_binding_sha256"]
            if result_digest is not None and (
                not isinstance(result_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", result_digest) is None
            ):
                raise ValueError
            return VisualPredicateShadowAttempt(
                gate=gate,
                status=status,
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
                result_binding_sha256=result_digest,
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                completed_at=(
                    datetime.fromisoformat(row["completed_at"])
                    if row["completed_at"] is not None
                    else None
                ),
            )
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise VisualPredicateShadowQueueError(
                "visual predicate shadow attempt row is invalid"
            ) from exc
