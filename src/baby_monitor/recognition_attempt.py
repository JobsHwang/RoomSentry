from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from .domain import RecognitionProvenance, RecognitionRunMetrics, SchemaError

RECOGNITION_ATTEMPT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS recognition_attempts (
    attempt_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    stage TEXT NOT NULL CHECK (
        stage IN ('local_primary', 'local_secondary', 'remote')
    ),
    provenance_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    metrics_json TEXT,
    error_code TEXT
);

CREATE INDEX IF NOT EXISTS idx_recognition_attempt_started_at
    ON recognition_attempts (started_at);
CREATE INDEX IF NOT EXISTS idx_recognition_attempt_event_id
    ON recognition_attempts (event_id, started_at);
"""


_SAFE_EVENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")
_SAFE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SAFE_ATTEMPT_ID = re.compile(r"[0-9a-f]{32}")


class RecognitionStage(StrEnum):
    LOCAL_PRIMARY = "local_primary"
    LOCAL_SECONDARY = "local_secondary"
    REMOTE = "remote"


class RecognitionAttemptStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RecognitionAttempt:
    attempt_id: str
    event_id: str
    stage: RecognitionStage
    provenance: RecognitionProvenance
    started_at: datetime
    completed_at: datetime | None
    status: RecognitionAttemptStatus
    metrics: RecognitionRunMetrics | None
    error_code: str | None
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("unsupported recognition attempt version")
        if _SAFE_ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise ValueError("recognition attempt_id is invalid")
        if _SAFE_EVENT_ID.fullmatch(self.event_id) is None:
            raise ValueError("recognition attempt event_id is invalid")
        if not isinstance(self.stage, RecognitionStage):
            raise TypeError("recognition attempt stage is invalid")
        if not isinstance(self.provenance, RecognitionProvenance):
            raise TypeError("recognition attempt provenance is invalid")
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("recognition attempt started_at must be timezone-aware")
        if self.completed_at is not None and (
            self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None
        ):
            raise ValueError("recognition attempt completed_at must be timezone-aware")
        if not isinstance(self.status, RecognitionAttemptStatus):
            raise TypeError("recognition attempt status is invalid")
        if self.metrics is not None and not isinstance(self.metrics, RecognitionRunMetrics):
            raise TypeError("recognition attempt metrics are invalid")
        if self.error_code is not None and (
            not isinstance(self.error_code, str)
            or _SAFE_ERROR_CODE.fullmatch(self.error_code) is None
        ):
            raise ValueError("recognition attempt error_code is invalid")
        if self.status is RecognitionAttemptStatus.RUNNING:
            if self.completed_at is not None or self.metrics is not None or self.error_code:
                raise ValueError("running recognition attempt cannot be completed")
        elif self.completed_at is None or self.metrics is None:
            raise ValueError("terminal recognition attempt requires completion metrics")
        elif self.status is RecognitionAttemptStatus.SUCCEEDED and (
            self.error_code is not None or not self.metrics.json_compliant
        ):
            raise ValueError("successful recognition attempt must be JSON compliant")
        elif self.status is RecognitionAttemptStatus.FAILED and self.error_code is None:
            raise ValueError("failed recognition attempt requires an error code")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "attempt_id": self.attempt_id,
            "event_id": self.event_id,
            "stage": self.stage.value,
            "provenance": self.provenance.to_dict(),
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "completed_at": (
                self.completed_at.astimezone(UTC).isoformat()
                if self.completed_at is not None
                else None
            ),
            "status": self.status.value,
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "error_code": self.error_code,
        }


class RecognitionAttemptAuditStore:
    """A path-free, media-free ledger for every pipeline VLM invocation."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _connect_read_only(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise FileNotFoundError("recognition attempt database does not exist")
        try:
            connection = sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro",
                uri=True,
            )
        except sqlite3.Error:
            raise RuntimeError("recognition attempt database is unreadable") from None
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
        except sqlite3.Error:
            connection.close()
            raise RuntimeError("recognition attempt database is unreadable") from None
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
            connection.executescript(RECOGNITION_ATTEMPT_SCHEMA_SQL)

    def start(
        self,
        *,
        event_id: str,
        stage: RecognitionStage,
        provenance: RecognitionProvenance,
        started_at: datetime,
    ) -> str:
        if _SAFE_EVENT_ID.fullmatch(event_id) is None:
            raise ValueError("recognition attempt event_id is invalid")
        if not isinstance(stage, RecognitionStage):
            raise TypeError("recognition attempt stage is invalid")
        if not isinstance(provenance, RecognitionProvenance):
            raise TypeError("recognition attempt provenance is invalid")
        if started_at.tzinfo is None or started_at.utcoffset() is None:
            raise ValueError("recognition attempt started_at must be timezone-aware")
        attempt_id = uuid.uuid4().hex
        with self._session() as connection:
            connection.execute(
                """
                INSERT INTO recognition_attempts (
                    attempt_id, event_id, stage, provenance_json, started_at,
                    completed_at, status, metrics_json, error_code
                ) VALUES (?, ?, ?, ?, ?, NULL, 'running', NULL, NULL)
                """,
                (
                    attempt_id,
                    event_id,
                    stage.value,
                    provenance.to_json(indent=None),
                    started_at.astimezone(UTC).isoformat(),
                ),
            )
        return attempt_id

    def complete(
        self,
        attempt_id: str,
        *,
        status: RecognitionAttemptStatus,
        completed_at: datetime,
        metrics: RecognitionRunMetrics,
        error_code: str | None = None,
    ) -> None:
        if status not in {
            RecognitionAttemptStatus.SUCCEEDED,
            RecognitionAttemptStatus.FAILED,
        }:
            raise ValueError("recognition attempt completion status must be terminal")
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise ValueError("recognition attempt completed_at must be timezone-aware")
        if not isinstance(metrics, RecognitionRunMetrics):
            raise TypeError("recognition attempt metrics are invalid")
        if status is RecognitionAttemptStatus.SUCCEEDED:
            if error_code is not None or not metrics.json_compliant:
                raise ValueError("successful recognition attempt must be JSON compliant")
        elif not isinstance(error_code, str) or _SAFE_ERROR_CODE.fullmatch(error_code) is None:
            raise ValueError("failed recognition attempt requires a safe error code")
        with self._session() as connection:
            cursor = connection.execute(
                """
                UPDATE recognition_attempts
                SET completed_at = ?, status = ?, metrics_json = ?, error_code = ?
                WHERE attempt_id = ? AND status = 'running'
                """,
                (
                    completed_at.astimezone(UTC).isoformat(),
                    status.value,
                    metrics.to_json(indent=None),
                    error_code,
                    attempt_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("recognition attempt is missing or already completed")

    def recent(self, *, limit: int = 20) -> tuple[RecognitionAttempt, ...]:
        if type(limit) is not int or not 1 <= limit <= 1_000:
            raise ValueError("recognition attempt limit must be between 1 and 1000")
        return self._read(
            "SELECT * FROM recognition_attempts ORDER BY started_at DESC, attempt_id DESC LIMIT ?",
            (limit,),
        )

    def between(self, start: datetime, end: datetime) -> tuple[RecognitionAttempt, ...]:
        if (
            start.tzinfo is None
            or start.utcoffset() is None
            or end.tzinfo is None
            or end.utcoffset() is None
            or end <= start
        ):
            raise ValueError("recognition attempt range must be timezone-aware and increasing")
        return self._read(
            "SELECT * FROM recognition_attempts "
            "WHERE started_at >= ? AND started_at < ? "
            "ORDER BY started_at, attempt_id",
            (start.astimezone(UTC).isoformat(), end.astimezone(UTC).isoformat()),
        )

    def _read(
        self,
        statement: str,
        parameters: tuple[object, ...],
    ) -> tuple[RecognitionAttempt, ...]:
        connection = self._connect_read_only()
        try:
            try:
                rows = connection.execute(statement, parameters).fetchall()
            except sqlite3.Error:
                raise RuntimeError(
                    "recognition attempt schema is missing or invalid; run init-db"
                ) from None
        finally:
            connection.close()
        return tuple(self._row_to_attempt(row) for row in rows)

    @staticmethod
    def _row_to_attempt(row: sqlite3.Row) -> RecognitionAttempt:
        try:
            attempt = RecognitionAttempt(
                attempt_id=row["attempt_id"],
                event_id=row["event_id"],
                stage=RecognitionStage(row["stage"]),
                provenance=RecognitionProvenance.from_json(row["provenance_json"]),
                started_at=datetime.fromisoformat(row["started_at"]),
                completed_at=(
                    datetime.fromisoformat(row["completed_at"])
                    if row["completed_at"] is not None
                    else None
                ),
                status=RecognitionAttemptStatus(row["status"]),
                metrics=(
                    RecognitionRunMetrics.from_json(row["metrics_json"])
                    if row["metrics_json"] is not None
                    else None
                ),
                error_code=row["error_code"],
            )
        except (IndexError, SchemaError, TypeError, ValueError):
            raise RuntimeError("recognition attempt row is invalid") from None
        if attempt.started_at.tzinfo is None or attempt.started_at.utcoffset() is None:
            raise RuntimeError("recognition attempt row is invalid")
        if attempt.status is RecognitionAttemptStatus.RUNNING:
            if (
                attempt.completed_at is not None
                or attempt.metrics is not None
                or attempt.error_code
            ):
                raise RuntimeError("recognition attempt row is invalid")
        else:
            if attempt.completed_at is None or attempt.metrics is None:
                raise RuntimeError("recognition attempt row is invalid")
            if attempt.status is RecognitionAttemptStatus.SUCCEEDED and (
                attempt.error_code is not None or not attempt.metrics.json_compliant
            ):
                raise RuntimeError("recognition attempt row is invalid")
            if attempt.status is RecognitionAttemptStatus.FAILED and (
                not isinstance(attempt.error_code, str)
                or _SAFE_ERROR_CODE.fullmatch(attempt.error_code) is None
            ):
                raise RuntimeError("recognition attempt row is invalid")
        return attempt
