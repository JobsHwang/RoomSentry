"""Transactional local-review reservations; no models or evidence-file reads.

An expired lease is only a write fence, not proof that an old process stopped.
Evidence bindings here are historical metadata, never current-byte validation.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .database import EventDatabase, StoredEvent
from .domain import (
    DecisionRecord,
    FinalSource,
    RecognitionProvenance,
    RecognitionRunMetrics,
    ReviewTier,
    VLMEvent,
)
from .evidence_binding import MAX_BINDING_BYTES, EvidenceBinding, media_fingerprint
from .report_queue import ReportQueueError, _day_bounds, _root, _timezone

_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}\Z")
_EVENT_ID = re.compile(r"[A-Za-z0-9_.:-]{1,256}\Z")
_CODES = frozenset(
    {
        "invalid_input",
        "database_unavailable",
        "invalid_record",
        "event_not_found",
        "binding_required",
        "not_eligible",
        "request_conflict",
        "event_busy",
        "request_not_found",
        "not_running",
        "lease_expired",
        "event_changed",
        "invalid_completion",
        "model_failed",
        "evidence_changed",
        "review_failed",
    }
)
_FAIL_CODES = frozenset({"model_failed", "evidence_changed", "review_failed"})
_MAX_ROW_BYTES = 8 * 1024 * 1024
_ATTEMPT_LIMITS = {
    "request_id": 64,
    "event_id": 256,
    "status": 16,
    "error_code": 32,
    "requested_at": 40,
    "completed_at": 40,
    "lease_until": 40,
    "event_sha256": 64,
    "binding_sha256": 64,
}


def _projection(fields):
    return ",".join(
        f"CASE WHEN typeof({key})='text' AND length(CAST({key} AS BLOB))<={_ATTEMPT_LIMITS[key]} "
        f"THEN {key} WHEN {key} IS NULL THEN NULL ELSE CAST(X'00' AS TEXT) END AS {key}"
        for key in fields
    )


def _validate_attempt(row, *, fingerprints=False):
    _identifier(row["request_id"])
    _identifier(row["event_id"], event=True)
    if row["status"] not in ("running", "succeeded", "failed"):
        raise LocalReviewError("invalid_record")
    error = row["error_code"]
    if (row["status"] == "failed" and error not in _FAIL_CODES | {"lease_expired"}) or (
        row["status"] != "failed" and error is not None
    ):
        raise LocalReviewError("invalid_record")
    for key in ("requested_at", "lease_until", "completed_at"):
        value = row[key]
        if key == "completed_at" and row["status"] == "running" and value is None:
            continue
        if not isinstance(value, str) or _now(datetime.fromisoformat(value)).isoformat() != value:
            raise LocalReviewError("invalid_record")
    if row["lease_until"] <= row["requested_at"]:
        raise LocalReviewError("invalid_record")
    if fingerprints:
        for key in ("event_sha256", "binding_sha256"):
            if not isinstance(row[key], str) or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None:
                raise LocalReviewError("invalid_record")


def _attempt(connection, request_id):
    row = connection.execute(
        "SELECT " + _projection(_ATTEMPT_LIMITS) + " FROM local_review_attempts WHERE request_id=?",
        (request_id,),
    ).fetchone()
    if row is not None:
        _validate_attempt(row, fingerprints=True)
    return row


LOCAL_REVIEW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS local_review_attempts (
 request_id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES events(event_id),
 status TEXT NOT NULL CHECK(status IN ('running','succeeded','failed')),
 event_sha256 TEXT NOT NULL, binding_sha256 TEXT NOT NULL,
 before_result_json TEXT NOT NULL, before_decision_json TEXT NOT NULL,
 after_result_json TEXT, after_decision_json TEXT,
 after_secondary_result_json TEXT, after_provenance_json TEXT, after_metrics_json TEXT,
 requested_at TEXT NOT NULL, completed_at TEXT, lease_until TEXT NOT NULL, error_code TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_local_review_active
 ON local_review_attempts(event_id) WHERE status='running';
CREATE INDEX IF NOT EXISTS idx_local_review_requested ON local_review_attempts(requested_at);
CREATE TABLE IF NOT EXISTS local_review_report_refreshes (
 request_id TEXT PRIMARY KEY REFERENCES local_review_attempts(request_id),
 storage_root TEXT NOT NULL, timezone_name TEXT NOT NULL, local_date TEXT NOT NULL,
 queued_job_id TEXT, enqueued_at TEXT, last_error_code TEXT, last_attempt_at TEXT
);
"""


class LocalReviewError(RuntimeError):
    def __init__(self, code: str):
        self.code = code if isinstance(code, str) and code in _CODES else "review_failed"
        super().__init__(f"local review: {self.code}")


@dataclass(frozen=True, slots=True)
class LocalReviewReservation:
    event: StoredEvent
    binding: EvidenceBinding
    request_id: str
    reused_existing: bool
    status: str
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class LocalReviewReportIntent:
    request_id: str
    local_date: date
    timezone_name: str
    storage_root: Path = field(repr=False)


_REFRESH_LIMITS = {
    "request_id": 64,
    "storage_root": 16384,
    "timezone_name": 512,
    "local_date": 10,
    "queued_job_id": 71,
    "enqueued_at": 40,
    "last_error_code": 32,
    "last_attempt_at": 40,
}


def _refresh_projection():
    return ",".join(
        f"CASE WHEN typeof(r.{key})='text' AND length(CAST(r.{key} AS BLOB))<={limit} "
        f"THEN r.{key} WHEN r.{key} IS NULL THEN NULL ELSE CAST(X'00' AS TEXT) END AS {key}"
        for key, limit in _REFRESH_LIMITS.items()
    )


def _refresh_exists(connection):
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='local_review_report_refreshes'"
        ).fetchone()
        is not None
    )


def _report_job_id(value):
    if not isinstance(value, str) or re.fullmatch(r"report_[0-9a-f]{64}", value) is None:
        raise LocalReviewError("invalid_input")


def _parse_refresh(row):
    _identifier(row["request_id"])
    root = _root(row["storage_root"], admit=False)
    timezone = _timezone(row["timezone_name"])
    day = date.fromisoformat(row["local_date"])
    if day.isoformat() != row["local_date"]:
        raise LocalReviewError("invalid_record")
    _day_bounds(day, timezone)
    if row["queued_job_id"] is not None:
        _report_job_id(row["queued_job_id"])
        stamp = row["enqueued_at"]
        if not isinstance(stamp, str) or _now(datetime.fromisoformat(stamp)).isoformat() != stamp:
            raise LocalReviewError("invalid_record")
    elif row["enqueued_at"] is not None:
        raise LocalReviewError("invalid_record")
    if row["last_error_code"] not in (None, "report_enqueue_failed"):
        raise LocalReviewError("invalid_record")
    if row["last_attempt_at"] is not None:
        stamp = row["last_attempt_at"]
        if not isinstance(stamp, str) or _now(datetime.fromisoformat(stamp)).isoformat() != stamp:
            raise LocalReviewError("invalid_record")
    return LocalReviewReportIntent(row["request_id"], day, timezone, root)


def _refresh(connection, request_id):
    if not _refresh_exists(connection):
        return None
    row = connection.execute(
        "SELECT "
        + _refresh_projection()
        + " FROM local_review_report_refreshes r WHERE r.request_id=?",
        (request_id,),
    ).fetchone()
    if row is not None:
        _parse_refresh(row)
    return row


def _identifier(value: object, *, event: bool = False) -> None:
    if not isinstance(value, str) or (_EVENT_ID if event else _ID).fullmatch(value) is None:
        raise LocalReviewError("invalid_input")


def _now(value: datetime | None) -> datetime:
    value = datetime.now(UTC) if value is None else value
    try:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise LocalReviewError("invalid_input")
        return value.astimezone(UTC)
    except (ValueError, OverflowError):
        raise LocalReviewError("invalid_input") from None


def _strict_json(payload: str) -> object:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise LocalReviewError("invalid_record")
            value[key] = item
        return value

    def constant(_value):
        raise LocalReviewError("invalid_record")

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)


def _row_fingerprint(row: sqlite3.Row) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


class LocalReviewStore:
    def __init__(self, path: str | Path):
        try:
            self.path = Path(path)
        except (TypeError, ValueError):
            raise LocalReviewError("invalid_input") from None

    @contextmanager
    def _transaction(self, *, readonly: bool = False):
        connection = None
        try:
            # mode=rw/ro, unlike sqlite's default, never creates a missing database.
            uri = self.path.absolute().as_uri() + ("?mode=ro" if readonly else "?mode=rw")
            connection = sqlite3.connect(uri, uri=True)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            if readonly:
                connection.execute("PRAGMA query_only = ON")
            connection.execute("BEGIN" if readonly else "BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except LocalReviewError:
            if connection is not None:
                connection.rollback()
            raise
        except (sqlite3.Error, OSError):
            if connection is not None:
                connection.rollback()
            raise LocalReviewError("database_unavailable") from None
        except (
            ReportQueueError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            OverflowError,
            RecursionError,
        ):
            if connection is not None:
                connection.rollback()
            raise LocalReviewError("invalid_record") from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _load(connection, event_id):
        columns = connection.execute("PRAGMA table_info(events)").fetchall()
        if not columns:
            raise LocalReviewError("invalid_record")
        # Bound the entire row before fetching it, in this same transaction.
        names = ['"' + row["name"].replace('"', '""') + '"' for row in columns]
        sizes = "+".join(f"coalesce(length(CAST({name} AS BLOB)),0)" for name in names)
        size = connection.execute(
            f"SELECT {sizes} FROM events WHERE event_id=?", (event_id,)
        ).fetchone()
        if size is None:
            raise LocalReviewError("event_not_found")
        if size[0] > _MAX_ROW_BYTES:
            raise LocalReviewError("invalid_record")
        row = connection.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        for key in dict(row):
            if key.endswith("_json") and row[key] is not None:
                _strict_json(row[key])
        event = EventDatabase._row_to_event(row)
        if (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_evidence_bindings'"
            ).fetchone()
            is None
        ):
            raise LocalReviewError("binding_required")
        bound = connection.execute(
            "SELECT CASE WHEN typeof(binding_json)='text' "
            "AND length(CAST(binding_json AS BLOB))<=? "
            "THEN binding_json END FROM event_evidence_bindings WHERE event_id=?",
            (MAX_BINDING_BYTES, event_id),
        ).fetchone()
        if bound is None:
            raise LocalReviewError("binding_required")
        binding = EvidenceBinding.from_json(bound[0])
        if binding.media_sha256 != media_fingerprint(event.evidence) or len(
            binding.assets
        ) != 1 + len(event.evidence.frame_paths):
            raise LocalReviewError("invalid_record")
        return row, event, binding

    @staticmethod
    def _eligible(event):
        if (
            event.decision.final_source is not FinalSource.LOCAL_PRIMARY
            or event.secondary_result is not None
            or event.remote_result is not None
            or event.remote_used
            or event.decision.remote_used
            or event.decision.review_tier.rank < ReviewTier.LOCAL_SECONDARY.rank
        ):
            raise LocalReviewError("not_eligible")

    @staticmethod
    def _expire(connection, event_id, now):
        connection.execute(
            "UPDATE local_review_attempts SET status='failed',error_code='lease_expired',"
            "completed_at=? "
            "WHERE event_id=? AND status='running' AND lease_until<=?",
            (now, event_id, now),
        )

    def reserve(
        self,
        event_id: str,
        request_id: str,
        lease_seconds: int = 600,
        now: datetime | None = None,
        *,
        report_storage_root: Path | None = None,
        report_timezone: str | None = None,
    ) -> LocalReviewReservation:
        _identifier(event_id, event=True)
        _identifier(request_id)
        current = _now(now)
        if (report_storage_root is None) != (report_timezone is None):
            raise LocalReviewError("invalid_input")
        report_target = None
        if report_storage_root is not None:
            try:
                report_target = (
                    _root(report_storage_root, admit=False),
                    _timezone(report_timezone),
                )
            except (ReportQueueError, ValueError, TypeError, OSError):
                raise LocalReviewError("invalid_input") from None
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 86400:
            raise LocalReviewError("invalid_input")
        try:
            deadline = (current + timedelta(seconds=lease_seconds)).isoformat()
        except OverflowError:
            raise LocalReviewError("invalid_input") from None
        with self._transaction() as connection:
            # execute, not executescript: schema creation remains inside this transaction.
            for statement in LOCAL_REVIEW_SCHEMA_SQL.split(";"):
                if statement.strip():
                    connection.execute(statement)
            refresh_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(local_review_report_refreshes)")
            }
            if "last_attempt_at" not in refresh_columns:
                connection.execute(
                    "ALTER TABLE local_review_report_refreshes ADD COLUMN last_attempt_at TEXT"
                )
            attempt = _attempt(connection, request_id)
            if attempt is not None and attempt["event_id"] != event_id:
                raise LocalReviewError("request_conflict")
            target = _refresh(connection, request_id)
            if (
                target is not None
                and report_target is not None
                and (
                    str(report_target[0]) != target["storage_root"]
                    or report_target[1] != target["timezone_name"]
                )
            ):
                raise LocalReviewError("request_conflict")
            row, event, binding = self._load(connection, event_id)
            self._expire(connection, event_id, current.isoformat())
            if attempt is not None:
                attempt = _attempt(connection, request_id)
                return LocalReviewReservation(
                    event, binding, request_id, True, attempt["status"], attempt["error_code"]
                )
            self._eligible(event)
            new_target = None
            if report_target is not None:
                root, timezone = report_target
                root = _root(root)
                day = event.occurred_at.astimezone(ZoneInfo(timezone)).date()
                _day_bounds(day, timezone)
                new_target = (request_id, str(root), timezone, day.isoformat())
            if connection.execute(
                "SELECT 1 FROM local_review_attempts WHERE event_id=? AND status='running'",
                (event_id,),
            ).fetchone():
                raise LocalReviewError("event_busy")
            connection.execute(
                "INSERT INTO local_review_attempts(request_id,event_id,status,event_sha256,"
                "binding_sha256,before_result_json,before_decision_json,requested_at,lease_until) "
                "VALUES(?,?,'running',?,?,?,?,?,?)",
                (
                    request_id,
                    event_id,
                    _row_fingerprint(row),
                    binding.fingerprint,
                    row["result_json"],
                    row["decision_json"] or event.decision.to_json(indent=None),
                    current.isoformat(),
                    deadline,
                ),
            )
            if new_target is not None:
                connection.execute(
                    "INSERT INTO local_review_report_refreshes "
                    "(request_id,storage_root,timezone_name,local_date) VALUES(?,?,?,?)",
                    new_target,
                )
            return LocalReviewReservation(event, binding, request_id, False, "running")

    def report_refresh_state(self, request_id: str) -> dict[str, object]:
        _identifier(request_id)
        if not self.path.exists():
            return {"status": "not_requested"}
        with self._transaction(readonly=True) as connection:
            row = _refresh(connection, request_id)
            if row is None:
                return {"status": "not_requested"}
            attempt = _attempt(connection, request_id)
            if attempt is None:
                raise LocalReviewError("invalid_record")
            if attempt["status"] == "failed":
                return {"status": "not_requested"}
            status = (
                "awaiting_review"
                if attempt["status"] != "succeeded"
                else ("queued" if row["queued_job_id"] is not None else "pending")
            )
            result = {"status": status}
            if row["queued_job_id"] is not None:
                result["job_id"] = row["queued_job_id"]
            if row["last_error_code"] is not None:
                result["error_code"] = row["last_error_code"]
            return result

    def pending_report_refreshes(
        self,
        limit: int = 20,
        request_id: str | None = None,
    ) -> tuple[LocalReviewReportIntent, ...]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise LocalReviewError("invalid_input")
        if request_id is not None:
            _identifier(request_id)
        if not self.path.exists():
            return ()
        with self._transaction(readonly=True) as connection:
            if not _refresh_exists(connection):
                return ()
            selection = " AND r.request_id=?" if request_id is not None else ""
            parameters = (request_id, limit) if request_id is not None else (limit,)
            rows = connection.execute(
                "SELECT " + _refresh_projection() + " FROM local_review_report_refreshes r "
                "JOIN local_review_attempts a ON a.request_id=r.request_id "
                "WHERE a.status='succeeded' AND r.queued_job_id IS NULL"
                + selection
                + " ORDER BY r.last_attempt_at IS NOT NULL,r.last_attempt_at,r.request_id LIMIT ?",
                parameters,
            ).fetchall()
            result = []
            for row in rows:
                _attempt(connection, row["request_id"])
                result.append(_parse_refresh(row))
            return tuple(result)

    def mark_report_enqueued(
        self,
        request_id: str,
        *,
        job_id: str,
        now: datetime | None = None,
    ) -> None:
        _identifier(request_id)
        _report_job_id(job_id)
        current = _now(now).isoformat()
        with self._transaction() as connection:
            row = _refresh(connection, request_id)
            if row is None:
                raise LocalReviewError("request_not_found")
            attempt = _attempt(connection, request_id)
            if attempt is None or attempt["status"] != "succeeded":
                raise LocalReviewError("not_running")
            if row["queued_job_id"] is not None:
                if row["queued_job_id"] != job_id:
                    raise LocalReviewError("request_conflict")
                return
            if current < attempt["completed_at"]:
                raise LocalReviewError("invalid_input")
            connection.execute(
                "UPDATE local_review_report_refreshes SET queued_job_id=?,enqueued_at=?,"
                "last_error_code=NULL,last_attempt_at=? WHERE request_id=?",
                (job_id, current, current, request_id),
            )

    def mark_report_failed(
        self,
        request_id: str,
        *,
        error_code: str = "report_enqueue_failed",
        now: datetime | None = None,
    ) -> None:
        _identifier(request_id)
        current = _now(now).isoformat()
        # Error messages are never persisted; only this controlled retry status is stored.
        with self._transaction() as connection:
            row = _refresh(connection, request_id)
            if row is None:
                raise LocalReviewError("request_not_found")
            attempt = _attempt(connection, request_id)
            if attempt is None or attempt["status"] != "succeeded":
                raise LocalReviewError("not_running")
            if row["queued_job_id"] is None:
                if current < attempt["completed_at"]:
                    raise LocalReviewError("invalid_input")
                connection.execute(
                    "UPDATE local_review_report_refreshes SET "
                    "last_error_code='report_enqueue_failed',last_attempt_at=? WHERE request_id=?",
                    (current, request_id),
                )

    def complete(
        self,
        request_id: str,
        *,
        secondary_result: VLMEvent,
        secondary_provenance: RecognitionProvenance,
        secondary_metrics: RecognitionRunMetrics,
        decision: DecisionRecord,
        now: datetime | None = None,
    ) -> None:
        _identifier(request_id)
        current = _now(now).isoformat()
        try:
            for value, expected in (
                (secondary_result, VLMEvent),
                (secondary_provenance, RecognitionProvenance),
                (secondary_metrics, RecognitionRunMetrics),
                (decision, DecisionRecord),
            ):
                if not isinstance(value, expected):
                    raise LocalReviewError("invalid_completion")
                expected.from_json(value.to_json(indent=None))
            if (
                decision.final_source is not FinalSource.LOCAL_SECONDARY
                or decision.remote_used
                or decision.risk_level is not secondary_result.risk_level
                or decision.review_tier.rank < ReviewTier.LOCAL_SECONDARY.rank
            ):
                raise LocalReviewError("invalid_completion")
            result = (
                secondary_result.require_remote_review()
                if decision.needs_remote_review
                else secondary_result
            )
            result_json, decision_json = result.to_json(indent=None), decision.to_json(indent=None)
            raw_json, provenance_json, metrics_json = (
                value.to_json(indent=None)
                for value in (secondary_result, secondary_provenance, secondary_metrics)
            )
            if (
                sum(
                    len(value.encode())
                    for value in (
                        result_json,
                        decision_json,
                        raw_json,
                        provenance_json,
                        metrics_json,
                    )
                )
                > _MAX_ROW_BYTES
            ):
                raise LocalReviewError("invalid_completion")
        except (ValueError, TypeError, AttributeError, OverflowError, RecursionError):
            raise LocalReviewError("invalid_completion") from None
        expired = False
        with self._transaction() as connection:
            attempt = _attempt(connection, request_id)
            if attempt is None:
                raise LocalReviewError("request_not_found")
            if attempt["status"] != "running":
                raise LocalReviewError("not_running")
            if current < attempt["requested_at"]:
                raise LocalReviewError("invalid_input")
            if current >= attempt["lease_until"]:
                self._expire(connection, attempt["event_id"], current)
                expired = True
            else:
                row, event, binding = self._load(connection, attempt["event_id"])
                if (
                    _row_fingerprint(row) != attempt["event_sha256"]
                    or binding.fingerprint != attempt["binding_sha256"]
                ):
                    raise LocalReviewError("event_changed")
                self._eligible(event)
                original = event.decision
                if (
                    decision.remote_allowed != original.remote_allowed
                    or decision.review_tier.rank < original.review_tier.rank
                    or not set(original.review_reasons).issubset(decision.review_reasons)
                    or not set(original.contradictions).issubset(decision.contradictions)
                    or decision.signal_frequency_count != original.signal_frequency_count
                    or decision.signal_frequency_window_seconds
                    != original.signal_frequency_window_seconds
                    or decision.frequent_signal_types != original.frequent_signal_types
                ):
                    raise LocalReviewError("invalid_completion")
                payload = result.to_dict()
                connection.execute(
                    "UPDATE events SET secondary_result_json=?,secondary_provenance_json=?,"
                    "secondary_metrics_json=?,"
                    "result_json=?,decision_json=?,event_type=?,subjects_json=?,observations_json=?,inferences_json=?,"
                    "risk_level=?,confidence=?,needs_remote_review=?,summary_zh=?,"
                    "evidence_frames_json=? WHERE event_id=?",
                    (
                        raw_json,
                        provenance_json,
                        metrics_json,
                        result_json,
                        decision_json,
                        result.event_type.value,
                        json.dumps(payload["subjects"], ensure_ascii=False),
                        json.dumps(payload["observations"], ensure_ascii=False),
                        json.dumps(payload["inferences"], ensure_ascii=False),
                        result.risk_level.value,
                        result.confidence,
                        int(result.needs_remote_review),
                        result.summary_zh,
                        json.dumps(payload["evidence_frames"], ensure_ascii=False),
                        event.event_id,
                    ),
                )
                connection.execute(
                    "UPDATE local_review_attempts SET status='succeeded',completed_at=?,"
                    "after_result_json=?,after_decision_json=?,after_secondary_result_json=?,"
                    "after_provenance_json=?,after_metrics_json=? WHERE request_id=?",
                    (
                        current,
                        result_json,
                        decision_json,
                        raw_json,
                        provenance_json,
                        metrics_json,
                        request_id,
                    ),
                )
        if expired:
            raise LocalReviewError("lease_expired")

    def fail(self, request_id: str, *, error_code: str, now: datetime | None = None) -> None:
        _identifier(request_id)
        current = _now(now).isoformat()
        code = (
            error_code
            if isinstance(error_code, str) and error_code in _FAIL_CODES
            else "review_failed"
        )
        with self._transaction() as connection:
            attempt = _attempt(connection, request_id)
            if attempt is None:
                raise LocalReviewError("request_not_found")
            if attempt["status"] != "running":
                return
            if current < attempt["requested_at"]:
                raise LocalReviewError("invalid_input")
            if current >= attempt["lease_until"]:
                code = "lease_expired"
            connection.execute(
                "UPDATE local_review_attempts SET status='failed',completed_at=?,error_code=? "
                "WHERE request_id=?",
                (current, code, request_id),
            )

    def recent(self, limit: int = 20) -> tuple[dict[str, object], ...]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise LocalReviewError("invalid_input")
        with self._transaction(readonly=True) as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='local_review_attempts'"
                ).fetchone()
                is None
            ):
                return ()
            rows = connection.execute(
                "SELECT "
                + _projection(tuple(_ATTEMPT_LIMITS)[:7])
                + " FROM local_review_attempts ORDER BY requested_at DESC,request_id LIMIT ?",
                (limit,),
            ).fetchall()
            result = []
            for row in rows:
                _validate_attempt(row)
                result.append(dict(row))
            return tuple(result)
