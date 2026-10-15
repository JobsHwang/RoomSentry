"""Persistent online binding for household event-policy decisions.

The pure rule engine in :mod:`baby_monitor.event_policy` remains the source
of routing semantics.  This module only gives those decisions an idempotent,
serialized SQLite ledger so an analysis retry cannot make a second routing
decision after a crash.

The ledger deliberately contains event summaries and routing metadata only:
no media, source URI, notification body, credential, or delivery authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .event_policy import (
    EventPolicyDecision,
    EventPolicyError,
    HouseholdEventPolicy,
    PolicyEventInput,
    evaluate_event_policy,
)

EVENT_POLICY_DECISION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS event_policy_decisions (
    event_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    event_json TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL,
    package_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('route', 'suppress')),
    rule_id TEXT,
    reason TEXT NOT NULL,
    sink_ids_json TEXT NOT NULL,
    local_only INTEGER NOT NULL CHECK (local_only IN (0, 1)),
    created_at TEXT NOT NULL,
    record_sha256 TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_policy_decisions_policy_time
    ON event_policy_decisions (policy_sha256, occurred_at, event_id);

CREATE INDEX IF NOT EXISTS idx_event_policy_decisions_policy_camera_time
    ON event_policy_decisions (policy_sha256, camera_id, occurred_at, event_id);

CREATE INDEX IF NOT EXISTS idx_event_policy_decisions_routed_history
    ON event_policy_decisions (
        policy_sha256, rule_id, camera_id, action, occurred_at, event_id
    );
"""

_EXPECTED_COLUMNS = {
    "event_id",
    "camera_id",
    "occurred_at",
    "event_json",
    "policy_sha256",
    "package_id",
    "action",
    "rule_id",
    "reason",
    "sink_ids_json",
    "local_only",
    "created_at",
    "record_sha256",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")


class EventPolicyRuntimeError(RuntimeError):
    """The online policy ledger is missing, corrupt, or inconsistent."""


def _utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EventPolicyRuntimeError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _record_sha256(
    *,
    event: PolicyEventInput,
    policy_sha256: str,
    package_id: str,
    decision: EventPolicyDecision,
    created_at: datetime,
) -> str:
    payload = {
        "event": event.to_dict(),
        "policy_sha256": policy_sha256,
        "package_id": package_id,
        "decision": decision.to_dict(),
        "created_at": _utc(created_at, field="decision created_at").isoformat(),
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def _strict_json(raw: Any, *, field: str) -> Any:
    if not isinstance(raw, str):
        raise EventPolicyRuntimeError(f"stored {field} is invalid")
    try:
        return json.loads(
            raw,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        raise EventPolicyRuntimeError(f"stored {field} is invalid") from None


def _normalized_event(event: PolicyEventInput) -> PolicyEventInput:
    if not isinstance(event, PolicyEventInput):
        raise EventPolicyRuntimeError("online event policy input is invalid")
    return replace(event, occurred_at=_utc(event.occurred_at, field="event occurred_at"))


@dataclass(frozen=True, slots=True)
class EventPolicyRoutingRecord:
    """One immutable online routing decision."""

    event: PolicyEventInput
    policy_sha256: str
    package_id: str
    decision: EventPolicyDecision
    created_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.event, PolicyEventInput):
            raise EventPolicyRuntimeError("stored event policy input is invalid")
        if not isinstance(self.policy_sha256, str) or _SHA256.fullmatch(self.policy_sha256) is None:
            raise EventPolicyRuntimeError("stored event policy fingerprint is invalid")
        if not isinstance(self.package_id, str) or _SAFE_ID.fullmatch(self.package_id) is None:
            raise EventPolicyRuntimeError("stored event policy package is invalid")
        if not isinstance(self.decision, EventPolicyDecision):
            raise EventPolicyRuntimeError("stored event policy decision is invalid")
        if self.decision.event_id != self.event.event_id:
            raise EventPolicyRuntimeError("stored event policy decision binding is invalid")
        _utc(self.created_at, field="decision created_at")


@dataclass(frozen=True, slots=True)
class EventPolicyRoutingResult:
    """A new or idempotently reused online policy decision."""

    record: EventPolicyRoutingRecord
    reused_existing: bool

    def to_public_dict(self) -> dict[str, object]:
        decision = self.record.decision
        return {
            "version": 1,
            "policy_sha256": self.record.policy_sha256,
            "package_id": self.record.package_id,
            "action": decision.action,
            "rule_id": decision.rule_id,
            "reason": decision.reason,
            "sink_ids": list(decision.sink_ids),
            "local_only": decision.local_only,
            "reused_existing": self.reused_existing,
            "routing_only": True,
            "facts_modified": False,
            "risk_inference": False,
            "notification_delivery_authority": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }


class EventPolicyDecisionStore:
    """Serialize and persist pure policy decisions before outbox enqueue."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
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
        try:
            with self._session() as connection:
                connection.executescript(EVENT_POLICY_DECISION_SCHEMA_SQL)
                columns = {
                    str(row["name"])
                    for row in connection.execute(
                        "PRAGMA table_info(event_policy_decisions)"
                    ).fetchall()
                }
                if columns != _EXPECTED_COLUMNS:
                    raise EventPolicyRuntimeError(
                        "event policy decision database schema is unsupported"
                    )
        except sqlite3.Error:
            raise EventPolicyRuntimeError(
                "event policy decision database is missing or invalid"
            ) from None

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> EventPolicyRoutingRecord:
        try:
            event = PolicyEventInput.from_dict(
                _strict_json(row["event_json"], field="policy event JSON")
            )
            event = _normalized_event(event)
            sink_ids = _strict_json(row["sink_ids_json"], field="policy sink JSON")
            if not isinstance(sink_ids, list):
                raise EventPolicyRuntimeError("stored policy sink JSON is invalid")
            local_only = row["local_only"]
            if type(local_only) is not int or local_only not in {0, 1}:
                raise EventPolicyRuntimeError("stored policy local-only flag is invalid")
            decision = EventPolicyDecision(
                event_id=row["event_id"],
                action=row["action"],
                rule_id=row["rule_id"],
                reason=row["reason"],
                sink_ids=tuple(sink_ids),
                local_only=bool(local_only),
            )
            occurred_at = datetime.fromisoformat(row["occurred_at"])
            created_at = datetime.fromisoformat(row["created_at"])
            record = EventPolicyRoutingRecord(
                event=event,
                policy_sha256=row["policy_sha256"],
                package_id=row["package_id"],
                decision=decision,
                created_at=_utc(created_at, field="decision created_at"),
            )
            record_sha256 = row["record_sha256"]
            if (
                row["event_id"] != event.event_id
                or row["camera_id"] != event.camera_id
                or row["occurred_at"] != event.occurred_at.isoformat()
                or _utc(occurred_at, field="decision occurred_at") != event.occurred_at
                or row["created_at"] != record.created_at.isoformat()
                or row["event_json"] != _json(event.to_dict())
                or row["sink_ids_json"] != _json(list(decision.sink_ids))
                or not isinstance(record_sha256, str)
                or _SHA256.fullmatch(record_sha256) is None
                or record_sha256
                != _record_sha256(
                    event=record.event,
                    policy_sha256=record.policy_sha256,
                    package_id=record.package_id,
                    decision=record.decision,
                    created_at=record.created_at,
                )
            ):
                raise EventPolicyRuntimeError("stored event policy decision is inconsistent")
            return record
        except EventPolicyRuntimeError:
            raise
        except (KeyError, TypeError, ValueError, EventPolicyError):
            raise EventPolicyRuntimeError("stored event policy decision is invalid") from None

    @staticmethod
    def _matching_rule(policy: HouseholdEventPolicy, event: PolicyEventInput):
        if not policy.enabled:
            return None
        return next((rule for rule in policy.rules if rule.matches(event)), None)

    def _routed_history(
        self,
        connection: sqlite3.Connection,
        *,
        policy: HouseholdEventPolicy,
        event: PolicyEventInput,
    ) -> tuple[EventPolicyRoutingRecord, ...]:
        rule = self._matching_rule(policy, event)
        if rule is None:
            return ()
        lookback_seconds = max(
            rule.cooldown_seconds,
            rule.aggregation_window_seconds,
        )
        if lookback_seconds <= 0.0:
            return ()
        start = event.occurred_at - timedelta(seconds=lookback_seconds)
        rows = connection.execute(
            """
            SELECT * FROM event_policy_decisions
            WHERE policy_sha256 = ? AND rule_id = ? AND camera_id = ?
              AND action = 'route' AND occurred_at >= ?
              AND (occurred_at < ? OR (occurred_at = ? AND event_id < ?))
            ORDER BY occurred_at ASC, event_id ASC
            LIMIT 102
            """,
            (
                policy.fingerprint,
                rule.rule_id,
                event.camera_id,
                start.isoformat(),
                event.occurred_at.isoformat(),
                event.occurred_at.isoformat(),
                event.event_id,
            ),
        ).fetchall()
        if len(rows) > 100:
            raise EventPolicyRuntimeError("event policy routed history exceeds its bound")
        return tuple(self._row_to_record(row) for row in rows)

    def decide_and_record(
        self,
        *,
        policy: HouseholdEventPolicy,
        event: PolicyEventInput,
        now: datetime | None = None,
    ) -> EventPolicyRoutingResult:
        """Evaluate one per-camera chronological event and durably bind its decision.

        Existing records win even if the configured policy changes between a
        crash and retry.  That preserves the original route/suppress outcome.
        A newly arriving event older than an already-recorded decision for the
        same policy fingerprint and camera fails closed because notifications
        already emitted for later events cannot be recalled safely. Independent
        camera streams do not impose a global arrival order on each other.
        """

        if not isinstance(policy, HouseholdEventPolicy):
            raise EventPolicyRuntimeError("online household event policy is invalid")
        normalized = _normalized_event(event)
        recorded_at = _utc(now or datetime.now(UTC), field="decision now")
        event_json = _json(normalized.to_dict())
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM event_policy_decisions WHERE event_id = ?",
                (normalized.event_id,),
            ).fetchone()
            if existing is not None:
                record = self._row_to_record(existing)
                if record.event != normalized:
                    raise EventPolicyRuntimeError(
                        "event policy decision already exists with different event facts"
                    )
                connection.commit()
                return EventPolicyRoutingResult(record=record, reused_existing=True)

            latest_row = connection.execute(
                """
                SELECT * FROM event_policy_decisions
                WHERE policy_sha256 = ? AND camera_id = ?
                ORDER BY occurred_at DESC, event_id DESC
                LIMIT 1
                """,
                (policy.fingerprint, normalized.camera_id),
            ).fetchone()
            if latest_row is not None:
                latest = self._row_to_record(latest_row)
                if (normalized.occurred_at, normalized.event_id) < (
                    latest.event.occurred_at,
                    latest.event.event_id,
                ):
                    raise EventPolicyRuntimeError(
                        "event policy input arrived out of chronological order"
                    )

            history = self._routed_history(
                connection,
                policy=policy,
                event=normalized,
            )
            evaluation = evaluate_event_policy(
                policy,
                tuple(record.event for record in history) + (normalized,),
            )
            decisions = {decision.event_id: decision for decision in evaluation.decisions}
            for record in history:
                if decisions.get(record.event.event_id) != record.decision:
                    raise EventPolicyRuntimeError(
                        "stored event policy routed history is inconsistent"
                    )
            decision = decisions.get(normalized.event_id)
            if decision is None:
                raise EventPolicyRuntimeError("online event policy decision disappeared")
            record_sha256 = _record_sha256(
                event=normalized,
                policy_sha256=policy.fingerprint,
                package_id=policy.package_id,
                decision=decision,
                created_at=recorded_at,
            )

            connection.execute(
                """
                INSERT INTO event_policy_decisions (
                    event_id, camera_id, occurred_at, event_json,
                    policy_sha256, package_id, action, rule_id, reason,
                    sink_ids_json, local_only, created_at, record_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized.event_id,
                    normalized.camera_id,
                    normalized.occurred_at.isoformat(),
                    event_json,
                    policy.fingerprint,
                    policy.package_id,
                    decision.action,
                    decision.rule_id,
                    decision.reason,
                    _json(list(decision.sink_ids)),
                    int(decision.local_only),
                    recorded_at.isoformat(),
                    record_sha256,
                ),
            )
            row = connection.execute(
                "SELECT * FROM event_policy_decisions WHERE event_id = ?",
                (normalized.event_id,),
            ).fetchone()
            if row is None:
                raise EventPolicyRuntimeError("event policy decision disappeared from SQLite")
            record = self._row_to_record(row)
            connection.commit()
            return EventPolicyRoutingResult(record=record, reused_existing=False)
        except EventPolicyRuntimeError:
            raise
        except (sqlite3.Error, EventPolicyError, TypeError, ValueError):
            raise EventPolicyRuntimeError(
                "event policy decision database is missing or invalid"
            ) from None
        finally:
            connection.close()
