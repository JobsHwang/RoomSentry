from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .domain import (
    CandidateEvent,
    DecisionRecord,
    EventMedia,
    FinalSource,
    RecognitionProvenance,
    RecognitionRunMetrics,
    ReviewTier,
    SignalEvent,
    SignalType,
    VLMEvent,
)
from .evidence_binding import (
    MAX_BINDING_BYTES,
    EvidenceBinding,
    EvidenceBindingError,
    media_fingerprint,
)
from .presence_history import PRESENCE_FACT_SCHEMA_SQL
from .recognition_attempt import RECOGNITION_ATTEMPT_SCHEMA_SQL
from .remote_review import REMOTE_REVIEW_SCHEMA_SQL
from .sampling_history import SAMPLING_HISTORY_SCHEMA_SQL

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    event_type TEXT NOT NULL,
    subjects_json TEXT NOT NULL,
    observations_json TEXT NOT NULL,
    inferences_json TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    needs_remote_review INTEGER NOT NULL CHECK (needs_remote_review IN (0, 1)),
    remote_used INTEGER NOT NULL CHECK (remote_used IN (0, 1)),
    summary_zh TEXT NOT NULL,
    evidence_frames_json TEXT NOT NULL,
    clip_path TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    local_result_json TEXT NOT NULL,
    local_provenance_json TEXT,
    local_metrics_json TEXT,
    secondary_result_json TEXT,
    secondary_provenance_json TEXT,
    secondary_metrics_json TEXT,
    remote_result_json TEXT,
    remote_provenance_json TEXT,
    remote_metrics_json TEXT,
    result_json TEXT NOT NULL,
    decision_json TEXT,
    evidence_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events (occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events (camera_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_events_risk_time ON events (risk_level, occurred_at);

CREATE TABLE IF NOT EXISTS event_evidence_bindings (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id),
    binding_json TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class StoredEvent:
    event_id: str
    camera_id: str
    occurred_at: datetime
    source_kind: str
    source_ref: str
    local_result: VLMEvent
    local_provenance: RecognitionProvenance
    local_metrics: RecognitionRunMetrics | None
    secondary_result: VLMEvent | None
    secondary_provenance: RecognitionProvenance | None
    secondary_metrics: RecognitionRunMetrics | None
    remote_result: VLMEvent | None
    remote_provenance: RecognitionProvenance | None
    remote_metrics: RecognitionRunMetrics | None
    result: VLMEvent
    decision: DecisionRecord
    evidence: EventMedia
    evidence_metadata_present: bool
    provenance_metadata_present: bool
    metrics_metadata_present: bool
    clip_path: str
    candidate: dict[str, Any]
    remote_used: bool
    created_at: datetime


class EventDatabase:
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
        with self._session() as connection:
            connection.executescript(SCHEMA_SQL)
            connection.executescript(REMOTE_REVIEW_SCHEMA_SQL)
            connection.executescript(RECOGNITION_ATTEMPT_SCHEMA_SQL)
            connection.executescript(PRESENCE_FACT_SCHEMA_SQL)
            connection.executescript(SAMPLING_HISTORY_SCHEMA_SQL)
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(events)").fetchall()
            }
            if "secondary_result_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN secondary_result_json TEXT")
            if "decision_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN decision_json TEXT")
            if "evidence_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN evidence_json TEXT")
            if "local_provenance_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN local_provenance_json TEXT")
            if "secondary_provenance_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN secondary_provenance_json TEXT")
            if "remote_provenance_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN remote_provenance_json TEXT")
            if "local_metrics_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN local_metrics_json TEXT")
            if "secondary_metrics_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN secondary_metrics_json TEXT")
            if "remote_metrics_json" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN remote_metrics_json TEXT")

    def insert_event(
        self,
        *,
        candidate: CandidateEvent,
        occurred_at: datetime,
        source_kind: str,
        source_ref: str,
        local_result: VLMEvent,
        local_provenance: RecognitionProvenance,
        local_metrics: RecognitionRunMetrics,
        secondary_result: VLMEvent | None,
        secondary_provenance: RecognitionProvenance | None,
        secondary_metrics: RecognitionRunMetrics | None,
        remote_result: VLMEvent | None,
        remote_provenance: RecognitionProvenance | None,
        remote_metrics: RecognitionRunMetrics | None,
        result: VLMEvent,
        decision: DecisionRecord,
        media: EventMedia,
        remote_used: bool,
        created_at: datetime,
        evidence_binding: EvidenceBinding | None = None,
    ) -> None:
        if occurred_at.tzinfo is None or created_at.tzinfo is None:
            raise ValueError("database timestamps must be timezone-aware")
        payload = result.to_dict()
        candidate_payload = candidate.to_dict()
        media_json = json.dumps(media.to_dict(), ensure_ascii=False)
        binding_json = None
        if evidence_binding is not None:
            if not isinstance(evidence_binding, EvidenceBinding) or (
                evidence_binding.media_sha256 != media_fingerprint(media)
                or len(evidence_binding.assets) != 1 + len(media.frame_paths)
            ):
                raise EvidenceBindingError("evidence binding does not match event metadata")
            if len(media_json.encode("utf-8")) > 65_536:
                raise EvidenceBindingError("bound event metadata exceeds its size limit")
            binding_json = evidence_binding.to_json()
        with self._session() as connection:
            connection.execute(
                """
                INSERT INTO events (
                    event_id, camera_id, occurred_at, source_kind, source_ref,
                    event_type, subjects_json, observations_json, inferences_json,
                    risk_level, confidence, needs_remote_review, remote_used,
                    summary_zh, evidence_frames_json, clip_path, candidate_json,
                    local_result_json, local_provenance_json, local_metrics_json,
                    secondary_result_json, secondary_provenance_json,
                    secondary_metrics_json,
                    remote_result_json, remote_provenance_json, remote_metrics_json,
                    result_json, decision_json, evidence_json, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    candidate.event_id,
                    candidate.camera_id,
                    occurred_at.astimezone(UTC).isoformat(),
                    source_kind,
                    source_ref,
                    result.event_type.value,
                    json.dumps(payload["subjects"], ensure_ascii=False),
                    json.dumps(payload["observations"], ensure_ascii=False),
                    json.dumps(payload["inferences"], ensure_ascii=False),
                    result.risk_level.value,
                    result.confidence,
                    int(result.needs_remote_review),
                    int(remote_used),
                    result.summary_zh,
                    json.dumps(payload["evidence_frames"], ensure_ascii=False),
                    media.clip_path,
                    json.dumps(candidate_payload, ensure_ascii=False),
                    local_result.to_json(indent=None),
                    local_provenance.to_json(indent=None),
                    local_metrics.to_json(indent=None),
                    secondary_result.to_json(indent=None) if secondary_result else None,
                    (secondary_provenance.to_json(indent=None) if secondary_provenance else None),
                    secondary_metrics.to_json(indent=None) if secondary_metrics else None,
                    remote_result.to_json(indent=None) if remote_result else None,
                    remote_provenance.to_json(indent=None) if remote_provenance else None,
                    remote_metrics.to_json(indent=None) if remote_metrics else None,
                    result.to_json(indent=None),
                    decision.to_json(indent=None),
                    media_json,
                    created_at.astimezone(UTC).isoformat(),
                ),
            )

            if binding_json is not None:
                connection.execute(
                    "INSERT INTO event_evidence_bindings (event_id, binding_json) VALUES (?, ?)",
                    (candidate.event_id, binding_json),
                )

    def get_evidence_binding(self, event_id: str) -> EvidenceBinding | None:
        """Read historical binding metadata; never hash or backfill current media."""
        try:
            return self._get_evidence_binding(event_id)
        except (sqlite3.Error, OSError):
            # SQLite decoding and connection errors can include private paths
            # or corrupt column contents before JSON validation even starts.
            raise EvidenceBindingError("stored evidence binding is unavailable") from None

    def _get_evidence_binding(self, event_id: str) -> EvidenceBinding | None:
        if not self.path.exists():
            return None
        with closing(
            sqlite3.connect(self.path.absolute().as_uri() + "?mode=ro", uri=True)
        ) as connection:
            connection.row_factory = sqlite3.Row
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'event_evidence_bindings'"
            ).fetchone()
            if exists is None:
                return None
            row = connection.execute(
                "SELECT CASE WHEN typeof(b.binding_json) = 'text' "
                "AND length(CAST(b.binding_json AS BLOB)) <= ? THEN b.binding_json END AS binding, "
                "CASE WHEN typeof(e.evidence_json) = 'text' "
                "AND length(CAST(e.evidence_json AS BLOB)) <= 65536 "
                "THEN e.evidence_json END AS media "
                "FROM event_evidence_bindings b LEFT JOIN events e ON e.event_id = b.event_id "
                "WHERE b.event_id = ?",
                (MAX_BINDING_BYTES, event_id),
            ).fetchone()
        if row is None:
            return None
        try:
            binding = EvidenceBinding.from_json(row["binding"])

            def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError("duplicate metadata key")
                    value[key] = item
                return value

            media = EventMedia.from_dict(json.loads(row["media"], object_pairs_hook=unique_object))
            if binding.media_sha256 != media_fingerprint(media) or len(binding.assets) != 1 + len(
                media.frame_paths
            ):
                raise EvidenceBindingError("evidence binding does not match event metadata")
            return binding
        except (ValueError, TypeError, AttributeError, UnicodeError, RecursionError):
            raise EvidenceBindingError("stored evidence binding is invalid") from None

    def count_recent_signal_events(
        self,
        *,
        camera_id: str,
        signal_types: tuple[SignalType, ...],
        occurred_at: datetime,
        window_seconds: int,
    ) -> int:
        if occurred_at.tzinfo is None:
            raise ValueError("signal frequency timestamp must be timezone-aware")
        if not 10 <= window_seconds <= 86_400:
            raise ValueError("signal frequency window must be between 10 and 86400 seconds")
        if not signal_types:
            return 0
        if any(not isinstance(value, SignalType) for value in signal_types):
            raise ValueError("signal_types must contain SignalType values")
        end = occurred_at.astimezone(UTC)
        start = end - timedelta(seconds=window_seconds)
        with self._session() as connection:
            rows = connection.execute(
                "SELECT candidate_json FROM events "
                "WHERE camera_id = ? AND occurred_at >= ? AND occurred_at <= ?",
                (camera_id, start.isoformat(), end.isoformat()),
            ).fetchall()
        wanted = set(signal_types)
        count = 0
        for row in rows:
            try:
                candidate = json.loads(row["candidate_json"])
                raw_signals = candidate.get("signals", [])
                if not isinstance(raw_signals, list):
                    raise ValueError
                stored_types = {SignalEvent.from_dict(value).signal_type for value in raw_signals}
            except (AttributeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError("stored candidate signal data is invalid") from exc
            count += int(bool(stored_types & wanted))
        return count

    def fetch_between(
        self,
        start: datetime,
        end: datetime,
        *,
        camera_id: str | None = None,
    ) -> list[StoredEvent]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("query timestamps must be timezone-aware")
        sql = "SELECT * FROM events WHERE occurred_at >= ? AND occurred_at < ?"
        parameters: list[Any] = [
            start.astimezone(UTC).isoformat(),
            end.astimezone(UTC).isoformat(),
        ]
        if camera_id:
            sql += " AND camera_id = ?"
            parameters.append(camera_id)
        sql += " ORDER BY occurred_at ASC, event_id ASC"
        with self._session() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._row_to_event(row) for row in rows]

    def list_recent(self, *, limit: int = 50) -> list[StoredEvent]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._session() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY occurred_at DESC, event_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_event(self, event_id: str) -> StoredEvent | None:
        if not event_id:
            raise ValueError("event_id cannot be empty")
        with self._session() as connection:
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_event(row) if row is not None else None

    def iter_media_before(self, cutoff: datetime) -> Iterable[tuple[str, str, tuple[str, ...]]]:
        if cutoff.tzinfo is None:
            raise ValueError("cutoff must be timezone-aware")
        with self._session() as connection:
            rows = connection.execute(
                "SELECT event_id, clip_path, evidence_frames_json "
                "FROM events WHERE occurred_at < ?",
                (cutoff.astimezone(UTC).isoformat(),),
            ).fetchall()
        for row in rows:
            frames = tuple(json.loads(row["evidence_frames_json"]))
            yield row["event_id"], row["clip_path"], frames

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> StoredEvent:
        result = VLMEvent.from_json(row["result_json"])
        remote_used = bool(row["remote_used"])
        decision_payload = row["decision_json"]
        if decision_payload is None:
            decision = DecisionRecord(
                risk_level=result.risk_level,
                review_tier=(
                    ReviewTier.REMOTE
                    if result.needs_remote_review or remote_used
                    else ReviewTier.LOCAL_PRIMARY
                ),
                review_reasons=(
                    ("legacy_result_needs_remote_review",) if result.needs_remote_review else ()
                ),
                contradictions=(),
                remote_allowed=remote_used,
                remote_used=remote_used,
                final_source=(FinalSource.REMOTE if remote_used else FinalSource.LOCAL_PRIMARY),
            )
        else:
            decision = DecisionRecord.from_json(decision_payload)
        evidence_payload = row["evidence_json"]
        evidence = (
            EventMedia.from_dict(json.loads(evidence_payload))
            if evidence_payload is not None
            else EventMedia(row["clip_path"], result.evidence_frames)
        )
        local_provenance_payload = row["local_provenance_json"]
        secondary_provenance_payload = row["secondary_provenance_json"]
        remote_provenance_payload = row["remote_provenance_json"]
        local_metrics_payload = row["local_metrics_json"]
        secondary_metrics_payload = row["secondary_metrics_json"]
        remote_metrics_payload = row["remote_metrics_json"]
        legacy_provenance = RecognitionProvenance(adapter_name="legacy_unknown")
        local_provenance = (
            RecognitionProvenance.from_json(local_provenance_payload)
            if local_provenance_payload is not None
            else legacy_provenance
        )
        secondary_result = (
            VLMEvent.from_json(row["secondary_result_json"])
            if row["secondary_result_json"] is not None
            else None
        )
        remote_result = (
            VLMEvent.from_json(row["remote_result_json"])
            if row["remote_result_json"] is not None
            else None
        )
        return StoredEvent(
            event_id=row["event_id"],
            camera_id=row["camera_id"],
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            source_kind=row["source_kind"],
            source_ref=row["source_ref"],
            local_result=VLMEvent.from_json(row["local_result_json"]),
            local_provenance=local_provenance,
            local_metrics=(
                RecognitionRunMetrics.from_json(local_metrics_payload)
                if local_metrics_payload is not None
                else None
            ),
            secondary_result=secondary_result,
            secondary_provenance=(
                RecognitionProvenance.from_json(secondary_provenance_payload)
                if secondary_provenance_payload is not None
                else (legacy_provenance if secondary_result is not None else None)
            ),
            secondary_metrics=(
                RecognitionRunMetrics.from_json(secondary_metrics_payload)
                if secondary_metrics_payload is not None
                else None
            ),
            remote_result=remote_result,
            remote_provenance=(
                RecognitionProvenance.from_json(remote_provenance_payload)
                if remote_provenance_payload is not None
                else (legacy_provenance if remote_result is not None else None)
            ),
            remote_metrics=(
                RecognitionRunMetrics.from_json(remote_metrics_payload)
                if remote_metrics_payload is not None
                else None
            ),
            result=result,
            decision=decision,
            evidence=evidence,
            evidence_metadata_present=evidence_payload is not None,
            provenance_metadata_present=local_provenance_payload is not None,
            metrics_metadata_present=local_metrics_payload is not None,
            clip_path=row["clip_path"],
            candidate=json.loads(row["candidate_json"]),
            remote_used=remote_used,
            created_at=datetime.fromisoformat(row["created_at"]),
        )
