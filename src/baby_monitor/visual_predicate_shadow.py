from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from baby_monitor.adapters.visual_predicate import (
    VisualPredicateAdapter,
    VisualPredicateFrame,
    VisualPredicateRequest,
)
from baby_monitor.domain import CandidateEvent
from baby_monitor.visual_predicate import VisualPredicate, VisualPredicateResult

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class VisualPredicateShadowError(ValueError):
    """Path-free error for shadow gate, orchestration, and persistence."""


class VisualPredicateShadowTrigger(StrEnum):
    PERSON_DETECTION = "person_detection"
    HARDWARE_SIGNAL = "hardware_signal"
    PERIODIC_SAMPLE = "periodic_sample"


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowGate:
    event_id: str
    selected: bool
    triggers: tuple[VisualPredicateShadowTrigger, ...]
    predicates: tuple[VisualPredicate, ...]
    person_detector_id: str | None
    person_detection_count: int
    hardware_signal_adapters: tuple[str, ...]
    hardware_signal_count: int
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise VisualPredicateShadowError("unsupported visual predicate shadow gate version")
        if not isinstance(self.event_id, str) or _SAFE_ID.fullmatch(self.event_id) is None:
            raise VisualPredicateShadowError("visual predicate shadow event_id is invalid")
        if (
            not isinstance(self.triggers, tuple)
            or len(self.triggers) > len(VisualPredicateShadowTrigger)
            or len(set(self.triggers)) != len(self.triggers)
            or any(not isinstance(item, VisualPredicateShadowTrigger) for item in self.triggers)
        ):
            raise VisualPredicateShadowError("visual predicate shadow triggers are invalid")
        if (
            not isinstance(self.predicates, tuple)
            or len(self.predicates) > len(VisualPredicate)
            or len(set(self.predicates)) != len(self.predicates)
            or any(not isinstance(item, VisualPredicate) for item in self.predicates)
        ):
            raise VisualPredicateShadowError("visual predicate shadow predicates are invalid")
        if type(self.selected) is not bool or self.selected != bool(
            self.triggers and self.predicates
        ):
            raise VisualPredicateShadowError("visual predicate shadow selection is inconsistent")
        if self.person_detector_id is not None and (
            not isinstance(self.person_detector_id, str)
            or _SAFE_ID.fullmatch(self.person_detector_id) is None
        ):
            raise VisualPredicateShadowError(
                "visual predicate shadow person_detector_id is invalid"
            )
        if (
            not isinstance(self.hardware_signal_adapters, tuple)
            or len(set(self.hardware_signal_adapters)) != len(self.hardware_signal_adapters)
            or any(
                not isinstance(item, str) or _SAFE_ID.fullmatch(item) is None
                for item in self.hardware_signal_adapters
            )
        ):
            raise VisualPredicateShadowError(
                "visual predicate shadow hardware signal adapters are invalid"
            )

        for value, field_name in (
            (self.person_detection_count, "person_detection_count"),
            (self.hardware_signal_count, "hardware_signal_count"),
        ):
            if type(value) is not int or not 0 <= value <= 1_000_000:
                raise VisualPredicateShadowError(f"visual predicate shadow {field_name} is invalid")
        has_person_trigger = VisualPredicateShadowTrigger.PERSON_DETECTION in self.triggers
        has_hardware_trigger = VisualPredicateShadowTrigger.HARDWARE_SIGNAL in self.triggers
        if has_person_trigger != bool(
            self.person_detector_id is not None and self.person_detection_count
        ) or has_hardware_trigger != bool(
            self.hardware_signal_adapters and self.hardware_signal_count
        ):
            raise VisualPredicateShadowError(
                "visual predicate shadow trigger provenance is inconsistent"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "event_id": self.event_id,
            "selected": self.selected,
            "triggers": [item.value for item in self.triggers],
            "predicates": [item.value for item in self.predicates],
            "person_detector_id": self.person_detector_id,
            "person_detection_count": self.person_detection_count,
            "hardware_signal_adapters": list(self.hardware_signal_adapters),
            "hardware_signal_count": self.hardware_signal_count,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
            "paths_disclosed": False,
        }


def select_visual_predicate_shadow(
    candidate: CandidateEvent,
    *,
    predicates: tuple[VisualPredicate, ...],
    person_detector_id: str | None = None,
    hardware_signal_source_adapters: tuple[str, ...] = ("home_assistant",),
    periodic_sample_due: bool = False,
    min_person_confidence: float = 0.25,
) -> VisualPredicateShadowGate:
    if not isinstance(candidate, CandidateEvent):
        raise VisualPredicateShadowError("visual predicate shadow candidate is invalid")
    if (
        not isinstance(predicates, tuple)
        or len(predicates) > len(VisualPredicate)
        or len(set(predicates)) != len(predicates)
        or any(not isinstance(item, VisualPredicate) for item in predicates)
    ):
        raise VisualPredicateShadowError("visual predicate shadow predicates are invalid")
    if person_detector_id is not None and (
        not isinstance(person_detector_id, str) or _SAFE_ID.fullmatch(person_detector_id) is None
    ):
        raise VisualPredicateShadowError("person_detector_id is invalid")
    if (
        not isinstance(hardware_signal_source_adapters, tuple)
        or len(set(hardware_signal_source_adapters)) != len(hardware_signal_source_adapters)
        or any(
            not isinstance(item, str) or _SAFE_ID.fullmatch(item) is None
            for item in hardware_signal_source_adapters
        )
    ):
        raise VisualPredicateShadowError("hardware_signal_source_adapters are invalid")

    if type(periodic_sample_due) is not bool:
        raise VisualPredicateShadowError("periodic_sample_due must be a boolean")
    if (
        type(min_person_confidence) not in {int, float}
        or not math.isfinite(min_person_confidence)
        or not 0.0 <= float(min_person_confidence) <= 1.0
    ):
        raise VisualPredicateShadowError("min_person_confidence is invalid")

    person_count = (
        sum(
            1
            for detection in candidate.detections
            if detection.label == "person" and detection.confidence >= float(min_person_confidence)
        )
        if person_detector_id is not None
        else 0
    )
    hardware_signals = tuple(
        signal
        for signal in candidate.signals
        if signal.source_adapter in hardware_signal_source_adapters
    )
    matched_hardware_adapters = tuple(
        sorted({signal.source_adapter for signal in hardware_signals})
    )
    triggers: list[VisualPredicateShadowTrigger] = []
    if person_count:
        triggers.append(VisualPredicateShadowTrigger.PERSON_DETECTION)
    if hardware_signals:
        triggers.append(VisualPredicateShadowTrigger.HARDWARE_SIGNAL)
    if periodic_sample_due:
        triggers.append(VisualPredicateShadowTrigger.PERIODIC_SAMPLE)
    return VisualPredicateShadowGate(
        event_id=candidate.event_id,
        selected=bool(triggers and predicates),
        triggers=tuple(triggers),
        predicates=predicates,
        person_detector_id=person_detector_id,
        person_detection_count=person_count,
        hardware_signal_adapters=matched_hardware_adapters,
        hardware_signal_count=len(hardware_signals),
    )


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowRun:
    gate: VisualPredicateShadowGate
    results: tuple[VisualPredicateResult, ...]
    recorded_at: datetime
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise VisualPredicateShadowError("unsupported visual predicate shadow run version")
        if not isinstance(self.gate, VisualPredicateShadowGate) or not self.gate.selected:
            raise VisualPredicateShadowError("visual predicate shadow run requires a selected gate")
        if (
            not isinstance(self.results, tuple)
            or tuple(item.predicate for item in self.results) != self.gate.predicates
            or any(item.event_id != self.gate.event_id for item in self.results)
        ):
            raise VisualPredicateShadowError("visual predicate shadow results are inconsistent")
        if not self.results:
            raise VisualPredicateShadowError("visual predicate shadow results are empty")
        bindings = {
            (item.model_id, item.model_revision, item.prompt_version) for item in self.results
        }
        if len(bindings) != 1:
            raise VisualPredicateShadowError("visual predicate shadow model binding changed")
        if (
            not isinstance(self.recorded_at, datetime)
            or self.recorded_at.tzinfo is None
            or self.recorded_at.utcoffset() is None
        ):
            raise VisualPredicateShadowError("visual predicate shadow recorded_at is invalid")
        object.__setattr__(self, "recorded_at", self.recorded_at.astimezone(UTC))

    def binding_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "gate": self.gate.to_dict(),
            "results": [item.to_dict() for item in self.results],
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
            "paths_disclosed": False,
            "media_metadata_included": False,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.binding_dict(),
            "recorded_at": self.recorded_at.isoformat(),
        }


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS visual_predicate_shadow_runs (
    event_id TEXT PRIMARY KEY,
    binding_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowPersistResult:
    event_id: str
    predicate_count: int
    reused_existing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "predicate_count": self.predicate_count,
            "reused_existing": self.reused_existing,
            "paths_disclosed": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }


class VisualPredicateShadowStore:
    def __init__(self, database_path: str | Path):
        path = Path(database_path).expanduser()
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise VisualPredicateShadowError("visual predicate shadow database is unavailable")
        try:
            self.path = path.resolve(strict=True)
        except OSError:
            raise VisualPredicateShadowError(
                "visual predicate shadow database is unavailable"
            ) from None

    def persist(self, run: VisualPredicateShadowRun) -> VisualPredicateShadowPersistResult:
        if not isinstance(run, VisualPredicateShadowRun):
            raise VisualPredicateShadowError("visual predicate shadow run is invalid")
        binding_json = json.dumps(
            run.binding_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        try:
            connection = sqlite3.connect(f"{self.path.as_uri()}?mode=rw", uri=True)
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(_SCHEMA_SQL)
            existing = connection.execute(
                "SELECT binding_json FROM visual_predicate_shadow_runs WHERE event_id = ?",
                (run.gate.event_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO visual_predicate_shadow_runs (
                        event_id, binding_json, recorded_at
                    ) VALUES (?, ?, ?)
                    """,
                    (run.gate.event_id, binding_json, run.recorded_at.isoformat()),
                )
                reused = False
            elif existing[0] == binding_json:
                reused = True
            else:
                raise VisualPredicateShadowError(
                    "visual predicate shadow event conflicts with an existing run"
                )
            connection.commit()
        except VisualPredicateShadowError:
            if "connection" in locals():
                connection.rollback()
            raise
        except sqlite3.Error:
            if "connection" in locals():
                connection.rollback()
            raise VisualPredicateShadowError(
                "visual predicate shadow database is unavailable"
            ) from None
        finally:
            if "connection" in locals():
                connection.close()
        return VisualPredicateShadowPersistResult(
            event_id=run.gate.event_id,
            predicate_count=len(run.results),
            reused_existing=reused,
        )


@dataclass(frozen=True, slots=True)
class VisualPredicateShadowSummary:
    gate: VisualPredicateShadowGate
    results: tuple[VisualPredicateResult, ...]
    persisted: bool
    reused_existing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "gate": self.gate.to_dict(),
            "results": [item.to_dict() for item in self.results],
            "persisted": self.persisted,
            "reused_existing": self.reused_existing,
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
            "paths_disclosed": False,
            "media_metadata_included": False,
        }


def run_visual_predicate_shadow(
    *,
    gate: VisualPredicateShadowGate,
    frames: tuple[VisualPredicateFrame, ...],
    adapter: VisualPredicateAdapter,
    store_factory: Callable[[], VisualPredicateShadowStore],
    recorded_at: datetime | None = None,
) -> VisualPredicateShadowSummary:
    if not isinstance(gate, VisualPredicateShadowGate):
        raise VisualPredicateShadowError("visual predicate shadow gate is invalid")
    if not gate.selected:
        return VisualPredicateShadowSummary(
            gate=gate,
            results=(),
            persisted=False,
            reused_existing=False,
        )
    if (
        not isinstance(frames, tuple)
        or not 1 <= len(frames) <= 12
        or any(not isinstance(item, VisualPredicateFrame) for item in frames)
    ):
        raise VisualPredicateShadowError("visual predicate shadow frames are invalid")
    if not isinstance(adapter, VisualPredicateAdapter):
        raise VisualPredicateShadowError("visual predicate shadow adapter is invalid")
    if not callable(store_factory):
        raise VisualPredicateShadowError("visual predicate shadow store factory is invalid")

    results: list[VisualPredicateResult] = []
    allowed_frame_ids = frozenset(item.frame_id for item in frames)
    for predicate in gate.predicates:
        try:
            result = adapter.assess(
                VisualPredicateRequest(
                    event_id=gate.event_id,
                    predicate=predicate,
                    frames=frames,
                )
            )
            if (
                not isinstance(result, VisualPredicateResult)
                or result.event_id != gate.event_id
                or result.predicate is not predicate
                or any(frame_id not in allowed_frame_ids for frame_id in result.evidence_frames)
            ):
                raise VisualPredicateShadowError("visual predicate shadow assessment failed")
        except Exception:
            raise VisualPredicateShadowError("visual predicate shadow assessment failed") from None
        results.append(result)

    run = VisualPredicateShadowRun(
        gate=gate,
        results=tuple(results),
        recorded_at=recorded_at or datetime.now(UTC),
    )
    try:
        store = store_factory()
    except Exception:
        raise VisualPredicateShadowError(
            "visual predicate shadow database is unavailable"
        ) from None
    if not isinstance(store, VisualPredicateShadowStore):
        raise VisualPredicateShadowError("visual predicate shadow store is invalid")
    persisted = store.persist(run)
    return VisualPredicateShadowSummary(
        gate=gate,
        results=run.results,
        persisted=True,
        reused_existing=persisted.reused_existing,
    )
