from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from baby_monitor.adapters.visual_predicate import (
    ResidentVisualPredicateHealth,
    VisualPredicateHealthError,
)


class ResidentVisualPredicateFreshness(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    CLOCK_SKEW = "clock_skew"


def _aware(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise VisualPredicateHealthError(f"resident health {field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _bounded_seconds(
    value: object,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or not minimum <= float(value) <= maximum
    ):
        raise VisualPredicateHealthError(f"resident health {field_name} is invalid")
    return float(value)


@dataclass(frozen=True, slots=True)
class ResidentVisualPredicateHealthAssessment:
    checked_at: datetime
    health_checked_at: datetime
    reported_state: str
    worker_running: bool
    freshness: ResidentVisualPredicateFreshness
    age_seconds: float
    stale_after_seconds: float
    future_tolerance_seconds: float
    operational: bool
    reason_code: str
    model_id: str
    model_revision: str
    prompt_version: str
    successful_jobs: int
    failed_jobs: int
    timeout_jobs: int
    current_worker_jobs: int
    version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "checked_at": self.checked_at.astimezone(UTC).isoformat(),
            "health_checked_at": self.health_checked_at.astimezone(UTC).isoformat(),
            "reported_state": self.reported_state,
            "worker_running": self.worker_running,
            "freshness": self.freshness.value,
            "age_seconds": self.age_seconds,
            "stale_after_seconds": self.stale_after_seconds,
            "future_tolerance_seconds": self.future_tolerance_seconds,
            "operational": self.operational,
            "reason_code": self.reason_code,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "successful_jobs": self.successful_jobs,
            "failed_jobs": self.failed_jobs,
            "timeout_jobs": self.timeout_jobs,
            "current_worker_jobs": self.current_worker_jobs,
            "paths_disclosed": False,
            "media_metadata_included": False,
        }


def assess_resident_visual_predicate_health(
    health: ResidentVisualPredicateHealth,
    *,
    checked_at: datetime | None = None,
    stale_after_seconds: float = 60.0,
    future_tolerance_seconds: float = 5.0,
) -> ResidentVisualPredicateHealthAssessment:
    if not isinstance(health, ResidentVisualPredicateHealth):
        raise VisualPredicateHealthError("resident health snapshot is invalid")
    checked = _aware(checked_at or datetime.now(UTC), "checked_at")
    stale_after = _bounded_seconds(
        stale_after_seconds,
        "stale_after_seconds",
        minimum=1.0,
        maximum=604_800.0,
    )
    future_tolerance = _bounded_seconds(
        future_tolerance_seconds,
        "future_tolerance_seconds",
        minimum=0.0,
        maximum=3_600.0,
    )
    observed = health.checked_at.astimezone(UTC)
    age = (checked - observed).total_seconds()

    if age < -future_tolerance:
        freshness = ResidentVisualPredicateFreshness.CLOCK_SKEW
        reason = "health_checked_at_in_future"
        operational = False
    elif age > stale_after:
        freshness = ResidentVisualPredicateFreshness.STALE
        reason = "resident_health_stale"
        operational = False
    else:
        freshness = ResidentVisualPredicateFreshness.FRESH
        reason = f"fresh_{health.state}"
        operational = health.worker_running and health.state in {"ready", "busy"}

    return ResidentVisualPredicateHealthAssessment(
        checked_at=checked,
        health_checked_at=observed,
        reported_state=health.state,
        worker_running=health.worker_running,
        freshness=freshness,
        age_seconds=round(age, 3),
        stale_after_seconds=stale_after,
        future_tolerance_seconds=future_tolerance,
        operational=operational,
        reason_code=reason,
        model_id=health.model_id,
        model_revision=health.model_revision,
        prompt_version=health.prompt_version,
        successful_jobs=health.successful_jobs,
        failed_jobs=health.failed_jobs,
        timeout_jobs=health.timeout_jobs,
        current_worker_jobs=health.current_worker_jobs,
    )
