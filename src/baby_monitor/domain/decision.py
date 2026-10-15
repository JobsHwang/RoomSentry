"""DecisionRecord v1: risk level and review tier decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from baby_monitor.domain._shared import (
    SchemaError,
    _reject_json_constant,
    _require_exact_keys,
    _string_list,
)
from baby_monitor.domain.enums import (
    FinalSource,
    ReviewTier,
    RiskLevel,
    SignalType,
)

_SAFE_DECISION_CODE = re.compile(r"[a-z0-9_.:-]{1,120}")


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    risk_level: RiskLevel
    review_tier: ReviewTier
    review_reasons: tuple[str, ...]
    contradictions: tuple[str, ...]
    remote_allowed: bool
    remote_used: bool
    final_source: FinalSource
    signal_frequency_count: int = 0
    signal_frequency_window_seconds: int = 0
    frequent_signal_types: tuple[SignalType, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise SchemaError("unsupported decision record version")
        if not isinstance(self.risk_level, RiskLevel):
            raise SchemaError("decision risk_level is invalid")
        if not isinstance(self.review_tier, ReviewTier):
            raise SchemaError("decision review_tier is invalid")
        if not isinstance(self.final_source, FinalSource):
            raise SchemaError("decision final_source is invalid")
        if type(self.remote_allowed) is not bool or type(self.remote_used) is not bool:
            raise SchemaError("decision remote flags must be booleans")
        if self.remote_used and not self.remote_allowed:
            raise SchemaError("decision cannot use remote review when it is not allowed")
        if self.remote_used != (self.final_source is FinalSource.REMOTE):
            raise SchemaError("decision remote_used and final_source must agree")
        for values, field in (
            (self.review_reasons, "review_reasons"),
            (self.contradictions, "contradictions"),
        ):
            if not isinstance(values, tuple) or len(set(values)) != len(values):
                raise SchemaError(f"decision {field} must be an immutable unique list")
            if any(
                not isinstance(value, str) or _SAFE_DECISION_CODE.fullmatch(value) is None
                for value in values
            ):
                raise SchemaError(f"decision {field} contains an invalid code")
        if (
            type(self.signal_frequency_count) is not int
            or not 0 <= self.signal_frequency_count <= 100_000
        ):
            raise SchemaError("decision signal_frequency_count is invalid")
        if (
            type(self.signal_frequency_window_seconds) is not int
            or not 0 <= self.signal_frequency_window_seconds <= 86_400
        ):
            raise SchemaError("decision signal_frequency_window_seconds is invalid")
        if self.signal_frequency_count == 0 and self.signal_frequency_window_seconds != 0:
            raise SchemaError("decision frequency window requires a signal count")
        if self.signal_frequency_count > 0 and self.signal_frequency_window_seconds == 0:
            raise SchemaError("decision signal count requires a frequency window")
        if not isinstance(self.frequent_signal_types, tuple) or len(
            set(self.frequent_signal_types)
        ) != len(self.frequent_signal_types):
            raise SchemaError("decision frequent_signal_types must be immutable and unique")
        if any(not isinstance(value, SignalType) for value in self.frequent_signal_types):
            raise SchemaError("decision frequent_signal_types contains an invalid type")
        if self.frequent_signal_types and self.signal_frequency_count == 0:
            raise SchemaError("decision frequent signal types require a signal count")

    @property
    def needs_remote_review(self) -> bool:
        return self.review_tier is ReviewTier.REMOTE and not self.remote_used

    @property
    def needs_local_review(self) -> bool:
        return (
            self.review_tier is ReviewTier.LOCAL_SECONDARY
            and self.final_source is FinalSource.LOCAL_PRIMARY
        )

    @property
    def needs_review(self) -> bool:
        """Outstanding review, without conflating local work and upload consent."""
        return self.needs_local_review or self.needs_remote_review

    @classmethod
    def from_dict(cls, data: Any) -> DecisionRecord:
        expected = {
            "version",
            "risk_level",
            "review_tier",
            "review_reasons",
            "contradictions",
            "remote_allowed",
            "remote_used",
            "final_source",
            "signal_frequency_count",
            "signal_frequency_window_seconds",
            "frequent_signal_types",
        }
        if not isinstance(data, dict):
            raise SchemaError("decision record must be an object")
        _require_exact_keys(data, expected, "decision record")
        try:
            risk_level = RiskLevel(data["risk_level"])
            review_tier = ReviewTier(data["review_tier"])
            final_source = FinalSource(data["final_source"])
            frequent_signal_types = tuple(
                SignalType(value)
                for value in _string_list(
                    data["frequent_signal_types"],
                    "decision.frequent_signal_types",
                    unique=True,
                )
            )
        except (TypeError, ValueError) as exc:
            raise SchemaError("decision record contains an invalid enum") from exc
        if type(data["remote_allowed"]) is not bool or type(data["remote_used"]) is not bool:
            raise SchemaError("decision remote flags must be booleans")
        return cls(
            version=data["version"],
            risk_level=risk_level,
            review_tier=review_tier,
            review_reasons=_string_list(
                data["review_reasons"], "decision.review_reasons", unique=True
            ),
            contradictions=_string_list(
                data["contradictions"], "decision.contradictions", unique=True
            ),
            remote_allowed=data["remote_allowed"],
            remote_used=data["remote_used"],
            final_source=final_source,
            signal_frequency_count=data["signal_frequency_count"],
            signal_frequency_window_seconds=data["signal_frequency_window_seconds"],
            frequent_signal_types=frequent_signal_types,
        )

    @classmethod
    def from_json(cls, payload: str) -> DecisionRecord:
        try:
            data = json.loads(payload, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SchemaError("decision record is not valid strict JSON") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "risk_level": self.risk_level.value,
            "review_tier": self.review_tier.value,
            "review_reasons": list(self.review_reasons),
            "contradictions": list(self.contradictions),
            "remote_allowed": self.remote_allowed,
            "remote_used": self.remote_used,
            "final_source": self.final_source.value,
            "signal_frequency_count": self.signal_frequency_count,
            "signal_frequency_window_seconds": self.signal_frequency_window_seconds,
            "frequent_signal_types": [value.value for value in self.frequent_signal_types],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)
