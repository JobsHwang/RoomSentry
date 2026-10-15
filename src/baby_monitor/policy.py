from __future__ import annotations

from dataclasses import dataclass, replace

from .domain import (
    DecisionRecord,
    FinalSource,
    ReviewTier,
    RiskLevel,
    SignalEvent,
    SignalType,
    VLMEvent,
)


@dataclass(frozen=True, slots=True)
class ReviewPolicy:
    confidence_threshold: float = 0.65
    remote_risk_threshold: RiskLevel = RiskLevel.MEDIUM
    signal_frequency_window_seconds: int = 600
    signal_frequency_secondary_threshold: int = 3
    priority_signal_types: tuple[SignalType, ...] = (SignalType.FALL_HINT,)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 10 <= self.signal_frequency_window_seconds <= 86_400:
            raise ValueError("signal frequency window must be between 10 and 86400 seconds")
        if not 2 <= self.signal_frequency_secondary_threshold <= 1_000:
            raise ValueError("signal frequency threshold must be between 2 and 1000")
        if len(set(self.priority_signal_types)) != len(self.priority_signal_types) or any(
            not isinstance(value, SignalType) for value in self.priority_signal_types
        ):
            raise ValueError("priority signal types must be unique SignalType values")

    def decide(
        self,
        result: VLMEvent,
        *,
        signals: tuple[SignalEvent, ...] = (),
        signal_frequency_count: int = 0,
        contradiction_detected: bool = False,
        remote_allowed: bool = False,
        evidence_complete: bool = True,
    ) -> DecisionRecord:
        if type(signal_frequency_count) is not int or signal_frequency_count < 0:
            raise ValueError("signal_frequency_count must be a non-negative integer")
        reasons: list[str] = []
        review_tier = ReviewTier.LOCAL_PRIMARY
        if result.needs_remote_review:
            reasons.append("local_model_requested_review")
        if result.confidence < self.confidence_threshold:
            reasons.append("low_confidence")
        if result.risk_level.rank >= self.remote_risk_threshold.rank:
            reasons.append("elevated_risk")
        if contradiction_detected:
            reasons.append("contradictory_results")
        if reasons:
            review_tier = ReviewTier.REMOTE

        if type(evidence_complete) is not bool:
            raise ValueError("evidence_complete must be a boolean")
        # The legacy v1 boolean only says whether gaps were recorded. True is
        # not a source continuity certificate; a second recognizer cannot fill
        # missing coverage metadata either. Retain this reason after review.
        reasons.append(
            "evidence_coverage_unverified" if evidence_complete else "evidence_coverage_gap"
        )
        if review_tier.rank < ReviewTier.LOCAL_SECONDARY.rank:
            review_tier = ReviewTier.LOCAL_SECONDARY

        signal_types = tuple(sorted({signal.signal_type for signal in signals}, key=str))
        severity_high = any(
            dict(signal.attributes).get("severity_hint") == "high" for signal in signals
        )
        if severity_high or any(
            signal_type in self.priority_signal_types for signal_type in signal_types
        ):
            reasons.append("priority_signal_requires_review")
            if review_tier.rank < ReviewTier.LOCAL_SECONDARY.rank:
                review_tier = ReviewTier.LOCAL_SECONDARY
        frequent_signal_types: tuple[SignalType, ...] = ()
        if signal_types and signal_frequency_count >= self.signal_frequency_secondary_threshold:
            reasons.append("repeated_signal_window")
            frequent_signal_types = signal_types
            if review_tier.rank < ReviewTier.LOCAL_SECONDARY.rank:
                review_tier = ReviewTier.LOCAL_SECONDARY

        return DecisionRecord(
            risk_level=result.risk_level,
            review_tier=review_tier,
            review_reasons=tuple(dict.fromkeys(reasons)),
            contradictions=("recognizer_results_disagree",) if contradiction_detected else (),
            remote_allowed=remote_allowed,
            remote_used=False,
            final_source=FinalSource.LOCAL_PRIMARY,
            signal_frequency_count=signal_frequency_count,
            signal_frequency_window_seconds=(
                self.signal_frequency_window_seconds if signal_frequency_count else 0
            ),
            frequent_signal_types=frequent_signal_types,
        )


def result_contradictions(local: VLMEvent, other: VLMEvent) -> tuple[str, ...]:
    contradictions: list[str] = []
    if local.event_type != other.event_type:
        contradictions.append("event_type_disagreement")
    if local.risk_level != other.risk_level:
        contradictions.append("risk_level_disagreement")
    if abs(local.confidence - other.confidence) >= 0.4:
        contradictions.append("confidence_gap_ge_0.4")
    return tuple(contradictions)


def has_semantic_contradiction(contradictions: tuple[str, ...]) -> bool:
    return bool({"event_type_disagreement", "risk_level_disagreement"} & set(contradictions))


def record_secondary_result(
    decision: DecisionRecord,
    *,
    primary: VLMEvent,
    secondary: VLMEvent,
    final: VLMEvent,
) -> DecisionRecord:
    contradictions = result_contradictions(primary, secondary)
    reasons = decision.review_reasons
    if has_semantic_contradiction(contradictions) and "contradictory_results" not in reasons:
        reasons += ("contradictory_results",)
    review_tier = decision.review_tier
    if review_tier.rank < ReviewTier.LOCAL_SECONDARY.rank:
        review_tier = ReviewTier.LOCAL_SECONDARY
    return replace(
        decision,
        risk_level=final.risk_level,
        review_tier=review_tier,
        review_reasons=reasons,
        contradictions=tuple(dict.fromkeys(decision.contradictions + contradictions)),
        remote_used=False,
        final_source=FinalSource.LOCAL_SECONDARY,
    )


def record_remote_result(
    decision: DecisionRecord,
    *,
    local: VLMEvent,
    remote: VLMEvent,
    final: VLMEvent,
) -> DecisionRecord:
    contradictions = result_contradictions(local, remote)
    reasons = decision.review_reasons
    if has_semantic_contradiction(contradictions) and "contradictory_results" not in reasons:
        reasons += ("contradictory_results",)
    return replace(
        decision,
        risk_level=final.risk_level,
        review_tier=ReviewTier.REMOTE,
        review_reasons=reasons,
        contradictions=tuple(dict.fromkeys(decision.contradictions + contradictions)),
        remote_allowed=True,
        remote_used=True,
        final_source=FinalSource.REMOTE,
    )


def reconcile_remote(local: VLMEvent, remote: VLMEvent) -> VLMEvent:
    """Prefer remote output; DecisionRecord separately preserves any disagreement."""
    return remote
