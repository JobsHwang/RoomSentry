from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from .visual_predicate import (
    VisualPredicate,
    VisualPredicateAnswer,
    VisualPredicateResult,
)
from .visual_predicate_shadow_queue import (
    VisualPredicateShadowAttemptStatus,
    VisualPredicateShadowObservation,
)

VISUAL_PREDICATE_REVIEW_POLICY_VERSION = "atomic_visual_review_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class VisualPredicateReviewError(ValueError):
    """Path-free failure for the shadow-only deterministic review policy."""


class VisualPredicateReviewState(StrEnum):
    REVIEW_RECOMMENDED = "review_recommended"
    NO_RULE_MATCH = "no_rule_match"
    UNKNOWN = "unknown"


class VisualPredicateReviewReason(StrEnum):
    FACE_NOT_VISIBLE = "face_not_visible"
    FACE_PARTIALLY_OBSCURED = "face_partially_obscured"
    COVERING_NEAR_FACE_VISIBLE = "covering_near_face_visible"
    FACE_VISIBILITY_UNKNOWN = "face_visibility_unknown"
    COVERING_NEAR_FACE_UNKNOWN = "covering_near_face_unknown"
    REQUIRED_FACE_VISIBILITY_MISSING = "required_face_visibility_missing"
    REQUIRED_COVERING_NEAR_FACE_MISSING = "required_covering_near_face_missing"


_REVIEW_REASONS = frozenset(
    {
        VisualPredicateReviewReason.FACE_NOT_VISIBLE,
        VisualPredicateReviewReason.FACE_PARTIALLY_OBSCURED,
        VisualPredicateReviewReason.COVERING_NEAR_FACE_VISIBLE,
    }
)
_UNKNOWN_REASONS = frozenset(
    {
        VisualPredicateReviewReason.FACE_VISIBILITY_UNKNOWN,
        VisualPredicateReviewReason.COVERING_NEAR_FACE_UNKNOWN,
        VisualPredicateReviewReason.REQUIRED_FACE_VISIBILITY_MISSING,
        VisualPredicateReviewReason.REQUIRED_COVERING_NEAR_FACE_MISSING,
    }
)


@dataclass(frozen=True, slots=True)
class VisualPredicateReviewRecommendation:
    event_id: str
    source_binding_sha256: str
    state: VisualPredicateReviewState
    reasons: tuple[VisualPredicateReviewReason, ...]
    manual_review_recommended: bool
    version: int = 1
    policy_version: str = VISUAL_PREDICATE_REVIEW_POLICY_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise VisualPredicateReviewError("unsupported visual review recommendation version")
        if self.policy_version != VISUAL_PREDICATE_REVIEW_POLICY_VERSION:
            raise VisualPredicateReviewError("unsupported visual review policy version")
        if not isinstance(self.event_id, str) or _SAFE_ID.fullmatch(self.event_id) is None:
            raise VisualPredicateReviewError("visual review event binding is invalid")
        if (
            not isinstance(self.source_binding_sha256, str)
            or _SHA256.fullmatch(self.source_binding_sha256) is None
        ):
            raise VisualPredicateReviewError("visual review source binding is invalid")
        if not isinstance(self.state, VisualPredicateReviewState):
            raise VisualPredicateReviewError("visual review state is invalid")
        if (
            not isinstance(self.reasons, tuple)
            or len(set(self.reasons)) != len(self.reasons)
            or any(not isinstance(item, VisualPredicateReviewReason) for item in self.reasons)
        ):
            raise VisualPredicateReviewError("visual review reasons are invalid")
        if type(self.manual_review_recommended) is not bool:
            raise VisualPredicateReviewError("visual review recommendation flag is invalid")
        has_review_reason = any(item in _REVIEW_REASONS for item in self.reasons)
        has_unknown_reason = any(item in _UNKNOWN_REASONS for item in self.reasons)
        expected_state = (
            VisualPredicateReviewState.REVIEW_RECOMMENDED
            if has_review_reason
            else VisualPredicateReviewState.UNKNOWN
            if has_unknown_reason
            else VisualPredicateReviewState.NO_RULE_MATCH
        )
        if self.state is not expected_state or self.manual_review_recommended != (
            expected_state is not VisualPredicateReviewState.NO_RULE_MATCH
        ):
            raise VisualPredicateReviewError("visual review recommendation is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "policy_version": self.policy_version,
            "event_id": self.event_id,
            "source_binding_sha256": self.source_binding_sha256,
            "state": self.state.value,
            "reasons": [item.value for item in self.reasons],
            "manual_review_recommended": self.manual_review_recommended,
            "model_confidence_calibrated": False,
            "confidence_threshold_applied": False,
            "media_read": False,
            "paths_disclosed": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "risk_level_assigned": False,
            "notification_decision": False,
            "notification_enqueued": False,
        }


def _results_by_predicate(
    observation: VisualPredicateShadowObservation,
) -> dict[VisualPredicate, VisualPredicateResult]:
    attempt = observation.attempt
    if (
        attempt.status is not VisualPredicateShadowAttemptStatus.COMPLETED
        or attempt.result_binding_sha256 is None
        or not observation.results
        or tuple(item.predicate for item in observation.results) != attempt.gate.predicates
        or any(item.event_id != attempt.event_id for item in observation.results)
    ):
        raise VisualPredicateReviewError(
            "visual review requires a bound completed shadow observation"
        )
    return {item.predicate: item for item in observation.results}


def assess_visual_predicate_review(
    observation: VisualPredicateShadowObservation,
) -> VisualPredicateReviewRecommendation:
    if not isinstance(observation, VisualPredicateShadowObservation):
        raise VisualPredicateReviewError("visual review observation is invalid")
    by_predicate = _results_by_predicate(observation)
    reasons: list[VisualPredicateReviewReason] = []

    face = by_predicate.get(VisualPredicate.FACE_VISIBILITY)
    if face is None:
        reasons.append(VisualPredicateReviewReason.REQUIRED_FACE_VISIBILITY_MISSING)
    elif face.answer is VisualPredicateAnswer.NOT_VISIBLE:
        reasons.append(VisualPredicateReviewReason.FACE_NOT_VISIBLE)
    elif face.answer is VisualPredicateAnswer.PARTIALLY_OBSCURED:
        reasons.append(VisualPredicateReviewReason.FACE_PARTIALLY_OBSCURED)
    elif face.answer is VisualPredicateAnswer.UNKNOWN:
        reasons.append(VisualPredicateReviewReason.FACE_VISIBILITY_UNKNOWN)

    covering = by_predicate.get(VisualPredicate.COVERING_NEAR_FACE_VISIBLE)
    if covering is None:
        reasons.append(VisualPredicateReviewReason.REQUIRED_COVERING_NEAR_FACE_MISSING)
    elif covering.answer is VisualPredicateAnswer.YES:
        reasons.append(VisualPredicateReviewReason.COVERING_NEAR_FACE_VISIBLE)
    elif covering.answer is VisualPredicateAnswer.UNKNOWN:
        reasons.append(VisualPredicateReviewReason.COVERING_NEAR_FACE_UNKNOWN)

    has_review_reason = any(item in _REVIEW_REASONS for item in reasons)
    has_unknown_reason = any(item in _UNKNOWN_REASONS for item in reasons)
    state = (
        VisualPredicateReviewState.REVIEW_RECOMMENDED
        if has_review_reason
        else VisualPredicateReviewState.UNKNOWN
        if has_unknown_reason
        else VisualPredicateReviewState.NO_RULE_MATCH
    )
    source_binding = observation.attempt.result_binding_sha256
    if source_binding is None:
        raise VisualPredicateReviewError("visual review source binding is unavailable")
    return VisualPredicateReviewRecommendation(
        event_id=observation.attempt.event_id,
        source_binding_sha256=source_binding,
        state=state,
        reasons=tuple(reasons),
        manual_review_recommended=(state is not VisualPredicateReviewState.NO_RULE_MATCH),
    )


@dataclass(frozen=True, slots=True)
class VisualPredicateReviewReport:
    observed_attempt_count: int
    assessments: tuple[VisualPredicateReviewRecommendation, ...]
    unassessed_counts: tuple[tuple[str, int], ...]
    version: int = 1
    policy_version: str = VISUAL_PREDICATE_REVIEW_POLICY_VERSION

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise VisualPredicateReviewError("unsupported visual review report version")
        if self.policy_version != VISUAL_PREDICATE_REVIEW_POLICY_VERSION:
            raise VisualPredicateReviewError("unsupported visual review policy version")
        if type(self.observed_attempt_count) is not int or self.observed_attempt_count < 0:
            raise VisualPredicateReviewError("visual review report count is invalid")
        if not isinstance(self.assessments, tuple) or any(
            not isinstance(item, VisualPredicateReviewRecommendation) for item in self.assessments
        ):
            raise VisualPredicateReviewError("visual review assessments are invalid")
        expected_statuses = tuple(
            item.value
            for item in VisualPredicateShadowAttemptStatus
            if item is not VisualPredicateShadowAttemptStatus.COMPLETED
        )
        if (
            not isinstance(self.unassessed_counts, tuple)
            or tuple(item[0] for item in self.unassessed_counts) != expected_statuses
            or any(type(item[1]) is not int or item[1] < 0 for item in self.unassessed_counts)
            or self.observed_attempt_count
            != len(self.assessments) + sum(item[1] for item in self.unassessed_counts)
        ):
            raise VisualPredicateReviewError("visual review unassessed counts are invalid")

    def to_dict(self) -> dict[str, object]:
        counts = Counter(item.state.value for item in self.assessments)
        return {
            "version": self.version,
            "policy_version": self.policy_version,
            "observed_attempt_count": self.observed_attempt_count,
            "assessed_completed_count": len(self.assessments),
            "review_recommended_count": counts[VisualPredicateReviewState.REVIEW_RECOMMENDED.value],
            "unknown_count": counts[VisualPredicateReviewState.UNKNOWN.value],
            "no_rule_match_count": counts[VisualPredicateReviewState.NO_RULE_MATCH.value],
            "unassessed_counts": dict(self.unassessed_counts),
            "assessments": [item.to_dict() for item in self.assessments],
            "model_confidence_calibrated": False,
            "media_read": False,
            "paths_disclosed": False,
            "subject_identity_inference": False,
            "risk_inference": False,
            "risk_level_assigned": False,
            "notification_decision": False,
            "notification_enqueued": False,
        }


def build_visual_predicate_review_report(
    observations: tuple[VisualPredicateShadowObservation, ...],
) -> VisualPredicateReviewReport:
    if (
        not isinstance(observations, tuple)
        or len(observations) > 1_000
        or any(not isinstance(item, VisualPredicateShadowObservation) for item in observations)
    ):
        raise VisualPredicateReviewError("visual review observations are invalid")
    assessments: list[VisualPredicateReviewRecommendation] = []
    unassessed = Counter[str]()
    for observation in observations:
        if observation.attempt.status is VisualPredicateShadowAttemptStatus.COMPLETED:
            assessments.append(assess_visual_predicate_review(observation))
        else:
            unassessed[observation.attempt.status.value] += 1
    ordered_unassessed = tuple(
        (status.value, unassessed[status.value])
        for status in VisualPredicateShadowAttemptStatus
        if status is not VisualPredicateShadowAttemptStatus.COMPLETED
    )
    return VisualPredicateReviewReport(
        observed_attempt_count=len(observations),
        assessments=tuple(assessments),
        unassessed_counts=ordered_unassessed,
    )
