from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .identity_domain import (
    DeclaredKind,
    FaceObservation,
    FaceQualityState,
    IdentityAssessment,
    IdentityInference,
    IdentityOutcome,
    IdentityReason,
)


class IdentityMatchingError(ValueError):
    """The caller supplied an invalid matcher request or policy."""


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise IdentityMatchingError(f"identity matcher {field} is invalid")
    return value


def _fingerprint(value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IdentityMatchingError("identity matcher model_fingerprint is invalid")
    return value


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise IdentityMatchingError(f"identity matcher {field} is invalid")
    return float(value)


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    """Private, in-memory embedding input for one face observation."""

    face_id: str
    model_fingerprint: str
    embedding: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _identifier(self.face_id, "face_id")
        _fingerprint(self.model_fingerprint)
        if not isinstance(self.embedding, tuple):
            raise IdentityMatchingError("identity matcher embedding must be immutable")


@dataclass(frozen=True, slots=True)
class IdentityTemplate:
    """Private, in-memory template; profile_id must be an opaque local identifier."""

    template_id: str
    profile_id: str
    declared_kind: DeclaredKind
    model_fingerprint: str
    embedding: tuple[float, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _identifier(self.template_id, "template_id")
        _identifier(self.profile_id, "profile_id")
        if (
            not isinstance(self.declared_kind, DeclaredKind)
            or self.declared_kind is DeclaredKind.UNKNOWN
        ):
            raise IdentityMatchingError("identity template declared_kind is invalid")
        _fingerprint(self.model_fingerprint)
        if not isinstance(self.embedding, tuple):
            raise IdentityMatchingError("identity template embedding must be immutable")


@dataclass(frozen=True, slots=True)
class IdentityMatchPolicy:
    model_fingerprint: str
    embedding_dimension: int
    match_threshold: float
    runner_up_margin_threshold: float
    matcher_revision: str
    threshold_revision: str

    def __post_init__(self) -> None:
        _fingerprint(self.model_fingerprint)
        if type(self.embedding_dimension) is not int or not 1 <= self.embedding_dimension <= 65_536:
            raise IdentityMatchingError("identity matcher embedding_dimension is invalid")
        object.__setattr__(
            self,
            "match_threshold",
            _number(self.match_threshold, "match_threshold", minimum=-1.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "runner_up_margin_threshold",
            _number(
                self.runner_up_margin_threshold,
                "runner_up_margin_threshold",
                minimum=0.0,
                maximum=2.0,
            ),
        )
        _identifier(self.matcher_revision, "matcher_revision")
        _identifier(self.threshold_revision, "threshold_revision")


def _vector_problem(
    value: tuple[float, ...],
    expected_dimension: int,
) -> IdentityReason | None:
    if len(value) != expected_dimension:
        return IdentityReason.EMBEDDING_DIMENSION_MISMATCH
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            return IdentityReason.NONFINITE_EMBEDDING
    norm = math.hypot(*(float(item) for item in value))
    if not math.isfinite(norm) or norm == 0.0:
        return IdentityReason.INVALID_EMBEDDING_NORM
    return None


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = math.hypot(*(float(item) for item in left))
    right_norm = math.hypot(*(float(item) for item in right))
    score = math.fsum(
        (float(a) / left_norm) * (float(b) / right_norm) for a, b in zip(left, right, strict=True)
    )
    return min(1.0, max(-1.0, score))


def _scoreless_inference(
    observation: FaceObservation,
    policy: IdentityMatchPolicy,
    reason: IdentityReason,
    *,
    outcome: IdentityOutcome = IdentityOutcome.UNKNOWN,
) -> IdentityInference:
    return IdentityInference(
        face_id=observation.face_id,
        outcome=outcome,
        profile_id=None,
        declared_kind=DeclaredKind.UNKNOWN,
        match_score=None,
        decision_threshold=policy.match_threshold,
        runner_up_margin=None,
        runner_up_margin_threshold=policy.runner_up_margin_threshold,
        reason_code=reason,
        support_frame_refs=(observation.frame_ref,),
    )


def _match_one(
    observation: FaceObservation,
    embedding: FaceEmbedding | None,
    templates: tuple[IdentityTemplate, ...],
    policy: IdentityMatchPolicy,
) -> IdentityInference:
    if observation.quality_state is FaceQualityState.REJECTED:
        return _scoreless_inference(
            observation,
            policy,
            IdentityReason.QUALITY_REJECTED,
            outcome=IdentityOutcome.QUALITY_REJECTED,
        )
    if embedding is None:
        return _scoreless_inference(
            observation,
            policy,
            IdentityReason.MISSING_EMBEDDING,
        )
    if embedding.model_fingerprint != policy.model_fingerprint:
        return _scoreless_inference(
            observation,
            policy,
            IdentityReason.MODEL_FINGERPRINT_MISMATCH,
        )
    problem = _vector_problem(embedding.embedding, policy.embedding_dimension)
    if problem is not None:
        return _scoreless_inference(observation, policy, problem)
    if not templates:
        return _scoreless_inference(observation, policy, IdentityReason.NO_TEMPLATES)
    if any(item.model_fingerprint != policy.model_fingerprint for item in templates):
        return _scoreless_inference(
            observation,
            policy,
            IdentityReason.MODEL_FINGERPRINT_MISMATCH,
        )
    if len({item.template_id for item in templates}) != len(templates):
        return _scoreless_inference(
            observation,
            policy,
            IdentityReason.INVALID_TEMPLATE_SET,
        )

    profile_kinds: dict[str, DeclaredKind] = {}
    for template in templates:
        prior_kind = profile_kinds.setdefault(template.profile_id, template.declared_kind)
        if prior_kind is not template.declared_kind:
            return _scoreless_inference(
                observation,
                policy,
                IdentityReason.INVALID_TEMPLATE_SET,
            )
        problem = _vector_problem(template.embedding, policy.embedding_dimension)
        if problem is not None:
            return _scoreless_inference(observation, policy, problem)

    if len(profile_kinds) < 2:
        return _scoreless_inference(
            observation,
            policy,
            IdentityReason.SINGLE_PROFILE_UNCALIBRATED,
        )

    scores: dict[str, float] = {}
    for template in templates:
        score = _cosine(embedding.embedding, template.embedding)
        scores[template.profile_id] = max(score, scores.get(template.profile_id, -1.0))
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_profile_id, top_score = ranked[0]
    margin = top_score - ranked[1][1] if len(ranked) > 1 else None
    if top_score < policy.match_threshold:
        return IdentityInference(
            face_id=observation.face_id,
            outcome=IdentityOutcome.UNKNOWN,
            profile_id=None,
            declared_kind=DeclaredKind.UNKNOWN,
            match_score=top_score,
            decision_threshold=policy.match_threshold,
            runner_up_margin=margin,
            runner_up_margin_threshold=policy.runner_up_margin_threshold,
            reason_code=IdentityReason.BELOW_MATCH_THRESHOLD,
            support_frame_refs=(observation.frame_ref,),
        )
    if margin is not None and margin < policy.runner_up_margin_threshold:
        return IdentityInference(
            face_id=observation.face_id,
            outcome=IdentityOutcome.AMBIGUOUS,
            profile_id=None,
            declared_kind=DeclaredKind.UNKNOWN,
            match_score=top_score,
            decision_threshold=policy.match_threshold,
            runner_up_margin=margin,
            runner_up_margin_threshold=policy.runner_up_margin_threshold,
            reason_code=IdentityReason.INSUFFICIENT_RUNNER_UP_MARGIN,
            support_frame_refs=(observation.frame_ref,),
        )
    return IdentityInference(
        face_id=observation.face_id,
        outcome=IdentityOutcome.MATCHED,
        profile_id=top_profile_id,
        declared_kind=profile_kinds[top_profile_id],
        match_score=top_score,
        decision_threshold=policy.match_threshold,
        runner_up_margin=margin,
        runner_up_margin_threshold=policy.runner_up_margin_threshold,
        reason_code=IdentityReason.THRESHOLD_AND_MARGIN_PASSED,
        support_frame_refs=(observation.frame_ref,),
    )


def assess_identities(
    *,
    assessment_id: str,
    event_id: str,
    camera_id: str,
    observations: tuple[FaceObservation, ...],
    embeddings: tuple[FaceEmbedding, ...],
    templates: tuple[IdentityTemplate, ...],
    policy: IdentityMatchPolicy,
) -> IdentityAssessment:
    """Match each face independently; never aggregate or propagate through a tracker."""

    _identifier(assessment_id, "assessment_id")
    _identifier(event_id, "event_id")
    _identifier(camera_id, "camera_id")
    if not isinstance(policy, IdentityMatchPolicy):
        raise IdentityMatchingError("identity matcher policy is invalid")
    if not isinstance(observations, tuple) or any(
        not isinstance(item, FaceObservation) for item in observations
    ):
        raise IdentityMatchingError("identity matcher observations are invalid")
    if not isinstance(embeddings, tuple) or any(
        not isinstance(item, FaceEmbedding) for item in embeddings
    ):
        raise IdentityMatchingError("identity matcher embeddings are invalid")
    if not isinstance(templates, tuple) or any(
        not isinstance(item, IdentityTemplate) for item in templates
    ):
        raise IdentityMatchingError("identity matcher templates are invalid")
    observation_ids = tuple(item.face_id for item in observations)
    if len(set(observation_ids)) != len(observation_ids):
        raise IdentityMatchingError("identity matcher face IDs must be unique")
    embedding_ids = tuple(item.face_id for item in embeddings)
    if len(set(embedding_ids)) != len(embedding_ids):
        raise IdentityMatchingError("identity matcher embedding face IDs must be unique")
    if not set(embedding_ids) <= set(observation_ids):
        raise IdentityMatchingError("identity matcher embedding has no face observation")
    embedding_by_face = {item.face_id: item for item in embeddings}
    inferences = tuple(
        _match_one(
            observation,
            embedding_by_face.get(observation.face_id),
            templates,
            policy,
        )
        for observation in observations
    )
    return IdentityAssessment(
        assessment_id=assessment_id,
        event_id=event_id,
        camera_id=camera_id,
        observations=observations,
        inferences=inferences,
        model_fingerprint=policy.model_fingerprint,
        embedding_dimension=policy.embedding_dimension,
        matcher_revision=policy.matcher_revision,
        threshold_revision=policy.threshold_revision,
        match_threshold=policy.match_threshold,
        runner_up_margin_threshold=policy.runner_up_margin_threshold,
    )
