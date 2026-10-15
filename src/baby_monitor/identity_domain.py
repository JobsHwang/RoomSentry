from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class IdentityDomainError(ValueError):
    """A public identity assessment is not strict or internally consistent."""


class FaceQualityState(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class IdentityOutcome(StrEnum):
    MATCHED = "matched"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    QUALITY_REJECTED = "quality_rejected"


class DeclaredKind(StrEnum):
    BABY = "baby"
    ADULT = "adult"
    PERSON = "person"
    UNKNOWN = "unknown"


class IdentityReason(StrEnum):
    THRESHOLD_AND_MARGIN_PASSED = "threshold_and_margin_passed"
    BELOW_MATCH_THRESHOLD = "below_match_threshold"
    INSUFFICIENT_RUNNER_UP_MARGIN = "insufficient_runner_up_margin"
    QUALITY_REJECTED = "quality_rejected"
    SINGLE_PROFILE_UNCALIBRATED = "single_profile_uncalibrated"
    NO_TEMPLATES = "no_templates"
    MISSING_EMBEDDING = "missing_embedding"
    MODEL_FINGERPRINT_MISMATCH = "model_fingerprint_mismatch"
    EMBEDDING_DIMENSION_MISMATCH = "embedding_dimension_mismatch"
    NONFINITE_EMBEDDING = "nonfinite_embedding"
    INVALID_EMBEDDING_NORM = "invalid_embedding_norm"
    INVALID_TEMPLATE_SET = "invalid_template_set"


MAX_IDENTITY_FACES = 1_000
MAX_IDENTITY_JSON_BYTES = 2 * 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SAFE_REASON = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _strict_json(raw: str) -> Any:
    if not isinstance(raw, str):
        raise IdentityDomainError("identity JSON must be text")
    if len(raw.encode("utf-8")) > MAX_IDENTITY_JSON_BYTES:
        raise IdentityDomainError("identity JSON exceeds the size limit")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise IdentityDomainError("identity JSON contains duplicate keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                IdentityDomainError("identity JSON contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise IdentityDomainError("identity JSON is invalid") from exc


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise IdentityDomainError(f"identity {field} is invalid")
    return value


def _fingerprint(value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IdentityDomainError("identity model_fingerprint is invalid")
    return value


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise IdentityDomainError(f"identity {field} is invalid")
    return float(value)


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise IdentityDomainError("identity face bbox must contain four normalized numbers")
    parsed = tuple(_number(item, "face bbox", minimum=0.0, maximum=1.0) for item in value)
    if parsed[0] >= parsed[2] or parsed[1] >= parsed[3]:
        raise IdentityDomainError("identity face bbox must have positive normalized area")
    return parsed


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """Visible face-detector and quality-gate facts, without an identity claim."""

    face_id: str
    frame_ref: str
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float
    quality_state: FaceQualityState
    rejection_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.face_id, "face_id")
        _identifier(self.frame_ref, "frame_ref")
        object.__setattr__(self, "bbox_xyxy", _bbox(self.bbox_xyxy))
        object.__setattr__(
            self,
            "detector_confidence",
            _number(
                self.detector_confidence,
                "detector_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if not isinstance(self.quality_state, FaceQualityState):
            raise IdentityDomainError("identity face quality_state is invalid")
        if not isinstance(self.rejection_codes, tuple) or any(
            not isinstance(code, str) or _SAFE_REASON.fullmatch(code) is None
            for code in self.rejection_codes
        ):
            raise IdentityDomainError("identity face rejection_codes are invalid")
        if len(set(self.rejection_codes)) != len(self.rejection_codes):
            raise IdentityDomainError("identity face rejection_codes must be unique")
        if self.quality_state is FaceQualityState.ACCEPTED and self.rejection_codes:
            raise IdentityDomainError("accepted identity face cannot have rejection codes")
        if self.quality_state is FaceQualityState.REJECTED and not self.rejection_codes:
            raise IdentityDomainError("rejected identity face requires a rejection code")

    @classmethod
    def from_dict(cls, value: Any) -> FaceObservation:
        expected = {
            "face_id",
            "frame_ref",
            "bbox_xyxy",
            "detector_confidence",
            "quality_state",
            "rejection_codes",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise IdentityDomainError("identity face observation fields do not match v1")
        rejection_codes = value["rejection_codes"]
        if not isinstance(rejection_codes, list):
            raise IdentityDomainError("identity face rejection_codes must be an array")
        try:
            quality_state = FaceQualityState(value["quality_state"])
        except (TypeError, ValueError) as exc:
            raise IdentityDomainError("identity face quality_state is invalid") from exc
        return cls(
            face_id=value["face_id"],
            frame_ref=value["frame_ref"],
            bbox_xyxy=_bbox(value["bbox_xyxy"]),
            detector_confidence=value["detector_confidence"],
            quality_state=quality_state,
            rejection_codes=tuple(rejection_codes),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "face_id": self.face_id,
            "frame_ref": self.frame_ref,
            "bbox_xyxy": list(self.bbox_xyxy),
            "detector_confidence": self.detector_confidence,
            "quality_state": self.quality_state.value,
            "rejection_codes": list(self.rejection_codes),
        }


@dataclass(frozen=True, slots=True)
class IdentityInference:
    """A conservative open-set decision for exactly one observed face."""

    face_id: str
    outcome: IdentityOutcome
    profile_id: str | None
    declared_kind: DeclaredKind
    match_score: float | None
    decision_threshold: float
    runner_up_margin: float | None
    runner_up_margin_threshold: float
    reason_code: IdentityReason
    support_frame_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.face_id, "inference face_id")
        if not isinstance(self.outcome, IdentityOutcome):
            raise IdentityDomainError("identity inference outcome is invalid")
        if self.profile_id is not None:
            _identifier(self.profile_id, "profile_id")
        if not isinstance(self.declared_kind, DeclaredKind):
            raise IdentityDomainError("identity inference declared_kind is invalid")
        if self.match_score is not None:
            object.__setattr__(
                self,
                "match_score",
                _number(self.match_score, "match_score", minimum=-1.0, maximum=1.0),
            )
        object.__setattr__(
            self,
            "decision_threshold",
            _number(
                self.decision_threshold,
                "decision_threshold",
                minimum=-1.0,
                maximum=1.0,
            ),
        )
        if self.runner_up_margin is not None:
            object.__setattr__(
                self,
                "runner_up_margin",
                _number(
                    self.runner_up_margin,
                    "runner_up_margin",
                    minimum=0.0,
                    maximum=2.0,
                ),
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
        if not isinstance(self.reason_code, IdentityReason):
            raise IdentityDomainError("identity inference reason_code is invalid")
        if not isinstance(self.support_frame_refs, tuple) or len(self.support_frame_refs) != 1:
            raise IdentityDomainError("identity inference requires exactly one support frame")
        _identifier(self.support_frame_refs[0], "support_frame_ref")

        if self.outcome is IdentityOutcome.MATCHED:
            if (
                self.profile_id is None
                or self.declared_kind is DeclaredKind.UNKNOWN
                or self.match_score is None
                or self.reason_code is not IdentityReason.THRESHOLD_AND_MARGIN_PASSED
            ):
                raise IdentityDomainError("matched identity inference is inconsistent")
            if self.match_score < self.decision_threshold:
                raise IdentityDomainError("matched identity inference is below threshold")
            if (
                self.runner_up_margin is not None
                and self.runner_up_margin < self.runner_up_margin_threshold
            ):
                raise IdentityDomainError("matched identity inference has insufficient margin")
        elif self.outcome is IdentityOutcome.AMBIGUOUS:
            if (
                self.profile_id is not None
                or self.declared_kind is not DeclaredKind.UNKNOWN
                or self.match_score is None
                or self.match_score < self.decision_threshold
                or self.runner_up_margin is None
                or self.runner_up_margin >= self.runner_up_margin_threshold
                or self.reason_code is not IdentityReason.INSUFFICIENT_RUNNER_UP_MARGIN
            ):
                raise IdentityDomainError("ambiguous identity inference is inconsistent")
        elif self.outcome is IdentityOutcome.QUALITY_REJECTED:
            if (
                self.profile_id is not None
                or self.declared_kind is not DeclaredKind.UNKNOWN
                or self.match_score is not None
                or self.runner_up_margin is not None
                or self.reason_code is not IdentityReason.QUALITY_REJECTED
            ):
                raise IdentityDomainError("quality-rejected identity inference is inconsistent")
        else:
            allowed = {
                IdentityReason.BELOW_MATCH_THRESHOLD,
                IdentityReason.NO_TEMPLATES,
                IdentityReason.MISSING_EMBEDDING,
                IdentityReason.MODEL_FINGERPRINT_MISMATCH,
                IdentityReason.EMBEDDING_DIMENSION_MISMATCH,
                IdentityReason.NONFINITE_EMBEDDING,
                IdentityReason.INVALID_EMBEDDING_NORM,
                IdentityReason.INVALID_TEMPLATE_SET,
                IdentityReason.SINGLE_PROFILE_UNCALIBRATED,
            }
            if (
                self.profile_id is not None
                or self.declared_kind is not DeclaredKind.UNKNOWN
                or self.reason_code not in allowed
            ):
                raise IdentityDomainError("unknown identity inference is inconsistent")
            if self.reason_code is IdentityReason.BELOW_MATCH_THRESHOLD:
                if self.match_score is None or self.match_score >= self.decision_threshold:
                    raise IdentityDomainError("below-threshold identity inference is inconsistent")
            elif self.match_score is not None or self.runner_up_margin is not None:
                raise IdentityDomainError("failed-closed identity inference cannot expose a score")

    @classmethod
    def from_dict(cls, value: Any) -> IdentityInference:
        expected = {
            "face_id",
            "outcome",
            "profile_id",
            "declared_kind",
            "match_score",
            "decision_threshold",
            "runner_up_margin",
            "runner_up_margin_threshold",
            "reason_code",
            "support_frame_refs",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise IdentityDomainError("identity inference fields do not match v1")
        frames = value["support_frame_refs"]
        if not isinstance(frames, list):
            raise IdentityDomainError("identity inference support_frame_refs must be an array")
        try:
            outcome = IdentityOutcome(value["outcome"])
            declared_kind = DeclaredKind(value["declared_kind"])
            reason_code = IdentityReason(value["reason_code"])
        except (TypeError, ValueError) as exc:
            raise IdentityDomainError("identity inference enum value is invalid") from exc
        return cls(
            face_id=value["face_id"],
            outcome=outcome,
            profile_id=value["profile_id"],
            declared_kind=declared_kind,
            match_score=value["match_score"],
            decision_threshold=value["decision_threshold"],
            runner_up_margin=value["runner_up_margin"],
            runner_up_margin_threshold=value["runner_up_margin_threshold"],
            reason_code=reason_code,
            support_frame_refs=tuple(frames),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "face_id": self.face_id,
            "outcome": self.outcome.value,
            "profile_id": self.profile_id,
            "declared_kind": self.declared_kind.value,
            "match_score": self.match_score,
            "decision_threshold": self.decision_threshold,
            "runner_up_margin": self.runner_up_margin,
            "runner_up_margin_threshold": self.runner_up_margin_threshold,
            "reason_code": self.reason_code.value,
            "support_frame_refs": list(self.support_frame_refs),
        }


@dataclass(frozen=True, slots=True)
class IdentityAssessment:
    """Path-free, camera-local identity inferences paired with their face facts."""

    assessment_id: str
    event_id: str
    camera_id: str
    observations: tuple[FaceObservation, ...]
    inferences: tuple[IdentityInference, ...]
    model_fingerprint: str
    embedding_dimension: int
    matcher_revision: str
    threshold_revision: str
    match_threshold: float
    runner_up_margin_threshold: float
    cross_camera_linking: bool = False
    tracker_identity_propagation: bool = False
    paths_disclosed: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise IdentityDomainError("unsupported identity assessment version")
        _identifier(self.assessment_id, "assessment_id")
        _identifier(self.event_id, "event_id")
        _identifier(self.camera_id, "camera_id")
        if (
            not isinstance(self.observations, tuple)
            or len(self.observations) > MAX_IDENTITY_FACES
            or any(not isinstance(item, FaceObservation) for item in self.observations)
        ):
            raise IdentityDomainError("identity observations are invalid or too large")
        if (
            not isinstance(self.inferences, tuple)
            or len(self.inferences) != len(self.observations)
            or any(not isinstance(item, IdentityInference) for item in self.inferences)
        ):
            raise IdentityDomainError("identity inferences must pair with observations")
        observation_ids = tuple(item.face_id for item in self.observations)
        if len(set(observation_ids)) != len(observation_ids):
            raise IdentityDomainError("identity observation face IDs must be unique")
        if tuple(item.face_id for item in self.inferences) != observation_ids:
            raise IdentityDomainError("identity inferences must preserve per-face order")
        for observation, inference in zip(self.observations, self.inferences, strict=True):
            if inference.support_frame_refs != (observation.frame_ref,):
                raise IdentityDomainError(
                    "identity inference support frame does not match its face"
                )
            rejected = observation.quality_state is FaceQualityState.REJECTED
            if rejected != (inference.outcome is IdentityOutcome.QUALITY_REJECTED):
                raise IdentityDomainError("identity quality fact and inference do not agree")
        _fingerprint(self.model_fingerprint)
        if type(self.embedding_dimension) is not int or not 1 <= self.embedding_dimension <= 65_536:
            raise IdentityDomainError("identity embedding_dimension is invalid")
        _identifier(self.matcher_revision, "matcher_revision")
        _identifier(self.threshold_revision, "threshold_revision")
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
        if any(
            inference.decision_threshold != self.match_threshold
            or inference.runner_up_margin_threshold != self.runner_up_margin_threshold
            for inference in self.inferences
        ):
            raise IdentityDomainError("identity inference thresholds conflict with assessment")
        for value, field in (
            (self.cross_camera_linking, "cross_camera_linking"),
            (self.tracker_identity_propagation, "tracker_identity_propagation"),
            (self.paths_disclosed, "paths_disclosed"),
        ):
            if type(value) is not bool or value:
                raise IdentityDomainError(f"identity {field} must be false")

    @classmethod
    def from_dict(cls, value: Any) -> IdentityAssessment:
        expected = {
            "version",
            "assessment_id",
            "event_id",
            "camera_id",
            "observations",
            "inferences",
            "model_fingerprint",
            "embedding_dimension",
            "matcher_revision",
            "threshold_revision",
            "match_threshold",
            "runner_up_margin_threshold",
            "cross_camera_linking",
            "tracker_identity_propagation",
            "paths_disclosed",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise IdentityDomainError("identity assessment fields do not match v1")
        observations = value["observations"]
        inferences = value["inferences"]
        if not isinstance(observations, list) or not isinstance(inferences, list):
            raise IdentityDomainError("identity assessment faces must be arrays")
        return cls(
            version=value["version"],
            assessment_id=value["assessment_id"],
            event_id=value["event_id"],
            camera_id=value["camera_id"],
            observations=tuple(FaceObservation.from_dict(item) for item in observations),
            inferences=tuple(IdentityInference.from_dict(item) for item in inferences),
            model_fingerprint=value["model_fingerprint"],
            embedding_dimension=value["embedding_dimension"],
            matcher_revision=value["matcher_revision"],
            threshold_revision=value["threshold_revision"],
            match_threshold=value["match_threshold"],
            runner_up_margin_threshold=value["runner_up_margin_threshold"],
            cross_camera_linking=value["cross_camera_linking"],
            tracker_identity_propagation=value["tracker_identity_propagation"],
            paths_disclosed=value["paths_disclosed"],
        )

    @classmethod
    def from_json(cls, raw: str) -> IdentityAssessment:
        return cls.from_dict(_strict_json(raw))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "assessment_id": self.assessment_id,
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "observations": [item.to_dict() for item in self.observations],
            "inferences": [item.to_dict() for item in self.inferences],
            "model_fingerprint": self.model_fingerprint,
            "embedding_dimension": self.embedding_dimension,
            "matcher_revision": self.matcher_revision,
            "threshold_revision": self.threshold_revision,
            "match_threshold": self.match_threshold,
            "runner_up_margin_threshold": self.runner_up_margin_threshold,
            "cross_camera_linking": False,
            "tracker_identity_propagation": False,
            "paths_disclosed": False,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
