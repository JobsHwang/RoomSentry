from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .domain import SchemaError

VISUAL_PREDICATE_VERSION = 1
MAX_VISUAL_PREDICATE_BYTES = 64 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")


class VisualPredicate(StrEnum):
    FACE_VISIBILITY = "face_visibility"
    SUPPORT_CONTACT_VISIBLE = "support_contact_visible"
    POSTURE_TRANSITION_VISIBLE = "posture_transition_visible"
    COVERING_NEAR_FACE_VISIBLE = "covering_near_face_visible"


class VisualPredicateAnswer(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"
    CLEAR = "clear"
    PARTIALLY_OBSCURED = "partially_obscured"
    NOT_VISIBLE = "not_visible"


_ANSWERS = {
    VisualPredicate.FACE_VISIBILITY: frozenset(
        {
            VisualPredicateAnswer.CLEAR,
            VisualPredicateAnswer.PARTIALLY_OBSCURED,
            VisualPredicateAnswer.NOT_VISIBLE,
            VisualPredicateAnswer.UNKNOWN,
        }
    ),
    VisualPredicate.SUPPORT_CONTACT_VISIBLE: frozenset(
        {
            VisualPredicateAnswer.YES,
            VisualPredicateAnswer.NO,
            VisualPredicateAnswer.UNKNOWN,
        }
    ),
    VisualPredicate.POSTURE_TRANSITION_VISIBLE: frozenset(
        {
            VisualPredicateAnswer.YES,
            VisualPredicateAnswer.NO,
            VisualPredicateAnswer.UNKNOWN,
        }
    ),
    VisualPredicate.COVERING_NEAR_FACE_VISIBLE: frozenset(
        {
            VisualPredicateAnswer.YES,
            VisualPredicateAnswer.NO,
            VisualPredicateAnswer.UNKNOWN,
        }
    ),
}


def _strict_json(raw: str) -> Any:
    if not isinstance(raw, str) or not 0 < len(raw.encode("utf-8")) <= MAX_VISUAL_PREDICATE_BYTES:
        raise SchemaError("visual predicate result exceeds the size limit")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise SchemaError("visual predicate result contains duplicate JSON keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                SchemaError("visual predicate result contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise SchemaError("visual predicate result is not valid JSON") from exc


def _id(value: object, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise SchemaError(f"visual predicate {field} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class VisualPredicateResult:
    event_id: str
    predicate: VisualPredicate
    answer: VisualPredicateAnswer
    confidence: float
    evidence_frames: tuple[str, ...]
    model_id: str
    model_revision: str
    prompt_version: str
    version: int = VISUAL_PREDICATE_VERSION
    subject_identity_inference: bool = False
    risk_inference: bool = False
    notification_decision: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != VISUAL_PREDICATE_VERSION:
            raise SchemaError("unsupported visual predicate result version")
        _id(self.event_id, "event_id")
        if not isinstance(self.predicate, VisualPredicate):
            raise SchemaError("visual predicate name is invalid")
        if not isinstance(self.answer, VisualPredicateAnswer):
            raise SchemaError("visual predicate answer is invalid")
        if self.answer not in _ANSWERS[self.predicate]:
            raise SchemaError("visual predicate answer does not match the predicate")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise SchemaError("visual predicate confidence is invalid")
        object.__setattr__(self, "confidence", float(self.confidence))
        if (
            not isinstance(self.evidence_frames, tuple)
            or len(self.evidence_frames) > 12
            or len(set(self.evidence_frames)) != len(self.evidence_frames)
            or any(_SAFE_ID.fullmatch(item) is None for item in self.evidence_frames)
        ):
            raise SchemaError("visual predicate evidence_frames are invalid")
        if self.answer is VisualPredicateAnswer.UNKNOWN:
            if self.evidence_frames or self.confidence > 0.5:
                raise SchemaError("unknown visual predicate must be low-confidence and path-free")
        elif not self.evidence_frames:
            raise SchemaError("non-unknown visual predicate requires evidence frames")
        _id(self.model_id, "model_id")
        if (
            not isinstance(self.model_revision, str)
            or _GIT_SHA.fullmatch(self.model_revision) is None
        ):
            raise SchemaError("visual predicate model_revision must be a pinned Git commit")
        _id(self.prompt_version, "prompt_version")
        if any(
            value is not False
            for value in (
                self.subject_identity_inference,
                self.risk_inference,
                self.notification_decision,
            )
        ):
            raise SchemaError("visual predicate inference boundary is invalid")

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        allowed_frame_ids: tuple[str, ...] | None = None,
    ) -> VisualPredicateResult:
        expected = {
            "version",
            "event_id",
            "predicate",
            "answer",
            "confidence",
            "evidence_frames",
            "model_id",
            "model_revision",
            "prompt_version",
            "subject_identity_inference",
            "risk_inference",
            "notification_decision",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise SchemaError("visual predicate result fields do not match v1")
        try:
            predicate = VisualPredicate(value["predicate"])
            answer = VisualPredicateAnswer(value["answer"])
        except (TypeError, ValueError) as exc:
            raise SchemaError("visual predicate result enum is invalid") from exc
        evidence = value["evidence_frames"]
        if not isinstance(evidence, list):
            raise SchemaError("visual predicate evidence_frames must be an array")
        result = cls(
            version=value["version"],
            event_id=value["event_id"],
            predicate=predicate,
            answer=answer,
            confidence=value["confidence"],
            evidence_frames=tuple(evidence),
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            prompt_version=value["prompt_version"],
            subject_identity_inference=value["subject_identity_inference"],
            risk_inference=value["risk_inference"],
            notification_decision=value["notification_decision"],
        )
        if allowed_frame_ids is not None:
            if (
                not isinstance(allowed_frame_ids, tuple)
                or not 1 <= len(allowed_frame_ids) <= 12
                or len(set(allowed_frame_ids)) != len(allowed_frame_ids)
                or any(_SAFE_ID.fullmatch(item) is None for item in allowed_frame_ids)
            ):
                raise SchemaError("allowed visual predicate frame IDs are invalid")
            if any(item not in allowed_frame_ids for item in result.evidence_frames):
                raise SchemaError("visual predicate references an unadmitted frame")
        return result

    @classmethod
    def from_json(
        cls,
        raw: str,
        *,
        allowed_frame_ids: tuple[str, ...] | None = None,
    ) -> VisualPredicateResult:
        return cls.from_dict(_strict_json(raw), allowed_frame_ids=allowed_frame_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "event_id": self.event_id,
            "predicate": self.predicate.value,
            "answer": self.answer.value,
            "confidence": self.confidence,
            "evidence_frames": list(self.evidence_frames),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "subject_identity_inference": self.subject_identity_inference,
            "risk_inference": self.risk_inference,
            "notification_decision": self.notification_decision,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
