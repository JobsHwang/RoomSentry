"""VLM/recognition contracts: subjects, events, provenance, run metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from baby_monitor.domain._shared import (
    _SAFE_PROVENANCE_IDENTIFIER,
    SchemaError,
    _confidence,
    _reject_json_constant,
    _require_exact_keys,
    _string_list,
)
from baby_monitor.domain.enums import (
    EventType,
    RiskLevel,
    SubjectKind,
)


@dataclass(frozen=True, slots=True)
class Subject:
    kind: SubjectKind
    confidence: float
    evidence_frames: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Any) -> Subject:
        if not isinstance(data, dict):
            raise SchemaError("subject must be an object")
        _require_exact_keys(data, {"kind", "confidence", "evidence_frames"}, "subject")
        try:
            kind = SubjectKind(data["kind"])
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"invalid subject kind: {data.get('kind')!r}") from exc
        return cls(
            kind=kind,
            confidence=_confidence(data["confidence"], "subject.confidence"),
            evidence_frames=_string_list(
                data["evidence_frames"], "subject.evidence_frames", unique=True
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "confidence": self.confidence,
            "evidence_frames": list(self.evidence_frames),
        }


VLM_EVENT_KEYS = {
    "event_type",
    "subjects",
    "observations",
    "inferences",
    "risk_level",
    "confidence",
    "needs_remote_review",
    "summary_zh",
    "evidence_frames",
}


@dataclass(frozen=True, slots=True)
class RecognitionProvenance:
    adapter_name: str
    model_revision: str | None = None
    prompt_revision: str = "baby_monitor_vlm_prompt_v2"
    output_schema_revision: str = "vlm_event_v1"
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise SchemaError("unsupported recognition provenance version")
        for value, field in (
            (self.adapter_name, "adapter_name"),
            (self.prompt_revision, "prompt_revision"),
            (self.output_schema_revision, "output_schema_revision"),
        ):
            if not isinstance(value, str) or _SAFE_PROVENANCE_IDENTIFIER.fullmatch(value) is None:
                raise SchemaError(f"recognition provenance {field} is invalid")
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str)
            or _SAFE_PROVENANCE_IDENTIFIER.fullmatch(self.model_revision) is None
        ):
            raise SchemaError("recognition provenance model_revision is invalid")

    @classmethod
    def from_dict(cls, data: Any) -> RecognitionProvenance:
        expected = {
            "version",
            "adapter_name",
            "model_revision",
            "prompt_revision",
            "output_schema_revision",
        }
        if not isinstance(data, dict):
            raise SchemaError("recognition provenance must be an object")
        _require_exact_keys(data, expected, "recognition provenance")
        if data["model_revision"] is not None and not isinstance(data["model_revision"], str):
            raise SchemaError("recognition provenance model_revision must be a string or null")
        return cls(
            version=data["version"],
            adapter_name=data["adapter_name"],
            model_revision=data["model_revision"],
            prompt_revision=data["prompt_revision"],
            output_schema_revision=data["output_schema_revision"],
        )

    @classmethod
    def from_json(cls, payload: str) -> RecognitionProvenance:
        try:
            data = json.loads(payload, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SchemaError("recognition provenance is not valid strict JSON") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "adapter_name": self.adapter_name,
            "model_revision": self.model_revision,
            "prompt_revision": self.prompt_revision,
            "output_schema_revision": self.output_schema_revision,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)


@dataclass(frozen=True, slots=True)
class RecognitionRunMetrics:
    duration_ms: int
    json_compliant: bool
    process_rss_peak_kib: int | None = None
    gpu_device_memory_peak_mib: int | None = None
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise SchemaError("unsupported recognition run metrics version")
        if type(self.duration_ms) is not int or not 0 <= self.duration_ms <= 86_400_000:
            raise SchemaError("recognition run duration_ms is invalid")
        if type(self.json_compliant) is not bool:
            raise SchemaError("recognition run json_compliant must be a boolean")
        for value, field, maximum in (
            (self.process_rss_peak_kib, "process_rss_peak_kib", 1_000_000_000),
            (
                self.gpu_device_memory_peak_mib,
                "gpu_device_memory_peak_mib",
                10_000_000,
            ),
        ):
            if value is not None and (type(value) is not int or not 0 <= value <= maximum):
                raise SchemaError(f"recognition run {field} is invalid")

    @classmethod
    def from_dict(cls, data: Any) -> RecognitionRunMetrics:
        expected = {
            "version",
            "duration_ms",
            "json_compliant",
            "process_rss_peak_kib",
            "gpu_device_memory_peak_mib",
        }
        if not isinstance(data, dict):
            raise SchemaError("recognition run metrics must be an object")
        _require_exact_keys(data, expected, "recognition run metrics")
        return cls(
            version=data["version"],
            duration_ms=data["duration_ms"],
            json_compliant=data["json_compliant"],
            process_rss_peak_kib=data["process_rss_peak_kib"],
            gpu_device_memory_peak_mib=data["gpu_device_memory_peak_mib"],
        )

    @classmethod
    def from_json(cls, payload: str) -> RecognitionRunMetrics:
        try:
            data = json.loads(payload, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SchemaError("recognition run metrics are not valid strict JSON") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "duration_ms": self.duration_ms,
            "json_compliant": self.json_compliant,
            "process_rss_peak_kib": self.process_rss_peak_kib,
            "gpu_device_memory_peak_mib": self.gpu_device_memory_peak_mib,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)


@dataclass(frozen=True, slots=True)
class VLMEvent:
    event_type: EventType
    subjects: tuple[Subject, ...]
    observations: tuple[str, ...]
    inferences: tuple[str, ...]
    risk_level: RiskLevel
    confidence: float
    needs_remote_review: bool
    summary_zh: str
    evidence_frames: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Any) -> VLMEvent:
        if not isinstance(data, dict):
            raise SchemaError("VLM response must be one JSON object")
        _require_exact_keys(data, VLM_EVENT_KEYS, "VLM response")

        try:
            event_type = EventType(data["event_type"])
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"invalid event_type: {data.get('event_type')!r}") from exc
        try:
            risk_level = RiskLevel(data["risk_level"])
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"invalid risk_level: {data.get('risk_level')!r}") from exc

        subjects_raw = data["subjects"]
        if not isinstance(subjects_raw, list):
            raise SchemaError("subjects must be an array")
        subjects = tuple(Subject.from_dict(item) for item in subjects_raw)

        if type(data["needs_remote_review"]) is not bool:
            raise SchemaError("needs_remote_review must be a boolean")
        if not isinstance(data["summary_zh"], str):
            raise SchemaError("summary_zh must be a string")

        return cls(
            event_type=event_type,
            subjects=subjects,
            observations=_string_list(data["observations"], "observations"),
            inferences=_string_list(data["inferences"], "inferences"),
            risk_level=risk_level,
            confidence=_confidence(data["confidence"], "confidence"),
            needs_remote_review=data["needs_remote_review"],
            summary_zh=data["summary_zh"],
            evidence_frames=_string_list(data["evidence_frames"], "evidence_frames", unique=True),
        )

    @classmethod
    def from_json(cls, raw: str) -> VLMEvent:
        if not isinstance(raw, str):
            raise SchemaError("VLM response must be text containing one JSON object")
        try:
            data = json.loads(raw, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SchemaError(f"invalid JSON: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "subjects": [subject.to_dict() for subject in self.subjects],
            "observations": list(self.observations),
            "inferences": list(self.inferences),
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "needs_remote_review": self.needs_remote_review,
            "summary_zh": self.summary_zh,
            "evidence_frames": list(self.evidence_frames),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def require_remote_review(self) -> VLMEvent:
        if self.needs_remote_review:
            return self
        return replace(self, needs_remote_review=True)

    @staticmethod
    def json_schema() -> dict[str, Any]:
        subject_kinds = [item.value for item in SubjectKind]
        return {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(VLM_EVENT_KEYS),
            "properties": {
                "event_type": {"type": "string", "enum": [item.value for item in EventType]},
                "subjects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["kind", "confidence", "evidence_frames"],
                        "properties": {
                            "kind": {"type": "string", "enum": subject_kinds},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "evidence_frames": {
                                "type": "array",
                                "items": {"type": "string"},
                                "uniqueItems": True,
                            },
                        },
                    },
                },
                "observations": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "inferences": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "risk_level": {"type": "string", "enum": [item.value for item in RiskLevel]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "needs_remote_review": {"type": "boolean"},
                "summary_zh": {"type": "string"},
                "evidence_frames": {
                    "type": "array",
                    "items": {"type": "string"},
                    "uniqueItems": True,
                },
            },
        }
