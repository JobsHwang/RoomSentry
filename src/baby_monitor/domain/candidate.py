"""Candidate media contracts: detections, candidate events, evidence, event media."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from baby_monitor.domain._shared import (
    _SAFE_PROVENANCE_IDENTIFIER,
    SchemaError,
    _confidence,
    _require_exact_keys,
    _string_list,
)
from baby_monitor.domain.signal import SignalEvent


@dataclass(frozen=True, slots=True)
class Detection:
    timestamp_seconds: float
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_seconds, bool)
            or not isinstance(self.timestamp_seconds, (int, float))
            or not math.isfinite(float(self.timestamp_seconds))
            or self.timestamp_seconds < 0
        ):
            raise ValueError("detection timestamp cannot be negative")
        if (
            not isinstance(self.label, str)
            or not self.label.strip()
            or self.label != self.label.strip()
            or len(self.label) > 80
            or any(character in self.label for character in ("\x00", "\r", "\n"))
        ):
            raise ValueError("detection label cannot be empty")
        _confidence(self.confidence, "detection.confidence")
        if (
            not isinstance(self.bbox_xyxy, tuple)
            or len(self.bbox_xyxy) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in self.bbox_xyxy
            )
        ):
            raise ValueError("bbox_xyxy must contain four normalized values")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox_xyxy must have positive normalized area")


@dataclass(frozen=True, slots=True)
class CandidateEvent:
    event_id: str
    camera_id: str
    start_seconds: float
    end_seconds: float
    detections: tuple[Detection, ...]
    signals: tuple[SignalEvent, ...] = ()

    detector_id: str | None = None

    def __post_init__(self) -> None:
        if self.detector_id is not None and (
            not isinstance(self.detector_id, str)
            or _SAFE_PROVENANCE_IDENTIFIER.fullmatch(self.detector_id) is None
            or not self.detections
        ):
            raise ValueError("candidate detector_id is invalid")

    @property
    def center_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2.0

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({item.label for item in self.detections}))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "detections": [
                {
                    "timestamp_seconds": item.timestamp_seconds,
                    "label": item.label,
                    "confidence": item.confidence,
                    "bbox_xyxy": list(item.bbox_xyxy),
                }
                for item in self.detections
            ],
        }
        if self.signals:
            payload["signals"] = [item.to_dict() for item in self.signals]
        if self.detector_id is not None:
            payload["detector_id"] = self.detector_id
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceGap:
    started_at: datetime
    ended_at: datetime
    reason: str

    def __post_init__(self) -> None:
        for value, field in (
            (self.started_at, "started_at"),
            (self.ended_at, "ended_at"),
        ):
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise SchemaError(f"evidence gap {field} must be timezone-aware")
        if self.ended_at <= self.started_at:
            raise SchemaError("evidence gap must have a positive duration")
        if (
            not isinstance(self.reason, str)
            or re.fullmatch(r"[a-z0-9_.-]{1,64}", self.reason) is None
        ):
            raise SchemaError("evidence gap reason is invalid")

    @classmethod
    def from_dict(cls, data: Any) -> EvidenceGap:
        if not isinstance(data, dict):
            raise SchemaError("evidence gap must be an object")
        _require_exact_keys(data, {"started_at", "ended_at", "reason"}, "evidence gap")
        try:
            started_at = datetime.fromisoformat(data["started_at"])
            ended_at = datetime.fromisoformat(data["ended_at"])
        except (TypeError, ValueError) as exc:
            raise SchemaError("evidence gap timestamp is invalid") from exc
        return cls(started_at=started_at, ended_at=ended_at, reason=data["reason"])

    def to_dict(self) -> dict[str, str]:
        return {
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "ended_at": self.ended_at.astimezone(UTC).isoformat(),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class EventMedia:
    clip_path: str
    frame_paths: tuple[str, ...]
    coverage_gaps: tuple[EvidenceGap, ...] = ()
    selected_segments: int = 0
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.clip_path, str):
            raise SchemaError("event media clip_path is invalid")
        if not isinstance(self.frame_paths, tuple) or any(
            not isinstance(value, str) or not value for value in self.frame_paths
        ):
            raise SchemaError("event media frame_paths are invalid")
        if not isinstance(self.coverage_gaps, tuple) or any(
            not isinstance(value, EvidenceGap) for value in self.coverage_gaps
        ):
            raise SchemaError("event media coverage_gaps are invalid")
        if type(self.selected_segments) is not int or not 0 <= self.selected_segments <= 100_000:
            raise SchemaError("event media selected_segments is invalid")
        if type(self.version) is not int or self.version != 1:
            raise SchemaError("unsupported event media version")

    @property
    def complete_coverage(self) -> bool:
        """Legacy v1 absence-of-recorded-gaps flag, not continuity verification.

        Kept for existing serialized wrapper consistency and idempotent jobs.
        Semantic consumers must use coverage_status instead of treating True
        as proof of continuous source frames or correct wallclock mapping.
        """
        return not self.coverage_gaps

    @property
    def coverage_status(self) -> str:
        # v1 has no authenticated continuous-coverage evidence. Reading an old
        # gap-free record must not silently certify it or invent a missing span.
        return "gaps_recorded" if self.coverage_gaps else "unverified"

    @classmethod
    def from_dict(cls, data: Any) -> EventMedia:
        if not isinstance(data, dict):
            raise SchemaError("event media must be an object")
        _require_exact_keys(
            data,
            {
                "version",
                "clip_path",
                "frame_paths",
                "coverage_gaps",
                "selected_segments",
            },
            "event media",
        )
        frame_paths = _string_list(data["frame_paths"], "event_media.frame_paths")
        gaps = data["coverage_gaps"]
        if not isinstance(gaps, list):
            raise SchemaError("event media coverage_gaps must be an array")
        return cls(
            clip_path=data["clip_path"],
            frame_paths=frame_paths,
            coverage_gaps=tuple(EvidenceGap.from_dict(value) for value in gaps),
            selected_segments=data["selected_segments"],
            version=data["version"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "clip_path": self.clip_path,
            "frame_paths": list(self.frame_paths),
            "coverage_gaps": [value.to_dict() for value in self.coverage_gaps],
            "selected_segments": self.selected_segments,
        }
