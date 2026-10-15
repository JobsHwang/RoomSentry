from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pose_protocol import (
    COCO_PERSON_KEYPOINTS,
    PoseFrame,
    PoseFrameBatch,
    PoseInstance,
)

MAX_POSE_QUALITY_REPORT_BYTES = 16 * 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_STATUSES = {
    "no_pose_evidence",
    "single_frame_only",
    "no_anonymous_pairs",
    "insufficient_shared_keypoints",
    "temporal_measurements_available",
}


class PoseQualityError(ValueError):
    """A path-free pose quality configuration or report is invalid."""


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PoseQualityError("pose quality JSON contains duplicate keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                PoseQualityError("pose quality JSON contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise PoseQualityError("pose quality JSON is invalid") from exc


def _read_json(path: str | Path) -> Any:
    source = Path(path).expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        if source.stat().st_size > MAX_POSE_QUALITY_REPORT_BYTES:
            raise PoseQualityError("pose quality report exceeds the 16 MiB limit")
        raw = source.read_text(encoding="utf-8")
    except PoseQualityError:
        raise
    except (OSError, UnicodeError) as exc:
        raise PoseQualityError("pose quality report is missing or unsafe") from exc
    return _strict_json(raw)


def _id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PoseQualityError(f"pose quality {field} is invalid")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PoseQualityError(f"pose quality {field} is invalid")
    return value


def _number(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise PoseQualityError(f"pose quality {field} is invalid")
    return float(value)


def _optional_number(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is None:
        return None
    return _number(value, field, minimum=minimum, maximum=maximum)


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PoseQualityError(f"pose quality {field} is invalid")
    return value


def _clean(value: float) -> float:
    return round(value, 12)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return _clean(numerator / denominator)


def _validate_threshold_aggregate(
    *,
    scope: str,
    comparable_pairs: int,
    change_pairs: int,
    median: float | None,
    p95: float | None,
    threshold: float,
) -> None:
    """Reject aggregate metrics that cannot produce the declared threshold count."""

    if comparable_pairs == 0:
        return
    if median is None or p95 is None:
        raise PoseQualityError(f"pose quality {scope} displacement metrics are missing")
    if comparable_pairs == 1 and median != p95:
        raise PoseQualityError(f"pose quality {scope} single-pair median and p95 are inconsistent")
    if change_pairs == 0 and p95 > threshold:
        raise PoseQualityError(f"pose quality {scope} change count is inconsistent")
    if change_pairs == comparable_pairs and median <= threshold:
        raise PoseQualityError(f"pose quality {scope} change count is inconsistent")
    if change_pairs * 2 > comparable_pairs and median <= threshold:
        raise PoseQualityError(f"pose quality {scope} median is inconsistent")
    if change_pairs * 2 < comparable_pairs and median > threshold:
        raise PoseQualityError(f"pose quality {scope} median is inconsistent")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return _clean(ordered[lower])
    weight = position - lower
    return _clean(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


@dataclass(frozen=True, slots=True)
class PoseQualityConfig:
    keypoint_confidence_threshold: float = 0.5
    min_usable_keypoints: int = 8
    min_pair_bbox_iou: float = 0.2
    min_shared_keypoints: int = 8
    image_change_threshold: float = 0.05
    shape_change_threshold: float = 0.05
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise PoseQualityError("unsupported pose quality config version")
        object.__setattr__(
            self,
            "keypoint_confidence_threshold",
            _number(
                self.keypoint_confidence_threshold,
                "keypoint_confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        _integer(
            self.min_usable_keypoints,
            "min_usable_keypoints",
            minimum=1,
            maximum=len(COCO_PERSON_KEYPOINTS),
        )
        object.__setattr__(
            self,
            "min_pair_bbox_iou",
            _number(
                self.min_pair_bbox_iou,
                "min_pair_bbox_iou",
                minimum=1e-12,
                maximum=1.0,
            ),
        )
        _integer(
            self.min_shared_keypoints,
            "min_shared_keypoints",
            minimum=1,
            maximum=len(COCO_PERSON_KEYPOINTS),
        )
        for field in ("image_change_threshold", "shape_change_threshold"):
            object.__setattr__(
                self,
                field,
                _number(getattr(self, field), field, minimum=0.0, maximum=100.0),
            )

    @classmethod
    def from_dict(cls, value: Any) -> PoseQualityConfig:
        expected = {
            "version",
            "keypoint_confidence_threshold",
            "min_usable_keypoints",
            "min_pair_bbox_iou",
            "min_shared_keypoints",
            "image_change_threshold",
            "shape_change_threshold",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise PoseQualityError("pose quality config fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "keypoint_confidence_threshold": self.keypoint_confidence_threshold,
            "min_usable_keypoints": self.min_usable_keypoints,
            "min_pair_bbox_iou": self.min_pair_bbox_iou,
            "min_shared_keypoints": self.min_shared_keypoints,
            "image_change_threshold": self.image_change_threshold,
            "shape_change_threshold": self.shape_change_threshold,
        }


@dataclass(frozen=True, slots=True)
class PoseFrameQuality:
    frame_id: str
    timestamp_seconds: float
    pose_count: int
    usable_pose_count: int
    in_frame_keypoint_count: int
    high_confidence_keypoint_count: int

    def __post_init__(self) -> None:
        _id(self.frame_id, "frame_id")
        object.__setattr__(
            self,
            "timestamp_seconds",
            _number(
                self.timestamp_seconds,
                "frame timestamp_seconds",
                minimum=0.0,
                maximum=31_536_000.0,
            ),
        )
        _integer(self.pose_count, "frame pose_count", minimum=0, maximum=100)
        _integer(
            self.usable_pose_count,
            "frame usable_pose_count",
            minimum=0,
            maximum=self.pose_count,
        )
        maximum_keypoints = self.pose_count * len(COCO_PERSON_KEYPOINTS)
        _integer(
            self.in_frame_keypoint_count,
            "frame in_frame_keypoint_count",
            minimum=0,
            maximum=maximum_keypoints,
        )
        _integer(
            self.high_confidence_keypoint_count,
            "frame high_confidence_keypoint_count",
            minimum=0,
            maximum=self.in_frame_keypoint_count,
        )

    @classmethod
    def from_dict(cls, value: Any) -> PoseFrameQuality:
        expected = {
            "frame_id",
            "timestamp_seconds",
            "pose_count",
            "usable_pose_count",
            "in_frame_keypoint_count",
            "high_confidence_keypoint_count",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise PoseQualityError("pose frame quality fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp_seconds": self.timestamp_seconds,
            "pose_count": self.pose_count,
            "usable_pose_count": self.usable_pose_count,
            "in_frame_keypoint_count": self.in_frame_keypoint_count,
            "high_confidence_keypoint_count": self.high_confidence_keypoint_count,
        }


@dataclass(frozen=True, slots=True)
class PoseTransitionQuality:
    from_frame_id: str
    to_frame_id: str
    delta_seconds: float
    from_pose_count: int
    to_pose_count: int
    matched_pose_pairs: int
    comparable_pose_pairs: int
    insufficient_shared_keypoint_pairs: int
    unmatched_from_poses: int
    unmatched_to_poses: int
    image_change_pairs: int
    shape_change_pairs: int
    image_displacement_median: float | None
    image_displacement_p95: float | None
    shape_displacement_median: float | None
    shape_displacement_p95: float | None

    def __post_init__(self) -> None:
        _id(self.from_frame_id, "from_frame_id")
        _id(self.to_frame_id, "to_frame_id")
        if self.from_frame_id == self.to_frame_id:
            raise PoseQualityError("pose quality transition frame IDs must differ")
        object.__setattr__(
            self,
            "delta_seconds",
            _number(
                self.delta_seconds,
                "transition delta_seconds",
                minimum=1e-12,
                maximum=31_536_000.0,
            ),
        )
        _integer(self.from_pose_count, "from_pose_count", minimum=0, maximum=100)
        _integer(self.to_pose_count, "to_pose_count", minimum=0, maximum=100)
        _integer(
            self.matched_pose_pairs,
            "matched_pose_pairs",
            minimum=0,
            maximum=min(self.from_pose_count, self.to_pose_count),
        )
        _integer(
            self.comparable_pose_pairs,
            "comparable_pose_pairs",
            minimum=0,
            maximum=self.matched_pose_pairs,
        )
        _integer(
            self.insufficient_shared_keypoint_pairs,
            "insufficient_shared_keypoint_pairs",
            minimum=0,
            maximum=self.matched_pose_pairs,
        )
        if (
            self.comparable_pose_pairs + self.insufficient_shared_keypoint_pairs
            != self.matched_pose_pairs
        ):
            raise PoseQualityError("pose quality matched pair counts are inconsistent")
        if self.unmatched_from_poses != self.from_pose_count - self.matched_pose_pairs:
            raise PoseQualityError("pose quality unmatched_from_poses is inconsistent")
        if self.unmatched_to_poses != self.to_pose_count - self.matched_pose_pairs:
            raise PoseQualityError("pose quality unmatched_to_poses is inconsistent")
        _integer(
            self.unmatched_from_poses,
            "unmatched_from_poses",
            minimum=0,
            maximum=self.from_pose_count,
        )
        _integer(
            self.unmatched_to_poses,
            "unmatched_to_poses",
            minimum=0,
            maximum=self.to_pose_count,
        )
        _integer(
            self.image_change_pairs,
            "image_change_pairs",
            minimum=0,
            maximum=self.comparable_pose_pairs,
        )
        _integer(
            self.shape_change_pairs,
            "shape_change_pairs",
            minimum=0,
            maximum=self.comparable_pose_pairs,
        )
        displacement_fields = (
            "image_displacement_median",
            "image_displacement_p95",
            "shape_displacement_median",
            "shape_displacement_p95",
        )
        parsed = {
            field: _optional_number(
                getattr(self, field),
                field,
                minimum=0.0,
                maximum=1_000_000.0,
            )
            for field in displacement_fields
        }
        for field, value in parsed.items():
            object.__setattr__(self, field, value)
        if self.comparable_pose_pairs == 0:
            if any(value is not None for value in parsed.values()):
                raise PoseQualityError("pose quality displacement metrics require comparable pairs")
        elif any(value is None for value in parsed.values()):
            raise PoseQualityError("pose quality comparable pairs require displacement metrics")
        elif (
            parsed["image_displacement_p95"] < parsed["image_displacement_median"]
            or parsed["shape_displacement_p95"] < parsed["shape_displacement_median"]
        ):
            raise PoseQualityError("pose quality p95 displacement cannot be below the median")

    @classmethod
    def from_dict(cls, value: Any) -> PoseTransitionQuality:
        expected = {
            "from_frame_id",
            "to_frame_id",
            "delta_seconds",
            "from_pose_count",
            "to_pose_count",
            "matched_pose_pairs",
            "comparable_pose_pairs",
            "insufficient_shared_keypoint_pairs",
            "unmatched_from_poses",
            "unmatched_to_poses",
            "image_change_pairs",
            "shape_change_pairs",
            "image_displacement_median",
            "image_displacement_p95",
            "shape_displacement_median",
            "shape_displacement_p95",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise PoseQualityError("pose transition quality fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {
            "from_frame_id": self.from_frame_id,
            "to_frame_id": self.to_frame_id,
            "delta_seconds": self.delta_seconds,
            "from_pose_count": self.from_pose_count,
            "to_pose_count": self.to_pose_count,
            "matched_pose_pairs": self.matched_pose_pairs,
            "comparable_pose_pairs": self.comparable_pose_pairs,
            "insufficient_shared_keypoint_pairs": self.insufficient_shared_keypoint_pairs,
            "unmatched_from_poses": self.unmatched_from_poses,
            "unmatched_to_poses": self.unmatched_to_poses,
            "image_change_pairs": self.image_change_pairs,
            "shape_change_pairs": self.shape_change_pairs,
            "image_displacement_median": self.image_displacement_median,
            "image_displacement_p95": self.image_displacement_p95,
            "shape_displacement_median": self.shape_displacement_median,
            "shape_displacement_p95": self.shape_displacement_p95,
        }


@dataclass(frozen=True, slots=True)
class PoseQualitySummary:
    evidence_status: str
    frame_count: int
    pose_count: int
    usable_pose_count: int
    frames_with_poses: int
    frames_with_usable_poses: int
    usable_pose_rate: float | None
    transition_count: int
    matched_pose_pairs: int
    comparable_pose_pairs: int
    insufficient_shared_keypoint_pairs: int
    temporal_comparability_rate: float | None
    image_change_pairs: int
    shape_change_pairs: int
    image_displacement_median: float | None
    image_displacement_p95: float | None
    shape_displacement_median: float | None
    shape_displacement_p95: float | None

    def __post_init__(self) -> None:
        if self.evidence_status not in _EVIDENCE_STATUSES:
            raise PoseQualityError("pose quality evidence_status is invalid")
        _integer(self.frame_count, "summary frame_count", minimum=1, maximum=10_000)
        _integer(self.pose_count, "summary pose_count", minimum=0, maximum=5_000)
        _integer(
            self.usable_pose_count,
            "summary usable_pose_count",
            minimum=0,
            maximum=self.pose_count,
        )
        _integer(
            self.frames_with_poses,
            "summary frames_with_poses",
            minimum=0,
            maximum=self.frame_count,
        )
        _integer(
            self.frames_with_usable_poses,
            "summary frames_with_usable_poses",
            minimum=0,
            maximum=self.frames_with_poses,
        )
        _integer(
            self.transition_count,
            "summary transition_count",
            minimum=0,
            maximum=self.frame_count - 1,
        )
        if self.transition_count != self.frame_count - 1:
            raise PoseQualityError("pose quality transition_count is inconsistent")
        for field in (
            "matched_pose_pairs",
            "comparable_pose_pairs",
            "insufficient_shared_keypoint_pairs",
            "image_change_pairs",
            "shape_change_pairs",
        ):
            _integer(getattr(self, field), f"summary {field}", minimum=0, maximum=5_000)
        if self.comparable_pose_pairs > self.matched_pose_pairs:
            raise PoseQualityError("pose quality comparable pair total is inconsistent")
        if (
            self.comparable_pose_pairs + self.insufficient_shared_keypoint_pairs
            != self.matched_pose_pairs
        ):
            raise PoseQualityError("pose quality matched pair total is inconsistent")
        if self.image_change_pairs > self.comparable_pose_pairs:
            raise PoseQualityError("pose quality image change total is inconsistent")
        if self.shape_change_pairs > self.comparable_pose_pairs:
            raise PoseQualityError("pose quality shape change total is inconsistent")
        expected_usable_rate = _ratio(self.usable_pose_count, self.pose_count)
        expected_comparability = _ratio(self.comparable_pose_pairs, self.matched_pose_pairs)
        parsed_usable_rate = _optional_number(
            self.usable_pose_rate,
            "usable_pose_rate",
            minimum=0.0,
            maximum=1.0,
        )
        parsed_comparability = _optional_number(
            self.temporal_comparability_rate,
            "temporal_comparability_rate",
            minimum=0.0,
            maximum=1.0,
        )
        if parsed_usable_rate != expected_usable_rate:
            raise PoseQualityError("pose quality usable_pose_rate is inconsistent")
        if parsed_comparability != expected_comparability:
            raise PoseQualityError("pose quality temporal_comparability_rate is inconsistent")
        object.__setattr__(self, "usable_pose_rate", parsed_usable_rate)
        object.__setattr__(self, "temporal_comparability_rate", parsed_comparability)
        displacement_fields = (
            "image_displacement_median",
            "image_displacement_p95",
            "shape_displacement_median",
            "shape_displacement_p95",
        )
        parsed = {
            field: _optional_number(
                getattr(self, field),
                field,
                minimum=0.0,
                maximum=1_000_000.0,
            )
            for field in displacement_fields
        }
        for field, value in parsed.items():
            object.__setattr__(self, field, value)
        if self.comparable_pose_pairs == 0:
            if any(value is not None for value in parsed.values()):
                raise PoseQualityError(
                    "pose quality summary displacement requires comparable pairs"
                )
        elif any(value is None for value in parsed.values()):
            raise PoseQualityError(
                "pose quality summary comparable pairs require displacement metrics"
            )
        elif (
            parsed["image_displacement_p95"] < parsed["image_displacement_median"]
            or parsed["shape_displacement_p95"] < parsed["shape_displacement_median"]
        ):
            raise PoseQualityError("pose quality summary p95 cannot be below the median")

        expected_status = _evidence_status(
            frame_count=self.frame_count,
            pose_count=self.pose_count,
            matched_pose_pairs=self.matched_pose_pairs,
            comparable_pose_pairs=self.comparable_pose_pairs,
        )
        if self.evidence_status != expected_status:
            raise PoseQualityError("pose quality evidence_status is inconsistent")

    @classmethod
    def from_dict(cls, value: Any) -> PoseQualitySummary:
        expected = {
            "evidence_status",
            "frame_count",
            "pose_count",
            "usable_pose_count",
            "frames_with_poses",
            "frames_with_usable_poses",
            "usable_pose_rate",
            "transition_count",
            "matched_pose_pairs",
            "comparable_pose_pairs",
            "insufficient_shared_keypoint_pairs",
            "temporal_comparability_rate",
            "image_change_pairs",
            "shape_change_pairs",
            "image_displacement_median",
            "image_displacement_p95",
            "shape_displacement_median",
            "shape_displacement_p95",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise PoseQualityError("pose quality summary fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_status": self.evidence_status,
            "frame_count": self.frame_count,
            "pose_count": self.pose_count,
            "usable_pose_count": self.usable_pose_count,
            "frames_with_poses": self.frames_with_poses,
            "frames_with_usable_poses": self.frames_with_usable_poses,
            "usable_pose_rate": self.usable_pose_rate,
            "transition_count": self.transition_count,
            "matched_pose_pairs": self.matched_pose_pairs,
            "comparable_pose_pairs": self.comparable_pose_pairs,
            "insufficient_shared_keypoint_pairs": self.insufficient_shared_keypoint_pairs,
            "temporal_comparability_rate": self.temporal_comparability_rate,
            "image_change_pairs": self.image_change_pairs,
            "shape_change_pairs": self.shape_change_pairs,
            "image_displacement_median": self.image_displacement_median,
            "image_displacement_p95": self.image_displacement_p95,
            "shape_displacement_median": self.shape_displacement_median,
            "shape_displacement_p95": self.shape_displacement_p95,
        }


@dataclass(frozen=True, slots=True)
class PoseQualityReport:
    camera_id: str
    pose_adapter: str
    model_id: str
    model_revision: str
    model_sha256: str
    pose_batch_sha256: str
    config: PoseQualityConfig
    frames: tuple[PoseFrameQuality, ...]
    transitions: tuple[PoseTransitionQuality, ...]
    summary: PoseQualitySummary
    anonymous_pairing_only: bool = True
    subject_identity_inference: bool = False
    posture_classification: bool = False
    risk_inference: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise PoseQualityError("unsupported pose quality report version")
        for field in ("camera_id", "pose_adapter", "model_id", "model_revision"):
            _id(getattr(self, field), field)
        _sha256(self.model_sha256, "model_sha256")
        _sha256(self.pose_batch_sha256, "pose_batch_sha256")
        if not isinstance(self.config, PoseQualityConfig):
            raise PoseQualityError("pose quality config is invalid")
        if (
            not isinstance(self.frames, tuple)
            or not self.frames
            or len(self.frames) > 10_000
            or any(not isinstance(item, PoseFrameQuality) for item in self.frames)
        ):
            raise PoseQualityError("pose quality frames are invalid")
        if len({item.frame_id for item in self.frames}) != len(self.frames):
            raise PoseQualityError("pose quality frame IDs must be unique")
        if any(
            current.timestamp_seconds <= previous.timestamp_seconds
            for previous, current in zip(self.frames, self.frames[1:], strict=False)
        ):
            raise PoseQualityError("pose quality frames must be strictly chronological")
        if (
            not isinstance(self.transitions, tuple)
            or len(self.transitions) != len(self.frames) - 1
            or any(not isinstance(item, PoseTransitionQuality) for item in self.transitions)
        ):
            raise PoseQualityError("pose quality transitions are invalid")
        for index, transition in enumerate(self.transitions):
            previous = self.frames[index]
            current = self.frames[index + 1]
            if (
                transition.from_frame_id != previous.frame_id
                or transition.to_frame_id != current.frame_id
                or transition.delta_seconds
                != _clean(current.timestamp_seconds - previous.timestamp_seconds)
                or transition.from_pose_count != previous.pose_count
                or transition.to_pose_count != current.pose_count
            ):
                raise PoseQualityError("pose quality transition does not match adjacent frames")
            _validate_threshold_aggregate(
                scope="transition image",
                comparable_pairs=transition.comparable_pose_pairs,
                change_pairs=transition.image_change_pairs,
                median=transition.image_displacement_median,
                p95=transition.image_displacement_p95,
                threshold=self.config.image_change_threshold,
            )
            _validate_threshold_aggregate(
                scope="transition shape",
                comparable_pairs=transition.comparable_pose_pairs,
                change_pairs=transition.shape_change_pairs,
                median=transition.shape_displacement_median,
                p95=transition.shape_displacement_p95,
                threshold=self.config.shape_change_threshold,
            )
        keypoint_count = len(COCO_PERSON_KEYPOINTS)
        for frame in self.frames:
            minimum_high_confidence = frame.usable_pose_count * self.config.min_usable_keypoints
            maximum_high_confidence = frame.usable_pose_count * keypoint_count + (
                frame.pose_count - frame.usable_pose_count
            ) * (self.config.min_usable_keypoints - 1)
            if not (
                minimum_high_confidence
                <= frame.high_confidence_keypoint_count
                <= maximum_high_confidence
            ):
                raise PoseQualityError(
                    "pose quality usable pose count is inconsistent with keypoint coverage"
                )
        if not isinstance(self.summary, PoseQualitySummary):
            raise PoseQualityError("pose quality summary is invalid")
        expected_totals = {
            "frame_count": len(self.frames),
            "pose_count": sum(item.pose_count for item in self.frames),
            "usable_pose_count": sum(item.usable_pose_count for item in self.frames),
            "frames_with_poses": sum(item.pose_count > 0 for item in self.frames),
            "frames_with_usable_poses": sum(item.usable_pose_count > 0 for item in self.frames),
            "transition_count": len(self.transitions),
            "matched_pose_pairs": sum(item.matched_pose_pairs for item in self.transitions),
            "comparable_pose_pairs": sum(item.comparable_pose_pairs for item in self.transitions),
            "insufficient_shared_keypoint_pairs": sum(
                item.insufficient_shared_keypoint_pairs for item in self.transitions
            ),
            "image_change_pairs": sum(item.image_change_pairs for item in self.transitions),
            "shape_change_pairs": sum(item.shape_change_pairs for item in self.transitions),
        }
        if any(getattr(self.summary, field) != value for field, value in expected_totals.items()):
            raise PoseQualityError("pose quality summary totals do not match the report")
        _validate_threshold_aggregate(
            scope="summary image",
            comparable_pairs=self.summary.comparable_pose_pairs,
            change_pairs=self.summary.image_change_pairs,
            median=self.summary.image_displacement_median,
            p95=self.summary.image_displacement_p95,
            threshold=self.config.image_change_threshold,
        )
        _validate_threshold_aggregate(
            scope="summary shape",
            comparable_pairs=self.summary.comparable_pose_pairs,
            change_pairs=self.summary.shape_change_pairs,
            median=self.summary.shape_displacement_median,
            p95=self.summary.shape_displacement_p95,
            threshold=self.config.shape_change_threshold,
        )
        if type(self.anonymous_pairing_only) is not bool or not self.anonymous_pairing_only:
            raise PoseQualityError("pose quality pairing must remain anonymous")
        for field in (
            "subject_identity_inference",
            "posture_classification",
            "risk_inference",
            "source_media_included",
            "paths_disclosed",
        ):
            value = getattr(self, field)
            if type(value) is not bool or value:
                raise PoseQualityError(f"pose quality {field} must be false")

    @classmethod
    def from_dict(cls, value: Any) -> PoseQualityReport:
        expected = {
            "version",
            "camera_id",
            "pose_adapter",
            "model_id",
            "model_revision",
            "model_sha256",
            "pose_batch_sha256",
            "config",
            "frames",
            "transitions",
            "summary",
            "anonymous_pairing_only",
            "subject_identity_inference",
            "posture_classification",
            "risk_inference",
            "source_media_included",
            "paths_disclosed",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise PoseQualityError("pose quality report fields do not match v1")
        frames = value["frames"]
        transitions = value["transitions"]
        if not isinstance(frames, list) or not isinstance(transitions, list):
            raise PoseQualityError("pose quality frame and transition metrics must be arrays")
        return cls(
            version=value["version"],
            camera_id=value["camera_id"],
            pose_adapter=value["pose_adapter"],
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            model_sha256=value["model_sha256"],
            pose_batch_sha256=value["pose_batch_sha256"],
            config=PoseQualityConfig.from_dict(value["config"]),
            frames=tuple(PoseFrameQuality.from_dict(item) for item in frames),
            transitions=tuple(PoseTransitionQuality.from_dict(item) for item in transitions),
            summary=PoseQualitySummary.from_dict(value["summary"]),
            anonymous_pairing_only=value["anonymous_pairing_only"],
            subject_identity_inference=value["subject_identity_inference"],
            posture_classification=value["posture_classification"],
            risk_inference=value["risk_inference"],
            source_media_included=value["source_media_included"],
            paths_disclosed=value["paths_disclosed"],
        )

    @classmethod
    def from_json(cls, raw: str) -> PoseQualityReport:
        if len(raw.encode("utf-8")) > MAX_POSE_QUALITY_REPORT_BYTES:
            raise PoseQualityError("pose quality report exceeds the 16 MiB limit")
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> PoseQualityReport:
        return cls.from_dict(_read_json(path))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "pose_adapter": self.pose_adapter,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "pose_batch_sha256": self.pose_batch_sha256,
            "config": self.config.to_dict(),
            "frames": [item.to_dict() for item in self.frames],
            "transitions": [item.to_dict() for item in self.transitions],
            "summary": self.summary.to_dict(),
            "anonymous_pairing_only": True,
            "subject_identity_inference": False,
            "posture_classification": False,
            "risk_inference": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }

    def to_json(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(encoded.encode("utf-8")) > MAX_POSE_QUALITY_REPORT_BYTES:
            raise PoseQualityError("pose quality report exceeds the 16 MiB limit")
        return encoded


def _bbox_iou(first: PoseInstance, second: PoseInstance) -> float:
    ax1, ay1, ax2, ay2 = first.bbox_xyxy
    bx1, by1, bx2, by2 = second.bbox_xyxy
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _anonymous_pairs(
    previous: PoseFrame,
    current: PoseFrame,
    *,
    minimum_iou: float,
) -> tuple[tuple[PoseInstance, PoseInstance], ...]:
    candidates: list[tuple[float, str, str, int, int]] = []
    for previous_index, previous_pose in enumerate(previous.poses):
        for current_index, current_pose in enumerate(current.poses):
            overlap = _bbox_iou(previous_pose, current_pose)
            if overlap >= minimum_iou:
                candidates.append(
                    (
                        -overlap,
                        previous_pose.instance_id,
                        current_pose.instance_id,
                        previous_index,
                        current_index,
                    )
                )
    used_previous: set[int] = set()
    used_current: set[int] = set()
    pairs: list[tuple[PoseInstance, PoseInstance]] = []
    for _negative_iou, _previous_id, _current_id, previous_index, current_index in sorted(
        candidates
    ):
        if previous_index in used_previous or current_index in used_current:
            continue
        used_previous.add(previous_index)
        used_current.add(current_index)
        pairs.append((previous.poses[previous_index], current.poses[current_index]))
    return tuple(pairs)


def _shared_keypoint_displacements(
    previous: PoseInstance,
    current: PoseInstance,
    *,
    confidence_threshold: float,
) -> tuple[list[float], list[float]]:
    previous_width = previous.bbox_xyxy[2] - previous.bbox_xyxy[0]
    previous_height = previous.bbox_xyxy[3] - previous.bbox_xyxy[1]
    current_width = current.bbox_xyxy[2] - current.bbox_xyxy[0]
    current_height = current.bbox_xyxy[3] - current.bbox_xyxy[1]
    previous_diagonal = math.hypot(previous_width, previous_height)
    current_diagonal = math.hypot(current_width, current_height)
    scale = (previous_diagonal + current_diagonal) / 2.0

    image_displacements: list[float] = []
    shape_displacements: list[float] = []
    for previous_point, current_point in zip(
        previous.keypoints,
        current.keypoints,
        strict=True,
    ):
        if (
            not previous_point.in_frame
            or not current_point.in_frame
            or previous_point.confidence < confidence_threshold
            or current_point.confidence < confidence_threshold
        ):
            continue
        assert previous_point.x is not None and previous_point.y is not None
        assert current_point.x is not None and current_point.y is not None
        image_displacements.append(
            math.hypot(
                current_point.x - previous_point.x,
                current_point.y - previous_point.y,
            )
            / scale
        )
        previous_relative = (
            (previous_point.x - previous.bbox_xyxy[0]) / previous_width,
            (previous_point.y - previous.bbox_xyxy[1]) / previous_height,
        )
        current_relative = (
            (current_point.x - current.bbox_xyxy[0]) / current_width,
            (current_point.y - current.bbox_xyxy[1]) / current_height,
        )
        shape_displacements.append(
            math.hypot(
                current_relative[0] - previous_relative[0],
                current_relative[1] - previous_relative[1],
            )
            / math.sqrt(2.0)
        )
    return image_displacements, shape_displacements


def _frame_quality(frame: PoseFrame, config: PoseQualityConfig) -> PoseFrameQuality:
    high_confidence_counts = [
        sum(
            point.in_frame and point.confidence >= config.keypoint_confidence_threshold
            for point in pose.keypoints
        )
        for pose in frame.poses
    ]
    return PoseFrameQuality(
        frame_id=frame.frame_id,
        timestamp_seconds=frame.timestamp_seconds,
        pose_count=len(frame.poses),
        usable_pose_count=sum(
            count >= config.min_usable_keypoints for count in high_confidence_counts
        ),
        in_frame_keypoint_count=sum(
            point.in_frame for pose in frame.poses for point in pose.keypoints
        ),
        high_confidence_keypoint_count=sum(high_confidence_counts),
    )


def _transition_quality(
    previous: PoseFrame,
    current: PoseFrame,
    config: PoseQualityConfig,
) -> tuple[PoseTransitionQuality, list[float], list[float]]:
    pairs = _anonymous_pairs(previous, current, minimum_iou=config.min_pair_bbox_iou)
    image_pair_medians: list[float] = []
    shape_pair_medians: list[float] = []
    insufficient = 0
    for previous_pose, current_pose in pairs:
        image_displacements, shape_displacements = _shared_keypoint_displacements(
            previous_pose,
            current_pose,
            confidence_threshold=config.keypoint_confidence_threshold,
        )
        if len(image_displacements) < config.min_shared_keypoints:
            insufficient += 1
            continue
        image_pair_medians.append(_percentile(image_displacements, 0.5) or 0.0)
        shape_pair_medians.append(_percentile(shape_displacements, 0.5) or 0.0)

    comparable = len(image_pair_medians)
    transition = PoseTransitionQuality(
        from_frame_id=previous.frame_id,
        to_frame_id=current.frame_id,
        delta_seconds=_clean(current.timestamp_seconds - previous.timestamp_seconds),
        from_pose_count=len(previous.poses),
        to_pose_count=len(current.poses),
        matched_pose_pairs=len(pairs),
        comparable_pose_pairs=comparable,
        insufficient_shared_keypoint_pairs=insufficient,
        unmatched_from_poses=len(previous.poses) - len(pairs),
        unmatched_to_poses=len(current.poses) - len(pairs),
        image_change_pairs=sum(
            value > config.image_change_threshold for value in image_pair_medians
        ),
        shape_change_pairs=sum(
            value > config.shape_change_threshold for value in shape_pair_medians
        ),
        image_displacement_median=_percentile(image_pair_medians, 0.5),
        image_displacement_p95=_percentile(image_pair_medians, 0.95),
        shape_displacement_median=_percentile(shape_pair_medians, 0.5),
        shape_displacement_p95=_percentile(shape_pair_medians, 0.95),
    )
    return transition, image_pair_medians, shape_pair_medians


def _evidence_status(
    *,
    frame_count: int,
    pose_count: int,
    matched_pose_pairs: int,
    comparable_pose_pairs: int,
) -> str:
    if pose_count == 0:
        return "no_pose_evidence"
    if frame_count == 1:
        return "single_frame_only"
    if matched_pose_pairs == 0:
        return "no_anonymous_pairs"
    if comparable_pose_pairs == 0:
        return "insufficient_shared_keypoints"
    return "temporal_measurements_available"


def evaluate_pose_quality(
    batch: PoseFrameBatch,
    config: PoseQualityConfig | None = None,
) -> PoseQualityReport:
    """Measure path-free pose coverage and pairwise coordinate stability.

    Pairing is only a deterministic bbox-overlap comparison between adjacent frames. It is not
    a persistent track, a person identity, a posture classifier, or a risk inference.
    """

    if not isinstance(batch, PoseFrameBatch):
        raise TypeError("batch must be a PoseFrameBatch")
    settings = config or PoseQualityConfig()
    if not isinstance(settings, PoseQualityConfig):
        raise TypeError("config must be a PoseQualityConfig")

    frames = tuple(_frame_quality(frame, settings) for frame in batch.frames)
    transitions: list[PoseTransitionQuality] = []
    image_pair_medians: list[float] = []
    shape_pair_medians: list[float] = []
    for previous, current in zip(batch.frames, batch.frames[1:], strict=False):
        transition, image_values, shape_values = _transition_quality(
            previous,
            current,
            settings,
        )
        transitions.append(transition)
        image_pair_medians.extend(image_values)
        shape_pair_medians.extend(shape_values)

    pose_count = sum(item.pose_count for item in frames)
    usable_pose_count = sum(item.usable_pose_count for item in frames)
    matched_pairs = sum(item.matched_pose_pairs for item in transitions)
    comparable_pairs = sum(item.comparable_pose_pairs for item in transitions)
    summary = PoseQualitySummary(
        evidence_status=_evidence_status(
            frame_count=len(frames),
            pose_count=pose_count,
            matched_pose_pairs=matched_pairs,
            comparable_pose_pairs=comparable_pairs,
        ),
        frame_count=len(frames),
        pose_count=pose_count,
        usable_pose_count=usable_pose_count,
        frames_with_poses=sum(item.pose_count > 0 for item in frames),
        frames_with_usable_poses=sum(item.usable_pose_count > 0 for item in frames),
        usable_pose_rate=_ratio(usable_pose_count, pose_count),
        transition_count=len(transitions),
        matched_pose_pairs=matched_pairs,
        comparable_pose_pairs=comparable_pairs,
        insufficient_shared_keypoint_pairs=sum(
            item.insufficient_shared_keypoint_pairs for item in transitions
        ),
        temporal_comparability_rate=_ratio(comparable_pairs, matched_pairs),
        image_change_pairs=sum(item.image_change_pairs for item in transitions),
        shape_change_pairs=sum(item.shape_change_pairs for item in transitions),
        image_displacement_median=_percentile(image_pair_medians, 0.5),
        image_displacement_p95=_percentile(image_pair_medians, 0.95),
        shape_displacement_median=_percentile(shape_pair_medians, 0.5),
        shape_displacement_p95=_percentile(shape_pair_medians, 0.95),
    )
    return PoseQualityReport(
        camera_id=batch.camera_id,
        pose_adapter=batch.pose_adapter,
        model_id=batch.model_id,
        model_revision=batch.model_revision,
        model_sha256=batch.model_sha256,
        pose_batch_sha256=hashlib.sha256(batch.to_json().encode("utf-8")).hexdigest(),
        config=settings,
        frames=frames,
        transitions=tuple(transitions),
        summary=summary,
    )
