from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .object_tracker import (
    MAX_PENDING_OBSERVATIONS,
    MAX_PROCESSED_FRAMES,
    MAX_TRACKER_FILE_BYTES,
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
    ProcessedFrame,
)
from .temporal_zone import TrackedObjectObservation

MOTION_TRACKER_ADAPTER_ID = "motion_two_stage_tracker_v1"
MOTION_TRACKER_REVISION = "constant_velocity_global_match_v2"
MAX_MOTION_CONFIG_BYTES = 1024 * 1024
MAX_MOTION_ACTIVE_TRACKS = 512
MAX_MOTION_FRAME_DETECTIONS = 512

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MotionTrackerError(ValueError):
    """A motion tracker config or association input is invalid."""


def _strict_json(raw: str, context: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MotionTrackerError(f"{context} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                MotionTrackerError(f"{context} contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise MotionTrackerError(f"{context} is invalid JSON") from exc


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise MotionTrackerError(f"{field} is outside the allowed range")
    return float(value)


def _integer(value: Any, field: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise MotionTrackerError(f"{field} is outside the allowed range")
    return value


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise MotionTrackerError(f"{field} is invalid")
    return value


def _safe_label(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 80
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise MotionTrackerError(f"{field} is invalid")
    return value


def _bbox(
    value: Any,
    field: str,
) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise MotionTrackerError(f"{field} must contain four normalized numbers")
    parsed = tuple(_number(item, field, minimum=0.0, maximum=1.0) for item in value)
    x1, y1, x2, y2 = parsed
    if x1 >= x2 or y1 >= y2:
        raise MotionTrackerError(f"{field} must have positive normalized area")
    return parsed


def _aware(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise MotionTrackerError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise MotionTrackerError(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise MotionTrackerError(f"{field} must be an ISO 8601 timestamp") from exc
    return _aware(parsed, field)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class MotionTrackerConfig:
    high_confidence_threshold: float = 0.65
    low_confidence_threshold: float = 0.35
    primary_match_threshold: float = 0.30
    secondary_match_threshold: float = 0.20
    iou_weight: float = 0.75
    max_normalized_center_distance: float = 1.50
    max_area_ratio: float = 6.0
    max_track_age_seconds: float = 2.0
    max_prediction_seconds: float = 1.0
    max_active_tracks: int = 128
    max_frame_detections: int = 128
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise MotionTrackerError("unsupported motion tracker config version")
        for field, minimum, maximum in (
            ("high_confidence_threshold", 0.01, 1.0),
            ("low_confidence_threshold", 0.0, 0.99),
            ("primary_match_threshold", 0.01, 1.0),
            ("secondary_match_threshold", 0.01, 1.0),
            ("iou_weight", 0.0, 1.0),
            ("max_normalized_center_distance", 0.1, 10.0),
            ("max_area_ratio", 1.0, 100.0),
            ("max_track_age_seconds", 0.1, 300.0),
            ("max_prediction_seconds", 0.0, 300.0),
        ):
            object.__setattr__(
                self,
                field,
                _number(getattr(self, field), field, minimum=minimum, maximum=maximum),
            )
        if self.low_confidence_threshold >= self.high_confidence_threshold:
            raise MotionTrackerError(
                "low_confidence_threshold must be below high_confidence_threshold"
            )
        if self.secondary_match_threshold > self.primary_match_threshold:
            raise MotionTrackerError(
                "secondary_match_threshold cannot exceed primary_match_threshold"
            )
        if self.max_prediction_seconds > self.max_track_age_seconds:
            raise MotionTrackerError("max_prediction_seconds cannot exceed max_track_age_seconds")
        _integer(
            self.max_active_tracks,
            "max_active_tracks",
            maximum=MAX_MOTION_ACTIVE_TRACKS,
        )
        _integer(
            self.max_frame_detections,
            "max_frame_detections",
            maximum=MAX_MOTION_FRAME_DETECTIONS,
        )

    @classmethod
    def from_dict(cls, value: Any) -> MotionTrackerConfig:
        fields = {
            "version",
            "high_confidence_threshold",
            "low_confidence_threshold",
            "primary_match_threshold",
            "secondary_match_threshold",
            "iou_weight",
            "max_normalized_center_distance",
            "max_area_ratio",
            "max_track_age_seconds",
            "max_prediction_seconds",
            "max_active_tracks",
            "max_frame_detections",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise MotionTrackerError("motion tracker config fields do not match v1")
        return cls(**value)

    @classmethod
    def read(cls, path: str | Path) -> MotionTrackerConfig:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_MOTION_CONFIG_BYTES:
                raise MotionTrackerError("motion tracker config exceeds the 1 MiB limit")
            raw = source.read_text(encoding="utf-8")
        except MotionTrackerError:
            raise
        except (OSError, UnicodeError) as exc:
            raise MotionTrackerError("motion tracker config is missing or unsafe") from exc
        return cls.from_dict(_strict_json(raw, "motion tracker config"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "high_confidence_threshold": self.high_confidence_threshold,
            "low_confidence_threshold": self.low_confidence_threshold,
            "primary_match_threshold": self.primary_match_threshold,
            "secondary_match_threshold": self.secondary_match_threshold,
            "iou_weight": self.iou_weight,
            "max_normalized_center_distance": self.max_normalized_center_distance,
            "max_area_ratio": self.max_area_ratio,
            "max_track_age_seconds": self.max_track_age_seconds,
            "max_prediction_seconds": self.max_prediction_seconds,
            "max_active_tracks": self.max_active_tracks,
            "max_frame_detections": self.max_frame_detections,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class MotionTrackAssignment:
    frame_id: str
    detection_id: str
    tracking_id: str


@dataclass(frozen=True, slots=True)
class MotionTrackSample:
    occurred_at: datetime
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "occurred_at",
            _aware(self.occurred_at, "motion sample occurred_at"),
        )
        object.__setattr__(
            self,
            "confidence",
            _number(
                self.confidence,
                "motion sample confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "bbox_xyxy",
            _bbox(self.bbox_xyxy, "motion sample bbox"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> MotionTrackSample:
        if not isinstance(value, dict) or set(value) != {
            "occurred_at",
            "confidence",
            "bbox_xyxy",
        }:
            raise MotionTrackerError("motion track sample fields do not match v1")
        return cls(
            occurred_at=_timestamp(
                value["occurred_at"],
                "motion sample occurred_at",
            ),
            confidence=value["confidence"],
            bbox_xyxy=_bbox(value["bbox_xyxy"], "motion sample bbox"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "occurred_at": self.occurred_at.isoformat(),
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
        }


@dataclass(frozen=True, slots=True)
class MotionActiveTrack:
    tracking_id: str
    label: str
    current: MotionTrackSample
    previous: MotionTrackSample | None = None

    def __post_init__(self) -> None:
        _safe_id(self.tracking_id, "motion active tracking_id")
        _safe_label(self.label, "motion active label")
        if not isinstance(self.current, MotionTrackSample):
            raise MotionTrackerError("motion active current sample is invalid")
        if self.previous is not None and (
            not isinstance(self.previous, MotionTrackSample)
            or self.previous.occurred_at >= self.current.occurred_at
        ):
            raise MotionTrackerError("motion active sample history is invalid")

    @classmethod
    def from_dict(cls, value: Any) -> MotionActiveTrack:
        if not isinstance(value, dict) or set(value) != {
            "tracking_id",
            "label",
            "current",
            "previous",
        }:
            raise MotionTrackerError("motion active track fields do not match v1")
        previous = value["previous"]
        return cls(
            tracking_id=value["tracking_id"],
            label=value["label"],
            current=MotionTrackSample.from_dict(value["current"]),
            previous=(MotionTrackSample.from_dict(previous) if previous is not None else None),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tracking_id": self.tracking_id,
            "label": self.label,
            "current": self.current.to_dict(),
            "previous": self.previous.to_dict() if self.previous else None,
        }

    def update(
        self,
        detection: FrameDetection,
        occurred_at: datetime,
    ) -> MotionActiveTrack:
        return MotionActiveTrack(
            tracking_id=self.tracking_id,
            label=self.label,
            previous=self.current,
            current=MotionTrackSample(
                occurred_at=occurred_at,
                confidence=detection.confidence,
                bbox_xyxy=detection.bbox_xyxy,
            ),
        )


@dataclass(frozen=True, slots=True)
class MotionTrackerState:
    camera_id: str
    detector_adapter: str
    config_fingerprint: str
    last_frame_at: datetime | None = None
    active_tracks: tuple[MotionActiveTrack, ...] = ()
    processed_frames: tuple[ProcessedFrame, ...] = ()
    pending_observations: tuple[TrackedObjectObservation, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise MotionTrackerError("unsupported motion tracker state version")
        _safe_id(self.camera_id, "motion state camera_id")
        _safe_id(self.detector_adapter, "motion state detector_adapter")
        if (
            not isinstance(self.config_fingerprint, str)
            or _SHA256.fullmatch(self.config_fingerprint) is None
        ):
            raise MotionTrackerError("motion state config fingerprint is invalid")
        if self.last_frame_at is not None:
            object.__setattr__(
                self,
                "last_frame_at",
                _aware(self.last_frame_at, "motion state last_frame_at"),
            )
        if (
            not isinstance(self.active_tracks, tuple)
            or len(self.active_tracks) > MAX_MOTION_ACTIVE_TRACKS
            or any(not isinstance(track, MotionActiveTrack) for track in self.active_tracks)
            or tuple(sorted(track.tracking_id for track in self.active_tracks))
            != tuple(track.tracking_id for track in self.active_tracks)
            or len({track.tracking_id for track in self.active_tracks}) != len(self.active_tracks)
        ):
            raise MotionTrackerError("motion state active tracks are invalid")
        if (
            not isinstance(self.processed_frames, tuple)
            or len(self.processed_frames) > MAX_PROCESSED_FRAMES
            or any(not isinstance(frame, ProcessedFrame) for frame in self.processed_frames)
            or len({frame.frame_id for frame in self.processed_frames})
            != len(self.processed_frames)
            or any(
                current.occurred_at <= previous.occurred_at
                for previous, current in zip(
                    self.processed_frames,
                    self.processed_frames[1:],
                    strict=False,
                )
            )
        ):
            raise MotionTrackerError("motion state processed frames are invalid")
        if (
            not isinstance(self.pending_observations, tuple)
            or len(self.pending_observations) > MAX_PENDING_OBSERVATIONS
            or any(
                not isinstance(observation, TrackedObjectObservation)
                for observation in self.pending_observations
            )
            or len({observation.observation_id for observation in self.pending_observations})
            != len(self.pending_observations)
        ):
            raise MotionTrackerError("motion state pending observations are invalid")
        if any(
            current.occurred_at < previous.occurred_at
            for previous, current in zip(
                self.pending_observations,
                self.pending_observations[1:],
                strict=False,
            )
        ):
            raise MotionTrackerError("motion state pending observations are not chronological")
        if any(
            observation.camera_id != self.camera_id
            or observation.source_adapter != MOTION_TRACKER_ADAPTER_ID
            for observation in self.pending_observations
        ):
            raise MotionTrackerError(
                "motion state pending observations have the wrong source identity"
            )
        if self.last_frame_at is None:
            if self.active_tracks or self.processed_frames or self.pending_observations:
                raise MotionTrackerError("empty motion tracker timeline contains state")
        else:
            if not self.processed_frames:
                raise MotionTrackerError("motion tracker timeline is missing processed frames")
            if self.processed_frames[-1].occurred_at != self.last_frame_at:
                raise MotionTrackerError("motion tracker last_frame_at is inconsistent")
            if any(track.current.occurred_at > self.last_frame_at for track in self.active_tracks):
                raise MotionTrackerError("motion active track is newer than the tracker timeline")
            if any(
                observation.occurred_at > self.last_frame_at
                for observation in self.pending_observations
            ):
                raise MotionTrackerError("motion observation is newer than the tracker timeline")

    @classmethod
    def empty(
        cls,
        *,
        camera_id: str,
        detector_adapter: str,
        config_fingerprint: str,
    ) -> MotionTrackerState:
        return cls(
            camera_id=camera_id,
            detector_adapter=detector_adapter,
            config_fingerprint=config_fingerprint,
        )

    @classmethod
    def from_dict(cls, value: Any) -> MotionTrackerState:
        if not isinstance(value, dict) or set(value) != {
            "version",
            "camera_id",
            "detector_adapter",
            "config_fingerprint",
            "last_frame_at",
            "active_tracks",
            "processed_frames",
            "pending_observations",
        }:
            raise MotionTrackerError("motion tracker state fields do not match v1")
        active = value["active_tracks"]
        processed = value["processed_frames"]
        pending = value["pending_observations"]
        if not all(isinstance(item, list) for item in (active, processed, pending)):
            raise MotionTrackerError("motion tracker state arrays are invalid")
        last_frame_at = value["last_frame_at"]
        if last_frame_at is not None:
            last_frame_at = _timestamp(
                last_frame_at,
                "motion state last_frame_at",
            )
        return cls(
            version=value["version"],
            camera_id=value["camera_id"],
            detector_adapter=value["detector_adapter"],
            config_fingerprint=value["config_fingerprint"],
            last_frame_at=last_frame_at,
            active_tracks=tuple(MotionActiveTrack.from_dict(item) for item in active),
            processed_frames=tuple(ProcessedFrame.from_dict(item) for item in processed),
            pending_observations=tuple(
                TrackedObjectObservation.from_dict(item) for item in pending
            ),
        )

    @classmethod
    def read(cls, path: str | Path) -> MotionTrackerState:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_TRACKER_FILE_BYTES:
                raise MotionTrackerError("motion tracker state exceeds the 4 MiB limit")
            raw = source.read_text(encoding="utf-8")
        except MotionTrackerError:
            raise
        except (OSError, UnicodeError) as exc:
            raise MotionTrackerError("motion tracker state is missing or unsafe") from exc
        return cls.from_dict(_strict_json(raw, "motion tracker state"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "detector_adapter": self.detector_adapter,
            "config_fingerprint": self.config_fingerprint,
            "last_frame_at": (self.last_frame_at.isoformat() if self.last_frame_at else None),
            "active_tracks": [track.to_dict() for track in self.active_tracks],
            "processed_frames": [frame.to_dict() for frame in self.processed_frames],
            "pending_observations": [
                observation.to_dict() for observation in self.pending_observations
            ],
        }

    def _encoded(self) -> bytes:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        if len(payload) + 1 > MAX_TRACKER_FILE_BYTES:
            raise MotionTrackerError("motion tracker state exceeds the 4 MiB limit")
        return payload + b"\n"

    def write_atomic(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        parent = target.parent
        if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
            raise MotionTrackerError("motion tracker state destination is unsafe")
        payload = self._encoded()
        temporary = parent / (f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "posix":
                with suppress(OSError):
                    temporary.chmod(0o600)
            os.replace(temporary, target)
            if os.name == "posix":
                directory_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except OSError as exc:
            raise MotionTrackerError("motion tracker state could not be written") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(OSError):
                temporary.unlink()

    def acknowledge(
        self,
        observation_ids: tuple[str, ...],
    ) -> MotionTrackerState:
        if (
            not isinstance(observation_ids, tuple)
            or not observation_ids
            or len(set(observation_ids)) != len(observation_ids)
        ):
            raise MotionTrackerError(
                "motion observation acknowledgements must be non-empty and unique"
            )
        for observation_id in observation_ids:
            _safe_id(observation_id, "acknowledged motion observation_id")
        known = {observation.observation_id for observation in self.pending_observations}
        if not set(observation_ids).issubset(known):
            raise MotionTrackerError("motion acknowledgement contains an unknown observation_id")
        acknowledged = set(observation_ids)
        state = replace(
            self,
            pending_observations=tuple(
                observation
                for observation in self.pending_observations
                if observation.observation_id not in acknowledged
            ),
        )
        state._encoded()
        return state


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _geometry(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def _bbox_from_geometry(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    width = min(1.0, max(1e-9, width))
    height = min(1.0, max(1e-9, height))
    center_x = min(1.0 - width / 2.0, max(width / 2.0, center_x))
    center_y = min(1.0 - height / 2.0, max(height / 2.0, center_y))
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _raw_prediction_geometry(
    track: MotionActiveTrack,
    occurred_at: datetime,
    max_prediction_seconds: float,
) -> tuple[float, float, float, float]:
    if track.previous is None or max_prediction_seconds == 0.0:
        return _geometry(track.current.bbox_xyxy)
    history_seconds = (track.current.occurred_at - track.previous.occurred_at).total_seconds()
    if history_seconds <= 0.0:
        raise MotionTrackerError("motion tracker history is not chronological")
    future_seconds = min(
        max_prediction_seconds,
        max(0.0, (occurred_at - track.current.occurred_at).total_seconds()),
    )
    previous = _geometry(track.previous.bbox_xyxy)
    current = _geometry(track.current.bbox_xyxy)
    return tuple(
        current[index] + (current[index] - previous[index]) / history_seconds * future_seconds
        for index in range(4)
    )


def _prediction_fallback_required(track, occurred_at, max_prediction_seconds) -> bool:
    predicted = _raw_prediction_geometry(track, occurred_at, max_prediction_seconds)
    return (
        not all(math.isfinite(value) for value in predicted)
        or predicted[2] <= 0.0
        or predicted[3] <= 0.0
    )


def _predict_bbox_without_fallback(track, occurred_at, max_prediction_seconds):
    """Original predictor, retained for qualified-edge protection and comparisons."""
    if track.previous is None or max_prediction_seconds == 0.0:
        return track.current.bbox_xyxy
    return _bbox_from_geometry(
        *_raw_prediction_geometry(track, occurred_at, max_prediction_seconds)
    )


def _predict_bbox(track, occurred_at, max_prediction_seconds):
    if track.previous is None or max_prediction_seconds == 0.0:
        return track.current.bbox_xyxy
    predicted = _raw_prediction_geometry(track, occurred_at, max_prediction_seconds)
    # A linear width/height extrapolation can cross zero after abrupt box
    # shrinkage or growth. Clamping that invalid prediction to 1e-9 turns a
    # recoverable track into a near-point and makes its match score zero.
    invalid_extent = (
        not all(math.isfinite(value) for value in predicted)
        or predicted[2] <= 0.0
        or predicted[3] <= 0.0
    )
    if invalid_extent:
        return track.current.bbox_xyxy
    return _bbox_from_geometry(*predicted)


def _match_score(
    track: MotionActiveTrack,
    detection: FrameDetection,
    *,
    occurred_at: datetime,
    config: MotionTrackerConfig,
    use_fallback: bool = True,
) -> float:
    if track.label != detection.label:
        return 0.0
    predictor = _predict_bbox if use_fallback else _predict_bbox_without_fallback
    predicted = predictor(track, occurred_at, config.max_prediction_seconds)
    predicted_geometry = _geometry(predicted)
    detection_geometry = _geometry(detection.bbox_xyxy)
    predicted_area = predicted_geometry[2] * predicted_geometry[3]
    detection_area = detection_geometry[2] * detection_geometry[3]
    area_ratio = max(predicted_area, detection_area) / min(
        predicted_area,
        detection_area,
    )
    if area_ratio > config.max_area_ratio:
        return 0.0
    center_distance = math.hypot(
        predicted_geometry[0] - detection_geometry[0],
        predicted_geometry[1] - detection_geometry[1],
    )
    scale = max(
        math.hypot(predicted_geometry[2], predicted_geometry[3]),
        math.hypot(detection_geometry[2], detection_geometry[3]),
        1e-9,
    )
    normalized_distance = center_distance / scale
    if normalized_distance > config.max_normalized_center_distance:
        return 0.0
    center_similarity = 1.0 - (normalized_distance / config.max_normalized_center_distance)
    iou = _bbox_iou(predicted, detection.bbox_xyxy)
    return config.iou_weight * iou + (1.0 - config.iou_weight) * center_similarity


def _maximum_weight_assignment(
    scores: list[list[float]],
    *,
    minimum_score: float = 0.0,
) -> tuple[tuple[int, int], ...]:
    if not scores or not scores[0]:
        return ()
    row_count = len(scores)
    column_count = len(scores[0])
    if any(len(row) != column_count for row in scores):
        raise MotionTrackerError("motion tracker score matrix is ragged")
    transposed = row_count > column_count
    eligible_scores = [
        [score if score + 1e-15 >= minimum_score else 0.0 for score in row] for row in scores
    ]
    matrix = (
        [
            [eligible_scores[row][column] for row in range(row_count)]
            for column in range(column_count)
        ]
        if transposed
        else eligible_scores
    )
    rows = len(matrix)
    columns = len(matrix[0])
    potentials_rows = [0.0] * (rows + 1)
    potentials_columns = [0.0] * (columns + 1)
    column_match = [0] * (columns + 1)
    previous_column = [0] * (columns + 1)
    epsilon = 1e-15

    for row in range(1, rows + 1):
        column_match[0] = row
        current_column = 0
        minimum = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[current_column] = True
            current_row = column_match[current_column]
            delta = math.inf
            next_column = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                cost = 1.0 - matrix[current_row - 1][column - 1]
                reduced = cost - potentials_rows[current_row] - potentials_columns[column]
                if reduced < minimum[column] - epsilon:
                    minimum[column] = reduced
                    previous_column[column] = current_column
                if minimum[column] < delta - epsilon:
                    delta = minimum[column]
                    next_column = column
            if not math.isfinite(delta):
                raise MotionTrackerError("motion tracker assignment is infeasible")
            for column in range(columns + 1):
                if used[column]:
                    potentials_rows[column_match[column]] += delta
                    potentials_columns[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if column_match[current_column] == 0:
                break
        while True:
            next_column = previous_column[current_column]
            column_match[current_column] = column_match[next_column]
            current_column = next_column
            if current_column == 0:
                break

    pairs: list[tuple[int, int]] = []
    for column in range(1, columns + 1):
        if column_match[column] == 0:
            continue
        row_index = column_match[column] - 1
        column_index = column - 1
        pairs.append((column_index, row_index) if transposed else (row_index, column_index))
    return tuple(
        pair for pair in sorted(pairs) if scores[pair[0]][pair[1]] + 1e-15 >= minimum_score
    )


def _matching_scores(
    tracks: tuple[MotionActiveTrack, ...],
    detections: tuple[FrameDetection, ...],
    *,
    occurred_at: datetime,
    minimum_score: float,
    config: MotionTrackerConfig,
) -> list[list[float]]:
    if not tracks or not detections:
        return [[] for _ in tracks]
    scores = [
        [
            _match_score(
                track,
                detection,
                occurred_at=occurred_at,
                config=config,
                use_fallback=False,
            )
            for detection in detections
        ]
        for track in tracks
    ]
    fallback_rows = tuple(
        _prediction_fallback_required(track, occurred_at, config.max_prediction_seconds)
        for track in tracks
    )
    if not any(fallback_rows):
        return scores
    protected = tuple(
        any(row[j] > 0.0 and row[j] + 1e-15 >= minimum_score for row in scores)
        for j in range(len(detections))
    )
    # Keep every original edge of an already-qualified detection, including
    # ties. A fallback must not let a dormant track steal that detection.
    for i, track in enumerate(tracks):
        if not fallback_rows[i]:
            continue
        for j, detection in enumerate(detections):
            if not protected[j]:
                scores[i][j] = _match_score(
                    track, detection, occurred_at=occurred_at, config=config
                )
    return scores


def _match(tracks, detections, *, occurred_at, minimum_score, config):
    return _maximum_weight_assignment(
        _matching_scores(
            tracks,
            detections,
            occurred_at=occurred_at,
            minimum_score=minimum_score,
            config=config,
        ),
        minimum_score=minimum_score,
    )


def _new_tracking_id(
    batch: DetectionFrameBatch,
    config: MotionTrackerConfig,
    *,
    frame_id: str,
    detection: FrameDetection,
    retained: bool,
) -> str:
    payload = {
        "tracker_adapter": MOTION_TRACKER_ADAPTER_ID,
        "tracker_revision": MOTION_TRACKER_REVISION,
        "camera_id": batch.camera_id,
        "detector_adapter": batch.detector_adapter,
        "tracker_config_sha256": config.fingerprint,
        "frame_id": frame_id,
        "detection": detection.to_dict(),
        "retained": retained,
    }
    return f"track_{_canonical_sha256(payload)[:24]}"


class MotionObjectTracker:
    """Persistent anonymous tracker using the evaluated two-stage matcher."""

    def __init__(
        self,
        *,
        config: MotionTrackerConfig,
        camera_id: str,
        detector_adapter: str,
        state: MotionTrackerState | None = None,
    ):
        if not isinstance(config, MotionTrackerConfig):
            raise MotionTrackerError("motion tracker config is invalid")
        _safe_id(camera_id, "camera_id")
        _safe_id(detector_adapter, "detector_adapter")
        self.config = config
        self.state = state or MotionTrackerState.empty(
            camera_id=camera_id,
            detector_adapter=detector_adapter,
            config_fingerprint=config.fingerprint,
        )
        if self.state.camera_id != camera_id or self.state.detector_adapter != detector_adapter:
            raise MotionTrackerError("motion tracker state belongs to a different input source")
        if self.state.config_fingerprint != config.fingerprint:
            raise MotionTrackerError("motion tracker state belongs to a different config")
        if len(self.state.active_tracks) > config.max_active_tracks:
            raise MotionTrackerError("motion tracker state exceeds configured active track limit")

    def _observation(
        self,
        *,
        frame: DetectionFrame,
        detection: FrameDetection,
        tracking_id: str,
    ) -> TrackedObjectObservation:
        digest = _canonical_sha256(
            {
                "version": 1,
                "camera_id": self.state.camera_id,
                "detector_adapter": self.state.detector_adapter,
                "frame_id": frame.frame_id,
                "occurred_at": frame.occurred_at.isoformat(),
                "detection": detection.to_dict(),
                "tracking_id": tracking_id,
            }
        )
        return TrackedObjectObservation(
            observation_id=f"observation_{digest[:24]}",
            camera_id=self.state.camera_id,
            occurred_at=frame.occurred_at,
            source_adapter=MOTION_TRACKER_ADAPTER_ID,
            tracking_id=tracking_id,
            label=detection.label,
            confidence=detection.confidence,
            bbox_xyxy=detection.bbox_xyxy,
        )

    def process(
        self,
        frame: DetectionFrame,
    ) -> tuple[TrackedObjectObservation, ...]:
        if not isinstance(frame, DetectionFrame):
            raise MotionTrackerError("motion tracker input must be a DetectionFrame")
        existing = next(
            (item for item in self.state.processed_frames if item.frame_id == frame.frame_id),
            None,
        )
        if existing is not None:
            if existing.fingerprint != frame.fingerprint:
                raise MotionTrackerError("motion frame_id was reused with conflicting content")
            return ()
        if self.state.last_frame_at is not None and frame.occurred_at <= self.state.last_frame_at:
            raise MotionTrackerError("new motion tracker frames must be strictly monotonic")
        if len(frame.detections) > self.config.max_frame_detections:
            raise MotionTrackerError("motion tracker frame exceeds max_frame_detections")

        retained_active = {
            track.tracking_id: track
            for track in self.state.active_tracks
            if (frame.occurred_at - track.current.occurred_at).total_seconds()
            <= self.config.max_track_age_seconds
        }
        ordered_tracks = tuple(retained_active[key] for key in sorted(retained_active))
        high = tuple(
            detection
            for detection in frame.detections
            if detection.confidence >= self.config.high_confidence_threshold
        )
        low = tuple(
            detection
            for detection in frame.detections
            if self.config.low_confidence_threshold
            <= detection.confidence
            < self.config.high_confidence_threshold
        )
        ignored = tuple(
            detection
            for detection in frame.detections
            if detection.confidence < self.config.low_confidence_threshold
        )

        primary_matches = _match(
            ordered_tracks,
            high,
            occurred_at=frame.occurred_at,
            minimum_score=self.config.primary_match_threshold,
            config=self.config,
        )
        matched_track_indexes = {item[0] for item in primary_matches}
        matched_high_indexes = {item[1] for item in primary_matches}
        remaining_tracks = tuple(
            track
            for index, track in enumerate(ordered_tracks)
            if index not in matched_track_indexes
        )
        secondary_matches = _match(
            remaining_tracks,
            low,
            occurred_at=frame.occurred_at,
            minimum_score=self.config.secondary_match_threshold,
            config=self.config,
        )
        matched_low_indexes = {item[1] for item in secondary_matches}

        detection_tracks: dict[str, str] = {}
        next_active = dict(retained_active)
        for track_index, detection_index in primary_matches:
            track = ordered_tracks[track_index]
            detection = high[detection_index]
            next_active[track.tracking_id] = track.update(
                detection,
                frame.occurred_at,
            )
            detection_tracks[detection.detection_id] = track.tracking_id
        for track_index, detection_index in secondary_matches:
            track = remaining_tracks[track_index]
            detection = low[detection_index]
            next_active[track.tracking_id] = track.update(
                detection,
                frame.occurred_at,
            )
            detection_tracks[detection.detection_id] = track.tracking_id

        source_batch = DetectionFrameBatch(
            camera_id=self.state.camera_id,
            detector_adapter=self.state.detector_adapter,
            frames=(frame,),
        )
        issued_ids = {track.tracking_id for track in self.state.active_tracks} | {
            observation.tracking_id for observation in self.state.pending_observations
        }
        unmatched_high = tuple(
            detection for index, detection in enumerate(high) if index not in matched_high_indexes
        )
        if len(next_active) + len(unmatched_high) > self.config.max_active_tracks:
            raise MotionTrackerError("motion tracker active track limit would be exceeded")
        for detection in unmatched_high:
            tracking_id = _new_tracking_id(
                source_batch,
                self.config,
                frame_id=frame.frame_id,
                detection=detection,
                retained=True,
            )
            if tracking_id in issued_ids or tracking_id in next_active:
                raise MotionTrackerError("motion tracker deterministic ID collision")
            issued_ids.add(tracking_id)
            next_active[tracking_id] = MotionActiveTrack(
                tracking_id=tracking_id,
                label=detection.label,
                current=MotionTrackSample(
                    occurred_at=frame.occurred_at,
                    confidence=detection.confidence,
                    bbox_xyxy=detection.bbox_xyxy,
                ),
            )
            detection_tracks[detection.detection_id] = tracking_id

        unmatched_nonretained = (
            tuple(
                detection for index, detection in enumerate(low) if index not in matched_low_indexes
            )
            + ignored
        )
        for detection in unmatched_nonretained:
            tracking_id = _new_tracking_id(
                source_batch,
                self.config,
                frame_id=frame.frame_id,
                detection=detection,
                retained=False,
            )
            if tracking_id in issued_ids or tracking_id in next_active:
                raise MotionTrackerError("motion tracker deterministic ID collision")
            issued_ids.add(tracking_id)
            detection_tracks[detection.detection_id] = tracking_id

        if set(detection_tracks) != {detection.detection_id for detection in frame.detections}:
            raise MotionTrackerError("motion tracker did not assign every detection exactly once")
        observations = tuple(
            self._observation(
                frame=frame,
                detection=detection,
                tracking_id=detection_tracks[detection.detection_id],
            )
            for detection in frame.detections
        )
        if len(self.state.pending_observations) + len(observations) > MAX_PENDING_OBSERVATIONS:
            raise MotionTrackerError("motion tracker pending observation outbox is full")

        processed = (
            *self.state.processed_frames,
            ProcessedFrame(
                frame_id=frame.frame_id,
                occurred_at=frame.occurred_at,
                fingerprint=frame.fingerprint,
            ),
        )[-MAX_PROCESSED_FRAMES:]
        candidate_state = replace(
            self.state,
            last_frame_at=frame.occurred_at,
            active_tracks=tuple(sorted(next_active.values(), key=lambda item: item.tracking_id)),
            processed_frames=processed,
            pending_observations=(
                *self.state.pending_observations,
                *observations,
            ),
        )
        candidate_state._encoded()
        self.state = candidate_state
        return observations


def build_motion_track_assignments(
    batch: DetectionFrameBatch,
    config: MotionTrackerConfig,
) -> tuple[MotionTrackAssignment, ...]:
    if not isinstance(batch, DetectionFrameBatch) or not isinstance(
        config,
        MotionTrackerConfig,
    ):
        raise MotionTrackerError("motion tracker inputs are invalid")

    active: dict[str, MotionActiveTrack] = {}
    assignments: dict[tuple[str, str], str] = {}
    issued_ids: set[str] = set()

    for frame in batch.frames:
        if len(frame.detections) > config.max_frame_detections:
            raise MotionTrackerError("motion tracker frame exceeds max_frame_detections")
        retained_active = {
            tracking_id: track
            for tracking_id, track in active.items()
            if (frame.occurred_at - track.current.occurred_at).total_seconds()
            <= config.max_track_age_seconds
        }
        ordered_tracks = tuple(retained_active[key] for key in sorted(retained_active))
        high = tuple(
            detection
            for detection in frame.detections
            if detection.confidence >= config.high_confidence_threshold
        )
        low = tuple(
            detection
            for detection in frame.detections
            if config.low_confidence_threshold
            <= detection.confidence
            < config.high_confidence_threshold
        )
        ignored = tuple(
            detection
            for detection in frame.detections
            if detection.confidence < config.low_confidence_threshold
        )

        primary_matches = _match(
            ordered_tracks,
            high,
            occurred_at=frame.occurred_at,
            minimum_score=config.primary_match_threshold,
            config=config,
        )
        matched_track_indexes = {item[0] for item in primary_matches}
        matched_high_indexes = {item[1] for item in primary_matches}
        remaining_tracks = tuple(
            track
            for index, track in enumerate(ordered_tracks)
            if index not in matched_track_indexes
        )
        secondary_matches = _match(
            remaining_tracks,
            low,
            occurred_at=frame.occurred_at,
            minimum_score=config.secondary_match_threshold,
            config=config,
        )
        matched_low_indexes = {item[1] for item in secondary_matches}

        next_active = dict(retained_active)
        for track_index, detection_index in primary_matches:
            track = ordered_tracks[track_index]
            detection = high[detection_index]
            next_active[track.tracking_id] = track.update(
                detection,
                frame.occurred_at,
            )
            assignments[(frame.frame_id, detection.detection_id)] = track.tracking_id
        for track_index, detection_index in secondary_matches:
            track = remaining_tracks[track_index]
            detection = low[detection_index]
            next_active[track.tracking_id] = track.update(
                detection,
                frame.occurred_at,
            )
            assignments[(frame.frame_id, detection.detection_id)] = track.tracking_id

        unmatched_high = tuple(
            detection for index, detection in enumerate(high) if index not in matched_high_indexes
        )
        if len(next_active) + len(unmatched_high) > config.max_active_tracks:
            raise MotionTrackerError("motion tracker active track limit would be exceeded")
        for detection in unmatched_high:
            tracking_id = _new_tracking_id(
                batch,
                config,
                frame_id=frame.frame_id,
                detection=detection,
                retained=True,
            )
            if tracking_id in issued_ids or tracking_id in next_active:
                raise MotionTrackerError("motion tracker deterministic ID collision")
            issued_ids.add(tracking_id)
            next_active[tracking_id] = MotionActiveTrack(
                tracking_id=tracking_id,
                label=detection.label,
                current=MotionTrackSample(
                    occurred_at=frame.occurred_at,
                    confidence=detection.confidence,
                    bbox_xyxy=detection.bbox_xyxy,
                ),
            )
            assignments[(frame.frame_id, detection.detection_id)] = tracking_id

        unmatched_nonretained = (
            tuple(
                detection for index, detection in enumerate(low) if index not in matched_low_indexes
            )
            + ignored
        )
        for detection in unmatched_nonretained:
            tracking_id = _new_tracking_id(
                batch,
                config,
                frame_id=frame.frame_id,
                detection=detection,
                retained=False,
            )
            if tracking_id in issued_ids or tracking_id in next_active:
                raise MotionTrackerError("motion tracker deterministic ID collision")
            issued_ids.add(tracking_id)
            assignments[(frame.frame_id, detection.detection_id)] = tracking_id

        active = next_active

    expected = tuple(
        (frame.frame_id, detection.detection_id)
        for frame in batch.frames
        for detection in frame.detections
    )
    if set(assignments) != set(expected):
        raise MotionTrackerError("motion tracker did not assign every detection exactly once")
    return tuple(
        MotionTrackAssignment(
            frame_id=frame_id,
            detection_id=detection_id,
            tracking_id=assignments[(frame_id, detection_id)],
        )
        for frame_id, detection_id in expected
    )
