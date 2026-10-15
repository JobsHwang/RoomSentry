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

from .temporal_zone import TrackedObjectObservation

MAX_TRACKER_FILE_BYTES = 4 * 1024 * 1024
MAX_TRACKING_FRAMES = 10_000
MAX_DETECTIONS_PER_FRAME = 1_000
MAX_TRACKING_DETECTIONS = 10_000
MAX_ACTIVE_TRACKS = 10_000
MAX_PROCESSED_FRAMES = 10_000
MAX_PENDING_OBSERVATIONS = 10_000
TRACKER_ADAPTER_ID = "iou_tracker_v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ObjectTrackerError(RuntimeError):
    pass


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ObjectTrackerError(f"{field} is invalid")
    return value


def _safe_label(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 80
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ObjectTrackerError(f"{field} is invalid")
    return value


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObjectTrackerError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ObjectTrackerError(f"{field} is outside the allowed range")
    return parsed


def _bbox(value: Any, field: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ObjectTrackerError(f"{field} must contain four normalized numbers")
    parsed = tuple(_number(item, field, minimum=0.0, maximum=1.0) for item in value)
    x1, y1, x2, y2 = parsed
    if x1 >= x2 or y1 >= y2:
        raise ObjectTrackerError(f"{field} must have positive normalized area")
    return parsed


def _aware(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ObjectTrackerError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ObjectTrackerError(f"{field} must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ObjectTrackerError(f"{field} must be an ISO 8601 timestamp") from exc
    return _aware(parsed, field)


def _strict_json(raw: str, context: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ObjectTrackerError(f"{context} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ObjectTrackerError(f"{context} contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ObjectTrackerError(f"{context} is invalid JSON") from exc


def _read_json(path: str | Path, *, context: str) -> Any:
    source = Path(path).expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        if source.stat().st_size > MAX_TRACKER_FILE_BYTES:
            raise ObjectTrackerError(f"{context} exceeds the 4 MiB limit")
        raw = source.read_text(encoding="utf-8")
    except ObjectTrackerError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ObjectTrackerError(f"{context} is missing or unsafe") from exc
    return _strict_json(raw, context)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FrameDetection:
    detection_id: str
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _safe_id(self.detection_id, "detection_id")
        _safe_label(self.label, "detection label")
        object.__setattr__(
            self,
            "confidence",
            _number(self.confidence, "detection confidence", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(self, "bbox_xyxy", _bbox(self.bbox_xyxy, "detection bbox"))

    @classmethod
    def from_dict(cls, value: Any) -> FrameDetection:
        if not isinstance(value, dict) or set(value) != {
            "detection_id",
            "label",
            "confidence",
            "bbox_xyxy",
        }:
            raise ObjectTrackerError("frame detection fields do not match v1")
        return cls(
            detection_id=value["detection_id"],
            label=value["label"],
            confidence=value["confidence"],
            bbox_xyxy=_bbox(value["bbox_xyxy"], "detection bbox"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
        }


@dataclass(frozen=True, slots=True)
class DetectionFrame:
    frame_id: str
    occurred_at: datetime
    detections: tuple[FrameDetection, ...]

    def __post_init__(self) -> None:
        _safe_id(self.frame_id, "frame_id")
        object.__setattr__(self, "occurred_at", _aware(self.occurred_at, "occurred_at"))
        if (
            not isinstance(self.detections, tuple)
            or len(self.detections) > MAX_DETECTIONS_PER_FRAME
            or any(not isinstance(item, FrameDetection) for item in self.detections)
        ):
            raise ObjectTrackerError("frame detections are invalid or exceed the limit")
        ordered = tuple(sorted(self.detections, key=lambda item: item.detection_id))
        if len({item.detection_id for item in ordered}) != len(ordered):
            raise ObjectTrackerError("frame detection IDs must be unique")
        object.__setattr__(self, "detections", ordered)

    @classmethod
    def from_dict(cls, value: Any) -> DetectionFrame:
        if not isinstance(value, dict) or set(value) != {
            "frame_id",
            "occurred_at",
            "detections",
        }:
            raise ObjectTrackerError("detection frame fields do not match v1")
        detections = value["detections"]
        if not isinstance(detections, list):
            raise ObjectTrackerError("frame detections must be an array")
        return cls(
            frame_id=value["frame_id"],
            occurred_at=_timestamp(value["occurred_at"], "occurred_at"),
            detections=tuple(FrameDetection.from_dict(item) for item in detections),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "occurred_at": self.occurred_at.isoformat(),
            "detections": [item.to_dict() for item in self.detections],
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class DetectionFrameBatch:
    camera_id: str
    detector_adapter: str
    frames: tuple[DetectionFrame, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ObjectTrackerError("unsupported detection frame batch version")
        _safe_id(self.camera_id, "camera_id")
        _safe_id(self.detector_adapter, "detector_adapter")
        if (
            not isinstance(self.frames, tuple)
            or not 1 <= len(self.frames) <= MAX_TRACKING_FRAMES
            or any(not isinstance(frame, DetectionFrame) for frame in self.frames)
        ):
            raise ObjectTrackerError("detection frames are invalid or exceed the limit")
        if len({frame.frame_id for frame in self.frames}) != len(self.frames):
            raise ObjectTrackerError("detection frame IDs must be unique")
        if any(
            current.occurred_at <= previous.occurred_at
            for previous, current in zip(self.frames, self.frames[1:], strict=False)
        ):
            raise ObjectTrackerError("detection frames must be strictly chronological")
        if sum(len(frame.detections) for frame in self.frames) > MAX_TRACKING_DETECTIONS:
            raise ObjectTrackerError("detection batch exceeds the total detection limit")

    @classmethod
    def from_dict(cls, value: Any) -> DetectionFrameBatch:
        if not isinstance(value, dict) or set(value) != {
            "version",
            "camera_id",
            "detector_adapter",
            "frames",
        }:
            raise ObjectTrackerError("detection frame batch fields do not match v1")
        frames = value["frames"]
        if not isinstance(frames, list):
            raise ObjectTrackerError("detection frames must be an array")
        return cls(
            version=value["version"],
            camera_id=value["camera_id"],
            detector_adapter=value["detector_adapter"],
            frames=tuple(DetectionFrame.from_dict(item) for item in frames),
        )

    @classmethod
    def read(cls, path: str | Path) -> DetectionFrameBatch:
        return cls.from_dict(_read_json(path, context="detection frame batch"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "detector_adapter": self.detector_adapter,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def write_new(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        parent = target.parent
        if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
            raise ObjectTrackerError("detection frame batch destination is unsafe")
        payload = f"{self.to_json()}\n".encode()
        if len(payload) > MAX_TRACKER_FILE_BYTES:
            raise ObjectTrackerError("detection frame batch exceeds the 4 MiB limit")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(target, flags, 0o600)
        except FileExistsError:
            raise FileExistsError("detection frame batch output already exists") from None
        except OSError as exc:
            raise ObjectTrackerError("detection frame batch output could not be created") from exc
        try:
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            with suppress(OSError):
                target.unlink()
            raise ObjectTrackerError("detection frame batch output could not be written") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)


@dataclass(frozen=True, slots=True)
class IoUTrackerConfig:
    iou_threshold: float = 0.3
    max_track_age_seconds: float = 2.0
    max_active_tracks: int = 128
    max_processed_frames: int = 1_000
    max_pending_observations: int = 1_000
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ObjectTrackerError("unsupported IoU tracker config version")
        object.__setattr__(
            self,
            "iou_threshold",
            _number(self.iou_threshold, "iou_threshold", minimum=0.01, maximum=1.0),
        )
        object.__setattr__(
            self,
            "max_track_age_seconds",
            _number(
                self.max_track_age_seconds,
                "max_track_age_seconds",
                minimum=0.1,
                maximum=300.0,
            ),
        )
        for value, field, maximum in (
            (self.max_active_tracks, "max_active_tracks", MAX_ACTIVE_TRACKS),
            (self.max_processed_frames, "max_processed_frames", MAX_PROCESSED_FRAMES),
            (
                self.max_pending_observations,
                "max_pending_observations",
                MAX_PENDING_OBSERVATIONS,
            ),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ObjectTrackerError(f"{field} is outside the allowed range")

    @classmethod
    def from_dict(cls, value: Any) -> IoUTrackerConfig:
        if not isinstance(value, dict) or set(value) != {
            "version",
            "iou_threshold",
            "max_track_age_seconds",
            "max_active_tracks",
            "max_processed_frames",
            "max_pending_observations",
        }:
            raise ObjectTrackerError("IoU tracker config fields do not match v1")
        return cls(
            version=value["version"],
            iou_threshold=value["iou_threshold"],
            max_track_age_seconds=value["max_track_age_seconds"],
            max_active_tracks=value["max_active_tracks"],
            max_processed_frames=value["max_processed_frames"],
            max_pending_observations=value["max_pending_observations"],
        )

    @classmethod
    def read(cls, path: str | Path) -> IoUTrackerConfig:
        return cls.from_dict(_read_json(path, context="IoU tracker config"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "iou_threshold": self.iou_threshold,
            "max_track_age_seconds": self.max_track_age_seconds,
            "max_active_tracks": self.max_active_tracks,
            "max_processed_frames": self.max_processed_frames,
            "max_pending_observations": self.max_pending_observations,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class ActiveTrack:
    tracking_id: str
    label: str
    last_seen_at: datetime
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _safe_id(self.tracking_id, "active tracking_id")
        _safe_label(self.label, "active track label")
        object.__setattr__(self, "last_seen_at", _aware(self.last_seen_at, "active last_seen_at"))
        object.__setattr__(
            self,
            "confidence",
            _number(self.confidence, "active confidence", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(self, "bbox_xyxy", _bbox(self.bbox_xyxy, "active bbox"))

    @classmethod
    def from_dict(cls, value: Any) -> ActiveTrack:
        if not isinstance(value, dict) or set(value) != {
            "tracking_id",
            "label",
            "last_seen_at",
            "confidence",
            "bbox_xyxy",
        }:
            raise ObjectTrackerError("active track fields do not match v1")
        return cls(
            tracking_id=value["tracking_id"],
            label=value["label"],
            last_seen_at=_timestamp(value["last_seen_at"], "active last_seen_at"),
            confidence=value["confidence"],
            bbox_xyxy=_bbox(value["bbox_xyxy"], "active bbox"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tracking_id": self.tracking_id,
            "label": self.label,
            "last_seen_at": self.last_seen_at.isoformat(),
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
        }


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    frame_id: str
    occurred_at: datetime
    fingerprint: str

    def __post_init__(self) -> None:
        _safe_id(self.frame_id, "processed frame_id")
        object.__setattr__(self, "occurred_at", _aware(self.occurred_at, "processed occurred_at"))
        if not isinstance(self.fingerprint, str) or _SHA256.fullmatch(self.fingerprint) is None:
            raise ObjectTrackerError("processed frame fingerprint is invalid")

    @classmethod
    def from_dict(cls, value: Any) -> ProcessedFrame:
        if not isinstance(value, dict) or set(value) != {
            "frame_id",
            "occurred_at",
            "fingerprint",
        }:
            raise ObjectTrackerError("processed frame fields do not match v1")
        return cls(
            frame_id=value["frame_id"],
            occurred_at=_timestamp(value["occurred_at"], "processed occurred_at"),
            fingerprint=value["fingerprint"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "occurred_at": self.occurred_at.isoformat(),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class IoUTrackerState:
    camera_id: str
    detector_adapter: str
    config_fingerprint: str
    last_frame_at: datetime | None = None
    active_tracks: tuple[ActiveTrack, ...] = ()
    processed_frames: tuple[ProcessedFrame, ...] = ()
    pending_observations: tuple[TrackedObjectObservation, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ObjectTrackerError("unsupported IoU tracker state version")
        _safe_id(self.camera_id, "state camera_id")
        _safe_id(self.detector_adapter, "state detector_adapter")
        if (
            not isinstance(self.config_fingerprint, str)
            or _SHA256.fullmatch(self.config_fingerprint) is None
        ):
            raise ObjectTrackerError("state config fingerprint is invalid")
        if self.last_frame_at is not None:
            object.__setattr__(
                self, "last_frame_at", _aware(self.last_frame_at, "state last_frame_at")
            )
        if (
            not isinstance(self.active_tracks, tuple)
            or len(self.active_tracks) > MAX_ACTIVE_TRACKS
            or any(not isinstance(track, ActiveTrack) for track in self.active_tracks)
            or tuple(sorted(track.tracking_id for track in self.active_tracks))
            != tuple(track.tracking_id for track in self.active_tracks)
            or len({track.tracking_id for track in self.active_tracks}) != len(self.active_tracks)
        ):
            raise ObjectTrackerError("state active tracks are invalid")
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
            raise ObjectTrackerError("state processed frames are invalid")
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
            raise ObjectTrackerError("state pending observations are invalid")
        if any(
            current.occurred_at < previous.occurred_at
            for previous, current in zip(
                self.pending_observations,
                self.pending_observations[1:],
                strict=False,
            )
        ):
            raise ObjectTrackerError("state pending observations are not chronological")
        if any(
            observation.camera_id != self.camera_id
            or observation.source_adapter != TRACKER_ADAPTER_ID
            for observation in self.pending_observations
        ):
            raise ObjectTrackerError("state pending observations have the wrong source identity")
        if self.last_frame_at is None:
            if self.active_tracks or self.processed_frames or self.pending_observations:
                raise ObjectTrackerError("empty tracker timeline contains state")
        else:
            if not self.processed_frames:
                raise ObjectTrackerError("tracker timeline is missing processed frames")
            if self.processed_frames[-1].occurred_at != self.last_frame_at:
                raise ObjectTrackerError("tracker last_frame_at is inconsistent")
            if any(track.last_seen_at > self.last_frame_at for track in self.active_tracks):
                raise ObjectTrackerError("active track is newer than the tracker timeline")
            if any(
                observation.occurred_at > self.last_frame_at
                for observation in self.pending_observations
            ):
                raise ObjectTrackerError("pending observation is newer than the tracker timeline")

    @classmethod
    def empty(
        cls,
        *,
        camera_id: str,
        detector_adapter: str,
        config_fingerprint: str,
    ) -> IoUTrackerState:
        return cls(
            camera_id=camera_id,
            detector_adapter=detector_adapter,
            config_fingerprint=config_fingerprint,
        )

    @classmethod
    def from_dict(cls, value: Any) -> IoUTrackerState:
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
            raise ObjectTrackerError("IoU tracker state fields do not match v1")
        active = value["active_tracks"]
        processed = value["processed_frames"]
        pending = value["pending_observations"]
        if not all(isinstance(item, list) for item in (active, processed, pending)):
            raise ObjectTrackerError("IoU tracker state arrays are invalid")
        last_frame_at = value["last_frame_at"]
        if last_frame_at is not None:
            last_frame_at = _timestamp(last_frame_at, "state last_frame_at")
        return cls(
            version=value["version"],
            camera_id=value["camera_id"],
            detector_adapter=value["detector_adapter"],
            config_fingerprint=value["config_fingerprint"],
            last_frame_at=last_frame_at,
            active_tracks=tuple(ActiveTrack.from_dict(item) for item in active),
            processed_frames=tuple(ProcessedFrame.from_dict(item) for item in processed),
            pending_observations=tuple(
                TrackedObjectObservation.from_dict(item) for item in pending
            ),
        )

    @classmethod
    def read(cls, path: str | Path) -> IoUTrackerState:
        return cls.from_dict(_read_json(path, context="IoU tracker state"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "detector_adapter": self.detector_adapter,
            "config_fingerprint": self.config_fingerprint,
            "last_frame_at": self.last_frame_at.isoformat() if self.last_frame_at else None,
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
            raise ObjectTrackerError("IoU tracker state exceeds the 4 MiB limit")
        return payload + b"\n"

    def write_atomic(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        parent = target.parent
        if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
            raise ObjectTrackerError("IoU tracker state destination is unsafe")
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
            raise ObjectTrackerError("IoU tracker state could not be written") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(OSError):
                temporary.unlink()

    def acknowledge(self, observation_ids: tuple[str, ...]) -> IoUTrackerState:
        if (
            not isinstance(observation_ids, tuple)
            or not observation_ids
            or len(set(observation_ids)) != len(observation_ids)
        ):
            raise ObjectTrackerError("observation acknowledgements must be non-empty and unique")
        for observation_id in observation_ids:
            _safe_id(observation_id, "acknowledged observation_id")
        known = {observation.observation_id for observation in self.pending_observations}
        if not set(observation_ids).issubset(known):
            raise ObjectTrackerError("acknowledgement contains an unknown observation_id")
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


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


class IoUObjectTracker:
    def __init__(
        self,
        *,
        config: IoUTrackerConfig,
        camera_id: str,
        detector_adapter: str,
        state: IoUTrackerState | None = None,
    ):
        _safe_id(camera_id, "camera_id")
        _safe_id(detector_adapter, "detector_adapter")
        self.config = config
        self.state = state or IoUTrackerState.empty(
            camera_id=camera_id,
            detector_adapter=detector_adapter,
            config_fingerprint=config.fingerprint,
        )
        if self.state.camera_id != camera_id or self.state.detector_adapter != detector_adapter:
            raise ObjectTrackerError("tracker state belongs to a different input source")
        if self.state.config_fingerprint != config.fingerprint:
            raise ObjectTrackerError("tracker state belongs to a different config")

    def _tracking_id(self, frame: DetectionFrame, detection: FrameDetection) -> str:
        digest = _fingerprint(
            {
                "version": 1,
                "camera_id": self.state.camera_id,
                "detector_adapter": self.state.detector_adapter,
                "config_fingerprint": self.state.config_fingerprint,
                "frame_id": frame.frame_id,
                "occurred_at": frame.occurred_at.isoformat(),
                "detection": detection.to_dict(),
            }
        )
        return f"track_{digest[:24]}"

    def _observation(
        self,
        *,
        frame: DetectionFrame,
        detection: FrameDetection,
        tracking_id: str,
    ) -> TrackedObjectObservation:
        digest = _fingerprint(
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
            source_adapter=TRACKER_ADAPTER_ID,
            tracking_id=tracking_id,
            label=detection.label,
            confidence=detection.confidence,
            bbox_xyxy=detection.bbox_xyxy,
        )

    def process(self, frame: DetectionFrame) -> tuple[TrackedObjectObservation, ...]:
        if not isinstance(frame, DetectionFrame):
            raise ObjectTrackerError("tracker input must be a DetectionFrame")
        existing = next(
            (item for item in self.state.processed_frames if item.frame_id == frame.frame_id),
            None,
        )
        if existing is not None:
            if existing.fingerprint != frame.fingerprint:
                raise ObjectTrackerError("frame_id was reused with conflicting content")
            return ()
        if self.state.last_frame_at is not None and frame.occurred_at <= self.state.last_frame_at:
            raise ObjectTrackerError("new frames must be strictly monotonic")

        active = tuple(
            track
            for track in self.state.active_tracks
            if (frame.occurred_at - track.last_seen_at).total_seconds()
            <= self.config.max_track_age_seconds
        )
        candidate_pairs: list[tuple[float, str, str]] = []
        tracks_by_id = {track.tracking_id: track for track in active}
        for track in active:
            for detection in frame.detections:
                if track.label != detection.label:
                    continue
                overlap = _iou(track.bbox_xyxy, detection.bbox_xyxy)
                if overlap >= self.config.iou_threshold:
                    candidate_pairs.append((-overlap, track.tracking_id, detection.detection_id))
        candidate_pairs.sort()
        assigned_tracks: set[str] = set()
        assigned_detections: set[str] = set()
        detection_tracks: dict[str, str] = {}
        for _negative_iou, tracking_id, detection_id in candidate_pairs:
            if tracking_id in assigned_tracks or detection_id in assigned_detections:
                continue
            assigned_tracks.add(tracking_id)
            assigned_detections.add(detection_id)
            detection_tracks[detection_id] = tracking_id

        next_tracks = {
            track.tracking_id: track for track in active if track.tracking_id not in assigned_tracks
        }
        observations: list[TrackedObjectObservation] = []
        for detection in frame.detections:
            tracking_id = detection_tracks.get(detection.detection_id)
            if tracking_id is None:
                tracking_id = self._tracking_id(frame, detection)
                if tracking_id in tracks_by_id or tracking_id in next_tracks:
                    raise ObjectTrackerError("deterministic tracking_id collision")
            next_tracks[tracking_id] = ActiveTrack(
                tracking_id=tracking_id,
                label=detection.label,
                last_seen_at=frame.occurred_at,
                confidence=detection.confidence,
                bbox_xyxy=detection.bbox_xyxy,
            )
            observations.append(
                self._observation(
                    frame=frame,
                    detection=detection,
                    tracking_id=tracking_id,
                )
            )
        if len(next_tracks) > self.config.max_active_tracks:
            raise ObjectTrackerError("active track limit would be exceeded")
        if (
            len(self.state.pending_observations) + len(observations)
            > self.config.max_pending_observations
        ):
            raise ObjectTrackerError("pending observation outbox is full")

        processed = (
            *self.state.processed_frames,
            ProcessedFrame(
                frame_id=frame.frame_id,
                occurred_at=frame.occurred_at,
                fingerprint=frame.fingerprint,
            ),
        )[-self.config.max_processed_frames :]
        candidate_state = replace(
            self.state,
            last_frame_at=frame.occurred_at,
            active_tracks=tuple(sorted(next_tracks.values(), key=lambda item: item.tracking_id)),
            processed_frames=processed,
            pending_observations=(
                *self.state.pending_observations,
                *observations,
            ),
        )
        candidate_state._encoded()
        self.state = candidate_state
        return tuple(observations)
