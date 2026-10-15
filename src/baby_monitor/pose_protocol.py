from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .frame_transform import validate_image_size, validate_rotation_degrees

COCO_PERSON_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

MAX_POSE_FILE_BYTES = 16 * 1024 * 1024
MAX_POSE_FRAMES = 10_000
MAX_POSES_PER_FRAME = 100
MAX_TOTAL_POSES = 5_000
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class PoseProtocolError(ValueError):
    """A bounded YOLO pose request or path-free result is invalid."""


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PoseProtocolError("pose JSON contains duplicate keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                PoseProtocolError("pose JSON contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise PoseProtocolError("pose JSON is invalid") from exc


def _read_json(path: str | Path, *, context: str) -> Any:
    source = Path(path).expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        if source.stat().st_size > MAX_POSE_FILE_BYTES:
            raise PoseProtocolError(f"{context} exceeds the 16 MiB limit")
        raw = source.read_text(encoding="utf-8")
    except PoseProtocolError:
        raise
    except (OSError, UnicodeError) as exc:
        raise PoseProtocolError(f"{context} is missing or unsafe") from exc
    return _strict_json(raw)


def _id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PoseProtocolError(f"pose {field} is invalid")
    return value


def _path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PoseProtocolError(f"pose {field} is invalid")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PoseProtocolError(f"pose {field} is unsafe")
    return str(path)


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise PoseProtocolError(f"pose {field} is invalid")
    return float(value)


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise PoseProtocolError("pose bbox must contain four normalized numbers")
    parsed = tuple(_number(item, "bbox", minimum=0.0, maximum=1.0) for item in value)
    if parsed[0] >= parsed[2] or parsed[1] >= parsed[3]:
        raise PoseProtocolError("pose bbox must have positive normalized area")
    return parsed


@dataclass(frozen=True, slots=True)
class PoseKeypoint:
    name: str
    x: float | None
    y: float | None
    confidence: float
    in_frame: bool

    def __post_init__(self) -> None:
        if self.name not in COCO_PERSON_KEYPOINTS:
            raise PoseProtocolError("pose keypoint name is not in COCO person order")
        if type(self.in_frame) is not bool:
            raise PoseProtocolError("pose keypoint in_frame must be a boolean")
        if self.in_frame:
            if self.x is None or self.y is None:
                raise PoseProtocolError("in-frame pose keypoint coordinates are required")
            object.__setattr__(self, "x", _number(self.x, "keypoint x", minimum=0.0, maximum=1.0))
            object.__setattr__(self, "y", _number(self.y, "keypoint y", minimum=0.0, maximum=1.0))
        elif self.x is not None or self.y is not None:
            raise PoseProtocolError("out-of-frame pose keypoint coordinates must be null")
        object.__setattr__(
            self,
            "confidence",
            _number(self.confidence, "keypoint confidence", minimum=0.0, maximum=1.0),
        )

    @classmethod
    def from_dict(cls, value: Any) -> PoseKeypoint:
        if not isinstance(value, dict) or set(value) != {
            "name",
            "x",
            "y",
            "confidence",
            "in_frame",
        }:
            raise PoseProtocolError("pose keypoint fields do not match v1")
        return cls(
            name=value["name"],
            x=value["x"],
            y=value["y"],
            confidence=value["confidence"],
            in_frame=value["in_frame"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "in_frame": self.in_frame,
        }


@dataclass(frozen=True, slots=True)
class PoseInstance:
    instance_id: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    keypoints: tuple[PoseKeypoint, ...]
    label: str = "person"

    def __post_init__(self) -> None:
        _id(self.instance_id, "instance_id")
        if self.label != "person":
            raise PoseProtocolError("COCO pose instances must retain the generic person label")
        object.__setattr__(
            self,
            "confidence",
            _number(self.confidence, "instance confidence", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(self, "bbox_xyxy", _bbox(self.bbox_xyxy))
        if (
            not isinstance(self.keypoints, tuple)
            or len(self.keypoints) != len(COCO_PERSON_KEYPOINTS)
            or any(not isinstance(item, PoseKeypoint) for item in self.keypoints)
            or tuple(item.name for item in self.keypoints) != COCO_PERSON_KEYPOINTS
        ):
            raise PoseProtocolError("pose keypoints must contain the ordered COCO-17 set")

    @classmethod
    def from_dict(cls, value: Any) -> PoseInstance:
        if not isinstance(value, dict) or set(value) != {
            "instance_id",
            "label",
            "confidence",
            "bbox_xyxy",
            "keypoints",
        }:
            raise PoseProtocolError("pose instance fields do not match v1")
        keypoints = value["keypoints"]
        if not isinstance(keypoints, list):
            raise PoseProtocolError("pose keypoints must be an array")
        return cls(
            instance_id=value["instance_id"],
            label=value["label"],
            confidence=value["confidence"],
            bbox_xyxy=_bbox(value["bbox_xyxy"]),
            keypoints=tuple(PoseKeypoint.from_dict(item) for item in keypoints),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "keypoints": [item.to_dict() for item in self.keypoints],
        }


@dataclass(frozen=True, slots=True)
class PoseFrame:
    frame_id: str
    timestamp_seconds: float
    poses: tuple[PoseInstance, ...]

    def __post_init__(self) -> None:
        _id(self.frame_id, "frame_id")
        object.__setattr__(
            self,
            "timestamp_seconds",
            _number(
                self.timestamp_seconds,
                "timestamp_seconds",
                minimum=0.0,
                maximum=31_536_000.0,
            ),
        )
        if (
            not isinstance(self.poses, tuple)
            or len(self.poses) > MAX_POSES_PER_FRAME
            or any(not isinstance(item, PoseInstance) for item in self.poses)
        ):
            raise PoseProtocolError("frame poses are invalid or exceed the limit")
        if len({item.instance_id for item in self.poses}) != len(self.poses):
            raise PoseProtocolError("pose instance IDs must be unique within one frame")

    @classmethod
    def from_dict(cls, value: Any) -> PoseFrame:
        if not isinstance(value, dict) or set(value) != {
            "frame_id",
            "timestamp_seconds",
            "poses",
        }:
            raise PoseProtocolError("pose frame fields do not match v1")
        poses = value["poses"]
        if not isinstance(poses, list):
            raise PoseProtocolError("frame poses must be an array")
        return cls(
            frame_id=value["frame_id"],
            timestamp_seconds=value["timestamp_seconds"],
            poses=tuple(PoseInstance.from_dict(item) for item in poses),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp_seconds": self.timestamp_seconds,
            "poses": [item.to_dict() for item in self.poses],
        }


@dataclass(frozen=True, slots=True)
class PoseFrameBatch:
    camera_id: str
    pose_adapter: str
    model_id: str
    model_revision: str
    model_sha256: str
    frames: tuple[PoseFrame, ...]
    subject_identity_inference: bool = False
    posture_classification: bool = False
    paths_disclosed: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise PoseProtocolError("unsupported pose frame batch version")
        _id(self.camera_id, "camera_id")
        _id(self.pose_adapter, "pose_adapter")
        _id(self.model_id, "model_id")
        _id(self.model_revision, "model_revision")
        if _SHA256.fullmatch(self.model_sha256) is None:
            raise PoseProtocolError("pose model_sha256 is invalid")
        if (
            not isinstance(self.frames, tuple)
            or not 1 <= len(self.frames) <= MAX_POSE_FRAMES
            or any(not isinstance(frame, PoseFrame) for frame in self.frames)
        ):
            raise PoseProtocolError("pose frames are invalid or exceed the limit")
        if len({frame.frame_id for frame in self.frames}) != len(self.frames):
            raise PoseProtocolError("pose frame IDs must be unique")
        if any(
            current.timestamp_seconds <= previous.timestamp_seconds
            for previous, current in zip(self.frames, self.frames[1:], strict=False)
        ):
            raise PoseProtocolError("pose frames must be strictly chronological")
        if sum(len(frame.poses) for frame in self.frames) > MAX_TOTAL_POSES:
            raise PoseProtocolError("pose batch exceeds the total pose limit")
        for value, field in (
            (self.subject_identity_inference, "identity inference"),
            (self.posture_classification, "posture classification"),
            (self.paths_disclosed, "paths_disclosed"),
        ):
            if type(value) is not bool or value:
                raise PoseProtocolError(f"pose {field} must be false")

    @classmethod
    def from_dict(cls, value: Any) -> PoseFrameBatch:
        expected = {
            "version",
            "camera_id",
            "pose_adapter",
            "model_id",
            "model_revision",
            "model_sha256",
            "frames",
            "subject_identity_inference",
            "posture_classification",
            "paths_disclosed",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise PoseProtocolError("pose frame batch fields do not match v1")
        frames = value["frames"]
        if not isinstance(frames, list):
            raise PoseProtocolError("pose frames must be an array")
        return cls(
            version=value["version"],
            camera_id=value["camera_id"],
            pose_adapter=value["pose_adapter"],
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            model_sha256=value["model_sha256"],
            frames=tuple(PoseFrame.from_dict(item) for item in frames),
            subject_identity_inference=value["subject_identity_inference"],
            posture_classification=value["posture_classification"],
            paths_disclosed=value["paths_disclosed"],
        )

    @classmethod
    def from_json(cls, raw: str) -> PoseFrameBatch:
        if len(raw.encode("utf-8")) > MAX_POSE_FILE_BYTES:
            raise PoseProtocolError("pose frame batch exceeds the 16 MiB limit")
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> PoseFrameBatch:
        return cls.from_dict(_read_json(path, context="pose frame batch"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "pose_adapter": self.pose_adapter,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "frames": [frame.to_dict() for frame in self.frames],
            "subject_identity_inference": False,
            "posture_classification": False,
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
        if len(encoded.encode("utf-8")) > MAX_POSE_FILE_BYTES:
            raise PoseProtocolError("pose frame batch exceeds the 16 MiB limit")
        return encoded

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class AdmittedPoseScan:
    """Path-free pose result bound to one exact admitted source and scan config."""

    source_admission_sha256: str
    confidence_threshold: float
    requested_sample_fps: float
    requested_max_frames: int
    image_size: int
    rotation_degrees: int
    result: PoseFrameBatch
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise PoseProtocolError("unsupported admitted pose scan version")
        if (
            not isinstance(self.source_admission_sha256, str)
            or _SHA256.fullmatch(self.source_admission_sha256) is None
        ):
            raise PoseProtocolError("admitted pose scan source admission hash is invalid")
        object.__setattr__(
            self,
            "confidence_threshold",
            _number(
                self.confidence_threshold,
                "admitted confidence threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "requested_sample_fps",
            _number(
                self.requested_sample_fps,
                "admitted requested sample fps",
                minimum=0.01,
                maximum=10.0,
            ),
        )
        if (
            type(self.requested_max_frames) is not int
            or not 1 <= self.requested_max_frames <= MAX_POSE_FRAMES
        ):
            raise PoseProtocolError("admitted pose scan requested max frames is invalid")
        try:
            validate_image_size(self.image_size)
            validate_rotation_degrees(self.rotation_degrees)
        except ValueError as exc:
            raise PoseProtocolError(str(exc)) from exc
        if not isinstance(self.result, PoseFrameBatch):
            raise PoseProtocolError("admitted pose scan result is invalid")
        if len(self.result.frames) > self.requested_max_frames:
            raise PoseProtocolError("admitted pose scan exceeds requested max frames")
        if any(
            pose.confidence < self.confidence_threshold
            for frame in self.result.frames
            for pose in frame.poses
        ):
            raise PoseProtocolError("admitted pose result conflicts with scan threshold")

    @classmethod
    def from_dict(cls, value: Any) -> AdmittedPoseScan:
        expected = {
            "version",
            "source_admission_sha256",
            "confidence_threshold",
            "requested_sample_fps",
            "requested_max_frames",
            "image_size",
            "rotation_degrees",
            "result",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise PoseProtocolError("admitted pose scan fields do not match v1")
        return cls(
            version=value["version"],
            source_admission_sha256=value["source_admission_sha256"],
            confidence_threshold=value["confidence_threshold"],
            requested_sample_fps=value["requested_sample_fps"],
            requested_max_frames=value["requested_max_frames"],
            image_size=value["image_size"],
            rotation_degrees=value["rotation_degrees"],
            result=PoseFrameBatch.from_dict(value["result"]),
        )

    @classmethod
    def from_json(cls, raw: str) -> AdmittedPoseScan:
        if len(raw.encode("utf-8")) > MAX_POSE_FILE_BYTES:
            raise PoseProtocolError("admitted pose scan exceeds the 16 MiB limit")
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> AdmittedPoseScan:
        return cls.from_dict(_read_json(path, context="admitted pose scan"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_admission_sha256": self.source_admission_sha256,
            "confidence_threshold": self.confidence_threshold,
            "requested_sample_fps": self.requested_sample_fps,
            "requested_max_frames": self.requested_max_frames,
            "image_size": self.image_size,
            "rotation_degrees": self.rotation_degrees,
            "result": self.result.to_dict(),
        }

    def to_json(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(encoded.encode("utf-8")) > MAX_POSE_FILE_BYTES:
            raise PoseProtocolError("admitted pose scan exceeds the 16 MiB limit")
        return encoded

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def write_new(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        parent = target.parent
        if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
            raise PoseProtocolError("admitted pose scan destination is unsafe")
        payload = f"{self.to_json()}\n".encode()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(target, flags, 0o600)
        except FileExistsError:
            raise FileExistsError("admitted pose scan output already exists") from None
        except OSError as exc:
            raise PoseProtocolError("admitted pose scan output could not be created") from exc
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
            raise PoseProtocolError("admitted pose scan output could not be written") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)


@dataclass(frozen=True, slots=True)
class PoseScanRequest:
    """Private path-bearing request for one short-lived local pose process."""

    camera_id: str
    pose_adapter: str
    source_path: str
    model_path: str
    manifest_path: str
    model_id: str
    model_revision: str
    model_sha256: str
    confidence: float
    sample_fps: float
    max_frames: int
    device: str
    image_size: int = 640
    rotation_degrees: int = 0
    version: int = 2

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 2:
            raise PoseProtocolError("unsupported pose request version")
        _id(self.camera_id, "camera_id")
        _id(self.pose_adapter, "pose_adapter")
        _path(self.source_path, "source_path")
        _path(self.model_path, "model_path")
        _path(self.manifest_path, "manifest_path")
        _id(self.model_id, "model_id")
        _id(self.model_revision, "model_revision")
        if _SHA256.fullmatch(self.model_sha256) is None:
            raise PoseProtocolError("pose model_sha256 is invalid")
        _number(self.confidence, "confidence", minimum=0.0, maximum=1.0)
        _number(self.sample_fps, "sample_fps", minimum=0.01, maximum=10.0)
        if type(self.max_frames) is not int or not 1 <= self.max_frames <= MAX_POSE_FRAMES:
            raise PoseProtocolError("pose max_frames is invalid")
        if self.device != "cpu" and re.fullmatch(r"\d+", self.device) is None:
            raise PoseProtocolError("pose device is invalid")
        try:
            validate_image_size(self.image_size)
            validate_rotation_degrees(self.rotation_degrees)
        except ValueError as exc:
            raise PoseProtocolError(str(exc)) from exc

    @classmethod
    def from_dict(cls, value: Any) -> PoseScanRequest:
        legacy_expected = {
            "version",
            "camera_id",
            "pose_adapter",
            "source_path",
            "model_path",
            "manifest_path",
            "model_id",
            "model_revision",
            "model_sha256",
            "confidence",
            "sample_fps",
            "max_frames",
            "device",
        }
        current_expected = legacy_expected | {"image_size", "rotation_degrees"}
        if not isinstance(value, dict):
            raise PoseProtocolError("pose request fields do not match v2")
        if value.get("version") == 1 and set(value) == legacy_expected:
            value = {
                **value,
                "version": 2,
                "image_size": 640,
                "rotation_degrees": 0,
            }
        elif value.get("version") != 2 or set(value) != current_expected:
            raise PoseProtocolError("pose request fields do not match v2")
        return cls(**value)

    @classmethod
    def from_json(cls, raw: str) -> PoseScanRequest:
        if len(raw.encode("utf-8")) > MAX_POSE_FILE_BYTES:
            raise PoseProtocolError("pose request exceeds the 16 MiB limit")
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> PoseScanRequest:
        return cls.from_dict(_read_json(path, context="pose request"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "pose_adapter": self.pose_adapter,
            "source_path": self.source_path,
            "model_path": self.model_path,
            "manifest_path": self.manifest_path,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "confidence": self.confidence,
            "sample_fps": self.sample_fps,
            "max_frames": self.max_frames,
            "device": self.device,
            "image_size": self.image_size,
            "rotation_degrees": self.rotation_degrees,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
