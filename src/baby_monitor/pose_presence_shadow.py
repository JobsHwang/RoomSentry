from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

from .adapters.camera import CameraSource
from .adapters.detector import Detector, DetectorError
from .adapters.pose import PoseEstimator, PoseEstimatorError
from .domain import Detection
from .pose_protocol import PoseFrameBatch
from .room_presence import StreamPresenceSampleState

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PosePresenceShadowError(RuntimeError):
    """A path-free detector/Pose shadow classification failure."""


def _safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise PosePresenceShadowError(f"pose presence shadow {field} is invalid")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PosePresenceShadowError(f"pose presence shadow {field} is invalid")
    return value


def _number(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise PosePresenceShadowError(f"pose presence shadow {field} is invalid")
    return float(value)


@dataclass(frozen=True, slots=True)
class PosePresenceShadowCameraRule:
    camera_id: str
    detector_confidence_threshold: float
    detector_hold_seconds: float
    pose_enabled: bool
    pose_confidence_threshold: float
    pose_hold_seconds: float
    minimum_pose_bbox_area: float
    minimum_pose_bbox_height: float
    keypoint_confidence_threshold: float
    minimum_keypoint_count: int

    def __post_init__(self) -> None:
        _safe_id(self.camera_id, field="camera_id")
        for field in (
            "detector_confidence_threshold",
            "pose_confidence_threshold",
            "minimum_pose_bbox_area",
            "minimum_pose_bbox_height",
            "keypoint_confidence_threshold",
        ):
            object.__setattr__(
                self,
                field,
                _number(
                    getattr(self, field),
                    field=field,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        for field in ("detector_hold_seconds", "pose_hold_seconds"):
            object.__setattr__(
                self,
                field,
                _number(
                    getattr(self, field),
                    field=field,
                    minimum=0.0,
                    maximum=30.0,
                ),
            )
        if type(self.pose_enabled) is not bool:
            raise PosePresenceShadowError("pose presence shadow pose_enabled is invalid")
        if type(self.minimum_keypoint_count) is not int or not (
            1 <= self.minimum_keypoint_count <= 17
        ):
            raise PosePresenceShadowError("pose presence shadow minimum_keypoint_count is invalid")


@dataclass(frozen=True, slots=True)
class PosePresenceShadowPolicy:
    profile_sha256: str
    temporal_holdout_attestation_sha256: str
    human_audit_report_sha256: str
    detector_adapter: str
    detector_model_id: str
    detector_model_revision: str
    detector_model_sha256: str
    detector_source_confidence_threshold: float
    detector_requested_sample_fps: float
    pose_adapter: str
    pose_model_id: str
    pose_model_revision: str
    pose_model_sha256: str
    pose_source_confidence_threshold: float
    pose_requested_sample_fps: float
    pose_requested_max_frames: int
    image_size: int
    rotation_degrees: int
    bin_seconds: float
    cameras: tuple[PosePresenceShadowCameraRule, ...]
    version: int = 1
    label: str = "person"
    fusion_operator: str = "detector_or_gated_pose"
    shadow_only: bool = True
    subject_identity_inference: bool = False
    posture_classification: bool = False
    risk_assessment: bool = False
    notification_authority: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise PosePresenceShadowError("unsupported pose presence shadow policy version")
        for field in (
            "profile_sha256",
            "temporal_holdout_attestation_sha256",
            "human_audit_report_sha256",
            "detector_model_sha256",
            "pose_model_sha256",
        ):
            _sha256(getattr(self, field), field=field)
        for field in (
            "detector_model_id",
            "detector_model_revision",
            "detector_adapter",
            "pose_adapter",
            "pose_model_id",
            "pose_model_revision",
        ):
            _safe_id(getattr(self, field), field=field)
        object.__setattr__(
            self,
            "detector_source_confidence_threshold",
            _number(
                self.detector_source_confidence_threshold,
                field="detector_source_confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "pose_source_confidence_threshold",
            _number(
                self.pose_source_confidence_threshold,
                field="pose_source_confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        for field in ("detector_requested_sample_fps", "pose_requested_sample_fps"):
            object.__setattr__(
                self,
                field,
                _number(
                    getattr(self, field),
                    field=field,
                    minimum=0.01,
                    maximum=10.0,
                ),
            )
        if type(self.pose_requested_max_frames) is not int or not (
            1 <= self.pose_requested_max_frames <= 10_000
        ):
            raise PosePresenceShadowError(
                "pose presence shadow pose_requested_max_frames is invalid"
            )
        if (
            type(self.image_size) is not int
            or not 320 <= self.image_size <= 1280
            or self.image_size % 32
        ):
            raise PosePresenceShadowError("pose presence shadow image_size is invalid")
        if type(self.rotation_degrees) is not int or self.rotation_degrees not in {
            0,
            90,
            180,
            270,
        }:
            raise PosePresenceShadowError("pose presence shadow rotation is invalid")
        object.__setattr__(
            self,
            "bin_seconds",
            _number(self.bin_seconds, field="bin_seconds", minimum=0.1, maximum=60.0),
        )
        if (
            not isinstance(self.cameras, tuple)
            or not self.cameras
            or any(not isinstance(item, PosePresenceShadowCameraRule) for item in self.cameras)
        ):
            raise PosePresenceShadowError("pose presence shadow cameras are invalid")
        camera_ids = tuple(item.camera_id for item in self.cameras)
        if camera_ids != tuple(sorted(camera_ids)) or len(set(camera_ids)) != len(camera_ids):
            raise PosePresenceShadowError("pose presence shadow cameras must be unique and sorted")
        if self.label != "person" or self.fusion_operator != "detector_or_gated_pose":
            raise PosePresenceShadowError("pose presence shadow semantics are invalid")
        for field, expected in (
            ("shadow_only", True),
            ("subject_identity_inference", False),
            ("posture_classification", False),
            ("risk_assessment", False),
            ("notification_authority", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise PosePresenceShadowError(
                    f"pose presence shadow {field} must be {str(expected).lower()}"
                )
        if any(
            item.detector_confidence_threshold < self.detector_source_confidence_threshold
            or item.pose_confidence_threshold < self.pose_source_confidence_threshold
            for item in self.cameras
        ):
            raise PosePresenceShadowError(
                "pose presence shadow rule is below an admitted scan threshold"
            )

    @property
    def fingerprint(self) -> str:
        payload = {
            field: (
                [
                    {
                        camera_field: getattr(camera, camera_field)
                        for camera_field in camera.__dataclass_fields__
                    }
                    for camera in self.cameras
                ]
                if field == "cameras"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def camera_rule(self, camera_id: str) -> PosePresenceShadowCameraRule:
        rule = next((item for item in self.cameras if item.camera_id == camera_id), None)
        if rule is None:
            raise PosePresenceShadowError("pose presence shadow camera is not configured")
        return rule


@dataclass(frozen=True, slots=True)
class PosePresenceShadowDecision:
    state: StreamPresenceSampleState
    detector_hit: bool
    pose_hit: bool
    pose_executed: bool

    def __post_init__(self) -> None:
        if self.state not in {
            StreamPresenceSampleState.PERSON_DETECTED,
            StreamPresenceSampleState.CLEAR,
        }:
            raise PosePresenceShadowError("pose presence shadow decision state is invalid")
        for field in ("detector_hit", "pose_hit", "pose_executed"):
            if type(getattr(self, field)) is not bool:
                raise PosePresenceShadowError("pose presence shadow decision is invalid")
        if self.pose_hit and not self.pose_executed:
            raise PosePresenceShadowError("pose presence shadow decision is inconsistent")
        if (self.detector_hit or self.pose_hit) != (
            self.state is StreamPresenceSampleState.PERSON_DETECTED
        ):
            raise PosePresenceShadowError("pose presence shadow decision is inconsistent")


def _pose_passes(rule: PosePresenceShadowCameraRule, pose: object) -> bool:
    try:
        x1, y1, x2, y2 = pose.bbox_xyxy  # type: ignore[attr-defined]
        confidence = pose.confidence  # type: ignore[attr-defined]
        keypoints = pose.keypoints  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError) as exc:
        raise PosePresenceShadowError("pose presence shadow pose is invalid") from exc
    usable_keypoints = sum(
        item.in_frame and item.confidence >= rule.keypoint_confidence_threshold
        for item in keypoints
    )
    return (
        confidence >= rule.pose_confidence_threshold
        and (x2 - x1) * (y2 - y1) >= rule.minimum_pose_bbox_area
        and y2 - y1 >= rule.minimum_pose_bbox_height
        and usable_keypoints >= rule.minimum_keypoint_count
    )


class PosePresenceShadowClassifier:
    """Evaluate only the most recent causal bin of one bounded segment."""

    def __init__(
        self,
        policy: PosePresenceShadowPolicy,
        detector: Detector,
        estimator: PoseEstimator,
    ):
        if not isinstance(policy, PosePresenceShadowPolicy):
            raise PosePresenceShadowError("pose presence shadow policy is invalid")
        if not isinstance(detector, Detector):
            raise PosePresenceShadowError("pose presence shadow detector is invalid")
        if not isinstance(estimator, PoseEstimator):
            raise PosePresenceShadowError("pose presence shadow estimator is invalid")
        detector_manifest = getattr(detector, "manifest", None)
        for field, actual, expected in (
            (
                "detector_adapter",
                getattr(detector, "runner_adapter", None),
                policy.detector_adapter,
            ),
            (
                "detector_model_id",
                getattr(detector_manifest, "model_id", None),
                policy.detector_model_id,
            ),
            (
                "detector_model_revision",
                getattr(detector_manifest, "revision", None),
                policy.detector_model_revision,
            ),
            (
                "detector_model_sha256",
                getattr(detector, "model_sha256", None),
                policy.detector_model_sha256,
            ),
            (
                "detector_source_confidence_threshold",
                getattr(detector, "confidence", None),
                policy.detector_source_confidence_threshold,
            ),
            (
                "detector_requested_sample_fps",
                getattr(detector, "sample_fps", None),
                policy.detector_requested_sample_fps,
            ),
            ("image_size", getattr(detector, "image_size", None), policy.image_size),
            (
                "rotation_degrees",
                getattr(detector, "rotation_degrees", None),
                policy.rotation_degrees,
            ),
        ):
            if actual != expected:
                raise PosePresenceShadowError(
                    f"pose presence shadow detector {field} does not match evidence"
                )
        pose_manifest = getattr(estimator, "manifest", None)
        for field, actual, expected in (
            ("pose_adapter", getattr(estimator, "runner_adapter", None), policy.pose_adapter),
            ("pose_model_id", getattr(pose_manifest, "model_id", None), policy.pose_model_id),
            (
                "pose_model_revision",
                getattr(pose_manifest, "revision", None),
                policy.pose_model_revision,
            ),
            (
                "pose_model_sha256",
                getattr(estimator, "model_sha256", None),
                policy.pose_model_sha256,
            ),
            (
                "pose_source_confidence_threshold",
                getattr(estimator, "confidence", None),
                policy.pose_source_confidence_threshold,
            ),
            (
                "pose_requested_sample_fps",
                getattr(estimator, "sample_fps", None),
                policy.pose_requested_sample_fps,
            ),
            (
                "pose_requested_max_frames",
                getattr(estimator, "max_frames", None),
                policy.pose_requested_max_frames,
            ),
            ("image_size", getattr(estimator, "image_size", None), policy.image_size),
            (
                "rotation_degrees",
                getattr(estimator, "rotation_degrees", None),
                policy.rotation_degrees,
            ),
        ):
            if actual != expected:
                raise PosePresenceShadowError(
                    f"pose presence shadow estimator {field} does not match evidence"
                )
        self.policy = policy
        self.detector = detector
        self.estimator = estimator
        self.scanner_identity = f"pose_shadow_v1_{policy.fingerprint[:16]}"
        self.detector_identity = f"detector_pose_or_v1_{policy.fingerprint[:16]}"

    def classify(
        self,
        source: CameraSource,
        *,
        duration_seconds: float,
    ) -> PosePresenceShadowDecision:
        if not isinstance(source, CameraSource) or source.kind != "file":
            raise PosePresenceShadowError("pose presence shadow source is invalid")
        duration = _number(
            duration_seconds,
            field="duration_seconds",
            minimum=0.1,
            maximum=3600.0,
        )
        rule = self.policy.camera_rule(source.camera_id)
        try:
            detections = tuple(self.detector.scan(source))
        except DetectorError as exc:
            raise PosePresenceShadowError("pose presence shadow detector failed") from exc
        if any(not isinstance(item, Detection) for item in detections):
            raise PosePresenceShadowError("pose presence shadow detections are invalid")
        if any(item.timestamp_seconds > duration + 0.25 for item in detections):
            raise PosePresenceShadowError(
                "pose presence shadow detector timestamp exceeds the segment"
            )
        bin_start = max(0.0, duration - self.policy.bin_seconds)
        detector_start = max(0.0, bin_start - rule.detector_hold_seconds)
        detector_hit = any(
            item.label == "person"
            and item.confidence >= rule.detector_confidence_threshold
            and detector_start <= item.timestamp_seconds <= duration + 0.25
            for item in detections
        )
        if detector_hit or not rule.pose_enabled:
            return PosePresenceShadowDecision(
                state=(
                    StreamPresenceSampleState.PERSON_DETECTED
                    if detector_hit
                    else StreamPresenceSampleState.CLEAR
                ),
                detector_hit=detector_hit,
                pose_hit=False,
                pose_executed=False,
            )
        try:
            batch = self.estimator.scan(source)
        except PoseEstimatorError as exc:
            raise PosePresenceShadowError("pose presence shadow estimator failed") from exc
        if not isinstance(batch, PoseFrameBatch) or batch.camera_id != source.camera_id:
            raise PosePresenceShadowError("pose presence shadow result is invalid")
        if any(frame.timestamp_seconds > duration + 0.25 for frame in batch.frames):
            raise PosePresenceShadowError("pose presence shadow timestamp exceeds the segment")
        pose_start = max(0.0, bin_start - rule.pose_hold_seconds)
        pose_hit = any(
            pose_start <= frame.timestamp_seconds <= duration + 0.25
            and any(_pose_passes(rule, pose) for pose in frame.poses)
            for frame in batch.frames
        )
        return PosePresenceShadowDecision(
            state=(
                StreamPresenceSampleState.PERSON_DETECTED
                if pose_hit
                else StreamPresenceSampleState.CLEAR
            ),
            detector_hit=False,
            pose_hit=pose_hit,
            pose_executed=True,
        )
