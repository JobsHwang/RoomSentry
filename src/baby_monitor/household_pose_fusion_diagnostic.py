"""Evaluation-only household detector and gated-pose fusion diagnostics.

The contracts in this module consume closed model-generated presence reviews,
verified local-only source admissions, and already-saved detector/pose scans.
They never read media, load a model, infer identity or posture, mutate runtime
configuration, or grant deployment authority.  Camera-specific profiles are
kept explicit because the household cameras have materially different views.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detector_protocol import AdmittedDetectorScan
from .household_model_diagnostic import (
    HOUSEHOLD_MODEL_REVIEW_SAMPLING_STRATEGIES,
    MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES,
    HouseholdModelDiagnosticError,
    HouseholdModelPresenceItem,
    HouseholdModelPresenceReview,
    _canonical_sha256,
    _count,
    _number,
    _optional_ratio,
    _ratio,
    _safe_id,
    _sha256,
    _strict_json,
    _validate_admissions,
    _validate_scans,
    _write_exclusive,
)
from .household_model_temporal_diagnostic import (
    MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLD_SECONDS,
    HouseholdModelTemporalCameraPoint,
    _f1,
)
from .pose_protocol import AdmittedPoseScan
from .source_admission import ExternalSourceAdmission

MAX_HOUSEHOLD_POSE_FUSION_PROFILE_BYTES = 128 * 1024
MAX_HOUSEHOLD_POSE_FUSION_REPORT_BYTES = 4 * 1024 * 1024
MAX_HOUSEHOLD_POSE_FUSION_BATCHES = 16


@dataclass(frozen=True, slots=True)
class HouseholdPoseFusionCameraProfile:
    """One explicit detector/pose operating profile for one anonymous camera."""

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
        _safe_id(self.camera_id, "pose fusion profile camera_id")
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
                    f"pose fusion profile {field}",
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
                    f"pose fusion profile {field}",
                    minimum=0.0,
                    maximum=MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLD_SECONDS,
                ),
            )
        if type(self.pose_enabled) is not bool:
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile pose_enabled is invalid"
            )
        if type(self.minimum_keypoint_count) is not int or not (
            1 <= self.minimum_keypoint_count <= 17
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile minimum_keypoint_count is invalid"
            )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdPoseFusionCameraProfile:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household pose fusion camera profile fields do not match v1"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdPoseFusionProfile:
    """Camera-specific profile that is evidence-only and never deployable by itself."""

    profile_id: str
    cameras: tuple[HouseholdPoseFusionCameraProfile, ...]
    version: int = 1
    label: str = "person"
    fusion_operator: str = "detector_or_gated_pose"
    selection_provenance: str = "same_household_model_generated_review"
    profile_selected_on_evaluation_data: bool = True
    diagnostic_only: bool = True
    deployment_approved: bool = False
    household_configuration_changed: bool = False
    subject_identity_inference: bool = False
    posture_classification: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HouseholdModelDiagnosticError("unsupported household pose fusion profile version")
        _safe_id(self.profile_id, "pose fusion profile_id")
        if (
            not isinstance(self.cameras, tuple)
            or not 1 <= len(self.cameras) <= MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES
            or any(not isinstance(item, HouseholdPoseFusionCameraProfile) for item in self.cameras)
        ):
            raise HouseholdModelDiagnosticError("household pose fusion profile cameras are invalid")
        camera_ids = tuple(item.camera_id for item in self.cameras)
        if camera_ids != tuple(sorted(camera_ids)) or len(set(camera_ids)) != len(camera_ids):
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile cameras must be unique and sorted"
            )
        if not any(item.pose_enabled for item in self.cameras):
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile must enable pose for at least one camera"
            )
        if (
            self.label != "person"
            or self.fusion_operator != "detector_or_gated_pose"
            or self.selection_provenance != "same_household_model_generated_review"
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile semantics are invalid"
            )
        for field, expected in (
            ("profile_selected_on_evaluation_data", True),
            ("diagnostic_only", True),
            ("deployment_approved", False),
            ("household_configuration_changed", False),
            ("subject_identity_inference", False),
            ("posture_classification", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdModelDiagnosticError(
                    f"household pose fusion profile {field} must be {str(expected).lower()}"
                )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: (
                [item.to_dict() for item in self.cameras]
                if field == "cameras"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        if include_fingerprint:
            value["profile_sha256"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def to_public_dict(self) -> dict[str, object]:
        return {"valid": True, **self.to_dict()}

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdPoseFusionProfile:
        fields = {*cls.__dataclass_fields__, "profile_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile fields do not match v1"
            )
        if not isinstance(value["cameras"], list):
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile cameras must be an array"
            )
        profile_sha256 = value["profile_sha256"]
        profile = cls(
            **{
                key: (
                    tuple(HouseholdPoseFusionCameraProfile.from_dict(item) for item in item_value)
                    if key == "cameras"
                    else item_value
                )
                for key, item_value in value.items()
                if key != "profile_sha256"
            }
        )
        if profile_sha256 != profile.fingerprint:
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile fingerprint is invalid"
            )
        return profile

    @classmethod
    def from_json(cls, raw: str) -> HouseholdPoseFusionProfile:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_POSE_FUSION_PROFILE_BYTES:
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile exceeds the size limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdPoseFusionProfile:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_HOUSEHOLD_POSE_FUSION_PROFILE_BYTES:
                raise HouseholdModelDiagnosticError(
                    "household pose fusion profile exceeds the size limit"
                )
            raw = source.read_text(encoding="utf-8")
        except HouseholdModelDiagnosticError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_POSE_FUSION_PROFILE_BYTES,
        )


@dataclass(frozen=True, slots=True)
class HouseholdPoseFusionMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float | None
    recall: float | None
    f1: float | None
    specificity: float | None
    false_positive_rate: float | None
    clear_positive_count: int
    clear_detected_count: int
    clear_visibility_recall: float | None
    partial_positive_count: int
    partial_detected_count: int
    partial_visibility_recall: float | None
    maximum_negative_camera_false_positive_rate: float | None
    camera_points: tuple[HouseholdModelTemporalCameraPoint, ...]

    def __post_init__(self) -> None:
        for field in (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "clear_positive_count",
            "clear_detected_count",
            "partial_positive_count",
            "partial_detected_count",
        ):
            _count(getattr(self, field), f"pose fusion metrics {field}")
        for field in (
            "precision",
            "recall",
            "f1",
            "specificity",
            "false_positive_rate",
            "clear_visibility_recall",
            "partial_visibility_recall",
            "maximum_negative_camera_false_positive_rate",
        ):
            object.__setattr__(
                self,
                field,
                _optional_ratio(getattr(self, field), f"pose fusion metrics {field}"),
            )
        if (
            not isinstance(self.camera_points, tuple)
            or not 1 <= len(self.camera_points) <= MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES
            or any(
                not isinstance(item, HouseholdModelTemporalCameraPoint)
                for item in self.camera_points
            )
        ):
            raise HouseholdModelDiagnosticError("household pose fusion camera metrics are invalid")
        camera_ids = tuple(item.camera_id for item in self.camera_points)
        if camera_ids != tuple(sorted(camera_ids)) or len(set(camera_ids)) != len(camera_ids):
            raise HouseholdModelDiagnosticError(
                "household pose fusion camera metrics must be unique and sorted"
            )
        for field in (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "clear_positive_count",
            "clear_detected_count",
            "partial_positive_count",
            "partial_detected_count",
        ):
            if getattr(self, field) != sum(getattr(item, field) for item in self.camera_points):
                raise HouseholdModelDiagnosticError(
                    "household pose fusion aggregate and camera accounting differ"
                )
        expected_precision = _ratio(
            self.true_positive,
            self.true_positive + self.false_positive,
        )
        expected_recall = _ratio(
            self.true_positive,
            self.true_positive + self.false_negative,
        )
        camera_rates = tuple(
            item.false_positive_rate
            for item in self.camera_points
            if item.false_positive_rate is not None
        )
        if (
            self.precision != expected_precision
            or self.recall != expected_recall
            or self.f1 != _f1(expected_precision, expected_recall)
            or self.specificity
            != _ratio(self.true_negative, self.true_negative + self.false_positive)
            or self.false_positive_rate
            != _ratio(self.false_positive, self.true_negative + self.false_positive)
            or self.clear_visibility_recall
            != _ratio(self.clear_detected_count, self.clear_positive_count)
            or self.partial_visibility_recall
            != _ratio(self.partial_detected_count, self.partial_positive_count)
            or self.maximum_negative_camera_false_positive_rate
            != (max(camera_rates) if camera_rates else None)
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion aggregate metrics are inconsistent"
            )

    @property
    def evaluated_bin_count(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @classmethod
    def from_camera_points(
        cls,
        camera_points: tuple[HouseholdModelTemporalCameraPoint, ...],
    ) -> HouseholdPoseFusionMetrics:
        totals = {
            field: sum(getattr(item, field) for item in camera_points)
            for field in (
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
                "clear_positive_count",
                "clear_detected_count",
                "partial_positive_count",
                "partial_detected_count",
            )
        }
        precision = _ratio(
            totals["true_positive"],
            totals["true_positive"] + totals["false_positive"],
        )
        recall = _ratio(
            totals["true_positive"],
            totals["true_positive"] + totals["false_negative"],
        )
        camera_rates = tuple(
            item.false_positive_rate
            for item in camera_points
            if item.false_positive_rate is not None
        )
        return cls(
            **totals,
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            specificity=_ratio(
                totals["true_negative"],
                totals["true_negative"] + totals["false_positive"],
            ),
            false_positive_rate=_ratio(
                totals["false_positive"],
                totals["true_negative"] + totals["false_positive"],
            ),
            clear_visibility_recall=_ratio(
                totals["clear_detected_count"],
                totals["clear_positive_count"],
            ),
            partial_visibility_recall=_ratio(
                totals["partial_detected_count"],
                totals["partial_positive_count"],
            ),
            maximum_negative_camera_false_positive_rate=(
                max(camera_rates) if camera_rates else None
            ),
            camera_points=camera_points,
        )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdPoseFusionMetrics:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household pose fusion metrics fields do not match v1"
            )
        if not isinstance(value["camera_points"], list):
            raise HouseholdModelDiagnosticError(
                "household pose fusion camera_points must be an array"
            )
        return cls(
            **{
                key: (
                    tuple(HouseholdModelTemporalCameraPoint.from_dict(item) for item in item_value)
                    if key == "camera_points"
                    else item_value
                )
                for key, item_value in value.items()
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            field: (
                [item.to_dict() for item in self.camera_points]
                if field == "camera_points"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class HouseholdPoseFusionDiagnosticReport:
    review_sha256: str
    media_admission_sha256s: tuple[str, ...]
    detector_scan_sha256s: tuple[str, ...]
    pose_scan_sha256s: tuple[str, ...]
    camera_ids: tuple[str, ...]
    profile: HouseholdPoseFusionProfile
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
    pose_image_size: int
    pose_rotation_degrees: int
    bin_seconds: float
    sampling_strategy: str
    reviewed_frame_count: int
    evaluated_bin_count: int
    positive_bin_count: int
    negative_bin_count: int
    unknown_bin_count: int
    conflicting_bin_count: int
    clear_positive_bin_count: int
    partial_positive_bin_count: int
    metrics: HouseholdPoseFusionMetrics
    version: int = 1
    label: str = "person"
    ground_truth_provenance: str = "model_generated_visual_review"
    fusion_operator: str = "detector_or_gated_pose"
    temporal_aggregation_strategy: str = "independent_causal_signal_holds"
    prediction_anchor: str = "bin_end"
    hold_window_semantics: str = "current_bin_plus_prior_seconds"
    conflict_resolution_strategy: str = "exclude_present_absent_conflicts"
    conflicting_bins_excluded: bool = True
    diagnostic_only: bool = True
    shadow_only: bool = True
    derived_without_model_rerun: bool = True
    model_generated_labels: bool = True
    human_verified: bool = False
    ground_truth: bool = False
    official_accuracy_claim: bool = False
    profile_selected_on_evaluation_data: bool = True
    frame_level_alignment_evaluated: bool = False
    candidate_trigger_coverage_evaluated: bool = False
    box_localization_accuracy_evaluated: bool = False
    danger_recognition_accuracy_evaluated: bool = False
    risk_accuracy_evaluated: bool = False
    winner_selected: bool = False
    deployment_approved: bool = False
    household_configuration_changed: bool = False
    dataset_specific_only: bool = True
    subject_identity_inference: bool = False
    posture_classification: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HouseholdModelDiagnosticError(
                "unsupported household pose fusion diagnostic report version"
            )
        _sha256(self.review_sha256, "pose fusion report review_sha256")
        if not isinstance(self.profile, HouseholdPoseFusionProfile):
            raise HouseholdModelDiagnosticError("household pose fusion report profile is invalid")
        profile_camera_ids = tuple(item.camera_id for item in self.profile.cameras)
        if (
            not isinstance(self.camera_ids, tuple)
            or self.camera_ids != profile_camera_ids
            or len(set(self.camera_ids)) != len(self.camera_ids)
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion report cameras do not match the profile"
            )
        for camera_id in self.camera_ids:
            _safe_id(camera_id, "pose fusion report camera_id")
        for field in (
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "pose_scan_sha256s",
        ):
            values = getattr(self, field)
            if (
                not isinstance(values, tuple)
                or len(values) != len(self.camera_ids)
                or len(set(values)) != len(values)
            ):
                raise HouseholdModelDiagnosticError(
                    f"household pose fusion report {field} is invalid"
                )
            for value in values:
                _sha256(value, f"pose fusion report {field}")
        for field in (
            "detector_adapter",
            "detector_model_id",
            "detector_model_revision",
            "pose_adapter",
            "pose_model_id",
            "pose_model_revision",
        ):
            _safe_id(getattr(self, field), f"pose fusion report {field}")
        _sha256(self.detector_model_sha256, "pose fusion report detector_model_sha256")
        _sha256(self.pose_model_sha256, "pose fusion report pose_model_sha256")
        for field in (
            "detector_source_confidence_threshold",
            "pose_source_confidence_threshold",
        ):
            object.__setattr__(
                self,
                field,
                _number(
                    getattr(self, field),
                    f"pose fusion report {field}",
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
                    f"pose fusion report {field}",
                    minimum=0.01,
                    maximum=10.0,
                ),
            )
        if any(
            item.detector_confidence_threshold < self.detector_source_confidence_threshold
            or item.pose_confidence_threshold < self.pose_source_confidence_threshold
            for item in self.profile.cameras
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion report profile is below a source scan threshold"
            )
        if type(self.pose_requested_max_frames) is not int or not (
            1 <= self.pose_requested_max_frames <= 10_000
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion report pose_requested_max_frames is invalid"
            )
        if type(self.pose_image_size) is not int or not (
            320 <= self.pose_image_size <= 1280 and self.pose_image_size % 32 == 0
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion report pose_image_size is invalid"
            )
        if self.pose_rotation_degrees not in (0, 90, 180, 270):
            raise HouseholdModelDiagnosticError(
                "household pose fusion report pose_rotation_degrees is invalid"
            )
        object.__setattr__(
            self,
            "bin_seconds",
            _number(
                self.bin_seconds,
                "pose fusion report bin_seconds",
                minimum=0.1,
                maximum=60.0,
            ),
        )
        if self.sampling_strategy not in HOUSEHOLD_MODEL_REVIEW_SAMPLING_STRATEGIES:
            raise HouseholdModelDiagnosticError(
                "household pose fusion report sampling_strategy is invalid"
            )
        for field in (
            "reviewed_frame_count",
            "evaluated_bin_count",
            "positive_bin_count",
            "negative_bin_count",
            "unknown_bin_count",
            "conflicting_bin_count",
            "clear_positive_bin_count",
            "partial_positive_bin_count",
        ):
            _count(getattr(self, field), f"pose fusion report {field}")
        if (
            self.reviewed_frame_count < 1
            or self.evaluated_bin_count < 1
            or self.positive_bin_count + self.negative_bin_count != self.evaluated_bin_count
            or self.clear_positive_bin_count + self.partial_positive_bin_count
            != self.positive_bin_count
            or self.evaluated_bin_count + self.unknown_bin_count + self.conflicting_bin_count
            > self.reviewed_frame_count
            or not isinstance(self.metrics, HouseholdPoseFusionMetrics)
            or self.metrics.evaluated_bin_count != self.evaluated_bin_count
            or self.metrics.true_positive + self.metrics.false_negative != self.positive_bin_count
            or self.metrics.true_negative + self.metrics.false_positive != self.negative_bin_count
            or self.metrics.clear_positive_count != self.clear_positive_bin_count
            or self.metrics.partial_positive_count != self.partial_positive_bin_count
            or tuple(item.camera_id for item in self.metrics.camera_points) != self.camera_ids
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion report count accounting is inconsistent"
            )
        if (
            self.label != "person"
            or self.ground_truth_provenance != "model_generated_visual_review"
            or self.fusion_operator != "detector_or_gated_pose"
            or self.temporal_aggregation_strategy != "independent_causal_signal_holds"
            or self.prediction_anchor != "bin_end"
            or self.hold_window_semantics != "current_bin_plus_prior_seconds"
            or self.conflict_resolution_strategy != "exclude_present_absent_conflicts"
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion report semantics are invalid"
            )
        for field, expected in (
            ("conflicting_bins_excluded", True),
            ("diagnostic_only", True),
            ("shadow_only", True),
            ("derived_without_model_rerun", True),
            ("model_generated_labels", True),
            ("human_verified", False),
            ("ground_truth", False),
            ("official_accuracy_claim", False),
            ("profile_selected_on_evaluation_data", True),
            ("frame_level_alignment_evaluated", False),
            ("candidate_trigger_coverage_evaluated", False),
            ("box_localization_accuracy_evaluated", False),
            ("danger_recognition_accuracy_evaluated", False),
            ("risk_accuracy_evaluated", False),
            ("winner_selected", False),
            ("deployment_approved", False),
            ("household_configuration_changed", False),
            ("dataset_specific_only", True),
            ("subject_identity_inference", False),
            ("posture_classification", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdModelDiagnosticError(
                    f"household pose fusion report {field} must be {str(expected).lower()}"
                )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        sequence_fields = {
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "pose_scan_sha256s",
            "camera_ids",
        }
        value: dict[str, object] = {
            field: (
                self.profile.to_dict()
                if field == "profile"
                else self.metrics.to_dict()
                if field == "metrics"
                else list(getattr(self, field))
                if field in sequence_fields
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        if include_fingerprint:
            value["report_sha256"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def to_public_dict(self) -> dict[str, object]:
        value = self.to_dict()
        value["valid"] = True
        value["source_count"] = len(self.camera_ids)
        return value

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdPoseFusionDiagnosticReport:
        fields = {*cls.__dataclass_fields__, "report_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household pose fusion diagnostic report fields do not match v1"
            )
        sequence_fields = {
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "pose_scan_sha256s",
            "camera_ids",
        }
        if any(not isinstance(value[field], list) for field in sequence_fields):
            raise HouseholdModelDiagnosticError(
                "household pose fusion diagnostic report arrays are invalid"
            )
        report_sha256 = value["report_sha256"]
        report = cls(
            **{
                key: (
                    tuple(item_value)
                    if key in sequence_fields
                    else HouseholdPoseFusionProfile.from_dict(item_value)
                    if key == "profile"
                    else HouseholdPoseFusionMetrics.from_dict(item_value)
                    if key == "metrics"
                    else item_value
                )
                for key, item_value in value.items()
                if key != "report_sha256"
            }
        )
        if report_sha256 != report.fingerprint:
            raise HouseholdModelDiagnosticError(
                "household pose fusion diagnostic report fingerprint is invalid"
            )
        return report

    @classmethod
    def from_json(cls, raw: str) -> HouseholdPoseFusionDiagnosticReport:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_POSE_FUSION_REPORT_BYTES:
            raise HouseholdModelDiagnosticError(
                "household pose fusion diagnostic report exceeds the size limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdPoseFusionDiagnosticReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_HOUSEHOLD_POSE_FUSION_REPORT_BYTES:
                raise HouseholdModelDiagnosticError(
                    "household pose fusion diagnostic report exceeds the size limit"
                )
            raw = source.read_text(encoding="utf-8")
        except HouseholdModelDiagnosticError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdModelDiagnosticError(
                "household pose fusion diagnostic report is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_POSE_FUSION_REPORT_BYTES,
        )


@dataclass(frozen=True, slots=True)
class HouseholdPoseFusionBatchBinding:
    report_sha256: str
    review_sha256: str
    media_admission_sha256s: tuple[str, ...]
    detector_scan_sha256s: tuple[str, ...]
    pose_scan_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256(self.report_sha256, "pose fusion batch binding report_sha256")
        _sha256(self.review_sha256, "pose fusion batch binding review_sha256")
        for field in (
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "pose_scan_sha256s",
        ):
            values = getattr(self, field)
            if (
                not isinstance(values, tuple)
                or not 1 <= len(values) <= MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES
                or len(set(values)) != len(values)
            ):
                raise HouseholdModelDiagnosticError(
                    f"household pose fusion batch binding {field} is invalid"
                )
            for value in values:
                _sha256(value, f"pose fusion batch binding {field}")

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdPoseFusionBatchBinding:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch binding fields do not match v1"
            )
        sequence_fields = {
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "pose_scan_sha256s",
        }
        if any(not isinstance(value[field], list) for field in sequence_fields):
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch binding arrays are invalid"
            )
        return cls(
            **{
                key: tuple(item_value) if key in sequence_fields else item_value
                for key, item_value in value.items()
            }
        )

    def to_dict(self) -> dict[str, object]:
        sequence_fields = {
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "pose_scan_sha256s",
        }
        return {
            field: (
                list(getattr(self, field)) if field in sequence_fields else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class HouseholdPoseFusionBatchDiagnosticReport:
    batches: tuple[HouseholdPoseFusionBatchBinding, ...]
    camera_ids: tuple[str, ...]
    profile: HouseholdPoseFusionProfile
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
    pose_image_size: int
    pose_rotation_degrees: int
    bin_seconds: float
    sampling_strategy: str
    reviewed_frame_count: int
    evaluated_bin_count: int
    positive_bin_count: int
    negative_bin_count: int
    unknown_bin_count: int
    conflicting_bin_count: int
    clear_positive_bin_count: int
    partial_positive_bin_count: int
    mixed_label_camera_count: int
    metrics: HouseholdPoseFusionMetrics
    version: int = 1
    label: str = "person"
    ground_truth_provenance: str = "model_generated_visual_review"
    aggregation_strategy: str = "sum_disjoint_pose_fusion_diagnostic_batches"
    fusion_operator: str = "detector_or_gated_pose"
    temporal_aggregation_strategy: str = "independent_causal_signal_holds"
    prediction_anchor: str = "bin_end"
    hold_window_semantics: str = "current_bin_plus_prior_seconds"
    conflict_resolution_strategy: str = "exclude_present_absent_conflicts"
    cross_batch_aggregation: bool = True
    input_reports_distinct: bool = True
    media_sets_disjoint: bool = True
    detector_scan_sets_disjoint: bool = True
    pose_scan_sets_disjoint: bool = True
    conflicting_bins_excluded: bool = True
    diagnostic_only: bool = True
    shadow_only: bool = True
    derived_without_model_rerun: bool = True
    model_generated_labels: bool = True
    human_verified: bool = False
    ground_truth: bool = False
    official_accuracy_claim: bool = False
    profile_selected_on_evaluation_data: bool = True
    frame_level_alignment_evaluated: bool = False
    candidate_trigger_coverage_evaluated: bool = False
    box_localization_accuracy_evaluated: bool = False
    danger_recognition_accuracy_evaluated: bool = False
    risk_accuracy_evaluated: bool = False
    winner_selected: bool = False
    deployment_approved: bool = False
    household_configuration_changed: bool = False
    dataset_specific_only: bool = True
    subject_identity_inference: bool = False
    posture_classification: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HouseholdModelDiagnosticError(
                "unsupported household pose fusion batch report version"
            )
        if (
            not isinstance(self.batches, tuple)
            or not 2 <= len(self.batches) <= MAX_HOUSEHOLD_POSE_FUSION_BATCHES
            or any(not isinstance(item, HouseholdPoseFusionBatchBinding) for item in self.batches)
        ):
            raise HouseholdModelDiagnosticError("household pose fusion batch bindings are invalid")
        report_hashes = tuple(item.report_sha256 for item in self.batches)
        review_hashes = tuple(item.review_sha256 for item in self.batches)
        if (
            report_hashes != tuple(sorted(report_hashes))
            or len(set(report_hashes)) != len(report_hashes)
            or len(set(review_hashes)) != len(review_hashes)
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch reports and reviews must be distinct"
            )
        if not isinstance(self.profile, HouseholdPoseFusionProfile):
            raise HouseholdModelDiagnosticError("household pose fusion batch profile is invalid")
        profile_camera_ids = tuple(item.camera_id for item in self.profile.cameras)
        if not isinstance(self.camera_ids, tuple) or self.camera_ids != profile_camera_ids:
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch cameras do not match the profile"
            )
        for binding in self.batches:
            if any(
                len(getattr(binding, field)) != len(self.camera_ids)
                for field in (
                    "media_admission_sha256s",
                    "detector_scan_sha256s",
                    "pose_scan_sha256s",
                )
            ):
                raise HouseholdModelDiagnosticError(
                    "household pose fusion batch binding source count is inconsistent"
                )
        for field in (
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "pose_scan_sha256s",
        ):
            values = tuple(value for item in self.batches for value in getattr(item, field))
            if len(set(values)) != len(values):
                raise HouseholdModelDiagnosticError(
                    "household pose fusion batch inputs must be disjoint"
                )
        for field in (
            "detector_adapter",
            "detector_model_id",
            "detector_model_revision",
            "pose_adapter",
            "pose_model_id",
            "pose_model_revision",
        ):
            _safe_id(getattr(self, field), f"pose fusion batch {field}")
        _sha256(self.detector_model_sha256, "pose fusion batch detector_model_sha256")
        _sha256(self.pose_model_sha256, "pose fusion batch pose_model_sha256")
        for field in (
            "detector_source_confidence_threshold",
            "pose_source_confidence_threshold",
        ):
            object.__setattr__(
                self,
                field,
                _number(
                    getattr(self, field),
                    f"pose fusion batch {field}",
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
                    f"pose fusion batch {field}",
                    minimum=0.01,
                    maximum=10.0,
                ),
            )
        if any(
            item.detector_confidence_threshold < self.detector_source_confidence_threshold
            or item.pose_confidence_threshold < self.pose_source_confidence_threshold
            for item in self.profile.cameras
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch profile is below a source scan threshold"
            )
        if type(self.pose_requested_max_frames) is not int or not (
            1 <= self.pose_requested_max_frames <= 10_000
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch pose_requested_max_frames is invalid"
            )
        if type(self.pose_image_size) is not int or not (
            320 <= self.pose_image_size <= 1280 and self.pose_image_size % 32 == 0
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch pose_image_size is invalid"
            )
        if self.pose_rotation_degrees not in (0, 90, 180, 270):
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch pose_rotation_degrees is invalid"
            )
        object.__setattr__(
            self,
            "bin_seconds",
            _number(
                self.bin_seconds,
                "pose fusion batch bin_seconds",
                minimum=0.1,
                maximum=60.0,
            ),
        )
        if self.sampling_strategy not in HOUSEHOLD_MODEL_REVIEW_SAMPLING_STRATEGIES:
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch sampling_strategy is invalid"
            )
        for field in (
            "reviewed_frame_count",
            "evaluated_bin_count",
            "positive_bin_count",
            "negative_bin_count",
            "unknown_bin_count",
            "conflicting_bin_count",
            "clear_positive_bin_count",
            "partial_positive_bin_count",
            "mixed_label_camera_count",
        ):
            _count(getattr(self, field), f"pose fusion batch {field}")
        if (
            self.reviewed_frame_count < len(self.batches)
            or self.evaluated_bin_count < 1
            or self.positive_bin_count + self.negative_bin_count != self.evaluated_bin_count
            or self.clear_positive_bin_count + self.partial_positive_bin_count
            != self.positive_bin_count
            or self.evaluated_bin_count + self.unknown_bin_count + self.conflicting_bin_count
            > self.reviewed_frame_count
            or self.mixed_label_camera_count > len(self.camera_ids)
            or not isinstance(self.metrics, HouseholdPoseFusionMetrics)
            or self.metrics.evaluated_bin_count != self.evaluated_bin_count
            or self.metrics.true_positive + self.metrics.false_negative != self.positive_bin_count
            or self.metrics.true_negative + self.metrics.false_positive != self.negative_bin_count
            or self.metrics.clear_positive_count != self.clear_positive_bin_count
            or self.metrics.partial_positive_count != self.partial_positive_bin_count
            or tuple(item.camera_id for item in self.metrics.camera_points) != self.camera_ids
            or self.mixed_label_camera_count
            != sum(
                item.positive_bin_count > 0 and item.negative_bin_count > 0
                for item in self.metrics.camera_points
            )
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch count accounting is inconsistent"
            )
        if (
            self.label != "person"
            or self.ground_truth_provenance != "model_generated_visual_review"
            or self.aggregation_strategy != "sum_disjoint_pose_fusion_diagnostic_batches"
            or self.fusion_operator != "detector_or_gated_pose"
            or self.temporal_aggregation_strategy != "independent_causal_signal_holds"
            or self.prediction_anchor != "bin_end"
            or self.hold_window_semantics != "current_bin_plus_prior_seconds"
            or self.conflict_resolution_strategy != "exclude_present_absent_conflicts"
        ):
            raise HouseholdModelDiagnosticError("household pose fusion batch semantics are invalid")
        for field, expected in (
            ("cross_batch_aggregation", True),
            ("input_reports_distinct", True),
            ("media_sets_disjoint", True),
            ("detector_scan_sets_disjoint", True),
            ("pose_scan_sets_disjoint", True),
            ("conflicting_bins_excluded", True),
            ("diagnostic_only", True),
            ("shadow_only", True),
            ("derived_without_model_rerun", True),
            ("model_generated_labels", True),
            ("human_verified", False),
            ("ground_truth", False),
            ("official_accuracy_claim", False),
            ("profile_selected_on_evaluation_data", True),
            ("frame_level_alignment_evaluated", False),
            ("candidate_trigger_coverage_evaluated", False),
            ("box_localization_accuracy_evaluated", False),
            ("danger_recognition_accuracy_evaluated", False),
            ("risk_accuracy_evaluated", False),
            ("winner_selected", False),
            ("deployment_approved", False),
            ("household_configuration_changed", False),
            ("dataset_specific_only", True),
            ("subject_identity_inference", False),
            ("posture_classification", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdModelDiagnosticError(
                    f"household pose fusion batch {field} must be {str(expected).lower()}"
                )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: (
                [item.to_dict() for item in self.batches]
                if field == "batches"
                else list(self.camera_ids)
                if field == "camera_ids"
                else self.profile.to_dict()
                if field == "profile"
                else self.metrics.to_dict()
                if field == "metrics"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        if include_fingerprint:
            value["report_sha256"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def to_public_dict(self) -> dict[str, object]:
        value = self.to_dict()
        value["valid"] = True
        value["batch_count"] = len(self.batches)
        return value

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdPoseFusionBatchDiagnosticReport:
        fields = {*cls.__dataclass_fields__, "report_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch report fields do not match v1"
            )
        if not isinstance(value["batches"], list) or not isinstance(value["camera_ids"], list):
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch report arrays are invalid"
            )
        report_sha256 = value["report_sha256"]
        report = cls(
            **{
                key: (
                    tuple(HouseholdPoseFusionBatchBinding.from_dict(item) for item in item_value)
                    if key == "batches"
                    else tuple(item_value)
                    if key == "camera_ids"
                    else HouseholdPoseFusionProfile.from_dict(item_value)
                    if key == "profile"
                    else HouseholdPoseFusionMetrics.from_dict(item_value)
                    if key == "metrics"
                    else item_value
                )
                for key, item_value in value.items()
                if key != "report_sha256"
            }
        )
        if report_sha256 != report.fingerprint:
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch report fingerprint is invalid"
            )
        return report

    @classmethod
    def from_json(cls, raw: str) -> HouseholdPoseFusionBatchDiagnosticReport:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_POSE_FUSION_REPORT_BYTES:
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch report exceeds the size limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdPoseFusionBatchDiagnosticReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_HOUSEHOLD_POSE_FUSION_REPORT_BYTES:
                raise HouseholdModelDiagnosticError(
                    "household pose fusion batch report exceeds the size limit"
                )
            raw = source.read_text(encoding="utf-8")
        except HouseholdModelDiagnosticError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdModelDiagnosticError(
                "household pose fusion batch report is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_POSE_FUSION_REPORT_BYTES,
        )


@dataclass(frozen=True, slots=True)
class _FusionSample:
    camera_id: str
    media_admission_sha256: str
    bin_index: int
    label: str
    partial: bool
    predicted: bool


def _validate_pose_scans(
    review: HouseholdModelPresenceReview,
    scans: tuple[AdmittedPoseScan, ...],
) -> dict[tuple[str, str], AdmittedPoseScan]:
    if (
        not isinstance(scans, tuple)
        or len(scans) != len(review.sources)
        or any(not isinstance(item, AdmittedPoseScan) for item in scans)
    ):
        raise HouseholdModelDiagnosticError(
            "household pose fusion admitted pose scans are incomplete"
        )
    by_source = {(scan.result.camera_id, scan.source_admission_sha256): scan for scan in scans}
    expected = {(source.camera_id, source.media_admission_sha256) for source in review.sources}
    if len(by_source) != len(scans) or set(by_source) != expected:
        raise HouseholdModelDiagnosticError(
            "household pose fusion admitted pose scans do not match review sources"
        )
    first = scans[0]
    identity = (
        first.result.pose_adapter,
        first.result.model_id,
        first.result.model_revision,
        first.result.model_sha256,
        first.confidence_threshold,
        first.requested_sample_fps,
        first.requested_max_frames,
        first.image_size,
        first.rotation_degrees,
    )
    for scan in scans:
        if (
            scan.result.pose_adapter,
            scan.result.model_id,
            scan.result.model_revision,
            scan.result.model_sha256,
            scan.confidence_threshold,
            scan.requested_sample_fps,
            scan.requested_max_frames,
            scan.image_size,
            scan.rotation_degrees,
        ) != identity:
            raise HouseholdModelDiagnosticError(
                "household pose fusion scans must share one exact pose configuration"
            )
    source_by_key = {
        (source.camera_id, source.media_admission_sha256): source for source in review.sources
    }
    for key, scan in by_source.items():
        duration = source_by_key[key].source_duration_seconds
        if any(frame.timestamp_seconds > duration + 1e-9 for frame in scan.result.frames):
            raise HouseholdModelDiagnosticError(
                "household pose fusion pose frame exceeds source duration"
            )
    return by_source


def _pose_passes(profile: HouseholdPoseFusionCameraProfile, pose: Any) -> bool:
    x1, y1, x2, y2 = pose.bbox_xyxy
    width = x2 - x1
    height = y2 - y1
    usable_keypoints = sum(
        item.in_frame and item.confidence >= profile.keypoint_confidence_threshold
        for item in pose.keypoints
    )
    return (
        pose.confidence >= profile.pose_confidence_threshold
        and width * height >= profile.minimum_pose_bbox_area
        and height >= profile.minimum_pose_bbox_height
        and usable_keypoints >= profile.minimum_keypoint_count
    )


def household_pose_fusion_bin_prediction(
    profile: HouseholdPoseFusionCameraProfile,
    detector_scan: AdmittedDetectorScan,
    pose_scan: AdmittedPoseScan,
    *,
    bin_index: int,
    bin_seconds: float,
) -> bool:
    """Replay the exact causal detector/Pose OR rule for one admitted source bin."""

    if not isinstance(profile, HouseholdPoseFusionCameraProfile):
        raise HouseholdModelDiagnosticError("household pose fusion camera profile is invalid")
    if not isinstance(detector_scan, AdmittedDetectorScan):
        raise HouseholdModelDiagnosticError("household pose fusion detector scan is invalid")
    if not isinstance(pose_scan, AdmittedPoseScan):
        raise HouseholdModelDiagnosticError("household pose fusion pose scan is invalid")
    if type(bin_index) is not int or not 0 <= bin_index <= 10_000_000:
        raise HouseholdModelDiagnosticError("household pose fusion bin_index is invalid")
    bin_seconds = _number(
        bin_seconds,
        "pose fusion bin_seconds",
        minimum=0.1,
        maximum=60.0,
    )
    detector_key = (
        detector_scan.result.camera_id,
        detector_scan.source_admission_sha256,
    )
    pose_key = (
        pose_scan.result.camera_id,
        pose_scan.source_admission_sha256,
    )
    if detector_key != pose_key or detector_key[0] != profile.camera_id:
        raise HouseholdModelDiagnosticError(
            "household pose fusion bin inputs do not share one profile source"
        )
    if (
        profile.detector_confidence_threshold < detector_scan.confidence_threshold
        or profile.pose_confidence_threshold < pose_scan.confidence_threshold
    ):
        raise HouseholdModelDiagnosticError(
            "household pose fusion profile is below an admitted scan threshold"
        )

    bin_start = bin_index * bin_seconds
    bin_end = (bin_index + 1) * bin_seconds
    detector_start = max(0.0, bin_start - profile.detector_hold_seconds)
    detector_hit = any(
        detection.confidence >= profile.detector_confidence_threshold
        and detector_start <= detection.timestamp_seconds < bin_end
        for detection in detector_scan.result.detections
    )
    if detector_hit or not profile.pose_enabled:
        return detector_hit
    pose_start = max(0.0, bin_start - profile.pose_hold_seconds)
    return any(
        pose_start <= frame.timestamp_seconds < bin_end
        and any(_pose_passes(profile, pose) for pose in frame.poses)
        for frame in pose_scan.result.frames
    )


def _camera_point(
    camera_id: str,
    samples: tuple[_FusionSample, ...],
) -> HouseholdModelTemporalCameraPoint:
    positive = tuple(item for item in samples if item.label == "present")
    negative = tuple(item for item in samples if item.label == "absent")
    true_positive = sum(item.predicted for item in positive)
    false_positive = sum(item.predicted for item in negative)
    clear = tuple(item for item in positive if not item.partial)
    partial = tuple(item for item in positive if item.partial)
    clear_detected = sum(item.predicted for item in clear)
    partial_detected = sum(item.predicted for item in partial)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, len(positive))
    return HouseholdModelTemporalCameraPoint(
        camera_id=camera_id,
        evaluated_bin_count=len(samples),
        positive_bin_count=len(positive),
        negative_bin_count=len(negative),
        clear_positive_count=len(clear),
        partial_positive_count=len(partial),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=len(positive) - true_positive,
        true_negative=len(negative) - false_positive,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        specificity=_ratio(len(negative) - false_positive, len(negative)),
        false_positive_rate=_ratio(false_positive, len(negative)),
        clear_detected_count=clear_detected,
        clear_visibility_recall=_ratio(clear_detected, len(clear)),
        partial_detected_count=partial_detected,
        partial_visibility_recall=_ratio(partial_detected, len(partial)),
    )


def evaluate_household_pose_fusion_diagnostic(
    review: HouseholdModelPresenceReview,
    *,
    media_admissions: tuple[ExternalSourceAdmission, ...],
    detector_scans: tuple[AdmittedDetectorScan, ...],
    pose_scans: tuple[AdmittedPoseScan, ...],
    profile: HouseholdPoseFusionProfile,
    bin_seconds: float,
) -> HouseholdPoseFusionDiagnosticReport:
    """Replay one explicit per-camera OR profile without reading source media."""

    if not isinstance(review, HouseholdModelPresenceReview):
        raise HouseholdModelDiagnosticError("household pose fusion diagnostic review is invalid")
    if not isinstance(profile, HouseholdPoseFusionProfile):
        raise HouseholdModelDiagnosticError("household pose fusion diagnostic profile is invalid")
    _validate_admissions(review, media_admissions)
    detector_by_source = _validate_scans(review, detector_scans)
    pose_by_source = _validate_pose_scans(review, pose_scans)
    bin_seconds = _number(
        bin_seconds,
        "pose fusion diagnostic bin_seconds",
        minimum=0.1,
        maximum=60.0,
    )
    ordered_sources = tuple(sorted(review.sources, key=lambda item: item.camera_id))
    camera_ids = tuple(item.camera_id for item in ordered_sources)
    if tuple(item.camera_id for item in profile.cameras) != camera_ids:
        raise HouseholdModelDiagnosticError(
            "household pose fusion profile does not match review cameras"
        )
    profile_by_camera = {item.camera_id: item for item in profile.cameras}
    first_detector = detector_scans[0]
    first_pose = pose_scans[0]
    for camera_profile in profile.cameras:
        if (
            camera_profile.detector_confidence_threshold < first_detector.confidence_threshold
            or camera_profile.pose_confidence_threshold < first_pose.confidence_threshold
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion profile is below an admitted scan threshold"
            )

    grouped_items: dict[tuple[str, str, int], list[HouseholdModelPresenceItem]] = defaultdict(list)
    for item in review.items:
        grouped_items[
            (
                item.camera_id,
                item.media_admission_sha256,
                int(item.timestamp_seconds // bin_seconds),
            )
        ].append(item)

    samples: list[_FusionSample] = []
    unknown_bin_count = 0
    conflicting_bin_count = 0
    for (camera_id, admission_sha256, bin_index), items in sorted(grouped_items.items()):
        labels = {item.visible_person for item in items}
        if "present" in labels and "absent" in labels:
            conflicting_bin_count += 1
            continue
        if "unknown" in labels:
            unknown_bin_count += 1
            continue
        label = labels.pop()
        camera_profile = profile_by_camera[camera_id]
        predicted = household_pose_fusion_bin_prediction(
            camera_profile,
            detector_by_source[(camera_id, admission_sha256)],
            pose_by_source[(camera_id, admission_sha256)],
            bin_index=bin_index,
            bin_seconds=bin_seconds,
        )
        samples.append(
            _FusionSample(
                camera_id=camera_id,
                media_admission_sha256=admission_sha256,
                bin_index=bin_index,
                label=label,
                partial=(
                    label == "present" and any(item.visibility == "partial" for item in items)
                ),
                predicted=predicted,
            )
        )
    if not samples:
        raise HouseholdModelDiagnosticError(
            "household pose fusion diagnostic contains no evaluable bins"
        )

    camera_points = tuple(
        _camera_point(
            camera_id,
            tuple(item for item in samples if item.camera_id == camera_id),
        )
        for camera_id in camera_ids
    )
    metrics = HouseholdPoseFusionMetrics.from_camera_points(camera_points)
    positive_bin_count = sum(item.label == "present" for item in samples)
    partial_positive_bin_count = sum(item.partial for item in samples)
    ordered_detector_scans = tuple(
        detector_by_source[(source.camera_id, source.media_admission_sha256)]
        for source in ordered_sources
    )
    ordered_pose_scans = tuple(
        pose_by_source[(source.camera_id, source.media_admission_sha256)]
        for source in ordered_sources
    )
    return HouseholdPoseFusionDiagnosticReport(
        review_sha256=review.fingerprint,
        media_admission_sha256s=tuple(source.media_admission_sha256 for source in ordered_sources),
        detector_scan_sha256s=tuple(item.fingerprint for item in ordered_detector_scans),
        pose_scan_sha256s=tuple(item.fingerprint for item in ordered_pose_scans),
        camera_ids=camera_ids,
        profile=profile,
        detector_adapter=first_detector.result.detector_adapter,
        detector_model_id=first_detector.result.model_id,
        detector_model_revision=first_detector.result.model_revision,
        detector_model_sha256=first_detector.result.model_sha256,
        detector_source_confidence_threshold=first_detector.confidence_threshold,
        detector_requested_sample_fps=first_detector.requested_sample_fps,
        pose_adapter=first_pose.result.pose_adapter,
        pose_model_id=first_pose.result.model_id,
        pose_model_revision=first_pose.result.model_revision,
        pose_model_sha256=first_pose.result.model_sha256,
        pose_source_confidence_threshold=first_pose.confidence_threshold,
        pose_requested_sample_fps=first_pose.requested_sample_fps,
        pose_requested_max_frames=first_pose.requested_max_frames,
        pose_image_size=first_pose.image_size,
        pose_rotation_degrees=first_pose.rotation_degrees,
        bin_seconds=bin_seconds,
        sampling_strategy=review.sampling_strategy,
        reviewed_frame_count=len(review.items),
        evaluated_bin_count=len(samples),
        positive_bin_count=positive_bin_count,
        negative_bin_count=len(samples) - positive_bin_count,
        unknown_bin_count=unknown_bin_count,
        conflicting_bin_count=conflicting_bin_count,
        clear_positive_bin_count=positive_bin_count - partial_positive_bin_count,
        partial_positive_bin_count=partial_positive_bin_count,
        metrics=metrics,
    )


def _combine_camera_points(
    camera_id: str,
    points: tuple[HouseholdModelTemporalCameraPoint, ...],
) -> HouseholdModelTemporalCameraPoint:
    totals = {
        field: sum(getattr(item, field) for item in points)
        for field in (
            "evaluated_bin_count",
            "positive_bin_count",
            "negative_bin_count",
            "clear_positive_count",
            "partial_positive_count",
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "clear_detected_count",
            "partial_detected_count",
        )
    }
    precision = _ratio(
        totals["true_positive"],
        totals["true_positive"] + totals["false_positive"],
    )
    recall = _ratio(totals["true_positive"], totals["positive_bin_count"])
    return HouseholdModelTemporalCameraPoint(
        camera_id=camera_id,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        specificity=_ratio(totals["true_negative"], totals["negative_bin_count"]),
        false_positive_rate=_ratio(totals["false_positive"], totals["negative_bin_count"]),
        clear_visibility_recall=_ratio(
            totals["clear_detected_count"], totals["clear_positive_count"]
        ),
        partial_visibility_recall=_ratio(
            totals["partial_detected_count"], totals["partial_positive_count"]
        ),
        **totals,
    )


def combine_household_pose_fusion_diagnostic_batches(
    reports: tuple[HouseholdPoseFusionDiagnosticReport, ...],
) -> HouseholdPoseFusionBatchDiagnosticReport:
    """Combine disjoint same-profile reports without reading media or rerunning models."""

    if (
        not isinstance(reports, tuple)
        or not 2 <= len(reports) <= MAX_HOUSEHOLD_POSE_FUSION_BATCHES
        or any(not isinstance(item, HouseholdPoseFusionDiagnosticReport) for item in reports)
    ):
        raise HouseholdModelDiagnosticError("household pose fusion batch reports are invalid")
    reports = tuple(sorted(reports, key=lambda item: item.fingerprint))
    first = reports[0]
    identity = (
        first.camera_ids,
        first.profile,
        first.detector_adapter,
        first.detector_model_id,
        first.detector_model_revision,
        first.detector_model_sha256,
        first.detector_source_confidence_threshold,
        first.detector_requested_sample_fps,
        first.pose_adapter,
        first.pose_model_id,
        first.pose_model_revision,
        first.pose_model_sha256,
        first.pose_source_confidence_threshold,
        first.pose_requested_sample_fps,
        first.pose_requested_max_frames,
        first.pose_image_size,
        first.pose_rotation_degrees,
        first.bin_seconds,
        first.sampling_strategy,
        first.label,
        first.fusion_operator,
        first.temporal_aggregation_strategy,
        first.prediction_anchor,
        first.hold_window_semantics,
        first.conflict_resolution_strategy,
    )
    for report in reports:
        if (
            report.camera_ids,
            report.profile,
            report.detector_adapter,
            report.detector_model_id,
            report.detector_model_revision,
            report.detector_model_sha256,
            report.detector_source_confidence_threshold,
            report.detector_requested_sample_fps,
            report.pose_adapter,
            report.pose_model_id,
            report.pose_model_revision,
            report.pose_model_sha256,
            report.pose_source_confidence_threshold,
            report.pose_requested_sample_fps,
            report.pose_requested_max_frames,
            report.pose_image_size,
            report.pose_rotation_degrees,
            report.bin_seconds,
            report.sampling_strategy,
            report.label,
            report.fusion_operator,
            report.temporal_aggregation_strategy,
            report.prediction_anchor,
            report.hold_window_semantics,
            report.conflict_resolution_strategy,
        ) != identity:
            raise HouseholdModelDiagnosticError(
                "household pose fusion batches must share one exact profile and model config"
            )
    if len({item.fingerprint for item in reports}) != len(reports) or len(
        {item.review_sha256 for item in reports}
    ) != len(reports):
        raise HouseholdModelDiagnosticError(
            "household pose fusion batches must bind distinct reports and reviews"
        )
    for field in (
        "media_admission_sha256s",
        "detector_scan_sha256s",
        "pose_scan_sha256s",
    ):
        values = tuple(value for report in reports for value in getattr(report, field))
        if len(set(values)) != len(values):
            raise HouseholdModelDiagnosticError(
                "household pose fusion batches require disjoint admitted inputs"
            )

    camera_points = tuple(
        _combine_camera_points(
            camera_id,
            tuple(
                next(
                    point for point in report.metrics.camera_points if point.camera_id == camera_id
                )
                for report in reports
            ),
        )
        for camera_id in first.camera_ids
    )
    metrics = HouseholdPoseFusionMetrics.from_camera_points(camera_points)
    bindings = tuple(
        HouseholdPoseFusionBatchBinding(
            report_sha256=report.fingerprint,
            review_sha256=report.review_sha256,
            media_admission_sha256s=report.media_admission_sha256s,
            detector_scan_sha256s=report.detector_scan_sha256s,
            pose_scan_sha256s=report.pose_scan_sha256s,
        )
        for report in reports
    )
    return HouseholdPoseFusionBatchDiagnosticReport(
        batches=bindings,
        camera_ids=first.camera_ids,
        profile=first.profile,
        detector_adapter=first.detector_adapter,
        detector_model_id=first.detector_model_id,
        detector_model_revision=first.detector_model_revision,
        detector_model_sha256=first.detector_model_sha256,
        detector_source_confidence_threshold=first.detector_source_confidence_threshold,
        detector_requested_sample_fps=first.detector_requested_sample_fps,
        pose_adapter=first.pose_adapter,
        pose_model_id=first.pose_model_id,
        pose_model_revision=first.pose_model_revision,
        pose_model_sha256=first.pose_model_sha256,
        pose_source_confidence_threshold=first.pose_source_confidence_threshold,
        pose_requested_sample_fps=first.pose_requested_sample_fps,
        pose_requested_max_frames=first.pose_requested_max_frames,
        pose_image_size=first.pose_image_size,
        pose_rotation_degrees=first.pose_rotation_degrees,
        bin_seconds=first.bin_seconds,
        sampling_strategy=first.sampling_strategy,
        reviewed_frame_count=sum(item.reviewed_frame_count for item in reports),
        evaluated_bin_count=sum(item.evaluated_bin_count for item in reports),
        positive_bin_count=sum(item.positive_bin_count for item in reports),
        negative_bin_count=sum(item.negative_bin_count for item in reports),
        unknown_bin_count=sum(item.unknown_bin_count for item in reports),
        conflicting_bin_count=sum(item.conflicting_bin_count for item in reports),
        clear_positive_bin_count=sum(item.clear_positive_bin_count for item in reports),
        partial_positive_bin_count=sum(item.partial_positive_bin_count for item in reports),
        mixed_label_camera_count=sum(
            item.positive_bin_count > 0 and item.negative_bin_count > 0 for item in camera_points
        ),
        metrics=metrics,
    )
