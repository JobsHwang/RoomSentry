"""Causal temporal and camera-stratified household detector diagnostics.

The evaluator in this module consumes only an already-closed model-generated
presence review, verified local-only admissions, and saved detector scans.  It
never reads media, reruns a model, mutates configuration, or grants deployment
authority.  A detection may affect its current review bin and, when a hold is
configured, later bins; a detection after a bin end can never affect that bin.
"""

from __future__ import annotations

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
    _thresholds,
    _validate_admissions,
    _validate_scans,
    _write_exclusive,
)
from .source_admission import ExternalSourceAdmission

MAX_HOUSEHOLD_MODEL_TEMPORAL_DIAGNOSTIC_BYTES = 4 * 1024 * 1024
MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLDS = 16
MAX_HOUSEHOLD_MODEL_TEMPORAL_POINTS = 512
MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLD_SECONDS = 300.0

HOUSEHOLD_MODEL_TEMPORAL_SELECTION_RULE = (
    "minimum_positive_camera_recall_then_recall_then_partial_recall_"
    "then_clear_recall_then_precision_then_f1_then_lower_negative_camera_"
    "false_positive_rate_then_shorter_hold_then_higher_threshold"
)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0.0:
        return None
    return round(2.0 * precision * recall / (precision + recall), 12)


@dataclass(frozen=True, slots=True)
class HouseholdModelTemporalCameraPoint:
    """One anonymous camera stratum at one threshold/hold point."""

    camera_id: str
    evaluated_bin_count: int
    positive_bin_count: int
    negative_bin_count: int
    clear_positive_count: int
    partial_positive_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float | None
    recall: float | None
    f1: float | None
    specificity: float | None
    false_positive_rate: float | None
    clear_detected_count: int
    clear_visibility_recall: float | None
    partial_detected_count: int
    partial_visibility_recall: float | None

    def __post_init__(self) -> None:
        _safe_id(self.camera_id, "temporal camera point camera_id")
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
        ):
            _count(getattr(self, field), f"temporal camera point {field}")
        if (
            self.positive_bin_count + self.negative_bin_count != self.evaluated_bin_count
            or self.clear_positive_count + self.partial_positive_count != self.positive_bin_count
            or self.true_positive + self.false_negative != self.positive_bin_count
            or self.true_negative + self.false_positive != self.negative_bin_count
            or self.clear_detected_count > self.clear_positive_count
            or self.partial_detected_count > self.partial_positive_count
            or self.clear_detected_count + self.partial_detected_count != self.true_positive
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal camera point accounting is inconsistent"
            )
        for field in (
            "precision",
            "recall",
            "f1",
            "specificity",
            "false_positive_rate",
            "clear_visibility_recall",
            "partial_visibility_recall",
        ):
            object.__setattr__(
                self,
                field,
                _optional_ratio(getattr(self, field), f"temporal camera point {field}"),
            )
        expected_precision = _ratio(
            self.true_positive,
            self.true_positive + self.false_positive,
        )
        expected_recall = _ratio(
            self.true_positive,
            self.true_positive + self.false_negative,
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
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal camera point metrics are inconsistent"
            )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelTemporalCameraPoint:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household temporal camera point fields do not match v1"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdModelTemporalDiagnosticPoint:
    """Aggregate and per-camera metrics for one causal temporal setting."""

    confidence_threshold: float
    hold_seconds: float
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
        object.__setattr__(
            self,
            "confidence_threshold",
            _number(
                self.confidence_threshold,
                "temporal point confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "hold_seconds",
            _number(
                self.hold_seconds,
                "temporal point hold_seconds",
                minimum=0.0,
                maximum=MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLD_SECONDS,
            ),
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
            _count(getattr(self, field), f"temporal point {field}")
        if (
            self.clear_detected_count > self.clear_positive_count
            or self.partial_detected_count > self.partial_positive_count
            or self.clear_detected_count + self.partial_detected_count != self.true_positive
            or self.clear_positive_count + self.partial_positive_count
            != self.true_positive + self.false_negative
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic point visibility accounting is inconsistent"
            )
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
                _optional_ratio(getattr(self, field), f"temporal point {field}"),
            )
        expected_precision = _ratio(
            self.true_positive,
            self.true_positive + self.false_positive,
        )
        expected_recall = _ratio(
            self.true_positive,
            self.true_positive + self.false_negative,
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
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic point metrics are inconsistent"
            )
        if (
            not isinstance(self.camera_points, tuple)
            or not 1 <= len(self.camera_points) <= MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES
            or any(
                not isinstance(point, HouseholdModelTemporalCameraPoint)
                for point in self.camera_points
            )
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic camera points are invalid"
            )
        camera_ids = tuple(point.camera_id for point in self.camera_points)
        if camera_ids != tuple(sorted(camera_ids)) or len(set(camera_ids)) != len(camera_ids):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic camera points are not unique and sorted"
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
            if getattr(self, field) != sum(getattr(point, field) for point in self.camera_points):
                raise HouseholdModelDiagnosticError(
                    "household temporal diagnostic camera and aggregate accounting differ"
                )
        camera_rates = tuple(
            point.false_positive_rate
            for point in self.camera_points
            if point.false_positive_rate is not None
        )
        expected_maximum = max(camera_rates) if camera_rates else None
        if self.maximum_negative_camera_false_positive_rate != expected_maximum:
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic maximum camera false-positive rate is inconsistent"
            )

    @property
    def evaluated_bin_count(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelTemporalDiagnosticPoint:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic point fields do not match v1"
            )
        if not isinstance(value["camera_points"], list):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic point camera_points is invalid"
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
                [point.to_dict() for point in self.camera_points]
                if field == "camera_points"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class HouseholdModelTemporalDiagnosticReport:
    """Strict path-free causal hold sweep over model-generated weak labels."""

    review_sha256: str
    media_admission_sha256s: tuple[str, ...]
    detector_scan_sha256s: tuple[str, ...]
    camera_ids: tuple[str, ...]
    detector_adapter: str
    model_id: str
    model_revision: str
    model_sha256: str
    source_confidence_threshold: float
    requested_sample_fps: float
    bin_seconds: float
    sampling_strategy: str
    reviewed_frame_count: int
    evaluated_bin_count: int
    positive_bin_count: int
    negative_bin_count: int
    unknown_bin_count: int
    clear_positive_bin_count: int
    partial_positive_bin_count: int
    thresholds: tuple[float, ...]
    hold_seconds_grid: tuple[float, ...]
    points: tuple[HouseholdModelTemporalDiagnosticPoint, ...]
    version: int = 1
    label: str = "person"
    ground_truth_provenance: str = "model_generated_visual_review"
    temporal_aggregation_strategy: str = "causal_detection_hold"
    prediction_anchor: str = "bin_end"
    hold_window_semantics: str = "current_bin_plus_prior_seconds"
    stratification_axes: tuple[str, ...] = ("camera_id", "visibility")
    causal_temporal_aggregation: bool = True
    future_detections_used: bool = False
    diagnostic_only: bool = True
    shadow_only: bool = True
    derived_without_model_rerun: bool = True
    model_generated_labels: bool = True
    human_verified: bool = False
    ground_truth: bool = False
    official_accuracy_claim: bool = False
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
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HouseholdModelDiagnosticError(
                "unsupported household temporal diagnostic report version"
            )
        _sha256(self.review_sha256, "temporal report review_sha256")
        for field in ("detector_adapter", "model_id", "model_revision"):
            _safe_id(getattr(self, field), f"temporal report {field}")
        _sha256(self.model_sha256, "temporal report model_sha256")
        if (
            not isinstance(self.camera_ids, tuple)
            or not 1 <= len(self.camera_ids) <= MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES
            or self.camera_ids != tuple(sorted(self.camera_ids))
            or len(set(self.camera_ids)) != len(self.camera_ids)
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic camera IDs are invalid"
            )
        for camera_id in self.camera_ids:
            _safe_id(camera_id, "temporal report camera_id")
        for field in ("media_admission_sha256s", "detector_scan_sha256s"):
            values = getattr(self, field)
            if (
                not isinstance(values, tuple)
                or len(values) != len(self.camera_ids)
                or len(set(values)) != len(values)
            ):
                raise HouseholdModelDiagnosticError(
                    f"household temporal diagnostic {field} is invalid"
                )
            for value in values:
                _sha256(value, f"temporal report {field}")
        object.__setattr__(
            self,
            "source_confidence_threshold",
            _number(
                self.source_confidence_threshold,
                "temporal report source_confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "requested_sample_fps",
            _number(
                self.requested_sample_fps,
                "temporal report requested_sample_fps",
                minimum=0.01,
                maximum=10.0,
            ),
        )
        object.__setattr__(
            self,
            "bin_seconds",
            _number(
                self.bin_seconds,
                "temporal report bin_seconds",
                minimum=0.1,
                maximum=60.0,
            ),
        )
        if self.sampling_strategy not in HOUSEHOLD_MODEL_REVIEW_SAMPLING_STRATEGIES:
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic sampling strategy is invalid"
            )
        for field in (
            "reviewed_frame_count",
            "evaluated_bin_count",
            "positive_bin_count",
            "negative_bin_count",
            "unknown_bin_count",
            "clear_positive_bin_count",
            "partial_positive_bin_count",
        ):
            _count(getattr(self, field), f"temporal report {field}")
        if (
            self.reviewed_frame_count < 1
            or self.evaluated_bin_count < 1
            or self.positive_bin_count + self.negative_bin_count != self.evaluated_bin_count
            or self.clear_positive_bin_count + self.partial_positive_bin_count
            != self.positive_bin_count
            or self.evaluated_bin_count + self.unknown_bin_count > self.reviewed_frame_count
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic report bin accounting is inconsistent"
            )
        if not isinstance(self.thresholds, tuple) or self.thresholds != _thresholds(
            self.thresholds,
            source_threshold=self.source_confidence_threshold,
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic thresholds are invalid"
            )
        if (
            not isinstance(self.hold_seconds_grid, tuple)
            or not 2 <= len(self.hold_seconds_grid) <= MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLDS
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic hold grid is invalid"
            )
        normalized_holds = tuple(
            _number(
                value,
                "temporal report hold_seconds_grid",
                minimum=0.0,
                maximum=MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLD_SECONDS,
            )
            for value in self.hold_seconds_grid
        )
        if (
            normalized_holds != tuple(sorted(normalized_holds))
            or len(set(normalized_holds)) != len(normalized_holds)
            or normalized_holds[0] != 0.0
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic holds must be unique, ascending, and start at zero"
            )
        object.__setattr__(self, "hold_seconds_grid", normalized_holds)
        expected_keys = tuple(
            (threshold, hold_seconds)
            for hold_seconds in self.hold_seconds_grid
            for threshold in self.thresholds
        )
        if (
            not isinstance(self.points, tuple)
            or len(self.points) != len(expected_keys)
            or len(self.points) > MAX_HOUSEHOLD_MODEL_TEMPORAL_POINTS
            or any(
                not isinstance(point, HouseholdModelTemporalDiagnosticPoint)
                for point in self.points
            )
            or tuple((point.confidence_threshold, point.hold_seconds) for point in self.points)
            != expected_keys
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic points do not match the canonical grid"
            )
        for point in self.points:
            if (
                point.evaluated_bin_count != self.evaluated_bin_count
                or point.true_positive + point.false_negative != self.positive_bin_count
                or point.true_negative + point.false_positive != self.negative_bin_count
                or point.clear_positive_count != self.clear_positive_bin_count
                or point.partial_positive_count != self.partial_positive_bin_count
                or tuple(camera.camera_id for camera in point.camera_points) != self.camera_ids
            ):
                raise HouseholdModelDiagnosticError(
                    "household temporal diagnostic point does not match report strata"
                )
        by_key = {(point.confidence_threshold, point.hold_seconds): point for point in self.points}
        monotonic_fields = (
            "true_positive",
            "false_positive",
            "clear_detected_count",
            "partial_detected_count",
        )
        for hold_seconds in self.hold_seconds_grid:
            previous: HouseholdModelTemporalDiagnosticPoint | None = None
            for threshold in self.thresholds:
                current = by_key[(threshold, hold_seconds)]
                if previous is not None and any(
                    getattr(current, field) > getattr(previous, field) for field in monotonic_fields
                ):
                    raise HouseholdModelDiagnosticError(
                        "household temporal diagnostic threshold counts are not monotonic"
                    )
                previous = current
        for threshold in self.thresholds:
            previous = None
            for hold_seconds in self.hold_seconds_grid:
                current = by_key[(threshold, hold_seconds)]
                if previous is not None and any(
                    getattr(current, field) < getattr(previous, field) for field in monotonic_fields
                ):
                    raise HouseholdModelDiagnosticError(
                        "household temporal diagnostic hold counts are not monotonic"
                    )
                previous = current
        if self.label != "person" or self.ground_truth_provenance != (
            "model_generated_visual_review"
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic label authority is invalid"
            )
        if (
            self.temporal_aggregation_strategy != "causal_detection_hold"
            or self.prediction_anchor != "bin_end"
            or self.hold_window_semantics != "current_bin_plus_prior_seconds"
            or self.stratification_axes != ("camera_id", "visibility")
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic evaluation semantics are invalid"
            )
        for field, expected in (
            ("causal_temporal_aggregation", True),
            ("future_detections_used", False),
            ("diagnostic_only", True),
            ("shadow_only", True),
            ("derived_without_model_rerun", True),
            ("model_generated_labels", True),
            ("human_verified", False),
            ("ground_truth", False),
            ("official_accuracy_claim", False),
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
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdModelDiagnosticError(
                    f"household temporal diagnostic {field} must be {str(expected).lower()}"
                )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        sequence_fields = {
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "camera_ids",
            "thresholds",
            "hold_seconds_grid",
            "stratification_axes",
        }
        value: dict[str, object] = {
            field: (
                [point.to_dict() for point in self.points]
                if field == "points"
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
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "valid": True,
            "version": self.version,
            "report_sha256": self.fingerprint,
            "review_sha256": self.review_sha256,
            "source_count": len(self.camera_ids),
            "camera_ids": list(self.camera_ids),
            "detector_adapter": self.detector_adapter,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "source_confidence_threshold": self.source_confidence_threshold,
            "requested_sample_fps": self.requested_sample_fps,
            "bin_seconds": self.bin_seconds,
            "sampling_strategy": self.sampling_strategy,
            "reviewed_frame_count": self.reviewed_frame_count,
            "evaluated_bin_count": self.evaluated_bin_count,
            "positive_bin_count": self.positive_bin_count,
            "negative_bin_count": self.negative_bin_count,
            "unknown_bin_count": self.unknown_bin_count,
            "clear_positive_bin_count": self.clear_positive_bin_count,
            "partial_positive_bin_count": self.partial_positive_bin_count,
            "thresholds": list(self.thresholds),
            "hold_seconds_grid": list(self.hold_seconds_grid),
            "points": [point.to_dict() for point in self.points],
            "temporal_aggregation_strategy": self.temporal_aggregation_strategy,
            "prediction_anchor": self.prediction_anchor,
            "hold_window_semantics": self.hold_window_semantics,
            "stratification_axes": list(self.stratification_axes),
            "causal_temporal_aggregation": True,
            "future_detections_used": False,
            "diagnostic_only": True,
            "shadow_only": True,
            "derived_without_model_rerun": True,
            "model_generated_labels": True,
            "human_verified": False,
            "ground_truth": False,
            "official_accuracy_claim": False,
            "frame_level_alignment_evaluated": False,
            "candidate_trigger_coverage_evaluated": False,
            "box_localization_accuracy_evaluated": False,
            "danger_recognition_accuracy_evaluated": False,
            "risk_accuracy_evaluated": False,
            "winner_selected": False,
            "deployment_approved": False,
            "household_configuration_changed": False,
            "dataset_specific_only": True,
            "subject_identity_inference": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelTemporalDiagnosticReport:
        fields = {*cls.__dataclass_fields__, "report_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic report fields do not match v1"
            )
        sequence_fields = {
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "camera_ids",
            "thresholds",
            "hold_seconds_grid",
            "stratification_axes",
            "points",
        }
        if any(not isinstance(value[field], list) for field in sequence_fields):
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic report arrays are invalid"
            )
        report_sha256 = value["report_sha256"]
        report = cls(
            **{
                key: (
                    tuple(
                        HouseholdModelTemporalDiagnosticPoint.from_dict(item) for item in item_value
                    )
                    if key == "points"
                    else tuple(item_value)
                    if key in sequence_fields
                    else item_value
                )
                for key, item_value in value.items()
                if key != "report_sha256"
            }
        )
        if report_sha256 != report.fingerprint:
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic report fingerprint is invalid"
            )
        return report

    @classmethod
    def from_json(cls, raw: str) -> HouseholdModelTemporalDiagnosticReport:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_MODEL_TEMPORAL_DIAGNOSTIC_BYTES:
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic report exceeds the 4 MiB limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdModelTemporalDiagnosticReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not (0 < source.stat().st_size <= MAX_HOUSEHOLD_MODEL_TEMPORAL_DIAGNOSTIC_BYTES):
                raise HouseholdModelDiagnosticError(
                    "household temporal diagnostic report exceeds the size limit"
                )
            raw = source.read_text(encoding="utf-8")
        except HouseholdModelDiagnosticError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdModelDiagnosticError(
                "household temporal diagnostic report is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_MODEL_TEMPORAL_DIAGNOSTIC_BYTES,
        )


@dataclass(frozen=True, slots=True)
class HouseholdModelTemporalDiagnosticPolicy:
    minimum_precision: float
    minimum_recall: float
    minimum_positive_camera_recall: float
    minimum_clear_visibility_recall: float
    minimum_partial_visibility_recall: float
    maximum_negative_camera_false_positive_rate: float
    maximum_hold_seconds: float
    require_clear_visibility_samples: bool = False
    require_partial_visibility_samples: bool = True
    require_negative_camera_samples: bool = True
    require_mixed_labels_per_camera: bool = True

    def __post_init__(self) -> None:
        for field in (
            "minimum_precision",
            "minimum_recall",
            "minimum_positive_camera_recall",
            "minimum_clear_visibility_recall",
            "minimum_partial_visibility_recall",
            "maximum_negative_camera_false_positive_rate",
        ):
            object.__setattr__(
                self,
                field,
                _number(getattr(self, field), field, minimum=0.0, maximum=1.0),
            )
        object.__setattr__(
            self,
            "maximum_hold_seconds",
            _number(
                self.maximum_hold_seconds,
                "maximum_hold_seconds",
                minimum=0.0,
                maximum=MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLD_SECONDS,
            ),
        )
        for field in (
            "require_clear_visibility_samples",
            "require_partial_visibility_samples",
            "require_negative_camera_samples",
            "require_mixed_labels_per_camera",
        ):
            if type(getattr(self, field)) is not bool:
                raise HouseholdModelDiagnosticError(
                    "household temporal diagnostic sample policy is invalid"
                )

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class _TemporalSample:
    camera_id: str
    media_admission_sha256: str
    bin_index: int
    label: str
    partial: bool


def _hold_grid(values: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not 2 <= len(values) <= MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLDS:
        raise HouseholdModelDiagnosticError("household temporal diagnostic hold grid is invalid")
    result = tuple(
        _number(
            value,
            "hold_seconds",
            minimum=0.0,
            maximum=MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLD_SECONDS,
        )
        for value in values
    )
    if result != tuple(sorted(result)) or len(set(result)) != len(result) or result[0] != 0.0:
        raise HouseholdModelDiagnosticError(
            "household temporal diagnostic holds must be unique, ascending, and start at zero"
        )
    return result


def _camera_point(
    camera_id: str,
    samples: tuple[tuple[_TemporalSample, float | None], ...],
    threshold: float,
) -> HouseholdModelTemporalCameraPoint:
    true_positive = false_positive = false_negative = true_negative = 0
    clear_positive_count = partial_positive_count = 0
    clear_detected_count = partial_detected_count = 0
    for sample, score in samples:
        predicted = score is not None and score >= threshold
        if sample.label == "present":
            if sample.partial:
                partial_positive_count += 1
                partial_detected_count += int(predicted)
            else:
                clear_positive_count += 1
                clear_detected_count += int(predicted)
            if predicted:
                true_positive += 1
            else:
                false_negative += 1
        elif predicted:
            false_positive += 1
        else:
            true_negative += 1
    positive_bin_count = true_positive + false_negative
    negative_bin_count = false_positive + true_negative
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, positive_bin_count)
    return HouseholdModelTemporalCameraPoint(
        camera_id=camera_id,
        evaluated_bin_count=positive_bin_count + negative_bin_count,
        positive_bin_count=positive_bin_count,
        negative_bin_count=negative_bin_count,
        clear_positive_count=clear_positive_count,
        partial_positive_count=partial_positive_count,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        specificity=_ratio(true_negative, negative_bin_count),
        false_positive_rate=_ratio(false_positive, negative_bin_count),
        clear_detected_count=clear_detected_count,
        clear_visibility_recall=_ratio(clear_detected_count, clear_positive_count),
        partial_detected_count=partial_detected_count,
        partial_visibility_recall=_ratio(partial_detected_count, partial_positive_count),
    )


def _aggregate_point(
    *,
    threshold: float,
    hold_seconds: float,
    camera_points: tuple[HouseholdModelTemporalCameraPoint, ...],
) -> HouseholdModelTemporalDiagnosticPoint:
    totals = {
        field: sum(getattr(point, field) for point in camera_points)
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
    negative_count = totals["true_negative"] + totals["false_positive"]
    camera_rates = tuple(
        point.false_positive_rate
        for point in camera_points
        if point.false_positive_rate is not None
    )
    return HouseholdModelTemporalDiagnosticPoint(
        confidence_threshold=threshold,
        hold_seconds=hold_seconds,
        true_positive=totals["true_positive"],
        false_positive=totals["false_positive"],
        false_negative=totals["false_negative"],
        true_negative=totals["true_negative"],
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        specificity=_ratio(totals["true_negative"], negative_count),
        false_positive_rate=_ratio(totals["false_positive"], negative_count),
        clear_positive_count=totals["clear_positive_count"],
        clear_detected_count=totals["clear_detected_count"],
        clear_visibility_recall=_ratio(
            totals["clear_detected_count"],
            totals["clear_positive_count"],
        ),
        partial_positive_count=totals["partial_positive_count"],
        partial_detected_count=totals["partial_detected_count"],
        partial_visibility_recall=_ratio(
            totals["partial_detected_count"],
            totals["partial_positive_count"],
        ),
        maximum_negative_camera_false_positive_rate=(max(camera_rates) if camera_rates else None),
        camera_points=camera_points,
    )


def evaluate_household_model_temporal_diagnostic(
    review: HouseholdModelPresenceReview,
    *,
    media_admissions: tuple[ExternalSourceAdmission, ...],
    detector_scans: tuple[AdmittedDetectorScan, ...],
    thresholds: tuple[float, ...],
    hold_seconds_grid: tuple[float, ...],
    bin_seconds: float,
) -> HouseholdModelTemporalDiagnosticReport:
    """Evaluate causal detection holds at bin ends, stratified by camera/visibility."""

    if not isinstance(review, HouseholdModelPresenceReview):
        raise HouseholdModelDiagnosticError("household temporal diagnostic review is invalid")
    _validate_admissions(review, media_admissions)
    scans_by_source = _validate_scans(review, detector_scans)
    bin_seconds = _number(
        bin_seconds,
        "temporal diagnostic bin_seconds",
        minimum=0.1,
        maximum=60.0,
    )
    first_scan = detector_scans[0]
    thresholds = _thresholds(
        thresholds,
        source_threshold=first_scan.confidence_threshold,
    )
    hold_seconds_grid = _hold_grid(hold_seconds_grid)
    if len(thresholds) * len(hold_seconds_grid) > MAX_HOUSEHOLD_MODEL_TEMPORAL_POINTS:
        raise HouseholdModelDiagnosticError(
            "household temporal diagnostic grid contains too many points"
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

    samples: list[_TemporalSample] = []
    unknown_bin_count = 0
    for (camera_id, admission_sha256, bin_index), items in sorted(grouped_items.items()):
        labels = {item.visible_person for item in items}
        if "present" in labels and "absent" in labels:
            raise HouseholdModelDiagnosticError(
                "household model presence review conflicts within one temporal bin"
            )
        if "unknown" in labels:
            unknown_bin_count += 1
            continue
        label = labels.pop()
        samples.append(
            _TemporalSample(
                camera_id=camera_id,
                media_admission_sha256=admission_sha256,
                bin_index=bin_index,
                label=label,
                partial=(
                    label == "present" and any(item.visibility == "partial" for item in items)
                ),
            )
        )
    if not samples:
        raise HouseholdModelDiagnosticError(
            "household temporal diagnostic contains no evaluable bins"
        )

    detections_by_source = {
        key: tuple(scan.result.detections) for key, scan in scans_by_source.items()
    }
    scores: dict[tuple[str, str, int, float], float | None] = {}
    for sample in samples:
        source_detections = detections_by_source[(sample.camera_id, sample.media_admission_sha256)]
        bin_start = sample.bin_index * bin_seconds
        bin_end = (sample.bin_index + 1) * bin_seconds
        for hold_seconds in hold_seconds_grid:
            window_start = max(0.0, bin_start - hold_seconds)
            eligible_scores = tuple(
                detection.confidence
                for detection in source_detections
                if window_start <= detection.timestamp_seconds < bin_end
            )
            scores[
                (
                    sample.camera_id,
                    sample.media_admission_sha256,
                    sample.bin_index,
                    hold_seconds,
                )
            ] = max(eligible_scores) if eligible_scores else None

    ordered_sources = tuple(sorted(review.sources, key=lambda source: source.camera_id))
    camera_ids = tuple(source.camera_id for source in ordered_sources)
    points: list[HouseholdModelTemporalDiagnosticPoint] = []
    for hold_seconds in hold_seconds_grid:
        scored_by_camera = {
            camera_id: tuple(
                (
                    sample,
                    scores[
                        (
                            sample.camera_id,
                            sample.media_admission_sha256,
                            sample.bin_index,
                            hold_seconds,
                        )
                    ],
                )
                for sample in samples
                if sample.camera_id == camera_id
            )
            for camera_id in camera_ids
        }
        for threshold in thresholds:
            camera_points = tuple(
                _camera_point(camera_id, scored_by_camera[camera_id], threshold)
                for camera_id in camera_ids
            )
            points.append(
                _aggregate_point(
                    threshold=threshold,
                    hold_seconds=hold_seconds,
                    camera_points=camera_points,
                )
            )

    positive_bin_count = sum(sample.label == "present" for sample in samples)
    partial_positive_bin_count = sum(sample.partial for sample in samples)
    ordered_scans = tuple(
        scans_by_source[(source.camera_id, source.media_admission_sha256)]
        for source in ordered_sources
    )
    return HouseholdModelTemporalDiagnosticReport(
        review_sha256=review.fingerprint,
        media_admission_sha256s=tuple(source.media_admission_sha256 for source in ordered_sources),
        detector_scan_sha256s=tuple(scan.fingerprint for scan in ordered_scans),
        camera_ids=camera_ids,
        detector_adapter=first_scan.result.detector_adapter,
        model_id=first_scan.result.model_id,
        model_revision=first_scan.result.model_revision,
        model_sha256=first_scan.result.model_sha256,
        source_confidence_threshold=first_scan.confidence_threshold,
        requested_sample_fps=first_scan.requested_sample_fps,
        bin_seconds=bin_seconds,
        sampling_strategy=review.sampling_strategy,
        reviewed_frame_count=len(review.items),
        evaluated_bin_count=len(samples),
        positive_bin_count=positive_bin_count,
        negative_bin_count=len(samples) - positive_bin_count,
        unknown_bin_count=unknown_bin_count,
        clear_positive_bin_count=positive_bin_count - partial_positive_bin_count,
        partial_positive_bin_count=partial_positive_bin_count,
        thresholds=thresholds,
        hold_seconds_grid=hold_seconds_grid,
        points=tuple(points),
    )


def _select_household_model_temporal_points(
    points: tuple[HouseholdModelTemporalDiagnosticPoint, ...],
    *,
    clear_positive_bin_count: int,
    partial_positive_bin_count: int,
    policy: HouseholdModelTemporalDiagnosticPolicy,
) -> tuple[HouseholdModelTemporalDiagnosticPoint | None, int]:
    if (
        not isinstance(points, tuple)
        or not points
        or any(not isinstance(point, HouseholdModelTemporalDiagnosticPoint) for point in points)
        or not isinstance(policy, HouseholdModelTemporalDiagnosticPolicy)
    ):
        raise HouseholdModelDiagnosticError(
            "household temporal diagnostic selection input is invalid"
        )
    if (policy.require_clear_visibility_samples and clear_positive_bin_count == 0) or (
        policy.require_partial_visibility_samples and partial_positive_bin_count == 0
    ):
        return None, 0
    has_negative_camera_samples = any(
        camera.negative_bin_count > 0 for camera in points[0].camera_points
    )
    if policy.require_negative_camera_samples and not has_negative_camera_samples:
        return None, 0
    if policy.require_mixed_labels_per_camera and any(
        camera.positive_bin_count == 0 or camera.negative_bin_count == 0
        for camera in points[0].camera_points
    ):
        return None, 0

    feasible = tuple(
        point
        for point in points
        if point.hold_seconds <= policy.maximum_hold_seconds
        and point.precision is not None
        and point.recall is not None
        and point.precision >= policy.minimum_precision
        and point.recall >= policy.minimum_recall
        and all(
            camera.recall is not None and camera.recall >= policy.minimum_positive_camera_recall
            for camera in point.camera_points
            if camera.positive_bin_count > 0
        )
        and (
            point.clear_positive_count == 0
            or (
                point.clear_visibility_recall is not None
                and point.clear_visibility_recall >= policy.minimum_clear_visibility_recall
            )
        )
        and (
            point.partial_positive_count == 0
            or (
                point.partial_visibility_recall is not None
                and point.partial_visibility_recall >= policy.minimum_partial_visibility_recall
            )
        )
        and (
            point.maximum_negative_camera_false_positive_rate is None
            or point.maximum_negative_camera_false_positive_rate
            <= policy.maximum_negative_camera_false_positive_rate
        )
    )
    if not feasible:
        return None, 0
    selected = max(
        feasible,
        key=lambda point: (
            min(
                camera.recall
                for camera in point.camera_points
                if camera.recall is not None and camera.positive_bin_count > 0
            ),
            point.recall if point.recall is not None else -1.0,
            (
                point.partial_visibility_recall
                if point.partial_visibility_recall is not None
                else -1.0
            ),
            (point.clear_visibility_recall if point.clear_visibility_recall is not None else -1.0),
            point.precision if point.precision is not None else -1.0,
            point.f1 if point.f1 is not None else -1.0,
            -(
                point.maximum_negative_camera_false_positive_rate
                if point.maximum_negative_camera_false_positive_rate is not None
                else 1.0
            ),
            -point.hold_seconds,
            point.confidence_threshold,
        ),
    )
    return selected, len(feasible)


def select_household_model_temporal_working_point(
    report: HouseholdModelTemporalDiagnosticReport,
    policy: HouseholdModelTemporalDiagnosticPolicy,
) -> tuple[HouseholdModelTemporalDiagnosticPoint | None, int]:
    """Select only a next shadow diagnostic candidate under explicit constraints."""

    if not isinstance(report, HouseholdModelTemporalDiagnosticReport):
        raise HouseholdModelDiagnosticError(
            "household temporal diagnostic selection input is invalid"
        )
    return _select_household_model_temporal_points(
        report.points,
        clear_positive_bin_count=report.clear_positive_bin_count,
        partial_positive_bin_count=report.partial_positive_bin_count,
        policy=policy,
    )
