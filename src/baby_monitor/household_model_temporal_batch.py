"""Cross-batch aggregation for causal household detector diagnostics.

The contract combines only already-saved temporal diagnostic reports.  Source
media and detector models are never opened.  Input reports must describe one
detector configuration and one threshold/hold/camera grid while binding
distinct reviews, media admissions, and detector scans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .household_model_diagnostic import (
    HOUSEHOLD_MODEL_REVIEW_SAMPLING_STRATEGIES,
    MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES,
    HouseholdModelDiagnosticError,
    _canonical_sha256,
    _count,
    _number,
    _ratio,
    _safe_id,
    _sha256,
    _strict_json,
    _thresholds,
    _write_exclusive,
)
from .household_model_temporal_diagnostic import (
    MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLDS,
    MAX_HOUSEHOLD_MODEL_TEMPORAL_POINTS,
    HouseholdModelTemporalCameraPoint,
    HouseholdModelTemporalDiagnosticPoint,
    HouseholdModelTemporalDiagnosticPolicy,
    HouseholdModelTemporalDiagnosticReport,
    _aggregate_point,
    _f1,
    _hold_grid,
    _select_household_model_temporal_points,
)

MAX_HOUSEHOLD_MODEL_TEMPORAL_BATCH_BYTES = 4 * 1024 * 1024
MAX_HOUSEHOLD_MODEL_TEMPORAL_BATCHES = 16


@dataclass(frozen=True, slots=True)
class HouseholdModelTemporalBatchBinding:
    report_sha256: str
    review_sha256: str
    media_admission_sha256s: tuple[str, ...]
    detector_scan_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256(self.report_sha256, "temporal batch binding report_sha256")
        _sha256(self.review_sha256, "temporal batch binding review_sha256")
        for field in ("media_admission_sha256s", "detector_scan_sha256s"):
            values = getattr(self, field)
            if (
                not isinstance(values, tuple)
                or not 1 <= len(values) <= MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES
                or len(set(values)) != len(values)
            ):
                raise HouseholdModelDiagnosticError(
                    f"household temporal batch binding {field} is invalid"
                )
            for value in values:
                _sha256(value, f"temporal batch binding {field}")

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelTemporalBatchBinding:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household temporal batch binding fields do not match v1"
            )
        for field in ("media_admission_sha256s", "detector_scan_sha256s"):
            if not isinstance(value[field], list):
                raise HouseholdModelDiagnosticError(
                    "household temporal batch binding arrays are invalid"
                )
        return cls(
            **{
                key: tuple(item) if key.endswith("_sha256s") else item
                for key, item in value.items()
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            field: (
                list(getattr(self, field)) if field.endswith("_sha256s") else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class HouseholdModelTemporalBatchDiagnosticReport:
    batches: tuple[HouseholdModelTemporalBatchBinding, ...]
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
    mixed_label_camera_count: int
    thresholds: tuple[float, ...]
    hold_seconds_grid: tuple[float, ...]
    points: tuple[HouseholdModelTemporalDiagnosticPoint, ...]
    version: int = 1
    label: str = "person"
    ground_truth_provenance: str = "model_generated_visual_review"
    aggregation_strategy: str = "sum_disjoint_temporal_diagnostic_batches"
    temporal_aggregation_strategy: str = "causal_detection_hold"
    prediction_anchor: str = "bin_end"
    hold_window_semantics: str = "current_bin_plus_prior_seconds"
    stratification_axes: tuple[str, ...] = ("camera_id", "visibility")
    cross_batch_aggregation: bool = True
    input_reports_distinct: bool = True
    media_sets_disjoint: bool = True
    detector_scan_sets_disjoint: bool = True
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
                "unsupported household temporal batch diagnostic report version"
            )
        if (
            not isinstance(self.batches, tuple)
            or not 2 <= len(self.batches) <= MAX_HOUSEHOLD_MODEL_TEMPORAL_BATCHES
            or any(
                not isinstance(item, HouseholdModelTemporalBatchBinding) for item in self.batches
            )
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic bindings are invalid"
            )
        report_hashes = tuple(item.report_sha256 for item in self.batches)
        review_hashes = tuple(item.review_sha256 for item in self.batches)
        if (
            report_hashes != tuple(sorted(report_hashes))
            or len(set(report_hashes)) != len(report_hashes)
            or len(set(review_hashes)) != len(review_hashes)
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic reports and reviews must be distinct"
            )
        if (
            not isinstance(self.camera_ids, tuple)
            or not 1 <= len(self.camera_ids) <= MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES
            or self.camera_ids != tuple(sorted(self.camera_ids))
            or len(set(self.camera_ids)) != len(self.camera_ids)
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic camera IDs are invalid"
            )
        for camera_id in self.camera_ids:
            _safe_id(camera_id, "temporal batch diagnostic camera_id")
        for binding in self.batches:
            if len(binding.media_admission_sha256s) != len(self.camera_ids) or len(
                binding.detector_scan_sha256s
            ) != len(self.camera_ids):
                raise HouseholdModelDiagnosticError(
                    "household temporal batch diagnostic binding source count is inconsistent"
                )
        media_hashes = tuple(
            value for binding in self.batches for value in binding.media_admission_sha256s
        )
        scan_hashes = tuple(
            value for binding in self.batches for value in binding.detector_scan_sha256s
        )
        if len(set(media_hashes)) != len(media_hashes) or len(set(scan_hashes)) != len(scan_hashes):
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic media and scans must be disjoint"
            )
        for field in ("detector_adapter", "model_id", "model_revision"):
            _safe_id(getattr(self, field), f"temporal batch diagnostic {field}")
        _sha256(self.model_sha256, "temporal batch diagnostic model_sha256")
        object.__setattr__(
            self,
            "source_confidence_threshold",
            _number(
                self.source_confidence_threshold,
                "temporal batch diagnostic source_confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "requested_sample_fps",
            _number(
                self.requested_sample_fps,
                "temporal batch diagnostic requested_sample_fps",
                minimum=0.01,
                maximum=10.0,
            ),
        )
        object.__setattr__(
            self,
            "bin_seconds",
            _number(
                self.bin_seconds,
                "temporal batch diagnostic bin_seconds",
                minimum=0.1,
                maximum=60.0,
            ),
        )
        if self.sampling_strategy not in HOUSEHOLD_MODEL_REVIEW_SAMPLING_STRATEGIES:
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic sampling strategy is invalid"
            )
        for field in (
            "reviewed_frame_count",
            "evaluated_bin_count",
            "positive_bin_count",
            "negative_bin_count",
            "unknown_bin_count",
            "clear_positive_bin_count",
            "partial_positive_bin_count",
            "mixed_label_camera_count",
        ):
            _count(getattr(self, field), f"temporal batch diagnostic {field}")
        if (
            self.reviewed_frame_count < len(self.batches)
            or self.evaluated_bin_count < 1
            or self.positive_bin_count + self.negative_bin_count != self.evaluated_bin_count
            or self.clear_positive_bin_count + self.partial_positive_bin_count
            != self.positive_bin_count
            or self.evaluated_bin_count + self.unknown_bin_count > self.reviewed_frame_count
            or self.mixed_label_camera_count > len(self.camera_ids)
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic count accounting is inconsistent"
            )
        if not isinstance(self.thresholds, tuple) or self.thresholds != _thresholds(
            self.thresholds,
            source_threshold=self.source_confidence_threshold,
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic thresholds are invalid"
            )
        if (
            not isinstance(self.hold_seconds_grid, tuple)
            or not 2 <= len(self.hold_seconds_grid) <= MAX_HOUSEHOLD_MODEL_TEMPORAL_HOLDS
            or self.hold_seconds_grid != _hold_grid(self.hold_seconds_grid)
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic hold grid is invalid"
            )
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
                "household temporal batch diagnostic points do not match the canonical grid"
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
                    "household temporal batch diagnostic point does not match report strata"
                )
        first_cameras = self.points[0].camera_points
        expected_mixed = sum(
            camera.positive_bin_count > 0 and camera.negative_bin_count > 0
            for camera in first_cameras
        )
        if self.mixed_label_camera_count != expected_mixed:
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic mixed camera count is inconsistent"
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
                        "household temporal batch threshold counts are not monotonic"
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
                        "household temporal batch hold counts are not monotonic"
                    )
                previous = current
        if (
            self.label != "person"
            or self.ground_truth_provenance != "model_generated_visual_review"
            or self.aggregation_strategy != "sum_disjoint_temporal_diagnostic_batches"
            or self.temporal_aggregation_strategy != "causal_detection_hold"
            or self.prediction_anchor != "bin_end"
            or self.hold_window_semantics != "current_bin_plus_prior_seconds"
            or self.stratification_axes != ("camera_id", "visibility")
        ):
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic semantics are invalid"
            )
        for field, expected in (
            ("cross_batch_aggregation", True),
            ("input_reports_distinct", True),
            ("media_sets_disjoint", True),
            ("detector_scan_sets_disjoint", True),
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
                    f"household temporal batch diagnostic {field} must be {str(expected).lower()}"
                )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        sequence_fields = {
            "camera_ids",
            "thresholds",
            "hold_seconds_grid",
            "stratification_axes",
        }
        value: dict[str, object] = {
            field: (
                [item.to_dict() for item in self.batches]
                if field == "batches"
                else [point.to_dict() for point in self.points]
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
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "valid": True,
            "version": self.version,
            "report_sha256": self.fingerprint,
            "input_report_sha256s": [item.report_sha256 for item in self.batches],
            "batch_count": len(self.batches),
            "source_count_per_batch": len(self.camera_ids),
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
            "mixed_label_camera_count": self.mixed_label_camera_count,
            "thresholds": list(self.thresholds),
            "hold_seconds_grid": list(self.hold_seconds_grid),
            "points": [point.to_dict() for point in self.points],
            "aggregation_strategy": self.aggregation_strategy,
            "cross_batch_aggregation": True,
            "input_reports_distinct": True,
            "media_sets_disjoint": True,
            "detector_scan_sets_disjoint": True,
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
    def from_dict(cls, value: Any) -> HouseholdModelTemporalBatchDiagnosticReport:
        fields = {*cls.__dataclass_fields__, "report_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic report fields do not match v1"
            )
        sequence_fields = {
            "batches",
            "camera_ids",
            "thresholds",
            "hold_seconds_grid",
            "stratification_axes",
            "points",
        }
        if any(not isinstance(value[field], list) for field in sequence_fields):
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic report arrays are invalid"
            )
        report_sha256 = value["report_sha256"]
        report = cls(
            **{
                key: (
                    tuple(HouseholdModelTemporalBatchBinding.from_dict(item) for item in item_value)
                    if key == "batches"
                    else tuple(
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
                "household temporal batch diagnostic report fingerprint is invalid"
            )
        return report

    @classmethod
    def from_json(cls, raw: str) -> HouseholdModelTemporalBatchDiagnosticReport:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_MODEL_TEMPORAL_BATCH_BYTES:
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic report exceeds the 4 MiB limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdModelTemporalBatchDiagnosticReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_HOUSEHOLD_MODEL_TEMPORAL_BATCH_BYTES:
                raise HouseholdModelDiagnosticError(
                    "household temporal batch diagnostic report exceeds the size limit"
                )
            raw = source.read_text(encoding="utf-8")
        except HouseholdModelDiagnosticError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostic report is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_MODEL_TEMPORAL_BATCH_BYTES,
        )


def _combine_camera_points(
    camera_id: str,
    points: tuple[HouseholdModelTemporalCameraPoint, ...],
) -> HouseholdModelTemporalCameraPoint:
    if not points or any(
        not isinstance(point, HouseholdModelTemporalCameraPoint) or point.camera_id != camera_id
        for point in points
    ):
        raise HouseholdModelDiagnosticError("household temporal batch camera points are invalid")
    totals = {
        field: sum(getattr(point, field) for point in points)
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
            totals["clear_detected_count"],
            totals["clear_positive_count"],
        ),
        partial_visibility_recall=_ratio(
            totals["partial_detected_count"],
            totals["partial_positive_count"],
        ),
        **totals,
    )


def combine_household_model_temporal_diagnostic_batches(
    reports: tuple[HouseholdModelTemporalDiagnosticReport, ...],
) -> HouseholdModelTemporalBatchDiagnosticReport:
    """Combine disjoint, same-grid temporal diagnostics without new media access."""

    if (
        not isinstance(reports, tuple)
        or not 2 <= len(reports) <= MAX_HOUSEHOLD_MODEL_TEMPORAL_BATCHES
        or any(not isinstance(report, HouseholdModelTemporalDiagnosticReport) for report in reports)
    ):
        raise HouseholdModelDiagnosticError(
            "household temporal batch diagnostic reports are invalid"
        )
    reports = tuple(sorted(reports, key=lambda report: report.fingerprint))
    first = reports[0]
    identity = (
        first.camera_ids,
        first.detector_adapter,
        first.model_id,
        first.model_revision,
        first.model_sha256,
        first.source_confidence_threshold,
        first.requested_sample_fps,
        first.bin_seconds,
        first.sampling_strategy,
        first.thresholds,
        first.hold_seconds_grid,
        first.label,
        first.temporal_aggregation_strategy,
        first.prediction_anchor,
        first.hold_window_semantics,
        first.stratification_axes,
    )
    for report in reports:
        if (
            report.camera_ids,
            report.detector_adapter,
            report.model_id,
            report.model_revision,
            report.model_sha256,
            report.source_confidence_threshold,
            report.requested_sample_fps,
            report.bin_seconds,
            report.sampling_strategy,
            report.thresholds,
            report.hold_seconds_grid,
            report.label,
            report.temporal_aggregation_strategy,
            report.prediction_anchor,
            report.hold_window_semantics,
            report.stratification_axes,
        ) != identity:
            raise HouseholdModelDiagnosticError(
                "household temporal batch diagnostics must share one exact grid and model config"
            )
    if len({report.fingerprint for report in reports}) != len(reports) or len(
        {report.review_sha256 for report in reports}
    ) != len(reports):
        raise HouseholdModelDiagnosticError(
            "household temporal batch diagnostics must bind distinct reports and reviews"
        )
    media_hashes = tuple(value for report in reports for value in report.media_admission_sha256s)
    scan_hashes = tuple(value for report in reports for value in report.detector_scan_sha256s)
    if len(set(media_hashes)) != len(media_hashes) or len(set(scan_hashes)) != len(scan_hashes):
        raise HouseholdModelDiagnosticError(
            "household temporal batch diagnostics require disjoint media and scans"
        )

    points: list[HouseholdModelTemporalDiagnosticPoint] = []
    for point_index, source_point in enumerate(first.points):
        camera_points = tuple(
            _combine_camera_points(
                camera_id,
                tuple(
                    next(
                        camera
                        for camera in report.points[point_index].camera_points
                        if camera.camera_id == camera_id
                    )
                    for report in reports
                ),
            )
            for camera_id in first.camera_ids
        )
        points.append(
            _aggregate_point(
                threshold=source_point.confidence_threshold,
                hold_seconds=source_point.hold_seconds,
                camera_points=camera_points,
            )
        )
    bindings = tuple(
        HouseholdModelTemporalBatchBinding(
            report_sha256=report.fingerprint,
            review_sha256=report.review_sha256,
            media_admission_sha256s=report.media_admission_sha256s,
            detector_scan_sha256s=report.detector_scan_sha256s,
        )
        for report in reports
    )
    first_point = points[0]
    return HouseholdModelTemporalBatchDiagnosticReport(
        batches=bindings,
        camera_ids=first.camera_ids,
        detector_adapter=first.detector_adapter,
        model_id=first.model_id,
        model_revision=first.model_revision,
        model_sha256=first.model_sha256,
        source_confidence_threshold=first.source_confidence_threshold,
        requested_sample_fps=first.requested_sample_fps,
        bin_seconds=first.bin_seconds,
        sampling_strategy=first.sampling_strategy,
        reviewed_frame_count=sum(report.reviewed_frame_count for report in reports),
        evaluated_bin_count=sum(report.evaluated_bin_count for report in reports),
        positive_bin_count=sum(report.positive_bin_count for report in reports),
        negative_bin_count=sum(report.negative_bin_count for report in reports),
        unknown_bin_count=sum(report.unknown_bin_count for report in reports),
        clear_positive_bin_count=sum(report.clear_positive_bin_count for report in reports),
        partial_positive_bin_count=sum(report.partial_positive_bin_count for report in reports),
        mixed_label_camera_count=sum(
            camera.positive_bin_count > 0 and camera.negative_bin_count > 0
            for camera in first_point.camera_points
        ),
        thresholds=first.thresholds,
        hold_seconds_grid=first.hold_seconds_grid,
        points=tuple(points),
    )


def select_household_model_temporal_batch_working_point(
    report: HouseholdModelTemporalBatchDiagnosticReport,
    policy: HouseholdModelTemporalDiagnosticPolicy,
) -> tuple[HouseholdModelTemporalDiagnosticPoint | None, int]:
    if not isinstance(report, HouseholdModelTemporalBatchDiagnosticReport):
        raise HouseholdModelDiagnosticError("household temporal batch selection input is invalid")
    return _select_household_model_temporal_points(
        report.points,
        clear_positive_bin_count=report.clear_positive_bin_count,
        partial_positive_bin_count=report.partial_positive_bin_count,
        policy=policy,
    )
