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

from .detector_evaluation import (
    DetectorEvaluationError,
    evaluate_detector_box_quality,
)
from .detector_protocol import AdmittedDetectorScan
from .object_tracker import DetectionFrameBatch
from .source_admission import ExternalSourceAdmission
from .tracker_evaluation import TrackerAssociationReview

MAX_DETECTOR_THRESHOLD_SWEEP_BYTES = 1024 * 1024
MAX_THRESHOLD_POINTS = 32
DETECTOR_OPERATING_POINT_OBJECTIVES = ("recall_first", "balanced_f1")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class DetectorThresholdSweepError(ValueError):
    """A detector threshold sweep is inconsistent or unsafe."""


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DetectorThresholdSweepError(
                    "detector threshold sweep JSON contains duplicate keys"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                DetectorThresholdSweepError(
                    "detector threshold sweep JSON contains a non-standard number"
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise DetectorThresholdSweepError("detector threshold sweep JSON is invalid") from exc


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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
        raise DetectorThresholdSweepError(f"detector threshold sweep {field} is invalid")
    return float(value)


def _optional_ratio(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field, minimum=0.0, maximum=1.0)


def _count(value: Any, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 1_000_000:
        raise DetectorThresholdSweepError(f"detector threshold sweep {field} is invalid")
    return value


def _positive_count(value: Any, field: str) -> int:
    value = _count(value, field)
    if value == 0:
        raise DetectorThresholdSweepError(f"detector threshold sweep {field} is invalid")
    return value


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise DetectorThresholdSweepError(f"detector threshold sweep {field} is invalid")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DetectorThresholdSweepError(f"detector threshold sweep {field} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class DetectorThresholdSweepPoint:
    confidence_threshold: float
    detector_box_count: int
    matched_box_count: int
    missed_ground_truth_box_count: int
    false_positive_box_count: int
    precision: float | None
    recall: float | None
    f1: float | None
    mean_matched_iou: float | None
    longest_consecutive_subject_miss_samples: int
    longest_consecutive_subject_miss_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence_threshold",
            _number(
                self.confidence_threshold,
                "point confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        for field in (
            "detector_box_count",
            "matched_box_count",
            "missed_ground_truth_box_count",
            "false_positive_box_count",
            "longest_consecutive_subject_miss_samples",
        ):
            _count(getattr(self, field), f"point {field}")
        for field in ("precision", "recall", "f1", "mean_matched_iou"):
            object.__setattr__(
                self,
                field,
                _optional_ratio(getattr(self, field), f"point {field}"),
            )
        object.__setattr__(
            self,
            "longest_consecutive_subject_miss_seconds",
            _number(
                self.longest_consecutive_subject_miss_seconds,
                "point longest_consecutive_subject_miss_seconds",
                minimum=0.0,
                maximum=86_400.0,
            ),
        )
        if self.matched_box_count + self.false_positive_box_count != (self.detector_box_count):
            raise DetectorThresholdSweepError(
                "detector threshold sweep point detector accounting is inconsistent"
            )
        expected_precision = (
            None
            if self.detector_box_count == 0
            else round(self.matched_box_count / self.detector_box_count, 12)
        )
        if self.precision != expected_precision:
            raise DetectorThresholdSweepError(
                "detector threshold sweep point precision is inconsistent"
            )

    @classmethod
    def from_dict(cls, value: Any) -> DetectorThresholdSweepPoint:
        fields = {
            "confidence_threshold",
            "detector_box_count",
            "matched_box_count",
            "missed_ground_truth_box_count",
            "false_positive_box_count",
            "precision",
            "recall",
            "f1",
            "mean_matched_iou",
            "longest_consecutive_subject_miss_samples",
            "longest_consecutive_subject_miss_seconds",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise DetectorThresholdSweepError(
                "detector threshold sweep point fields do not match v1"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class DetectorOperatingPointPolicy:
    objective: str
    minimum_precision: float
    minimum_recall: float
    maximum_longest_miss_seconds: float

    def __post_init__(self) -> None:
        if self.objective not in DETECTOR_OPERATING_POINT_OBJECTIVES:
            raise DetectorThresholdSweepError("detector operating point objective is invalid")
        for field in ("minimum_precision", "minimum_recall"):
            object.__setattr__(
                self,
                field,
                _number(
                    getattr(self, field),
                    f"operating point {field}",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        object.__setattr__(
            self,
            "maximum_longest_miss_seconds",
            _number(
                self.maximum_longest_miss_seconds,
                "operating point maximum_longest_miss_seconds",
                minimum=0.0,
                maximum=86_400.0,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class DetectorThresholdSweepReport:
    ground_truth_batch_sha256: str
    ground_truth_review_sha256: str
    ground_truth_admission_sha256: str
    media_admission_sha256: str
    detector_scan_sha256: str
    camera_id: str
    detector_adapter: str
    model_id: str
    model_revision: str
    model_sha256: str
    label: str
    source_confidence_threshold: float
    requested_sample_fps: float
    source_fps: float
    effective_sample_fps: float
    source_frame_count: int
    annotation_frame_count: int
    annotation_frame_offset: int
    evaluated_frame_count: int
    ground_truth_box_count: int
    iou_threshold: float
    points: tuple[DetectorThresholdSweepPoint, ...]
    best_f1_threshold: float | None
    best_f1: float | None
    version: int = 1
    selection_rule: str = "max_f1_then_recall_then_precision_then_lower_threshold"
    ground_truth_provenance: str = "admitted_publisher_human"
    frame_alignment_confirmed: bool = True
    derived_without_model_rerun: bool = True
    detection_accuracy_evaluated: bool = True
    subject_identity_inference: bool = False
    age_role_inference: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise DetectorThresholdSweepError("unsupported detector threshold sweep version")
        for field in (
            "ground_truth_batch_sha256",
            "ground_truth_review_sha256",
            "ground_truth_admission_sha256",
            "media_admission_sha256",
            "detector_scan_sha256",
            "model_sha256",
        ):
            _sha256(getattr(self, field), field)
        for field in (
            "camera_id",
            "detector_adapter",
            "model_id",
            "model_revision",
        ):
            _safe_id(getattr(self, field), field)
        if self.label != "person":
            raise DetectorThresholdSweepError(
                "detector threshold sweep label must remain generic person"
            )
        for field, minimum, maximum in (
            ("source_confidence_threshold", 0.0, 1.0),
            ("requested_sample_fps", 0.01, 10.0),
            ("source_fps", 0.01, 240.0),
            ("effective_sample_fps", 0.01, 240.0),
            ("iou_threshold", 0.01, 1.0),
        ):
            object.__setattr__(
                self,
                field,
                _number(
                    getattr(self, field),
                    field,
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        for field in (
            "source_frame_count",
            "annotation_frame_count",
            "evaluated_frame_count",
            "ground_truth_box_count",
        ):
            _positive_count(getattr(self, field), field)
        _count(self.annotation_frame_offset, "annotation_frame_offset")
        if (
            not isinstance(self.points, tuple)
            or not 2 <= len(self.points) <= MAX_THRESHOLD_POINTS
            or any(not isinstance(point, DetectorThresholdSweepPoint) for point in self.points)
        ):
            raise DetectorThresholdSweepError("detector threshold sweep points are invalid")
        thresholds = tuple(point.confidence_threshold for point in self.points)
        if (
            thresholds != tuple(sorted(thresholds))
            or len(set(thresholds)) != len(thresholds)
            or thresholds[0] != self.source_confidence_threshold
        ):
            raise DetectorThresholdSweepError(
                "detector threshold sweep thresholds must be unique, ascending, "
                "and start at the source threshold"
            )
        previous_detector_count: int | None = None
        previous_matched_count: int | None = None
        for point in self.points:
            if point.matched_box_count + point.missed_ground_truth_box_count != (
                self.ground_truth_box_count
            ):
                raise DetectorThresholdSweepError(
                    "detector threshold sweep point ground-truth accounting is inconsistent"
                )
            expected_recall = round(
                point.matched_box_count / self.ground_truth_box_count,
                12,
            )
            expected_f1 = (
                None
                if point.precision is None or point.precision + expected_recall == 0.0
                else round(
                    2.0 * point.precision * expected_recall / (point.precision + expected_recall),
                    12,
                )
            )
            if point.recall != expected_recall or point.f1 != expected_f1:
                raise DetectorThresholdSweepError(
                    "detector threshold sweep point recall/F1 is inconsistent"
                )
            if not math.isclose(
                point.longest_consecutive_subject_miss_seconds,
                point.longest_consecutive_subject_miss_samples / self.effective_sample_fps,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise DetectorThresholdSweepError(
                    "detector threshold sweep miss duration is inconsistent"
                )
            if (
                previous_detector_count is not None
                and point.detector_box_count > previous_detector_count
            ) or (
                previous_matched_count is not None
                and point.matched_box_count > previous_matched_count
            ):
                raise DetectorThresholdSweepError(
                    "detector threshold sweep counts are not monotonic"
                )
            previous_detector_count = point.detector_box_count
            previous_matched_count = point.matched_box_count
        eligible = tuple(point for point in self.points if point.f1 is not None)
        selected = (
            None
            if not eligible
            else max(
                eligible,
                key=lambda point: (
                    point.f1,
                    point.recall if point.recall is not None else -1.0,
                    point.precision if point.precision is not None else -1.0,
                    -point.confidence_threshold,
                ),
            )
        )
        expected_best_threshold = None if selected is None else selected.confidence_threshold
        expected_best_f1 = None if selected is None else selected.f1
        if self.best_f1_threshold != expected_best_threshold or self.best_f1 != expected_best_f1:
            raise DetectorThresholdSweepError(
                "detector threshold sweep best F1 selection is inconsistent"
            )
        if self.selection_rule != ("max_f1_then_recall_then_precision_then_lower_threshold"):
            raise DetectorThresholdSweepError("detector threshold sweep selection rule is invalid")
        if self.ground_truth_provenance != "admitted_publisher_human":
            raise DetectorThresholdSweepError(
                "detector threshold sweep ground truth provenance is invalid"
            )
        for field, expected in (
            ("frame_alignment_confirmed", True),
            ("derived_without_model_rerun", True),
            ("detection_accuracy_evaluated", True),
            ("subject_identity_inference", False),
            ("age_role_inference", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise DetectorThresholdSweepError(
                    f"detector threshold sweep {field} must be {str(expected).lower()}"
                )

    @classmethod
    def from_dict(cls, value: Any) -> DetectorThresholdSweepReport:
        fields = {
            *cls.__dataclass_fields__,
            "report_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise DetectorThresholdSweepError(
                "detector threshold sweep report fields do not match v1"
            )
        report_sha256 = value["report_sha256"]
        report = cls(
            **{
                key: (
                    tuple(DetectorThresholdSweepPoint.from_dict(point) for point in item)
                    if key == "points" and isinstance(item, list)
                    else item
                )
                for key, item in value.items()
                if key != "report_sha256"
            }
        )
        if report_sha256 != report.fingerprint:
            raise DetectorThresholdSweepError(
                "detector threshold sweep report fingerprint is invalid"
            )
        return report

    @classmethod
    def from_json(cls, raw: str) -> DetectorThresholdSweepReport:
        if len(raw.encode("utf-8")) > MAX_DETECTOR_THRESHOLD_SWEEP_BYTES:
            raise DetectorThresholdSweepError(
                "detector threshold sweep report exceeds the 1 MiB limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> DetectorThresholdSweepReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if source.stat().st_size > MAX_DETECTOR_THRESHOLD_SWEEP_BYTES:
                raise DetectorThresholdSweepError(
                    "detector threshold sweep report exceeds the 1 MiB limit"
                )
            raw = source.read_text(encoding="utf-8")
        except DetectorThresholdSweepError:
            raise
        except (OSError, UnicodeError) as exc:
            raise DetectorThresholdSweepError(
                "detector threshold sweep report is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: (
                [point.to_dict() for point in self.points]
                if field == "points"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        if include_fingerprint:
            value["report_sha256"] = self.fingerprint
        return value

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    def write_new(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        parent = target.parent
        if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
            raise DetectorThresholdSweepError("detector threshold sweep destination is unsafe")
        payload = f"{self.to_json()}\n".encode()
        if len(payload) > MAX_DETECTOR_THRESHOLD_SWEEP_BYTES:
            raise DetectorThresholdSweepError(
                "detector threshold sweep report exceeds the 1 MiB limit"
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = os.open(target, flags, 0o600)
            if os.name == "posix":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise FileExistsError("detector threshold sweep output already exists") from None
        except OSError as exc:
            with suppress(OSError):
                target.unlink()
            raise DetectorThresholdSweepError(
                "detector threshold sweep output could not be written"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def to_public_dict(self) -> dict[str, object]:
        return self.to_dict()


def select_detector_operating_point(
    report: DetectorThresholdSweepReport,
    policy: DetectorOperatingPointPolicy,
) -> tuple[DetectorThresholdSweepPoint | None, int]:
    if not isinstance(report, DetectorThresholdSweepReport):
        raise DetectorThresholdSweepError(
            "detector operating point requires a threshold sweep report"
        )
    if not isinstance(policy, DetectorOperatingPointPolicy):
        raise DetectorThresholdSweepError("detector operating point policy is invalid")
    eligible = tuple(
        point
        for point in report.points
        if point.precision is not None
        and point.recall is not None
        and point.f1 is not None
        and point.precision >= policy.minimum_precision
        and point.recall >= policy.minimum_recall
        and point.longest_consecutive_subject_miss_seconds <= policy.maximum_longest_miss_seconds
    )
    if not eligible:
        return None, 0
    if policy.objective == "recall_first":

        def selection_key(
            point: DetectorThresholdSweepPoint,
        ) -> tuple[float, float, float, float, float]:
            return (
                point.recall or 0.0,
                point.precision or 0.0,
                point.f1 or 0.0,
                -point.longest_consecutive_subject_miss_seconds,
                -point.confidence_threshold,
            )

    else:

        def selection_key(
            point: DetectorThresholdSweepPoint,
        ) -> tuple[float, float, float, float, float]:
            return (
                point.f1 or 0.0,
                point.recall or 0.0,
                point.precision or 0.0,
                -point.longest_consecutive_subject_miss_seconds,
                -point.confidence_threshold,
            )

    return max(eligible, key=selection_key), len(eligible)


def evaluate_detector_threshold_sweep(
    ground_truth_batch: DetectionFrameBatch,
    ground_truth_review: TrackerAssociationReview,
    detector_scan: AdmittedDetectorScan,
    *,
    ground_truth_admission: ExternalSourceAdmission,
    media_admission: ExternalSourceAdmission,
    source_fps: float,
    source_frame_count: int,
    annotation_frame_offset: int,
    thresholds: tuple[float, ...],
    iou_threshold: float = 0.5,
    frame_alignment_confirmed: bool = False,
) -> DetectorThresholdSweepReport:
    parsed_thresholds = tuple(
        _number(value, "threshold", minimum=0.0, maximum=1.0) for value in thresholds
    )
    if (
        not 2 <= len(parsed_thresholds) <= MAX_THRESHOLD_POINTS
        or parsed_thresholds != tuple(sorted(parsed_thresholds))
        or len(set(parsed_thresholds)) != len(parsed_thresholds)
        or parsed_thresholds[0] != detector_scan.confidence_threshold
    ):
        raise DetectorThresholdSweepError(
            "thresholds must be 2-32 unique ascending values starting at "
            "the admitted scan threshold"
        )
    reports = []
    for threshold in parsed_thresholds:
        filtered_result = detector_scan.result.at_confidence_threshold(threshold)
        filtered_scan = AdmittedDetectorScan(
            source_admission_sha256=detector_scan.source_admission_sha256,
            labels=detector_scan.labels,
            confidence_threshold=threshold,
            requested_sample_fps=detector_scan.requested_sample_fps,
            result=filtered_result,
            version=detector_scan.version,
        )
        try:
            reports.append(
                evaluate_detector_box_quality(
                    ground_truth_batch,
                    ground_truth_review,
                    filtered_scan,
                    ground_truth_admission=ground_truth_admission,
                    media_admission=media_admission,
                    source_fps=source_fps,
                    source_frame_count=source_frame_count,
                    annotation_frame_offset=annotation_frame_offset,
                    iou_threshold=iou_threshold,
                    frame_alignment_confirmed=frame_alignment_confirmed,
                )
            )
        except DetectorEvaluationError as exc:
            raise DetectorThresholdSweepError(str(exc)) from exc
    points = tuple(
        DetectorThresholdSweepPoint(
            confidence_threshold=report.confidence_threshold,
            detector_box_count=report.detector_box_count,
            matched_box_count=report.matched_box_count,
            missed_ground_truth_box_count=report.missed_ground_truth_box_count,
            false_positive_box_count=report.false_positive_box_count,
            precision=report.precision,
            recall=report.recall,
            f1=report.f1,
            mean_matched_iou=report.mean_matched_iou,
            longest_consecutive_subject_miss_samples=(
                report.longest_consecutive_subject_miss_samples
            ),
            longest_consecutive_subject_miss_seconds=(
                report.longest_consecutive_subject_miss_seconds
            ),
        )
        for report in reports
    )
    eligible = tuple(point for point in points if point.f1 is not None)
    selected = (
        None
        if not eligible
        else max(
            eligible,
            key=lambda point: (
                point.f1,
                point.recall if point.recall is not None else -1.0,
                point.precision if point.precision is not None else -1.0,
                -point.confidence_threshold,
            ),
        )
    )
    first = reports[0]
    return DetectorThresholdSweepReport(
        ground_truth_batch_sha256=ground_truth_batch.fingerprint,
        ground_truth_review_sha256=ground_truth_review.fingerprint,
        ground_truth_admission_sha256=ground_truth_admission.fingerprint,
        media_admission_sha256=media_admission.fingerprint,
        detector_scan_sha256=detector_scan.fingerprint,
        camera_id=first.camera_id,
        detector_adapter=first.detector_adapter,
        model_id=first.model_id,
        model_revision=first.model_revision,
        model_sha256=first.model_sha256,
        label=first.label,
        source_confidence_threshold=detector_scan.confidence_threshold,
        requested_sample_fps=first.requested_sample_fps,
        source_fps=first.source_fps,
        effective_sample_fps=first.effective_sample_fps,
        source_frame_count=first.source_frame_count,
        annotation_frame_count=first.annotation_frame_count,
        annotation_frame_offset=first.annotation_frame_offset,
        evaluated_frame_count=first.evaluated_frame_count,
        ground_truth_box_count=first.ground_truth_box_count,
        iou_threshold=first.iou_threshold,
        points=points,
        best_f1_threshold=(None if selected is None else selected.confidence_threshold),
        best_f1=None if selected is None else selected.f1,
    )
