from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .detector_protocol import AdmittedDetectorScan
from .detector_sampling import DetectorSamplingError, validate_nominal_sampling_grid
from .motion_tracker import _maximum_weight_assignment
from .object_tracker import (
    MAX_DETECTIONS_PER_FRAME,
    MAX_TRACKING_DETECTIONS,
    MAX_TRACKING_FRAMES,
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
    ObjectTrackerError,
)
from .source_admission import ExternalSourceAdmission
from .tracker_evaluation import (
    TrackerAssociationReview,
    TrackerReviewAssignment,
)

MAX_DETECTOR_QUALITY_REPORT_BYTES = 1024 * 1024
MAX_SOURCE_FRAMES = 1_000_000

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class DetectorEvaluationError(ValueError):
    """Detector evaluation inputs are inconsistent or outside the safe profile."""


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DetectorEvaluationError("detector quality JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                DetectorEvaluationError("detector quality JSON contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise DetectorEvaluationError("detector quality JSON is invalid") from exc


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise DetectorEvaluationError(f"detector quality {field} is invalid")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DetectorEvaluationError(f"detector quality {field} is invalid")
    return value


def _count(value: Any, field: str, *, maximum: int = MAX_SOURCE_FRAMES) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise DetectorEvaluationError(f"detector quality {field} is invalid")
    return value


def _positive_count(value: Any, field: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_SOURCE_FRAMES:
        raise DetectorEvaluationError(f"detector quality {field} is invalid")
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
        raise DetectorEvaluationError(f"detector quality {field} is invalid")
    return float(value)


def _optional_ratio(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field, minimum=0.0, maximum=1.0)


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


def build_detector_tracking_batch(
    detector_scan: AdmittedDetectorScan,
    *,
    media_admission: ExternalSourceAdmission,
    source_fps: float,
    source_frame_count: int,
    timeline_start: datetime,
) -> DetectionFrameBatch:
    """Convert one exact admitted scan into tracker frames, including empty samples.

    ``timeline_start`` is only a local ordering anchor. The source frame index and
    declared frame rate determine every following timestamp; no wall-clock capture
    time is inferred from media metadata.
    """

    if not isinstance(detector_scan, AdmittedDetectorScan):
        raise DetectorEvaluationError("detector scan is invalid")
    if not isinstance(media_admission, ExternalSourceAdmission):
        raise DetectorEvaluationError("media admission is invalid")
    if (
        media_admission.asset_kind != "media_asset"
        or media_admission.admission.status != "verified_local"
    ):
        raise DetectorEvaluationError("detector tracking conversion requires verified local media")
    if detector_scan.source_admission_sha256 != media_admission.fingerprint:
        raise DetectorEvaluationError("detector scan belongs to a different admitted media asset")
    source_fps = _number(source_fps, "source_fps", minimum=0.01, maximum=240.0)
    source_frame_count = _positive_count(source_frame_count, "source_frame_count")
    if detector_scan.result.sampling is not None:
        try:
            validate_nominal_sampling_grid(
                detector_scan.result.sampling,
                source_fps=source_fps,
                source_frame_count=source_frame_count,
            )
        except DetectorSamplingError as error:
            raise DetectorEvaluationError(str(error)) from None
    if (
        not isinstance(timeline_start, datetime)
        or timeline_start.tzinfo is None
        or timeline_start.utcoffset() is None
    ):
        raise DetectorEvaluationError("timeline_start must be timezone-aware")
    timeline_start = timeline_start.astimezone(UTC)

    sample_step_frames = max(
        1,
        round(source_fps / detector_scan.requested_sample_fps),
    )
    expected_source_indexes = tuple(range(0, source_frame_count, sample_step_frames))
    expected_set = set(expected_source_indexes)
    detections_by_source_index: dict[int, list[Any]] = {}
    for detection in detector_scan.result.detections:
        source_position = detection.timestamp_seconds * source_fps
        source_index = round(source_position)
        if abs(source_position - source_index) > 1e-6:
            raise DetectorEvaluationError(
                "detector timestamp does not map to an exact source frame"
            )
        if not 0 <= source_index < source_frame_count:
            raise DetectorEvaluationError("detector timestamp is outside the admitted source")
        if source_index not in expected_set:
            raise DetectorEvaluationError(
                "detector timestamp conflicts with the declared sampling step"
            )
        detections_by_source_index.setdefault(source_index, []).append(detection)

    frames = tuple(
        DetectionFrame(
            frame_id=f"sample_{source_index:08d}",
            occurred_at=timeline_start + timedelta(seconds=source_index / source_fps),
            detections=tuple(
                FrameDetection(
                    detection_id=f"det_{index:04d}",
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox_xyxy=detection.bbox_xyxy,
                )
                for index, detection in enumerate(detections_by_source_index.get(source_index, ()))
            ),
        )
        for source_index in expected_source_indexes
    )
    return DetectionFrameBatch(
        camera_id=detector_scan.result.camera_id,
        detector_adapter=detector_scan.result.detector_adapter,
        frames=frames,
    )


def build_detector_sampled_tracking_batch(
    detector_scan: AdmittedDetectorScan,
    *,
    media_admission: ExternalSourceAdmission,
    timeline_start: datetime,
) -> DetectionFrameBatch:
    """Convert actual v2 invocations to an offline tracker batch using source PTS.

    The UTC anchor represents source PTS zero, not the first sample or capture
    wall time. No nominal frames are inserted, and partial decoded input is not
    upgraded to source completeness. Duplicate/fallback clocks cannot satisfy
    the tracker's strict chronology and are rejected, never repaired.
    """
    if not isinstance(detector_scan, AdmittedDetectorScan):
        raise DetectorEvaluationError("detector scan is invalid")
    if not isinstance(media_admission, ExternalSourceAdmission):
        raise DetectorEvaluationError("media admission is invalid")
    if (
        media_admission.asset_kind != "media_asset"
        or media_admission.admission.status != "verified_local"
    ):
        raise DetectorEvaluationError("detector tracking conversion requires verified local media")
    if detector_scan.source_admission_sha256 != media_admission.fingerprint:
        raise DetectorEvaluationError("detector scan belongs to a different admitted media asset")
    ledger = detector_scan.result.sampling
    if detector_scan.version != 2 or ledger is None:
        raise DetectorEvaluationError("actual PTS tracking requires a v2 sampling ledger")
    if (
        not isinstance(timeline_start, datetime)
        or timeline_start.tzinfo is None
        or timeline_start.utcoffset() is None
    ):
        raise DetectorEvaluationError("timeline_start must be timezone-aware")
    try:
        timeline_start = timeline_start.astimezone(UTC)
    except (OverflowError, ValueError):
        raise DetectorEvaluationError(
            "tracking timeline anchor is outside the supported range"
        ) from None
    if (
        len(ledger.samples) > MAX_TRACKING_FRAMES
        or len(detector_scan.result.detections) > MAX_TRACKING_DETECTIONS
        or any(sample.detection_count > MAX_DETECTIONS_PER_FRAME for sample in ledger.samples)
    ):
        raise DetectorEvaluationError("actual samples exceed the tracking batch limits")
    frames: list[DetectionFrame] = []
    previous_pts = -1.0
    previous_time: datetime | None = None
    for sample in ledger.samples:
        if sample.timestamp_source != "container_pts":
            raise DetectorEvaluationError("actual PTS tracking cannot use a fallback sample clock")
        if sample.timestamp_seconds <= previous_pts:
            raise DetectorEvaluationError(
                "actual PTS tracking requires strictly increasing samples"
            )
        try:
            occurred_at = timeline_start + timedelta(seconds=sample.timestamp_seconds)
        except (OverflowError, ValueError):
            raise DetectorEvaluationError(
                "sample PTS exceeds the tracking datetime range"
            ) from None
        if previous_time is not None and occurred_at <= previous_time:
            raise DetectorEvaluationError(
                "sample PTS cannot preserve chronology at microsecond precision"
            )
        detections = detector_scan.result.detections[
            sample.detection_start : sample.detection_start + sample.detection_count
        ]
        try:
            frames.append(
                DetectionFrame(
                    frame_id=f"sample_{sample.decoded_frame_index:08d}",
                    occurred_at=occurred_at,
                    detections=tuple(
                        FrameDetection(
                            detection_id=f"det_{index:04d}",
                            label=detection.label,
                            confidence=detection.confidence,
                            bbox_xyxy=detection.bbox_xyxy,
                        )
                        for index, detection in enumerate(detections)
                    ),
                )
            )
        except ObjectTrackerError:
            raise DetectorEvaluationError(
                "actual sample is incompatible with the tracker contract"
            ) from None
        previous_pts = sample.timestamp_seconds
        previous_time = occurred_at
    try:
        return DetectionFrameBatch(
            camera_id=detector_scan.result.camera_id,
            detector_adapter=detector_scan.result.detector_adapter,
            frames=tuple(frames),
        )
    except ObjectTrackerError:
        raise DetectorEvaluationError(
            "actual samples are incompatible with the tracker contract"
        ) from None


@dataclass(frozen=True, slots=True)
class DetectorBoxQualityReport:
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
    confidence_threshold: float
    requested_sample_fps: float
    source_fps: float
    effective_sample_fps: float
    sample_step_frames: int
    source_frame_count: int
    annotation_frame_count: int
    annotation_frame_offset: int
    ignored_trailing_source_frame_count: int
    ignored_trailing_detector_box_count: int
    iou_threshold: float
    evaluated_frame_count: int
    evaluated_frames_with_ground_truth: int
    ground_truth_box_count: int
    detector_box_count: int
    matched_box_count: int
    missed_ground_truth_box_count: int
    false_positive_box_count: int
    precision: float | None
    recall: float | None
    f1: float | None
    mean_matched_iou: float | None
    frames_without_detector_boxes: int
    frames_with_missed_boxes: int
    frames_with_false_positive_boxes: int
    ground_truth_subject_count: int
    subjects_with_missed_boxes: int
    longest_consecutive_subject_miss_samples: int
    longest_consecutive_subject_miss_seconds: float
    version: int = 1
    ground_truth_provenance: str = "admitted_publisher_human"
    frame_alignment_provenance: str = "explicit_local_ffprobe_and_visual_spot_check"
    frame_alignment_confirmed: bool = True
    detection_accuracy_evaluated: bool = True
    missed_detections_evaluated: bool = True
    subject_identity_inference: bool = False
    age_role_inference: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise DetectorEvaluationError("unsupported detector box quality report version")
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
            raise DetectorEvaluationError("detector quality label must remain generic person")
        for field, minimum, maximum in (
            ("confidence_threshold", 0.0, 1.0),
            ("requested_sample_fps", 0.01, 10.0),
            ("source_fps", 0.01, 240.0),
            ("effective_sample_fps", 0.01, 240.0),
            ("iou_threshold", 0.01, 1.0),
            ("longest_consecutive_subject_miss_seconds", 0.0, 86_400.0),
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
        _positive_count(self.sample_step_frames, "sample_step_frames")
        _positive_count(self.source_frame_count, "source_frame_count")
        _positive_count(self.annotation_frame_count, "annotation_frame_count")
        _count(self.annotation_frame_offset, "annotation_frame_offset")
        for field in (
            "ignored_trailing_source_frame_count",
            "ignored_trailing_detector_box_count",
            "evaluated_frame_count",
            "evaluated_frames_with_ground_truth",
            "ground_truth_box_count",
            "detector_box_count",
            "matched_box_count",
            "missed_ground_truth_box_count",
            "false_positive_box_count",
            "frames_without_detector_boxes",
            "frames_with_missed_boxes",
            "frames_with_false_positive_boxes",
            "ground_truth_subject_count",
            "subjects_with_missed_boxes",
            "longest_consecutive_subject_miss_samples",
        ):
            _count(getattr(self, field), field)
        for field in ("precision", "recall", "f1", "mean_matched_iou"):
            object.__setattr__(
                self,
                field,
                _optional_ratio(getattr(self, field), field),
            )
        if self.source_frame_count < (self.annotation_frame_offset + self.annotation_frame_count):
            raise DetectorEvaluationError(
                "detector quality source does not contain all annotation frames"
            )
        if self.ignored_trailing_source_frame_count != self.source_frame_count - (
            self.annotation_frame_offset + self.annotation_frame_count
        ):
            raise DetectorEvaluationError(
                "detector quality trailing source frame count is inconsistent"
            )
        if self.matched_box_count + self.missed_ground_truth_box_count != (
            self.ground_truth_box_count
        ):
            raise DetectorEvaluationError(
                "detector quality ground-truth box accounting is inconsistent"
            )
        if self.matched_box_count + self.false_positive_box_count != (self.detector_box_count):
            raise DetectorEvaluationError(
                "detector quality detector box accounting is inconsistent"
            )
        if self.evaluated_frames_with_ground_truth > self.evaluated_frame_count:
            raise DetectorEvaluationError("detector quality frame accounting is inconsistent")
        expected_evaluated_frame_count = sum(
            source_index % self.sample_step_frames == 0
            for source_index in range(
                self.annotation_frame_offset,
                self.annotation_frame_offset + self.annotation_frame_count,
            )
        )
        if self.evaluated_frame_count != expected_evaluated_frame_count:
            raise DetectorEvaluationError("detector quality sampled frame count is inconsistent")
        if not math.isclose(
            self.effective_sample_fps,
            self.source_fps / self.sample_step_frames,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DetectorEvaluationError("detector quality effective sample fps is inconsistent")
        if any(
            count > self.evaluated_frame_count
            for count in (
                self.frames_without_detector_boxes,
                self.frames_with_missed_boxes,
                self.frames_with_false_positive_boxes,
            )
        ):
            raise DetectorEvaluationError("detector quality per-frame counts are inconsistent")
        if (
            self.frames_with_missed_boxes > self.missed_ground_truth_box_count
            or self.frames_with_false_positive_boxes > self.false_positive_box_count
        ):
            raise DetectorEvaluationError("detector quality error frame counts exceed error boxes")
        if self.subjects_with_missed_boxes > self.ground_truth_subject_count:
            raise DetectorEvaluationError("detector quality subject accounting is inconsistent")
        expected_precision = (
            None
            if self.detector_box_count == 0
            else round(self.matched_box_count / self.detector_box_count, 12)
        )
        expected_recall = (
            None
            if self.ground_truth_box_count == 0
            else round(self.matched_box_count / self.ground_truth_box_count, 12)
        )
        expected_f1 = (
            None
            if (
                expected_precision is None
                or expected_recall is None
                or expected_precision + expected_recall == 0.0
            )
            else round(
                2.0 * expected_precision * expected_recall / (expected_precision + expected_recall),
                12,
            )
        )
        if (
            self.precision != expected_precision
            or self.recall != expected_recall
            or self.f1 != expected_f1
        ):
            raise DetectorEvaluationError(
                "detector quality precision/recall accounting is inconsistent"
            )
        if (self.matched_box_count == 0) != (self.mean_matched_iou is None):
            raise DetectorEvaluationError(
                "detector quality mean matched IoU presence is inconsistent"
            )
        if not math.isclose(
            self.longest_consecutive_subject_miss_seconds,
            self.longest_consecutive_subject_miss_samples / self.effective_sample_fps,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise DetectorEvaluationError(
                "detector quality consecutive miss duration is inconsistent"
            )
        if self.ground_truth_provenance != "admitted_publisher_human":
            raise DetectorEvaluationError("detector quality ground truth provenance is invalid")
        if self.frame_alignment_provenance != ("explicit_local_ffprobe_and_visual_spot_check"):
            raise DetectorEvaluationError("detector quality frame alignment provenance is invalid")
        for field, expected in (
            ("frame_alignment_confirmed", True),
            ("detection_accuracy_evaluated", True),
            ("missed_detections_evaluated", True),
            ("subject_identity_inference", False),
            ("age_role_inference", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise DetectorEvaluationError(
                    f"detector quality {field} must be {str(expected).lower()}"
                )

    @classmethod
    def from_dict(cls, value: Any) -> DetectorBoxQualityReport:
        fields = {
            "version",
            "ground_truth_batch_sha256",
            "ground_truth_review_sha256",
            "ground_truth_admission_sha256",
            "media_admission_sha256",
            "detector_scan_sha256",
            "camera_id",
            "detector_adapter",
            "model_id",
            "model_revision",
            "model_sha256",
            "label",
            "confidence_threshold",
            "requested_sample_fps",
            "source_fps",
            "effective_sample_fps",
            "sample_step_frames",
            "source_frame_count",
            "annotation_frame_count",
            "annotation_frame_offset",
            "ignored_trailing_source_frame_count",
            "ignored_trailing_detector_box_count",
            "iou_threshold",
            "evaluated_frame_count",
            "evaluated_frames_with_ground_truth",
            "ground_truth_box_count",
            "detector_box_count",
            "matched_box_count",
            "missed_ground_truth_box_count",
            "false_positive_box_count",
            "precision",
            "recall",
            "f1",
            "mean_matched_iou",
            "frames_without_detector_boxes",
            "frames_with_missed_boxes",
            "frames_with_false_positive_boxes",
            "ground_truth_subject_count",
            "subjects_with_missed_boxes",
            "longest_consecutive_subject_miss_samples",
            "longest_consecutive_subject_miss_seconds",
            "ground_truth_provenance",
            "frame_alignment_provenance",
            "frame_alignment_confirmed",
            "detection_accuracy_evaluated",
            "missed_detections_evaluated",
            "subject_identity_inference",
            "age_role_inference",
            "source_media_included",
            "paths_disclosed",
            "report_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise DetectorEvaluationError("detector box quality report fields do not match v1")
        report_sha256 = value["report_sha256"]
        report = cls(**{key: item for key, item in value.items() if key != "report_sha256"})
        if report_sha256 != report.fingerprint:
            raise DetectorEvaluationError("detector box quality report fingerprint is invalid")
        return report

    @classmethod
    def from_json(cls, raw: str) -> DetectorBoxQualityReport:
        if len(raw.encode("utf-8")) > MAX_DETECTOR_QUALITY_REPORT_BYTES:
            raise DetectorEvaluationError("detector box quality report exceeds the 1 MiB limit")
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> DetectorBoxQualityReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_DETECTOR_QUALITY_REPORT_BYTES:
                raise DetectorEvaluationError("detector box quality report exceeds the 1 MiB limit")
            raw = source.read_text(encoding="utf-8")
        except DetectorEvaluationError:
            raise
        except (OSError, UnicodeError) as exc:
            raise DetectorEvaluationError(
                "detector box quality report is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value = {field: getattr(self, field) for field in self.__dataclass_fields__}
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
            raise DetectorEvaluationError("detector box quality report destination is unsafe")
        payload = f"{self.to_json()}\n".encode()
        if len(payload) > MAX_DETECTOR_QUALITY_REPORT_BYTES:
            raise DetectorEvaluationError("detector box quality report exceeds the 1 MiB limit")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(target, flags, 0o600)
        except FileExistsError:
            raise FileExistsError("detector box quality report output already exists") from None
        except OSError as exc:
            raise DetectorEvaluationError(
                "detector box quality report output could not be created"
            ) from exc
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
            raise DetectorEvaluationError(
                "detector box quality report output could not be written"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def to_public_dict(self) -> dict[str, object]:
        return self.to_dict()


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 12)


def evaluate_detector_box_quality(
    ground_truth_batch: DetectionFrameBatch,
    ground_truth_review: TrackerAssociationReview,
    detector_scan: AdmittedDetectorScan,
    *,
    ground_truth_admission: ExternalSourceAdmission,
    media_admission: ExternalSourceAdmission,
    source_fps: float,
    source_frame_count: int,
    annotation_frame_offset: int,
    iou_threshold: float = 0.5,
    frame_alignment_confirmed: bool = False,
) -> DetectorBoxQualityReport:
    if not frame_alignment_confirmed:
        raise DetectorEvaluationError(
            "detector evaluation requires explicit local frame alignment confirmation"
        )
    if not isinstance(ground_truth_batch, DetectionFrameBatch):
        raise DetectorEvaluationError("detector ground-truth batch is invalid")
    if not isinstance(ground_truth_review, TrackerAssociationReview):
        raise DetectorEvaluationError("detector ground-truth review is invalid")
    if not isinstance(detector_scan, AdmittedDetectorScan):
        raise DetectorEvaluationError("admitted detector scan is invalid")
    if not isinstance(ground_truth_admission, ExternalSourceAdmission) or not isinstance(
        media_admission,
        ExternalSourceAdmission,
    ):
        raise DetectorEvaluationError("detector source admissions are invalid")
    if ground_truth_review.review_status != "approved":
        raise DetectorEvaluationError("detector evaluation requires approved ground-truth review")
    if any(item.review_state != "assigned" for item in ground_truth_review.assignments):
        raise DetectorEvaluationError(
            "detector evaluation requires every ground-truth box to be assigned"
        )
    ground_truth_review.bind(ground_truth_batch)
    ground_truth_review.bind_source_admission(ground_truth_admission)
    if ground_truth_review.ground_truth_provenance != "admitted_publisher_human":
        raise DetectorEvaluationError(
            "detector evaluation requires admitted publisher-human ground truth"
        )
    if media_admission.asset_kind != "media_asset" or (
        media_admission.admission.status != "verified_local"
    ):
        raise DetectorEvaluationError("detector evaluation media admission must be verified_local")
    if detector_scan.source_admission_sha256 != media_admission.fingerprint:
        raise DetectorEvaluationError("detector scan belongs to a different media admission")
    if detector_scan.result.camera_id != ground_truth_batch.camera_id:
        raise DetectorEvaluationError("detector scan and ground truth camera IDs differ")
    if detector_scan.labels != ("person",):
        raise DetectorEvaluationError("detector quality evaluation requires a person-only scan")

    source_fps = _number(source_fps, "source_fps", minimum=0.01, maximum=240.0)
    source_frame_count = _positive_count(source_frame_count, "source_frame_count")
    annotation_frame_offset = _count(
        annotation_frame_offset,
        "annotation_frame_offset",
    )
    iou_threshold = _number(
        iou_threshold,
        "iou_threshold",
        minimum=0.01,
        maximum=1.0,
    )
    annotation_frame_count = len(ground_truth_batch.frames)
    if source_frame_count < annotation_frame_offset + annotation_frame_count:
        raise DetectorEvaluationError("source frame count does not cover the ground-truth batch")
    if detector_scan.result.sampling is not None:
        try:
            validate_nominal_sampling_grid(
                detector_scan.result.sampling,
                source_fps=source_fps,
                source_frame_count=source_frame_count,
            )
        except DetectorSamplingError as error:
            raise DetectorEvaluationError(str(error)) from None
    frame_seconds = 1.0 / source_fps
    for previous, current in zip(
        ground_truth_batch.frames,
        ground_truth_batch.frames[1:],
        strict=False,
    ):
        delta = (current.occurred_at - previous.occurred_at).total_seconds()
        if abs(delta - frame_seconds) > 1e-6:
            raise DetectorEvaluationError(
                "ground-truth batch does not match the declared source fps"
            )

    sample_step_frames = max(
        1,
        round(source_fps / detector_scan.requested_sample_fps),
    )
    effective_sample_fps = source_fps / sample_step_frames
    expected_source_indexes = tuple(
        source_index
        for source_index in range(
            annotation_frame_offset,
            annotation_frame_offset + annotation_frame_count,
        )
        if source_index % sample_step_frames == 0
    )
    expected_set = set(expected_source_indexes)
    detections_by_source_index: dict[int, list[Any]] = {}
    ignored_trailing_detector_box_count = 0
    for detection in detector_scan.result.detections:
        source_position = detection.timestamp_seconds * source_fps
        source_index = round(source_position)
        if abs(source_position - source_index) > 1e-6:
            raise DetectorEvaluationError(
                "detector timestamp does not map to an exact source frame"
            )
        if not 0 <= source_index < source_frame_count:
            raise DetectorEvaluationError("detector timestamp is outside the admitted source")
        if source_index % sample_step_frames != 0:
            raise DetectorEvaluationError(
                "detector timestamp conflicts with the declared sampling step"
            )
        if source_index not in expected_set:
            if source_index >= annotation_frame_offset + annotation_frame_count:
                ignored_trailing_detector_box_count += 1
                continue
            raise DetectorEvaluationError(
                "detector timestamp falls outside the aligned annotation window"
            )
        detections_by_source_index.setdefault(source_index, []).append(detection)

    subject_by_detection = {
        (assignment.frame_id, assignment.detection_id): assignment.subject_id
        for assignment in ground_truth_review.assignments
        if assignment.review_state == "assigned"
    }
    all_subjects = {subject for subject in subject_by_detection.values() if subject}
    matched_box_count = 0
    missed_ground_truth_box_count = 0
    false_positive_box_count = 0
    matched_ious: list[float] = []
    evaluated_frames_with_ground_truth = 0
    frames_without_detector_boxes = 0
    frames_with_missed_boxes = 0
    frames_with_false_positive_boxes = 0
    ground_truth_box_count = 0
    detector_box_count = 0
    missed_subjects: set[str] = set()
    current_miss_run: dict[str, int] = {subject: 0 for subject in all_subjects}
    longest_miss_run = 0

    for source_index in expected_source_indexes:
        frame = ground_truth_batch.frames[source_index - annotation_frame_offset]
        ground_truth = frame.detections
        detected = tuple(detections_by_source_index.get(source_index, ()))
        if ground_truth:
            evaluated_frames_with_ground_truth += 1
        if not detected:
            frames_without_detector_boxes += 1
        ground_truth_box_count += len(ground_truth)
        detector_box_count += len(detected)
        scores = [
            [_bbox_iou(expected.bbox_xyxy, actual.bbox_xyxy) for actual in detected]
            for expected in ground_truth
        ]
        matches = _maximum_weight_assignment(
            scores,
            minimum_score=iou_threshold,
        )
        matched_ground_truth = {pair[0] for pair in matches}
        matched_box_count += len(matches)
        missed = len(ground_truth) - len(matches)
        false_positive = len(detected) - len(matches)
        missed_ground_truth_box_count += missed
        false_positive_box_count += false_positive
        matched_ious.extend(scores[left][right] for left, right in matches)
        if missed:
            frames_with_missed_boxes += 1
        if false_positive:
            frames_with_false_positive_boxes += 1

        present_subjects: set[str] = set()
        missed_in_frame: set[str] = set()
        for index, expected in enumerate(ground_truth):
            subject = subject_by_detection[(frame.frame_id, expected.detection_id)]
            if subject is None:
                raise DetectorEvaluationError(
                    "ground-truth assignment is missing an anonymous subject"
                )
            present_subjects.add(subject)
            if index not in matched_ground_truth:
                missed_in_frame.add(subject)
                missed_subjects.add(subject)
        for subject in all_subjects:
            if subject in missed_in_frame:
                current_miss_run[subject] += 1
                longest_miss_run = max(
                    longest_miss_run,
                    current_miss_run[subject],
                )
            elif subject in present_subjects:
                current_miss_run[subject] = 0
            else:
                current_miss_run[subject] = 0

    precision = _ratio(matched_box_count, detector_box_count)
    recall = _ratio(matched_box_count, ground_truth_box_count)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0.0
        else round(2.0 * precision * recall / (precision + recall), 12)
    )
    return DetectorBoxQualityReport(
        ground_truth_batch_sha256=ground_truth_batch.fingerprint,
        ground_truth_review_sha256=ground_truth_review.fingerprint,
        ground_truth_admission_sha256=ground_truth_admission.fingerprint,
        media_admission_sha256=media_admission.fingerprint,
        detector_scan_sha256=detector_scan.fingerprint,
        camera_id=ground_truth_batch.camera_id,
        detector_adapter=detector_scan.result.detector_adapter,
        model_id=detector_scan.result.model_id,
        model_revision=detector_scan.result.model_revision,
        model_sha256=detector_scan.result.model_sha256,
        label="person",
        confidence_threshold=detector_scan.confidence_threshold,
        requested_sample_fps=detector_scan.requested_sample_fps,
        source_fps=source_fps,
        effective_sample_fps=effective_sample_fps,
        sample_step_frames=sample_step_frames,
        source_frame_count=source_frame_count,
        annotation_frame_count=annotation_frame_count,
        annotation_frame_offset=annotation_frame_offset,
        ignored_trailing_source_frame_count=source_frame_count
        - (annotation_frame_offset + annotation_frame_count),
        ignored_trailing_detector_box_count=ignored_trailing_detector_box_count,
        iou_threshold=iou_threshold,
        evaluated_frame_count=len(expected_source_indexes),
        evaluated_frames_with_ground_truth=evaluated_frames_with_ground_truth,
        ground_truth_box_count=ground_truth_box_count,
        detector_box_count=detector_box_count,
        matched_box_count=matched_box_count,
        missed_ground_truth_box_count=missed_ground_truth_box_count,
        false_positive_box_count=false_positive_box_count,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_matched_iou=(
            None if not matched_ious else round(sum(matched_ious) / len(matched_ious), 12)
        ),
        frames_without_detector_boxes=frames_without_detector_boxes,
        frames_with_missed_boxes=frames_with_missed_boxes,
        frames_with_false_positive_boxes=frames_with_false_positive_boxes,
        ground_truth_subject_count=len(all_subjects),
        subjects_with_missed_boxes=len(missed_subjects),
        longest_consecutive_subject_miss_samples=longest_miss_run,
        longest_consecutive_subject_miss_seconds=round(
            longest_miss_run / effective_sample_fps,
            12,
        ),
    )


def build_detector_tracking_evaluation_set(
    ground_truth_batch: DetectionFrameBatch,
    ground_truth_review: TrackerAssociationReview,
    detector_scan: AdmittedDetectorScan,
    quality_report: DetectorBoxQualityReport,
    *,
    ground_truth_admission: ExternalSourceAdmission,
    media_admission: ExternalSourceAdmission,
) -> tuple[DetectionFrameBatch, TrackerAssociationReview]:
    if not isinstance(quality_report, DetectorBoxQualityReport):
        raise DetectorEvaluationError("detector quality report is invalid")
    recomputed = evaluate_detector_box_quality(
        ground_truth_batch,
        ground_truth_review,
        detector_scan,
        ground_truth_admission=ground_truth_admission,
        media_admission=media_admission,
        source_fps=quality_report.source_fps,
        source_frame_count=quality_report.source_frame_count,
        annotation_frame_offset=quality_report.annotation_frame_offset,
        iou_threshold=quality_report.iou_threshold,
        frame_alignment_confirmed=quality_report.frame_alignment_confirmed,
    )
    if recomputed.fingerprint != quality_report.fingerprint:
        raise DetectorEvaluationError(
            "detector quality report does not match the exact evaluation inputs"
        )

    sample_step = quality_report.sample_step_frames
    annotation_start = quality_report.annotation_frame_offset
    annotation_end = annotation_start + quality_report.annotation_frame_count
    expected_source_indexes = tuple(
        index for index in range(annotation_start, annotation_end) if index % sample_step == 0
    )
    expected_set = set(expected_source_indexes)
    detections_by_source_index: dict[int, list[Any]] = {}
    for detection in detector_scan.result.detections:
        source_index = round(detection.timestamp_seconds * quality_report.source_fps)
        if source_index in expected_set:
            detections_by_source_index.setdefault(source_index, []).append(detection)

    subject_by_ground_truth = {
        (assignment.frame_id, assignment.detection_id): assignment.subject_id
        for assignment in ground_truth_review.assignments
        if assignment.review_state == "assigned"
    }
    derived_frames: list[DetectionFrame] = []
    assignments: list[TrackerReviewAssignment] = []
    for source_index in expected_source_indexes:
        ground_truth_frame = ground_truth_batch.frames[source_index - annotation_start]
        detected = tuple(detections_by_source_index.get(source_index, ()))
        frame_detections = tuple(
            FrameDetection(
                detection_id=f"det_{index:04d}",
                label=detection.label,
                confidence=detection.confidence,
                bbox_xyxy=detection.bbox_xyxy,
            )
            for index, detection in enumerate(detected)
        )
        derived_frames.append(
            DetectionFrame(
                frame_id=ground_truth_frame.frame_id,
                occurred_at=ground_truth_frame.occurred_at,
                detections=frame_detections,
            )
        )
        scores = [
            [_bbox_iou(expected.bbox_xyxy, actual.bbox_xyxy) for actual in frame_detections]
            for expected in ground_truth_frame.detections
        ]
        matches = _maximum_weight_assignment(
            scores,
            minimum_score=quality_report.iou_threshold,
        )
        ground_truth_by_detector = {
            detector_index: ground_truth_index for ground_truth_index, detector_index in matches
        }
        for detector_index, detection in enumerate(frame_detections):
            ground_truth_index = ground_truth_by_detector.get(detector_index)
            if ground_truth_index is None:
                assignments.append(
                    TrackerReviewAssignment(
                        frame_id=ground_truth_frame.frame_id,
                        detection_id=detection.detection_id,
                        review_state="false_positive",
                        subject_id=None,
                    )
                )
                continue
            ground_truth_detection = ground_truth_frame.detections[ground_truth_index]
            subject = subject_by_ground_truth.get(
                (
                    ground_truth_frame.frame_id,
                    ground_truth_detection.detection_id,
                )
            )
            if subject is None:
                raise DetectorEvaluationError(
                    "matched ground-truth box has no anonymous publisher subject"
                )
            assignments.append(
                TrackerReviewAssignment(
                    frame_id=ground_truth_frame.frame_id,
                    detection_id=detection.detection_id,
                    review_state="assigned",
                    subject_id=subject,
                )
            )

    derived_batch = DetectionFrameBatch(
        camera_id=ground_truth_batch.camera_id,
        detector_adapter=detector_scan.result.detector_adapter,
        frames=tuple(derived_frames),
    )
    derived_review = TrackerAssociationReview(
        camera_id=derived_batch.camera_id,
        detector_adapter=derived_batch.detector_adapter,
        detection_batch_sha256=derived_batch.fingerprint,
        source_admission_sha256=ground_truth_admission.fingerprint,
        review_status="approved",
        assignments=tuple(assignments),
        reviews=(),
        human_labels=True,
        ground_truth_provenance="admitted_publisher_human",
    )
    derived_review.bind(derived_batch)
    derived_review.bind_source_admission(ground_truth_admission)
    assigned_count = sum(
        assignment.review_state == "assigned" for assignment in derived_review.assignments
    )
    false_positive_count = sum(
        assignment.review_state == "false_positive" for assignment in derived_review.assignments
    )
    if (
        assigned_count != quality_report.matched_box_count
        or false_positive_count != quality_report.false_positive_box_count
        or len(derived_review.assignments) != quality_report.detector_box_count
    ):
        raise DetectorEvaluationError(
            "detector tracking conversion does not reproduce quality box accounting"
        )
    return derived_batch, derived_review
