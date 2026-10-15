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

from .detector_evaluation import DetectorEvaluationError, evaluate_detector_box_quality
from .detector_protocol import AdmittedDetectorScan
from .object_tracker import DetectionFrameBatch
from .source_admission import ExternalSourceAdmission
from .tracker_evaluation import TrackerAssociationReview

MAX_CANDIDATE_TRIGGER_REPORT_BYTES = 1024 * 1024
MAX_CANDIDATE_TRIGGER_POINTS = 32
_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CandidateTriggerEvaluationError(ValueError):
    """A candidate trigger coverage evaluation is inconsistent or unsafe."""


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CandidateTriggerEvaluationError(
                    "candidate trigger report JSON contains duplicate keys"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                CandidateTriggerEvaluationError(
                    "candidate trigger report JSON contains a non-standard number"
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise CandidateTriggerEvaluationError("candidate trigger report JSON is invalid") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise CandidateTriggerEvaluationError(f"candidate trigger report {field} is invalid")
    return float(value)


def _optional_ratio(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field, minimum=0.0, maximum=1.0)


def _optional_seconds(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field, minimum=0.0, maximum=86_400.0)


def _count(value: Any, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 1_000_000:
        raise CandidateTriggerEvaluationError(f"candidate trigger report {field} is invalid")
    return value


def _positive_count(value: Any, field: str) -> int:
    value = _count(value, field)
    if value == 0:
        raise CandidateTriggerEvaluationError(f"candidate trigger report {field} is invalid")
    return value


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise CandidateTriggerEvaluationError(f"candidate trigger report {field} is invalid")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CandidateTriggerEvaluationError(f"candidate trigger report {field} is invalid")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 12)


@dataclass(frozen=True, slots=True)
class CandidateStartGateSweepPoint:
    candidate_start_confidence: float
    candidate_group_count: int
    emitted_group_count: int
    suppressed_group_count: int
    emitted_groups_with_positive_sample: int
    emitted_groups_without_positive_sample: int
    covered_positive_sample_count: int
    uncovered_positive_sample_count: int
    positive_sample_recall: float
    activated_negative_sample_count: int
    negative_sample_false_activation_rate: float | None
    covered_ground_truth_episode_count: int
    missed_ground_truth_episode_count: int
    ground_truth_episode_recall: float
    maximum_detection_delay_seconds: float | None
    longest_uncovered_positive_samples: int
    longest_uncovered_positive_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_start_confidence",
            _number(
                self.candidate_start_confidence,
                "point candidate_start_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        for field in (
            "candidate_group_count",
            "emitted_group_count",
            "suppressed_group_count",
            "emitted_groups_with_positive_sample",
            "emitted_groups_without_positive_sample",
            "covered_positive_sample_count",
            "uncovered_positive_sample_count",
            "activated_negative_sample_count",
            "covered_ground_truth_episode_count",
            "missed_ground_truth_episode_count",
            "longest_uncovered_positive_samples",
        ):
            _count(getattr(self, field), f"point {field}")
        object.__setattr__(
            self,
            "positive_sample_recall",
            _number(
                self.positive_sample_recall,
                "point positive_sample_recall",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "negative_sample_false_activation_rate",
            _optional_ratio(
                self.negative_sample_false_activation_rate,
                "point negative_sample_false_activation_rate",
            ),
        )
        object.__setattr__(
            self,
            "ground_truth_episode_recall",
            _number(
                self.ground_truth_episode_recall,
                "point ground_truth_episode_recall",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "maximum_detection_delay_seconds",
            _optional_seconds(
                self.maximum_detection_delay_seconds,
                "point maximum_detection_delay_seconds",
            ),
        )
        object.__setattr__(
            self,
            "longest_uncovered_positive_seconds",
            _number(
                self.longest_uncovered_positive_seconds,
                "point longest_uncovered_positive_seconds",
                minimum=0.0,
                maximum=86_400.0,
            ),
        )
        if self.emitted_group_count + self.suppressed_group_count != (self.candidate_group_count):
            raise CandidateTriggerEvaluationError(
                "candidate trigger point group accounting is inconsistent"
            )
        if (
            self.emitted_groups_with_positive_sample + self.emitted_groups_without_positive_sample
            != self.emitted_group_count
        ):
            raise CandidateTriggerEvaluationError(
                "candidate trigger point emitted-group accounting is inconsistent"
            )

    @classmethod
    def from_dict(cls, value: Any) -> CandidateStartGateSweepPoint:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise CandidateTriggerEvaluationError("candidate trigger point fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CandidateStartGateSweepReport:
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
    evaluated_sample_count: int
    positive_sample_count: int
    negative_sample_count: int
    ground_truth_episode_count: int
    candidate_gap_seconds: float
    max_event_span_seconds: float
    validation_iou_threshold: float
    points: tuple[CandidateStartGateSweepPoint, ...]
    version: int = 1
    ground_truth_provenance: str = "admitted_publisher_human"
    grouping_rule: str = "same_as_candidate_worker_gap_and_max_span"
    derived_without_model_rerun: bool = True
    candidate_trigger_coverage_evaluated: bool = True
    box_localization_accuracy_evaluated: bool = False
    risk_accuracy_evaluated: bool = False
    deployment_approved: bool = False
    subject_identity_inference: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise CandidateTriggerEvaluationError("unsupported candidate trigger report version")
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
            raise CandidateTriggerEvaluationError(
                "candidate trigger report label must remain generic person"
            )
        for field, minimum, maximum in (
            ("source_confidence_threshold", 0.0, 1.0),
            ("requested_sample_fps", 0.01, 10.0),
            ("source_fps", 0.01, 240.0),
            ("effective_sample_fps", 0.01, 240.0),
            ("candidate_gap_seconds", 0.0, 300.0),
            ("max_event_span_seconds", 1.0, 300.0),
            ("validation_iou_threshold", 0.01, 1.0),
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
            "evaluated_sample_count",
            "positive_sample_count",
            "ground_truth_episode_count",
        ):
            _positive_count(getattr(self, field), field)
        for field in ("annotation_frame_offset", "negative_sample_count"):
            _count(getattr(self, field), field)
        if self.positive_sample_count + self.negative_sample_count != (self.evaluated_sample_count):
            raise CandidateTriggerEvaluationError(
                "candidate trigger report sample accounting is inconsistent"
            )
        if (
            not isinstance(self.points, tuple)
            or not 2 <= len(self.points) <= MAX_CANDIDATE_TRIGGER_POINTS
            or any(not isinstance(point, CandidateStartGateSweepPoint) for point in self.points)
        ):
            raise CandidateTriggerEvaluationError("candidate trigger report points are invalid")
        thresholds = tuple(point.candidate_start_confidence for point in self.points)
        if (
            thresholds != tuple(sorted(thresholds))
            or len(set(thresholds)) != len(thresholds)
            or thresholds[0] != self.source_confidence_threshold
        ):
            raise CandidateTriggerEvaluationError(
                "candidate start thresholds must be unique, ascending, and start "
                "at the source threshold"
            )
        previous_emitted: int | None = None
        previous_covered: int | None = None
        previous_negative: int | None = None
        for point in self.points:
            if (
                point.covered_positive_sample_count + point.uncovered_positive_sample_count
                != self.positive_sample_count
                or point.covered_ground_truth_episode_count
                + point.missed_ground_truth_episode_count
                != self.ground_truth_episode_count
                or point.activated_negative_sample_count > self.negative_sample_count
                or point.positive_sample_recall
                != _ratio(point.covered_positive_sample_count, self.positive_sample_count)
                or point.negative_sample_false_activation_rate
                != _ratio(point.activated_negative_sample_count, self.negative_sample_count)
                or point.ground_truth_episode_recall
                != _ratio(
                    point.covered_ground_truth_episode_count,
                    self.ground_truth_episode_count,
                )
                or not math.isclose(
                    point.longest_uncovered_positive_seconds,
                    point.longest_uncovered_positive_samples / self.effective_sample_fps,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or (
                    point.covered_ground_truth_episode_count == 0
                    and point.maximum_detection_delay_seconds is not None
                )
                or (
                    point.covered_ground_truth_episode_count > 0
                    and point.maximum_detection_delay_seconds is None
                )
            ):
                raise CandidateTriggerEvaluationError(
                    "candidate trigger point coverage accounting is inconsistent"
                )
            if (
                (previous_emitted is not None and point.emitted_group_count > previous_emitted)
                or (
                    previous_covered is not None
                    and point.covered_positive_sample_count > previous_covered
                )
                or (
                    previous_negative is not None
                    and point.activated_negative_sample_count > previous_negative
                )
            ):
                raise CandidateTriggerEvaluationError(
                    "candidate trigger point counts are not monotonic"
                )
            previous_emitted = point.emitted_group_count
            previous_covered = point.covered_positive_sample_count
            previous_negative = point.activated_negative_sample_count
        if self.ground_truth_provenance != "admitted_publisher_human":
            raise CandidateTriggerEvaluationError(
                "candidate trigger ground truth provenance is invalid"
            )
        if self.grouping_rule != "same_as_candidate_worker_gap_and_max_span":
            raise CandidateTriggerEvaluationError("candidate trigger grouping rule is invalid")
        for field, expected in (
            ("derived_without_model_rerun", True),
            ("candidate_trigger_coverage_evaluated", True),
            ("box_localization_accuracy_evaluated", False),
            ("risk_accuracy_evaluated", False),
            ("deployment_approved", False),
            ("subject_identity_inference", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise CandidateTriggerEvaluationError(
                    f"candidate trigger report {field} must be {str(expected).lower()}"
                )

    @classmethod
    def from_dict(cls, value: Any) -> CandidateStartGateSweepReport:
        fields = {*cls.__dataclass_fields__, "report_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise CandidateTriggerEvaluationError("candidate trigger report fields do not match v1")
        report_sha256 = value["report_sha256"]
        report = cls(
            **{
                key: (
                    tuple(CandidateStartGateSweepPoint.from_dict(point) for point in item)
                    if key == "points" and isinstance(item, list)
                    else item
                )
                for key, item in value.items()
                if key != "report_sha256"
            }
        )
        if report_sha256 != report.fingerprint:
            raise CandidateTriggerEvaluationError("candidate trigger report fingerprint is invalid")
        return report

    @classmethod
    def from_json(cls, raw: str) -> CandidateStartGateSweepReport:
        if len(raw.encode("utf-8")) > MAX_CANDIDATE_TRIGGER_REPORT_BYTES:
            raise CandidateTriggerEvaluationError(
                "candidate trigger report exceeds the 1 MiB limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> CandidateStartGateSweepReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if source.stat().st_size > MAX_CANDIDATE_TRIGGER_REPORT_BYTES:
                raise CandidateTriggerEvaluationError(
                    "candidate trigger report exceeds the 1 MiB limit"
                )
            raw = source.read_text(encoding="utf-8")
        except CandidateTriggerEvaluationError:
            raise
        except (OSError, UnicodeError) as exc:
            raise CandidateTriggerEvaluationError(
                "candidate trigger report is missing or unsafe"
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
            raise CandidateTriggerEvaluationError("candidate trigger report destination is unsafe")
        payload = f"{self.to_json()}\n".encode()
        if len(payload) > MAX_CANDIDATE_TRIGGER_REPORT_BYTES:
            raise CandidateTriggerEvaluationError(
                "candidate trigger report exceeds the 1 MiB limit"
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
            raise FileExistsError("candidate trigger report output already exists") from None
        except OSError as exc:
            with suppress(OSError):
                target.unlink()
            raise CandidateTriggerEvaluationError(
                "candidate trigger report output could not be written"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def to_public_dict(self) -> dict[str, object]:
        return self.to_dict()


def _group_detections(
    detections: tuple[Any, ...],
    *,
    candidate_gap_seconds: float,
    max_event_span_seconds: float,
) -> tuple[tuple[Any, ...], ...]:
    groups: list[list[Any]] = []
    for detection in detections:
        if not groups:
            groups.append([detection])
            continue
        group = groups[-1]
        gap = detection.timestamp_seconds - group[-1].timestamp_seconds
        span = detection.timestamp_seconds - group[0].timestamp_seconds
        if gap <= candidate_gap_seconds and span <= max_event_span_seconds:
            group.append(detection)
        else:
            groups.append([detection])
    return tuple(tuple(group) for group in groups)


def _ground_truth_episodes(
    positive_indexes: set[int],
    *,
    sample_step_frames: int,
) -> tuple[tuple[int, ...], ...]:
    episodes: list[list[int]] = []
    for source_index in sorted(positive_indexes):
        if not episodes or source_index - episodes[-1][-1] != sample_step_frames:
            episodes.append([source_index])
        else:
            episodes[-1].append(source_index)
    return tuple(tuple(episode) for episode in episodes)


def evaluate_candidate_start_gate_sweep(
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
    candidate_gap_seconds: float,
    max_event_span_seconds: float,
    validation_iou_threshold: float = 0.5,
    frame_alignment_confirmed: bool = False,
) -> CandidateStartGateSweepReport:
    parsed_thresholds = tuple(
        _number(value, "threshold", minimum=0.0, maximum=1.0) for value in thresholds
    )
    if (
        not 2 <= len(parsed_thresholds) <= MAX_CANDIDATE_TRIGGER_POINTS
        or parsed_thresholds != tuple(sorted(parsed_thresholds))
        or len(set(parsed_thresholds)) != len(parsed_thresholds)
        or parsed_thresholds[0] != detector_scan.confidence_threshold
    ):
        raise CandidateTriggerEvaluationError(
            "thresholds must be 2-32 unique ascending values starting at "
            "the admitted scan threshold"
        )
    candidate_gap_seconds = _number(
        candidate_gap_seconds,
        "candidate_gap_seconds",
        minimum=0.0,
        maximum=300.0,
    )
    max_event_span_seconds = _number(
        max_event_span_seconds,
        "max_event_span_seconds",
        minimum=1.0,
        maximum=300.0,
    )
    validation_iou_threshold = _number(
        validation_iou_threshold,
        "validation_iou_threshold",
        minimum=0.01,
        maximum=1.0,
    )
    try:
        quality = evaluate_detector_box_quality(
            ground_truth_batch,
            ground_truth_review,
            detector_scan,
            ground_truth_admission=ground_truth_admission,
            media_admission=media_admission,
            source_fps=source_fps,
            source_frame_count=source_frame_count,
            annotation_frame_offset=annotation_frame_offset,
            iou_threshold=validation_iou_threshold,
            frame_alignment_confirmed=frame_alignment_confirmed,
        )
    except DetectorEvaluationError as exc:
        raise CandidateTriggerEvaluationError(str(exc)) from exc

    sample_step_frames = max(
        1,
        round(quality.source_fps / detector_scan.requested_sample_fps),
    )
    expected_source_indexes = tuple(
        source_index
        for source_index in range(
            annotation_frame_offset,
            annotation_frame_offset + len(ground_truth_batch.frames),
        )
        if source_index % sample_step_frames == 0
    )
    expected_set = set(expected_source_indexes)
    assigned = {
        (assignment.frame_id, assignment.detection_id)
        for assignment in ground_truth_review.assignments
        if assignment.review_state == "assigned"
    }
    positive_indexes: set[int] = set()
    for source_index in expected_source_indexes:
        frame = ground_truth_batch.frames[source_index - annotation_frame_offset]
        if any(
            (frame.frame_id, detection.detection_id) in assigned for detection in frame.detections
        ):
            positive_indexes.add(source_index)
    negative_indexes = expected_set - positive_indexes
    detections = tuple(
        sorted(
            (
                detection
                for detection in detector_scan.result.detections
                if round(detection.timestamp_seconds * quality.source_fps) in expected_set
            ),
            key=lambda detection: (
                detection.timestamp_seconds,
                -detection.confidence,
                detection.bbox_xyxy,
            ),
        )
    )
    groups = _group_detections(
        detections,
        candidate_gap_seconds=candidate_gap_seconds,
        max_event_span_seconds=max_event_span_seconds,
    )
    episodes = _ground_truth_episodes(
        positive_indexes,
        sample_step_frames=sample_step_frames,
    )
    points: list[CandidateStartGateSweepPoint] = []
    for threshold in parsed_thresholds:
        emitted_groups = tuple(
            group
            for group in groups
            if any(detection.confidence >= threshold for detection in group)
        )
        emitted_indexes = {
            round(detection.timestamp_seconds * quality.source_fps)
            for group in emitted_groups
            for detection in group
        }
        covered_positive = positive_indexes & emitted_indexes
        activated_negative = negative_indexes & emitted_indexes
        covered_episodes = tuple(episode for episode in episodes if set(episode) & covered_positive)
        delays = tuple(
            (min(set(episode) & covered_positive) - episode[0]) / quality.source_fps
            for episode in covered_episodes
        )
        current_uncovered = 0
        longest_uncovered = 0
        for source_index in expected_source_indexes:
            if source_index in positive_indexes and source_index not in covered_positive:
                current_uncovered += 1
                longest_uncovered = max(longest_uncovered, current_uncovered)
            else:
                current_uncovered = 0
        points.append(
            CandidateStartGateSweepPoint(
                candidate_start_confidence=threshold,
                candidate_group_count=len(groups),
                emitted_group_count=len(emitted_groups),
                suppressed_group_count=len(groups) - len(emitted_groups),
                emitted_groups_with_positive_sample=sum(
                    1
                    for group in emitted_groups
                    if any(
                        round(detection.timestamp_seconds * quality.source_fps) in positive_indexes
                        for detection in group
                    )
                ),
                emitted_groups_without_positive_sample=sum(
                    1
                    for group in emitted_groups
                    if not any(
                        round(detection.timestamp_seconds * quality.source_fps) in positive_indexes
                        for detection in group
                    )
                ),
                covered_positive_sample_count=len(covered_positive),
                uncovered_positive_sample_count=(len(positive_indexes) - len(covered_positive)),
                positive_sample_recall=_ratio(
                    len(covered_positive),
                    len(positive_indexes),
                )
                or 0.0,
                activated_negative_sample_count=len(activated_negative),
                negative_sample_false_activation_rate=_ratio(
                    len(activated_negative),
                    len(negative_indexes),
                ),
                covered_ground_truth_episode_count=len(covered_episodes),
                missed_ground_truth_episode_count=(len(episodes) - len(covered_episodes)),
                ground_truth_episode_recall=_ratio(
                    len(covered_episodes),
                    len(episodes),
                )
                or 0.0,
                maximum_detection_delay_seconds=(None if not delays else round(max(delays), 12)),
                longest_uncovered_positive_samples=longest_uncovered,
                longest_uncovered_positive_seconds=round(
                    longest_uncovered / quality.effective_sample_fps,
                    12,
                ),
            )
        )

    return CandidateStartGateSweepReport(
        ground_truth_batch_sha256=ground_truth_batch.fingerprint,
        ground_truth_review_sha256=ground_truth_review.fingerprint,
        ground_truth_admission_sha256=ground_truth_admission.fingerprint,
        media_admission_sha256=media_admission.fingerprint,
        detector_scan_sha256=detector_scan.fingerprint,
        camera_id=quality.camera_id,
        detector_adapter=quality.detector_adapter,
        model_id=quality.model_id,
        model_revision=quality.model_revision,
        model_sha256=quality.model_sha256,
        label=quality.label,
        source_confidence_threshold=detector_scan.confidence_threshold,
        requested_sample_fps=quality.requested_sample_fps,
        source_fps=quality.source_fps,
        effective_sample_fps=quality.effective_sample_fps,
        source_frame_count=quality.source_frame_count,
        annotation_frame_count=quality.annotation_frame_count,
        annotation_frame_offset=quality.annotation_frame_offset,
        evaluated_sample_count=len(expected_source_indexes),
        positive_sample_count=len(positive_indexes),
        negative_sample_count=len(negative_indexes),
        ground_truth_episode_count=len(episodes),
        candidate_gap_seconds=candidate_gap_seconds,
        max_event_span_seconds=max_event_span_seconds,
        validation_iou_threshold=validation_iou_threshold,
        points=tuple(points),
    )


CANDIDATE_GATE_SELECTION_RULE = (
    "episode_recall_then_sample_recall_then_delay_then_uncovered_"
    "then_negative_activation_then_workload_then_lower_threshold"
)


@dataclass(frozen=True, slots=True)
class CandidateStartGatePolicy:
    minimum_ground_truth_episode_recall: float
    minimum_positive_sample_recall: float
    maximum_detection_delay_seconds: float
    maximum_longest_uncovered_positive_seconds: float
    maximum_negative_false_activation_rate: float | None = None
    require_negative_samples: bool = False

    def __post_init__(self) -> None:
        for field in (
            "minimum_ground_truth_episode_recall",
            "minimum_positive_sample_recall",
        ):
            object.__setattr__(
                self,
                field,
                _number(
                    getattr(self, field),
                    f"policy {field}",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        for field in (
            "maximum_detection_delay_seconds",
            "maximum_longest_uncovered_positive_seconds",
        ):
            object.__setattr__(
                self,
                field,
                _number(
                    getattr(self, field),
                    f"policy {field}",
                    minimum=0.0,
                    maximum=86_400.0,
                ),
            )
        object.__setattr__(
            self,
            "maximum_negative_false_activation_rate",
            _optional_ratio(
                self.maximum_negative_false_activation_rate,
                "policy maximum_negative_false_activation_rate",
            ),
        )
        if type(self.require_negative_samples) is not bool:
            raise CandidateTriggerEvaluationError(
                "policy require_negative_samples must be a boolean"
            )
        if (
            self.maximum_negative_false_activation_rate is not None
            and not self.require_negative_samples
        ):
            raise CandidateTriggerEvaluationError(
                "a negative false-activation limit requires negative samples"
            )

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def select_candidate_start_gate_operating_point(
    report: CandidateStartGateSweepReport,
    policy: CandidateStartGatePolicy,
) -> tuple[CandidateStartGateSweepPoint | None, int]:
    if not isinstance(report, CandidateStartGateSweepReport):
        raise CandidateTriggerEvaluationError("candidate gate selection report is invalid")
    if not isinstance(policy, CandidateStartGatePolicy):
        raise CandidateTriggerEvaluationError("candidate gate selection policy is invalid")
    if policy.require_negative_samples and report.negative_sample_count == 0:
        return None, 0
    feasible: list[CandidateStartGateSweepPoint] = []
    for point in report.points:
        delay = point.maximum_detection_delay_seconds
        false_activation = point.negative_sample_false_activation_rate
        if (
            point.ground_truth_episode_recall < policy.minimum_ground_truth_episode_recall
            or point.positive_sample_recall < policy.minimum_positive_sample_recall
            or delay is None
            or delay > policy.maximum_detection_delay_seconds
            or point.longest_uncovered_positive_seconds
            > policy.maximum_longest_uncovered_positive_seconds
            or (
                policy.maximum_negative_false_activation_rate is not None
                and (
                    false_activation is None
                    or false_activation > policy.maximum_negative_false_activation_rate
                )
            )
        ):
            continue
        feasible.append(point)
    if not feasible:
        return None, 0
    selected = min(
        feasible,
        key=lambda point: (
            -point.ground_truth_episode_recall,
            -point.positive_sample_recall,
            point.maximum_detection_delay_seconds
            if point.maximum_detection_delay_seconds is not None
            else math.inf,
            point.longest_uncovered_positive_seconds,
            point.activated_negative_sample_count,
            point.emitted_group_count,
            point.candidate_start_confidence,
        ),
    )
    return selected, len(feasible)
