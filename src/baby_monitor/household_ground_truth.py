"""Household time-range ground truth and person-trigger coverage evaluation.

The existing detector and candidate-gate evaluations depend on per-box human
ground truth (admitted publisher boxes or a full tracker association review).
That cost is not realistic for household clips, where the user can realistically
mark time ranges: empty-room negatives, person-present periods, and the
confirmed dangerous-event window. This module provides that cheaper, strict
ingestion contract and a coverage evaluator that reuses the live candidate
worker's grouping semantics.

The annotation and report stay path-free, anonymous, and carry no identity,
box-localization, danger-recognition, risk, or notification authority.
"""

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

from .candidate_trigger_evaluation import _group_detections
from .detector_protocol import AdmittedDetectorScan
from .detector_sampling import DetectorSamplingError, validate_nominal_sampling_grid
from .source_admission import ExternalSourceAdmission

MAX_HOUSEHOLD_ANNOTATION_BYTES = 1024 * 1024
MAX_HOUSEHOLD_TRIGGER_REPORT_BYTES = 1024 * 1024
MAX_HOUSEHOLD_TRIGGER_POINTS = 32
MAX_HOUSEHOLD_ANNOTATION_RANGES = 256

HOUSEHOLD_RANGE_KINDS = ("danger_event", "empty_room", "person_present")
HOUSEHOLD_SCENARIO_TAGS = (
    "crib_overhead_view",
    "daytime",
    "eye_level_view",
    "large_subject_scale",
    "night_vision",
    "partial_occlusion",
    "small_subject_scale",
)

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_REVIEWER_ID = re.compile(r"reviewer_[A-Za-z0-9_.-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class HouseholdGroundTruthError(ValueError):
    """A household annotation or trigger coverage report is unsafe."""


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HouseholdGroundTruthError(
                    "household ground truth JSON contains duplicate keys"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                HouseholdGroundTruthError(
                    "household ground truth JSON contains a non-standard number"
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise HouseholdGroundTruthError("household ground truth JSON is invalid") from exc


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
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HouseholdGroundTruthError(f"household ground truth {field} is invalid")
    try:
        converted = float(value)
        valid = math.isfinite(converted) and minimum <= converted <= maximum
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise HouseholdGroundTruthError(f"household ground truth {field} is invalid")
    return converted


def _count(value: Any, field: str, *, maximum: int = 1_000_000) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise HouseholdGroundTruthError(f"household ground truth {field} is invalid")
    return value


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise HouseholdGroundTruthError(f"household ground truth {field} is invalid")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HouseholdGroundTruthError(f"household ground truth {field} is invalid")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 12)


def _write_exclusive(payload: str, path: str | Path, *, limit: int) -> None:
    target = Path(path).expanduser()
    parent = target.parent
    if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
        raise HouseholdGroundTruthError("household ground truth destination is unsafe")
    data = f"{payload}\n".encode()
    if len(data) > limit:
        raise HouseholdGroundTruthError("household ground truth output exceeds the size limit")
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
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FileExistsError("household ground truth output already exists") from None
    except OSError as exc:
        with suppress(OSError):
            target.unlink()
        raise HouseholdGroundTruthError(
            "household ground truth output could not be written"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class HouseholdEpisodeRange:
    start_seconds: float
    end_seconds: float
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "start_seconds",
            _number(self.start_seconds, "range start_seconds", minimum=0.0, maximum=86_400.0),
        )
        object.__setattr__(
            self,
            "end_seconds",
            _number(self.end_seconds, "range end_seconds", minimum=0.0, maximum=86_400.0),
        )
        if self.kind not in HOUSEHOLD_RANGE_KINDS:
            raise HouseholdGroundTruthError("household episode range kind is invalid")
        if self.start_seconds >= self.end_seconds:
            raise HouseholdGroundTruthError("household episode range end must exceed start")

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdEpisodeRange:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdGroundTruthError("household episode range fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdEpisodeAnnotation:
    """Human time-range labels bound to one admitted household media asset.

    Ranges form one mutually exclusive timeline: ``danger_event`` marks the
    user-confirmed dangerous window, ``person_present`` any other period with a
    person visible, and ``empty_room`` declared no-person negatives. Time not
    covered by any range is excluded from evaluation denominators. The kinds
    describe human-visible presence only; they carry no identity, action, or
    risk semantics.

    V1 retains its nominal frame-grid metadata and wire fingerprint. V2 uses
    elapsed source PTS and has no nominal FPS/count in its wire representation;
    those constructor fields must be None. Its duration is a declared temporal
    horizon, not proof of complete video. Source PTS zero is bound separately.
    """

    annotation_id: str
    media_admission_sha256: str
    camera_id: str
    source_fps: float | None
    source_frame_count: int | None
    source_duration_seconds: float
    scenario_tags: tuple[str, ...]
    ranges: tuple[HouseholdEpisodeRange, ...]
    review_state: str
    reviewers: tuple[str, ...] = ()
    version: int = 1
    human_verified: bool = False
    model_generated_labels: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False
    subject_identity_inference: bool = False
    timebase: str = "nominal_frame_grid"

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version not in (1, 2):
            raise HouseholdGroundTruthError("unsupported household episode annotation version")
        _safe_id(self.annotation_id, "annotation_id")
        _sha256(self.media_admission_sha256, "media_admission_sha256")
        _safe_id(self.camera_id, "camera_id")
        if self.version == 1:
            if self.timebase != "nominal_frame_grid":
                raise HouseholdGroundTruthError(
                    "v1 household annotation requires a nominal timebase"
                )
            object.__setattr__(
                self,
                "source_fps",
                _number(self.source_fps, "source_fps", minimum=0.01, maximum=240.0),
            )
            object.__setattr__(
                self,
                "source_frame_count",
                _count(self.source_frame_count, "source_frame_count", maximum=10_000_000),
            )
            if self.source_frame_count < 1:
                raise HouseholdGroundTruthError(
                    "household episode annotation source_frame_count is invalid"
                )
        elif (
            self.timebase != "elapsed_source_pts"
            or self.source_fps is not None
            or self.source_frame_count is not None
        ):
            raise HouseholdGroundTruthError(
                "v2 household annotation requires elapsed source PTS without nominal FPS/count"
            )
        object.__setattr__(
            self,
            "source_duration_seconds",
            _number(
                self.source_duration_seconds,
                "source_duration_seconds",
                minimum=0.01,
                maximum=86_400.0,
            ),
        )
        if self.version == 1:
            frame_duration = self.source_frame_count / self.source_fps
            if abs(self.source_duration_seconds - frame_duration) > 1.0 / self.source_fps + 1e-9:
                raise HouseholdGroundTruthError(
                    "household episode annotation duration conflicts with the frame grid"
                )
        if (
            not isinstance(self.scenario_tags, tuple)
            or len(self.scenario_tags) > len(HOUSEHOLD_SCENARIO_TAGS)
            or any(tag not in HOUSEHOLD_SCENARIO_TAGS for tag in self.scenario_tags)
            or tuple(sorted(self.scenario_tags)) != self.scenario_tags
            or len(set(self.scenario_tags)) != len(self.scenario_tags)
        ):
            raise HouseholdGroundTruthError(
                "household episode annotation scenario tags are invalid"
            )
        if (
            not isinstance(self.ranges, tuple)
            or not 1 <= len(self.ranges) <= MAX_HOUSEHOLD_ANNOTATION_RANGES
            or any(not isinstance(item, HouseholdEpisodeRange) for item in self.ranges)
        ):
            raise HouseholdGroundTruthError("household episode annotation ranges are invalid")
        previous_end = 0.0
        for item in self.ranges:
            if item.start_seconds < previous_end:
                raise HouseholdGroundTruthError(
                    "household episode annotation ranges overlap or are unsorted"
                )
            if item.end_seconds > self.source_duration_seconds + (1e-9 if self.version == 1 else 0):
                raise HouseholdGroundTruthError(
                    "household episode annotation range exceeds the source duration"
                )
            previous_end = item.end_seconds
        if not isinstance(self.review_state, str) or self.review_state not in {
            "pending",
            "approved",
        }:
            raise HouseholdGroundTruthError("household episode annotation review state is invalid")
        if type(self.human_verified) is not bool:
            raise HouseholdGroundTruthError(
                "household episode annotation human_verified is invalid"
            )
        if (
            not isinstance(self.reviewers, tuple)
            or len(self.reviewers) > 8
            or any(
                not isinstance(reviewer, str) or _REVIEWER_ID.fullmatch(reviewer) is None
                for reviewer in self.reviewers
            )
            or len(set(self.reviewers)) != len(self.reviewers)
        ):
            raise HouseholdGroundTruthError("household episode annotation reviewers are invalid")
        if self.review_state == "approved":
            if not self.reviewers or self.human_verified is not True:
                raise HouseholdGroundTruthError(
                    "approved household annotation requires anonymous reviewers "
                    "and human_verified=true"
                )
        elif self.reviewers or self.human_verified:
            raise HouseholdGroundTruthError(
                "pending household annotation must have no reviewer and human_verified=false"
            )
        for field, expected in (
            ("model_generated_labels", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
            ("subject_identity_inference", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdGroundTruthError(
                    f"household episode annotation {field} must be {str(expected).lower()}"
                )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdEpisodeAnnotation:
        if not isinstance(value, dict) or type(value.get("version")) is not int:
            raise HouseholdGroundTruthError("unsupported household episode annotation version")
        version = value["version"]
        if version not in (1, 2):
            raise HouseholdGroundTruthError("unsupported household episode annotation version")
        omitted = {"timebase"} if version == 1 else {"source_fps", "source_frame_count"}
        fields = (set(cls.__dataclass_fields__) - omitted) | {"annotation_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdGroundTruthError(
                "household episode annotation fields do not match version"
            )
        if any(
            not isinstance(value[field], list) for field in ("ranges", "scenario_tags", "reviewers")
        ):
            raise HouseholdGroundTruthError("household episode annotation arrays are invalid")
        annotation_sha256 = value["annotation_sha256"]
        annotation = cls(
            **({"source_fps": None, "source_frame_count": None} if version == 2 else {}),
            **{
                key: (
                    tuple(HouseholdEpisodeRange.from_dict(item) for item in value[key])
                    if key == "ranges"
                    else tuple(value[key])
                    if key in {"scenario_tags", "reviewers"}
                    else value[key]
                )
                for key in value
                if key != "annotation_sha256"
            },
        )
        if annotation_sha256 != annotation.fingerprint:
            raise HouseholdGroundTruthError("household episode annotation fingerprint is invalid")
        return annotation

    @classmethod
    def from_json(cls, raw: str) -> HouseholdEpisodeAnnotation:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_ANNOTATION_BYTES:
            raise HouseholdGroundTruthError("household episode annotation exceeds the 1 MiB limit")
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdEpisodeAnnotation:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if source.stat().st_size > MAX_HOUSEHOLD_ANNOTATION_BYTES:
                raise HouseholdGroundTruthError(
                    "household episode annotation exceeds the 1 MiB limit"
                )
            raw = source.read_text(encoding="utf-8")
        except HouseholdGroundTruthError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdGroundTruthError(
                "household episode annotation is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: (
                [item.to_dict() for item in self.ranges]
                if field == "ranges"
                else list(getattr(self, field))
                if field in {"scenario_tags", "reviewers"}
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
            if field
            not in ({"timebase"} if self.version == 1 else {"source_fps", "source_frame_count"})
        }
        if include_fingerprint:
            value["annotation_sha256"] = self.fingerprint
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
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_ANNOTATION_BYTES,
        )


@dataclass(frozen=True, slots=True)
class HouseholdTriggerCoveragePoint:
    candidate_start_confidence: float
    candidate_group_count: int
    emitted_group_count: int
    suppressed_group_count: int
    emitted_groups_with_positive_sample: int
    emitted_groups_without_positive_sample: int
    covered_positive_sample_count: int
    uncovered_positive_sample_count: int
    positive_sample_recall: float | None
    activated_negative_sample_count: int
    negative_sample_false_activation_rate: float | None
    activated_unannotated_sample_count: int
    covered_person_episode_count: int
    missed_person_episode_count: int
    person_episode_recall: float | None
    covered_danger_episode_count: int
    missed_danger_episode_count: int
    danger_episode_recall: float | None
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
            "activated_unannotated_sample_count",
            "covered_person_episode_count",
            "missed_person_episode_count",
            "covered_danger_episode_count",
            "missed_danger_episode_count",
            "longest_uncovered_positive_samples",
        ):
            _count(getattr(self, field), f"point {field}")
        for field in (
            "positive_sample_recall",
            "negative_sample_false_activation_rate",
            "person_episode_recall",
            "danger_episode_recall",
        ):
            object.__setattr__(
                self,
                field,
                _number(getattr(self, field), f"point {field}", minimum=0.0, maximum=1.0)
                if getattr(self, field) is not None
                else None,
            )
        object.__setattr__(
            self,
            "maximum_detection_delay_seconds",
            _number(
                self.maximum_detection_delay_seconds,
                "point maximum_detection_delay_seconds",
                minimum=0.0,
                maximum=86_400.0,
            )
            if self.maximum_detection_delay_seconds is not None
            else None,
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
            raise HouseholdGroundTruthError(
                "household trigger point group accounting is inconsistent"
            )
        if (
            self.emitted_groups_with_positive_sample + self.emitted_groups_without_positive_sample
            != self.emitted_group_count
        ):
            raise HouseholdGroundTruthError(
                "household trigger point emitted-group accounting is inconsistent"
            )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdTriggerCoveragePoint:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdGroundTruthError("household trigger point fields do not match v1")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdTriggerCoverageReport:
    annotation_sha256: str
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
    source_duration_seconds: float
    scenario_tags: tuple[str, ...]
    evaluated_sample_count: int
    positive_sample_count: int
    negative_sample_count: int
    unannotated_sample_count: int
    person_episode_count: int
    danger_episode_count: int
    candidate_gap_seconds: float
    max_event_span_seconds: float
    points: tuple[HouseholdTriggerCoveragePoint, ...]
    version: int = 1
    ground_truth_provenance: str = "household_human_time_range"
    grouping_rule: str = "same_as_candidate_worker_gap_and_max_span"
    derived_without_model_rerun: bool = True
    candidate_trigger_coverage_evaluated: bool = True
    box_localization_accuracy_evaluated: bool = False
    danger_recognition_accuracy_evaluated: bool = False
    risk_accuracy_evaluated: bool = False
    deployment_approved: bool = False
    dataset_specific_only: bool = True
    subject_identity_inference: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HouseholdGroundTruthError("unsupported household trigger coverage report version")
        for field in (
            "annotation_sha256",
            "media_admission_sha256",
            "detector_scan_sha256",
            "model_sha256",
        ):
            _sha256(getattr(self, field), field)
        for field in ("camera_id", "detector_adapter", "model_id", "model_revision"):
            _safe_id(getattr(self, field), field)
        if self.label != "person":
            raise HouseholdGroundTruthError(
                "household trigger coverage label must remain generic person"
            )
        for field, minimum, maximum in (
            ("source_confidence_threshold", 0.0, 1.0),
            ("requested_sample_fps", 0.01, 10.0),
            ("source_fps", 0.01, 240.0),
            ("effective_sample_fps", 0.01, 240.0),
            ("source_duration_seconds", 0.01, 86_400.0),
            ("candidate_gap_seconds", 0.0, 300.0),
            ("max_event_span_seconds", 1.0, 300.0),
        ):
            object.__setattr__(
                self,
                field,
                _number(getattr(self, field), field, minimum=minimum, maximum=maximum),
            )
        for field in (
            "source_frame_count",
            "evaluated_sample_count",
        ):
            value = _count(getattr(self, field), field, maximum=10_000_000)
            if value < 1:
                raise HouseholdGroundTruthError(f"household trigger report {field} is invalid")
        for field in (
            "positive_sample_count",
            "negative_sample_count",
            "unannotated_sample_count",
            "person_episode_count",
            "danger_episode_count",
        ):
            _count(getattr(self, field), field)
        if (
            self.positive_sample_count + self.negative_sample_count + self.unannotated_sample_count
            != self.evaluated_sample_count
        ):
            raise HouseholdGroundTruthError(
                "household trigger report sample accounting is inconsistent"
            )
        if (
            not isinstance(self.scenario_tags, tuple)
            or len(self.scenario_tags) > len(HOUSEHOLD_SCENARIO_TAGS)
            or any(tag not in HOUSEHOLD_SCENARIO_TAGS for tag in self.scenario_tags)
            or tuple(sorted(self.scenario_tags)) != self.scenario_tags
            or len(set(self.scenario_tags)) != len(self.scenario_tags)
        ):
            raise HouseholdGroundTruthError("household trigger report scenario tags are invalid")
        if (
            not isinstance(self.points, tuple)
            or not 2 <= len(self.points) <= MAX_HOUSEHOLD_TRIGGER_POINTS
            or any(not isinstance(point, HouseholdTriggerCoveragePoint) for point in self.points)
        ):
            raise HouseholdGroundTruthError("household trigger report points are invalid")
        thresholds = tuple(point.candidate_start_confidence for point in self.points)
        if (
            thresholds != tuple(sorted(thresholds))
            or len(set(thresholds)) != len(thresholds)
            or thresholds[0] != self.source_confidence_threshold
        ):
            raise HouseholdGroundTruthError(
                "household trigger start thresholds must be unique, ascending, and "
                "start at the source threshold"
            )
        previous_emitted: int | None = None
        previous_covered: int | None = None
        previous_negative: int | None = None
        previous_unannotated: int | None = None
        for point in self.points:
            covered_episode_total = (
                point.covered_person_episode_count + point.covered_danger_episode_count
            )
            if (
                point.covered_positive_sample_count + point.uncovered_positive_sample_count
                != self.positive_sample_count
                or point.covered_person_episode_count + point.missed_person_episode_count
                != self.person_episode_count
                or point.covered_danger_episode_count + point.missed_danger_episode_count
                != self.danger_episode_count
                or point.activated_negative_sample_count > self.negative_sample_count
                or point.activated_unannotated_sample_count > self.unannotated_sample_count
                or point.positive_sample_recall
                != _ratio(
                    point.covered_positive_sample_count,
                    self.positive_sample_count,
                )
                or point.negative_sample_false_activation_rate
                != _ratio(
                    point.activated_negative_sample_count,
                    self.negative_sample_count,
                )
                or point.person_episode_recall
                != _ratio(
                    point.covered_person_episode_count,
                    self.person_episode_count,
                )
                or point.danger_episode_recall
                != _ratio(
                    point.covered_danger_episode_count,
                    self.danger_episode_count,
                )
                or not math.isclose(
                    point.longest_uncovered_positive_seconds,
                    point.longest_uncovered_positive_samples / self.effective_sample_fps,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or (
                    covered_episode_total == 0 and point.maximum_detection_delay_seconds is not None
                )
                or (covered_episode_total > 0 and point.maximum_detection_delay_seconds is None)
            ):
                raise HouseholdGroundTruthError(
                    "household trigger point coverage accounting is inconsistent"
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
                or (
                    previous_unannotated is not None
                    and point.activated_unannotated_sample_count > previous_unannotated
                )
            ):
                raise HouseholdGroundTruthError("household trigger point counts are not monotonic")
            previous_emitted = point.emitted_group_count
            previous_covered = point.covered_positive_sample_count
            previous_negative = point.activated_negative_sample_count
            previous_unannotated = point.activated_unannotated_sample_count
        if self.ground_truth_provenance != "household_human_time_range":
            raise HouseholdGroundTruthError("household trigger ground truth provenance is invalid")
        if self.grouping_rule != "same_as_candidate_worker_gap_and_max_span":
            raise HouseholdGroundTruthError("household trigger grouping rule is invalid")
        for field, expected in (
            ("derived_without_model_rerun", True),
            ("candidate_trigger_coverage_evaluated", True),
            ("box_localization_accuracy_evaluated", False),
            ("danger_recognition_accuracy_evaluated", False),
            ("risk_accuracy_evaluated", False),
            ("deployment_approved", False),
            ("dataset_specific_only", True),
            ("subject_identity_inference", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdGroundTruthError(
                    f"household trigger report {field} must be {str(expected).lower()}"
                )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdTriggerCoverageReport:
        fields = {*cls.__dataclass_fields__, "report_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdGroundTruthError("household trigger report fields do not match v1")
        report_sha256 = value["report_sha256"]
        report = cls(
            **{
                key: (
                    tuple(HouseholdTriggerCoveragePoint.from_dict(item) for item in value[key])
                    if key == "points"
                    else tuple(value[key])
                    if key == "scenario_tags"
                    else value[key]
                )
                for key, item in value.items()
                if key != "report_sha256"
            }
        )
        if report_sha256 != report.fingerprint:
            raise HouseholdGroundTruthError("household trigger report fingerprint is invalid")
        return report

    @classmethod
    def from_json(cls, raw: str) -> HouseholdTriggerCoverageReport:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_TRIGGER_REPORT_BYTES:
            raise HouseholdGroundTruthError("household trigger report exceeds the 1 MiB limit")
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdTriggerCoverageReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if source.stat().st_size > MAX_HOUSEHOLD_TRIGGER_REPORT_BYTES:
                raise HouseholdGroundTruthError("household trigger report exceeds the 1 MiB limit")
            raw = source.read_text(encoding="utf-8")
        except HouseholdGroundTruthError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdGroundTruthError(
                "household trigger report is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: (
                [point.to_dict() for point in self.points]
                if field == "points"
                else list(getattr(self, field))
                if field == "scenario_tags"
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
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_TRIGGER_REPORT_BYTES,
        )

    def to_public_dict(self) -> dict[str, object]:
        return self.to_dict()


def _range_kind_at(annotation: HouseholdEpisodeAnnotation, seconds: float) -> str | None:
    for item in annotation.ranges:
        if item.start_seconds <= seconds < item.end_seconds:
            return item.kind
    return None


def evaluate_household_trigger_coverage(
    annotation: HouseholdEpisodeAnnotation,
    detector_scan: AdmittedDetectorScan,
    *,
    media_admission: ExternalSourceAdmission,
    thresholds: tuple[float, ...],
    candidate_gap_seconds: float,
    max_event_span_seconds: float,
) -> HouseholdTriggerCoverageReport:
    """Measure person-trigger coverage against human time-range ground truth.

    A sampled grid point counts as detector-covered when an emitted candidate
    group (same gap/max-span semantics as the live worker) contains a person
    detection at that point. No box IoU matching is performed: this evaluates
    trigger coverage only, not box localization or danger recognition.

    This v1 evaluator supports the declared nominal source-frame grid only.
    Off-grid PTS must be rejected, not snapped to a nearby point or discarded.
    Scan v1 has no empty-frame/execution ledger: the annotation-derived grid is
    an evaluation denominator, not proof that the detector decoded every point.
    """

    if not isinstance(annotation, HouseholdEpisodeAnnotation) or annotation.version != 1:
        raise HouseholdGroundTruthError(
            "nominal trigger coverage requires v1 annotation; use actual sampled coverage for PTS"
        )
    if annotation.review_state != "approved":
        raise HouseholdGroundTruthError("pending household annotations cannot be evaluated")
    if (
        media_admission.fingerprint != annotation.media_admission_sha256
        or detector_scan.source_admission_sha256 != annotation.media_admission_sha256
    ):
        raise HouseholdGroundTruthError(
            "household annotation, media admission, and detector scan do not "
            "bind to the same admitted media"
        )
    if detector_scan.result.camera_id != annotation.camera_id:
        raise HouseholdGroundTruthError(
            "household annotation camera does not match the detector scan"
        )
    if "person" not in detector_scan.labels:
        raise HouseholdGroundTruthError(
            "household trigger coverage requires an admitted person scan"
        )
    parsed_thresholds = tuple(
        _number(value, "threshold", minimum=0.0, maximum=1.0) for value in thresholds
    )
    if (
        not 2 <= len(parsed_thresholds) <= MAX_HOUSEHOLD_TRIGGER_POINTS
        or parsed_thresholds != tuple(sorted(parsed_thresholds))
        or len(set(parsed_thresholds)) != len(parsed_thresholds)
        or parsed_thresholds[0] != detector_scan.confidence_threshold
    ):
        raise HouseholdGroundTruthError(
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

    source_fps = annotation.source_fps
    if detector_scan.result.sampling is not None:
        try:
            validate_nominal_sampling_grid(
                detector_scan.result.sampling,
                source_fps=source_fps,
                source_frame_count=annotation.source_frame_count,
            )
        except DetectorSamplingError as error:
            raise HouseholdGroundTruthError(str(error)) from None
    sample_step_frames = max(
        1,
        round(source_fps / detector_scan.requested_sample_fps),
    )
    effective_sample_fps = source_fps / sample_step_frames
    grid_indexes = tuple(range(0, annotation.source_frame_count, sample_step_frames))
    grid_set = set(grid_indexes)

    # Match the exact source-frame alignment gate used by box evaluation.
    # Validate before filtering labels or annotation ranges so incompatible
    # scan evidence cannot silently vanish from the nominal-grid report.
    for detection in detector_scan.result.detections:
        source_position = detection.timestamp_seconds * source_fps
        source_index = round(source_position)
        if abs(source_position - source_index) > 1e-6:
            raise HouseholdGroundTruthError(
                "detector timestamp does not map to an exact nominal source frame; "
                "off-grid PTS coverage is not supported"
            )
        if not 0 <= source_index < annotation.source_frame_count:
            raise HouseholdGroundTruthError("detector timestamp is outside the admitted source")
        if source_index % sample_step_frames != 0:
            raise HouseholdGroundTruthError(
                "detector timestamp conflicts with the declared sampling step"
            )

    positive_indexes: set[int] = set()
    negative_indexes: set[int] = set()
    person_episode_samples: list[list[int]] = []
    danger_episode_samples: list[list[int]] = []
    for item in annotation.ranges:
        episode: list[int] = []
        for source_index in grid_indexes:
            seconds = source_index / source_fps
            if item.start_seconds <= seconds < item.end_seconds:
                if item.kind == "empty_room":
                    negative_indexes.add(source_index)
                else:
                    positive_indexes.add(source_index)
                    episode.append(source_index)
        if item.kind == "person_present":
            person_episode_samples.append(episode)
        elif item.kind == "danger_event":
            danger_episode_samples.append(episode)
    unannotated_indexes = grid_set - positive_indexes - negative_indexes
    if any(not episode for episode in person_episode_samples) or any(
        not episode for episode in danger_episode_samples
    ):
        raise HouseholdGroundTruthError(
            "a labeled positive range contains no sampled grid point; extend "
            "the range or increase the scan sample rate"
        )

    detections = tuple(
        sorted(
            (
                detection
                for detection in detector_scan.result.detections
                if detection.label == "person"
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

    points: list[HouseholdTriggerCoveragePoint] = []
    for threshold in parsed_thresholds:
        emitted_groups = tuple(
            group
            for group in groups
            if any(detection.confidence >= threshold for detection in group)
        )
        emitted_indexes = {
            round(detection.timestamp_seconds * source_fps)
            for group in emitted_groups
            for detection in group
        }
        covered_positive = positive_indexes & emitted_indexes
        activated_negative = negative_indexes & emitted_indexes
        activated_unannotated = unannotated_indexes & emitted_indexes

        covered_person = 0
        covered_danger = 0
        delays: list[float] = []
        for episode in person_episode_samples:
            covered = set(episode) & covered_positive
            if covered:
                covered_person += 1
                delays.append((min(covered) - episode[0]) / source_fps)
        for episode in danger_episode_samples:
            covered = set(episode) & covered_positive
            if covered:
                covered_danger += 1
                delays.append((min(covered) - episode[0]) / source_fps)

        current_uncovered = 0
        longest_uncovered = 0
        for source_index in grid_indexes:
            if source_index in positive_indexes and source_index not in covered_positive:
                current_uncovered += 1
                longest_uncovered = max(longest_uncovered, current_uncovered)
            else:
                current_uncovered = 0

        points.append(
            HouseholdTriggerCoveragePoint(
                candidate_start_confidence=threshold,
                candidate_group_count=len(groups),
                emitted_group_count=len(emitted_groups),
                suppressed_group_count=len(groups) - len(emitted_groups),
                emitted_groups_with_positive_sample=sum(
                    1
                    for group in emitted_groups
                    if any(
                        round(detection.timestamp_seconds * source_fps) in positive_indexes
                        for detection in group
                    )
                ),
                emitted_groups_without_positive_sample=sum(
                    1
                    for group in emitted_groups
                    if not any(
                        round(detection.timestamp_seconds * source_fps) in positive_indexes
                        for detection in group
                    )
                ),
                covered_positive_sample_count=len(covered_positive),
                uncovered_positive_sample_count=(len(positive_indexes) - len(covered_positive)),
                positive_sample_recall=_ratio(
                    len(covered_positive),
                    len(positive_indexes),
                ),
                activated_negative_sample_count=len(activated_negative),
                negative_sample_false_activation_rate=_ratio(
                    len(activated_negative),
                    len(negative_indexes),
                ),
                activated_unannotated_sample_count=len(activated_unannotated),
                covered_person_episode_count=covered_person,
                missed_person_episode_count=len(person_episode_samples) - covered_person,
                person_episode_recall=_ratio(
                    covered_person,
                    len(person_episode_samples),
                ),
                covered_danger_episode_count=covered_danger,
                missed_danger_episode_count=len(danger_episode_samples) - covered_danger,
                danger_episode_recall=_ratio(
                    covered_danger,
                    len(danger_episode_samples),
                ),
                maximum_detection_delay_seconds=(None if not delays else round(max(delays), 12)),
                longest_uncovered_positive_samples=longest_uncovered,
                longest_uncovered_positive_seconds=round(
                    longest_uncovered / effective_sample_fps,
                    12,
                ),
            )
        )

    return HouseholdTriggerCoverageReport(
        annotation_sha256=annotation.fingerprint,
        media_admission_sha256=media_admission.fingerprint,
        detector_scan_sha256=detector_scan.fingerprint,
        camera_id=detector_scan.result.camera_id,
        detector_adapter=detector_scan.result.detector_adapter,
        model_id=detector_scan.result.model_id,
        model_revision=detector_scan.result.model_revision,
        model_sha256=detector_scan.result.model_sha256,
        label="person",
        source_confidence_threshold=detector_scan.confidence_threshold,
        requested_sample_fps=detector_scan.requested_sample_fps,
        source_fps=source_fps,
        effective_sample_fps=effective_sample_fps,
        source_frame_count=annotation.source_frame_count,
        source_duration_seconds=annotation.source_duration_seconds,
        scenario_tags=annotation.scenario_tags,
        evaluated_sample_count=len(grid_indexes),
        positive_sample_count=len(positive_indexes),
        negative_sample_count=len(negative_indexes),
        unannotated_sample_count=len(unannotated_indexes),
        person_episode_count=len(person_episode_samples),
        danger_episode_count=len(danger_episode_samples),
        candidate_gap_seconds=candidate_gap_seconds,
        max_event_span_seconds=max_event_span_seconds,
        points=tuple(points),
    )


HOUSEHOLD_TRIGGER_SELECTION_RULE = (
    "danger_episode_recall_then_person_episode_recall_then_sample_recall_"
    "then_delay_then_uncovered_then_negative_activation_then_workload_"
    "then_lower_threshold"
)


@dataclass(frozen=True, slots=True)
class HouseholdTriggerPolicy:
    minimum_danger_episode_recall: float
    minimum_person_episode_recall: float
    minimum_positive_sample_recall: float
    maximum_detection_delay_seconds: float
    maximum_longest_uncovered_positive_seconds: float
    maximum_negative_false_activation_rate: float | None = None
    require_negative_samples: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_danger_episode_recall, bool)
            or not isinstance(self.minimum_danger_episode_recall, (int, float))
            or not math.isfinite(float(self.minimum_danger_episode_recall))
            or float(self.minimum_danger_episode_recall) != 1.0
        ):
            raise HouseholdGroundTruthError(
                "household trigger policy must require danger episode recall 1.0"
            )
        object.__setattr__(
            self,
            "minimum_danger_episode_recall",
            1.0,
        )
        for field in (
            "minimum_person_episode_recall",
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
        maximum_negative = self.maximum_negative_false_activation_rate
        if maximum_negative is not None:
            object.__setattr__(
                self,
                "maximum_negative_false_activation_rate",
                _number(
                    maximum_negative,
                    "policy maximum_negative_false_activation_rate",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        if type(self.require_negative_samples) is not bool:
            raise HouseholdGroundTruthError("policy require_negative_samples must be a boolean")
        if (
            self.maximum_negative_false_activation_rate is not None
            and not self.require_negative_samples
        ):
            raise HouseholdGroundTruthError(
                "a negative false-activation limit requires negative samples"
            )

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def select_household_trigger_operating_point(
    report: HouseholdTriggerCoverageReport,
    policy: HouseholdTriggerPolicy,
) -> tuple[HouseholdTriggerCoveragePoint | None, int]:
    if not isinstance(report, HouseholdTriggerCoverageReport):
        raise HouseholdGroundTruthError("household trigger selection report is invalid")
    if not isinstance(policy, HouseholdTriggerPolicy):
        raise HouseholdGroundTruthError("household trigger selection policy is invalid")
    if policy.require_negative_samples and report.negative_sample_count == 0:
        return None, 0
    feasible: list[HouseholdTriggerCoveragePoint] = []
    for point in report.points:
        delay = point.maximum_detection_delay_seconds
        if (
            point.danger_episode_recall is not None
            and point.danger_episode_recall < policy.minimum_danger_episode_recall
        ):
            continue
        if (
            point.person_episode_recall is not None
            and point.person_episode_recall < policy.minimum_person_episode_recall
        ):
            continue
        if (
            point.positive_sample_recall is not None
            and point.positive_sample_recall < policy.minimum_positive_sample_recall
        ):
            continue
        if delay is None or delay > policy.maximum_detection_delay_seconds:
            continue
        if (
            point.longest_uncovered_positive_seconds
            > policy.maximum_longest_uncovered_positive_seconds
        ):
            continue
        false_activation = point.negative_sample_false_activation_rate
        if policy.maximum_negative_false_activation_rate is not None and (
            false_activation is None
            or false_activation > policy.maximum_negative_false_activation_rate
        ):
            continue
        feasible.append(point)
    if not feasible:
        return None, 0

    def person_recall_or_inf(point: HouseholdTriggerCoveragePoint) -> float:
        return (
            -point.person_episode_recall if point.person_episode_recall is not None else -math.inf
        )

    def sample_recall_or_inf(point: HouseholdTriggerCoveragePoint) -> float:
        return (
            -point.positive_sample_recall if point.positive_sample_recall is not None else -math.inf
        )

    def danger_recall_key(point: HouseholdTriggerCoveragePoint) -> float:
        return -point.danger_episode_recall if point.danger_episode_recall is not None else -1.0

    selected = min(
        feasible,
        key=lambda point: (
            danger_recall_key(point),
            person_recall_or_inf(point),
            sample_recall_or_inf(point),
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


HOUSEHOLD_TRIGGER_COMPARISON_RULE = "same_annotation_media_grid_and_thresholds"


def compare_household_trigger_coverage(
    reports: tuple[HouseholdTriggerCoverageReport, ...],
) -> dict[str, object]:
    """Compare models that scanned the same admitted clip on the same grid.

    The comparison only verifies that every report binds the same approved
    annotation, media admission, sample grid, episode accounting, and start
    threshold sequence, then lists the per-threshold metrics side by side. It
    never selects a winner and never approves deployment.
    """

    if (
        not isinstance(reports, tuple)
        or not 2 <= len(reports) <= 8
        or any(not isinstance(report, HouseholdTriggerCoverageReport) for report in reports)
    ):
        raise HouseholdGroundTruthError("household trigger comparison requires 2-8 valid reports")
    first = reports[0]
    grid_fields = (
        "annotation_sha256",
        "media_admission_sha256",
        "source_fps",
        "effective_sample_fps",
        "source_frame_count",
        "source_duration_seconds",
        "scenario_tags",
        "evaluated_sample_count",
        "positive_sample_count",
        "negative_sample_count",
        "unannotated_sample_count",
        "person_episode_count",
        "danger_episode_count",
        "candidate_gap_seconds",
        "max_event_span_seconds",
    )
    for report in reports[1:]:
        for field in grid_fields:
            if getattr(report, field) != getattr(first, field):
                raise HouseholdGroundTruthError(
                    "household trigger comparison reports do not share the same "
                    "annotation, media, grid, or grouping"
                )
        if tuple(point.candidate_start_confidence for point in report.points) != tuple(
            point.candidate_start_confidence for point in first.points
        ):
            raise HouseholdGroundTruthError(
                "household trigger comparison requires the same start threshold "
                "sequence for every model"
            )
    model_identities = {
        (report.model_id, report.model_revision, report.model_sha256) for report in reports
    }
    if len(model_identities) != len(reports):
        raise HouseholdGroundTruthError(
            "household trigger comparison requires distinct model identities"
        )

    points_by_threshold: list[dict[str, object]] = []
    for index, threshold in enumerate(point.candidate_start_confidence for point in first.points):
        points_by_threshold.append(
            {
                "candidate_start_confidence": threshold,
                "models": [
                    {
                        "model_id": report.model_id,
                        "positive_sample_recall": (report.points[index].positive_sample_recall),
                        "person_episode_recall": (report.points[index].person_episode_recall),
                        "danger_episode_recall": (report.points[index].danger_episode_recall),
                        "maximum_detection_delay_seconds": (
                            report.points[index].maximum_detection_delay_seconds
                        ),
                        "longest_uncovered_positive_seconds": (
                            report.points[index].longest_uncovered_positive_seconds
                        ),
                        "activated_negative_sample_count": (
                            report.points[index].activated_negative_sample_count
                        ),
                        "activated_unannotated_sample_count": (
                            report.points[index].activated_unannotated_sample_count
                        ),
                        "emitted_group_count": (report.points[index].emitted_group_count),
                    }
                    for report in reports
                ],
            }
        )
    return {
        "valid": True,
        "version": 1,
        "comparison_rule": HOUSEHOLD_TRIGGER_COMPARISON_RULE,
        "annotation_sha256": first.annotation_sha256,
        "media_admission_sha256": first.media_admission_sha256,
        "scenario_tags": list(first.scenario_tags),
        "evaluated_sample_count": first.evaluated_sample_count,
        "positive_sample_count": first.positive_sample_count,
        "negative_sample_count": first.negative_sample_count,
        "person_episode_count": first.person_episode_count,
        "danger_episode_count": first.danger_episode_count,
        "models": [
            {
                "model_id": report.model_id,
                "model_revision": report.model_revision,
                "model_sha256": report.model_sha256,
                "detector_adapter": report.detector_adapter,
                "detector_scan_sha256": report.detector_scan_sha256,
            }
            for report in reports
        ],
        "thresholds": [point.candidate_start_confidence for point in first.points],
        "points_by_threshold": points_by_threshold,
        "same_grid_and_media_verified": True,
        "winner_selected": False,
        "dataset_specific_only": True,
        "deployment_approved": False,
        "household_configuration_changed": False,
        "danger_recognition_accuracy_evaluated": False,
        "box_localization_accuracy_evaluated": False,
        "risk_accuracy_evaluated": False,
        "subject_identity_inference": False,
        "paths_disclosed": False,
    }
