"""Model-generated household presence review and diagnostic threshold sweeps.

This module deliberately does not extend :mod:`household_ground_truth`.
Visual-model labels are useful for candidate triage and iterative shadow
calibration, but they are not independent human ground truth.  The contracts
below keep that lower authority explicit while still making the automated
YOLO -> selected stills -> visual review -> threshold replay loop repeatable.

All inputs and outputs are path-free, media-free, identity-free offline
artifacts.  Evaluation consumes only already-admitted source records and
already-saved detector scans; it never reads media, loads a model, accesses the
network, mutates household configuration, or selects a production winner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detector_protocol import AdmittedDetectorScan
from .source_admission import ExternalSourceAdmission

MAX_HOUSEHOLD_MODEL_REVIEW_BYTES = 1024 * 1024
MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_BYTES = 1024 * 1024
MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_DRIFT_BYTES = 1024 * 1024
MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES = 32
MAX_HOUSEHOLD_MODEL_REVIEW_ITEMS = 4096
MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_POINTS = 32

HOUSEHOLD_MODEL_REVIEW_MEDIA_TRANSFERS = (
    "local_model_only",
    "selected_stills_remote",
)
HOUSEHOLD_MODEL_REVIEW_SAMPLING_STRATEGIES = (
    "candidate_disagreement_and_control",
    "candidate_targeted",
    "uniform_grid",
)
VISIBLE_PERSON_LABELS = ("absent", "present", "unknown")
VISIBLE_PERSON_VISIBILITIES = ("clear", "partial", "unknown")
HOUSEHOLD_MODEL_DIAGNOSTIC_SELECTION_RULE = (
    "recall_then_partial_recall_then_precision_then_f1_then_specificity_then_higher_threshold"
)

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class HouseholdModelDiagnosticError(ValueError):
    """A model-generated review or diagnostic report is unsafe."""


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic JSON contains duplicate keys"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                HouseholdModelDiagnosticError(
                    "household model diagnostic JSON contains a non-standard number"
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise HouseholdModelDiagnosticError("household model diagnostic JSON is invalid") from exc


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
        raise HouseholdModelDiagnosticError(f"household model diagnostic {field} is invalid")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise HouseholdModelDiagnosticError(f"household model diagnostic {field} is invalid")
    return value


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise HouseholdModelDiagnosticError(f"household model diagnostic {field} is invalid")
    return float(value)


def _optional_ratio(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field, minimum=0.0, maximum=1.0)


def _count(value: Any, field: str, *, maximum: int = 10_000_000) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise HouseholdModelDiagnosticError(f"household model diagnostic {field} is invalid")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 12)


def _write_exclusive(payload: str, path: str | Path, *, limit: int) -> None:
    target = Path(path).expanduser()
    parent = target.parent
    if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
        raise HouseholdModelDiagnosticError("household model diagnostic destination is unsafe")
    data = f"{payload}\n".encode()
    if len(data) > limit:
        raise HouseholdModelDiagnosticError(
            "household model diagnostic output exceeds the size limit"
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
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FileExistsError("household model diagnostic output already exists") from None
    except OSError as exc:
        with suppress(OSError):
            target.unlink()
        raise HouseholdModelDiagnosticError(
            "household model diagnostic output could not be written"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class HouseholdModelPresenceSource:
    camera_id: str
    media_admission_sha256: str
    source_duration_seconds: float

    def __post_init__(self) -> None:
        _safe_id(self.camera_id, "source camera_id")
        _sha256(self.media_admission_sha256, "source media_admission_sha256")
        object.__setattr__(
            self,
            "source_duration_seconds",
            _number(
                self.source_duration_seconds,
                "source source_duration_seconds",
                minimum=0.01,
                maximum=86_400.0,
            ),
        )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelPresenceSource:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household model presence source fields do not match v1"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdModelPresenceItem:
    review_item_id: str
    camera_id: str
    media_admission_sha256: str
    timestamp_seconds: float
    source_frame_sha256: str
    visible_person: str
    visibility: str
    label_confidence: float

    def __post_init__(self) -> None:
        _safe_id(self.review_item_id, "item review_item_id")
        _safe_id(self.camera_id, "item camera_id")
        _sha256(self.media_admission_sha256, "item media_admission_sha256")
        _sha256(self.source_frame_sha256, "item source_frame_sha256")
        object.__setattr__(
            self,
            "timestamp_seconds",
            _number(
                self.timestamp_seconds,
                "item timestamp_seconds",
                minimum=0.0,
                maximum=86_400.0,
            ),
        )
        object.__setattr__(
            self,
            "label_confidence",
            _number(
                self.label_confidence,
                "item label_confidence",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if self.visible_person not in VISIBLE_PERSON_LABELS:
            raise HouseholdModelDiagnosticError(
                "household model presence item visible_person is invalid"
            )
        if self.visibility not in VISIBLE_PERSON_VISIBILITIES:
            raise HouseholdModelDiagnosticError(
                "household model presence item visibility is invalid"
            )
        if self.visible_person == "present" and self.visibility not in {
            "clear",
            "partial",
        }:
            raise HouseholdModelDiagnosticError(
                "present household model label requires clear or partial visibility"
            )
        if self.visible_person == "absent" and self.visibility != "clear":
            raise HouseholdModelDiagnosticError(
                "absent household model label requires clear visibility"
            )
        if self.visible_person == "unknown" and self.visibility != "unknown":
            raise HouseholdModelDiagnosticError(
                "unknown household model label requires unknown visibility"
            )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelPresenceItem:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household model presence item fields do not match v1"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdModelPresenceReview:
    review_id: str
    selection_manifest_sha256: str
    review_model_id: str
    review_model_revision: str
    review_media_transfer: str
    sampling_strategy: str
    sources: tuple[HouseholdModelPresenceSource, ...]
    items: tuple[HouseholdModelPresenceItem, ...]
    version: int = 1
    review_scope: str = "visible_person_presence_only"
    model_generated_labels: bool = True
    human_verified: bool = False
    ground_truth: bool = False
    approved_for_official_evaluation: bool = False
    selected_stills_only: bool = True
    clips_shared: bool = False
    audio_shared: bool = False
    identity_inference: bool = False
    biometric_processing: bool = False
    training_allowed: bool = False
    redistribution_allowed: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HouseholdModelDiagnosticError(
                "unsupported household model presence review version"
            )
        for field in ("review_id", "review_model_id", "review_model_revision"):
            _safe_id(getattr(self, field), field)
        _sha256(self.selection_manifest_sha256, "selection_manifest_sha256")
        if self.review_media_transfer not in HOUSEHOLD_MODEL_REVIEW_MEDIA_TRANSFERS:
            raise HouseholdModelDiagnosticError("household model review media transfer is invalid")
        if self.sampling_strategy not in HOUSEHOLD_MODEL_REVIEW_SAMPLING_STRATEGIES:
            raise HouseholdModelDiagnosticError(
                "household model review sampling strategy is invalid"
            )
        if (
            not isinstance(self.sources, tuple)
            or not 1 <= len(self.sources) <= MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES
            or any(not isinstance(item, HouseholdModelPresenceSource) for item in self.sources)
        ):
            raise HouseholdModelDiagnosticError(
                "household model presence review sources are invalid"
            )
        source_order = tuple(source.camera_id for source in self.sources)
        if source_order != tuple(sorted(source_order)) or len(set(source_order)) != len(
            source_order
        ):
            raise HouseholdModelDiagnosticError(
                "household model presence review sources must use sorted unique cameras"
            )
        source_keys = {
            (source.camera_id, source.media_admission_sha256): source for source in self.sources
        }
        if len(source_keys) != len(self.sources):
            raise HouseholdModelDiagnosticError(
                "household model presence review sources are duplicated"
            )
        if (
            not isinstance(self.items, tuple)
            or not 1 <= len(self.items) <= MAX_HOUSEHOLD_MODEL_REVIEW_ITEMS
            or any(not isinstance(item, HouseholdModelPresenceItem) for item in self.items)
        ):
            raise HouseholdModelDiagnosticError("household model presence review items are invalid")
        expected_order = tuple(
            sorted(
                self.items,
                key=lambda item: (
                    item.camera_id,
                    item.timestamp_seconds,
                    item.review_item_id,
                ),
            )
        )
        if self.items != expected_order:
            raise HouseholdModelDiagnosticError(
                "household model presence review items must be canonically sorted"
            )
        if len({item.review_item_id for item in self.items}) != len(self.items):
            raise HouseholdModelDiagnosticError(
                "household model presence review item IDs are duplicated"
            )
        if len({item.source_frame_sha256 for item in self.items}) != len(self.items):
            raise HouseholdModelDiagnosticError(
                "household model presence review frame hashes are duplicated"
            )
        if len(
            {
                (item.camera_id, item.media_admission_sha256, item.timestamp_seconds)
                for item in self.items
            }
        ) != len(self.items):
            raise HouseholdModelDiagnosticError(
                "household model presence review source timestamps are duplicated"
            )
        used_sources: set[tuple[str, str]] = set()
        for item in self.items:
            key = (item.camera_id, item.media_admission_sha256)
            source = source_keys.get(key)
            if source is None:
                raise HouseholdModelDiagnosticError(
                    "household model presence review item source is not declared"
                )
            if item.timestamp_seconds > source.source_duration_seconds + 1e-9:
                raise HouseholdModelDiagnosticError(
                    "household model presence review item exceeds source duration"
                )
            used_sources.add(key)
        if used_sources != set(source_keys):
            raise HouseholdModelDiagnosticError(
                "household model presence review contains an unused source"
            )
        if self.review_scope != "visible_person_presence_only":
            raise HouseholdModelDiagnosticError("household model presence review scope is invalid")
        for field, expected in (
            ("model_generated_labels", True),
            ("human_verified", False),
            ("ground_truth", False),
            ("approved_for_official_evaluation", False),
            ("selected_stills_only", True),
            ("clips_shared", False),
            ("audio_shared", False),
            ("identity_inference", False),
            ("biometric_processing", False),
            ("training_allowed", False),
            ("redistribution_allowed", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdModelDiagnosticError(
                    f"household model presence review {field} must be {str(expected).lower()}"
                )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    @property
    def present_frame_count(self) -> int:
        return sum(item.visible_person == "present" for item in self.items)

    @property
    def absent_frame_count(self) -> int:
        return sum(item.visible_person == "absent" for item in self.items)

    @property
    def unknown_frame_count(self) -> int:
        return sum(item.visible_person == "unknown" for item in self.items)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: (
                [item.to_dict() for item in self.sources]
                if field == "sources"
                else [item.to_dict() for item in self.items]
                if field == "items"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        if include_fingerprint:
            value["review_sha256"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "valid": True,
            "version": self.version,
            "review_id": self.review_id,
            "review_sha256": self.fingerprint,
            "review_model_id": self.review_model_id,
            "review_model_revision": self.review_model_revision,
            "review_media_transfer": self.review_media_transfer,
            "sampling_strategy": self.sampling_strategy,
            "source_count": len(self.sources),
            "reviewed_frame_count": len(self.items),
            "present_frame_count": self.present_frame_count,
            "absent_frame_count": self.absent_frame_count,
            "unknown_frame_count": self.unknown_frame_count,
            "model_generated_labels": True,
            "human_verified": False,
            "ground_truth": False,
            "approved_for_official_evaluation": False,
            "selected_stills_only": True,
            "clips_shared": False,
            "audio_shared": False,
            "identity_inference": False,
            "biometric_processing": False,
            "training_allowed": False,
            "redistribution_allowed": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelPresenceReview:
        fields = {*cls.__dataclass_fields__, "review_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household model presence review fields do not match v1"
            )
        if not isinstance(value["sources"], list) or not isinstance(value["items"], list):
            raise HouseholdModelDiagnosticError(
                "household model presence review arrays are invalid"
            )
        review_sha256 = value["review_sha256"]
        review = cls(
            **{
                key: (
                    tuple(HouseholdModelPresenceSource.from_dict(item) for item in value[key])
                    if key == "sources"
                    else tuple(HouseholdModelPresenceItem.from_dict(item) for item in value[key])
                    if key == "items"
                    else value[key]
                )
                for key in value
                if key != "review_sha256"
            }
        )
        if review_sha256 != review.fingerprint:
            raise HouseholdModelDiagnosticError(
                "household model presence review fingerprint is invalid"
            )
        return review

    @classmethod
    def from_json(cls, raw: str) -> HouseholdModelPresenceReview:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_MODEL_REVIEW_BYTES:
            raise HouseholdModelDiagnosticError(
                "household model presence review exceeds the 1 MiB limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdModelPresenceReview:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_HOUSEHOLD_MODEL_REVIEW_BYTES:
                raise HouseholdModelDiagnosticError(
                    "household model presence review exceeds the size limit"
                )
            raw = source.read_text(encoding="utf-8")
        except HouseholdModelDiagnosticError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdModelDiagnosticError(
                "household model presence review is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_MODEL_REVIEW_BYTES,
        )


@dataclass(frozen=True, slots=True)
class HouseholdModelDiagnosticPoint:
    confidence_threshold: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float | None
    recall: float | None
    f1: float | None
    specificity: float | None
    partial_positive_count: int
    partial_detected_count: int
    partial_visibility_recall: float | None

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
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "partial_positive_count",
            "partial_detected_count",
        ):
            _count(getattr(self, field), f"point {field}")
        if (
            self.partial_detected_count > self.partial_positive_count
            or self.partial_detected_count > self.true_positive
        ):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic partial detection count is inconsistent"
            )
        for field in (
            "precision",
            "recall",
            "f1",
            "specificity",
            "partial_visibility_recall",
        ):
            object.__setattr__(
                self,
                field,
                _optional_ratio(getattr(self, field), f"point {field}"),
            )
        expected_precision = _ratio(
            self.true_positive,
            self.true_positive + self.false_positive,
        )
        expected_recall = _ratio(
            self.true_positive,
            self.true_positive + self.false_negative,
        )
        expected_f1 = (
            None
            if expected_precision is None
            or expected_recall is None
            or expected_precision + expected_recall == 0.0
            else round(
                2.0 * expected_precision * expected_recall / (expected_precision + expected_recall),
                12,
            )
        )
        expected_specificity = _ratio(
            self.true_negative,
            self.true_negative + self.false_positive,
        )
        expected_partial_recall = _ratio(
            self.partial_detected_count,
            self.partial_positive_count,
        )
        if (
            self.precision != expected_precision
            or self.recall != expected_recall
            or self.f1 != expected_f1
            or self.specificity != expected_specificity
            or self.partial_visibility_recall != expected_partial_recall
        ):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic point metrics are inconsistent"
            )

    @property
    def evaluated_bin_count(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelDiagnosticPoint:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic point fields do not match v1"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdModelDiagnosticReport:
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
    partial_positive_bin_count: int
    points: tuple[HouseholdModelDiagnosticPoint, ...]
    version: int = 1
    label: str = "person"
    ground_truth_provenance: str = "model_generated_visual_review"
    diagnostic_only: bool = True
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
                "unsupported household model diagnostic report version"
            )
        _sha256(self.review_sha256, "review_sha256")
        for field in ("detector_adapter", "model_id", "model_revision"):
            _safe_id(getattr(self, field), field)
        _sha256(self.model_sha256, "model_sha256")
        if (
            not isinstance(self.camera_ids, tuple)
            or not 1 <= len(self.camera_ids) <= MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES
            or any(
                not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None
                for value in self.camera_ids
            )
            or self.camera_ids != tuple(sorted(self.camera_ids))
            or len(set(self.camera_ids)) != len(self.camera_ids)
        ):
            raise HouseholdModelDiagnosticError("household model diagnostic camera IDs are invalid")
        for field in ("media_admission_sha256s", "detector_scan_sha256s"):
            values = getattr(self, field)
            if (
                not isinstance(values, tuple)
                or len(values) != len(self.camera_ids)
                or any(
                    not isinstance(value, str) or _SHA256.fullmatch(value) is None
                    for value in values
                )
                or len(set(values)) != len(values)
            ):
                raise HouseholdModelDiagnosticError(
                    f"household model diagnostic {field} is invalid"
                )
        object.__setattr__(
            self,
            "source_confidence_threshold",
            _number(
                self.source_confidence_threshold,
                "source_confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "requested_sample_fps",
            _number(
                self.requested_sample_fps,
                "requested_sample_fps",
                minimum=0.01,
                maximum=10.0,
            ),
        )
        object.__setattr__(
            self,
            "bin_seconds",
            _number(self.bin_seconds, "bin_seconds", minimum=0.1, maximum=60.0),
        )
        if self.sampling_strategy not in HOUSEHOLD_MODEL_REVIEW_SAMPLING_STRATEGIES:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic sampling strategy is invalid"
            )
        for field in (
            "reviewed_frame_count",
            "evaluated_bin_count",
            "positive_bin_count",
            "negative_bin_count",
            "unknown_bin_count",
            "partial_positive_bin_count",
        ):
            _count(getattr(self, field), field)
        if self.reviewed_frame_count < 1 or self.evaluated_bin_count < 1:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic requires reviewed frames and evaluable bins"
            )
        if self.positive_bin_count + self.negative_bin_count != self.evaluated_bin_count:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic evaluated bin accounting is inconsistent"
            )
        if self.partial_positive_bin_count > self.positive_bin_count:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic partial bin accounting is inconsistent"
            )
        if self.evaluated_bin_count + self.unknown_bin_count > self.reviewed_frame_count:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic bin count exceeds reviewed frames"
            )
        if (
            not isinstance(self.points, tuple)
            or not 2 <= len(self.points) <= MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_POINTS
            or any(not isinstance(point, HouseholdModelDiagnosticPoint) for point in self.points)
        ):
            raise HouseholdModelDiagnosticError("household model diagnostic points are invalid")
        thresholds = tuple(point.confidence_threshold for point in self.points)
        if (
            thresholds != tuple(sorted(thresholds))
            or len(set(thresholds)) != len(thresholds)
            or thresholds[0] != self.source_confidence_threshold
        ):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic thresholds must be unique, ascending, "
                "and start at the source threshold"
            )
        previous_predicted_positive: int | None = None
        previous_true_positive: int | None = None
        previous_partial_detected: int | None = None
        for point in self.points:
            if point.evaluated_bin_count != self.evaluated_bin_count:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic point bin count is inconsistent"
                )
            if point.true_positive + point.false_negative != self.positive_bin_count:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic positive accounting is inconsistent"
                )
            if point.true_negative + point.false_positive != self.negative_bin_count:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic negative accounting is inconsistent"
                )
            if point.partial_positive_count != self.partial_positive_bin_count:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic partial accounting is inconsistent"
                )
            predicted_positive = point.true_positive + point.false_positive
            if (
                (
                    previous_predicted_positive is not None
                    and predicted_positive > previous_predicted_positive
                )
                or (
                    previous_true_positive is not None
                    and point.true_positive > previous_true_positive
                )
                or (
                    previous_partial_detected is not None
                    and point.partial_detected_count > previous_partial_detected
                )
            ):
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic counts are not monotonic"
                )
            previous_predicted_positive = predicted_positive
            previous_true_positive = point.true_positive
            previous_partial_detected = point.partial_detected_count
        if self.label != "person":
            raise HouseholdModelDiagnosticError(
                "household model diagnostic label must remain generic person"
            )
        if self.ground_truth_provenance != "model_generated_visual_review":
            raise HouseholdModelDiagnosticError("household model diagnostic provenance is invalid")
        for field, expected in (
            ("diagnostic_only", True),
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
                    f"household model diagnostic {field} must be {str(expected).lower()}"
                )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: (
                list(getattr(self, field))
                if field in {"media_admission_sha256s", "detector_scan_sha256s", "camera_ids"}
                else [point.to_dict() for point in self.points]
                if field == "points"
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
            "review_sha256": self.review_sha256,
            "source_count": len(self.camera_ids),
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
            "partial_positive_bin_count": self.partial_positive_bin_count,
            "points": [point.to_dict() for point in self.points],
            "diagnostic_only": True,
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
    def from_dict(cls, value: Any) -> HouseholdModelDiagnosticReport:
        fields = {*cls.__dataclass_fields__, "report_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic report fields do not match v1"
            )
        for field in (
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "camera_ids",
            "points",
        ):
            if not isinstance(value[field], list):
                raise HouseholdModelDiagnosticError(
                    f"household model diagnostic report {field} is invalid"
                )
        report_sha256 = value["report_sha256"]
        report = cls(
            **{
                key: (
                    tuple(HouseholdModelDiagnosticPoint.from_dict(item) for item in value[key])
                    if key == "points"
                    else tuple(value[key])
                    if key in {"media_admission_sha256s", "detector_scan_sha256s", "camera_ids"}
                    else value[key]
                )
                for key in value
                if key != "report_sha256"
            }
        )
        if report_sha256 != report.fingerprint:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic report fingerprint is invalid"
            )
        return report

    @classmethod
    def from_json(cls, raw: str) -> HouseholdModelDiagnosticReport:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_BYTES:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic report exceeds the 1 MiB limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdModelDiagnosticReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_BYTES:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic report exceeds the size limit"
                )
            raw = source.read_text(encoding="utf-8")
        except HouseholdModelDiagnosticError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic report is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_BYTES,
        )


@dataclass(frozen=True, slots=True)
class HouseholdModelDiagnosticPolicy:
    minimum_precision: float
    minimum_recall: float
    minimum_partial_visibility_recall: float
    require_partial_visibility_samples: bool = True

    def __post_init__(self) -> None:
        for field in (
            "minimum_precision",
            "minimum_recall",
            "minimum_partial_visibility_recall",
        ):
            object.__setattr__(
                self,
                field,
                _number(getattr(self, field), field, minimum=0.0, maximum=1.0),
            )
        if type(self.require_partial_visibility_samples) is not bool:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic partial visibility policy is invalid"
            )

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _validate_admissions(
    review: HouseholdModelPresenceReview,
    admissions: tuple[ExternalSourceAdmission, ...],
) -> dict[str, ExternalSourceAdmission]:
    if (
        not isinstance(admissions, tuple)
        or len(admissions) != len(review.sources)
        or any(not isinstance(item, ExternalSourceAdmission) for item in admissions)
    ):
        raise HouseholdModelDiagnosticError("household model diagnostic admissions are incomplete")
    by_hash = {admission.fingerprint: admission for admission in admissions}
    if len(by_hash) != len(admissions):
        raise HouseholdModelDiagnosticError("household model diagnostic admissions are duplicated")
    expected = {source.media_admission_sha256 for source in review.sources}
    if set(by_hash) != expected:
        raise HouseholdModelDiagnosticError(
            "household model diagnostic admissions do not match review sources"
        )
    for admission in admissions:
        decision = admission.admission
        if (
            admission.asset_kind != "media_asset"
            or decision.status != "verified_local"
            or decision.tier != "C"
            or decision.purpose != "model_evaluation"
            or not decision.requires_user_approval
            or not decision.user_approval_recorded
            or decision.git_allowed
            or decision.ci_allowed
            or decision.redistribution_allowed
            or decision.training_allowed
            or admission.privacy.private_household_data != "yes"
            or admission.labels.human_verified
            or admission.labels.ground_truth
        ):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic requires verified local-only Tier C media"
            )
    return by_hash


def _validate_scans(
    review: HouseholdModelPresenceReview,
    scans: tuple[AdmittedDetectorScan, ...],
) -> dict[tuple[str, str], AdmittedDetectorScan]:
    if (
        not isinstance(scans, tuple)
        or len(scans) != len(review.sources)
        or any(not isinstance(item, AdmittedDetectorScan) for item in scans)
    ):
        raise HouseholdModelDiagnosticError("household model diagnostic scans are incomplete")
    by_source = {(scan.result.camera_id, scan.source_admission_sha256): scan for scan in scans}
    if len(by_source) != len(scans):
        raise HouseholdModelDiagnosticError("household model diagnostic scans are duplicated")
    expected = {(source.camera_id, source.media_admission_sha256) for source in review.sources}
    if set(by_source) != expected:
        raise HouseholdModelDiagnosticError(
            "household model diagnostic scans do not match review sources"
        )
    first = scans[0]
    identity = (
        first.result.detector_adapter,
        first.result.model_id,
        first.result.model_revision,
        first.result.model_sha256,
        first.labels,
        first.confidence_threshold,
        first.requested_sample_fps,
    )
    for scan in scans:
        if (
            scan.result.detector_adapter,
            scan.result.model_id,
            scan.result.model_revision,
            scan.result.model_sha256,
            scan.labels,
            scan.confidence_threshold,
            scan.requested_sample_fps,
        ) != identity or scan.labels != ("person",):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic scans must share one person detector config"
            )
    source_by_key = {
        (source.camera_id, source.media_admission_sha256): source for source in review.sources
    }
    for key, scan in by_source.items():
        duration = source_by_key[key].source_duration_seconds
        if any(
            detection.timestamp_seconds > duration + 1e-9 for detection in scan.result.detections
        ):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic detection exceeds source duration"
            )
    return by_source


def _thresholds(
    values: tuple[float, ...],
    *,
    source_threshold: float,
) -> tuple[float, ...]:
    if (
        not isinstance(values, tuple)
        or not 2 <= len(values) <= MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_POINTS
    ):
        raise HouseholdModelDiagnosticError("household model diagnostic thresholds are invalid")
    result = tuple(
        _number(value, "threshold", minimum=source_threshold, maximum=1.0) for value in values
    )
    if (
        result != tuple(sorted(result))
        or len(set(result)) != len(result)
        or result[0] != source_threshold
    ):
        raise HouseholdModelDiagnosticError(
            "household model diagnostic thresholds must be unique, ascending, "
            "and start at the scan threshold"
        )
    return result


def evaluate_household_model_presence_diagnostic(
    review: HouseholdModelPresenceReview,
    *,
    media_admissions: tuple[ExternalSourceAdmission, ...],
    detector_scans: tuple[AdmittedDetectorScan, ...],
    thresholds: tuple[float, ...],
    bin_seconds: float,
) -> HouseholdModelDiagnosticReport:
    """Evaluate one detector identity against model-generated visible facts.

    Labels sharing one source/bin are collapsed.  Conflicting present/absent
    labels fail closed; a bin containing any ``unknown`` label is excluded and
    counted explicitly.  The detector score for an eligible bin is the maximum
    saved person confidence in that same bin, or no score when the admitted
    scan emitted nothing there.
    """

    if not isinstance(review, HouseholdModelPresenceReview):
        raise HouseholdModelDiagnosticError("household model diagnostic review is invalid")
    _validate_admissions(review, media_admissions)
    scans_by_source = _validate_scans(review, detector_scans)
    bin_seconds = _number(bin_seconds, "bin_seconds", minimum=0.1, maximum=60.0)
    first_scan = detector_scans[0]
    thresholds = _thresholds(
        thresholds,
        source_threshold=first_scan.confidence_threshold,
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

    scores: dict[tuple[str, str, int], float] = {}
    for (camera_id, admission_sha256), scan in scans_by_source.items():
        for detection in scan.result.detections:
            key = (
                camera_id,
                admission_sha256,
                int(detection.timestamp_seconds // bin_seconds),
            )
            scores[key] = max(scores.get(key, 0.0), detection.confidence)

    samples: list[tuple[str, bool, float | None]] = []
    unknown_bin_count = 0
    partial_positive_bin_count = 0
    for key in sorted(grouped_items):
        items = grouped_items[key]
        labels = {item.visible_person for item in items}
        if "present" in labels and "absent" in labels:
            raise HouseholdModelDiagnosticError(
                "household model presence review conflicts within one diagnostic bin"
            )
        if "unknown" in labels:
            unknown_bin_count += 1
            continue
        label = labels.pop()
        partial = label == "present" and any(item.visibility == "partial" for item in items)
        partial_positive_bin_count += int(partial)
        samples.append((label, partial, scores.get(key)))
    if not samples:
        raise HouseholdModelDiagnosticError("household model diagnostic contains no evaluable bins")

    positive_bin_count = sum(label == "present" for label, _partial, _score in samples)
    negative_bin_count = sum(label == "absent" for label, _partial, _score in samples)
    points: list[HouseholdModelDiagnosticPoint] = []
    for threshold in thresholds:
        true_positive = false_positive = false_negative = true_negative = 0
        partial_detected_count = 0
        for label, partial, score in samples:
            predicted = score is not None and score >= threshold
            if label == "present" and predicted:
                true_positive += 1
            elif label == "present":
                false_negative += 1
            elif predicted:
                false_positive += 1
            else:
                true_negative += 1
            if partial and predicted:
                partial_detected_count += 1
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = (
            None
            if precision is None or recall is None or precision + recall == 0.0
            else round(2.0 * precision * recall / (precision + recall), 12)
        )
        points.append(
            HouseholdModelDiagnosticPoint(
                confidence_threshold=threshold,
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                true_negative=true_negative,
                precision=precision,
                recall=recall,
                f1=f1,
                specificity=_ratio(true_negative, true_negative + false_positive),
                partial_positive_count=partial_positive_bin_count,
                partial_detected_count=partial_detected_count,
                partial_visibility_recall=_ratio(
                    partial_detected_count,
                    partial_positive_bin_count,
                ),
            )
        )

    ordered_sources = tuple(review.sources)
    ordered_scans = tuple(
        scans_by_source[(source.camera_id, source.media_admission_sha256)]
        for source in ordered_sources
    )
    return HouseholdModelDiagnosticReport(
        review_sha256=review.fingerprint,
        media_admission_sha256s=tuple(source.media_admission_sha256 for source in ordered_sources),
        detector_scan_sha256s=tuple(scan.fingerprint for scan in ordered_scans),
        camera_ids=tuple(source.camera_id for source in ordered_sources),
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
        negative_bin_count=negative_bin_count,
        unknown_bin_count=unknown_bin_count,
        partial_positive_bin_count=partial_positive_bin_count,
        points=tuple(points),
    )


def select_household_model_diagnostic_working_point(
    report: HouseholdModelDiagnosticReport,
    policy: HouseholdModelDiagnosticPolicy,
) -> tuple[HouseholdModelDiagnosticPoint | None, int]:
    if not isinstance(report, HouseholdModelDiagnosticReport) or not isinstance(
        policy, HouseholdModelDiagnosticPolicy
    ):
        raise HouseholdModelDiagnosticError("household model diagnostic selection input is invalid")
    if policy.require_partial_visibility_samples and report.partial_positive_bin_count == 0:
        return None, 0
    feasible = tuple(
        point
        for point in report.points
        if point.precision is not None
        and point.recall is not None
        and point.precision >= policy.minimum_precision
        and point.recall >= policy.minimum_recall
        and (
            (
                point.partial_visibility_recall is not None
                and point.partial_visibility_recall >= policy.minimum_partial_visibility_recall
            )
            or (
                point.partial_visibility_recall is None
                and not policy.require_partial_visibility_samples
            )
        )
    )
    if not feasible:
        return None, 0
    selected = max(
        feasible,
        key=lambda point: (
            point.recall if point.recall is not None else -1.0,
            point.partial_visibility_recall
            if point.partial_visibility_recall is not None
            else -1.0,
            point.precision if point.precision is not None else -1.0,
            point.f1 if point.f1 is not None else -1.0,
            point.specificity if point.specificity is not None else -1.0,
            point.confidence_threshold,
        ),
    )
    return selected, len(feasible)


def compare_household_model_presence_diagnostics(
    reports: tuple[HouseholdModelDiagnosticReport, ...],
) -> dict[str, object]:
    if (
        not isinstance(reports, tuple)
        or len(reports) < 2
        or any(not isinstance(report, HouseholdModelDiagnosticReport) for report in reports)
    ):
        raise HouseholdModelDiagnosticError(
            "household model diagnostic comparison requires at least two reports"
        )
    first = reports[0]
    comparable = (
        first.review_sha256,
        first.media_admission_sha256s,
        first.camera_ids,
        first.source_confidence_threshold,
        first.requested_sample_fps,
        first.bin_seconds,
        first.sampling_strategy,
        first.reviewed_frame_count,
        first.evaluated_bin_count,
        first.positive_bin_count,
        first.negative_bin_count,
        first.unknown_bin_count,
        first.partial_positive_bin_count,
        tuple(point.confidence_threshold for point in first.points),
    )
    identities: set[tuple[str, str, str]] = set()
    for report in reports:
        if (
            report.review_sha256,
            report.media_admission_sha256s,
            report.camera_ids,
            report.source_confidence_threshold,
            report.requested_sample_fps,
            report.bin_seconds,
            report.sampling_strategy,
            report.reviewed_frame_count,
            report.evaluated_bin_count,
            report.positive_bin_count,
            report.negative_bin_count,
            report.unknown_bin_count,
            report.partial_positive_bin_count,
            tuple(point.confidence_threshold for point in report.points),
        ) != comparable:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic reports do not share one review and grid"
            )
        identity = (report.model_id, report.model_revision, report.model_sha256)
        if identity in identities:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic comparison repeats a model identity"
            )
        identities.add(identity)
    ordered = tuple(sorted(reports, key=lambda report: (report.model_id, report.model_revision)))
    return {
        "valid": True,
        "version": 1,
        "review_sha256": first.review_sha256,
        "source_count": len(first.camera_ids),
        "reviewed_frame_count": first.reviewed_frame_count,
        "evaluated_bin_count": first.evaluated_bin_count,
        "positive_bin_count": first.positive_bin_count,
        "negative_bin_count": first.negative_bin_count,
        "unknown_bin_count": first.unknown_bin_count,
        "partial_positive_bin_count": first.partial_positive_bin_count,
        "bin_seconds": first.bin_seconds,
        "sampling_strategy": first.sampling_strategy,
        "same_review_media_grid_and_thresholds_verified": True,
        "reports": [
            {
                "report_sha256": report.fingerprint,
                "model_id": report.model_id,
                "model_revision": report.model_revision,
                "model_sha256": report.model_sha256,
                "points": [point.to_dict() for point in report.points],
            }
            for report in ordered
        ],
        "diagnostic_only": True,
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
        "shadow_preference_selected": False,
        "deployment_approved": False,
        "household_configuration_changed": False,
        "dataset_specific_only": True,
        "subject_identity_inference": False,
        "source_media_included": False,
        "paths_disclosed": False,
    }


def _optional_signed_delta(
    baseline: float | None,
    current: float | None,
) -> float | None:
    if baseline is None or current is None:
        return None
    return round(current - baseline, 12)


@dataclass(frozen=True, slots=True)
class HouseholdModelDiagnosticDriftPoint:
    confidence_threshold: float
    baseline_precision: float | None
    current_precision: float | None
    precision_delta: float | None
    baseline_recall: float | None
    current_recall: float | None
    recall_delta: float | None
    baseline_specificity: float | None
    current_specificity: float | None
    specificity_delta: float | None
    baseline_partial_visibility_recall: float | None
    current_partial_visibility_recall: float | None
    partial_visibility_recall_delta: float | None
    precision_drop_exceeded: bool
    recall_drop_exceeded: bool
    specificity_drop_exceeded: bool
    partial_visibility_recall_drop_exceeded: bool
    metric_availability_changed: bool
    attention_required: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence_threshold",
            _number(
                self.confidence_threshold,
                "drift point confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        metric_pairs = (
            ("precision", self.baseline_precision, self.current_precision),
            ("recall", self.baseline_recall, self.current_recall),
            ("specificity", self.baseline_specificity, self.current_specificity),
            (
                "partial_visibility_recall",
                self.baseline_partial_visibility_recall,
                self.current_partial_visibility_recall,
            ),
        )
        for name, baseline, current in metric_pairs:
            object.__setattr__(
                self,
                f"baseline_{name}",
                _optional_ratio(baseline, f"drift point baseline_{name}"),
            )
            object.__setattr__(
                self,
                f"current_{name}",
                _optional_ratio(current, f"drift point current_{name}"),
            )
            delta_field = f"{name}_delta"
            delta = getattr(self, delta_field)
            if delta is not None:
                delta = _number(
                    delta,
                    f"drift point {delta_field}",
                    minimum=-1.0,
                    maximum=1.0,
                )
                object.__setattr__(self, delta_field, delta)
            expected_delta = _optional_signed_delta(baseline, current)
            if delta != expected_delta:
                raise HouseholdModelDiagnosticError(
                    f"household model diagnostic drift point {delta_field} is inconsistent"
                )
        boolean_fields = (
            "precision_drop_exceeded",
            "recall_drop_exceeded",
            "specificity_drop_exceeded",
            "partial_visibility_recall_drop_exceeded",
            "metric_availability_changed",
            "attention_required",
        )
        if any(type(getattr(self, field)) is not bool for field in boolean_fields):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift point flags are invalid"
            )
        expected_availability_change = any(
            (baseline is None) != (current is None) for _name, baseline, current in metric_pairs
        )
        if self.metric_availability_changed is not expected_availability_change:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift metric availability flag is inconsistent"
            )
        expected_attention = self.metric_availability_changed or any(
            getattr(self, field)
            for field in (
                "precision_drop_exceeded",
                "recall_drop_exceeded",
                "specificity_drop_exceeded",
                "partial_visibility_recall_drop_exceeded",
            )
        )
        if self.attention_required is not expected_attention:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift attention flag is inconsistent"
            )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelDiagnosticDriftPoint:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift point fields do not match v1"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdModelDiagnosticDriftPolicy:
    maximum_precision_drop: float
    maximum_recall_drop: float
    maximum_specificity_drop: float
    maximum_partial_visibility_recall_drop: float
    require_partial_visibility_samples: bool = True

    def __post_init__(self) -> None:
        for field in (
            "maximum_precision_drop",
            "maximum_recall_drop",
            "maximum_specificity_drop",
            "maximum_partial_visibility_recall_drop",
        ):
            object.__setattr__(
                self,
                field,
                _number(getattr(self, field), field, minimum=0.0, maximum=1.0),
            )
        if type(self.require_partial_visibility_samples) is not bool:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift partial visibility policy is invalid"
            )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelDiagnosticDriftPolicy:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift policy fields do not match v1"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdModelDiagnosticDriftReport:
    baseline_report_sha256: str
    current_report_sha256: str
    baseline_review_sha256: str
    current_review_sha256: str
    baseline_media_admission_sha256s: tuple[str, ...]
    current_media_admission_sha256s: tuple[str, ...]
    baseline_detector_scan_sha256s: tuple[str, ...]
    current_detector_scan_sha256s: tuple[str, ...]
    camera_ids: tuple[str, ...]
    detector_adapter: str
    model_id: str
    model_revision: str
    model_sha256: str
    source_confidence_threshold: float
    requested_sample_fps: float
    bin_seconds: float
    sampling_strategy: str
    policy: HouseholdModelDiagnosticDriftPolicy
    baseline_reviewed_frame_count: int
    baseline_evaluated_bin_count: int
    baseline_positive_bin_count: int
    baseline_negative_bin_count: int
    baseline_unknown_bin_count: int
    baseline_partial_positive_bin_count: int
    current_reviewed_frame_count: int
    current_evaluated_bin_count: int
    current_positive_bin_count: int
    current_negative_bin_count: int
    current_unknown_bin_count: int
    current_partial_positive_bin_count: int
    points: tuple[HouseholdModelDiagnosticDriftPoint, ...]
    attention_threshold_count: int
    metric_availability_change_count: int
    attention_required: bool
    version: int = 1
    label: str = "person"
    ground_truth_provenance: str = "model_generated_visual_review"
    diagnostic_only: bool = True
    cross_batch_comparison: bool = True
    same_detector_configuration_verified: bool = True
    same_camera_set_verified: bool = True
    same_sampling_and_threshold_grid_verified: bool = True
    disjoint_source_media_verified: bool = True
    disjoint_detector_scans_verified: bool = True
    derived_without_model_rerun: bool = True
    model_generated_labels: bool = True
    human_verified: bool = False
    ground_truth: bool = False
    official_accuracy_claim: bool = False
    regression_confirmed: bool = False
    frame_level_alignment_evaluated: bool = False
    candidate_trigger_coverage_evaluated: bool = False
    box_localization_accuracy_evaluated: bool = False
    danger_recognition_accuracy_evaluated: bool = False
    risk_accuracy_evaluated: bool = False
    winner_selected: bool = False
    shadow_preference_selected: bool = False
    deployment_approved: bool = False
    household_configuration_changed: bool = False
    dataset_specific_only: bool = True
    subject_identity_inference: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HouseholdModelDiagnosticError(
                "unsupported household model diagnostic drift report version"
            )
        for field in (
            "baseline_report_sha256",
            "current_report_sha256",
            "baseline_review_sha256",
            "current_review_sha256",
            "model_sha256",
        ):
            _sha256(getattr(self, field), f"drift {field}")
        if self.baseline_report_sha256 == self.current_report_sha256:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift reports must be distinct"
            )
        if self.baseline_review_sha256 == self.current_review_sha256:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift reviews must be distinct"
            )
        for field in ("detector_adapter", "model_id", "model_revision"):
            _safe_id(getattr(self, field), f"drift {field}")
        if (
            not isinstance(self.camera_ids, tuple)
            or not 1 <= len(self.camera_ids) <= MAX_HOUSEHOLD_MODEL_REVIEW_SOURCES
            or any(
                not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None
                for value in self.camera_ids
            )
            or self.camera_ids != tuple(sorted(self.camera_ids))
            or len(set(self.camera_ids)) != len(self.camera_ids)
        ):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift camera IDs are invalid"
            )
        hash_tuple_fields = (
            "baseline_media_admission_sha256s",
            "current_media_admission_sha256s",
            "baseline_detector_scan_sha256s",
            "current_detector_scan_sha256s",
        )
        for field in hash_tuple_fields:
            values = getattr(self, field)
            if (
                not isinstance(values, tuple)
                or len(values) != len(self.camera_ids)
                or len(set(values)) != len(values)
                or any(
                    not isinstance(value, str) or _SHA256.fullmatch(value) is None
                    for value in values
                )
            ):
                raise HouseholdModelDiagnosticError(
                    f"household model diagnostic drift {field} is invalid"
                )
        if set(self.baseline_media_admission_sha256s) & set(self.current_media_admission_sha256s):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift source media must be disjoint"
            )
        if set(self.baseline_detector_scan_sha256s) & set(self.current_detector_scan_sha256s):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift detector scans must be disjoint"
            )
        object.__setattr__(
            self,
            "source_confidence_threshold",
            _number(
                self.source_confidence_threshold,
                "drift source_confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "requested_sample_fps",
            _number(
                self.requested_sample_fps,
                "drift requested_sample_fps",
                minimum=0.01,
                maximum=10.0,
            ),
        )
        object.__setattr__(
            self,
            "bin_seconds",
            _number(self.bin_seconds, "drift bin_seconds", minimum=0.1, maximum=60.0),
        )
        if self.sampling_strategy not in HOUSEHOLD_MODEL_REVIEW_SAMPLING_STRATEGIES:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift sampling strategy is invalid"
            )
        if not isinstance(self.policy, HouseholdModelDiagnosticDriftPolicy):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift policy is invalid"
            )
        for prefix in ("baseline", "current"):
            for suffix in (
                "reviewed_frame_count",
                "evaluated_bin_count",
                "positive_bin_count",
                "negative_bin_count",
                "unknown_bin_count",
                "partial_positive_bin_count",
            ):
                _count(getattr(self, f"{prefix}_{suffix}"), f"drift {prefix}_{suffix}")
            reviewed = getattr(self, f"{prefix}_reviewed_frame_count")
            evaluated = getattr(self, f"{prefix}_evaluated_bin_count")
            positive = getattr(self, f"{prefix}_positive_bin_count")
            negative = getattr(self, f"{prefix}_negative_bin_count")
            unknown = getattr(self, f"{prefix}_unknown_bin_count")
            partial = getattr(self, f"{prefix}_partial_positive_bin_count")
            if reviewed < 1 or evaluated < 1 or positive < 1 or negative < 1:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic drift requires positive and negative bins "
                    "in both batches"
                )
            if positive + negative != evaluated or evaluated + unknown > reviewed:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic drift batch accounting is inconsistent"
                )
            if partial > positive:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic drift partial accounting is inconsistent"
                )
            if self.policy.require_partial_visibility_samples and partial < 1:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic drift requires partial visibility samples"
                )
        if (
            not isinstance(self.points, tuple)
            or not 2 <= len(self.points) <= MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_POINTS
            or any(
                not isinstance(point, HouseholdModelDiagnosticDriftPoint) for point in self.points
            )
        ):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift points are invalid"
            )
        thresholds = tuple(point.confidence_threshold for point in self.points)
        if (
            thresholds != tuple(sorted(thresholds))
            or len(set(thresholds)) != len(thresholds)
            or thresholds[0] != self.source_confidence_threshold
        ):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift thresholds are invalid"
            )
        tolerances = {
            "precision": self.policy.maximum_precision_drop,
            "recall": self.policy.maximum_recall_drop,
            "specificity": self.policy.maximum_specificity_drop,
            "partial_visibility_recall": (self.policy.maximum_partial_visibility_recall_drop),
        }
        for point in self.points:
            for metric, tolerance in tolerances.items():
                delta = getattr(point, f"{metric}_delta")
                expected = delta is not None and delta < -tolerance
                if getattr(point, f"{metric}_drop_exceeded") is not expected:
                    raise HouseholdModelDiagnosticError(
                        f"household model diagnostic drift {metric} drop flag is inconsistent"
                    )
        _count(self.attention_threshold_count, "drift attention_threshold_count", maximum=32)
        _count(
            self.metric_availability_change_count,
            "drift metric_availability_change_count",
            maximum=32,
        )
        expected_attention_count = sum(point.attention_required for point in self.points)
        expected_availability_count = sum(
            point.metric_availability_changed for point in self.points
        )
        if self.attention_threshold_count != expected_attention_count:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift attention count is inconsistent"
            )
        if self.metric_availability_change_count != expected_availability_count:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift metric availability count is inconsistent"
            )
        if type(self.attention_required) is not bool or self.attention_required is not (
            expected_attention_count > 0
        ):
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift report attention flag is inconsistent"
            )
        if self.label != "person":
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift label must remain generic person"
            )
        if self.ground_truth_provenance != "model_generated_visual_review":
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift provenance is invalid"
            )
        for field, expected in (
            ("diagnostic_only", True),
            ("cross_batch_comparison", True),
            ("same_detector_configuration_verified", True),
            ("same_camera_set_verified", True),
            ("same_sampling_and_threshold_grid_verified", True),
            ("disjoint_source_media_verified", True),
            ("disjoint_detector_scans_verified", True),
            ("derived_without_model_rerun", True),
            ("model_generated_labels", True),
            ("human_verified", False),
            ("ground_truth", False),
            ("official_accuracy_claim", False),
            ("regression_confirmed", False),
            ("frame_level_alignment_evaluated", False),
            ("candidate_trigger_coverage_evaluated", False),
            ("box_localization_accuracy_evaluated", False),
            ("danger_recognition_accuracy_evaluated", False),
            ("risk_accuracy_evaluated", False),
            ("winner_selected", False),
            ("shadow_preference_selected", False),
            ("deployment_approved", False),
            ("household_configuration_changed", False),
            ("dataset_specific_only", True),
            ("subject_identity_inference", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdModelDiagnosticError(
                    f"household model diagnostic drift {field} must be {str(expected).lower()}"
                )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        tuple_fields = {
            "baseline_media_admission_sha256s",
            "current_media_admission_sha256s",
            "baseline_detector_scan_sha256s",
            "current_detector_scan_sha256s",
            "camera_ids",
        }
        value: dict[str, object] = {
            field: (
                list(getattr(self, field))
                if field in tuple_fields
                else [point.to_dict() for point in self.points]
                if field == "points"
                else self.policy.to_dict()
                if field == "policy"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        if include_fingerprint:
            value["drift_report_sha256"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def to_public_dict(self) -> dict[str, object]:
        hidden = {
            "baseline_media_admission_sha256s",
            "current_media_admission_sha256s",
            "baseline_detector_scan_sha256s",
            "current_detector_scan_sha256s",
            "camera_ids",
        }
        value = {
            key: item
            for key, item in self.to_dict(include_fingerprint=False).items()
            if key not in hidden
        }
        value["valid"] = True
        value["drift_report_sha256"] = self.fingerprint
        value["source_count"] = len(self.camera_ids)
        return value

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdModelDiagnosticDriftReport:
        fields = {*cls.__dataclass_fields__, "drift_report_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift report fields do not match v1"
            )
        tuple_fields = {
            "baseline_media_admission_sha256s",
            "current_media_admission_sha256s",
            "baseline_detector_scan_sha256s",
            "current_detector_scan_sha256s",
            "camera_ids",
        }
        for field in (*tuple_fields, "points"):
            if not isinstance(value[field], list):
                raise HouseholdModelDiagnosticError(
                    f"household model diagnostic drift report {field} is invalid"
                )
        report_sha256 = value["drift_report_sha256"]
        report = cls(
            **{
                key: (
                    tuple(HouseholdModelDiagnosticDriftPoint.from_dict(item) for item in value[key])
                    if key == "points"
                    else HouseholdModelDiagnosticDriftPolicy.from_dict(value[key])
                    if key == "policy"
                    else tuple(value[key])
                    if key in tuple_fields
                    else value[key]
                )
                for key in value
                if key != "drift_report_sha256"
            }
        )
        if report_sha256 != report.fingerprint:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift report fingerprint is invalid"
            )
        return report

    @classmethod
    def from_json(cls, raw: str) -> HouseholdModelDiagnosticDriftReport:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_DRIFT_BYTES:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift report exceeds the 1 MiB limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdModelDiagnosticDriftReport:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_DRIFT_BYTES:
                raise HouseholdModelDiagnosticError(
                    "household model diagnostic drift report exceeds the size limit"
                )
            raw = source.read_text(encoding="utf-8")
        except HouseholdModelDiagnosticError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdModelDiagnosticError(
                "household model diagnostic drift report is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_MODEL_DIAGNOSTIC_DRIFT_BYTES,
        )


def compare_household_model_diagnostic_batches(
    baseline: HouseholdModelDiagnosticReport,
    current: HouseholdModelDiagnosticReport,
    policy: HouseholdModelDiagnosticDriftPolicy,
) -> HouseholdModelDiagnosticDriftReport:
    if (
        not isinstance(baseline, HouseholdModelDiagnosticReport)
        or not isinstance(current, HouseholdModelDiagnosticReport)
        or not isinstance(policy, HouseholdModelDiagnosticDriftPolicy)
    ):
        raise HouseholdModelDiagnosticError(
            "household model diagnostic drift comparison input is invalid"
        )
    if baseline.fingerprint == current.fingerprint:
        raise HouseholdModelDiagnosticError(
            "household model diagnostic drift reports must be distinct"
        )
    if baseline.review_sha256 == current.review_sha256:
        raise HouseholdModelDiagnosticError(
            "household model diagnostic drift reviews must be distinct"
        )
    if set(baseline.media_admission_sha256s) & set(current.media_admission_sha256s):
        raise HouseholdModelDiagnosticError(
            "household model diagnostic drift source media must be disjoint"
        )
    if set(baseline.detector_scan_sha256s) & set(current.detector_scan_sha256s):
        raise HouseholdModelDiagnosticError(
            "household model diagnostic drift detector scans must be disjoint"
        )
    baseline_identity = (
        baseline.detector_adapter,
        baseline.model_id,
        baseline.model_revision,
        baseline.model_sha256,
    )
    current_identity = (
        current.detector_adapter,
        current.model_id,
        current.model_revision,
        current.model_sha256,
    )
    if baseline_identity != current_identity:
        raise HouseholdModelDiagnosticError(
            "household model diagnostic drift requires one detector configuration"
        )
    if baseline.camera_ids != current.camera_ids:
        raise HouseholdModelDiagnosticError(
            "household model diagnostic drift requires the same camera set"
        )
    baseline_grid = (
        baseline.source_confidence_threshold,
        baseline.requested_sample_fps,
        baseline.bin_seconds,
        baseline.sampling_strategy,
        tuple(point.confidence_threshold for point in baseline.points),
    )
    current_grid = (
        current.source_confidence_threshold,
        current.requested_sample_fps,
        current.bin_seconds,
        current.sampling_strategy,
        tuple(point.confidence_threshold for point in current.points),
    )
    if baseline_grid != current_grid:
        raise HouseholdModelDiagnosticError(
            "household model diagnostic drift requires the same sampling and threshold grid"
        )
    tolerances = {
        "precision": policy.maximum_precision_drop,
        "recall": policy.maximum_recall_drop,
        "specificity": policy.maximum_specificity_drop,
        "partial_visibility_recall": policy.maximum_partial_visibility_recall_drop,
    }
    points: list[HouseholdModelDiagnosticDriftPoint] = []
    for baseline_point, current_point in zip(baseline.points, current.points, strict=True):
        values: dict[str, object] = {
            "confidence_threshold": baseline_point.confidence_threshold,
        }
        availability_changed = False
        drop_flags: list[bool] = []
        for metric, tolerance in tolerances.items():
            baseline_value = getattr(baseline_point, metric)
            current_value = getattr(current_point, metric)
            delta = _optional_signed_delta(baseline_value, current_value)
            drop_exceeded = delta is not None and delta < -tolerance
            values[f"baseline_{metric}"] = baseline_value
            values[f"current_{metric}"] = current_value
            values[f"{metric}_delta"] = delta
            values[f"{metric}_drop_exceeded"] = drop_exceeded
            availability_changed = availability_changed or (
                (baseline_value is None) != (current_value is None)
            )
            drop_flags.append(drop_exceeded)
        values["metric_availability_changed"] = availability_changed
        values["attention_required"] = availability_changed or any(drop_flags)
        points.append(HouseholdModelDiagnosticDriftPoint(**values))
    attention_count = sum(point.attention_required for point in points)
    availability_count = sum(point.metric_availability_changed for point in points)
    return HouseholdModelDiagnosticDriftReport(
        baseline_report_sha256=baseline.fingerprint,
        current_report_sha256=current.fingerprint,
        baseline_review_sha256=baseline.review_sha256,
        current_review_sha256=current.review_sha256,
        baseline_media_admission_sha256s=baseline.media_admission_sha256s,
        current_media_admission_sha256s=current.media_admission_sha256s,
        baseline_detector_scan_sha256s=baseline.detector_scan_sha256s,
        current_detector_scan_sha256s=current.detector_scan_sha256s,
        camera_ids=baseline.camera_ids,
        detector_adapter=baseline.detector_adapter,
        model_id=baseline.model_id,
        model_revision=baseline.model_revision,
        model_sha256=baseline.model_sha256,
        source_confidence_threshold=baseline.source_confidence_threshold,
        requested_sample_fps=baseline.requested_sample_fps,
        bin_seconds=baseline.bin_seconds,
        sampling_strategy=baseline.sampling_strategy,
        policy=policy,
        baseline_reviewed_frame_count=baseline.reviewed_frame_count,
        baseline_evaluated_bin_count=baseline.evaluated_bin_count,
        baseline_positive_bin_count=baseline.positive_bin_count,
        baseline_negative_bin_count=baseline.negative_bin_count,
        baseline_unknown_bin_count=baseline.unknown_bin_count,
        baseline_partial_positive_bin_count=baseline.partial_positive_bin_count,
        current_reviewed_frame_count=current.reviewed_frame_count,
        current_evaluated_bin_count=current.evaluated_bin_count,
        current_positive_bin_count=current.positive_bin_count,
        current_negative_bin_count=current.negative_bin_count,
        current_unknown_bin_count=current.unknown_bin_count,
        current_partial_positive_bin_count=current.partial_positive_bin_count,
        points=tuple(points),
        attention_threshold_count=attention_count,
        metric_availability_change_count=availability_count,
        attention_required=attention_count > 0,
    )
