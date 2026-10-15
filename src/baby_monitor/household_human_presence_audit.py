"""Blind single-human frame audits for household detector/Pose diagnostics.

The browser review format is an intentionally narrow local hand-off: it binds
uniformly sampled stills without exposing detector, Pose, fusion, or existing
model-review outputs.  Import turns that hand-off into a strict, path-free
artifact and verifies every immutable item against the original selection
manifest.  The resulting labels are a useful independent human *reference*,
but one reviewer and isolated frames do not establish temporal episode ground
truth, official accuracy, or deployment authority.

Evaluation replays the already-frozen detector/Pose profile over the exact
admitted scans bound by a temporal holdout attestation.  It never reads media,
loads a model, accesses the network, changes household configuration, or
selects a production winner.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .detector_protocol import AdmittedDetectorScan
from .household_model_diagnostic import (
    HouseholdModelDiagnosticError,
    HouseholdModelPresenceReview,
    _canonical_sha256,
    _count,
    _number,
    _optional_ratio,
    _ratio,
    _safe_id,
    _sha256,
    _strict_json,
    _write_exclusive,
)
from .household_model_temporal_diagnostic import _f1
from .household_pose_fusion_diagnostic import (
    HouseholdPoseFusionCameraProfile,
    household_pose_fusion_bin_prediction,
)
from .household_pose_fusion_holdout import (
    HouseholdPoseFusionTemporalHoldoutAttestation,
)
from .pose_protocol import AdmittedPoseScan
from .source_admission import ExternalSourceAdmission

MAX_HOUSEHOLD_HUMAN_AUDIT_INPUT_BYTES = 1024 * 1024
MAX_HOUSEHOLD_HUMAN_AUDIT_BYTES = 2 * 1024 * 1024
MAX_HOUSEHOLD_HUMAN_AUDIT_REPORT_BYTES = 2 * 1024 * 1024
MAX_HOUSEHOLD_HUMAN_AUDIT_ITEMS = 4096
MAX_HOUSEHOLD_HUMAN_AUDIT_CAMERAS = 32
HUMAN_PRESENCE_LABELS = ("absent", "present", "unknown")

_RAW_MANIFEST_FIELDS = {
    "version",
    "audit_type",
    "audit_id",
    "created_at_utc",
    "source_selection_manifest_sha256",
    "sampling",
    "privacy",
    "claims",
    "expected_download_filename",
    "items",
    "audit_manifest_sha256",
}
_RAW_SAMPLING_FIELDS = {
    "source",
    "camera_count",
    "frames_per_camera",
    "frame_count",
    "within_camera_strategy",
    "presentation_order",
}
_RAW_PRIVACY = {
    "local_only": True,
    "network_requests_required": False,
    "clips_included": False,
    "audio_included": False,
    "identity_requested": False,
    "model_outputs_disclosed": False,
    "existing_model_labels_disclosed": False,
    "fusion_profile_disclosed": False,
}
_RAW_CLAIMS = {
    "human_review_completed": False,
    "human_verified": False,
    "ground_truth": False,
    "official_accuracy_claim": False,
    "deployment_change_authorized": False,
}
_RAW_RESULT_FIELDS = {
    "version",
    "review_type",
    "audit_id",
    "completed_at_utc",
    "source_selection_manifest_sha256",
    "audit_manifest_sha256",
    "human_review_completed",
    "human_verified",
    "labels_created_without_model_outputs",
    "ground_truth",
    "official_accuracy_claim",
    "deployment_change_authorized",
    "items",
}
_RAW_BINDING_ITEM_FIELDS = {
    "review_item_id",
    "source_review_id",
    "camera_id",
    "media_admission_sha256",
    "bin_index",
    "timestamp_seconds",
    "source_frame_sha256",
}
_SELECTION_MANIFEST_FIELDS = {
    "version",
    "batch_id",
    "profile_sha256",
    "profile_frozen_before_capture",
    "source_count",
    "frame_count",
    "max_frames",
    "bin_seconds",
    "sampling_strategy",
    "model_generated_selection",
    "consent_recorded",
    "selected_stills_only",
    "clips_shared",
    "audio_shared",
    "identity_inference",
    "biometric_processing",
    "training_allowed",
    "redistribution_allowed",
    "human_verified",
    "ground_truth",
    "production_remote_vlm_changed",
    "items",
    "selection_manifest_sha256",
}
_SELECTION_ITEM_FIELDS = {
    "review_id",
    "camera_id",
    "media_admission_sha256",
    "detector_scan_sha256",
    "pose_scan_sha256",
    "timestamp_seconds",
    "bin_index",
    "source_frame_sha256",
    "frame_filename",
    "selection_source",
    "detector_max_confidence_nearby",
    "pose_max_confidence_nearby",
    "sheet_id",
    "sheet_panel_index",
}


def _read_json_artifact(
    path: str | Path,
    *,
    context: str,
    limit: int,
) -> tuple[bytes, Any]:
    source = Path(path).expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        size = source.stat().st_size
        if not 0 < size <= limit:
            raise HouseholdModelDiagnosticError(f"{context} exceeds the size limit")
        raw = source.read_bytes()
        text = raw.decode("utf-8")
    except HouseholdModelDiagnosticError:
        raise
    except (OSError, UnicodeError) as exc:
        raise HouseholdModelDiagnosticError(f"{context} is missing or unsafe") from exc
    return raw, _strict_json(text)


def _exact_fields(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise HouseholdModelDiagnosticError(f"{context} fields do not match v1")
    return value


def _utc_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not 20 <= len(value) <= 40 or not value.endswith("Z"):
        raise HouseholdModelDiagnosticError(f"household human audit {field} is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise HouseholdModelDiagnosticError(f"household human audit {field} is invalid") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise HouseholdModelDiagnosticError(f"household human audit {field} is invalid")
    return parsed


def _canonical_ascii_sha256(value: object, *, trailing_newline: bool = False) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if trailing_newline:
        payload += b"\n"
    return hashlib.sha256(payload).hexdigest()


def _fixed_boolean_fields(value: object, expected: dict[str, bool], context: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise HouseholdModelDiagnosticError(f"{context} fields do not match v1")
    for field, expected_value in expected.items():
        if type(value[field]) is not bool or value[field] is not expected_value:
            raise HouseholdModelDiagnosticError(
                f"{context} {field} must be {str(expected_value).lower()}"
            )


@dataclass(frozen=True, slots=True)
class _HouseholdHumanAuditBindingItem:
    review_item_id: str
    source_review_id: str
    camera_id: str
    media_admission_sha256: str
    bin_index: int
    timestamp_seconds: float
    source_frame_sha256: str

    def __post_init__(self) -> None:
        for field in ("review_item_id", "source_review_id", "camera_id"):
            _safe_id(getattr(self, field), f"human audit {field}")
        _sha256(self.media_admission_sha256, "human audit media_admission_sha256")
        _sha256(self.source_frame_sha256, "human audit source_frame_sha256")
        if type(self.bin_index) is not int or not 0 <= self.bin_index <= 10_000_000:
            raise HouseholdModelDiagnosticError("household human audit bin_index is invalid")
        object.__setattr__(
            self,
            "timestamp_seconds",
            _number(
                self.timestamp_seconds,
                "human audit timestamp_seconds",
                minimum=0.0,
                maximum=86_400.0,
            ),
        )

    @classmethod
    def from_dict(cls, value: Any) -> _HouseholdHumanAuditBindingItem:
        _exact_fields(value, _RAW_BINDING_ITEM_FIELDS, "household human audit binding item")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdHumanPresenceAuditItem:
    review_item_id: str
    source_review_id: str
    camera_id: str
    media_admission_sha256: str
    bin_index: int
    timestamp_seconds: float
    source_frame_sha256: str
    visible_person: str

    def __post_init__(self) -> None:
        _HouseholdHumanAuditBindingItem(
            review_item_id=self.review_item_id,
            source_review_id=self.source_review_id,
            camera_id=self.camera_id,
            media_admission_sha256=self.media_admission_sha256,
            bin_index=self.bin_index,
            timestamp_seconds=self.timestamp_seconds,
            source_frame_sha256=self.source_frame_sha256,
        )
        object.__setattr__(self, "timestamp_seconds", float(self.timestamp_seconds))
        if self.visible_person not in HUMAN_PRESENCE_LABELS:
            raise HouseholdModelDiagnosticError("household human audit visible_person is invalid")

    @property
    def binding(self) -> _HouseholdHumanAuditBindingItem:
        return _HouseholdHumanAuditBindingItem(
            **{
                field: getattr(self, field)
                for field in _HouseholdHumanAuditBindingItem.__dataclass_fields__
            }
        )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdHumanPresenceAuditItem:
        fields = {*_RAW_BINDING_ITEM_FIELDS, "visible_person"}
        _exact_fields(value, fields, "household human audit item")
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdHumanPresenceAudit:
    audit_id: str
    created_at_utc: str
    completed_at_utc: str
    source_batch_id: str
    source_selection_manifest_sha256: str
    source_profile_sha256: str
    audit_manifest_sha256: str
    result_file_sha256: str
    bin_seconds: float
    camera_count: int
    frames_per_camera: int
    sampling_source: str
    within_camera_strategy: str
    presentation_order: str
    items: tuple[HouseholdHumanPresenceAuditItem, ...]
    version: int = 1
    audit_type: str = "independent_human_frame_presence_audit"
    review_scope: str = "visible_person_presence_only"
    human_reviewer_count: int = 1
    human_review_completed: bool = True
    human_verified: bool = True
    labels_created_without_model_outputs: bool = True
    blind_ui_operator_attested: bool = True
    blindness_cryptographically_proven: bool = False
    single_human_reference: bool = True
    model_generated_labels: bool = False
    ground_truth: bool = False
    independent_human_ground_truth: bool = False
    temporal_episode_ground_truth: bool = False
    official_accuracy_claim: bool = False
    selected_stills_only: bool = True
    clips_included: bool = False
    audio_included: bool = False
    identity_inference: bool = False
    biometric_processing: bool = False
    training_allowed: bool = False
    redistribution_allowed: bool = False
    deployment_change_authorized: bool = False
    deployment_approved: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HouseholdModelDiagnosticError(
                "unsupported household human presence audit version"
            )
        for field in ("audit_id", "source_batch_id"):
            _safe_id(getattr(self, field), f"human audit {field}")
        for field in (
            "source_selection_manifest_sha256",
            "source_profile_sha256",
            "audit_manifest_sha256",
            "result_file_sha256",
        ):
            _sha256(getattr(self, field), f"human audit {field}")
        created = _utc_datetime(self.created_at_utc, "created_at_utc")
        completed = _utc_datetime(self.completed_at_utc, "completed_at_utc")
        if completed < created:
            raise HouseholdModelDiagnosticError(
                "household human audit completed_at_utc precedes creation"
            )
        object.__setattr__(
            self,
            "bin_seconds",
            _number(
                self.bin_seconds,
                "human audit bin_seconds",
                minimum=0.1,
                maximum=60.0,
            ),
        )
        if type(self.camera_count) is not int or not (
            1 <= self.camera_count <= MAX_HOUSEHOLD_HUMAN_AUDIT_CAMERAS
        ):
            raise HouseholdModelDiagnosticError("household human audit camera_count is invalid")
        if type(self.frames_per_camera) is not int or not (
            1 <= self.frames_per_camera <= MAX_HOUSEHOLD_HUMAN_AUDIT_ITEMS
        ):
            raise HouseholdModelDiagnosticError(
                "household human audit frames_per_camera is invalid"
            )
        for field, expected in (
            ("sampling_source", "uniform_holdout_only"),
            ("within_camera_strategy", "evenly_spaced_order_statistics"),
            ("presentation_order", "deterministic_blinded_shuffle"),
            ("audit_type", "independent_human_frame_presence_audit"),
            ("review_scope", "visible_person_presence_only"),
        ):
            if getattr(self, field) != expected:
                raise HouseholdModelDiagnosticError(f"household human audit {field} is invalid")
        if type(self.human_reviewer_count) is not int or self.human_reviewer_count != 1:
            raise HouseholdModelDiagnosticError(
                "household human audit human_reviewer_count must be one"
            )
        if (
            not isinstance(self.items, tuple)
            or not 1 <= len(self.items) <= MAX_HOUSEHOLD_HUMAN_AUDIT_ITEMS
            or any(not isinstance(item, HouseholdHumanPresenceAuditItem) for item in self.items)
        ):
            raise HouseholdModelDiagnosticError("household human audit items are invalid")
        if self.items != tuple(sorted(self.items, key=lambda item: item.review_item_id)):
            raise HouseholdModelDiagnosticError(
                "household human audit items must use canonical review order"
            )
        for values, message in (
            (
                [item.review_item_id for item in self.items],
                "review item IDs",
            ),
            (
                [item.source_review_id for item in self.items],
                "source review item IDs",
            ),
            (
                [item.source_frame_sha256 for item in self.items],
                "source frame hashes",
            ),
            (
                [
                    (item.camera_id, item.media_admission_sha256, item.bin_index)
                    for item in self.items
                ],
                "source bins",
            ),
        ):
            if len(set(values)) != len(values):
                raise HouseholdModelDiagnosticError(
                    f"household human audit {message} are duplicated"
                )
        camera_counts = Counter(item.camera_id for item in self.items)
        if (
            len(camera_counts) != self.camera_count
            or set(camera_counts.values()) != {self.frames_per_camera}
            or len(self.items) != self.camera_count * self.frames_per_camera
        ):
            raise HouseholdModelDiagnosticError(
                "household human audit per-camera sampling is inconsistent"
            )
        for item in self.items:
            if item.bin_index != int(item.timestamp_seconds // self.bin_seconds):
                raise HouseholdModelDiagnosticError(
                    "household human audit item does not match the declared bin grid"
                )
        for field, expected in (
            ("human_review_completed", True),
            ("human_verified", True),
            ("labels_created_without_model_outputs", True),
            ("blind_ui_operator_attested", True),
            ("blindness_cryptographically_proven", False),
            ("single_human_reference", True),
            ("model_generated_labels", False),
            ("ground_truth", False),
            ("independent_human_ground_truth", False),
            ("temporal_episode_ground_truth", False),
            ("official_accuracy_claim", False),
            ("selected_stills_only", True),
            ("clips_included", False),
            ("audio_included", False),
            ("identity_inference", False),
            ("biometric_processing", False),
            ("training_allowed", False),
            ("redistribution_allowed", False),
            ("deployment_change_authorized", False),
            ("deployment_approved", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdModelDiagnosticError(
                    f"household human audit {field} must be {str(expected).lower()}"
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
                [item.to_dict() for item in self.items]
                if field == "items"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        if include_fingerprint:
            value["audit_sha256"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def to_public_dict(self) -> dict[str, object]:
        return {
            "valid": True,
            "version": self.version,
            "audit_type": self.audit_type,
            "audit_id": self.audit_id,
            "audit_sha256": self.fingerprint,
            "source_selection_manifest_sha256": self.source_selection_manifest_sha256,
            "camera_count": self.camera_count,
            "reviewed_frame_count": len(self.items),
            "present_frame_count": self.present_frame_count,
            "absent_frame_count": self.absent_frame_count,
            "unknown_frame_count": self.unknown_frame_count,
            "human_reviewer_count": 1,
            "human_review_completed": True,
            "human_verified": True,
            "labels_created_without_model_outputs": True,
            "blind_ui_operator_attested": True,
            "blindness_cryptographically_proven": False,
            "single_human_reference": True,
            "ground_truth": False,
            "independent_human_ground_truth": False,
            "temporal_episode_ground_truth": False,
            "official_accuracy_claim": False,
            "deployment_approved": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdHumanPresenceAudit:
        fields = {*cls.__dataclass_fields__, "audit_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household human presence audit fields do not match v1"
            )
        if not isinstance(value["items"], list):
            raise HouseholdModelDiagnosticError("household human presence audit items are invalid")
        audit_sha256 = value["audit_sha256"]
        audit = cls(
            **{
                key: (
                    tuple(HouseholdHumanPresenceAuditItem.from_dict(item) for item in item_value)
                    if key == "items"
                    else item_value
                )
                for key, item_value in value.items()
                if key != "audit_sha256"
            }
        )
        if audit_sha256 != audit.fingerprint:
            raise HouseholdModelDiagnosticError(
                "household human presence audit fingerprint is invalid"
            )
        return audit

    @classmethod
    def from_json(cls, raw: str) -> HouseholdHumanPresenceAudit:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_HUMAN_AUDIT_BYTES:
            raise HouseholdModelDiagnosticError(
                "household human presence audit exceeds the size limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdHumanPresenceAudit:
        _raw, value = _read_json_artifact(
            path,
            context="household human presence audit",
            limit=MAX_HOUSEHOLD_HUMAN_AUDIT_BYTES,
        )
        return cls.from_dict(value)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(self.to_json(), path, limit=MAX_HOUSEHOLD_HUMAN_AUDIT_BYTES)


def _validate_selection_manifest(value: Any) -> dict[str, Any]:
    manifest = _exact_fields(
        value,
        _SELECTION_MANIFEST_FIELDS,
        "household human audit selection manifest",
    )
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise HouseholdModelDiagnosticError(
            "unsupported household human audit selection manifest version"
        )
    for field in ("batch_id", "sampling_strategy"):
        _safe_id(manifest[field], f"human audit selection {field}")
    for field in (
        "profile_sha256",
        "selection_manifest_sha256",
    ):
        _sha256(manifest[field], f"human audit selection {field}")
    unsigned = dict(manifest)
    claimed_sha256 = unsigned.pop("selection_manifest_sha256")
    if _canonical_ascii_sha256(unsigned) != claimed_sha256:
        raise HouseholdModelDiagnosticError(
            "household human audit selection manifest fingerprint is invalid"
        )
    source_count = _count(
        manifest["source_count"],
        "human audit selection source_count",
        maximum=MAX_HOUSEHOLD_HUMAN_AUDIT_CAMERAS,
    )
    frame_count = _count(
        manifest["frame_count"],
        "human audit selection frame_count",
        maximum=MAX_HOUSEHOLD_HUMAN_AUDIT_ITEMS,
    )
    max_frames = _count(
        manifest["max_frames"],
        "human audit selection max_frames",
        maximum=MAX_HOUSEHOLD_HUMAN_AUDIT_ITEMS,
    )
    if source_count < 1 or frame_count < 1 or max_frames < frame_count:
        raise HouseholdModelDiagnosticError("household human audit selection counts are invalid")
    _number(
        manifest["bin_seconds"],
        "human audit selection bin_seconds",
        minimum=0.1,
        maximum=60.0,
    )
    for field, expected in (
        ("profile_frozen_before_capture", True),
        ("model_generated_selection", True),
        ("consent_recorded", True),
        ("selected_stills_only", True),
        ("clips_shared", False),
        ("audio_shared", False),
        ("identity_inference", False),
        ("biometric_processing", False),
        ("training_allowed", False),
        ("redistribution_allowed", False),
        ("human_verified", False),
        ("ground_truth", False),
        ("production_remote_vlm_changed", False),
    ):
        if type(manifest[field]) is not bool or manifest[field] is not expected:
            raise HouseholdModelDiagnosticError(
                f"household human audit selection {field} must be {str(expected).lower()}"
            )
    if not isinstance(manifest["items"], list) or len(manifest["items"]) != frame_count:
        raise HouseholdModelDiagnosticError("household human audit selection items are invalid")
    item_by_id: dict[str, dict[str, Any]] = {}
    cameras: set[str] = set()
    frame_hashes: set[str] = set()
    source_timestamps: set[tuple[str, str, float]] = set()
    for raw_item in manifest["items"]:
        item = _exact_fields(
            raw_item,
            _SELECTION_ITEM_FIELDS,
            "household human audit selection item",
        )
        for field in ("review_id", "camera_id", "selection_source", "sheet_id"):
            _safe_id(item[field], f"human audit selection item {field}")
        for field in (
            "media_admission_sha256",
            "detector_scan_sha256",
            "pose_scan_sha256",
            "source_frame_sha256",
        ):
            _sha256(item[field], f"human audit selection item {field}")
        timestamp = _number(
            item["timestamp_seconds"],
            "human audit selection item timestamp_seconds",
            minimum=0.0,
            maximum=86_400.0,
        )
        if type(item["bin_index"]) is not int or not 0 <= item["bin_index"] <= 10_000_000:
            raise HouseholdModelDiagnosticError(
                "household human audit selection item bin_index is invalid"
            )
        if item["bin_index"] != int(timestamp // float(manifest["bin_seconds"])):
            raise HouseholdModelDiagnosticError(
                "household human audit selection item does not match the bin grid"
            )
        if (
            not isinstance(item["frame_filename"], str)
            or Path(item["frame_filename"]).name != item["frame_filename"]
            or len(item["frame_filename"]) > 255
        ):
            raise HouseholdModelDiagnosticError(
                "household human audit selection frame filename is invalid"
            )
        for field in (
            "detector_max_confidence_nearby",
            "pose_max_confidence_nearby",
        ):
            if item[field] is not None:
                _number(
                    item[field],
                    f"human audit selection item {field}",
                    minimum=0.0,
                    maximum=1.0,
                )
        if type(item["sheet_panel_index"]) is not int or not (
            0 <= item["sheet_panel_index"] <= MAX_HOUSEHOLD_HUMAN_AUDIT_ITEMS
        ):
            raise HouseholdModelDiagnosticError(
                "household human audit selection sheet_panel_index is invalid"
            )
        if item["review_id"] in item_by_id:
            raise HouseholdModelDiagnosticError(
                "household human audit selection review IDs are duplicated"
            )
        if item["source_frame_sha256"] in frame_hashes:
            raise HouseholdModelDiagnosticError(
                "household human audit selection frame hashes are duplicated"
            )
        source_timestamp = (item["camera_id"], item["media_admission_sha256"], timestamp)
        if source_timestamp in source_timestamps:
            raise HouseholdModelDiagnosticError(
                "household human audit selection source timestamps are duplicated"
            )
        item_by_id[item["review_id"]] = item
        frame_hashes.add(item["source_frame_sha256"])
        source_timestamps.add(source_timestamp)
        cameras.add(item["camera_id"])
    if len(cameras) != source_count:
        raise HouseholdModelDiagnosticError(
            "household human audit selection source count is inconsistent"
        )
    return item_by_id


def import_household_human_presence_audit(
    audit_manifest_path: str | Path,
    result_path: str | Path,
    selection_manifest_path: str | Path,
) -> HouseholdHumanPresenceAudit:
    """Validate a blind browser hand-off and return one strict audit artifact."""

    _manifest_raw, manifest_value = _read_json_artifact(
        audit_manifest_path,
        context="household human audit manifest",
        limit=MAX_HOUSEHOLD_HUMAN_AUDIT_INPUT_BYTES,
    )
    result_raw, result_value = _read_json_artifact(
        result_path,
        context="household human audit result",
        limit=MAX_HOUSEHOLD_HUMAN_AUDIT_INPUT_BYTES,
    )
    _selection_raw, selection_value = _read_json_artifact(
        selection_manifest_path,
        context="household human audit selection manifest",
        limit=MAX_HOUSEHOLD_HUMAN_AUDIT_INPUT_BYTES,
    )
    selection_by_id = _validate_selection_manifest(selection_value)

    manifest = _exact_fields(
        manifest_value,
        _RAW_MANIFEST_FIELDS,
        "household human audit manifest",
    )
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise HouseholdModelDiagnosticError("unsupported household human audit manifest version")
    if manifest["audit_type"] != "independent_human_frame_presence_audit":
        raise HouseholdModelDiagnosticError("household human audit manifest type is invalid")
    _safe_id(manifest["audit_id"], "human audit manifest audit_id")
    _sha256(
        manifest["source_selection_manifest_sha256"],
        "human audit manifest source_selection_manifest_sha256",
    )
    _utc_datetime(manifest["created_at_utc"], "manifest created_at_utc")
    if manifest["source_selection_manifest_sha256"] != selection_value["selection_manifest_sha256"]:
        raise HouseholdModelDiagnosticError(
            "household human audit manifest does not match the selection manifest"
        )
    _fixed_boolean_fields(manifest["privacy"], _RAW_PRIVACY, "household human audit privacy")
    _fixed_boolean_fields(manifest["claims"], _RAW_CLAIMS, "household human audit claims")
    sampling = _exact_fields(
        manifest["sampling"],
        _RAW_SAMPLING_FIELDS,
        "household human audit sampling",
    )
    if (
        sampling["source"] != "uniform_holdout_only"
        or sampling["within_camera_strategy"] != "evenly_spaced_order_statistics"
        or sampling["presentation_order"] != "deterministic_blinded_shuffle"
    ):
        raise HouseholdModelDiagnosticError("household human audit sampling strategy is invalid")
    camera_count = _count(
        sampling["camera_count"],
        "human audit camera_count",
        maximum=MAX_HOUSEHOLD_HUMAN_AUDIT_CAMERAS,
    )
    frames_per_camera = _count(
        sampling["frames_per_camera"],
        "human audit frames_per_camera",
        maximum=MAX_HOUSEHOLD_HUMAN_AUDIT_ITEMS,
    )
    frame_count = _count(
        sampling["frame_count"],
        "human audit frame_count",
        maximum=MAX_HOUSEHOLD_HUMAN_AUDIT_ITEMS,
    )
    if (
        camera_count < 1
        or frames_per_camera < 1
        or frame_count != (camera_count * frames_per_camera)
    ):
        raise HouseholdModelDiagnosticError("household human audit sampling counts are invalid")
    if manifest["expected_download_filename"] != f"{manifest['audit_id']}.completed.json":
        raise HouseholdModelDiagnosticError(
            "household human audit expected download filename is invalid"
        )
    if not isinstance(manifest["items"], list) or len(manifest["items"]) != frame_count:
        raise HouseholdModelDiagnosticError("household human audit manifest items are invalid")
    binding_items = tuple(
        _HouseholdHumanAuditBindingItem.from_dict(item) for item in manifest["items"]
    )
    if binding_items != tuple(sorted(binding_items, key=lambda item: item.review_item_id)):
        raise HouseholdModelDiagnosticError(
            "household human audit manifest items must use canonical review order"
        )
    unsigned_manifest = dict(manifest)
    claimed_manifest_sha256 = unsigned_manifest.pop("audit_manifest_sha256")
    _sha256(claimed_manifest_sha256, "human audit manifest audit_manifest_sha256")
    if _canonical_ascii_sha256(unsigned_manifest, trailing_newline=True) != (
        claimed_manifest_sha256
    ):
        raise HouseholdModelDiagnosticError("household human audit manifest fingerprint is invalid")

    result = _exact_fields(
        result_value,
        _RAW_RESULT_FIELDS,
        "household human audit result",
    )
    if type(result["version"]) is not int or result["version"] != 1:
        raise HouseholdModelDiagnosticError("unsupported household human audit result version")
    if result["review_type"] != "independent_human_frame_presence_audit_result":
        raise HouseholdModelDiagnosticError("household human audit result type is invalid")
    for field, expected in (
        ("human_review_completed", True),
        ("human_verified", True),
        ("labels_created_without_model_outputs", True),
        ("ground_truth", False),
        ("official_accuracy_claim", False),
        ("deployment_change_authorized", False),
    ):
        if type(result[field]) is not bool or result[field] is not expected:
            raise HouseholdModelDiagnosticError(
                f"household human audit result {field} must be {str(expected).lower()}"
            )
    if (
        result["audit_id"] != manifest["audit_id"]
        or result["source_selection_manifest_sha256"]
        != manifest["source_selection_manifest_sha256"]
        or result["audit_manifest_sha256"] != claimed_manifest_sha256
    ):
        raise HouseholdModelDiagnosticError(
            "household human audit result does not match its manifest"
        )
    completed = _utc_datetime(result["completed_at_utc"], "result completed_at_utc")
    if completed < _utc_datetime(manifest["created_at_utc"], "manifest created_at_utc"):
        raise HouseholdModelDiagnosticError("household human audit result predates its manifest")
    if not isinstance(result["items"], list) or len(result["items"]) != len(binding_items):
        raise HouseholdModelDiagnosticError("household human audit result items are invalid")
    items = tuple(HouseholdHumanPresenceAuditItem.from_dict(item) for item in result["items"])
    if tuple(item.binding for item in items) != binding_items:
        raise HouseholdModelDiagnosticError(
            "household human audit result item binding differs from its manifest"
        )

    for binding in binding_items:
        selected = selection_by_id.get(binding.source_review_id)
        if selected is None or selected["selection_source"] != "uniform_holdout":
            raise HouseholdModelDiagnosticError(
                "household human audit item is not a uniform holdout selection"
            )
        expected = (
            selected["camera_id"],
            selected["media_admission_sha256"],
            selected["bin_index"],
            float(selected["timestamp_seconds"]),
            selected["source_frame_sha256"],
        )
        actual = (
            binding.camera_id,
            binding.media_admission_sha256,
            binding.bin_index,
            binding.timestamp_seconds,
            binding.source_frame_sha256,
        )
        if actual != expected:
            raise HouseholdModelDiagnosticError(
                "household human audit item differs from its source selection"
            )

    return HouseholdHumanPresenceAudit(
        audit_id=manifest["audit_id"],
        created_at_utc=manifest["created_at_utc"],
        completed_at_utc=result["completed_at_utc"],
        source_batch_id=selection_value["batch_id"],
        source_selection_manifest_sha256=selection_value["selection_manifest_sha256"],
        source_profile_sha256=selection_value["profile_sha256"],
        audit_manifest_sha256=claimed_manifest_sha256,
        result_file_sha256=hashlib.sha256(result_raw).hexdigest(),
        bin_seconds=selection_value["bin_seconds"],
        camera_count=camera_count,
        frames_per_camera=frames_per_camera,
        sampling_source=sampling["source"],
        within_camera_strategy=sampling["within_camera_strategy"],
        presentation_order=sampling["presentation_order"],
        items=items,
    )


_AGREEMENT_COUNT_FIELDS = (
    "reviewed_frame_count",
    "human_present_frame_count",
    "human_absent_frame_count",
    "human_unknown_frame_count",
    "model_present_frame_count",
    "model_absent_frame_count",
    "model_unknown_frame_count",
    "comparable_frame_count",
    "agreement_count",
    "disagreement_count",
    "model_present_human_present",
    "model_present_human_absent",
    "model_present_human_unknown",
    "model_absent_human_present",
    "model_absent_human_absent",
    "model_absent_human_unknown",
    "model_unknown_human_present",
    "model_unknown_human_absent",
    "model_unknown_human_unknown",
)


def _validate_agreement_counts(value: Any, context: str) -> None:
    for field in _AGREEMENT_COUNT_FIELDS:
        _count(getattr(value, field), f"{context} {field}")
    cells = {
        ("present", "present"): value.model_present_human_present,
        ("present", "absent"): value.model_present_human_absent,
        ("present", "unknown"): value.model_present_human_unknown,
        ("absent", "present"): value.model_absent_human_present,
        ("absent", "absent"): value.model_absent_human_absent,
        ("absent", "unknown"): value.model_absent_human_unknown,
        ("unknown", "present"): value.model_unknown_human_present,
        ("unknown", "absent"): value.model_unknown_human_absent,
        ("unknown", "unknown"): value.model_unknown_human_unknown,
    }
    human_counts = {
        label: sum(count for (model, human), count in cells.items() if human == label)
        for label in HUMAN_PRESENCE_LABELS
    }
    model_counts = {
        label: sum(count for (model, human), count in cells.items() if model == label)
        for label in HUMAN_PRESENCE_LABELS
    }
    comparable = sum(
        count
        for (model, human), count in cells.items()
        if model != "unknown" and human != "unknown"
    )
    agreement = cells[("present", "present")] + cells[("absent", "absent")]
    if (
        value.reviewed_frame_count != sum(cells.values())
        or value.reviewed_frame_count != sum(human_counts.values())
        or value.reviewed_frame_count != sum(model_counts.values())
        or value.human_present_frame_count != human_counts["present"]
        or value.human_absent_frame_count != human_counts["absent"]
        or value.human_unknown_frame_count != human_counts["unknown"]
        or value.model_present_frame_count != model_counts["present"]
        or value.model_absent_frame_count != model_counts["absent"]
        or value.model_unknown_frame_count != model_counts["unknown"]
        or value.comparable_frame_count != comparable
        or value.agreement_count != agreement
        or value.disagreement_count != comparable - agreement
        or value.agreement_rate != _ratio(agreement, comparable)
    ):
        raise HouseholdModelDiagnosticError(
            f"household human audit {context} accounting is inconsistent"
        )


@dataclass(frozen=True, slots=True)
class HouseholdHumanModelAgreementCameraPoint:
    camera_id: str
    reviewed_frame_count: int
    human_present_frame_count: int
    human_absent_frame_count: int
    human_unknown_frame_count: int
    model_present_frame_count: int
    model_absent_frame_count: int
    model_unknown_frame_count: int
    comparable_frame_count: int
    agreement_count: int
    disagreement_count: int
    model_present_human_present: int
    model_present_human_absent: int
    model_present_human_unknown: int
    model_absent_human_present: int
    model_absent_human_absent: int
    model_absent_human_unknown: int
    model_unknown_human_present: int
    model_unknown_human_absent: int
    model_unknown_human_unknown: int
    agreement_rate: float | None

    def __post_init__(self) -> None:
        _safe_id(self.camera_id, "human audit agreement camera_id")
        object.__setattr__(
            self,
            "agreement_rate",
            _optional_ratio(self.agreement_rate, "human audit agreement_rate"),
        )
        _validate_agreement_counts(self, "agreement camera point")

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdHumanModelAgreementCameraPoint:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household human audit agreement camera fields do not match v1"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdHumanModelAgreementMetrics:
    camera_points: tuple[HouseholdHumanModelAgreementCameraPoint, ...]
    reviewed_frame_count: int
    human_present_frame_count: int
    human_absent_frame_count: int
    human_unknown_frame_count: int
    model_present_frame_count: int
    model_absent_frame_count: int
    model_unknown_frame_count: int
    comparable_frame_count: int
    agreement_count: int
    disagreement_count: int
    model_present_human_present: int
    model_present_human_absent: int
    model_present_human_unknown: int
    model_absent_human_present: int
    model_absent_human_absent: int
    model_absent_human_unknown: int
    model_unknown_human_present: int
    model_unknown_human_absent: int
    model_unknown_human_unknown: int
    agreement_rate: float | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.camera_points, tuple)
            or not 1 <= len(self.camera_points) <= MAX_HOUSEHOLD_HUMAN_AUDIT_CAMERAS
            or any(
                not isinstance(item, HouseholdHumanModelAgreementCameraPoint)
                for item in self.camera_points
            )
        ):
            raise HouseholdModelDiagnosticError(
                "household human audit agreement camera points are invalid"
            )
        camera_ids = tuple(item.camera_id for item in self.camera_points)
        if camera_ids != tuple(sorted(camera_ids)) or len(set(camera_ids)) != len(camera_ids):
            raise HouseholdModelDiagnosticError(
                "household human audit agreement cameras must be sorted and unique"
            )
        object.__setattr__(
            self,
            "agreement_rate",
            _optional_ratio(self.agreement_rate, "human audit aggregate agreement_rate"),
        )
        _validate_agreement_counts(self, "aggregate agreement")
        for field in _AGREEMENT_COUNT_FIELDS:
            if getattr(self, field) != sum(getattr(item, field) for item in self.camera_points):
                raise HouseholdModelDiagnosticError(
                    "household human audit aggregate agreement does not match cameras"
                )

    @classmethod
    def from_camera_points(
        cls,
        camera_points: tuple[HouseholdHumanModelAgreementCameraPoint, ...],
    ) -> HouseholdHumanModelAgreementMetrics:
        values = {
            field: sum(getattr(item, field) for item in camera_points)
            for field in _AGREEMENT_COUNT_FIELDS
        }
        return cls(
            camera_points=camera_points,
            **values,
            agreement_rate=_ratio(values["agreement_count"], values["comparable_frame_count"]),
        )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdHumanModelAgreementMetrics:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household human audit agreement metric fields do not match v1"
            )
        if not isinstance(value["camera_points"], list):
            raise HouseholdModelDiagnosticError(
                "household human audit agreement camera points are invalid"
            )
        return cls(
            **{
                key: (
                    tuple(
                        HouseholdHumanModelAgreementCameraPoint.from_dict(item)
                        for item in item_value
                    )
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


_FUSION_COUNT_FIELDS = (
    "reviewed_bin_count",
    "evaluated_bin_count",
    "positive_bin_count",
    "negative_bin_count",
    "unknown_bin_count",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
)


def _validate_fusion_counts(value: Any, context: str) -> None:
    for field in _FUSION_COUNT_FIELDS:
        _count(getattr(value, field), f"{context} {field}")
    if (
        value.reviewed_bin_count != value.evaluated_bin_count + value.unknown_bin_count
        or value.evaluated_bin_count != value.positive_bin_count + value.negative_bin_count
        or value.positive_bin_count != value.true_positive + value.false_negative
        or value.negative_bin_count != value.false_positive + value.true_negative
    ):
        raise HouseholdModelDiagnosticError(
            f"household human audit {context} count accounting is inconsistent"
        )
    precision = _ratio(value.true_positive, value.true_positive + value.false_positive)
    recall = _ratio(value.true_positive, value.positive_bin_count)
    expected = {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "specificity": _ratio(value.true_negative, value.negative_bin_count),
        "false_positive_rate": _ratio(value.false_positive, value.negative_bin_count),
    }
    if any(getattr(value, field) != expected_value for field, expected_value in expected.items()):
        raise HouseholdModelDiagnosticError(
            f"household human audit {context} ratios are inconsistent"
        )


@dataclass(frozen=True, slots=True)
class HouseholdHumanFusionCameraPoint:
    camera_id: str
    reviewed_bin_count: int
    evaluated_bin_count: int
    positive_bin_count: int
    negative_bin_count: int
    unknown_bin_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float | None
    recall: float | None
    f1: float | None
    specificity: float | None
    false_positive_rate: float | None

    def __post_init__(self) -> None:
        _safe_id(self.camera_id, "human audit fusion camera_id")
        for field in (
            "precision",
            "recall",
            "f1",
            "specificity",
            "false_positive_rate",
        ):
            object.__setattr__(
                self,
                field,
                _optional_ratio(getattr(self, field), f"human audit fusion {field}"),
            )
        _validate_fusion_counts(self, "fusion camera point")

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdHumanFusionCameraPoint:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household human audit fusion camera fields do not match v1"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HouseholdHumanFusionMetrics:
    camera_points: tuple[HouseholdHumanFusionCameraPoint, ...]
    reviewed_bin_count: int
    evaluated_bin_count: int
    positive_bin_count: int
    negative_bin_count: int
    unknown_bin_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float | None
    recall: float | None
    f1: float | None
    specificity: float | None
    false_positive_rate: float | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.camera_points, tuple)
            or not 1 <= len(self.camera_points) <= MAX_HOUSEHOLD_HUMAN_AUDIT_CAMERAS
            or any(
                not isinstance(item, HouseholdHumanFusionCameraPoint) for item in self.camera_points
            )
        ):
            raise HouseholdModelDiagnosticError(
                "household human audit fusion camera points are invalid"
            )
        camera_ids = tuple(item.camera_id for item in self.camera_points)
        if camera_ids != tuple(sorted(camera_ids)) or len(set(camera_ids)) != len(camera_ids):
            raise HouseholdModelDiagnosticError(
                "household human audit fusion cameras must be sorted and unique"
            )
        for field in (
            "precision",
            "recall",
            "f1",
            "specificity",
            "false_positive_rate",
        ):
            object.__setattr__(
                self,
                field,
                _optional_ratio(getattr(self, field), f"human audit aggregate fusion {field}"),
            )
        _validate_fusion_counts(self, "aggregate fusion")
        for field in _FUSION_COUNT_FIELDS:
            if getattr(self, field) != sum(getattr(item, field) for item in self.camera_points):
                raise HouseholdModelDiagnosticError(
                    "household human audit aggregate fusion does not match cameras"
                )

    @classmethod
    def from_camera_points(
        cls,
        camera_points: tuple[HouseholdHumanFusionCameraPoint, ...],
    ) -> HouseholdHumanFusionMetrics:
        values = {
            field: sum(getattr(item, field) for item in camera_points)
            for field in _FUSION_COUNT_FIELDS
        }
        precision = _ratio(
            values["true_positive"],
            values["true_positive"] + values["false_positive"],
        )
        recall = _ratio(values["true_positive"], values["positive_bin_count"])
        return cls(
            camera_points=camera_points,
            **values,
            precision=precision,
            recall=recall,
            f1=_f1(precision, recall),
            specificity=_ratio(values["true_negative"], values["negative_bin_count"]),
            false_positive_rate=_ratio(values["false_positive"], values["negative_bin_count"]),
        )

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdHumanFusionMetrics:
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise HouseholdModelDiagnosticError(
                "household human audit fusion metric fields do not match v1"
            )
        if not isinstance(value["camera_points"], list):
            raise HouseholdModelDiagnosticError(
                "household human audit fusion camera points are invalid"
            )
        return cls(
            **{
                key: (
                    tuple(HouseholdHumanFusionCameraPoint.from_dict(item) for item in item_value)
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
class HouseholdHumanPoseFusionAuditReport:
    audit_sha256: str
    model_review_sha256: str
    temporal_holdout_attestation_sha256: str
    holdout_report_sha256: str
    profile_sha256: str
    media_admission_sha256s: tuple[str, ...]
    detector_scan_sha256s: tuple[str, ...]
    pose_scan_sha256s: tuple[str, ...]
    camera_ids: tuple[str, ...]
    bin_seconds: float
    model_agreement: HouseholdHumanModelAgreementMetrics
    fusion_metrics: HouseholdHumanFusionMetrics
    version: int = 1
    report_type: str = "single_human_blind_pose_fusion_audit"
    label: str = "person"
    reference_provenance: str = "single_human_blind_frame_review"
    fusion_operator: str = "detector_or_gated_pose"
    temporal_aggregation_strategy: str = "independent_causal_signal_holds"
    prediction_anchor: str = "bin_end"
    hold_window_semantics: str = "current_bin_plus_prior_seconds"
    temporal_holdout_attested: bool = True
    profile_frozen_before_holdout_capture: bool = True
    profile_selected_on_human_reference: bool = False
    capture_order_operator_confirmed: bool = True
    capture_order_cryptographically_proven: bool = False
    human_review_completed: bool = True
    human_verified: bool = True
    human_reviewer_count: int = 1
    single_human_reference: bool = True
    labels_created_without_model_outputs: bool = True
    blind_ui_operator_attested: bool = True
    blindness_cryptographically_proven: bool = False
    frame_sample_only: bool = True
    model_generated_reference_labels: bool = False
    ground_truth: bool = False
    independent_human_ground_truth: bool = False
    temporal_episode_ground_truth: bool = False
    official_accuracy_claim: bool = False
    diagnostic_only: bool = True
    shadow_only: bool = True
    winner_selected: bool = False
    deployment_approved: bool = False
    household_configuration_changed: bool = False
    dataset_specific_only: bool = True
    subject_identity_inference: bool = False
    posture_classification: bool = False
    risk_accuracy_evaluated: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HouseholdModelDiagnosticError(
                "unsupported household human pose fusion audit report version"
            )
        for field in (
            "audit_sha256",
            "model_review_sha256",
            "temporal_holdout_attestation_sha256",
            "holdout_report_sha256",
            "profile_sha256",
        ):
            _sha256(getattr(self, field), f"human pose fusion audit {field}")
        if (
            not isinstance(self.camera_ids, tuple)
            or not 1 <= len(self.camera_ids) <= MAX_HOUSEHOLD_HUMAN_AUDIT_CAMERAS
            or self.camera_ids != tuple(sorted(self.camera_ids))
            or len(set(self.camera_ids)) != len(self.camera_ids)
        ):
            raise HouseholdModelDiagnosticError(
                "household human pose fusion audit camera IDs are invalid"
            )
        for camera_id in self.camera_ids:
            _safe_id(camera_id, "human pose fusion audit camera_id")
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
                    f"household human pose fusion audit {field} is invalid"
                )
            for value in values:
                _sha256(value, f"human pose fusion audit {field}")
        object.__setattr__(
            self,
            "bin_seconds",
            _number(
                self.bin_seconds,
                "human pose fusion audit bin_seconds",
                minimum=0.1,
                maximum=60.0,
            ),
        )
        if not isinstance(self.model_agreement, HouseholdHumanModelAgreementMetrics):
            raise HouseholdModelDiagnosticError(
                "household human pose fusion audit model agreement is invalid"
            )
        if not isinstance(self.fusion_metrics, HouseholdHumanFusionMetrics):
            raise HouseholdModelDiagnosticError(
                "household human pose fusion audit fusion metrics are invalid"
            )
        if (
            tuple(item.camera_id for item in self.model_agreement.camera_points) != self.camera_ids
            or tuple(item.camera_id for item in self.fusion_metrics.camera_points)
            != self.camera_ids
            or self.model_agreement.reviewed_frame_count != self.fusion_metrics.reviewed_bin_count
        ):
            raise HouseholdModelDiagnosticError(
                "household human pose fusion audit camera metric coverage is inconsistent"
            )
        for field, expected in (
            ("report_type", "single_human_blind_pose_fusion_audit"),
            ("label", "person"),
            ("reference_provenance", "single_human_blind_frame_review"),
            ("fusion_operator", "detector_or_gated_pose"),
            ("temporal_aggregation_strategy", "independent_causal_signal_holds"),
            ("prediction_anchor", "bin_end"),
            ("hold_window_semantics", "current_bin_plus_prior_seconds"),
        ):
            if getattr(self, field) != expected:
                raise HouseholdModelDiagnosticError(
                    f"household human pose fusion audit {field} is invalid"
                )
        if type(self.human_reviewer_count) is not int or self.human_reviewer_count != 1:
            raise HouseholdModelDiagnosticError(
                "household human pose fusion audit human_reviewer_count must be one"
            )
        for field, expected in (
            ("temporal_holdout_attested", True),
            ("profile_frozen_before_holdout_capture", True),
            ("profile_selected_on_human_reference", False),
            ("capture_order_operator_confirmed", True),
            ("capture_order_cryptographically_proven", False),
            ("human_review_completed", True),
            ("human_verified", True),
            ("single_human_reference", True),
            ("labels_created_without_model_outputs", True),
            ("blind_ui_operator_attested", True),
            ("blindness_cryptographically_proven", False),
            ("frame_sample_only", True),
            ("model_generated_reference_labels", False),
            ("ground_truth", False),
            ("independent_human_ground_truth", False),
            ("temporal_episode_ground_truth", False),
            ("official_accuracy_claim", False),
            ("diagnostic_only", True),
            ("shadow_only", True),
            ("winner_selected", False),
            ("deployment_approved", False),
            ("household_configuration_changed", False),
            ("dataset_specific_only", True),
            ("subject_identity_inference", False),
            ("posture_classification", False),
            ("risk_accuracy_evaluated", False),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdModelDiagnosticError(
                    f"household human pose fusion audit {field} must be {str(expected).lower()}"
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
                self.model_agreement.to_dict()
                if field == "model_agreement"
                else self.fusion_metrics.to_dict()
                if field == "fusion_metrics"
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
            "report_type": self.report_type,
            "report_sha256": self.fingerprint,
            "audit_sha256": self.audit_sha256,
            "temporal_holdout_attestation_sha256": (self.temporal_holdout_attestation_sha256),
            "profile_sha256": self.profile_sha256,
            "camera_count": len(self.camera_ids),
            "reviewed_frame_count": self.model_agreement.reviewed_frame_count,
            "human_present_frame_count": self.model_agreement.human_present_frame_count,
            "human_absent_frame_count": self.model_agreement.human_absent_frame_count,
            "human_unknown_frame_count": self.model_agreement.human_unknown_frame_count,
            "model_label_agreement_count": self.model_agreement.agreement_count,
            "model_label_disagreement_count": self.model_agreement.disagreement_count,
            "model_label_agreement_rate": self.model_agreement.agreement_rate,
            "fusion_evaluated_bin_count": self.fusion_metrics.evaluated_bin_count,
            "fusion_true_positive": self.fusion_metrics.true_positive,
            "fusion_false_positive": self.fusion_metrics.false_positive,
            "fusion_false_negative": self.fusion_metrics.false_negative,
            "fusion_true_negative": self.fusion_metrics.true_negative,
            "fusion_precision": self.fusion_metrics.precision,
            "fusion_recall": self.fusion_metrics.recall,
            "fusion_specificity": self.fusion_metrics.specificity,
            "human_reviewer_count": 1,
            "single_human_reference": True,
            "labels_created_without_model_outputs": True,
            "blindness_cryptographically_proven": False,
            "frame_sample_only": True,
            "ground_truth": False,
            "independent_human_ground_truth": False,
            "temporal_episode_ground_truth": False,
            "official_accuracy_claim": False,
            "deployment_approved": False,
            "household_configuration_changed": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }

    @classmethod
    def from_dict(cls, value: Any) -> HouseholdHumanPoseFusionAuditReport:
        fields = {*cls.__dataclass_fields__, "report_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household human pose fusion audit report fields do not match v1"
            )
        sequence_fields = {
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "pose_scan_sha256s",
            "camera_ids",
        }
        if any(not isinstance(value[field], list) for field in sequence_fields):
            raise HouseholdModelDiagnosticError(
                "household human pose fusion audit report arrays are invalid"
            )
        report_sha256 = value["report_sha256"]
        report = cls(
            **{
                key: (
                    tuple(item_value)
                    if key in sequence_fields
                    else HouseholdHumanModelAgreementMetrics.from_dict(item_value)
                    if key == "model_agreement"
                    else HouseholdHumanFusionMetrics.from_dict(item_value)
                    if key == "fusion_metrics"
                    else item_value
                )
                for key, item_value in value.items()
                if key != "report_sha256"
            }
        )
        if report_sha256 != report.fingerprint:
            raise HouseholdModelDiagnosticError(
                "household human pose fusion audit report fingerprint is invalid"
            )
        return report

    @classmethod
    def from_json(cls, raw: str) -> HouseholdHumanPoseFusionAuditReport:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_HUMAN_AUDIT_REPORT_BYTES:
            raise HouseholdModelDiagnosticError(
                "household human pose fusion audit report exceeds the size limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(cls, path: str | Path) -> HouseholdHumanPoseFusionAuditReport:
        _raw, value = _read_json_artifact(
            path,
            context="household human pose fusion audit report",
            limit=MAX_HOUSEHOLD_HUMAN_AUDIT_REPORT_BYTES,
        )
        return cls.from_dict(value)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_HUMAN_AUDIT_REPORT_BYTES,
        )


def _agreement_camera_point(
    camera_id: str,
    labels: tuple[tuple[str, str], ...],
) -> HouseholdHumanModelAgreementCameraPoint:
    cells = Counter(labels)
    values = {
        "reviewed_frame_count": len(labels),
        "human_present_frame_count": sum(human == "present" for _model, human in labels),
        "human_absent_frame_count": sum(human == "absent" for _model, human in labels),
        "human_unknown_frame_count": sum(human == "unknown" for _model, human in labels),
        "model_present_frame_count": sum(model == "present" for model, _human in labels),
        "model_absent_frame_count": sum(model == "absent" for model, _human in labels),
        "model_unknown_frame_count": sum(model == "unknown" for model, _human in labels),
        "model_present_human_present": cells[("present", "present")],
        "model_present_human_absent": cells[("present", "absent")],
        "model_present_human_unknown": cells[("present", "unknown")],
        "model_absent_human_present": cells[("absent", "present")],
        "model_absent_human_absent": cells[("absent", "absent")],
        "model_absent_human_unknown": cells[("absent", "unknown")],
        "model_unknown_human_present": cells[("unknown", "present")],
        "model_unknown_human_absent": cells[("unknown", "absent")],
        "model_unknown_human_unknown": cells[("unknown", "unknown")],
    }
    comparable = sum(
        count
        for (model, human), count in cells.items()
        if model != "unknown" and human != "unknown"
    )
    agreement = cells[("present", "present")] + cells[("absent", "absent")]
    return HouseholdHumanModelAgreementCameraPoint(
        camera_id=camera_id,
        **values,
        comparable_frame_count=comparable,
        agreement_count=agreement,
        disagreement_count=comparable - agreement,
        agreement_rate=_ratio(agreement, comparable),
    )


def _fusion_camera_point(
    camera_id: str,
    labels: tuple[tuple[str, bool], ...],
) -> HouseholdHumanFusionCameraPoint:
    evaluated = tuple(item for item in labels if item[0] != "unknown")
    positive = tuple(item for item in evaluated if item[0] == "present")
    negative = tuple(item for item in evaluated if item[0] == "absent")
    true_positive = sum(predicted for _label, predicted in positive)
    false_positive = sum(predicted for _label, predicted in negative)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, len(positive))
    return HouseholdHumanFusionCameraPoint(
        camera_id=camera_id,
        reviewed_bin_count=len(labels),
        evaluated_bin_count=len(evaluated),
        positive_bin_count=len(positive),
        negative_bin_count=len(negative),
        unknown_bin_count=len(labels) - len(evaluated),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=len(positive) - true_positive,
        true_negative=len(negative) - false_positive,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        specificity=_ratio(len(negative) - false_positive, len(negative)),
        false_positive_rate=_ratio(false_positive, len(negative)),
    )


def evaluate_household_human_pose_fusion_audit(
    audit: HouseholdHumanPresenceAudit,
    *,
    model_review: HouseholdModelPresenceReview,
    temporal_holdout_attestation: HouseholdPoseFusionTemporalHoldoutAttestation,
    media_admissions: tuple[ExternalSourceAdmission, ...],
    detector_scans: tuple[AdmittedDetectorScan, ...],
    pose_scans: tuple[AdmittedPoseScan, ...],
) -> HouseholdHumanPoseFusionAuditReport:
    """Compare a frozen fusion profile with a blinded single-human frame audit."""

    if not isinstance(audit, HouseholdHumanPresenceAudit):
        raise HouseholdModelDiagnosticError("household human presence audit is invalid")
    if not isinstance(model_review, HouseholdModelPresenceReview):
        raise HouseholdModelDiagnosticError("household human audit model review is invalid")
    if not isinstance(
        temporal_holdout_attestation,
        HouseholdPoseFusionTemporalHoldoutAttestation,
    ):
        raise HouseholdModelDiagnosticError(
            "household human audit temporal holdout attestation is invalid"
        )
    holdout = temporal_holdout_attestation.holdout_report
    if (
        audit.source_selection_manifest_sha256 != model_review.selection_manifest_sha256
        or audit.source_profile_sha256 != holdout.profile.fingerprint
        or model_review.fingerprint != holdout.review_sha256
        or audit.bin_seconds != holdout.bin_seconds
    ):
        raise HouseholdModelDiagnosticError(
            "household human audit does not match the frozen holdout review"
        )
    camera_ids = tuple(sorted({item.camera_id for item in audit.items}))
    if camera_ids != holdout.camera_ids or camera_ids != tuple(
        source.camera_id for source in model_review.sources
    ):
        raise HouseholdModelDiagnosticError(
            "household human audit cameras do not match the frozen holdout"
        )
    if (
        not isinstance(media_admissions, tuple)
        or len(media_admissions) != len(camera_ids)
        or any(not isinstance(item, ExternalSourceAdmission) for item in media_admissions)
    ):
        raise HouseholdModelDiagnosticError("household human audit media admissions are incomplete")
    admission_by_sha = {item.fingerprint: item for item in media_admissions}
    if len(admission_by_sha) != len(media_admissions) or set(admission_by_sha) != set(
        holdout.media_admission_sha256s
    ):
        raise HouseholdModelDiagnosticError(
            "household human audit media admissions do not match the holdout"
        )
    if (
        not isinstance(detector_scans, tuple)
        or len(detector_scans) != len(camera_ids)
        or any(not isinstance(item, AdmittedDetectorScan) for item in detector_scans)
    ):
        raise HouseholdModelDiagnosticError("household human audit detector scans are incomplete")
    if (
        not isinstance(pose_scans, tuple)
        or len(pose_scans) != len(camera_ids)
        or any(not isinstance(item, AdmittedPoseScan) for item in pose_scans)
    ):
        raise HouseholdModelDiagnosticError("household human audit pose scans are incomplete")

    detector_by_source = {
        (item.result.camera_id, item.source_admission_sha256): item for item in detector_scans
    }
    pose_by_source = {
        (item.result.camera_id, item.source_admission_sha256): item for item in pose_scans
    }
    source_by_camera = {source.camera_id: source for source in model_review.sources}
    expected_keys = {
        (source.camera_id, source.media_admission_sha256) for source in model_review.sources
    }
    if (
        len(detector_by_source) != len(detector_scans)
        or set(detector_by_source) != expected_keys
        or len(pose_by_source) != len(pose_scans)
        or set(pose_by_source) != expected_keys
    ):
        raise HouseholdModelDiagnosticError(
            "household human audit scans do not match the holdout sources"
        )
    ordered_detector_scans = tuple(
        detector_by_source[(camera_id, source_by_camera[camera_id].media_admission_sha256)]
        for camera_id in camera_ids
    )
    ordered_pose_scans = tuple(
        pose_by_source[(camera_id, source_by_camera[camera_id].media_admission_sha256)]
        for camera_id in camera_ids
    )
    if (
        tuple(item.fingerprint for item in ordered_detector_scans) != holdout.detector_scan_sha256s
        or tuple(item.fingerprint for item in ordered_pose_scans) != holdout.pose_scan_sha256s
    ):
        raise HouseholdModelDiagnosticError(
            "household human audit scan fingerprints do not match the holdout"
        )
    ordered_admission_sha256s = tuple(
        source_by_camera[camera_id].media_admission_sha256 for camera_id in camera_ids
    )
    if ordered_admission_sha256s != holdout.media_admission_sha256s:
        raise HouseholdModelDiagnosticError(
            "household human audit admission order does not match the holdout"
        )

    model_item_by_id = {item.review_item_id: item for item in model_review.items}
    profile_by_camera: dict[str, HouseholdPoseFusionCameraProfile] = {
        item.camera_id: item for item in holdout.profile.cameras
    }
    agreement_labels: dict[str, list[tuple[str, str]]] = {camera_id: [] for camera_id in camera_ids}
    fusion_labels: dict[str, list[tuple[str, bool]]] = {camera_id: [] for camera_id in camera_ids}
    for item in audit.items:
        model_item = model_item_by_id.get(item.source_review_id)
        if model_item is None or (
            model_item.camera_id,
            model_item.media_admission_sha256,
            model_item.timestamp_seconds,
            model_item.source_frame_sha256,
        ) != (
            item.camera_id,
            item.media_admission_sha256,
            item.timestamp_seconds,
            item.source_frame_sha256,
        ):
            raise HouseholdModelDiagnosticError(
                "household human audit item does not match the model review"
            )
        if item.timestamp_seconds > source_by_camera[item.camera_id].source_duration_seconds + 1e-9:
            raise HouseholdModelDiagnosticError(
                "household human audit item exceeds its admitted source duration"
            )
        source_key = (item.camera_id, item.media_admission_sha256)
        predicted = household_pose_fusion_bin_prediction(
            profile_by_camera[item.camera_id],
            detector_by_source[source_key],
            pose_by_source[source_key],
            bin_index=item.bin_index,
            bin_seconds=audit.bin_seconds,
        )
        agreement_labels[item.camera_id].append((model_item.visible_person, item.visible_person))
        fusion_labels[item.camera_id].append((item.visible_person, predicted))

    agreement_points = tuple(
        _agreement_camera_point(camera_id, tuple(agreement_labels[camera_id]))
        for camera_id in camera_ids
    )
    fusion_points = tuple(
        _fusion_camera_point(camera_id, tuple(fusion_labels[camera_id])) for camera_id in camera_ids
    )
    return HouseholdHumanPoseFusionAuditReport(
        audit_sha256=audit.fingerprint,
        model_review_sha256=model_review.fingerprint,
        temporal_holdout_attestation_sha256=temporal_holdout_attestation.fingerprint,
        holdout_report_sha256=holdout.fingerprint,
        profile_sha256=holdout.profile.fingerprint,
        media_admission_sha256s=holdout.media_admission_sha256s,
        detector_scan_sha256s=holdout.detector_scan_sha256s,
        pose_scan_sha256s=holdout.pose_scan_sha256s,
        camera_ids=camera_ids,
        bin_seconds=holdout.bin_seconds,
        model_agreement=HouseholdHumanModelAgreementMetrics.from_camera_points(agreement_points),
        fusion_metrics=HouseholdHumanFusionMetrics.from_camera_points(fusion_points),
    )
