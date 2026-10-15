"""Strict provenance for a frozen-profile temporal holdout diagnostic.

The underlying detector/Pose fusion report remains a model-generated,
dataset-specific diagnostic.  This contract records that its profile was
selected on a separate, disjoint batch and was frozen before the holdout media
was captured.  The ordering is an explicit operator confirmation rather than
a cryptographic timestamp proof, and the result never becomes human ground
truth or deployment authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .household_model_diagnostic import (
    HouseholdModelDiagnosticError,
    _canonical_sha256,
    _strict_json,
    _write_exclusive,
)
from .household_pose_fusion_diagnostic import (
    HouseholdPoseFusionBatchDiagnosticReport,
    HouseholdPoseFusionDiagnosticReport,
)

MAX_HOUSEHOLD_POSE_FUSION_HOLDOUT_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class HouseholdPoseFusionTemporalHoldoutAttestation:
    """Embed selection and holdout evidence with conservative authority flags."""

    selection_batch: HouseholdPoseFusionBatchDiagnosticReport
    holdout_report: HouseholdPoseFusionDiagnosticReport
    version: int = 1
    attestation_type: str = "frozen_profile_temporal_holdout"
    temporal_holdout: bool = True
    selection_batch_used_for_profile_selection: bool = True
    profile_frozen_before_holdout_capture: bool = True
    profile_selected_on_holdout: bool = False
    selection_and_holdout_reports_distinct: bool = True
    selection_and_holdout_reviews_distinct: bool = True
    media_sets_disjoint: bool = True
    detector_scan_sets_disjoint: bool = True
    pose_scan_sets_disjoint: bool = True
    capture_order_operator_confirmed: bool = True
    capture_order_cryptographically_proven: bool = False
    review_model_independence_established: bool = False
    attestation_derived_without_model_rerun: bool = True
    model_generated_labels: bool = True
    human_verified: bool = False
    ground_truth: bool = False
    independent_human_ground_truth: bool = False
    official_accuracy_claim: bool = False
    deployment_approved: bool = False
    household_configuration_changed: bool = False
    dataset_specific_only: bool = True
    source_media_included: bool = False
    paths_disclosed: bool = False

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise HouseholdModelDiagnosticError(
                "unsupported household pose fusion holdout attestation version"
            )
        if not isinstance(
            self.selection_batch,
            HouseholdPoseFusionBatchDiagnosticReport,
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion holdout selection batch is invalid"
            )
        if not isinstance(self.holdout_report, HouseholdPoseFusionDiagnosticReport):
            raise HouseholdModelDiagnosticError("household pose fusion holdout report is invalid")
        if self.attestation_type != "frozen_profile_temporal_holdout":
            raise HouseholdModelDiagnosticError(
                "household pose fusion holdout attestation type is invalid"
            )
        if (
            self.selection_batch.camera_ids != self.holdout_report.camera_ids
            or self.selection_batch.profile != self.holdout_report.profile
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion holdout profile does not match selection data"
            )
        selection_identity = (
            self.selection_batch.detector_adapter,
            self.selection_batch.detector_model_id,
            self.selection_batch.detector_model_revision,
            self.selection_batch.detector_model_sha256,
            self.selection_batch.detector_source_confidence_threshold,
            self.selection_batch.detector_requested_sample_fps,
            self.selection_batch.pose_adapter,
            self.selection_batch.pose_model_id,
            self.selection_batch.pose_model_revision,
            self.selection_batch.pose_model_sha256,
            self.selection_batch.pose_source_confidence_threshold,
            self.selection_batch.pose_requested_sample_fps,
            self.selection_batch.pose_requested_max_frames,
            self.selection_batch.pose_image_size,
            self.selection_batch.pose_rotation_degrees,
            self.selection_batch.bin_seconds,
            self.selection_batch.sampling_strategy,
            self.selection_batch.label,
            self.selection_batch.fusion_operator,
            self.selection_batch.temporal_aggregation_strategy,
            self.selection_batch.prediction_anchor,
            self.selection_batch.hold_window_semantics,
            self.selection_batch.conflict_resolution_strategy,
        )
        holdout_identity = (
            self.holdout_report.detector_adapter,
            self.holdout_report.detector_model_id,
            self.holdout_report.detector_model_revision,
            self.holdout_report.detector_model_sha256,
            self.holdout_report.detector_source_confidence_threshold,
            self.holdout_report.detector_requested_sample_fps,
            self.holdout_report.pose_adapter,
            self.holdout_report.pose_model_id,
            self.holdout_report.pose_model_revision,
            self.holdout_report.pose_model_sha256,
            self.holdout_report.pose_source_confidence_threshold,
            self.holdout_report.pose_requested_sample_fps,
            self.holdout_report.pose_requested_max_frames,
            self.holdout_report.pose_image_size,
            self.holdout_report.pose_rotation_degrees,
            self.holdout_report.bin_seconds,
            self.holdout_report.sampling_strategy,
            self.holdout_report.label,
            self.holdout_report.fusion_operator,
            self.holdout_report.temporal_aggregation_strategy,
            self.holdout_report.prediction_anchor,
            self.holdout_report.hold_window_semantics,
            self.holdout_report.conflict_resolution_strategy,
        )
        if selection_identity != holdout_identity:
            raise HouseholdModelDiagnosticError(
                "household pose fusion holdout model configuration differs from selection data"
            )
        report_hashes = {item.report_sha256 for item in self.selection_batch.batches}
        review_hashes = {item.review_sha256 for item in self.selection_batch.batches}
        if (
            self.holdout_report.fingerprint in report_hashes
            or self.holdout_report.review_sha256 in review_hashes
        ):
            raise HouseholdModelDiagnosticError(
                "household pose fusion selection and holdout reports must be distinct"
            )
        for field in (
            "media_admission_sha256s",
            "detector_scan_sha256s",
            "pose_scan_sha256s",
        ):
            selection_values = {
                value
                for binding in self.selection_batch.batches
                for value in getattr(binding, field)
            }
            holdout_values = set(getattr(self.holdout_report, field))
            if selection_values & holdout_values:
                raise HouseholdModelDiagnosticError(
                    "household pose fusion selection and holdout inputs must be disjoint"
                )
        for field, expected in (
            ("temporal_holdout", True),
            ("selection_batch_used_for_profile_selection", True),
            ("profile_frozen_before_holdout_capture", True),
            ("profile_selected_on_holdout", False),
            ("selection_and_holdout_reports_distinct", True),
            ("selection_and_holdout_reviews_distinct", True),
            ("media_sets_disjoint", True),
            ("detector_scan_sets_disjoint", True),
            ("pose_scan_sets_disjoint", True),
            ("capture_order_operator_confirmed", True),
            ("capture_order_cryptographically_proven", False),
            ("review_model_independence_established", False),
            ("attestation_derived_without_model_rerun", True),
            ("model_generated_labels", True),
            ("human_verified", False),
            ("ground_truth", False),
            ("independent_human_ground_truth", False),
            ("official_accuracy_claim", False),
            ("deployment_approved", False),
            ("household_configuration_changed", False),
            ("dataset_specific_only", True),
            ("source_media_included", False),
            ("paths_disclosed", False),
        ):
            if type(getattr(self, field)) is not bool or getattr(self, field) is not expected:
                raise HouseholdModelDiagnosticError(
                    f"household pose fusion holdout {field} must be {str(expected).lower()}"
                )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            field: (
                self.selection_batch.to_dict()
                if field == "selection_batch"
                else self.holdout_report.to_dict()
                if field == "holdout_report"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }
        if include_fingerprint:
            value["attestation_sha256"] = self.fingerprint
        return value

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    def to_public_dict(self) -> dict[str, object]:
        metrics = self.holdout_report.metrics
        return {
            "valid": True,
            "version": self.version,
            "attestation_type": self.attestation_type,
            "attestation_sha256": self.fingerprint,
            "profile_sha256": self.holdout_report.profile.fingerprint,
            "selection_batch_report_sha256": self.selection_batch.fingerprint,
            "selection_batch_count": len(self.selection_batch.batches),
            "holdout_report_sha256": self.holdout_report.fingerprint,
            "camera_count": len(self.holdout_report.camera_ids),
            "holdout_reviewed_frame_count": self.holdout_report.reviewed_frame_count,
            "holdout_evaluated_bin_count": self.holdout_report.evaluated_bin_count,
            "holdout_positive_bin_count": self.holdout_report.positive_bin_count,
            "holdout_negative_bin_count": self.holdout_report.negative_bin_count,
            "holdout_conflicting_bin_count": self.holdout_report.conflicting_bin_count,
            "holdout_true_positive": metrics.true_positive,
            "holdout_false_positive": metrics.false_positive,
            "holdout_false_negative": metrics.false_negative,
            "holdout_true_negative": metrics.true_negative,
            "temporal_holdout": True,
            "selection_batch_used_for_profile_selection": True,
            "profile_frozen_before_holdout_capture": True,
            "profile_selected_on_holdout": False,
            "selection_and_holdout_reports_distinct": True,
            "selection_and_holdout_reviews_distinct": True,
            "media_sets_disjoint": True,
            "detector_scan_sets_disjoint": True,
            "pose_scan_sets_disjoint": True,
            "capture_order_operator_confirmed": True,
            "capture_order_cryptographically_proven": False,
            "review_model_independence_established": False,
            "attestation_derived_without_model_rerun": True,
            "model_generated_labels": True,
            "human_verified": False,
            "ground_truth": False,
            "independent_human_ground_truth": False,
            "official_accuracy_claim": False,
            "deployment_approved": False,
            "household_configuration_changed": False,
            "dataset_specific_only": True,
            "source_media_included": False,
            "paths_disclosed": False,
        }

    @classmethod
    def from_dict(
        cls,
        value: Any,
    ) -> HouseholdPoseFusionTemporalHoldoutAttestation:
        fields = {*cls.__dataclass_fields__, "attestation_sha256"}
        if not isinstance(value, dict) or set(value) != fields:
            raise HouseholdModelDiagnosticError(
                "household pose fusion holdout attestation fields do not match v1"
            )
        attestation_sha256 = value["attestation_sha256"]
        attestation = cls(
            **{
                key: (
                    HouseholdPoseFusionBatchDiagnosticReport.from_dict(item)
                    if key == "selection_batch"
                    else HouseholdPoseFusionDiagnosticReport.from_dict(item)
                    if key == "holdout_report"
                    else item
                )
                for key, item in value.items()
                if key != "attestation_sha256"
            }
        )
        if attestation_sha256 != attestation.fingerprint:
            raise HouseholdModelDiagnosticError(
                "household pose fusion holdout attestation fingerprint is invalid"
            )
        return attestation

    @classmethod
    def from_json(
        cls,
        raw: str,
    ) -> HouseholdPoseFusionTemporalHoldoutAttestation:
        if len(raw.encode("utf-8")) > MAX_HOUSEHOLD_POSE_FUSION_HOLDOUT_BYTES:
            raise HouseholdModelDiagnosticError(
                "household pose fusion holdout attestation exceeds the size limit"
            )
        return cls.from_dict(_strict_json(raw))

    @classmethod
    def read(
        cls,
        path: str | Path,
    ) -> HouseholdPoseFusionTemporalHoldoutAttestation:
        source = Path(path).expanduser()
        try:
            if source.is_symlink() or not source.is_file():
                raise OSError
            if not 0 < source.stat().st_size <= MAX_HOUSEHOLD_POSE_FUSION_HOLDOUT_BYTES:
                raise HouseholdModelDiagnosticError(
                    "household pose fusion holdout attestation exceeds the size limit"
                )
            raw = source.read_text(encoding="utf-8")
        except HouseholdModelDiagnosticError:
            raise
        except (OSError, UnicodeError) as exc:
            raise HouseholdModelDiagnosticError(
                "household pose fusion holdout attestation is missing or unsafe"
            ) from exc
        return cls.from_json(raw)

    def write_new(self, path: str | Path) -> None:
        _write_exclusive(
            self.to_json(),
            path,
            limit=MAX_HOUSEHOLD_POSE_FUSION_HOLDOUT_BYTES,
        )


def attest_household_pose_fusion_temporal_holdout(
    selection_batch: HouseholdPoseFusionBatchDiagnosticReport,
    holdout_report: HouseholdPoseFusionDiagnosticReport,
    *,
    confirm_selection_batch_used_for_profile_selection: bool,
    confirm_profile_frozen_before_holdout_capture: bool,
) -> HouseholdPoseFusionTemporalHoldoutAttestation:
    """Create an attestation only after both ordering facts are explicit."""

    if confirm_selection_batch_used_for_profile_selection is not True:
        raise HouseholdModelDiagnosticError(
            "household pose fusion holdout requires explicit selection-batch confirmation"
        )
    if confirm_profile_frozen_before_holdout_capture is not True:
        raise HouseholdModelDiagnosticError(
            "household pose fusion holdout requires explicit frozen-profile confirmation"
        )
    return HouseholdPoseFusionTemporalHoldoutAttestation(
        selection_batch=selection_batch,
        holdout_report=holdout_report,
    )
