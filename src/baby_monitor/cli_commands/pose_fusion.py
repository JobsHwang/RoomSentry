"""Evaluation-only household detector/pose fusion commands."""

from __future__ import annotations

import argparse

from baby_monitor.cli_support import _json_print

from ..detector_protocol import AdmittedDetectorScan
from ..household_human_presence_audit import (
    HouseholdHumanPoseFusionAuditReport,
    HouseholdHumanPresenceAudit,
    evaluate_household_human_pose_fusion_audit,
    import_household_human_presence_audit,
)
from ..household_model_diagnostic import HouseholdModelPresenceReview
from ..household_pose_fusion_diagnostic import (
    HouseholdPoseFusionBatchDiagnosticReport,
    HouseholdPoseFusionDiagnosticReport,
    HouseholdPoseFusionProfile,
    combine_household_pose_fusion_diagnostic_batches,
    evaluate_household_pose_fusion_diagnostic,
)
from ..household_pose_fusion_holdout import (
    HouseholdPoseFusionTemporalHoldoutAttestation,
    attest_household_pose_fusion_temporal_holdout,
)
from ..pose_protocol import AdmittedPoseScan
from ..source_admission import ExternalSourceAdmission


def command_validate_household_pose_fusion_profile(args: argparse.Namespace) -> int:
    profile = HouseholdPoseFusionProfile.read(args.path)
    _json_print(profile.to_public_dict())
    return 0


def command_evaluate_household_pose_fusion_diagnostic(args: argparse.Namespace) -> int:
    report = evaluate_household_pose_fusion_diagnostic(
        HouseholdModelPresenceReview.read(args.review),
        media_admissions=tuple(ExternalSourceAdmission.read(path) for path in args.media_admission),
        detector_scans=tuple(AdmittedDetectorScan.read(path) for path in args.detector_scan),
        pose_scans=tuple(AdmittedPoseScan.read(path) for path in args.pose_scan),
        profile=HouseholdPoseFusionProfile.read(args.profile),
        bin_seconds=args.bin_seconds,
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_household_pose_fusion_diagnostic(args: argparse.Namespace) -> int:
    report = HouseholdPoseFusionDiagnosticReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_combine_household_pose_fusion_diagnostic_batches(
    args: argparse.Namespace,
) -> int:
    report = combine_household_pose_fusion_diagnostic_batches(
        tuple(HouseholdPoseFusionDiagnosticReport.read(path) for path in args.paths)
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_household_pose_fusion_batch_diagnostic(
    args: argparse.Namespace,
) -> int:
    report = HouseholdPoseFusionBatchDiagnosticReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_attest_household_pose_fusion_temporal_holdout(
    args: argparse.Namespace,
) -> int:
    attestation = attest_household_pose_fusion_temporal_holdout(
        HouseholdPoseFusionBatchDiagnosticReport.read(args.selection_batch),
        HouseholdPoseFusionDiagnosticReport.read(args.holdout_report),
        confirm_selection_batch_used_for_profile_selection=(
            args.confirm_selection_batch_used_for_profile_selection
        ),
        confirm_profile_frozen_before_holdout_capture=(
            args.confirm_profile_frozen_before_holdout_capture
        ),
    )
    attestation.write_new(args.output)
    _json_print(attestation.to_public_dict())
    return 0


def command_validate_household_pose_fusion_temporal_holdout(
    args: argparse.Namespace,
) -> int:
    attestation = HouseholdPoseFusionTemporalHoldoutAttestation.read(args.path)
    _json_print(attestation.to_public_dict())
    return 0


def command_import_household_human_presence_audit(args: argparse.Namespace) -> int:
    audit = import_household_human_presence_audit(
        args.audit_manifest,
        args.result,
        args.selection_manifest,
    )
    audit.write_new(args.output)
    _json_print(audit.to_public_dict())
    return 0


def command_validate_household_human_presence_audit(args: argparse.Namespace) -> int:
    audit = HouseholdHumanPresenceAudit.read(args.path)
    _json_print(audit.to_public_dict())
    return 0


def command_evaluate_household_human_pose_fusion_audit(
    args: argparse.Namespace,
) -> int:
    report = evaluate_household_human_pose_fusion_audit(
        HouseholdHumanPresenceAudit.read(args.audit),
        model_review=HouseholdModelPresenceReview.read(args.model_review),
        temporal_holdout_attestation=(
            HouseholdPoseFusionTemporalHoldoutAttestation.read(args.temporal_holdout_attestation)
        ),
        media_admissions=tuple(ExternalSourceAdmission.read(path) for path in args.media_admission),
        detector_scans=tuple(AdmittedDetectorScan.read(path) for path in args.detector_scan),
        pose_scans=tuple(AdmittedPoseScan.read(path) for path in args.pose_scan),
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_household_human_pose_fusion_audit(
    args: argparse.Namespace,
) -> int:
    report = HouseholdHumanPoseFusionAuditReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0
