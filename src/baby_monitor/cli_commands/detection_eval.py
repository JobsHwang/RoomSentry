"""Detector scan, threshold sweep, candidate gate, and household evaluation commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

from baby_monitor.cli_support import (  # noqa: F401
    _absolute_runner_executable,
    _aware_datetime,
    _configured_remote_adapter,
    _database_from_path,
    _enqueue_buffered_candidate,
    _json_print,
    _local_adapter,
    _remote_adapter,
    _review_policy,
    _secondary_adapter,
    _storage_override,
    _yolo_detector,
)

from ..adapters.camera import LocalFileCamera
from ..adapters.detector import (
    CommandYoloDetector,
)
from ..candidate_trigger_evaluation import (
    CANDIDATE_GATE_SELECTION_RULE,
    CandidateStartGatePolicy,
    CandidateStartGateSweepReport,
    evaluate_candidate_start_gate_sweep,
    select_candidate_start_gate_operating_point,
)
from ..detector_evaluation import (
    DetectorBoxQualityReport,
    build_detector_sampled_tracking_batch,
    build_detector_tracking_batch,
    build_detector_tracking_evaluation_set,
    evaluate_detector_box_quality,
)
from ..detector_protocol import AdmittedDetectorScan, DetectorScanResult
from ..detector_threshold_sweep import (
    DetectorOperatingPointPolicy,
    DetectorThresholdSweepReport,
    evaluate_detector_threshold_sweep,
    select_detector_operating_point,
)
from ..household_ground_truth import (
    HOUSEHOLD_TRIGGER_SELECTION_RULE,
    MAX_HOUSEHOLD_TRIGGER_REPORT_BYTES,
    HouseholdEpisodeAnnotation,
    HouseholdTriggerCoverageReport,
    HouseholdTriggerPolicy,
    _strict_json,
    _write_exclusive,
    compare_household_trigger_coverage,
    evaluate_household_trigger_coverage,
    select_household_trigger_operating_point,
)
from ..household_model_diagnostic import (
    HOUSEHOLD_MODEL_DIAGNOSTIC_SELECTION_RULE,
    HouseholdModelDiagnosticDriftPolicy,
    HouseholdModelDiagnosticDriftReport,
    HouseholdModelDiagnosticPolicy,
    HouseholdModelDiagnosticReport,
    HouseholdModelPresenceReview,
    compare_household_model_diagnostic_batches,
    compare_household_model_presence_diagnostics,
    evaluate_household_model_presence_diagnostic,
    select_household_model_diagnostic_working_point,
)
from ..household_model_temporal_batch import (
    HouseholdModelTemporalBatchDiagnosticReport,
    combine_household_model_temporal_diagnostic_batches,
    select_household_model_temporal_batch_working_point,
)
from ..household_model_temporal_diagnostic import (
    HOUSEHOLD_MODEL_TEMPORAL_SELECTION_RULE,
    HouseholdModelTemporalDiagnosticPolicy,
    HouseholdModelTemporalDiagnosticReport,
    evaluate_household_model_temporal_diagnostic,
    select_household_model_temporal_working_point,
)
from ..household_sampled_coverage import evaluate_household_sampled_coverage
from ..object_tracker import (
    DetectionFrameBatch,
)
from ..source_admission import ExternalSourceAdmission
from ..tracker_evaluation import (
    TrackerAssociationReview,
)
from ..yolo_model import YoloModelManifest
from ..yolo_pose_model import YoloPoseModelManifest


def command_verify_yolo_model(args: argparse.Namespace) -> int:
    manifest = YoloModelManifest.read(args.manifest)
    verification = manifest.verify_weight(args.weights)
    _json_print(verification.to_public_dict())
    return 0


def _sampling_summary(result: DetectorScanResult) -> dict[str, object]:
    ledger = result.sampling
    return {
        "sampling_evidence_available": ledger is not None,
        "sampled_frame_count": len(ledger.samples) if ledger else None,
        "zero_detection_sample_count": sum(s.detection_count == 0 for s in ledger.samples)
        if ledger
        else None,
        "decoded_frame_count": ledger.decoded_frame_count if ledger else None,
        "max_adjacent_container_pts_gap_seconds": ledger.max_adjacent_container_pts_gap_seconds
        if ledger
        else None,
        "timestamp_fallback_frame_count": ledger.timestamp_fallback_frame_count if ledger else None,
        "end_reason": ledger.end_reason if ledger else "unknown",
        "source_exhaustion_verified": False,
    }


def command_yolo_scan_file(args: argparse.Namespace) -> int:
    admission = ExternalSourceAdmission.read(args.source_admission)
    if admission.asset_kind != "media_asset":
        raise ValueError("YOLO scan source admission must describe a media asset")
    admission.verify_asset(args.source)
    source_path = Path(args.source).expanduser().resolve()
    runner_input = Path(args.runner_executable).expanduser()
    if not runner_input.is_absolute() or not runner_input.is_file():
        raise ValueError("YOLO runner executable must be an absolute regular file")
    runner = runner_input
    manifest = YoloModelManifest.read(args.manifest)
    labels = tuple(args.label or ("person",))
    detector = CommandYoloDetector(
        model_path=args.weights,
        manifest_path=args.manifest,
        manifest=manifest,
        labels=labels,
        confidence=args.confidence,
        sample_fps=args.sample_fps,
        device=args.device,
        image_size=args.image_size,
        rotation_degrees=args.rotation_degrees,
        command=(
            str(runner),
            "-m",
            "baby_monitor.local_runners.yolo_detector",
            "{request}",
        ),
        timeout_seconds=args.timeout_seconds,
    )
    source = LocalFileCamera(args.camera_id, source_path).source()
    result = detector.scan_result(source)
    admission.verify_asset(args.source)
    scan = AdmittedDetectorScan(
        source_admission_sha256=admission.fingerprint,
        labels=labels,
        confidence_threshold=args.confidence,
        requested_sample_fps=args.sample_fps,
        result=result,
        version=result.version,
    )
    scan.write_new(args.output)
    _json_print(
        {
            "valid": True,
            "version": 1,
            "camera_id": result.camera_id,
            "detector_adapter": result.detector_adapter,
            "model_id": result.model_id,
            "model_revision": result.model_revision,
            "model_sha256": result.model_sha256,
            "source_admission_sha256": admission.fingerprint,
            "scan_sha256": scan.fingerprint,
            "labels": list(labels),
            "confidence_threshold": args.confidence,
            "requested_sample_fps": args.sample_fps,
            "detection_count": len(result.detections),
            "scan_version": scan.version,
            **_sampling_summary(result),
            "detection_timestamp_count": len(
                {detection.timestamp_seconds for detection in result.detections}
            ),
            "subject_identity_inference": False,
            "source_media_included": False,
            "paths_disclosed": False,
            "network_used": False,
        }
    )
    return 0


def command_validate_detector_scan(args: argparse.Namespace) -> int:
    scan = AdmittedDetectorScan.read(args.path)
    result = scan.result
    _json_print(
        {
            "valid": True,
            "version": scan.version,
            "camera_id": result.camera_id,
            "detector_adapter": result.detector_adapter,
            "model_id": result.model_id,
            "model_revision": result.model_revision,
            "model_sha256": result.model_sha256,
            "source_admission_sha256": scan.source_admission_sha256,
            "scan_sha256": scan.fingerprint,
            "labels": list(scan.labels),
            "confidence_threshold": scan.confidence_threshold,
            "requested_sample_fps": scan.requested_sample_fps,
            "detection_count": len(result.detections),
            **_sampling_summary(result),
            "detection_timestamp_count": len(
                {detection.timestamp_seconds for detection in result.detections}
            ),
            "subject_identity_inference": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_evaluate_detector_box_quality(args: argparse.Namespace) -> int:
    ground_truth_batch = DetectionFrameBatch.read(args.ground_truth_frames)
    ground_truth_review = TrackerAssociationReview.read(args.ground_truth_review)
    ground_truth_admission = ExternalSourceAdmission.read(args.ground_truth_admission)
    media_admission = ExternalSourceAdmission.read(args.media_admission)
    media_admission.verify_asset(args.media_asset)
    detector_scan = AdmittedDetectorScan.read(args.detector_scan)
    report = evaluate_detector_box_quality(
        ground_truth_batch,
        ground_truth_review,
        detector_scan,
        ground_truth_admission=ground_truth_admission,
        media_admission=media_admission,
        source_fps=args.source_fps,
        source_frame_count=args.source_frame_count,
        annotation_frame_offset=args.annotation_frame_offset,
        iou_threshold=args.iou_threshold,
        frame_alignment_confirmed=args.confirm_frame_alignment,
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_evaluate_detector_threshold_sweep(args: argparse.Namespace) -> int:
    ground_truth_batch = DetectionFrameBatch.read(args.ground_truth_frames)
    ground_truth_review = TrackerAssociationReview.read(args.ground_truth_review)
    ground_truth_admission = ExternalSourceAdmission.read(args.ground_truth_admission)
    media_admission = ExternalSourceAdmission.read(args.media_admission)
    media_admission.verify_asset(args.media_asset)
    detector_scan = AdmittedDetectorScan.read(args.detector_scan)
    report = evaluate_detector_threshold_sweep(
        ground_truth_batch,
        ground_truth_review,
        detector_scan,
        ground_truth_admission=ground_truth_admission,
        media_admission=media_admission,
        source_fps=args.source_fps,
        source_frame_count=args.source_frame_count,
        annotation_frame_offset=args.annotation_frame_offset,
        thresholds=tuple(args.threshold),
        iou_threshold=args.iou_threshold,
        frame_alignment_confirmed=args.confirm_frame_alignment,
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_select_detector_operating_point(args: argparse.Namespace) -> int:
    report = DetectorThresholdSweepReport.read(args.path)
    policy = DetectorOperatingPointPolicy(
        objective=args.objective,
        minimum_precision=args.minimum_precision,
        minimum_recall=args.minimum_recall,
        maximum_longest_miss_seconds=args.maximum_longest_miss_seconds,
    )
    selected, feasible_point_count = select_detector_operating_point(
        report,
        policy,
    )
    _json_print(
        {
            "valid": True,
            "version": 1,
            "sweep_report_sha256": report.fingerprint,
            "model_id": report.model_id,
            "model_revision": report.model_revision,
            "model_sha256": report.model_sha256,
            "policy": policy.to_dict(),
            "feasible_point_count": feasible_point_count,
            "selected_point": None if selected is None else selected.to_dict(),
            "dataset_specific_only": True,
            "deployment_approved": False,
            "subject_identity_inference": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_validate_detector_threshold_sweep(args: argparse.Namespace) -> int:
    report = DetectorThresholdSweepReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_evaluate_candidate_start_gate_sweep(
    args: argparse.Namespace,
) -> int:
    ground_truth_batch = DetectionFrameBatch.read(args.ground_truth_frames)
    ground_truth_review = TrackerAssociationReview.read(args.ground_truth_review)
    ground_truth_admission = ExternalSourceAdmission.read(args.ground_truth_admission)
    media_admission = ExternalSourceAdmission.read(args.media_admission)
    media_admission.verify_asset(args.media_asset)
    detector_scan = AdmittedDetectorScan.read(args.detector_scan)
    report = evaluate_candidate_start_gate_sweep(
        ground_truth_batch,
        ground_truth_review,
        detector_scan,
        ground_truth_admission=ground_truth_admission,
        media_admission=media_admission,
        source_fps=args.source_fps,
        source_frame_count=args.source_frame_count,
        annotation_frame_offset=args.annotation_frame_offset,
        thresholds=tuple(args.start_threshold),
        candidate_gap_seconds=args.candidate_gap_seconds,
        max_event_span_seconds=args.max_event_span_seconds,
        validation_iou_threshold=args.validation_iou_threshold,
        frame_alignment_confirmed=args.confirm_frame_alignment,
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_candidate_start_gate_sweep(
    args: argparse.Namespace,
) -> int:
    report = CandidateStartGateSweepReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_select_candidate_start_gate_operating_point(
    args: argparse.Namespace,
) -> int:
    report = CandidateStartGateSweepReport.read(args.path)
    policy = CandidateStartGatePolicy(
        minimum_ground_truth_episode_recall=(args.minimum_ground_truth_episode_recall),
        minimum_positive_sample_recall=args.minimum_positive_sample_recall,
        maximum_detection_delay_seconds=args.maximum_detection_delay_seconds,
        maximum_longest_uncovered_positive_seconds=(
            args.maximum_longest_uncovered_positive_seconds
        ),
        maximum_negative_false_activation_rate=(args.maximum_negative_false_activation_rate),
        require_negative_samples=args.require_negative_samples,
    )
    selected, feasible_point_count = select_candidate_start_gate_operating_point(report, policy)
    _json_print(
        {
            "valid": True,
            "version": 1,
            "sweep_report_sha256": report.fingerprint,
            "model_id": report.model_id,
            "model_revision": report.model_revision,
            "model_sha256": report.model_sha256,
            "policy": policy.to_dict(),
            "feasible_point_count": feasible_point_count,
            "selected_point": None if selected is None else selected.to_dict(),
            "selection_rule": CANDIDATE_GATE_SELECTION_RULE,
            "dataset_specific_only": True,
            "deployment_approved": False,
            "household_configuration_changed": False,
            "end_to_end_latency_evaluated": False,
            "risk_accuracy_evaluated": False,
            "subject_identity_inference": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_validate_household_episode_annotation(
    args: argparse.Namespace,
) -> int:
    annotation = HouseholdEpisodeAnnotation.read(args.path)
    _json_print(annotation.to_dict())
    return 0


def command_evaluate_household_trigger_coverage(
    args: argparse.Namespace,
) -> int:
    annotation = HouseholdEpisodeAnnotation.read(args.annotation)
    media_admission = ExternalSourceAdmission.read(args.media_admission)
    if args.media_asset:
        media_admission.verify_asset(args.media_asset)
    detector_scan = AdmittedDetectorScan.read(args.detector_scan)
    report = evaluate_household_trigger_coverage(
        annotation,
        detector_scan,
        media_admission=media_admission,
        thresholds=tuple(args.start_threshold),
        candidate_gap_seconds=args.candidate_gap_seconds,
        max_event_span_seconds=args.max_event_span_seconds,
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def _sampled_coverage_input(path: str, *, maximum: int) -> bytes:
    """Read one bounded regular input without disclosing a path on failure."""
    source = Path(path).expanduser()
    descriptor = -1
    try:
        metadata = source.lstat()
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum:
            raise ValueError("sampled coverage input is unsafe or exceeds its size limit")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(source, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise ValueError("sampled coverage input changed during reading")
        chunks = bytearray()
        while len(chunks) <= maximum:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        final = os.fstat(descriptor)
        path_metadata = source.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            not 0 < len(chunks) <= maximum
            or len(chunks) != opened.st_size
            or any(getattr(opened, field) != getattr(final, field) for field in fields)
            or any(getattr(opened, field) != getattr(path_metadata, field) for field in fields)
        ):
            raise ValueError("sampled coverage input changed or exceeds its size limit")
        return bytes(chunks)
    except OSError:
        raise ValueError("sampled coverage input is missing or unsafe") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def command_evaluate_household_sampled_coverage(args: argparse.Namespace) -> int:
    raw_binding = _sampled_coverage_input(args.timeline_binding, maximum=64 * 1024)
    try:
        timeline_binding = _strict_json(raw_binding.decode("utf-8"))
    except UnicodeError:
        raise ValueError("sampled coverage timeline binding must be UTF-8 JSON") from None
    # This opaque reference is only byte-hashed: no interpretation or disclosure.
    reference_sha256 = hashlib.sha256(
        _sampled_coverage_input(args.timeline_reference, maximum=8 * 1024 * 1024)
    ).hexdigest()
    annotation = HouseholdEpisodeAnnotation.read(args.annotation)
    media_admission = ExternalSourceAdmission.read(args.media_admission)
    media_admission.verify_asset(args.media_asset)
    scan = AdmittedDetectorScan.read(args.detector_scan)
    report = evaluate_household_sampled_coverage(
        annotation,
        scan,
        media_admission=media_admission,
        timeline_binding=timeline_binding,
        reference_sha256=reference_sha256,
        confidence_threshold=args.confidence_threshold,
    )
    _write_exclusive(
        json.dumps(
            report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        args.output,
        limit=MAX_HOUSEHOLD_TRIGGER_REPORT_BYTES,
    )
    _json_print(report)
    return 0


def command_validate_household_trigger_coverage(
    args: argparse.Namespace,
) -> int:
    report = HouseholdTriggerCoverageReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_compare_household_trigger_coverage(
    args: argparse.Namespace,
) -> int:
    reports = tuple(HouseholdTriggerCoverageReport.read(path) for path in args.paths)
    _json_print(compare_household_trigger_coverage(reports))
    return 0


def command_select_household_trigger_operating_point(
    args: argparse.Namespace,
) -> int:
    report = HouseholdTriggerCoverageReport.read(args.path)
    policy = HouseholdTriggerPolicy(
        minimum_danger_episode_recall=args.minimum_danger_episode_recall,
        minimum_person_episode_recall=args.minimum_person_episode_recall,
        minimum_positive_sample_recall=args.minimum_positive_sample_recall,
        maximum_detection_delay_seconds=args.maximum_detection_delay_seconds,
        maximum_longest_uncovered_positive_seconds=(
            args.maximum_longest_uncovered_positive_seconds
        ),
        maximum_negative_false_activation_rate=(args.maximum_negative_false_activation_rate),
        require_negative_samples=args.require_negative_samples,
    )
    selected, feasible_point_count = select_household_trigger_operating_point(
        report,
        policy,
    )
    _json_print(
        {
            "valid": True,
            "version": 1,
            "coverage_report_sha256": report.fingerprint,
            "model_id": report.model_id,
            "model_revision": report.model_revision,
            "model_sha256": report.model_sha256,
            "scenario_tags": list(report.scenario_tags),
            "policy": policy.to_dict(),
            "feasible_point_count": feasible_point_count,
            "selected_point": None if selected is None else selected.to_dict(),
            "selection_rule": HOUSEHOLD_TRIGGER_SELECTION_RULE,
            "dataset_specific_only": True,
            "deployment_approved": False,
            "household_configuration_changed": False,
            "end_to_end_latency_evaluated": False,
            "danger_recognition_accuracy_evaluated": False,
            "box_localization_accuracy_evaluated": False,
            "risk_accuracy_evaluated": False,
            "subject_identity_inference": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_validate_household_model_presence_review(
    args: argparse.Namespace,
) -> int:
    review = HouseholdModelPresenceReview.read(args.path)
    _json_print(review.to_public_dict())
    return 0


def command_evaluate_household_model_presence_diagnostic(
    args: argparse.Namespace,
) -> int:
    review = HouseholdModelPresenceReview.read(args.review)
    media_admissions = tuple(ExternalSourceAdmission.read(path) for path in args.media_admission)
    detector_scans = tuple(AdmittedDetectorScan.read(path) for path in args.detector_scan)
    report = evaluate_household_model_presence_diagnostic(
        review,
        media_admissions=media_admissions,
        detector_scans=detector_scans,
        thresholds=tuple(args.threshold),
        bin_seconds=args.bin_seconds,
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_household_model_presence_diagnostic(
    args: argparse.Namespace,
) -> int:
    report = HouseholdModelDiagnosticReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_compare_household_model_presence_diagnostics(
    args: argparse.Namespace,
) -> int:
    reports = tuple(HouseholdModelDiagnosticReport.read(path) for path in args.paths)
    _json_print(compare_household_model_presence_diagnostics(reports))
    return 0


def command_compare_household_model_diagnostic_batches(
    args: argparse.Namespace,
) -> int:
    baseline = HouseholdModelDiagnosticReport.read(args.baseline)
    current = HouseholdModelDiagnosticReport.read(args.current)
    policy = HouseholdModelDiagnosticDriftPolicy(
        maximum_precision_drop=args.maximum_precision_drop,
        maximum_recall_drop=args.maximum_recall_drop,
        maximum_specificity_drop=args.maximum_specificity_drop,
        maximum_partial_visibility_recall_drop=(args.maximum_partial_visibility_recall_drop),
        require_partial_visibility_samples=(not args.allow_no_partial_visibility_samples),
    )
    report = compare_household_model_diagnostic_batches(baseline, current, policy)
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_household_model_diagnostic_drift(
    args: argparse.Namespace,
) -> int:
    report = HouseholdModelDiagnosticDriftReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_select_household_model_presence_working_point(
    args: argparse.Namespace,
) -> int:
    report = HouseholdModelDiagnosticReport.read(args.path)
    policy = HouseholdModelDiagnosticPolicy(
        minimum_precision=args.minimum_precision,
        minimum_recall=args.minimum_recall,
        minimum_partial_visibility_recall=args.minimum_partial_visibility_recall,
        require_partial_visibility_samples=(not args.allow_no_partial_visibility_samples),
    )
    selected, feasible_point_count = select_household_model_diagnostic_working_point(
        report,
        policy,
    )
    _json_print(
        {
            "valid": True,
            "version": 1,
            "diagnostic_report_sha256": report.fingerprint,
            "model_id": report.model_id,
            "model_revision": report.model_revision,
            "model_sha256": report.model_sha256,
            "policy": policy.to_dict(),
            "feasible_point_count": feasible_point_count,
            "selected_point": None if selected is None else selected.to_dict(),
            "selection_rule": HOUSEHOLD_MODEL_DIAGNOSTIC_SELECTION_RULE,
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
            "deployment_approved": False,
            "household_configuration_changed": False,
            "dataset_specific_only": True,
            "subject_identity_inference": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_evaluate_household_model_temporal_diagnostic(
    args: argparse.Namespace,
) -> int:
    review = HouseholdModelPresenceReview.read(args.review)
    media_admissions = tuple(ExternalSourceAdmission.read(path) for path in args.media_admission)
    detector_scans = tuple(AdmittedDetectorScan.read(path) for path in args.detector_scan)
    report = evaluate_household_model_temporal_diagnostic(
        review,
        media_admissions=media_admissions,
        detector_scans=detector_scans,
        thresholds=tuple(args.threshold),
        hold_seconds_grid=tuple(args.hold_seconds),
        bin_seconds=args.bin_seconds,
    )
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_household_model_temporal_diagnostic(
    args: argparse.Namespace,
) -> int:
    report = HouseholdModelTemporalDiagnosticReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_select_household_model_temporal_working_point(
    args: argparse.Namespace,
) -> int:
    report = HouseholdModelTemporalDiagnosticReport.read(args.path)
    policy = HouseholdModelTemporalDiagnosticPolicy(
        minimum_precision=args.minimum_precision,
        minimum_recall=args.minimum_recall,
        minimum_positive_camera_recall=args.minimum_positive_camera_recall,
        minimum_clear_visibility_recall=args.minimum_clear_visibility_recall,
        minimum_partial_visibility_recall=args.minimum_partial_visibility_recall,
        maximum_negative_camera_false_positive_rate=(
            args.maximum_negative_camera_false_positive_rate
        ),
        maximum_hold_seconds=args.maximum_hold_seconds,
        require_clear_visibility_samples=args.require_clear_visibility_samples,
        require_partial_visibility_samples=(not args.allow_no_partial_visibility_samples),
        require_negative_camera_samples=(not args.allow_no_negative_camera_samples),
        require_mixed_labels_per_camera=(not args.allow_single_label_camera_strata),
    )
    selected, feasible_point_count = select_household_model_temporal_working_point(
        report,
        policy,
    )
    _json_print(
        {
            "valid": True,
            "version": 1,
            "diagnostic_report_sha256": report.fingerprint,
            "model_id": report.model_id,
            "model_revision": report.model_revision,
            "model_sha256": report.model_sha256,
            "policy": policy.to_dict(),
            "feasible_point_count": feasible_point_count,
            "selected_point": None if selected is None else selected.to_dict(),
            "selection_rule": HOUSEHOLD_MODEL_TEMPORAL_SELECTION_RULE,
            "temporal_aggregation_strategy": "causal_detection_hold",
            "causal_temporal_aggregation": True,
            "future_detections_used": False,
            "diagnostic_only": True,
            "shadow_only": True,
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
    )
    return 0


def command_combine_household_model_temporal_diagnostic_batches(
    args: argparse.Namespace,
) -> int:
    reports = tuple(HouseholdModelTemporalDiagnosticReport.read(path) for path in args.paths)
    report = combine_household_model_temporal_diagnostic_batches(reports)
    report.write_new(args.output)
    _json_print(report.to_public_dict())
    return 0


def command_validate_household_model_temporal_batch_diagnostic(
    args: argparse.Namespace,
) -> int:
    report = HouseholdModelTemporalBatchDiagnosticReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_select_household_model_temporal_batch_working_point(
    args: argparse.Namespace,
) -> int:
    report = HouseholdModelTemporalBatchDiagnosticReport.read(args.path)
    policy = HouseholdModelTemporalDiagnosticPolicy(
        minimum_precision=args.minimum_precision,
        minimum_recall=args.minimum_recall,
        minimum_positive_camera_recall=args.minimum_positive_camera_recall,
        minimum_clear_visibility_recall=args.minimum_clear_visibility_recall,
        minimum_partial_visibility_recall=args.minimum_partial_visibility_recall,
        maximum_negative_camera_false_positive_rate=(
            args.maximum_negative_camera_false_positive_rate
        ),
        maximum_hold_seconds=args.maximum_hold_seconds,
        require_clear_visibility_samples=args.require_clear_visibility_samples,
        require_partial_visibility_samples=(not args.allow_no_partial_visibility_samples),
        require_negative_camera_samples=(not args.allow_no_negative_camera_samples),
        require_mixed_labels_per_camera=(not args.allow_single_label_camera_strata),
    )
    selected, feasible_point_count = select_household_model_temporal_batch_working_point(
        report,
        policy,
    )
    _json_print(
        {
            "valid": True,
            "version": 1,
            "diagnostic_report_sha256": report.fingerprint,
            "input_batch_count": len(report.batches),
            "mixed_label_camera_count": report.mixed_label_camera_count,
            "model_id": report.model_id,
            "model_revision": report.model_revision,
            "model_sha256": report.model_sha256,
            "policy": policy.to_dict(),
            "feasible_point_count": feasible_point_count,
            "selected_point": None if selected is None else selected.to_dict(),
            "selection_rule": HOUSEHOLD_MODEL_TEMPORAL_SELECTION_RULE,
            "aggregation_strategy": report.aggregation_strategy,
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
    )
    return 0


def command_validate_detector_box_quality(args: argparse.Namespace) -> int:
    report = DetectorBoxQualityReport.read(args.path)
    _json_print(report.to_public_dict())
    return 0


def command_prepare_detector_tracking_batch(args: argparse.Namespace) -> int:
    timebase = args.timebase
    if timebase == "nominal":
        if args.source_fps is None or args.source_frame_count is None:
            raise ValueError("nominal timebase requires --source-fps and --source-frame-count")
    elif timebase == "actual_pts":
        if args.source_fps is not None or args.source_frame_count is not None:
            raise ValueError("actual_pts timebase forbids --source-fps and --source-frame-count")
    else:
        raise ValueError("detector tracking timebase is invalid")
    timeline_start = _aware_datetime(args.timeline_start, field="timeline start")
    media_admission = ExternalSourceAdmission.read(args.media_admission)
    media_admission.verify_asset(args.media_asset)
    detector_scan = AdmittedDetectorScan.read(args.detector_scan)
    if timebase == "actual_pts":
        batch = build_detector_sampled_tracking_batch(
            detector_scan,
            media_admission=media_admission,
            timeline_start=timeline_start,
        )
    else:
        batch = build_detector_tracking_batch(
            detector_scan,
            media_admission=media_admission,
            source_fps=args.source_fps,
            source_frame_count=args.source_frame_count,
            timeline_start=timeline_start,
        )
    batch.write_new(args.output)
    _json_print(
        {
            "valid": True,
            "version": 1,
            "detector_scan_sha256": detector_scan.fingerprint,
            "source_admission_sha256": media_admission.fingerprint,
            "detection_batch_sha256": batch.fingerprint,
            "camera_id": batch.camera_id,
            "detector_adapter": batch.detector_adapter,
            "frame_count": len(batch.frames),
            "frames_with_detections": sum(bool(frame.detections) for frame in batch.frames),
            "empty_frame_count": sum(not frame.detections for frame in batch.frames),
            "detection_count": sum(len(frame.detections) for frame in batch.frames),
            "timeline_is_ordering_anchor": True,
            "subject_identity_inference": False,
            "source_media_included": False,
            "paths_disclosed": False,
            "network_used": False,
            **(
                {
                    "timebase": "actual_pts",
                    "actual_samples_only": True,
                    "source_exhaustion_verified": False,
                    "accuracy_evaluated": False,
                }
                if timebase == "actual_pts"
                else {}
            ),
        }
    )
    return 0


def command_prepare_detector_tracking_evaluation(args: argparse.Namespace) -> int:
    ground_truth_batch = DetectionFrameBatch.read(args.ground_truth_frames)
    ground_truth_review = TrackerAssociationReview.read(args.ground_truth_review)
    ground_truth_admission = ExternalSourceAdmission.read(args.ground_truth_admission)
    media_admission = ExternalSourceAdmission.read(args.media_admission)
    media_admission.verify_asset(args.media_asset)
    detector_scan = AdmittedDetectorScan.read(args.detector_scan)
    quality_report = DetectorBoxQualityReport.read(args.quality_report)
    batch, review = build_detector_tracking_evaluation_set(
        ground_truth_batch,
        ground_truth_review,
        detector_scan,
        quality_report,
        ground_truth_admission=ground_truth_admission,
        media_admission=media_admission,
    )
    batch.write_new(args.frames_output)
    review.write_new(args.review_output)
    assigned_count = sum(assignment.review_state == "assigned" for assignment in review.assignments)
    false_positive_count = sum(
        assignment.review_state == "false_positive" for assignment in review.assignments
    )
    _json_print(
        {
            "valid": True,
            "version": 1,
            "detector_scan_sha256": detector_scan.fingerprint,
            "detector_quality_report_sha256": quality_report.fingerprint,
            "detection_batch_sha256": batch.fingerprint,
            "association_review_sha256": review.fingerprint,
            "frame_count": len(batch.frames),
            "detection_count": len(review.assignments),
            "assigned_detection_count": assigned_count,
            "false_positive_detection_count": false_positive_count,
            "missed_ground_truth_box_count": (quality_report.missed_ground_truth_box_count),
            "ground_truth_provenance": review.ground_truth_provenance,
            "detection_accuracy_reported_separately": True,
            "missed_detections_reported_separately": True,
            "subject_identity_inference": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_verify_yolo_pose_model(args: argparse.Namespace) -> int:
    manifest = YoloPoseModelManifest.read(args.manifest)
    verification = manifest.verify_weight(args.weights)
    _json_print(verification.to_public_dict())
    return 0
