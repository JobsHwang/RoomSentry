"""Pose scan and pose quality report commands."""

from __future__ import annotations

import argparse
import os
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
from ..adapters.pose import CommandYoloPoseEstimator
from ..pose_protocol import AdmittedPoseScan, PoseFrameBatch
from ..pose_quality import (
    PoseQualityConfig,
    PoseQualityReport,
    evaluate_pose_quality,
)
from ..source_admission import ExternalSourceAdmission
from ..yolo_pose_model import YoloPoseModelManifest


def _pose_quality_public_summary(report: PoseQualityReport) -> dict[str, object]:
    summary = report.summary
    return {
        "valid": True,
        "version": report.version,
        "camera_id": report.camera_id,
        "pose_adapter": report.pose_adapter,
        "model_id": report.model_id,
        "model_revision": report.model_revision,
        "model_sha256": report.model_sha256,
        "pose_batch_sha256": report.pose_batch_sha256,
        "evidence_status": summary.evidence_status,
        "frames": summary.frame_count,
        "poses": summary.pose_count,
        "usable_poses": summary.usable_pose_count,
        "matched_pose_pairs": summary.matched_pose_pairs,
        "comparable_pose_pairs": summary.comparable_pose_pairs,
        "image_displacement_median": summary.image_displacement_median,
        "shape_displacement_median": summary.shape_displacement_median,
        "anonymous_pairing_only": True,
        "subject_identity_inference": False,
        "posture_classification": False,
        "risk_inference": False,
        "source_media_included": False,
        "paths_disclosed": False,
    }


def _scan_pose_batch(args: argparse.Namespace) -> PoseFrameBatch:
    manifest = YoloPoseModelManifest.read(args.manifest)
    runner = Path(args.runner_executable).expanduser()
    if not runner.is_absolute() or not runner.is_file():
        raise ValueError("YOLO pose runner executable must be an absolute local file")
    estimator = CommandYoloPoseEstimator(
        model_path=args.weights,
        manifest_path=args.manifest,
        manifest=manifest,
        confidence=args.confidence,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        device=args.device,
        image_size=args.image_size,
        rotation_degrees=args.rotation_degrees,
        command=(str(runner), "-m", "baby_monitor.local_runners.yolo_pose", "{request}"),
        timeout_seconds=args.timeout_seconds,
    )
    return estimator.scan(LocalFileCamera(args.camera_id, args.source).source())


def _pose_scan_public_summary(
    batch: PoseFrameBatch,
    *,
    admitted_scan: AdmittedPoseScan | None = None,
) -> dict[str, object]:
    poses = tuple(pose for frame in batch.frames for pose in frame.poses)
    keypoints = tuple(keypoint for pose in poses for keypoint in pose.keypoints)
    summary: dict[str, object] = {
        "valid": True,
        "version": 1,
        "camera_id": batch.camera_id,
        "pose_adapter": batch.pose_adapter,
        "model_id": batch.model_id,
        "model_revision": batch.model_revision,
        "model_sha256": batch.model_sha256,
        "frames": len(batch.frames),
        "frames_with_poses": sum(bool(frame.poses) for frame in batch.frames),
        "poses": len(poses),
        "keypoints": len(keypoints),
        "in_frame_keypoints": sum(item.in_frame for item in keypoints),
        "subject_identity_inference": False,
        "posture_classification": False,
        "paths_disclosed": False,
    }
    if admitted_scan is not None:
        summary.update(
            {
                "source_admission_sha256": admitted_scan.source_admission_sha256,
                "scan_sha256": admitted_scan.fingerprint,
                "confidence_threshold": admitted_scan.confidence_threshold,
                "requested_sample_fps": admitted_scan.requested_sample_fps,
                "requested_max_frames": admitted_scan.requested_max_frames,
                "image_size": admitted_scan.image_size,
                "rotation_degrees": admitted_scan.rotation_degrees,
            }
        )
    return summary


def command_pose_scan_file(args: argparse.Namespace) -> int:
    batch = _scan_pose_batch(args)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{batch.to_json()}\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output, flags, 0o600)
    except FileExistsError:
        raise FileExistsError("YOLO pose output already exists") from None
    except OSError:
        raise RuntimeError("YOLO pose output could not be created") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
    except OSError:
        output.unlink(missing_ok=True)
        raise RuntimeError("YOLO pose output could not be written") from None
    summary = _pose_scan_public_summary(batch)
    summary["network_used"] = False
    _json_print(summary)
    return 0


def command_pose_scan_admitted_file(args: argparse.Namespace) -> int:
    admission = ExternalSourceAdmission.read(args.source_admission)
    if admission.asset_kind != "media_asset":
        raise ValueError("YOLO pose scan source admission must describe a media asset")
    admission.verify_asset(args.source)
    batch = _scan_pose_batch(args)
    scan = AdmittedPoseScan(
        source_admission_sha256=admission.fingerprint,
        confidence_threshold=args.confidence,
        requested_sample_fps=args.sample_fps,
        requested_max_frames=args.max_frames,
        image_size=args.image_size,
        rotation_degrees=args.rotation_degrees,
        result=batch,
    )
    scan.write_new(args.output)
    summary = _pose_scan_public_summary(batch, admitted_scan=scan)
    summary["network_used"] = False
    _json_print(summary)
    return 0


def command_validate_pose_frames(args: argparse.Namespace) -> int:
    batch = PoseFrameBatch.read(args.path)
    poses = tuple(pose for frame in batch.frames for pose in frame.poses)
    keypoints = tuple(keypoint for pose in poses for keypoint in pose.keypoints)
    _json_print(
        {
            "valid": True,
            "version": 1,
            "camera_id": batch.camera_id,
            "pose_adapter": batch.pose_adapter,
            "frames": len(batch.frames),
            "poses": len(poses),
            "keypoints": len(keypoints),
            "in_frame_keypoints": sum(item.in_frame for item in keypoints),
            "subject_identity_inference": False,
            "posture_classification": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_validate_admitted_pose_scan(args: argparse.Namespace) -> int:
    scan = AdmittedPoseScan.read(args.path)
    _json_print(_pose_scan_public_summary(scan.result, admitted_scan=scan))
    return 0


def command_evaluate_pose_quality(args: argparse.Namespace) -> int:
    batch = PoseFrameBatch.read(args.pose_frames)
    report = evaluate_pose_quality(
        batch,
        PoseQualityConfig(
            keypoint_confidence_threshold=args.keypoint_confidence,
            min_usable_keypoints=args.min_usable_keypoints,
            min_pair_bbox_iou=args.min_pair_bbox_iou,
            min_shared_keypoints=args.min_shared_keypoints,
            image_change_threshold=args.image_change_threshold,
            shape_change_threshold=args.shape_change_threshold,
        ),
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{report.to_json()}\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output, flags, 0o600)
    except FileExistsError:
        raise FileExistsError("pose quality output already exists") from None
    except OSError:
        raise RuntimeError("pose quality output could not be created") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
    except OSError:
        output.unlink(missing_ok=True)
        raise RuntimeError("pose quality output could not be written") from None
    _json_print(_pose_quality_public_summary(report))
    return 0


def command_validate_pose_quality(args: argparse.Namespace) -> int:
    report = PoseQualityReport.read(args.path)
    _json_print(_pose_quality_public_summary(report))
    return 0
