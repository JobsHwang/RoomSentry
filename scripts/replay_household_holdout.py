"""Offline frozen-parameter regression on an existing admitted household holdout.

Explicit invocation reads private local media. It does not transfer pixels,
select thresholds, replace prior evidence, or authorize a deployment. Run with
the core environment; model interpreters must already exist and stay offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

from household_debug_common import (
    new_output_directory,
    private_directory,
    private_file,
    runtime_versions,
    write_new,
)

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.detector import CommandYoloDetector
from baby_monitor.adapters.pose import CommandYoloPoseEstimator
from baby_monitor.detector_protocol import AdmittedDetectorScan, DetectorScanResult
from baby_monitor.household_human_presence_audit import (
    HouseholdHumanPresenceAudit,
    evaluate_household_human_pose_fusion_audit,
)
from baby_monitor.household_model_diagnostic import HouseholdModelPresenceReview
from baby_monitor.household_pose_fusion_diagnostic import household_pose_fusion_bin_prediction
from baby_monitor.household_pose_fusion_holdout import (
    HouseholdPoseFusionTemporalHoldoutAttestation,
)
from baby_monitor.pose_protocol import AdmittedPoseScan
from baby_monitor.source_admission import ExternalSourceAdmission
from baby_monitor.yolo_model import YoloModelManifest
from baby_monitor.yolo_pose_model import YoloPoseModelManifest


def read_private(kind, path):
    private_file(path)
    return kind.read(path)


def confusion(labels: list[tuple[str, bool]]) -> dict[str, int]:
    return {
        "tp": sum(label == "present" and prediction for label, prediction in labels),
        "fn": sum(label == "present" and not prediction for label, prediction in labels),
        "fp": sum(label == "absent" and prediction for label, prediction in labels),
        "tn": sum(label == "absent" and not prediction for label, prediction in labels),
        "unknown_excluded": sum(label == "unknown" for label, _prediction in labels),
    }


def replay(
    *, source_root: Path, output_root: Path, vision_python: Path, weights: Path, device: str
) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    private_directory(source_root)
    new_output_directory(output_root)
    unshare = shutil.which("unshare")
    if unshare is None:
        raise ValueError("network isolation is required")
    isolation = (unshare, "--user", "--map-root-user", "--net")
    subprocess.run((*isolation, "true"), check=True, capture_output=True, timeout=5)
    attestation = read_private(
        HouseholdPoseFusionTemporalHoldoutAttestation,
        source_root / "household-pose-fusion-temporal-holdout-attestation-v1.json",
    )
    audit = read_private(
        HouseholdHumanPresenceAudit,
        source_root
        / "visual-review/independent-human-audit/household-human-presence-audit-v1.json",
    )
    review = read_private(
        HouseholdModelPresenceReview,
        source_root / "visual-review/household-model-presence-review-v1.json",
    )
    cameras = tuple(item.camera_id for item in attestation.holdout_report.profile.cameras)
    admissions = tuple(
        read_private(
            ExternalSourceAdmission, source_root / "admissions" / f"{camera}.source-admission.json"
        )
        for camera in cameras
    )
    old_detectors = tuple(
        read_private(
            AdmittedDetectorScan, source_root / "detector-scans" / f"{camera}.detector-scan.json"
        )
        for camera in cameras
    )
    old_poses = tuple(
        read_private(
            AdmittedPoseScan, source_root / "pose-scans" / f"{camera}.admitted-pose-scan.json"
        )
        for camera in cameras
    )
    baseline = evaluate_household_human_pose_fusion_audit(
        audit,
        model_review=review,
        temporal_holdout_attestation=attestation,
        media_admissions=admissions,
        detector_scans=old_detectors,
        pose_scans=old_poses,
    )
    detector_path = repository / "config/yolo26m-coco.reviewed.example.json"
    pose_path = repository / "config/yolo26m-pose-coco.reviewed.example.json"
    detector_manifest = YoloModelManifest.read(detector_path)
    pose_manifest = YoloPoseModelManifest.read(pose_path)
    detector_manifest.verify_weight(weights / "yolo26m.pt")
    pose_manifest.verify_weight(weights / "yolo26m-pose.pt")
    sources = []
    for camera, admission, old_d, old_p in zip(
        cameras, admissions, old_detectors, old_poses, strict=True
    ):
        matches = tuple(source_root.glob(f"*_{camera}_*.mp4"))
        if len(matches) != 1:
            raise ValueError("expected exactly one admitted local source per camera")
        private_file(matches[0], limit=1024 * 1024 * 1024)
        admission.verify_asset(matches[0])
        if (
            old_d.result.model_sha256 != detector_manifest.sha256
            or old_p.result.model_sha256 != pose_manifest.sha256
            or old_d.confidence_threshold != 0.1
            or old_p.confidence_threshold != 0.1
            or old_d.requested_sample_fps != 0.5
            or old_p.requested_sample_fps != 0.5
            or old_p.image_size != 640
            or old_p.rotation_degrees != 0
            or old_p.requested_max_frames != 128
        ):
            raise ValueError("holdout is not compatible with this frozen replay recipe")
        sources.append(matches[0])
    source_files = (
        "src/baby_monitor/adapters/detector.py",
        "src/baby_monitor/local_runners/yolo_detector.py",
        "src/baby_monitor/local_runners/yolo_pose.py",
        "src/baby_monitor/video_timeline.py",
        "scripts/replay_household_holdout.py",
        "scripts/household_debug_common.py",
    )
    code_hashes = {
        name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in source_files
    }
    output_root.mkdir(mode=0o700)
    plan = {
        "version": 1,
        "purpose": "frozen_holdout_runtime_regression",
        "baseline_report_sha256": baseline.fingerprint,
        "audit_sha256": audit.fingerprint,
        "attestation_sha256": attestation.fingerprint,
        "profile_sha256": attestation.holdout_report.profile.fingerprint,
        "media_admission_sha256s": [item.fingerprint for item in admissions],
        "model_sha256s": [detector_manifest.sha256, pose_manifest.sha256],
        "confidence": 0.1,
        "sample_fps": 0.5,
        "image_size": 640,
        "rotation_degrees": 0,
        "pose_max_frames": 128,
        "device": device,
        "historical_detector_transform_source": "historical verification prose; v1 scan omits it",
        "code_sha256s": code_hashes,
        "runtime_versions": runtime_versions(vision_python, isolation),
        "prediction_head_option": "runtime_and_weight_default_not_explicitly_overridden",
        "network_isolation": "user_and_network_namespace",
        "threshold_selection_performed": False,
        "deployment_approved": False,
    }
    write_new(output_root / "replay-plan.json", plan)
    rows = []
    predictions = []
    for camera, admission, source_path, old_d, old_p, profile in zip(
        cameras,
        admissions,
        sources,
        old_detectors,
        old_poses,
        attestation.holdout_report.profile.cameras,
        strict=True,
    ):
        if any(
            hashlib.sha256((repository / name).read_bytes()).hexdigest() != digest
            for name, digest in code_hashes.items()
        ):
            raise ValueError("production scan code changed during replay")
        private_file(source_path, limit=1024 * 1024 * 1024)
        admission.verify_asset(source_path)
        source = LocalFileCamera(camera, source_path).source()
        detector = CommandYoloDetector(
            model_path=weights / "yolo26m.pt",
            manifest_path=detector_path,
            manifest=detector_manifest,
            labels=("person",),
            confidence=0.1,
            sample_fps=0.5,
            device=device,
            image_size=640,
            rotation_degrees=0,
            timeout_seconds=240,
            command=(
                *isolation,
                str(vision_python),
                "-m",
                "baby_monitor.local_runners.yolo_detector",
                "{request}",
            ),
        )
        start = time.perf_counter()
        detections = tuple(detector.scan(source))
        detector_elapsed = time.perf_counter() - start
        admission.verify_asset(source_path)
        new_d = AdmittedDetectorScan(
            source_admission_sha256=admission.fingerprint,
            labels=("person",),
            confidence_threshold=0.1,
            requested_sample_fps=0.5,
            result=DetectorScanResult(
                camera_id=camera,
                detector_adapter=detector.runner_adapter,
                model_id=detector_manifest.model_id,
                model_revision=detector_manifest.revision,
                model_sha256=detector_manifest.sha256,
                detections=detections,
            ),
        )
        new_d.write_new(output_root / f"{camera}.detector-scan.json")
        pose = CommandYoloPoseEstimator(
            model_path=weights / "yolo26m-pose.pt",
            manifest_path=pose_path,
            manifest=pose_manifest,
            confidence=0.1,
            sample_fps=0.5,
            max_frames=128,
            device=device,
            image_size=640,
            rotation_degrees=0,
            timeout_seconds=240,
            command=(
                *isolation,
                str(vision_python),
                "-m",
                "baby_monitor.local_runners.yolo_pose",
                "{request}",
            ),
        )
        admission.verify_asset(source_path)
        start = time.perf_counter()
        batch = pose.scan(source)
        pose_elapsed = time.perf_counter() - start
        admission.verify_asset(source_path)
        new_p = AdmittedPoseScan(
            source_admission_sha256=admission.fingerprint,
            confidence_threshold=0.1,
            requested_sample_fps=0.5,
            requested_max_frames=128,
            image_size=640,
            rotation_degrees=0,
            result=batch,
        )
        new_p.write_new(output_root / f"{camera}.admitted-pose-scan.json")
        labels = []
        changed = 0
        for item in audit.items:
            if item.camera_id != camera:
                continue
            options = {"bin_index": item.bin_index, "bin_seconds": audit.bin_seconds}
            before = household_pose_fusion_bin_prediction(profile, old_d, old_p, **options)
            after = household_pose_fusion_bin_prediction(profile, new_d, new_p, **options)
            labels.append((item.visible_person, after))
            changed += before != after
        predictions.extend(labels)
        row = {
            "camera_id": camera,
            "confusion": confusion(labels),
            "changed_predictions": changed,
            "detector_boxes": len(detections),
            "previous_detector_boxes": len(old_d.result.detections),
            "detector_scan_unchanged": new_d.fingerprint == old_d.fingerprint,
            "pose_frames": len(batch.frames),
            "pose_scan_unchanged": new_p.fingerprint == old_p.fingerprint,
            "poses": sum(len(frame.poses) for frame in batch.frames),
            "pose_max_timestamp_shift_seconds": max(
                (
                    abs(a.timestamp_seconds - b.timestamp_seconds)
                    for a, b in zip(batch.frames, old_p.result.frames, strict=True)
                ),
                default=0.0,
            ),
            "detector_seconds": round(detector_elapsed, 3),
            "pose_seconds": round(pose_elapsed, 3),
            "detector_scan_sha256": new_d.fingerprint,
            "pose_scan_sha256": new_p.fingerprint,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    if any(
        hashlib.sha256((repository / name).read_bytes()).hexdigest() != digest
        for name, digest in code_hashes.items()
    ):
        raise ValueError("production scan code changed during replay")
    result = {
        "version": 1,
        "purpose": plan["purpose"],
        "plan": plan,
        "cameras": rows,
        "confusion": confusion(predictions),
        "single_human_reference": True,
        "camera_label_confounding": True,
        "official_accuracy_claim": False,
        "new_temporal_holdout_attestation": False,
        "deployment_approved": False,
        "source_media_included": False,
        "paths_disclosed": False,
    }
    write_new(output_root / "replay-result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "output-root", "vision-python", "weights"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "0"), default="cpu")
    args = parser.parse_args()
    result = replay(
        source_root=args.source_root.absolute(),
        output_root=args.output_root.absolute(),
        vision_python=args.vision_python.absolute(),
        weights=args.weights.absolute(),
        device=args.device,
    )
    print(json.dumps({"completed": True, "confusion": result["confusion"]}, sort_keys=True))


if __name__ == "__main__":
    main()
