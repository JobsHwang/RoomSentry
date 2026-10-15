"""Compare cached model candidates on explicitly selected, hash-bound local stills.

This is a small, biased development diagnostic, not a held-out accuracy test.
No upload, training, automatic winner selection, or runtime reconfiguration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from household_debug_common import (
    new_output_directory,
    private_directory,
    private_file,
    read_private_json_with_sha,
    runtime_versions,
    safe_id,
    write_new,
)

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.detector import CommandYoloDetector
from baby_monitor.adapters.pose import CommandYoloPoseEstimator
from baby_monitor.source_admission import ExternalSourceAdmission
from baby_monitor.yolo_model import YoloModelManifest
from baby_monitor.yolo_pose_model import YoloPoseModelManifest

VARIANTS = (
    ("yolo11n", 640, False),
    ("yolo26m", 640, False),
    ("yolo26m", 960, False),
    ("yolo26m-pose", 640, True),
    ("yolo26m-pose", 960, True),
)
THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.35, 0.5)


def box_evidence(bbox: object) -> list[float]:
    """Retain adapter coordinates, rejecting invalid evidence instead of clipping it."""
    if (
        not isinstance(bbox, (tuple, list))
        or len(bbox) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
            or not math.isfinite(value)
            for value in bbox
        )
    ):
        raise ValueError("object bbox_xyxy must contain four finite normalized values")
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        raise ValueError("object bbox_xyxy must have positive normalized area")
    return [float(value) for value in bbox]


def verify_frame(path: Path, expected_sha256: str) -> None:
    private_file(path, limit=20 * 1024 * 1024)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("selected frame bytes do not match the review")


def compare(
    *,
    source_root: Path,
    labels_path: Path,
    output_root: Path,
    vision_python: Path,
    weights: Path,
    device: str,
) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    private_directory(source_root)
    new_output_directory(output_root)
    review_root = source_root / "visual-review"
    selection_path = review_root / "selection-manifest.json"
    selection, selection_sha = read_private_json_with_sha(selection_path)
    labels, labels_sha = read_private_json_with_sha(labels_path)
    if (
        labels.get("model_generated_labels") is not True
        or labels.get("human_verified") is not False
        or labels.get("ground_truth") is not False
        or not 1 <= len(labels["items"]) <= 3
    ):
        raise ValueError("only up to three explicit model-generated reference labels are allowed")
    frames = []
    seen = set()
    for item in labels["items"]:
        safe_id(item["review_id"])
        safe_id(item["camera_id"])
        if item["review_id"] in seen or item["visible_person"] not in (
            "present",
            "absent",
            "unknown",
        ):
            raise ValueError("reference labels are invalid")
        seen.add(item["review_id"])
        count = item.get("visible_person_count")
        if count is not None and (
            type(count) is not int
            or not 0 <= count <= 100
            or (item["visible_person"] == "present" and count == 0)
            or (item["visible_person"] == "absent" and count != 0)
            or item["visible_person"] == "unknown"
        ):
            raise ValueError("reference count is invalid or claims an unknown count")
        matches = [i for i in selection["items"] if i["review_id"] == item["review_id"]]
        if len(matches) != 1:
            raise ValueError("selected frame is not uniquely bound")
        source = matches[0]
        for key in ("camera_id", "timestamp_seconds", "source_frame_sha256"):
            if item[key] != source[key]:
                raise ValueError("label/source binding mismatch")
        filename = source["frame_filename"]
        if Path(filename).name != filename:
            raise ValueError("selected frame filename must be a basename")
        frame = review_root / "frames" / filename
        verify_frame(frame, item["source_frame_sha256"])
        camera = source["camera_id"]
        private_file(source_root / "admissions" / f"{camera}.source-admission.json")
        admission = ExternalSourceAdmission.read(
            source_root / "admissions" / f"{camera}.source-admission.json"
        )
        media = tuple(source_root.glob(f"*_{camera}_*.mp4"))
        if len(media) != 1 or admission.fingerprint != source["media_admission_sha256"]:
            raise ValueError("parent media admission does not match the selection")
        private_file(media[0], limit=1024 * 1024 * 1024)
        admission.verify_asset(media[0])
        frames.append((item, frame))
    unshare = shutil.which("unshare")
    if unshare is None:
        raise ValueError("network isolation is required")
    isolation = (unshare, "--user", "--map-root-user", "--net")
    subprocess.run((*isolation, "true"), check=True, capture_output=True, timeout=5)
    manifests = {}
    for model, _size, is_pose in VARIANTS:
        path = repository / "config" / f"{model}-coco.reviewed.example.json"
        manifest = (YoloPoseModelManifest if is_pose else YoloModelManifest).read(path)
        manifest.verify_weight(weights / f"{model}.pt")
        manifests[model] = (path, manifest)
    output_root.mkdir(mode=0o700)
    plan = {
        "version": 1,
        "purpose": "biased_selected_still_candidate_diagnostic",
        "variants": [list(item) for item in VARIANTS],
        "thresholds": list(THRESHOLDS),
        "source_threshold": 0.1,
        "device": device,
        "label_file_sha256": labels_sha,
        "selection_file_sha256": selection_sha,
        "runtime_versions": runtime_versions(vision_python, isolation),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_head_option": "runtime_and_weight_default_not_explicitly_overridden",
        "source_frame_sha256s": [item["source_frame_sha256"] for item, _ in frames],
        "model_sha256s": {model: manifest.sha256 for model, (_path, manifest) in manifests.items()},
        "derived_stills_not_native_video": True,
        "bbox_coordinate_space": "normalized_selected_still_xyxy",
        "bbox_reference_matching_performed": False,
        "sample_is_independent_holdout": False,
        "human_verified": False,
        "official_accuracy_claim": False,
        "winner_selected": False,
        "deployment_approved": False,
    }
    write_new(output_root / "comparison-plan.json", plan)
    rows = []
    for model, size, is_pose in VARIANTS:
        manifest_path, manifest = manifests[model]
        for item, frame in frames:
            verify_frame(frame, item["source_frame_sha256"])
            common = {
                "model_path": weights / f"{model}.pt",
                "manifest_path": manifest_path,
                "manifest": manifest,
                "confidence": 0.1,
                "sample_fps": 1.0,
                "device": device,
                "image_size": size,
                "timeout_seconds": 90,
                "command": (
                    *isolation,
                    str(vision_python),
                    "-m",
                    "baby_monitor.local_runners." + ("yolo_pose" if is_pose else "yolo_detector"),
                    "{request}",
                ),
            }
            source = LocalFileCamera(item["camera_id"], frame).source()
            started = time.perf_counter()
            if is_pose:
                result = CommandYoloPoseEstimator(**common, max_frames=1).scan(source)
                objects = [pose for batch_frame in result.frames for pose in batch_frame.poses]
                details = [
                    {
                        "confidence": p.confidence,
                        "bbox_xyxy": box_evidence(p.bbox_xyxy),
                        "keypoints_at_010": sum(
                            k.in_frame and k.confidence >= 0.1 for k in p.keypoints
                        ),
                    }
                    for p in objects
                ]
            else:
                objects = CommandYoloDetector(**common, labels=("person",)).scan(source)
                details = [
                    {
                        "confidence": detection.confidence,
                        "bbox_xyxy": box_evidence(detection.bbox_xyxy),
                    }
                    for detection in objects
                ]
            verify_frame(frame, item["source_frame_sha256"])
            row = {
                "model": model,
                "image_size": size,
                "camera_id": item["camera_id"],
                "review_id": item["review_id"],
                "source_timestamp_seconds": item["timestamp_seconds"],
                "source_frame_sha256": item["source_frame_sha256"],
                "bbox_coordinate_space": "normalized_selected_still_xyxy",
                "reference_visible_person": item["visible_person"],
                "reference_count": item.get("visible_person_count"),
                "objects": details,
                "counts_at_thresholds": {
                    str(t): sum(d["confidence"] >= t for d in details) for t in THRESHOLDS
                },
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            write_new(output_root / f"{model}-{size}-{item['review_id']}.json", row)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    result = {
        "plan": plan,
        "rows": rows,
        "model_network_isolated": True,
        "host_network_usage_measured": False,
        "paths_disclosed": False,
    }
    write_new(output_root / "comparison-result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-root", "labels-path", "output-root", "vision-python", "weights"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "0"), default="cpu")
    args = parser.parse_args()
    compare(
        **{
            name: value.absolute() if isinstance(value, Path) else value
            for name, value in vars(args).items()
        }
    )
