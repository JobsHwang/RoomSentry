"""Load already reviewed local YOLO weights sequentially on synthetic video.

Explicit opt-in invocation only. Linux network namespaces fail closed; no model
download, household media, CUDA, service startup, or accuracy evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.detector import CommandYoloDetector
from baby_monitor.adapters.pose import CommandYoloPoseEstimator
from baby_monitor.yolo_model import YoloModelManifest
from baby_monitor.yolo_pose_model import YoloPoseModelManifest


def smoke(*, detector_python: Path, pose_python: Path, weights: Path) -> dict[str, object]:
    unshare = shutil.which("unshare")
    if unshare is None:
        raise RuntimeError("Linux network namespace isolation is required")
    isolation = [unshare, "--user", "--map-root-user", "--net"]
    subprocess.run(isolation + ["true"], check=True, capture_output=True, timeout=5)
    repository = Path(__file__).resolve().parents[1]
    detector_manifest_path = repository / "config/yolo11n-coco.reviewed.example.json"
    pose_manifest_path = repository / "config/yolo11n-pose-coco.reviewed.example.json"
    detector_manifest = YoloModelManifest.read(detector_manifest_path)
    pose_manifest = YoloPoseModelManifest.read(pose_manifest_path)
    # Verify both before starting either model; constructors verify again.
    detector_manifest.verify_weight(weights / "yolo11n.pt")
    pose_manifest.verify_weight(weights / "yolo11n-pose.pt")
    with tempfile.TemporaryDirectory(prefix="baby-monitor-yolo-smoke-") as directory:
        video = Path(directory) / "synthetic_vfr.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x96:rate=10:duration=2",
                "-vf",
                "select=not(mod(n\\,3))+not(mod(n\\,7))",
                "-fps_mode",
                "vfr",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        source = LocalFileCamera("synthetic_cam", video).source()
        detector = CommandYoloDetector(
            model_path=weights / "yolo11n.pt",
            manifest_path=detector_manifest_path,
            manifest=detector_manifest,
            labels=("person",),
            confidence=0.35,
            sample_fps=2,
            device="cpu",
            image_size=320,
            timeout_seconds=45,
            command=isolation
            + [str(detector_python), "-m", "baby_monitor.local_runners.yolo_detector", "{request}"],
        )
        began = time.perf_counter()
        detections = detector.scan(source)
        detector_seconds = time.perf_counter() - began
        # scan() waits for child exit and validates strict JSON before returning.
        pose = CommandYoloPoseEstimator(
            model_path=weights / "yolo11n-pose.pt",
            manifest_path=pose_manifest_path,
            manifest=pose_manifest,
            confidence=0.35,
            sample_fps=2,
            max_frames=5,
            device="cpu",
            image_size=320,
            timeout_seconds=45,
            command=isolation
            + [str(pose_python), "-m", "baby_monitor.local_runners.yolo_pose", "{request}"],
        )
        began = time.perf_counter()
        batch = pose.scan(source)
        pose_seconds = time.perf_counter() - began
        expected = (0.0, 0.6, 0.9, 1.4, 1.8)
        if len(batch.frames) != len(expected) or not all(
            math.isclose(frame.timestamp_seconds, timestamp, abs_tol=0.00001, rel_tol=0)
            for frame, timestamp in zip(batch.frames, expected, strict=True)
        ):
            raise AssertionError("real pose runner differs from synthetic VFR fixture PTS")
    return {
        "version": 1,
        "device": "cpu",
        "models_loaded_sequentially": True,
        "network_isolation": "Linux user+network namespace per model subprocess",
        "strict_subprocess_json_validated": True,
        "detector": {
            "model_sha256": detector_manifest.sha256,
            "detections": len(detections),
            "elapsed_seconds": round(detector_seconds, 6),
        },
        "pose": {
            "model_sha256": pose_manifest.sha256,
            "frames": len(batch.frames),
            "poses": sum(len(frame.poses) for frame in batch.frames),
            "timestamps_seconds": [frame.timestamp_seconds for frame in batch.frames],
            "elapsed_seconds": round(pose_seconds, 6),
        },
        "household_media_read": False,
        "accuracy_evaluated": False,
        "temporary_synthetic_media_removed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-python", required=True, type=Path)
    parser.add_argument("--pose-python", required=True, type=Path)
    parser.add_argument("--weights-directory", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            smoke(
                detector_python=args.detector_python.absolute(),
                pose_python=args.pose_python.absolute(),
                weights=args.weights_directory.absolute(),
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
