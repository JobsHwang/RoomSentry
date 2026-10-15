"""Offline same-source sampling-schedule experiment; never a production ledger.

Decode twice: freeze a time-only plan before inference, then run one prediction
per frame in the union of both schedules. Shared frames have identical results.
Each source has a fresh, bounded, network-isolated model process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import socket
import subprocess
import sys
from pathlib import Path

from baby_monitor.detector_protocol import AdmittedDetectorScan
from baby_monitor.source_admission import ExternalSourceAdmission
from baby_monitor.yolo_model import YoloModelManifest
from scripts.compare_yolo_prediction_heads import (
    ISOLATION,
    REPO,
    RUNTIME_VERSION,
    _unit_number,
    boxes_from_result,
    head_state,
    run_child,
)
from scripts.household_debug_common import (
    new_output_directory,
    private_directory,
    private_file,
    read_private_json_with_sha,
    runtime_versions,
    write_new,
)
from scripts.pts_sampling_diagnostic import plan_pts_sampling

CAMERAS = ("camera_03", "camera_04")
MAX_FRAMES = 10_000
MAX_UNION_SAMPLES = 256
SOURCE_CONFIDENCE = 0.1
REPORT_THRESHOLD = 0.35


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic_code_sha256s() -> dict[str, str]:
    return {
        name: _sha(REPO / "scripts" / name)
        for name in (
            "compare_yolo_pts_sampling.py",
            "pts_sampling_diagnostic.py",
            "compare_yolo_prediction_heads.py",
            "household_debug_common.py",
        )
    }


def decode_timeline(source: Path, *, cv2_module: object) -> tuple[float, tuple[float, ...]]:
    capture = cv2_module.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError("PTS comparison could not open the source")
        native_fps = float(capture.get(cv2_module.CAP_PROP_FPS))
        pts = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or getattr(frame, "size", None) == 0 or len(pts) >= MAX_FRAMES:
                raise ValueError("PTS comparison decoded an empty or unbounded source")
            pts.append(float(capture.get(cv2_module.CAP_PROP_POS_MSEC)) / 1000)
        # Strict clock/native FPS checks are shared with the pure plan helper.
        plan_pts_sampling(tuple(pts), native_fps=native_fps, sample_fps=0.5)
        return native_fps, tuple(pts)
    finally:
        capture.release()


def predict_union(
    source: Path,
    *,
    cv2_module: object,
    model: object,
    reference_pts: tuple[float, ...],
    selected_indices: tuple[int, ...],
) -> list[dict[str, object]]:
    if (
        not selected_indices
        or len(selected_indices) > MAX_UNION_SAMPLES
        or tuple(sorted(set(selected_indices))) != selected_indices
        or any(
            type(index) is not int or not 0 <= index < len(reference_pts)
            for index in selected_indices
        )
    ):
        raise ValueError("PTS comparison selection is invalid")
    capture = cv2_module.VideoCapture(str(source))
    selected = set(selected_indices)
    rows = []
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError("PTS comparison could not reopen the source")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or getattr(frame, "size", None) == 0 or count >= len(reference_pts):
                raise ValueError("PTS comparison replay frame count changed")
            pts = float(capture.get(cv2_module.CAP_PROP_POS_MSEC)) / 1000
            if abs(pts - reference_pts[count]) > 0.00001 or not 0 <= pts <= 86400:
                raise ValueError("PTS comparison replay timestamp changed")
            if count in selected:
                predictions = model.predict(
                    source=frame,
                    conf=SOURCE_CONFIDENCE,
                    classes=[0],
                    imgsz=640,
                    device="0",
                    end2end=True,
                    iou=0.7,
                    max_det=300,
                    agnostic_nms=False,
                    compile=False,
                    verbose=False,
                    save=False,
                    save_txt=False,
                    save_crop=False,
                    show=False,
                )
                if not isinstance(predictions, (tuple, list)) or len(predictions) != 1:
                    raise ValueError("PTS comparison requires exactly one prediction")
                rows.append(
                    {
                        "decoded_frame_index": count,
                        "timestamp_seconds": pts,
                        "boxes": boxes_from_result(predictions[0]),
                    }
                )
                head_state(model, True)
            count += 1
        if count != len(reference_pts) or len(rows) != len(selected):
            raise ValueError("PTS comparison replay was truncated")
        return rows
    finally:
        capture.release()


def summarize_schedule(plan: dict, rows: list[dict], *, threshold: float) -> dict:
    if not _unit_number(threshold) or threshold < SOURCE_CONFIDENCE:
        raise ValueError("PTS comparison threshold is below the saved floor or invalid")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_UNION_SAMPLES:
        raise ValueError("PTS comparison prediction rows are invalid")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"decoded_frame_index", "timestamp_seconds", "boxes"}
            or type(row["decoded_frame_index"]) is not int
            or isinstance(row["timestamp_seconds"], bool)
            or not isinstance(row["timestamp_seconds"], (float, int))
            or not 0 <= row["timestamp_seconds"] <= 86400
            or not isinstance(row["boxes"], list)
            or len(row["boxes"]) > 300
        ):
            raise ValueError("PTS comparison prediction row is invalid")
        for box in row["boxes"]:
            if (
                not isinstance(box, dict)
                or set(box) != {"confidence", "bbox_xyxy"}
                or not _unit_number(box["confidence"])
                or box["confidence"] < SOURCE_CONFIDENCE
                or not isinstance(box["bbox_xyxy"], list)
                or len(box["bbox_xyxy"]) != 4
                or any(not _unit_number(value) for value in box["bbox_xyxy"])
                or box["bbox_xyxy"][0] >= box["bbox_xyxy"][2]
                or box["bbox_xyxy"][1] >= box["bbox_xyxy"][3]
            ):
                raise ValueError("PTS comparison prediction box is invalid")
    by_index = {row["decoded_frame_index"]: row for row in rows}
    union = set(plan["decoded_stride"]["decoded_frame_indices"]) | set(
        plan["source_pts_deadline"]["decoded_frame_indices"]
    )
    if len(by_index) != len(rows) or set(by_index) != union:
        raise ValueError("PTS comparison predictions do not match the full planned union")
    branches = {}
    for branch in ("decoded_stride", "source_pts_deadline"):
        indices = plan[branch]["decoded_frame_indices"]
        timestamps = plan[branch]["pts_seconds"]
        if any(
            abs(by_index[index]["timestamp_seconds"] - timestamp) > 0.00001
            for index, timestamp in zip(indices, timestamps, strict=True)
        ):
            raise ValueError("PTS comparison prediction timestamp is not plan-bound")
        counts = [
            sum(box["confidence"] >= threshold for box in by_index[index]["boxes"])
            for index in indices
        ]
        branches[branch] = {
            "sample_count": len(indices),
            "zero_person_box_samples": counts.count(0),
            "person_box_count": sum(counts),
            "max_adjacent_sample_pts_gap_seconds": plan[branch][
                "max_adjacent_sample_pts_gap_seconds"
            ],
        }
    return {
        "threshold": threshold,
        "branches": branches,
        "unique_prediction_count": len(rows),
        "shared_frame_results_identical": True,
        "accuracy_evaluated": False,
        "ground_truth": False,
        "deployment_approved": False,
    }


def worker(request_path: Path) -> None:
    request, request_sha = read_private_json_with_sha(request_path)
    expected = {
        "version",
        "camera_id",
        "source",
        "admission",
        "admission_sha256",
        "baseline_scan",
        "baseline_scan_sha256",
        "weight",
        "weight_sha256",
        "parent_net_namespace",
        "output_root",
    }
    if (
        not isinstance(request, dict)
        or set(request) != expected
        or type(request["version"]) is not int
        or request["version"] != 1
    ):
        raise ValueError("PTS comparison request is invalid")
    if (
        request["camera_id"] not in CAMERAS
        or os.readlink("/proc/self/ns/net") == request["parent_net_namespace"]
        or {name for _, name in socket.if_nameindex()} != {"lo"}
        or importlib.metadata.version("ultralytics") != RUNTIME_VERSION
    ):
        raise ValueError("PTS comparison requires the reviewed isolated runtime")
    source, baseline_path = Path(request["source"]), Path(request["baseline_scan"])
    admission_path, output = Path(request["admission"]), Path(request["output_root"])
    private_directory(output)
    private_file(source, limit=1024 * 1024 * 1024)
    private_file(admission_path)
    private_file(baseline_path)
    admission = ExternalSourceAdmission.read(admission_path)
    baseline = AdmittedDetectorScan.read(baseline_path)
    if (
        admission.fingerprint != request["admission_sha256"]
        or baseline.source_admission_sha256 != admission.fingerprint
        or baseline.result.camera_id != request["camera_id"]
        or baseline.version != 2
        or baseline.result.sampling is None
        or baseline.requested_sample_fps != 0.5
        or _sha(baseline_path) != request["baseline_scan_sha256"]
    ):
        raise ValueError("PTS comparison source/scan binding changed")
    admission.verify_asset(source)
    manifest = YoloModelManifest.read(REPO / "config/yolo26m-coco.reviewed.example.json")
    manifest.verify_weight(Path(request["weight"]))
    if manifest.sha256 != request["weight_sha256"]:
        raise ValueError("PTS comparison weight binding changed")

    import cv2
    from ultralytics import YOLO

    native_fps, pts = decode_timeline(source, cv2_module=cv2)
    plan = plan_pts_sampling(pts, native_fps=native_fps, sample_fps=0.5)
    ledger = baseline.result.sampling
    if (
        len(pts) != ledger.decoded_frame_count
        or abs(native_fps - ledger.native_fps) > 1e-9
        or ledger.native_fps_source != "reported"
        or ledger.timestamp_fallback_frame_count
        or ledger.container_pts_regression_count
        or plan["decoded_stride"]["decoded_frame_indices"]
        != [s.decoded_frame_index for s in ledger.samples]
        or any(
            s.timestamp_source != "container_pts"
            or abs(pts[s.decoded_frame_index] - s.timestamp_seconds) > 0.00001
            for s in ledger.samples
        )
    ):
        raise ValueError("PTS comparison reference no longer matches the baseline ledger")
    union = tuple(
        sorted(
            set(plan["decoded_stride"]["decoded_frame_indices"])
            | set(plan["source_pts_deadline"]["decoded_frame_indices"])
        )
    )
    if len(union) > MAX_UNION_SAMPLES:
        raise ValueError("PTS comparison planned inference exceeds the bound")
    write_new(
        output / "sampling-plan.json",
        {
            "request_sha256": request_sha,
            "plan": plan,
            "source_admission_sha256": admission.fingerprint,
            "model_sha256": manifest.sha256,
            "source_confidence": SOURCE_CONFIDENCE,
            "report_threshold": REPORT_THRESHOLD,
            "classes": [0],
            "imgsz": 640,
            "rotation": 0,
            "end2end": True,
            "iou": 0.7,
            "max_det": 300,
            "agnostic_nms": False,
            "compile": False,
            "device": "0",
            "union_prediction_count": len(union),
            "reference_pts_seconds": pts,
            "plan_frozen_before_inference": True,
            "runner_sha256": _sha(Path(__file__)),
            "planner_sha256": _sha(REPO / "scripts/pts_sampling_diagnostic.py"),
            "debug_code_sha256s": _diagnostic_code_sha256s(),
        },
    )
    model = YOLO(request["weight"])
    if (
        model.task != "detect"
        or model.model.model[-1].__class__.__name__ != "Detect"
        or model.predictor is not None
        or model.names.get(0) != "person"
    ):
        raise ValueError("PTS comparison loaded unexpected model task/head")
    model.model.end2end = True
    model.model.model[-1].end2end = True
    rows = predict_union(
        source, cv2_module=cv2, model=model, reference_pts=pts, selected_indices=union
    )
    admission.verify_asset(source)
    manifest.verify_weight(Path(request["weight"]))
    package_root = Path(importlib.metadata.distribution("ultralytics").locate_file("ultralytics"))
    runtime_code_sha256s = {
        name: _sha(package_root / name)
        for name in (
            "engine/predictor.py",
            "nn/tasks.py",
            "nn/modules/head.py",
            "nn/autobackend.py",
            "nn/backends/pytorch.py",
            "utils/nms.py",
        )
    }
    write_new(
        output / "result.json",
        {
            "request_sha256": request_sha,
            "plan_sha256": plan["report_sha256"],
            "rows": rows,
            "summary": summarize_schedule(plan, rows, threshold=REPORT_THRESHOLD),
            "effective_head": head_state(model, True),
            "source_exhaustion_verified": False,
            "new_pixels_uploaded": False,
            "network_namespace_isolated": True,
            "media_bytes_verified_before_and_after": True,
            "runtime_code_sha256s": runtime_code_sha256s,
        },
    )


def compare(*, source_root: Path, output_root: Path, vision_python: Path, weights: Path) -> dict:
    private_directory(source_root)
    new_output_directory(output_root)
    versions = runtime_versions(vision_python, ISOLATION)
    if versions["ultralytics"] != RUNTIME_VERSION:
        raise ValueError("PTS comparison requires the reviewed runtime")
    manifest = YoloModelManifest.read(REPO / "config/yolo26m-coco.reviewed.example.json")
    weight = weights / "yolo26m.pt"
    manifest.verify_weight(weight)
    output_root.mkdir(mode=0o700)
    settings = output_root / "runtime-settings"
    settings.mkdir(mode=0o700)
    write_new(
        output_root / "plan.json",
        {
            "cameras": list(CAMERAS),
            "runtime_versions": versions,
            "model_sha256": manifest.sha256,
            "sample_fps": 0.5,
            "source_confidence": SOURCE_CONFIDENCE,
            "report_threshold": REPORT_THRESHOLD,
            "runner_sha256": _sha(Path(__file__)),
            "debug_code_sha256s": _diagnostic_code_sha256s(),
            "new_pixels_uploaded": False,
            "deployment_approved": False,
            "child_timeout_seconds": 120,
        },
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "YOLO_CONFIG_DIR": str(settings),
        "YOLO_AUTOINSTALL": "false",
        "YOLO_VERBOSE": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    summaries = []
    for camera in CAMERAS:
        media = tuple(source_root.glob(f"*_{camera}_*.mp4"))
        if len(media) != 1:
            raise ValueError("PTS comparison requires one bound source per camera")
        admission_path = source_root / "admissions" / f"{camera}.source-admission.json"
        private_file(admission_path)
        private_file(media[0], limit=1024 * 1024 * 1024)
        admission = ExternalSourceAdmission.read(admission_path)
        admission.verify_asset(media[0])
        baseline = source_root / "sampling-ledger-v2" / f"{camera}.scan.json"
        private_file(baseline)
        output = output_root / camera
        output.mkdir(mode=0o700)
        request = output / "request.json"
        write_new(
            request,
            {
                "version": 1,
                "camera_id": camera,
                "source": str(media[0]),
                "admission": str(admission_path),
                "admission_sha256": admission.fingerprint,
                "baseline_scan": str(baseline),
                "baseline_scan_sha256": _sha(baseline),
                "weight": str(weight),
                "weight_sha256": manifest.sha256,
                "parent_net_namespace": os.readlink("/proc/self/ns/net"),
                "output_root": str(output),
            },
        )
        elapsed = run_child(
            (
                *ISOLATION,
                str(vision_python),
                "-m",
                "scripts.compare_yolo_pts_sampling",
                "worker",
                str(request),
            ),
            environment=environment,
        )
        result, result_sha = read_private_json_with_sha(output / "result.json")
        frozen, _ = read_private_json_with_sha(output / "sampling-plan.json")
        recomputed_plan = plan_pts_sampling(
            tuple(frozen["reference_pts_seconds"]),
            native_fps=frozen["plan"]["native_fps"],
            sample_fps=0.5,
        )
        if (
            result["request_sha256"] != _sha(request)
            or frozen["request_sha256"] != _sha(request)
            or result["plan_sha256"] != frozen["plan"]["report_sha256"]
            or recomputed_plan != frozen["plan"]
            or frozen["runner_sha256"] != _sha(Path(__file__))
            or frozen["planner_sha256"] != _sha(REPO / "scripts/pts_sampling_diagnostic.py")
            or frozen["debug_code_sha256s"] != _diagnostic_code_sha256s()
            or result["summary"]
            != summarize_schedule(frozen["plan"], result["rows"], threshold=REPORT_THRESHOLD)
        ):
            raise ValueError("PTS comparison child result/plan binding is invalid")
        admission.verify_asset(media[0])
        summaries.append(
            {
                "camera_id": camera,
                "result_sha256": result_sha,
                "process_elapsed_seconds": elapsed,
                **result["summary"],
            }
        )
    report = {
        "rows": summaries,
        "accuracy_evaluated": False,
        "deployment_approved": False,
        "new_pixels_uploaded": False,
        "production_configuration_changed": False,
    }
    write_new(output_root / "comparison.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("compare", "worker"))
    parser.add_argument("request", nargs="?")
    for option in ("source-root", "output-root", "vision-python", "weights"):
        parser.add_argument(f"--{option}", type=Path)
    args = parser.parse_args()
    if args.action == "worker":
        worker(Path(args.request))
    else:
        print(
            compare(
                source_root=args.source_root,
                output_root=args.output_root,
                vision_python=args.vision_python,
                weights=args.weights,
            )
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, TypeError, KeyError, subprocess.SubprocessError):
        print("pts_sampling_comparison_failed; no private logs disclosed", file=sys.stderr)
        raise SystemExit(2) from None
