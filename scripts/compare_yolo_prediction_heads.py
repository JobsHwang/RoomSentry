"""Bounded offline comparison of fresh YOLO26 prediction heads on reviewed stills.

Experimental only: no installed-package edits, training, uploads or production
configuration. Run as a module from a checkout. Each model/head has a separate
network-isolated child; a fused instance is never reused for another head.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

from baby_monitor.source_admission import ExternalSourceAdmission
from baby_monitor.yolo_model import YoloModelManifest
from baby_monitor.yolo_pose_model import YoloPoseModelManifest
from scripts.household_debug_common import (
    new_output_directory,
    private_directory,
    private_file,
    read_private_json_with_sha,
    runtime_versions,
    safe_id,
    write_new,
)

REPO = Path(__file__).resolve().parents[1]
MODELS = ("yolo26m", "yolo26m-pose")
ISOLATION = ("/usr/bin/unshare", "--user", "--map-root-user", "--net")
THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.35, 0.5)
RUNTIME_VERSION = "8.4.115"


def verify_frame(path: Path, digest: str) -> None:
    private_file(path, limit=20 * 1024 * 1024)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("reviewed frame bytes changed")


def head_state(model: object, expected: bool) -> dict[str, object]:
    """Verify effective branches after first setup/fusion, not just input args."""
    predictor = model.predictor
    head = model.model.model[-1]
    flags = {
        "predictor_argument": predictor.args.end2end,
        "autobackend": predictor.model.end2end,
        "backend_snapshot": predictor.model.backend.end2end,
        "wrapped_model": predictor.model.model.end2end,
        "wrapped_head": predictor.model.model.model[-1].end2end,
        "task_model": model.model.end2end,
        "task_head": head.end2end,
    }
    if any(type(value) is not bool or value is not expected for value in flags.values()):
        raise ValueError("effective prediction head does not match requested branch")
    one_to_many_present = all(getattr(head, key, None) is not None for key in ("cv2", "cv3"))
    if not expected and not one_to_many_present:
        raise ValueError("one-to-many heads are missing after model setup")
    if not expected and model.task == "pose" and getattr(head, "cv4", None) is None:
        raise ValueError("one-to-many pose head is missing after model setup")
    return {**flags, "one_to_many_heads_present": one_to_many_present}


def boxes_from_result(result: object) -> list[dict[str, object]]:
    boxes = result.boxes
    if boxes is None:
        return []
    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    coordinates = boxes.xyxyn.detach().cpu().tolist()
    if (
        not all(isinstance(value, list) for value in (classes, confidences, coordinates))
        or not len(classes) == len(confidences) == len(coordinates)
        or len(classes) > 300
    ):
        raise ValueError("prediction box arrays are inconsistent or unbounded")
    output = []
    for label, confidence, bbox in zip(classes, confidences, coordinates, strict=True):
        if not _unit_number(label) or label != 0:
            raise ValueError("prediction head returned a non-person class")
        if not _unit_number(confidence) or confidence < 0.1:
            raise ValueError("prediction confidence is invalid")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not _unit_number(value) for value in bbox)
            or bbox[0] >= bbox[2]
            or bbox[1] >= bbox[3]
        ):
            raise ValueError("prediction box coordinates are invalid")
        output.append({"confidence": float(confidence), "bbox_xyxy": bbox})
    return output


def _unit_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and 0 <= value <= 1
    except (OverflowError, ValueError):
        return False


def worker(request_path: Path) -> None:
    request, _ = read_private_json_with_sha(request_path)
    if (
        not isinstance(request, dict)
        or set(request)
        != {
            "version",
            "model_id",
            "weight_path",
            "weight_sha256",
            "end2end",
            "frames",
            "parent_net_namespace",
            "output",
            "device",
            "labels_source_sha256",
        }
        or type(request["version"]) is not int
        or request["version"] != 1
    ):
        raise ValueError("prediction head worker request is invalid")
    if (
        request["model_id"] not in MODELS
        or type(request["end2end"]) is not bool
        or request["device"] not in ("cpu", "0")
        or not isinstance(request["frames"], list)
        or not 1 <= len(request["frames"]) <= 32
        or os.readlink("/proc/self/ns/net") == request["parent_net_namespace"]
        or {name for _, name in socket.if_nameindex()} != {"lo"}
    ):
        raise ValueError("prediction head worker requires bounded isolated inputs")
    if importlib.metadata.version("ultralytics") != RUNTIME_VERSION:
        raise ValueError("prediction head diagnostic requires the reviewed runtime version")
    name = request["model_id"]
    manifest_class = YoloPoseModelManifest if name.endswith("-pose") else YoloModelManifest
    manifest = manifest_class.read(REPO / "config" / f"{name}-coco.reviewed.example.json")
    weight = Path(request["weight_path"])
    manifest.verify_weight(weight)
    if request["weight_sha256"] != manifest.sha256:
        raise ValueError("prediction head weight binding does not match")
    for frame in request["frames"]:
        if set(frame) != {"review_id", "frame_sha256", "path"}:
            raise ValueError("prediction head frame request is invalid")
        safe_id(frame["review_id"])
        verify_frame(Path(frame["path"]), frame["frame_sha256"])
    # Optional imports occur only after all provenance and namespace checks.
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(1)
    model = YOLO(str(weight))
    if model.predictor is not None or model.names.get(0) != "person":
        raise ValueError("prediction head requires a fresh person model")
    original_head = model.model.model[-1]
    expected_task = "pose" if name.endswith("-pose") else "detect"
    if model.task != expected_task or type(original_head).__name__ != (
        "Pose26" if expected_task == "pose" else "Detect"
    ):
        raise ValueError("prediction head checkpoint architecture is not the reviewed type")
    if any(
        getattr(original_head, key, None) is None
        for key in ("cv2", "cv3", "one2one_cv2", "one2one_cv3")
    ):
        raise ValueError("prediction head checkpoint does not preserve both detector branches")
    if expected_task == "pose" and any(
        getattr(original_head, key, None) is None for key in ("cv4", "one2one_cv4")
    ):
        raise ValueError("prediction head checkpoint does not preserve both pose branches")
    model.model.end2end = request["end2end"]
    model.add_callback("on_predict_start", lambda _predictor: head_state(model, request["end2end"]))
    rows = []
    state = None
    started = time.monotonic()
    for frame in request["frames"]:
        path = Path(frame["path"])
        verify_frame(path, frame["frame_sha256"])
        results = model.predict(
            source=str(path),
            conf=0.1,
            classes=[0],
            imgsz=640,
            device=request["device"],
            end2end=request["end2end"],
            iou=0.7,
            max_det=300,
            verbose=False,
            save=False,
            save_txt=False,
            save_crop=False,
            show=False,
            agnostic_nms=False,
            compile=False,
        )
        if len(results) != 1:
            raise ValueError("prediction head did not return exactly one still result")
        current = head_state(model, request["end2end"])
        if state is not None and current != state:
            raise ValueError("prediction head state changed within a branch")
        state = current
        rows.append(
            {
                "model_id": name,
                "model_sha256": manifest.sha256,
                "review_id": frame["review_id"],
                "frame_sha256": frame["frame_sha256"],
                "boxes": boxes_from_result(results[0]),
            }
        )
        verify_frame(path, frame["frame_sha256"])
    manifest.verify_weight(weight)
    package_root = Path(ultralytics.__file__).parent
    code_sha256s = {
        relative: hashlib.sha256((package_root / relative).read_bytes()).hexdigest()
        for relative in (
            "engine/predictor.py",
            "nn/tasks.py",
            "nn/modules/head.py",
            "nn/autobackend.py",
            "nn/backends/pytorch.py",
            "utils/nms.py",
        )
    }
    write_new(
        Path(request["output"]),
        {
            "rows": rows,
            "effective_head": state,
            "end2end": request["end2end"],
            "labels_source_sha256": request["labels_source_sha256"],
            "prediction_elapsed_seconds": time.monotonic() - started,
            "network_namespace_isolated": True,
            "fresh_model_instance": True,
            "model_weights_loaded": True,
            "new_pixels_uploaded": False,
            "nms_backend": "not_used_end2end" if request["end2end"] else "torchvision.ops.nms",
            "torchvision_version": torchvision.__version__,
            "runtime_code_sha256s": code_sha256s,
        },
    )


def run_child(arguments: tuple[str, ...], *, environment: dict[str, str]) -> float:
    started = time.monotonic()
    with subprocess.Popen(
        arguments,
        cwd=REPO,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    ) as process:
        try:
            code = process.wait(timeout=120)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code:
        raise ValueError("isolated prediction head child failed; no runtime logs disclosed")
    return time.monotonic() - started


def compare(
    *, source_root: Path, output_root: Path, vision_python: Path, weights: Path, device: str = "0"
) -> dict[str, object]:
    private_directory(source_root)
    new_output_directory(output_root)
    if device not in ("cpu", "0"):
        raise ValueError("prediction head device is invalid")
    selection, selection_sha = read_private_json_with_sha(
        source_root / "visual-review" / "selection-manifest.json"
    )
    labels, labels_sha = read_private_json_with_sha(
        source_root / "visual-review" / "assistant-frame-labels.json"
    )
    if (
        labels.get("model_generated_labels") is not True
        or labels.get("human_verified") is not False
        or labels.get("ground_truth") is not False
        or not 1 <= len(labels["items"]) <= 16
    ):
        raise ValueError("prediction head comparison requires bounded existing weak labels")
    frames, weak_labels, admissions = [], [], []
    for label in labels["items"]:
        ident, camera = safe_id(label["review_id"]), safe_id(label["camera_id"])
        matches = [item for item in selection["items"] if item["review_id"] == ident]
        if len(matches) != 1 or label["visible_person"] not in ("present", "absent", "unknown"):
            raise ValueError("reviewed prediction head selection is inconsistent")
        item = matches[0]
        if any(
            item[key] != label[key]
            for key in ("camera_id", "timestamp_seconds", "source_frame_sha256")
        ):
            raise ValueError("prediction head label/source binding does not match")
        filename = item["frame_filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("prediction head frame name is invalid")
        path = source_root / "visual-review" / "frames" / filename
        verify_frame(path, item["source_frame_sha256"])
        admission_path = source_root / "admissions" / f"{camera}.source-admission.json"
        private_file(admission_path)
        admission = ExternalSourceAdmission.read(admission_path)
        media = tuple(source_root.glob(f"*_{camera}_*.mp4"))
        if len(media) != 1 or admission.fingerprint != item["media_admission_sha256"]:
            raise ValueError("prediction head parent media binding does not match")
        private_file(media[0], limit=1024 * 1024 * 1024)
        admission.verify_asset(media[0])
        admissions.append((admission, media[0]))
        frames.append(
            {"review_id": ident, "frame_sha256": item["source_frame_sha256"], "path": str(path)}
        )
        weak_labels.append(
            {
                "review_id": ident,
                "frame_sha256": item["source_frame_sha256"],
                "presence": label["visible_person"],
            }
        )
    if len({frame["review_id"] for frame in frames}) != len(frames):
        raise ValueError("prediction head frame IDs must be unique")
    versions = runtime_versions(vision_python, ISOLATION)
    if versions["ultralytics"] != RUNTIME_VERSION:
        raise ValueError("prediction head comparison requires the reviewed runtime version")
    manifests = {}
    for name in MODELS:
        cls = YoloPoseModelManifest if name.endswith("-pose") else YoloModelManifest
        manifests[name] = cls.read(REPO / "config" / f"{name}-coco.reviewed.example.json")
        manifests[name].verify_weight(weights / f"{name}.pt")
    output_root.mkdir(mode=0o700)
    settings = output_root / "runtime-settings"
    settings.mkdir(mode=0o700)
    environment = {
        **os.environ,
        "YOLO_CONFIG_DIR": str(settings),
        "PYTHONPATH": str(REPO / "src"),
        "YOLO_AUTOINSTALL": "false",
        "YOLO_VERBOSE": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    plan = {
        "version": 1,
        "models": list(MODELS),
        "branches": ["end2end", "one_to_many"],
        "source_confidence": 0.1,
        "image_size": 640,
        "rotation": 0,
        "classes": [0],
        "iou": 0.7,
        "max_det": 300,
        "agnostic_nms": False,
        "compile": False,
        "device": device,
        "thresholds": list(THRESHOLDS),
        "labels_source_sha256": labels_sha,
        "selection_sha256": selection_sha,
        "frames": weak_labels,
        "weight_sha256s": {name: item.sha256 for name, item in manifests.items()},
        "runtime_versions": versions,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "derived_stills_not_native_video": True,
        "human_verified": False,
        "ground_truth": False,
        "prediction_head_selected_before_first_predict": True,
        "new_pixels_uploaded": False,
        "deployment_approved": False,
    }
    write_new(output_root / "plan.json", plan)
    branches = {name: [] for name in ("end2end", "one_to_many")}
    for name, manifest in manifests.items():
        for branch, flag in (("end2end", True), ("one_to_many", False)):
            result_path = output_root / f"{name}-{branch}.json"
            request_path = output_root / f"{name}-{branch}.request.json"
            write_new(
                request_path,
                {
                    "version": 1,
                    "model_id": name,
                    "weight_path": str(weights / f"{name}.pt"),
                    "weight_sha256": manifest.sha256,
                    "end2end": flag,
                    "frames": frames,
                    "parent_net_namespace": os.readlink("/proc/self/ns/net"),
                    "output": str(result_path),
                    "device": device,
                    "labels_source_sha256": labels_sha,
                },
            )
            elapsed = run_child(
                (
                    *ISOLATION,
                    str(vision_python),
                    "-m",
                    "scripts.compare_yolo_prediction_heads",
                    "--worker",
                    str(request_path),
                ),
                environment=environment,
            )
            result, _ = read_private_json_with_sha(result_path)
            branches[branch].extend(result["rows"])
            print(
                json.dumps(
                    {
                        "model": name,
                        "branch": branch,
                        "frames": len(result["rows"]),
                        "boxes": sum(len(row["boxes"]) for row in result["rows"]),
                        "process_elapsed_seconds": round(elapsed, 3),
                        "effective_head": result["effective_head"],
                    }
                ),
                flush=True,
            )
    for admission, media in admissions:
        admission.verify_asset(media)
    from scripts.prediction_head_diagnostic import diagnose_prediction_heads

    inputs = [
        {
            "branch_id": branch,
            "labels_source_sha256": labels_sha,
            "source_confidence": 0.1,
            "rows": tuple({**row, "boxes": tuple(row["boxes"])} for row in rows),
        }
        for branch, rows in branches.items()
    ]
    report = diagnose_prediction_heads(
        *inputs,
        weak_labels=tuple(weak_labels),
        label_source_sha256=labels_sha,
        thresholds=THRESHOLDS,
    )
    write_new(output_root / "comparison.json", report)
    return report


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--vision-python", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", choices=("cpu", "0"), default="0")
    args = parser.parse_args()
    try:
        if args.worker:
            worker(args.worker)
        else:
            if any(
                getattr(args, key) is None
                for key in ("source_root", "output_root", "vision_python", "weights")
            ):
                raise ValueError("prediction head comparison arguments are incomplete")
            compare(
                source_root=args.source_root,
                output_root=args.output_root,
                vision_python=args.vision_python,
                weights=args.weights,
                device=args.device,
            )
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(error).__name__, "details_disclosed": False}
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
