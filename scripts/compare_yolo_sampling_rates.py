"""Bounded offline rate sweep over admitted, previously decoded household clips.

One fresh isolated model per source, one prediction per union frame. Timing is
of that union execution, NOT independent per-rate throughput or live latency.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

from baby_monitor.source_admission import ExternalSourceAdmission
from baby_monitor.yolo_model import YoloModelManifest
from scripts.analyze_pts_sampling_opportunity import _object_json, analyze
from scripts.compare_yolo_prediction_heads import ISOLATION, REPO, RUNTIME_VERSION, run_child
from scripts.compare_yolo_pts_sampling import (
    CAMERAS,
    REPORT_THRESHOLD,
    _diagnostic_code_sha256s,
    _sha,
    head_state,
    predict_union,
    summarize_schedule,
)
from scripts.household_debug_common import (
    new_output_directory,
    private_directory,
    private_file,
    runtime_versions,
    write_new,
)
from scripts.pts_sampling_diagnostic import _hash
from scripts.sampling_rate_diagnostic import plan_sampling_rates


def code_hashes() -> dict[str, str]:
    return {
        **_diagnostic_code_sha256s(),
        **{
            name: _sha(REPO / "scripts" / name)
            for name in (
                "compare_yolo_sampling_rates.py",
                "sampling_rate_diagnostic.py",
                "sampling_opportunity_diagnostic.py",
                "analyze_pts_sampling_opportunity.py",
            )
        },
    }


class TimedModel:
    """Synchronize actual GPU work; keep cold call separate from later calls."""

    def __init__(self, model, synchronize, clock=time.perf_counter):
        self.wrapped = model
        self.synchronize = synchronize
        self.clock = clock
        self.seconds = []

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def predict(self, **kwargs):
        self.synchronize()
        start = self.clock()
        result = self.wrapped.predict(**kwargs)
        self.synchronize()
        self.seconds.append(self.clock() - start)
        return result


def summarize_rates(plan: dict, reference: tuple, rows: list, seconds: list) -> dict:
    indices = plan["union_decoded_frame_indices"]
    selection = {
        "decoded_frame_indices": indices,
        "pts_seconds": [reference[i] for i in indices],
        "max_adjacent_sample_pts_gap_seconds": None,
    }
    # Reuse strict row/box validation, including zero-box calls and exact union.
    summarize_schedule(
        {"decoded_stride": selection, "source_pts_deadline": selection},
        rows,
        threshold=REPORT_THRESHOLD,
    )
    if (
        [row["decoded_frame_index"] for row in rows] != indices
        or not isinstance(seconds, list)
        or len(seconds) != len(rows)
        or any(
            isinstance(t, bool)
            or not isinstance(t, (int, float))
            or not 0 <= t <= 120
            or not math.isfinite(t)
            for t in seconds
        )
    ):
        raise ValueError("rate execution order/timing does not match union")
    by_index = {row["decoded_frame_index"]: row for row in rows}
    branches = {}
    for name, branch in plan["branches"].items():
        chosen = branch["selection"]
        branch_rows = [by_index[i] for i in chosen["decoded_frame_indices"]]
        result = summarize_schedule(
            {"decoded_stride": chosen, "source_pts_deadline": chosen},
            branch_rows,
            threshold=REPORT_THRESHOLD,
        )
        branches[name] = {
            "requested_sample_fps": branch["sample_fps"],
            **result["branches"]["source_pts_deadline"],
        }
    later = seconds[1:]
    return {
        "branches": branches,
        "unique_prediction_count": len(rows),
        "threshold": REPORT_THRESHOLD,
        "first_prediction_wall_seconds": seconds[0],
        "subsequent_prediction_count": len(later),
        "subsequent_prediction_median_seconds": statistics.median(later) if later else None,
        "subsequent_prediction_p95_nearest_rank_seconds": (
            sorted(later)[math.ceil(0.95 * len(later)) - 1] if later else None
        ),
        "sum_prediction_wall_seconds": math.fsum(seconds),
        "timing_includes_gpu_synchronization": True,
        "timing_excludes_decode_box_serialization_and_delivery": True,
        "independent_per_rate_throughput_measured": False,
        "live_latency_evaluated": False,
        "shared_frame_results_identical": True,
        "accuracy_evaluated": False,
        "ground_truth": False,
        "deployment_approved": False,
    }


def load_inputs(request: dict):
    expected = {
        "version",
        "camera_id",
        "source",
        "admission",
        "admission_sha256",
        "reference_artifact",
        "reference_artifact_sha256",
        "source_duration_seconds",
        "weight",
        "weight_sha256",
        "parent_net_namespace",
        "output_root",
        "code_sha256s",
    }
    if (
        not isinstance(request, dict)
        or set(request) != expected
        or type(request["version"]) is not int
        or request["version"] != 1
        or request["camera_id"] not in CAMERAS
        or request["code_sha256s"] != code_hashes()
    ):
        raise ValueError("rate request identity/code changed")
    source = Path(request["source"])
    private_file(source, limit=1024 * 1024 * 1024)
    admission_path = Path(request["admission"])
    private_file(admission_path)
    admission = ExternalSourceAdmission.read(admission_path)
    admission.verify_asset(source)
    frozen, frozen_sha = _object_json(Path(request["reference_artifact"]))
    if (
        admission.fingerprint != request["admission_sha256"]
        or frozen_sha != request["reference_artifact_sha256"]
        or frozen.get("source_admission_sha256") != admission.fingerprint
        or frozen.get("debug_code_sha256s") != _diagnostic_code_sha256s()
    ):
        raise ValueError("rate request source/reference binding changed")
    reference = tuple(frozen["reference_pts_seconds"])
    plan = plan_sampling_rates(
        reference,
        native_fps=frozen["plan"]["native_fps"],
        source_duration_seconds=request["source_duration_seconds"],
    )
    manifest = YoloModelManifest.read(REPO / "config/yolo26m-coco.reviewed.example.json")
    manifest.verify_weight(Path(request["weight"]))
    if manifest.sha256 != request["weight_sha256"]:
        raise ValueError("rate model binding changed")
    return source, admission, manifest, reference, plan


def worker(request_path: Path) -> None:
    request, request_sha = _object_json(request_path)
    if (
        os.readlink("/proc/self/ns/net") == request.get("parent_net_namespace")
        or {name for _, name in socket.if_nameindex()} != {"lo"}
        or importlib.metadata.version("ultralytics") != RUNTIME_VERSION
    ):
        raise ValueError("rate worker requires reviewed isolated runtime")
    source, admission, manifest, reference, plan = load_inputs(request)
    output = Path(request["output_root"])
    private_directory(output)
    write_new(
        output / "sampling-plan.json",
        {
            "request_sha256": request_sha,
            "plan": plan,
            "plan_frozen_before_inference": True,
            "reference_artifact_sha256": request["reference_artifact_sha256"],
            "code_sha256s": code_hashes(),
            "source_confidence": 0.1,
            "report_threshold": REPORT_THRESHOLD,
            "imgsz": 640,
            "classes": [0],
            "end2end": True,
            "rotation": 0,
            "iou": 0.7,
            "max_det": 300,
            "agnostic_nms": False,
            "compile": False,
            "device": "0",
        },
    )
    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(request["weight"])
    if (
        model.task != "detect"
        or model.model.model[-1].__class__.__name__ != "Detect"
        or model.predictor is not None
        or model.names.get(0) != "person"
    ):
        raise ValueError("rate worker loaded unexpected task/head")
    model.model.end2end = True
    model.model.model[-1].end2end = True
    timed = TimedModel(model, lambda: torch.cuda.synchronize(0))
    started = time.perf_counter()
    rows = predict_union(
        source,
        cv2_module=cv2,
        model=timed,
        reference_pts=reference,
        selected_indices=tuple(plan["union_decoded_frame_indices"]),
    )
    replay_seconds = time.perf_counter() - started
    admission.verify_asset(source)
    manifest.verify_weight(Path(request["weight"]))
    package_root = Path(importlib.metadata.distribution("ultralytics").locate_file("ultralytics"))
    runtime_hashes = {
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
            "prediction_wall_seconds": timed.seconds,
            "decode_and_union_replay_wall_seconds": replay_seconds,
            "summary": summarize_rates(plan, reference, rows, timed.seconds),
            "effective_head": head_state(model, True),
            "runtime_code_sha256s": runtime_hashes,
            "network_namespace_isolated": True,
            "media_bytes_verified_before_and_after": True,
            "full_reference_decode_count_and_pts_matched": True,
            "source_exhaustion_verified": False,
            "new_pixels_uploaded": False,
        },
    )


def compare(
    *,
    source_root: Path,
    experiment_root: Path,
    output_root: Path,
    vision_python: Path,
    weights: Path,
) -> dict:
    new_output_directory(output_root)
    output_root.mkdir(mode=0o700)
    # Admission, prior actual execution and time provenance are revalidated by
    # the existing reader, before any optional model runtime is started.
    audit = analyze(
        source_root=source_root,
        experiment_root=experiment_root,
        output_root=output_root / "prior-execution-audit",
    )
    versions = runtime_versions(vision_python, ISOLATION)
    if versions["ultralytics"] != RUNTIME_VERSION:
        raise ValueError("rate comparison requires reviewed runtime")
    manifest = YoloModelManifest.read(REPO / "config/yolo26m-coco.reviewed.example.json")
    weight = weights / "yolo26m.pt"
    manifest.verify_weight(weight)
    settings = output_root / "runtime-settings"
    settings.mkdir(mode=0o700)
    write_new(
        output_root / "plan.json",
        {
            "cameras": list(CAMERAS),
            "rates": [0.5, 1.0, 2.0, 4.0],
            "prior_execution_audit_sha256": audit["report_sha256"],
            "runtime_versions": versions,
            "code_sha256s": code_hashes(),
            "model_sha256": manifest.sha256,
            "child_timeout_seconds": 120,
            "maximum_union_predictions_per_source": 256,
            "duration_from_prior_probe_not_reprobed": True,
            "deployment_approved": False,
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
    for row in audit["rows"]:
        camera = row["camera_id"]
        old_request, _ = _object_json(experiment_root / camera / "request.json")
        output = output_root / camera
        output.mkdir(mode=0o700)
        request = {
            "version": 1,
            "camera_id": camera,
            "source": old_request["source"],
            "admission": old_request["admission"],
            "admission_sha256": row["source_admission_sha256"],
            "reference_artifact": str(experiment_root / camera / "sampling-plan.json"),
            "reference_artifact_sha256": row["frozen_plan_sha256"],
            "source_duration_seconds": row["opportunity"]["source_duration_seconds"],
            "weight": str(weight),
            "weight_sha256": manifest.sha256,
            "parent_net_namespace": os.readlink("/proc/self/ns/net"),
            "output_root": str(output),
            "code_sha256s": code_hashes(),
        }
        _, _, _, reference, plan = load_inputs(request)
        request_path = output / "request.json"
        write_new(request_path, request)
        elapsed = run_child(
            (
                *ISOLATION,
                str(vision_python),
                "-m",
                "scripts.compare_yolo_sampling_rates",
                "worker",
                str(request_path),
            ),
            environment=environment,
        )
        result, result_sha = _object_json(output / "result.json")
        frozen, _ = _object_json(output / "sampling-plan.json")
        replay_seconds = result.get("decode_and_union_replay_wall_seconds")
        if (
            isinstance(replay_seconds, bool)
            or not isinstance(replay_seconds, (int, float))
            or not 0 <= replay_seconds <= 120
        ):
            raise ValueError("rate replay timing is invalid")
        recomputed_summary = summarize_rates(
            plan, reference, result["rows"], result["prediction_wall_seconds"]
        )
        if recomputed_summary["sum_prediction_wall_seconds"] > replay_seconds + 1e-8:
            raise ValueError("rate prediction timing exceeds replay wall time")
        # JSON round-trip normalizes tuples to arrays without weakening fields.
        if (
            frozen.get("request_sha256") != _sha(request_path)
            or result.get("request_sha256") != _sha(request_path)
            or _hash(frozen.get("plan")) != _hash(plan)
            or frozen.get("plan_frozen_before_inference") is not True
            or frozen.get("code_sha256s") != code_hashes()
            or result.get("plan_sha256") != plan["report_sha256"]
            or result.get("network_namespace_isolated") is not True
            or result.get("media_bytes_verified_before_and_after") is not True
            or result.get("full_reference_decode_count_and_pts_matched") is not True
            or result.get("source_exhaustion_verified") is not False
            or result.get("new_pixels_uploaded") is not False
            or result.get("summary") != recomputed_summary
        ):
            raise ValueError("rate execution result binding changed")
        load_inputs(request)
        summaries.append(
            {
                "camera_id": camera,
                "result_sha256": result_sha,
                "process_elapsed_seconds": elapsed,
                "decode_and_union_replay_wall_seconds": result[
                    "decode_and_union_replay_wall_seconds"
                ],
                "opportunity": plan["opportunity"],
                **result["summary"],
            }
        )
    report = {
        "schema_version": "executed-sampling-rate-comparison-v1",
        "rows": summaries,
        "code_sha256s": code_hashes(),
        "prior_execution_audit_sha256": audit["report_sha256"],
        "duration_from_prior_probe_not_reprobed": True,
        "rate_selection_fixed_before_inference": True,
        "new_pixels_uploaded": False,
        "accuracy_evaluated": False,
        "event_recall_evaluated": False,
        "deployment_approved": False,
        "production_configuration_changed": False,
    }
    report = {**report, "report_sha256": _hash(report)}
    write_new(output_root / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("compare", "worker"))
    parser.add_argument("request", nargs="?")
    for option in ("source-root", "experiment-root", "output-root", "vision-python", "weights"):
        parser.add_argument(f"--{option}", type=Path)
    args = parser.parse_args()
    if args.action == "worker":
        worker(Path(args.request))
    else:
        report = compare(
            source_root=args.source_root,
            experiment_root=args.experiment_root,
            output_root=args.output_root,
            vision_python=args.vision_python,
            weights=args.weights,
        )
        print(
            {
                "status": "generated",
                "sources": len(report["rows"]),
                "report_sha256": report["report_sha256"],
                "event_recall_evaluated": False,
            }
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, TypeError, KeyError, subprocess.SubprocessError):
        print("sampling_rate_comparison_failed; no private logs disclosed", file=sys.stderr)
        raise SystemExit(2) from None
