"""Bind hypothetical short-window timing analysis to an executed local ablation.

Reads prior private artifacts and hashes admitted media bytes; never decodes,
loads a model, uploads, approves labels or changes a production sampling policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from baby_monitor.source_admission import ExternalSourceAdmission
from scripts.compare_yolo_pts_sampling import (
    CAMERAS,
    REPORT_THRESHOLD,
    _diagnostic_code_sha256s,
    _sha,
    summarize_schedule,
)
from scripts.household_debug_common import (
    new_output_directory,
    private_directory,
    private_file,
    read_private_json_with_sha,
    write_new,
)
from scripts.pts_sampling_diagnostic import _hash, plan_pts_sampling
from scripts.sampling_opportunity_diagnostic import evaluate_sampling_opportunity

EVENT_DURATIONS = (0.25, 0.5, 1.0, 2.0, 5.0)


def _object_json(path: Path) -> tuple[dict, str]:
    value, digest = read_private_json_with_sha(path)
    if not isinstance(value, dict):
        raise ValueError("sampling opportunity artifact must be an object")
    return value, digest


def analyze(*, source_root: Path, experiment_root: Path, output_root: Path) -> dict:
    private_directory(source_root)
    private_directory(experiment_root)
    new_output_directory(output_root)
    metadata, metadata_sha = _object_json(source_root / "capture-verification.json")
    comparison, comparison_sha = _object_json(experiment_root / "comparison.json")
    experiment_plan, experiment_plan_sha = _object_json(experiment_root / "plan.json")
    code_hashes = _diagnostic_code_sha256s()
    if (
        experiment_plan.get("cameras") != list(CAMERAS)
        or experiment_plan.get("debug_code_sha256s") != code_hashes
        or not isinstance(comparison.get("rows"), list)
        or len(comparison["rows"]) != len(CAMERAS)
        or comparison.get("accuracy_evaluated") is not False
        or comparison.get("deployment_approved") is not False
        or not isinstance(metadata.get("rows"), list)
        or not 1 <= len(metadata["rows"]) <= 32
        or any(not isinstance(item, dict) for item in (*metadata["rows"], *comparison["rows"]))
    ):
        raise ValueError("sampling opportunity experiment identity is invalid")
    rows = []
    for camera in CAMERAS:
        directory = experiment_root / camera
        request, request_sha = _object_json(directory / "request.json")
        frozen, frozen_sha = _object_json(directory / "sampling-plan.json")
        result, result_sha = _object_json(directory / "result.json")
        media = tuple(source_root.glob(f"*_{camera}_*.mp4"))
        expected_admission = source_root / "admissions" / f"{camera}.source-admission.json"
        if (
            len(media) != 1
            or request.get("camera_id") != camera
            or request.get("source") != str(media[0])
            or request.get("admission") != str(expected_admission)
            or frozen.get("request_sha256") != request_sha
            or result.get("request_sha256") != request_sha
            or frozen.get("debug_code_sha256s") != code_hashes
        ):
            raise ValueError("sampling opportunity request/source binding does not match")
        private_file(expected_admission)
        private_file(media[0], limit=1024 * 1024 * 1024)
        admission = ExternalSourceAdmission.read(expected_admission)
        verification = admission.verify_asset(media[0])
        matching_metadata = [item for item in metadata["rows"] if item["camera_id"] == camera]
        matching_comparison = [item for item in comparison["rows"] if item["camera_id"] == camera]
        if len(matching_metadata) != 1 or len(matching_comparison) != 1:
            raise ValueError("sampling opportunity requires one bound source row")
        source_info = matching_metadata[0]
        if (
            request.get("admission_sha256") != admission.fingerprint
            or frozen.get("source_admission_sha256") != admission.fingerprint
            or source_info["source_admission_sha256"] != admission.fingerprint
            or source_info["sha256"] != verification.sha256
            or type(source_info["bytes"]) is not int
            or source_info["bytes"] != verification.observed_bytes
            or matching_comparison[0]["result_sha256"] != result_sha
        ):
            raise ValueError(
                "sampling opportunity prior duration/media/result binding does not match"
            )
        reference = tuple(frozen["reference_pts_seconds"])
        plan = plan_pts_sampling(reference, native_fps=frozen["plan"]["native_fps"], sample_fps=0.5)
        if (
            plan != frozen["plan"]
            or result.get("plan_sha256") != plan["report_sha256"]
            or result.get("summary")
            != summarize_schedule(plan, result["rows"], threshold=REPORT_THRESHOLD)
            or result.get("source_exhaustion_verified") is not False
            or result.get("network_namespace_isolated") is not True
            or result.get("media_bytes_verified_before_and_after") is not True
        ):
            raise ValueError("sampling opportunity recorded execution does not match the plan")
        selections = {
            branch: tuple(plan[branch]["decoded_frame_indices"])
            for branch in ("decoded_stride", "source_pts_deadline")
        }
        # summarize_schedule requires the exact complete union of actual rows,
        # including zero-box calls. A plan alone is never accepted as execution.
        opportunity = evaluate_sampling_opportunity(
            reference,
            selections,
            source_duration_seconds=source_info["duration_seconds"],
            event_durations_seconds=EVENT_DURATIONS,
        )
        rows.append(
            {
                "camera_id": camera,
                "source_admission_sha256": admission.fingerprint,
                "request_sha256": request_sha,
                "frozen_plan_sha256": frozen_sha,
                "execution_result_sha256": result_sha,
                "opportunity": opportunity,
            }
        )
    report = {
        "schema_version": "executed-sampling-opportunity-v1",
        "rows": rows,
        "capture_verification_sha256": metadata_sha,
        "comparison_sha256": comparison_sha,
        "experiment_plan_sha256": experiment_plan_sha,
        "analysis_code_sha256s": {
            **code_hashes,
            "analyze_pts_sampling_opportunity.py": _sha(Path(__file__)),
            "sampling_opportunity_diagnostic.py": _sha(
                Path(__file__).with_name("sampling_opportunity_diagnostic.py")
            ),
        },
        "duration_from_prior_probe_not_reprobed": True,
        "recorded_execution_union_verified": True,
        "media_bytes_verified": True,
        "event_durations_fixed_before_source_analysis_seconds": EVENT_DURATIONS,
        "source_exhaustion_verified": False,
        "actual_event_labels_used": False,
        "accuracy_evaluated": False,
        "event_recall_evaluated": False,
        "model_loaded": False,
        "video_decoded": False,
        "new_pixels_uploaded": False,
        "deployment_approved": False,
        "production_configuration_changed": False,
    }
    report = {**report, "report_sha256": _hash(report)}
    output_root.mkdir(mode=0o700)
    write_new(output_root / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for option in ("source-root", "experiment-root", "output-root"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        source_root=args.source_root,
        experiment_root=args.experiment_root,
        output_root=args.output_root,
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
    except (ValueError, OSError, RuntimeError, TypeError, KeyError):
        raise SystemExit("sampling_opportunity_failed; no private logs disclosed") from None
