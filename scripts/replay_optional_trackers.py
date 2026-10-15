#!/usr/bin/env python3
"""Bounded offline box-only tracker replay in an opt-in vision runtime.

Run inside a network namespace with loopback only, YOLO_AUTOINSTALL=false and
YOLO_OFFLINE=true. Inputs and exclusive outputs must be private and Git-out.
No image/video input, weight download, service mutation or role assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

from baby_monitor.local_runners.tracker_replay import TrackerReplayConfig, replay_tracker
from baby_monitor.object_tracker import DetectionFrameBatch
from scripts.household_debug_common import (
    REPOSITORY,
    new_output_directory,
    read_private_json_with_sha,
    write_new,
)


def code_hashes():
    paths = (
        "scripts/replay_optional_trackers.py",
        "scripts/household_debug_common.py",
        "src/baby_monitor/local_runners/tracker_replay.py",
        "src/baby_monitor/object_tracker.py",
        "src/baby_monitor/tracker_evaluation/prediction.py",
    )
    return {path: hashlib.sha256((REPOSITORY / path).read_bytes()).hexdigest() for path in paths}


def run(batch_path: Path, config_path: Path, output: Path):
    new_output_directory(output)
    batch_value, batch_raw_sha = read_private_json_with_sha(batch_path)
    config_value, config_raw_sha = read_private_json_with_sha(config_path)
    batch = DetectionFrameBatch.from_dict(batch_value)
    if not isinstance(config_value, dict) or set(config_value) != {
        field.name for field in fields(TrackerReplayConfig)
    }:
        raise ValueError("offline tracker config must explicitly freeze every parameter")
    config = TrackerReplayConfig(**config_value)
    code_before = code_hashes()
    prediction, audit = replay_tracker(batch, config)
    if (
        read_private_json_with_sha(batch_path)[1] != batch_raw_sha
        or read_private_json_with_sha(config_path)[1] != config_raw_sha
        or code_hashes() != code_before
    ):
        raise ValueError("offline tracker inputs or adapter changed during replay")
    output.mkdir(mode=0o700)
    write_new(output / "predictions.local.json", prediction.to_dict())
    write_new(output / "audit.local.json", audit)
    write_new(
        output / "protocol.local.json",
        {
            "version": 1,
            "input_raw_sha256": batch_raw_sha,
            "config_raw_sha256": config_raw_sha,
            "detection_batch_sha256": batch.fingerprint,
            "config_sha256": config.fingerprint,
            "prediction_sha256": prediction.fingerprint,
            "code_sha256": code_before,
            "inputs_unchanged_before_after": True,
            "same_uid_concurrent_mutation_prevented": False,
            "config": config.to_dict(),
            "household_upload": False,
            "image_or_video_input": False,
            "runtime_id": prediction.runtime_id,
            "deployment_approved": False,
        },
    )
    return {
        "status": "completed_offline_replay",
        "algorithm": config.algorithm,
        "frames": audit["frame_count"],
        "assignments": len(prediction.assignments),
        "native_emitted": audit["native_emitted_assignment_count"],
        "native_tentative": audit["native_tentative_assignment_count"],
        "unassociated_singletons": audit["unassociated_singleton_count"],
        "detection_accuracy_evaluated": False,
        "deployment_approved": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        summary = run(args.batch, args.config, args.output)
    except Exception:
        # Do not leak private input paths or optional dependency tracebacks.
        print(json.dumps({"status": "offline_tracker_replay_failed"}))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
