"""Strict VLM event validation and sequential A/B evaluation commands."""

from __future__ import annotations

import argparse
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

from ..ab_eval import load_dataset, run_ab
from ..domain import (
    VLMEvent,
)


def command_validate_json(args: argparse.Namespace) -> int:
    raw = Path(args.path).read_text(encoding="utf-8")
    event = VLMEvent.from_json(raw)
    _json_print({"valid": True, "event": event.to_dict()})
    return 0


def command_ab_run(args: argparse.Namespace) -> int:
    artifacts = run_ab(args.config)
    _json_print(
        {
            "backends": artifacts.backend_count,
            "cases": artifacts.case_count,
            "results": str(artifacts.results_path),
            "summary": str(artifacts.summary_path),
        }
    )
    return 0


def command_validate_ab_dataset(args: argparse.Namespace) -> int:
    cases = load_dataset(args.path)
    _json_print(
        {
            "valid": True,
            "version": 1,
            "case_versions": sorted({case.schema_version for case in cases}),
            "cases": len(cases),
            "frames": sum(len(case.frames) for case in cases),
            "hashes_verified": True,
            "paths_disclosed": False,
        }
    )
    return 0
