"""Release audit, SBOM, and deterministic bundle commands."""

from __future__ import annotations

import argparse

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

from ..release_audit import audit_release
from ..release_bundle import build_source_bundle
from ..release_sbom import source_sbom_json, write_source_sbom


def command_release_audit(args: argparse.Namespace) -> int:
    report = audit_release(
        args.root,
        max_file_bytes=args.max_file_bytes,
        source="worktree" if args.worktree else "git_index",
    )
    _json_print(report.to_dict())
    return 0 if report.passed else 1


def command_release_sbom(args: argparse.Namespace) -> int:
    if args.output:
        _json_print(write_source_sbom(args.root, args.output))
    else:
        print(source_sbom_json(args.root))
    return 0


def command_release_bundle(args: argparse.Namespace) -> int:
    _json_print(build_source_bundle(args.root, args.output))
    return 0
