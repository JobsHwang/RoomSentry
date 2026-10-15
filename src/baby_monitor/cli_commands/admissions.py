"""External source admission validation commands."""

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

from ..source_admission import ExternalSourceAdmission


def command_validate_source_admission(args: argparse.Namespace) -> int:
    record = ExternalSourceAdmission.read(args.path)
    _json_print(record.to_public_dict())
    return 0


def command_verify_source_asset(args: argparse.Namespace) -> int:
    record = ExternalSourceAdmission.read(args.manifest)
    verification = record.verify_asset(args.asset)
    _json_print(verification.to_public_dict())
    return 0
