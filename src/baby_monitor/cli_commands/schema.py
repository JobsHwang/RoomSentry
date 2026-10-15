"""Public JSON schema listing command."""

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

from ..domain import (
    VLMEvent,
)


def command_schema(args: argparse.Namespace) -> int:
    del args
    _json_print(VLMEvent.json_schema())
    return 0
