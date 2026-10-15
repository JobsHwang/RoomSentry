"""Remote review provider, audit, and mock server commands."""

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

from ..adapters.vlm import (
    VLMRequest,
)
from ..config import load_settings
from ..domain import (
    CandidateEvent,
    EventMedia,
)
from ..recognition_attempt import RecognitionAttemptAuditStore
from ..remote_mock import serve_remote_mock
from ..remote_review import RemoteReviewAuditStore


def command_remote_preview(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    adapter = _configured_remote_adapter(settings)
    request = VLMRequest(
        candidate=CandidateEvent(
            event_id="remote_preview",
            camera_id="anonymous_camera",
            start_seconds=0.0,
            end_seconds=0.0,
            detections=(),
        ),
        media=EventMedia("", tuple(str(Path(path).expanduser()) for path in args.frame)),
    )
    _json_print({"preview": adapter.preview(request).to_dict(), "network_request_sent": False})
    return 0


def command_remote_audit_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    attempts = RemoteReviewAuditStore(database_path).recent(limit=args.limit)
    counts = {"reserved": 0, "succeeded": 0, "failed": 0}
    for attempt in attempts:
        counts[attempt.status.value] += 1
    _json_print(
        {
            "database": str(database_path),
            "returned": len(attempts),
            "counts": counts,
            "attempts": [attempt.to_public_dict() for attempt in attempts],
            "contains_media_content": False,
            "contains_api_keys": False,
        }
    )
    return 0


def command_recognition_audit_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    database_path = Path(args.database) if args.database else settings.storage.database
    attempts = RecognitionAttemptAuditStore(database_path).recent(limit=args.limit)
    counts = {"running": 0, "succeeded": 0, "failed": 0}
    compliant = 0
    completed = 0
    for attempt in attempts:
        counts[attempt.status.value] += 1
        if attempt.metrics is not None:
            completed += 1
            compliant += int(attempt.metrics.json_compliant)
    _json_print(
        {
            "database": str(database_path),
            "returned": len(attempts),
            "counts": counts,
            "json_compliant": compliant,
            "completed": completed,
            "json_compliance_rate": (compliant / completed if completed else None),
            "attempts": [attempt.to_public_dict() for attempt in attempts],
            "contains_media_content": False,
            "contains_media_paths": False,
            "contains_error_text": False,
        }
    )
    return 0


def command_remote_mock_server(args: argparse.Namespace) -> int:
    serve_remote_mock(host=args.host, port=args.port, max_requests=args.max_requests)
    return 0
