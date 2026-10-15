"""Domain signal/decision event validation and buffer enqueue commands."""

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

from ..adapters.signal import JSONSignalSource, signal_event_id
from ..config import load_settings
from ..domain import (
    DecisionRecord,
    SignalEvent,
)
from ..ring_buffer import (
    BufferWindowUnavailableError,
    RingBufferError,
)


def _one_signal(path: str | Path) -> SignalEvent:
    events = tuple(JSONSignalSource(path, max_events=1).events())
    if len(events) != 1:
        raise ValueError("exactly one signal event is required")
    return events[0]


def command_validate_decision_json(args: argparse.Namespace) -> int:
    raw = Path(args.path).read_text(encoding="utf-8")
    decision = DecisionRecord.from_json(raw)
    _json_print({"valid": True, "decision": decision.to_dict()})
    return 0


def command_validate_signal_json(args: argparse.Namespace) -> int:
    signal = _one_signal(args.path)
    _json_print({"valid": True, "signal": signal.to_dict()})
    return 0


def command_enqueue_signal_event(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(args.config)
    except (OSError, ValueError, RuntimeError):
        _json_print(
            {
                "queued": False,
                "reason": "buffer_config_invalid",
                "media_paths_disclosed": False,
                "signal_payload_disclosed": False,
            }
        )
        return 7
    try:
        signal = _one_signal(args.signal_json)
    except (OSError, ValueError, RuntimeError):
        _json_print(
            {
                "queued": False,
                "reason": "signal_staging_invalid",
                "media_paths_disclosed": False,
                "signal_payload_disclosed": False,
            }
        )
        return 8
    try:
        return _enqueue_buffered_candidate(
            args,
            settings=settings,
            event_time=signal.occurred_at,
            event_id=signal_event_id(signal),
            signals=(signal,),
        )
    except BufferWindowUnavailableError:
        _json_print(
            {
                "queued": False,
                "reason": "buffer_window_unavailable",
                "media_paths_disclosed": False,
                "signal_payload_disclosed": False,
            }
        )
        return 4
    except FileNotFoundError:
        _json_print(
            {
                "queued": False,
                "reason": "media_command_unavailable",
                "media_paths_disclosed": False,
                "signal_payload_disclosed": False,
            }
        )
        return 5
    except RingBufferError:
        _json_print(
            {
                "queued": False,
                "reason": "buffer_media_processing_failed",
                "media_paths_disclosed": False,
                "signal_payload_disclosed": False,
            }
        )
        return 6
    except OSError:
        _json_print(
            {
                "queued": False,
                "reason": "buffer_storage_unavailable",
                "media_paths_disclosed": False,
                "signal_payload_disclosed": False,
            }
        )
        return 9
    except ValueError:
        _json_print(
            {
                "queued": False,
                "reason": "buffer_request_invalid",
                "media_paths_disclosed": False,
                "signal_payload_disclosed": False,
            }
        )
        return 10
    except RuntimeError:
        _json_print(
            {
                "queued": False,
                "reason": "buffer_runtime_failed",
                "media_paths_disclosed": False,
                "signal_payload_disclosed": False,
            }
        )
        return 11
