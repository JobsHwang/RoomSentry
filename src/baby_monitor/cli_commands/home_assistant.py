"""Home Assistant signal discovery, authorization, watcher, and dispatch commands."""

from __future__ import annotations

import argparse
import getpass
import json
import sys
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

from ..adapters.home_assistant import (
    HomeAssistantJSONSignalSource,
    HomeAssistantSignalMap,
)
from ..adapters.home_assistant_live import (
    HomeAssistantWatchSettings,
    begin_home_assistant_authorization,
    begin_home_assistant_token_capture,
    discover_home_assistant_entities,
    listen_home_assistant_signals,
    watch_home_assistant_signals,
    write_home_assistant_authorization_request,
    write_home_assistant_discovery,
    write_home_assistant_token,
    write_home_assistant_token_capture_request,
)
from ..domain import (
    SignalEvent,
)
from ..signal_dispatch import (
    BufferedSignalCommand,
    HASignalRouteMap,
    dispatch_one_signal,
    run_signal_dispatch_worker,
)
from ..signal_outbox import SignalOutbox


def command_normalize_ha_signals(args: argparse.Namespace) -> int:
    signals = HomeAssistantJSONSignalSource(
        args.mapping,
        args.events,
        max_events=args.max_events,
    ).events()
    _json_print(
        {
            "version": 1,
            "source_adapter": "home_assistant",
            "emitted_count": len(signals),
            "paths_disclosed": False,
            "signals": [signal.to_dict() for signal in signals],
        }
    )
    return 0


def command_discover_ha_signals(args: argparse.Namespace) -> int:
    if not args.confirm_private_output:
        raise ValueError(
            "HA discovery requires --confirm-private-output because entity IDs "
            "are written to a local private file"
        )
    candidates = discover_home_assistant_entities(
        args.url,
        args.token_file,
        timeout_seconds=args.timeout_seconds,
    )
    write_home_assistant_discovery(args.output, candidates)
    _json_print(
        {
            "version": 1,
            "candidate_entity_count": len(candidates),
            "event_entity_count": sum(item.domain == "event" for item in candidates),
            "binary_sensor_count": sum(item.domain == "binary_sensor" for item in candidates),
            "suggested_signal_count": sum(
                item.suggested_signal_type is not None for item in candidates
            ),
            "private_output_written": True,
            "entity_ids_disclosed": False,
            "token_disclosed": False,
            "paths_disclosed": False,
            "network_used": True,
        }
    )
    return 0


def command_store_ha_token(args: argparse.Namespace) -> int:
    access_value = getpass.getpass("HA access value (hidden): ")
    try:
        write_home_assistant_token(args.output, access_value)
    finally:
        access_value = ""
    _json_print(
        {
            "version": 1,
            "token_file_written": True,
            "mode": "0600",
            "token_disclosed": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_capture_ha_token(args: argparse.Namespace) -> int:
    capture = begin_home_assistant_token_capture(
        args.output,
        callback_port=args.callback_port,
    )
    if args.capture_request_output is not None:
        write_home_assistant_token_capture_request(
            args.capture_request_output,
            capture,
        )
    print(
        json.dumps(
            {
                "version": 1,
                "phase": "capture_required",
                "capture_url": capture.capture_url,
                "loopback_only": True,
                "token_disclosed": False,
                "paths_disclosed": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
    capture.complete(timeout_seconds=args.timeout_seconds)
    _json_print(
        {
            "version": 1,
            "phase": "complete",
            "token_file_written": True,
            "mode": "0600",
            "token_disclosed": False,
            "paths_disclosed": False,
        }
    )
    return 0


def command_authorize_ha(args: argparse.Namespace) -> int:
    authorization = begin_home_assistant_authorization(
        args.url,
        args.credentials_output,
        callback_port=args.callback_port,
    )
    if args.authorization_request_output is not None:
        write_home_assistant_authorization_request(
            args.authorization_request_output,
            authorization,
        )
    print(
        json.dumps(
            {
                "version": 1,
                "phase": "authorization_required",
                "authorization_url": authorization.authorization_url,
                "callback_loopback_only": True,
                "token_disclosed": False,
                "paths_disclosed": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
    authorization.complete(timeout_seconds=args.timeout_seconds)
    _json_print(
        {
            "version": 1,
            "phase": "complete",
            "credentials_file_written": True,
            "mode": "0600",
            "refresh_supported": True,
            "token_disclosed": False,
            "paths_disclosed": False,
            "network_used": True,
        }
    )
    return 0


def command_listen_ha_signals(args: argparse.Namespace) -> int:
    signal_map = HomeAssistantSignalMap.read(args.mapping)
    signals = listen_home_assistant_signals(
        args.url,
        args.token_file,
        signal_map,
        duration_seconds=args.duration_seconds,
        max_messages=args.max_messages,
        max_signals=args.max_signals,
        timeout_seconds=args.timeout_seconds,
    )
    _json_print(
        {
            "version": 1,
            "source_adapter": "home_assistant_live",
            "emitted_count": len(signals),
            "entity_ids_disclosed": False,
            "token_disclosed": False,
            "paths_disclosed": False,
            "network_used": True,
            "signals": [signal.to_dict() for signal in signals],
        }
    )
    return 0


def command_watch_ha_signals(args: argparse.Namespace) -> int:
    signal_map = HomeAssistantSignalMap.read(args.mapping)
    outbox = SignalOutbox(args.outbox_database) if args.outbox_database else None
    if any(
        not isinstance(value, str)
        or not 1 <= len(value) <= 80
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
            for character in value
        )
        for value in args.outbox_camera_id
    ):
        raise ValueError("outbox camera filter is invalid")
    outbox_camera_ids = frozenset(args.outbox_camera_id)
    outbox_enqueued = 0
    outbox_reused = 0
    outbox_filtered = 0

    def emit_signal(signal: object) -> None:
        nonlocal outbox_enqueued, outbox_filtered, outbox_reused
        if not isinstance(signal, SignalEvent):
            raise ValueError("HA watcher emitted an invalid signal")
        if outbox is not None and (not outbox_camera_ids or signal.camera_id in outbox_camera_ids):
            queued = outbox.enqueue(
                signal,
                max_active=args.outbox_max_active,
                max_completed=args.outbox_max_completed,
            )
            if queued.reused_existing:
                outbox_reused += 1
            else:
                outbox_enqueued += 1
        elif outbox is not None:
            outbox_filtered += 1
        if args.stream_json_lines:
            print(
                json.dumps(
                    {
                        "version": 1,
                        "record_type": "signal",
                        "entity_ids_disclosed": False,
                        "token_disclosed": False,
                        "paths_disclosed": False,
                        "signal": signal.to_dict(),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                flush=True,
            )

    result = watch_home_assistant_signals(
        args.url,
        args.token_file,
        signal_map,
        settings=HomeAssistantWatchSettings(
            max_runtime_seconds=args.max_runtime_seconds,
            poll_seconds=args.poll_seconds,
            reconnect_initial_seconds=args.reconnect_initial_seconds,
            reconnect_max_seconds=args.reconnect_max_seconds,
            max_connection_attempts=args.max_connection_attempts,
            max_messages=args.max_messages,
            max_signals=args.max_signals,
        ),
        timeout_seconds=args.timeout_seconds,
        on_signal=emit_signal if args.stream_json_lines or outbox is not None else None,
    )
    payload = result.to_dict()
    payload["outbox_enabled"] = outbox is not None
    payload["outbox_enqueued"] = outbox_enqueued
    payload["outbox_reused"] = outbox_reused
    payload["outbox_filtered"] = outbox_filtered
    if args.stream_json_lines:
        payload["signals"] = []
        payload["signals_streamed"] = len(result.signals)
    _json_print(payload)
    return 0


def command_ha_signal_outbox_status(args: argparse.Namespace) -> int:
    snapshot = SignalOutbox(args.database).snapshot_read_only()
    _json_print(snapshot.to_public_dict())
    return 0


def command_ha_signal_outbox_claim(args: argparse.Namespace) -> int:
    claim = SignalOutbox(args.database).claim_next(
        lease_seconds=args.lease_seconds,
        max_attempts=args.max_attempts,
    )
    if claim is None:
        _json_print(
            {
                "version": 1,
                "status": "idle",
                "paths_disclosed": False,
                "entity_ids_disclosed": False,
            }
        )
    else:
        _json_print(claim.to_public_dict())
    return 0


def command_ha_signal_outbox_ack(args: argparse.Namespace) -> int:
    SignalOutbox(args.database).acknowledge(args.outbox_id, args.lease_token)
    _json_print(
        {
            "version": 1,
            "status": "acknowledged",
            "outbox_id": args.outbox_id,
            "payload_scrubbed": True,
            "paths_disclosed": False,
            "entity_ids_disclosed": False,
        }
    )
    return 0


def command_ha_signal_outbox_fail(args: argparse.Namespace) -> int:
    status = SignalOutbox(args.database).record_failure(
        args.outbox_id,
        args.lease_token,
        error_code=args.error_code,
        retry_delay_seconds=args.retry_delay_seconds,
        max_attempts=args.max_attempts,
    )
    _json_print(
        {
            "version": 1,
            "status": status.value,
            "outbox_id": args.outbox_id,
            "paths_disclosed": False,
            "entity_ids_disclosed": False,
        }
    )
    return 0


def command_ha_signal_outbox_retry(args: argparse.Namespace) -> int:
    SignalOutbox(args.database).retry_failed(
        args.outbox_id,
        max_active=args.max_active,
    )
    _json_print(
        {
            "version": 1,
            "status": "pending",
            "outbox_id": args.outbox_id,
            "paths_disclosed": False,
            "entity_ids_disclosed": False,
        }
    )
    return 0


def command_ha_signal_dispatch_once(args: argparse.Namespace) -> int:
    routes = HASignalRouteMap.read(args.route_map)
    command = BufferedSignalCommand(
        python_executable=Path(sys.executable).resolve(),
        config_path=Path(args.config).expanduser().resolve(),
        staging_root=Path(args.staging_root).expanduser().resolve(),
        database_path=(
            Path(args.analysis_database).expanduser().resolve() if args.analysis_database else None
        ),
        output_root=(Path(args.output_root).expanduser().resolve() if args.output_root else None),
        timeout_seconds=args.command_timeout_seconds,
    )
    attempt = dispatch_one_signal(
        outbox=SignalOutbox(args.outbox_database),
        routes=routes,
        command=command,
        lease_seconds=args.lease_seconds,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    _json_print(attempt.to_public_dict())
    return 2 if attempt.status == "failed" else 0


def command_ha_signal_dispatch_worker(args: argparse.Namespace) -> int:
    routes = HASignalRouteMap.read(args.route_map)
    command = BufferedSignalCommand(
        python_executable=Path(sys.executable).resolve(),
        config_path=Path(args.config).expanduser().resolve(),
        staging_root=Path(args.staging_root).expanduser().resolve(),
        database_path=(
            Path(args.analysis_database).expanduser().resolve() if args.analysis_database else None
        ),
        output_root=(Path(args.output_root).expanduser().resolve() if args.output_root else None),
        timeout_seconds=args.command_timeout_seconds,
    )
    summary = run_signal_dispatch_worker(
        outbox=SignalOutbox(args.outbox_database),
        routes=routes,
        command=command,
        lease_seconds=args.lease_seconds,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
        max_jobs=args.max_jobs,
        max_runtime_seconds=args.max_runtime_seconds,
        poll_seconds=args.poll_seconds,
    )
    _json_print(summary.to_public_dict())
    return 0
