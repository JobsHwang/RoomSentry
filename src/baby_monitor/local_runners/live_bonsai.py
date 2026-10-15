"""Persistent offline Bonsai over an inherited socket; parent owns the GPU gate."""

from __future__ import annotations

import argparse
import os
import socket
import stat
import uuid
from pathlib import Path

from baby_monitor.live_detector_protocol import receive_packet, send_packet
from baby_monitor.live_vlm_protocol import VLMDiagnostics, validate_vlm_diagnostics
from baby_monitor.local_runners.bonsai_27b import (
    MODEL_REVISION,
    BonsaiResponseError,
    BonsaiSession,
    BonsaiSettings,
    prompt_revision,
    strict_json,
)
from baby_monitor.local_runners.common import (
    load_request,
    report_runner_failure,
    runner_failure_status,
)
from baby_monitor.local_runners.subject_probe import PROMPT_VERSION as SUBJECT_PROBE_VERSION
from baby_monitor.local_runners.subject_state_probe import PROMPT_VERSION as SUBJECT_STATE_VERSION

ERROR_CODES = frozenset(
    {
        "bonsai_response_incomplete",
        "bonsai_output_token_limit",
        "bonsai_invalid_envelope",
        "bonsai_invalid_response_json",
        "bonsai_server_load_failed",
        "bonsai_server_load_timeout",
        "bonsai_server_exited",
        "bonsai_invalid_request",
        "bonsai_invalid_protocol",
        "runner_oom",
        "invalid_structured_output",
        "runner_failed",
    }
)
MAX_REQUEST_BYTES = 256 * 1024


def _request_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid request ID")
    parsed = uuid.UUID(value)
    if value not in (parsed.hex, str(parsed)):
        raise ValueError("invalid request ID")
    return value


def _load_private_request(value: object):
    if not isinstance(value, str):
        raise ValueError("invalid request path")
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("invalid request path")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not 0 < metadata.st_size <= MAX_REQUEST_BYTES
            or metadata.st_mode & 0o077
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise ValueError("unsafe request file")
        # Read the admitted descriptor, not a path that could be swapped after lstat.
        return load_request(Path("/proc/self/fd") / str(descriptor))
    finally:
        os.close(descriptor)


def _error(
    channel: socket.socket,
    request_id: str | None,
    code: str,
    diagnostics: VLMDiagnostics | None = None,
) -> bool:
    if code not in ERROR_CODES:
        code = "runner_failed"
    fatal = code != "bonsai_output_token_limit"
    header = {"type": "error", "request_id": request_id, "error_code": code, "fatal": fatal}
    if diagnostics is not None:
        header["diagnostics"] = diagnostics
    send_packet(channel, header)
    return fatal


def run(
    channel: socket.socket,
    settings: BonsaiSettings,
    *,
    execution_lock_fd: int | None = None,
) -> None:
    with BonsaiSession(settings, execution_lock_fd=execution_lock_fd) as session:
        send_packet(
            channel,
            {
                "type": "ready",
                "model_revision": MODEL_REVISION,
                "prompt_revision": prompt_revision(settings),
                "model_loads": 1,
                "network_enabled": False,
                "subject_probe_revision": SUBJECT_PROBE_VERSION,
                "subject_state_revision": SUBJECT_STATE_VERSION,
            },
        )
        while True:
            try:
                header, payload = receive_packet(channel)
            except (RuntimeError, ConnectionError):
                return  # Peer closed: the context manager still reaps the model.
            except (ValueError, UnicodeError):
                _error(channel, None, "bonsai_invalid_protocol")
                return
            if header == {"type": "stop"} and not payload:
                return
            request_id = None
            try:
                request_id = _request_id(header.get("request_id"))
                if (
                    set(header) != {"type", "request_id", "request_path"}
                    or header["type"] not in ("analyze", "probe_subjects", "probe_subject_state")
                    or payload
                ):
                    raise ValueError("invalid request packet")
            except (ValueError, AttributeError):
                _error(channel, request_id, "bonsai_invalid_protocol")
                return
            # Reset here as well as in the real session so legacy/mock sessions
            # cannot accidentally return diagnostics from an earlier request.
            session.last_diagnostics = None
            try:
                request = _load_private_request(header["request_path"])
            except Exception:
                _error(channel, request_id, "bonsai_invalid_request")
                return
            generation_error = None
            try:
                result = (
                    session.probe_subject_state(request)
                    if header["type"] == "probe_subject_state"
                    else session.probe_subjects(request)
                    if header["type"] == "probe_subjects"
                    else strict_json(session.generate(request))
                )
                if not isinstance(result, dict):
                    raise ValueError("invalid result")
            except Exception as error:
                generation_error = error
            try:
                diagnostics = validate_vlm_diagnostics(session.last_diagnostics)
            except ValueError:
                _error(channel, request_id, "runner_failed")
                return
            if generation_error is not None:
                code = (
                    generation_error.code
                    if isinstance(generation_error, BonsaiResponseError)
                    else runner_failure_status(generation_error)["error_code"]
                )
                if _error(channel, request_id, code, diagnostics):
                    return
                continue
            header = {
                "type": "subject_result"
                if header["type"] in ("probe_subjects", "probe_subject_state")
                else "result",
                "request_id": request_id,
                "result": result,
            }
            if diagnostics is not None:
                header["diagnostics"] = diagnostics
            send_packet(channel, header)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--execution-lock-fd", type=int, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--runtime-directory", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--cuda-library-directory", type=Path, action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--image-tokens", type=int, default=1024)
    parser.add_argument("--independent-visual-review", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = BonsaiSettings(
            model_directory=args.model_directory,
            runtime_directory=args.runtime_directory,
            runtime_manifest=args.runtime_manifest,
            cuda_library_directories=tuple(args.cuda_library_directory),
            max_new_tokens=args.max_new_tokens,
            independent_visual_review=args.independent_visual_review,
            context_tokens=args.context_tokens,
            image_tokens=args.image_tokens,
        )
        with socket.socket(fileno=args.fd) as channel:
            run(channel, settings, execution_lock_fd=args.execution_lock_fd)
    except Exception as error:
        report_runner_failure(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
