from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from baby_monitor.local_runners.common import report_runner_failure
from baby_monitor.local_runners.mage_vl_4b import (
    OFFICIAL_REVISION,
    MageVLSettings,
    TrustedModelCodeManifest,
    isolated_transformers_module_cache,
)
from baby_monitor.local_runners.mage_vl_predicate import (
    ATOMIC_MODEL_ID,
    PROMPT_VERSION,
    MagePredicateRunnerError,
    MagePredicateSession,
    load_predicate_request,
)

WORKER_PROTOCOL_VERSION = 1
_MAX_CONTROL_LINE_BYTES = 16 * 1024


class MagePredicateWorkerError(MagePredicateRunnerError):
    pass


def _strict_control(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= _MAX_CONTROL_LINE_BYTES:
        raise MagePredicateWorkerError("resident worker control message is invalid")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise MagePredicateWorkerError("resident worker control message is invalid")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                MagePredicateWorkerError("resident worker control message is invalid")
            ),
        )
    except (json.JSONDecodeError, UnicodeError):
        raise MagePredicateWorkerError("resident worker control message is invalid") from None
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "request_path"}
        or type(value["version"]) is not int
        or value["version"] != WORKER_PROTOCOL_VERSION
        or not isinstance(value["request_path"], str)
        or "\x00" in value["request_path"]
        or not Path(value["request_path"]).is_absolute()
    ):
        raise MagePredicateWorkerError("resident worker control message is invalid")
    return value


def _emit(stream: TextIO, value: dict[str, object]) -> None:
    stream.write(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()


def _serve_one(session: MagePredicateSession, raw: bytes) -> dict[str, object]:
    try:
        control = _strict_control(raw)
        request = load_predicate_request(control["request_path"])
        result = session.assess(request)
    except (MagePredicateRunnerError, OSError, RuntimeError, ValueError):
        return {
            "version": WORKER_PROTOCOL_VERSION,
            "status": "error",
            "error_code": "runner_failed",
        }
    return {
        "version": WORKER_PROTOCOL_VERSION,
        "status": "ok",
        "result": result.to_dict(),
    }


def _readline_with_timeout(stream: BinaryIO, timeout: float) -> bytes | None:
    if os.name != "posix":
        raise MagePredicateWorkerError("resident worker requires a POSIX runtime")
    try:
        readable, _, _ = select.select((stream,), (), (), max(0.0, timeout))
    except (OSError, ValueError):
        raise MagePredicateWorkerError("resident worker input failed") from None
    if not readable:
        return None
    raw = stream.readline(_MAX_CONTROL_LINE_BYTES + 1)
    if raw == b"":
        raise EOFError
    if len(raw) > _MAX_CONTROL_LINE_BYTES or not raw.endswith(b"\n"):
        return b""
    return raw[:-1]


def serve(
    session: MagePredicateSession,
    *,
    input_stream: BinaryIO,
    output_stream: TextIO,
    max_jobs: int,
    idle_exit_seconds: float,
    max_runtime_seconds: float,
) -> int:
    started = time.monotonic()
    last_activity = started
    completed = 0
    _emit(
        output_stream,
        {
            "version": WORKER_PROTOCOL_VERSION,
            "status": "ready",
            "model_id": ATOMIC_MODEL_ID,
            "model_revision": OFFICIAL_REVISION,
            "prompt_version": PROMPT_VERSION,
            "max_jobs": max_jobs,
            "idle_exit_seconds": idle_exit_seconds,
            "max_runtime_seconds": max_runtime_seconds,
        },
    )
    while completed < max_jobs:
        now = time.monotonic()
        runtime_remaining = max_runtime_seconds - (now - started)
        idle_remaining = idle_exit_seconds - (now - last_activity)
        if runtime_remaining <= 0 or idle_remaining <= 0:
            break
        try:
            raw = _readline_with_timeout(
                input_stream,
                min(runtime_remaining, idle_remaining),
            )
        except EOFError:
            break
        if raw is None:
            break
        response = _serve_one(session, raw)
        _emit(output_stream, response)
        completed += 1
        last_activity = time.monotonic()
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baby-monitor-mage-vl-predicate-worker")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-manifest", required=True)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--confirm-reviewed-remote-code", action="store_true")
    parser.add_argument("--language-linear-precision", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-gpu-memory-mib", type=int, default=10_240)
    parser.add_argument("--max-cpu-memory-mib", type=int, default=12_288)
    parser.add_argument("--max-jobs", type=int, default=32)
    parser.add_argument("--idle-exit-seconds", type=float, default=30.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=3600.0)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.revision != OFFICIAL_REVISION:
        raise MagePredicateWorkerError("resident worker revision has not been reviewed")
    if args.confirm_reviewed_remote_code is not True:
        raise MagePredicateWorkerError(
            "resident worker remote code requires explicit one-shot confirmation"
        )
    if type(args.max_jobs) is not int or not 1 <= args.max_jobs <= 256:
        raise MagePredicateWorkerError("resident worker max jobs is invalid")
    if not 1.0 <= args.idle_exit_seconds <= 3600.0:
        raise MagePredicateWorkerError("resident worker idle timeout is invalid")
    if not 1.0 <= args.max_runtime_seconds <= 86_400.0:
        raise MagePredicateWorkerError("resident worker maximum runtime is invalid")
    manifest = TrustedModelCodeManifest.read(args.code_manifest)
    settings = MageVLSettings(
        model_path=Path(args.model_path).expanduser(),
        code_manifest=manifest,
        confirm_reviewed_remote_code=True,
        language_linear_precision=args.language_linear_precision,
        max_new_tokens=args.max_new_tokens,
        max_gpu_memory_mib=args.max_gpu_memory_mib,
        max_cpu_memory_mib=args.max_cpu_memory_mib,
    )
    protocol_output = sys.stdout
    with (
        open(os.devnull, "w", encoding="utf-8") as output_sink,
        isolated_transformers_module_cache(),
    ):
        from contextlib import redirect_stderr, redirect_stdout

        with (
            redirect_stdout(output_sink),
            redirect_stderr(output_sink),
            MagePredicateSession(settings) as session,
        ):
            serve(
                session,
                input_stream=sys.stdin.buffer,
                output_stream=protocol_output,
                max_jobs=args.max_jobs,
                idle_exit_seconds=args.idle_exit_seconds,
                max_runtime_seconds=args.max_runtime_seconds,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(build_parser().parse_args(argv))
    except (MagePredicateWorkerError, OSError, RuntimeError, ValueError) as exc:
        report_runner_failure(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
