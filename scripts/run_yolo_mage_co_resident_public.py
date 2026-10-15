from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import stat
import subprocess
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

from baby_monitor.adapters.visual_predicate import (
    CommandVisualPredicateAdapter,
    ResidentCommandVisualPredicateAdapter,
    VisualPredicateFrame,
    VisualPredicateRequest,
)
from baby_monitor.local_runners.mage_vl_4b import OFFICIAL_REVISION
from baby_monitor.local_runners.mage_vl_predicate import ATOMIC_MODEL_ID, PROMPT_VERSION
from baby_monitor.visual_predicate import VisualPredicate

_SHA256_LENGTH = 64
_SAMPLE_SECONDS = 0.1
_YOLO_CHILD = r"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

weight = Path(sys.argv[1])
frame = Path(sys.argv[2])
expected_weight = sys.argv[3]
expected_frame = sys.argv[4]
if weight.is_symlink() or frame.is_symlink():
    raise SystemExit(2)
if hashlib.sha256(weight.read_bytes()).hexdigest() != expected_weight:
    raise SystemExit(2)
if hashlib.sha256(frame.read_bytes()).hexdigest() != expected_frame:
    raise SystemExit(2)
image = cv2.imread(str(frame))
if image is None:
    raise SystemExit(2)
model = YOLO(str(weight))
results = model.predict(source=image, conf=0.25, device="0", verbose=False)
detections = sum(0 if result.boxes is None else len(result.boxes) for result in results)
print(json.dumps(
    {"version": 1, "status": "ready", "detections": detections},
    separators=(",", ":"),
), flush=True)
sys.stdin.buffer.read(1)
"""


def _safe_hash(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _bounded_request_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("request count must be an integer") from None
    if not 1 <= parsed <= 32:
        raise argparse.ArgumentTypeError("request count must be between 1 and 32")
    return parsed


def _python_executable(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or "\x00" in str(path):
        raise RuntimeError("benchmark Python runtime is unsafe")
    target = path.resolve(strict=True)
    metadata = target.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or not os.access(target, os.X_OK)
        or (os.name == "posix" and metadata.st_mode & 0o022)
    ):
        raise RuntimeError("benchmark Python runtime is unsafe")
    return path


def _private_file(path: Path, *, expected_hash: str, max_bytes: int) -> bytes:
    if path.is_symlink():
        raise RuntimeError("local benchmark input is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= max_bytes
            or (os.name == "posix" and metadata.st_uid != os.getuid())
        ):
            raise RuntimeError("local benchmark input is unsafe")
        payload = b""
        while len(payload) <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - len(payload) + 1))
            if not chunk:
                break
            payload += chunk
        if len(payload) != metadata.st_size or hashlib.sha256(payload).hexdigest() != expected_hash:
            raise RuntimeError("local benchmark input admission failed")
        return payload
    finally:
        os.close(descriptor)


def _environment() -> dict[str, str]:
    environment = CommandVisualPredicateAdapter._environment()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "ULTRALYTICS_OFFLINE": "true",
        }
    )
    return environment


def _readline_timeout(stream: TextIO, timeout: float) -> str:
    responses: queue.Queue[str | None] = queue.Queue(maxsize=1)

    def read() -> None:
        responses.put(stream.readline() or None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        raw = responses.get(timeout=timeout)
    except queue.Empty:
        raise RuntimeError("YOLO holder startup timed out") from None
    if raw is None or len(raw.encode("utf-8")) > 4096:
        raise RuntimeError("YOLO holder stopped before ready")
    return raw


def _gpu_used_mib() -> int:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=_environment(),
        check=False,
        timeout=5,
    )
    values = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or len(values) != 1:
        raise RuntimeError("GPU memory sample failed")
    try:
        value = int(values[0].strip())
    except ValueError:
        raise RuntimeError("GPU memory sample failed") from None
    if not 0 <= value <= 1024 * 1024:
        raise RuntimeError("GPU memory sample failed")
    return value


def _write_private(path: Path, value: dict[str, Any]) -> None:
    parent = path.parent.resolve(strict=True)
    metadata = parent.stat(follow_symlinks=False)
    if (
        path.exists()
        or path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or parent.is_symlink()
        or (os.name == "posix" and metadata.st_uid != os.getuid())
        or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o700)
    ):
        raise RuntimeError("benchmark output is unsafe")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-python", required=True)
    parser.add_argument("--yolo-weight", required=True)
    parser.add_argument("--yolo-weight-sha256", required=True)
    parser.add_argument("--mage-python", required=True)
    parser.add_argument("--mage-model", required=True)
    parser.add_argument("--mage-manifest", required=True)
    parser.add_argument("--public-frame", action="append", required=True)
    parser.add_argument("--public-frame-sha256", action="append", required=True)
    parser.add_argument(
        "--predicate",
        choices=tuple(item.value for item in VisualPredicate),
        default="face_visibility",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-count", type=_bounded_request_count, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    worker_max_jobs = max(4, args.request_count)
    yolo_python = _python_executable(args.yolo_python)
    mage_python = _python_executable(args.mage_python)
    yolo_weight = Path(args.yolo_weight).expanduser().resolve(strict=True)
    public_frames = tuple(
        Path(value).expanduser().resolve(strict=True) for value in args.public_frame
    )
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        raise RuntimeError("benchmark output must be absolute")
    weight_hash = _safe_hash(args.yolo_weight_sha256, "YOLO weight hash")
    frame_hashes = tuple(
        _safe_hash(value, "public frame hash") for value in args.public_frame_sha256
    )
    if not 1 <= len(public_frames) <= 12 or len(public_frames) != len(frame_hashes):
        raise RuntimeError("public benchmark frame set is invalid")
    _private_file(yolo_weight, expected_hash=weight_hash, max_bytes=64 * 1024 * 1024)
    for public_frame, frame_hash in zip(public_frames, frame_hashes, strict=True):
        _private_file(
            public_frame,
            expected_hash=frame_hash,
            max_bytes=25 * 1024 * 1024,
        )

    baseline_mib = _gpu_used_mib()
    holder = subprocess.Popen(
        (
            str(yolo_python),
            "-c",
            _YOLO_CHILD,
            str(yolo_weight),
            str(public_frames[0]),
            weight_hash,
            frame_hashes[0],
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_environment(),
        start_new_session=os.name == "posix",
    )
    assert holder.stdin is not None and holder.stdout is not None
    adapter: ResidentCommandVisualPredicateAdapter | None = None
    samples: list[int] = []
    stop_sampler = threading.Event()

    def sample() -> None:
        while not stop_sampler.wait(_SAMPLE_SECONDS):
            try:
                samples.append(_gpu_used_mib())
            except (OSError, RuntimeError, subprocess.SubprocessError):
                return

    try:
        ready = json.loads(_readline_timeout(holder.stdout, 60.0))
        if (
            not isinstance(ready, dict)
            or set(ready) != {"version", "status", "detections"}
            or ready["version"] != 1
            or ready["status"] != "ready"
            or type(ready["detections"]) is not int
            or ready["detections"] < 0
        ):
            raise RuntimeError("YOLO holder handshake is invalid")
        yolo_only_mib = max(_gpu_used_mib() for _ in range(3))
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        adapter = ResidentCommandVisualPredicateAdapter(
            command=(
                str(mage_python),
                "-m",
                "baby_monitor.local_runners.mage_vl_predicate_worker",
                "--model-path",
                str(Path(args.mage_model).expanduser().resolve(strict=True)),
                "--code-manifest",
                str(Path(args.mage_manifest).expanduser().resolve(strict=True)),
                "--confirm-reviewed-remote-code",
                "--language-linear-precision",
                "int8",
                "--max-jobs",
                str(worker_max_jobs),
                "--idle-exit-seconds",
                "60",
                "--max-runtime-seconds",
                "300",
            ),
            model_id=ATOMIC_MODEL_ID,
            model_revision=OFFICIAL_REVISION,
            prompt_version=PROMPT_VERSION,
            max_jobs=worker_max_jobs,
            idle_exit_seconds=60,
            max_runtime_seconds=300,
            startup_timeout_seconds=180,
            request_timeout_seconds=60,
        )
        request = VisualPredicateRequest(
            event_id="public_yolo_int8_co_resident",
            predicate=VisualPredicate(args.predicate),
            frames=tuple(
                VisualPredicateFrame(
                    frame_id=f"t{index:02d}_full",
                    sha256=frame_hash,
                    media_type="image/jpeg",
                    path=str(public_frame),
                )
                for index, (public_frame, frame_hash) in enumerate(
                    zip(public_frames, frame_hashes, strict=True)
                )
            ),
        )
        request_seconds: list[float] = []
        results = []
        post_job_gpu_mib: list[int] = []
        health_after_job: list[dict[str, Any]] = []
        for job_index in range(args.request_count):
            job_request = VisualPredicateRequest(
                event_id=f"public_yolo_int8_co_resident_{job_index:02d}",
                predicate=request.predicate,
                frames=request.frames,
            )
            started = time.perf_counter()
            results.append(adapter.assess(job_request))
            request_seconds.append(time.perf_counter() - started)
            post_job_gpu_mib.append(_gpu_used_mib())
            health_after_job.append(adapter.health_snapshot().to_dict())
        first_request_seconds = request_seconds[0]
        result = results[-1]
        time.sleep(1.0)
        stable_mib = max(_gpu_used_mib() for _ in range(3))
        stop_sampler.set()
        sampler.join(timeout=5.0)
        if not samples:
            raise RuntimeError("GPU sampler produced no values")
        health = adapter.health_snapshot().to_dict()
        resident_post_job_gpu_mib = [
            used
            for used, snapshot in zip(post_job_gpu_mib, health_after_job, strict=True)
            if snapshot["worker_running"]
        ]
        warm_request_seconds = request_seconds[1:]
        warm_sorted = sorted(warm_request_seconds)
        warm_p95_seconds = (
            warm_sorted[(95 * len(warm_sorted) + 99) // 100 - 1] if warm_sorted else None
        )
        answer_counts: dict[str, int] = {}
        for item in results:
            answer_counts[item.answer.value] = answer_counts.get(item.answer.value, 0) + 1
        report = {
            "version": 1,
            "precision": "int8",
            "predicate": args.predicate,
            "frame_count": len(public_frames),
            "request_count": args.request_count,
            "successful_result_count": len(results),
            "baseline_gpu_mib": baseline_mib,
            "yolo_only_gpu_mib": yolo_only_mib,
            "co_resident_peak_gpu_mib": max(samples),
            "co_resident_stable_gpu_mib": (
                max(resident_post_job_gpu_mib[-3:]) if resident_post_job_gpu_mib else stable_mib
            ),
            "post_bound_gpu_mib": stable_mib,
            "yolo_incremental_mib": yolo_only_mib - baseline_mib,
            "co_resident_incremental_mib": max(samples) - baseline_mib,
            "first_request_seconds": round(first_request_seconds, 4),
            "warm_request_mean_seconds": (
                round(sum(warm_request_seconds) / len(warm_request_seconds), 4)
                if warm_request_seconds
                else None
            ),
            "warm_request_p95_seconds": (
                round(warm_p95_seconds, 4) if warm_p95_seconds is not None else None
            ),
            "answer_counts": answer_counts,
            "answer": result.answer.value,
            "post_job_gpu_mib": post_job_gpu_mib,
            "resident_post_job_gpu_mib": resident_post_job_gpu_mib,
            "resident_gpu_range_mib": (
                max(resident_post_job_gpu_mib) - min(resident_post_job_gpu_mib)
                if resident_post_job_gpu_mib
                else 0
            ),
            "resident_gpu_drift_mib": (
                resident_post_job_gpu_mib[-1] - resident_post_job_gpu_mib[0]
                if resident_post_job_gpu_mib
                else 0
            ),
            "identity_inference": result.subject_identity_inference,
            "risk_inference": result.risk_inference,
            "notification_decision": result.notification_decision,
            "yolo_detection_count": ready["detections"],
            "worker_health": health,
            "household_media": False,
            "uploads_performed": False,
            "paths_disclosed": False,
        }
        _write_private(output, report)
    finally:
        stop_sampler.set()
        if adapter is not None:
            adapter.close()
        try:
            holder.stdin.write("\n")
            holder.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            holder.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            CommandVisualPredicateAdapter._kill_process(holder)
            holder.wait(timeout=10.0)
        holder.stdout.close()
        with suppress(BrokenPipeError, OSError):
            holder.stdin.close()

    cleanup_mib = _gpu_used_mib()
    print(
        json.dumps(
            {
                "version": 1,
                "status": "completed",
                "co_resident_peak_gpu_mib": report["co_resident_peak_gpu_mib"],
                "cleanup_gpu_mib": cleanup_mib,
                "answer": report["answer"],
                "request_count": report["request_count"],
                "household_media": False,
                "uploads_performed": False,
                "paths_disclosed": False,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
