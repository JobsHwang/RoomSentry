from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from baby_monitor.adapters.camera import redact_uri


@dataclass(frozen=True, slots=True)
class RTSPSoakSettings:
    uri: str
    duration_seconds: float = 7200.0
    startup_timeout_seconds: float = 30.0
    completion_grace_seconds: float = 60.0
    poll_interval_seconds: float = 0.5
    ffmpeg: str = "ffmpeg"
    allow_non_loopback: bool = False

    def __post_init__(self) -> None:
        try:
            parts = urlsplit(self.uri)
        except ValueError as exc:
            raise ValueError("invalid RTSP read-back URI") from exc
        if parts.scheme.lower() not in {"rtsp", "rtsps"}:
            raise ValueError("read-back URI must use rtsp:// or rtsps://")
        if not parts.hostname or not parts.path.strip("/"):
            raise ValueError("read-back URI must include a host and stream path")
        if parts.username is not None or parts.password is not None:
            raise ValueError("RTSP credentials are disabled because process arguments are visible")
        if parts.query or parts.fragment:
            raise ValueError("read-back URI cannot contain query or fragment values")
        is_loopback = parts.hostname.lower() == "localhost" or parts.hostname in {
            "127.0.0.1",
            "::1",
        }
        if not self.allow_non_loopback and not is_loopback:
            raise ValueError("read-back URI must be loopback unless explicitly allowed")
        if not 1 <= self.duration_seconds <= 86400:
            raise ValueError("duration_seconds must be between 1 and 86400")
        if not 1 <= self.startup_timeout_seconds <= 300:
            raise ValueError("startup_timeout_seconds must be between 1 and 300")
        if not 1 <= self.completion_grace_seconds <= 300:
            raise ValueError("completion_grace_seconds must be between 1 and 300")
        if not 0.05 <= self.poll_interval_seconds <= 10:
            raise ValueError("poll_interval_seconds must be between 0.05 and 10")

    @property
    def audit_ref(self) -> str:
        return redact_uri(self.uri)

    def command(self) -> list[str]:
        return [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.uri,
            "-map",
            "0:v:0",
            "-an",
            "-t",
            f"{self.duration_seconds:g}",
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ]


@dataclass(frozen=True, slots=True)
class RTSPSoakResult:
    ok: bool
    stream: str
    started_at: str
    requested_seconds: float
    elapsed_seconds: float
    frames: int
    media_seconds: float
    speed: float | None
    ffmpeg_peak_rss_mb: float | None
    ffmpeg_exit_code: int
    progress_end: bool
    error_type: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _process_rss_mb(pid: int) -> float | None:
    try:
        with (Path("/proc") / str(pid) / "status").open(encoding="ascii") as handle:
            line = next(item for item in handle if item.startswith("VmRSS:"))
        return int(line.split()[1]) / 1024
    except (OSError, StopIteration, ValueError):
        return None


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _parse_speed(value: str) -> float | None:
    normalized = value.removesuffix("x")
    try:
        return float(normalized)
    except ValueError:
        return None


def _parse_progress_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def _completion_budget_seconds(settings: RTSPSoakSettings) -> float:
    # FFmpeg's -t follows media timestamps, not wall time. Live sources can run
    # slightly below 1.0x without being stalled, so a fixed one-minute grace is
    # too short for multi-hour soaks. Five percent still fails boundedly while
    # allowing ordinary clock drift and publisher pacing.
    drift_grace = settings.duration_seconds * 0.05
    return (
        settings.startup_timeout_seconds
        + settings.duration_seconds
        + max(settings.completion_grace_seconds, drift_grace)
    )


def run_rtsp_soak(settings: RTSPSoakSettings) -> RTSPSoakResult:
    executable = settings.ffmpeg
    if shutil.which(executable) is None and not Path(executable).is_file():
        raise FileNotFoundError("FFmpeg is not installed or configured")

    progress: dict[str, str] = {}
    progress_lock = threading.Lock()
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    peak_rss: float | None = None
    error_type: str | None = None

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as error_log:
        process = subprocess.Popen(
            settings.command(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=error_log,
            text=True,
            encoding="utf-8",
        )
        if process.stdout is None:
            _stop(process)
            raise RuntimeError("FFmpeg progress pipe is unavailable")

        def consume_progress() -> None:
            for raw_line in process.stdout:
                key, separator, value = raw_line.strip().partition("=")
                if separator and key:
                    with progress_lock:
                        progress[key] = value

        reader = threading.Thread(
            target=consume_progress,
            name="rtsp-soak-progress-reader",
            daemon=True,
        )
        reader.start()
        startup_deadline = started + settings.startup_timeout_seconds
        completion_deadline = started + _completion_budget_seconds(settings)
        progress_stall_seconds = max(
            settings.startup_timeout_seconds,
            settings.completion_grace_seconds,
        )
        last_media_microseconds = 0
        last_media_progress_at = started
        try:
            while process.poll() is None:
                rss = _process_rss_mb(process.pid)
                if rss is not None:
                    peak_rss = rss if peak_rss is None else max(peak_rss, rss)
                with progress_lock:
                    frame_count = _parse_progress_int(progress.get("frame", "0") or "0")
                    media_microseconds = _parse_progress_int(
                        progress.get("out_time_us", progress.get("out_time_ms", "0")) or "0"
                    )
                now = time.monotonic()
                if media_microseconds > last_media_microseconds:
                    last_media_microseconds = media_microseconds
                    last_media_progress_at = now
                if frame_count == 0 and now >= startup_deadline:
                    error_type = "startup_timeout"
                    _stop(process)
                    break
                if frame_count > 0 and now - last_media_progress_at >= progress_stall_seconds:
                    error_type = "progress_stall"
                    _stop(process)
                    break
                if now >= completion_deadline:
                    error_type = "completion_timeout"
                    _stop(process)
                    break
                time.sleep(settings.poll_interval_seconds)
        except BaseException:
            _stop(process)
            raise
        finally:
            with suppress(OSError):
                process.stdout.close()
            reader.join(timeout=3)

        return_code = process.wait(timeout=5)
        elapsed = time.monotonic() - started
        with progress_lock:
            frames = _parse_progress_int(progress.get("frame", "0") or "0")
            media_microseconds = _parse_progress_int(
                progress.get("out_time_us", progress.get("out_time_ms", "0")) or "0"
            )
            speed = _parse_speed(progress.get("speed", ""))
            progress_end = progress.get("progress") == "end"
        if error_type is None and return_code != 0:
            error_type = "ffmpeg_exit"
        ok = return_code == 0 and progress_end and frames > 0 and error_type is None
        if not ok and error_type is None:
            error_type = "incomplete_progress"
        return RTSPSoakResult(
            ok=ok,
            stream=settings.audit_ref,
            started_at=started_at,
            requested_seconds=settings.duration_seconds,
            elapsed_seconds=round(elapsed, 3),
            frames=frames,
            media_seconds=round(media_microseconds / 1_000_000, 3),
            speed=speed,
            ffmpeg_peak_rss_mb=round(peak_rss, 3) if peak_rss is not None else None,
            ffmpeg_exit_code=return_code,
            progress_end=progress_end,
            error_type=error_type,
        )
