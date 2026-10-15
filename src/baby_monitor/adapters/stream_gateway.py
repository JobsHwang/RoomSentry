from __future__ import annotations

import shutil
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from baby_monitor.adapters.camera import redact_uri


class StreamPublisherError(RuntimeError):
    """Raised when the local stream publisher cannot accept or forward video."""


class VideoCodec(StrEnum):
    H264 = "h264"
    H265 = "h265"

    @property
    def ffmpeg_demuxer(self) -> str:
        return "h264" if self is VideoCodec.H264 else "hevc"


@dataclass(frozen=True, slots=True)
class AnnexBAccessUnit:
    """One vendor-neutral Annex-B video access unit.

    Vendor adapters must attach the real codec and source timestamps. The
    generic gateway deliberately does not guess a Xiaomi channel or codec.
    """

    camera_id: str
    channel: int
    codec: VideoCodec
    timestamp_ms: int
    sequence: int
    data: bytes

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id cannot be empty")
        if self.channel < 0:
            raise ValueError("channel cannot be negative")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms cannot be negative")
        if self.sequence < 0:
            raise ValueError("sequence cannot be negative")
        if not self.data:
            raise ValueError("Annex-B access unit cannot be empty")


@dataclass(frozen=True, slots=True)
class RTSPPublisherSettings:
    target_uri: str
    codec: VideoCodec
    input_fps: float = 15.0
    ffmpeg: str = "ffmpeg"
    allow_non_loopback: bool = False

    def __post_init__(self) -> None:
        try:
            parts = urlsplit(self.target_uri)
        except ValueError as exc:
            raise ValueError("invalid RTSP publisher target") from exc
        if parts.scheme.lower() not in {"rtsp", "rtsps"}:
            raise ValueError("publisher target must use rtsp:// or rtsps://")
        if not parts.hostname or not parts.path.strip("/"):
            raise ValueError("publisher target must include a host and stream path")
        if parts.username is not None or parts.password is not None:
            raise ValueError(
                "publisher credentials are disabled because command-line URIs "
                "leak via process lists"
            )
        if parts.query or parts.fragment:
            raise ValueError("publisher target cannot contain query or fragment values")
        is_loopback = parts.hostname.lower() == "localhost" or parts.hostname in {
            "127.0.0.1",
            "::1",
        }
        if not self.allow_non_loopback and not is_loopback:
            raise ValueError("publisher target must be loopback unless explicitly allowed")
        if self.input_fps <= 0:
            raise ValueError("input_fps must be greater than zero")

    @property
    def audit_ref(self) -> str:
        return redact_uri(self.target_uri)


class AnnexBRTSPPublisher:
    """Forward already-encoded Annex-B access units to an RTSP media router.

    This class does not listen on a network port and does not decode, analyze,
    record, or schedule work. It launches FFmpeg as an RTSP publishing client.
    MediaMTX (or another RTSP server) remains a separate, replaceable process.

    FFmpeg uses packet arrival wallclock time for PTS so a variable-rate camera
    cannot make event time drift merely because ``input_fps`` is an estimate.
    Source timestamps remain present in :class:`AnnexBAccessUnit` for a future
    publisher that can preserve the vendor clock directly.
    """

    def __init__(self, settings: RTSPPublisherSettings):
        self.settings = settings
        self._process: subprocess.Popen[bytes] | None = None

    def command(self) -> list[str]:
        return [
            self.settings.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-use_wallclock_as_timestamps",
            "1",
            "-fflags",
            "+genpts",
            "-probesize",
            "8192",
            "-analyzeduration",
            "100000",
            "-f",
            self.settings.codec.ffmpeg_demuxer,
            "-r",
            f"{self.settings.input_fps:.3f}",
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            self.settings.target_uri,
        ]

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.running:
            return
        executable = self.settings.ffmpeg
        if shutil.which(executable) is None and not Path(executable).is_file():
            raise StreamPublisherError("FFmpeg is not installed or configured")
        self._process = subprocess.Popen(
            self.command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        if self._process.poll() is not None:
            return_code = self._process.returncode
            self._process = None
            raise StreamPublisherError(
                f"FFmpeg publisher exited during startup with code {return_code}"
            )

    def write(self, access_unit: AnnexBAccessUnit) -> None:
        if access_unit.codec is not self.settings.codec:
            raise StreamPublisherError(
                f"codec changed from {self.settings.codec.value} to {access_unit.codec.value}"
            )
        if not self.running or self._process is None or self._process.stdin is None:
            raise StreamPublisherError("RTSP publisher is not running")
        try:
            self._process.stdin.write(access_unit.data)
        except (BrokenPipeError, OSError) as exc:
            code = self._process.poll()
            raise StreamPublisherError(
                f"FFmpeg publisher stopped accepting video (exit={code})"
            ) from exc

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        try:
            process.wait(timeout=timeout_seconds)
            return
        except subprocess.TimeoutExpired:
            process.terminate()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_seconds)

    def __enter__(self) -> AnnexBRTSPPublisher:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
