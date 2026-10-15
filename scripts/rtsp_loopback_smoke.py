#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TextIO

from baby_monitor.rtsp_soak import RTSPSoakSettings, run_rtsp_soak


def _wait_for_port(host: str, port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"MediaMTX exited during startup with code {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("MediaMTX did not open the loopback RTSP listener")


def _stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _tail(log: TextIO, limit: int = 20) -> str:
    log.seek(0)
    return "\n".join(log.read().splitlines()[-limit:])


def _probe_live(uri: str, *, deadline_seconds: float = 8.0) -> dict[str, object]:
    deadline = time.monotonic() + deadline_seconds
    with tempfile.TemporaryDirectory(prefix="baby-monitor-live-probe-") as directory:
        sample = Path(directory) / "sample.mkv"
        while time.monotonic() < deadline:
            capture = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-rtsp_transport",
                    "tcp",
                    "-i",
                    uri,
                    "-t",
                    "1",
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "copy",
                    str(sample),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=6,
                check=False,
            )
            if capture.returncode == 0 and sample.is_file() and sample.stat().st_size > 0:
                probe = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name,width,height",
                        "-of",
                        "json",
                        str(sample),
                    ],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                try:
                    data = json.loads(probe.stdout)
                except json.JSONDecodeError:
                    data = {}
                if (
                    probe.returncode == 0
                    and isinstance(data.get("streams"), list)
                    and data["streams"]
                ):
                    return data
            sample.unlink(missing_ok=True)
            time.sleep(0.2)
    raise RuntimeError("ffprobe could not read the live RTSP stream")


def run(
    mediamtx: Path,
    config: Path,
    codec: str = "h264",
    read_seconds: int = 3,
) -> dict[str, object]:
    mediamtx = mediamtx.resolve()
    config = config.resolve()
    if not mediamtx.is_file():
        raise FileNotFoundError(f"MediaMTX binary does not exist: {mediamtx}")
    if not config.is_file():
        raise FileNotFoundError(f"MediaMTX config does not exist: {config}")
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            raise FileNotFoundError(f"required command is missing: {command}")

    if codec not in {"h264", "h265"}:
        raise ValueError("codec must be h264 or h265")
    if not 1 <= read_seconds <= 300:
        raise ValueError("read_seconds must be between 1 and 300")
    encoder = "libx264" if codec == "h264" else "libx265"
    expected_probe_codec = "h264" if codec == "h264" else "hevc"
    stream_name = f"smoke/synthetic-{codec}"
    stream_uri = f"rtsp://127.0.0.1:8554/{stream_name}"
    server: subprocess.Popen[bytes] | None = None
    publisher: subprocess.Popen[bytes] | None = None
    with (
        tempfile.TemporaryDirectory(prefix="baby-monitor-mediamtx-") as runtime_directory,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as server_log,
    ):
        try:
            server = subprocess.Popen(
                [str(mediamtx), str(config)],
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                cwd=runtime_directory,
            )
            _wait_for_port("127.0.0.1", 8554, server)

            publisher = subprocess.Popen(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-re",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=160x90:rate=10",
                    "-t",
                    str(read_seconds + 6),
                    "-pix_fmt",
                    "yuv420p",
                    "-c:v",
                    encoder,
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-g",
                    "10",
                    *(
                        ["-x265-params", "log-level=error:keyint=10:min-keyint=10:scenecut=0"]
                        if codec == "h265"
                        else []
                    ),
                    "-f",
                    "rtsp",
                    "-rtsp_transport",
                    "tcp",
                    stream_uri,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            probe_data = _probe_live(stream_uri)
            streams = probe_data.get("streams")
            if (
                not isinstance(streams, list)
                or not streams
                or streams[0].get("codec_name") != expected_probe_codec
            ):
                raise RuntimeError("RTSP probe codec did not match the synthetic publisher")

            reader = run_rtsp_soak(
                RTSPSoakSettings(
                    uri=stream_uri,
                    duration_seconds=read_seconds,
                    startup_timeout_seconds=5,
                    completion_grace_seconds=10,
                    poll_interval_seconds=0.1,
                )
            )
            if not reader.ok:
                raise RuntimeError(f"RTSP read-back failed safely: {reader.error_type}")

            version = subprocess.run(
                [str(mediamtx), "--version"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            return {
                "ok": True,
                "server": f"MediaMTX {version}",
                "listener": "rtsp://127.0.0.1:8554",
                "stream": stream_name,
                "codec": codec,
                "probe": probe_data,
                "read_back": reader.to_dict(),
            }
        except Exception as exc:
            safe_log_tail = _tail(server_log)
            if safe_log_tail:
                raise RuntimeError(f"{exc}\nMediaMTX log tail:\n{safe_log_tail}") from exc
            raise
        finally:
            _stop(publisher)
            _stop(server)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local-only MediaMTX RTSP smoke test")
    parser.add_argument(
        "--mediamtx",
        type=Path,
        default=Path.home() / ".local/opt/mediamtx/1.19.3/mediamtx",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/mediamtx.loopback.yml"),
    )
    parser.add_argument("--codec", choices=("h264", "h265"), default="h264")
    parser.add_argument("--read-seconds", type=int, default=3)
    args = parser.parse_args()
    result = run(
        args.mediamtx.expanduser(),
        args.config,
        args.codec,
        args.read_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
