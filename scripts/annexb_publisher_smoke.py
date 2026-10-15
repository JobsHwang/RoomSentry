#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from rtsp_loopback_smoke import _probe_live, _stop, _wait_for_port

from baby_monitor.adapters.stream_gateway import (
    AnnexBAccessUnit,
    AnnexBRTSPPublisher,
    RTSPPublisherSettings,
    VideoCodec,
)


def _nal_units(data: bytes) -> list[bytes]:
    starts: list[int] = []
    index = 0
    while index + 3 < len(data):
        if data[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append(index)
            index += 4
        elif data[index : index + 3] == b"\x00\x00\x01":
            starts.append(index)
            index += 3
        else:
            index += 1
    return [data[start:end] for start, end in zip(starts, starts[1:] + [len(data)], strict=True)]


def _nal_type(nal: bytes, codec: VideoCodec) -> int:
    prefix = 4 if nal.startswith(b"\x00\x00\x00\x01") else 3
    header = nal[prefix]
    return header & 0x1F if codec is VideoCodec.H264 else (header >> 1) & 0x3F


def _access_units(data: bytes, codec: VideoCodec) -> list[bytes]:
    aud_type = 9 if codec is VideoCodec.H264 else 35
    result: list[bytes] = []
    current = bytearray()
    seen_aud = False
    for nal in _nal_units(data):
        is_aud = _nal_type(nal, codec) == aud_type
        if is_aud and seen_aud and current:
            result.append(bytes(current))
            current.clear()
        current.extend(nal)
        if is_aud:
            seen_aud = True
    if current:
        result.append(bytes(current))
    if not seen_aud or len(result) < 10:
        raise RuntimeError("synthetic Annex-B stream did not contain enough access units")
    return result


def _generate_annexb(path: Path, codec: VideoCodec, fps: float) -> list[bytes]:
    encoder = "libx264" if codec is VideoCodec.H264 else "libx265"
    muxer = "h264" if codec is VideoCodec.H264 else "hevc"
    codec_options = (
        "aud=1:repeat-headers=1:keyint=10:min-keyint=10:scenecut=0"
        if codec is VideoCodec.H264
        else "log-level=error:aud=1:repeat-headers=1:keyint=10:min-keyint=10:scenecut=0"
    )
    option_name = "-x264-params" if codec is VideoCodec.H264 else "-x265-params"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=160x90:rate={fps}",
            "-t",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            encoder,
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            option_name,
            codec_options,
            "-f",
            muxer,
            str(path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=True,
    )
    return _access_units(path.read_bytes(), codec)


def run(mediamtx: Path, config: Path, codec: VideoCodec) -> dict[str, object]:
    mediamtx = mediamtx.expanduser().resolve()
    config = config.resolve()
    fps = 10.0
    stream_name = f"smoke/core-publisher-{codec.value}"
    target = f"rtsp://127.0.0.1:8554/{stream_name}"
    expected_codec = "h264" if codec is VideoCodec.H264 else "hevc"
    server: subprocess.Popen[bytes] | None = None
    publisher: AnnexBRTSPPublisher | None = None
    feeder: threading.Thread | None = None
    stop_feeder = threading.Event()
    feed_errors: list[str] = []
    sent = {"count": 0}

    with (
        tempfile.TemporaryDirectory(prefix="baby-monitor-core-publisher-") as runtime_directory,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as server_log,
    ):
        runtime = Path(runtime_directory)
        access_units = _generate_annexb(runtime / f"input.{codec.value}", codec, fps)
        try:
            server = subprocess.Popen(
                [str(mediamtx), str(config)],
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                cwd=runtime,
            )
            _wait_for_port("127.0.0.1", 8554, server)
            publisher = AnnexBRTSPPublisher(
                RTSPPublisherSettings(
                    target_uri=target,
                    codec=codec,
                    input_fps=fps,
                )
            )
            publisher.start()

            def feed() -> None:
                try:
                    for sequence, data in enumerate(access_units):
                        if stop_feeder.is_set():
                            return
                        publisher.write(
                            AnnexBAccessUnit(
                                camera_id="synthetic_cam",
                                channel=0,
                                codec=codec,
                                timestamp_ms=int(sequence * 1000 / fps),
                                sequence=sequence,
                                data=data,
                            )
                        )
                        sent["count"] += 1
                        time.sleep(1 / fps)
                except Exception as exc:
                    feed_errors.append(type(exc).__name__)

            feeder = threading.Thread(target=feed, name="annexb-smoke-feeder", daemon=True)
            feeder.start()

            probe_data = _probe_live(target)
            streams = probe_data.get("streams", [])
            if not streams or streams[0].get("codec_name") != expected_codec:
                raise RuntimeError("core publisher codec did not match")

            reader = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-rtsp_transport",
                    "tcp",
                    "-i",
                    target,
                    "-t",
                    "3",
                    "-f",
                    "null",
                    "-",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if reader.returncode != 0:
                raise RuntimeError("FFmpeg could not read from the core publisher")
            if feed_errors:
                raise RuntimeError(f"access-unit feeder failed: {feed_errors[-1]}")
            return {
                "ok": True,
                "codec": codec.value,
                "listener": "rtsp://127.0.0.1:8554",
                "stream": stream_name,
                "generated_access_units": len(access_units),
                "sent_access_units": sent["count"],
                "probe": probe_data,
                "read_back_seconds": 3,
            }
        finally:
            stop_feeder.set()
            if feeder is not None:
                feeder.join(timeout=2)
            if publisher is not None:
                publisher.stop()
            _stop(server)


def main() -> int:
    parser = argparse.ArgumentParser(description="Real Annex-B core publisher smoke test")
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
    args = parser.parse_args()
    result = run(args.mediamtx, args.config, VideoCodec(args.codec))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
