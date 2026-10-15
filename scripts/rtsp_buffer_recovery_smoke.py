#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path

from baby_monitor.media_health import MediaHealthError, MediaSourceHealth, MediaSourceState
from baby_monitor.ring_buffer import (
    RTSPSegmentBuffer,
    SegmentBufferSettings,
    SegmentManifest,
)


def _stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _publisher(uri: str, duration: float) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
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
            "testsrc2=size=320x180:rate=10",
            "-t",
            f"{duration:g}",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-g",
            "10",
            "-keyint_min",
            "10",
            "-sc_threshold",
            "0",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            uri,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def run(output_root: Path, *, run_seconds: float = 16.0) -> dict[str, object]:
    stream_id = uuid.uuid4().hex[:12]
    uri = f"rtsp://127.0.0.1:8554/smoke/buffer-recovery-{stream_id}"
    publishers: list[subprocess.Popen[bytes]] = []
    publisher_codes: list[int] = []
    cycle_errors: list[str] = []

    first = _publisher(uri, 5.0)
    publishers.append(first)

    def cycle() -> None:
        try:
            publisher_codes.append(first.wait(timeout=10))
            time.sleep(2.0)
            second = _publisher(uri, 10.0)
            publishers.append(second)
            publisher_codes.append(second.wait(timeout=15))
        except Exception as exc:
            cycle_errors.append(type(exc).__name__)

    thread = threading.Thread(target=cycle, name="rtsp-publisher-recovery-cycle", daemon=True)
    thread.start()
    time.sleep(0.5)
    settings = SegmentBufferSettings(
        uri=uri,
        camera_id="cam_recovery_test",
        buffer_root=output_root,
        segment_seconds=1.0,
        max_segments=30,
        run_seconds=run_seconds,
        reconnect_initial_seconds=0.25,
        reconnect_max_seconds=1.0,
        poll_interval_seconds=0.1,
        eviction_enabled=False,
        eviction_confirmed=False,
    )
    observed_health_states: list[str] = []
    observer_stop = threading.Event()

    def observe_health() -> None:
        while not observer_stop.wait(0.05):
            try:
                state = MediaSourceHealth.read(settings.health_path).state.value
            except MediaHealthError:
                continue
            if not observed_health_states or observed_health_states[-1] != state:
                observed_health_states.append(state)

    observer = threading.Thread(
        target=observe_health,
        name="rtsp-buffer-health-observer",
        daemon=True,
    )
    observer.start()
    try:
        result = RTSPSegmentBuffer(settings).run()
    finally:
        observer_stop.set()
        observer.join(timeout=2)
        thread.join(timeout=18)
        for publisher in publishers:
            _stop(publisher)

    if cycle_errors:
        raise RuntimeError(f"publisher cycle failed safely: {cycle_errors[-1]}")
    manifest = SegmentManifest.read(settings.manifest_path)
    health = MediaSourceHealth.read(settings.health_path)
    sessions = {item.session_id for item in manifest.segments}
    ordered = sorted(manifest.segments, key=lambda item: item.started_at)
    media_gap_seconds = sum(
        max(0.0, (current.started_at - previous.ended_at).total_seconds())
        for previous, current in zip(ordered, ordered[1:], strict=False)
    )
    ok = (
        result.ok
        and result.reconnects >= 1
        and len(sessions) >= 2
        and len(manifest.segments) >= 2
        and media_gap_seconds > 0
        and health.state is MediaSourceState.STOPPED
        and health.reconnect_count == result.reconnects
        and health.segments == result.segments
        and MediaSourceState.RUNNING.value in observed_health_states
        and MediaSourceState.DEGRADED.value in observed_health_states
    )
    return {
        "ok": ok,
        "worker": result.to_dict(),
        "publisher_exit_codes": publisher_codes,
        "session_count": len(sessions),
        "media_gap_seconds": round(media_gap_seconds, 3),
        "manifest_gap_records": len(manifest.gaps),
        "manifest": str(settings.manifest_path),
        "health": health.to_dict(),
        "health_path": str(settings.health_path),
        "observed_health_states": observed_health_states,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify RTSP segment-buffer recovery using a synthetic loopback publisher"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=16.0)
    args = parser.parse_args()
    result = run(args.output_root.expanduser(), run_seconds=args.seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
