"""Continuous FFmpeg decoding; model processes never receive network access."""

from __future__ import annotations

import subprocess
import threading
import time
from datetime import UTC, datetime
from urllib.parse import urlsplit

from baby_monitor.adapters.camera import CameraSource
from baby_monitor.live_detection import FrameMailbox, LiveFrame


def is_uniform_gray_bgr(payload: bytes) -> bool:
    """Exact information-free gray image only, not darkness/blur heuristics.

    A tiny precheck avoids a complete count on ordinary frames. All bytes must
    still be checked before rejection; a single differing byte is admitted.
    """
    if type(payload) is not bytes or not payload or len(payload) % 3:
        raise ValueError("invalid BGR payload for image-quality check")
    value = payload[0]
    if any(payload[index] != value for index in (1, 2, len(payload) // 2, len(payload) - 1)):
        return False
    return payload.count(value) == len(payload)


class FFmpegLiveSource:
    def __init__(
        self,
        source: CameraSource,
        *,
        width: int,
        height: int,
        mailbox: FrameMailbox,
        reject_uniform_frames: bool = False,
    ):
        parts = urlsplit(source.read_uri)
        if source.kind != "rtsp" or parts.scheme not in ("rtsp", "rtsps"):
            raise ValueError("live source requires RTSP")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("live source requires a credential-free bridge endpoint")
        if not 16 <= width <= 4096 or not 16 <= height <= 4096 or width * height > 4_000_000:
            raise ValueError("invalid live source dimensions")
        if type(reject_uniform_frames) is not bool:
            raise ValueError("invalid uniform-frame guard configuration")
        self.source, self.width, self.height, self.mailbox = source, width, height, mailbox
        self.reject_uniform_frames = reject_uniform_frames
        self.decoded_payloads = self.unusable_frames = self.quality_boundaries = 0
        self.image_unavailable = False
        self.last_unusable_at: str | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="decode-" + source.camera_id, daemon=True
        )
        self.process: subprocess.Popen[bytes] | None = None
        self.lock = threading.Lock()
        self.generation, self.failures, self.sequence = 0, 0, 0
        self.last_received_at: str | None = None
        self.phase = "starting"

    def start(self) -> None:
        self.thread.start()

    def _admit_payload(self, payload: bytes, received: datetime, monotonic: float) -> bool:
        """Source-thread admission; a quality gap is not a decoder restart."""
        if len(payload) != self.width * self.height * 3:
            raise ValueError("invalid live decoded payload length")
        unusable = self.reject_uniform_frames and is_uniform_gray_bgr(payload)
        with self.lock:
            frame = LiveFrame(
                self.source.camera_id,
                self.sequence,
                self.width,
                self.height,
                received,
                monotonic,
                payload,
                self.generation,
            )
            self.decoded_payloads += 1
            self.last_received_at = received.isoformat()
            if unusable:
                self.unusable_frames += 1
                self.last_unusable_at = received.isoformat()
                new_boundary = not self.image_unavailable
                self.image_unavailable = True
                self.phase = "image_unavailable"
                if new_boundary:
                    # Invalidate in-flight and queued old evidence before any
                    # valid recovery input can be dispatched. No zero-person
                    # observation is fabricated for this unavailable interval.
                    self.generation += 1
                    self.quality_boundaries += 1
                    self.mailbox.discard_unavailable_source(self.source.camera_id)
                self.sequence += 1
                return False
            self.mailbox.put(frame)
            self.sequence += 1
            self.image_unavailable = False
            self.phase = "receiving"
            return True

    def _run(self) -> None:
        frame_size = self.width * self.height * 3
        while not self.stop.is_set():
            command = [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-err_detect",
                "explode",
                "-rtsp_transport",
                "tcp",
                "-timeout",
                "12000000",
                "-allowed_media_types",
                "video",
                "-threads",
                "1",
                "-i",
                self.source.read_uri,
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-map_metadata",
                "-1",
                "-vf",
                f"scale={self.width}:{self.height}",
                "-vsync",
                "0",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    env={"PATH": "/usr/bin:/bin"},
                )
                with self.lock:
                    self.process, self.phase = process, "decoding"
                assert process.stdout is not None
                while not self.stop.is_set():
                    payload = process.stdout.read(frame_size)
                    if len(payload) != frame_size:
                        break
                    received = datetime.now(UTC)
                    self._admit_payload(payload, received, time.monotonic())
            except (OSError, ValueError):
                pass
            finally:
                with self.lock:
                    process = self.process
                    self.process = None
                    self.image_unavailable = False
                    if not self.stop.is_set():
                        self.failures += 1
                        self.generation += 1
                        self.phase = "reconnecting"
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                    if process.stdout is not None:
                        process.stdout.close()
            self.stop.wait(1)
        with self.lock:
            self.phase = "stopped"

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            snapshot = {
                "phase": self.phase,
                "last_received_at": self.last_received_at,
                "decoder_restarts": self.failures,
                "generation": self.generation,
            }
            if self.reject_uniform_frames:
                snapshot["image_quality"] = {
                    "revision": "exact_uniform_gray_guard_v1",
                    "decoded_payloads": self.decoded_payloads,
                    "unusable_frames": self.unusable_frames,
                    "quality_generation_boundaries": self.quality_boundaries,
                    "last_unusable_at": self.last_unusable_at,
                    "reason": "exact_uniform_gray" if self.image_unavailable else None,
                    "absence_of_people_established": False,
                }
            return snapshot

    def close(self) -> None:
        self.stop.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
        self.thread.join(timeout=8)
        if self.thread.is_alive():
            raise RuntimeError("live decoder did not stop")
