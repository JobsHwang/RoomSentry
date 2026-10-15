from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from baby_monitor.adapters.camera import CameraSource
from baby_monitor.domain import CandidateEvent, EventMedia
from baby_monitor.storage import StorageResolver, secure_media_file


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    duration_seconds: float
    video_codec: str
    width: int
    height: int
    average_fps: float | None
    time_base: str
    has_audio: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "video_codec": self.video_codec,
            "width": self.width,
            "height": self.height,
            "average_fps": self.average_fps,
            "time_base": self.time_base,
            "has_audio": self.has_audio,
        }


class FFprobeMediaInspector:
    def __init__(self, *, ffprobe: str = "ffprobe", timeout_seconds: float = 30.0):
        if not 1.0 <= timeout_seconds <= 300.0:
            raise ValueError("ffprobe timeout must be between 1 and 300 seconds")
        self.ffprobe = ffprobe
        self.timeout_seconds = timeout_seconds

    def probe(self, source: CameraSource) -> MediaMetadata:
        if source.kind != "file":
            raise MediaError("bounded media inspection currently supports local files only")
        if shutil.which(self.ffprobe) is None and not Path(self.ffprobe).is_file():
            raise MediaError("ffprobe is not installed or configured")
        try:
            completed = subprocess.run(
                [
                    self.ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    (
                        "format=duration:"
                        "stream=index,codec_type,codec_name,width,height,"
                        "avg_frame_rate,time_base,duration"
                    ),
                    "-of",
                    "json",
                    source.read_uri,
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaError("ffprobe timed out while inspecting the local video") from exc
        if completed.returncode != 0:
            raise MediaError("ffprobe could not read the local video")
        try:
            payload = json.loads(completed.stdout)
            streams = payload["streams"]
            video = next(item for item in streams if item.get("codec_type") == "video")
            duration_raw = payload.get("format", {}).get("duration") or video.get("duration")
            duration = float(duration_raw)
            width = int(video["width"])
            height = int(video["height"])
            codec = str(video["codec_name"])
            time_base = str(video["time_base"])
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise MediaError("ffprobe returned incomplete video metadata") from exc
        if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
            raise MediaError("ffprobe returned invalid video metadata")

        average_fps: float | None = None
        frame_rate = str(video.get("avg_frame_rate", ""))
        try:
            parsed_rate = float(Fraction(frame_rate))
            if math.isfinite(parsed_rate) and parsed_rate > 0:
                average_fps = parsed_rate
        except (ValueError, ZeroDivisionError):
            pass
        return MediaMetadata(
            duration_seconds=duration,
            video_codec=codec,
            width=width,
            height=height,
            average_fps=average_fps,
            time_base=time_base,
            has_audio=any(item.get("codec_type") == "audio" for item in streams),
        )


class MediaProcessor(ABC):
    @abstractmethod
    def prepare(
        self,
        *,
        source: CameraSource,
        candidate: CandidateEvent,
        occurred_at: datetime,
        storage: StorageResolver,
    ) -> EventMedia:
        raise NotImplementedError


class FFmpegMediaProcessor(MediaProcessor):
    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        keyframe_count: int = 5,
    ):
        if not 3 <= keyframe_count <= 8:
            raise ValueError("keyframe_count must be between 3 and 8")
        self.ffmpeg = ffmpeg
        self.keyframe_count = keyframe_count
        self.inspector = FFprobeMediaInspector(ffprobe=ffprobe)
        self._metadata_cache: dict[str, MediaMetadata] = {}

    def _ensure_available(self) -> None:
        if shutil.which(self.ffmpeg) is None and not Path(self.ffmpeg).is_file():
            raise MediaError("FFmpeg is not installed or configured")

    @staticmethod
    def _run(args: list[str], action: str) -> None:
        try:
            completed = subprocess.run(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaError(f"{action} timed out") from exc
        if completed.returncode != 0:
            tail = (completed.stderr or "").strip().splitlines()[-1:] or ["unknown FFmpeg error"]
            raise MediaError(f"{action} failed: {tail[0]}")

    def probe(self, source: CameraSource) -> MediaMetadata:
        metadata = self._metadata_cache.get(source.read_uri)
        if metadata is None:
            metadata = self.inspector.probe(source)
            self._metadata_cache[source.read_uri] = metadata
        return metadata

    def prepare(
        self,
        *,
        source: CameraSource,
        candidate: CandidateEvent,
        occurred_at: datetime,
        storage: StorageResolver,
    ) -> EventMedia:
        if source.kind != "file":
            raise MediaError("RTSP extraction requires the not-yet-enabled ring buffer worker")
        self._ensure_available()
        metadata = self.probe(source)
        window_start = max(0.0, candidate.start_seconds)
        window_end = min(metadata.duration_seconds, candidate.end_seconds)
        if window_start >= metadata.duration_seconds or window_end - window_start < 0.05:
            raise MediaError("candidate event window does not overlap readable video")
        allocated = storage.allocate_media(
            occurred_at=occurred_at,
            camera_id=candidate.camera_id,
            event_id=candidate.event_id,
            frame_count=self.keyframe_count,
        )
        duration = window_end - window_start
        self._run(
            [
                self.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-ss",
                f"{window_start:.3f}",
                "-i",
                source.read_uri,
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                # Do not synthesize CFR observations across missing source PTS.
                "-fps_mode:v",
                "passthrough",
                "-enc_time_base:v",
                "1:1000000",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-movflags",
                "+faststart",
                str(allocated.clip_path),
            ],
            "event clip extraction",
        )
        if not allocated.clip_path.is_file() or allocated.clip_path.stat().st_size == 0:
            raise MediaError("event clip extraction produced no playable output")
        try:
            secure_media_file(allocated.clip_path)
        except ValueError as exc:
            raise MediaError("event clip extraction produced unsafe output") from exc

        frame_interval = 1.0 / metadata.average_fps if metadata.average_fps is not None else 0.1
        end_guard = min(max(0.05, frame_interval + 0.001), duration / 2.0)
        last_timestamp = max(window_start, window_end - end_guard)
        spacing = (last_timestamp - window_start) / (self.keyframe_count - 1)
        timestamps = [window_start + spacing * index for index in range(self.keyframe_count)]
        for timestamp, output in zip(timestamps, allocated.frame_paths, strict=True):
            self._run(
                [
                    self.ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-n",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    source.read_uri,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(output),
                ],
                "keyframe extraction",
            )
            if not output.is_file() or output.stat().st_size == 0:
                raise MediaError("keyframe extraction produced no frame")
            try:
                secure_media_file(output)
            except ValueError as exc:
                raise MediaError("keyframe extraction produced unsafe output") from exc
        return EventMedia(
            clip_path=str(allocated.clip_path),
            frame_paths=tuple(str(path) for path in allocated.frame_paths),
        )


class FixtureMediaProcessor(MediaProcessor):
    """Creates unmistakable test artifacts; never use it for real monitoring."""

    _ONE_PIXEL_JPEG = base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABBQJ//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAwEBPwF//8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAgBAgEBPwF//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQAGPwJ//8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPyF//9oADAMBAAIAAwAAABD/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/EH//xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/EH//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/EH//2Q=="
    )

    def __init__(self, *, frame_count: int = 3):
        self.frame_count = frame_count

    def prepare(
        self,
        *,
        source: CameraSource,
        candidate: CandidateEvent,
        occurred_at: datetime,
        storage: StorageResolver,
    ) -> EventMedia:
        del source
        allocated = storage.allocate_media(
            occurred_at=occurred_at,
            camera_id=candidate.camera_id,
            event_id=candidate.event_id,
            frame_count=self.frame_count,
        )
        allocated.clip_path.write_bytes(b"SMOKE-DEMO-NOT-A-REAL-VIDEO\n")
        secure_media_file(allocated.clip_path)
        for frame_path in allocated.frame_paths:
            frame_path.write_bytes(self._ONE_PIXEL_JPEG)
            secure_media_file(frame_path)
        return EventMedia(
            clip_path=str(allocated.clip_path),
            frame_paths=tuple(str(path) for path in allocated.frame_paths),
        )
