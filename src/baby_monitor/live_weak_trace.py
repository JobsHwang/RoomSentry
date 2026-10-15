"""One-shot, private metadata capture for an independent low-score experiment.

This is not a new detector operating point. Native predictions remain untouched;
the same resident YOLO performs one extra original-view inference per admitted
batch. No pixels, roles, events or identity labels are stored here.
"""

from __future__ import annotations

import json
import math
import os
import stat
import time
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.domain import Detection
from baby_monitor.live_detection import LiveFrame


class BoundedWeakTrace:
    """Explicitly constructed debug sink; no file overwrite or automatic renewal."""

    def __init__(self, path: Path, *, seconds: float = 60, max_frames: int = 4096):
        if (
            type(seconds) not in (float, int)
            or not math.isfinite(seconds)
            or not 1 <= seconds <= 180
        ):
            raise ValueError("invalid weak trace duration")
        if type(max_frames) is not int or not 1 <= max_frames <= 12000:
            raise ValueError("invalid weak trace frame limit")
        path = Path(path)
        if not path.is_absolute() or path.suffix != ".jsonl" or os.name != "posix":
            raise ValueError("weak trace requires a private POSIX file")
        if any(part in ("..", ".env") for part in path.parts) or str(path).startswith("/mnt/"):
            raise ValueError("unsafe weak trace path")
        # Refuse symlink parents, including ancestors. Only the immediate private
        # parent is required to be owner-only; system ancestors need not be 0700.
        for parent in path.parents:
            info = parent.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("unsafe weak trace parent")
        info = path.parent.lstat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("weak trace parent must be owner-only")
        repo = Path(__file__).resolve().parents[2]
        if path.is_relative_to(repo):
            raise ValueError("weak trace must stay outside the repository")
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.seconds = float(seconds)
        self.max_frames = max_frames
        self.frames = self.batches = self.failed_batches = 0
        self.byte_count = 0
        self.started = None
        self.started_utc = None
        self.status = "armed"
        try:
            self._write(
                {
                    "type": "protocol",
                    "version": 1,
                    "seconds": self.seconds,
                    "max_frames": max_frames,
                    "max_bytes": 32 * 1024 * 1024,
                    "strong_confidence": 0.35,
                    "weak_confidence": 0.20,
                    "view": "original",
                    "source_media_included": False,
                    "human_ground_truth_used": False,
                    "production_modified": False,
                }
            )
        except BaseException:
            os.close(self.fd)
            self.fd = None
            raise

    def _write(self, value):
        data = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        if self.byte_count + len(data) > 32 * 1024 * 1024:
            raise ValueError("weak trace byte limit")
        # Short writes are possible even on a local fd. Never count a partial row
        # as a complete sample, or silently discard the remaining bytes.
        view = memoryview(data)
        while view:
            count = os.write(self.fd, view)
            if count <= 0:
                raise OSError("weak trace short write")
            view = view[count:]
        self.byte_count += len(data)

    def due(self, batch_size: int) -> bool:
        if self.fd is None:
            return False
        now, wall = time.monotonic(), datetime.now(UTC)
        if self.started is None:
            self.started, self.started_utc = now, wall
            self.status = "collecting"
        elapsed = now - self.started
        wall_elapsed = (wall - self.started_utc).total_seconds()
        if min(elapsed, wall_elapsed) < 0:
            self.close("clock_invalid")
            return False
        if max(elapsed, wall_elapsed) >= self.seconds:
            self.close("duration_limit")
            return False
        if self.frames + batch_size > self.max_frames:
            self.close("frame_limit")
            return False
        return True

    def record(self, frames: tuple[LiveFrame, ...], native, diagnostic) -> None:
        """Reject malformed extra output without altering the native result."""
        if self.fd is None:
            raise ValueError("weak trace is closed")
        fields = {"version", "status", "frames", "primary_ms", "weak_ms"}
        if (
            not isinstance(diagnostic, dict)
            or set(diagnostic) != fields
            or type(diagnostic["version"]) is not int
            or diagnostic["version"] != 1
        ):
            raise ValueError("invalid weak trace response")
        for key in ("primary_ms", "weak_ms"):
            value = diagnostic[key]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 15000
            ):
                raise ValueError("invalid weak trace timing")
        rows = diagnostic["frames"]
        failed = diagnostic["status"] == "inference_failed"
        if failed:
            if rows != []:
                raise ValueError("failed weak inference has results")
        elif (
            diagnostic["status"] != "completed"
            or not isinstance(rows, list)
            or len(rows) != len(frames)
        ):
            raise ValueError("invalid weak trace batch")
        output = []
        for index, frame in enumerate(frames):
            original = [
                {"confidence": d.confidence, "bbox_xyxy": list(d.bbox_xyxy)} for d in native[index]
            ]
            weak = None
            equal = None
            if not failed:
                row = rows[index]
                if not isinstance(row, dict) or set(row) != {
                    "camera_id",
                    "sequence",
                    "sha256",
                    "detections",
                }:
                    raise ValueError("invalid weak trace frame")
                if (row["camera_id"], row["sequence"], row["sha256"]) != (
                    frame.camera_id,
                    frame.sequence,
                    frame.sha256,
                ):
                    raise ValueError("weak trace frame binding failed")
                weak = row["detections"]
                if not isinstance(weak, list) or len(weak) > 100:
                    raise ValueError("invalid weak trace detections")
                for item in weak:
                    if not isinstance(item, dict) or set(item) != {"confidence", "bbox_xyxy"}:
                        raise ValueError("invalid weak trace box")
                    if type(item["confidence"]) not in (int, float):
                        raise ValueError("invalid weak trace confidence")
                    d = Detection(0.0, "person", item["confidence"], tuple(item["bbox_xyxy"]))
                    if d.confidence < 0.20:
                        raise ValueError("weak trace box below admitted threshold")
                equal = [item for item in weak if item["confidence"] >= 0.35] == original
            output.append(
                {
                    "camera_id": frame.camera_id,
                    "sequence": frame.sequence,
                    "generation": frame.generation,
                    "raw_sha256": frame.sha256,
                    "received_at": frame.received_at.isoformat(),
                    "received_monotonic": frame.received_monotonic,
                    "native": original,
                    "low": weak,
                    "strong_output_equal": equal,
                }
            )
        self._write(
            {
                "type": "batch",
                **diagnostic,
                "frames": output,
                "finished_at": datetime.now(UTC).isoformat(),
                "finished_monotonic": time.monotonic(),
            }
        )
        self.frames += len(frames)
        self.batches += 1
        self.failed_batches += int(failed)

    def close(self, reason: str = "stopped") -> None:
        if self.fd is None:
            return
        self.status = reason
        try:
            self._write(
                {
                    "type": "end",
                    "reason": reason,
                    "frames": self.frames,
                    "batches": self.batches,
                    "failed_batches": self.failed_batches,
                }
            )
            os.fsync(self.fd)
        finally:
            os.close(self.fd)
            self.fd = None
