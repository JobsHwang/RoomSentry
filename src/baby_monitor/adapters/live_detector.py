"""Standard-library client for one persistent, offline YOLO process."""

from __future__ import annotations

import socket
import subprocess
from contextlib import suppress

from baby_monitor.adapters.detector import CommandYoloDetector
from baby_monitor.domain import Detection
from baby_monitor.live_detection import LiveFrame
from baby_monitor.live_detector_protocol import receive_packet, send_packet
from baby_monitor.live_review_cadence import validate_empty_review_intervals
from baby_monitor.live_weak_trace import BoundedWeakTrace


class PersistentYoloDetector:
    def __init__(
        self,
        *,
        command: tuple[str, ...],
        model_path: str,
        manifest_path: str,
        device: str = "0",
        confidence: float = 0.35,
        image_size: int = 640,
        half_precision: bool = False,
        image_height: int | None = None,
        review_rotation_180: bool = False,
        review_nonempty_shadow: bool = False,
        review_empty_scale_fallback: bool = False,
        review_empty_interval_seconds_by_camera: dict[str, float] | None = None,
        weak_continuation_trace: BoundedWeakTrace | None = None,
    ):
        if weak_continuation_trace is not None and (
            not isinstance(weak_continuation_trace, BoundedWeakTrace) or confidence != 0.35
        ):
            raise ValueError("weak trace requires explicit sink and native .35")
        self.weak_continuation_trace = weak_continuation_trace
        if type(half_precision) is not bool or (half_precision and device == "cpu"):
            raise ValueError("FP16 requires an explicitly selected CUDA device")
        if image_height is not None and (
            type(image_height) is not int
            or not 320 <= image_height <= image_size
            or image_height % 32
        ):
            raise ValueError("invalid explicit live input height")
        if type(review_rotation_180) is not bool:
            raise ValueError("invalid alternate review view")
        empty_intervals = validate_empty_review_intervals(
            review_empty_interval_seconds_by_camera, enabled=review_rotation_180
        )
        if type(review_nonempty_shadow) is not bool or (
            review_nonempty_shadow and not review_rotation_180
        ):
            raise ValueError("nonempty shadow review requires alternate review view")
        self.command = command
        if type(review_empty_scale_fallback) is not bool or (
            review_empty_scale_fallback
            and (not review_rotation_180 or image_size != 640 or image_height != 384)
        ):
            raise ValueError("scale fallback requires the admitted 384x640 alternate view")
        self.settings = {
            "type": "initialize",
            "version": 1,
            "model_path": model_path,
            "manifest_path": manifest_path,
            "device": device,
            "confidence": confidence,
            "image_size": image_size,
            "half_precision": half_precision,
            "image_height": image_height,
            "review_rotation_180": review_rotation_180,
            "review_nonempty_shadow": review_nonempty_shadow,
            "review_empty_scale_fallback": review_empty_scale_fallback,
            "review_empty_interval_seconds_by_camera": empty_intervals,
        }
        self.process: subprocess.Popen[bytes] | None = None
        self.channel: socket.socket | None = None
        self.identity: dict[str, object] = {}
        self.review_hints: dict[tuple[str, int, int, str], tuple[Detection, ...]] = {}
        self.shadow_hints: dict[tuple[str, int, int, str], tuple[Detection, ...]] = {}
        self.shadow_status: dict[tuple[str, int, int, str], str] = {}
        self.fallback_frames: set[tuple[str, int, int, str]] = set()

    def start(self) -> None:
        if self.channel is not None or self.process is not None:
            raise RuntimeError("live detector already started")
        parent, child = socket.socketpair()
        self.channel = parent
        try:
            self.process = subprocess.Popen(
                (*self.command, "--fd", str(child.fileno())),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=(child.fileno(),),
                env=CommandYoloDetector._subprocess_environment(),
            )
            child.close()
            parent.settimeout(120)
            send_packet(parent, self.settings)
            response, payload = receive_packet(parent)
            if response.get("type") != "ready" or payload:
                raise RuntimeError("live detector initialization failed")
            requested_intervals = self.settings["review_empty_interval_seconds_by_camera"]
            acknowledged_intervals = validate_empty_review_intervals(
                response.get("review_empty_interval_seconds_by_camera"),
                enabled=response.get("review_rotation_180") is True,
            )
            if acknowledged_intervals != requested_intervals or (
                requested_intervals
                and (
                    response.get("review_rotation_180") is not True
                    or type(response.get("review_shadow_interval_seconds")) not in (int, float)
                    or type(response.get("review_fallback_interval_seconds")) not in (int, float)
                    or response.get("review_shadow_interval_seconds") != 1.0
                    or response.get("review_fallback_interval_seconds") != 1.0
                )
            ):
                raise RuntimeError("live detector cadence initialization mismatch")
            shadow = response.get("review_nonempty_shadow", False)
            if (
                type(shadow) is not bool
                or shadow != self.settings["review_nonempty_shadow"]
                or (shadow and response.get("review_rotation_180") is not True)
            ):
                raise RuntimeError("live detector shadow initialization mismatch")
            self.identity = response
            if (
                self.weak_continuation_trace is not None
                and response.get("weak_trace_protocol") != 1
            ):
                raise RuntimeError("live detector weak trace protocol unavailable")
            fallback = response.get("review_empty_scale_fallback", False)
            if (
                type(fallback) is not bool
                or fallback != self.settings["review_empty_scale_fallback"]
            ):
                raise RuntimeError("live detector fallback initialization mismatch")
            parent.settimeout(15)
        except BaseException:
            child.close()
            self.close()
            raise

    def predict(self, frames: tuple[LiveFrame, ...]) -> tuple[tuple[Detection, ...], ...]:
        self.review_hints = {}
        self.shadow_hints = {}
        self.shadow_status = {}
        self.fallback_frames = set()
        if self.channel is None or not 1 <= len(frames) <= 4:
            raise RuntimeError("invalid live inference batch")
        if len({frame.camera_id for frame in frames}) != len(frames):
            raise RuntimeError("duplicate live inference camera")
        entries = [
            {
                "camera_id": frame.camera_id,
                "sequence": frame.sequence,
                "width": frame.width,
                "height": frame.height,
                "sha256": frame.sha256,
                "byte_count": len(frame.pixels),
            }
            for frame in frames
        ]
        trace = self.weak_continuation_trace
        weak_requested = False
        if trace is not None:
            try:
                weak_requested = trace.due(len(frames))
            except (OSError, ValueError):
                self._close_weak_trace("capture_failed")
        send_packet(
            self.channel,
            {
                "type": "predict",
                "frames": entries,
                **({"weak_trace": True} if weak_requested else {}),
            },
            b"".join(frame.pixels for frame in frames),
        )
        response, payload = receive_packet(self.channel)
        rows = response.get("frames")
        if (
            response.get("type") != "predictions"
            or payload
            or not isinstance(rows, list)
            or len(rows) != len(frames)
        ):
            raise RuntimeError("invalid live detector response")
        result = []
        review_hints, shadow_hints, shadow_status = {}, {}, {}
        fallback_frames = set()
        for frame, row in zip(frames, rows, strict=True):
            if (row.get("camera_id"), row.get("sequence"), row.get("sha256")) != (
                frame.camera_id,
                frame.sequence,
                frame.sha256,
            ):
                raise RuntimeError("live detector frame binding failed")
            detections = row.get("detections")
            if not isinstance(detections, list) or len(detections) > 100:
                raise RuntimeError("invalid live detections")
            result.append(
                tuple(
                    Detection(0.0, "person", item["confidence"], tuple(item["bbox_xyxy"]))
                    for item in detections
                )
            )
            checked = row.get("review_checked", False)
            if type(checked) is not bool:
                raise RuntimeError("invalid alternate review check marker")
            if detections and (checked or "review_detections" in row):
                raise RuntimeError("native detections cannot use alternate review hints")
            key = (frame.camera_id, frame.generation, frame.sequence, frame.sha256)
            fallback = row.get("review_fallback_checked", False)
            if type(fallback) is not bool or (
                fallback
                and (not self.settings["review_empty_scale_fallback"] or detections or not checked)
            ):
                raise RuntimeError("invalid scale fallback marker")
            if fallback:
                fallback_frames.add(key)
            if checked:
                hints = row.get("review_detections")
                if (
                    not self.settings["review_rotation_180"]
                    or detections
                    or not isinstance(hints, list)
                    or len(hints) > 100
                ):
                    raise RuntimeError("invalid alternate review hints")
                review_hints[key] = tuple(
                    Detection(0.0, "person", item["confidence"], tuple(item["bbox_xyxy"]))
                    for item in hints
                )
            shadow = self._read_shadow(row, native_present=bool(detections))
            if shadow is not None:
                status, hints = shadow
                shadow_status[key] = status
                if hints is not None:
                    shadow_hints[key] = hints
        # A malformed later row must not expose a partially accepted batch.
        self.review_hints = review_hints
        self.shadow_hints = shadow_hints
        self.shadow_status = shadow_status
        self.fallback_frames = fallback_frames
        if weak_requested:
            try:
                trace.record(frames, tuple(result), response.get("weak_trace"))
            except (OSError, ValueError, TypeError, KeyError, OverflowError):
                self._close_weak_trace("capture_failed")
        return tuple(result)

    def _close_weak_trace(self, reason):
        if self.weak_continuation_trace is not None:
            with suppress(OSError, ValueError):
                self.weak_continuation_trace.close(reason)

    def review_fallback_for(self, frame: LiveFrame) -> bool:
        return (
            frame.camera_id,
            frame.generation,
            frame.sequence,
            frame.sha256,
        ) in self.fallback_frames

    def _read_shadow(
        self, row: dict[str, object], *, native_present: bool
    ) -> tuple[str, tuple[Detection, ...] | None] | None:
        fields = {"shadow_checked", "shadow_skip_reason", "shadow_detections"}
        if not fields.intersection(row) and not self.settings["review_nonempty_shadow"]:
            # Legacy opt-out runners have no shadow channel.
            return None
        checked = row.get("shadow_checked")
        if type(checked) is not bool or "shadow_skip_reason" not in row:
            raise RuntimeError("invalid shadow review marker")
        reason = row["shadow_skip_reason"]
        if checked:
            hints = row.get("shadow_detections")
            if (
                not self.settings["review_nonempty_shadow"]
                or not native_present
                or reason is not None
                or not isinstance(hints, list)
                or len(hints) > 100
            ):
                raise RuntimeError("invalid shadow review hints")
            return "checked", tuple(
                Detection(0.0, "person", item["confidence"], tuple(item["bbox_xyxy"]))
                for item in hints
            )
        expected = (
            "disabled"
            if not self.settings["review_nonempty_shadow"]
            else "rate_limited"
            if native_present
            else "native_empty"
        )
        if reason != expected or "shadow_detections" in row:
            raise RuntimeError("invalid shadow review skip")
        return reason, None

    def review_hints_for(self, frame: LiveFrame) -> tuple[Detection, ...] | None:
        """None means not checked; an empty tuple means checked without candidates."""
        return self.review_hints.get(
            (frame.camera_id, frame.generation, frame.sequence, frame.sha256)
        )

    def shadow_hints_for(self, frame: LiveFrame) -> tuple[Detection, ...] | None:
        """Diagnostic only; never feed these hints to ordinary review triggers.

        None means not checked; an empty tuple means checked without candidates.
        """
        return self.shadow_hints.get(
            (frame.camera_id, frame.generation, frame.sequence, frame.sha256)
        )

    def shadow_status_for(self, frame: LiveFrame) -> str | None:
        """Return checked/disabled/native_empty/rate_limited, or no bound result."""
        return self.shadow_status.get(
            (frame.camera_id, frame.generation, frame.sequence, frame.sha256)
        )

    def close(self) -> None:
        self._close_weak_trace("stopped")
        self.review_hints = {}
        self.shadow_hints = {}
        self.shadow_status = {}
        self.fallback_frames = set()
        if self.channel is not None:
            with suppress(OSError, ValueError):
                send_packet(self.channel, {"type": "stop"})
            self.channel.close()
            self.channel = None
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
