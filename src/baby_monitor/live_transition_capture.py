"""Opt-in, bounded local evidence for actual detector/tracker transitions.

This is a diagnostic sampler, not a detector, tracker, identity source or truth.
It retains the actual admitted input bytes; encoding and disk I/O run separately.
No camera access, model calls, role inheritance, media deletion or network access.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import queue
import stat
import threading
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC

from baby_monitor.live_detection import LiveObservation, encode_bgr_png

MAX_FRAME_BYTES = 2 * 1024**2
MAX_INPUT_METADATA_BYTES = 64 * 1024
MAX_CAPTURE_ATTEMPTS = 32
REASON_LIMITS = {
    "positive_to_zero": 16,
    "two_to_one": 8,
    "dormant_with_eligible_recent_edge": 8,
}
QUEUE_CAPACITY = 2
FOLLOWUP_SECONDS = 1.0
COOLDOWN_SECONDS = 15.0
VIEWS = ("original", "rotation_180", "rotation_180_scale_fallback")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class CaptureInput:
    observation: LiveObservation
    metadata: dict

    @classmethod
    def bind(cls, observation, trace, view):
        """Bind a single just-processed row, before any later input is consumed."""
        if not isinstance(observation, LiveObservation) or view not in VIEWS:
            raise ValueError("invalid transition input")
        frame = observation.frame
        if type(frame.pixels) is not bytes or len(frame.pixels) > MAX_FRAME_BYTES:
            raise ValueError("transition frame exceeds immutable byte limit")
        if (
            trace.get("status") != "bound_interval"
            or trace.get("camera_id") != frame.camera_id
            or trace.get("detection_view") != view
            or trace.get("exported_frames") != 1
            or trace.get("interval_frames") != 1
            or trace.get("omitted_middle_frames") != 0
            or len(trace.get("frames", ())) != 1
        ):
            raise ValueError("transition trace is not an exact single input")
        row = trace["frames"][0]
        if (
            (row["generation"], row["sequence"], row["raw_sha256"], row["view"])
            != (frame.generation, frame.sequence, frame.sha256, view)
            or row["received_at"] != frame.received_at.isoformat()
            or row["received_monotonic_seconds"] != frame.received_monotonic
            or row["completed_monotonic_seconds"] != observation.completed_monotonic
            or [(d["confidence"], tuple(d["bbox_xyxy"])) for d in row["detections"]]
            != [(d.confidence, d.bbox_xyxy) for d in observation.detections]
        ):
            raise ValueError("transition trace differs from actual input")
        metadata = copy.deepcopy(row)
        if len(_json(metadata)) > MAX_INPUT_METADATA_BYTES:
            raise ValueError("transition metadata exceeds limit")
        return cls(observation, metadata)


@dataclass(frozen=True)
class TransitionBundle:
    reason: str
    before: CaptureInput
    at: CaptureInput
    after: CaptureInput | None
    end_reason: str

    @property
    def inputs(self):
        return (self.before, self.at) + (() if self.after is None else (self.after,))


class TransitionSelector:
    """At most one previous and two pending images per configured lane.

    Empty detector input is not an empty room. Two-to-one is not a true merge.
    Dormant competition is not an incorrect person assignment. End markers keep
    missing/censored follow-up distinct from observed recovery.
    """

    def __init__(self, camera_ids):
        if not 1 <= len(camera_ids) <= 4 or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("invalid transition cameras")
        self.camera_ids = tuple(camera_ids)
        self.previous = {}
        self.pending = {}
        self.cooldowns = {}
        self.counts = Counter()
        self.attempts = 0
        self.attempts_by_reason = Counter()

    def gap(self):
        self.counts["pending_discarded_on_pause"] += len(self.pending)
        self.previous.clear()
        self.pending.clear()
        self.cooldowns.clear()

    def finish(self):
        bundles = [
            TransitionBundle(reason, before, at, after, "stopped_before_followup")
            for reason, before, at, after in self.pending.values()
        ]
        self.pending.clear()
        self.previous.clear()
        return bundles

    def observe(self, item):
        frame = item.observation.frame
        if frame.camera_id not in self.camera_ids:
            raise ValueError("unconfigured transition camera")
        view = item.metadata["view"]
        key = (frame.camera_id, view == "original")
        previous = self.previous.get(key)
        if previous is not None:
            old = previous.observation.frame
            if frame.generation < old.generation or (
                frame.generation == old.generation and frame.sequence <= old.sequence
            ):
                self.counts["rejected_order_inputs"] += 1
                return ()
            elapsed = frame.received_monotonic - old.received_monotonic
            wall = (frame.received_at - old.received_at).total_seconds()
            reset = (
                frame.generation != old.generation
                or view != previous.metadata["view"]
                or min(elapsed, wall) <= 0
                or max(elapsed, wall) > 2.0
            )
        else:
            reset = True
        result = []
        pending = self.pending.get(key)
        if reset:
            self.counts["lane_resets"] += 1
            if pending is not None:
                reason, before, at, after = self.pending.pop(key)
                result.append(
                    TransitionBundle(reason, before, at, after, "input_gap_or_view_change")
                )
            for reason in REASON_LIMITS:
                self.cooldowns.pop((*key, reason), None)
            previous = None
        elif pending is not None:
            reason, before, at, after = pending
            seed = at.observation.frame
            age = max(
                frame.received_monotonic - seed.received_monotonic,
                (frame.received_at - seed.received_at).total_seconds(),
            )
            if age > FOLLOWUP_SECONDS:
                result.append(TransitionBundle(reason, before, at, after, "followup_deadline"))
                del self.pending[key]
            elif reason != "positive_to_zero" or item.observation.detections:
                result.append(TransitionBundle(reason, before, at, item, "observed_followup"))
                del self.pending[key]
            else:
                self.pending[key] = (reason, before, at, item)
        self.previous[key] = item
        if previous is None or key in self.pending:
            return tuple(result)
        before_count = len(previous.observation.detections)
        count = len(item.observation.detections)
        reason = (
            "positive_to_zero"
            if before_count and not count
            else "two_to_one"
            if before_count == 2 and count == 1
            else "dormant_with_eligible_recent_edge"
            if any(
                d["selected_source"] == "dormant_retained_track"
                and d["eligible_previous_input_track_count"] > 0
                for d in item.metadata.get("association_audit", {}).get("detections", ())
            )
            else None
        )
        if reason is not None:
            self.counts["observed_" + reason] += 1
            cooldown_key = (*key, reason)
            if self.attempts >= MAX_CAPTURE_ATTEMPTS:
                self.counts["attempt_limit_rejections"] += 1
            elif self.attempts_by_reason[reason] >= REASON_LIMITS[reason]:
                self.counts["reason_limit_rejections_" + reason] += 1
            elif frame.received_monotonic < self.cooldowns.get(cooldown_key, -1):
                self.counts["cooldown_rejections"] += 1
            else:
                self.attempts += 1
                self.attempts_by_reason[reason] += 1
                self.pending[key] = (reason, previous, item, None)
                self.cooldowns[cooldown_key] = frame.received_monotonic + COOLDOWN_SECONDS
        return tuple(result)


def write_transition_bundle(root, bundle, session_started_at):
    """Exclusive private output, receipt last; failed partial files are retained."""
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise ValueError("unsafe transition output root")
        frame = bundle.at.observation.frame
        event_id = uuid.uuid4().hex
        name = frame.received_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S_%fZ_")
        name += frame.camera_id + "_" + event_id
        os.mkdir(name, 0o700, dir_fd=descriptor)
        event = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            entries = []
            for index, item in enumerate(bundle.inputs):
                png = encode_bgr_png(item.observation.frame, profile="fast")
                filename = f"{name}_f{index:02}.png"
                _write_exclusive(event, filename, png)
                entries.append(
                    {
                        "filename": filename,
                        "size_bytes": len(png),
                        "png_sha256": hashlib.sha256(png).hexdigest(),
                        "width": item.observation.frame.width,
                        "height": item.observation.frame.height,
                        "input": item.metadata,
                    }
                )
            receipt = {
                "version": 1,
                "semantics": "actual_input_transition_not_person_truth",
                "event_id": event_id,
                "camera_id": frame.camera_id,
                "worker_pid": os.getpid(),
                "session_started_at": session_started_at,
                "reason": bundle.reason,
                "end_reason": bundle.end_reason,
                "source_images": "exact_admitted_decoder_bgr_not_new_rtsp_snapshots",
                "frames": entries,
                "human_ground_truth": False,
                "accuracy_measured": False,
                "role_inheritance": False,
                "model_calls_added": 0,
                "remote_enabled": False,
                "production_writeback": False,
            }
            _write_exclusive(event, "receipt.json", _json(receipt))
            os.fsync(event)
            return sum(e["size_bytes"] for e in entries)
        finally:
            os.close(event)
    finally:
        os.close(descriptor)


def _write_exclusive(directory, name, payload):
    fd = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
    )
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class TransitionCaptureWorker:
    """One local encoder thread; bounded nonblocking submissions, no VLM."""

    def __init__(
        self, camera_ids, *, root, session_started_at, admission, writer=write_transition_bundle
    ):
        self.selector = TransitionSelector(camera_ids)
        self.root, self.session_started_at = root, session_started_at
        self.admission, self.writer = admission, writer
        self.queue = queue.Queue(maxsize=QUEUE_CAPACITY)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.disabled = False
        self.closed = False
        self.counts = Counter()
        self.thread = threading.Thread(target=self._run, name="transition-evidence", daemon=True)

    def start(self):
        self.thread.start()

    def gap(self):
        with self.lock:
            self.selector.gap()
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
                self.queue.task_done()
                self.counts["queued_discarded_on_pause"] += 1

    def observe(self, observation, trace, *, view):
        with self.lock:
            if self.disabled or self.closed:
                return
            try:
                item = CaptureInput.bind(observation, trace, view)
                bundles = self.selector.observe(item)
            except (ValueError, KeyError, TypeError, AttributeError):
                self.counts["unbound_or_oversized_inputs"] += 1
                self.selector.gap()
                return
            for bundle in bundles:
                self._submit(bundle)

    def _submit(self, bundle):
        try:
            self.queue.put_nowait(bundle)
            self.counts["queued"] += 1
        except queue.Full:
            self.counts["queue_full_rejections"] += 1

    def _run(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                bundle = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self.disabled:
                    with self.lock:
                        self.counts["discarded_after_error"] += 1
                elif not self.admission():
                    with self.lock:
                        self.counts["storage_deferred"] += 1
                else:
                    size = self.writer(self.root, bundle, self.session_started_at)
                    with self.lock:
                        self.counts["completed"] += 1
                        self.counts["png_bytes_written"] += size
            except Exception:
                # No raw exception, paths, model text or credential-bearing data.
                with self.lock:
                    self.disabled = True
                    self.counts["write_failures"] += 1
                    self.selector.gap()
            finally:
                self.queue.task_done()

    def snapshot(self):
        with self.lock:
            return {
                "enabled": True,
                "disabled_after_error": self.disabled,
                "worker_running": self.thread.is_alive(),
                "capture_attempts": self.selector.attempts,
                "pending_lanes": len(self.selector.pending),
                "maximum_attempts": MAX_CAPTURE_ATTEMPTS,
                "reason_limits": dict(REASON_LIMITS),
                "attempts_by_reason": dict(self.selector.attempts_by_reason),
                "maximum_frame_bytes": MAX_FRAME_BYTES,
                "queue_capacity": QUEUE_CAPACITY,
                "queued_bundles": self.queue.qsize(),
                "counts": {**dict(self.selector.counts), **dict(self.counts)},
                "accuracy_measured": False,
                "model_calls_added": 0,
                "production_writeback": False,
                "paths_disclosed": False,
            }

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for bundle in self.selector.finish():
                self._submit(bundle)
            self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise RuntimeError("transition encoder join timeout")
