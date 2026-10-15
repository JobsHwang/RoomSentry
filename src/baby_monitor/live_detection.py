"""Bounded frame scheduling and event triggers for continuous local detection."""

from __future__ import annotations

import hashlib
import math
import re
import struct
import threading
import time
import zlib
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

from baby_monitor.domain import Detection


@dataclass(frozen=True, slots=True)
class LiveFrame:
    camera_id: str
    sequence: int
    width: int
    height: int
    received_at: datetime
    received_monotonic: float
    pixels: bytes
    generation: int = 0

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.camera_id) is None:
            raise ValueError("invalid live camera ID")
        if any(type(value) is not int or value < 0 for value in (self.sequence, self.generation)):
            raise ValueError("invalid live frame sequence")
        if any(
            type(value) is not int or not 1 <= value <= 4096 for value in (self.width, self.height)
        ):
            raise ValueError("invalid live frame dimensions")
        if self.width * self.height > 4_000_000 or len(self.pixels) != self.width * self.height * 3:
            raise ValueError("invalid live BGR frame")
        if self.received_at.tzinfo is None or not math.isfinite(self.received_monotonic):
            raise ValueError("invalid live receipt time")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.pixels).hexdigest()


def receipt_age_seconds(
    frame: LiveFrame, now: float, now_utc: datetime | None = None
) -> float | None:
    """Conservative decoder-receipt age; invalid/future clocks are unavailable.

    Elapsed-only callers are useful for deterministic scheduling simulations.
    Real model admission must supply both clocks, sampled under the queue lock.
    """
    if type(now) not in (int, float) or not 0 <= now <= 10**12:
        return None
    age = now - frame.received_monotonic
    if not math.isfinite(age) or age < 0:
        return None
    if now_utc is not None:
        try:
            if now_utc.utcoffset() is None or frame.received_at.utcoffset() is None:
                return None
            wall_age = (now_utc - frame.received_at).total_seconds()
            if not math.isfinite(wall_age) or wall_age < 0:
                return None
            age = max(age, wall_age)
        except (TypeError, ValueError, OverflowError, AttributeError):
            return None
    return age


@dataclass(frozen=True, slots=True)
class MotionBurstSettings:
    """Sampling hints only: pixel changes are not person or safety observations."""

    fps: float = 20.0
    hold_seconds: float = 3.0
    pixel_delta: float = 20.0
    changed_fraction: float = 0.01
    pending_capacity: int = 1

    def __post_init__(self) -> None:
        for value, lower, upper in (
            (self.fps, 0.5, 30),
            (self.hold_seconds, 0.1, 10),
            (self.pixel_delta, 1, 255),
            (self.changed_fraction, 0.001, 1),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError("invalid motion burst settings")
        if type(self.pending_capacity) is not int or not 1 <= self.pending_capacity <= 4:
            raise ValueError("invalid motion burst pending capacity")


@dataclass(slots=True)
class _MotionState:
    generation: int
    last_received: float
    thumbnail: bytes
    burst_until: float = -math.inf
    burst_started: float = -math.inf
    trigger_frames: int = 0
    episodes: int = 0


class MotionBurstPolicy:
    """Cheap change hint on every decoded frame, independent of YOLO admission.

    A 32x24 luminance grid and common brightness-shift subtraction reduce lighting
    noise. Small, slow, occluded or low-contrast movement can still be missed.
    Continuous movement renews the hold; configured rate and queue bounds still apply.
    """

    def __init__(self, settings: MotionBurstSettings):
        self.settings = settings
        self.state: dict[str, _MotionState] = {}

    def observe(self, frame: LiveFrame) -> None:
        thumbnail = bytes(
            (
                29 * frame.pixels[offset]
                + 150 * frame.pixels[offset + 1]
                + 77 * frame.pixels[offset + 2]
            )
            // 256
            for y in range(24)
            for x in range(32)
            for offset in [((y * frame.height // 24) * frame.width + x * frame.width // 32) * 3]
        )
        now = frame.received_monotonic
        state = self.state.get(frame.camera_id)
        if (
            state is None
            or state.generation != frame.generation
            or not 0 <= now - state.last_received <= 1.0
        ):
            self.state[frame.camera_id] = _MotionState(
                frame.generation,
                now,
                thumbnail,
                trigger_frames=0 if state is None else state.trigger_frames,
                episodes=0 if state is None else state.episodes,
            )
            return
        deltas = tuple(new - old for new, old in zip(thumbnail, state.thumbnail, strict=True))
        common_shift = sum(deltas) / len(deltas)
        changed = sum(abs(delta - common_shift) >= self.settings.pixel_delta for delta in deltas)
        if changed >= math.ceil(len(deltas) * self.settings.changed_fraction):
            state.trigger_frames += 1
            if now >= state.burst_until:
                state.episodes += 1
                state.burst_started = now
            state.burst_until = now + self.settings.hold_seconds
        state.last_received = now
        state.thumbnail = thumbnail

    def active(self, camera: str, now: float) -> bool:
        state = self.state.get(camera)
        return state is not None and now < state.burst_until


class FrameMailbox:
    """Fair full-frame FIFO or bounded scheduled evidence slots, with separate counters."""

    def __init__(
        self,
        camera_ids: tuple[str, ...],
        *,
        capacity: int = 32,
        max_age_seconds: float = 1.0,
        sample_fps: float | None = None,
        motion_burst: MotionBurstSettings | None = None,
        batch_coalesce_seconds: float = 0.0,
    ):
        if not camera_ids or len(set(camera_ids)) != len(camera_ids) or not 1 <= capacity <= 256:
            raise ValueError("invalid frame mailbox")
        if not 0 < max_age_seconds <= 10:
            raise ValueError("invalid frame age limit")
        if sample_fps is not None and (
            isinstance(sample_fps, bool)
            or not isinstance(sample_fps, (int, float))
            or not math.isfinite(sample_fps)
            or not 0.5 <= sample_fps <= 30
        ):
            raise ValueError("invalid live sampling rate")
        if motion_burst is not None and (
            not isinstance(motion_burst, MotionBurstSettings)
            or sample_fps is None
            or motion_burst.fps <= sample_fps
        ):
            raise ValueError("motion burst requires a lower baseline sampling rate")
        if (
            isinstance(batch_coalesce_seconds, bool)
            or not isinstance(batch_coalesce_seconds, (int, float))
            or not math.isfinite(batch_coalesce_seconds)
            or not 0 <= batch_coalesce_seconds <= 0.02
        ):
            raise ValueError("invalid live batch coalescing window")
        self.capacity = capacity
        self.max_age_seconds = max_age_seconds
        self.sample_fps = sample_fps
        self.batch_coalesce_seconds = batch_coalesce_seconds
        self.motion = None if motion_burst is None else MotionBurstPolicy(motion_burst)
        self.effective_fps = dict.fromkeys(camera_ids, sample_fps)
        self.last_selected: dict[str, float | None] = dict.fromkeys(camera_ids)
        self.next_due: dict[str, float | None] = dict.fromkeys(camera_ids)
        self.generation: dict[str, int | None] = dict.fromkeys(camera_ids)
        self.condition = threading.Condition()
        self.frames: dict[str, deque[LiveFrame]] = {camera: deque() for camera in camera_ids}
        self.received = dict.fromkeys(camera_ids, 0)
        self.dropped = dict.fromkeys(camera_ids, 0)
        self.sampling_skipped = dict.fromkeys(camera_ids, 0)
        self.eligible = dict.fromkeys(camera_ids, 0)
        self.dispatched = dict.fromkeys(camera_ids, 0)
        self.overload_dropped = dict.fromkeys(camera_ids, 0)
        self.stale_dropped = dict.fromkeys(camera_ids, 0)
        self.receipt_clock_dropped = dict.fromkeys(camera_ids, 0)
        self.source_unavailable_dropped = dict.fromkeys(camera_ids, 0)

    def discard_unavailable_source(self, camera_id: str) -> int:
        """Purge only pending input, not media, after an explicit source-quality gap.

        Already dispatched frames are isolated by the source generation. This
        does not create a synthetic zero-detection observation or delete files.
        """
        with self.condition:
            queue = self.frames[camera_id]
            discarded = len(queue)
            queue.clear()
            self.dropped[camera_id] += discarded
            self.source_unavailable_dropped[camera_id] += discarded
            self.next_due[camera_id] = None
            self.last_selected[camera_id] = None
            self.generation[camera_id] = None
            self.effective_fps[camera_id] = self.sample_fps
            if self.motion is not None:
                # Keep cumulative motion counters; the next accepted generation
                # rebuilds its thumbnail and cannot treat a gap as movement.
                state = self.motion.state.get(camera_id)
                if state is not None:
                    state.burst_until = -math.inf
            self.condition.notify_all()
            return discarded

    def put(self, frame: LiveFrame) -> None:
        with self.condition:
            queue = self.frames[frame.camera_id]
            burst_active = False
            self.received[frame.camera_id] += 1
            if self.sample_fps is not None:
                now = frame.received_monotonic
                if self.generation[frame.camera_id] != frame.generation:
                    self.next_due[frame.camera_id] = None
                    self.last_selected[frame.camera_id] = None
                    self.effective_fps[frame.camera_id] = self.sample_fps
                    self.generation[frame.camera_id] = frame.generation
                rate = self.sample_fps
                if self.motion is not None:
                    self.motion.observe(frame)
                    if self.motion.active(frame.camera_id, now):
                        rate = self.motion.settings.fps
                        burst_active = True
                if rate != self.effective_fps[frame.camera_id]:
                    previous = self.last_selected[frame.camera_id]
                    # Admit the triggering frame when the new rate allows it, not
                    # after the old 5 FPS deadline; transitions cannot exceed the cap.
                    self.next_due[frame.camera_id] = (
                        None if previous is None else previous + 1 / rate
                    )
                    self.effective_fps[frame.camera_id] = rate
                due = self.next_due[frame.camera_id]
                if due is not None and now + 1e-9 < due:
                    self.sampling_skipped[frame.camera_id] += 1
                    return
                interval = 1 / rate
                due = now if due is None else due
                # Keep the original phase without a catch-up burst after source gaps.
                periods = max(1, math.floor((now - due + 1e-9) / interval) + 1)
                self.next_due[frame.camera_id] = due + periods * interval
                self.last_selected[frame.camera_id] = now
            self.eligible[frame.camera_id] += 1
            capacity = 1 if self.sample_fps is not None else self.capacity
            if burst_active and self.motion is not None:
                capacity = self.motion.settings.pending_capacity
                if capacity > 1:
                    started = self.motion.state[frame.camera_id].burst_started
                    # A queued pre-change baseline must not occupy the protected
                    # first-burst slot and push out the triggering evidence.
                    while queue and queue[0].received_monotonic < started:
                        queue.popleft()
                        self.dropped[frame.camera_id] += 1
                        self.overload_dropped[frame.camera_id] += 1
            while len(queue) >= capacity:
                # In a multi-slot burst preserve the earliest pending evidence and
                # replace an intermediate with the newest. Outside bursts retain
                # the ordinary single latest-eligible slot (or full-frame FIFO).
                if burst_active and capacity > 1:
                    queue.pop()
                else:
                    queue.popleft()
                self.dropped[frame.camera_id] += 1
                self.overload_dropped[frame.camera_id] += 1
            queue.append(frame)
            self.condition.notify()

    def wait_for_batch(self) -> None:
        """Briefly group staggered streams; never wait for an empty or complete batch.

        Condition waiting releases the lock so decode threads can keep admitting
        frames. A missing/quiet camera cannot hold up hot cameras indefinitely.
        """
        with self.condition:
            if self.batch_coalesce_seconds and any(self.frames.values()):
                self.condition.wait_for(
                    lambda: all(self.frames.values()), timeout=self.batch_coalesce_seconds
                )

    def take_live(self) -> tuple[LiveFrame, ...]:
        # Admission cannot race a producer that adds a frame after a caller's
        # timestamp sample. Condition's reentrant lock pins the whole cohort.
        with self.condition:
            return self.take(time.monotonic(), now_utc=datetime.now(UTC))

    def take(self, now: float, *, now_utc: datetime | None = None) -> tuple[LiveFrame, ...]:
        batch = []
        with self.condition:
            for camera, queue in self.frames.items():
                while queue:
                    age = receipt_age_seconds(queue[0], now, now_utc)
                    if age is not None and age <= self.max_age_seconds:
                        break
                    queue.popleft()
                    self.dropped[camera] += 1
                    self.stale_dropped[camera] += 1
                    self.receipt_clock_dropped[camera] += age is None
                if queue:
                    batch.append(queue.popleft())
                    self.dispatched[camera] += 1
        return tuple(batch)

    def snapshot(self, now: float | None = None) -> dict[str, dict[str, object]]:
        with self.condition:
            result: dict[str, dict[str, object]] = {
                camera: {
                    "decoded_frames": self.received[camera],
                    "discarded_frames": self.dropped[camera],
                    "queued_frames": len(queue),
                    "sampling_skipped_frames": self.sampling_skipped[camera],
                    "eligible_frames": self.eligible[camera],
                    "dispatched_frames": self.dispatched[camera],
                    "overload_discarded_frames": self.overload_dropped[camera],
                    "stale_discarded_frames": self.stale_dropped[camera],
                    "receipt_clock_discarded_frames": self.receipt_clock_dropped[camera],
                }
                for camera, queue in self.frames.items()
            }
            if self.motion is not None:
                for camera, row in result.items():
                    state = self.motion.state.get(camera)
                    clock = now if now is not None else (state.last_received if state else 0)
                    active = self.motion.active(camera, clock)
                    row.update(
                        sampling_target_fps=self.motion.settings.fps if active else self.sample_fps,
                        sampling_burst_active=active,
                        sampling_pending_capacity=self.motion.settings.pending_capacity
                        if active
                        else 1,
                        motion_trigger_frames=state.trigger_frames if state else 0,
                        motion_burst_episodes=state.episodes if state else 0,
                    )
            for camera, row in result.items():
                if self.source_unavailable_dropped[camera]:
                    row["source_unavailable_discarded_frames"] = self.source_unavailable_dropped[
                        camera
                    ]
            return result


@dataclass(frozen=True, slots=True)
class LiveObservation:
    frame: LiveFrame
    detections: tuple[Detection, ...]
    completed_monotonic: float


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    observation: LiveObservation
    reason: str
    priority: int
    queued_monotonic: float
    evidence_view: str = "original"

    @property
    def evidence_role(self) -> str:
        return "trigger_frame" if self.priority > 0 else "latest_scene"


class ReviewMailbox:
    """Keep first/latest change evidence per camera while bounding review backlog."""

    def __init__(self, camera_ids: tuple[str, ...], *, max_evidence_age_seconds: float = 120.0):
        if (
            isinstance(max_evidence_age_seconds, bool)
            or not isinstance(max_evidence_age_seconds, (int, float))
            or not math.isfinite(max_evidence_age_seconds)
            or not 10 <= max_evidence_age_seconds <= 600
        ):
            raise ValueError("invalid review evidence age limit")
        self.camera_ids = frozenset(camera_ids)
        self.max_evidence_age_seconds = max_evidence_age_seconds
        self.condition = threading.Condition()
        self.pending: dict[str, list[ReviewCandidate]] = {}
        self.generations: dict[str, int] = {}
        self.last_camera: str | None = None
        self.last_served = dict.fromkeys(camera_ids, 0)
        self.service_sequence = 0
        self.coalesced = 0
        self.expired = 0
        self.receipt_clock_discarded = 0
        self.generation_discarded = 0
        self.control_discarded = 0

    def _accept_generation(self, observation: LiveObservation) -> bool:
        camera, generation = observation.frame.camera_id, observation.frame.generation
        if camera not in self.camera_ids:
            raise ValueError("unknown review camera")
        previous = self.generations.get(camera)
        if previous is not None and generation < previous:
            return False
        if previous is not None and generation != previous:
            self.generation_discarded += len(self.pending.pop(camera, []))
        self.generations[camera] = generation
        return True

    def put(self, candidate: ReviewCandidate) -> None:
        camera = candidate.observation.frame.camera_id
        if camera not in self.camera_ids:
            raise ValueError("unknown review camera")
        with self.condition:
            if not self._accept_generation(candidate.observation):
                return
            entries = self.pending.setdefault(camera, [])
            if entries and entries[0].priority == 0:
                # A periodic/initial scene may advance until a real trigger pins it.
                previous = entries[0]
                self.coalesced += 1
                candidate = ReviewCandidate(
                    candidate.observation,
                    candidate.reason,
                    candidate.priority,
                    previous.queued_monotonic,
                    candidate.evidence_view,
                )
                entries[0] = candidate
            elif len(entries) < 2:
                entries.append(candidate)
            else:
                # First trigger stays immutable; only the newest pending slot coalesces.
                previous = entries[-1]
                self.coalesced += 1
                if candidate.priority == 0 and previous.priority > 0:
                    return
                entries[-1] = ReviewCandidate(
                    candidate.observation,
                    candidate.reason,
                    candidate.priority,
                    previous.queued_monotonic,
                    candidate.evidence_view,
                )
            self.condition.notify()

    def take_live(self) -> ReviewCandidate | None:
        with self.condition:
            return self.take(time.monotonic(), now_utc=datetime.now(UTC))

    def take(
        self, now: float | None = None, *, now_utc: datetime | None = None
    ) -> ReviewCandidate | None:
        """Explicit clocks for simulations; take_live() is the model-admission path."""
        with self.condition:
            if now is not None or now_utc is not None:
                for camera, entries in list(self.pending.items()):
                    kept = []
                    for item in entries:
                        age = receipt_age_seconds(item.observation.frame, now, now_utc)
                        if age is not None and age <= self.max_evidence_age_seconds:
                            kept.append(item)
                        elif age is None:
                            self.receipt_clock_discarded += 1
                    self.expired += len(entries) - len(kept)
                    if kept:
                        self.pending[camera] = kept
                    else:
                        del self.pending[camera]
            candidates = [entries[0] for entries in self.pending.values()]
            if len(candidates) > 1:
                candidates = [
                    item
                    for item in candidates
                    if item.observation.frame.camera_id != self.last_camera
                ]
            if not candidates:
                return None
            # Queue age wins: two busy cameras must not starve a quiet-camera baseline.
            selected = min(
                candidates,
                key=lambda item: (
                    item.queued_monotonic,
                    self.last_served[item.observation.frame.camera_id],
                    -item.priority,
                ),
            )
            self.last_camera = selected.observation.frame.camera_id
            self.service_sequence += 1
            self.last_served[self.last_camera] = self.service_sequence
            self.pending[self.last_camera].pop(0)
            if not self.pending[self.last_camera]:
                del self.pending[self.last_camera]
            return selected

    def discard_pending_for_suspend(self) -> None:
        """Drop only unsaved RAM candidates; preserve generations and counters.

        The caller must block new review admission until resume. This is an
        explicit review coverage gap, not media retention or a source reset.
        """
        with self.condition:
            self.control_discarded += sum(len(items) for items in self.pending.values())
            self.pending.clear()

    def snapshot(self) -> dict[str, object]:
        with self.condition:
            return {
                "pending_cameras": sorted(self.pending),
                "pending_candidates": sum(len(entries) for entries in self.pending.values()),
                "pinned_trigger_frames": sum(
                    item.priority > 0 for entries in self.pending.values() for item in entries
                ),
                "capacity_per_camera": 2,
                "max_evidence_age_seconds": self.max_evidence_age_seconds,
                "coalesced_candidates": self.coalesced,
                "expired_candidates": self.expired,
                "receipt_clock_discarded_candidates": self.receipt_clock_discarded,
                "generation_discarded_candidates": self.generation_discarded,
                "control_discarded_candidates": self.control_discarded,
            }

    def refresh(self, observation: LiveObservation) -> None:
        """Refresh periodic scenes; detected-change evidence stays bound to its frame."""
        camera = observation.frame.camera_id
        with self.condition:
            if not self._accept_generation(observation):
                return
            for index, previous in enumerate(self.pending.get(camera, [])):
                if previous.priority == 0:
                    self.pending[camera][index] = ReviewCandidate(
                        observation, previous.reason, previous.priority, previous.queued_monotonic
                    )


class AlternateViewReviewPolicy:
    """Confirmed alternate-view hints request local review without changing native counts."""

    def __init__(
        self, *, minimum_frame_confidence: float = 0.0, minimum_confirmation_seconds: float = 1.0
    ):
        if (
            type(minimum_frame_confidence) not in (int, float)
            or not 0 <= minimum_frame_confidence <= 1
            or not math.isfinite(minimum_frame_confidence)
        ):
            raise ValueError("invalid alternate frame confidence gate")
        if (
            type(minimum_confirmation_seconds) not in (int, float)
            or not math.isfinite(minimum_confirmation_seconds)
            or not 1.0 <= minimum_confirmation_seconds <= 3.0
        ):
            raise ValueError("invalid alternate confirmation interval")
        self.minimum_frame_confidence = minimum_frame_confidence
        self.minimum_confirmation_seconds = minimum_confirmation_seconds
        self.state: dict[str, dict[str, object]] = {}

    def frame_eligible(self, detections: tuple[Detection, ...]) -> bool:
        """Trigger eligibility only; retain all raw candidates, not a person fact."""
        return (
            bool(detections)
            and max(d.confidence for d in detections) >= self.minimum_frame_confidence
        )

    def observe(self, observation: LiveObservation) -> ReviewCandidate | None:
        frame = observation.frame
        now = observation.completed_monotonic
        signature = (
            _signature(observation.detections)
            if self.frame_eligible(observation.detections)
            else ()
        )
        state = self.state.get(frame.camera_id)
        if state is None or state["generation"] != frame.generation:
            state = {
                "generation": frame.generation,
                "signature": None,
                "votes": 0,
                "last_seen": -math.inf,
                "last_trigger": -math.inf,
                "accepted": None,
                "first_seen": now,
                "first_received": frame.received_monotonic,
                "last_sequence": -1,
                "last_received": -math.inf,
                "last_completed": -math.inf,
            }
            self.state[frame.camera_id] = state
        if (
            frame.sequence <= int(state["last_sequence"])
            or frame.received_monotonic <= float(state["last_received"])
            or now < float(state["last_completed"])
        ):
            # Invalid order is a continuity gap, including empty/weak frames.
            # Keep high-water marks and cooldown, but require a fresh first vote.
            state["signature"], state["accepted"], state["votes"] = None, None, 0
            state["last_seen"] = -math.inf
            state["last_sequence"] = max(int(state["last_sequence"]), frame.sequence)
            state["last_received"] = max(float(state["last_received"]), frame.received_monotonic)
            state["last_completed"] = max(float(state["last_completed"]), now)
            return None
        continuity_gap = (
            now - float(state["last_seen"]) > 3
            or frame.received_monotonic - float(state["last_received"]) > 3
        )
        state["last_sequence"] = frame.sequence
        state["last_received"], state["last_completed"] = frame.received_monotonic, now
        if not signature or continuity_gap:
            state["signature"] = None
            state["votes"], state["accepted"] = 0, None
            state["first_seen"] = now
            state["first_received"] = frame.received_monotonic
        if signature and signature == state["signature"]:
            if (
                now - float(state["first_seen"]) >= self.minimum_confirmation_seconds
                and frame.received_monotonic - float(state["first_received"])
                >= self.minimum_confirmation_seconds
            ):
                state["votes"] = int(state["votes"]) + 1
        else:
            state["votes"] = 1 if signature else 0
            state["first_seen"] = now
            state["first_received"] = frame.received_monotonic
        state["signature"], state["last_seen"] = signature, now
        if (
            signature
            and int(state["votes"]) >= 2
            and now - float(state["first_seen"]) >= self.minimum_confirmation_seconds
            and frame.received_monotonic - float(state["first_received"])
            >= self.minimum_confirmation_seconds
            and signature != state["accepted"]
            and now - float(state["last_trigger"]) >= 30
        ):
            state["accepted"], state["last_trigger"] = signature, now
            return ReviewCandidate(
                observation, "alternate_view_person_hint", 1, now, "rotation_180"
            )
        return None


def _signature(detections: tuple[Detection, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            (
                round((item.bbox_xyxy[0] + item.bbox_xyxy[2]) * 4),
                round((item.bbox_xyxy[1] + item.bbox_xyxy[3]) * 4),
            )
            for item in detections
        )
    )


def _thumbnail(frame: LiveFrame) -> bytes:
    # Fixed pixel samples, independent of frame dimensions; no optional dependencies.
    return bytes(
        frame.pixels[((y * frame.height // 12) * frame.width + x * frame.width // 16) * 3 + channel]
        for y in range(12)
        for x in range(16)
        for channel in range(3)
    )


class ReviewTriggerPolicy:
    """Trigger on confirmed detection changes, scene changes, and supplemental patrol."""

    def __init__(self, *, patrol_seconds: float = 120.0):
        if not 10 <= patrol_seconds <= 3600:
            raise ValueError("invalid supplemental patrol interval")
        self.patrol_seconds = patrol_seconds
        self.state: dict[str, dict[str, object]] = {}

    def observe(self, observation: LiveObservation) -> ReviewCandidate | None:
        frame = observation.frame
        signature = _signature(observation.detections)
        thumbnail = _thumbnail(frame)
        state = self.state.get(frame.camera_id)
        now = observation.completed_monotonic
        if state is None or state["generation"] != frame.generation:
            self.state[frame.camera_id] = {
                "generation": frame.generation,
                "accepted": signature,
                "pending": signature,
                "votes": 0,
                "thumbnail": thumbnail,
                "last_review": now,
            }
            return ReviewCandidate(observation, "initial_or_reconnected_scene", 0, now)
        if signature == state["pending"]:
            state["votes"] = int(state["votes"]) + 1
        else:
            state["pending"], state["votes"] = signature, 1
        reason = None
        priority = 0
        if signature != state["accepted"] and int(state["votes"]) >= 2:
            reason = (
                "person_count_changed"
                if len(signature) != len(state["accepted"])
                else "person_position_changed"
            )
            priority = 2
            state["accepted"] = signature
        previous_thumbnail = state["thumbnail"]
        difference = sum(abs(a - b) for a, b in zip(thumbnail, previous_thumbnail, strict=True)) / (
            len(thumbnail) * 255
        )
        if reason is None and not signature and difference >= 0.12:
            reason, priority = "scene_changed_without_person", 1
        if reason is None and now - float(state["last_review"]) >= self.patrol_seconds:
            reason = "supplemental_missed_detection_review"
        if reason is None:
            return None
        state["last_review"], state["thumbnail"] = now, thumbnail
        return ReviewCandidate(observation, reason, priority, now)


def encode_bgr_png(frame: LiveFrame, *, profile: str = "compact") -> bytes:
    """Lossless PNG; compact uses Sub/zlib9, fast uses unfiltered rows/zlib1.

    Encoding belongs to the local-review worker, not the live detection thread.
    Both fixed profiles avoid optional image dependencies and preserve pixels.
    """
    if profile not in ("compact", "fast"):
        raise ValueError("invalid PNG encoding profile")
    rgb = bytearray(len(frame.pixels))
    rgb[0::3], rgb[1::3], rgb[2::3] = frame.pixels[2::3], frame.pixels[1::3], frame.pixels[0::3]
    stride = frame.width * 3
    scanlines = bytearray()
    for start in range(0, len(rgb), stride):
        row = rgb[start : start + stride]
        if profile == "fast":
            scanlines.append(0)
            scanlines.extend(row)
        else:
            # PNG Sub predicts each byte from the same previous-pixel channel.
            scanlines.append(1)
            scanlines.extend(row[:3])
            scanlines.extend(
                (value - previous) & 255 for value, previous in zip(row[3:], row[:-3], strict=True)
            )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", frame.width, frame.height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, 1 if profile == "fast" else 9))
        + chunk(b"IEND", b"")
    )
