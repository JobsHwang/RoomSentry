"""Offline experimental confirmation of measured, anonymous person candidates.

Not installed in the live pipeline. A confirmed geometric chain is only a local
review hint, never person presence, a unique identity, a role, or a safety fact.
No prediction, missed-frame filling, score fusion, or cross-camera association.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from baby_monitor.domain import Detection


def _number(value: object, lower: float, upper: float) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and lower <= value <= upper
    except (OverflowError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class ConfirmationSettings:
    minimum_candidate_confidence: float = 0.4
    minimum_confirmation_seconds: float = 1.0
    minimum_iou: float = 0.5
    maximum_gap_seconds: float = 3.0
    review_cooldown_seconds: float = 30.0

    def __post_init__(self) -> None:
        for value, lower, upper in (
            (self.minimum_candidate_confidence, 0, 1),
            (self.minimum_confirmation_seconds, 1, 3),
            (self.minimum_iou, 0.01, 1),
            (self.maximum_gap_seconds, 0.01, 3),
            (self.review_cooldown_seconds, 30, 120),
        ):
            if not _number(value, lower, upper):
                raise ValueError("invalid candidate confirmation settings")


@dataclass(frozen=True, slots=True)
class ConfirmationFrame:
    """Metadata from one checked view, all boxes in unchanged source coordinates.

    Receipt/completion are explicit elapsed clocks, not source PTS. The caller
    authenticates frame content and view bindings. Unchecked views must not be
    fabricated as zero detections; report a gap before the next measured frame.
    """

    camera_id: str
    generation: int
    sequence: int
    received_monotonic: float
    completed_monotonic: float
    detections: tuple[Detection, ...]

    def __post_init__(self) -> None:
        if (
            type(self.camera_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.camera_id) is None
            or any(type(v) is not int or v < 0 for v in (self.generation, self.sequence))
        ):
            raise ValueError("invalid candidate confirmation frame identity")
        if (
            not _number(self.received_monotonic, 0, 10**12)
            or not _number(self.completed_monotonic, 0, 10**12)
            or self.completed_monotonic < self.received_monotonic
        ):
            raise ValueError("invalid candidate confirmation clocks")
        if (
            type(self.detections) is not tuple
            or len(self.detections) > 100
            or any(type(d) is not Detection or d.label != "person" for d in self.detections)
        ):
            raise ValueError("confirmation requires at most 100 person detections")


@dataclass(frozen=True, slots=True)
class CandidateChainEvidence:
    chain_id: int
    candidate_index: int
    measured_frames: int
    receipt_span_seconds: float
    completion_span_seconds: float
    confirmed: bool


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    chains: tuple[CandidateChainEvidence, ...]
    review_requested_indices: tuple[int, ...]
    cooldown_blocked_indices: tuple[int, ...]
    ambiguous_indices: tuple[int, ...]
    below_gate_indices: tuple[int, ...]
    reset_reason: str | None
    source_at_box_limit: bool


@dataclass(slots=True)
class _Chain:
    chain_id: int
    bbox: tuple[float, float, float, float]
    first_received: float
    first_completed: float
    measured_frames: int = 1
    review_requested: bool = False


@dataclass(slots=True)
class _Camera:
    generation: int = -1
    last_sequence: int = -1
    last_received: float = -math.inf
    last_completed: float = -math.inf
    last_request: float = -math.inf
    next_chain_id: int = 0
    chains: list[_Chain] = field(default_factory=list)


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    union = (
        (left[2] - left[0]) * (left[3] - left[1])
        + (right[2] - right[0]) * (right[3] - right[1])
        - intersection
    )
    return intersection / union


class CandidateConfirmationPolicy:
    """Bounded, single-owner offline experiment; no production adapter imports it.

    Only an unambiguous one-to-one IoU edge can extend a chain. Overlapping
    current boxes and many-to-one/one-to-many edges reset affected chains rather
    than assigning identity. Even a unique edge can switch people; opaque chain
    IDs must NEVER carry age/role labels. Static false boxes can still confirm.
    """

    def __init__(self, camera_ids: tuple[str, ...], settings: ConfirmationSettings | None = None):
        if (
            type(camera_ids) is not tuple
            or not 1 <= len(camera_ids) <= 64
            or any(
                type(c) is not str or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", c) is None
                for c in camera_ids
            )
            or len(set(camera_ids)) != len(camera_ids)
        ):
            raise ValueError("invalid confirmation camera allowlist")
        if settings is not None and type(settings) is not ConfirmationSettings:
            raise ValueError("invalid confirmation settings type")
        self.settings = settings or ConfirmationSettings()
        self._state = {camera: _Camera() for camera in camera_ids}

    def gap(self, camera_id: str) -> None:
        """Explicit unchecked/decode/review gap; keep high-water marks and cooldown."""
        self._state[camera_id].chains.clear()

    def observe(self, frame: ConfirmationFrame) -> ConfirmationResult:
        if type(frame) is not ConfirmationFrame or frame.camera_id not in self._state:
            raise ValueError("unadmitted confirmation frame")
        state = self._state[frame.camera_id]
        below = tuple(
            i
            for i, d in enumerate(frame.detections)
            if d.confidence < self.settings.minimum_candidate_confidence
        )
        eligible = [i for i in range(len(frame.detections)) if i not in below]
        reset = None
        if frame.generation < state.generation:
            state.chains.clear()
            return ConfirmationResult(
                (), (), (), (), below, "stale_generation", len(frame.detections) == 100
            )
        if frame.generation > state.generation:
            state.chains.clear()
            state.generation = frame.generation
            state.last_sequence = -1
            state.last_received = state.last_completed = -math.inf
            reset = "generation"
        received, completed = frame.received_monotonic, frame.completed_monotonic
        if (
            frame.sequence <= state.last_sequence
            or received <= state.last_received
            or completed < state.last_completed
        ):
            state.chains.clear()
            state.last_sequence = max(state.last_sequence, frame.sequence)
            state.last_received = max(state.last_received, received)
            state.last_completed = max(state.last_completed, completed)
            return ConfirmationResult(
                (), (), (), (), below, "invalid_order", len(frame.detections) == 100
            )
        if (
            received - state.last_received > self.settings.maximum_gap_seconds
            or completed - state.last_completed > self.settings.maximum_gap_seconds
        ) and reset is None:
            state.chains.clear()
            reset = "clock_gap"
        state.last_sequence, state.last_received, state.last_completed = (
            frame.sequence,
            received,
            completed,
        )
        edges = {
            index: [
                j
                for j, previous in enumerate(state.chains)
                if _iou(frame.detections[index].bbox_xyxy, previous.bbox)
                >= self.settings.minimum_iou
            ]
            for index in eligible
        }
        degrees = [
            sum(j in neighbors for neighbors in edges.values()) for j in range(len(state.chains))
        ]
        ambiguous = {
            i for i in eligible if len(edges[i]) > 1 or any(degrees[j] > 1 for j in edges[i])
        }
        for offset, i in enumerate(eligible):
            for j in eligible[offset + 1 :]:
                if (
                    _iou(frame.detections[i].bbox_xyxy, frame.detections[j].bbox_xyxy)
                    >= self.settings.minimum_iou
                ):
                    ambiguous.update((i, j))
        new_chains = []
        evidence = []
        pending = []
        for index in eligible:
            if index in ambiguous:
                continue
            if edges[index]:
                chain = state.chains[edges[index][0]]
                chain.bbox = frame.detections[index].bbox_xyxy
                chain.measured_frames += 1
            else:
                chain = _Chain(
                    state.next_chain_id,
                    frame.detections[index].bbox_xyxy,
                    received,
                    completed,
                )
                state.next_chain_id += 1
            receipt_span = received - chain.first_received
            completion_span = completed - chain.first_completed
            confirmed = (
                chain.measured_frames >= 2
                and receipt_span >= self.settings.minimum_confirmation_seconds
                and completion_span >= self.settings.minimum_confirmation_seconds
            )
            evidence.append(
                CandidateChainEvidence(
                    chain.chain_id,
                    index,
                    chain.measured_frames,
                    receipt_span,
                    completion_span,
                    confirmed,
                )
            )
            if confirmed and not chain.review_requested:
                pending.append((index, chain))
            new_chains.append(chain)
        state.chains = new_chains
        requested = ()
        blocked = ()
        if pending:
            indices = tuple(index for index, _ in pending)
            if completed - state.last_request >= self.settings.review_cooldown_seconds:
                requested = indices
                state.last_request = completed
                for _, chain in pending:
                    chain.review_requested = True
            else:
                blocked = indices
        return ConfirmationResult(
            tuple(evidence),
            requested,
            blocked,
            tuple(sorted(ambiguous)),
            below,
            reset,
            len(frame.detections) == 100,
        )
