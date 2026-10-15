"""Bounded, incremental, single-camera association experiment (not deployment).

No future batch hash is needed to name current anonymous assignments. Actual
elapsed input time resets gaps and expires exported IDs, but does NOT change
the upstream fixed-step Kalman model. Expiration is not an accuracy fix.
The native global ID counter requires exclusive process ownership; use a
context manager and one dedicated process per camera/label in future shadow
integrations. This module neither reads media nor carries person roles.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

from baby_monitor.local_runners.tracker_replay import (
    _REPLAY_LOCK,
    TrackerReplayConfig,
    TrackerReplayError,
    _finite_number,
    _validated_backend_assignments,
    make_backend,
)
from baby_monitor.object_tracker import DetectionFrame
from baby_monitor.tracker_evaluation.prediction import TrackerPredictionAssignment


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TrackerStreamConfig:
    native: TrackerReplayConfig
    max_export_age_seconds: float = 1.0
    duplicate_cache_frames: int = 32

    def __post_init__(self):
        if not isinstance(self.native, TrackerReplayConfig):
            raise TrackerReplayError("invalid incremental native config")
        if (
            not _finite_number(self.max_export_age_seconds)
            or not 0 < self.max_export_age_seconds <= 10
        ):
            raise TrackerReplayError("invalid incremental export age")
        if (
            type(self.duplicate_cache_frames) is not int
            or not 1 <= self.duplicate_cache_frames <= 256
        ):
            raise TrackerReplayError("invalid incremental duplicate cache bound")

    def to_dict(self):
        return {
            "version": 1,
            "native": self.native.to_dict(),
            "max_export_age_seconds": self.max_export_age_seconds,
            "duplicate_cache_frames": self.duplicate_cache_frames,
            "max_frames": 5000,
            "max_detections": 10000,
            "max_live_exports": 4096,
            "deployment_approved": False,
            "kalman_elapsed_dt_used": False,
            "native_history_expired_by_export_age": False,
            "export_age_clock_policy": "either_elapsed_or_utc_exceeds_limit",
            "identity_or_role_propagation": False,
            "process_policy": "exclusive_one_camera_one_label_bounded_experiment",
        }

    @property
    def fingerprint(self):
        return _sha(self.to_dict())


@dataclass(frozen=True, slots=True)
class TrackerStreamResult:
    """Current-input assignments only; phases are native lifecycle, not truth."""

    camera_id: str
    detector_adapter: str
    frame_sha256: str
    config_sha256: str
    generation: int
    segment: int
    elapsed_seconds: float
    assignments: tuple[TrackerPredictionAssignment, ...]
    phases: tuple[str, ...]
    reset_reason: str | None
    expired_export_count: int
    backend_update_count: int

    def bind(self, frame: DetectionFrame):
        if (
            not isinstance(frame, DetectionFrame)
            or frame.fingerprint != self.frame_sha256
            or tuple((a.frame_id, a.detection_id) for a in self.assignments)
            != tuple((frame.frame_id, d.detection_id) for d in frame.detections)
            or len(self.phases) != len(self.assignments)
            or any(
                p not in ("native_emitted", "native_tentative", "unassociated_singleton")
                for p in self.phases
            )
            or len({a.tracking_id for a in self.assignments}) != len(self.assignments)
        ):
            raise TrackerReplayError("incremental result does not bind to frame")

    def to_dict(self):
        return {
            "version": 1,
            "camera_id": self.camera_id,
            "detector_adapter": self.detector_adapter,
            "frame_sha256": self.frame_sha256,
            "config_sha256": self.config_sha256,
            "generation": self.generation,
            "segment": self.segment,
            "elapsed_seconds": self.elapsed_seconds,
            "assignments": [a.to_dict() for a in self.assignments],
            "assignment_phases": list(self.phases),
            "reset_reason": self.reset_reason,
            "expired_export_count": self.expired_export_count,
            "backend_update_count": self.backend_update_count,
            "deployment_approved": False,
            "identity_or_role_propagation": False,
            "kalman_elapsed_dt_used": False,
        }


class IncrementalTrackerExperiment:
    def __init__(
        self,
        *,
        camera_id: str,
        detector_adapter: str,
        label: str,
        session_id: str,
        config: TrackerStreamConfig,
        _backend_factory=None,
    ):
        if any(
            type(value) is not str or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value) is None
            for value in (camera_id, detector_adapter, label)
        ):
            raise TrackerReplayError("invalid incremental camera, adapter or label")
        if type(session_id) is not str or re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
            raise TrackerReplayError("incremental session must be opaque 32-hex ID")
        if not isinstance(config, TrackerStreamConfig):
            raise TrackerReplayError("invalid incremental config")
        self._camera, self._detector, self._label, self._config = (
            camera_id,
            detector_adapter,
            label,
            config,
        )
        self._prefix = (
            "s_" + _sha([camera_id, detector_adapter, label, session_id, config.fingerprint])[:16]
        )
        self._factory = make_backend if _backend_factory is None else _backend_factory
        self._runtime_id = (
            "ultralytics_8_4_115_lap_0_5_12"
            if _backend_factory is None
            else "injected_stdlib_test_backend"
        )
        self._backend = None
        self._runtime_provenance = None
        self._previous = None
        self._segment = -1
        self._exports = {}
        self._cache = OrderedDict()
        self._seen_keys = set()
        self._frames = self._detections = self._serial = 0
        self._attempted_updates = 0
        self._call_lock = threading.Lock()
        self._closed = False
        if not _REPLAY_LOCK.acquire(blocking=False):
            raise TrackerReplayError("native tracker process already in use")

    def __enter__(self):
        if self._closed:
            raise TrackerReplayError("incremental tracker is closed")
        return self

    def __exit__(self, *args):
        self.close()

    def _close_unlocked(self):
        if not self._closed:
            self._closed = True
            self._backend = None
            self._exports.clear()
            self._cache.clear()
            self._seen_keys.clear()
            _REPLAY_LOCK.release()

    def close(self):
        with self._call_lock:
            self._close_unlocked()

    @property
    def runtime_id(self):
        return self._runtime_id if self._attempted_updates else "no_backend_invoked"

    @property
    def attempted_update_count(self):
        return self._attempted_updates

    @property
    def runtime_provenance(self):
        # Never expose a mutable alias into the adapter's runtime checks.
        return json.loads(json.dumps(self._runtime_provenance))

    def update(self, frame: DetectionFrame, *, generation: int, elapsed_seconds: float):
        if not self._call_lock.acquire(blocking=False):
            raise TrackerReplayError("concurrent incremental update refused")
        try:
            return self._update(frame, generation, elapsed_seconds)
        finally:
            self._call_lock.release()

    def _update(self, frame, generation, elapsed):
        if self._closed:
            raise TrackerReplayError("incremental tracker is closed")
        if (
            not isinstance(frame, DetectionFrame)
            or type(generation) is not int
            or not 0 <= generation <= 2**31 - 1
            or not _finite_number(elapsed)
            or not 0 <= elapsed <= 1e9
            or len(frame.detections) > 128
            or any(d.label != self._label for d in frame.detections)
        ):
            raise TrackerReplayError("invalid incremental input")
        if self._previous is not None and generation < self._previous[0]:
            raise TrackerReplayError("stale incremental source generation")
        key = (generation, frame.frame_id)
        cached = self._cache.get(key)
        if cached is not None:
            if cached.frame_sha256 != frame.fingerprint or cached.elapsed_seconds != elapsed:
                raise TrackerReplayError("conflicting incremental duplicate")
            return cached
        if key in self._seen_keys:
            raise TrackerReplayError("incremental duplicate outside retry cache")
        previous = self._previous
        if previous is not None and (
            generation < previous[0]
            or elapsed <= previous[1]
            or (generation == previous[0] and frame.occurred_at <= previous[2])
        ):
            raise TrackerReplayError("stale incremental input")
        if self._frames >= 5000 or self._detections + len(frame.detections) > 10000:
            raise TrackerReplayError("incremental experiment bound reached")
        reset = (
            "first_input"
            if previous is None
            else "source_generation_changed"
            if generation != previous[0]
            else "elapsed_input_gap"
            if elapsed - previous[1] > self._config.native.max_input_gap_seconds
            else "utc_input_gap"
            if (frame.occurred_at - previous[2]).total_seconds()
            > self._config.native.max_input_gap_seconds
            else None
        )
        # From backend construction onward, a failure may have mutated native
        # state. Fail-stop the session instead of pretending the input was atomic.
        try:
            if reset is not None:
                self._backend = self._factory(self._config.native)
                provenance = getattr(self._backend, "provenance", None)
                frozen = json.loads(json.dumps(provenance, allow_nan=False))
                if self._segment >= 0 and frozen != self._runtime_provenance:
                    raise TrackerReplayError("incremental runtime changed between segments")
                self._runtime_provenance = frozen
                self._segment += 1
                self._exports.clear()
            expired = [
                native_id
                for native_id, (_, last_elapsed, last_utc) in self._exports.items()
                if max(elapsed - last_elapsed, (frame.occurred_at - last_utc).total_seconds())
                > self._config.max_export_age_seconds
            ]
            for native_id in expired:
                del self._exports[native_id]
            self._attempted_updates += 1
            rows = self._backend.update(frame.detections)
            emitted, tentative = _validated_backend_assignments(
                self._backend, rows, frame.detections, self._config.native
            )
            assignments, phases = [], []
            for index, detection in enumerate(frame.detections):
                native_id = emitted.get(index)
                if native_id is None:
                    tracking_id = (
                        self._prefix
                        + "_u_"
                        + _sha([generation, frame.frame_id, detection.detection_id])[:32]
                    )
                    phase = "unassociated_singleton"
                else:
                    existing = self._exports.get(native_id)
                    if existing is None:
                        self._serial += 1
                        tracking_id = f"{self._prefix}_g{self._segment}_e{self._serial}"
                    else:
                        tracking_id = existing[0]
                    self._exports[native_id] = (tracking_id, elapsed, frame.occurred_at)
                    phase = "native_tentative" if index in tentative else "native_emitted"
                assignments.append(
                    TrackerPredictionAssignment(frame.frame_id, detection.detection_id, tracking_id)
                )
                phases.append(phase)
            if len(self._exports) > 4096:
                raise TrackerReplayError("incremental live export bound reached")
            result = TrackerStreamResult(
                self._camera,
                self._detector,
                frame.fingerprint,
                self._config.fingerprint,
                generation,
                self._segment,
                elapsed,
                tuple(assignments),
                tuple(phases),
                reset,
                len(expired),
                self._frames + 1,
            )
            result.bind(frame)
        except Exception:
            self._close_unlocked()
            raise TrackerReplayError("incremental backend failed; session closed") from None
        self._previous = (generation, elapsed, frame.occurred_at)
        self._frames += 1
        self._detections += len(frame.detections)
        self._cache[key] = result
        self._seen_keys.add(key)
        while len(self._cache) > self._config.duplicate_cache_frames:
            self._cache.popitem(last=False)
        return result
