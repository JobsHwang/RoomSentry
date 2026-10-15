"""Offline Ultralytics association adapter, not detection or person identity.

The fixed upstream uses one Kalman step per sampled update, not elapsed dt.
Actual input gaps only reset a segment. All original detections are retained;
upstream-unreturned detections get explicit one-frame singleton assignments.
No image, ReID encoder, model weights or VLM are used by this adapter.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import socket
import threading
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from baby_monitor.object_tracker import DetectionFrameBatch
from baby_monitor.tracker_evaluation.prediction import (
    TrackerPredictionAssignment,
    TrackerPredictionBatch,
)

UPSTREAM_VERSION = "8.4.115"
LAP_VERSION = "0.5.12"
_REPLAY_LOCK = threading.Lock()


def _finite_number(value):
    return type(value) in (int, float) and -1e308 <= value <= 1e308 and math.isfinite(value)


class TrackerReplayError(ValueError):
    """Path-free failure in a bounded offline association experiment."""


@dataclass(frozen=True, slots=True)
class TrackerReplayConfig:
    algorithm: str
    canvas_width: int
    canvas_height: int
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    max_input_gap_seconds: float = 1.0
    emit_tentative_assignments: bool = False

    def __post_init__(self):
        if self.algorithm not in ("bytetrack", "botsort"):
            raise TrackerReplayError("unsupported offline tracker")
        for value in (self.canvas_width, self.canvas_height):
            if type(value) is not int or not 1 <= value <= 4096:
                raise TrackerReplayError("invalid tracker canvas")
        for value in (
            self.track_high_thresh,
            self.track_low_thresh,
            self.new_track_thresh,
            self.match_thresh,
        ):
            if not _finite_number(value) or not 0 <= value <= 1:
                raise TrackerReplayError("invalid tracker threshold")
        if not self.track_low_thresh < self.track_high_thresh <= self.new_track_thresh:
            raise TrackerReplayError("invalid tracker threshold order")
        if type(self.track_buffer) is not int or not 1 <= self.track_buffer <= 300:
            raise TrackerReplayError("invalid tracker buffer")
        if type(self.fuse_score) is not bool:
            raise TrackerReplayError("invalid tracker score fusion")
        if type(self.emit_tentative_assignments) is not bool:
            raise TrackerReplayError("invalid tentative tracker output policy")
        if (
            not _finite_number(self.max_input_gap_seconds)
            or not 0 < self.max_input_gap_seconds <= 10
        ):
            raise TrackerReplayError("invalid tracker gap reset")

    def to_dict(self):
        return {
            **asdict(self),
            "version": 1,
            "upstream_version": UPSTREAM_VERSION,
            "lap_version": LAP_VERSION,
            "with_reid": False,
            "gmc_method": "none",
            "time_model": "native_fixed_step_per_sampled_update",
            "unreturned_policy": "one_frame_singleton_not_backend_track",
            "tentative_output_policy": "explicit_phase_current_detection_only_opt_in",
            "label_state_policy": "sequential_independent_labels",
            "invocation_state_policy": "serialized_native_global_counter",
        }

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def make_backend(config: TrackerReplayConfig):
    """Load only pinned association code inside an offline optional runtime."""
    if (
        {name for _, name in socket.if_nameindex()} != {"lo"}
        or os.environ.get("YOLO_AUTOINSTALL") != "false"
        or os.environ.get("YOLO_OFFLINE") != "true"
    ):
        raise TrackerReplayError("tracker requires isolated offline runtime")
    if (
        importlib.metadata.version("ultralytics") != UPSTREAM_VERSION
        or importlib.metadata.version("lap") != LAP_VERSION
    ):
        raise TrackerReplayError("tracker dependency version mismatch")
    import numpy as np
    from ultralytics.trackers.basetrack import TrackState
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker

    args = SimpleNamespace(
        **asdict(config),
        with_reid=False,
        gmc_method="none",
        proximity_thresh=0.5,
        appearance_thresh=0.8,
        model="auto",
    )
    tracker = (BYTETracker if config.algorithm == "bytetrack" else BOTSORT)(args)
    if getattr(tracker, "encoder", None) is not None:
        raise TrackerReplayError("unexpected tracker encoder")
    distribution = importlib.metadata.distribution("ultralytics")
    source_files = (
        "ultralytics/trackers/byte_tracker.py",
        "ultralytics/trackers/bot_sort.py",
        "ultralytics/trackers/basetrack.py",
        "ultralytics/trackers/utils/stracks.py",
        "ultralytics/trackers/utils/kalman_filter.py",
        "ultralytics/trackers/utils/matching.py",
        "ultralytics/trackers/utils/gmc.py",
    )
    runtime_provenance = {
        "ultralytics_version": UPSTREAM_VERSION,
        "lap_version": LAP_VERSION,
        "numpy_version": importlib.metadata.version("numpy"),
        "tracker_source_sha256": {
            name: hashlib.sha256(distribution.locate_file(name).read_bytes()).hexdigest()
            for name in source_files
        },
    }

    class Boxes:
        def __init__(self, rows):
            self.rows = np.asarray(rows, dtype=np.float32).reshape((-1, 5))
            self.conf = self.rows[:, 4]
            self.cls = np.zeros(len(self.rows), dtype=np.float32)
            self.xyxy = self.rows[:, :4]
            self.xywh = self.xyxy.copy()
            self.xywh[:, :2] = (self.xyxy[:, :2] + self.xyxy[:, 2:]) / 2
            self.xywh[:, 2:] = self.xyxy[:, 2:] - self.xyxy[:, :2]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, selector):
            return Boxes(self.rows[selector])

    class Backend:
        provenance = runtime_provenance
        tentative_indices = ()

        def update(self, detections):
            rows = []
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox_xyxy
                rows.append(
                    [
                        x1 * config.canvas_width,
                        y1 * config.canvas_height,
                        x2 * config.canvas_width,
                        y2 * config.canvas_height,
                        detection.confidence,
                    ]
                )
            emitted = tracker.update(Boxes(rows), img=None, feats=None).tolist()
            self.tentative_indices = ()
            if config.emit_tentative_assignments:
                tentative = _current_tentative_rows(tracker, TrackState.Tracked)
                # Convert NumPy scalar fields to the same bounded row format as
                # native update(). This does not change native state or geometry.
                tentative = np.asarray(tentative, dtype=np.float32).reshape((-1, 8)).tolist()
                self.tentative_indices = tuple(row[7] for row in tentative)
                emitted.extend(tentative)
            return emitted

    return Backend()


def _current_tentative_rows(tracker, tracked_state):
    """Expose only actual current-frame new candidates, never lost predictions.

    This opt-in phase remains tentative. It is not a confirmed subject, a role
    or an identity, and must not be promoted into production by this replay.
    """
    return [
        track.result
        for track in tracker.tracked_stracks
        if track.is_activated is False
        and track.state == tracked_state
        and track.frame_id == tracker.frame_id
    ]


def _integer(value, minimum, maximum):
    if not _finite_number(value) or not minimum <= value <= maximum or value != int(value):
        raise TrackerReplayError("invalid tracker output index or ID")
    return int(value)


def replay_tracker(
    batch: DetectionFrameBatch,
    config: TrackerReplayConfig,
    *,
    _backend_factory=None,
) -> tuple[TrackerPredictionBatch, dict]:
    # Native STrack IDs are process-global. Serialize *whole* replays as well as
    # labels/segments, so concurrent callers cannot reset one another's counter.
    if not _REPLAY_LOCK.acquire(blocking=False):
        raise TrackerReplayError("native tracker process already in use")
    try:
        return _replay_tracker(batch, config, _backend_factory=_backend_factory)
    finally:
        _REPLAY_LOCK.release()


def _validated_backend_assignments(backend, rows, detections, config):
    """Validate the explicit input indices before using any native association."""
    if type(rows) is not list or len(rows) > len(detections):
        raise TrackerReplayError("invalid tracker output count")
    values = getattr(backend, "tentative_indices", ())
    if type(values) is not tuple or len(values) > len(rows):
        raise TrackerReplayError("invalid tentative tracker index list")
    tentative = {_integer(v, 0, len(detections) - 1) for v in values}
    if len(tentative) != len(values) or (tentative and not config.emit_tentative_assignments):
        raise TrackerReplayError("unexpected tentative tracker assignment")
    emitted, used_tracks = {}, set()
    for row in rows:
        if (
            type(row) is not list
            or len(row) != 8
            or any(not _finite_number(v) for v in row)
            or row[2] <= row[0]
            or row[3] <= row[1]
        ):
            raise TrackerReplayError("invalid tracker output row")
        native_id = _integer(row[4], 1, 2**31 - 1)
        index = _integer(row[7], 0, len(detections) - 1)
        if (
            index in emitted
            or native_id in used_tracks
            or row[6] != 0
            or not math.isclose(row[5], detections[index].confidence, abs_tol=1e-6)
        ):
            raise TrackerReplayError("tracker output mapping mismatch")
        emitted[index] = native_id
        used_tracks.add(native_id)
    if not tentative.issubset(emitted):
        raise TrackerReplayError("unbound tentative tracker assignment")
    return emitted, tentative


def _replay_tracker(batch, config, *, _backend_factory=None):
    if not isinstance(batch, DetectionFrameBatch) or not isinstance(config, TrackerReplayConfig):
        raise TrackerReplayError("invalid tracker replay inputs")
    if len(batch.frames) > 5000 or sum(len(f.detections) for f in batch.frames) > 10000:
        raise TrackerReplayError("tracker replay exceeds bounded experiment")
    if any(len(frame.detections) > 128 for frame in batch.frames):
        raise TrackerReplayError("tracker replay frame exceeds bounded experiment")
    factory = make_backend if _backend_factory is None else _backend_factory
    batch_sha = batch.fingerprint
    config_sha = config.fingerprint
    labels = sorted({d.label for f in batch.frames for d in f.detections})
    if len(labels) > 16:
        raise TrackerReplayError("tracker replay label count exceeds bound")
    assignments = {}
    provenance = {}
    resets = 0
    update_count = 0
    runtime_provenance = None
    # Upstream constructors reset a class-level ID counter. Finish a whole label
    # before constructing another instance, and namespace each reset generation.
    for label_index, label in enumerate(labels):
        backend, previous, segment = None, None, -1
        for frame in batch.frames:
            if (
                previous is None
                or (frame.occurred_at - previous).total_seconds() > config.max_input_gap_seconds
            ):
                backend = factory(config)
                current_provenance = getattr(backend, "provenance", None)
                if runtime_provenance is not None and current_provenance != runtime_provenance:
                    raise TrackerReplayError("tracker runtime changed between segments")
                runtime_provenance = current_provenance
                segment += 1
                resets += 1
            detections = tuple(d for d in frame.detections if d.label == label)
            rows = backend.update(detections)
            update_count += 1
            emitted, tentative_indices = _validated_backend_assignments(
                backend, rows, detections, config
            )
            for index, detection in enumerate(detections):
                key = (frame.frame_id, detection.detection_id)
                native_id = emitted.get(index)
                if native_id is None:
                    opaque = hashlib.sha256(
                        json.dumps([batch_sha, config_sha, *key], separators=(",", ":")).encode()
                    ).hexdigest()[:32]
                    tracking_id = "unassociated_" + opaque
                    provenance[key] = "unassociated_singleton"
                else:
                    tracking_id = f"track_l{label_index}_s{segment}_n{native_id}"
                    provenance[key] = (
                        "native_tentative" if index in tentative_indices else "native_emitted"
                    )
                assignments[key] = TrackerPredictionAssignment(*key, tracking_id)
            previous = frame.occurred_at
    ordered = tuple(
        assignments[(frame.frame_id, detection.detection_id)]
        for frame in batch.frames
        for detection in frame.detections
    )
    # A batch/config-scoped prefix prevents IDs accidentally linking two cameras
    # or experiments whose upstream numeric counters happen to be equal.
    prefix = hashlib.sha256((batch_sha + config_sha).encode()).hexdigest()[:16]
    ordered = tuple(
        TrackerPredictionAssignment(a.frame_id, a.detection_id, f"t_{prefix}_{a.tracking_id}")
        for a in ordered
    )
    result = TrackerPredictionBatch(
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
        detection_batch_sha256=batch_sha,
        tracker_adapter="ultralytics_" + config.algorithm + "_offline_v1",
        tracker_revision="fixed_8_4_115_no_reid_gmc_phased_v2",
        tracker_config_sha256=config_sha,
        runtime_id=(
            "no_backend_invoked"
            if not update_count
            else (
                "ultralytics_8_4_115_lap_0_5_12"
                if _backend_factory is None
                else "injected_stdlib_test_backend"
            )
        ),
        assignments=ordered,
    )
    result.bind(batch)
    gaps = [
        (b.occurred_at - a.occurred_at).total_seconds()
        for a, b in zip(batch.frames, batch.frames[1:], strict=False)
    ]
    emitted_count = sum(value == "native_emitted" for value in provenance.values())
    tentative_count = sum(value == "native_tentative" for value in provenance.values())
    audit = {
        "version": 1,
        "detection_batch_sha256": batch_sha,
        "prediction_sha256": result.fingerprint,
        "config": config.to_dict(),
        "frame_count": len(batch.frames),
        "empty_frame_count": sum(not f.detections for f in batch.frames),
        "backend_update_count": update_count,
        "backend_invoked": update_count > 0,
        "dependencies_verified": update_count > 0 and _backend_factory is None,
        "runtime_provenance": runtime_provenance,
        "label_segment_count": resets,
        "minimum_input_gap_seconds": min(gaps) if gaps else None,
        "maximum_input_gap_seconds": max(gaps) if gaps else None,
        "native_emitted_assignment_count": emitted_count,
        "native_tentative_assignment_count": tentative_count,
        "unassociated_singleton_count": len(ordered) - emitted_count - tentative_count,
        "assignment_provenance": [
            {
                "frame_id": a.frame_id,
                "detection_id": a.detection_id,
                "status": provenance[(a.frame_id, a.detection_id)],
            }
            for a in ordered
        ],
        "kalman_elapsed_dt_used": False,
        "source_exposure_continuity_established": False,
        "detection_accuracy_evaluated": False,
        "identity_or_age_evaluated": False,
        "deployment_approved": False,
        "media_read": False,
        "model_weights_loaded": False,
    }
    return result, audit
