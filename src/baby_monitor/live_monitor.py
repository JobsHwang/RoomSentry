"""Four-stream continuous detection with one independent local review worker."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import stat
import threading
import time
import uuid
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from baby_monitor.adapters.camera import CameraSource
from baby_monitor.adapters.live_media import FFmpegLiveSource
from baby_monitor.adapters.vlm import VLMRequest
from baby_monitor.database import EventDatabase
from baby_monitor.domain import CandidateEvent, Detection, EventMedia
from baby_monitor.live_confirmation_shadow import LiveConfirmationShadow, summarize_shadow_probe
from baby_monitor.live_detection import (
    AlternateViewReviewPolicy,
    FrameMailbox,
    LiveObservation,
    MotionBurstSettings,
    ReviewCandidate,
    ReviewMailbox,
    ReviewTriggerPolicy,
    encode_bgr_png,
)
from baby_monitor.live_people import LivePeople
from baby_monitor.live_review_control import LiveReviewSuspension, process_start_ticks
from baby_monitor.live_transition_capture import TransitionCaptureWorker
from baby_monitor.live_vlm_protocol import validate_vlm_diagnostics
from baby_monitor.local_runners import subject_state_probe
from baby_monitor.local_runners.subject_probe import validate_bound_result
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy


def _write_private(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=str
    ).encode()
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    os.replace(temporary, path)


def _local_vlm_diagnostics_snapshot(local_vlm: Any) -> object:
    try:
        return getattr(local_vlm, "last_diagnostics", None)
    except Exception:
        # Optional telemetry must not change the review's business outcome.
        return None


def _updated_review_diagnostics(local_vlm: Any, previous: object) -> dict[str, Any] | None:
    current = _local_vlm_diagnostics_snapshot(local_vlm)
    if current is previous:
        # Encoding, evidence/audit validation, or reuse may bypass analyze().
        return None
    try:
        return validate_vlm_diagnostics(current)
    except ValueError:
        return None


_REVIEW_STORAGE_RESERVE_BYTES = 12 * 1024 * 1024
_REVIEW_MINIMUM_DISK_FREE_BYTES = 2 * 1024**3
_REVIEW_DIAGNOSTIC_RECENCY_SECONDS = 30.0


def _subject_projection(attempt, completed, claims, **context):
    """Keep a still-young completed frame only while its successor is analyzing."""
    evidence = _review_evidence_snapshot(attempt, **context)
    evidence["update_in_progress"] = False
    if evidence["status"] == "analyzing" and completed is not None:
        previous = _review_evidence_snapshot(completed["attempt"], **context)
        if previous["status"] == "recent_completed_review":
            evidence = {**previous, "update_in_progress": True}
            claims = completed["claims"]
    if evidence["status"] != "recent_completed_review":
        claims = None
    return {"subject_evidence": evidence, "subject_claims": claims}


def _review_evidence_snapshot(
    attempt: dict[str, Any] | None,
    *,
    now: float,
    source_phase: str | None,
    source_generation: int | None,
    monitor_phase: str,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Session-local freshness of the latest attempt, not a baby-state TTL.

    No persisted review is imported on startup. A newer attempt (including a
    failure) replaces the previous success for this diagnostic. Recency uses
    decoder-receipt monotonic time, never completion time or camera exposure.
    Live publication also supplies UTC: the larger of both nonnegative ages
    is used, conservatively expiring evidence when either clock says it is old.
    Synthetic elapsed-only callers may omit UTC, but live publication may not.
    The 30-second label is a diagnostic window, not validated live accuracy.
    """
    snapshot = {
        "status": "not_reviewed_this_session",
        "attempt_status": None,
        "frame_received_at": None,
        "completed_at": None,
        "frame_generation": None,
        "evidence_age_seconds": None,
        "diagnostic_recency_window_seconds": _REVIEW_DIAGNOSTIC_RECENCY_SECONDS,
        "time_semantics": "decoder_receipt_not_camera_exposure",
        "model_output_verified": False,
        "live_baby_state_established": False,
    }
    if attempt is not None:
        snapshot.update(
            attempt_status=attempt["status"],
            frame_received_at=attempt["frame_received_at"],
            completed_at=attempt["completed_at"],
            frame_generation=attempt["frame_generation"],
        )
        received = attempt["received_monotonic"]
        age = now - received
        clock_valid = math.isfinite(age) and age >= 0
        if clock_valid and now_utc is not None:
            try:
                stamp = datetime.fromisoformat(attempt["frame_received_at"])
                if now_utc.utcoffset() is None or stamp.utcoffset() is None:
                    raise ValueError
                utc_age = (now_utc - stamp).total_seconds()
                clock_valid = math.isfinite(utc_age) and utc_age >= 0
                age = max(age, utc_age)
            except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
                clock_valid = False
        if clock_valid:
            snapshot["evidence_age_seconds"] = round(age, 3)
        if not clock_valid:
            snapshot["status"] = "receipt_clock_unavailable"
        elif type(source_generation) is not int or source_generation < 0:
            snapshot["status"] = "source_generation_unavailable"
        elif source_generation != attempt["frame_generation"]:
            snapshot["status"] = "previous_source_generation"
        elif attempt["status"] == "analyzing":
            snapshot["status"] = "analyzing"
        elif attempt["status"] == "analysis_failed":
            snapshot["status"] = "latest_analysis_failed"
        elif attempt["status"] != "analyzed":
            snapshot["status"] = "attempt_unavailable"
        else:
            snapshot["status"] = (
                "recent_completed_review"
                if age <= _REVIEW_DIAGNOSTIC_RECENCY_SECONDS
                else "stale_completed_review"
            )
    # A dead/stopped source cannot become live merely because its result is young.
    if monitor_phase == "local_review_suspended":
        snapshot["status"] = "local_review_suspended"
    elif monitor_phase != "detecting":
        snapshot["status"] = "monitor_not_detecting"
    elif source_phase != "receiving":
        snapshot["status"] = "source_not_receiving"
    return snapshot


def _validate_review_storage_limits(max_storage_bytes: int, max_storage_entries: int) -> None:
    if (
        type(max_storage_bytes) is not int
        or not 64 * 1024 * 1024 <= max_storage_bytes <= 64 * 1024**3
        or type(max_storage_entries) is not int
        or not 1 <= max_storage_entries <= 200000
    ):
        raise ValueError("invalid review storage budget")


def _review_storage_summary(
    root: Path, max_storage_bytes: int, max_storage_entries: int = 5000
) -> dict[str, Any]:
    """One bounded scan, with fixed reasons and no paths or raw exception text.

    Entry and byte counts are lower bounds when ``scan_complete`` is false.
    The entry budget includes directories and transient atomic-write files,
    preserving the original admission limit semantics.
    """
    _validate_review_storage_limits(max_storage_bytes, max_storage_entries)
    total = 0
    entries = 0
    free = None
    reason = "available"
    complete = False
    try:
        with ExitStack() as resources:

            def open_directory(path, *, parent=None):
                scope = ExitStack()
                resources.callback(scope.close)
                descriptor = os.open(
                    path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent
                )
                scope.callback(os.close, descriptor)
                iterator = scope.enter_context(os.scandir(descriptor))
                return descriptor, iterator, scope

            # Stream each directory rather than pathlib's eager/suppressing glob.
            # Parent descriptors anchor child opens and O_NOFOLLOW rejects a
            # directory swapped for a symlink after its no-follow stat.
            pending = [open_directory(root)]
            while pending:
                descriptor, iterator, scope = pending[-1]
                try:
                    item = next(iterator)
                except StopIteration:
                    scope.close()
                    pending.pop()
                    continue
                entries += 1
                if entries >= max_storage_entries:
                    reason = "entry_limit"
                    break
                try:
                    metadata = item.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # Atomic replacement can remove a listed temporary filename.
                    continue
                if stat.S_ISLNK(metadata.st_mode):
                    reason = "unsafe_symlink"
                    break
                if stat.S_ISREG(metadata.st_mode):
                    total += metadata.st_size
                elif stat.S_ISDIR(metadata.st_mode):
                    pending.append(open_directory(item.name, parent=descriptor))
            complete = reason == "available"
    except OSError:
        reason = "io_error"
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        if reason == "available":
            reason = "io_error"
    if reason == "available":
        if total + _REVIEW_STORAGE_RESERVE_BYTES >= max_storage_bytes:
            reason = "capacity_limit"
        elif free < _REVIEW_MINIMUM_DISK_FREE_BYTES:
            reason = "disk_space_low"
    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "available": reason == "available",
        "reason": reason,
        "used_bytes": total,
        "entries": entries,
        "scan_complete": complete,
        "limit_bytes": max_storage_bytes,
        "limit_entries": max_storage_entries,
        "remaining_budget_bytes": max(0, max_storage_bytes - total),
        "remaining_entries": max(0, max_storage_entries - entries),
        "reserve_bytes": _REVIEW_STORAGE_RESERVE_BYTES,
        "free_disk_bytes": free,
        "minimum_free_disk_bytes": _REVIEW_MINIMUM_DISK_FREE_BYTES,
    }


def _review_storage_available(
    root: Path, max_storage_bytes: int, max_storage_entries: int = 5000
) -> bool:
    """Compatibility wrapper; admission diagnostics use the same scan result."""
    return _review_storage_summary(root, max_storage_bytes, max_storage_entries)["available"]


def _shadow_state() -> dict[str, Any]:
    return {
        "semantics": "diagnostic_only_not_additional_people_or_review_trigger",
        "checks": 0,
        "positive_frames": 0,
        "count_disagreement_frames": 0,
        "skipped": dict.fromkeys(("disabled", "native_empty", "rate_limited", "unsupported"), 0),
        "last_pair": None,
    }


def _record_shadow_observation(
    state: dict[str, Any],
    observation: LiveObservation,
    status: str | None,
    hints: tuple[Detection, ...] | None,
) -> None:
    """Bounded diagnostics only; this function cannot enqueue reviews or alter detections."""
    if status is None:
        status = "unsupported"
    if status != "checked":
        if status not in state["skipped"] or hints is not None:
            raise RuntimeError("invalid shadow skip result")
        state["skipped"][status] += 1
        return
    if (
        not observation.detections
        or not isinstance(hints, tuple)
        or len(hints) > 100
        or any(not isinstance(item, Detection) for item in hints)
    ):
        raise RuntimeError("invalid shadow diagnostic result")
    state["checks"] += 1
    state["positive_frames"] += bool(hints)
    state["count_disagreement_frames"] += len(hints) != len(observation.detections)
    frame = observation.frame
    state["last_pair"] = {
        "camera_id": frame.camera_id,
        "generation": frame.generation,
        "sequence": frame.sequence,
        "raw_bgr_sha256": frame.sha256,
        "frame_received_at": frame.received_at.isoformat(),
        "image_orientation": "original",
        "shadow_view": "rotation_180",
        "native_boxes": [
            {"confidence": item.confidence, "bbox_xyxy": list(item.bbox_xyxy)}
            for item in observation.detections
        ],
        "shadow_boxes": [
            {"confidence": item.confidence, "bbox_xyxy": list(item.bbox_xyxy)} for item in hints
        ],
    }


def _enqueue_review(queue, observation, candidate, *, review_task, evidence_view="original"):
    """Current-scene diagnostics coalesce; event triggers retain their evidence."""
    if review_task == "subjects":
        queue.put(
            ReviewCandidate(
                observation,
                "subject_current_scene",
                0,
                observation.completed_monotonic,
                evidence_view,
            )
        )
    elif candidate is not None:
        queue.put(candidate)
    else:
        queue.refresh(observation)


def _take_next_review(event_queue, subject_queue, preferred, now=None, *, now_utc=None):
    """One serialized slot, alternate when both tasks have work; never drain the other."""
    queues = {"event": event_queue, "subjects": subject_queue}
    other = "event" if preferred == "subjects" else "subjects"
    for task in (preferred, other):
        selected = (
            queues[task].take_live()
            if now is None and now_utc is None
            else queues[task].take(now, now_utc=now_utc)
        )
        if selected is not None:
            return task, selected, "event" if task == "subjects" else "subjects"
    return preferred, None, preferred


class ReviewTaskCycle:
    """Bounded subject preference without dropping or starving queued event work."""

    def __init__(self, subject_slots_per_event=1):
        if type(subject_slots_per_event) is not int or not 1 <= subject_slots_per_event <= 4:
            raise ValueError("invalid subject slots per event")
        self.subject_slots = subject_slots_per_event
        self.remaining = subject_slots_per_event

    def take(self, event_queue, subject_queue, now=None, *, now_utc=None):
        preferred = "subjects" if self.remaining > 0 else "event"
        task, selected, _ = _take_next_review(
            event_queue, subject_queue, preferred, now, now_utc=now_utc
        )
        if selected is not None:
            self.remaining = self.subject_slots if task == "event" else max(0, self.remaining - 1)
        return task, selected


def run_live_monitor(
    *,
    streams: tuple[tuple[CameraSource, int, int], ...],
    detector: Any,
    local_vlm: Any,
    root: Path,
    review_admission: Any,
    max_storage_bytes: int = 512 * 1024 * 1024,
    max_storage_entries: int = 5000,
    max_runtime_seconds: float | None = None,
    sample_fps: float | None = None,
    motion_burst: MotionBurstSettings | None = None,
    batch_coalesce_seconds: float = 0.0,
    review_task: str = "event",
    subject_slots_per_event: int = 1,
    alternate_review_min_frame_confidence: float = 0.0,
    review_png_profile: str = "compact",
    allow_review_suspend_signals: bool = False,
    confirmation_shadow: bool = False,
    people_tracking: bool = False,
    people_review_trace: bool = False,
    people_utc_geometry_continuity: bool = False,
    people_transition_capture: bool = False,
    reject_uniform_frames: bool = False,
    room_registry: Any = None,
) -> None:
    if type(reject_uniform_frames) is not bool:
        raise ValueError("invalid uniform-frame guard configuration")
    if type(people_utc_geometry_continuity) is not bool or (
        people_utc_geometry_continuity and not (people_tracking and people_review_trace)
    ):
        raise ValueError("UTC geometry continuity requires people tracking and review trace")
    if type(people_transition_capture) is not bool or (
        people_transition_capture and not people_review_trace
    ):
        raise ValueError("transition capture requires explicit people review trace")
    if type(people_review_trace) is not bool or (people_review_trace and not people_tracking):
        raise ValueError("people review trace requires explicit people tracking")
    if type(people_tracking) is not bool or (
        people_tracking
        and (
            review_task not in ("subjects", "hybrid")
            or not callable(getattr(local_vlm, "probe_subject_state", None))
        )
    ):
        raise ValueError("invalid people tracking or missing state probe capability")
    if type(confirmation_shadow) is not bool or (
        confirmation_shadow and not callable(getattr(local_vlm, "probe_subjects", None))
    ):
        raise ValueError("invalid confirmation shadow configuration")
    if type(allow_review_suspend_signals) is not bool or (
        allow_review_suspend_signals
        and (
            threading.current_thread() is not threading.main_thread()
            or not callable(getattr(local_vlm, "close", None))
        )
    ):
        raise ValueError("invalid review suspension configuration")
    suspension = LiveReviewSuspension() if allow_review_suspend_signals else None
    start_ticks = process_start_ticks(os.getpid()) if suspension is not None else None
    if review_png_profile not in ("compact", "fast"):
        raise ValueError("invalid review PNG encoding profile")
    task_cycle = ReviewTaskCycle(subject_slots_per_event)
    alternate_triggers = AlternateViewReviewPolicy(
        minimum_frame_confidence=alternate_review_min_frame_confidence
    )
    if review_task not in ("event", "subjects", "hybrid") or (
        review_task in ("subjects", "hybrid")
        and not callable(getattr(local_vlm, "probe_subjects", None))
    ):
        raise ValueError("invalid live review task or missing subject capability")
    if (
        not 1 <= len(streams) <= 4
        or len({item[0].camera_id for item in streams}) != len(streams)
        or root.resolve() != root
        or not root.is_dir()
        or root.stat().st_mode & 0o077
        or root.stat().st_uid != os.getuid()
    ):
        raise ValueError("invalid private live monitor configuration")
    _validate_review_storage_limits(max_storage_bytes, max_storage_entries)
    camera_ids = tuple(item[0].camera_id for item in streams)
    people = (
        LivePeople(
            camera_ids,
            room_registry=room_registry,
            auxiliary=True,
            association_diagnostics=people_review_trace,
            utc_geometry_continuity=people_utc_geometry_continuity,
        )
        if people_tracking
        else None
    )
    confirmation = (
        LiveConfirmationShadow(
            camera_ids, minimum_frame_confidence=alternate_review_min_frame_confidence
        )
        if confirmation_shadow
        else None
    )
    frame_mailbox = FrameMailbox(
        camera_ids,
        sample_fps=sample_fps,
        motion_burst=motion_burst,
        batch_coalesce_seconds=batch_coalesce_seconds,
    )
    lock_descriptor = os.open(root / "worker.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ("frames", "records", "reviews"):
        directory = root / name
        if directory.is_symlink():
            raise ValueError("unsafe live output directory")
        directory.mkdir(mode=0o700, exist_ok=True)
    review_mailbox = ReviewMailbox(
        camera_ids, max_evidence_age_seconds=10.0 if review_task == "subjects" else 120.0
    )
    subject_mailbox = (
        ReviewMailbox(camera_ids, max_evidence_age_seconds=10.0)
        if review_task == "hybrid"
        else None
    )
    triggers = ReviewTriggerPolicy()
    sources = [
        FFmpegLiveSource(
            source,
            width=width,
            height=height,
            mailbox=frame_mailbox,
            **({"reject_uniform_frames": True} if reject_uniform_frames else {}),
        )
        for source, width, height in streams
    ]
    database_path = root / "debug-events.sqlite3"
    descriptor = os.open(database_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o077
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise ValueError("unsafe live event database")
    finally:
        os.close(descriptor)
    database = EventDatabase(database_path)
    database.initialize()
    stop = threading.Event()
    review_worker_failed = threading.Event()
    state_lock = threading.Lock()
    session_started = datetime.now(UTC)
    transition_capture = None
    if people_transition_capture:
        capture_root = root / "transition-captures"
        if capture_root.is_symlink():
            raise ValueError("unsafe transition capture directory")
        capture_root.mkdir(mode=0o700, exist_ok=True)
        if capture_root.stat().st_mode & 0o077 or capture_root.stat().st_uid != os.getuid():
            raise ValueError("unsafe transition capture directory")

        def capture_admission():
            storage = _review_storage_summary(root, max_storage_bytes, max_storage_entries)
            # A <=3-image bundle fits in 8 MiB including metadata; additionally
            # retain the normal review reserve while the sole encoder writes.
            return (
                storage["available"]
                and storage["remaining_budget_bytes"] > storage["reserve_bytes"] + 8 * 1024**2
                and storage["remaining_entries"] >= 8
                and storage["free_disk_bytes"] >= storage["minimum_free_disk_bytes"] + 8 * 1024**2
            )

        transition_capture = TransitionCaptureWorker(
            camera_ids,
            root=capture_root,
            session_started_at=session_started.isoformat(),
            admission=capture_admission,
        )
    began = time.monotonic()
    detecting_began: float | None = None
    recent_frames: dict[str, deque[float]] = {camera: deque() for camera in camera_ids}
    review_attempts: dict[str, dict[str, Any] | None] = dict.fromkeys(camera_ids)
    subject_attempts: dict[str, dict[str, Any] | None] = dict.fromkeys(camera_ids)
    subject_completed: dict[str, dict[str, Any] | None] = dict.fromkeys(camera_ids)
    state: dict[str, Any] = {
        "version": 1,
        "worker_pid": os.getpid(),
        "worker_start_ticks": start_ticks,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "scope": "four_stream_continuous_local_debug",
        "session_started_at": session_started.isoformat(),
        "phase": "initializing_detector",
        "remote_enabled": False,
        "production_writeback": False,
        "source_clock_verified": False,
        "baby_fields_basis": "unverified_track_bound_model_inferences"
        if people is not None
        else "unimplemented_state_estimator_unknown_placeholders",
        "people_tracking_enabled": people_tracking,
        "people_review_trace_enabled": people_review_trace,
        "people_transition_capture_enabled": people_transition_capture,
        "time_semantics": "decoder_receipt_not_camera_exposure",
        "sampling": {
            "mode": "all_decoded_frames" if sample_fps is None else "scheduled_latest_eligible",
            "target_fps_per_camera": sample_fps,
            "clock": "decoder_receipt_monotonic",
            "pending_capacity_per_camera": 32 if sample_fps is None else 1,
            "burst_pending_capacity_per_camera": None
            if motion_burst is None
            else motion_burst.pending_capacity,
            "max_receipt_age_seconds": 1.0,
            "admission_age_basis": "maximum_decoder_receipt_elapsed_and_utc",
            "adaptive_burst_enabled": motion_burst is not None,
            "motion_burst": None if motion_burst is None else asdict(motion_burst),
            "motion_semantics": "sampling_hint_not_person_identity_or_safety",
            "batch_coalesce_seconds": batch_coalesce_seconds,
        },
        "review_successes": 0,
        "review_failures": 0,
        "review_task": review_task,
        "review_png_profile": review_png_profile,
        "subject_slots_per_event": subject_slots_per_event if review_task == "hybrid" else None,
        "alternate_review_min_frame_confidence": alternate_review_min_frame_confidence,
        "review_scheduling": {
            "subjects": "latest_scene_fair",
            "event": "event_evidence",
            "hybrid": "weighted_event_and_latest_scene",
        }[review_task],
        "subject_probe_successes": 0,
        "subject_probe_failures": 0,
        "subject_not_due_discards": 0,
        "review_suspended_frames": 0,
        "review_pre_resume_frame_drops": 0,
        "review_storage_limit_bytes": max_storage_bytes,
        "review_storage_limit_entries": max_storage_entries,
        "review_storage": None,
        "local_review_mode": getattr(local_vlm, "execution_mode", "per_request"),
        "inference_batches": 0,
        "inference_batch_sizes": dict.fromkeys(("1", "2", "3", "4"), 0),
        "active_review_camera": None,
        "active_review_task": None,
        "detection_failures": 0,
        "last_inference_batch_ms": None,
        "last_inference_batch_size": None,
        "cameras": {
            camera: {
                "inferred_frames": 0,
                "last_detection_received_at": None,
                "last_person_count": None,
                "last_boxes": [],
                "max_receipt_to_detection_ms": 0.0,
                "last_receipt_to_detection_ms": None,
                "last_review_frame_received_at": None,
                "last_review_completed_at": None,
                "last_review_event_type": None,
                "last_review_error": None,
                "last_review_duration_ms": None,
                "alternate_view_checks": 0,
                "alternate_view_positive_frames": 0,
                "alternate_view_review_triggers": 0,
                "alternate_view_below_gate_frames": 0,
                "alternate_scale_fallback_checks": 0,
                "alternate_scale_fallback_positive_frames": 0,
                "nonempty_shadow": _shadow_state(),
                "baby_location": "unknown",
                "baby_state": "unknown",
                "subject_claims": None,
            }
            for camera in camera_ids
        },
    }
    if people_utc_geometry_continuity:
        state["people_utc_geometry_continuity"] = {
            "enabled": True,
            "revision": "utc_only_geometry_role_scope_v1",
            "anonymous_geometry_only": True,
            "role_evidence_scope_reset_unchanged": True,
            "dual_clock_freshness_unchanged": True,
        }
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, lambda *_: stop.set())
        if suspension is not None:
            for signum, action in (
                (signal.SIGUSR1, suspension.request_suspend),
                (signal.SIGUSR2, suspension.request_resume),
            ):
                old_handlers[signum] = signal.signal(signum, lambda *_, action=action: action())

    def review_loop() -> None:
        regular_since_shadow = 0
        while not stop.is_set():
            if suspension is not None:
                if suspension.requested:
                    if not suspension.unloaded:
                        with state_lock:
                            state["review_phase"] = "suspending"
                        # The sole review thread has completed its current call.
                        # A close failure cannot acknowledge a released model.
                        local_vlm.close()
                        review_mailbox.discard_pending_for_suspend()
                        if subject_mailbox is not None:
                            subject_mailbox.discard_pending_for_suspend()
                        if confirmation is not None:
                            confirmation.gap(reason="suspension")
                        with state_lock:
                            if people is not None:
                                people.clear_roles()
                            for camera in camera_ids:
                                review_attempts[camera] = None
                                subject_attempts[camera] = None
                                subject_completed[camera] = None
                                state["cameras"][camera]["subject_claims"] = None
                            suspension.acknowledge_unloaded()
                    with state_lock:
                        state["review_phase"] = "suspended"
                    stop.wait(0.2)
                    continue
                if suspension.unloaded:
                    # Resume never imports work/claims retained before the gap.
                    review_mailbox.discard_pending_for_suspend()
                    if subject_mailbox is not None:
                        subject_mailbox.discard_pending_for_suspend()
                    if confirmation is not None:
                        confirmation.gap(reason="resume")
                    with state_lock:
                        if not suspension.release():
                            continue
                        state["review_phase"] = "waiting"
            storage = _review_storage_summary(root, max_storage_bytes, max_storage_entries)
            with state_lock:
                state["review_storage"] = storage
            if not storage["available"]:
                with state_lock:
                    state["review_phase"] = "storage_deferred"
                stop.wait(1)
                continue
            if not review_admission():
                with state_lock:
                    state["review_phase"] = "memory_deferred"
                stop.wait(1)
                continue
            with state_lock:
                state["review_phase"] = "waiting"
            task = review_task
            shadow_request = (
                confirmation.take(time.monotonic(), datetime.now(UTC))
                if confirmation is not None and regular_since_shadow >= 4
                else None
            )
            if shadow_request is not None:
                task = "confirmation_shadow"
                selected = ReviewCandidate(
                    shadow_request.observation,
                    "confirmation_shadow_diagnostic",
                    1,
                    shadow_request.observation.completed_monotonic,
                    shadow_request.comparison["detector_view"],
                )
            elif subject_mailbox is not None:
                task, selected = task_cycle.take(review_mailbox, subject_mailbox)
            else:
                selected = review_mailbox.take_live()
            if selected is None and confirmation is not None:
                shadow_request = confirmation.take(time.monotonic(), datetime.now(UTC))
                if shadow_request is not None:
                    task = "confirmation_shadow"
                    selected = ReviewCandidate(
                        shadow_request.observation,
                        "confirmation_shadow_diagnostic",
                        1,
                        shadow_request.observation.completed_monotonic,
                        shadow_request.comparison["detector_view"],
                    )
            if selected is None:
                with review_mailbox.condition:
                    review_mailbox.condition.wait(timeout=0.2)
                continue
            observation = selected.observation
            frame = observation.frame
            if suspension is not None and (
                suspension.blocked or frame.received_monotonic < suspension.resumed_after
            ):
                with state_lock:
                    state["review_pre_resume_frame_drops"] += 1
                if shadow_request is not None:
                    confirmation.complete(shadow_request, status="pre_resume_discarded", summary={})
                continue
            if task == "subjects" and people is not None:
                with state_lock:
                    if not people.review_due(
                        frame.camera_id, now=time.monotonic(), now_utc=datetime.now(UTC)
                    ):
                        # Frames queued during an in-flight call may no longer
                        # need review after it completes. Recheck before reading
                        # media or consuming a regular Bonsai/shadow quota slot.
                        state["subject_not_due_discards"] += 1
                        continue
            if task == "confirmation_shadow":
                regular_since_shadow = 0
                attempts = dict.fromkeys(camera_ids)
            else:
                regular_since_shadow = min(4, regular_since_shadow + 1)
                attempts = subject_attempts if task == "subjects" else review_attempts
            previous_diagnostics = _local_vlm_diagnostics_snapshot(local_vlm)
            identifier = uuid.uuid4().hex
            filename = (
                frame.received_at.strftime("%Y%m%dT%H%M%SZ")
                + "_"
                + frame.camera_id
                + "_"
                + identifier
                + ".png"
            )
            record: dict[str, Any] = {
                "event_id": identifier,
                "camera_id": frame.camera_id,
                "frame_sequence": frame.sequence,
                "raw_bgr_sha256": frame.sha256,
                "frame_received_at": frame.received_at.isoformat(),
                "trigger": selected.reason,
                "evidence_role": selected.evidence_role,
                "detector_view": selected.evidence_view,
                "image_orientation": "original",
                "frame_generation": frame.generation,
                "review_detection_count": len(observation.detections),
                "review_boxes": [
                    {"confidence": d.confidence, "bbox_xyxy": list(d.bbox_xyxy)}
                    for d in observation.detections
                ],
                "queue_wait_seconds": round(
                    max(0.0, time.monotonic() - selected.queued_monotonic), 3
                ),
                "evidence_age_at_selection_seconds": round(
                    max(0.0, time.monotonic() - frame.received_monotonic), 3
                ),
                "selected_at": datetime.now(UTC).isoformat(),
                "remote_used": False,
                "time_semantics": "decoder_receipt_not_camera_exposure",
                "status": "running",
                "review_task": task,
                "review_png_profile": review_png_profile,
                "local_review_mode": state["local_review_mode"],
                "local_vlm_diagnostics": None,
            }
            if shadow_request is not None:
                record["confirmation_comparison"] = shadow_request.comparison
                record["production_writeback"] = False
                record["shadow_model_summary"] = None
            with state_lock:
                state["active_review_camera"] = frame.camera_id
                state["active_review_task"] = task
                state["review_phase"] = "analyzing"
                # New attempts own a new claim slot. The display projection may
                # separately retain a still-valid completed frame while analyzing.
                if task == "subjects":
                    state["cameras"][frame.camera_id]["subject_claims"] = None
                attempts[frame.camera_id] = {
                    "status": "analyzing",
                    "frame_received_at": frame.received_at.isoformat(),
                    "received_monotonic": frame.received_monotonic,
                    "frame_generation": frame.generation,
                    "completed_at": None,
                }
            try:
                path = root / "frames" / filename
                encoding_started = time.monotonic()
                payload = (
                    encode_bgr_png(frame)
                    if review_png_profile == "compact"
                    else encode_bgr_png(frame, profile=review_png_profile)
                )
                record["evidence_encoding_ms"] = round(
                    (time.monotonic() - encoding_started) * 1000, 3
                )
                descriptor = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
                )
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                candidate = CandidateEvent(
                    identifier, frame.camera_id, 0.0, 0.0, observation.detections
                )
                review_started = time.monotonic()
                if task in ("subjects", "confirmation_shadow"):
                    use_state = task == "subjects" and people is not None
                    probe = local_vlm.probe_subject_state if use_state else local_vlm.probe_subjects
                    validator = (
                        subject_state_probe.validate_bound_result
                        if use_state
                        else validate_bound_result
                    )
                    result = validator(
                        probe(VLMRequest(candidate, EventMedia("", (str(path),)))),
                        frame_id=path.name,
                        input_sha256=hashlib.sha256(payload).hexdigest(),
                    )
                    record.update(
                        status="analyzed",
                        subject_result=result,
                        review_duration_ms=round((time.monotonic() - review_started) * 1000, 3),
                        local_vlm_diagnostics=_updated_review_diagnostics(
                            local_vlm, previous_diagnostics
                        ),
                        evidence_age_at_completion_seconds=round(
                            max(0.0, time.monotonic() - frame.received_monotonic), 3
                        ),
                    )
                    completed = datetime.now(UTC).isoformat()
                    if task == "confirmation_shadow":
                        record["shadow_model_summary"] = summarize_shadow_probe(
                            shadow_request,
                            result,
                            frame_id=path.name,
                            png_sha256=hashlib.sha256(payload).hexdigest(),
                        )
                        # No database write, latest-review file, live age projection or alert.
                        continue
                    if confirmation is not None and not use_state:
                        try:
                            record["confirmation_subject_agreement"] = (
                                confirmation.record_subject_probe(
                                    observation,
                                    result,
                                    frame_id=path.name,
                                    png_sha256=hashlib.sha256(payload).hexdigest(),
                                )
                            )
                        except Exception:
                            # Valid regular subject output survives failed optional telemetry.
                            confirmation.gap(reason="telemetry_failed")
                            record["confirmation_subject_agreement"] = {
                                "status": "telemetry_failed"
                            }
                    if people is not None:
                        with state_lock:
                            if people_review_trace:
                                record["people_trace"] = people.review_trace(
                                    observation, view=selected.evidence_view
                                )
                            record["people_binding"] = people.apply_review(
                                observation,
                                result["claims"],
                                evidence_ref=path.name,
                                now=time.monotonic(),
                                now_utc=datetime.now(UTC),
                                view=selected.evidence_view,
                            )
                    _write_private(
                        root / "reviews" / (frame.camera_id + "_subjects.json"),
                        {**record, "completed_at": completed, "subject_result": result},
                    )
                    # Status contains bounded enums/coordinates only, never model prose.
                    with state_lock:
                        state["subject_probe_successes"] += 1
                        state["cameras"][frame.camera_id].update(
                            subject_claims=[
                                {
                                    "age_claim": person["age_group"],
                                    "head_center": person["head_center"],
                                }
                                for person in result["claims"]["people"]
                            ],
                            last_subject_frame_received_at=frame.received_at.isoformat(),
                            last_subject_completed_at=completed,
                            last_subject_error=None,
                            last_subject_duration_ms=record["review_duration_ms"],
                        )
                    continue
                processed = process_prepared_event(
                    candidate=candidate,
                    occurred_at=frame.received_at,
                    source_kind="live_detection_review",
                    source_ref="live_camera:" + frame.camera_id,
                    media=EventMedia("", (str(path),)),
                    local_vlm=local_vlm,
                    database=database,
                    policy=ReviewPolicy(),
                    secondary_vlm=None,
                    remote_vlm=None,
                    remote_uploads_allowed=False,
                )
                stored = database.get_event(identifier)
                if stored is None or stored.remote_used:
                    raise RuntimeError("invalid local review writeback")
                record.update(
                    status="analyzed",
                    review_tier=processed.review_tier.value,
                    model_event_type=stored.result.event_type.value,
                    review_duration_ms=round((time.monotonic() - review_started) * 1000, 3),
                    local_vlm_diagnostics=_updated_review_diagnostics(
                        local_vlm, previous_diagnostics
                    ),
                )
                completed = datetime.now(UTC).isoformat()
                record["evidence_age_at_completion_seconds"] = round(
                    max(0.0, time.monotonic() - frame.received_monotonic), 3
                )
                _write_private(
                    root / "reviews" / (frame.camera_id + ".json"),
                    {
                        **record,
                        "completed_at": completed,
                        "result": stored.result.to_dict(),
                        "decision": stored.decision.to_dict(),
                    },
                )
                with state_lock:
                    state["review_successes"] += 1
                    state["cameras"][frame.camera_id].update(
                        last_review_frame_received_at=frame.received_at.isoformat(),
                        last_review_completed_at=completed,
                        last_review_event_type=stored.result.event_type.value,
                        last_review_error=None,
                        last_review_duration_ms=record["review_duration_ms"],
                    )
            except Exception as error:
                record.update(
                    status="analysis_failed",
                    error_code="confirmation_shadow_review_failed"
                    if task == "confirmation_shadow"
                    else getattr(error, "code", "local_review_failed"),
                )
                if task != "confirmation_shadow":
                    with state_lock:
                        if people is not None and task == "subjects":
                            people.clear_roles(frame.camera_id)
                        state[
                            "subject_probe_failures" if task == "subjects" else "review_failures"
                        ] += 1
                        state["cameras"][frame.camera_id][
                            "last_subject_error" if task == "subjects" else "last_review_error"
                        ] = record["error_code"]
            finally:
                record["local_vlm_diagnostics"] = _updated_review_diagnostics(
                    local_vlm, previous_diagnostics
                )
                record["completed_at"] = datetime.now(UTC).isoformat()
                _write_private(root / "records" / (identifier + ".json"), record)
                if shadow_request is not None:
                    confirmation.complete(
                        shadow_request,
                        status=record["status"],
                        summary=record.get("shadow_model_summary")
                        or {"error_code": "confirmation_shadow_review_failed"},
                    )
                with state_lock:
                    attempts[frame.camera_id].update(
                        status=record["status"], completed_at=record["completed_at"]
                    )
                    if task == "subjects":
                        subject_completed[frame.camera_id] = (
                            {
                                "attempt": dict(attempts[frame.camera_id]),
                                "claims": state["cameras"][frame.camera_id]["subject_claims"],
                            }
                            if record["status"] == "analyzed"
                            else None
                        )
                    state["active_review_camera"] = None
                    state["active_review_task"] = None
                    state["review_phase"] = "waiting"

    def review_worker() -> None:
        try:
            review_loop()
        except Exception:
            # Never print raw model/source exceptions containing private paths or media text.
            with state_lock:
                state["review_phase"] = "worker_failed"
                state["fatal_error_code"] = "review_worker_failed"
            review_worker_failed.set()

    def publish() -> None:
        sampling = frame_mailbox.snapshot(time.monotonic())
        media = {source.source.camera_id: source.snapshot() for source in sources}
        with state_lock:
            elapsed = max(0.001, time.monotonic() - began)
            metric_now = time.monotonic()
            published_at = datetime.now(UTC)
            window = min(10.0, max(0.001, metric_now - (detecting_began or metric_now)))
            for history in recent_frames.values():
                while history and metric_now - history[0] >= 10:
                    history.popleft()
            snapshot = {
                **state,
                "heartbeat_at": published_at.isoformat(),
                "heartbeat_monotonic_seconds": metric_now,
                "elapsed_seconds": round(elapsed, 3),
                "review_queue": review_mailbox.snapshot(),
                "subject_queue": subject_mailbox.snapshot()
                if subject_mailbox is not None
                else None,
                "review_control": {"enabled": False}
                if suspension is None
                else suspension.snapshot(),
                "confirmation_shadow": {"enabled": False}
                if confirmation is None
                else confirmation.summary(),
            }
            review_suspended = (
                state["phase"] == "detecting" and suspension is not None and suspension.blocked
            )
            if review_suspended:
                snapshot["review_phase"] = "suspended" if suspension.unloaded else "suspending"
            review_monitor_phase = "local_review_suspended" if review_suspended else state["phase"]
            snapshot["cameras"] = {
                camera: {
                    **data,
                    **sampling[camera],
                    **media[camera],
                    "review_evidence": _review_evidence_snapshot(
                        subject_attempts[camera]
                        if review_task == "subjects"
                        else review_attempts[camera],
                        now=metric_now,
                        now_utc=published_at,
                        source_phase=media[camera].get("phase"),
                        source_generation=media[camera].get("generation"),
                        monitor_phase=review_monitor_phase,
                    ),
                    **_subject_projection(
                        subject_attempts[camera],
                        subject_completed[camera],
                        data["subject_claims"],
                        now=metric_now,
                        now_utc=published_at,
                        source_phase=media[camera].get("phase"),
                        source_generation=media[camera].get("generation"),
                        monitor_phase="review_worker_failed"
                        if state.get("review_phase") == "worker_failed"
                        else review_monitor_phase,
                    ),
                    "average_inference_fps": round(data["inferred_frames"] / elapsed, 3),
                    "recent_inference_fps": round(len(recent_frames[camera]) / window, 3),
                    "recent_fps_window_seconds": round(window, 3),
                }
                for camera, data in state["cameras"].items()
            }
            for data in snapshot["cameras"].values():
                # Never publish old-generation/stopped/stale claims as current results.
                if data["subject_evidence"]["status"] != "recent_completed_review":
                    data["subject_claims"] = None
            if people is not None:
                projection = people.snapshot(
                    now=metric_now,
                    now_utc=published_at,
                    sources=media,
                    monitor_phase=state["phase"],
                    review_available=not review_suspended
                    and state.get("review_phase") != "worker_failed",
                )
                snapshot["baby_supervision"] = projection
                current = projection["current"]
                snapshot["baby_location"] = (
                    current["room"]["room_id"]
                    if current and current["room"]
                    else current["camera_id"]
                    if current
                    else "unknown"
                )
                snapshot["baby_state"] = current["visible_posture"] if current else "unknown"
                for camera, data in snapshot["cameras"].items():
                    data["tracked_people"] = projection["cameras"][camera]
        if confirmation is not None:
            try:
                _write_private(
                    root / "confirmation-shadow.json",
                    {
                        "worker_pid": os.getpid(),
                        "session_started_at": session_started.isoformat(),
                        "heartbeat_at": published_at.isoformat(),
                        "remote_enabled": False,
                        "production_writeback": False,
                        **confirmation.snapshot(),
                    },
                )
            except Exception:
                confirmation.gap(reason="telemetry_failed")
                snapshot["confirmation_shadow"] = confirmation.summary()
        if transition_capture is not None:
            snapshot["transition_capture"] = transition_capture.snapshot()
        _write_private(root / "status.json", snapshot)

    review_thread = threading.Thread(target=review_worker, name="local-review", daemon=True)
    started_sources: list[FFmpegLiveSource] = []
    try:
        publish()
        detector.start()
        if transition_capture is not None:
            transition_capture.start()
        detecting_began = time.monotonic()
        with state_lock:
            state["detector"] = detector.identity
            state["phase"] = "detecting"
        for source in sources:
            source.start()
            started_sources.append(source)
        review_thread.start()
        last_publish = 0.0
        while not stop.is_set():
            now = time.monotonic()
            if review_worker_failed.is_set():
                raise RuntimeError("local review worker failed")
            if max_runtime_seconds is not None and now - began >= max_runtime_seconds:
                break
            frame_mailbox.wait_for_batch()
            frames = frame_mailbox.take_live()
            if frames:
                try:
                    inference_started = time.monotonic()
                    predictions = detector.predict(frames)
                    completed = time.monotonic()
                    if len(predictions) != len(frames):
                        raise RuntimeError("live inference batch mismatch")
                    with state_lock:
                        state["inference_batches"] += 1
                        state["inference_batch_sizes"][str(len(frames))] += 1
                        state["last_inference_batch_ms"] = round(
                            (completed - inference_started) * 1000, 3
                        )
                        state["last_inference_batch_size"] = len(frames)
                    for frame, detections in zip(frames, predictions, strict=True):
                        observation = LiveObservation(frame, detections, completed)
                        latency = (completed - frame.received_monotonic) * 1000
                        with state_lock:
                            if people is not None:
                                accepted = people.observe(observation)
                                if transition_capture is not None and accepted:
                                    if suspension is not None and suspension.blocked:
                                        transition_capture.gap()
                                    else:
                                        transition_capture.observe(
                                            observation,
                                            people.review_trace(observation),
                                            view="original",
                                        )
                            camera = state["cameras"][frame.camera_id]
                            camera["inferred_frames"] += 1
                            recent_frames[frame.camera_id].append(completed)
                            camera.update(
                                last_detection_received_at=frame.received_at.isoformat(),
                                last_person_count=len(detections),
                                last_boxes=[list(item.bbox_xyxy) for item in detections],
                                last_receipt_to_detection_ms=round(latency, 3),
                                max_receipt_to_detection_ms=max(
                                    camera["max_receipt_to_detection_ms"], round(latency, 3)
                                ),
                            )
                        candidate = triggers.observe(observation)
                        # Keep shadow results completely outside trigger/presence semantics.
                        shadow_status_reader = getattr(detector, "shadow_status_for", None)
                        shadow_hint_reader = getattr(detector, "shadow_hints_for", None)
                        shadow_status = (
                            shadow_status_reader(frame)
                            if shadow_status_reader is not None
                            else None
                        )
                        shadow_hints = (
                            shadow_hint_reader(frame) if shadow_hint_reader is not None else None
                        )
                        with state_lock:
                            _record_shadow_observation(
                                state["cameras"][frame.camera_id]["nonempty_shadow"],
                                observation,
                                shadow_status,
                                shadow_hints,
                            )
                        hint_reader = getattr(detector, "review_hints_for", None)
                        hints = hint_reader(frame) if hint_reader is not None else None
                        actual_alternate_requested = None
                        fallback_used = False
                        if hints is not None:
                            fallback_reader = getattr(detector, "review_fallback_for", None)
                            fallback_used = (
                                fallback_reader is not None and fallback_reader(frame) is True
                            )
                            with state_lock:
                                state["cameras"][frame.camera_id]["alternate_view_checks"] += 1
                                state["cameras"][frame.camera_id][
                                    "alternate_view_below_gate_frames"
                                ] += bool(hints) and not alternate_triggers.frame_eligible(hints)
                                state["cameras"][frame.camera_id][
                                    "alternate_scale_fallback_checks"
                                ] += fallback_used
                                state["cameras"][frame.camera_id][
                                    "alternate_scale_fallback_positive_frames"
                                ] += fallback_used and bool(hints)
                                state["cameras"][frame.camera_id][
                                    "alternate_view_positive_frames"
                                ] += bool(hints)
                            alternate = alternate_triggers.observe(
                                LiveObservation(frame, hints, completed)
                            )
                            actual_alternate_requested = alternate is not None
                            if alternate is not None:
                                if fallback_used:
                                    alternate = replace(
                                        alternate, evidence_view="rotation_180_scale_fallback"
                                    )
                                with state_lock:
                                    state["cameras"][frame.camera_id][
                                        "alternate_view_review_triggers"
                                    ] += 1
                                if candidate is None or candidate.priority <= alternate.priority:
                                    candidate = alternate
                        if confirmation is not None and (
                            suspension is None or not suspension.blocked
                        ):
                            try:
                                if hints is not None:
                                    confirmation.observe(
                                        LiveObservation(frame, hints, completed),
                                        detector_view="rotation_180_scale_fallback"
                                        if fallback_used
                                        else "rotation_180",
                                        actual_alternate_requested=actual_alternate_requested,
                                    )
                                elif shadow_status == "checked":
                                    confirmation.observe(
                                        LiveObservation(frame, shadow_hints, completed),
                                        detector_view="rotation_180",
                                        actual_alternate_requested=None,
                                    )
                            except Exception:
                                # An optional experiment cannot take down native monitoring.
                                confirmation.gap(reason="telemetry_failed")
                        if suspension is not None and suspension.blocked:
                            with state_lock:
                                state["review_suspended_frames"] += 1
                            continue
                        with state_lock:
                            if people is not None and hints is not None and not detections:
                                people.observe_auxiliary(
                                    LiveObservation(frame, hints, completed),
                                    view="rotation_180_scale_fallback"
                                    if fallback_used
                                    else "rotation_180",
                                )
                                if transition_capture is not None:
                                    auxiliary_observation = LiveObservation(frame, hints, completed)
                                    auxiliary_view = (
                                        "rotation_180_scale_fallback"
                                        if fallback_used
                                        else "rotation_180"
                                    )
                                    transition_capture.observe(
                                        auxiliary_observation,
                                        people.review_trace(
                                            auxiliary_observation, view=auxiliary_view
                                        ),
                                        view=auxiliary_view,
                                    )
                            subject_observation, subject_view = (
                                people.review_observation(
                                    observation, now=completed, now_utc=datetime.now(UTC)
                                )
                                if people is not None
                                else (observation, "original")
                            )
                            subjects_due = people is None or people.review_due(
                                frame.camera_id, now=completed, now_utc=datetime.now(UTC)
                            )
                        if review_task != "subjects" or subjects_due:
                            _enqueue_review(
                                review_mailbox,
                                subject_observation if review_task == "subjects" else observation,
                                candidate,
                                review_task=review_task,
                                evidence_view=subject_view
                                if review_task == "subjects"
                                else "original",
                            )
                        if subject_mailbox is not None and subjects_due:
                            _enqueue_review(
                                subject_mailbox,
                                subject_observation,
                                None,
                                review_task="subjects",
                                evidence_view=subject_view,
                            )
                except Exception:
                    with state_lock:
                        state["detection_failures"] += 1
                        state["phase"] = "detector_failed"
                    raise
            else:
                with frame_mailbox.condition:
                    frame_mailbox.condition.wait(timeout=0.02)
            if now - last_publish >= 1:
                publish()
                last_publish = now
    finally:
        stop.set()
        cleanup_errors: list[str] = []

        def cleanup(action, code: str) -> None:
            try:
                action()
            except Exception:
                # Raw source/model exceptions may contain private paths or URLs.
                if code not in cleanup_errors:
                    cleanup_errors.append(code)

        for source in started_sources:
            cleanup(source.close, "source_close_failed")
        cleanup(detector.close, "detector_close_failed")
        if transition_capture is not None:
            cleanup(transition_capture.close, "transition_capture_close_failed")

        def join_review() -> None:
            if review_thread.is_alive():
                review_thread.join(timeout=220)
                if review_thread.is_alive():
                    cleanup_errors.append("review_join_timeout")

        cleanup(join_review, "review_join_failed")

        def close_review() -> None:
            close_vlm = getattr(local_vlm, "close", None)
            if close_vlm is not None:
                close_vlm()

        cleanup(close_review, "vlm_close_failed")
        try:
            with state_lock:
                state["phase"] = "cleanup_failed" if cleanup_errors else "stopped"
                state["cleanup_error_codes"] = list(cleanup_errors)
            cleanup(publish, "status_publish_failed")
        finally:
            for signum, handler in old_handlers.items():
                cleanup(
                    lambda signum=signum, handler=handler: signal.signal(signum, handler),
                    "signal_restore_failed",
                )
            cleanup(lambda: os.close(lock_descriptor), "worker_lock_close_failed")
        if cleanup_errors:
            # Never return success merely because the primary loop completed.
            raise RuntimeError("live monitor cleanup failed: " + ",".join(cleanup_errors)) from None
