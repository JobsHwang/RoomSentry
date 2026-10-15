"""Session-local anonymous people and explicitly labelled baby inferences.

Reuses the time-aware motion tracker. A role may follow only an uninterrupted,
unique geometric association, never a reused ID, a crossing, or another camera.
History contains bounded metadata only, not images. All times are decoder receipt
times; they do not prove camera exposure time or sleep.
"""

from __future__ import annotations

import copy
import math
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from baby_monitor.live_detection import LiveObservation, receipt_age_seconds
from baby_monitor.live_people_association import describe_association
from baby_monitor.motion_tracker import (
    MotionObjectTracker,
    MotionTrackerConfig,
    MotionTrackerError,
    _bbox_iou,
)
from baby_monitor.object_tracker import DetectionFrame, FrameDetection


@dataclass(frozen=True)
class PeopleSettings:
    track_gap_seconds: float = 2.0
    role_ttl_seconds: float = 30.0
    confirmation_spacing_seconds: float = 1.0
    frame_history_per_camera: int = 1024

    def __post_init__(self):
        for value, lower, upper in (
            (self.track_gap_seconds, 0.1, 10),
            (self.role_ttl_seconds, 1, 60),
            (self.confirmation_spacing_seconds, 0.1, 10),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError("invalid people timing")
        if (
            type(self.frame_history_per_camera) is not int
            or not 16 <= self.frame_history_per_camera <= 2048
        ):
            raise ValueError("invalid people history limit")


def _age(mono, stamp, now, now_utc):
    try:
        wall = (now_utc - stamp).total_seconds()
        elapsed = now - mono
        if not all(math.isfinite(n) and n >= 0 for n in (wall, elapsed)):
            return None
        return max(wall, elapsed)
    except (TypeError, ValueError, OverflowError):
        return None


class LivePeople:
    """Called under the monitor's state lock by detection, review and publication."""

    def __init__(
        self,
        camera_ids,
        *,
        room_registry=None,
        settings=None,
        auxiliary=False,
        association_diagnostics=False,
        utc_geometry_continuity=False,
    ):
        self.settings = settings or PeopleSettings()
        if type(association_diagnostics) is not bool:
            raise ValueError("invalid association diagnostics flag")
        self.association_diagnostics = association_diagnostics
        if type(utc_geometry_continuity) is not bool:
            raise ValueError("invalid UTC geometry continuity flag")
        self.utc_geometry_continuity = utc_geometry_continuity
        if not 1 <= len(camera_ids) <= 4 or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("invalid people cameras")
        self.session = uuid.uuid4().hex[:12]
        self.rooms = {
            camera: None
            if room_registry is None
            else room_registry.room_for_stream(camera).to_dict()
            for camera in camera_ids
        }
        self.cameras = {
            camera: {
                "previous": None,
                "tracker": None,
                "tracks": {},
                "history": OrderedDict(),
                "segment": 0,
                "last_seen": None,
                "untracked": [],
                "review_at": None,
                "epochs": {},
                "epoch_causes": {},
                "processed_frames": 0,
                "history_evicted_frames": 0,
                "history_resets": 0,
                "history_reset_discarded_frames": 0,
                "history_reset_reasons": {},
                "latest_history_reset": None,
                "latest_reset_checkpoint": None,
                "geometry_resets": 0,
                "utc_geometry_retained": 0,
                "geometry_candidate_failures": 0,
                "latest_geometry_boundary": None,
            }
            for camera in camera_ids
        }
        self.counters = dict.fromkeys(
            (
                "frames",
                "resets",
                "ambiguous_associations",
                "bound_reviews",
                "unbound_people",
                "rejected_reviews",
            ),
            0,
        )
        self.auxiliary = (
            LivePeople(
                camera_ids,
                room_registry=room_registry,
                settings=self.settings,
                association_diagnostics=association_diagnostics,
                utc_geometry_continuity=utc_geometry_continuity,
            )
            if auxiliary
            else None
        )

    @staticmethod
    def _key(frame):
        return frame.generation, frame.sequence

    def clear_roles(self, camera_id=None, *, reason="explicit_role_clear"):
        if reason not in ("explicit_role_clear", "native_lane_took_over"):
            raise ValueError("invalid role clear reason")
        if self.auxiliary is not None:
            self.auxiliary.clear_roles(camera_id, reason=reason)
        selected = self.cameras.values() if camera_id is None else (self.cameras[camera_id],)
        for camera in selected:
            camera["epochs"] = {k: v + 1 for k, v in camera["epochs"].items()}
            for causes in camera["epoch_causes"].values():
                causes[reason] = causes.get(reason, 0) + 1
            for track in camera["tracks"].values():
                track["epoch"] = camera["epochs"].get(track["track_id"], track["epoch"] + 1)
                track["role"] = None
            camera["untracked"] = []

    def review_due(self, camera_id, *, now, now_utc):
        """Refresh roles on change; retain periodic full-image zero-box patrols."""
        camera = self.cameras[camera_id]
        previous = camera["previous"]
        if not camera["tracks"] and previous is not None:
            alternate = self._eligible_auxiliary(
                camera_id, generation=previous.generation, now=now, now_utc=now_utc
            )
            if alternate is not None:
                return self.auxiliary.review_due(camera_id, now=now, now_utc=now_utc)
        reviewed = camera["review_at"]
        if reviewed is None:
            return True
        tracks = camera["tracks"].values()
        unsettled = any(
            t["role"] is None
            or t["role"]["support_count"] < 2
            or t["role"]["age_claim"] == "unknown"
            for t in tracks
        ) or bool(camera["untracked"])
        interval = 1.0 if unsettled else 10.0 if camera["tracks"] else 15.0
        if now - reviewed >= interval or now < reviewed:
            return True
        # A forward UTC jump cannot postpone review just because mono is young.
        for track in tracks:
            role = track["role"]
            if role is not None:
                age = _age(role["mono"], role["stamp"], now, now_utc)
                if age is None or age >= interval:
                    return True
        return False

    def observe(self, observation: LiveObservation, *, view="original") -> bool:
        """Consume a new input; rejected ordering must have no lane side effects."""
        frame = observation.frame
        if view not in ("original", "rotation_180", "rotation_180_scale_fallback"):
            raise ValueError("invalid people detection view")
        if any(item.label != "person" for item in observation.detections):
            raise ValueError("live people requires person detections")
        camera = self.cameras[frame.camera_id]
        previous = camera["previous"]
        if previous is not None:
            if frame.generation < previous.generation or (
                frame.generation == previous.generation and frame.sequence <= previous.sequence
            ):
                return False
            elapsed = frame.received_monotonic - previous.received_monotonic
            wall = (frame.received_at - previous.received_at).total_seconds()
            reset = (
                frame.generation != previous.generation
                or min(elapsed, wall) <= 0
                or max(elapsed, wall) > self.settings.track_gap_seconds
            )
        else:
            reset = True
        reset_checkpoint = None
        # Receipt UTC is an evidence boundary, but need not be a motion-clock
        # boundary. Never preserve roles, review history or support votes here.
        old_input = None if previous is None else camera["history"].get(self._key(previous))
        keep_geometry = bool(
            self.utc_geometry_continuity
            and reset
            and previous is not None
            and camera["tracker"] is not None
            and frame.generation == previous.generation
            and math.isfinite(elapsed)
            and 0 < elapsed <= self.settings.track_gap_seconds
            and math.isfinite(wall)
            and (wall <= 0 or wall > self.settings.track_gap_seconds)
            and old_input is not None
            and (old_input["view"] == "original") == (view == "original")
        )
        if self.auxiliary is not None and observation.detections and not camera["tracks"]:
            # Only an admitted new input can take over the native lane. Duplicate
            # or older input must not revoke a more recent auxiliary role.
            self.auxiliary.clear_roles(frame.camera_id, reason="native_lane_took_over")
        if reset:
            if self.association_diagnostics:
                # This is the actual admission boundary, not a guess made from
                # sparse exported intervals after the old history is gone.
                reasons = []
                if previous is None:
                    reasons.append("initial_input")
                else:
                    if frame.generation != previous.generation:
                        reasons.append("generation_changed")
                    if elapsed <= 0:
                        reasons.append("non_increasing_monotonic")
                    if wall <= 0:
                        reasons.append("non_increasing_utc")
                    if elapsed > self.settings.track_gap_seconds:
                        reasons.append("monotonic_gap")
                    if wall > self.settings.track_gap_seconds:
                        reasons.append("utc_gap")
                for reason in reasons:
                    camera["history_reset_reasons"][reason] = (
                        camera["history_reset_reasons"].get(reason, 0) + 1
                    )
                camera["latest_history_reset"] = {
                    "version": 1,
                    "reasons": reasons,
                    "previous_generation": None if previous is None else previous.generation,
                    "generation": frame.generation,
                    "previous_sequence": None if previous is None else previous.sequence,
                    "sequence": frame.sequence,
                    "previous_received_at": None
                    if previous is None
                    else previous.received_at.isoformat(),
                    "received_at": frame.received_at.isoformat(),
                    "previous_received_monotonic_seconds": None
                    if previous is None
                    else previous.received_monotonic,
                    "received_monotonic_seconds": frame.received_monotonic,
                    "elapsed_receipt_gap_seconds": None
                    if previous is None or not math.isfinite(elapsed)
                    else elapsed,
                    "utc_receipt_gap_seconds": None if previous is None else wall,
                    "track_gap_seconds": self.settings.track_gap_seconds,
                    "discarded_history_frames": len(camera["history"]),
                    "discarded_visible_tracks": len(camera["tracks"]),
                    "diagnostic_only": True,
                    "camera_exposure_clock_verified": False,
                }
                camera["latest_reset_checkpoint"] = None
                if not keep_geometry and (
                    reasons == ["utc_gap"] or reasons == ["non_increasing_utc"]
                ):
                    # Capture the actual discarded matching state, not a cold
                    # guess made from a later review. One bounded latest event,
                    # no role/identity evidence and no image objects retained.
                    old_tracker = camera["tracker"]
                    before = old_tracker.state
                    old_input = camera["history"][self._key(previous)]
                    previous_input = DetectionFrame(
                        old_input["association_audit"]["input_frame_id"],
                        before.last_frame_at,
                        tuple(
                            FrameDetection(f"d_{i:03}", "person", confidence, track["bbox"])
                            for i, (track, confidence) in enumerate(
                                zip(old_input["tracks"], old_input["confidences"], strict=True)
                            )
                        ),
                    )
                    reset_checkpoint = {
                        "version": 1,
                        "camera_id": frame.camera_id,
                        "detection_view": view,
                        "previous_detection_view": old_input["view"],
                        "boundary": copy.deepcopy(camera["latest_history_reset"]),
                        "tracker_config": old_tracker.config.to_dict(),
                        "before_state": replace(
                            before,
                            processed_frames=before.processed_frames[-1:],
                            pending_observations=(),
                        ).to_dict(),
                        "previous_input": previous_input.to_dict(),
                        "previous_raw_sha256": old_input["raw_sha256"],
                        "previous_ordinal": camera["processed_frames"],
                        "previous_assignment_ids": [t["track_id"] for t in old_input["tracks"]],
                        "full_active_track_state": True,
                        "omitted_processed_dedup_frames": max(0, len(before.processed_frames) - 1),
                        "pending_delivery_outbox_included": False,
                        "role_evidence_included": False,
                        "matching_replay_only": True,
                    }
            camera["history_resets"] += 1
            camera["history_reset_discarded_frames"] += len(camera["history"])
            camera["segment"] += 1
            camera["history"].clear()
            camera["tracks"] = {}
            camera["untracked"] = []
            camera["review_at"] = None
            if keep_geometry:
                # The new visible tracks have no prior role. Reused anonymous
                # IDs still get a new epoch below; old async reviews lost their
                # frozen input above and cannot cross this evidence scope.
                camera["epoch_causes"] = {
                    key: {"utc_evidence_scope_reset": 1} for key in camera["epochs"]
                }
            else:
                camera["epochs"] = {}
                camera["epoch_causes"] = {}
            # All exported YOLO boxes seed anonymous tracks. The detector's own
            # threshold is untouched; the tracker never adds boxes or people.
            if not keep_geometry:
                camera["tracker"] = self._empty_tracker(frame.camera_id)
            if self.utc_geometry_continuity:
                camera["geometry_resets"] += not keep_geometry
                camera["latest_geometry_boundary"] = {
                    "generation": frame.generation,
                    "sequence": frame.sequence,
                    "segment": camera["segment"],
                    "action": "pending" if keep_geometry else "reset",
                    "fallback_code": None,
                    "role_scope_invalidated": True,
                }
            self.counters["resets"] += 1
        tracker = camera["tracker"]
        identifier = f"f_{self.session}_{camera['segment']}_{frame.generation}_{frame.sequence}"
        detected = tuple(
            FrameDetection(f"d_{i:03}", "person", item.confidence, item.bbox_xyxy)
            for i, item in enumerate(observation.detections)
        )
        # Association velocity uses actual monotonic intervals, even if UTC steps.
        clock = datetime(2000, 1, 1, tzinfo=UTC) + timedelta(seconds=frame.received_monotonic)
        tracker_input = DetectionFrame(identifier, clock, detected)
        before_state = tracker.state
        try:
            try:
                rows = tracker.process(tracker_input)
            except MotionTrackerError:
                if not keep_geometry:
                    raise
                # Carry can exceed the bounded active-track capacity even when
                # the old empty baseline would admit this input. Fall back to
                # that exact configuration, never expand limits or swallow a
                # second failure. The matcher commits state transactionally.
                camera["geometry_candidate_failures"] += 1
                camera["geometry_resets"] += 1
                camera["latest_geometry_boundary"].update(
                    action="reset", fallback_code="matcher_rejected"
                )
                camera["epochs"] = {}
                camera["epoch_causes"] = {}
                tracker = self._empty_tracker(frame.camera_id, config=tracker.config)
                camera["tracker"] = tracker
                before_state = tracker.state
                rows = tracker.process(tracker_input)
                keep_geometry = False
            association_audit = (
                describe_association(before_state, tracker_input, rows, tracker.config)
                if self.association_diagnostics
                else None
            )
        except Exception:
            if self.utc_geometry_continuity and reset:
                # Nothing from the failed evidence boundary may remain current
                # or be reviewed. Do not publish a retained outcome on failure.
                boundary = camera["latest_geometry_boundary"]
                if boundary["action"] == "pending":
                    camera["geometry_candidate_failures"] += 1
                    camera["geometry_resets"] += 1
                boundary.update(action="failed", fallback_code="processing_failed")
                camera["previous"] = None
                camera["tracker"] = self._empty_tracker(frame.camera_id)
                camera["epochs"] = {}
                camera["epoch_causes"] = {}
            raise
        if keep_geometry:
            camera["utc_geometry_retained"] += 1
            camera["latest_geometry_boundary"]["action"] = "retained"
        if reset_checkpoint is not None:
            reset_checkpoint.update(
                input_frame=tracker_input.to_dict(),
                ordinal=camera["processed_frames"] + 1,
                original_assignment_ids=[row.tracking_id for row in rows],
                original_association_audit=association_audit,
            )
            camera["latest_reset_checkpoint"] = reset_checkpoint
        tracker.state = replace(
            tracker.state,
            pending_observations=(),
            processed_frames=tracker.state.processed_frames[-64:],
        )
        old = camera["tracks"]
        tracks = {}
        old_boxes = [t["bbox"] for t in old.values()]
        boxes = [row.bbox_xyxy for row in rows]
        for index, row in enumerate(rows):
            box = row.bbox_xyxy
            prior = old.get(row.tracking_id)
            overlaps = [p for p in old_boxes if _bbox_iou(p, box) >= 0.30]
            reverse = (
                [] if prior is None else [b for b in boxes if _bbox_iou(prior["bbox"], b) >= 0.30]
            )
            shared = any(i != index and _bbox_iou(b, box) >= 0.30 for i, b in enumerate(boxes))
            unique = (
                prior is not None
                and len(overlaps) == len(reverse) == 1
                and not shared
                and overlaps[0] == prior["bbox"]
                and reverse[0] == box
            )
            epoch = (
                camera["epochs"].get(row.tracking_id, -1) + 1
                if prior is None
                else prior["epoch"] + (not unique)
            )
            causes = camera["epoch_causes"].setdefault(row.tracking_id, {})
            cause = (
                "detection_gap"
                if prior is None and row.tracking_id in camera["epochs"] and not keep_geometry
                else "overlapping_current_boxes"
                if shared
                else "nonunique_geometry"
                if prior is not None and not unique
                else None
            )
            if cause is not None:
                causes[cause] = causes.get(cause, 0) + 1
            camera["epochs"][row.tracking_id] = epoch
            tracks[row.tracking_id] = {
                "track_id": row.tracking_id,
                "bbox": box,
                "epoch": epoch,
                "role": prior["role"] if unique else None,
                "hits": 1 if prior is None else prior["hits"] + 1,
                "ambiguous": shared or (prior is not None and not unique),
            }
            self.counters["ambiguous_associations"] += tracks[row.tracking_id]["ambiguous"]
        camera["tracks"] = tracks
        retained_ids = {t.tracking_id for t in tracker.state.active_tracks} | set(tracks)
        camera["epochs"] = {k: v for k, v in camera["epochs"].items() if k in retained_ids}
        camera["epoch_causes"] = {
            k: v for k, v in camera["epoch_causes"].items() if k in retained_ids
        }
        camera["previous"] = frame
        camera["processed_frames"] += 1
        # Retain no image object: review binding uses its raw digest plus both clocks.
        camera["history"][self._key(frame)] = {
            "raw_sha256": frame.sha256,
            "ordinal": camera["processed_frames"],
            "received_at": frame.received_at.isoformat(),
            "received_monotonic_seconds": frame.received_monotonic,
            "completed_monotonic_seconds": observation.completed_monotonic,
            "view": view,
            "confidences": tuple(row.confidence for row in rows),
            "tracks": tuple(
                {**t, "role": None, "epoch_causes": dict(camera["epoch_causes"][t["track_id"]])}
                for t in tracks.values()
            ),
        }
        if reset_checkpoint is not None:
            # Reuse the normal history digest; checkpoint capture must not hash
            # the old/current pixel buffers an extra time.
            reset_checkpoint["input_raw_sha256"] = camera["history"][self._key(frame)]["raw_sha256"]
        if association_audit is not None:
            camera["history"][self._key(frame)]["association_audit"] = association_audit
            # Immutable metadata only. Serialize a matching checkpoint once per
            # exported interval, rather than duplicating all old tracks per row.
            camera["history"][self._key(frame)]["association_before_state"] = before_state
        while len(camera["history"]) > self.settings.frame_history_per_camera:
            camera["history"].popitem(last=False)
            camera["history_evicted_frames"] += 1
        self.counters["frames"] += 1
        self._remember_baby(frame.camera_id, frame.received_monotonic, frame.received_at)
        return True

    @staticmethod
    def _empty_tracker(camera_id, *, config=None):
        return MotionObjectTracker(
            config=config
            or MotionTrackerConfig(high_confidence_threshold=0.01, low_confidence_threshold=0.0),
            camera_id=camera_id,
            detector_adapter="live_yolo",
        )

    def observe_auxiliary(self, observation, *, view):
        if self.auxiliary is None or view not in ("rotation_180", "rotation_180_scale_fallback"):
            raise ValueError("invalid auxiliary people input")
        if not self.auxiliary.observe(observation, view=view):
            return
        self.auxiliary.cameras[observation.frame.camera_id]["view"] = view
        self.auxiliary.cameras[observation.frame.camera_id]["observation"] = observation

    def review_trace(self, observation, *, view="original", maximum_frames=256):
        """Copy a bounded exact-input interval, without pixels, claims or mutation.

        This is private opt-in diagnostic data, not camera coverage or truth.
        Preserve the submitted row and the newest suffix when truncation is needed.
        Frame sequence gaps may be deliberate sampling, not detector input loss.
        """
        if type(maximum_frames) is not int or not 2 <= maximum_frames <= 256:
            raise ValueError("invalid people trace limit")
        if view not in ("original", "rotation_180", "rotation_180_scale_fallback"):
            raise ValueError("invalid review detection view")
        lane = self if view == "original" else self.auxiliary
        if lane is None:
            raise ValueError("auxiliary people trace unavailable")
        frame = observation.frame
        camera = lane.cameras[frame.camera_id]
        frozen = camera["history"].get(self._key(frame))
        result = {
            "version": 1,
            "semantics": "actual_people_tracker_inputs_not_video_coverage_or_truth",
            "camera_id": frame.camera_id,
            "detection_view": view,
            "review_frame_generation": frame.generation,
            "review_frame_sequence": frame.sequence,
            "review_raw_bgr_sha256": frame.sha256,
            "maximum_exported_frames": maximum_frames,
            "processed_frames_total": camera["processed_frames"],
            "history_evicted_frames_total": camera["history_evicted_frames"],
            "history_resets_total": camera["history_resets"],
            "history_reset_discarded_frames_total": camera["history_reset_discarded_frames"],
            "interval_frames": None,
            "omitted_middle_frames": None,
            "exported_frames": 0,
            "frames": [],
            "status": "review_frame_unavailable"
            if frozen is None
            else "frame_bytes_mismatch"
            if frozen["raw_sha256"] != frame.sha256
            else "input_view_mismatch"
            if frozen["view"] != view
            else "detector_input_mismatch"
            if tuple(
                (t["bbox"], c) for t, c in zip(frozen["tracks"], frozen["confidences"], strict=True)
            )
            != tuple((d.bbox_xyxy, d.confidence) for d in observation.detections)
            else "bound_interval",
            "media_included": False,
            "accuracy_measured": False,
            "paths_disclosed": False,
        }
        if lane.association_diagnostics:
            # At most six cumulative cause counters and one latest event per
            # lane. Missing old protocol fields are unknown, never zero causes.
            result["history_reset_reasons_total"] = dict(camera["history_reset_reasons"])
            result["latest_history_reset"] = copy.deepcopy(camera["latest_history_reset"])
            result["latest_reset_checkpoint"] = copy.deepcopy(camera["latest_reset_checkpoint"])
            result["reset_diagnostics"] = {
                "version": 1,
                "cause_counts_may_overlap": True,
                "latest_event_only": True,
                "complete_reset_event_history": False,
                "production_rules_changed": lane.utc_geometry_continuity,
            }
        if lane.utc_geometry_continuity:
            result["geometry_continuity"] = {
                "version": 1,
                "revision": "utc_only_geometry_role_scope_v1",
                "enabled": True,
                "history_resets_are_evidence_scope_resets": True,
                "geometry_resets_total": camera["geometry_resets"],
                "utc_geometry_retained_total": camera["utc_geometry_retained"],
                "candidate_failures_total": camera["geometry_candidate_failures"],
                "latest_boundary": copy.deepcopy(camera["latest_geometry_boundary"]),
                "complete_boundary_event_history": False,
                "role_ttl_or_dual_clock_admission_changed": False,
                "same_id_proves_role_continuity": False,
            }
        if result["status"] != "bound_interval":
            return result
        interval = [
            (key, value)
            for key, value in camera["history"].items()
            if value["ordinal"] >= frozen["ordinal"]
        ]
        result["interval_frames"] = len(interval)
        if len(interval) > maximum_frames:
            interval = interval[:1] + interval[-(maximum_frames - 1) :]
        result["exported_frames"] = len(interval)
        result["omitted_middle_frames"] = result["interval_frames"] - len(interval)
        if "association_before_state" in frozen:
            before = frozen["association_before_state"]
            compact = replace(
                before,
                processed_frames=before.processed_frames[-1:],
                pending_observations=(),
            )
            result["association_checkpoint"] = {
                "version": 1,
                "before_state": compact.to_dict(),
                "tracker_config": camera["tracker"].config.to_dict(),
                "input_frame_id": frozen["association_audit"]["input_frame_id"],
                "full_active_track_state": True,
                "omitted_processed_dedup_frames": max(0, len(before.processed_frames) - 1),
                "pending_delivery_outbox_included": False,
                "matching_replay_only": True,
            }
        for (generation, sequence), value in interval:
            result["frames"].append(
                {
                    "generation": generation,
                    "sequence": sequence,
                    **{
                        k: value[k]
                        for k in (
                            "ordinal",
                            "raw_sha256",
                            "received_at",
                            "received_monotonic_seconds",
                            "completed_monotonic_seconds",
                            "view",
                        )
                    },
                    "detections": [
                        {
                            "track_id": track["track_id"],
                            "bbox_xyxy": list(track["bbox"]),
                            "confidence": confidence,
                            "epoch": track["epoch"],
                            "epoch_causes": dict(track["epoch_causes"]),
                            "association_ambiguous": track["ambiguous"],
                        }
                        for track, confidence in zip(
                            value["tracks"], value["confidences"], strict=True
                        )
                    ],
                }
            )
            if "association_audit" in value:
                result["frames"][-1]["association_audit"] = copy.deepcopy(
                    value["association_audit"]
                )
        return result

    def review_observation(self, observation, *, now, now_utc):
        """Prefer an exact recently checked supplement only when native is empty."""
        if not observation.detections:
            alternate = self._eligible_auxiliary(
                observation.frame.camera_id,
                generation=observation.frame.generation,
                now=now,
                now_utc=now_utc,
            )
            if alternate is not None:
                return alternate, self.auxiliary.cameras[observation.frame.camera_id]["view"]
        return observation, "original"

    def _eligible_auxiliary(self, camera_id, *, generation, now, now_utc):
        """Selection and cadence must use the same positive, current input lane."""
        if self.auxiliary is None:
            return None
        alternate = self.auxiliary.cameras[camera_id].get("observation")
        if (
            alternate is None
            or not alternate.detections
            or alternate.frame.generation != generation
        ):
            return None
        age = receipt_age_seconds(alternate.frame, now, now_utc)
        return alternate if age is not None and age <= 1 else None

    def _remember_baby(self, camera_id, now, now_utc):
        camera = self.cameras[camera_id]
        candidates = []
        for track in camera["tracks"].values():
            role = track["role"]
            age = None if role is None else _age(role["mono"], role["stamp"], now, now_utc)
            if (
                role
                and role["age_claim"] == "baby"
                and age is not None
                and age <= self.settings.role_ttl_seconds
            ):
                candidates.append(track)
        if len(candidates) != 1 or candidates[0]["role"]["support_count"] < 2:
            return
        track, frame = candidates[0], camera["previous"]
        camera["last_seen"] = {
            "camera_id": camera_id,
            "room": self.rooms[camera_id],
            "track_id": track["track_id"],
            "last_detected_at": frame.received_at.isoformat(),
            "evidence_ref": track["role"]["evidence_ref"],
            "evidence_frame_received_at": track["role"]["stamp"].isoformat(),
        }

    def apply_review(self, observation, claims, *, evidence_ref, now, now_utc, view="original"):
        """Bind already contract-validated claims to the exact submitted frame."""
        if view != "original":
            if self.auxiliary is None or view not in (
                "rotation_180",
                "rotation_180_scale_fallback",
            ):
                raise ValueError("invalid review detection view")
            return self.auxiliary.apply_review(
                observation, claims, evidence_ref=evidence_ref, now=now, now_utc=now_utc
            )
        frame = observation.frame
        camera = self.cameras[frame.camera_id]
        frozen = camera["history"].get(self._key(frame))
        age = receipt_age_seconds(frame, now, now_utc)
        previous = camera["previous"]
        if (
            frozen is None
            or frozen["raw_sha256"] != frame.sha256
            or age is None
            or age > self.settings.role_ttl_seconds
            or previous is None
            or previous.generation != frame.generation
            or (camera["review_at"] is not None and frame.received_monotonic <= camera["review_at"])
        ):
            self.counters["rejected_reviews"] += 1
            return {"status": "stale_or_unbound_review", "bound_people": 0}
        camera["review_at"] = frame.received_monotonic
        people = claims["people"]
        origins = frozen["tracks"]
        matches = []
        for person in people:
            head = person["head_center"]
            options = (
                []
                if head is None
                else [
                    t
                    for t in origins
                    if t["bbox"][0] <= head[0] / 1000 <= t["bbox"][2]
                    and t["bbox"][1] <= head[1] / 1000 <= t["bbox"][3]
                ]
            )
            matches.append(options)
        frequencies = {}
        for options in matches:
            for t in options:
                frequencies[t["track_id"]] = frequencies.get(t["track_id"], 0) + 1
        untracked, bound, diagnostics = [], 0, []
        mapped = set()
        for person, options in zip(people, matches, strict=True):
            origin = options[0] if len(options) == 1 else None
            track = None if origin is None else camera["tracks"].get(origin["track_id"])
            duplicate_head = (
                person["head_center"] is not None
                and sum(
                    p["head_center"] is not None
                    and math.dist(person["head_center"], p["head_center"]) <= 20
                    for p in people
                )
                > 1
            )
            epoch_changes = (
                {
                    k: v - origin["epoch_causes"].get(k, 0)
                    for k, v in camera["epoch_causes"].get(origin["track_id"], {}).items()
                    if v > origin["epoch_causes"].get(k, 0)
                }
                if origin is not None
                else {}
            )
            if (
                origin is None
                or track is None
                or frequencies[origin["track_id"]] != 1
                or origin["ambiguous"]
                or track["ambiguous"]
                or duplicate_head
                or track["epoch"] != origin["epoch"]
            ):
                reason = (
                    "head_not_located"
                    if person["head_center"] is None
                    else "no_detector_box_at_head"
                    if not options
                    else "overlapping_detector_boxes"
                    if len(options) > 1
                    else "multiple_heads_in_one_box"
                    if frequencies[origin["track_id"]] != 1
                    else "duplicate_model_head"
                    if duplicate_head
                    else "review_scope_invalidated"
                    if track is not None
                    and track["epoch"] != origin["epoch"]
                    and any(
                        k in epoch_changes for k in ("explicit_role_clear", "native_lane_took_over")
                    )
                    else "track_continuity_lost"
                    if track is None or track["epoch"] != origin["epoch"]
                    else "ambiguous_tracking"
                )
                untracked.append(
                    {
                        "age_claim": person["age_group"],
                        "head_center": person["head_center"],
                        "binding_reason": reason,
                    }
                )
                diagnostics.append(
                    {
                        "reason": reason,
                        "origin_track_id": None if origin is None else origin["track_id"],
                        "origin_epoch": None if origin is None else origin["epoch"],
                        "current_epoch": None if track is None else track["epoch"],
                        "current_track_present": track is not None,
                        "origin_ambiguous": False if origin is None else origin["ambiguous"],
                        "current_ambiguous": False if track is None else track["ambiguous"],
                        "epoch_changes_since_input": epoch_changes,
                    }
                )
                self.counters["unbound_people"] += 1
                continue
            mapped.add(track["track_id"])
            role = track["role"]
            role_age = None if role is None else _age(role["mono"], role["stamp"], now, now_utc)
            same = (
                role is not None
                and role_age is not None
                and role_age <= self.settings.role_ttl_seconds
                and role["age_claim"] == person["age_group"]
            )
            count = role["support_count"] if same else 0
            last_support = role["support_mono"] if same else frame.received_monotonic
            if (
                count == 0
                or frame.received_monotonic - last_support
                >= self.settings.confirmation_spacing_seconds
            ):
                count = min(2, count + 1)
                last_support = frame.received_monotonic
            posture = person.get("visible_posture", "unknown")
            visible = person.get("visibility", "unknown")
            same_posture = same and role.get("posture_claim") == posture
            posture_count = role.get("posture_support_count", 0) if same_posture else 0
            posture_support = (
                role.get("posture_support_mono", frame.received_monotonic)
                if same_posture
                else frame.received_monotonic
            )
            if visible != "clear" or posture == "unknown":
                posture_count = 0
            elif (
                posture_count == 0
                or frame.received_monotonic - posture_support
                >= self.settings.confirmation_spacing_seconds
            ):
                posture_count = min(2, posture_count + 1)
                posture_support = frame.received_monotonic
            # A changed age immediately invalidates support. It takes two new
            # independent reviews to establish another supported role estimate.
            track["role"] = {
                "age_claim": person["age_group"],
                "support_count": count,
                "support_mono": last_support,
                "mono": frame.received_monotonic,
                "stamp": frame.received_at,
                "evidence_ref": evidence_ref,
                "visible_posture": posture if posture_count >= 2 else "unknown",
                "posture_claim": posture,
                "posture_support_count": posture_count,
                "posture_support_mono": posture_support,
                "visibility": visible,
            }
            bound += 1
        # A fresh full-image omission is evidence against carrying old roles.
        for origin in origins:
            track = camera["tracks"].get(origin["track_id"])
            if (
                track is not None
                and track["epoch"] == origin["epoch"]
                and track["track_id"] not in mapped
            ):
                track["role"] = None
        camera["untracked"] = [
            {
                **p,
                "mono": frame.received_monotonic,
                "stamp": frame.received_at,
                "evidence_ref": evidence_ref,
            }
            for p in untracked
        ]
        self.counters["bound_reviews"] += 1
        self._remember_baby(frame.camera_id, now, now_utc)
        return {
            "status": "bound_model_claims",
            "bound_people": bound,
            "unbound_people": len(untracked),
            "input_detector_boxes": len(origins),
            "unbound_reasons": {
                reason: sum(p["binding_reason"] == reason for p in untracked)
                for reason in sorted({p["binding_reason"] for p in untracked})
            },
            "unbound_diagnostics": diagnostics,
        }

    def project(self, *, now, now_utc, sources, monitor_phase, review_available=True):
        output = {}
        supported, candidates, last_seen = [], [], []
        for camera_id, camera in self.cameras.items():
            source = sources.get(camera_id, {})
            frame = camera["previous"]
            age = None if frame is None else receipt_age_seconds(frame, now, now_utc)
            current = (
                monitor_phase == "detecting"
                and review_available
                and source.get("phase") == "receiving"
                and frame is not None
                and source.get("generation") == frame.generation
                and age is not None
                and age <= self.settings.track_gap_seconds
            )
            rows = []
            for track in camera["tracks"].values() if current else ():
                role = track["role"]
                role_age = None if role is None else _age(role["mono"], role["stamp"], now, now_utc)
                valid = role_age is not None and role_age <= self.settings.role_ttl_seconds
                item = {
                    "track_id": track["track_id"],
                    "bbox_xyxy": list(track["bbox"]),
                    "track_hits": track["hits"],
                    "association_ambiguous": track["ambiguous"],
                    "age_claim": role["age_claim"] if valid else "unknown",
                    "role_status": "supported"
                    if valid and role["support_count"] >= 2 and role["age_claim"] != "unknown"
                    else "candidate"
                    if valid and role["age_claim"] != "unknown"
                    else "unknown",
                    "visible_posture": role["visible_posture"] if valid else "unknown",
                    "visibility": role["visibility"] if valid else "unknown",
                    "evidence_age_seconds": None if not valid else round(role_age, 3),
                    "evidence_frame_received_at": None if not valid else role["stamp"].isoformat(),
                    "evidence_ref": None if not valid else role["evidence_ref"],
                }
                rows.append(item)
            room = self.rooms[camera_id]
            baby = [r for r in rows if r["age_claim"] == "baby"]
            for row in baby:
                entry = {
                    **row,
                    "camera_id": camera_id,
                    "room": room,
                    "last_detected_at": frame.received_at.isoformat(),
                }
                (supported if row["role_status"] == "supported" else candidates).append(entry)
            untracked = []
            if current:
                for person in camera["untracked"]:
                    probe_age = _age(person["mono"], person["stamp"], now, now_utc)
                    if probe_age is not None and probe_age <= self.settings.role_ttl_seconds:
                        item = {k: v for k, v in person.items() if k not in ("mono", "stamp")}
                        item.update(
                            evidence_age_seconds=round(probe_age, 3),
                            evidence_frame_received_at=person["stamp"].isoformat(),
                        )
                        untracked.append(item)
                        if item["age_claim"] == "baby":
                            candidates.append(
                                {
                                    **item,
                                    "camera_id": camera_id,
                                    "room": room,
                                    "role_status": "untracked_candidate",
                                }
                            )
            if camera["last_seen"] is not None:
                last_seen.append(camera["last_seen"])
            output[camera_id] = {
                "status": "current" if current else "unavailable",
                "semantics": "anonymous_tracks_and_unverified_model_inferences",
                "room": room,
                "generation": None if frame is None else frame.generation,
                "people": rows,
                "untracked_people": untracked,
                "last_seen_baby": camera["last_seen"],
                "detection_age_seconds": None if age is None else round(age, 3),
            }
        total = supported + candidates
        status = (
            "ambiguous"
            if len(total) > 1
            else "current_model_estimate"
            if len(supported) == 1
            else "candidate"
            if candidates
            else "last_seen"
            if last_seen
            else "unknown"
        )
        current_baby = supported[0] if status == "current_model_estimate" else None
        last_seen.sort(key=lambda x: x["last_detected_at"], reverse=True)
        return {
            "enabled": True,
            "version": 1,
            "semantics": "unverified_model_inferences",
            "status": status,
            "current": current_baby,
            "candidates": candidates,
            "last_seen": last_seen[0] if last_seen else None,
            "cameras": output,
            "accuracy_verified": False,
            "sleep_established": False,
            "time_semantics": "decoder_receipt_not_camera_exposure",
            "counters": dict(self.counters),
        }

    def snapshot(self, *, now, now_utc, sources, monitor_phase, review_available=True):
        """Choose one detection lane per camera; never union overlapping bodies."""
        context = dict(
            now=now,
            now_utc=now_utc,
            sources=sources,
            monitor_phase=monitor_phase,
            review_available=review_available,
        )
        native = self.project(**context)
        if self.auxiliary is None:
            return native
        alternate = self.auxiliary.project(**context)
        cameras, supported, candidates, historical = {}, [], [], []
        for camera_id, row in native["cameras"].items():
            auxiliary = alternate["cameras"][camera_id]
            if not row["people"] and auxiliary["status"] == "current" and auxiliary["people"]:
                row = {**auxiliary, "detection_view": self.auxiliary.cameras[camera_id].get("view")}
            else:
                row = {**row, "detection_view": "original"}
            cameras[camera_id] = row
            for person in row["people"]:
                if person["age_claim"] == "baby":
                    entry = {
                        **person,
                        "camera_id": camera_id,
                        "room": row["room"],
                        "detection_view": row["detection_view"],
                    }
                    (supported if person["role_status"] == "supported" else candidates).append(
                        entry
                    )
            for person in row["untracked_people"]:
                if person["age_claim"] == "baby":
                    candidates.append(
                        {
                            **person,
                            "camera_id": camera_id,
                            "room": row["room"],
                            "role_status": "untracked_candidate",
                        }
                    )
            camera_history = []
            for old in (
                native["cameras"][camera_id]["last_seen_baby"],
                auxiliary["last_seen_baby"],
            ):
                if old is not None:
                    historical.append(old)
                    camera_history.append(old)
            camera_history.sort(key=lambda old: old["last_detected_at"], reverse=True)
            row["last_seen_baby"] = camera_history[0] if camera_history else None
        total = supported + candidates
        status = (
            "ambiguous"
            if len(total) > 1
            else "current_model_estimate"
            if supported
            else "candidate"
            if candidates
            else "last_seen"
            if historical
            else "unknown"
        )
        historical.sort(key=lambda row: row["last_detected_at"], reverse=True)
        return {
            **native,
            "cameras": cameras,
            "status": status,
            "current": supported[0] if status == "current_model_estimate" else None,
            "candidates": candidates,
            "last_seen": historical[0] if historical else None,
            "auxiliary_counters": dict(self.auxiliary.counters),
        }
