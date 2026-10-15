"""Opt-in live comparison telemetry and a separate bounded local-probe queue.

Never changes native detections, production triggers, facts, or notifications.
One existing review worker must serialize probes with its other model tasks.
"""

from __future__ import annotations

import copy
import threading
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime

from baby_monitor.candidate_confirmation import (
    CandidateConfirmationPolicy,
    ConfirmationFrame,
    ConfirmationSettings,
)
from baby_monitor.live_detection import (
    AlternateViewReviewPolicy,
    LiveFrame,
    LiveObservation,
    receipt_age_seconds,
)
from baby_monitor.local_runners.subject_probe import validate_bound_result


@dataclass(frozen=True, slots=True)
class ConfirmationShadowRequest:
    observation: LiveObservation
    comparison: dict


class LiveConfirmationShadow:
    """Single source-coordinate auxiliary view, bounded history and first-evidence slots.

    The cloned old rule sees ALL checked auxiliary frames, including nonempty
    native shadow checks. actual_alternate_requested separately records whether
    the existing production alternate policy participated/requested on this frame.
    These denominators must not be conflated. A changed scale starts a fresh chain.
    """

    def __init__(self, camera_ids: tuple[str, ...], *, minimum_frame_confidence: float):
        self.settings = ConfirmationSettings(minimum_candidate_confidence=minimum_frame_confidence)
        self._candidate = CandidateConfirmationPolicy(camera_ids, self.settings)
        self._whole = AlternateViewReviewPolicy(minimum_frame_confidence=minimum_frame_confidence)
        self._filtered = AlternateViewReviewPolicy(
            minimum_frame_confidence=minimum_frame_confidence
        )
        self._cameras = camera_ids
        self._lock = threading.RLock()
        self._pending: dict[str, ConfirmationShadowRequest] = {}
        self._last_admitted = dict.fromkeys(camera_ids, -float("inf"))
        self._generation: dict[str, int] = {}
        self._view: dict[str, str] = {}
        self._counts = Counter()
        self._by_camera = {camera: Counter() for camera in camera_ids}
        self._checks = deque(maxlen=128)
        self._reviews = deque(maxlen=32)
        self._subject_probes = deque(maxlen=32)
        self._disabled = False

    def gap(self, *, reason: str) -> None:
        if reason not in ("suspension", "resume", "telemetry_failed"):
            raise ValueError("invalid confirmation shadow gap")
        with self._lock:
            self._counts["pending_discarded_for_" + reason] += len(self._pending)
            self._pending.clear()
            for camera in self._cameras:
                self._candidate.gap(camera)
                self._whole.state.pop(camera, None)
                self._filtered.state.pop(camera, None)
            self._counts["gaps_" + reason] += 1
            if reason == "telemetry_failed":
                self._disabled = True

    def observe(
        self,
        observation: LiveObservation,
        *,
        detector_view: str,
        actual_alternate_requested: bool | None,
    ) -> None:
        if (
            type(observation) is not LiveObservation
            or type(observation.frame) is not LiveFrame
            or observation.frame.camera_id not in self._cameras
            or detector_view not in ("rotation_180", "rotation_180_scale_fallback")
            or actual_alternate_requested is not None
            and type(actual_alternate_requested) is not bool
        ):
            raise ValueError("invalid confirmation shadow observation")
        frame = observation.frame
        checked = ConfirmationFrame(
            frame.camera_id,
            frame.generation,
            frame.sequence,
            frame.received_monotonic,
            observation.completed_monotonic,
            observation.detections,
        )
        with self._lock:
            if self._disabled:
                return
            previous_view = self._view.get(frame.camera_id)
            if previous_view is not None and previous_view != detector_view:
                self._candidate.gap(frame.camera_id)
                self._whole.state.pop(frame.camera_id, None)
                self._filtered.state.pop(frame.camera_id, None)
                self._counts["view_change_gaps"] += 1
            self._view[frame.camera_id] = detector_view
            new = self._candidate.observe(checked)
            rejected = new.reset_reason in ("stale_generation", "invalid_order")
            if rejected:
                self._whole.state.pop(frame.camera_id, None)
                self._filtered.state.pop(frame.camera_id, None)
                whole = filtered = False
                self._counts["rejected_checks"] += 1
            else:
                whole = self._whole.observe(observation) is not None
                filtered = (
                    self._filtered.observe(
                        LiveObservation(
                            frame,
                            tuple(
                                d
                                for d in observation.detections
                                if d.confidence >= self.settings.minimum_candidate_confidence
                            ),
                            observation.completed_monotonic,
                        )
                    )
                    is not None
                )
                previous_generation = self._generation.get(frame.camera_id)
                if previous_generation is not None and previous_generation != frame.generation:
                    self._counts["generation_discarded_requests"] += (
                        self._pending.pop(frame.camera_id, None) is not None
                    )
                self._generation[frame.camera_id] = frame.generation
            candidate = bool(new.review_requested_indices)
            counts = self._by_camera[frame.camera_id]
            counts["checked_frames"] += 1
            counts["zero_frames"] += not observation.detections
            counts["whole_requests"] += whole
            counts["filtered_whole_requests"] += filtered
            counts["candidate_requests"] += candidate
            counts["request_matrix_" + str(int(whole)) + str(int(candidate))] += 1
            counts["production_alternate_checks"] += actual_alternate_requested is not None
            counts["production_alternate_requests"] += actual_alternate_requested is True
            counts["nonempty_auxiliary_checks"] += actual_alternate_requested is None
            counts["ambiguous_frames"] += bool(new.ambiguous_indices)
            self._counts["checked_frames"] += 1
            row = {
                "camera_id": frame.camera_id,
                "generation": frame.generation,
                "sequence": frame.sequence,
                "raw_bgr_sha256": frame.sha256,
                "frame_received_at": frame.received_at.isoformat(),
                "received_monotonic": frame.received_monotonic,
                "completed_monotonic": observation.completed_monotonic,
                "width": frame.width,
                "height": frame.height,
                "detector_view": detector_view,
                "coordinate_space": "source_image_normalized",
                "whole_requested": whole,
                "filtered_whole_requested": filtered,
                "per_candidate": asdict(new),
                "actual_alternate_requested": actual_alternate_requested,
                "raw_boxes": [
                    {"confidence": d.confidence, "bbox_xyxy": list(d.bbox_xyxy)}
                    for d in observation.detections
                ],
                "probe_id": None,
            }
            # Sample the union, not just successful new-rule cases; agreement is a control.
            if not rejected and (whole or filtered or candidate):
                if frame.camera_id in self._pending:
                    self._counts["pending_slot_rejected"] += 1
                elif observation.completed_monotonic - self._last_admitted[frame.camera_id] < 30:
                    self._counts["probe_cooldown_rejected"] += 1
                else:
                    row["probe_id"] = uuid.uuid4().hex
                    self._pending[frame.camera_id] = ConfirmationShadowRequest(
                        observation,
                        copy.deepcopy(row),
                    )
                    self._last_admitted[frame.camera_id] = observation.completed_monotonic
                    self._counts["probe_queued"] += 1
            if len(self._checks) == self._checks.maxlen:
                self._counts["check_history_evicted"] += 1
            self._checks.append(row)

    def take(self, now: float, now_utc: datetime) -> ConfirmationShadowRequest | None:
        with self._lock:
            for camera, request in list(self._pending.items()):
                age = receipt_age_seconds(request.observation.frame, now, now_utc)
                if age is None or age > 120:
                    del self._pending[camera]
                    self._counts["probe_expired_or_clock_invalid"] += 1
            if not self._pending or self._disabled:
                return None
            camera = min(
                self._pending, key=lambda c: self._pending[c].observation.completed_monotonic
            )
            self._counts["probe_selected"] += 1
            return self._pending.pop(camera)

    def complete(self, request: ConfirmationShadowRequest, *, status: str, summary: dict) -> None:
        if status not in ("analyzed", "analysis_failed", "pre_resume_discarded"):
            raise ValueError("invalid confirmation shadow completion")
        with self._lock:
            self._counts["probe_" + status] += 1
            if len(self._reviews) == self._reviews.maxlen:
                self._counts["review_history_evicted"] += 1
            self._reviews.append(
                {
                    "probe_id": request.comparison["probe_id"],
                    "camera_id": request.observation.frame.camera_id,
                    "raw_bgr_sha256": request.observation.frame.sha256,
                    "status": status,
                    "summary": copy.deepcopy(summary),
                }
            )

    def record_subject_probe(
        self, observation: LiveObservation, result: object, *, frame_id: str, png_sha256: str
    ) -> dict:
        """Reuse an existing independent full-scene probe, including native zero frames.

        Only EXACT same-frame auxiliary checks may join it; a nearby check is not
        a substitute. Model disagreement is a hard-case lead, never ground truth.
        """
        if (
            type(observation) is not LiveObservation
            or type(observation.frame) is not LiveFrame
            or observation.frame.camera_id not in self._cameras
        ):
            raise ValueError("unadmitted shadow subject probe")
        frame = observation.frame
        comparison = {
            "raw_boxes": [
                {"confidence": d.confidence, "bbox_xyxy": list(d.bbox_xyxy)}
                for d in observation.detections
            ]
        }
        agreement = summarize_shadow_probe(
            ConfirmationShadowRequest(observation, comparison),
            result,
            frame_id=frame_id,
            png_sha256=png_sha256,
        )
        with self._lock:
            if self._disabled:
                return {"status": "disabled_after_error"}
            matched = next(
                (
                    row
                    for row in reversed(self._checks)
                    if (
                        row["camera_id"] == frame.camera_id
                        and row["generation"] == frame.generation
                        and row["sequence"] == frame.sequence
                        and row["raw_bgr_sha256"] == frame.sha256
                    )
                ),
                None,
            )
            native_present = bool(observation.detections)
            model_present = agreement["claimed_people"] > 0
            category = "native_" + str(int(native_present)) + "_model_" + str(int(model_present))
            self._counts["subject_probe_checks"] += 1
            self._by_camera[frame.camera_id][category] += 1
            row = {
                "camera_id": frame.camera_id,
                "generation": frame.generation,
                "sequence": frame.sequence,
                "raw_bgr_sha256": frame.sha256,
                "frame_received_at": frame.received_at.isoformat(),
                "native_candidate_count": len(observation.detections),
                "native_model_category": category,
                "auxiliary_check_same_frame": matched is not None,
                "auxiliary_candidate_count": None if matched is None else len(matched["raw_boxes"]),
                "auxiliary_probe_id": None if matched is None else matched["probe_id"],
                "model_agreement": agreement,
            }
            if len(self._subject_probes) == self._subject_probes.maxlen:
                self._counts["subject_probe_history_evicted"] += 1
            self._subject_probes.append(row)
            return copy.deepcopy(row)

    def summary(self) -> dict:
        with self._lock:
            return {
                "enabled": True,
                "disabled_after_error": self._disabled,
                "semantics": "shadow_local_review_opportunities_not_person_facts",
                "settings": asdict(self.settings),
                "counts": dict(self._counts),
                "cameras": {c: dict(v) for c, v in self._by_camera.items()},
                "pending_requests": len(self._pending),
                "capacity_per_camera": 1,
                "probe_max_receipt_age_seconds": 120,
                "probe_cooldown_seconds": 30,
                "regular_slots_before_shadow": 4,
                "check_history_capacity": 128,
                "review_history_capacity": 32,
                "subject_probe_history_capacity": 32,
                "production_triggers_changed": False,
                "accuracy_verified": False,
                "baby_location": "unknown",
                "baby_state": "unknown",
            }

    def snapshot(self) -> dict:
        with self._lock:
            return {
                **self.summary(),
                "checks": copy.deepcopy(list(self._checks)),
                "reviews": copy.deepcopy(list(self._reviews)),
                "subject_probes": copy.deepcopy(list(self._subject_probes)),
            }


def summarize_shadow_probe(
    request: ConfirmationShadowRequest,
    result: object,
    *,
    frame_id: str,
    png_sha256: str,
) -> dict:
    """Exact-image model agreement only; null/no matching heads remain unresolved."""
    bound = validate_bound_result(result, frame_id=frame_id, input_sha256=png_sha256)
    people = bound["claims"]["people"]
    boxes = request.comparison["raw_boxes"]
    matches = []
    point_degrees = Counter()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["bbox_xyxy"]
        inside = [
            j
            for j, p in enumerate(people)
            if p["head_center"] is not None
            and x1 <= p["head_center"][0] / 1000 <= x2
            and y1 <= p["head_center"][1] / 1000 <= y2
        ]
        matches.append((i, inside))
        point_degrees.update(inside)
    support = []
    for i, indices in matches:
        kind = (
            "unique_head_claim_support"
            if len(indices) == 1 and point_degrees[indices[0]] == 1
            else "ambiguous_head_claim_support"
            if indices
            else "no_located_head_claim_support"
        )
        support.append({"candidate_index": i, "head_claim_indices": indices, "status": kind})
    return {
        "semantics": "same_image_unverified_model_agreement_not_ground_truth",
        "claimed_people": len(people),
        "unlocated_head_claims": sum(p["head_center"] is None for p in people),
        "has_uncertainties": bool(bound["claims"]["uncertainties"]),
        "head_support": support,
        "model_person_presence_claim": "present_claim" if people else "no_person_claim",
        "no_head_support_means_false_detection": False,
        "unique_head_support_means_unique_identity": False,
        "person_truth_available": False,
        "accuracy_verified": False,
    }
