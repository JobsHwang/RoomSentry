"""Separate saved native detector, model claims and anonymous tracking evidence.

No model execution, pixels, labels, configuration writes or deployment decisions.
Callers must bind records to private source images before presenting a review.
These development strata are deliberately biased, not an accuracy denominator.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime

from baby_monitor.head_point_review import fingerprint
from baby_monitor.live_people_association import describe_association
from baby_monitor.local_runners.subject_head_match import match_head_centers
from baby_monitor.local_runners.subject_state_probe import validate_bound_result
from baby_monitor.motion_tracker import MotionObjectTracker
from baby_monitor.tracker_evaluation.association_checkpoint import _inputs

REVISION = "native_subject_layer_diagnostic_v1"
STRATA = (
    "baby_claim_zero_boxes",
    "baby_claim_outside_nonempty_boxes",
    "baby_claim_shared_box",
    "baby_claim_overlapping_boxes",
    "baby_claim_unique_origin_then_gap",
    "camera_01_control",
    "camera_02_control",
    "camera_03_control",
)


def _event(row, first, index):
    return {
        "input_index": index,
        "sequence": row["sequence"],
        "seconds_from_source": (
            row["received_monotonic_seconds"] - first["received_monotonic_seconds"]
        ),
        "box_count": len(row["detections"]),
    }


def _origin_path(origin, frames):
    first_missing, first_epoch_change, first_return = None, None, None
    previous_present = True
    observed_inputs = 0
    transitions_to_missing = 0
    for index, row in enumerate(frames):
        matches = [d for d in row["detections"] if d["track_id"] == origin["track_id"]]
        if len(matches) > 1:
            raise ValueError("duplicate anonymous track in one input")
        present = bool(matches)
        observed_inputs += present
        if not present and previous_present:
            transitions_to_missing += 1
            if first_missing is None:
                first_missing = _event(row, frames[0], index)
        if present and first_missing is not None and first_return is None:
            first_return = _event(row, frames[0], index)
        if present and matches[0]["epoch"] != origin["epoch"] and first_epoch_change is None:
            first_epoch_change = _event(row, frames[0], index)
        previous_present = present
    return {
        "observed_inputs": observed_inputs,
        "first_missing": first_missing,
        "first_epoch_change": first_epoch_change,
        "first_same_id_return": first_return,
        "transitions_to_missing": transitions_to_missing,
        "present_at_interval_end": previous_present,
        "same_id_is_not_identity": True,
    }


def diagnose(record):
    """Require complete actual matching replay before deriving any path diagnosis.

    Native original-image lane only: auxiliary geometry needs its own explicit
    coordinate transform and is rejected, not silently pooled with native boxes.
    Source PNG/BGR verification remains a separate caller responsibility.
    """
    try:
        trace = record["people_trace"]
        if re.fullmatch(r"[0-9a-f]{64}", record["record_sha256"]) is None:
            raise ValueError("invalid source record digest")
        if datetime.fromisoformat(record["selected_at"]).utcoffset() is None:
            raise ValueError("missing selected time zone")
        if record["detector_view"] != "original" or trace["detection_view"] != "original":
            raise ValueError("native original lane required")
        if (
            trace["camera_id"],
            trace["review_frame_generation"],
            trace["review_frame_sequence"],
            trace["review_raw_bgr_sha256"],
        ) != (
            record["camera_id"],
            record["frame_generation"],
            record["frame_sequence"],
            record["raw_bgr_sha256"],
        ):
            raise ValueError("record trace differs")
        result = record["subject_result"]
        validate_bound_result(
            result, frame_id=result["frame_id"], input_sha256=result["input_sha256"]
        )
        duration = record["review_duration_ms"]
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid review duration")
        config, before, batch, expected = _inputs(trace)
        tracker = MotionObjectTracker(
            config=config,
            camera_id=before.camera_id,
            detector_adapter=before.detector_adapter,
            state=before,
        )
        actual = []
        for row, frame in zip(trace["frames"], batch.frames, strict=True):
            state = tracker.state
            observations = tracker.process(frame)
            if describe_association(state, frame, observations, config) != row["association_audit"]:
                raise ValueError("actual association audit differs")
            ids = []
            for d, observation, saved in zip(
                frame.detections, observations, row["detections"], strict=True
            ):
                actual.append([frame.frame_id, d.detection_id, observation.tracking_id])
                if type(saved.get("epoch")) is not int or saved["epoch"] < 0:
                    raise ValueError("invalid evidence epoch")
                ids.append(saved["track_id"])
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate track")
        if actual != expected:
            raise ValueError("actual assignment differs")
        frames = trace["frames"]
        first = frames[0]
        people = result["claims"]["people"]
        matched = match_head_centers(
            probe_input_sha256=result["input_sha256"],
            head_input_sha256=result["input_sha256"],
            centers=[p["head_center"] for p in people],
            heads=[d["bbox_xyxy"] for d in first["detections"]],
        )["subjects"]
        frequencies = Counter(i for m in matched for i in m["head_indices"])
        claims = []
        for index, (person, match) in enumerate(zip(people, matched, strict=True)):
            options = match["head_indices"]
            spatial = (
                "head_not_located"
                if person["head_center"] is None
                else "no_detector_box_at_head"
                if not options
                else "overlapping_detector_boxes"
                if len(options) > 1
                else "multiple_heads_in_one_box"
                if frequencies[options[0]] > 1
                else "unique_point_containment"
            )
            path = (
                _origin_path(first["detections"][options[0]], frames) if len(options) == 1 else None
            )
            claims.append(
                {
                    "claim_index": index,
                    "model_age_claim": person["age_group"],
                    "model_posture_claim": person["visible_posture"],
                    "model_visibility_claim": person["visibility"],
                    "spatial_diagnostic": spatial,
                    "source_box_indices": options,
                    "anonymous_origin_path": path,
                }
            )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid or nonreproducible subject layer input") from exc
    tags = []
    for claim in claims:
        if claim["model_age_claim"] != "baby":
            continue
        spatial, path = claim["spatial_diagnostic"], claim["anonymous_origin_path"]
        if not first["detections"]:
            tags.append(STRATA[0])
        elif spatial == "no_detector_box_at_head":
            tags.append(STRATA[1])
        elif spatial == "multiple_heads_in_one_box":
            tags.append(STRATA[2])
        elif spatial == "overlapping_detector_boxes":
            tags.append(STRATA[3])
        elif spatial == "unique_point_containment" and path["first_missing"] is not None:
            tags.append(STRATA[4])
    if record["camera_id"] in ("camera_01", "camera_02", "camera_03"):
        tags.append(record["camera_id"] + "_control")
    return {
        "revision": REVISION,
        "record_projection_sha256": fingerprint(record),
        "source_record_sha256": record["record_sha256"],
        "camera_id": record["camera_id"],
        "frame_sequence": record["frame_sequence"],
        "png_sha256": result["input_sha256"],
        "bgr_sha256": record["raw_bgr_sha256"],
        "selected_at": record["selected_at"],
        "strata": sorted(set(tags)),
        "detector": {
            "source_box_count": len(first["detections"]),
            "reference_status": "pending_independent_visible_people_reference",
            "false_negatives": None,
            "false_positives": None,
        },
        "bonsai": {
            "claims": claims,
            "review_duration_ms": duration,
            "reference_status": "pending_independent_role_and_posture_reference",
            "role_correct": None,
            "role_incorrect": None,
        },
        "tracking": {
            "actual_assignments_reproduced": True,
            "actual_audits_reproduced": True,
            "input_count": len(frames),
            "zero_box_inputs": sum(not row["detections"] for row in frames),
            "multiple_box_inputs": sum(len(row["detections"]) > 1 for row in frames),
            "monotonic_span_seconds": (
                frames[-1]["received_monotonic_seconds"] - first["received_monotonic_seconds"]
            ),
            "epoch_values_are_exported_not_independently_replayed": True,
            "reference_status": "pending_independent_temporal_identity_reference",
            "id_switches": None,
        },
        "accuracy_measured": False,
        "model_claims_are_ground_truth": False,
        "deployment_approved": False,
        "pixels_verified_by_this_function": False,
    }


def select_development_cases(reports):
    """First unused exact-image hash per fixed stratum; missing stays explicit."""
    if not 1 <= len(reports) <= 128:
        raise ValueError("bounded development reports required")
    if any(r.get("revision") != REVISION for r in reports):
        raise ValueError("different diagnostic revision")
    ordered = sorted(reports, key=lambda r: (r["selected_at"], r["source_record_sha256"]))
    used, items, missing = set(), [], []
    for stratum in STRATA:
        eligible = [r for r in ordered if stratum in r["strata"] and r["png_sha256"] not in used]
        if not eligible:
            missing.append(stratum)
            continue
        chosen = eligible[0]
        used.add(chosen["png_sha256"])
        items.append(
            {
                "item_id": f"frame_{len(items) + 1:02}",
                "selection_stratum": stratum,
                "eligible_unused_count": len(eligible),
                "report": chosen,
            }
        )
    return {
        "revision": REVISION,
        "items": items,
        "missing_strata": missing,
        "input_reports": len(reports),
        "unique_image_count": len({r["png_sha256"] for r in reports}),
        "selection": "first_unused_image_per_fixed_stratum_in_selected_utc_order",
        "selection_purpose": "development_counterexamples",
        "independent_holdout": False,
        "representative_accuracy_denominator": False,
        "negative_ground_truth_inferred": False,
    }
