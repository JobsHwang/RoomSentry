"""Independent offline candidate: weak samples cannot rewrite strong kinematics.

Reserve strong-reference assignments before weak-geometry recovery. This is a
cold-start diagnostic, not a resident tracker or an identity/role inference.
The frozen original three-arm replay remains separate and unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from baby_monitor.motion_tracker import (
    MotionActiveTrack,
    MotionTrackerConfig,
    MotionTrackSample,
    _bbox_iou,
    _match,
    _matching_scores,
    _new_tracking_id,
)
from baby_monitor.object_tracker import DetectionFrame, DetectionFrameBatch, FrameDetection
from baby_monitor.tracker_evaluation.weak_continuation import compare_weak_continuation


def _primary_pairs(tracks, references, strong, clock, config):
    """Reserve reference matches; recover only completely unqualified columns."""
    ordered_references = tuple(references[track.tracking_id] for track in tracks)
    scores = _matching_scores(
        ordered_references, strong, occurred_at=clock, minimum_score=0.30, config=config
    )
    reserved = _match(
        ordered_references, strong, occurred_at=clock, minimum_score=0.30, config=config
    )
    used_tracks = {i for i, _ in reserved}
    protected = {
        col for col in range(len(strong)) if any(row[col] + 1e-15 >= 0.30 for row in scores)
    }
    remaining_rows = [i for i in range(len(tracks)) if i not in used_tracks]
    remaining_cols = [j for j in range(len(strong)) if j not in protected]
    recovered = _match(
        tuple(tracks[i] for i in remaining_rows),
        tuple(strong[j] for j in remaining_cols),
        occurred_at=clock,
        minimum_score=0.30,
        config=config,
    )
    return (
        (*reserved, *((remaining_rows[i], remaining_cols[j]) for i, j in recovered)),
        len(reserved),
        len(recovered),
        len(protected),
    )


def _select_weak(remaining, strong, weak, clock, config):
    unique, duplicate = [], 0
    for item in weak:
        if any(
            _bbox_iou(item.bbox_xyxy, other.bbox_xyxy) >= 0.70
            for other in (*strong, *weak)
            if other is not item
        ):
            duplicate += 1
        else:
            unique.append(item)
    scores = _matching_scores(
        remaining, unique, occurred_at=clock, minimum_score=0.30, config=config
    )
    edges = [
        [i for i, values in enumerate(scores) if values[col] >= 0.30] for col in range(len(unique))
    ]
    selected = tuple(
        item
        for col, item in enumerate(unique)
        if len(edges[col]) == 1 and sum(edges[col][0] in edge for edge in edges) == 1
    )
    return selected, duplicate, len(unique) - len(selected)


def compare_guarded_weak_continuation(rows: list[dict]) -> dict:
    # Reuse ALL validation/causal-break rules and preserve the old baseline as
    # an explicit result. No changed old matcher or hidden cold-start checkpoint.
    baseline = compare_weak_continuation(rows)
    config = MotionTrackerConfig.from_dict(baseline["configs"]["strong"])
    cameras, frames = {}, []
    for row, control in zip(rows, baseline["frames"], strict=True):
        camera = row["camera_id"]
        state = cameras.setdefault(
            camera, {"segment": 0, "active": {}, "references": {}, "anchors": {}}
        )
        if control["reset"] or control["rejected"]:
            state.update(segment=state["segment"] + 1, active={}, references={}, anchors={})
        output = {
            **{
                key: control[key]
                for key in (
                    "camera_id",
                    "generation",
                    "sequence",
                    "native_count",
                    "weak_count",
                    "reset",
                    "rejected",
                )
            },
            "strong_links": [],
            "weak_continuations": 0,
            "duplicate_weak_rejected": 0,
            "ambiguous_weak_rejected": 0,
            "reserved_reference_matches": 0,
            "weak_recovery_matches": 0,
            "protected_strong_columns": 0,
        }
        frames.append(output)
        if control["rejected"]:
            continue
        clock = datetime(2000, 1, 1, tzinfo=UTC) + timedelta(seconds=row["received_monotonic"])
        references, anchors = state["references"], state["anchors"]
        # Only a strong observation renews the two-second seed. Weak updates
        # never create tracks or extend that deadline, including zero-box input.
        active = {
            identifier: track
            for identifier, track in state["active"].items()
            if (clock - references[identifier].current.occurred_at).total_seconds() <= 2
        }
        references = {identifier: references[identifier] for identifier in active}
        anchors = {identifier: anchors[identifier] for identifier in active}
        tracks = tuple(active[key] for key in sorted(active))
        strong = tuple(
            FrameDetection(f"s_{i}", "person", item["confidence"], tuple(item["bbox_xyxy"]))
            for i, item in enumerate(row["native"])
        )
        weak = tuple(
            FrameDetection(f"w_{i}", "person", item["confidence"], tuple(item["bbox_xyxy"]))
            for i, item in enumerate(row["low"])
            if item["confidence"] < 0.35
        )
        primary, reserved, recovered, protected = _primary_pairs(
            tracks, references, strong, clock, config
        )
        remaining = tuple(
            track for i, track in enumerate(tracks) if i not in {a for a, _ in primary}
        )
        selected, duplicate, ambiguous = _select_weak(remaining, strong, weak, clock, config)
        secondary = _match(
            remaining, selected, occurred_at=clock, minimum_score=0.30, config=config
        )
        frame = DetectionFrame(
            f"f_{state['segment']}_{row['generation']}_{row['sequence']}",
            clock,
            (*strong, *selected),
        )
        batch = DetectionFrameBatch(
            camera_id=camera, detector_adapter="weak_continuation_shadow", frames=(frame,)
        )
        matched_high = {}
        for i, j in primary:
            track, detection = tracks[i], strong[j]
            identifier = track.tracking_id
            active[identifier] = track.update(detection, clock)
            # Deliberately use the previous STRONG reference, never a weak
            # sample, to update the reference velocity/extent estimate.
            references[identifier] = references[identifier].update(detection, clock)
            matched_high[j] = identifier
        for i, j in secondary:
            track = remaining[i]
            active[track.tracking_id] = track.update(selected[j], clock)
        for index, detection in enumerate(strong):
            if index not in matched_high:
                identifier = _new_tracking_id(
                    batch, config, frame_id=frame.frame_id, detection=detection, retained=True
                )
                if identifier in active:
                    raise ValueError("guarded weak shadow ID collision")
                track = MotionActiveTrack(
                    identifier,
                    "person",
                    MotionTrackSample(clock, detection.confidence, detection.bbox_xyxy),
                )
                active[identifier] = references[identifier] = track
                matched_high[index] = identifier
            identifier = matched_high[index]
            endpoint = [row["generation"], row["sequence"], index]
            if identifier in anchors:
                output["strong_links"].append([anchors[identifier], endpoint])
            anchors[identifier] = endpoint
        if len(active) > config.max_active_tracks:
            raise ValueError("guarded weak shadow active track limit")
        state.update(active=active, references=references, anchors=anchors)
        output.update(
            weak_continuations=len(secondary),
            duplicate_weak_rejected=duplicate,
            ambiguous_weak_rejected=ambiguous,
            reserved_reference_matches=reserved,
            weak_recovery_matches=recovered,
            protected_strong_columns=protected,
        )
    return {
        "version": 1,
        "candidate_revision": "strong_reference_reservation_v1",
        "baseline": baseline,
        "frames": frames,
        "config": config.to_dict(),
        "cold_start_shadow": True,
        "live_assignments_reproduced": False,
        "human_ground_truth_used": False,
        "accuracy_evaluated": False,
        "role_propagation": False,
        "production_modified": False,
        "new_model_calls": 0,
    }
