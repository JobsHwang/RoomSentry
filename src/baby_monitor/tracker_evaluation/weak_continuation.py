"""Causal three-arm replay of one explicitly bounded, low-score debug trace.

Cold-start independent shadow, NOT reproduction of the resident people's state.
Strong-only, unrestricted weak continuation, and duplicate/ambiguity-controlled
weak continuation share every admitted input. No VLM role is propagated.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from baby_monitor.motion_tracker import (
    MotionObjectTracker,
    MotionTrackerConfig,
    _bbox_iou,
    _match,
    _matching_scores,
)
from baby_monitor.object_tracker import DetectionFrame, FrameDetection


def compare_weak_continuation(rows: list[dict]) -> dict:
    if not isinstance(rows, list) or not 1 <= len(rows) <= 12000:
        raise ValueError("invalid weak continuation input count")
    configs = {
        name: MotionTrackerConfig(
            high_confidence_threshold=0.35,
            low_confidence_threshold=0.20,
            primary_match_threshold=0.30,
            secondary_match_threshold=0.30,
        )
        for name in ("strong", "weak", "controlled")
    }
    cameras = {}
    reports = []
    for row in rows:
        camera = row["camera_id"]
        if camera not in cameras:
            if len(cameras) >= 4:
                raise ValueError("weak continuation camera limit")
            cameras[camera] = {"previous": None, "segment": 0, "arms": {}}
        state = cameras[camera]
        wall = datetime.fromisoformat(row["received_at"])
        mono = row["received_monotonic"]
        sequence, generation = row["sequence"], row["generation"]
        if (
            wall.utcoffset() is None
            or type(mono) not in (int, float)
            or not math.isfinite(mono)
            or mono < 0
        ):
            raise ValueError("invalid weak continuation clock")
        if any(type(value) is not int or value < 0 for value in (sequence, generation)):
            raise ValueError("invalid weak continuation sequence")
        clock = datetime(2000, 1, 1, tzinfo=UTC) + timedelta(seconds=mono)
        previous = state["previous"]
        reset = previous is None
        if previous is not None:
            old_generation, old_sequence, old_wall, old_mono = previous
            if generation < old_generation or (
                generation == old_generation and sequence <= old_sequence
            ):
                raise ValueError("weak continuation input is not chronological")
            elapsed, wall_elapsed = mono - old_mono, (wall - old_wall).total_seconds()
            reset = (
                generation != old_generation
                or min(elapsed, wall_elapsed) <= 0
                or max(elapsed, wall_elapsed) > 2
            )
        strong = tuple(
            FrameDetection(f"s_{index}", "person", item["confidence"], tuple(item["bbox_xyxy"]))
            for index, item in enumerate(row["native"])
        )
        if len(strong) > 100 or any(item.confidence < 0.35 for item in strong):
            raise ValueError("invalid native strong boxes")
        low = row["low"]
        equal = (
            None
            if low is None
            else [item for item in low if item["confidence"] >= 0.35] == row["native"]
        )
        if equal != row["strong_output_equal"]:
            raise ValueError("weak trace equality marker mismatch")
        # Any incomplete or changed strong output is a causal break for ALL arms,
        # not a silently removed difficult frame. Preserve its raw denominator.
        rejected = low is None or not equal
        weak = ()
        if not rejected:
            if len(low) > 100:
                raise ValueError("weak continuation detection limit")
            parsed = tuple(
                FrameDetection(f"w_{index}", "person", item["confidence"], tuple(item["bbox_xyxy"]))
                for index, item in enumerate(low)
            )
            if any(item.confidence < 0.20 for item in parsed):
                raise ValueError("low box below threshold")
            weak = tuple(item for item in parsed if item.confidence < 0.35)
        if reset or rejected:
            state["segment"] += 1
            state["arms"] = {
                name: {
                    "tracker": MotionObjectTracker(
                        config=config, camera_id=camera, detector_adapter="weak_continuation_shadow"
                    ),
                    "anchors": {},
                }
                for name, config in configs.items()
            }
        state["previous"] = generation, sequence, wall, mono
        result = {
            "camera_id": camera,
            "generation": generation,
            "sequence": sequence,
            "native_count": len(strong),
            "weak_count": len(weak),
            "reset": reset,
            "rejected": rejected,
            "arms": {},
        }
        if rejected:
            reports.append(result)
            continue
        for name, arm in state["arms"].items():
            tracker, anchors = arm["tracker"], arm["anchors"]
            # Weak evidence may bridge a short dip, never renew its strong seed
            # indefinitely. This is independent of the latest weak update age.
            tracker.state = replace(
                tracker.state,
                active_tracks=tuple(
                    track
                    for track in tracker.state.active_tracks
                    if track.tracking_id in anchors
                    and (clock - anchors[track.tracking_id][1]).total_seconds() <= 2
                ),
            )
            selected, duplicate, ambiguous = (), 0, 0
            if name == "weak":
                selected = weak
            elif name == "controlled":
                tracks = sorted(tracker.state.active_tracks, key=lambda track: track.tracking_id)
                high_pairs = _match(
                    tracks,
                    list(strong),
                    occurred_at=clock,
                    minimum_score=0.30,
                    config=tracker.config,
                )
                used = {pair[0] for pair in high_pairs}
                remaining = [track for index, track in enumerate(tracks) if index not in used]
                unique = []
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
                    remaining, unique, occurred_at=clock, minimum_score=0.30, config=tracker.config
                )
                edges = [
                    [index for index, values in enumerate(scores) if values[col] >= 0.30]
                    for col in range(len(unique))
                ]
                selected = tuple(
                    item
                    for col, item in enumerate(unique)
                    if len(edges[col]) == 1 and sum(edges[col][0] in edge for edge in edges) == 1
                )
                ambiguous = len(unique) - len(selected)
            batch = DetectionFrame(
                f"f_{state['segment']}_{generation}_{sequence}", clock, (*strong, *selected)
            )
            observations = tracker.process(batch)
            links, continuing_weak = [], 0
            for item, observation in zip(batch.detections, observations, strict=True):
                identifier = observation.tracking_id
                if item.detection_id.startswith("s_"):
                    endpoint = [generation, sequence, int(item.detection_id[2:])]
                    if identifier in anchors:
                        links.append([anchors[identifier][0], endpoint])
                    anchors[identifier] = endpoint, clock
                elif identifier in {track.tracking_id for track in tracker.state.active_tracks}:
                    continuing_weak += 1
            active = {track.tracking_id for track in tracker.state.active_tracks}
            arm["anchors"] = {
                identifier: anchor for identifier, anchor in anchors.items() if identifier in active
            }
            tracker.state = replace(
                tracker.state,
                pending_observations=(),
                processed_frames=tracker.state.processed_frames[-64:],
            )
            result["arms"][name] = {
                "strong_links": links,
                "weak_continuations": continuing_weak,
                "duplicate_weak_rejected": duplicate,
                "ambiguous_weak_rejected": ambiguous,
            }
        reports.append(result)
    return {
        "version": 1,
        "frames": reports,
        "configs": {name: config.to_dict() for name, config in configs.items()},
        "cold_start_shadow": True,
        "live_assignments_reproduced": False,
        "human_ground_truth_used": False,
        "accuracy_evaluated": False,
        "role_propagation": False,
        "production_modified": False,
    }
