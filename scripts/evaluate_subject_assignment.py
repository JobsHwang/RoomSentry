"""Bounded saved-record graph audit, separate from live roles and visual accuracy.

First reproduce each native record's real assignments and complete audit. Compare
only its exact source-frame point/box graph, then inspect the original exported
anonymous path. No pixels, model calls, training labels or temporal role replay.
"""

from __future__ import annotations

import re
from collections import Counter

from scripts.subject_assignment_diagnostic import diagnose_global_assignment
from scripts.subject_layer_diagnostic import _origin_path, diagnose

REVISION = "saved_subject_assignment_audit_v1"


def evaluate_records(records, *, unusable_png_sha256=()):
    if (
        type(records) is not list
        or not 1 <= len(records) <= 128
        or any(type(r) is not dict for r in records)
    ):
        raise ValueError("bounded saved subject records required")
    digests = [r.get("record_sha256") for r in records]
    if any(type(d) is not str or re.fullmatch("[0-9a-f]{64}", d) is None for d in digests):
        raise ValueError("invalid saved record digest")
    if len(set(digests)) != len(digests):
        raise ValueError("duplicate saved source records")
    if (
        type(unusable_png_sha256) not in (tuple, list)
        or len(unusable_png_sha256) > 8
        or any(
            type(d) is not str or re.fullmatch("[0-9a-f]{64}", d) is None
            for d in unusable_png_sha256
        )
        or len(set(unusable_png_sha256)) != len(unusable_png_sha256)
    ):
        raise ValueError("invalid explicit unusable-image exclusions")
    counts = Counter(records_supplied=len(records))
    rows = []
    input_keys = set()
    for record in records:
        row = {"source_record_sha256": record["record_sha256"]}
        rows.append(row)
        if record.get("detector_view") != "original":
            row["status"] = "non_native_lane_excluded"
            counts["non_native_lane_excluded"] += 1
            continue
        counts["native_records"] += 1
        try:
            actual = diagnose(record)
        except ValueError:
            row["status"] = "invalid_or_nonreproducible_native_input"
            counts["native_replay_rejected"] += 1
            continue
        counts["matching_replays_passed"] += 1
        row.update(
            actual_assignment_and_audit_reproduced=True,
            record_projection_sha256=actual["record_projection_sha256"],
            camera_id=actual["camera_id"],
            frame_sequence=actual["frame_sequence"],
            png_sha256=actual["png_sha256"],
            bgr_sha256=actual["bgr_sha256"],
            actual_online_binding_status=record.get("people_binding", {}).get("status"),
        )
        input_keys.add(
            (
                record["camera_id"],
                record["detector_view"],
                record["frame_generation"],
                record["frame_sequence"],
                record["raw_bgr_sha256"],
            )
        )
        if actual["png_sha256"] in unusable_png_sha256:
            row["status"] = "known_unusable_visual_input_excluded"
            counts["known_unusable_visual_input_excluded"] += 1
            continue
        frames = record["people_trace"]["frames"]
        source_boxes = frames[0]["detections"]
        people = record["subject_result"]["claims"]["people"]
        try:
            graph = diagnose_global_assignment(
                probe_input_sha256=actual["png_sha256"],
                detector_input_sha256=actual["png_sha256"],
                centers=[p["head_center"] for p in people],
                boxes=[d["bbox_xyxy"] for d in source_boxes],
            )
        except ValueError:
            row["status"] = "graph_input_rejected_after_replay"
            counts["graph_input_rejected_after_replay"] += 1
            continue
        counts["graph_evaluated_records"] += 1
        counts["graph_status_" + graph["status"]] += 1
        row.update(status="evaluated", graph=graph, claims=[])
        assignment = graph["candidate_box_indices"]
        for index, (person, legacy) in enumerate(
            zip(people, graph["legacy"]["subjects"], strict=True)
        ):
            counts["model_claims"] += 1
            old_unique = legacy["status"] == "unique_spatial_match"
            counts["legacy_unique_claims"] += old_unique
            # Unresolved global components cannot revoke an existing unique edge.
            box_index = (
                legacy["head_indices"][0]
                if old_unique
                else assignment[index]
                if assignment is not None
                else None
            )
            added = not old_unique and box_index is not None
            counts["graph_added_candidate_edges"] += added
            counts[person["age_group"] + "_model_added_candidate_edges"] += added
            if old_unique:
                assert box_index == legacy["head_indices"][0]
                counts["legacy_unique_preserved"] += 1
            path = None if box_index is None else _origin_path(source_boxes[box_index], frames)
            if path is not None:
                counts["effective_origin_candidates"] += 1
                counts["origin_candidates_with_observed_gap"] += path["first_missing"] is not None
                if added:
                    counts["added_candidates_with_observed_gap"] += (
                        path["first_missing"] is not None
                    )
                    counts["added_candidates_absent_at_last_saved_input"] += not path[
                        "present_at_interval_end"
                    ]
            row["claims"].append(
                {
                    "claim_index": index,
                    "model_age_claim": person["age_group"],
                    "legacy_unique": old_unique,
                    "candidate_added": added,
                    "effective_source_box_index": box_index,
                    "anonymous_origin_path": path,
                }
            )
        row["review_duration_ms"] = actual["bonsai"]["review_duration_ms"]
        row["tracking"] = actual["tracking"]
    return {
        "revision": REVISION,
        "counts": dict(counts),
        "rows": rows,
        "explicit_unusable_png_sha256": list(unusable_png_sha256),
        "unique_native_review_input_keys": len(input_keys),
        "semantics": "saved_record_source_graph_and_original_path_only",
        "reference_provenance": "unverified_model_claims",
        "pixels_reverified_by_this_function": False,
        "epochs_are_exported_not_role_policy_replayed": True,
        "full_video_coverage": False,
        "human_verified": False,
        "ground_truth": False,
        "new_model_calls": 0,
        "pixel_reads": 0,
        "accuracy_measured": False,
        "production_changed": False,
        "roles_propagated": False,
        "deployment_approved": False,
    }
