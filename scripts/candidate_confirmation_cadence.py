"""Bounded elapsed-receipt cadence counterfactuals over admitted detection metadata.

These are overlapping replays, not independent trials, measured live throughput,
source PTS coverage, or permission to interpolate unsampled detections.
"""

from __future__ import annotations

import math

from scripts.compare_candidate_confirmation import compare_confirmation


def compare_cadence(
    frames: object,
    *,
    interval_seconds: float = 1.0,
    phases: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8),
) -> dict:
    if (
        type(interval_seconds) not in (int, float)
        or not 0.2 <= interval_seconds <= 3
        or not math.isfinite(interval_seconds)
        or type(phases) is not tuple
        or not 1 <= len(phases) <= 8
        or any(
            type(p) not in (int, float) or not 0 <= p < interval_seconds or not math.isfinite(p)
            for p in phases
        )
        or len(set(phases)) != len(phases)
    ):
        raise ValueError("invalid cadence interval or phases")
    # Strict box/clock/field validation of ALL rows, including skipped/zero rows.
    full = compare_confirmation(frames)
    admitted = [row["input"] for row in full["rows"]]
    previous = {}
    for row in admitted:
        key = row["camera_id"]
        old = previous.get(key)
        if old is not None and (
            row["generation"] < old["generation"]
            or row["generation"] == old["generation"]
            and (
                row["sequence"] <= old["sequence"]
                or row["received_monotonic"] <= old["received_monotonic"]
                or row["completed_monotonic"] < old["completed_monotonic"]
            )
        ):
            raise ValueError("cadence selection cannot hide invalid source order")
        previous[key] = row
    reports = []
    for phase in phases:
        origins, last, selected = {}, {}, []
        for row in admitted:
            key = row["camera_id"], row["generation"]
            now = row["received_monotonic"]
            origin = origins.setdefault(key, now)
            if now - origin < phase:
                continue
            if key not in last or now - last[key] >= interval_seconds:
                selected.append(row)
                last[key] = now
        reports.append(
            {
                "phase_seconds": phase,
                "selected_frame_ids": [row["frame_id"] for row in selected],
                "unselected_frames": len(admitted) - len(selected),
                "comparison": compare_confirmation(selected) if selected else None,
            }
        )
    return {
        "semantics": "overlapping_elapsed_receipt_cadence_counterfactuals",
        "interval_seconds": interval_seconds,
        "selection": "first_at_phase_then_at_least_interval_since_previous_selected_receipt",
        "source_frames": len(admitted),
        "phases": reports,
        "independent_trials": False,
        "measured_live_auxiliary_cadence": False,
        "source_pts_continuity_verified": False,
        "accuracy_verified": False,
        "deployment_approved": False,
    }
