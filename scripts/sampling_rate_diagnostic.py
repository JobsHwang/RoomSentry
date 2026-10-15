"""Fixed-rate local experiment plans, not actual detector-call evidence.

Existing PTS and opportunity helpers remain unchanged and their report hashes
are retained. A selected slot is an available reference frame planned for model
execution, not proof that inference happened, a person was seen, or gaps closed.
"""

from __future__ import annotations

import hashlib
import json
import math

from scripts.pts_sampling_diagnostic import plan_pts_sampling
from scripts.sampling_opportunity_diagnostic import evaluate_sampling_opportunity

RATES = (("rate_0_5", 0.5), ("rate_1", 1.0), ("rate_2", 2.0), ("rate_4", 4.0))
EVENT_DURATIONS = (0.25, 0.5, 1.0, 2.0, 5.0)
MAX_UNION_SLOTS = 256


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def plan_sampling_rates(
    reference_pts: tuple,
    *,
    native_fps: float,
    source_duration_seconds: float,
) -> dict[str, object]:
    """Plan four fixed absolute-PTS rates on identical supplied reference frames.

    Reference validity and scheduling are delegated to plan_pts_sampling, without
    changing its protocol. Each retained selection.actual_sample_count
    means the number of selected reference frames, NOT actual model invocations.
    The unique union allows one future inference per selected source frame; this
    function never runs that inference or treats overlapping schedules as repeats.

    D must exceed the fixed longest hypothetical window (5 s). Opportunity is
    measured on common possible-start domains, not empirical event frequencies.
    Plans exceeding 256 unique slots fail before any experiment should execute.
    """
    if isinstance(source_duration_seconds, bool) or not isinstance(
        source_duration_seconds, (int, float)
    ):
        raise ValueError("sampling rate diagnostic source duration is invalid")
    try:
        duration = float(source_duration_seconds)
    except (OverflowError, ValueError):
        raise ValueError("sampling rate diagnostic source duration is invalid") from None
    if not math.isfinite(duration) or not 5.0 < duration <= 86400.0:
        raise ValueError("sampling rate diagnostic source duration must exceed five seconds")

    branches = {}
    selections = {}
    union = set()
    first_plan = None
    for name, rate in RATES:
        plan = plan_pts_sampling(reference_pts, native_fps=native_fps, sample_fps=rate)
        if first_plan is None:
            first_plan = plan
        selection = plan["source_pts_deadline"]
        indices = tuple(selection["decoded_frame_indices"])
        selections[name] = indices
        union.update(indices)
        if len(union) > MAX_UNION_SLOTS:
            raise ValueError("sampling rate diagnostic union exceeds 256 planned slots")
        branches[name] = {
            "sample_fps": rate,
            "planned_slot_count": len(indices),
            "pts_plan_sha256": plan["report_sha256"],
            "selection": selection,
        }
    opportunity = evaluate_sampling_opportunity(
        reference_pts,
        selections,
        source_duration_seconds=duration,
        event_durations_seconds=EVENT_DURATIONS,
    )
    union_indices = sorted(union)
    inputs = {
        "reference_pts_seconds": tuple(float(value) for value in reference_pts),
        "native_fps": first_plan["native_fps"],
        "source_duration_seconds": duration,
        "rates_fps": [rate for _, rate in RATES],
        "hypothetical_event_durations_seconds": list(EVENT_DURATIONS),
    }
    report = {
        "schema_version": "sampling-rate-diagnostic-v1",
        "input_sha256": _hash(inputs),
        "reference_pts_sha256": first_plan["reference_pts_sha256"],
        "native_fps": first_plan["native_fps"],
        "source_duration_seconds": duration,
        "rates_fps": inputs["rates_fps"],
        "hypothetical_event_durations_seconds": list(EVENT_DURATIONS),
        "branches": branches,
        "union_decoded_frame_indices": union_indices,
        "unique_union_pts_seconds": [float(reference_pts[index]) for index in union_indices],
        "unique_union_planned_slot_count": len(union_indices),
        "opportunity": opportunity,
        "planned_slots_only": True,
        "actual_model_calls_verified": False,
        "offline_diagnostic_only": True,
        "empirical_event_probability_evaluated": False,
        "accuracy_evaluated": False,
        "event_recall_evaluated": False,
        "ground_truth": False,
        "source_exhaustion_verified": False,
        "deployment_approved": False,
        "equal_stride_ledger_claimed": False,
    }
    return {**report, "report_sha256": _hash(report)}
