"""Exact temporal opportunity for hypothetical windows, never event recall.

No labels, boxes, media or inference. The measure is over possible event START
positions, not observed video duration or an empirical event distribution.
"""

from __future__ import annotations

import math
import re

from scripts.pts_sampling_diagnostic import MAX_REFERENCE_FRAMES, _hash, _number


def _opportunity_measure(
    timestamps: tuple[float, ...],
    *,
    event_duration: float,
    start_domain_end: float,
) -> float:
    # A half-open window [s, s+d) contains t iff s is in (t-d, t].
    # Open/closed endpoints have zero Lebesgue measure. No discrete test-grid
    # is used, so a narrow unsampled window cannot disappear between test bins.
    lengths = []
    left = right = None
    for timestamp in timestamps:
        low = max(0.0, timestamp - event_duration)
        high = min(start_domain_end, timestamp)
        if low >= high:
            continue
        if right is None:
            left, right = low, high
        elif low <= right:
            right = max(right, high)
        else:
            lengths.append(right - left)
            left, right = low, high
    if right is not None:
        lengths.append(right - left)
    # Guard only floating-point summation roundoff; all input indices/times
    # were already validated. This is not inferred continuous observation.
    measure = math.fsum(lengths)
    if not 0 <= measure <= start_domain_end + 1e-8:
        raise ValueError("sampling opportunity measure exceeds its start domain")
    return min(start_domain_end, measure)


def evaluate_sampling_opportunity(
    reference_pts: tuple[float, ...],
    selections: dict[str, tuple[int, ...]],
    *,
    source_duration_seconds: float,
    event_durations_seconds: tuple[float, ...],
) -> dict[str, object]:
    """Compare every possible start position using an identical denominator.

    For duration d, the hypothetical event fits wholly in [0,D): starts range
    from 0 through D-d. Measure union(t-d,t] clipped to that start domain.
    Using every reference frame is a temporal upper bound over this supplied
    decoded reference, not evidence that any source is complete or any person
    would be recognized. No external event-frequency prior is assumed true.
    """
    duration = _number(source_duration_seconds, minimum=0.001, maximum=86400)
    if not isinstance(reference_pts, tuple) or not 1 <= len(reference_pts) <= MAX_REFERENCE_FRAMES:
        raise ValueError("sampling opportunity reference is invalid")
    reference = tuple(_number(t, minimum=0, maximum=duration) for t in reference_pts)
    if reference[-1] >= duration or any(
        right <= left for left, right in zip(reference, reference[1:], strict=False)
    ):
        raise ValueError("sampling opportunity requires increasing PTS inside the source")
    if (
        not isinstance(event_durations_seconds, tuple)
        or not 1 <= len(event_durations_seconds) <= 16
    ):
        raise ValueError("sampling opportunity event durations are invalid")
    durations = tuple(_number(d, minimum=0.001, maximum=duration) for d in event_durations_seconds)
    if tuple(sorted(set(durations))) != durations or durations[-1] >= duration:
        raise ValueError(
            "sampling opportunity durations must increase and be shorter than the source"
        )
    if not isinstance(selections, dict) or not 1 <= len(selections) <= 8:
        raise ValueError("sampling opportunity selections are invalid")
    for name, indices in selections.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) is None
            or not isinstance(indices, tuple)
            or len(indices) > len(reference)
            or any(type(index) is not int or not 0 <= index < len(reference) for index in indices)
            or tuple(sorted(set(indices))) != indices
        ):
            raise ValueError("sampling opportunity selection must be a bounded unique index tuple")
    inputs = {
        "reference_pts_seconds": reference,
        "selections": selections,
        "source_duration_seconds": duration,
        "event_durations_seconds": durations,
    }
    rows = []
    for length in durations:
        domain = duration - length
        source_measure = _opportunity_measure(
            reference, event_duration=length, start_domain_end=domain
        )
        branches = {}
        for name, indices in sorted(selections.items()):
            measure = _opportunity_measure(
                tuple(reference[index] for index in indices),
                event_duration=length,
                start_domain_end=domain,
            )
            if measure > source_measure + 1e-8:
                raise ValueError("sampling opportunity exceeds supplied reference opportunity")
            branches[name] = {
                "sample_count": len(indices),
                "covered_start_position_measure_seconds": measure,
                "uncovered_start_position_measure_seconds": max(0.0, domain - measure),
                "uniform_start_position_opportunity_fraction": measure / domain,
                "extra_unsampled_start_position_measure_seconds": max(
                    0.0, source_measure - measure
                ),
                "extra_unsampled_start_position_fraction": max(0.0, source_measure - measure)
                / domain,
            }
        rows.append(
            {
                "hypothetical_event_duration_seconds": length,
                "possible_start_position_measure_seconds": domain,
                "reference_frame_opportunity_measure_seconds": source_measure,
                "no_reference_frame_start_position_measure_seconds": max(
                    0.0, domain - source_measure
                ),
                "reference_uniform_start_position_opportunity_fraction": source_measure / domain,
                "branches": branches,
            }
        )
    report = {
        "schema_version": "sampling-opportunity-diagnostic-v1",
        "input_sha256": _hash(inputs),
        "reference_pts_sha256": _hash(reference),
        "source_duration_seconds": duration,
        "reference_frame_count": len(reference),
        "reference_first_pts_seconds": reference[0],
        "reference_last_pts_seconds": reference[-1],
        "equal_selected_sample_counts": len({len(indices) for indices in selections.values()}) == 1,
        "window_convention": "half_open_start_inclusive_end_exclusive",
        "start_domain": "all_starts_from_zero_through_source_duration_minus_window_duration",
        "measure_unit": "seconds_of_possible_start_positions_not_observation_duration",
        "uniform_start_position_assumption_is_not_empirical_event_distribution": True,
        "endpoint_only_opportunities_have_zero_measure": True,
        "fraction_one_does_not_guarantee_every_start": True,
        "timed_sample_is_only_a_necessary_not_sufficient_detection_opportunity": True,
        "occlusion_model_errors_queueing_and_delivery_not_evaluated": True,
        "floating_point_measure_tolerance_seconds": 1e-8,
        "reference_is_decoded_frames_not_complete_source": True,
        "rows": rows,
        "offline_diagnostic_only": True,
        "model_calls_verified_by_helper": False,
        "media_bytes_verified_by_helper": False,
        "actual_event_labels_used": False,
        "ground_truth": False,
        "accuracy_evaluated": False,
        "event_recall_evaluated": False,
        "source_exhaustion_verified": False,
        "deployment_approved": False,
    }
    return {**report, "report_sha256": _hash(report)}
