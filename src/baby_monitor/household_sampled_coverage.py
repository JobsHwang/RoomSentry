"""Fixed human intervals measured against actual detector invocations.

This is raw person-detection coverage, not candidate-group coverage, identity,
localization, continuous observation or danger recognition. A timeline binding
is an explicit caller assertion tied to reference bytes, not proof that the
reference establishes the asserted origin. Never infer it from the first PTS.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from typing import Any

from .detector_protocol import AdmittedDetectorScan
from .household_ground_truth import (
    HouseholdEpisodeAnnotation,
    HouseholdGroundTruthError,
    _canonical_sha256,
)
from .source_admission import ExternalSourceAdmission

_BINDING_FIELDS = {
    "version",
    "annotation_sha256",
    "media_admission_sha256",
    "reference_sha256",
    "source_pts_at_annotation_zero_seconds",
    "alignment_confirmed",
}


def _number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HouseholdGroundTruthError(f"sampled coverage {field} is invalid")
    try:
        converted = float(value)
        valid = math.isfinite(converted) and minimum <= converted <= maximum
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise HouseholdGroundTruthError(f"sampled coverage {field} is invalid")
    return converted


def evaluate_household_sampled_coverage(
    annotation: HouseholdEpisodeAnnotation,
    detector_scan: AdmittedDetectorScan,
    *,
    media_admission: ExternalSourceAdmission,
    timeline_binding: dict[str, object],
    reference_sha256: str,
    confidence_threshold: float,
) -> dict[str, object]:
    """Keep every fixed interval, including those with no known timed calls.

    Reference bytes are authenticated by the CLI; the pure API only checks their
    caller-supplied digest. The origin is an explicit assertion, never inferred
    or approved here. Fallback clocks remain unknown and cannot enter a range.
    Duplicate known PTS are separate invocations, not deduplicated evidence.
    """
    if not isinstance(annotation, HouseholdEpisodeAnnotation):
        raise HouseholdGroundTruthError("sampled coverage requires a household annotation")
    if annotation.review_state != "approved":
        raise HouseholdGroundTruthError("sampled coverage requires approved human intervals")
    if not isinstance(detector_scan, AdmittedDetectorScan):
        raise HouseholdGroundTruthError("sampled coverage requires an admitted detector scan")
    if (
        not isinstance(media_admission, ExternalSourceAdmission)
        or media_admission.asset_kind != "media_asset"
        or media_admission.admission.status != "verified_local"
    ):
        raise HouseholdGroundTruthError("sampled coverage requires verified local media")
    if (
        media_admission.fingerprint != annotation.media_admission_sha256
        or detector_scan.source_admission_sha256 != annotation.media_admission_sha256
    ):
        raise HouseholdGroundTruthError("sampled coverage inputs do not bind to the same media")
    if detector_scan.result.camera_id != annotation.camera_id:
        raise HouseholdGroundTruthError("sampled coverage camera does not match the annotation")
    if "person" not in detector_scan.labels:
        raise HouseholdGroundTruthError("sampled coverage requires a person scan")
    ledger = detector_scan.result.sampling
    if detector_scan.version != 2 or ledger is None:
        raise HouseholdGroundTruthError("sampled coverage requires a v2 sampling ledger")
    threshold = _number(
        confidence_threshold,
        "confidence_threshold",
        minimum=detector_scan.confidence_threshold,
        maximum=1,
    )
    if not isinstance(timeline_binding, dict) or set(timeline_binding) != _BINDING_FIELDS:
        raise HouseholdGroundTruthError("sampled coverage timeline binding fields are invalid")
    if (
        type(timeline_binding["version"]) is not int
        or timeline_binding["version"] != 1
        or timeline_binding["alignment_confirmed"] is not True
    ):
        raise HouseholdGroundTruthError("sampled coverage requires an explicit timeline assertion")
    if (
        timeline_binding["annotation_sha256"] != annotation.fingerprint
        or timeline_binding["media_admission_sha256"] != media_admission.fingerprint
        or not isinstance(reference_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", reference_sha256) is None
        or timeline_binding["reference_sha256"] != reference_sha256
    ):
        raise HouseholdGroundTruthError("sampled coverage timeline binding does not match inputs")
    origin = _number(
        timeline_binding["source_pts_at_annotation_zero_seconds"],
        "source_pts_at_annotation_zero_seconds",
        minimum=0,
        maximum=86_400,
    )
    starts = tuple(origin + item.start_seconds for item in annotation.ranges)
    ends = tuple(origin + item.end_seconds for item in annotation.ranges)
    source_end = origin + annotation.source_duration_seconds
    if source_end <= origin or any(end <= start for start, end in zip(starts, ends, strict=True)):
        raise HouseholdGroundTruthError("sampled coverage cannot represent shifted intervals")
    counts = [0] * len(starts)
    detected_counts = [0] * len(starts)
    first_detection: list[float | None] = [None] * len(starts)
    unknown_clock = outside = unannotated = 0
    zero_boxes = 0
    for sample in ledger.samples:
        zero_boxes += sample.detection_count == 0
        if sample.timestamp_source != "container_pts":
            unknown_clock += 1
            continue
        pts = sample.timestamp_seconds
        if pts < origin or pts >= source_end:
            outside += 1
            continue
        index = bisect_right(starts, pts) - 1
        if index < 0 or pts >= ends[index]:
            unannotated += 1
            continue
        counts[index] += 1
        detected = any(
            detection.label == "person" and detection.confidence >= threshold
            for detection in detector_scan.result.detections[
                sample.detection_start : sample.detection_start + sample.detection_count
            ]
        )
        if detected:
            detected_counts[index] += 1
            if first_detection[index] is None:
                first_detection[index] = pts

    intervals: list[dict[str, object]] = []
    positive = negative = positive_samples = negative_samples = 0
    detected_positive = observed_miss = unobserved_positive = 0
    activated_negative = unobserved_negative = positive_hits = negative_hits = 0
    for index, item in enumerate(annotation.ranges):
        count = counts[index]
        hits = detected_counts[index]
        state = (
            "person_detected"
            if hits
            else "observed_without_person_detection"
            if count
            else "unobserved"
        )
        intervals.append(
            {
                "range_index": index,
                "kind": item.kind,
                "start_seconds": item.start_seconds,
                "end_seconds": item.end_seconds,
                "actual_sample_count": count,
                "person_detection_sample_count": hits,
                "state": state,
                "first_person_detection_delay_seconds": (
                    None
                    if first_detection[index] is None
                    else first_detection[index] - starts[index]
                ),
            }
        )
        if item.kind == "empty_room":
            negative += 1
            negative_samples += count
            negative_hits += hits
            activated_negative += hits > 0
            unobserved_negative += count == 0
        else:
            positive += 1
            positive_samples += count
            positive_hits += hits
            detected_positive += hits > 0
            observed_miss += count > 0 and hits == 0
            unobserved_positive += count == 0

    report: dict[str, object] = {
        "version": 1,
        "measurement": "actual_pts_raw_person_interval_coverage",
        "annotation_sha256": annotation.fingerprint,
        "detector_scan_sha256": detector_scan.fingerprint,
        "media_admission_sha256": media_admission.fingerprint,
        "timeline_binding_sha256": _canonical_sha256(timeline_binding),
        "reference_sha256": reference_sha256,
        "alignment_asserted_by_input": True,
        "reference_semantics_verified": False,
        "source_pts_at_annotation_zero_seconds": origin,
        "camera_id": annotation.camera_id,
        "confidence_threshold": threshold,
        "actual_sample_count": len(ledger.samples),
        "zero_box_sample_count": zero_boxes,
        "unknown_clock_sample_count": unknown_clock,
        "outside_source_sample_count": outside,
        "unannotated_sample_count": unannotated,
        "positive_sample_count": positive_samples,
        "negative_sample_count": negative_samples,
        "detected_positive_sample_count": positive_hits,
        "activated_negative_sample_count": negative_hits,
        "positive_interval_count": positive,
        "detected_positive_interval_count": detected_positive,
        "observed_positive_without_detection_count": observed_miss,
        "unobserved_positive_interval_count": unobserved_positive,
        "positive_interval_detection_rate": None if positive == 0 else detected_positive / positive,
        "negative_interval_count": negative,
        "activated_negative_interval_count": activated_negative,
        "unobserved_negative_interval_count": unobserved_negative,
        "intervals": intervals,
        "human_ground_truth": True,
        "dataset_specific_only": True,
        "winner_selected": False,
        "candidate_grouping_evaluated": False,
        "source_exhaustion_verified": False,
        "baby_identity_evaluated": False,
        "box_localization_evaluated": False,
        "independent_person_coverage_evaluated": False,
        "danger_recognition_evaluated": False,
        "deployment_approved": False,
        "source_media_included": False,
        "paths_disclosed": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report
