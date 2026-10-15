"""Pure frame-index timeline diagnostics; no media, model, or file access.

The caller must establish media byte identity, admission provenance, and the
origin/units of reference timestamps. Comparing metadata does not establish that
all source content was decoded, or that people were correctly detected.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

from baby_monitor.detector_protocol import AdmittedDetectorScan, DetectorProtocolError
from baby_monitor.detector_sampling import DetectorSamplingError

MAX_REFERENCE_FRAMES = 200_000
TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS = 0.00001
MAX_MISMATCH_DETAILS = 16


def _timeline(value: object) -> tuple[float | None, ...]:
    if not isinstance(value, tuple) or not 1 <= len(value) <= MAX_REFERENCE_FRAMES:
        raise ValueError("timeline requires a bounded nonempty tuple")
    result = []
    for item in value:
        if item is None:
            result.append(None)
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("timeline timestamp must be finite nonnegative seconds or None")
        try:
            converted = float(item)
        except (OverflowError, ValueError):
            raise ValueError(
                "timeline timestamp must be finite nonnegative seconds or None"
            ) from None
        if not math.isfinite(converted) or converted < 0:
            raise ValueError("timeline timestamp must be finite nonnegative seconds or None")
        result.append(converted)
    return tuple(result)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def diagnose_detector_timeline(
    scan: AdmittedDetectorScan,
    reference_pts: tuple[float | None, ...],
    *,
    reference_source_admission_sha256: str,
) -> dict[str, object]:
    """Compare sampled PTS by decoded frame index without shifting either clock.

    The four sample categories partition the actual ledger samples. A nominal
    fallback is always ``fallback`` even if its frame index lacks a reference;
    ``missing_reference`` applies only to container-PTS samples. Matching uses
    absolute tolerance 1e-5 seconds, no relative tolerance. The all-match flag is
    independent of decoded-frame-count agreement and neither proves completeness.

    Reference adjacent gaps/regressions require consecutive frame indices with
    valid PTS; missing values break adjacency. Duplicate timestamps are retained.
    Reference fingerprints hash the compact JSON array after conversion of valid
    numeric timestamps to floats. No complete timeline is copied to the report.
    """
    if not isinstance(scan, AdmittedDetectorScan):
        raise ValueError("timeline diagnostic requires an admitted detector scan")
    try:
        original_scan_fingerprint = scan.fingerprint
        scan = AdmittedDetectorScan.from_dict(scan.to_dict())
    except (DetectorProtocolError, DetectorSamplingError, TypeError, ValueError, AttributeError):
        raise ValueError("timeline diagnostic scan is invalid") from None
    if scan.version != 2 or scan.result.sampling is None:
        raise ValueError("timeline diagnostic requires a v2 sampling ledger")
    if (
        not isinstance(reference_source_admission_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", reference_source_admission_sha256) is None
        or reference_source_admission_sha256 != scan.source_admission_sha256
    ):
        raise ValueError("timeline diagnostic source admission binding does not match")
    reference = _timeline(reference_pts)
    ledger = scan.result.sampling
    matched = mismatched = fallback = missing = 0
    mismatch_details = []
    for sample in ledger.samples:
        if sample.timestamp_source == "nominal_fps":
            fallback += 1
            continue
        index = sample.decoded_frame_index
        expected = reference[index] if index < len(reference) else None
        if expected is None:
            missing += 1
        elif abs(sample.timestamp_seconds - expected) <= TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS:
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_details) < MAX_MISMATCH_DETAILS:
                mismatch_details.append(
                    {
                        "decoded_frame_index": index,
                        "actual_timestamp_seconds": sample.timestamp_seconds,
                        "reference_timestamp_seconds": expected,
                    }
                )
    maximum_gap = None
    maximum_gap_indices = None
    regression_count = duplicate_count = adjacent_pair_count = 0
    for index in range(1, len(reference)):
        previous, current = reference[index - 1], reference[index]
        if previous is None or current is None:
            continue
        adjacent_pair_count += 1
        gap = current - previous
        if gap < 0:
            regression_count += 1
        else:
            duplicate_count += gap == 0
            if maximum_gap is None or gap > maximum_gap:
                maximum_gap = gap
                maximum_gap_indices = [index - 1, index]
    sampled_maximum_gap = max(
        (
            current.timestamp_seconds - previous.timestamp_seconds
            for previous, current in zip(ledger.samples, ledger.samples[1:], strict=False)
        ),
        default=None,
    )
    valid_reference = [value for value in reference if value is not None]
    return {
        "schema_version": "detector-timeline-diagnostic-v1",
        "offline_diagnostic_only": True,
        "source_admission_sha256": scan.source_admission_sha256,
        "admitted_scan_sha256": original_scan_fingerprint,
        "sampling_ledger_sha256": _fingerprint(ledger.to_dict()),
        "reference_timeline_sha256": _fingerprint(reference),
        "media_bytes_verified_by_helper": False,
        "reference_provenance_verified_by_helper": False,
        "timestamp_unit": "seconds",
        "alignment": "decoded_frame_index_without_timestamp_origin_shift",
        "absolute_tolerance_seconds": TIMESTAMP_ABSOLUTE_TOLERANCE_SECONDS,
        "relative_tolerance": 0.0,
        "reference_frame_count": len(reference),
        "decoded_frame_count": ledger.decoded_frame_count,
        "decoded_frame_count_matches_reference": ledger.decoded_frame_count == len(reference),
        "sampled_frame_count": len(ledger.samples),
        "matched_sample_count": matched,
        "mismatched_sample_count": mismatched,
        "fallback_sample_count": fallback,
        "missing_reference_sample_count": missing,
        "comparable_container_pts_sample_count": matched + mismatched,
        "all_sample_timestamps_match": matched == len(ledger.samples),
        "mismatch_details": mismatch_details,
        "mismatch_details_truncated": mismatched > MAX_MISMATCH_DETAILS,
        "reference_missing_pts_count": len(reference) - len(valid_reference),
        "reference_adjacent_valid_pair_count": adjacent_pair_count,
        "reference_pts_regression_count": regression_count,
        "reference_adjacent_duplicate_pts_count": duplicate_count,
        "reference_max_adjacent_nonnegative_pts_gap_seconds": maximum_gap,
        "reference_max_gap_first_frame_indices": maximum_gap_indices,
        "reference_first_valid_pts_seconds": valid_reference[0] if valid_reference else None,
        "reference_last_valid_pts_seconds": valid_reference[-1] if valid_reference else None,
        "sampled_max_adjacent_timestamp_gap_seconds": sampled_maximum_gap,
        "source_exhaustion_verified": False,
        "continuous_observation_seconds": None,
        "accuracy_verified": False,
        "deployment_authorized": False,
        "subject_identity_inference": False,
        "paths_disclosed": False,
    }


def diagnose_packet_timestamps(
    reference_pts: tuple[float | None, ...],
    packet_pts: tuple[float | None, ...],
) -> dict[str, object]:
    """Compare valid timestamp multisets, not packets-as-frames or missing time.

    Packets can be emitted in decode order with B-frame reordering. Presentation
    gaps therefore use sorted valid PTS, preserving multiplicity. Missing values
    are reported separately and do not become synthetic timestamps. Equal valid
    multisets do not establish a one-packet-per-frame relationship/completeness.
    """
    reference, packets = _timeline(reference_pts), _timeline(packet_pts)
    valid_reference = [value for value in reference if value is not None]
    presented = sorted(value for value in packets if value is not None)
    packet_counter = Counter(presented)
    maximum_gap = max(
        (current - previous for previous, current in zip(presented, presented[1:], strict=False)),
        default=None,
    )
    return {
        "schema_version": "packet-timeline-diagnostic-v1",
        "offline_diagnostic_only": True,
        "reference_timeline_sha256": _fingerprint(reference),
        "packet_timeline_sha256": _fingerprint(packets),
        "reference_frame_count": len(reference),
        "reference_missing_pts_count": len(reference) - len(valid_reference),
        "packet_count": len(packets),
        "missing_pts_count": len(packets) - len(presented),
        "valid_packet_pts_count": len(presented),
        "duplicate_present_pts_count": len(presented) - len(packet_counter),
        "max_presentation_pts_gap_seconds": maximum_gap,
        "valid_pts_multisets_match": Counter(valid_reference) == packet_counter,
        "presentation_order": "ascending_valid_packet_pts_with_multiplicity",
        "one_packet_per_frame_assumed": False,
        "media_bytes_verified_by_helper": False,
        "reference_provenance_verified_by_helper": False,
        "source_exhaustion_verified": False,
        "missing_content_duration_seconds": None,
        "accuracy_verified": False,
        "deployment_authorized": False,
        "subject_identity_inference": False,
        "paths_disclosed": False,
    }
