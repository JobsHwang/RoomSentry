"""Join frame-bound head references to frozen detector, Bonsai and tracking results.

Standard library only. Reports visible-head correspondence in development
examples; they neither establish whole-person box accuracy nor grant live roles.
No image reads, model execution, camera capture or production changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from baby_monitor.head_point_review import (
    diagnose_point_coverage,
    fingerprint,
    validate_manifest,
    validate_review,
)
from scripts.household_debug_common import new_output_directory, write_new
from scripts.prepare_head_point_review import read_blob
from scripts.subject_layer_diagnostic import _origin_path, diagnose

REVISION = "reviewed_subject_layers_v1"


def _require(condition):
    if not condition:
        raise ValueError("subject layer reference or binding rejected")


def _radius(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and 1 <= value <= 64
    except (ValueError, OverflowError):
        valid = False
    _require(valid)
    return float(value)


def bind_records(manifest, records):
    """Bind all requested frames, preserving explicit zero boxes and full replay."""
    manifest = validate_manifest(manifest)
    _require(type(records) is list and 1 <= len(records) <= 48)
    _require(all(type(r) is dict and type(r.get("subject_result")) is dict for r in records))
    bound, reports = [], []
    for item in manifest["items"]:
        candidates = [
            r
            for r in records
            if type(r) is dict
            and r.get("subject_result", {}).get("input_sha256") == item["frame_sha256"]
        ]
        _require(len(candidates) == 1)
        record = candidates[0]
        report = diagnose(record)
        _require(record["camera_id"] == item["camera_id"])
        _require(record["people_trace"]["frames"][0]["received_at"] == item["frame_received_at"])
        bound.append(record)
        reports.append(report)
    return bound, reports


def _model_links(item, answer, people, radius):
    # Use image pixels, not normalized distance: rectangular images have
    # different x/y scales. Never force a nearest match in a crowd.
    links = []
    distances = []
    for point in answer["points"]:
        row = [
            None
            if person["head_center"] is None
            else math.hypot(
                (point["point_xy"][0] - person["head_center"][0] / 1000) * item["width"],
                (point["point_xy"][1] - person["head_center"][1] / 1000) * item["height"],
            )
            for person in people
        ]
        distances.append(row)
        links.append([i for i, d in enumerate(row) if d is not None and d <= radius])
    degrees = [sum(i in row for row in links) for i in range(len(people))]
    return links, degrees, distances


def evaluate(manifest, review, records, *, maximum_head_distance_pixels):
    """Evaluate only references supplied for the exact saved image.

    A one-to-one head match permits an age comparison. Unknown ages and
    ambiguous/unlocated heads stay unresolved. A missing head match is not a
    false-positive person; head references cannot describe hidden people.
    Tracking describes the source box's anonymous path, not temporal identity.
    """
    radius = _radius(maximum_head_distance_pixels)
    manifest = validate_manifest(manifest)
    review = validate_review(review, manifest)
    bound, reports = bind_records(manifest, records)
    predictions = [
        {
            "item_id": item["item_id"],
            "frame_sha256": item["frame_sha256"],
            "boxes": [d["bbox_xyxy"] for d in record["people_trace"]["frames"][0]["detections"]],
        }
        for item, record in zip(manifest["items"], bound, strict=True)
    ]
    coverage = diagnose_point_coverage(manifest, review, predictions)
    counts = Counter()
    rows = []
    for item, answer, record, report, detector in zip(
        manifest["items"], review["answers"], bound, reports, coverage["rows"], strict=True
    ):
        frames = record["people_trace"]["frames"]
        first = frames[0]["detections"]
        people = record["subject_result"]["claims"]["people"]
        links, degrees, distances = _model_links(item, answer, people, radius)
        counts["reviewed_frames"] += 1
        counts["unknown_frames"] += answer["decision"] == "unknown"
        counts["complete_visible_head_inventories"] += answer["inventory_complete"]
        details = []
        for index, (point, detected, options) in enumerate(
            zip(answer["points"], detector["points"], links, strict=True)
        ):
            count_name = "reference_" + point["role"] + "_heads"
            counts[count_name] += 1
            counts["reference_heads"] += 1
            counts["detector_" + detected["state"]] += 1
            counts[point["role"] + "_detector_" + detected["state"]] += 1
            box_options = [
                i
                for i, d in enumerate(first)
                if d["bbox_xyxy"][0] <= point["point_xy"][0] <= d["bbox_xyxy"][2]
                and d["bbox_xyxy"][1] <= point["point_xy"][1] <= d["bbox_xyxy"][3]
            ]
            detector_reason = (
                "no_box_at_reference_head"
                if not box_options
                else "overlapping_boxes_at_reference_head"
                if len(box_options) > 1
                else "multiple_reference_heads_in_one_box"
                if detected["state"] == "ambiguous_point_containment"
                else "unique_reference_head_box"
            )
            counts["detector_" + detector_reason] += 1
            path = (
                _origin_path(first[box_options[0]], frames)
                if detected["state"] == "unique_point_containment"
                else None
            )
            if path is not None:
                counts["tracking_unique_source_boxes"] += 1
                counts["tracking_source_boxes_with_gap"] += path["first_missing"] is not None
                counts["tracking_source_boxes_with_epoch_change"] += (
                    path["first_epoch_change"] is not None
                )
            model_index = options[0] if len(options) == 1 and degrees[options[0]] == 1 else None
            correspondence = (
                "no_nearby_located_claim"
                if not options
                else "ambiguous_head_correspondence"
                if model_index is None
                else "unique_head_correspondence"
            )
            role = None if model_index is None else people[model_index]["age_group"]
            comparison = (
                "unresolved_correspondence"
                if model_index is None
                else "reference_age_unknown"
                if point["role"] == "unknown"
                else "model_age_unknown"
                if role == "unknown"
                else "agrees_with_reference"
                if role == point["role"]
                else "disagrees_with_reference"
            )
            counts["bonsai_" + correspondence] += 1
            counts["bonsai_age_" + comparison] += 1
            counts[point["role"] + "_bonsai_age_" + comparison] += 1
            details.append(
                {
                    "point_id": point["point_id"],
                    "reference_role": point["role"],
                    "detector_state": detected["state"],
                    "detector_reason": detector_reason,
                    "detector_box_indices": box_options,
                    "bonsai_correspondence": correspondence,
                    "bonsai_claim_index": model_index,
                    "bonsai_head_distance_pixels": None
                    if model_index is None
                    else distances[index][model_index],
                    "bonsai_role": role,
                    "bonsai_age_comparison": comparison,
                    "anonymous_source_box_path": path,
                }
            )
        rows.append(
            {
                "item_id": item["item_id"],
                "camera_id": item["camera_id"],
                "frame_sha256": item["frame_sha256"],
                "source_record_sha256": record["record_sha256"],
                "record_projection_sha256": report["record_projection_sha256"],
                "decision": answer["decision"],
                "inventory_complete_declared": answer["inventory_complete"],
                "detector_box_count": len(first),
                "model_claim_count": len(people),
                "model_claims_without_located_head": sum(p["head_center"] is None for p in people),
                "points": details,
                "tracking_input_count": len(frames),
                "actual_assignments_and_audits_reproduced": True,
                "review_duration_ms": report["bonsai"]["review_duration_ms"],
            }
        )
    # Keep absent categories explicit; absence of observations is not an
    # unreported denominator or an invented zero false-positive rate.
    for name in (
        "reference_heads",
        "reference_baby_heads",
        "reference_adult_heads",
        "reference_unknown_heads",
        "reviewed_frames",
        "unknown_frames",
        "complete_visible_head_inventories",
        "detector_point_not_covered",
        "detector_unique_point_containment",
        "detector_ambiguous_point_containment",
        "tracking_unique_source_boxes",
        "tracking_source_boxes_with_gap",
        "tracking_source_boxes_with_epoch_change",
        "bonsai_unique_head_correspondence",
        "bonsai_no_nearby_located_claim",
        "bonsai_ambiguous_head_correspondence",
        "bonsai_age_agrees_with_reference",
        "bonsai_age_disagrees_with_reference",
        "bonsai_age_model_age_unknown",
        "bonsai_age_reference_age_unknown",
        "bonsai_age_unresolved_correspondence",
    ):
        counts.setdefault(name, 0)
    return {
        "revision": REVISION,
        "manifest_sha256": fingerprint(manifest),
        "review_sha256": fingerprint(review),
        "records_sha256": fingerprint(bound),
        "maximum_head_distance_pixels": radius,
        "reference_provenance_declared": review["provenance"],
        "reference_authenticated_by_this_function": False,
        "counts": dict(counts),
        "rows": rows,
        "measurement": "development_visible_head_and_age_correspondence",
        "formal_person_box_precision": None,
        "formal_person_box_recall": None,
        "temporal_id_switches": None,
        "posture_reference_available": False,
        "sleep_established": False,
        "epoch_values_are_exported_not_independently_replayed": True,
        "independent_holdout": False,
        "pixels_verified_by_this_function": False,
        "deployment_approved": False,
        "identity_or_live_state_authorized": False,
        "new_model_calls": 0,
    }


def _read_json(path, limit):
    raw = read_blob(path, limit=limit)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result)
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("non-finite layer input rejected")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        _require(depth <= 32)
        if type(item) is dict:
            pending.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            pending.extend((v, depth + 1) for v in item)
    return value, hashlib.sha256(raw).hexdigest()


def run(cohort, output, *, answers=None, maximum_head_distance_pixels):
    radius = _radius(maximum_head_distance_pixels)
    repository = Path(__file__).resolve().parents[1]
    implementations = (
        "scripts/evaluate_subject_layers.py",
        "scripts/subject_layer_diagnostic.py",
        "src/baby_monitor/head_point_review.py",
        "src/baby_monitor/motion_tracker.py",
        "src/baby_monitor/live_people_association.py",
        "src/baby_monitor/tracker_evaluation/association_checkpoint.py",
        "src/baby_monitor/local_runners/subject_state_probe.py",
    )
    code_sha = {
        name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
        for name in implementations
    }
    manifest, manifest_file_sha = _read_json(cohort / "blind-review" / "manifest.json", 128 * 1024)
    source, source_file_sha = _read_json(cohort / "source-records.json", 8 * 1024 * 1024)
    _require(type(source) is dict and set(source) == {"records"})
    records, _reports = bind_records(manifest, source["records"])
    package, _package_file_sha = _read_json(cohort / "blind-review" / "package.json", 128 * 1024)
    _require(type(package) is dict and type(package.get("synthetic")) is bool)
    _require(package.get("manifest_sha256") == fingerprint(manifest))
    _require(type(package.get("item_count")) is int and package["item_count"] == len(records))
    bindings, bindings_file_sha = _read_json(cohort / "image-bindings.json", 128 * 1024)
    _require(type(bindings) is dict and set(bindings) == {"items"})
    _require(type(bindings["items"]) is list and len(bindings["items"]) == len(records))
    for item, record, binding in zip(manifest["items"], records, bindings["items"], strict=True):
        _require(type(binding) is dict)
        _require(binding.get("item_id") == item["item_id"])
        _require(binding.get("source_record_sha256") == record["record_sha256"])
        _require(binding.get("png_sha256") == item["frame_sha256"])
        _require(binding.get("bgr_sha256") == record["raw_bgr_sha256"])
        _require(binding.get("dimensions") == [item["width"], item["height"]])
        _require(binding.get("exact_png_bgr_verified") is True)
    result = {
        "revision": REVISION,
        "status": "awaiting_independent_answers",
        "manifest_sha256": fingerprint(manifest),
        "source_records_sha256": fingerprint(source["records"]),
        "manifest_file_sha256": manifest_file_sha,
        "source_file_sha256": source_file_sha,
        "image_bindings_file_sha256": bindings_file_sha,
        "image_verification_is_prior_caller_declaration": True,
        "selected_frames": len(records),
        "complete_actual_matching_replays": len(records),
        "maximum_head_distance_pixels": radius,
        "implementation_sha256": code_sha,
        "new_model_calls": 0,
        "pixel_reads": 0,
        "independent_holdout": False,
        "metrics_available": False,
    }
    if answers is not None:
        review, answer_file_sha = _read_json(answers, 128 * 1024)
        _require(type(review) is dict)
        _require(
            review.get("provenance")
            == ("synthetic" if package["synthetic"] else "human_user_review")
        )
        result = {
            "status": "evaluated_supplied_references",
            "answer_file_sha256": answer_file_sha,
            "image_bindings_file_sha256": bindings_file_sha,
            "image_verification_is_prior_caller_declaration": True,
            "implementation_sha256": code_sha,
            **evaluate(manifest, review, records, maximum_head_distance_pixels=radius),
        }
    _require(
        all(
            hashlib.sha256((repository / name).read_bytes()).hexdigest() == digest
            for name, digest in code_sha.items()
        )
    )
    new_output_directory(output)
    output.mkdir(mode=0o700)
    write_new(output / "report.json", result)
    return {"status": result["status"], "selected_frames": len(records), "new_model_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-head-distance-pixels", required=True, type=float)
    args = parser.parse_args(argv)
    try:
        result = run(
            args.cohort,
            args.output,
            answers=args.answers,
            maximum_head_distance_pixels=args.maximum_head_distance_pixels,
        )
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError, RecursionError, OverflowError):
        print(json.dumps({"ok": False, "error_code": "subject_layer_acceptance_rejected"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
