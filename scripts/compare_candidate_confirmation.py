"""Metadata-only, fixed-gate offline comparison; no live policy is changed.

Whole-frame baseline (.35 boxes, .4 frame gate), a per-box-gate control (.4),
and experimental per-candidate confirmation (.4) use the same ordered frames.
Receipt timestamps may be replayed as synthetic completion clocks only when
explicitly declared by the caller; this helper authenticates neither clock nor
media binding and produces no detector accuracy or presence assertion.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from types import SimpleNamespace

from baby_monitor.candidate_confirmation import (
    CandidateConfirmationPolicy,
    ConfirmationFrame,
    ConfirmationSettings,
)
from baby_monitor.domain import Detection
from baby_monitor.live_detection import AlternateViewReviewPolicy

POLICY_SPEC = {
    "raw_candidate_confidence": 0.35,
    "review_confidence": 0.4,
    "minimum_confirmation_seconds": 1.0,
    "minimum_iou": 0.5,
    "maximum_gap_seconds": 3.0,
    "review_cooldown_seconds": 30.0,
    "association": "unique_mutual_iou_edge_only_no_prediction",
    "ambiguity": "reset_affected_chains_no_role_or_identity_propagation",
}
ARMS = ("whole_frame", "whole_frame_candidate_gate", "per_candidate")


def _checked_frames(frames: object) -> list[tuple[dict, ConfirmationFrame]]:
    if type(frames) is not list or not 1 <= len(frames) <= 5000:
        raise ValueError("comparison requires a bounded nonempty frame list")
    checked = []
    ids = set()
    for row in frames:
        if type(row) is not dict or set(row) != {
            "frame_id",
            "frame_sha256",
            "camera_id",
            "generation",
            "sequence",
            "received_monotonic",
            "completed_monotonic",
            "boxes",
        }:
            raise ValueError("invalid comparison frame fields")
        frame_id, digest = row["frame_id"], row["frame_sha256"]
        if (
            type(frame_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", frame_id) is None
            or frame_id in ids
            or type(digest) is not str
            or re.fullmatch(r"[a-f0-9]{64}", digest) is None
        ):
            raise ValueError("invalid or duplicate comparison frame binding")
        ids.add(frame_id)
        boxes = row["boxes"]
        if type(boxes) is not list or len(boxes) > 100:
            raise ValueError("comparison requires at most 100 raw boxes per frame")
        detections = []
        for box in boxes:
            if type(box) is not dict or set(box) != {"confidence", "bbox_xyxy"}:
                raise ValueError("invalid comparison box fields")
            if type(box["bbox_xyxy"]) is not list:
                raise ValueError("comparison xyxy must be a list")
            detections.append(Detection(0, "person", box["confidence"], tuple(box["bbox_xyxy"])))
        checked.append(
            (
                row,
                ConfirmationFrame(
                    row["camera_id"],
                    row["generation"],
                    row["sequence"],
                    row["received_monotonic"],
                    row["completed_monotonic"],
                    tuple(detections),
                ),
            )
        )
    return checked


def compare_confirmation(frames: object) -> dict:
    """Validate the whole batch before replay, retaining all boxes and zero rows.

    The filtered whole-frame control isolates confidence filtering from changing
    the confirmation unit. No hard geometric support filter across views is used.
    Outputs are simulated local-review requests, NOT confirmed people or alarms.
    """
    checked = _checked_frames(frames)
    cameras = tuple(sorted({frame.camera_id for _, frame in checked}))
    candidate_policy = CandidateConfirmationPolicy(cameras, ConfirmationSettings())
    frame_policies = {
        name: AlternateViewReviewPolicy(
            minimum_frame_confidence=POLICY_SPEC["review_confidence"],
            minimum_confirmation_seconds=POLICY_SPEC["minimum_confirmation_seconds"],
        )
        for name in ARMS[:2]
    }
    metrics = {
        name: {"checked_frames": 0, "eligible_frames": 0, "request_frames": []} for name in ARMS
    }
    rows = []
    for row, frame in checked:
        outputs = {}
        for name, threshold in (
            ("whole_frame", POLICY_SPEC["raw_candidate_confidence"]),
            ("whole_frame_candidate_gate", POLICY_SPEC["review_confidence"]),
        ):
            detections = tuple(d for d in frame.detections if d.confidence >= threshold)
            # Deliberate metadata proxy, never constructed as an image-bearing LiveFrame.
            observation = SimpleNamespace(
                frame=frame, detections=detections, completed_monotonic=frame.completed_monotonic
            )
            policy = frame_policies[name]
            requested = policy.observe(observation) is not None
            metrics[name]["checked_frames"] += 1
            metrics[name]["eligible_frames"] += policy.frame_eligible(detections)
            if requested:
                metrics[name]["request_frames"].append(row["frame_id"])
            outputs[name] = {"requested": requested}
        result = candidate_policy.observe(frame)
        requested = bool(result.review_requested_indices)
        metrics["per_candidate"]["checked_frames"] += 1
        metrics["per_candidate"]["eligible_frames"] += any(
            d.confidence >= 0.4 for d in frame.detections
        )
        if requested:
            metrics["per_candidate"]["request_frames"].append(row["frame_id"])
        outputs["per_candidate"] = asdict(result)
        outputs["per_candidate"]["requested"] = requested
        rows.append({"input": row, "arms": outputs})
    return {
        "version": 1,
        "semantics": "offline_simulated_local_review_requests_not_person_facts",
        "policy_spec": dict(POLICY_SPEC),
        "metrics": metrics,
        "rows": rows,
        "helper_authenticates_media_or_timing": False,
        "real_inference_completion_timing": False,
        "person_truth_available": False,
        "accuracy_verified": False,
        "presence": "unknown",
        "baby_location": "unknown",
        "baby_state": "unknown",
        "deployment_approved": False,
    }


def synthetic_confirmation_cases() -> list[dict]:
    """Fixed, labelled mechanism stress cases, not camera/model benchmark data.

    expected_request asks whether THIS episode contains a permitted continuous
    true candidate lasting >=1s. Artifact-only/static-artefact scenes are negative;
    the latter deliberately demonstrates that persistence cannot reject them.
    Neither input boxes nor labels are produced by a model.
    """
    a = {"confidence": 0.8, "bbox_xyxy": [0.1, 0.1, 0.4, 0.6]}
    b = {"confidence": 0.8, "bbox_xyxy": [0.7, 0.1, 0.95, 0.6]}
    weak = {**a, "confidence": 0.385}
    times = [i / 5 for i in range(16)]
    specifications = [
        ("stable_true", True, times, [[a] for _ in times], {}),
        (
            "true_with_strong_flicker",
            True,
            times,
            [[a] if i % 2 else [a, b] for i in range(16)],
            {},
        ),
        (
            "true_with_weak_flicker",
            True,
            times,
            [[a] if i % 2 else [a, {**b, "confidence": 0.36}] for i in range(16)],
            {},
        ),
        (
            "smooth_true_grid_crossing",
            True,
            times,
            [[{**a, "bbox_xyxy": [0.1 + i * 0.02, 0.1, 0.4 + i * 0.02, 0.6]}] for i in range(16)],
            {},
        ),
        ("one_frame_artifact", False, times, [[a]] + [[] for _ in times[1:]], {}),
        ("two_frame_artifact", False, times, [[a], [a]] + [[] for _ in times[2:]], {}),
        (
            "alternating_disjoint_artifacts",
            False,
            times,
            [[a] if i % 2 else [b] for i in range(16)],
            {},
        ),
        (
            "same_cell_alternating_artifacts",
            False,
            times,
            [
                [{**a, "bbox_xyxy": [0.1, 0.1, 0.11, 0.11]}]
                if i % 2
                else [{**a, "bbox_xyxy": [0.13, 0.1, 0.14, 0.11]}]
                for i in range(16)
            ],
            {},
        ),
        ("weak_static_artifact", False, times, [[weak] for _ in times], {}),
        ("strong_static_artifact", False, times, [[a] for _ in times], {}),
        (
            "explicit_zero_gap",
            False,
            [0, 0.2, 0.4, 0.6, 1, 1.2, 1.4, 1.6],
            [[a], [a], [a], [], [a], [a], [a], [a]],
            {},
        ),
        (
            "generation_split",
            False,
            [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4],
            [[a] for _ in range(8)],
            {"generations": [0, 0, 0, 0, 1, 1, 1, 1]},
        ),
        ("receipt_clock_gap", False, [0, 0.2, 0.4, 4, 4.2, 4.4], [[a] for _ in range(6)], {}),
        (
            "completion_clock_gap",
            False,
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            [[a] for _ in range(6)],
            {"completions": [0, 0.2, 0.4, 4, 4.2, 4.4]},
        ),
        (
            "ambiguous_crossing",
            False,
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            [
                [a, b],
                [a, b],
                [a, b],
                [
                    {**a, "bbox_xyxy": [0.25, 0.1, 0.65, 0.6]},
                    {**b, "bbox_xyxy": [0.3, 0.1, 0.7, 0.6]},
                ],
                [a, b],
                [a, b],
            ],
            {},
        ),
    ]
    cases = []
    for name, expected, receipts, boxes, extra in specifications:
        frames = []
        for i, (received, candidates) in enumerate(zip(receipts, boxes, strict=True)):
            frame_id = f"{name}_{i}"
            frames.append(
                {
                    "frame_id": frame_id,
                    # A synthetic identifier, NOT a checksum of camera pixels.
                    "frame_sha256": hashlib.sha256(frame_id.encode()).hexdigest(),
                    "camera_id": "synthetic_camera",
                    "generation": extra.get("generations", [0] * len(receipts))[i],
                    "sequence": i,
                    "received_monotonic": received,
                    "completed_monotonic": extra.get("completions", receipts)[i],
                    "boxes": candidates,
                }
            )
        cases.append(
            {
                "case_id": name,
                "expected_request": expected,
                "frames": frames,
                "synthetic_inputs_and_labels": True,
            }
        )
    return cases


def compare_synthetic_cases() -> dict:
    cases = []
    totals = {name: dict(tp=0, fn=0, fp=0, tn=0) for name in ARMS}
    for case in synthetic_confirmation_cases():
        report = compare_confirmation(case["frames"])
        expected = case["expected_request"]
        for name in ARMS:
            requested = bool(report["metrics"][name]["request_frames"])
            key = ("tp" if requested else "fn") if expected else ("fp" if requested else "tn")
            totals[name][key] += 1
        cases.append(
            {"case_id": case["case_id"], "expected_request": expected, "comparison": report}
        )
    return {
        "semantics": "synthetic_review_eligibility_episode_tests_not_detector_accuracy",
        "cases": cases,
        "totals": totals,
        "household_accuracy_verified": False,
    }
