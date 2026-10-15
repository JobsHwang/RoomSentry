#!/usr/bin/env python3
"""Export fully approved household reviews as a strict VLM A/B dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from baby_monitor.ab_eval import load_dataset

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_CASE_KEYS = {
    "version",
    "review_status",
    "case_id",
    "frames",
    "question",
    "expected",
    "reviews",
}
_REVIEW_KEYS = {"reviewer_id", "verdict", "reviewed_at"}


def _strict_json(line: str, *, line_number: int) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key on line {line_number}")
            value[key] = item
        return value

    try:
        value = json.loads(
            line,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON number on line {line_number}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid review JSON on line {line_number}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"review case must be an object on line {line_number}")
    return value


def _reviewer_ids(value: object, *, line_number: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 5:
        raise ValueError(f"approved review list is invalid on line {line_number}")
    identifiers: list[str] = []
    for review in value:
        if not isinstance(review, dict) or set(review) != _REVIEW_KEYS:
            raise ValueError(f"review record is invalid on line {line_number}")
        reviewer_id = review["reviewer_id"]
        if (
            not isinstance(reviewer_id, str)
            or _SAFE_ID.fullmatch(reviewer_id) is None
            or review["verdict"] != "approve"
            or not isinstance(review["reviewed_at"], str)
            or not review["reviewed_at"]
        ):
            raise ValueError(f"review record is invalid on line {line_number}")
        identifiers.append(reviewer_id)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"reviewer IDs must be unique on line {line_number}")
    return tuple(identifiers)


def _is_safety_critical(expected: object, *, line_number: int) -> bool:
    if not isinstance(expected, dict):
        raise ValueError(f"approved expected labels are missing on line {line_number}")
    event_type = expected.get("event_type")
    risk_level = expected.get("risk_level")
    return event_type == "possible_risk" or risk_level in {"medium", "high"}


def export(review_path: Path, output_path: Path) -> tuple[int, int, int, str]:
    if review_path.is_symlink() or not review_path.is_file():
        raise ValueError("completed review must be a regular file, not a symlink")
    if not 0 < review_path.stat().st_size <= 10 * 1024 * 1024:
        raise ValueError("completed review size is outside the allowed range")
    review_root = review_path.resolve().parent
    if output_path.parent.resolve() != review_root:
        raise ValueError("exported dataset must stay beside its reviewed frames")
    if output_path.exists():
        raise ValueError("output file already exists; refusing to overwrite")
    try:
        lines = review_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("completed review must be readable UTF-8") from exc

    dataset_cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    safety_cases = 0
    all_reviewers: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        case = _strict_json(line, line_number=line_number)
        if set(case) != _CASE_KEYS or case["version"] != 1:
            raise ValueError(f"review fields are invalid on line {line_number}")
        case_id = case["case_id"]
        if (
            case["review_status"] != "approved"
            or not isinstance(case_id, str)
            or _SAFE_ID.fullmatch(case_id) is None
            or case_id in case_ids
        ):
            raise ValueError(f"every review case must be uniquely approved on line {line_number}")
        reviewers = _reviewer_ids(case["reviews"], line_number=line_number)
        if _is_safety_critical(case["expected"], line_number=line_number):
            safety_cases += 1
            if len(reviewers) < 2:
                raise ValueError(
                    f"safety-critical case requires two reviewers on line {line_number}"
                )
        all_reviewers.update(reviewers)
        dataset_cases.append(
            {
                "version": 2,
                "case_id": case_id,
                "frames": case["frames"],
                "question": case["question"],
                "expected": case["expected"],
            }
        )
        case_ids.add(case_id)
    if not 10 <= len(dataset_cases) <= 30:
        raise ValueError("completed household review must contain 10 to 30 cases")

    partial = output_path.with_name(f".{output_path.name}.partial-{os.getpid()}")
    if partial.exists():
        raise ValueError("private export partial already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(partial, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for case in dataset_cases:
                handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
        validated = load_dataset(partial)
        if len(validated) != len(dataset_cases):
            raise ValueError("exported dataset case count changed during validation")
        os.replace(partial, output_path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return len(dataset_cases), sum(len(case.frames) for case in validated), safety_cases, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    cases, frames, safety_cases, digest = export(args.review, args.output)
    print(
        json.dumps(
            {
                "version": 1,
                "cases": cases,
                "frames": frames,
                "safety_cases": safety_cases,
                "dataset_sha256": digest,
                "all_cases_approved": True,
                "hashes_verified": True,
                "ready_for_ab": True,
                "paths_disclosed": False,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
