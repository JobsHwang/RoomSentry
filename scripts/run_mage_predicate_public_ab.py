from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from baby_monitor.local_runners.mage_vl_4b import (
    OFFICIAL_REVISION,
    MageVLSettings,
    TrustedModelCodeManifest,
    isolated_transformers_module_cache,
)
from baby_monitor.local_runners.mage_vl_predicate import (
    ATOMIC_MODEL_ID,
    PROMPT_VERSION,
    MagePredicateFrame,
    MagePredicateRequest,
    MagePredicateSession,
)
from baby_monitor.visual_predicate import VisualPredicate


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _private_file(path: Path, *, root: Path, expected_hash: str) -> bytes:
    if path.is_symlink():
        raise RuntimeError("dataset frame path is unsafe")
    resolved = path.resolve(strict=True)
    if resolved.parent != root:
        raise RuntimeError("dataset frame path is unsafe")
    metadata = resolved.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > 25 * 1024 * 1024
        or (os.name == "posix" and metadata.st_uid != os.getuid())
        or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600)
    ):
        raise RuntimeError("dataset frame is unsafe")
    payload = resolved.read_bytes()
    if _sha256(payload) != expected_hash:
        raise RuntimeError("dataset frame hash mismatch")
    return payload


def _load_dataset(path: Path) -> tuple[Path, dict[str, Any], str]:
    if path.is_symlink():
        raise RuntimeError("dataset path is unsafe")
    root = path.parent.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved.parent != root or path.parent.is_symlink():
        raise RuntimeError("dataset root is unsafe")
    root_metadata = root.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or (os.name == "posix" and root_metadata.st_uid != os.getuid())
        or (os.name == "posix" and stat.S_IMODE(root_metadata.st_mode) != 0o700)
    ):
        raise RuntimeError("dataset root must be private")
    dataset_metadata = resolved.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(dataset_metadata.st_mode)
        or dataset_metadata.st_nlink != 1
        or not 0 < dataset_metadata.st_size <= 1024 * 1024
        or (os.name == "posix" and dataset_metadata.st_uid != os.getuid())
        or (os.name == "posix" and stat.S_IMODE(dataset_metadata.st_mode) != 0o600)
    ):
        raise RuntimeError("dataset path is unsafe")
    payload = resolved.read_bytes()
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or not isinstance(value.get("cases"), list)
        or value.get("source", {}).get("household_media") is not False
        or value.get("source", {}).get("uploads_performed") is not False
    ):
        raise RuntimeError("dataset contract is invalid")
    return root, value, _sha256(payload)


def _request(
    root: Path,
    case: dict[str, Any],
    variant_name: str,
) -> MagePredicateRequest:
    frames = []
    for value in case["variants"][variant_name]:
        expected = {"path", "sha256", "media_type", "bytes", "frame_id"}
        optional = {"detector_model_id", "detector_model_revision", "detector_confidence"}
        if not expected.issubset(value) or set(value) - expected - optional:
            raise RuntimeError("dataset frame fields are invalid")
        payload = _private_file(
            root / value["path"],
            root=root,
            expected_hash=value["sha256"],
        )
        if type(value["bytes"]) is not int or value["bytes"] != len(payload):
            raise RuntimeError("dataset frame byte count is invalid")
        frames.append(
            MagePredicateFrame(
                frame_id=value["frame_id"],
                sha256=value["sha256"],
                media_type=value["media_type"],
                payload=payload,
            )
        )
    return MagePredicateRequest(
        event_id=f"{case['case_id']}.{variant_name}",
        predicate=VisualPredicate(case["predicate"]),
        frames=tuple(frames),
        model_id=ATOMIC_MODEL_ID,
        model_revision=OFFICIAL_REVISION,
        prompt_version=PROMPT_VERSION,
    )


def _write_private(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", choices=("bf16", "int8"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-manifest", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.repetitions <= 5:
        raise RuntimeError("repetitions must be between 1 and 5")
    dataset_path = Path(args.dataset).expanduser().resolve(strict=True)
    root, dataset, dataset_sha256 = _load_dataset(dataset_path)
    output = Path(args.output).expanduser()
    if not output.is_absolute() or output.parent.resolve(strict=True) != root:
        raise RuntimeError("output must be a new file in the private dataset root")
    manifest = TrustedModelCodeManifest.read(args.code_manifest)
    settings = MageVLSettings(
        model_path=Path(args.model_path).expanduser(),
        code_manifest=manifest,
        confirm_reviewed_remote_code=True,
        language_linear_precision=args.precision,
        max_new_tokens=64,
    )
    results = []
    started = time.perf_counter()
    warmup_seconds = 0.0
    with (
        open(os.devnull, "w", encoding="utf-8") as output_sink,
        redirect_stdout(output_sink),
        redirect_stderr(output_sink),
        isolated_transformers_module_cache(),
        MagePredicateSession(settings) as session,
    ):
        loaded = time.perf_counter()
        warmup_request = _request(root, dataset["cases"][0], "full")
        warmup_started = time.perf_counter()
        session.assess(warmup_request)
        warmup_seconds = time.perf_counter() - warmup_started
        for repetition in range(args.repetitions):
            for case_index, case in enumerate(dataset["cases"]):
                variant_names = ("full", "full_yolo_crop")
                if (repetition + case_index) % 2:
                    variant_names = tuple(reversed(variant_names))
                for variant_name in variant_names:
                    request = _request(root, case, variant_name)
                    inference_started = time.perf_counter()
                    result = session.assess(request)
                    elapsed = time.perf_counter() - inference_started
                    results.append(
                        {
                            "case_id": case["case_id"],
                            "predicate": case["predicate"],
                            "variant": variant_name,
                            "repetition": repetition,
                            "expected_answer": case["expected_answer"],
                            "answer": result.answer.value,
                            "correct": result.answer.value == case["expected_answer"],
                            "unknown": result.answer.value == "unknown",
                            "elapsed_seconds": round(elapsed, 4),
                            "frame_count": len(request.frames),
                            "identity_inference": result.subject_identity_inference,
                            "risk_inference": result.risk_inference,
                            "notification_decision": result.notification_decision,
                        }
                    )
    total = time.perf_counter() - started
    by_variant = {}
    for variant_name in ("full", "full_yolo_crop"):
        selected = [item for item in results if item["variant"] == variant_name]
        grouped_answers = {
            case["case_id"]: {
                item["answer"] for item in selected if item["case_id"] == case["case_id"]
            }
            for case in dataset["cases"]
        }
        by_variant[variant_name] = {
            "cases": len(dataset["cases"]),
            "trials": len(selected),
            "correct_trials": sum(item["correct"] for item in selected),
            "unknown_trials": sum(item["unknown"] for item in selected),
            "stable_cases": sum(len(answers) == 1 for answers in grouped_answers.values()),
            "mean_inference_seconds": round(
                sum(item["elapsed_seconds"] for item in selected) / len(selected),
                4,
            ),
        }
    report = {
        "version": 1,
        "model_id": ATOMIC_MODEL_ID,
        "model_revision": OFFICIAL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "precision": args.precision,
        "dataset_sha256": dataset_sha256,
        "repetitions": args.repetitions,
        "model_load_seconds": round(loaded - started, 4),
        "warmup_seconds": round(warmup_seconds, 4),
        "total_seconds": round(total, 4),
        "household_media": False,
        "uploads_performed": False,
        "identity_used": False,
        "risk_used": False,
        "notification_used": False,
        "by_variant": by_variant,
        "results": results,
    }
    _write_private(output, report)
    print(
        json.dumps(
            {
                "version": 1,
                "precision": args.precision,
                "cases": len(dataset["cases"]) * 2,
                "trials": len(results),
                "correct_trials": sum(item["correct"] for item in results),
                "unknown_trials": sum(item["unknown"] for item in results),
                "total_seconds": round(total, 4),
                "paths_disclosed": False,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
