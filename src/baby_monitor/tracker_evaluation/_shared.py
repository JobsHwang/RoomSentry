"""Shared strict-JSON validation helpers and limits for tracker evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..object_tracker import (
    DetectionFrameBatch,
)

MAX_TRACKER_EVAL_BYTES = 8 * 1024 * 1024


MAX_TRACKER_EVAL_ASSIGNMENTS = 10_000


_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")


_SUBJECT_ID = re.compile(r"subject_[A-Za-z0-9_.-]{1,64}")


_REVIEWER_ID = re.compile(r"reviewer_[A-Za-z0-9_.-]{1,64}")


_SHA256 = re.compile(r"[0-9a-f]{64}")


class TrackerEvaluationError(ValueError):
    """A tracker review, prediction, or association report is unsafe."""


def _strict_json(raw: str, context: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise TrackerEvaluationError(f"{context} contains duplicate JSON keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                TrackerEvaluationError(f"{context} contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise TrackerEvaluationError(f"{context} is invalid JSON") from exc


def _read_json(path: str | Path, context: str) -> Any:
    source = Path(path).expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        if not 0 < source.stat().st_size <= MAX_TRACKER_EVAL_BYTES:
            raise TrackerEvaluationError(f"{context} exceeds the 8 MiB limit")
        raw = source.read_text(encoding="utf-8")
    except TrackerEvaluationError:
        raise
    except (OSError, UnicodeError) as exc:
        raise TrackerEvaluationError(f"{context} is missing or unsafe") from exc
    return _strict_json(raw, context)


def _write_new_json(path: str | Path, value: object, context: str) -> None:
    target = Path(path).expanduser()
    parent = target.parent
    if target.is_symlink() or not parent.is_dir() or parent.is_symlink():
        raise TrackerEvaluationError(f"{context} destination is unsafe")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if len(payload) > MAX_TRACKER_EVAL_BYTES:
        raise TrackerEvaluationError(f"{context} exceeds the 8 MiB limit")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError:
        raise FileExistsError(f"{context} output already exists") from None
    except OSError as exc:
        raise TrackerEvaluationError(f"{context} could not be created") from exc
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        with suppress(OSError):
            target.unlink()
        raise TrackerEvaluationError(f"{context} could not be written") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _exact_object(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise TrackerEvaluationError(f"{context} fields do not match v1")
    return value


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise TrackerEvaluationError(f"{field} is invalid")
    return value


def _subject_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SUBJECT_ID.fullmatch(value) is None:
        raise TrackerEvaluationError(f"{field} must be an anonymous subject_ identifier")
    return value


def _reviewer_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _REVIEWER_ID.fullmatch(value) is None:
        raise TrackerEvaluationError(f"{field} must be an anonymous reviewer_ identifier")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrackerEvaluationError(f"{field} is invalid")
    return value


def _optional_sha256(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field)


def _aware_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise TrackerEvaluationError(f"{field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TrackerEvaluationError(f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TrackerEvaluationError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _integer(value: Any, field: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise TrackerEvaluationError(f"{field} is invalid")
    return value


def _optional_ratio(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise TrackerEvaluationError(f"{field} is invalid")
    return float(value)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 12)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _detection_keys(batch: DetectionFrameBatch) -> tuple[tuple[str, str], ...]:
    return tuple(
        (frame.frame_id, detection.detection_id)
        for frame in batch.frames
        for detection in frame.detections
    )


MAX_TRACKER_DISAGREEMENT_FRAMES = 200
