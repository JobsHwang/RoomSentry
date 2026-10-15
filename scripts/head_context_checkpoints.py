"""Pure finite checkpoint bindings for offline head-context comparisons.

No filesystem, media, model calls, resume policy, or production authority.
Individual source checkpoints are partial evidence, never a completed matrix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

from scripts.head_conditioned_roi_geometry import SIDE_PIXELS

SOURCE_COUNT = 20
CALLS_PER_SOURCE = 14
_SEMANTICS = "offline_unverified_head_conditioned_yolo_checkpoint"


def _digest(value: object) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("checkpoint requires finite JSON data") from error
    if len(payload) > 1024 * 1024:
        raise ValueError("source checkpoint exceeds one MiB")
    return hashlib.sha256(payload).hexdigest()


def _sha(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("invalid checkpoint digest")
    return value


def make_case_checkpoint(
    case_index: object, case: dict, rows: object, *, protocol_sha256: object
) -> dict:
    """Copy one complete 14-call source, binding it to its frozen protocol."""
    _sha(protocol_sha256)
    if type(case_index) is not int or not 0 <= case_index < SOURCE_COUNT:
        raise ValueError("invalid checkpoint source ordinal")
    if type(case) is not dict:
        raise ValueError("invalid checkpoint source")
    frame = case.get("frame")
    if (
        type(frame) is not str
        or not 1 <= len(frame) <= 200
        or PurePosixPath(frame).name != frame
        or "\\" in frame
        or not frame.endswith(".png")
    ):
        raise ValueError("invalid checkpoint frame")
    for field in ("png_sha256", "raw_bgr_sha256"):
        _sha(case.get(field))
    if type(case.get("camera_id")) is not str or case["camera_id"] not in {
        f"camera_{i:02}" for i in range(1, 5)
    }:
        raise ValueError("invalid checkpoint camera")
    if type(rows) is not list or len(rows) != CALLS_PER_SOURCE:
        raise ValueError("source checkpoint must contain all fourteen calls")
    expected = {(None, None, rot) for rot in (0, 180)} | {
        (anchor, side, rot) for anchor in (0, 1) for side in SIDE_PIXELS for rot in (0, 180)
    }
    keys = []
    for row in rows:
        if type(row) is not dict or any(
            row.get(field) != value
            for field, value in (
                ("frame", frame),
                ("camera_id", case["camera_id"]),
                ("frame_sha256", case["png_sha256"]),
                ("raw_bgr_sha256", case["raw_bgr_sha256"]),
            )
        ):
            raise ValueError("checkpoint source binding mismatch")
        if not {"anchor_index", "side_pixels", "rotation"}.issubset(row):
            raise ValueError("checkpoint matrix key missing")
        anchor, side, rotation = (row[k] for k in ("anchor_index", "side_pixels", "rotation"))
        if (
            (anchor is not None and (type(anchor) is not int or anchor not in (0, 1)))
            or (side is not None and (type(side) is not int or side not in SIDE_PIXELS))
            or type(rotation) is not int
            or rotation not in (0, 180)
        ):
            raise ValueError("invalid checkpoint matrix key")
        for field in ("pixels_sha256", "tensor_sha256"):
            _sha(row.get(field))
        keys.append((anchor, side, rotation))
    if len(set(keys)) != CALLS_PER_SOURCE or set(keys) != expected:
        raise ValueError("missing, duplicate, or undeclared checkpoint calls")
    rows_sha = _digest(rows)
    copied_rows = json.loads(json.dumps(rows, allow_nan=False))
    return {
        "version": 1,
        "semantics": _SEMANTICS,
        "protocol_sha256": protocol_sha256,
        "case_index": case_index,
        "frame": frame,
        "rows_sha256": rows_sha,
        "rows": copied_rows,
        "accuracy_measured": False,
        "production_writeback": False,
        "complete_matrix": False,
    }


def complete_checkpoint_rows(
    cases: object, checkpoints: object, *, protocol_sha256: object
) -> list:
    """Require twenty protocol-bound sources; partial data cannot pass."""
    _sha(protocol_sha256)
    if (
        type(cases) is not list
        or len(cases) != SOURCE_COUNT
        or type(checkpoints) is not list
        or len(checkpoints) != SOURCE_COUNT
    ):
        raise ValueError("complete comparison requires twenty source checkpoints")
    if (
        any(type(c) is not dict or type(c.get("frame")) is not str for c in cases)
        or len({c["frame"] for c in cases}) != SOURCE_COUNT
    ):
        raise ValueError("duplicate or invalid frozen sources")
    indexed = {}
    for item in checkpoints:
        if type(item) is not dict or type(item.get("version")) is not int or item["version"] != 1:
            raise ValueError("invalid checkpoint version")
        index = item.get("case_index")
        if type(index) is not int or not 0 <= index < SOURCE_COUNT or index in indexed:
            raise ValueError("duplicate or invalid source ordinal")
        expected = make_case_checkpoint(
            index, cases[index], item.get("rows"), protocol_sha256=protocol_sha256
        )
        if item != expected or any(
            item[k] is not False
            for k in ("accuracy_measured", "production_writeback", "complete_matrix")
        ):
            raise ValueError("checkpoint digest, protocol, or semantics mismatch")
        indexed[index] = expected["rows"]
    return [row for index in range(SOURCE_COUNT) for row in indexed[index]]
