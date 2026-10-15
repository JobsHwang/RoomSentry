"""Shared validation helpers for the domain contract objects."""

from __future__ import annotations

import re
from typing import Any


class SchemaError(ValueError):
    """Raised when a VLM response is not strict, schema-compliant JSON."""


def _reject_json_constant(value: str) -> None:
    raise SchemaError(f"JSON constant {value!r} is not allowed")


def _require_exact_keys(data: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise SchemaError(f"{context} has invalid keys: {', '.join(details)}")


def _confidence(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{field} must be a number")
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise SchemaError(f"{field} must be between 0.0 and 1.0")
    return parsed


def _string_list(value: Any, field: str, *, unique: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaError(f"{field} must be an array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise SchemaError(f"every {field} item must be a non-empty string")
        result.append(item)
    if unique and len(set(result)) != len(result):
        raise SchemaError(f"{field} items must be unique")
    return tuple(result)


_SAFE_PROVENANCE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
