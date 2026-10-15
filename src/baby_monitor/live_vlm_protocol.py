"""Closed, content-free diagnostics shared by local VLM IPC participants."""

from __future__ import annotations

import math
from typing import TypeAlias

VLMDiagnostics: TypeAlias = dict[str, int | float | bool | str | None]

_COUNT_FIELDS = (
    "prompt_tokens_reported",
    "completion_tokens_reported",
    "total_tokens_reported",
    "cached_tokens_reported",
    "image_tokens_reported",
    "prompt_tokens_timed",
    "cached_tokens_timed",
    "predicted_tokens_timed",
)
_MILLISECOND_FIELDS = ("prompt_ms_reported", "predicted_ms_reported")
_FIELDS = frozenset(
    (
        "context_tokens_configured",
        "image_tokens_configured",
        "max_new_tokens_configured",
        "context_truncated_reported",
        "finish_reason_reported",
        *_COUNT_FIELDS,
        *_MILLISECOND_FIELDS,
    )
)
_FINISH_REASONS = frozenset(("stop", "length", "tool_calls", "content_filter", "function_call"))
_INVALID_DIAGNOSTICS = "invalid VLM diagnostics"


def validate_vlm_diagnostics(value: object) -> VLMDiagnostics | None:
    """Copy a closed diagnostic record or reject it without echoing its contents.

    Configured budgets are not measurements; reported values may remain unknown.
    This coarse configuration check does not guarantee an actual prompt fits.
    """
    if value is None:
        return None
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or value.keys() != _FIELDS
    ):
        raise ValueError(_INVALID_DIAGNOSTICS)

    context = value["context_tokens_configured"]
    image = value["image_tokens_configured"]
    output = value["max_new_tokens_configured"]
    if (
        type(context) is not int
        or context not in (4096, 8192)
        or type(image) is not int
        or image not in (1024, 2048, 4096)
        or type(output) is not int
        or not 64 <= output <= 1024
        or image + output + 1024 >= context
    ):
        raise ValueError(_INVALID_DIAGNOSTICS)

    for field in _COUNT_FIELDS:
        item = value[field]
        if item is not None and (type(item) is not int or not 0 <= item <= 1_000_000):
            raise ValueError(_INVALID_DIAGNOSTICS)
    for field in _MILLISECOND_FIELDS:
        item = value[field]
        # Check bounds before isfinite: converting an enormous int may overflow.
        if item is not None and (
            type(item) not in (int, float) or not 0 <= item <= 600_000 or not math.isfinite(item)
        ):
            raise ValueError(_INVALID_DIAGNOSTICS)

    truncated = value["context_truncated_reported"]
    finish = value["finish_reason_reported"]
    if truncated is not None and type(truncated) is not bool:
        raise ValueError(_INVALID_DIAGNOSTICS)
    if finish is not None and (type(finish) is not str or finish not in _FINISH_REASONS):
        raise ValueError(_INVALID_DIAGNOSTICS)
    return dict(value)
