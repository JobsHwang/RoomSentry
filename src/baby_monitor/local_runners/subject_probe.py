"""Experimental, single-view subject claims; never identity or state facts.

Coordinates are integer thousandths of the submitted image, not body boxes or
track IDs. Callers must retain the exact image binding. No confidence is invented.
"""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "subject_probe_v1_20261005"
PROMPT = (
    "逐人检查这张图，返回唯一JSON对象。只报告像素直接支持的现场实体人物；衣物、家具、"
    "被褥、屏幕和倒影不能自动算人物。检查全图及边缘，局部身体可见但头部不可见时仍可报告人物，"
    "head_center设null。每个人只列一次，无法分清时在uncertainties说明，不补全被遮挡的人。"
    "people每项有head_center、age_group、visual_evidence三个字段。head_center是可见头部中心"
    "[x,y]，左上角[0,0]、右下角[1000,1000]，整数坐标，无法定位用null。"
    "age_group只能为baby、adult、unknown：清晰外观支持婴幼儿用baby，清晰外观支持成年人用adult，"
    "年龄依据不足用unknown。不能从床、婴儿用品、体积远近或有人照料推断年龄。"
    "visual_evidence用简短中文写该人物可见的面部/身体特征和遮挡。uncertainties为简短字符串数组。"
    "每条文字最多80字，最多8个人；无法确认人物时people为空。单图不判断睡眠、身份、亲属关系或风险。"
)


def schema() -> dict[str, Any]:
    """Return a fresh schema matching the frozen pilot, without event fields."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["people", "uncertainties"],
        "properties": {
            "people": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["head_center", "age_group", "visual_evidence"],
                    "properties": {
                        "head_center": {
                            "anyOf": [
                                {"type": "null"},
                                {
                                    "type": "array",
                                    "minItems": 2,
                                    "maxItems": 2,
                                    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                                },
                            ]
                        },
                        "age_group": {"type": "string", "enum": ["baby", "adult", "unknown"]},
                        "visual_evidence": {"type": "string", "minLength": 1, "maxLength": 240},
                    },
                },
            },
            "uncertainties": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "maxLength": 240},
            },
        },
    }


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate probe field")
        result[key] = value
    return result


def validate_bound_result(value, *, frame_id: str, input_sha256: str):
    """Validate the explicit probe IPC result without interpreting claims as facts."""
    if type(value) is not dict or set(value) != {
        "version",
        "prompt_revision",
        "semantics",
        "frame_id",
        "input_sha256",
        "coordinate_space",
        "claims",
    }:
        raise ValueError("invalid bound probe fields")
    if (
        type(value["version"]) is not int
        or value["version"] != 1
        or value["prompt_revision"] != PROMPT_VERSION
        or value["semantics"] != "unverified_model_claims"
        or value["coordinate_space"] != "submitted_image_thousandths"
        or value["frame_id"] != frame_id
        or value["input_sha256"] != input_sha256
    ):
        raise ValueError("probe input or contract mismatch")
    claims = parse_response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(value["claims"], allow_nan=False),
                    },
                }
            ]
        }
    )
    return {**value, "claims": claims}


def _constant(_value):
    raise ValueError("nonfinite probe value")


def parse_response(response: Any) -> dict[str, Any]:
    """Reject malformed/truncated envelopes without echoing private model text."""
    if type(response) is not dict or "error" in response:
        raise ValueError("invalid probe response")
    choices = response.get("choices")
    if (
        type(choices) is not list
        or len(choices) != 1
        or type(choices[0]) is not dict
        or choices[0].get("finish_reason") != "stop"
    ):
        raise ValueError("incomplete probe response")
    message = choices[0].get("message")
    if type(message) is not dict or message.get("tool_calls") or message.get("refusal"):
        raise ValueError("invalid probe message")
    raw = message.get("content")
    try:
        if type(raw) is not str or len(raw.encode("utf-8")) > 32768:
            raise ValueError("invalid probe content")
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("invalid probe JSON") from None
    if type(value) is not dict or set(value) != {"people", "uncertainties"}:
        raise ValueError("invalid probe fields")
    people, uncertainties = value["people"], value["uncertainties"]
    if (
        type(people) is not list
        or len(people) > 8
        or type(uncertainties) is not list
        or len(uncertainties) > 8
    ):
        raise ValueError("invalid probe arrays")
    for person in people:
        if type(person) is not dict or set(person) != {
            "head_center",
            "age_group",
            "visual_evidence",
        }:
            raise ValueError("invalid person fields")
        center = person["head_center"]
        if center is not None and (
            type(center) is not list
            or len(center) != 2
            or any(type(n) is not int or not 0 <= n <= 1000 for n in center)
        ):
            raise ValueError("invalid head center")
        if person["age_group"] not in ("baby", "adult", "unknown"):
            raise ValueError("invalid age group")
        evidence = person["visual_evidence"]
        if type(evidence) is not str or not 1 <= len(evidence) <= 240:
            raise ValueError("invalid evidence text")
    if any(type(s) is not str or len(s) > 240 for s in uncertainties):
        raise ValueError("invalid uncertainty text")
    return value
