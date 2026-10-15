"""Experimental whole-image visibility claims, not identity or live room facts.

This smaller task deliberately asks for neither head coordinates nor sleep. A
model's ``no`` means no supported visible target in this image, not an empty or
safe room. Callers retain the exact input binding and failures separately.
"""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "scene_visibility_v1_20261010"
ANSWERS = ("yes", "no", "unknown")
FIELDS = ("person_visible", "baby_visible", "adult_visible")
PROMPT = (
    "只看这张图，回答现场是否直接可见实体人物、婴幼儿、成年人。返回唯一JSON对象，"
    "person_visible、baby_visible、adult_visible各用yes/no/unknown。"
    "yes需有可辨的外观依据；no表示图中未见该类目标，不表示房间无人或安全；"
    "存在模糊、遮挡或年龄歧义时用unknown，不把无法判断当作no。"
    "人物局部可见也需检查；家具、被褥、玩具、屏幕和倒影不自动算现场人物。"
    "不根据床、婴儿用品、距离远近或有人照料猜年龄。"
    "visual_evidence用一句中文描述可见依据或不确定原因，最多80字。"
    "不输出坐标、人数、置信分数、身份、亲属关系、睡眠或危险结论。"
)


def schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [*FIELDS, "visual_evidence"],
        "properties": {
            **{field: {"type": "string", "enum": list(ANSWERS)} for field in FIELDS},
            "visual_evidence": {"type": "string", "minLength": 1, "maxLength": 240},
        },
    }


def _pairs(items):
    result = {}
    for key, item in items:
        if key in result:
            raise ValueError("duplicate scene field")
        result[key] = item
    return result


def _constant(_value):
    raise ValueError("nonfinite scene value")


def validate_claims(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {*FIELDS, "visual_evidence"}:
        raise ValueError("invalid scene fields")
    if any(type(value[field]) is not str or value[field] not in ANSWERS for field in FIELDS):
        raise ValueError("invalid scene answer")
    text = value["visual_evidence"]
    if type(text) is not str or not text.strip() or len(text) > 240:
        raise ValueError("invalid scene evidence")
    try:
        text.encode("utf-8")
    except UnicodeError:
        raise ValueError("invalid scene evidence") from None
    if value["person_visible"] != "yes" and any(
        value[field] == "yes" for field in ("baby_visible", "adult_visible")
    ):
        raise ValueError("inconsistent scene presence")
    if value["person_visible"] == "no" and any(
        value[field] != "no" for field in ("baby_visible", "adult_visible")
    ):
        raise ValueError("inconsistent scene absence")
    return dict(value)


def parse_response(response: Any) -> dict[str, Any]:
    if type(response) is not dict or "error" in response:
        raise ValueError("invalid scene response")
    choices = response.get("choices")
    if (
        type(choices) is not list
        or len(choices) != 1
        or type(choices[0]) is not dict
        or choices[0].get("finish_reason") != "stop"
    ):
        raise ValueError("incomplete scene response")
    message = choices[0].get("message")
    if type(message) is not dict or message.get("tool_calls") or message.get("refusal"):
        raise ValueError("invalid scene message")
    raw = message.get("content")
    try:
        if type(raw) is not str or not 0 < len(raw.encode("utf-8")) <= 32768:
            raise ValueError("invalid scene content")
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
        return validate_claims(value)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("invalid scene JSON or claims") from None
