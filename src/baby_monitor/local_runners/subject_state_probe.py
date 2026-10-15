"""Bound single-image role and visible posture claims for local supervision.

This is an opt-in successor to the frozen v1 diagnostic. Sleep, identity and
kinship are deliberately outside its contract. Model claims remain inferences.
"""

from __future__ import annotations

import json

from baby_monitor.local_runners import subject_probe

PROMPT_VERSION = "subject_state_probe_v2_20261012"
POSTURES = ("lying", "sitting", "standing", "held", "unknown")
VISIBILITIES = ("clear", "partial", "unknown")
PROMPT = subject_probe.PROMPT + (
    "另外每个人必须有visible_posture和visibility。visible_posture只能为lying（身体可见躺卧）、"
    "sitting（坐着）、standing（站立）、held（可见被另一人抱着）或unknown。"
    "visibility只能为clear、partial或unknown。只露头或身体关系无法辨认时姿态用unknown；"
    "床上物品不能作为躺卧依据。躺卧不等于睡着。每人visual_evidence最多40字。"
)


def schema():
    value = subject_probe.schema()
    person = value["properties"]["people"]["items"]
    person["required"].extend(("visible_posture", "visibility"))
    person["properties"].update(
        visible_posture={"type": "string", "enum": list(POSTURES)},
        visibility={"type": "string", "enum": list(VISIBILITIES)},
    )
    return value


def parse_response(response):
    if type(response) is not dict or "error" in response:
        raise ValueError("invalid state probe response")
    choices = response.get("choices")
    if (
        type(choices) is not list
        or len(choices) != 1
        or type(choices[0]) is not dict
        or choices[0].get("finish_reason") != "stop"
    ):
        raise ValueError("incomplete state probe response")
    message = choices[0].get("message")
    if type(message) is not dict or message.get("tool_calls") or message.get("refusal"):
        raise ValueError("invalid state probe message")
    raw = message.get("content")
    if type(raw) is not str or len(raw.encode("utf-8")) > 32768:
        raise ValueError("invalid state probe content")
    try:
        value = json.loads(
            raw, object_pairs_hook=subject_probe._pairs, parse_constant=subject_probe._constant
        )
    except (ValueError, RecursionError):
        raise ValueError("invalid state probe JSON") from None
    if type(value) is not dict or set(value) != {"people", "uncertainties"}:
        raise ValueError("invalid state probe fields")
    if type(value["people"]) is not list or len(value["people"]) > 8:
        raise ValueError("invalid state probe people")
    base_people = []
    for person in value["people"]:
        if type(person) is not dict or set(person) != {
            "head_center",
            "age_group",
            "visual_evidence",
            "visible_posture",
            "visibility",
        }:
            raise ValueError("invalid state person fields")
        if person["visible_posture"] not in POSTURES or person["visibility"] not in VISIBILITIES:
            raise ValueError("invalid visible state")
        base_people.append({k: person[k] for k in ("head_center", "age_group", "visual_evidence")})
    subject_probe.parse_response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"people": base_people, "uncertainties": value["uncertainties"]},
                            allow_nan=False,
                        )
                    },
                }
            ]
        }
    )
    return value


def validate_bound_result(value, *, frame_id, input_sha256):
    fields = {
        "version",
        "prompt_revision",
        "semantics",
        "frame_id",
        "input_sha256",
        "coordinate_space",
        "claims",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("invalid bound state probe fields")
    if (
        type(value["version"]) is not int
        or value["version"] != 2
        or value["prompt_revision"] != PROMPT_VERSION
        or value["semantics"] != "unverified_model_claims"
        or value["coordinate_space"] != "submitted_image_thousandths"
        or value["frame_id"] != frame_id
        or value["input_sha256"] != input_sha256
    ):
        raise ValueError("state probe binding mismatch")
    claims = parse_response(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(value["claims"], allow_nan=False)},
                }
            ]
        }
    )
    return {**value, "claims": claims}
