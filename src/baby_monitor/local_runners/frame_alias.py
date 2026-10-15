"""Opt-in single-frame wire alias experiment; never changes source media or facts."""

import copy
import json
import re

from baby_monitor.local_runners.common import validate_result

ALIAS = "f0.png"


def _source(source):
    if type(source) is not str or re.fullmatch(r"[A-Za-z0-9_.-]{1,240}", source) is None:
        raise ValueError("invalid source frame name")
    if source in (".", "..", ALIAS):
        raise ValueError("source frame alias collision")


def alias_payload(payload, source):
    """Replace only the explicit frame-ID line and both constrained enum lists."""
    _source(source)
    result = copy.deepcopy(payload)
    try:
        messages = result["messages"]
        if len(messages) != 1 or len(messages[0]["content"]) != 2:
            raise ValueError("unsupported payload")
        content = messages[0]["content"]
        if content[0]["type"] != "image_url" or content[1]["type"] != "text":
            raise ValueError("unsupported content")
        marker = "帧 ID（输入顺序）：" + json.dumps((source,), ensure_ascii=False)
        lines = content[1]["text"].split("\n")
        if lines.count(marker) != 1:
            raise ValueError("frame marker mismatch")
        lines[lines.index(marker)] = "帧 ID（输入顺序）：" + json.dumps((ALIAS,))
        content[1]["text"] = "\n".join(lines)
        props = result["response_format"]["json_schema"]["schema"]["properties"]
        for field in (
            props["evidence_frames"],
            props["subjects"]["items"]["properties"]["evidence_frames"],
        ):
            if field["items"] != {"type": "string", "enum": [source]}:
                raise ValueError("frame enum mismatch")
            field["items"]["enum"] = [ALIAS]
    except (KeyError, TypeError, IndexError, AttributeError):
        raise ValueError("unsupported frame alias payload") from None
    return result


def restore_event(raw, source):
    """Validate alias references first, map only evidence fields, validate again."""
    _source(source)
    event = validate_result(raw, allowed_frame_ids=(ALIAS,)).to_dict()
    event["evidence_frames"] = [source for _ in event["evidence_frames"]]
    for subject in event["subjects"]:
        subject["evidence_frames"] = [source for _ in subject["evidence_frames"]]
    return validate_result(json.dumps(event, ensure_ascii=False), allowed_frame_ids=(source,))
