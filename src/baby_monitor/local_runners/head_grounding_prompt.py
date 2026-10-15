"""Opt-in head-guided subject experiment; detector hints are never role facts."""

from __future__ import annotations

import json
import math

from baby_monitor.local_runners import subject_probe
from baby_monitor.local_runners.subject_head_match import match_head_centers

PROMPT_VERSION = "head_guided_subject_v1_20261006"


def build_prompt(*, frame_sha256: str, head_input_sha256: str, heads: list) -> str:
    """Use only same-image geometric hints and preserve the full-scene task.

    Input boxes are normalized in the original submitted image. No labels,
    identity, age guesses, detector confidence or private prose enter the prompt.
    An empty candidate list still asks the model to inspect the entire image.
    """
    if type(heads) is not list or len(heads) > 8:
        raise ValueError("invalid head candidate list")
    match_head_centers(
        probe_input_sha256=frame_sha256,
        head_input_sha256=head_input_sha256,
        centers=[],
        heads=heads,
    )
    boxes = [
        [
            math.floor(box[0] * 1000),
            math.floor(box[1] * 1000),
            math.ceil(box[2] * 1000),
            math.ceil(box[3] * 1000),
        ]
        for box in heads
    ]
    return (
        subject_probe.PROMPT
        + "\n独立头部检测器给出以下候选矩形[x1,y1,x2,y2]，坐标仍为本图千分位："
        + json.dumps(boxes, ensure_ascii=True, separators=(",", ":"))
        + "。这些只是可能误报或漏报的几何提示，不是已确认人物或年龄。"
        "请逐个查看候选附近实际可见的头部与身体，再独立定位head_center并判断年龄；"
        "没有真实人物的候选不要报告。继续检查全图与边缘，候选之外的人也要报告，"
        "不得把一个候选补成一个人，不得因为多个候选而重复报告同一个人。"
    )
