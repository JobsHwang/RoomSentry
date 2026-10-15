"""Rules that separate visible facts from unsupported identity/action claims.

Pure transformations over domain records: no adapters, storage or model calls.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .domain import CandidateEvent, EventType, SignalType, SubjectKind, VLMEvent

_TEMPORAL_OBSERVATION_PATTERN = re.compile(
    r"\b(?:crying|falling|moving|movement|running|walking|"
    r"walks?|walked|moves?|moved|runs?|ran|falls?|fell|"
    r"rolling|rolled|climbing|climbed|leaving|left|entering|"
    r"exiting|picking\s+up|carrying)\b",
    flags=re.IGNORECASE,
)
_TEMPORAL_OBSERVATION_MARKERS = (
    "行走",
    "走动",
    "正在走",
    "移动",
    "跑动",
    "奔跑",
    "翻身",
    "离床",
    "下床",
    "进入",
    "离开",
    "抱起",
    "跌落",
    "摔倒",
    "哭闹",
    "正在哭",
)
_IDENTITY_OBSERVATION_PATTERNS = {
    SubjectKind.BABY: re.compile(
        r"\b(?:baby|babies|infant|infants)\b|宝宝|婴(?:幼)?儿",
        flags=re.IGNORECASE,
    ),
    SubjectKind.ADULT: re.compile(
        r"\b(?:adult|adults)\b|成人|成年人|大人",
        flags=re.IGNORECASE,
    ),
}
_IDENTITY_DISCLAIMER_PATTERNS = (
    re.compile(
        r"(?:不是|不代表|不能(?:证明|确定)|并非)[^。.;；,，:：!！?？\r\n]{0,32}?(?:身份|结论)",
    ),
    re.compile(
        r"\b(?:not|does\s+not|cannot|can't)\b[^。.;；,，:：!！?？\r\n]{0,64}?"
        r"\b(?:identity|conclusion)\b",
        flags=re.IGNORECASE,
    ),
)
_EXPLICIT_IDENTITY_LABELS = {
    SubjectKind.BABY: frozenset({"baby", "infant"}),
    SubjectKind.ADULT: frozenset({"adult"}),
}
_HUMAN_PRESENCE_LABELS = frozenset({"person", "baby", "infant", "adult"})
_UNCERTAIN_SUMMARY = "模型摘要含未证实的描述，请查看推测及证据。"


def _has_temporal_claim(text: str) -> bool:
    return bool(_TEMPORAL_OBSERVATION_PATTERN.search(text)) or any(
        marker in text for marker in _TEMPORAL_OBSERVATION_MARKERS
    )


def _has_identity_claim(text: str, kind: SubjectKind) -> bool:
    # A disclaimer only exempts mentions inside its own bounded span, never
    # other assertions elsewhere in the same observation or summary.
    disclaimers = tuple(
        match.span()
        for pattern in _IDENTITY_DISCLAIMER_PATTERNS
        for match in pattern.finditer(text)
    )
    return any(
        not any(start <= match.start() and match.end() <= end for start, end in disclaimers)
        for match in _IDENTITY_OBSERVATION_PATTERNS[kind].finditer(text)
    )


def enforce_temporal_fact_boundary(
    result: VLMEvent,
    candidate: CandidateEvent,
) -> VLMEvent:
    """Downgrade unsupported action prose before it can become an observation.

    A local VLM receives selected still frames, not optical-flow or audio
    evidence.  Repeated detections and zone dwell facts prove sampled presence,
    but do not prove a body action.  Keep the original model text for auditability
    while moving it into explicitly uncertain inferences.
    """

    observations: list[str] = []
    downgraded: list[str] = []
    for observation in result.observations:
        if _has_temporal_claim(observation):
            downgraded.append(f"时序证据不足，模型推测：{observation}")
        else:
            observations.append(observation)

    summary = result.summary_zh
    if _has_temporal_claim(summary):
        downgraded.append(f"时序证据不足，模型推测：{summary}")
        summary = _UNCERTAIN_SUMMARY

    signal_types = {signal.signal_type for signal in candidate.signals}
    event_type = result.event_type
    if downgraded and event_type is EventType.MOVEMENT and SignalType.MOTION not in signal_types:
        event_type = EventType.UNKNOWN
    if (
        downgraded
        and event_type is EventType.CRYING
        and SignalType.HARDWARE_CRY not in signal_types
    ):
        event_type = EventType.UNKNOWN

    if not downgraded and event_type is result.event_type:
        return result
    inferences = list(result.inferences)
    for inference in downgraded:
        if inference not in inferences:
            inferences.append(inference)
    return replace(
        result,
        event_type=event_type,
        observations=tuple(observations),
        summary_zh=summary,
        inferences=tuple(inferences),
        confidence=min(result.confidence, 0.49),
        needs_remote_review=True,
    )


def _candidate_object_labels(candidate: CandidateEvent) -> frozenset[str]:
    labels = {detection.label.casefold() for detection in candidate.detections}
    for signal in candidate.signals:
        object_label = dict(signal.attributes).get("object_label")
        if isinstance(object_label, str):
            labels.add(object_label.casefold())
    return frozenset(labels)


def enforce_subject_identity_boundary(
    result: VLMEvent,
    candidate: CandidateEvent,
) -> VLMEvent:
    """Keep unsupported baby/adult identity claims out of stored facts.

    Generic person detections prove human presence, not age or household role.
    Until an upstream adapter supplies an explicit reviewed baby/adult label,
    retain the model claim only as an inference and downgrade the structured
    subject to ``person`` (or ``unknown`` when even human presence is not
    supported by the candidate facts).
    """

    candidate_labels = _candidate_object_labels(candidate)

    def has_explicit_identity(kind: SubjectKind) -> bool:
        return bool(candidate_labels & _EXPLICIT_IDENTITY_LABELS[kind])

    has_human_presence = bool(candidate_labels & _HUMAN_PRESENCE_LABELS) or any(
        signal.signal_type is SignalType.PERSON for signal in candidate.signals
    )
    inferences = list(result.inferences)
    subjects = []
    changed = False

    for subject in result.subjects:
        if subject.kind not in _EXPLICIT_IDENTITY_LABELS or has_explicit_identity(subject.kind):
            subjects.append(subject)
            continue
        changed = True
        inference = f"身份证据不足，模型推测：模型将主体标记为 {subject.kind.value}。"
        if inference not in inferences:
            inferences.append(inference)
        subjects.append(
            replace(
                subject,
                kind=(SubjectKind.PERSON if has_human_presence else SubjectKind.UNKNOWN),
                confidence=min(subject.confidence, 0.49),
            )
        )

    observations = []
    for observation in result.observations:
        unsupported_kinds = tuple(
            kind
            for kind in _IDENTITY_OBSERVATION_PATTERNS
            if not has_explicit_identity(kind) and _has_identity_claim(observation, kind)
        )
        if not unsupported_kinds:
            observations.append(observation)
            continue
        changed = True
        inference = f"身份证据不足，模型推测：{observation}"
        if inference not in inferences:
            inferences.append(inference)

    summary = result.summary_zh
    if any(
        not has_explicit_identity(kind) and _has_identity_claim(summary, kind)
        for kind in _IDENTITY_OBSERVATION_PATTERNS
    ):
        changed = True
        inference = f"身份证据不足，模型推测：{summary}"
        if inference not in inferences:
            inferences.append(inference)
        summary = _UNCERTAIN_SUMMARY

    event_type = result.event_type
    if event_type is EventType.ADULT_INTERACTION and not has_explicit_identity(SubjectKind.ADULT):
        changed = True
        event_type = EventType.UNKNOWN
        inference = "身份证据不足，模型推测：事件被标记为 adult_interaction。"
        if inference not in inferences:
            inferences.append(inference)

    if not changed:
        return result
    return replace(
        result,
        event_type=event_type,
        subjects=tuple(subjects),
        observations=tuple(observations),
        summary_zh=summary,
        inferences=tuple(inferences),
        confidence=min(result.confidence, 0.49),
        needs_remote_review=True,
    )
