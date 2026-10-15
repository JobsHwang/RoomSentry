from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baby_monitor.domain import EventMedia, SchemaError, VLMEvent


class LocalVLMRunnerError(RuntimeError):
    pass


PROMPT_VERSION = "vlm_event_v5_20260912_coverage_unverified"
RUNNER_FAILURE_PROTOCOL = "baby_monitor_local_runner_failure_v1"


JSON_SYSTEM_PROMPT = (
    "你是一个保守的本地视觉事件分析器。必须返回完整且唯一的 JSON 对象，"
    "结构服从用户提供的模板。格式正确不代表事实正确；证据不足时保持 unknown、低置信度并请求复核。"
)


def _is_explicit_cuda_oom(error: BaseException) -> bool:
    """Recognize a caught PyTorch OOM without importing the optional runtime."""
    current: BaseException | None = error
    seen: set[int] = set()
    for _ in range(16):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        error_type = type(current)
        if error_type.__name__ == "OutOfMemoryError" and (
            error_type.__module__ == "torch" or error_type.__module__.startswith("torch.")
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def runner_failure_status(error: BaseException) -> dict[str, object]:
    """Return a path-free status; never include exception or model-generated text."""
    if _is_explicit_cuda_oom(error):
        error_code = "runner_oom"
        oom_status = "observed"
    elif isinstance(error, SchemaError):
        error_code = "invalid_structured_output"
        oom_status = "not_observed"
    else:
        error_code = "runner_failed"
        oom_status = "unknown"
    return {
        "protocol": RUNNER_FAILURE_PROTOCOL,
        "error_code": error_code,
        "oom_status": oom_status,
    }


def report_runner_failure(error: BaseException) -> None:
    print(
        json.dumps(runner_failure_status(error), separators=(",", ":")),
        file=sys.stderr,
    )


@dataclass(frozen=True, slots=True)
class LocalVLMRequest:
    candidate: dict[str, Any]
    evidence: EventMedia
    frame_paths: tuple[Path, ...]
    question: str
    output_schema: dict[str, Any]

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return tuple(path.name for path in self.frame_paths)


def load_request(path: str | Path) -> LocalVLMRequest:
    request_path = Path(path).expanduser()
    try:
        value = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalVLMRunnerError("request JSON is missing or invalid") from exc
    if not isinstance(value, dict):
        raise LocalVLMRunnerError("request must be one JSON object")

    candidate = value.get("candidate")
    frame_values = value.get("frame_paths")
    question = value.get("question")
    output_schema = value.get("output_schema")
    evidence_value = value.get("evidence")
    if not isinstance(candidate, dict):
        raise LocalVLMRunnerError("request candidate must be an object")
    if not isinstance(frame_values, list) or not 1 <= len(frame_values) <= 8:
        raise LocalVLMRunnerError("request must contain between 1 and 8 frames")
    if not isinstance(question, str) or not question.strip():
        raise LocalVLMRunnerError("request question must be non-empty text")
    if not isinstance(output_schema, dict):
        raise LocalVLMRunnerError("request output_schema must be an object")
    if output_schema != VLMEvent.json_schema():
        raise LocalVLMRunnerError("request output_schema does not match the project contract")
    try:
        evidence = EventMedia.from_dict(evidence_value)
    except SchemaError as exc:
        raise LocalVLMRunnerError("request evidence metadata is invalid") from exc

    frames: list[Path] = []
    names: set[str] = set()
    for raw in frame_values:
        if not isinstance(raw, str) or not raw:
            raise LocalVLMRunnerError("every frame path must be non-empty text")
        frame = Path(raw).expanduser()
        if not frame.is_absolute():
            raise LocalVLMRunnerError("frame paths must be absolute")
        if frame.is_symlink() or not frame.is_file():
            raise LocalVLMRunnerError("a requested frame is missing or unsafe")
        if frame.name in names:
            raise LocalVLMRunnerError("frame basenames must be unique")
        names.add(frame.name)
        frames.append(frame)
    if tuple(str(frame) for frame in frames) != evidence.frame_paths:
        raise LocalVLMRunnerError("request evidence frame paths do not match")
    return LocalVLMRequest(
        candidate=candidate,
        evidence=evidence,
        frame_paths=tuple(frames),
        question=question.strip(),
        output_schema=output_schema,
    )


def build_prompt(
    request: LocalVLMRequest,
    *,
    include_default_template: bool = False,
) -> str:
    detections = request.candidate.get("detections", [])
    signals = request.candidate.get("signals", [])
    if include_default_template:
        output_template = {
            "event_type": "unknown",
            "subjects": [],
            "observations": [],
            "inferences": [],
            "risk_level": "none",
            "confidence": 0.0,
            "needs_remote_review": True,
            "summary_zh": "",
            "evidence_frames": [request.frame_ids[0]],
        }
        structure_guidance = [
            "必须从下面的单行模板开始，只替换值；不能增加、删除、重复或改写字段名：",
            json.dumps(output_template, ensure_ascii=False, separators=(",", ":")),
            "模板中的值只是格式占位，必须先观察画面再替换；不得把整个模板当作视觉答案。",
        ]
    else:
        structure_guidance = [
            "JSON 必须依次且仅包含这 9 个根字段：event_type、subjects、observations、"
            "inferences、risk_level、confidence、needs_remote_review、summary_zh、"
            "evidence_frames；不能增加、删除、重复或改写字段名。",
            "必须先观察画面再填写所有字段，不提供任何可照抄的默认答案。",
        ]
    return "\n".join(
        [
            request.question,
            "你正在分析按时间顺序排列的本地事件关键帧。",
            "只输出一个 JSON 对象，不要 Markdown、代码围栏、解释或 JSON 之外的文字。",
            "输出的第一个非空字符必须是 {，最后一个非空字符必须是 }；绝对禁止 ```json、"
            "``` 或任何代码围栏。",
            *structure_guidance,
            "即使无法判断家庭事件而令 event_type=unknown，"
            "也必须在 observations 中列出画面确实可见的颜色、形状、数量、相对位置、物体以及"
            "可信的跨帧变化；只有画面没有任何可靠可见内容时 observations 才能为空。",
            "summary_zh 必须用非空中文短句概括直接可见内容和不确定性；不得因为事件未知而留空。",
            "event_type 只能是 sleeping、awake、crying、movement、adult_interaction、"
            "possible_risk、unknown 之一。",
            "risk_level 只能是 none、low、medium、high 之一；confidence 必须是 0 到 1 的数字；"
            "needs_remote_review 必须是 JSON 小写布尔值 true 或 false。",
            "subjects 必须是 JSON 数组。若非空，每项只能有 kind、confidence、evidence_frames "
            "三个字段；kind 只能是 baby、adult、person、pet、unknown。",
            "observations、inferences、evidence_frames 必须是 JSON 字符串数组；"
            "不要把 observations、inferences 等字段名当成数组内容。",
            "数组必须是原生 JSON 数组，不能把整个数组包在字符串里。字段片段"
            ' "observations":"[\\"事实\\"]" 是错误格式；'
            '"observations":["事实"] 才是正确格式。',
            "observations 只能写关键帧中直接可见的事实，或下面明确提供的系统检测事实。",
            "宝宝/成人身份、睡眠、醒来、哭闹、翻身、离床、抱起和风险若证据不足，"
            "必须放入 inferences 并使用不确定措辞，或令 event_type 为 unknown。",
            "静态无声关键帧不能证明哭闹。不得把未看到的声音、人物、动作或物品写成事实。",
            "静态或重复关键帧不能证明 walking、movement、翻身、离床等连续动作。只有不同时间帧"
            "显示可比较的位置或姿态变化时，才能把该变化写入 observations；仅凭单帧姿势猜测的"
            "动作必须放入 inferences 并使用不确定措辞。没有可见跨帧变化时，event_type 应为 "
            "unknown。",
            "重复的 person 检测以及 zone_enter、zone_dwell 只证明系统在多个采样时刻记录到目标或"
            "区域状态，不证明人物身份或身体动作。",
            "置信度不得因为必须作答而提高。证据不足时 confidence 应较低，"
            "并设置 needs_remote_review。",
            "evidence_frames 以及每个 subject.evidence_frames 只能使用给出的帧 ID。",
            "根级 evidence_frames 必须至少包含一个实际依据帧；每个 subject 的依据帧也必须"
            "包含在根级 evidence_frames 中。",
            "帧 ID（输入顺序）：" + json.dumps(request.frame_ids, ensure_ascii=False),
            "系统检测事实：" + json.dumps(detections, ensure_ascii=False),
            "系统信号事实：" + json.dumps(signals, ensure_ascii=False),
            "证据覆盖元数据："
            + json.dumps(
                {
                    "coverage_status": request.evidence.coverage_status,
                    "continuous_coverage_verified": False,
                    "coverage_gap_reasons": [gap.reason for gap in request.evidence.coverage_gaps],
                    "selected_segments": request.evidence.selected_segments,
                },
                ensure_ascii=False,
            ),
            "若存在覆盖缺口，不得猜测缺失时段发生了什么。",
            "coverage_status=unverified 表示未验证连续覆盖；未知不等于已记录缺失，也不等于"
            "连续覆盖。gaps_recorded 仅表示已有缺口记录；未记录缺口不能证明没有缺口。",
        ]
    )


def build_constrained_schema(request: LocalVLMRequest) -> dict[str, Any]:
    schema = copy.deepcopy(request.output_schema)
    frame_items = {"type": "string", "enum": list(request.frame_ids)}
    schema["properties"]["evidence_frames"]["items"] = frame_items
    schema["properties"]["evidence_frames"]["minItems"] = 1
    subject_properties = schema["properties"]["subjects"]["items"]["properties"]
    subject_properties["evidence_frames"]["items"] = copy.deepcopy(frame_items)
    subject_properties["evidence_frames"]["minItems"] = 1
    return schema


def validate_result(raw: str, *, allowed_frame_ids: tuple[str, ...]) -> VLMEvent:
    result = VLMEvent.from_json(raw.strip())
    allowed = set(allowed_frame_ids)
    root_references = set(result.evidence_frames)
    if not root_references:
        raise SchemaError("model did not cite an evidence frame")
    references = set(root_references)
    for subject in result.subjects:
        if not subject.evidence_frames:
            raise SchemaError("model subject did not cite an evidence frame")
        if not set(subject.evidence_frames).issubset(root_references):
            raise SchemaError("subject evidence is missing from root evidence_frames")
        references.update(subject.evidence_frames)
    if not references.issubset(allowed):
        raise SchemaError("model referenced an evidence frame that was not provided")
    return result
