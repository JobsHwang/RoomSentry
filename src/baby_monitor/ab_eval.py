from __future__ import annotations

import gc
import hashlib
import json
import re
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .adapters.vlm import (
    CommandVLMAdapter,
    MockVLMAdapter,
    VLMAdapter,
    VLMError,
    VLMRequest,
)
from .domain import (
    CandidateEvent,
    EventMedia,
    EventType,
    RiskLevel,
    SubjectKind,
    VLMEvent,
)
from .local_runners.common import PROMPT_VERSION
from .resources import ResourceSampler

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_SAFE_PROVENANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
_SAFE_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}(?:/[A-Za-z0-9][A-Za-z0-9_.-]{0,79})?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EXPECTED_KEYS_V1 = {
    "event_type",
    "risk_level",
    "allowed_subjects",
    "allowed_observations",
    "allowed_inferences",
}
_EXPECTED_KEYS_V2 = _EXPECTED_KEYS_V1 | {
    "required_observation_groups",
    "required_inference_groups",
}
_MAX_DATASET_BYTES = 10 * 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024
_MAX_FRAME_BYTES = 50 * 1024 * 1024
_MAX_CASES = 1000


@dataclass(frozen=True, slots=True)
class ABFrame:
    frame_id: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ABExpected:
    event_type: EventType
    risk_level: RiskLevel
    allowed_subjects: tuple[SubjectKind, ...]
    allowed_observations: tuple[str, ...]
    allowed_inferences: tuple[str, ...]
    required_observation_groups: tuple[tuple[str, ...], ...]
    required_inference_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class ABCase:
    schema_version: int
    case_id: str
    frames: tuple[ABFrame, ...]
    question: str
    expected: ABExpected

    @property
    def frame_paths(self) -> tuple[str, ...]:
        return tuple(frame.path for frame in self.frames)


@dataclass(frozen=True, slots=True)
class ABRunArtifacts:
    results_path: Path
    summary_path: Path
    backend_count: int
    case_count: int


@dataclass(frozen=True, slots=True)
class ABBackend:
    adapter: VLMAdapter
    model_id: str
    model_revision: str
    prompt_version: str
    runtime_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, VLMAdapter):
            raise ValueError("A/B backend adapter is invalid")
        _model_identifier(self.model_id)
        _path_free_identifier(self.model_revision, field="revision")
        _path_free_identifier(self.prompt_version, field="prompt_version")
        _path_free_identifier(self.runtime_id, field="runtime_id")

    def provenance(self) -> dict[str, str | int]:
        return {
            "version": 1,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "runtime_id": self.runtime_id,
        }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _exact_object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{context} must contain exactly the v1 fields")
    return value


def _bounded_strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError(f"{field} must be an array with at most 100 items")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > 500:
            raise ValueError(f"{field} entries must be non-empty strings up to 500 characters")
        result.append(item.strip())
    if len(set(result)) != len(result):
        raise ValueError(f"{field} entries must be unique")
    return tuple(result)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_frame(value: object, *, dataset_root: Path, line_number: int) -> ABFrame:
    data = _exact_object(value, {"frame_id", "path", "sha256"}, "A/B frame")
    frame_id = data["frame_id"]
    raw_path = data["path"]
    sha256 = data["sha256"]
    if not isinstance(frame_id, str) or _SAFE_ID.fullmatch(frame_id) is None:
        raise ValueError(f"invalid frame_id on line {line_number}")
    if not isinstance(raw_path, str) or not raw_path.strip() or len(raw_path) > 4096:
        raise ValueError(f"invalid frame path on line {line_number}")
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise ValueError(f"invalid frame sha256 on line {line_number}")

    unresolved = Path(raw_path).expanduser()
    was_absolute = unresolved.is_absolute()
    if not was_absolute:
        if ".." in unresolved.parts:
            raise ValueError(f"relative frame path escapes the dataset on line {line_number}")
        unresolved = dataset_root / unresolved
    if unresolved.is_symlink():
        raise ValueError(f"frame symlinks are not allowed on line {line_number}")
    try:
        resolved = unresolved.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise ValueError(f"frame is missing or unreadable on line {line_number}") from exc
    if not resolved.is_file() or not 0 < stat.st_size <= _MAX_FRAME_BYTES:
        raise ValueError(f"frame must be a bounded regular file on line {line_number}")
    if not was_absolute and not resolved.is_relative_to(dataset_root.resolve()):
        raise ValueError(f"relative frame path escapes the dataset on line {line_number}")
    if _sha256_file(resolved) != sha256:
        raise ValueError(f"frame sha256 mismatch on line {line_number}")
    return ABFrame(frame_id=frame_id, path=str(resolved), sha256=sha256)


def _bounded_string_groups(value: object, field: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list) or len(value) > 100:
        raise ValueError(f"{field} must be an array with at most 100 groups")
    groups: list[tuple[str, ...]] = []
    for group in value:
        parsed = _bounded_strings(group, field)
        if not parsed or len(parsed) > 20:
            raise ValueError(f"each {field} group must contain 1 to 20 alternatives")
        groups.append(parsed)
    return tuple(groups)


def _parse_expected(value: object, *, line_number: int, version: int) -> ABExpected:
    expected_keys = _EXPECTED_KEYS_V1 if version == 1 else _EXPECTED_KEYS_V2
    data = _exact_object(value, expected_keys, "A/B expected labels")
    try:
        event_type = EventType(data["event_type"])
        risk_level = RiskLevel(data["risk_level"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid event/risk label on line {line_number}") from exc
    raw_subjects = data["allowed_subjects"]
    if not isinstance(raw_subjects, list) or len(raw_subjects) > len(SubjectKind):
        raise ValueError(f"allowed_subjects is invalid on line {line_number}")
    try:
        subjects = tuple(SubjectKind(item) for item in raw_subjects)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"allowed_subjects is invalid on line {line_number}") from exc
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"allowed_subjects must be unique on line {line_number}")
    allowed_observations = _bounded_strings(data["allowed_observations"], "allowed_observations")
    allowed_inferences = _bounded_strings(data["allowed_inferences"], "allowed_inferences")
    observation_groups = (
        _bounded_string_groups(
            data["required_observation_groups"],
            "required_observation_groups",
        )
        if version == 2
        else ()
    )
    inference_groups = (
        _bounded_string_groups(
            data["required_inference_groups"],
            "required_inference_groups",
        )
        if version == 2
        else ()
    )
    if any(
        alternative not in allowed_observations
        for group in observation_groups
        for alternative in group
    ):
        raise ValueError(
            f"required observation alternatives must also be allowed on line {line_number}"
        )
    if any(
        alternative not in allowed_inferences for group in inference_groups for alternative in group
    ):
        raise ValueError(
            f"required inference alternatives must also be allowed on line {line_number}"
        )
    return ABExpected(
        event_type=event_type,
        risk_level=risk_level,
        allowed_subjects=subjects,
        allowed_observations=allowed_observations,
        allowed_inferences=allowed_inferences,
        required_observation_groups=observation_groups,
        required_inference_groups=inference_groups,
    )


def load_dataset(path: str | Path) -> tuple[ABCase, ...]:
    dataset_path = Path(path).expanduser()
    try:
        size = dataset_path.stat().st_size
    except OSError as exc:
        raise ValueError("A/B dataset is missing or unreadable") from exc
    if (
        dataset_path.is_symlink()
        or not dataset_path.is_file()
        or not 0 < size <= _MAX_DATASET_BYTES
    ):
        raise ValueError("A/B dataset must be a bounded regular file, not a symlink")
    dataset_root = dataset_path.resolve().parent
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("A/B dataset must be a readable UTF-8 file") from exc
    cases: list[ABCase] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            if len(line.encode("utf-8")) > _MAX_LINE_BYTES:
                raise ValueError("dataset line exceeds 64 KiB")
            data = json.loads(line, parse_constant=_reject_constant)
            data = _exact_object(
                data,
                {"version", "case_id", "frames", "question", "expected"},
                "A/B case",
            )
            if type(data["version"]) is not int or data["version"] not in {1, 2}:
                raise ValueError("unsupported A/B case version")
            case_id = data["case_id"]
            question = data["question"]
            raw_frames = data["frames"]
            if not isinstance(case_id, str) or _SAFE_ID.fullmatch(case_id) is None:
                raise ValueError("case_id is invalid")
            if case_id in case_ids:
                raise ValueError("case_id is duplicated")
            if not isinstance(question, str) or not question.strip() or len(question) > 4000:
                raise ValueError("question must be 1 to 4000 characters")
            if not isinstance(raw_frames, list) or not 1 <= len(raw_frames) <= 8:
                raise ValueError("frames must contain 1 to 8 items")
            frames = tuple(
                _parse_frame(item, dataset_root=dataset_root, line_number=line_number)
                for item in raw_frames
            )
            if len({item.frame_id for item in frames}) != len(frames):
                raise ValueError("frame_id values must be unique within a case")
            if len({item.path for item in frames}) != len(frames):
                raise ValueError("frame paths must be unique within a case")
            if len({Path(item.path).name for item in frames}) != len(frames):
                raise ValueError("frame basenames must be unique within a case")
            cases.append(
                ABCase(
                    schema_version=data["version"],
                    case_id=case_id,
                    frames=frames,
                    question=question.strip(),
                    expected=_parse_expected(
                        data["expected"],
                        line_number=line_number,
                        version=data["version"],
                    ),
                )
            )
            case_ids.add(case_id)
            if len(cases) > _MAX_CASES:
                raise ValueError("A/B dataset exceeds 1000 cases")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid A/B dataset item on line {line_number}: {exc}") from exc
    if not cases:
        raise ValueError("A/B dataset is empty")
    return tuple(cases)


def _path_free_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_PROVENANCE_ID.fullmatch(value) is None:
        raise ValueError(f"A/B backend {field} must be a path-free identifier")
    return value


def _model_identifier(value: object) -> str:
    if not isinstance(value, str) or _SAFE_MODEL_ID.fullmatch(value) is None:
        raise ValueError("A/B backend model_id must be a path-free model identifier")
    return value


def _build_backends(data: dict[str, Any]) -> dict[str, ABBackend]:
    if not isinstance(data, dict):
        raise ValueError("A/B backends must be an object")
    backends: dict[str, ABBackend] = {}
    for name, raw in data.items():
        if (
            not isinstance(name, str)
            or _SAFE_ID.fullmatch(name) is None
            or not isinstance(raw, dict)
        ):
            raise ValueError("A/B backend name or configuration is invalid")
        enabled = raw.get("enabled", False)
        if type(enabled) is not bool:
            raise ValueError(f"backend {name!r} enabled must be boolean")
        if not enabled:
            continue
        kind = raw.get("kind")
        if kind == "mock":
            if set(raw) != {"enabled", "kind"}:
                raise ValueError(f"backend {name!r} mock fields are invalid")
            backends[name] = ABBackend(
                adapter=MockVLMAdapter(),
                model_id="builtin/mock",
                model_revision="builtin_mock_v1",
                prompt_version=PROMPT_VERSION,
                runtime_id="stdlib",
            )
        elif kind == "command":
            expected = {
                "enabled",
                "kind",
                "command",
                "model_id",
                "revision",
                "prompt_version",
                "runtime_id",
            }
            if set(raw) != expected:
                raise ValueError(f"backend {name!r} command fields are invalid")
            command = raw.get("command")
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                raise ValueError(f"backend {name!r} requires a string command array")
            backends[name] = ABBackend(
                adapter=CommandVLMAdapter(name=name, command=command),
                model_id=_model_identifier(raw["model_id"]),
                model_revision=_path_free_identifier(
                    raw["revision"],
                    field="revision",
                ),
                prompt_version=_path_free_identifier(
                    raw["prompt_version"],
                    field="prompt_version",
                ),
                runtime_id=_path_free_identifier(
                    raw["runtime_id"],
                    field="runtime_id",
                ),
            )
        else:
            raise ValueError(f"backend {name!r} has unsupported kind {kind!r}")
    if not backends:
        raise ValueError("no A/B backend is enabled")
    return backends


def _evidence_id(case: ABCase, reference: str) -> str | None:
    exact = {frame.path: frame.frame_id for frame in case.frames}
    if reference in exact:
        return exact[reference]
    if reference == Path(reference).name:
        by_name = {Path(frame.path).name: frame.frame_id for frame in case.frames}
        return by_name.get(reference)
    return None


def _sanitize_result(result: VLMEvent, case: ABCase) -> tuple[dict[str, Any], bool]:
    payload = result.to_dict()
    root_ids = tuple(_evidence_id(case, reference) for reference in result.evidence_frames)
    evidence_compliant = bool(root_ids) and all(value is not None for value in root_ids)
    payload["evidence_frames"] = [value or "unbound" for value in root_ids]
    root_set = {value for value in root_ids if value is not None}
    for subject_payload, subject in zip(payload["subjects"], result.subjects, strict=True):
        subject_ids = tuple(_evidence_id(case, reference) for reference in subject.evidence_frames)
        if (
            not subject_ids
            or any(value is None for value in subject_ids)
            or not {value for value in subject_ids if value is not None}.issubset(root_set)
        ):
            evidence_compliant = False
        subject_payload["evidence_frames"] = [value or "unbound" for value in subject_ids]
    return payload, evidence_compliant


def _score(
    result: VLMEvent,
    expected: ABExpected,
    *,
    evidence_compliant: bool,
) -> dict[str, Any]:
    allowed_observations = set(expected.allowed_observations)
    allowed_inferences = set(expected.allowed_inferences)
    allowed_subjects = set(expected.allowed_subjects)
    observations = {observation.strip() for observation in result.observations}
    inferences = {inference.strip() for inference in result.inferences}
    observation_group_hits = sum(
        bool(observations & set(group)) for group in expected.required_observation_groups
    )
    inference_group_hits = sum(
        bool(inferences & set(group)) for group in expected.required_inference_groups
    )
    return {
        "event_type_match": result.event_type is expected.event_type,
        "risk_level_match": result.risk_level is expected.risk_level,
        "evidence_compliant": evidence_compliant,
        "unsupported_subject_count": sum(
            1 for subject in result.subjects if subject.kind not in allowed_subjects
        ),
        "unsupported_observation_count": sum(
            1
            for observation in result.observations
            if observation.strip() not in allowed_observations
        ),
        "unsupported_inference_count": sum(
            1 for inference in result.inferences if inference.strip() not in allowed_inferences
        ),
        "required_observation_group_hits": observation_group_hits,
        "required_observation_group_total": len(expected.required_observation_groups),
        "required_inference_group_hits": inference_group_hits,
        "required_inference_group_total": len(expected.required_inference_groups),
        "hallucination_metric_note": (
            "exact-match proxy against human-authored allowed subjects/facts/inferences; "
            "manual review is still required"
        ),
    }


def run_ab(config_path: str | Path) -> ABRunArtifacts:
    config_file = Path(config_path)
    with config_file.open("rb") as handle:
        config = tomllib.load(handle)
    run_settings = config.get("run", {})
    dataset_path = Path(run_settings["dataset"]).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = config_file.resolve().parent / dataset_path
    output_dir = Path(run_settings.get("output_dir", "./artifacts/vlm-ab")).expanduser()
    if not output_dir.is_absolute():
        output_dir = config_file.resolve().parent / output_dir
    poll_interval = float(run_settings.get("poll_interval_seconds", 0.25))
    cases = load_dataset(dataset_path)
    dataset_sha256 = _sha256_file(dataset_path.resolve())
    case_schema_versions = sorted({case.schema_version for case in cases})
    backends = _build_backends(config.get("backends", {}))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    results_path = output_dir / f"{timestamp}_results.jsonl"
    summary_path = output_dir / f"{timestamp}_summary.md"

    result_rows: list[dict[str, Any]] = []
    # This loop is deliberately sequential: one backend/case invocation at a time.
    for backend_name, backend in backends.items():
        for case in cases:
            candidate = CandidateEvent(
                event_id=case.case_id,
                camera_id="ab_fixture",
                start_seconds=0.0,
                end_seconds=0.0,
                detections=(),
            )
            request = VLMRequest(
                candidate=candidate,
                media=EventMedia(clip_path="", frame_paths=case.frame_paths),
                question=case.question,
            )
            sampler = ResourceSampler(poll_interval)
            start = time.perf_counter()
            row: dict[str, Any] = {
                "version": 1,
                "backend": backend_name,
                "provenance": backend.provenance(),
                "case_id": case.case_id,
                "case_schema_version": case.schema_version,
                "dataset_sha256": dataset_sha256,
                "json_compliant": False,
                "error": None,
                "error_code": None,
                "recognition_metrics": None,
                "oom_status": "unknown",
            }
            try:
                with sampler:
                    analysis = backend.adapter.analyze_with_metrics(request)
                result = analysis.result
                sanitized_result, evidence_compliant = _sanitize_result(result, case)
                row.update(
                    {
                        "json_compliant": True,
                        "recognition_metrics": analysis.metrics.to_dict(),
                        "oom_status": "not_observed",
                        "result": sanitized_result,
                        "score": _score(
                            result,
                            case.expected,
                            evidence_compliant=evidence_compliant,
                        ),
                    }
                )
            # Record a failed case while allowing the remaining sequential comparisons to run.
            except Exception as exc:
                row["error"] = type(exc).__name__
                if isinstance(exc, VLMError):
                    row["error_code"] = exc.code
                    row["oom_status"] = exc.oom_status
                    if exc.metrics is not None:
                        row["recognition_metrics"] = exc.metrics.to_dict()
            row["latency_seconds"] = round(time.perf_counter() - start, 4)
            row["resources"] = sampler.result().to_dict()
            result_rows.append(row)
            gc.collect()

    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result_rows),
        encoding="utf-8",
    )
    lines = [
        "# VLM A/B 测试摘要",
        "",
        f"> 数据集 SHA-256：`{dataset_sha256}`；案例 Schema："
        f"{', '.join(str(value) for value in case_schema_versions)}。",
        "",
        "> 后端顺序执行；显存为整块 GPU 的采样值，内存为整机/当前编排进程采样值。"
        "幻觉指标是人工允许事实的精确匹配代理，仍需人工看片复核。",
        "",
        "| 后端 | 用例数 | JSON 合规率 | 失败数/OOM未知数 | 证据合规率 | "
        "平均端到端(s) | 平均模型尝试(s) | 子进程RSS峰值(MiB) | GPU峰值(MiB) | "
        "系统内存峰值(MiB) | 全案例证据/事件/风险命中率 | 事实/推测覆盖率 | "
        "不支持事实/推测/主体 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend_name in backends:
        rows = [row for row in result_rows if row["backend"] == backend_name]
        compliant = sum(bool(row["json_compliant"]) for row in rows)
        latencies = [float(row["latency_seconds"]) for row in rows]
        scored = [row.get("score", {}) for row in rows]
        event_values = [bool(item.get("event_type_match", False)) for item in scored]
        risk_values = [bool(item.get("risk_level_match", False)) for item in scored]
        evidence_values = [bool(item.get("evidence_compliant", False)) for item in scored]
        unsupported_observations = sum(
            int(item.get("unsupported_observation_count", 0)) for item in scored
        )
        unsupported_inferences = sum(
            int(item.get("unsupported_inference_count", 0)) for item in scored
        )
        unsupported_subjects = sum(int(item.get("unsupported_subject_count", 0)) for item in scored)
        recognition_metrics = [
            row["recognition_metrics"]
            for row in rows
            if isinstance(row.get("recognition_metrics"), dict)
        ]
        model_durations = [
            int(metrics["duration_ms"])
            for metrics in recognition_metrics
            if type(metrics.get("duration_ms")) is int
        ]
        rss_peaks = [
            int(metrics["process_rss_peak_kib"]) / 1024
            for metrics in recognition_metrics
            if type(metrics.get("process_rss_peak_kib")) is int
        ]
        gpu_peaks = [
            int(metrics["gpu_device_memory_peak_mib"])
            for metrics in recognition_metrics
            if type(metrics.get("gpu_device_memory_peak_mib")) is int
        ]
        system_peaks = [
            float(row["resources"]["system_used_peak_mb"])
            for row in rows
            if isinstance(row.get("resources"), dict)
            and type(row["resources"].get("system_used_peak_mb")) in {int, float}
        ]
        failure_count = len(rows) - compliant
        oom_unknown_count = sum(row.get("oom_status") == "unknown" for row in rows)
        observation_group_hits = sum(
            int(item.get("required_observation_group_hits", 0)) for item in scored
        )
        observation_group_total = sum(
            len(case.expected.required_observation_groups) for case in cases
        )
        inference_group_hits = sum(
            int(item.get("required_inference_group_hits", 0)) for item in scored
        )
        inference_group_total = sum(len(case.expected.required_inference_groups) for case in cases)

        def rate(values: list[bool]) -> str:
            return f"{sum(values) / len(values):.1%}" if values else "n/a"

        def average(values: list[float | int], *, scale: float = 1.0) -> str:
            if not values:
                return "n/a"
            return f"{sum(values) / len(values) / scale:.3f}"

        def maximum(values: list[float | int]) -> str:
            return f"{max(values):.1f}" if values else "n/a"

        def coverage(hits: int, total: int) -> str:
            return f"{hits}/{total} ({hits / total:.1%})" if total else "n/a"

        lines.append(
            f"| {backend_name} | {len(rows)} | {compliant / len(rows):.1%} | "
            f"{failure_count}/{oom_unknown_count} | {rate(evidence_values)} | "
            f"{sum(latencies) / len(latencies):.3f} | "
            f"{average(model_durations, scale=1000)} | {maximum(rss_peaks)} | "
            f"{maximum(gpu_peaks)} | {maximum(system_peaks)} | "
            f"{rate(evidence_values)}/{rate(event_values)}/{rate(risk_values)} | "
            f"{coverage(observation_group_hits, observation_group_total)}/"
            f"{coverage(inference_group_hits, inference_group_total)} | "
            f"{unsupported_observations}/{unsupported_inferences}/{unsupported_subjects} |"
        )
    lines.extend(["", f"原始结果：`{results_path.name}`", ""])
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return ABRunArtifacts(
        results_path=results_path,
        summary_path=summary_path,
        backend_count=len(backends),
        case_count=len(cases),
    )
