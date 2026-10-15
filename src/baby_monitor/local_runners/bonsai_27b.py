"""Experimental single-frame Bonsai runner; no downloader or remote fallback.

Run through CommandVLMAdapter (which owns the cross-process GPU gate), inside
``unshare --user --map-root-user --net``. Children stay in the adapter's process
group so its timeout kills the whole tree. This module never launches unshare,
changes host networking, or authorizes household input by itself.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import socket
import stat
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from baby_monitor.local_runners import scene_probe, subject_probe, subject_state_probe
from baby_monitor.local_runners.common import (
    LocalVLMRequest,
    LocalVLMRunnerError,
    build_constrained_schema,
    build_prompt,
    load_request,
    report_runner_failure,
    runner_failure_status,
    validate_result,
)

MODEL_REVISION = "b072e1d3b35a0a630cece372c2127528e0994386"
RUNTIME_REVISION = "adfffbe41b2cabcd51fff326ab045662265062bb"
PROMPT_VERSION = "bonsai_single_frame_v1_20260929"
INDEPENDENT_PROMPT_VERSION = "bonsai_independent_single_frame_v1_20261002"
MODEL_ASSETS = (
    (
        "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        5_946_648_928,
        "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
    ),
    (
        "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf",
        629_246_976,
        "6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903",
    ),
)
MAX_FRAME_BYTES = 10 * 1024 * 1024
MAX_RESPONSE_BYTES = 262_144
ENDPOINT = "http://127.0.0.1:18791"


class BonsaiResponseError(LocalVLMRunnerError):
    """Closed-set diagnostics, never raw model output or private exception text."""

    def __init__(self, code: str):
        if code not in {
            "bonsai_response_incomplete",
            "bonsai_output_token_limit",
            "bonsai_invalid_envelope",
            "bonsai_invalid_response_json",
            "bonsai_server_load_failed",
            "bonsai_server_load_timeout",
            "bonsai_server_exited",
        }:
            raise ValueError("invalid Bonsai diagnostic code")
        self.code = code
        super().__init__(code)


def strict_json(raw: str | bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("non-finite JSON")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)


def _directory(path: Path) -> None:
    if not path.is_absolute() or path.resolve() != path or not path.is_dir():
        raise LocalVLMRunnerError("local asset directory is missing or unsafe")


def _verify_file(path: Path, size: int, digest: str) -> None:
    if type(size) is not int or size <= 0 or not isinstance(digest, str):
        raise LocalVLMRunnerError("asset metadata is invalid")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise LocalVLMRunnerError("asset digest is invalid")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise LocalVLMRunnerError("asset size or type does not match")
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
        if actual != digest or (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise LocalVLMRunnerError("asset integrity check failed")


@dataclass(frozen=True, slots=True)
class BonsaiSettings:
    model_directory: Path
    runtime_directory: Path
    runtime_manifest: Path
    cuda_library_directories: tuple[Path, ...]
    max_new_tokens: int = 512
    request_timeout_seconds: int = 120
    independent_visual_review: bool = False
    context_tokens: int = 4096
    image_tokens: int = 1024

    def __post_init__(self):
        if type(self.independent_visual_review) is not bool:
            raise ValueError("Bonsai independent visual review setting must be boolean")
        for value, minimum, maximum in (
            (self.max_new_tokens, 64, 1024),
            (self.request_timeout_seconds, 1, 120),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("Bonsai generation budget is invalid")
        if (
            type(self.context_tokens) is not int
            or self.context_tokens not in (4096, 8192)
            or type(self.image_tokens) is not int
            or self.image_tokens not in (1024, 2048, 4096)
            or self.image_tokens + self.max_new_tokens + 1024 >= self.context_tokens
        ):
            raise ValueError("Bonsai input budget is invalid")
        if not 1 <= len(self.cuda_library_directories) <= 4:
            raise ValueError("Bonsai requires explicit local CUDA library directories")
        for path in (
            self.model_directory,
            self.runtime_directory,
            *self.cuda_library_directories,
        ):
            _directory(path)
        if (
            not self.runtime_manifest.is_absolute()
            or self.runtime_manifest.resolve() != self.runtime_manifest
            or not self.runtime_manifest.is_file()
        ):
            raise ValueError("runtime manifest must be an absolute regular file")

    def verify_assets(self) -> None:
        for name, size, digest in MODEL_ASSETS:
            _verify_file(self.model_directory / name, size, digest)
        fd = os.open(self.runtime_manifest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise LocalVLMRunnerError("runtime manifest is not a regular file")
            raw = stream.read(65_537)
        if len(raw) > 65_536:
            raise LocalVLMRunnerError("runtime manifest exceeds size limit")
        manifest = strict_json(raw)
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"version", "revision", "files", "symlinks"}
            or type(manifest["version"]) is not int
            or manifest["version"] != 1
            or manifest["revision"] != RUNTIME_REVISION
        ):
            raise LocalVLMRunnerError("runtime manifest is invalid")
        files, links = manifest["files"], manifest["symlinks"]
        if (
            not isinstance(files, dict)
            or not isinstance(links, dict)
            or not 1 <= len(files) + len(links) <= 128
            or set(files) & set(links)
            or "llama-server" not in files
        ):
            raise LocalVLMRunnerError("runtime member list is invalid")
        names = set(files) | set(links)
        if any(
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            for name in names
        ):
            raise LocalVLMRunnerError("runtime member name is invalid")
        if {path.name for path in self.runtime_directory.iterdir()} != names:
            raise LocalVLMRunnerError("runtime contents changed")
        for name, entry in files.items():
            if not isinstance(entry, dict) or set(entry) != {"bytes", "sha256"}:
                raise LocalVLMRunnerError("runtime file record is invalid")
            _verify_file(self.runtime_directory / name, entry["bytes"], entry["sha256"])
        for name, target in links.items():
            path = self.runtime_directory / name
            if (
                not isinstance(target, str)
                or target not in names
                or not path.is_symlink()
                or os.readlink(path) != target
                or path.resolve().parent != self.runtime_directory
                or path.resolve().name not in files
            ):
                raise LocalVLMRunnerError("runtime link changed")


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    # Bound native decoder allocation before model launch. This header check is
    # not full image validation; the pinned decoder still validates the pixels.
    position = 2
    start_of_frame = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while position < len(data):
        if data[position] != 0xFF:
            break
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker in {0xD9, 0xDA}:
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(data):
            break
        length = int.from_bytes(data[position : position + 2], "big")
        if length < 2 or position + length > len(data):
            break
        if marker in start_of_frame:
            if length < 8:
                break
            height, width = struct.unpack(">HH", data[position + 3 : position + 7])
            return width, height
        position += length
    raise LocalVLMRunnerError("JPEG dimensions are unavailable")


def _frame(request: LocalVLMRequest) -> tuple[bytes, str]:
    # Never silently select one out of a multi-frame request: that would imply
    # inspection of evidence this experimental runner actually omitted.
    if len(request.frame_paths) != 1:
        raise LocalVLMRunnerError("Bonsai currently requires exactly one frame")
    path = request.frame_paths[0]
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_FRAME_BYTES:
            raise LocalVLMRunnerError("frame size or type is invalid")
        data = stream.read(MAX_FRAME_BYTES + 1)
    if len(data) != info.st_size or len(data) > MAX_FRAME_BYTES:
        raise LocalVLMRunnerError("frame changed or exceeds size limit")
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        if data[12:16] != b"IHDR" or not 0 < width * height <= 16_000_000:
            raise LocalVLMRunnerError("PNG dimensions exceed input budget")
        return data, "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        width, height = _jpeg_dimensions(data)
        if not 0 < width * height <= 16_000_000:
            raise LocalVLMRunnerError("JPEG dimensions exceed input budget")
        return data, "image/jpeg"
    raise LocalVLMRunnerError("only admitted PNG or JPEG frames are supported")


def prompt_revision(settings: BonsaiSettings) -> str:
    return INDEPENDENT_PROMPT_VERSION if settings.independent_visual_review else PROMPT_VERSION


def build_payload(settings: BonsaiSettings, request: LocalVLMRequest) -> dict[str, Any]:
    data, mime = _frame(request)
    prompt_request = request
    if settings.independent_visual_review:
        # Hide upstream hints only in the model's prompt. Retain the original
        # request, original pixels, evidence IDs and result schema unchanged.
        prompt_request = replace(
            request, candidate={**request.candidate, "detections": [], "signals": []}
        )
    prompt = build_prompt(prompt_request) + (
        "\n本轮只有一张静态图。请简洁回答：observations 最多四条，只写与人物存在、可见位置、"
        "遮挡和直接可见物体有关的内容；不转录招牌文字，不猜背景用途、人物身份或行为意图。"
        "event_type=unknown 时仍保留可见事实，不把未知当无人或安全。"
    )
    if settings.independent_visual_review:
        prompt += (
            "\n本轮是独立视觉复核：未提供任何上游检测或系统信号，前面的空列表不表示无人。"
            "请先只依原图像素独立判断是否有直接可见的实体人物，不能复述、猜测或引用检测器"
            "类别、分数和框；你的置信度只能反映这张图的视觉证据。显示器中的人物、倒影、"
            "衣物或其他类人形状不能自动算作现场实体人物，不能确定时保留 unknown。"
            "年龄无法从清晰可见证据确定时，subject.kind 使用 person，不猜 baby 或 adult。"
            "单张静态图不能确认睡眠；睡眠状态保持 unknown，不能因躺卧、闭眼或静止而确认 sleeping。"
        )
    return {
        "model": "local-bonsai",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": settings.max_new_tokens,
        "stream": False,
        "reasoning_effort": "none",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "vlm_event",
                "strict": True,
                "schema": build_constrained_schema(request),
            },
        },
    }


def build_subject_probe_payload(
    settings: BonsaiSettings, request: LocalVLMRequest, *, state: bool = False
) -> dict[str, Any]:
    """Explicit diagnostic task, with no upstream labels or private question text.

    This does not replace the event contract and is never selected implicitly.
    The existing image admission and generation limits still apply.
    """
    payload = build_payload(settings, request)
    contract = subject_state_probe if state else subject_probe
    payload["messages"][0]["content"][1]["text"] = contract.PROMPT
    payload["response_format"]["json_schema"] = {
        "name": "subject_probe",
        "strict": True,
        "schema": contract.schema(),
    }
    payload["cache_prompt"] = False
    return payload


def build_scene_probe_payload(settings: BonsaiSettings, request: LocalVLMRequest) -> dict[str, Any]:
    """Explicit whole-image diagnostic; leave normal event/subject tasks unchanged."""
    payload = build_payload(settings, request)
    payload["messages"][0]["content"][1]["text"] = scene_probe.PROMPT
    payload["response_format"]["json_schema"] = {
        "name": "scene_visibility",
        "strict": True,
        "schema": scene_probe.schema(),
    }
    payload["cache_prompt"] = False
    return payload


def parse_response(value: Any, request: LocalVLMRequest) -> str:
    if not isinstance(value, dict) or "error" in value:
        raise BonsaiResponseError("bonsai_invalid_envelope")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise BonsaiResponseError("bonsai_invalid_envelope")
    choice = choices[0]
    if isinstance(choice, dict) and choice.get("finish_reason") == "length":
        raise BonsaiResponseError("bonsai_output_token_limit")
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise BonsaiResponseError("bonsai_response_incomplete")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("tool_calls"):
        raise LocalVLMRunnerError("Bonsai response is not a final answer")
    raw = message.get("content")
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise LocalVLMRunnerError("Bonsai result is missing or exceeds size limit")
    try:
        strict_json(raw)
    except ValueError:
        raise BonsaiResponseError("bonsai_invalid_response_json") from None
    return validate_result(raw, allowed_frame_ids=request.frame_ids).to_json()


def response_diagnostics(
    value: Any, settings: BonsaiSettings
) -> dict[str, int | float | bool | str | None]:
    """Keep only bounded numeric native metadata, never model text or prompt logs.

    Missing/invalid native fields remain unknown. Prompt usage may include image
    tokens; the configured image budget is not a measurement of those tokens.
    """
    envelope = value if isinstance(value, dict) else {}
    usage = envelope.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    timings = envelope.get("timings")
    timings = timings if isinstance(timings, dict) else {}

    def count(value):
        return value if type(value) is int and 0 <= value <= 1_000_000 else None

    def milliseconds(value):
        if type(value) not in (int, float) or not 0 <= value <= 600_000 or not math.isfinite(value):
            return None
        return float(value)

    choices = envelope.get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
    stop_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if not isinstance(stop_reason, str) or stop_reason not in (
        "stop",
        "length",
        "tool_calls",
        "content_filter",
        "function_call",
    ):
        stop_reason = None

    return {
        "context_tokens_configured": settings.context_tokens,
        "image_tokens_configured": settings.image_tokens,
        "max_new_tokens_configured": settings.max_new_tokens,
        "prompt_tokens_reported": count(usage.get("prompt_tokens")),
        "completion_tokens_reported": count(usage.get("completion_tokens")),
        "total_tokens_reported": count(usage.get("total_tokens")),
        "cached_tokens_reported": count(details.get("cached_tokens")),
        "image_tokens_reported": count(details.get("image_tokens")),
        "prompt_tokens_timed": count(timings.get("prompt_n")),
        "cached_tokens_timed": count(timings.get("cache_n")),
        "predicted_tokens_timed": count(timings.get("predicted_n")),
        "prompt_ms_reported": milliseconds(timings.get("prompt_ms")),
        "predicted_ms_reported": milliseconds(timings.get("predicted_ms")),
        # Native `truncated` denotes generation-time context exhaustion, not
        # silently trimmed input. Plain OpenAI responses normally omit it.
        "context_truncated_reported": envelope.get("truncated")
        if type(envelope.get("truncated")) is bool
        else None,
        "finish_reason_reported": stop_reason,
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise LocalVLMRunnerError("local model redirects are not allowed")


def _http(endpoint: str, payload: dict | None, timeout: float) -> Any:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT + endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    with opener.open(request, timeout=timeout) as response:
        data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise LocalVLMRunnerError("local model response exceeds size limit")
    return strict_json(data)


def _isolated_loopback() -> None:
    if sys.platform != "linux" or {name for _, name in socket.if_nameindex()} != {"lo"}:
        raise LocalVLMRunnerError("an isolated Linux network namespace is required")
    subprocess.run(
        ["/usr/sbin/ip", "link", "set", "lo", "up"],
        check=True,
        timeout=5,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _server_command(settings: BonsaiSettings) -> list[str]:
    return [
        str(settings.runtime_directory / "llama-server"),
        "-m",
        str(settings.model_directory / MODEL_ASSETS[0][0]),
        "--mmproj",
        str(settings.model_directory / MODEL_ASSETS[1][0]),
        "--host",
        "127.0.0.1",
        "--port",
        "18791",
        "-ngl",
        "99",
        "-fa",
        "on",
        "-c",
        str(settings.context_tokens),
        "-np",
        "1",
        "--fit",
        "off",
        "--offline",
        "--no-webui",
        "--no-webui-mcp-proxy",
        "--jinja",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--image-min-tokens",
        str(settings.image_tokens),
        "--image-max-tokens",
        str(settings.image_tokens),
        "-b",
        "512",
        "-ub",
        "256",
    ]


def _environment(settings: BonsaiSettings) -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "CUDA_VISIBLE_DEVICES"}
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result["LD_LIBRARY_PATH"] = ":".join(
        str(path) for path in (settings.runtime_directory, *settings.cuda_library_directories)
    )
    return result


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class BonsaiSession:
    """One verified offline server; callers own the lifetime-wide GPU gate.

    Requests remain independent single-image conversations. No prior messages or
    request payloads are retained here, and failed startup always reaps its child.
    """

    def __init__(
        self,
        settings: BonsaiSettings,
        *,
        disable_prompt_cache: bool = True,
        execution_lock_fd: int | None = None,
    ):
        if type(disable_prompt_cache) is not bool:
            raise ValueError("invalid Bonsai prompt cache setting")
        self.settings = settings
        self.disable_prompt_cache = disable_prompt_cache
        self.execution_lock_fd = execution_lock_fd
        self._validate_execution_lock()
        self.process: subprocess.Popen | None = None
        self.last_diagnostics: dict[str, int | float | bool | str | None] | None = None
        self._entered = False

    def _validate_execution_lock(self) -> None:
        if self.execution_lock_fd is None:
            return  # Legacy one-shot callers retain their existing parent gate.
        if type(self.execution_lock_fd) is not int or self.execution_lock_fd < 0:
            raise ValueError("invalid Bonsai execution lock descriptor")
        metadata = os.fstat(self.execution_lock_fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
        ):
            raise ValueError("unsafe Bonsai execution lock descriptor")

    def __enter__(self) -> BonsaiSession:
        if self._entered:
            raise LocalVLMRunnerError("Bonsai session cannot be reused")
        self._entered = True
        self._validate_execution_lock()
        _isolated_loopback()
        self.settings.verify_assets()
        self.process = subprocess.Popen(
            _server_command(self.settings),
            env=_environment(self.settings),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=False,
            # The server must retain the same open-file-description lease even
            # if either Python parent exits; never unlock or close the caller's fd.
            pass_fds=() if self.execution_lock_fd is None else (self.execution_lock_fd,),
        )
        try:
            self._wait_ready()
        except BaseException:
            self.close()
            raise
        return self

    def _wait_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise BonsaiResponseError("bonsai_server_load_failed")
            try:
                health = _http("/health", None, 1)
                if isinstance(health, dict) and health.get("status") == "ok":
                    break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(0.1)
        else:
            raise BonsaiResponseError("bonsai_server_load_timeout")

    def count_input_tokens(self, request: LocalVLMRequest) -> int:
        """Explicit diagnostic call using the pinned server's multimodal counter.

        Counts the same rendered chat input through the multimodal placeholder
        path, without visual encoding, token strings or answer generation. This
        is not separate image-token telemetry. Normal generation never calls it.
        """
        if self.process is None or self.process.poll() is not None:
            raise BonsaiResponseError("bonsai_server_exited")
        payload = build_payload(self.settings, request)
        if self.disable_prompt_cache:
            payload["cache_prompt"] = False
        value = _http(
            "/v1/chat/completions/input_tokens", payload, self.settings.request_timeout_seconds
        )
        if self.process.poll() is not None:
            raise BonsaiResponseError("bonsai_server_exited")
        if (
            not isinstance(value, dict)
            or value.get("object") != "response.input_tokens"
            or type(value.get("input_tokens")) is not int
            or not 1 <= value["input_tokens"] <= 1_000_000
        ):
            raise LocalVLMRunnerError("invalid native input token diagnostic")
        return value["input_tokens"]

    def generate(self, request: LocalVLMRequest) -> str:
        self.last_diagnostics = None
        if self.process is None or self.process.poll() is not None:
            raise BonsaiResponseError("bonsai_server_exited")
        payload = build_payload(self.settings, request)
        if self.disable_prompt_cache:
            payload["cache_prompt"] = False
        response = _http("/v1/chat/completions", payload, self.settings.request_timeout_seconds)
        self.last_diagnostics = response_diagnostics(response, self.settings)
        if self.process.poll() is not None:
            raise BonsaiResponseError("bonsai_server_exited")
        return parse_response(response, request)

    def probe_subjects(self, request: LocalVLMRequest) -> dict[str, Any]:
        return self._probe_people(request, state=False)

    def probe_subject_state(self, request: LocalVLMRequest) -> dict[str, Any]:
        return self._probe_people(request, state=True)

    def _probe_people(self, request: LocalVLMRequest, *, state: bool) -> dict[str, Any]:
        """One explicit independent diagnostic call on the already leased model.

        Returned coordinates bind to the exact submitted bytes, not a later file
        read. There is no conversion to VLMEvent, identity, room or sleep state.
        The caller owns camera/time/source bindings and any crop transforms.
        """
        self.last_diagnostics = None
        if self.process is None or self.process.poll() is not None:
            raise BonsaiResponseError("bonsai_server_exited")
        contract = subject_state_probe if state else subject_probe
        payload = build_subject_probe_payload(self.settings, request, state=state)
        image_url = payload["messages"][0]["content"][0]["image_url"]["url"]
        image_sha256 = hashlib.sha256(
            base64.b64decode(image_url.split(",", 1)[1], validate=True)
        ).hexdigest()
        response = _http("/v1/chat/completions", payload, self.settings.request_timeout_seconds)
        self.last_diagnostics = response_diagnostics(response, self.settings)
        if self.process.poll() is not None:
            raise BonsaiResponseError("bonsai_server_exited")
        try:
            claims = contract.parse_response(response)
        except ValueError:
            raise LocalVLMRunnerError("invalid subject probe response") from None
        return {
            "version": 2 if state else 1,
            "prompt_revision": contract.PROMPT_VERSION,
            "semantics": "unverified_model_claims",
            "frame_id": request.frame_ids[0],
            "input_sha256": image_sha256,
            "coordinate_space": "submitted_image_thousandths",
            "claims": claims,
        }

    def probe_scene(self, request: LocalVLMRequest) -> dict[str, Any]:
        """Bound one independent scene claim to the actual submitted image bytes."""
        self.last_diagnostics = None
        if self.process is None or self.process.poll() is not None:
            raise BonsaiResponseError("bonsai_server_exited")
        payload = build_scene_probe_payload(self.settings, request)
        image_url = payload["messages"][0]["content"][0]["image_url"]["url"]
        image_sha256 = hashlib.sha256(
            base64.b64decode(image_url.split(",", 1)[1], validate=True)
        ).hexdigest()
        response = _http("/v1/chat/completions", payload, self.settings.request_timeout_seconds)
        self.last_diagnostics = response_diagnostics(response, self.settings)
        if self.process.poll() is not None:
            raise BonsaiResponseError("bonsai_server_exited")
        try:
            claims = scene_probe.parse_response(response)
        except ValueError:
            raise LocalVLMRunnerError("invalid scene probe response") from None
        return {
            "version": 1,
            "prompt_revision": scene_probe.PROMPT_VERSION,
            "semantics": "unverified_model_claims",
            "frame_id": request.frame_ids[0],
            "input_sha256": image_sha256,
            "claims": claims,
        }

    def close(self) -> None:
        if self.process is not None:
            process, self.process = self.process, None
            _stop_server(process)

    def __exit__(self, *_args) -> None:
        self.close()


def generate(settings: BonsaiSettings, request: LocalVLMRequest) -> str:
    # Preserve rejection of bad/multi-frame requests before loading the model.
    build_payload(settings, request)
    with BonsaiSession(settings, disable_prompt_cache=False) as session:
        return session.generate(request)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--runtime-directory", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--cuda-library-directory", type=Path, action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--image-tokens", type=int, default=1024)
    parser.add_argument("--independent-visual-review", action="store_true")
    args = parser.parse_args(argv)
    try:
        settings = BonsaiSettings(
            model_directory=args.model_directory,
            runtime_directory=args.runtime_directory,
            runtime_manifest=args.runtime_manifest,
            cuda_library_directories=tuple(args.cuda_library_directory),
            max_new_tokens=args.max_new_tokens,
            independent_visual_review=args.independent_visual_review,
            context_tokens=args.context_tokens,
            image_tokens=args.image_tokens,
        )
        result = generate(settings, load_request(args.request))
    except Exception as exc:
        if isinstance(exc, BonsaiResponseError):
            status = runner_failure_status(exc)
            status["error_code"] = exc.code
            print(json.dumps(status, separators=(",", ":")), file=sys.stderr)
        else:
            report_runner_failure(exc)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
