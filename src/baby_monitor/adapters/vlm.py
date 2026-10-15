from __future__ import annotations

import base64
import ipaddress
import json
import mimetypes
import os
import re
import shutil
import signal as signal_module
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from baby_monitor.adapters.vlm_execution import VLMExecutionError, local_vlm_execution
from baby_monitor.domain import (
    CandidateEvent,
    EventMedia,
    EventType,
    RecognitionProvenance,
    RecognitionRunMetrics,
    RiskLevel,
    SchemaError,
    Subject,
    SubjectKind,
    VLMEvent,
)
from baby_monitor.remote_review_port import (
    RemoteAttemptStatus,
    RemoteBudgetError,
    RemoteReviewAudit,
)

DEFAULT_QUESTION_ZH = (
    "只根据提供的关键帧判断。observations 只能写可见事实，inferences 只能写推测；"
    "证据不足时输出 unknown。"
)


_SAFE_VLM_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_RUNNER_FAILURE_PROTOCOL = "baby_monitor_local_runner_failure_v1"
_OOM_STATUSES = frozenset({"observed", "not_observed", "unknown"})


class VLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "vlm_error",
        oom_status: str = "unknown",
        metrics: RecognitionRunMetrics | None = None,
    ):
        if _SAFE_VLM_ERROR_CODE.fullmatch(code) is None:
            raise ValueError("VLM error code must be a safe identifier")
        if oom_status not in _OOM_STATUSES:
            raise ValueError("VLM OOM status is invalid")
        if metrics is not None and not isinstance(metrics, RecognitionRunMetrics):
            raise TypeError("VLM failure metrics are invalid")
        self.code = code
        self.oom_status = oom_status
        self.metrics = metrics
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class VLMRequest:
    candidate: CandidateEvent
    media: EventMedia
    question: str = DEFAULT_QUESTION_ZH

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "clip_path": self.media.clip_path,
            "frame_paths": list(self.media.frame_paths),
            "evidence": self.media.to_dict(),
            "question": self.question,
            "output_schema": VLMEvent.json_schema(),
        }


@dataclass(frozen=True, slots=True)
class VLMAnalysis:
    result: VLMEvent
    metrics: RecognitionRunMetrics


class VLMAdapter(ABC):
    name: str

    @property
    def provenance(self) -> RecognitionProvenance:
        return RecognitionProvenance(adapter_name=self.name)

    def analyze_with_metrics(self, request: VLMRequest) -> VLMAnalysis:
        started_ns = time.monotonic_ns()
        result = self.analyze(request)
        duration_ms = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
        return VLMAnalysis(
            result=result,
            metrics=RecognitionRunMetrics(
                duration_ms=duration_ms,
                json_compliant=True,
            ),
        )

    @abstractmethod
    def analyze(self, request: VLMRequest) -> VLMEvent:
        raise NotImplementedError


class MockVLMAdapter(VLMAdapter):
    """Conservative deterministic adapter for plumbing tests, not AI analysis."""

    name = "mock"

    def __init__(self, *, name: str = "mock"):
        self._provenance = RecognitionProvenance(
            adapter_name=name,
            model_revision="builtin_mock_v1",
            prompt_revision="baby_monitor_vlm_prompt_v3_coverage_unverified",
        )
        self.name = name

    @property
    def provenance(self) -> RecognitionProvenance:
        return getattr(
            self,
            "_provenance",
            RecognitionProvenance(
                adapter_name=self.name,
                model_revision="builtin_mock_v1",
                prompt_revision="baby_monitor_vlm_prompt_v3_coverage_unverified",
            ),
        )

    def analyze(self, request: VLMRequest) -> VLMEvent:
        labels = request.candidate.labels
        detection_observations = tuple(
            f"系统候选检测器记录到类别 {label!r}；这不是宝宝/成人身份结论。" for label in labels
        )
        signal_observations = tuple(
            f"系统信号适配器 {signal.source_adapter!r} 记录到 {signal.signal_type.value!r}；"
            "这不是视觉身份或最终事件结论。"
            for signal in request.candidate.signals
        )
        coverage_observations = (
            (
                "系统取证窗口存在覆盖缺口："
                + ", ".join(gap.reason for gap in request.media.coverage_gaps)
                + "；缺失时段不能由模型补全。",
            )
            if request.media.coverage_gaps
            else ("系统取证窗口未验证连续覆盖；未验证不等于已记录缺失，也不等于连续覆盖。",)
        )
        subjects: list[Subject] = []
        frame_paths = tuple(request.media.frame_paths)
        if "person" in labels:
            subjects.append(
                Subject(
                    kind=SubjectKind.PERSON,
                    confidence=max(item.confidence for item in request.candidate.detections),
                    evidence_frames=frame_paths[:1],
                )
            )
        if any(label in {"cat", "dog"} for label in labels):
            subjects.append(
                Subject(
                    kind=SubjectKind.PET,
                    confidence=0.5,
                    evidence_frames=frame_paths[:1],
                )
            )
        return VLMEvent(
            event_type=EventType.MOVEMENT if labels else EventType.UNKNOWN,
            subjects=tuple(subjects),
            observations=(detection_observations + signal_observations + coverage_observations),
            inferences=(),
            risk_level=RiskLevel.NONE,
            confidence=0.35,
            needs_remote_review=False,
            summary_zh="模拟适配器仅验证流程，未执行视觉语义判断。",
            evidence_frames=frame_paths,
        )


class _GPUDeviceMemorySampler:
    """Sample whole-device memory with one bounded nvidia-smi subprocess."""

    def __init__(self, *, device: str, enabled: bool):
        self.device = device
        self.executable = shutil.which("nvidia-smi") if enabled else None
        self.peak_mib: int | None = None
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None

    @staticmethod
    def _environment() -> dict[str, str]:
        allowed = {"LD_LIBRARY_PATH", "PATH", "SYSTEMROOT", "WINDIR"}
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        environment.update({"LANG": "C", "LC_ALL": "C"})
        return environment

    @staticmethod
    def _parse_sample(line: str) -> int | None:
        stripped = line.strip()
        if not stripped.isascii() or not stripped.isdigit():
            return None
        value = int(stripped)
        return value if 0 <= value <= 10_000_000 else None

    def _read_samples(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                value = self._parse_sample(line)
                if value is not None and (self.peak_mib is None or value > self.peak_mib):
                    self.peak_mib = value
        except (OSError, UnicodeError, ValueError):
            return

    def __enter__(self) -> _GPUDeviceMemorySampler:
        if self.executable is None:
            return self
        try:
            self._process = subprocess.Popen(
                [
                    self.executable,
                    f"--id={self.device}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "--loop-ms=100",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="strict",
                env=self._environment(),
                start_new_session=os.name == "posix",
            )
        except OSError:
            self._process = None
            return self
        self._reader = threading.Thread(
            target=self._read_samples,
            name="baby-monitor-gpu-sampler",
            daemon=True,
        )
        self._reader.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            with suppress(OSError):
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal_module.SIGKILL)
                else:
                    with suppress(OSError):
                        process.kill()
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=2)
        if self._reader is not None:
            self._reader.join(timeout=2)
        if process is not None and process.stdout is not None:
            with suppress(OSError):
                process.stdout.close()


class CommandVLMAdapter(VLMAdapter):
    """Run a local model wrapper as a subprocess and parse strict stdout JSON."""

    def __init__(
        self,
        *,
        name: str,
        command: Sequence[str],
        timeout_seconds: int = 180,
        model_revision: str | None = None,
        gpu_device: str = "0",
        prompt_revision: str = "baby_monitor_vlm_prompt_v2",
    ):
        if not command:
            raise ValueError("command cannot be empty")
        if sum(part.count("{request}") for part in command) != 1:
            raise ValueError("command must contain {request} exactly once")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if re.fullmatch(r"[0-9]{1,3}", gpu_device) is None:
            raise ValueError("gpu_device must be a numeric index")
        self._provenance = RecognitionProvenance(
            adapter_name=name,
            model_revision=model_revision,
            prompt_revision=prompt_revision,
        )
        self.name = name
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.gpu_device = gpu_device

    @property
    def provenance(self) -> RecognitionProvenance:
        return self._provenance

    @staticmethod
    def _parse_runner_failure_status(stderr: str) -> tuple[str, str] | None:
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        if not lines or len(lines[-1].encode("utf-8")) > 4096:
            return None
        try:
            value = json.loads(lines[-1])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(value, dict) or set(value) != {
            "protocol",
            "error_code",
            "oom_status",
        }:
            return None
        code = value["error_code"]
        oom_status = value["oom_status"]
        if (
            value["protocol"] != _RUNNER_FAILURE_PROTOCOL
            or not isinstance(code, str)
            or _SAFE_VLM_ERROR_CODE.fullmatch(code) is None
            or not isinstance(oom_status, str)
            or oom_status not in _OOM_STATUSES
        ):
            return None
        return code, oom_status

    @staticmethod
    def _parse_process_rss_peak(path: Path) -> int | None:
        try:
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 65_536:
                return None
            for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
                prefix = "Maximum resident set size (kbytes):"
                stripped = line.strip()
                if stripped.startswith(prefix):
                    value = int(stripped.removeprefix(prefix).strip())
                    return value if 0 <= value <= 1_000_000_000 else None
        except (OSError, UnicodeError, ValueError):
            return None
        return None

    @staticmethod
    def _timed_command(command: list[str], metrics_path: Path) -> list[str]:
        time_executable = Path("/usr/bin/time")
        if not time_executable.is_file() or not os.access(time_executable, os.X_OK):
            return command
        return [
            str(time_executable),
            "-v",
            "-o",
            str(metrics_path),
            "--",
            *command,
        ]

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal_module.SIGKILL)
        else:
            with suppress(OSError):
                process.kill()

    @staticmethod
    def _subprocess_environment() -> dict[str, str]:
        # Local runners get only runtime necessities, never arbitrary inherited
        # API keys, camera credentials, or provider-specific environment values.
        allowed = {
            "CUDA_VISIBLE_DEVICES",
            "HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "LD_LIBRARY_PATH",
            "PATH",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "WINDIR",
        }
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        environment.update(
            {
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        return environment

    def _terminate_runner(self, process: subprocess.Popen[str]) -> None:
        self._kill_process_tree(process)
        # A second interruption while draining pipes must not release the
        # execution gate before the killed model has actually been reaped.
        with suppress(BaseException):
            process.communicate()
        while True:
            try:
                process.wait()
                return
            except (KeyboardInterrupt, SystemExit, InterruptedError):
                continue

    def _analyze(
        self,
        request: VLMRequest,
        *,
        measure_process_rss: bool,
    ) -> tuple[VLMEvent, int | None, int | None]:
        try:
            with local_vlm_execution(self.gpu_device):
                return self._analyze_unlocked(request, measure_process_rss=measure_process_rss)
        except VLMExecutionError as exc:
            raise VLMError(str(exc), code=exc.code) from None

    def _analyze_unlocked(
        self,
        request: VLMRequest,
        *,
        measure_process_rss: bool,
    ) -> tuple[VLMEvent, int | None, int | None]:
        started_ns = time.monotonic_ns()
        with tempfile.TemporaryDirectory(prefix="baby-monitor-vlm-") as directory:
            request_path = Path(directory) / "request.json"
            metrics_path = Path(directory) / "process-metrics.txt"
            request_path.write_text(
                json.dumps(request.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            with suppress(OSError):
                request_path.chmod(0o600)
            command = [part.replace("{request}", str(request_path)) for part in self.command]
            executed_command = (
                self._timed_command(command, metrics_path) if measure_process_rss else command
            )
            environment = self._subprocess_environment()
            if executed_command != command:
                environment.update(
                    {
                        "LANG": "C",
                        "LC_ALL": "C",
                        "PYTHONUTF8": "1",
                    }
                )
            gpu_sampler = _GPUDeviceMemorySampler(
                device=self.gpu_device,
                enabled=measure_process_rss,
            )
            gpu_sampler.__enter__()

            def failure_metrics() -> RecognitionRunMetrics:
                return RecognitionRunMetrics(
                    duration_ms=min(
                        86_400_000,
                        max(0, (time.monotonic_ns() - started_ns) // 1_000_000),
                    ),
                    json_compliant=False,
                    process_rss_peak_kib=(
                        self._parse_process_rss_peak(metrics_path) if measure_process_rss else None
                    ),
                    gpu_device_memory_peak_mib=gpu_sampler.peak_mib,
                )

            try:
                try:
                    process = subprocess.Popen(
                        executed_command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=environment,
                        start_new_session=os.name == "posix",
                    )
                except OSError:
                    raise VLMError(
                        f"local VLM runner {self.name!r} could not start",
                        code="runner_start_failed",
                        metrics=failure_metrics(),
                    ) from None
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._terminate_runner(process)
                    raise VLMError(
                        f"local VLM runner {self.name!r} timed out",
                        code="runner_timeout",
                        metrics=failure_metrics(),
                    ) from None
                except BaseException:
                    self._terminate_runner(process)
                    raise
                if process.returncode != 0:
                    failure_status = self._parse_runner_failure_status(stderr)
                    raise VLMError(
                        f"local VLM runner {self.name!r} failed",
                        code=(failure_status[0] if failure_status is not None else "runner_failed"),
                        oom_status=(failure_status[1] if failure_status is not None else "unknown"),
                        metrics=failure_metrics(),
                    )
                if len(stdout.encode("utf-8")) > 1_048_576:
                    raise VLMError(
                        f"local VLM runner {self.name!r} returned too much output",
                        code="runner_output_too_large",
                        metrics=failure_metrics(),
                    )
                try:
                    result = VLMEvent.from_json(stdout.strip())
                except SchemaError:
                    raise VLMError(
                        f"local VLM runner {self.name!r} returned invalid structured output",
                        code="invalid_structured_output",
                        oom_status="not_observed",
                        metrics=failure_metrics(),
                    ) from None
                process_rss_peak_kib = (
                    self._parse_process_rss_peak(metrics_path) if measure_process_rss else None
                )
            finally:
                gpu_sampler.__exit__(None, None, None)
            return result, process_rss_peak_kib, gpu_sampler.peak_mib

    def analyze(self, request: VLMRequest) -> VLMEvent:
        result, _process_rss_peak_kib, _gpu_device_memory_peak_mib = self._analyze(
            request,
            measure_process_rss=False,
        )
        return result

    def analyze_with_metrics(self, request: VLMRequest) -> VLMAnalysis:
        started_ns = time.monotonic_ns()
        result, process_rss_peak_kib, gpu_device_memory_peak_mib = self._analyze(
            request,
            measure_process_rss=True,
        )
        duration_ms = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
        return VLMAnalysis(
            result=result,
            metrics=RecognitionRunMetrics(
                duration_ms=duration_ms,
                json_compliant=True,
                process_rss_peak_kib=process_rss_peak_kib,
                gpu_device_memory_peak_mib=gpu_device_memory_peak_mib,
            ),
        )


class MiniCPMVAdapter(CommandVLMAdapter):
    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: int = 180,
        model_revision: str | None = None,
        gpu_device: str = "0",
    ):
        super().__init__(
            name="minicpm_v",
            command=command,
            timeout_seconds=timeout_seconds,
            model_revision=model_revision,
            gpu_device=gpu_device,
        )


class MageVL4BAdapter(CommandVLMAdapter):
    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: int = 180,
        model_revision: str | None = None,
        gpu_device: str = "0",
    ):
        super().__init__(
            name="mage_vl_4b",
            command=command,
            timeout_seconds=timeout_seconds,
            model_revision=model_revision,
            gpu_device=gpu_device,
        )


@dataclass(frozen=True, slots=True)
class RemoteUploadPreview:
    target_origin: str
    model: str
    selected_frame_count: int
    selected_frame_bytes: int
    omitted_frame_count: int
    upload_enabled: bool
    upload_consent: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "target_origin": self.target_origin,
            "model": self.model,
            "selected_frame_count": self.selected_frame_count,
            "selected_frame_bytes": self.selected_frame_bytes,
            "omitted_frame_count": self.omitted_frame_count,
            "upload_enabled": self.upload_enabled,
            "upload_consent": self.upload_consent,
            "uploads_allowed": self.upload_enabled and self.upload_consent,
            "includes_video": False,
            "includes_audio": False,
            "includes_local_paths": False,
        }


@dataclass(frozen=True, slots=True)
class _RemoteFrame:
    alias: str
    path: str
    mime_type: str
    size_bytes: int


class _RemoteCallError(RuntimeError):
    def __init__(self, code: str, public_message: str):
        self.code = code
        super().__init__(public_message)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            request.full_url,
            code,
            "remote redirects are disabled",
            headers,
            file_pointer,
        )


class OpenAICompatibleVLMAdapter(VLMAdapter):
    name = "openai_compatible_remote"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        enabled: bool,
        upload_consent: bool,
        max_frames: int = 3,
        timeout_seconds: int = 60,
        max_upload_bytes: int = 5_242_880,
        max_response_bytes: int = 262_144,
        max_requests_per_day: int = 20,
        min_interval_seconds: int = 60,
        audit_store: RemoteReviewAudit | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.enabled = enabled
        self.upload_consent = upload_consent
        self.max_frames = max_frames
        self.timeout_seconds = timeout_seconds
        self.max_upload_bytes = max_upload_bytes
        self.max_response_bytes = max_response_bytes
        self.max_requests_per_day = max_requests_per_day
        self.min_interval_seconds = min_interval_seconds
        self.audit_store = audit_store

        if type(enabled) is not bool or type(upload_consent) is not bool:
            raise ValueError("remote upload switches must be booleans")
        if type(max_frames) is not int or not 1 <= max_frames <= 3:
            raise ValueError("remote max_frames must be between 1 and 3")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
            raise ValueError("remote timeout_seconds must be between 1 and 600")
        if type(max_upload_bytes) is not int or not 1 <= max_upload_bytes <= 52_428_800:
            raise ValueError("remote max_upload_bytes must be between 1 and 52428800")
        if type(max_response_bytes) is not int or not 1_024 <= max_response_bytes <= 1_048_576:
            raise ValueError("remote max_response_bytes must be between 1024 and 1048576")
        if type(max_requests_per_day) is not int or not 1 <= max_requests_per_day <= 10_000:
            raise ValueError("remote max_requests_per_day must be between 1 and 10000")
        if type(min_interval_seconds) is not int or not 0 <= min_interval_seconds <= 86_400:
            raise ValueError("remote min_interval_seconds must be between 0 and 86400")

    @property
    def uploads_allowed(self) -> bool:
        return self.enabled and self.upload_consent

    @property
    def provenance(self) -> RecognitionProvenance:
        return RecognitionProvenance(
            adapter_name=self.name,
            prompt_revision="baby_monitor_vlm_prompt_v3_coverage_unverified",
        )

    def _endpoint(self) -> tuple[str, str, str]:
        if not self.base_url or not self.model:
            raise VLMError("remote base URL and model must be configured")
        if len(self.base_url) > 2_048 or any(ord(char) < 32 for char in self.base_url):
            raise VLMError("remote base URL is invalid")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise VLMError("remote base URL must use HTTP or HTTPS")
        if parsed.username is not None or parsed.password is not None:
            raise VLMError("remote base URL must not contain credentials")
        if parsed.query or parsed.fragment:
            raise VLMError("remote base URL must not contain query or fragment data")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise VLMError("remote base URL port is invalid") from exc
        if parsed.scheme == "http":
            try:
                is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                is_loopback = False
            if not is_loopback:
                raise VLMError("plain HTTP remote review is allowed only on an IP loopback address")

        base_path = parsed.path.rstrip("/")
        endpoint_path = f"{base_path}/chat/completions"
        endpoint = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))
        target_origin = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        return endpoint, target_origin, endpoint_path

    @staticmethod
    def _selected_indexes(frame_count: int, limit: int) -> tuple[int, ...]:
        selected_count = min(frame_count, limit)
        if selected_count == 0:
            return ()
        if selected_count == 1:
            return (0,)
        denominator = selected_count - 1
        return tuple(
            (index * (frame_count - 1) + denominator // 2) // denominator
            for index in range(selected_count)
        )

    def _prepare_frames(self, frame_paths: Sequence[str]) -> tuple[_RemoteFrame, ...]:
        indexes = self._selected_indexes(len(frame_paths), self.max_frames)
        if not indexes:
            raise VLMError("remote review requires at least one evidence frame")
        frames: list[_RemoteFrame] = []
        total_bytes = 0
        for output_index, source_index in enumerate(indexes, start=1):
            path = Path(frame_paths[source_index])
            try:
                stat = path.stat()
            except OSError:
                raise VLMError("remote evidence frame is unavailable") from None
            if not path.is_file():
                raise VLMError("remote evidence frame is not a regular file")
            mime_type = mimetypes.guess_type(path.name)[0]
            if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise VLMError("remote evidence frame type is not allowed")
            total_bytes += stat.st_size
            if total_bytes > self.max_upload_bytes:
                raise VLMError("remote evidence exceeds the configured byte budget")
            frames.append(
                _RemoteFrame(
                    alias=f"frame_{output_index:03d}",
                    path=str(path),
                    mime_type=mime_type,
                    size_bytes=stat.st_size,
                )
            )
        return tuple(frames)

    def preview(self, request: VLMRequest) -> RemoteUploadPreview:
        _, target_origin, _ = self._endpoint()
        frames = self._prepare_frames(request.media.frame_paths)
        return RemoteUploadPreview(
            target_origin=target_origin,
            model=self.model,
            selected_frame_count=len(frames),
            selected_frame_bytes=sum(frame.size_bytes for frame in frames),
            omitted_frame_count=len(request.media.frame_paths) - len(frames),
            upload_enabled=self.enabled,
            upload_consent=self.upload_consent,
        )

    @staticmethod
    def _candidate_facts(request: VLMRequest) -> dict[str, object]:
        detections = [
            {
                "timestamp_seconds": detection.timestamp_seconds,
                "label": (
                    detection.label
                    if re.fullmatch(r"[a-z0-9_.-]{1,64}", detection.label)
                    else "unknown_label"
                ),
                "confidence": detection.confidence,
                "bbox_xyxy": list(detection.bbox_xyxy),
            }
            for detection in request.candidate.detections
        ]
        signals = []
        for signal in request.candidate.signals:
            attributes = dict(signal.attributes)
            signal_fact: dict[str, object] = {"signal_type": signal.signal_type.value}
            if attributes.get("severity_hint") in {"low", "medium", "high"}:
                signal_fact["severity_hint"] = attributes["severity_hint"]
            signals.append(signal_fact)
        return {
            "event_duration_seconds": (
                request.candidate.end_seconds - request.candidate.start_seconds
            ),
            "detections": detections,
            "signals": signals,
        }

    @staticmethod
    def _coverage_facts(request: VLMRequest) -> dict[str, object]:
        return {
            "coverage_status": request.media.coverage_status,
            "continuous_coverage_verified": False,
            "selected_segments": request.media.selected_segments,
            "gaps": [
                {
                    "reason": gap.reason,
                    "duration_seconds": (gap.ended_at - gap.started_at).total_seconds(),
                }
                for gap in request.media.coverage_gaps
            ],
        }

    @staticmethod
    def _image_data_url(frame: _RemoteFrame) -> str:
        try:
            raw = Path(frame.path).read_bytes()
        except OSError:
            raise _RemoteCallError(
                "media_unavailable",
                "remote evidence frame became unavailable",
            ) from None
        if len(raw) != frame.size_bytes:
            raise _RemoteCallError("media_changed", "remote evidence frame changed before upload")
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{frame.mime_type};base64,{encoded}"

    def _payload(self, request: VLMRequest, frames: tuple[_RemoteFrame, ...]) -> dict[str, object]:
        aliases = [frame.alias for frame in frames]
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    DEFAULT_QUESTION_ZH
                    + "\n系统候选事实（不是最终语义结论）："
                    + json.dumps(self._candidate_facts(request), ensure_ascii=False)
                    + "\n证据覆盖事实（未验证不等于已记录缺失，也不等于连续覆盖；"
                    "已记录缺口不得补全，未记录缺口不能证明没有缺口）："
                    + json.dumps(self._coverage_facts(request), ensure_ascii=False)
                    + "\n只可在 evidence_frames 中引用这些匿名帧编号："
                    + json.dumps(aliases, ensure_ascii=False)
                ),
            }
        ]
        for frame in frames:
            content.extend(
                [
                    {"type": "text", "text": f"evidence frame_ref={frame.alias}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_data_url(frame)},
                    },
                ]
            )
        return {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是隐私敏感的视频事件复核器。严格输出一个 JSON 对象。"
                        "不得把推测写入 observations；证据不足时使用 unknown。"
                        "evidence_frames 和 subject.evidence_frames 只能引用提供的匿名 frame_ref。"
                    ),
                },
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "baby_monitor_event",
                    "strict": True,
                    "schema": VLMEvent.json_schema(),
                },
            },
        }

    @staticmethod
    def _parse_response(
        response_payload: object,
        frames: tuple[_RemoteFrame, ...],
    ) -> VLMEvent:
        try:
            if not isinstance(response_payload, dict):
                raise TypeError
            message_content = response_payload["choices"][0]["message"]["content"]
            if isinstance(message_content, list):
                message_content = "".join(
                    item.get("text", "")
                    for item in message_content
                    if isinstance(item, dict) and item.get("type", "text") == "text"
                )
            if not isinstance(message_content, str):
                raise TypeError
            event = VLMEvent.from_json(message_content.strip())
        except (KeyError, IndexError, TypeError, SchemaError):
            raise _RemoteCallError(
                "invalid_response",
                "remote response is not a compatible strict VLM event",
            ) from None

        by_alias = {frame.alias: frame.path for frame in frames}

        def map_references(references: tuple[str, ...]) -> tuple[str, ...]:
            try:
                mapped = tuple(by_alias[reference] for reference in references)
            except KeyError:
                raise _RemoteCallError(
                    "invalid_evidence_reference",
                    "remote response cited an unknown evidence frame",
                ) from None
            if len(set(mapped)) != len(mapped):
                raise _RemoteCallError(
                    "invalid_evidence_reference",
                    "remote response cited duplicate evidence frames",
                )
            return mapped

        if not event.evidence_frames:
            raise _RemoteCallError(
                "missing_evidence_reference",
                "remote response did not cite evidence",
            )
        root_frames = map_references(event.evidence_frames)
        root_set = set(root_frames)
        subjects: list[Subject] = []
        for subject in event.subjects:
            if not subject.evidence_frames:
                raise _RemoteCallError(
                    "missing_evidence_reference",
                    "remote response subject did not cite evidence",
                )
            subject_frames = map_references(subject.evidence_frames)
            if not set(subject_frames).issubset(root_set):
                raise _RemoteCallError(
                    "invalid_evidence_reference",
                    "remote subject evidence is outside root evidence",
                )
            subjects.append(replace(subject, evidence_frames=subject_frames))
        return replace(event, subjects=tuple(subjects), evidence_frames=root_frames)

    def analyze(self, request: VLMRequest) -> VLMEvent:
        if not self.uploads_allowed:
            raise VLMError(
                "remote upload is disabled or upload consent is missing",
                code="remote_upload_disabled",
            )
        endpoint, target_origin, _ = self._endpoint()
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise VLMError(
                "remote API key is not configured",
                code="remote_api_key_missing",
            )
        if self.audit_store is None:
            raise VLMError(
                "remote audit store must be configured before uploads",
                code="remote_audit_unavailable",
            )

        frames = self._prepare_frames(request.media.frame_paths)
        frame_bytes = sum(frame.size_bytes for frame in frames)
        try:
            self.audit_store.initialize()
            attempt_id = self.audit_store.reserve(
                event_id=request.candidate.event_id,
                requested_at=datetime.now(UTC),
                target_origin=target_origin,
                model=self.model,
                frame_count=len(frames),
                frame_bytes=frame_bytes,
                max_requests_per_day=self.max_requests_per_day,
                min_interval_seconds=self.min_interval_seconds,
            )
        except RemoteBudgetError as exc:
            raise VLMError(str(exc), code=exc.code) from None
        except Exception:
            raise VLMError(
                "remote audit ledger is unavailable; upload was refused",
                code="remote_audit_unavailable",
            ) from None

        started = time.monotonic()
        try:
            payload = self._payload(request, frames)
            try:
                request_data = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError):
                raise _RemoteCallError(
                    "invalid_request",
                    "remote request could not be serialized safely",
                ) from None
            request_object = urllib.request.Request(
                endpoint,
                data=request_data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({}),
                _NoRedirectHandler(),
            )
            try:
                with opener.open(request_object, timeout=self.timeout_seconds) as response:
                    response_bytes = response.read(self.max_response_bytes + 1)
            except urllib.error.HTTPError:
                raise _RemoteCallError(
                    "http_error", "remote provider returned an HTTP error"
                ) from None
            except (TimeoutError, OSError):
                raise _RemoteCallError(
                    "transport_error", "remote provider request failed"
                ) from None
            if len(response_bytes) > self.max_response_bytes:
                raise _RemoteCallError(
                    "response_too_large", "remote provider response exceeded the size limit"
                )
            try:
                response_payload = json.loads(response_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise _RemoteCallError(
                    "invalid_response", "remote response is not valid UTF-8 JSON"
                ) from None
            event = self._parse_response(response_payload, frames)
        except _RemoteCallError as exc:
            duration_ms = max(0, round((time.monotonic() - started) * 1_000))
            try:
                self.audit_store.complete(
                    attempt_id,
                    status=RemoteAttemptStatus.FAILED,
                    completed_at=datetime.now(UTC),
                    duration_ms=duration_ms,
                    error_code=exc.code,
                )
            except Exception:
                raise VLMError(
                    "remote audit ledger could not record the failed call",
                    code="remote_audit_write_failed",
                ) from None
            raise VLMError(str(exc), code=exc.code) from None

        duration_ms = max(0, round((time.monotonic() - started) * 1_000))
        try:
            self.audit_store.complete(
                attempt_id,
                status=RemoteAttemptStatus.SUCCEEDED,
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
            )
        except Exception:
            raise VLMError(
                "remote audit ledger could not record the completed call",
                code="remote_audit_write_failed",
            ) from None
        return event
