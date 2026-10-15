"""One serialized resident local VLM, with a lifetime GPU gate and private IPC.

Deployment supplies an offline command. No HTTP endpoint is exposed by this
adapter; requests use inherited socket pairs and private temporary files.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import subprocess
import tempfile
import threading
import uuid
from contextlib import suppress
from pathlib import Path

from baby_monitor.adapters.vlm import CommandVLMAdapter, VLMAdapter, VLMError, VLMRequest
from baby_monitor.adapters.vlm_execution import VLMExecutionError, local_vlm_execution
from baby_monitor.domain import RecognitionProvenance, SchemaError, VLMEvent
from baby_monitor.live_detector_protocol import receive_packet, send_packet
from baby_monitor.live_vlm_protocol import VLMDiagnostics, validate_vlm_diagnostics
from baby_monitor.local_runners import subject_state_probe
from baby_monitor.local_runners.subject_probe import PROMPT_VERSION as SUBJECT_PROBE_VERSION
from baby_monitor.local_runners.subject_probe import validate_bound_result

_MODEL_FAILURE_CODES = frozenset(
    {
        "bonsai_response_incomplete",
        "bonsai_invalid_envelope",
        "bonsai_invalid_response_json",
        "bonsai_server_load_failed",
        "bonsai_server_load_timeout",
        "bonsai_server_exited",
        "bonsai_invalid_request",
        "bonsai_invalid_protocol",
        "runner_oom",
        "invalid_structured_output",
        "runner_failed",
    }
)


class PersistentLocalVLM(VLMAdapter):
    """Explicit opt-in for the pinned single-frame Bonsai IPC runner.

    The gate spans startup, every request and full process-group cleanup. A
    recoverable token-limit response keeps the resident session; transport,
    binding and other model errors close it before any future startup.
    """

    execution_mode = "resident_single_model"

    def __init__(
        self,
        *,
        command: tuple[str, ...],
        model_revision: str,
        prompt_revision: str,
        gpu_device: str = "0",
        startup_timeout_seconds: int = 90,
        request_timeout_seconds: int = 120,
    ):
        if (
            not command
            or any(not isinstance(part, str) or not part or "{request}" in part for part in command)
            or type(startup_timeout_seconds) is not int
            or not 1 <= startup_timeout_seconds <= 90
            or type(request_timeout_seconds) is not int
            or not 1 <= request_timeout_seconds <= 120
        ):
            raise ValueError("invalid resident VLM configuration")
        self.name = "bonsai_2_27b_persistent"
        self.command = command
        self.gpu_device = gpu_device
        self.startup_timeout_seconds = startup_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self._provenance = RecognitionProvenance(
            adapter_name=self.name, model_revision=model_revision, prompt_revision=prompt_revision
        )
        self.process: subprocess.Popen | None = None
        self.channel: socket.socket | None = None
        self.identity: dict[str, object] = {}
        self.last_diagnostics: VLMDiagnostics | None = None
        self._gate = None
        self._gate_fd: int | None = None
        self._mutex = threading.RLock()

    @property
    def provenance(self) -> RecognitionProvenance:
        return self._provenance

    @property
    def ready(self) -> bool:
        with self._mutex:
            return (
                self.channel is not None
                and self.process is not None
                and self.process.poll() is None
                and bool(self.identity)
            )

    def start(self) -> None:
        with self._mutex:
            if self.ready:
                return
            self.close()
            gate = local_vlm_execution(self.gpu_device)
            try:
                lease = gate.__enter__()
            except VLMExecutionError as error:
                raise VLMError("resident VLM gate unavailable", code=error.code) from None
            self._gate = gate
            self._gate_fd = lease
            child = None
            try:
                parent, child = socket.socketpair()
                self.channel = parent
                parent.settimeout(self.startup_timeout_seconds)
                self.process = subprocess.Popen(
                    (*self.command, "--execution-lock-fd", str(lease), "--fd", str(child.fileno())),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    pass_fds=(child.fileno(), lease),
                    env=CommandVLMAdapter._subprocess_environment(),
                    start_new_session=True,
                )
                child.close()
                response, payload = receive_packet(parent)
                if (
                    response.get("type") != "ready"
                    or payload
                    or response.get("network_enabled") is not False
                    or type(response.get("model_loads")) is not int
                    or response["model_loads"] != 1
                    or response.get("model_revision") != self.provenance.model_revision
                    or response.get("prompt_revision") != self.provenance.prompt_revision
                ):
                    raise VLMError(
                        "resident VLM identity mismatch", code="resident_identity_mismatch"
                    )
                self.identity = response
                parent.settimeout(self.request_timeout_seconds)
            except BaseException as error:
                self.close()
                if isinstance(error, VLMError):
                    raise
                if isinstance(error, Exception):
                    raise VLMError(
                        "resident VLM startup failed", code="resident_start_failed"
                    ) from None
                raise
            finally:
                if child is not None:
                    child.close()

    def analyze(self, request: VLMRequest) -> VLMEvent:
        return self._request(request, probe=False)

    def probe_subjects(self, request: VLMRequest) -> dict:
        """Explicit independent diagnostic; never implicitly added to analyze()."""
        return self._request(request, probe=True)

    def probe_subject_state(self, request: VLMRequest) -> dict:
        return self._request(request, probe=True, state=True)

    @staticmethod
    def _frame_digest(path: Path) -> str:
        if not path.is_absolute() or path.resolve() != path:
            raise ValueError("unsafe probe frame")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 10 * 1024**2:
                raise ValueError("unsafe probe frame")
            data = stream.read(10 * 1024**2 + 1)
        if len(data) != info.st_size:
            raise ValueError("probe frame changed")
        return hashlib.sha256(data).hexdigest()

    def _request(self, request: VLMRequest, *, probe: bool, state: bool = False):
        with self._mutex:
            self.last_diagnostics = None
            if len(request.media.frame_paths) != 1:
                raise VLMError(
                    "resident runner requires one frame", code="resident_request_invalid"
                )
            identifier = uuid.uuid4().hex
            try:
                frame = Path(request.media.frame_paths[0])
                digest = self._frame_digest(frame) if probe else None
                self.start()
                contract_revision = (
                    subject_state_probe.PROMPT_VERSION if state else SUBJECT_PROBE_VERSION
                )
                capability = "subject_state_revision" if state else "subject_probe_revision"
                if probe and self.identity.get(capability) != contract_revision:
                    raise VLMError(
                        "probe capability unavailable", code="resident_identity_mismatch"
                    )
                with tempfile.TemporaryDirectory(
                    prefix="baby-monitor-resident-request-"
                ) as directory:
                    path = Path(directory) / "request.json"
                    payload = json.dumps(
                        request.to_dict(), ensure_ascii=False, allow_nan=False
                    ).encode()
                    if len(payload) > 256 * 1024:
                        raise VLMError(
                            "resident request exceeds bounds", code="resident_request_invalid"
                        )
                    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(payload)
                    send_packet(
                        self.channel,
                        {
                            "type": "probe_subject_state"
                            if state
                            else "probe_subjects"
                            if probe
                            else "analyze",
                            "request_id": identifier,
                            "request_path": str(path),
                        },
                    )
                    response, extra = receive_packet(self.channel)
                if extra or response.get("request_id") != identifier:
                    raise VLMError(
                        "resident response binding failed", code="resident_binding_failed"
                    )
                try:
                    diagnostics = validate_vlm_diagnostics(response.get("diagnostics"))
                except ValueError:
                    raise VLMError(
                        "resident response diagnostics invalid", code="resident_response_invalid"
                    ) from None
                if response.get("type") == "error":
                    if (
                        response.get("error_code") == "bonsai_output_token_limit"
                        and response.get("fatal") is False
                    ):
                        self.last_diagnostics = diagnostics
                        raise VLMError(
                            "local output reached token limit", code="bonsai_output_token_limit"
                        )
                    code = response.get("error_code")
                    if isinstance(code, str) and code in _MODEL_FAILURE_CODES:
                        self.last_diagnostics = diagnostics
                        raise VLMError(
                            "resident model request failed",
                            code=code,
                            oom_status="observed" if code == "runner_oom" else "unknown",
                        )
                    raise VLMError("resident model request failed", code="resident_model_failed")
                expected_type = "subject_result" if probe else "result"
                if response.get("type") != expected_type or not isinstance(
                    response.get("result"), dict
                ):
                    raise VLMError("resident response invalid", code="resident_response_invalid")
                if probe:
                    validator = (
                        subject_state_probe.validate_bound_result
                        if state
                        else validate_bound_result
                    )
                    result = validator(response["result"], frame_id=frame.name, input_sha256=digest)
                    if self._frame_digest(frame) != digest:
                        raise VLMError("probe frame changed", code="resident_binding_failed")
                    self.last_diagnostics = diagnostics
                    return result
                result = VLMEvent.from_dict(response["result"])
                allowed = {Path(path).name for path in request.media.frame_paths}
                references = set(result.evidence_frames)
                for subject in result.subjects:
                    references.update(subject.evidence_frames)
                if not references <= allowed:
                    raise VLMError(
                        "resident evidence binding failed", code="resident_binding_failed"
                    )
                self.last_diagnostics = diagnostics
                return result
            except BaseException as error:
                if not isinstance(error, VLMError) or error.code != "bonsai_output_token_limit":
                    self.close()
                if isinstance(error, VLMError):
                    raise
                if isinstance(error, (OSError, RuntimeError, ValueError, SchemaError, TypeError)):
                    raise VLMError(
                        "resident request failed", code="resident_request_failed"
                    ) from None
                raise

    def close(self) -> None:
        with self._mutex:
            if self.channel is not None:
                with suppress(OSError, RuntimeError, ValueError):
                    self.channel.settimeout(1)
                    send_packet(self.channel, {"type": "stop"})
                self.channel.close()
                self.channel = None
            if self.process is not None:
                process = self.process
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    CommandVLMAdapter._kill_process_tree(process)
                    process.wait(timeout=10)
                # Even a dead parent may have left a server in its process group.
                CommandVLMAdapter._kill_process_tree(process)
                self.process = None
            self.identity = {}
            if self._gate is not None:
                self._gate.__exit__(None, None, None)
                self._gate = None
                self._gate_fd = None
