from __future__ import annotations

import os
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from baby_monitor.identity_enrollment_contracts import (
    EnrollmentImage,
    ValidatedEnrollmentEmbedding,
)
from baby_monitor.identity_protocol import (
    FaceEmbeddingBatch,
    FaceEmbeddingMode,
    FaceEmbeddingProtocolError,
    FaceEmbeddingRequest,
    FaceEmbeddingStagingRoot,
    FaceImageInput,
    FaceModelReference,
)

MAX_FACE_COMMAND_CAPTURE_BYTES = 64 * 1024
MAX_FACE_COMMAND_PARTS = 32
MAX_FACE_COMMAND_PART_BYTES = 4096
MAX_CONSUMED_REQUESTS = 100_000
_SHELL_EXECUTABLES = frozenset(
    {
        "bash",
        "cmd",
        "cmd.exe",
        "dash",
        "fish",
        "ksh",
        "powershell",
        "powershell.exe",
        "pwsh",
        "sh",
        "tcsh",
        "zsh",
    }
)


class FaceCommandAdapterError(RuntimeError):
    """A fixed, path-free failure from the local face command boundary."""

    _MESSAGES = {
        "invalid_configuration": "local face command configuration was rejected",
        "invalid_request": "local face command request was rejected",
        "replay_rejected": "local face command request replay was rejected",
        "replay_state_full": "local face command replay state is full",
        "staging_failed": "local face command private staging failed",
        "start_failed": "local face command could not start",
        "timeout": "local face command timed out",
        "process_failed": "local face command process failed",
        "capture_failed": "local face command capture failed",
        "output_too_large": "local face command output exceeded its limit",
        "output_disclosure": "local face command output violated disclosure policy",
        "result_count_invalid": "local face command produced an invalid result count",
        "result_invalid": "local face command result was rejected",
        "cleanup_failed": "local face command private cleanup failed",
    }

    def __init__(self, code: str):
        message = self._MESSAGES.get(code)
        if message is None:
            code = "result_invalid"
            message = self._MESSAGES[code]
        self.code = code
        super().__init__(message)


ProcessFactory = Callable[..., Any]
ProcessTerminator = Callable[[Any], None]
BoundResult = TypeVar("BoundResult")


def _contains_control(value: str) -> bool:
    return any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def _default_terminate_process(process: Any) -> None:
    try:
        if process.poll() is not None:
            return
    except (AttributeError, OSError):
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except (AttributeError, OSError, ProcessLookupError):
            pass
    with suppress(AttributeError, OSError):
        process.kill()


@dataclass(frozen=True, slots=True)
class FaceEnrollmentCommandConfig:
    """Pinned local models and quality thresholds for one-image enrollment."""

    detector_model: FaceModelReference = field(repr=False)
    embedding_model: FaceModelReference = field(repr=False)
    face_adapter: str = "opencv_yunet_sface_v1"
    detector_score_threshold: float = 0.8
    detector_nms_threshold: float = 0.3
    detector_top_k: int = 100
    min_face_size_pixels: int = 40
    min_eye_distance_pixels: float = 12.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.detector_model, FaceModelReference)
            or self.detector_model.role != "detector"
            or not isinstance(self.embedding_model, FaceModelReference)
            or self.embedding_model.role != "embedding"
            or self.face_adapter != "opencv_yunet_sface_v1"
        ):
            raise FaceCommandAdapterError("invalid_configuration")
        for value, minimum, maximum in (
            (self.detector_score_threshold, 0.01, 1.0),
            (self.detector_nms_threshold, 0.01, 1.0),
            (self.min_eye_distance_pixels, 1.0, 2048.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not minimum <= float(value) <= maximum
            ):
                raise FaceCommandAdapterError("invalid_configuration")
        if (
            type(self.detector_top_k) is not int
            or not 1 <= self.detector_top_k <= 5_000
            or type(self.min_face_size_pixels) is not int
            or not 8 <= self.min_face_size_pixels <= 4096
        ):
            raise FaceCommandAdapterError("invalid_configuration")
        object.__setattr__(
            self,
            "detector_score_threshold",
            float(self.detector_score_threshold),
        )
        object.__setattr__(
            self,
            "detector_nms_threshold",
            float(self.detector_nms_threshold),
        )
        object.__setattr__(
            self,
            "min_eye_distance_pixels",
            float(self.min_eye_distance_pixels),
        )

    def _request_for(self, image: EnrollmentImage) -> FaceEmbeddingRequest:
        if not isinstance(image, EnrollmentImage):
            raise FaceCommandAdapterError("invalid_request")
        try:
            admitted_image = FaceImageInput(
                frame_ref=image.frame_ref,
                source_path=image.source_path,
                source_sha256=image.source_sha256,
            )
            return FaceEmbeddingRequest(
                request_id=f"enrollment_{secrets.token_hex(16)}",
                mode=FaceEmbeddingMode.ENROLLMENT,
                face_adapter=self.face_adapter,
                images=(admitted_image,),
                detector_model=self.detector_model,
                embedding_model=self.embedding_model,
                detector_score_threshold=self.detector_score_threshold,
                detector_nms_threshold=self.detector_nms_threshold,
                detector_top_k=self.detector_top_k,
                min_face_size_pixels=self.min_face_size_pixels,
                min_eye_distance_pixels=self.min_eye_distance_pixels,
            )
        except (OSError, ValueError):
            raise FaceCommandAdapterError("invalid_request") from None


class CommandFaceEmbeddingAdapter:
    """Run the reviewed local face runner through a short-lived private IPC root.

    The adapter itself performs no network calls, but it is not an OS network
    sandbox. Production execution still requires the systemd deployment gate
    ``IPAddressDeny=any`` for the child process.
    """

    def __init__(
        self,
        *,
        command: Sequence[str],
        staging_parent: str | Path,
        timeout_seconds: float = 120.0,
        max_captured_bytes: int = MAX_FACE_COMMAND_CAPTURE_BYTES,
        enrollment: FaceEnrollmentCommandConfig | None = None,
        _process_factory: ProcessFactory | None = None,
        _process_terminator: ProcessTerminator | None = None,
    ):
        self._command = self._admit_command(command)
        try:
            approved_parent = FaceEmbeddingStagingRoot(str(staging_parent))
        except (OSError, ValueError):
            raise FaceCommandAdapterError("invalid_configuration") from None
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0.01 <= float(timeout_seconds) <= 600.0
        ):
            raise FaceCommandAdapterError("invalid_configuration")
        if type(max_captured_bytes) is not int or not 1024 <= max_captured_bytes <= 1024 * 1024:
            raise FaceCommandAdapterError("invalid_configuration")
        if enrollment is not None and not isinstance(enrollment, FaceEnrollmentCommandConfig):
            raise FaceCommandAdapterError("invalid_configuration")
        self._staging_parent = Path(approved_parent.root)
        self._timeout_seconds = float(timeout_seconds)
        self._max_captured_bytes = max_captured_bytes
        self._process_factory = _process_factory or subprocess.Popen
        self._process_terminator = _process_terminator or _default_terminate_process
        self._enrollment = enrollment
        self._consumed_requests: set[str] = set()
        self._consumed_lock = threading.Lock()

    @staticmethod
    def _admit_command(command: Sequence[str]) -> tuple[str, ...]:
        if isinstance(command, (str, bytes)):
            raise FaceCommandAdapterError("invalid_configuration")
        parts = tuple(command)
        if not 1 <= len(parts) <= MAX_FACE_COMMAND_PARTS:
            raise FaceCommandAdapterError("invalid_configuration")
        for part in parts:
            if (
                not isinstance(part, str)
                or not part
                or len(part.encode("utf-8")) > MAX_FACE_COMMAND_PART_BYTES
                or _contains_control(part)
                or "{" in part
                or "}" in part
            ):
                raise FaceCommandAdapterError("invalid_configuration")
        executable = Path(parts[0])
        if not executable.is_absolute() or ".." in executable.parts or str(executable) != parts[0]:
            raise FaceCommandAdapterError("invalid_configuration")
        try:
            resolved = executable.resolve(strict=True)
            metadata = resolved.stat()
        except OSError:
            raise FaceCommandAdapterError("invalid_configuration") from None
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not os.access(resolved, os.X_OK)
            or not hasattr(os, "geteuid")
            or metadata.st_uid not in {0, os.geteuid()}
            or bool(stat.S_IMODE(metadata.st_mode) & 0o022)
            or resolved.name.casefold() in _SHELL_EXECUTABLES
        ):
            raise FaceCommandAdapterError("invalid_configuration")
        # Preserve an absolute venv interpreter path: resolving that symlink
        # would silently escape the venv and change the admitted runtime.
        return (str(executable), *parts[1:])

    @staticmethod
    def _minimal_environment() -> dict[str, str]:
        environment = {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "LANG": "C",
            "LC_ALL": "C",
            "NO_PROXY": "*",
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        return environment

    def _claim_request(self, request: FaceEmbeddingRequest) -> str:
        if not isinstance(request, FaceEmbeddingRequest):
            raise FaceCommandAdapterError("invalid_request")
        try:
            request_sha256 = request.canonical_sha256()
        except (OSError, ValueError):
            raise FaceCommandAdapterError("invalid_request") from None
        with self._consumed_lock:
            if request_sha256 in self._consumed_requests:
                raise FaceCommandAdapterError("replay_rejected")
            if len(self._consumed_requests) >= MAX_CONSUMED_REQUESTS:
                raise FaceCommandAdapterError("replay_state_full")
            self._consumed_requests.add(request_sha256)
        return request_sha256

    def _create_run_root(self) -> tuple[Path, FaceEmbeddingStagingRoot]:
        directory: Path | None = None
        try:
            approved_parent = FaceEmbeddingStagingRoot(str(self._staging_parent))
            directory = Path(
                tempfile.mkdtemp(
                    prefix="face-command-",
                    dir=approved_parent.root,
                )
            )
            directory.chmod(0o700)
            return directory, FaceEmbeddingStagingRoot(str(directory))
        except (OSError, ValueError):
            if directory is not None:
                with suppress(OSError):
                    shutil.rmtree(directory)
            raise FaceCommandAdapterError("staging_failed") from None

    def _cleanup_run_root(self, directory: Path) -> None:
        if directory.parent != self._staging_parent or not directory.name.startswith(
            "face-command-"
        ):
            raise FaceCommandAdapterError("cleanup_failed")
        try:
            FaceEmbeddingStagingRoot(str(self._staging_parent))
            metadata = directory.lstat()
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            raise FaceCommandAdapterError("cleanup_failed") from None
        try:
            if stat.S_ISLNK(metadata.st_mode):
                directory.unlink()
            elif stat.S_ISDIR(metadata.st_mode):
                shutil.rmtree(directory)
            else:
                directory.unlink()
        except OSError:
            raise FaceCommandAdapterError("cleanup_failed") from None

    def _capture_process(self, process: Any) -> tuple[bytes, bytes]:
        stdout = getattr(process, "stdout", None)
        stderr = getattr(process, "stderr", None)
        if stdout is None or stderr is None:
            self._process_terminator(process)
            raise FaceCommandAdapterError("capture_failed")

        buffers = (bytearray(), bytearray())
        lock = threading.Lock()
        overflow = threading.Event()
        read_failed = threading.Event()
        total = 0

        def read_stream(stream: Any, destination: bytearray) -> None:
            nonlocal total
            try:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        return
                    if not isinstance(chunk, bytes):
                        read_failed.set()
                        return
                    with lock:
                        remaining = self._max_captured_bytes - total
                        if remaining > 0:
                            destination.extend(chunk[:remaining])
                        total += len(chunk)
                        if total > self._max_captured_bytes:
                            overflow.set()
            except (OSError, ValueError):
                read_failed.set()

        readers = (
            threading.Thread(target=read_stream, args=(stdout, buffers[0]), daemon=True),
            threading.Thread(target=read_stream, args=(stderr, buffers[1]), daemon=True),
        )
        for reader in readers:
            reader.start()

        deadline = time.monotonic() + self._timeout_seconds
        timed_out = False
        try:
            while process.poll() is None:
                if overflow.is_set():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                overflow.wait(timeout=min(0.05, remaining))
            if overflow.is_set() or timed_out:
                self._process_terminator(process)
            with suppress(OSError, subprocess.TimeoutExpired):
                process.wait(timeout=2.0)
        except (AttributeError, OSError):
            self._process_terminator(process)
            raise FaceCommandAdapterError("capture_failed") from None
        finally:
            for reader in readers:
                reader.join(timeout=2.0)
            for stream in (stdout, stderr):
                with suppress(OSError):
                    stream.close()

        if any(reader.is_alive() for reader in readers) or read_failed.is_set():
            raise FaceCommandAdapterError("capture_failed")
        if overflow.is_set():
            raise FaceCommandAdapterError("output_too_large")
        if timed_out:
            raise FaceCommandAdapterError("timeout")
        return bytes(buffers[0]), bytes(buffers[1])

    @staticmethod
    def _has_disclosure(
        stdout: bytes,
        stderr: bytes,
        *,
        request: FaceEmbeddingRequest,
        private_paths: tuple[Path, ...],
    ) -> bool:
        output = stdout + b"\n" + stderr
        lowered = output.lower()
        for marker in (
            b"embedding",
            b"source_path",
            b"model_path",
            b"manifest_path",
        ):
            if marker in lowered:
                return True
        sensitive_paths = (
            *(image.source_path for image in request.images),
            request.detector_model.model_path,
            request.detector_model.manifest_path,
            request.embedding_model.model_path,
            request.embedding_model.manifest_path,
            *(str(path) for path in private_paths),
        )
        return any(path.encode("utf-8") in output for path in sensitive_paths)

    def _execute_unvalidated(
        self,
        request: FaceEmbeddingRequest,
        directory: Path,
        staging: FaceEmbeddingStagingRoot,
    ) -> FaceEmbeddingBatch:
        request_path = directory / "request.json"
        result_path = directory / "result.json"
        try:
            staging.write_request(request, request_path)
        except (OSError, ValueError):
            raise FaceCommandAdapterError("staging_failed") from None

        command = [*self._command, staging.root, str(request_path), str(result_path)]
        try:
            process = self._process_factory(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=staging.root,
                env=self._minimal_environment(),
                shell=False,
                close_fds=True,
                start_new_session=True,
                bufsize=0,
                umask=0o077,
            )
        except (OSError, ValueError):
            raise FaceCommandAdapterError("start_failed") from None

        try:
            stdout, stderr = self._capture_process(process)
        except BaseException:
            self._process_terminator(process)
            with suppress(AttributeError, OSError, subprocess.TimeoutExpired):
                process.wait(timeout=2.0)
            raise
        if getattr(process, "returncode", None) != 0:
            raise FaceCommandAdapterError("process_failed")
        if self._has_disclosure(
            stdout,
            stderr,
            request=request,
            private_paths=(directory, request_path, result_path),
        ):
            raise FaceCommandAdapterError("output_disclosure")

        try:
            entries = {entry.name for entry in directory.iterdir()}
        except OSError:
            raise FaceCommandAdapterError("result_count_invalid") from None
        if entries != {request_path.name, result_path.name}:
            raise FaceCommandAdapterError("result_count_invalid")

        try:
            if staging.read_request(request_path) != request:
                raise FaceEmbeddingProtocolError("staged request changed")
            batch = staging.read_batch(result_path)
        except (OSError, ValueError):
            raise FaceCommandAdapterError("result_invalid") from None
        return batch

    def _run_bound(
        self,
        request: FaceEmbeddingRequest,
        consumer: Callable[
            [FaceEmbeddingRequest, FaceEmbeddingBatch, FaceEmbeddingStagingRoot],
            BoundResult,
        ],
    ) -> BoundResult:
        self._claim_request(request)
        directory, staging = self._create_run_root()
        try:
            batch = self._execute_unvalidated(request, directory, staging)
            try:
                return consumer(request, batch, staging)
            except FaceCommandAdapterError:
                raise
            except Exception:
                raise FaceCommandAdapterError("result_invalid") from None
        finally:
            self._cleanup_run_root(directory)

    def embed(self, request: FaceEmbeddingRequest) -> FaceEmbeddingBatch:
        """Execute and consume one request; callers must mint a new nonce to retry."""

        def consume(
            bound_request: FaceEmbeddingRequest,
            batch: FaceEmbeddingBatch,
            staging: FaceEmbeddingStagingRoot,
        ) -> FaceEmbeddingBatch:
            batch.validate_against_request(bound_request, replay_guard=staging)
            return batch

        return self._run_bound(request, consume)

    def extract_validated_enrollment(
        self,
        image: EnrollmentImage,
    ) -> ValidatedEnrollmentEmbedding:
        """Return a path-free proof after the coordinator consumes the binding once."""

        if self._enrollment is None:
            raise FaceCommandAdapterError("invalid_configuration")
        request = self._enrollment._request_for(image)

        def consume(
            bound_request: FaceEmbeddingRequest,
            batch: FaceEmbeddingBatch,
            staging: FaceEmbeddingStagingRoot,
        ) -> ValidatedEnrollmentEmbedding:
            return ValidatedEnrollmentEmbedding.consume_bound_batch(
                bound_request,
                batch,
                replay_guard=staging,
            )

        return self._run_bound(request, consume)
