from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import signal
import stat
import subprocess
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from baby_monitor.domain import SchemaError
from baby_monitor.visual_predicate import (
    MAX_VISUAL_PREDICATE_BYTES,
    VisualPredicate,
    VisualPredicateAnswer,
    VisualPredicateResult,
)

_MAX_HEALTH_BYTES = 64 * 1024
_HEALTH_STATES = frozenset({"stopped", "starting", "ready", "busy", "failed"})
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_MEDIA_SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png"}
_MAX_FRAME_BYTES = 25 * 1024 * 1024
_MAX_REQUEST_MEDIA_BYTES = 100 * 1024 * 1024
_WORKER_PROTOCOL_VERSION = 1
_MAX_WORKER_LINE_BYTES = 128 * 1024


class VisualPredicateAdapterError(RuntimeError):
    """Path-free failure raised by a local visual-predicate adapter."""

    def __init__(self, message: str, *, code: str):
        if _SAFE_ID.fullmatch(code) is None:
            raise ValueError("visual predicate adapter error code is invalid")
        self.code = code
        super().__init__(message)


class VisualPredicateHealthError(ValueError):
    """Path-free failure for resident worker health persistence."""


@dataclass(frozen=True, slots=True)
class VisualPredicateFrame:
    frame_id: str
    sha256: str
    media_type: str
    path: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or _SAFE_ID.fullmatch(self.frame_id) is None:
            raise ValueError("visual predicate frame_id is invalid")
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("visual predicate frame hash is invalid")
        if self.media_type not in _MEDIA_SUFFIX:
            raise ValueError("visual predicate frame media_type is invalid")
        if (
            not isinstance(self.path, str)
            or "\x00" in self.path
            or not Path(self.path).is_absolute()
        ):
            raise ValueError("visual predicate frame path must be absolute")


@dataclass(frozen=True, slots=True)
class VisualPredicateRequest:
    event_id: str
    predicate: VisualPredicate
    frames: tuple[VisualPredicateFrame, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or _SAFE_ID.fullmatch(self.event_id) is None:
            raise ValueError("visual predicate request event_id is invalid")
        if not isinstance(self.predicate, VisualPredicate):
            raise ValueError("visual predicate request predicate is invalid")
        if (
            not isinstance(self.frames, tuple)
            or not 1 <= len(self.frames) <= 12
            or any(not isinstance(frame, VisualPredicateFrame) for frame in self.frames)
            or len({frame.frame_id for frame in self.frames}) != len(self.frames)
        ):
            raise ValueError("visual predicate request frames are invalid")

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return tuple(frame.frame_id for frame in self.frames)


@dataclass(frozen=True, slots=True)
class ResidentVisualPredicateHealth:
    checked_at: datetime
    state: str
    worker_running: bool
    worker_starts: int
    worker_restarts: int
    successful_jobs: int
    failed_jobs: int
    timeout_jobs: int
    current_worker_jobs: int
    max_jobs: int
    idle_exit_seconds: float
    max_runtime_seconds: float
    model_id: str
    model_revision: str
    prompt_version: str
    last_error_code: str | None
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise VisualPredicateHealthError("unsupported resident health version")
        if (
            not isinstance(self.checked_at, datetime)
            or self.checked_at.tzinfo is None
            or self.checked_at.utcoffset() is None
        ):
            raise VisualPredicateHealthError("resident health checked_at is invalid")
        object.__setattr__(self, "checked_at", self.checked_at.astimezone(UTC))
        if not isinstance(self.state, str) or self.state not in _HEALTH_STATES:
            raise VisualPredicateHealthError("resident health state is invalid")
        if type(self.worker_running) is not bool:
            raise VisualPredicateHealthError("resident health worker_running is invalid")
        if self.worker_running and self.state in {"stopped", "failed"}:
            raise VisualPredicateHealthError("resident health state is inconsistent")
        if not self.worker_running and self.state in {"ready", "busy"}:
            raise VisualPredicateHealthError("resident health state is inconsistent")
        for value, field_name in (
            (self.worker_starts, "worker_starts"),
            (self.worker_restarts, "worker_restarts"),
            (self.successful_jobs, "successful_jobs"),
            (self.failed_jobs, "failed_jobs"),
            (self.timeout_jobs, "timeout_jobs"),
            (self.current_worker_jobs, "current_worker_jobs"),
        ):
            if type(value) is not int or not 0 <= value <= 1_000_000_000:
                raise VisualPredicateHealthError(f"resident health {field_name} is invalid")
        if type(self.max_jobs) is not int or not 1 <= self.max_jobs <= 256:
            raise VisualPredicateHealthError("resident health max_jobs is invalid")
        if self.worker_restarts != max(0, self.worker_starts - 1):
            raise VisualPredicateHealthError("resident health restart count is inconsistent")
        if self.timeout_jobs > self.failed_jobs:
            raise VisualPredicateHealthError("resident health timeout count is inconsistent")
        if self.current_worker_jobs > self.max_jobs or (
            not self.worker_running and self.current_worker_jobs != 0
        ):
            raise VisualPredicateHealthError("resident health current jobs are inconsistent")
        for value, field_name, maximum in (
            (self.idle_exit_seconds, "idle_exit_seconds", 3_600.0),
            (self.max_runtime_seconds, "max_runtime_seconds", 86_400.0),
        ):
            if (
                type(value) not in {int, float}
                or not math.isfinite(value)
                or not 1.0 <= float(value) <= maximum
            ):
                raise VisualPredicateHealthError(f"resident health {field_name} is invalid")
            object.__setattr__(self, field_name, float(value))
        if not isinstance(self.model_id, str) or _SAFE_ID.fullmatch(self.model_id) is None:
            raise VisualPredicateHealthError("resident health model_id is invalid")
        if (
            not isinstance(self.model_revision, str)
            or _GIT_SHA.fullmatch(self.model_revision) is None
        ):
            raise VisualPredicateHealthError("resident health model_revision is invalid")
        if (
            not isinstance(self.prompt_version, str)
            or _SAFE_ID.fullmatch(self.prompt_version) is None
        ):
            raise VisualPredicateHealthError("resident health prompt_version is invalid")
        if self.last_error_code is not None and (
            not isinstance(self.last_error_code, str)
            or _SAFE_ID.fullmatch(self.last_error_code) is None
        ):
            raise VisualPredicateHealthError("resident health last_error_code is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "checked_at": self.checked_at.astimezone(UTC).isoformat(),
            "state": self.state,
            "worker_running": self.worker_running,
            "worker_starts": self.worker_starts,
            "worker_restarts": self.worker_restarts,
            "successful_jobs": self.successful_jobs,
            "failed_jobs": self.failed_jobs,
            "timeout_jobs": self.timeout_jobs,
            "current_worker_jobs": self.current_worker_jobs,
            "max_jobs": self.max_jobs,
            "idle_exit_seconds": self.idle_exit_seconds,
            "max_runtime_seconds": self.max_runtime_seconds,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "prompt_version": self.prompt_version,
            "last_error_code": self.last_error_code,
            "paths_disclosed": False,
            "media_metadata_included": False,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ResidentVisualPredicateHealth:
        expected = {
            "version",
            "checked_at",
            "state",
            "worker_running",
            "worker_starts",
            "worker_restarts",
            "successful_jobs",
            "failed_jobs",
            "timeout_jobs",
            "current_worker_jobs",
            "max_jobs",
            "idle_exit_seconds",
            "max_runtime_seconds",
            "model_id",
            "model_revision",
            "prompt_version",
            "last_error_code",
            "paths_disclosed",
            "media_metadata_included",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise VisualPredicateHealthError("resident health fields do not match v1")
        if value["paths_disclosed"] is not False or value["media_metadata_included"] is not False:
            raise VisualPredicateHealthError("resident health privacy flags are invalid")
        checked_at = value["checked_at"]
        if not isinstance(checked_at, str):
            raise VisualPredicateHealthError("resident health checked_at is invalid")
        try:
            parsed_time = datetime.fromisoformat(checked_at)
        except ValueError:
            raise VisualPredicateHealthError("resident health checked_at is invalid") from None
        return cls(
            version=value["version"],
            checked_at=parsed_time,
            state=value["state"],
            worker_running=value["worker_running"],
            worker_starts=value["worker_starts"],
            worker_restarts=value["worker_restarts"],
            successful_jobs=value["successful_jobs"],
            failed_jobs=value["failed_jobs"],
            timeout_jobs=value["timeout_jobs"],
            current_worker_jobs=value["current_worker_jobs"],
            max_jobs=value["max_jobs"],
            idle_exit_seconds=value["idle_exit_seconds"],
            max_runtime_seconds=value["max_runtime_seconds"],
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            prompt_version=value["prompt_version"],
            last_error_code=value["last_error_code"],
        )

    @staticmethod
    def _private_path(path: str | Path, *, existing: bool) -> Path:
        health_path = Path(path).expanduser()
        if not health_path.is_absolute() or "\x00" in str(health_path):
            raise VisualPredicateHealthError("resident health path is unsafe")
        parent = health_path.parent
        try:
            resolved_parent = parent.resolve(strict=True)
            parent_metadata = parent.stat(follow_symlinks=False)
        except OSError:
            raise VisualPredicateHealthError("resident health path is unsafe") from None
        if (
            resolved_parent != parent
            or parent.is_symlink()
            or not stat.S_ISDIR(parent_metadata.st_mode)
            or (os.name == "posix" and parent_metadata.st_uid != os.getuid())
            or (os.name == "posix" and stat.S_IMODE(parent_metadata.st_mode) != 0o700)
            or (os.name == "posix" and resolved_parent.parts[:3] == ("/", "mnt", "c"))
        ):
            raise VisualPredicateHealthError("resident health path is unsafe")
        if health_path.is_symlink():
            raise VisualPredicateHealthError("resident health path is unsafe")
        if existing or health_path.exists():
            try:
                metadata = health_path.stat(follow_symlinks=False)
            except OSError:
                raise VisualPredicateHealthError("resident health file is unavailable") from None
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or not 0 < metadata.st_size <= _MAX_HEALTH_BYTES
                or (os.name == "posix" and metadata.st_uid != os.getuid())
                or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600)
            ):
                raise VisualPredicateHealthError("resident health file is unsafe")
        return health_path

    @classmethod
    def read_private(cls, path: str | Path) -> ResidentVisualPredicateHealth:
        health_path = cls._private_path(path, existing=True)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(health_path, flags)
        except OSError:
            raise VisualPredicateHealthError("resident health file is unavailable") from None
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or not 0 < metadata.st_size <= _MAX_HEALTH_BYTES
                or (os.name == "posix" and metadata.st_uid != os.getuid())
                or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600)
            ):
                raise VisualPredicateHealthError("resident health file is unsafe")
            payload = os.read(descriptor, _MAX_HEALTH_BYTES + 1)
            if len(payload) != metadata.st_size:
                raise VisualPredicateHealthError("resident health file is invalid")
        finally:
            os.close(descriptor)

        def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise VisualPredicateHealthError("resident health file is invalid")
                result[key] = item
            return result

        try:
            value = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=pairs_hook,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    VisualPredicateHealthError("resident health file is invalid")
                ),
            )
        except (json.JSONDecodeError, UnicodeError):
            raise VisualPredicateHealthError("resident health file is invalid") from None
        return cls.from_dict(value)

    def write_atomic_private(self, path: str | Path) -> None:
        health_path = self._private_path(path, existing=False)
        payload = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        if len(payload) > _MAX_HEALTH_BYTES:
            raise VisualPredicateHealthError("resident health file is invalid")
        temporary = health_path.with_name(f".{health_path.name}.{uuid.uuid4().hex}.tmp")
        descriptor: int | None = None
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, health_path)
            directory_flags = (
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
            )
            directory_descriptor = os.open(health_path.parent, directory_flags)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            raise VisualPredicateHealthError("resident health file is unavailable") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(OSError):
                temporary.unlink()


class VisualPredicateAdapter(ABC):
    @abstractmethod
    def assess(self, request: VisualPredicateRequest) -> VisualPredicateResult:
        raise NotImplementedError


class ConservativeVisualPredicateAdapter(VisualPredicateAdapter):
    """Deterministic unknown adapter for plumbing; it performs no visual analysis."""

    def __init__(self, *, model_id: str, model_revision: str, prompt_version: str):
        self.model_id = _validated_id(model_id, "model_id")
        self.model_revision = _validated_revision(model_revision)
        self.prompt_version = _validated_id(prompt_version, "prompt_version")

    def assess(self, request: VisualPredicateRequest) -> VisualPredicateResult:
        if not isinstance(request, VisualPredicateRequest):
            raise TypeError("visual predicate request is invalid")
        return VisualPredicateResult(
            event_id=request.event_id,
            predicate=request.predicate,
            answer=VisualPredicateAnswer.UNKNOWN,
            confidence=0.0,
            evidence_frames=(),
            model_id=self.model_id,
            model_revision=self.model_revision,
            prompt_version=self.prompt_version,
        )


class CommandVisualPredicateAdapter(VisualPredicateAdapter):
    """Run one local, offline predicate request through a bounded subprocess."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        model_id: str,
        model_revision: str,
        prompt_version: str,
        timeout_seconds: int = 180,
    ):
        if (
            not isinstance(command, Sequence)
            or isinstance(command, (str, bytes))
            or not command
            or any(not isinstance(part, str) or not part for part in command)
            or sum(part.count("{request}") for part in command) != 1
        ):
            raise ValueError("visual predicate command must contain {request} exactly once")
        if isinstance(timeout_seconds, bool) or not 1 <= timeout_seconds <= 3600:
            raise ValueError("visual predicate timeout is invalid")
        self.command = tuple(command)
        self.model_id = _validated_id(model_id, "model_id")
        self.model_revision = _validated_revision(model_revision)
        self.prompt_version = _validated_id(prompt_version, "prompt_version")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _environment() -> dict[str, str]:
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
                "DO_NOT_TRACK": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        return environment

    @staticmethod
    def _kill_process(process: subprocess.Popen[str]) -> None:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            with suppress(OSError):
                process.kill()

    @staticmethod
    def _read_admitted_frame(frame: VisualPredicateFrame) -> bytes:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(frame.path, flags)
        except OSError:
            raise VisualPredicateAdapterError(
                "visual predicate frame admission failed",
                code="frame_admission_failed",
            ) from None
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or not 0 < metadata.st_size <= _MAX_FRAME_BYTES
                or (
                    os.name == "posix"
                    and (
                        metadata.st_uid != os.getuid()
                        or stat.S_IMODE(metadata.st_mode) & 0o077 != 0
                    )
                )
            ):
                raise VisualPredicateAdapterError(
                    "visual predicate frame admission failed",
                    code="frame_admission_failed",
                )
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, _MAX_FRAME_BYTES - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_FRAME_BYTES:
                    raise VisualPredicateAdapterError(
                        "visual predicate frame admission failed",
                        code="frame_admission_failed",
                    )
                digest.update(chunk)
                chunks.append(chunk)
            if total != metadata.st_size or digest.hexdigest() != frame.sha256:
                raise VisualPredicateAdapterError(
                    "visual predicate frame admission failed",
                    code="frame_admission_failed",
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _write_private(path: Path, payload: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(descriptor)

    def _private_request(
        self,
        request: VisualPredicateRequest,
        directory: Path,
    ) -> Path:
        return _stage_private_request(
            request,
            directory,
            model_id=self.model_id,
            model_revision=self.model_revision,
            prompt_version=self.prompt_version,
        )

    def assess(self, request: VisualPredicateRequest) -> VisualPredicateResult:
        if not isinstance(request, VisualPredicateRequest):
            raise TypeError("visual predicate request is invalid")
        with tempfile.TemporaryDirectory(prefix="baby-monitor-predicate-") as raw_directory:
            directory = Path(raw_directory)
            with suppress(OSError):
                directory.chmod(0o700)
            request_path = self._private_request(request, directory)
            command = [part.replace("{request}", str(request_path)) for part in self.command]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=self._environment(),
                    start_new_session=os.name == "posix",
                )
            except OSError:
                raise VisualPredicateAdapterError(
                    "visual predicate runner could not start",
                    code="runner_start_failed",
                ) from None
            try:
                stdout, _stderr = process.communicate(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                self._kill_process(process)
                process.communicate()
                raise VisualPredicateAdapterError(
                    "visual predicate runner timed out",
                    code="runner_timeout",
                ) from None
            except BaseException:
                self._kill_process(process)
                process.communicate()
                raise
            if process.returncode != 0:
                raise VisualPredicateAdapterError(
                    "visual predicate runner failed",
                    code="runner_failed",
                )
            if not 0 < len(stdout.encode("utf-8")) <= MAX_VISUAL_PREDICATE_BYTES:
                raise VisualPredicateAdapterError(
                    "visual predicate runner output is invalid",
                    code="invalid_structured_output",
                )
            try:
                result = VisualPredicateResult.from_json(
                    stdout.strip(),
                    allowed_frame_ids=request.frame_ids,
                )
            except SchemaError:
                raise VisualPredicateAdapterError(
                    "visual predicate runner output is invalid",
                    code="invalid_structured_output",
                ) from None
            if (
                result.event_id != request.event_id
                or result.predicate is not request.predicate
                or result.model_id != self.model_id
                or result.model_revision != self.model_revision
                or result.prompt_version != self.prompt_version
            ):
                raise VisualPredicateAdapterError(
                    "visual predicate runner output binding failed",
                    code="result_binding_failed",
                )
            return result


def _stage_private_request(
    request: VisualPredicateRequest,
    directory: Path,
    *,
    model_id: str,
    model_revision: str,
    prompt_version: str,
) -> Path:
    staged_frames: list[dict[str, str]] = []
    total = 0
    for index, frame in enumerate(request.frames):
        payload = CommandVisualPredicateAdapter._read_admitted_frame(frame)
        total += len(payload)
        if total > _MAX_REQUEST_MEDIA_BYTES:
            raise VisualPredicateAdapterError(
                "visual predicate request media exceeds the limit",
                code="request_media_too_large",
            )
        staged_path = directory / f"frame_{index:02d}{_MEDIA_SUFFIX[frame.media_type]}"
        CommandVisualPredicateAdapter._write_private(staged_path, payload)
        staged_frames.append(
            {
                "frame_id": frame.frame_id,
                "sha256": frame.sha256,
                "media_type": frame.media_type,
                "path": str(staged_path),
            }
        )
    payload: dict[str, Any] = {
        "version": 1,
        "event_id": request.event_id,
        "predicate": request.predicate.value,
        "frames": staged_frames,
        "model_id": model_id,
        "model_revision": model_revision,
        "prompt_version": prompt_version,
        "result_contract": "visual_predicate_result_v1",
    }
    request_path = directory / "request.json"
    CommandVisualPredicateAdapter._write_private(
        request_path,
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )
    return request_path


def _strict_worker_json(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= _MAX_WORKER_LINE_BYTES:
        raise VisualPredicateAdapterError(
            "resident visual predicate worker output is invalid",
            code="invalid_worker_output",
        )

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise VisualPredicateAdapterError(
                    "resident visual predicate worker output is invalid",
                    code="invalid_worker_output",
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                VisualPredicateAdapterError(
                    "resident visual predicate worker output is invalid",
                    code="invalid_worker_output",
                )
            ),
        )
    except (json.JSONDecodeError, UnicodeError):
        raise VisualPredicateAdapterError(
            "resident visual predicate worker output is invalid",
            code="invalid_worker_output",
        ) from None
    if not isinstance(value, dict):
        raise VisualPredicateAdapterError(
            "resident visual predicate worker output is invalid",
            code="invalid_worker_output",
        )
    return value


class ResidentCommandVisualPredicateAdapter(VisualPredicateAdapter):
    """Reuse one bounded local worker while keeping every request privately staged."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        model_id: str,
        model_revision: str,
        prompt_version: str,
        max_jobs: int = 32,
        idle_exit_seconds: float = 30.0,
        max_runtime_seconds: float = 3600.0,
        startup_timeout_seconds: float = 180.0,
        request_timeout_seconds: float = 180.0,
        health_path: str | Path | None = None,
    ):
        if (
            not isinstance(command, Sequence)
            or isinstance(command, (str, bytes))
            or not command
            or any(
                not isinstance(part, str) or not part or "\x00" in part or "{request}" in part
                for part in command
            )
            or not Path(command[0]).is_absolute()
        ):
            raise ValueError("resident visual predicate command is invalid")
        if type(max_jobs) is not int or not 1 <= max_jobs <= 256:
            raise ValueError("resident visual predicate max_jobs is invalid")
        for value, name, maximum in (
            (idle_exit_seconds, "idle_exit_seconds", 3600.0),
            (max_runtime_seconds, "max_runtime_seconds", 86_400.0),
            (startup_timeout_seconds, "startup_timeout_seconds", 3600.0),
            (request_timeout_seconds, "request_timeout_seconds", 3600.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 1.0 <= float(value) <= maximum
            ):
                raise ValueError(f"resident visual predicate {name} is invalid")
        self.command = tuple(command)
        self.model_id = _validated_id(model_id, "model_id")
        self.model_revision = _validated_revision(model_revision)
        self.prompt_version = _validated_id(prompt_version, "prompt_version")
        self.max_jobs = max_jobs
        self.idle_exit_seconds = float(idle_exit_seconds)
        self.max_runtime_seconds = float(max_runtime_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self._lock = threading.Lock()
        self._health_lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._responses: queue.Queue[bytes | None] | None = None
        self._reader: threading.Thread | None = None
        self._served_jobs = 0
        self._health_state = "stopped"
        self._worker_starts = 0
        self._successful_jobs = 0
        self._failed_jobs = 0
        self._timeout_jobs = 0
        self._last_error_code: str | None = None
        if health_path is not None and not isinstance(health_path, (str, Path)):
            raise ValueError("resident visual predicate health_path is invalid")
        self._health_path = Path(health_path).expanduser() if health_path is not None else None
        if self._health_path is not None:
            self.health_snapshot().write_atomic_private(self._health_path)

    def _set_health_state(self, state: str) -> None:
        with self._health_lock:
            self._health_state = state

    def _record_success(self) -> None:
        with self._health_lock:
            self._successful_jobs += 1
            self._last_error_code = None
            self._health_state = "ready"

    def _record_failure(self, code: str) -> None:
        with self._health_lock:
            self._failed_jobs += 1
            if code == "worker_timeout":
                self._timeout_jobs += 1
            self._last_error_code = code
            self._health_state = "failed"

    def _persist_health(self) -> None:
        if self._health_path is None:
            return
        try:
            self.health_snapshot().write_atomic_private(self._health_path)
        except VisualPredicateHealthError:
            raise VisualPredicateAdapterError(
                "resident visual predicate health is unavailable",
                code="health_persistence_failed",
            ) from None

    def health_snapshot(self) -> ResidentVisualPredicateHealth:
        process = self._process
        worker_running = process is not None and process.poll() is None
        with self._health_lock:
            state = self._health_state
            if not worker_running and state in {"starting", "ready", "busy"}:
                state = "stopped"
            return ResidentVisualPredicateHealth(
                checked_at=datetime.now(UTC),
                state=state,
                worker_running=worker_running,
                worker_starts=self._worker_starts,
                worker_restarts=max(0, self._worker_starts - 1),
                successful_jobs=self._successful_jobs,
                failed_jobs=self._failed_jobs,
                timeout_jobs=self._timeout_jobs,
                current_worker_jobs=self._served_jobs if worker_running else 0,
                max_jobs=self.max_jobs,
                idle_exit_seconds=self.idle_exit_seconds,
                max_runtime_seconds=self.max_runtime_seconds,
                model_id=self.model_id,
                model_revision=self.model_revision,
                prompt_version=self.prompt_version,
                last_error_code=self._last_error_code,
            )

    @staticmethod
    def _reader_loop(
        process: subprocess.Popen[bytes],
        responses: queue.Queue[bytes | None],
    ) -> None:
        assert process.stdout is not None
        while True:
            raw = process.stdout.readline(_MAX_WORKER_LINE_BYTES + 1)
            if raw == b"":
                item: bytes | None = None
            elif len(raw) > _MAX_WORKER_LINE_BYTES or not raw.endswith(b"\n"):
                item = b""
            else:
                item = raw[:-1]
            try:
                responses.put(item, timeout=1.0)
            except queue.Full:
                CommandVisualPredicateAdapter._kill_process(process)
                return
            if item is None or item == b"":
                return

    def _read_worker_line(self, timeout: float) -> bytes:
        assert self._responses is not None
        try:
            raw = self._responses.get(timeout=timeout)
        except queue.Empty:
            raise VisualPredicateAdapterError(
                "resident visual predicate worker timed out",
                code="worker_timeout",
            ) from None
        if raw is None:
            raise VisualPredicateAdapterError(
                "resident visual predicate worker stopped",
                code="worker_stopped",
            )
        if raw == b"":
            raise VisualPredicateAdapterError(
                "resident visual predicate worker output is invalid",
                code="invalid_worker_output",
            )
        return raw

    def _stop_locked(self) -> None:
        process = self._process
        reader = self._reader
        self._process = None
        self._responses = None
        self._reader = None
        self._served_jobs = 0
        if process is None:
            self._set_health_state("stopped")
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            CommandVisualPredicateAdapter._kill_process(process)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5.0)
        if reader is not None:
            reader.join(timeout=1.0)
        if process.stdout is not None:
            with suppress(OSError):
                process.stdout.close()
        self._set_health_state("stopped")

    def _start_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._stop_locked()
        self._set_health_state("starting")
        try:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=CommandVisualPredicateAdapter._environment(),
                start_new_session=os.name == "posix",
            )
        except OSError:
            raise VisualPredicateAdapterError(
                "resident visual predicate worker could not start",
                code="worker_start_failed",
            ) from None
        responses: queue.Queue[bytes | None] = queue.Queue(maxsize=4)
        reader = threading.Thread(
            target=self._reader_loop,
            args=(process, responses),
            name="visual-predicate-worker-reader",
            daemon=True,
        )
        self._process = process
        self._responses = responses
        self._reader = reader
        reader.start()
        try:
            ready = _strict_worker_json(self._read_worker_line(self.startup_timeout_seconds))
            expected = {
                "version",
                "status",
                "model_id",
                "model_revision",
                "prompt_version",
                "max_jobs",
                "idle_exit_seconds",
                "max_runtime_seconds",
            }
            if (
                set(ready) != expected
                or ready["version"] != _WORKER_PROTOCOL_VERSION
                or ready["status"] != "ready"
                or ready["model_id"] != self.model_id
                or ready["model_revision"] != self.model_revision
                or ready["prompt_version"] != self.prompt_version
                or ready["max_jobs"] != self.max_jobs
                or ready["idle_exit_seconds"] != self.idle_exit_seconds
                or ready["max_runtime_seconds"] != self.max_runtime_seconds
            ):
                raise VisualPredicateAdapterError(
                    "resident visual predicate worker handshake is invalid",
                    code="invalid_worker_output",
                )
            with self._health_lock:
                self._worker_starts += 1
                self._health_state = "ready"
        except BaseException:
            self._stop_locked()
            raise

    def assess(self, request: VisualPredicateRequest) -> VisualPredicateResult:
        if not isinstance(request, VisualPredicateRequest):
            raise TypeError("visual predicate request is invalid")
        with self._lock:
            try:
                self._start_locked()
                self._set_health_state("busy")
                assert self._process is not None and self._process.stdin is not None
                with tempfile.TemporaryDirectory(
                    prefix="baby-monitor-resident-predicate-"
                ) as raw_directory:
                    directory = Path(raw_directory)
                    with suppress(OSError):
                        directory.chmod(0o700)
                    request_path = _stage_private_request(
                        request,
                        directory,
                        model_id=self.model_id,
                        model_revision=self.model_revision,
                        prompt_version=self.prompt_version,
                    )
                    control = (
                        json.dumps(
                            {
                                "version": _WORKER_PROTOCOL_VERSION,
                                "request_path": str(request_path),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n"
                    )
                    try:
                        self._process.stdin.write(control)
                        self._process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        raise VisualPredicateAdapterError(
                            "resident visual predicate worker stopped",
                            code="worker_stopped",
                        ) from None
                    envelope = _strict_worker_json(
                        self._read_worker_line(self.request_timeout_seconds)
                    )
                    self._served_jobs += 1
                    if (
                        set(envelope) == {"version", "status", "error_code"}
                        and envelope["version"] == _WORKER_PROTOCOL_VERSION
                        and envelope["status"] == "error"
                        and envelope["error_code"] == "runner_failed"
                    ):
                        raise VisualPredicateAdapterError(
                            "resident visual predicate worker failed",
                            code="runner_failed",
                        )
                    if (
                        set(envelope) != {"version", "status", "result"}
                        or envelope["version"] != _WORKER_PROTOCOL_VERSION
                        or envelope["status"] != "ok"
                    ):
                        raise VisualPredicateAdapterError(
                            "resident visual predicate worker output is invalid",
                            code="invalid_worker_output",
                        )
                    try:
                        result = VisualPredicateResult.from_dict(
                            envelope["result"],
                            allowed_frame_ids=request.frame_ids,
                        )
                    except SchemaError:
                        raise VisualPredicateAdapterError(
                            "resident visual predicate worker output is invalid",
                            code="invalid_worker_output",
                        ) from None
                    if (
                        result.event_id != request.event_id
                        or result.predicate is not request.predicate
                        or result.model_id != self.model_id
                        or result.model_revision != self.model_revision
                        or result.prompt_version != self.prompt_version
                    ):
                        raise VisualPredicateAdapterError(
                            "resident visual predicate worker output binding failed",
                            code="result_binding_failed",
                        )
            except VisualPredicateAdapterError as exc:
                self._stop_locked()
                self._record_failure(exc.code)
                self._persist_health()
                raise
            except BaseException:
                self._stop_locked()
                raise
            self._record_success()
            if self._served_jobs >= self.max_jobs:
                self._stop_locked()
            try:
                self._persist_health()
            except VisualPredicateAdapterError:
                self._stop_locked()
                raise
            return result

    def close(self) -> None:
        with self._lock:
            self._stop_locked()
            self._persist_health()

    def __enter__(self) -> ResidentCommandVisualPredicateAdapter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _validated_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"visual predicate adapter {field_name} is invalid")
    return value


def _validated_revision(value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise ValueError("visual predicate adapter model_revision is invalid")
    return value
