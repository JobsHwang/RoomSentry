from __future__ import annotations

import json
import os
import re
import secrets
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .domain import SignalEvent
from .signal_outbox import SignalOutbox, SignalOutboxStatus

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_MAX_ROUTE_MAP_BYTES = 65_536
_MAX_ROUTES = 16
_SAFE_COMMAND_PATH = "/usr/bin:/bin"
_PACKAGE_SOURCE_ROOT = str(Path(__file__).resolve().parents[1])


class SignalDispatchError(RuntimeError):
    pass


def _strict_json(path: Path) -> object:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-standard JSON number")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise SignalDispatchError("HA signal route map is invalid") from exc


def _private_file(path: Path, *, label: str, max_bytes: int) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SignalDispatchError(f"{label} is missing or unsafe") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not 1 <= metadata.st_size <= max_bytes
    ):
        raise SignalDispatchError(f"{label} is missing or unsafe")
    if os.name == "posix" and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise SignalDispatchError(f"{label} permissions are unsafe")


def _require_git_external_posix_path(path: Path, *, label: str) -> None:
    if os.name != "posix":
        return
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SignalDispatchError(f"{label} is missing or unsafe") from exc
    parts = resolved.parts
    if len(parts) >= 3 and parts[1] == "mnt" and parts[2].casefold() == "c":
        raise SignalDispatchError(f"{label} must remain on the private Linux filesystem")
    for parent in (resolved, *resolved.parents):
        if (parent / ".git").exists():
            raise SignalDispatchError(f"{label} must remain outside the source worktree")


@dataclass(frozen=True, slots=True)
class HASignalRoute:
    camera_id: str
    manifest_path: Path = field(repr=False)
    signal_camera_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or _SAFE_ID.fullmatch(self.camera_id) is None:
            raise SignalDispatchError("HA signal route camera_id is invalid")
        if self.signal_camera_id is not None and (
            not isinstance(self.signal_camera_id, str)
            or _SAFE_ID.fullmatch(self.signal_camera_id) is None
        ):
            raise SignalDispatchError("HA signal route signal_camera_id is invalid")
        if not isinstance(self.manifest_path, Path) or not self.manifest_path.is_absolute():
            raise SignalDispatchError("HA signal route manifest path is invalid")
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise SignalDispatchError("HA signal route manifest is missing or unsafe")


@dataclass(frozen=True, slots=True)
class HASignalRouteMap:
    routes: tuple[HASignalRoute, ...]
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version not in {1, 2}:
            raise SignalDispatchError("unsupported HA signal route map version")
        if not isinstance(self.routes, tuple) or not 1 <= len(self.routes) <= _MAX_ROUTES:
            raise SignalDispatchError("HA signal route count is invalid")
        if any(not isinstance(route, HASignalRoute) for route in self.routes):
            raise SignalDispatchError("HA signal route is invalid")
        if self.version == 1 and any(route.signal_camera_id is not None for route in self.routes):
            raise SignalDispatchError("HA signal route version fields are inconsistent")
        if self.version == 2 and any(route.signal_camera_id is None for route in self.routes):
            raise SignalDispatchError("HA signal route version fields are inconsistent")
        signal_camera_ids = tuple(
            route.signal_camera_id or route.camera_id for route in self.routes
        )
        if len(set(signal_camera_ids)) != len(signal_camera_ids):
            raise SignalDispatchError("HA signal route signal_camera_id values must be unique")

    @classmethod
    def read(cls, path: str | Path) -> HASignalRouteMap:
        route_path = Path(path).expanduser()
        if not route_path.is_absolute():
            raise SignalDispatchError("HA signal route map path must be absolute")
        _require_git_external_posix_path(route_path, label="HA signal route map")
        _private_file(route_path, label="HA signal route map", max_bytes=_MAX_ROUTE_MAP_BYTES)
        value = _strict_json(route_path)
        if not isinstance(value, dict) or set(value) != {"version", "routes"}:
            raise SignalDispatchError("HA signal route map fields are invalid")
        if type(value["version"]) is not int or value["version"] not in {1, 2}:
            raise SignalDispatchError("unsupported HA signal route map version")
        if not isinstance(value["routes"], list):
            raise SignalDispatchError("HA signal route map routes are invalid")
        routes: list[HASignalRoute] = []
        for item in value["routes"]:
            expected_fields = (
                {"camera_id", "manifest_path"}
                if value["version"] == 1
                else {"signal_camera_id", "camera_id", "manifest_path"}
            )
            if not isinstance(item, dict) or set(item) != expected_fields:
                raise SignalDispatchError("HA signal route fields are invalid")
            if not isinstance(item["camera_id"], str) or not isinstance(item["manifest_path"], str):
                raise SignalDispatchError("HA signal route fields are invalid")
            signal_camera_id = item.get("signal_camera_id")
            if signal_camera_id is not None and not isinstance(signal_camera_id, str):
                raise SignalDispatchError("HA signal route fields are invalid")
            routes.append(
                HASignalRoute(
                    camera_id=item["camera_id"],
                    manifest_path=Path(item["manifest_path"]).expanduser(),
                    signal_camera_id=signal_camera_id,
                )
            )
        return cls(routes=tuple(routes), version=value["version"])

    def for_camera(self, camera_id: str) -> HASignalRoute | None:
        return next(
            (
                route
                for route in self.routes
                if (route.signal_camera_id or route.camera_id) == camera_id
            ),
            None,
        )


CommandRunner = Callable[[Sequence[str], float, dict[str, str]], int]


def _classify_child_failure(return_code: int, stderr: bytes) -> int:
    if return_code in {0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}:
        return return_code
    detail = stderr[:8_192].decode("utf-8", errors="replace")
    if "buffer has no segments for the requested event window" in detail:
        return 4
    if "required media command is missing" in detail:
        return 5
    if "FFmpeg could not" in detail or "assembled evidence is not readable" in detail:
        return 6
    if "config" in detail and "invalid" in detail:
        return 7
    if "signal camera_id does not match" in detail:
        return 10
    if (
        "Permission denied" in detail
        or "PermissionError:" in detail
        or "Read-only file system" in detail
    ):
        return 9
    if "usage:" in detail:
        return 12
    if "No module named baby_monitor" in detail:
        return 17
    if "TypeError:" in detail:
        return 13
    if "sqlite3." in detail:
        return 14
    if "TimeoutExpired" in detail:
        return 15
    if "Traceback (most recent call last)" in detail:
        return 16
    return return_code


def _default_runner(argv: Sequence[str], timeout_seconds: float, env: dict[str, str]) -> int:
    try:
        with tempfile.TemporaryFile(mode="w+b") as private_stderr:
            completed = subprocess.run(
                tuple(argv),
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=private_stderr,
                timeout=timeout_seconds,
                env=env,
            )
            private_stderr.seek(0)
            stderr_prefix = private_stderr.read(8_192)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SignalDispatchError("buffer evidence command failed") from exc
    return _classify_child_failure(completed.returncode, stderr_prefix)


def _existing_absolute_file(path: str | Path, *, label: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or value.is_symlink() or not value.is_file():
        raise SignalDispatchError(f"{label} is missing or unsafe")
    return value


def _private_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise SignalDispatchError("signal dispatch staging root is missing or unsafe")
    if os.name == "posix":
        metadata = root.stat()
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise SignalDispatchError("signal dispatch staging root permissions are unsafe")
    return root


@dataclass(frozen=True, slots=True)
class BufferedSignalCommand:
    python_executable: Path = field(repr=False)
    config_path: Path = field(repr=False)
    staging_root: Path = field(repr=False)
    database_path: Path | None = field(default=None, repr=False)
    output_root: Path | None = field(default=None, repr=False)
    timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        _existing_absolute_file(self.python_executable, label="Python executable")
        _existing_absolute_file(self.config_path, label="baby monitor config")
        _private_root(self.staging_root)
        _require_git_external_posix_path(
            self.staging_root,
            label="signal dispatch staging root",
        )
        if self.database_path is not None:
            if not self.database_path.is_absolute():
                raise SignalDispatchError("analysis database path must be absolute")
            _private_root(self.database_path.parent)
        if self.output_root is not None:
            _private_root(self.output_root)
        if not isinstance(self.timeout_seconds, (int, float)) or not (
            10.0 <= float(self.timeout_seconds) <= 3_600.0
        ):
            raise SignalDispatchError("buffer evidence timeout is invalid")

    def argv(self, *, signal_path: Path, route: HASignalRoute) -> tuple[str, ...]:
        arguments = [
            str(self.python_executable),
            "-m",
            "baby_monitor",
            "enqueue-signal-event",
            "--config",
            str(self.config_path),
            "--signal-json",
            str(signal_path),
            "--manifest",
            str(route.manifest_path),
        ]
        if self.database_path is not None:
            arguments.extend(("--database", str(self.database_path)))
        if self.output_root is not None:
            arguments.extend(("--output-root", str(self.output_root)))
        return tuple(arguments)


@dataclass(frozen=True, slots=True)
class SignalDispatchAttempt:
    status: str
    outbox_id: str | None
    error_code: str | None

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": self.status,
            "outbox_id": self.outbox_id,
            "error_code": self.error_code,
            "media_paths_disclosed": False,
            "entity_ids_disclosed": False,
            "signal_payload_disclosed": False,
            "remote_used": False,
            "model_loaded": False,
        }


@dataclass(frozen=True, slots=True)
class SignalDispatchWorkerSummary:
    processed: int
    completed: int
    retry_scheduled: int
    failed: int
    idle_polls: int
    stop_reason: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": "complete",
            "processed": self.processed,
            "completed": self.completed,
            "retry_scheduled": self.retry_scheduled,
            "failed": self.failed,
            "idle_polls": self.idle_polls,
            "stop_reason": self.stop_reason,
            "media_paths_disclosed": False,
            "entity_ids_disclosed": False,
            "signal_payload_disclosed": False,
            "remote_used": False,
            "model_loaded": False,
        }


def _write_signal(root: Path, outbox_id: str, signal: SignalEvent) -> Path:
    if _SAFE_ID.fullmatch(outbox_id) is None:
        raise SignalDispatchError("outbox identity is invalid")
    path = root / f"signal-{outbox_id}-{secrets.token_hex(8)}.json"
    payload = json.dumps(
        signal.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise SignalDispatchError("private signal staging failed") from exc
    return path


def _route_signal(signal: SignalEvent, route: HASignalRoute) -> SignalEvent:
    if signal.camera_id == route.camera_id:
        return signal
    payload = signal.to_dict()
    payload["camera_id"] = route.camera_id
    return SignalEvent.from_dict(payload)


def dispatch_one_signal(
    *,
    outbox: SignalOutbox,
    routes: HASignalRouteMap,
    command: BufferedSignalCommand,
    lease_seconds: int = 240,
    max_attempts: int = 5,
    retry_delay_seconds: int = 30,
    runner: CommandRunner = _default_runner,
) -> SignalDispatchAttempt:
    if lease_seconds < int(command.timeout_seconds) + 10:
        raise SignalDispatchError("signal lease must outlive the buffer evidence timeout")
    claim = outbox.claim_next(
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
    )
    if claim is None:
        return SignalDispatchAttempt(status="idle", outbox_id=None, error_code=None)

    route = routes.for_camera(claim.signal.camera_id)
    if route is None:
        status = outbox.record_failure(
            claim.outbox_id,
            claim.lease_token,
            error_code="route_missing",
            retry_delay_seconds=retry_delay_seconds,
            max_attempts=max_attempts,
        )
        return SignalDispatchAttempt(
            status="failed" if status is SignalOutboxStatus.FAILED else "retry_scheduled",
            outbox_id=claim.outbox_id,
            error_code="route_missing",
        )

    signal_path: Path | None = None
    try:
        signal_path = _write_signal(
            command.staging_root,
            claim.outbox_id,
            _route_signal(claim.signal, route),
        )
        return_code = runner(
            command.argv(signal_path=signal_path, route=route),
            float(command.timeout_seconds),
            {
                "HOME": str(command.staging_root),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": _SAFE_COMMAND_PATH,
                "PYTHONPATH": _PACKAGE_SOURCE_ROOT,
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    except SignalDispatchError:
        return_code = 2
    finally:
        if signal_path is not None:
            try:
                signal_path.unlink()
            except OSError as exc:
                raise SignalDispatchError("private signal staging cleanup failed") from exc

    if return_code == 0:
        outbox.acknowledge(claim.outbox_id, claim.lease_token)
        return SignalDispatchAttempt(
            status="completed",
            outbox_id=claim.outbox_id,
            error_code=None,
        )
    if return_code == 3:
        error_code = "analysis_queue_full"
    elif return_code == 4:
        error_code = "buffer_window_unavailable"
    elif return_code == 5:
        error_code = "media_command_unavailable"
    elif return_code == 6:
        error_code = "buffer_media_processing_failed"
    elif return_code == 7:
        error_code = "buffer_config_invalid"
    elif return_code == 8:
        error_code = "signal_staging_invalid"
    elif return_code == 9:
        error_code = "buffer_storage_unavailable"
    elif return_code == 10:
        error_code = "buffer_request_invalid"
    elif return_code == 11:
        error_code = "buffer_runtime_failed"
    elif return_code == 12:
        error_code = "buffer_command_invalid"
    elif return_code == 13:
        error_code = "buffer_contract_type_error"
    elif return_code == 14:
        error_code = "buffer_database_failed"
    elif return_code == 15:
        error_code = "buffer_command_timeout"
    elif return_code == 16:
        error_code = "buffer_child_exception"
    elif return_code == 17:
        error_code = "buffer_module_unavailable"
    else:
        error_code = "buffer_evidence_failed"
    status = outbox.record_failure(
        claim.outbox_id,
        claim.lease_token,
        error_code=error_code,
        retry_delay_seconds=retry_delay_seconds,
        max_attempts=max_attempts,
    )
    return SignalDispatchAttempt(
        status="failed" if status is SignalOutboxStatus.FAILED else "retry_scheduled",
        outbox_id=claim.outbox_id,
        error_code=error_code,
    )


def run_signal_dispatch_worker(
    *,
    outbox: SignalOutbox,
    routes: HASignalRouteMap,
    command: BufferedSignalCommand,
    lease_seconds: int = 240,
    max_attempts: int = 5,
    retry_delay_seconds: int = 30,
    max_jobs: int | None = None,
    max_runtime_seconds: float = 86_400.0,
    poll_seconds: float = 2.0,
    runner: CommandRunner = _default_runner,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> SignalDispatchWorkerSummary:
    if max_jobs is not None and (type(max_jobs) is not int or not 1 <= max_jobs <= 100_000):
        raise SignalDispatchError("signal dispatch max_jobs is invalid")
    if not isinstance(max_runtime_seconds, (int, float)) or not (
        1.0 <= float(max_runtime_seconds) <= 604_800.0
    ):
        raise SignalDispatchError("signal dispatch max runtime is invalid")
    if not isinstance(poll_seconds, (int, float)) or not (0.05 <= float(poll_seconds) <= 60.0):
        raise SignalDispatchError("signal dispatch poll interval is invalid")

    started_at = monotonic()
    processed = completed = retry_scheduled = failed = idle_polls = 0
    stop_reason = "max_runtime"
    while monotonic() - started_at < float(max_runtime_seconds):
        attempt = dispatch_one_signal(
            outbox=outbox,
            routes=routes,
            command=command,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            runner=runner,
        )
        if attempt.status == "idle":
            idle_polls += 1
            remaining = float(max_runtime_seconds) - (monotonic() - started_at)
            if remaining <= 0:
                break
            sleeper(min(float(poll_seconds), remaining))
            continue

        processed += 1
        if attempt.status == "completed":
            completed += 1
        elif attempt.status == "retry_scheduled":
            retry_scheduled += 1
        elif attempt.status == "failed":
            failed += 1
        else:
            raise SignalDispatchError("signal dispatch returned an invalid status")
        if max_jobs is not None and processed >= max_jobs:
            stop_reason = "max_jobs"
            break

    return SignalDispatchWorkerSummary(
        processed=processed,
        completed=completed,
        retry_scheduled=retry_scheduled,
        failed=failed,
        idle_polls=idle_polls,
        stop_reason=stop_reason,
    )
