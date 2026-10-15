from __future__ import annotations

import os
import re
import signal as signal_module
import stat
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

from baby_monitor.adapters.camera import CameraSource
from baby_monitor.frame_transform import validate_image_size, validate_rotation_degrees
from baby_monitor.pose_protocol import (
    MAX_POSE_FILE_BYTES,
    PoseFrameBatch,
    PoseProtocolError,
    PoseScanRequest,
)
from baby_monitor.yolo_pose_model import YoloPoseModelManifest


class PoseEstimatorError(RuntimeError):
    pass


class PoseEstimator(ABC):
    @abstractmethod
    def scan(self, source: CameraSource) -> PoseFrameBatch:
        raise NotImplementedError


class CommandYoloPoseEstimator(PoseEstimator):
    """Run a reviewed pose checkpoint in one bounded offline subprocess."""

    runner_adapter = "ultralytics_yolo_pose_subprocess_v1"
    _execution_lock_poll_seconds = 0.1
    _execution_lock_handoff_seconds = 0.12

    def __init__(
        self,
        *,
        model_path: str | Path,
        manifest_path: str | Path,
        manifest: YoloPoseModelManifest,
        confidence: float,
        sample_fps: float,
        max_frames: int,
        device: str,
        command: Sequence[str],
        image_size: int = 640,
        rotation_degrees: int = 0,
        timeout_seconds: float = 600.0,
        execution_lock_path: str | Path | None = None,
        execution_lock_timeout_seconds: float = 600.0,
    ):
        self.model_path = Path(model_path).expanduser()
        try:
            verification = manifest.verify_weight(self.model_path)
        except RuntimeError as exc:
            raise PoseEstimatorError(f"{exc}; automatic downloads are disabled") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("YOLO pose confidence must be between 0 and 1")
        if not 0.01 <= sample_fps <= 10.0:
            raise ValueError("YOLO pose sample_fps must be between 0.01 and 10")
        if type(max_frames) is not int or not 1 <= max_frames <= 10_000:
            raise ValueError("YOLO pose max_frames must be between 1 and 10000")
        if device != "cpu" and re.fullmatch(r"\d+", device) is None:
            raise ValueError("YOLO pose device must be 'cpu' or one numeric GPU index")
        validate_image_size(image_size)
        validate_rotation_degrees(rotation_degrees)
        if not command or not all(isinstance(part, str) and part for part in command):
            raise ValueError("YOLO pose runner command cannot be empty")
        if not Path(command[0]).expanduser().is_absolute():
            raise ValueError("YOLO pose runner command must start with an absolute executable")
        if sum(part.count("{request}") for part in command) != 1:
            raise ValueError("YOLO pose runner command must contain {request} exactly once")
        if any("\x00" in part for part in command):
            raise ValueError("YOLO pose runner command contains an invalid character")
        if not 1.0 <= timeout_seconds <= 3_600.0:
            raise ValueError("YOLO pose runner timeout must be between 1 and 3600 seconds")
        if not 1.0 <= execution_lock_timeout_seconds <= 3_600.0:
            raise ValueError("YOLO pose execution lock timeout must be between 1 and 3600 seconds")
        lock_path = (
            Path(execution_lock_path).expanduser() if execution_lock_path is not None else None
        )
        if lock_path is not None and (
            not lock_path.is_absolute()
            or lock_path.name in {"", ".", ".."}
            or os.name == "posix"
            and len(lock_path.parts) >= 3
            and tuple(part.casefold() for part in lock_path.parts[1:3]) == ("mnt", "c")
        ):
            raise ValueError("YOLO pose execution lock path is invalid")

        manifest_file = Path(manifest_path).expanduser()
        if manifest_file.is_symlink() or not manifest_file.is_file():
            raise PoseEstimatorError("YOLO pose subprocess manifest must be a regular local file")
        self.manifest = manifest
        self.manifest_path = manifest_file.resolve()
        self.model_path = self.model_path.resolve()
        self.model_sha256 = verification.sha256
        self.confidence = confidence
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.device = device
        self.image_size = image_size
        self.rotation_degrees = rotation_degrees
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.execution_lock_path = lock_path
        self.execution_lock_timeout_seconds = execution_lock_timeout_seconds
        self.last_execution_lock_wait_seconds: float | None = None

    @contextmanager
    def _execution_gate(self) -> Iterator[None]:
        path = self.execution_lock_path
        if path is None:
            self.last_execution_lock_wait_seconds = 0.0
            yield
            return
        if os.name != "posix":
            raise PoseEstimatorError("YOLO pose execution lock is unavailable")
        try:
            import fcntl

            parent = path.parent
            parent_metadata = parent.lstat()
            if (
                parent.is_symlink()
                or not parent.is_dir()
                or parent_metadata.st_uid != os.geteuid()
                or (parent_metadata.st_mode & 0o777) != 0o700
                or parent.resolve(strict=True) != parent
            ):
                raise PoseEstimatorError("YOLO pose execution lock is unsafe")
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
        except (OSError, PoseEstimatorError):
            raise PoseEstimatorError("YOLO pose execution lock is unsafe") from None
        acquired = False
        wait_started = time.monotonic()
        self.last_execution_lock_wait_seconds = None
        try:
            try:
                metadata = os.fstat(descriptor)
                path_metadata = path.lstat()
            except OSError:
                raise PoseEstimatorError("YOLO pose execution lock is unsafe") from None
            if (
                not stat.S_ISREG(metadata.st_mode)
                or path.is_symlink()
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o777 != 0o600
                or (metadata.st_dev, metadata.st_ino)
                != (path_metadata.st_dev, path_metadata.st_ino)
            ):
                raise PoseEstimatorError("YOLO pose execution lock is unsafe")
            deadline = time.monotonic() + self.execution_lock_timeout_seconds
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    self.last_execution_lock_wait_seconds = round(
                        time.monotonic() - wait_started,
                        6,
                    )
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise PoseEstimatorError("YOLO pose execution lock timed out") from None
                    time.sleep(self._execution_lock_poll_seconds)
                except OSError:
                    raise PoseEstimatorError("YOLO pose execution lock is unavailable") from None
            yield
        finally:
            if acquired:
                with suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            with suppress(OSError):
                os.close(descriptor)
            if acquired:
                time.sleep(self._execution_lock_handoff_seconds)

    @staticmethod
    def _subprocess_environment() -> dict[str, str]:
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
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                "YOLO_AUTOINSTALL": "false",
                "YOLO_OFFLINE": "true",
            }
        )
        return environment

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal_module.SIGKILL)
        else:
            with suppress(OSError):
                process.kill()

    def _request(self, source: CameraSource) -> PoseScanRequest:
        source_path = Path(source.read_uri).expanduser()
        if not source_path.is_absolute() or source_path.is_symlink() or not source_path.is_file():
            raise PoseEstimatorError("YOLO pose subprocess source must be a regular local file")
        return PoseScanRequest(
            camera_id=source.camera_id,
            pose_adapter=self.runner_adapter,
            source_path=str(source_path.resolve()),
            model_path=str(self.model_path),
            manifest_path=str(self.manifest_path),
            model_id=self.manifest.model_id,
            model_revision=self.manifest.revision,
            model_sha256=self.model_sha256,
            confidence=self.confidence,
            sample_fps=self.sample_fps,
            max_frames=self.max_frames,
            device=self.device,
            image_size=self.image_size,
            rotation_degrees=self.rotation_degrees,
        )

    def scan(self, source: CameraSource) -> PoseFrameBatch:
        if source.kind != "file":
            raise PoseEstimatorError("the YOLO pose scanner supports bounded local files only")
        request = self._request(source)
        with tempfile.TemporaryDirectory(prefix="baby-monitor-pose-") as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(request.to_json(), encoding="utf-8")
            with suppress(OSError):
                request_path.chmod(0o600)
            command = [part.replace("{request}", str(request_path)) for part in self.command]
            with self._execution_gate():
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=self._subprocess_environment(),
                        start_new_session=os.name == "posix",
                    )
                except OSError:
                    raise PoseEstimatorError("YOLO pose subprocess could not start") from None
                try:
                    stdout, _ = process.communicate(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._kill_process_tree(process)
                    process.communicate()
                    raise PoseEstimatorError("YOLO pose subprocess timed out") from None
                except BaseException:
                    self._kill_process_tree(process)
                    process.communicate()
                    raise
                if process.returncode != 0:
                    raise PoseEstimatorError("YOLO pose subprocess failed")
            if len(stdout.encode("utf-8")) > MAX_POSE_FILE_BYTES:
                raise PoseEstimatorError("YOLO pose subprocess returned too much output")
            try:
                result = PoseFrameBatch.from_json(stdout.strip())
            except PoseProtocolError:
                raise PoseEstimatorError("YOLO pose subprocess returned invalid JSON") from None

        if (
            result.camera_id != source.camera_id
            or result.pose_adapter != self.runner_adapter
            or result.model_id != self.manifest.model_id
            or result.model_revision != self.manifest.revision
            or result.model_sha256 != self.model_sha256
        ):
            raise PoseEstimatorError(
                "YOLO pose subprocess metadata does not match the reviewed model"
            )
        return result
