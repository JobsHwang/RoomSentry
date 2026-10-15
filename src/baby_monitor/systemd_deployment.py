from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config import Settings, load_settings
from .visual_predicate_shadow_runtime import (
    VisualPredicateShadowRuntimeSettings,
    load_visual_predicate_shadow_runtime_settings,
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
_PROFILE_REQUIRED_KEYS = {
    "version",
    "deployment_id",
    "project_root",
    "config_path",
    "database_path",
    "output_root",
    "buffer_run_seconds",
    "confirm_ephemeral_buffer_eviction",
    "cameras",
}
_PROFILE_OPTIONAL_KEYS = {
    "candidate_detector",
    "deferred_reports",
    "pose_presence_shadow",
    "trial_run_seconds",
}
_CAMERA_REQUIRED_KEYS = {"camera_id", "rtsp_uri", "buffer_root"}
_CAMERA_OPTIONAL_KEYS = {"image_size", "rotation_degrees"}
_POSE_PRESENCE_SHADOW_REQUIRED_KEYS = {
    "profile_path",
    "temporal_attestation_path",
    "human_audit_report_path",
    "detector_manifest_path",
    "detector_weights_path",
    "detector_runner_executable",
    "pose_manifest_path",
    "pose_weights_path",
    "pose_runner_executable",
}
_POSE_PRESENCE_SHADOW_OPTIONAL_KEYS = {
    "detector_device",
    "pose_device",
    "pose_timeout_seconds",
    "sample_interval_seconds",
}


class DeploymentProfileError(ValueError):
    pass


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DeploymentProfileError(f"{label} fields are invalid")


def _absolute_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or len(value) > 4_096:
        raise DeploymentProfileError(f"{field} must be a bounded absolute path")
    if any(ord(character) < 32 for character in value):
        raise DeploymentProfileError(f"{field} contains control characters")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise DeploymentProfileError(f"{field} must be an absolute path")
    return path


def _regular_input_path(value: object, *, field: str, executable: bool = False) -> Path:
    path = _absolute_path(value, field=field)
    if (
        not path.is_file()
        or path.is_symlink()
        and not executable
        or executable
        and not os.access(path, os.X_OK)
    ):
        raise DeploymentProfileError(f"{field} must be an existing regular local file")
    return path


def _loopback_rtsp_uri(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise DeploymentProfileError("camera rtsp_uri is invalid")
    if any(ord(character) < 32 for character in value):
        raise DeploymentProfileError("camera rtsp_uri contains control characters")
    parsed = urlsplit(value)
    if parsed.scheme != "rtsp" or not parsed.hostname or not parsed.path:
        raise DeploymentProfileError("camera rtsp_uri must be a complete RTSP URI")
    if parsed.username is not None or parsed.password is not None:
        raise DeploymentProfileError("camera rtsp_uri must not contain credentials")
    if parsed.query or parsed.fragment:
        raise DeploymentProfileError("camera rtsp_uri must not contain query or fragment data")
    try:
        host = ipaddress.ip_address(parsed.hostname)
        port = parsed.port
        if port is not None and not 1 <= port <= 65_535:
            raise ValueError
    except ValueError:
        raise DeploymentProfileError("camera rtsp_uri host or port is invalid") from None
    if not host.is_loopback:
        raise DeploymentProfileError("camera rtsp_uri must use an IP loopback address")
    return value


@dataclass(frozen=True, slots=True)
class DeploymentCamera:
    camera_id: str
    rtsp_uri: str
    buffer_root: Path
    image_size: int = 640
    rotation_degrees: int = 0

    @classmethod
    def from_dict(cls, value: object) -> DeploymentCamera:
        if not isinstance(value, dict):
            raise DeploymentProfileError("deployment camera must be an object")
        fields = set(value)
        if not _CAMERA_REQUIRED_KEYS.issubset(fields) or not fields.issubset(
            _CAMERA_REQUIRED_KEYS | _CAMERA_OPTIONAL_KEYS
        ):
            raise DeploymentProfileError("deployment camera fields are invalid")
        camera_id = value["camera_id"]
        if not isinstance(camera_id, str) or _SAFE_ID.fullmatch(camera_id) is None:
            raise DeploymentProfileError("camera_id must be a safe non-personal identifier")
        image_size = value.get("image_size", 640)
        if type(image_size) is not int or not 320 <= image_size <= 1280 or image_size % 32:
            raise DeploymentProfileError(
                "camera image_size must be a multiple of 32 between 320 and 1280"
            )
        rotation_degrees = value.get("rotation_degrees", 0)
        if type(rotation_degrees) is not int or rotation_degrees not in {0, 90, 180, 270}:
            raise DeploymentProfileError(
                "camera rotation_degrees must be one of 0, 90, 180, or 270"
            )
        return cls(
            camera_id=camera_id,
            rtsp_uri=_loopback_rtsp_uri(value["rtsp_uri"]),
            buffer_root=_absolute_path(value["buffer_root"], field="camera buffer_root"),
            image_size=image_size,
            rotation_degrees=rotation_degrees,
        )

    @property
    def camera_directory(self) -> Path:
        return self.buffer_root / self.camera_id

    @property
    def manifest_path(self) -> Path:
        return self.camera_directory / "manifest.json"

    @property
    def health_path(self) -> Path:
        return self.camera_directory / "health.json"

    @property
    def state_path(self) -> Path:
        return self.camera_directory / "candidate-scan-state.json"

    @property
    def presence_path(self) -> Path:
        return self.camera_directory / "stream-presence-batch.json"

    @property
    def shadow_presence_path(self) -> Path:
        return self.camera_directory / "pose-presence-shadow-batch.json"


@dataclass(frozen=True, slots=True)
class PosePresenceShadowDeployment:
    profile_path: Path
    temporal_attestation_path: Path
    human_audit_report_path: Path
    detector_manifest_path: Path
    detector_weights_path: Path
    detector_runner_executable: Path
    pose_manifest_path: Path
    pose_weights_path: Path
    pose_runner_executable: Path
    detector_device: str = "0"
    pose_device: str = "0"
    pose_timeout_seconds: float = 600.0
    sample_interval_seconds: float = 300.0

    @classmethod
    def from_dict(cls, value: object) -> PosePresenceShadowDeployment:
        if not isinstance(value, dict):
            raise DeploymentProfileError("pose_presence_shadow must be an object")
        fields = set(value)
        if not _POSE_PRESENCE_SHADOW_REQUIRED_KEYS.issubset(fields) or not fields.issubset(
            _POSE_PRESENCE_SHADOW_REQUIRED_KEYS | _POSE_PRESENCE_SHADOW_OPTIONAL_KEYS
        ):
            raise DeploymentProfileError("pose_presence_shadow fields are invalid")
        devices: dict[str, str] = {}
        for field in ("detector_device", "pose_device"):
            device = value.get(field, "0")
            if not isinstance(device, str) or (
                device != "cpu" and re.fullmatch(r"\d+", device) is None
            ):
                raise DeploymentProfileError(f"pose_presence_shadow {field} is invalid")
            devices[field] = device
        timeout = value.get("pose_timeout_seconds", 600.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 1.0 <= float(timeout) <= 3_600.0
        ):
            raise DeploymentProfileError("pose_presence_shadow timeout is invalid")
        sample_interval = value.get("sample_interval_seconds", 300.0)
        if (
            isinstance(sample_interval, bool)
            or not isinstance(sample_interval, (int, float))
            or not 30.0 <= float(sample_interval) <= 3_600.0
        ):
            raise DeploymentProfileError("pose_presence_shadow sample interval is invalid")
        return cls(
            profile_path=_regular_input_path(
                value["profile_path"], field="pose_presence_shadow profile_path"
            ),
            temporal_attestation_path=_regular_input_path(
                value["temporal_attestation_path"],
                field="pose_presence_shadow temporal_attestation_path",
            ),
            human_audit_report_path=_regular_input_path(
                value["human_audit_report_path"],
                field="pose_presence_shadow human_audit_report_path",
            ),
            detector_manifest_path=_regular_input_path(
                value["detector_manifest_path"],
                field="pose_presence_shadow detector_manifest_path",
            ),
            detector_weights_path=_regular_input_path(
                value["detector_weights_path"],
                field="pose_presence_shadow detector_weights_path",
            ),
            detector_runner_executable=_regular_input_path(
                value["detector_runner_executable"],
                field="pose_presence_shadow detector_runner_executable",
                executable=True,
            ),
            pose_manifest_path=_regular_input_path(
                value["pose_manifest_path"],
                field="pose_presence_shadow pose_manifest_path",
            ),
            pose_weights_path=_regular_input_path(
                value["pose_weights_path"],
                field="pose_presence_shadow pose_weights_path",
            ),
            pose_runner_executable=_regular_input_path(
                value["pose_runner_executable"],
                field="pose_presence_shadow pose_runner_executable",
                executable=True,
            ),
            detector_device=devices["detector_device"],
            pose_device=devices["pose_device"],
            pose_timeout_seconds=float(timeout),
            sample_interval_seconds=float(sample_interval),
        )


@dataclass(frozen=True, slots=True)
class RuntimeDeploymentProfile:
    deployment_id: str
    project_root: Path
    config_path: Path
    database_path: Path
    output_root: Path
    buffer_run_seconds: int
    confirm_ephemeral_buffer_eviction: bool
    cameras: tuple[DeploymentCamera, ...]
    candidate_detector: str = "scene"
    pose_presence_shadow: PosePresenceShadowDeployment | None = None
    trial_run_seconds: int | None = None
    version: int = 1
    deferred_reports: bool = False

    def __post_init__(self) -> None:
        if type(self.deferred_reports) is not bool:
            raise DeploymentProfileError("deferred_reports must be a boolean")

    @property
    def python_executable(self) -> Path:
        return self.project_root / ".venv" / "bin" / "python"

    @property
    def runtime_trial_ledger_path(self) -> Path:
        return self.database_path.parent / f"{self.deployment_id}-runtime-trial.json"

    @classmethod
    def read(cls, path: str | Path) -> RuntimeDeploymentProfile:
        profile_path = Path(path).expanduser()
        if (
            not profile_path.is_file()
            or profile_path.is_symlink()
            or profile_path.stat().st_size > 65_536
        ):
            raise DeploymentProfileError("deployment profile must be a small regular file")
        try:
            value = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise DeploymentProfileError("deployment profile is not valid UTF-8 JSON") from None
        if not isinstance(value, dict):
            raise DeploymentProfileError("deployment profile must be an object")
        fields = set(value)
        if not _PROFILE_REQUIRED_KEYS.issubset(fields) or not fields.issubset(
            _PROFILE_REQUIRED_KEYS | _PROFILE_OPTIONAL_KEYS
        ):
            raise DeploymentProfileError("deployment profile fields are invalid")
        if type(value["version"]) is not int or value["version"] != 1:
            raise DeploymentProfileError("unsupported deployment profile version")
        deployment_id = value["deployment_id"]
        if not isinstance(deployment_id, str) or _SAFE_ID.fullmatch(deployment_id) is None:
            raise DeploymentProfileError("deployment_id is invalid")
        if type(value["buffer_run_seconds"]) is not int or not (
            300 <= value["buffer_run_seconds"] <= 86_400
        ):
            raise DeploymentProfileError("buffer_run_seconds must be between 300 and 86400")
        if type(value["confirm_ephemeral_buffer_eviction"]) is not bool:
            raise DeploymentProfileError("confirm_ephemeral_buffer_eviction must be a boolean")
        candidate_detector = value.get("candidate_detector", "scene")
        if candidate_detector not in {"scene", "yolo_command"}:
            raise DeploymentProfileError("candidate_detector must be scene or yolo_command")
        trial_run_seconds = value.get("trial_run_seconds")
        if trial_run_seconds is not None and (
            type(trial_run_seconds) is not int or not 3_600 <= trial_run_seconds <= 604_800
        ):
            raise DeploymentProfileError(
                "trial_run_seconds must be null or between 3600 and 604800"
            )
        if not isinstance(value["cameras"], list) or not 1 <= len(value["cameras"]) <= 16:
            raise DeploymentProfileError("deployment must contain 1 to 16 cameras")
        cameras = tuple(DeploymentCamera.from_dict(item) for item in value["cameras"])
        if len({camera.camera_id for camera in cameras}) != len(cameras):
            raise DeploymentProfileError("deployment camera_id values must be unique")
        profile = cls(
            version=value["version"],
            deployment_id=deployment_id,
            project_root=_absolute_path(value["project_root"], field="project_root"),
            config_path=_absolute_path(value["config_path"], field="config_path"),
            database_path=_absolute_path(value["database_path"], field="database_path"),
            output_root=_absolute_path(value["output_root"], field="output_root"),
            buffer_run_seconds=value["buffer_run_seconds"],
            confirm_ephemeral_buffer_eviction=value["confirm_ephemeral_buffer_eviction"],
            cameras=cameras,
            candidate_detector=candidate_detector,
            pose_presence_shadow=(
                PosePresenceShadowDeployment.from_dict(value["pose_presence_shadow"])
                if value.get("pose_presence_shadow") is not None
                else None
            ),
            trial_run_seconds=trial_run_seconds,
            deferred_reports=value.get("deferred_reports", False),
        )
        profile.validate_local_paths()
        return profile

    def validate_local_paths(self) -> None:
        if not self.project_root.is_dir() or self.project_root.is_symlink():
            raise DeploymentProfileError("project_root must be an existing regular directory")
        try:
            resolved_python = self.python_executable.resolve(strict=True)
        except (OSError, RuntimeError):
            raise DeploymentProfileError("project venv Python executable is unavailable") from None
        if not resolved_python.is_file() or not os.access(resolved_python, os.X_OK):
            raise DeploymentProfileError("project venv Python executable is unavailable")
        if not self.config_path.is_file() or self.config_path.is_symlink():
            raise DeploymentProfileError("config_path must be an existing regular file")
        if not self.database_path.parent.is_dir():
            raise DeploymentProfileError("database parent directory must already exist")
        if self.database_path.exists() and (
            not self.database_path.is_file() or self.database_path.is_symlink()
        ):
            raise DeploymentProfileError("database_path must be a regular file when present")
        for path, field in ((self.output_root, "output_root"),):
            if not path.is_dir() or path.is_symlink():
                raise DeploymentProfileError(f"{field} must be an existing regular directory")
        for camera in self.cameras:
            if not camera.buffer_root.is_dir() or camera.buffer_root.is_symlink():
                raise DeploymentProfileError(
                    "camera buffer_root must be an existing regular directory"
                )


@dataclass(frozen=True, slots=True)
class RenderedDeployment:
    deployment_id: str
    unit_files: tuple[str, ...]
    manifest_file: str
    camera_count: int
    candidate_detector: str
    pose_presence_shadow: bool
    trial_run_seconds: int | None
    confirm_ephemeral_buffer_eviction: bool
    deferred_reports: bool = False

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": "rendered",
            "deployment_id": self.deployment_id,
            "camera_count": self.camera_count,
            "candidate_detector": self.candidate_detector,
            "pose_presence_shadow": self.pose_presence_shadow,
            "trial_run_seconds": self.trial_run_seconds,
            "bounded_trial": self.trial_run_seconds is not None,
            "unit_count": len(self.unit_files),
            "unit_files": list(self.unit_files),
            "manifest_file": self.manifest_file,
            "confirm_ephemeral_buffer_eviction": (self.confirm_ephemeral_buffer_eviction),
            "continuous_buffer_ready": self.confirm_ephemeral_buffer_eviction,
            **({"deferred_reports": True} if self.deferred_reports else {}),
            "requires_user_eviction_confirmation": (not self.confirm_ephemeral_buffer_eviction),
            "installed": False,
            "enabled": False,
            "started": False,
            "contains_credentials": False,
            "allows_remote_network": False,
        }


def _quote(value: str | Path) -> str:
    text = str(value)
    if not text or any(ord(character) < 32 for character in text):
        raise DeploymentProfileError("systemd argument is invalid")
    escaped = text.replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _exec_start(arguments: list[str | Path]) -> str:
    return "ExecStart=" + " ".join(_quote(argument) for argument in arguments)


def _directive_path(value: Path) -> str:
    safe = b"/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    parts: list[str] = []
    for byte in str(value).encode("utf-8"):
        if byte == ord("%"):
            parts.append("%%")
        elif byte in safe:
            parts.append(chr(byte))
        else:
            parts.append(f"\\x{byte:02x}")
    return "".join(parts)


def _common_service(
    profile: RuntimeDeploymentProfile,
    *,
    writable_paths: tuple[Path, ...],
    memory_max: str,
    allow_loopback: bool,
) -> list[str]:
    lines = [
        f"WorkingDirectory={_directive_path(profile.project_root)}",
        "UMask=0077",
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "ProtectKernelTunables=yes",
        "ProtectKernelModules=yes",
        "ProtectControlGroups=yes",
        "RestrictSUIDSGID=yes",
        "LockPersonality=yes",
        "RestrictRealtime=yes",
        "SystemCallArchitectures=native",
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=any",
        f"MemoryMax={memory_max}",
        "TasksMax=512",
        "TimeoutStopSec=30s",
        "KillMode=control-group",
        "Environment=PYTHONUNBUFFERED=1",
    ]
    if allow_loopback:
        lines.append("IPAddressAllow=localhost")
    for path in sorted({item.resolve() for item in writable_paths}, key=str):
        lines.append(f"ReadWritePaths={_directive_path(path)}")
    return lines


def _service_header(
    description: str,
    *,
    after: tuple[str, ...] = (),
    before: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
    part_of: tuple[str, ...] = (),
) -> list[str]:
    lines = [
        "[Unit]",
        f"Description={description}",
    ]
    if after:
        lines.append("After=" + " ".join(after))
    if before:
        lines.append("Before=" + " ".join(before))
    if requires:
        lines.append("Requires=" + " ".join(requires))
    if part_of:
        lines.append("PartOf=" + " ".join(part_of))
    lines.extend(
        [
            "StartLimitIntervalSec=300",
            "StartLimitBurst=10",
            "",
            "[Service]",
        ]
    )
    return lines


def _render_units(
    profile: RuntimeDeploymentProfile,
    *,
    settings: Settings,
    shadow_runtime: VisualPredicateShadowRuntimeSettings,
) -> dict[str, str]:
    prefix = f"baby-monitor-{profile.deployment_id}"
    python = profile.python_executable
    database_parent = profile.database_path.parent
    units: dict[str, str] = {}
    init_name = f"{prefix}-init.service"
    target_name = f"{prefix}.target"
    target_wants: list[str] = [init_name]
    candidate_names: list[str] = []

    init_lines = _service_header(
        "Baby monitor idempotent database initialization",
        part_of=(target_name,),
    )
    init_lines.extend(
        _common_service(
            profile,
            writable_paths=(database_parent,),
            memory_max="1G",
            allow_loopback=False,
        )
    )
    init_lines.extend(
        [
            "Type=oneshot",
            "RemainAfterExit=yes",
            _exec_start(
                [
                    python,
                    "-m",
                    "baby_monitor",
                    "init-db",
                    "--config",
                    profile.config_path,
                    "--database",
                    profile.database_path,
                ]
            ),
            "",
            "[Install]",
            f"WantedBy={prefix}.target",
            "",
        ]
    )
    units[init_name] = "\n".join(init_lines)

    for camera in profile.cameras:
        buffer_name = f"{prefix}-{camera.camera_id}-buffer.service"
        candidate_name = f"{prefix}-{camera.camera_id}-candidate.service"
        candidate_names.append(candidate_name)
        target_wants.extend((buffer_name, candidate_name))

        buffer_lines = _service_header(
            f"Baby monitor bounded RTSP buffer ({camera.camera_id})",
            part_of=(target_name,),
        )
        buffer_lines.extend(
            _common_service(
                profile,
                writable_paths=(camera.buffer_root,),
                memory_max="1G",
                allow_loopback=True,
            )
        )
        command: list[str | Path] = [
            python,
            "-m",
            "baby_monitor",
            "rtsp-buffer",
            "--config",
            profile.config_path,
            "--uri",
            camera.rtsp_uri,
            "--camera-id",
            camera.camera_id,
            "--seconds",
            str(profile.buffer_run_seconds),
            "--buffer-root",
            camera.buffer_root,
        ]
        if profile.confirm_ephemeral_buffer_eviction:
            command.append("--confirm-ephemeral-eviction")
        buffer_lines.extend(
            [
                _exec_start(command),
                "Restart=always",
                "RestartSec=5s",
                "Nice=10",
                "",
                "[Install]",
                f"WantedBy={prefix}.target",
                "",
            ]
        )
        units[buffer_name] = "\n".join(buffer_lines)

        candidate_lines = [
            "[Unit]",
            f"Description=Baby monitor candidate worker ({camera.camera_id})",
            f"After={init_name} {buffer_name}",
            f"Requires={init_name}",
            f"PartOf={target_name}",
            f"Wants={buffer_name}",
            "StartLimitIntervalSec=300",
            "StartLimitBurst=10",
            "",
            "[Service]",
        ]
        candidate_lines.extend(
            _common_service(
                profile,
                writable_paths=(
                    camera.buffer_root,
                    database_parent,
                    profile.output_root,
                    *(
                        (settings.detector.yolo_execution_lock_path.parent,)
                        if settings.detector.yolo_execution_lock_path is not None
                        else ()
                    ),
                ),
                memory_max=(
                    "8G"
                    if profile.pose_presence_shadow is not None
                    else "4G"
                    if profile.candidate_detector == "yolo_command"
                    else "2G"
                ),
                allow_loopback=False,
            )
        )
        candidate_command: list[str | Path] = [
            python,
            "-m",
            "baby_monitor",
            "candidate-worker",
            "--config",
            profile.config_path,
            "--manifest",
            camera.manifest_path,
            "--state",
            camera.state_path,
            "--presence",
            camera.presence_path,
            "--database",
            profile.database_path,
            "--output-root",
            profile.output_root,
            "--detector",
            profile.candidate_detector,
            "--image-size",
            str(camera.image_size),
            "--rotation-degrees",
            str(camera.rotation_degrees),
            "--idle-exit-seconds",
            "0",
            "--max-runtime-seconds",
            "3600",
        ]
        pose_shadow = profile.pose_presence_shadow
        if pose_shadow is not None:
            candidate_command.extend(
                (
                    "--pose-presence-shadow-profile",
                    pose_shadow.profile_path,
                    "--pose-presence-shadow-attestation",
                    pose_shadow.temporal_attestation_path,
                    "--pose-presence-shadow-audit",
                    pose_shadow.human_audit_report_path,
                    "--pose-presence-shadow-output",
                    camera.shadow_presence_path,
                    "--pose-shadow-detector-manifest",
                    pose_shadow.detector_manifest_path,
                    "--pose-shadow-detector-weights",
                    pose_shadow.detector_weights_path,
                    "--pose-shadow-detector-runner-executable",
                    pose_shadow.detector_runner_executable,
                    "--pose-shadow-detector-device",
                    pose_shadow.detector_device,
                    "--pose-shadow-manifest",
                    pose_shadow.pose_manifest_path,
                    "--pose-shadow-weights",
                    pose_shadow.pose_weights_path,
                    "--pose-shadow-runner-executable",
                    pose_shadow.pose_runner_executable,
                    "--pose-shadow-device",
                    pose_shadow.pose_device,
                    "--pose-shadow-timeout-seconds",
                    str(pose_shadow.pose_timeout_seconds),
                    "--pose-shadow-min-interval-seconds",
                    str(pose_shadow.sample_interval_seconds),
                )
            )
        candidate_lines.extend(
            [
                _exec_start(candidate_command),
                "Restart=always",
                "RestartSec=5s",
                "Nice=10",
                "",
                "[Install]",
                f"WantedBy={prefix}.target",
                "",
            ]
        )
        units[candidate_name] = "\n".join(candidate_lines)

    analysis_name = f"{prefix}-analysis.service"
    target_wants.append(analysis_name)
    analysis_lines = _service_header(
        "Baby monitor bounded sequential VLM worker",
        after=(init_name,),
        requires=(init_name,),
        part_of=(target_name,),
    )
    analysis_lines.extend(
        _common_service(
            profile,
            writable_paths=(database_parent, profile.output_root),
            memory_max="12G",
            allow_loopback=False,
        )
    )
    analysis_lines.extend(
        [
            # A shared user-runtime inode must survive PrivateTmp and service
            # restarts so CLI retries coordinate with the analysis worker.
            "RuntimeDirectory=baby-monitor-vlm-execution",
            "RuntimeDirectoryMode=0700",
            "RuntimeDirectoryPreserve=yes",
            _exec_start(
                [
                    python,
                    "-m",
                    "baby_monitor",
                    "analysis-worker",
                    "--config",
                    profile.config_path,
                    "--database",
                    profile.database_path,
                    "--idle-exit-seconds",
                    "0",
                    "--max-runtime-seconds",
                    "3600",
                    *(["--defer-reports"] if profile.deferred_reports else []),
                ]
            ),
            "Restart=always",
            "RestartSec=10s",
            "Nice=10",
            "",
            "[Install]",
            f"WantedBy={prefix}.target",
            "",
        ]
    )
    units[analysis_name] = "\n".join(analysis_lines)
    if profile.deferred_reports:
        refresh_name = f"{prefix}-report-refresh.service"
        target_wants.append(refresh_name)
        refresh_lines = _service_header(
            "Baby monitor bounded deferred daily report worker",
            after=(init_name,),
            requires=(init_name,),
            part_of=(target_name,),
        )
        refresh_lines.extend(
            _common_service(
                profile,
                writable_paths=(database_parent, profile.output_root),
                memory_max="1G",
                allow_loopback=False,
            )
        )
        refresh_lines.extend(
            [
                "PrivateDevices=yes",
                _exec_start(
                    [
                        python,
                        "-m",
                        "baby_monitor",
                        "report-refresh-worker",
                        "--config",
                        profile.config_path,
                        "--database",
                        profile.database_path,
                        "--max-jobs",
                        "0",
                        "--idle-exit-seconds",
                        "0",
                        "--max-runtime-seconds",
                        "3600",
                        "--poll-seconds",
                        "5",
                    ]
                ),
                "Restart=always",
                "RestartSec=10s",
                "Nice=15",
                "",
                "[Install]",
                f"WantedBy={prefix}.target",
                "",
            ]
        )
        units[refresh_name] = "\n".join(refresh_lines)
    if shadow_runtime.enabled:
        shadow_name = f"{prefix}-visual-predicate-shadow.service"
        target_wants.append(shadow_name)
        shadow_lines = _service_header(
            "Baby monitor isolated visual-predicate shadow worker",
            after=(init_name, analysis_name),
            requires=(init_name,),
            part_of=(target_name,),
        )
        shadow_lines.extend(
            _common_service(
                profile,
                writable_paths=(database_parent, profile.output_root),
                memory_max="12G",
                allow_loopback=False,
            )
        )
        shadow_lines.extend(
            [
                _exec_start(
                    [
                        python,
                        "-m",
                        "baby_monitor",
                        "visual-predicate-shadow-worker",
                        "--config",
                        profile.config_path,
                        "--database",
                        profile.database_path,
                        "--max-jobs",
                        str(shadow_runtime.max_jobs),
                        "--idle-exit-seconds",
                        "0",
                        "--max-runtime-seconds",
                        str(shadow_runtime.max_runtime_seconds),
                        "--poll-seconds",
                        "2",
                    ]
                ),
                "Restart=always",
                "RestartSec=10s",
                "Nice=12",
                "",
                "[Install]",
                f"WantedBy={prefix}.target",
                "",
            ]
        )
        units[shadow_name] = "\n".join(shadow_lines)

    notification_settings = settings.notification
    if notification_settings.auto_file_enabled:
        notification_name = f"{prefix}-notification.service"
        target_wants.append(notification_name)
        notification_lines = _service_header(
            "Baby monitor local file notification worker",
            after=(init_name,),
            requires=(init_name,),
            part_of=(target_name,),
        )
        notification_lines.extend(
            _common_service(
                profile,
                writable_paths=(database_parent, profile.output_root),
                memory_max="1G",
                allow_loopback=False,
            )
        )
        notification_lines.extend(
            [
                _exec_start(
                    [
                        python,
                        "-m",
                        "baby_monitor",
                        "notification-worker",
                        "--config",
                        profile.config_path,
                        "--database",
                        profile.database_path,
                        "--sink",
                        "file",
                        "--sink-id",
                        notification_settings.auto_file_sink_id,
                        "--output-dir",
                        profile.output_root,
                        "--max-jobs",
                        "0",
                        "--idle-exit-seconds",
                        "0",
                        "--max-runtime-seconds",
                        "3600",
                    ]
                ),
                "Restart=always",
                "RestartSec=10s",
                "Nice=10",
                "",
                "[Install]",
                f"WantedBy={prefix}.target",
                "",
            ]
        )
        units[notification_name] = "\n".join(notification_lines)

    report_name = f"{prefix}-daily-report.service"
    report_timer_name = f"{prefix}-daily-report.timer"
    report_lines = _service_header(
        "Baby monitor previous-day Markdown/HTML report",
        after=(init_name,),
        requires=(init_name,),
        part_of=(target_name,),
    )
    report_lines.extend(
        _common_service(
            profile,
            writable_paths=(database_parent, profile.output_root),
            memory_max="1G",
            allow_loopback=False,
        )
    )
    report_lines.extend(
        [
            "Type=oneshot",
            _exec_start(
                [
                    python,
                    "-m",
                    "baby_monitor",
                    "report",
                    "--config",
                    profile.config_path,
                    "--database",
                    profile.database_path,
                    "--output-root",
                    profile.output_root,
                    "--day-offset",
                    "-1",
                ]
            ),
            "",
        ]
    )
    units[report_name] = "\n".join(report_lines)
    units[report_timer_name] = "\n".join(
        [
            "[Unit]",
            "Description=Generate previous-day baby monitor report",
            f"PartOf={target_name}",
            "",
            "[Timer]",
            "OnCalendar=*-*-* 00:10:00",
            "Persistent=true",
            "RandomizedDelaySec=120",
            f"Unit={report_name}",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )

    health_name = f"{prefix}-health.service"
    health_timer_name = f"{prefix}-health.timer"
    health_lines = _service_header(
        "Baby monitor read-only runtime health snapshot",
        after=(init_name,),
        requires=(init_name,),
        part_of=(target_name,),
    )
    health_lines.extend(
        _common_service(
            profile,
            writable_paths=(),
            memory_max="1G",
            allow_loopback=False,
        )
    )
    health_command: list[str | Path] = [
        python,
        "-m",
        "baby_monitor",
        "runtime-status",
        "--config",
        profile.config_path,
    ]
    for camera in profile.cameras:
        health_command.extend(("--health", camera.health_path))
    health_command.extend(
        (
            "--database",
            profile.database_path,
            "--storage-root",
            profile.output_root,
        )
    )
    health_lines.extend(["Type=oneshot", _exec_start(health_command), ""])
    units[health_name] = "\n".join(health_lines)
    units[health_timer_name] = "\n".join(
        [
            "[Unit]",
            "Description=Check baby monitor runtime health every five minutes",
            f"PartOf={target_name}",
            "",
            "[Timer]",
            "OnBootSec=2min",
            "OnUnitActiveSec=5min",
            f"Unit={health_name}",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ]
    )

    additional_target_wants: list[str] = []
    additional_target_requires: list[str] = []
    if profile.trial_run_seconds is not None:
        baseline_name = f"{prefix}-trial-baseline.service"
        stop_name = f"{prefix}-trial-stop.service"
        stop_timer_name = f"{prefix}-trial-stop.timer"
        ledger_path = profile.runtime_trial_ledger_path
        baseline_lines = _service_header(
            "Capture the bounded baby monitor trial baseline",
            after=tuple(candidate_names),
            before=(target_name, stop_timer_name),
            requires=tuple(candidate_names),
            part_of=(target_name,),
        )
        baseline_lines.extend(
            _common_service(
                profile,
                writable_paths=(database_parent,),
                memory_max="256M",
                allow_loopback=False,
            )
        )
        baseline_command: list[str | Path] = [
            python,
            "-m",
            "baby_monitor",
            "runtime-trial-start",
            "--ledger",
            ledger_path,
            "--database",
            profile.database_path,
            "--expected-duration-seconds",
            str(profile.trial_run_seconds),
            "--health-wait-seconds",
            "30",
        ]
        for camera in profile.cameras:
            baseline_command.extend(("--health", camera.health_path))
        if profile.pose_presence_shadow is not None:
            baseline_command.append("--shadow-expected")
        baseline_lines.extend(
            [
                "Type=oneshot",
                "RemainAfterExit=yes",
                _exec_start(baseline_command),
                "",
            ]
        )
        units[baseline_name] = "\n".join(baseline_lines)

        stop_lines = _service_header(
            "Stop and settle the bounded baby monitor trial",
            after=(target_name,),
        )
        stop_lines.insert(2, f"Conflicts={target_name}")
        stop_lines.extend(
            _common_service(
                profile,
                writable_paths=(database_parent,),
                memory_max="256M",
                allow_loopback=False,
            )
        )
        finish_command: list[str | Path] = [
            python,
            "-m",
            "baby_monitor",
            "runtime-trial-finish",
            "--ledger",
            ledger_path,
            "--database",
            profile.database_path,
        ]
        for camera in profile.cameras:
            finish_command.extend(("--health", camera.health_path))
        stop_lines.extend(["Type=oneshot", _exec_start(finish_command), ""])
        units[stop_name] = "\n".join(stop_lines)
        units[stop_timer_name] = "\n".join(
            [
                "[Unit]",
                "Description=Stop the bounded baby monitor trial on schedule",
                f"After={baseline_name}",
                f"Requires={baseline_name}",
                f"PartOf={target_name}",
                "",
                "[Timer]",
                f"OnActiveSec={profile.trial_run_seconds}s",
                "AccuracySec=1s",
                f"Unit={stop_name}",
                "",
                "[Install]",
                f"WantedBy={target_name}",
                "",
            ]
        )
        additional_target_wants.append(stop_timer_name)
        additional_target_requires.append(baseline_name)

    target_lines = [
        "[Unit]",
        f"Description=Baby monitor runtime ({profile.deployment_id})",
        "Wants=" + " ".join(sorted(target_wants)),
        "Wants=" + " ".join((report_timer_name, health_timer_name, *additional_target_wants)),
    ]
    if additional_target_requires:
        target_lines.append("Requires=" + " ".join(additional_target_requires))
    target_lines.extend(
        [
            "After=network-online.target",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )
    units[target_name] = "\n".join(target_lines)
    return units


def render_systemd_user_units(
    profile: RuntimeDeploymentProfile,
    output_directory: str | Path,
) -> RenderedDeployment:
    settings = load_settings(profile.config_path)
    shadow_runtime = load_visual_predicate_shadow_runtime_settings(profile.config_path)
    if shadow_runtime.enabled and (
        settings.local_vlm.backend != "mock"
        or settings.secondary_vlm.enabled
        or settings.remote_vlm.enabled
    ):
        raise DeploymentProfileError(
            "visual predicate shadow deployment requires all other VLM backends disabled"
        )
    if profile.candidate_detector == "yolo_command" and settings.detector.backend != "yolo_command":
        raise DeploymentProfileError(
            "candidate_detector yolo_command requires detector.backend=yolo_command"
        )
    if profile.pose_presence_shadow is not None:
        if profile.candidate_detector != "yolo_command":
            raise DeploymentProfileError(
                "pose presence shadow requires candidate_detector=yolo_command"
            )
        if settings.detector.yolo_execution_lock_path is None:
            raise DeploymentProfileError(
                "pose presence shadow requires detector.yolo_execution_lock_path"
            )
        lock_parent = settings.detector.yolo_execution_lock_path.parent
        if (
            not lock_parent.is_dir()
            or lock_parent.is_symlink()
            or os.name == "posix"
            and (
                lock_parent.stat().st_uid != os.geteuid()
                or lock_parent.stat().st_mode & 0o777 != 0o700
            )
        ):
            raise DeploymentProfileError(
                "pose presence shadow execution lock parent must be a private local directory"
            )
    if profile.confirm_ephemeral_buffer_eviction and not settings.buffer.eviction_enabled:
        raise DeploymentProfileError(
            "persistent buffer.eviction_enabled must accompany one-shot confirmation"
        )
    output = Path(output_directory).expanduser()
    if not output.is_dir() or output.is_symlink():
        raise DeploymentProfileError("output directory must already exist and not be a symlink")
    units = _render_units(profile, settings=settings, shadow_runtime=shadow_runtime)
    manifest_name = f"baby-monitor-{profile.deployment_id}-units.json"
    names = (*sorted(units), manifest_name)
    if any((output / name).exists() for name in names):
        raise FileExistsError("deployment renderer never overwrites an existing file")

    manifest = {
        "version": 1,
        "deployment_id": profile.deployment_id,
        "unit_files": [
            {
                "name": name,
                "sha256": hashlib.sha256(units[name].encode("utf-8")).hexdigest(),
            }
            for name in sorted(units)
        ],
        "installed": False,
        "enabled": False,
        "started": False,
        "confirm_ephemeral_buffer_eviction": (profile.confirm_ephemeral_buffer_eviction),
        "trial_run_seconds": profile.trial_run_seconds,
        "pose_presence_shadow": profile.pose_presence_shadow is not None,
        **({"deferred_reports": True} if profile.deferred_reports else {}),
    }
    payloads = {**units, manifest_name: json.dumps(manifest, indent=2) + "\n"}
    created: list[Path] = []
    try:
        for name in names:
            destination = output / name
            with destination.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payloads[name])
            destination.chmod(0o600)
            created.append(destination)
    except Exception:
        for path in reversed(created):
            with suppress(FileNotFoundError):
                path.unlink()
        raise
    return RenderedDeployment(
        deployment_id=profile.deployment_id,
        unit_files=tuple(sorted(units)),
        manifest_file=manifest_name,
        camera_count=len(profile.cameras),
        candidate_detector=profile.candidate_detector,
        pose_presence_shadow=profile.pose_presence_shadow is not None,
        trial_run_seconds=profile.trial_run_seconds,
        confirm_ephemeral_buffer_eviction=(profile.confirm_ephemeral_buffer_eviction),
        deferred_reports=profile.deferred_reports,
    )
