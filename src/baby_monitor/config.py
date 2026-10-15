from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity_service import IdentitySettings


class ConfigurationError(ValueError):
    pass


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ConfigurationError(f"{name} must be true or false")


def _as_int(value: Any, name: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}")
    return parsed


def _as_float(value: Any, name: str, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if parsed < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}")
    return parsed


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"[{name}] must be a TOML table")
    return value


@dataclass(frozen=True, slots=True)
class StorageSettings:
    local_root: Path = Path("./data")
    nas_root: Path | None = None
    prefer_nas: bool = False
    database: Path = Path("./data/events.sqlite3")
    retention_days: int = 14
    retention_scan_enabled: bool = False
    retention_delete_enabled: bool = False


@dataclass(frozen=True, slots=True)
class CameraSettings:
    camera_id: str = "nursery_cam_01"
    kind: str = "file"
    source: str = ""


@dataclass(frozen=True, slots=True)
class DetectorSettings:
    backend: str = "mock"
    yolo_model_path: Path | None = None
    yolo_manifest_path: Path | None = None
    yolo_runner_command: tuple[str, ...] = ()
    yolo_runner_timeout_seconds: float = 600.0
    yolo_execution_lock_path: Path | None = None
    yolo_execution_lock_timeout_seconds: float = 600.0
    device: str = "0"
    confidence: float = 0.35
    sample_fps: float = 1.0
    image_size: int = 640
    rotation_degrees: int = 0
    candidate_gap_seconds: float = 3.0
    scene_threshold: float = 0.15
    scene_max_detections: int = 200
    labels: tuple[str, ...] = ("person", "cat", "dog")


@dataclass(frozen=True, slots=True)
class MediaSettings:
    pre_event_seconds: float = 5.0
    post_event_seconds: float = 7.0
    keyframe_count: int = 5
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


@dataclass(frozen=True, slots=True)
class BufferSettings:
    root: Path = Path("./data/buffer")
    segment_seconds: float = 2.0
    max_segments: int = 30
    reconnect_initial_seconds: float = 0.5
    reconnect_max_seconds: float = 8.0
    poll_interval_seconds: float = 0.25
    eviction_enabled: bool = False


@dataclass(frozen=True, slots=True)
class MediaHealthSettings:
    stale_after_seconds: float = 10.0
    future_tolerance_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class CandidateWorkerRuntimeSettings:
    max_segments_per_run: int = 32
    idle_exit_seconds: float = 30.0
    max_runtime_seconds: float = 3_600.0
    poll_seconds: float = 1.0
    max_event_span_seconds: float = 30.0
    candidate_start_confidence: float = 0.35
    max_pending_detections: int = 1_000
    detector_timeout_seconds: float = 120.0


@dataclass(frozen=True, slots=True)
class AnalysisQueueSettings:
    max_pending: int = 32
    lease_seconds: int = 300
    retry_delay_seconds: int = 30
    max_attempts: int = 3


@dataclass(frozen=True, slots=True)
class AnalysisWorkerSettings:
    max_jobs_per_run: int = 32
    idle_exit_seconds: float = 30.0
    max_runtime_seconds: float = 3600.0
    poll_seconds: float = 2.0
    gpu_guard_enabled: bool = False
    gpu_device: str = "0"
    min_free_vram_mib: int = 6_000
    max_gpu_utilization_percent: int = 20
    gpu_probe_timeout_seconds: int = 5


@dataclass(frozen=True, slots=True)
class ArchiveQueueSettings:
    max_pending: int = 64
    lease_seconds: int = 3_600
    retry_delay_seconds: int = 300
    max_attempts: int = 20


@dataclass(frozen=True, slots=True)
class ArchiveWorkerSettings:
    max_jobs_per_run: int = 32
    idle_exit_seconds: float = 30.0
    max_runtime_seconds: float = 3_600.0
    poll_seconds: float = 5.0


@dataclass(frozen=True, slots=True)
class LocalVLMSettings:
    backend: str = "mock"
    name: str = "mock"
    revision: str = ""
    command: tuple[str, ...] = ()
    timeout_seconds: int = 180
    confidence_threshold: float = 0.65
    gpu_device: str = "0"


@dataclass(frozen=True, slots=True)
class ReviewPolicySettings:
    signal_frequency_window_seconds: int = 600
    signal_frequency_secondary_threshold: int = 3
    priority_signal_types: tuple[str, ...] = ("fall_hint",)


@dataclass(frozen=True, slots=True)
class SecondaryVLMSettings:
    enabled: bool = False
    backend: str = "mock"
    name: str = "secondary_mock"
    revision: str = ""
    command: tuple[str, ...] = ()
    timeout_seconds: int = 180
    gpu_device: str = "0"


@dataclass(frozen=True, slots=True)
class RemoteVLMSettings:
    enabled: bool = False
    upload_consent: bool = False
    base_url: str = ""
    model: str = ""
    api_key_env: str = "BABY_MONITOR_REMOTE_API_KEY"
    max_frames: int = 3
    timeout_seconds: int = 60
    max_upload_bytes: int = 5_242_880
    max_response_bytes: int = 262_144
    max_requests_per_day: int = 20
    min_interval_seconds: int = 60

    @property
    def uploads_allowed(self) -> bool:
        return self.enabled and self.upload_consent


@dataclass(frozen=True, slots=True)
class NotificationSettings:
    enabled: bool = False
    external_delivery_enabled: bool = False
    auto_file_enabled: bool = False
    auto_file_sink_id: str = "shadow_local"
    event_policy_path: Path | None = None
    max_pending: int = 64
    max_pending_age_seconds: int = 3600
    lease_seconds: int = 300
    retry_delay_seconds: int = 30
    max_attempts: int = 5


@dataclass(frozen=True, slots=True)
class Settings:
    timezone: str = "Asia/Shanghai"
    storage: StorageSettings = StorageSettings()
    camera: CameraSettings = CameraSettings()
    detector: DetectorSettings = DetectorSettings()
    media: MediaSettings = MediaSettings()
    buffer: BufferSettings = BufferSettings()
    media_health: MediaHealthSettings = MediaHealthSettings()
    candidate_worker: CandidateWorkerRuntimeSettings = CandidateWorkerRuntimeSettings()
    analysis_queue: AnalysisQueueSettings = AnalysisQueueSettings()
    analysis_worker: AnalysisWorkerSettings = AnalysisWorkerSettings()
    archive_queue: ArchiveQueueSettings = ArchiveQueueSettings()
    archive_worker: ArchiveWorkerSettings = ArchiveWorkerSettings()
    local_vlm: LocalVLMSettings = LocalVLMSettings()
    review_policy: ReviewPolicySettings = ReviewPolicySettings()
    secondary_vlm: SecondaryVLMSettings = SecondaryVLMSettings()
    remote_vlm: RemoteVLMSettings = RemoteVLMSettings()
    identity: IdentitySettings = IdentitySettings()
    notification: NotificationSettings = NotificationSettings()


def _env(name: str, fallback: Any) -> Any:
    return os.environ.get(name, fallback)


def _optional_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser()


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else Path("config/default.toml")
    if config_path.exists():
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    else:
        data = {}

    general = _section(data, "general")
    storage = _section(data, "storage")
    camera = _section(data, "camera")
    detector = _section(data, "detector")
    media = _section(data, "media")
    buffer = _section(data, "buffer")
    media_health = _section(data, "media_health")
    candidate_worker = _section(data, "candidate_worker")
    analysis_queue = _section(data, "analysis_queue")
    analysis_worker = _section(data, "analysis_worker")
    archive_queue = _section(data, "archive_queue")
    archive_worker = _section(data, "archive_worker")
    local_vlm = _section(data, "local_vlm")
    review_policy = _section(data, "review_policy")
    secondary_vlm = _section(data, "secondary_vlm")
    remote_vlm = _section(data, "remote_vlm")
    identity = _section(data, "identity")
    notification = _section(data, "notification")

    labels_value = detector.get("labels", ["person", "cat", "dog"])
    if not isinstance(labels_value, list) or not all(
        isinstance(item, str) and item for item in labels_value
    ):
        raise ConfigurationError("detector.labels must be an array of non-empty strings")

    local_command_value = local_vlm.get("command", [])
    if not isinstance(local_command_value, list) or not all(
        isinstance(item, str) and item for item in local_command_value
    ):
        raise ConfigurationError("local_vlm.command must be an array of non-empty strings")
    local_command = tuple(local_command_value)
    if local_command:
        local_command = (
            str(Path(local_command[0]).expanduser()),
            *local_command[1:],
        )

    secondary_command_value = secondary_vlm.get("command", [])
    if not isinstance(secondary_command_value, list) or not all(
        isinstance(item, str) and item for item in secondary_command_value
    ):
        raise ConfigurationError("secondary_vlm.command must be an array of non-empty strings")
    secondary_command = tuple(secondary_command_value)
    if secondary_command:
        secondary_command = (
            str(Path(secondary_command[0]).expanduser()),
            *secondary_command[1:],
        )

    yolo_runner_command_value = detector.get("yolo_runner_command", [])
    if not isinstance(yolo_runner_command_value, list) or not all(
        isinstance(item, str) and item for item in yolo_runner_command_value
    ):
        raise ConfigurationError(
            "detector.yolo_runner_command must be an array of non-empty strings"
        )
    yolo_runner_command = tuple(yolo_runner_command_value)
    if yolo_runner_command:
        yolo_runner_command = (
            str(Path(yolo_runner_command[0]).expanduser()),
            *yolo_runner_command[1:],
        )

    priority_signal_types = review_policy.get("priority_signal_types", ["fall_hint"])
    if not isinstance(priority_signal_types, list) or not all(
        isinstance(item, str) and item for item in priority_signal_types
    ):
        raise ConfigurationError(
            "review_policy.priority_signal_types must be an array of non-empty strings"
        )

    settings = Settings(
        timezone=str(general.get("timezone", "Asia/Shanghai")),
        storage=StorageSettings(
            local_root=Path(
                _env("BABY_MONITOR_LOCAL_ROOT", storage.get("local_root", "./data"))
            ).expanduser(),
            nas_root=_optional_path(_env("BABY_MONITOR_NAS_ROOT", storage.get("nas_root", ""))),
            prefer_nas=_as_bool(
                _env("BABY_MONITOR_PREFER_NAS", storage.get("prefer_nas", False)),
                "BABY_MONITOR_PREFER_NAS",
            ),
            database=Path(
                _env("BABY_MONITOR_DATABASE", storage.get("database", "./data/events.sqlite3"))
            ).expanduser(),
            retention_days=_as_int(
                _env("BABY_MONITOR_RETENTION_DAYS", storage.get("retention_days", 14)),
                "BABY_MONITOR_RETENTION_DAYS",
                minimum=1,
            ),
            retention_scan_enabled=_as_bool(
                _env(
                    "BABY_MONITOR_RETENTION_SCAN_ENABLED",
                    storage.get("retention_scan_enabled", False),
                ),
                "BABY_MONITOR_RETENTION_SCAN_ENABLED",
            ),
            retention_delete_enabled=_as_bool(
                _env(
                    "BABY_MONITOR_RETENTION_DELETE_ENABLED",
                    storage.get("retention_delete_enabled", False),
                ),
                "BABY_MONITOR_RETENTION_DELETE_ENABLED",
            ),
        ),
        camera=CameraSettings(
            camera_id=str(_env("BABY_MONITOR_CAMERA_ID", camera.get("id", "nursery_cam_01"))),
            kind=str(_env("BABY_MONITOR_CAMERA_KIND", camera.get("kind", "file"))),
            source=str(_env("BABY_MONITOR_CAMERA_SOURCE", camera.get("source", ""))),
        ),
        detector=DetectorSettings(
            backend=str(_env("BABY_MONITOR_DETECTOR_BACKEND", detector.get("backend", "mock"))),
            yolo_model_path=_optional_path(
                _env(
                    "BABY_MONITOR_YOLO_MODEL_PATH",
                    detector.get("yolo_model_path", ""),
                )
            ),
            yolo_manifest_path=_optional_path(
                _env(
                    "BABY_MONITOR_YOLO_MANIFEST_PATH",
                    detector.get("yolo_manifest_path", ""),
                )
            ),
            yolo_runner_command=yolo_runner_command,
            yolo_runner_timeout_seconds=_as_float(
                _env(
                    "BABY_MONITOR_YOLO_RUNNER_TIMEOUT_SECONDS",
                    detector.get("yolo_runner_timeout_seconds", 600.0),
                ),
                "detector.yolo_runner_timeout_seconds",
                minimum=1.0,
            ),
            yolo_execution_lock_path=_optional_path(
                _env(
                    "BABY_MONITOR_YOLO_EXECUTION_LOCK_PATH",
                    detector.get("yolo_execution_lock_path", ""),
                )
            ),
            yolo_execution_lock_timeout_seconds=_as_float(
                _env(
                    "BABY_MONITOR_YOLO_EXECUTION_LOCK_TIMEOUT_SECONDS",
                    detector.get("yolo_execution_lock_timeout_seconds", 600.0),
                ),
                "detector.yolo_execution_lock_timeout_seconds",
                minimum=1.0,
            ),
            device=str(_env("BABY_MONITOR_YOLO_DEVICE", detector.get("device", "0"))),
            confidence=_as_float(
                _env("BABY_MONITOR_YOLO_CONFIDENCE", detector.get("confidence", 0.35)),
                "BABY_MONITOR_YOLO_CONFIDENCE",
            ),
            sample_fps=_as_float(
                _env("BABY_MONITOR_SAMPLE_FPS", detector.get("sample_fps", 1.0)),
                "BABY_MONITOR_SAMPLE_FPS",
                minimum=0.01,
            ),
            image_size=_as_int(
                _env(
                    "BABY_MONITOR_YOLO_IMAGE_SIZE",
                    detector.get("image_size", 640),
                ),
                "BABY_MONITOR_YOLO_IMAGE_SIZE",
                minimum=0,
            ),
            rotation_degrees=_as_int(
                _env(
                    "BABY_MONITOR_ROTATION_DEGREES",
                    detector.get("rotation_degrees", 0),
                ),
                "BABY_MONITOR_ROTATION_DEGREES",
                minimum=0,
            ),
            candidate_gap_seconds=_as_float(
                detector.get("candidate_gap_seconds", 3.0),
                "detector.candidate_gap_seconds",
            ),
            scene_threshold=_as_float(
                _env(
                    "BABY_MONITOR_SCENE_THRESHOLD",
                    detector.get("scene_threshold", 0.15),
                ),
                "BABY_MONITOR_SCENE_THRESHOLD",
            ),
            scene_max_detections=_as_int(
                _env(
                    "BABY_MONITOR_SCENE_MAX_DETECTIONS",
                    detector.get("scene_max_detections", 200),
                ),
                "BABY_MONITOR_SCENE_MAX_DETECTIONS",
                minimum=1,
            ),
            labels=tuple(labels_value),
        ),
        media=MediaSettings(
            pre_event_seconds=_as_float(
                _env(
                    "BABY_MONITOR_PRE_EVENT_SECONDS",
                    media.get("pre_event_seconds", 5.0),
                ),
                "BABY_MONITOR_PRE_EVENT_SECONDS",
            ),
            post_event_seconds=_as_float(
                _env(
                    "BABY_MONITOR_POST_EVENT_SECONDS",
                    media.get("post_event_seconds", 7.0),
                ),
                "BABY_MONITOR_POST_EVENT_SECONDS",
            ),
            keyframe_count=_as_int(
                _env("BABY_MONITOR_KEYFRAME_COUNT", media.get("keyframe_count", 5)),
                "BABY_MONITOR_KEYFRAME_COUNT",
                minimum=1,
            ),
            ffmpeg=str(media.get("ffmpeg", "ffmpeg")),
            ffprobe=str(media.get("ffprobe", "ffprobe")),
        ),
        buffer=BufferSettings(
            root=Path(
                _env("BABY_MONITOR_BUFFER_ROOT", buffer.get("root", "./data/buffer"))
            ).expanduser(),
            segment_seconds=_as_float(
                _env(
                    "BABY_MONITOR_BUFFER_SEGMENT_SECONDS",
                    buffer.get("segment_seconds", 2.0),
                ),
                "BABY_MONITOR_BUFFER_SEGMENT_SECONDS",
                minimum=1.0,
            ),
            max_segments=_as_int(
                _env(
                    "BABY_MONITOR_BUFFER_MAX_SEGMENTS",
                    buffer.get("max_segments", 30),
                ),
                "BABY_MONITOR_BUFFER_MAX_SEGMENTS",
                minimum=3,
            ),
            reconnect_initial_seconds=_as_float(
                buffer.get("reconnect_initial_seconds", 0.5),
                "buffer.reconnect_initial_seconds",
                minimum=0.1,
            ),
            reconnect_max_seconds=_as_float(
                buffer.get("reconnect_max_seconds", 8.0),
                "buffer.reconnect_max_seconds",
                minimum=0.1,
            ),
            poll_interval_seconds=_as_float(
                buffer.get("poll_interval_seconds", 0.25),
                "buffer.poll_interval_seconds",
                minimum=0.05,
            ),
            eviction_enabled=_as_bool(
                _env(
                    "BABY_MONITOR_BUFFER_EVICTION_ENABLED",
                    buffer.get("eviction_enabled", False),
                ),
                "BABY_MONITOR_BUFFER_EVICTION_ENABLED",
            ),
        ),
        media_health=MediaHealthSettings(
            stale_after_seconds=_as_float(
                _env(
                    "BABY_MONITOR_MEDIA_HEALTH_STALE_SECONDS",
                    media_health.get("stale_after_seconds", 10.0),
                ),
                "BABY_MONITOR_MEDIA_HEALTH_STALE_SECONDS",
                minimum=0.001,
            ),
            future_tolerance_seconds=_as_float(
                _env(
                    "BABY_MONITOR_MEDIA_HEALTH_FUTURE_TOLERANCE_SECONDS",
                    media_health.get("future_tolerance_seconds", 5.0),
                ),
                "BABY_MONITOR_MEDIA_HEALTH_FUTURE_TOLERANCE_SECONDS",
                minimum=0.0,
            ),
        ),
        candidate_worker=CandidateWorkerRuntimeSettings(
            max_segments_per_run=_as_int(
                _env(
                    "BABY_MONITOR_CANDIDATE_MAX_SEGMENTS_PER_RUN",
                    candidate_worker.get("max_segments_per_run", 32),
                ),
                "BABY_MONITOR_CANDIDATE_MAX_SEGMENTS_PER_RUN",
                minimum=1,
            ),
            idle_exit_seconds=_as_float(
                _env(
                    "BABY_MONITOR_CANDIDATE_IDLE_EXIT_SECONDS",
                    candidate_worker.get("idle_exit_seconds", 30.0),
                ),
                "BABY_MONITOR_CANDIDATE_IDLE_EXIT_SECONDS",
                minimum=0.0,
            ),
            max_runtime_seconds=_as_float(
                _env(
                    "BABY_MONITOR_CANDIDATE_MAX_RUNTIME_SECONDS",
                    candidate_worker.get("max_runtime_seconds", 3_600.0),
                ),
                "BABY_MONITOR_CANDIDATE_MAX_RUNTIME_SECONDS",
                minimum=1.0,
            ),
            poll_seconds=_as_float(
                _env(
                    "BABY_MONITOR_CANDIDATE_POLL_SECONDS",
                    candidate_worker.get("poll_seconds", 1.0),
                ),
                "BABY_MONITOR_CANDIDATE_POLL_SECONDS",
                minimum=0.05,
            ),
            max_event_span_seconds=_as_float(
                _env(
                    "BABY_MONITOR_CANDIDATE_MAX_EVENT_SPAN_SECONDS",
                    candidate_worker.get("max_event_span_seconds", 30.0),
                ),
                "BABY_MONITOR_CANDIDATE_MAX_EVENT_SPAN_SECONDS",
                minimum=1.0,
            ),
            candidate_start_confidence=_as_float(
                _env(
                    "BABY_MONITOR_CANDIDATE_START_CONFIDENCE",
                    candidate_worker.get("candidate_start_confidence", 0.35),
                ),
                "BABY_MONITOR_CANDIDATE_START_CONFIDENCE",
                minimum=0.0,
            ),
            max_pending_detections=_as_int(
                _env(
                    "BABY_MONITOR_CANDIDATE_MAX_PENDING_DETECTIONS",
                    candidate_worker.get("max_pending_detections", 1_000),
                ),
                "BABY_MONITOR_CANDIDATE_MAX_PENDING_DETECTIONS",
                minimum=1,
            ),
            detector_timeout_seconds=_as_float(
                _env(
                    "BABY_MONITOR_CANDIDATE_DETECTOR_TIMEOUT_SECONDS",
                    candidate_worker.get("detector_timeout_seconds", 120.0),
                ),
                "BABY_MONITOR_CANDIDATE_DETECTOR_TIMEOUT_SECONDS",
                minimum=1.0,
            ),
        ),
        analysis_queue=AnalysisQueueSettings(
            max_pending=_as_int(
                _env(
                    "BABY_MONITOR_ANALYSIS_QUEUE_MAX_PENDING",
                    analysis_queue.get("max_pending", 32),
                ),
                "BABY_MONITOR_ANALYSIS_QUEUE_MAX_PENDING",
                minimum=1,
            ),
            lease_seconds=_as_int(
                _env(
                    "BABY_MONITOR_ANALYSIS_QUEUE_LEASE_SECONDS",
                    analysis_queue.get("lease_seconds", 300),
                ),
                "BABY_MONITOR_ANALYSIS_QUEUE_LEASE_SECONDS",
                minimum=10,
            ),
            retry_delay_seconds=_as_int(
                _env(
                    "BABY_MONITOR_ANALYSIS_QUEUE_RETRY_DELAY_SECONDS",
                    analysis_queue.get("retry_delay_seconds", 30),
                ),
                "BABY_MONITOR_ANALYSIS_QUEUE_RETRY_DELAY_SECONDS",
                minimum=0,
            ),
            max_attempts=_as_int(
                _env(
                    "BABY_MONITOR_ANALYSIS_QUEUE_MAX_ATTEMPTS",
                    analysis_queue.get("max_attempts", 3),
                ),
                "BABY_MONITOR_ANALYSIS_QUEUE_MAX_ATTEMPTS",
                minimum=1,
            ),
        ),
        analysis_worker=AnalysisWorkerSettings(
            max_jobs_per_run=_as_int(
                _env(
                    "BABY_MONITOR_ANALYSIS_MAX_JOBS_PER_RUN",
                    analysis_worker.get("max_jobs_per_run", 32),
                ),
                "BABY_MONITOR_ANALYSIS_MAX_JOBS_PER_RUN",
                minimum=0,
            ),
            idle_exit_seconds=_as_float(
                _env(
                    "BABY_MONITOR_ANALYSIS_IDLE_EXIT_SECONDS",
                    analysis_worker.get("idle_exit_seconds", 30.0),
                ),
                "BABY_MONITOR_ANALYSIS_IDLE_EXIT_SECONDS",
            ),
            max_runtime_seconds=_as_float(
                _env(
                    "BABY_MONITOR_ANALYSIS_MAX_RUNTIME_SECONDS",
                    analysis_worker.get("max_runtime_seconds", 3600.0),
                ),
                "BABY_MONITOR_ANALYSIS_MAX_RUNTIME_SECONDS",
            ),
            poll_seconds=_as_float(
                _env(
                    "BABY_MONITOR_ANALYSIS_POLL_SECONDS",
                    analysis_worker.get("poll_seconds", 2.0),
                ),
                "BABY_MONITOR_ANALYSIS_POLL_SECONDS",
                minimum=0.01,
            ),
            gpu_guard_enabled=_as_bool(
                _env(
                    "BABY_MONITOR_ANALYSIS_GPU_GUARD_ENABLED",
                    analysis_worker.get("gpu_guard_enabled", False),
                ),
                "BABY_MONITOR_ANALYSIS_GPU_GUARD_ENABLED",
            ),
            gpu_device=str(
                _env(
                    "BABY_MONITOR_ANALYSIS_GPU_DEVICE",
                    analysis_worker.get("gpu_device", "0"),
                )
            ).strip(),
            min_free_vram_mib=_as_int(
                _env(
                    "BABY_MONITOR_ANALYSIS_MIN_FREE_VRAM_MIB",
                    analysis_worker.get("min_free_vram_mib", 6_000),
                ),
                "BABY_MONITOR_ANALYSIS_MIN_FREE_VRAM_MIB",
                minimum=1,
            ),
            max_gpu_utilization_percent=_as_int(
                _env(
                    "BABY_MONITOR_ANALYSIS_MAX_GPU_UTILIZATION_PERCENT",
                    analysis_worker.get("max_gpu_utilization_percent", 20),
                ),
                "BABY_MONITOR_ANALYSIS_MAX_GPU_UTILIZATION_PERCENT",
                minimum=0,
            ),
            gpu_probe_timeout_seconds=_as_int(
                analysis_worker.get("gpu_probe_timeout_seconds", 5),
                "analysis_worker.gpu_probe_timeout_seconds",
                minimum=1,
            ),
        ),
        archive_queue=ArchiveQueueSettings(
            max_pending=_as_int(
                _env(
                    "BABY_MONITOR_ARCHIVE_QUEUE_MAX_PENDING",
                    archive_queue.get("max_pending", 64),
                ),
                "BABY_MONITOR_ARCHIVE_QUEUE_MAX_PENDING",
                minimum=1,
            ),
            lease_seconds=_as_int(
                _env(
                    "BABY_MONITOR_ARCHIVE_QUEUE_LEASE_SECONDS",
                    archive_queue.get("lease_seconds", 3_600),
                ),
                "BABY_MONITOR_ARCHIVE_QUEUE_LEASE_SECONDS",
                minimum=10,
            ),
            retry_delay_seconds=_as_int(
                _env(
                    "BABY_MONITOR_ARCHIVE_QUEUE_RETRY_DELAY_SECONDS",
                    archive_queue.get("retry_delay_seconds", 300),
                ),
                "BABY_MONITOR_ARCHIVE_QUEUE_RETRY_DELAY_SECONDS",
                minimum=0,
            ),
            max_attempts=_as_int(
                _env(
                    "BABY_MONITOR_ARCHIVE_QUEUE_MAX_ATTEMPTS",
                    archive_queue.get("max_attempts", 20),
                ),
                "BABY_MONITOR_ARCHIVE_QUEUE_MAX_ATTEMPTS",
                minimum=1,
            ),
        ),
        archive_worker=ArchiveWorkerSettings(
            max_jobs_per_run=_as_int(
                _env(
                    "BABY_MONITOR_ARCHIVE_MAX_JOBS_PER_RUN",
                    archive_worker.get("max_jobs_per_run", 32),
                ),
                "BABY_MONITOR_ARCHIVE_MAX_JOBS_PER_RUN",
                minimum=0,
            ),
            idle_exit_seconds=_as_float(
                _env(
                    "BABY_MONITOR_ARCHIVE_IDLE_EXIT_SECONDS",
                    archive_worker.get("idle_exit_seconds", 30.0),
                ),
                "BABY_MONITOR_ARCHIVE_IDLE_EXIT_SECONDS",
                minimum=0.0,
            ),
            max_runtime_seconds=_as_float(
                _env(
                    "BABY_MONITOR_ARCHIVE_MAX_RUNTIME_SECONDS",
                    archive_worker.get("max_runtime_seconds", 3_600.0),
                ),
                "BABY_MONITOR_ARCHIVE_MAX_RUNTIME_SECONDS",
                minimum=0.0,
            ),
            poll_seconds=_as_float(
                _env(
                    "BABY_MONITOR_ARCHIVE_POLL_SECONDS",
                    archive_worker.get("poll_seconds", 5.0),
                ),
                "BABY_MONITOR_ARCHIVE_POLL_SECONDS",
                minimum=0.01,
            ),
        ),
        local_vlm=LocalVLMSettings(
            backend=str(_env("BABY_MONITOR_LOCAL_VLM_BACKEND", local_vlm.get("backend", "mock")))
            .strip()
            .lower(),
            name=str(_env("BABY_MONITOR_LOCAL_VLM_NAME", local_vlm.get("name", "mock"))).strip(),
            revision=str(
                _env(
                    "BABY_MONITOR_LOCAL_VLM_REVISION",
                    local_vlm.get("revision", ""),
                )
            ).strip(),
            command=local_command,
            timeout_seconds=_as_int(
                _env(
                    "BABY_MONITOR_LOCAL_VLM_TIMEOUT_SECONDS",
                    local_vlm.get("timeout_seconds", 180),
                ),
                "BABY_MONITOR_LOCAL_VLM_TIMEOUT_SECONDS",
                minimum=1,
            ),
            confidence_threshold=_as_float(
                _env(
                    "BABY_MONITOR_LOCAL_CONFIDENCE_THRESHOLD",
                    local_vlm.get("confidence_threshold", 0.65),
                ),
                "BABY_MONITOR_LOCAL_CONFIDENCE_THRESHOLD",
            ),
            gpu_device=str(
                _env(
                    "BABY_MONITOR_LOCAL_VLM_GPU_DEVICE",
                    local_vlm.get("gpu_device", "0"),
                )
            ).strip(),
        ),
        review_policy=ReviewPolicySettings(
            signal_frequency_window_seconds=_as_int(
                _env(
                    "BABY_MONITOR_SIGNAL_FREQUENCY_WINDOW_SECONDS",
                    review_policy.get("signal_frequency_window_seconds", 600),
                ),
                "BABY_MONITOR_SIGNAL_FREQUENCY_WINDOW_SECONDS",
                minimum=10,
            ),
            signal_frequency_secondary_threshold=_as_int(
                _env(
                    "BABY_MONITOR_SIGNAL_FREQUENCY_SECONDARY_THRESHOLD",
                    review_policy.get("signal_frequency_secondary_threshold", 3),
                ),
                "BABY_MONITOR_SIGNAL_FREQUENCY_SECONDARY_THRESHOLD",
                minimum=2,
            ),
            priority_signal_types=tuple(priority_signal_types),
        ),
        secondary_vlm=SecondaryVLMSettings(
            enabled=_as_bool(
                _env(
                    "BABY_MONITOR_SECONDARY_VLM_ENABLED",
                    secondary_vlm.get("enabled", False),
                ),
                "BABY_MONITOR_SECONDARY_VLM_ENABLED",
            ),
            backend=str(
                _env(
                    "BABY_MONITOR_SECONDARY_VLM_BACKEND",
                    secondary_vlm.get("backend", "mock"),
                )
            )
            .strip()
            .lower(),
            name=str(
                _env(
                    "BABY_MONITOR_SECONDARY_VLM_NAME",
                    secondary_vlm.get("name", "secondary_mock"),
                )
            ).strip(),
            revision=str(
                _env(
                    "BABY_MONITOR_SECONDARY_VLM_REVISION",
                    secondary_vlm.get("revision", ""),
                )
            ).strip(),
            command=secondary_command,
            timeout_seconds=_as_int(
                _env(
                    "BABY_MONITOR_SECONDARY_VLM_TIMEOUT_SECONDS",
                    secondary_vlm.get("timeout_seconds", 180),
                ),
                "BABY_MONITOR_SECONDARY_VLM_TIMEOUT_SECONDS",
                minimum=1,
            ),
            gpu_device=str(
                _env(
                    "BABY_MONITOR_SECONDARY_VLM_GPU_DEVICE",
                    secondary_vlm.get("gpu_device", "0"),
                )
            ).strip(),
        ),
        remote_vlm=RemoteVLMSettings(
            enabled=_as_bool(
                _env("BABY_MONITOR_REMOTE_ENABLED", remote_vlm.get("enabled", False)),
                "BABY_MONITOR_REMOTE_ENABLED",
            ),
            upload_consent=_as_bool(
                _env(
                    "BABY_MONITOR_REMOTE_UPLOAD_CONSENT",
                    remote_vlm.get("upload_consent", False),
                ),
                "BABY_MONITOR_REMOTE_UPLOAD_CONSENT",
            ),
            base_url=str(_env("BABY_MONITOR_REMOTE_BASE_URL", remote_vlm.get("base_url", ""))),
            model=str(_env("BABY_MONITOR_REMOTE_MODEL", remote_vlm.get("model", ""))),
            api_key_env=str(
                _env(
                    "BABY_MONITOR_REMOTE_API_KEY_ENV",
                    remote_vlm.get("api_key_env", "BABY_MONITOR_REMOTE_API_KEY"),
                )
            ),
            max_frames=_as_int(
                _env(
                    "BABY_MONITOR_REMOTE_MAX_FRAMES",
                    remote_vlm.get("max_frames", 3),
                ),
                "BABY_MONITOR_REMOTE_MAX_FRAMES",
                minimum=1,
            ),
            timeout_seconds=_as_int(
                _env(
                    "BABY_MONITOR_REMOTE_TIMEOUT_SECONDS",
                    remote_vlm.get("timeout_seconds", 60),
                ),
                "BABY_MONITOR_REMOTE_TIMEOUT_SECONDS",
                minimum=1,
            ),
            max_upload_bytes=_as_int(
                _env(
                    "BABY_MONITOR_REMOTE_MAX_UPLOAD_BYTES",
                    remote_vlm.get("max_upload_bytes", 5_242_880),
                ),
                "BABY_MONITOR_REMOTE_MAX_UPLOAD_BYTES",
                minimum=1,
            ),
            max_response_bytes=_as_int(
                _env(
                    "BABY_MONITOR_REMOTE_MAX_RESPONSE_BYTES",
                    remote_vlm.get("max_response_bytes", 262_144),
                ),
                "BABY_MONITOR_REMOTE_MAX_RESPONSE_BYTES",
                minimum=1_024,
            ),
            max_requests_per_day=_as_int(
                _env(
                    "BABY_MONITOR_REMOTE_MAX_REQUESTS_PER_DAY",
                    remote_vlm.get("max_requests_per_day", 20),
                ),
                "BABY_MONITOR_REMOTE_MAX_REQUESTS_PER_DAY",
                minimum=1,
            ),
            min_interval_seconds=_as_int(
                _env(
                    "BABY_MONITOR_REMOTE_MIN_INTERVAL_SECONDS",
                    remote_vlm.get("min_interval_seconds", 60),
                ),
                "BABY_MONITOR_REMOTE_MIN_INTERVAL_SECONDS",
                minimum=0,
            ),
        ),
        identity=IdentitySettings(
            enabled=_as_bool(
                _env(
                    "BABY_MONITOR_IDENTITY_ENABLED",
                    identity.get("enabled", False),
                ),
                "BABY_MONITOR_IDENTITY_ENABLED",
            ),
            biometric_enrollment_consent=_as_bool(
                _env(
                    "BABY_MONITOR_BIOMETRIC_ENROLLMENT_CONSENT",
                    identity.get("biometric_enrollment_consent", False),
                ),
                "BABY_MONITOR_BIOMETRIC_ENROLLMENT_CONSENT",
            ),
            event_matching_enabled=_as_bool(
                _env(
                    "BABY_MONITOR_IDENTITY_EVENT_MATCHING_ENABLED",
                    identity.get("event_matching_enabled", False),
                ),
                "BABY_MONITOR_IDENTITY_EVENT_MATCHING_ENABLED",
            ),
            external_identity_disclosure_enabled=_as_bool(
                _env(
                    "BABY_MONITOR_IDENTITY_EXTERNAL_DISCLOSURE_ENABLED",
                    identity.get("external_identity_disclosure_enabled", False),
                ),
                "BABY_MONITOR_IDENTITY_EXTERNAL_DISCLOSURE_ENABLED",
            ),
        ),
        notification=NotificationSettings(
            enabled=_as_bool(
                _env(
                    "BABY_MONITOR_NOTIFICATION_ENABLED",
                    notification.get("enabled", False),
                ),
                "BABY_MONITOR_NOTIFICATION_ENABLED",
            ),
            external_delivery_enabled=_as_bool(
                _env(
                    "BABY_MONITOR_NOTIFICATION_EXTERNAL_DELIVERY_ENABLED",
                    notification.get("external_delivery_enabled", False),
                ),
                "BABY_MONITOR_NOTIFICATION_EXTERNAL_DELIVERY_ENABLED",
            ),
            auto_file_enabled=_as_bool(
                _env(
                    "BABY_MONITOR_NOTIFICATION_AUTO_FILE_ENABLED",
                    notification.get("auto_file_enabled", False),
                ),
                "BABY_MONITOR_NOTIFICATION_AUTO_FILE_ENABLED",
            ),
            auto_file_sink_id=str(
                _env(
                    "BABY_MONITOR_NOTIFICATION_AUTO_FILE_SINK_ID",
                    notification.get("auto_file_sink_id", "shadow_local"),
                )
            ).strip(),
            event_policy_path=_optional_path(
                _env(
                    "BABY_MONITOR_NOTIFICATION_EVENT_POLICY_PATH",
                    notification.get("event_policy_path"),
                )
            ),
            max_pending=_as_int(
                _env(
                    "BABY_MONITOR_NOTIFICATION_MAX_PENDING",
                    notification.get("max_pending", 64),
                ),
                "BABY_MONITOR_NOTIFICATION_MAX_PENDING",
                minimum=1,
            ),
            max_pending_age_seconds=_as_int(
                _env(
                    "BABY_MONITOR_NOTIFICATION_MAX_PENDING_AGE_SECONDS",
                    notification.get("max_pending_age_seconds", 3600),
                ),
                "BABY_MONITOR_NOTIFICATION_MAX_PENDING_AGE_SECONDS",
                minimum=1,
            ),
            lease_seconds=_as_int(
                _env(
                    "BABY_MONITOR_NOTIFICATION_LEASE_SECONDS",
                    notification.get("lease_seconds", 300),
                ),
                "BABY_MONITOR_NOTIFICATION_LEASE_SECONDS",
                minimum=10,
            ),
            retry_delay_seconds=_as_int(
                _env(
                    "BABY_MONITOR_NOTIFICATION_RETRY_DELAY_SECONDS",
                    notification.get("retry_delay_seconds", 30),
                ),
                "BABY_MONITOR_NOTIFICATION_RETRY_DELAY_SECONDS",
                minimum=0,
            ),
            max_attempts=_as_int(
                _env(
                    "BABY_MONITOR_NOTIFICATION_MAX_ATTEMPTS",
                    notification.get("max_attempts", 5),
                ),
                "BABY_MONITOR_NOTIFICATION_MAX_ATTEMPTS",
                minimum=1,
            ),
        ),
    )

    if not settings.camera.camera_id.strip():
        raise ConfigurationError("camera id cannot be empty")
    if not 0.0 <= settings.detector.confidence <= 1.0:
        raise ConfigurationError("detector confidence must be between 0 and 1")
    if not 0.01 <= settings.detector.sample_fps <= 10.0:
        raise ConfigurationError("detector sample FPS must be between 0.01 and 10")
    if not 320 <= settings.detector.image_size <= 1280 or settings.detector.image_size % 32:
        raise ConfigurationError("YOLO image size must be a multiple of 32 between 320 and 1280")
    if settings.detector.rotation_degrees not in {0, 90, 180, 270}:
        raise ConfigurationError("YOLO rotation degrees must be one of 0, 90, 180, or 270")
    if not 1.0 <= settings.detector.yolo_runner_timeout_seconds <= 3_600.0:
        raise ConfigurationError("YOLO runner timeout must be between 1 and 3600 seconds")
    if not 1.0 <= settings.detector.yolo_execution_lock_timeout_seconds <= 3_600.0:
        raise ConfigurationError("YOLO execution lock timeout must be between 1 and 3600 seconds")
    if settings.detector.yolo_execution_lock_path is not None and not (
        settings.detector.yolo_execution_lock_path.is_absolute()
    ):
        raise ConfigurationError("YOLO execution lock path must be absolute")
    if settings.detector.backend == "yolo_command" and not settings.detector.yolo_runner_command:
        raise ConfigurationError(
            "detector.yolo_runner_command is required for yolo_command backend"
        )
    if settings.detector.device != "cpu" and not re.fullmatch(r"\d+", settings.detector.device):
        raise ConfigurationError("YOLO device must be cpu or one numeric GPU index")
    if not settings.detector.labels or len(set(settings.detector.labels)) != len(
        settings.detector.labels
    ):
        raise ConfigurationError("detector labels must be non-empty and unique")
    if any(
        not value.strip()
        or value != value.strip()
        or len(value) > 80
        or any(character in value for character in ("\x00", "\r", "\n"))
        for value in settings.detector.labels
    ):
        raise ConfigurationError("detector labels contain an invalid value")
    if not 0.0 < settings.detector.scene_threshold <= 1.0:
        raise ConfigurationError("scene threshold must be greater than 0 and at most 1")
    if settings.detector.scene_max_detections > 10_000:
        raise ConfigurationError("scene max detections must be at most 10000")
    if not 0.0 <= settings.local_vlm.confidence_threshold <= 1.0:
        raise ConfigurationError("local VLM confidence threshold must be between 0 and 1")
    if settings.review_policy.signal_frequency_window_seconds > 86_400:
        raise ConfigurationError("signal frequency window must be at most 86400 seconds")
    if settings.review_policy.signal_frequency_secondary_threshold > 1_000:
        raise ConfigurationError("signal frequency threshold must be at most 1000")
    allowed_signal_types = {
        "motion",
        "person",
        "face",
        "sound_peak",
        "hardware_cry",
        "zone_enter",
        "zone_dwell",
        "zone_exit",
        "fall_hint",
        "manual_trigger",
    }
    if len(set(settings.review_policy.priority_signal_types)) != len(
        settings.review_policy.priority_signal_types
    ) or any(
        value not in allowed_signal_types for value in settings.review_policy.priority_signal_types
    ):
        raise ConfigurationError("review policy priority signal types are invalid")
    if settings.local_vlm.backend not in {"mock", "command"}:
        raise ConfigurationError("local VLM backend must be mock or command")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", settings.local_vlm.name):
        raise ConfigurationError(
            "local VLM name must contain only letters, digits, dot, underscore, or dash"
        )
    if settings.local_vlm.revision and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", settings.local_vlm.revision
    ):
        raise ConfigurationError(
            "local VLM revision must be empty or a path-free revision identifier"
        )
    if settings.local_vlm.timeout_seconds > 3_600:
        raise ConfigurationError("local VLM timeout must be at most 3600 seconds")
    if not re.fullmatch(r"[0-9]{1,3}", settings.local_vlm.gpu_device):
        raise ConfigurationError("local VLM GPU device must be a numeric index")
    if settings.local_vlm.backend == "command":
        if not settings.local_vlm.command:
            raise ConfigurationError("local_vlm.command is required for command backend")
        if sum(part.count("{request}") for part in settings.local_vlm.command) != 1:
            raise ConfigurationError("local_vlm.command must contain {request} exactly once")
        if any(
            "\x00" in part or "\r" in part or "\n" in part for part in settings.local_vlm.command
        ):
            raise ConfigurationError("local_vlm.command cannot contain control characters")
        executable = Path(settings.local_vlm.command[0])
        if not executable.is_absolute() or not executable.is_file():
            raise ConfigurationError(
                "local_vlm.command executable must be an existing absolute file"
            )
    elif settings.local_vlm.command:
        raise ConfigurationError("local_vlm.command must be empty for mock backend")
    if settings.secondary_vlm.backend not in {"mock", "command"}:
        raise ConfigurationError("secondary VLM backend must be mock or command")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", settings.secondary_vlm.name):
        raise ConfigurationError(
            "secondary VLM name must contain only letters, digits, dot, underscore, or dash"
        )
    if settings.secondary_vlm.revision and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", settings.secondary_vlm.revision
    ):
        raise ConfigurationError(
            "secondary VLM revision must be empty or a path-free revision identifier"
        )
    if settings.secondary_vlm.timeout_seconds > 3_600:
        raise ConfigurationError("secondary VLM timeout must be at most 3600 seconds")
    if not re.fullmatch(r"[0-9]{1,3}", settings.secondary_vlm.gpu_device):
        raise ConfigurationError("secondary VLM GPU device must be a numeric index")
    if settings.secondary_vlm.backend == "command":
        if settings.secondary_vlm.enabled and not settings.secondary_vlm.command:
            raise ConfigurationError(
                "secondary_vlm.command is required for an enabled command backend"
            )
        if (
            settings.secondary_vlm.command
            and sum(part.count("{request}") for part in settings.secondary_vlm.command) != 1
        ):
            raise ConfigurationError("secondary_vlm.command must contain {request} exactly once")
        if any(
            "\x00" in part or "\r" in part or "\n" in part
            for part in settings.secondary_vlm.command
        ):
            raise ConfigurationError("secondary_vlm.command cannot contain control characters")
        if settings.secondary_vlm.command:
            executable = Path(settings.secondary_vlm.command[0])
            if not executable.is_absolute() or not executable.is_file():
                raise ConfigurationError(
                    "secondary_vlm.command executable must be an existing absolute file"
                )
    elif settings.secondary_vlm.command:
        raise ConfigurationError("secondary_vlm.command must be empty for mock backend")
    if not 3 <= settings.media.keyframe_count <= 8:
        raise ConfigurationError("keyframe_count must be between 3 and 8")
    if not 1.0 <= settings.buffer.segment_seconds <= 60.0:
        raise ConfigurationError("buffer segment seconds must be between 1 and 60")
    if settings.buffer.max_segments > 10_000:
        raise ConfigurationError("buffer max segments must be at most 10000")
    if not (
        settings.buffer.reconnect_initial_seconds <= settings.buffer.reconnect_max_seconds <= 300.0
    ):
        raise ConfigurationError("buffer reconnect delays are invalid")
    if settings.buffer.poll_interval_seconds > 5.0:
        raise ConfigurationError("buffer poll interval must be at most 5 seconds")
    if settings.media_health.stale_after_seconds > 604_800:
        raise ConfigurationError("media health stale threshold must be at most 604800 seconds")
    if settings.media_health.future_tolerance_seconds > 3_600:
        raise ConfigurationError("media health future tolerance must be at most 3600 seconds")
    if settings.candidate_worker.max_segments_per_run > 10_000:
        raise ConfigurationError("candidate worker max segments must be at most 10000")
    if settings.candidate_worker.idle_exit_seconds > 86_400:
        raise ConfigurationError("candidate worker idle exit must be at most 86400 seconds")
    if settings.candidate_worker.max_runtime_seconds > 604_800:
        raise ConfigurationError("candidate worker runtime must be at most 604800 seconds")
    if settings.candidate_worker.poll_seconds > 60:
        raise ConfigurationError("candidate worker poll interval must be at most 60 seconds")
    if settings.candidate_worker.max_event_span_seconds > 300:
        raise ConfigurationError("candidate event span must be at most 300 seconds")
    if not 0.0 <= settings.candidate_worker.candidate_start_confidence <= 1.0:
        raise ConfigurationError("candidate start confidence must be between 0 and 1")
    if settings.candidate_worker.max_pending_detections > 10_000:
        raise ConfigurationError("candidate pending detections must be at most 10000")
    if settings.candidate_worker.detector_timeout_seconds > 3_600:
        raise ConfigurationError("candidate detector timeout must be at most 3600 seconds")
    if (
        settings.candidate_worker.max_event_span_seconds
        + settings.media.pre_event_seconds
        + settings.media.post_event_seconds
        > 300
    ):
        raise ConfigurationError("candidate evidence window must be at most 300 seconds")
    if settings.analysis_queue.max_pending > 10_000:
        raise ConfigurationError("analysis queue max pending must be at most 10000")
    if settings.analysis_queue.lease_seconds > 86_400:
        raise ConfigurationError("analysis queue lease must be at most 86400 seconds")
    if settings.analysis_queue.retry_delay_seconds > 86_400:
        raise ConfigurationError("analysis queue retry delay must be at most 86400 seconds")
    if settings.analysis_queue.max_attempts > 100:
        raise ConfigurationError("analysis queue max attempts must be at most 100")
    if settings.analysis_worker.max_jobs_per_run > 10_000:
        raise ConfigurationError("analysis worker max jobs must be at most 10000")
    if settings.analysis_worker.idle_exit_seconds > 86_400:
        raise ConfigurationError("analysis worker idle exit must be at most 86400 seconds")
    if settings.analysis_worker.max_runtime_seconds > 604_800:
        raise ConfigurationError("analysis worker max runtime must be at most 604800 seconds")
    if settings.analysis_worker.poll_seconds > 60:
        raise ConfigurationError("analysis worker poll interval must be at most 60 seconds")
    if (
        settings.analysis_worker.max_jobs_per_run == 0
        and settings.analysis_worker.idle_exit_seconds == 0
        and settings.analysis_worker.max_runtime_seconds == 0
    ):
        raise ConfigurationError("analysis worker must have at least one stopping bound")
    if not re.fullmatch(r"[0-9]{1,3}", settings.analysis_worker.gpu_device):
        raise ConfigurationError("analysis worker GPU device must be a numeric index")
    if settings.analysis_worker.min_free_vram_mib > 1_000_000:
        raise ConfigurationError("analysis worker minimum free VRAM is invalid")
    if settings.analysis_worker.max_gpu_utilization_percent > 100:
        raise ConfigurationError("analysis worker maximum GPU utilization must be at most 100")
    if settings.analysis_worker.gpu_probe_timeout_seconds > 30:
        raise ConfigurationError("analysis worker GPU probe timeout must be at most 30 seconds")
    if settings.archive_queue.max_pending > 10_000:
        raise ConfigurationError("archive queue max pending must be at most 10000")
    if settings.archive_queue.lease_seconds > 86_400:
        raise ConfigurationError("archive queue lease must be at most 86400 seconds")
    if settings.archive_queue.retry_delay_seconds > 86_400:
        raise ConfigurationError("archive queue retry delay must be at most 86400 seconds")
    if settings.archive_queue.max_attempts > 100:
        raise ConfigurationError("archive queue max attempts must be at most 100")
    if settings.archive_worker.max_jobs_per_run > 10_000:
        raise ConfigurationError("archive worker max jobs must be at most 10000")
    if settings.archive_worker.idle_exit_seconds > 86_400:
        raise ConfigurationError("archive worker idle exit must be at most 86400 seconds")
    if settings.archive_worker.max_runtime_seconds > 604_800:
        raise ConfigurationError("archive worker max runtime must be at most 604800 seconds")
    if settings.archive_worker.poll_seconds > 60:
        raise ConfigurationError("archive worker poll interval must be at most 60 seconds")
    if (
        settings.archive_worker.max_jobs_per_run == 0
        and settings.archive_worker.idle_exit_seconds == 0
        and settings.archive_worker.max_runtime_seconds == 0
    ):
        raise ConfigurationError("archive worker must have at least one stopping bound")
    if settings.remote_vlm.max_frames > 3:
        raise ConfigurationError("remote VLM may upload at most 3 frames")
    if settings.notification.max_pending_age_seconds > 604_800:
        raise ConfigurationError("notification max_pending_age_seconds must be <= 604800")
    if settings.remote_vlm.timeout_seconds > 600:
        raise ConfigurationError("remote VLM timeout must be at most 600 seconds")
    if settings.remote_vlm.max_upload_bytes > 52_428_800:
        raise ConfigurationError("remote VLM upload byte budget must be at most 52428800")
    if settings.remote_vlm.max_response_bytes > 1_048_576:
        raise ConfigurationError("remote VLM response byte limit must be at most 1048576")
    if settings.remote_vlm.max_requests_per_day > 10_000:
        raise ConfigurationError("remote VLM daily request budget must be at most 10000")
    if settings.remote_vlm.min_interval_seconds > 86_400:
        raise ConfigurationError("remote VLM request interval must be at most 86400 seconds")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", settings.remote_vlm.api_key_env) is None:
        raise ConfigurationError("remote VLM API key environment name is invalid")
    if settings.notification.max_pending > 10_000:
        raise ConfigurationError("notification max pending must be at most 10000")
    if settings.notification.auto_file_enabled and not settings.notification.enabled:
        raise ConfigurationError(
            "notification auto file delivery requires notification.enabled=true"
        )
    if settings.notification.auto_file_enabled and settings.notification.event_policy_path is None:
        raise ConfigurationError(
            "notification auto file delivery requires notification.event_policy_path"
        )
    if re.fullmatch(r"[a-z0-9_.-]{1,64}", settings.notification.auto_file_sink_id) is None:
        raise ConfigurationError("notification auto file sink id is invalid")
    if settings.notification.lease_seconds > 86_400:
        raise ConfigurationError("notification lease must be at most 86400 seconds")
    if settings.notification.retry_delay_seconds > 86_400:
        raise ConfigurationError("notification retry delay must be at most 86400 seconds")
    if settings.notification.max_attempts > 100:
        raise ConfigurationError("notification max attempts must be at most 100")
    analysis_timeout_budget = 0
    if settings.local_vlm.backend == "command":
        analysis_timeout_budget += settings.local_vlm.timeout_seconds
    if settings.secondary_vlm.enabled and settings.secondary_vlm.backend == "command":
        analysis_timeout_budget += settings.secondary_vlm.timeout_seconds
    if settings.remote_vlm.uploads_allowed:
        analysis_timeout_budget += settings.remote_vlm.timeout_seconds
    if analysis_timeout_budget and settings.analysis_queue.lease_seconds <= analysis_timeout_budget:
        raise ConfigurationError(
            "analysis queue lease must be longer than the sequential review timeout budget"
        )
    return settings
