from __future__ import annotations

import math
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.analysis_queue import QueueSnapshot
from baby_monitor.archive_queue import ArchiveQueueSnapshot
from baby_monitor.gpu_guard import GPUCapacity
from baby_monitor.media_health import MediaHealthAssessment
from baby_monitor.notification import NotificationQueueSnapshot


class RuntimeStatusError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StorageCapacitySnapshot:
    total_bytes: int
    used_bytes: int
    free_bytes: int
    free_percent: float
    minimum_free_bytes: int
    sufficient: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueueHealthSnapshot:
    snapshot: QueueSnapshot
    max_active: int
    max_pending_age_seconds: float
    healthy: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **self.snapshot.to_dict(),
            "max_active": self.max_active,
            "max_pending_age_seconds": self.max_pending_age_seconds,
            "healthy": self.healthy,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class ArchiveQueueHealthSnapshot:
    snapshot: ArchiveQueueSnapshot
    max_active: int
    max_pending_age_seconds: float
    healthy: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **self.snapshot.to_dict(),
            "max_active": self.max_active,
            "max_pending_age_seconds": self.max_pending_age_seconds,
            "healthy": self.healthy,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class NotificationQueueHealthSnapshot:
    snapshot: NotificationQueueSnapshot
    enabled: bool
    max_active: int
    max_pending_age_seconds: float
    healthy: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            **self.snapshot.to_dict(),
            "enabled": self.enabled,
            "max_active": self.max_active,
            "max_pending_age_seconds": self.max_pending_age_seconds,
            "healthy": self.healthy,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class AnalysisCapacitySnapshot:
    gpu_guard_enabled: bool
    admission_allowed: bool | None
    reason_code: str
    gpu: GPUCapacity | None

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu_guard_enabled": self.gpu_guard_enabled,
            "admission_allowed": self.admission_allowed,
            "reason_code": self.reason_code,
            "gpu": self.gpu.to_dict() if self.gpu is not None else None,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSafetySnapshot:
    remote_enabled: bool
    remote_upload_consent: bool
    remote_uploads_allowed: bool
    notification_enabled: bool
    external_notification_delivery_enabled: bool
    retention_scan_enabled: bool
    retention_delete_enabled: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    checked_at: datetime
    media: tuple[MediaHealthAssessment, ...]
    queue: QueueHealthSnapshot
    archive_queue: ArchiveQueueHealthSnapshot
    notification_queue: NotificationQueueHealthSnapshot
    storage: StorageCapacitySnapshot
    analysis: AnalysisCapacitySnapshot
    safety: RuntimeSafetySnapshot
    operational: bool
    attention_required: bool
    reason_codes: tuple[str, ...]
    advisory_codes: tuple[str, ...]
    version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "checked_at": self.checked_at.astimezone(UTC).isoformat(),
            "status": "operational" if self.operational else "attention",
            "operational": self.operational,
            "attention_required": self.attention_required,
            "reason_codes": list(self.reason_codes),
            "advisory_codes": list(self.advisory_codes),
            "media": [item.to_dict() for item in self.media],
            "queue": self.queue.to_dict(),
            "archive_queue": self.archive_queue.to_dict(),
            "notification_queue": self.notification_queue.to_dict(),
            "storage": self.storage.to_dict(),
            "analysis": self.analysis.to_dict(),
            "safety": self.safety.to_dict(),
            "paths_disclosed": False,
        }


def inspect_storage_capacity(
    path: str | Path,
    *,
    minimum_free_bytes: int,
) -> StorageCapacitySnapshot:
    if type(minimum_free_bytes) is not int or not 1 <= minimum_free_bytes <= 10**15:
        raise RuntimeStatusError("minimum_free_bytes must be between 1 and 1000000000000000")
    root = Path(path).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise RuntimeStatusError("local storage root is missing or unsafe")
    try:
        usage = shutil.disk_usage(root)
    except OSError as exc:
        raise RuntimeStatusError("local storage capacity is unavailable") from exc
    free_percent = round((usage.free / usage.total) * 100, 3) if usage.total else 0.0
    return StorageCapacitySnapshot(
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        free_percent=free_percent,
        minimum_free_bytes=minimum_free_bytes,
        sufficient=usage.free >= minimum_free_bytes,
    )


def assess_queue_health(
    snapshot: QueueSnapshot,
    *,
    max_active: int,
    max_pending_age_seconds: float,
) -> QueueHealthSnapshot:
    if type(max_active) is not int or not 1 <= max_active <= 10_000:
        raise RuntimeStatusError("max_active must be between 1 and 10000")
    if (
        type(max_pending_age_seconds) not in {int, float}
        or not math.isfinite(max_pending_age_seconds)
        or not 1 <= max_pending_age_seconds <= 604_800
    ):
        raise RuntimeStatusError("max_pending_age_seconds must be between 1 and 604800")

    reasons: list[str] = []
    if snapshot.counts.active >= max_active:
        reasons.append("queue_at_capacity")
    if snapshot.counts.failed:
        reasons.append("queue_failed_jobs")
    if snapshot.missing_media_files:
        reasons.append("queue_missing_media")
    if (
        snapshot.oldest_pending_age_seconds is not None
        and snapshot.oldest_pending_age_seconds > max_pending_age_seconds
    ):
        reasons.append("queue_pending_too_old")
    return QueueHealthSnapshot(
        snapshot=snapshot,
        max_active=max_active,
        max_pending_age_seconds=float(max_pending_age_seconds),
        healthy=not reasons,
        reason_codes=tuple(reasons),
    )


def assess_archive_queue_health(
    snapshot: ArchiveQueueSnapshot,
    *,
    max_active: int,
    max_pending_age_seconds: float,
) -> ArchiveQueueHealthSnapshot:
    if type(max_active) is not int or not 1 <= max_active <= 10_000:
        raise RuntimeStatusError("archive max_active must be between 1 and 10000")
    if (
        type(max_pending_age_seconds) not in {int, float}
        or not math.isfinite(max_pending_age_seconds)
        or not 1 <= max_pending_age_seconds <= 2_592_000
    ):
        raise RuntimeStatusError("archive max_pending_age_seconds must be between 1 and 2592000")
    reasons: list[str] = []
    if snapshot.counts.active >= max_active:
        reasons.append("archive_queue_at_capacity")
    if snapshot.counts.failed:
        reasons.append("archive_failed_jobs")
    if (
        snapshot.oldest_pending_age_seconds is not None
        and snapshot.oldest_pending_age_seconds > max_pending_age_seconds
    ):
        reasons.append("archive_pending_too_old")
    return ArchiveQueueHealthSnapshot(
        snapshot=snapshot,
        max_active=max_active,
        max_pending_age_seconds=float(max_pending_age_seconds),
        healthy=not reasons,
        reason_codes=tuple(reasons),
    )


def assess_notification_queue_health(
    snapshot: NotificationQueueSnapshot,
    *,
    enabled: bool,
    max_active: int,
    max_pending_age_seconds: float,
) -> NotificationQueueHealthSnapshot:
    if type(enabled) is not bool:
        raise RuntimeStatusError("notification enabled must be a boolean")
    if type(max_active) is not int or not 1 <= max_active <= 10_000:
        raise RuntimeStatusError("notification max_active must be between 1 and 10000")
    if (
        type(max_pending_age_seconds) not in {int, float}
        or not math.isfinite(max_pending_age_seconds)
        or not 1 <= max_pending_age_seconds <= 604_800
    ):
        raise RuntimeStatusError(
            "notification max_pending_age_seconds must be between 1 and 604800"
        )

    reasons: list[str] = []
    if not enabled and snapshot.counts.active:
        reasons.append("notification_disabled_with_backlog")
    if snapshot.counts.active >= max_active:
        reasons.append("notification_queue_at_capacity")
    if snapshot.counts.failed:
        reasons.append("notification_failed_jobs")
    if (
        snapshot.oldest_pending_age_seconds is not None
        and snapshot.oldest_pending_age_seconds > max_pending_age_seconds
    ):
        reasons.append("notification_pending_too_old")
    return NotificationQueueHealthSnapshot(
        snapshot=snapshot,
        enabled=enabled,
        max_active=max_active,
        max_pending_age_seconds=float(max_pending_age_seconds),
        healthy=not reasons,
        reason_codes=tuple(reasons),
    )


def analysis_capacity_snapshot(
    *,
    gpu_guard_enabled: bool,
    capacity: GPUCapacity | None,
) -> AnalysisCapacitySnapshot:
    if type(gpu_guard_enabled) is not bool:
        raise RuntimeStatusError("gpu_guard_enabled must be a boolean")
    if not gpu_guard_enabled:
        return AnalysisCapacitySnapshot(
            gpu_guard_enabled=False,
            admission_allowed=None,
            reason_code="gpu_guard_disabled",
            gpu=None,
        )
    if capacity is None:
        raise RuntimeStatusError("GPU capacity is required when the guard is enabled")
    return AnalysisCapacitySnapshot(
        gpu_guard_enabled=True,
        admission_allowed=capacity.allowed,
        reason_code=capacity.reason,
        gpu=capacity,
    )


def build_runtime_status(
    *,
    checked_at: datetime,
    media: tuple[MediaHealthAssessment, ...],
    queue: QueueHealthSnapshot,
    archive_queue: ArchiveQueueHealthSnapshot,
    notification_queue: NotificationQueueHealthSnapshot,
    storage: StorageCapacitySnapshot,
    analysis: AnalysisCapacitySnapshot,
    safety: RuntimeSafetySnapshot,
) -> RuntimeStatus:
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise RuntimeStatusError("checked_at must be timezone-aware")
    if not media:
        raise RuntimeStatusError("at least one media health assessment is required")
    camera_ids = [item.camera_id for item in media]
    if len(camera_ids) != len(set(camera_ids)):
        raise RuntimeStatusError("media health camera_id values must be unique")

    reasons: list[str] = []
    if any(not item.operational for item in media):
        reasons.append("media_not_operational")
    if not queue.healthy:
        reasons.extend(queue.reason_codes)
    if not archive_queue.healthy:
        reasons.extend(archive_queue.reason_codes)
    if not notification_queue.healthy:
        reasons.extend(notification_queue.reason_codes)
    if not storage.sufficient:
        reasons.append("storage_low_free_space")
    advisories = ("analysis_deferred",) if analysis.admission_allowed is False else ()
    operational = not reasons
    return RuntimeStatus(
        checked_at=checked_at.astimezone(UTC),
        media=media,
        queue=queue,
        archive_queue=archive_queue,
        notification_queue=notification_queue,
        storage=storage,
        analysis=analysis,
        safety=safety,
        operational=operational,
        attention_required=not operational,
        reason_codes=tuple(dict.fromkeys(reasons)),
        advisory_codes=advisories,
    )
