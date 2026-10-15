from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from baby_monitor.adapters.camera import CameraSource, redact_uri
from baby_monitor.adapters.detector import Detector
from baby_monitor.analysis_queue import AnalysisQueue, QueueFullError
from baby_monitor.candidate_health import (
    CandidateHealthLifetime,
    CandidateHealthState,
    CandidateSamplingHealth,
    CandidateWorkerHealth,
)
from baby_monitor.detector_protocol import DetectorScanResult
from baby_monitor.detector_sampling import DetectorSamplingLedger
from baby_monitor.domain import CandidateEvent, Detection, EventMedia
from baby_monitor.media_health import MediaHealthError, MediaSourceHealth
from baby_monitor.pose_presence_shadow import (
    PosePresenceShadowClassifier,
    PosePresenceShadowError,
)
from baby_monitor.presence_history import PresenceFactMode, PresenceFactStore
from baby_monitor.presence_sampling import PresenceSamplingObservation, project_presence_sampling
from baby_monitor.ring_buffer import (
    BufferedEvidenceResult,
    BufferSegment,
    SegmentManifest,
    extract_buffered_evidence,
)
from baby_monitor.room_presence import (
    StreamPresenceBatch,
    StreamPresenceSample,
    StreamPresenceSampleState,
)
from baby_monitor.sampling_history import SamplingEvidenceStore
from baby_monitor.storage import StorageResolver


class CandidateWorkerError(RuntimeError):
    pass


def _utc(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CandidateWorkerError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise CandidateWorkerError(f"{field} must be an ISO timestamp")
    try:
        return _utc(datetime.fromisoformat(value), field)
    except ValueError as exc:
        raise CandidateWorkerError(f"{field} must be an ISO timestamp") from exc


def _safe_segment_name(value: Any) -> str:
    if not isinstance(value, str) or Path(value).name != value:
        raise CandidateWorkerError("candidate state segment filename is unsafe")
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", value) is None:
        raise CandidateWorkerError("candidate state segment filename is unsafe")
    return value


@dataclass(frozen=True, slots=True)
class SegmentFingerprint:
    filename: str
    started_at: datetime
    ended_at: datetime
    size_bytes: int

    def __post_init__(self) -> None:
        _safe_segment_name(self.filename)
        started = _utc(self.started_at, "segment.started_at")
        ended = _utc(self.ended_at, "segment.ended_at")
        if ended <= started or type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise CandidateWorkerError("candidate state segment fingerprint is invalid")

    @classmethod
    def from_segment(cls, segment: BufferSegment) -> SegmentFingerprint:
        return cls(
            filename=segment.filename,
            started_at=segment.started_at,
            ended_at=segment.ended_at,
            size_bytes=segment.size_bytes,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "started_at": self.started_at.astimezone(UTC).isoformat(),
            "ended_at": self.ended_at.astimezone(UTC).isoformat(),
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SegmentFingerprint:
        if not isinstance(value, dict) or set(value) != {
            "filename",
            "started_at",
            "ended_at",
            "size_bytes",
        }:
            raise CandidateWorkerError("candidate state segment fields do not match v1")
        return cls(
            filename=_safe_segment_name(value["filename"]),
            started_at=_parse_time(value["started_at"], "segment.started_at"),
            ended_at=_parse_time(value["ended_at"], "segment.ended_at"),
            size_bytes=value["size_bytes"],
        )


@dataclass(frozen=True, slots=True)
class TimedDetection:
    occurred_at: datetime
    segment_filename: str
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        _utc(self.occurred_at, "detection.occurred_at")
        _safe_segment_name(self.segment_filename)
        if (
            not isinstance(self.label, str)
            or not self.label.strip()
            or len(self.label) > 80
            or any(value in self.label for value in ("\x00", "\r", "\n"))
        ):
            raise CandidateWorkerError("candidate state detection label is invalid")
        if (
            not isinstance(self.bbox_xyxy, tuple)
            or len(self.bbox_xyxy) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in self.bbox_xyxy
            )
        ):
            raise CandidateWorkerError("candidate state detection bbox is invalid")
        try:
            Detection(0.0, self.label, self.confidence, self.bbox_xyxy)
        except (TypeError, ValueError) as exc:
            raise CandidateWorkerError("candidate state detection is invalid") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "occurred_at": self.occurred_at.astimezone(UTC).isoformat(),
            "segment_filename": self.segment_filename,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
        }

    @classmethod
    def from_dict(cls, value: Any) -> TimedDetection:
        if not isinstance(value, dict) or set(value) != {
            "occurred_at",
            "segment_filename",
            "label",
            "confidence",
            "bbox_xyxy",
        }:
            raise CandidateWorkerError("candidate state detection fields do not match v1")
        bbox = value["bbox_xyxy"]
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in bbox)
        ):
            raise CandidateWorkerError("candidate state detection bbox is invalid")
        try:
            coordinates = tuple(float(item) for item in bbox)
        except (TypeError, ValueError) as exc:
            raise CandidateWorkerError("candidate state detection bbox is invalid") from exc
        return cls(
            occurred_at=_parse_time(value["occurred_at"], "detection.occurred_at"),
            segment_filename=_safe_segment_name(value["segment_filename"]),
            label=value["label"],
            confidence=value["confidence"],
            bbox_xyxy=coordinates,  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CandidateScanState:
    camera_id: str
    source_ref: str
    scanner_identity: str
    updated_at: datetime
    watermark_at: datetime | None = None
    processed_segments: tuple[SegmentFingerprint, ...] = ()
    pending_detections: tuple[TimedDetection, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise CandidateWorkerError("unsupported candidate scan state version")
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.camera_id) is None:
            raise CandidateWorkerError("candidate state camera_id is invalid")
        if (
            not isinstance(self.source_ref, str)
            or not self.source_ref.lower().startswith(("rtsp://", "rtsps://"))
            or redact_uri(self.source_ref) != self.source_ref
        ):
            raise CandidateWorkerError("candidate state source_ref must be redacted")
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.scanner_identity) is None:
            raise CandidateWorkerError("candidate state scanner_identity is invalid")
        _utc(self.updated_at, "updated_at")
        watermark = (
            _utc(self.watermark_at, "watermark_at") if self.watermark_at is not None else None
        )
        if not isinstance(self.processed_segments, tuple) or len(self.processed_segments) > 10_000:
            raise CandidateWorkerError("candidate state processed segments are invalid")
        names = [item.filename for item in self.processed_segments]
        if len(names) != len(set(names)):
            raise CandidateWorkerError("candidate state processed segments are duplicated")
        if not isinstance(self.pending_detections, tuple) or len(self.pending_detections) > 10_000:
            raise CandidateWorkerError("candidate state pending detections are invalid")
        if tuple(sorted(self.pending_detections, key=lambda item: item.occurred_at)) != (
            self.pending_detections
        ):
            raise CandidateWorkerError("candidate state detections must be ordered")
        if self.pending_detections and watermark is None:
            raise CandidateWorkerError("candidate state pending detections require a watermark")
        if watermark is not None and any(
            item.occurred_at > watermark for item in self.pending_detections
        ):
            raise CandidateWorkerError("candidate state detection exceeds the watermark")
        processed_names = set(names)
        if any(item.segment_filename not in processed_names for item in self.pending_detections):
            raise CandidateWorkerError("candidate state detection origin is not processed")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "source_ref": self.source_ref,
            "scanner_identity": self.scanner_identity,
            "updated_at": self.updated_at.astimezone(UTC).isoformat(),
            "watermark_at": (
                self.watermark_at.astimezone(UTC).isoformat()
                if self.watermark_at is not None
                else None
            ),
            "processed_segments": [item.to_dict() for item in self.processed_segments],
            "pending_detections": [item.to_dict() for item in self.pending_detections],
        }

    @classmethod
    def from_dict(cls, value: Any) -> CandidateScanState:
        expected = {
            "version",
            "camera_id",
            "source_ref",
            "scanner_identity",
            "updated_at",
            "watermark_at",
            "processed_segments",
            "pending_detections",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise CandidateWorkerError("candidate scan state fields do not match v1")
        watermark = value["watermark_at"]
        processed = value["processed_segments"]
        pending = value["pending_detections"]
        if not isinstance(processed, list) or not isinstance(pending, list):
            raise CandidateWorkerError("candidate scan state arrays are invalid")
        return cls(
            version=value["version"],
            camera_id=value["camera_id"],
            source_ref=value["source_ref"],
            scanner_identity=value["scanner_identity"],
            updated_at=_parse_time(value["updated_at"], "updated_at"),
            watermark_at=(
                _parse_time(watermark, "watermark_at") if watermark is not None else None
            ),
            processed_segments=tuple(SegmentFingerprint.from_dict(item) for item in processed),
            pending_detections=tuple(TimedDetection.from_dict(item) for item in pending),
        )

    @classmethod
    def read(cls, path: str | Path) -> CandidateScanState:
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CandidateWorkerError("candidate scan state is missing or invalid") from exc
        return cls.from_dict(value)

    def write_atomic(self, path: str | Path) -> None:
        state_path = Path(path).expanduser()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_name(f".{state_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, state_path)
        finally:
            if temporary.exists():
                temporary.unlink()


@dataclass(frozen=True, slots=True)
class CandidateWorkerSettings:
    manifest_path: Path
    state_path: Path
    database_path: Path
    storage: StorageResolver
    detector: Detector
    scanner_identity: str
    detector_identity: str
    health_path: Path | None = None
    presence_path: Path | None = None
    presence_store: PresenceFactStore | None = None
    shadow_presence_path: Path | None = None
    shadow_presence_store: PresenceFactStore | None = None
    shadow_presence_classifier: PosePresenceShadowClassifier | None = None
    shadow_min_interval_seconds: float = 0.0
    pre_event_seconds: float = 5.0
    post_event_seconds: float = 7.0
    candidate_gap_seconds: float = 3.0
    candidate_start_confidence: float = 0.0
    max_event_span_seconds: float = 30.0
    keyframe_count: int = 5
    max_pending_detections: int = 1_000
    max_segments_per_run: int = 32
    idle_exit_seconds: float = 30.0
    max_runtime_seconds: float = 3_600.0
    poll_seconds: float = 1.0
    max_active_jobs: int = 32
    max_presence_samples_per_stream: int = 64
    presence_live_edge_segments: int = 3
    emit_candidate_events: bool = True
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    def __post_init__(self) -> None:
        if type(self.emit_candidate_events) is not bool:
            raise ValueError("emit_candidate_events must be a boolean")
        if not self.emit_candidate_events and self.presence_path is None:
            raise ValueError("presence-only candidate worker requires a presence path")
        if self.presence_store is not None and (
            not isinstance(self.presence_store, PresenceFactStore)
            or self.presence_path is None
            or not self.emit_candidate_events
        ):
            raise ValueError(
                "presence fact persistence requires a normal candidate worker presence path"
            )
        shadow_values = (
            self.shadow_presence_path,
            self.shadow_presence_store,
            self.shadow_presence_classifier,
        )
        if any(item is not None for item in shadow_values) and (
            any(item is None for item in shadow_values)
            or not self.emit_candidate_events
            or self.presence_path is None
            or not isinstance(self.shadow_presence_store, PresenceFactStore)
            or not isinstance(
                self.shadow_presence_classifier,
                PosePresenceShadowClassifier,
            )
        ):
            raise ValueError(
                "pose presence shadow requires a complete normal candidate worker configuration"
            )
        if (
            isinstance(self.shadow_min_interval_seconds, bool)
            or not isinstance(self.shadow_min_interval_seconds, (int, float))
            or not math.isfinite(float(self.shadow_min_interval_seconds))
            or not 0.0 <= float(self.shadow_min_interval_seconds) <= 3_600.0
        ):
            raise ValueError("shadow_min_interval_seconds must be between 0 and 3600")
        if self.manifest_path.expanduser().resolve() == self.state_path.expanduser().resolve():
            raise ValueError("candidate state path must differ from the manifest path")
        if self.health_path is not None and self.health_path.expanduser().resolve() in {
            self.manifest_path.expanduser().resolve(),
            self.state_path.expanduser().resolve(),
            self.database_path.expanduser().resolve(),
        }:
            raise ValueError("candidate health path must be dedicated")
        if self.presence_path is not None:
            if not self.presence_path.is_absolute():
                raise ValueError("candidate presence path must be absolute")
            if self.presence_path.expanduser().resolve() in {
                self.manifest_path.expanduser().resolve(),
                self.state_path.expanduser().resolve(),
                self.database_path.expanduser().resolve(),
                *(
                    (self.health_path.expanduser().resolve(),)
                    if self.health_path is not None
                    else ()
                ),
            }:
                raise ValueError("candidate presence path must be dedicated")
        if self.shadow_presence_path is not None:
            if not self.shadow_presence_path.is_absolute():
                raise ValueError("candidate shadow presence path must be absolute")
            if self.shadow_presence_path.expanduser().resolve() in {
                self.manifest_path.expanduser().resolve(),
                self.state_path.expanduser().resolve(),
                self.database_path.expanduser().resolve(),
                *(
                    (self.health_path.expanduser().resolve(),)
                    if self.health_path is not None
                    else ()
                ),
                *(
                    (self.presence_path.expanduser().resolve(),)
                    if self.presence_path is not None
                    else ()
                ),
            }:
                raise ValueError("candidate shadow presence path must be dedicated")
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.scanner_identity) is None:
            raise ValueError("scanner_identity must be a safe identifier")
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.detector_identity) is None:
            raise ValueError("detector_identity must be a safe identifier")
        for value, field in (
            (self.pre_event_seconds, "pre_event_seconds"),
            (self.post_event_seconds, "post_event_seconds"),
            (self.candidate_gap_seconds, "candidate_gap_seconds"),
        ):
            if not math.isfinite(value) or not 0 <= value <= 300:
                raise ValueError(f"{field} must be between 0 and 300")
        if (
            type(self.candidate_start_confidence) not in {int, float}
            or not math.isfinite(float(self.candidate_start_confidence))
            or not 0.0 <= float(self.candidate_start_confidence) <= 1.0
        ):
            raise ValueError("candidate_start_confidence must be between 0 and 1")
        if not 1 <= self.max_event_span_seconds <= 300:
            raise ValueError("max_event_span_seconds must be between 1 and 300")
        if self.max_event_span_seconds + self.pre_event_seconds + self.post_event_seconds > 300:
            raise ValueError("candidate evidence window must be at most 300 seconds")
        if not 3 <= self.keyframe_count <= 8:
            raise ValueError("keyframe_count must be between 3 and 8")
        if not 1 <= self.max_pending_detections <= 10_000:
            raise ValueError("max_pending_detections must be between 1 and 10000")
        if not 1 <= self.max_segments_per_run <= 10_000:
            raise ValueError("max_segments_per_run must be between 1 and 10000")
        if not 0 <= self.idle_exit_seconds <= 86_400:
            raise ValueError("idle_exit_seconds must be between 0 and 86400")
        if not 1 <= self.max_runtime_seconds <= 604_800:
            raise ValueError("max_runtime_seconds must be between 1 and 604800")
        if not 0.05 <= self.poll_seconds <= 60:
            raise ValueError("poll_seconds must be between 0.05 and 60")
        if not 1 <= self.max_active_jobs <= 10_000:
            raise ValueError("max_active_jobs must be between 1 and 10000")
        if not 1 <= self.max_presence_samples_per_stream <= 128:
            raise ValueError("max_presence_samples_per_stream must be between 1 and 128")
        if not 1 <= self.presence_live_edge_segments <= 16:
            raise ValueError("presence_live_edge_segments must be between 1 and 16")


class CandidateWorkerStopReason(StrEnum):
    SEGMENT_LIMIT = "segment_limit"
    IDLE_TIMEOUT = "idle_timeout"
    RUNTIME_LIMIT = "runtime_limit"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class CandidateWorkerSummary:
    stop_reason: CandidateWorkerStopReason
    scanned_segments: int
    detections: int
    emitted_events: int
    suppressed_candidate_groups: int
    queued_events: int
    reused_events: int
    queue_full_events: int
    pending_detections: int
    duration_seconds: float
    state_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stop_reason": self.stop_reason.value,
            "scanned_segments": self.scanned_segments,
            "detections": self.detections,
            "emitted_events": self.emitted_events,
            "suppressed_candidate_groups": self.suppressed_candidate_groups,
            "queued_events": self.queued_events,
            "reused_events": self.reused_events,
            "queue_full_events": self.queue_full_events,
            "pending_detections": self.pending_detections,
            "duration_seconds": self.duration_seconds,
            "state_path": self.state_path,
        }


@dataclass(frozen=True, slots=True)
class ScanProgress:
    scanned_segments: int = 0
    detections: int = 0
    emitted_events: int = 0
    suppressed_candidate_groups: int = 0
    queued_events: int = 0
    reused_events: int = 0
    queue_full_events: int = 0
    pending_detections: int = 0


def _shadow_sample_due(
    path: Path,
    *,
    camera_id: str,
    observed_at: datetime,
    minimum_interval_seconds: float,
) -> bool:
    if minimum_interval_seconds <= 0 or not (path.exists() or path.is_symlink()):
        return True
    batch = StreamPresenceBatch.read_private(path)
    previous = tuple(
        sample.observed_at for sample in batch.samples if sample.camera_id == camera_id
    )
    if not previous:
        return True
    return (observed_at - max(previous)).total_seconds() >= minimum_interval_seconds


def _group_detections(
    detections: tuple[TimedDetection, ...],
    *,
    gap_seconds: float,
    max_span_seconds: float,
) -> tuple[tuple[TimedDetection, ...], ...]:
    if not detections:
        return ()
    groups: list[list[TimedDetection]] = [[detections[0]]]
    for detection in detections[1:]:
        group = groups[-1]
        gap = (detection.occurred_at - group[-1].occurred_at).total_seconds()
        span = (detection.occurred_at - group[0].occurred_at).total_seconds()
        if gap <= gap_seconds and span <= max_span_seconds:
            group.append(detection)
        else:
            groups.append([detection])
    return tuple(tuple(group) for group in groups)


def _event_id(scanner_identity: str, camera_id: str, group: tuple[TimedDetection, ...]) -> str:
    identity = {
        "version": 1,
        "scanner": scanner_identity,
        "camera_id": camera_id,
        "detections": [item.to_dict() for item in group],
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"candidate_{digest[:24]}"


class CandidateWorker:
    def __init__(self, settings: CandidateWorkerSettings):
        self.settings = settings
        self.queue: AnalysisQueue | None = None
        if settings.emit_candidate_events:
            self.queue = AnalysisQueue(settings.database_path)
            self.queue.initialize()
        self._health: CandidateWorkerHealth | None = None

    def _persist_health(self, health: CandidateWorkerHealth) -> None:
        if self.settings.health_path is None:
            return
        health.write_atomic_private(self.settings.health_path)
        self._health = health

    def _ensure_health(self, camera_id: str) -> CandidateWorkerHealth | None:
        if self.settings.health_path is None:
            return None
        if self._health is None:
            previous: CandidateWorkerHealth | None = None
            health_path = self.settings.health_path
            if health_path.exists() or health_path.is_symlink():
                previous = CandidateWorkerHealth.read_private(health_path)
                if (
                    previous.camera_id != camera_id
                    or previous.scanner_identity != self.settings.scanner_identity
                    or previous.detector_identity != self.settings.detector_identity
                ):
                    raise CandidateWorkerError("candidate health identity does not match manifest")
            now = datetime.now(UTC)
            self._persist_health(
                CandidateWorkerHealth(
                    version=4,
                    camera_id=camera_id,
                    scanner_identity=self.settings.scanner_identity,
                    detector_identity=self.settings.detector_identity,
                    checked_at=now,
                    state=CandidateHealthState.STARTING,
                    worker_running=True,
                    lifetime=CandidateHealthLifetime.begin_session(
                        previous,
                        updated_at=now,
                    ),
                )
            )
        elif self._health.camera_id != camera_id:
            raise CandidateWorkerError("candidate health identity does not match manifest")
        return self._health

    def _health_scan_started(self, camera_id: str) -> float:
        started = time.monotonic()
        health = self._ensure_health(camera_id)
        if health is not None:
            now = datetime.now(UTC)
            self._persist_health(
                replace(
                    health,
                    checked_at=now,
                    state=CandidateHealthState.SCANNING,
                    worker_running=True,
                    scans_started=health.scans_started + 1,
                    last_scan_started_at=now,
                    last_sampling=None,
                    last_error_code=None,
                )
            )
        return started

    def _detector_lock_wait(self) -> float | None:
        lock_wait = getattr(
            self.settings.detector,
            "last_execution_lock_wait_seconds",
            None,
        )
        if type(lock_wait) not in {int, float} or not math.isfinite(float(lock_wait)):
            return None
        return round(float(lock_wait), 6)

    def _health_scan_failed(self, *, started: float) -> None:
        health = self._health
        if health is None:
            return
        now = datetime.now(UTC)
        assert health.lifetime is not None
        lifetime = health.lifetime.record_scan(
            updated_at=now,
            succeeded=False,
            duration_seconds=round(time.monotonic() - started, 6),
            execution_lock_wait_seconds=self._detector_lock_wait(),
        ).finish_session(updated_at=now, failed=True)
        self._persist_health(
            replace(
                health,
                checked_at=now,
                state=CandidateHealthState.FAILED,
                worker_running=False,
                scans_failed=health.scans_failed + 1,
                last_error_code="candidate_scan_failed",
                last_sampling=None,
                lifetime=lifetime,
            )
        )

    def _health_scan_succeeded(
        self,
        *,
        started: float,
        detections: int,
        sampling: DetectorSamplingLedger | None = None,
    ) -> None:
        health = self._health
        if health is None:
            return
        lock_wait = self._detector_lock_wait()
        now = datetime.now(UTC)
        duration = round(time.monotonic() - started, 6)
        assert health.lifetime is not None
        self._persist_health(
            replace(
                health,
                checked_at=now,
                state=CandidateHealthState.IDLE,
                worker_running=True,
                scans_succeeded=health.scans_succeeded + 1,
                segments_scanned=health.segments_scanned + 1,
                detections=health.detections + detections,
                last_scan_completed_at=now,
                last_scan_duration_seconds=duration,
                last_execution_lock_wait_seconds=lock_wait,
                last_sampling=CandidateSamplingHealth.from_ledger(sampling) if sampling else None,
                lifetime=health.lifetime.record_scan(
                    updated_at=now,
                    succeeded=True,
                    duration_seconds=duration,
                    execution_lock_wait_seconds=lock_wait,
                ),
                last_error_code=None,
            )
        )

    def _health_event_progress(self, progress: ScanProgress) -> None:
        health = self._health
        if health is None or not (
            progress.emitted_events
            or progress.suppressed_candidate_groups
            or progress.queue_full_events
        ):
            return
        now = datetime.now(UTC)
        assert health.lifetime is not None
        self._persist_health(
            replace(
                health,
                checked_at=now,
                emitted_events=health.emitted_events + progress.emitted_events,
                suppressed_candidate_groups=(
                    health.suppressed_candidate_groups + progress.suppressed_candidate_groups
                ),
                queue_full_events=(health.queue_full_events + progress.queue_full_events),
                lifetime=health.lifetime.record_candidate_groups(
                    updated_at=now,
                    emitted=progress.emitted_events,
                    suppressed=progress.suppressed_candidate_groups,
                ),
            )
        )

    def _health_stopped(self) -> None:
        health = self._health
        if health is None or health.state is CandidateHealthState.FAILED:
            return
        now = datetime.now(UTC)
        assert health.lifetime is not None
        self._persist_health(
            replace(
                health,
                checked_at=now,
                state=CandidateHealthState.STOPPED,
                worker_running=False,
                last_error_code=None,
                lifetime=health.lifetime.finish_session(
                    updated_at=now,
                    failed=False,
                ),
            )
        )

    def _initial_state(self, manifest: SegmentManifest) -> CandidateScanState:
        return CandidateScanState(
            camera_id=manifest.camera_id,
            source_ref=manifest.source_ref,
            scanner_identity=self.settings.scanner_identity,
            updated_at=datetime.now(UTC),
        )

    def _load_state(self, manifest: SegmentManifest) -> CandidateScanState:
        if not self.settings.state_path.exists():
            return self._initial_state(manifest)
        state = CandidateScanState.read(self.settings.state_path)
        if (
            state.camera_id != manifest.camera_id
            or state.source_ref != manifest.source_ref
            or state.scanner_identity != self.settings.scanner_identity
        ):
            raise CandidateWorkerError("candidate scan state identity does not match manifest")
        if not self.settings.emit_candidate_events and state.pending_detections:
            raise CandidateWorkerError(
                "presence-only candidate state cannot contain pending events"
            )
        current = {
            item.filename: SegmentFingerprint.from_segment(item) for item in manifest.segments
        }
        for fingerprint in state.processed_segments:
            value = current.get(fingerprint.filename)
            if value is not None and value != fingerprint:
                raise CandidateWorkerError("processed segment fingerprint changed")
        pending_origins = {item.segment_filename for item in state.pending_detections}
        if any(filename not in current for filename in pending_origins):
            raise CandidateWorkerError("pending detection segment disappeared from the manifest")
        retained = tuple(
            item
            for item in state.processed_segments
            if item.filename in current or item.filename in pending_origins
        )
        if retained == state.processed_segments:
            return state
        pruned = CandidateScanState(
            camera_id=state.camera_id,
            source_ref=state.source_ref,
            scanner_identity=state.scanner_identity,
            updated_at=datetime.now(UTC),
            watermark_at=state.watermark_at,
            processed_segments=retained,
            pending_detections=state.pending_detections,
        )
        pruned.write_atomic(self.settings.state_path)
        return pruned

    def _segment_path(self, segment: BufferSegment) -> Path:
        try:
            root = self.settings.manifest_path.expanduser().resolve().parent
            path = root / segment.filename
            resolved = path.resolve()
            unsafe = (
                path.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file()
            )
            size = resolved.stat().st_size
        except OSError as exc:
            raise CandidateWorkerError("manifest references an unreadable segment") from exc
        if unsafe:
            raise CandidateWorkerError("manifest references an unsafe or missing segment")
        if size != segment.size_bytes:
            raise CandidateWorkerError("segment size does not match manifest")
        return resolved

    def _scan_segment(
        self,
        manifest: SegmentManifest,
        segment: BufferSegment,
    ) -> tuple[TimedDetection, ...]:
        scan_started = self._health_scan_started(manifest.camera_id)
        try:
            path = self._segment_path(segment)
            if not math.isclose(
                (segment.ended_at - segment.started_at).total_seconds(),
                segment.duration_seconds,
                rel_tol=0,
                abs_tol=0.000001,
            ):
                raise CandidateWorkerError("segment duration does not match its time boundaries")
            source = CameraSource(
                camera_id=manifest.camera_id,
                kind="file",
                read_uri=str(path),
                audit_ref=str(path),
            )
            scan_result = getattr(self.settings.detector, "scan_result", None)
            sampling = None
            if callable(scan_result):
                result = scan_result(source)
                if (
                    not isinstance(result, DetectorScanResult)
                    or result.camera_id != manifest.camera_id
                ):
                    raise CandidateWorkerError("detector sampling result does not match the stream")
                detections = result.detections
                sampling = result.sampling
                if sampling is not None and any(
                    sample.timestamp_seconds > segment.duration_seconds + 0.25
                    for sample in sampling.samples
                ):
                    raise CandidateWorkerError(
                        "detector sampling timestamp exceeds the segment duration"
                    )
            else:
                detections = tuple(self.settings.detector.scan(source))
            timed: list[TimedDetection] = []
            for detection in detections:
                if (
                    not math.isfinite(detection.timestamp_seconds)
                    or detection.timestamp_seconds < 0
                    or detection.timestamp_seconds > segment.duration_seconds + 0.25
                ):
                    raise CandidateWorkerError("detector timestamp exceeds the segment duration")
                timed.append(
                    TimedDetection(
                        occurred_at=segment.started_at
                        + timedelta(seconds=detection.timestamp_seconds),
                        segment_filename=segment.filename,
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox_xyxy=detection.bbox_xyxy,
                    )
                )
            presence_observations = (
                project_presence_sampling(
                    detections=detections,
                    ledger=sampling,
                    duration_seconds=segment.duration_seconds,
                )
                if self.settings.presence_path is not None
                else ()
            )
            if any(
                segment.started_at + timedelta(seconds=item.offset_seconds) > segment.ended_at
                for item in presence_observations
                if item.state is not StreamPresenceSampleState.COVERAGE_GAP
            ):
                raise CandidateWorkerError("presence observation exceeds the segment boundary")
            if self.settings.presence_store is not None:
                SamplingEvidenceStore(self.settings.presence_store.path).record(
                    camera_id=manifest.camera_id,
                    scanner_id=self.settings.scanner_identity,
                    source_fingerprint=self._segment_source_fingerprint(manifest, segment),
                    started_at=segment.started_at,
                    ended_at=segment.ended_at,
                    ledger=sampling,
                )
            self._record_presence_sample(
                manifest=manifest,
                segment=segment,
                observations=presence_observations,
            )
            self._record_shadow_presence_sample(
                manifest=manifest,
                segment=segment,
                source=source,
            )
        except Exception:
            # Failure reporting must not mask the original detector/storage
            # error or prevent the independent best-effort presence gap.
            with suppress(Exception):
                self._health_scan_failed(started=scan_started)
            # The same storage fault may prevent this best-effort gap.
            # Keep the original scan failure and failed health state.
            with suppress(Exception):
                self._record_presence_sample(
                    manifest=manifest,
                    segment=segment,
                    observations=(
                        PresenceSamplingObservation(
                            StreamPresenceSampleState.COVERAGE_GAP,
                            segment.duration_seconds,
                            "candidate_scan_failed",
                        ),
                    ),
                )
            raise
        self._health_scan_succeeded(started=scan_started, detections=len(timed), sampling=sampling)
        return tuple(timed)

    @staticmethod
    def _segment_source_fingerprint(manifest: SegmentManifest, segment: BufferSegment) -> str:
        # Metadata identity only; do not present this as a media-byte digest.
        source_identity = {
            "version": 1,
            "camera_id": manifest.camera_id,
            "segment": SegmentFingerprint.from_segment(segment).to_dict(),
        }
        return hashlib.sha256(
            json.dumps(source_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _record_presence_sample(
        self,
        *,
        manifest: SegmentManifest,
        segment: BufferSegment,
        observations: tuple[PresenceSamplingObservation, ...],
    ) -> None:
        if self.settings.presence_path is None:
            return
        source_fingerprint = self._segment_source_fingerprint(manifest, segment)
        sample_identity = {
            "version": 1,
            "scanner_id": self.settings.scanner_identity,
            "detector_id": self.settings.detector_identity,
            "source_fingerprint": source_fingerprint,
        }
        sample_digest = hashlib.sha256(
            json.dumps(
                sample_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        # Persist a gap first: interruption between the two writes must not
        # expose a new positive fact while hiding its known coverage defect.
        for observation in reversed(observations):
            is_gap = observation.state is StreamPresenceSampleState.COVERAGE_GAP
            sample = StreamPresenceSample(
                sample_id=(
                    f"presence_gap_{sample_digest[:24]}_{observation.reason_code}"
                    if is_gap
                    else f"presence_{sample_digest[:24]}"
                ),
                camera_id=manifest.camera_id,
                observed_at=(
                    segment.ended_at
                    if is_gap
                    else segment.started_at + timedelta(seconds=observation.offset_seconds)
                ),
                state=observation.state,
                scanner_id=self.settings.scanner_identity,
                detector_id=None if is_gap else self.settings.detector_identity,
                source_fingerprint=source_fingerprint,
                reason_code=observation.reason_code,
            )
            StreamPresenceBatch.append_private(
                self.settings.presence_path,
                sample,
                max_samples_per_stream=self.settings.max_presence_samples_per_stream,
            )
            if self.settings.presence_store is not None:
                self.settings.presence_store.record(sample)

    def _record_shadow_presence_sample(
        self,
        *,
        manifest: SegmentManifest,
        segment: BufferSegment,
        source: CameraSource,
    ) -> None:
        classifier = self.settings.shadow_presence_classifier
        path = self.settings.shadow_presence_path
        store = self.settings.shadow_presence_store
        if classifier is None or path is None or store is None:
            return
        if not _shadow_sample_due(
            path,
            camera_id=manifest.camera_id,
            observed_at=segment.ended_at,
            minimum_interval_seconds=self.settings.shadow_min_interval_seconds,
        ):
            return
        source_identity = {
            "version": 1,
            "camera_id": manifest.camera_id,
            "segment": SegmentFingerprint.from_segment(segment).to_dict(),
        }
        source_fingerprint = hashlib.sha256(
            json.dumps(
                source_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        reason_code: str | None = None
        detector_id: str | None = classifier.detector_identity
        try:
            decision = classifier.classify(
                source,
                duration_seconds=segment.duration_seconds,
            )
            state = decision.state
        except PosePresenceShadowError:
            state = StreamPresenceSampleState.COVERAGE_GAP
            detector_id = None
            reason_code = "pose_shadow_failed"
        sample_identity = {
            "version": 1,
            "scanner_id": classifier.scanner_identity,
            "detector_id": detector_id,
            "source_fingerprint": source_fingerprint,
            "state": state.value,
            "reason_code": reason_code,
        }
        sample_digest = hashlib.sha256(
            json.dumps(
                sample_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        sample = StreamPresenceSample(
            sample_id=f"presence_shadow_{sample_digest[:24]}",
            camera_id=manifest.camera_id,
            observed_at=segment.ended_at,
            state=state,
            scanner_id=classifier.scanner_identity,
            detector_id=detector_id,
            source_fingerprint=source_fingerprint,
            reason_code=reason_code,
        )
        StreamPresenceBatch.append_private(
            path,
            sample,
            max_samples_per_stream=self.settings.max_presence_samples_per_stream,
        )
        store.record(sample, mode=PresenceFactMode.SHADOW)

    def _candidate(
        self,
        camera_id: str,
        group: tuple[TimedDetection, ...],
    ) -> tuple[CandidateEvent, datetime, float]:
        first = group[0].occurred_at
        span = (group[-1].occurred_at - first).total_seconds()
        pre = self.settings.pre_event_seconds
        detections = tuple(
            Detection(
                timestamp_seconds=pre + (item.occurred_at - first).total_seconds(),
                label=item.label,
                confidence=item.confidence,
                bbox_xyxy=item.bbox_xyxy,
            )
            for item in group
        )
        candidate = CandidateEvent(
            event_id=_event_id(
                self.settings.scanner_identity,
                camera_id,
                group,
            ),
            camera_id=camera_id,
            start_seconds=pre,
            end_seconds=pre + span,
            detections=detections,
            detector_id=self.settings.detector_identity,
        )
        return candidate, first, span + self.settings.post_event_seconds

    def _extract(
        self,
        candidate: CandidateEvent,
        occurred_at: datetime,
        post_seconds: float,
    ) -> BufferedEvidenceResult:
        return extract_buffered_evidence(
            manifest_path=self.settings.manifest_path,
            event_time=occurred_at,
            event_id=candidate.event_id,
            pre_event_seconds=self.settings.pre_event_seconds,
            post_event_seconds=post_seconds,
            keyframe_count=self.settings.keyframe_count,
            storage=self.settings.storage,
            ffmpeg=self.settings.ffmpeg,
            ffprobe=self.settings.ffprobe,
        )

    def _emit_ready(
        self,
        manifest: SegmentManifest,
        state: CandidateScanState,
        *,
        flush_pending: bool = False,
    ) -> tuple[CandidateScanState, ScanProgress]:
        if not self.settings.emit_candidate_events:
            if state.pending_detections:
                raise CandidateWorkerError(
                    "presence-only candidate state cannot contain pending events"
                )
            return state, ScanProgress()
        if state.watermark_at is None or not state.pending_detections:
            return state, ScanProgress(pending_detections=len(state.pending_detections))
        emitted = suppressed = queued = reused = queue_full = 0
        current = state
        while current.pending_detections:
            groups = _group_detections(
                current.pending_detections,
                gap_seconds=self.settings.candidate_gap_seconds,
                max_span_seconds=self.settings.max_event_span_seconds,
            )
            group = groups[0]
            ready_at = group[-1].occurred_at + timedelta(
                seconds=max(
                    self.settings.candidate_gap_seconds,
                    self.settings.post_event_seconds,
                )
            )
            if current.watermark_at < ready_at and not flush_pending:
                break
            if not any(
                item.confidence >= self.settings.candidate_start_confidence for item in group
            ):
                current = CandidateScanState(
                    camera_id=current.camera_id,
                    source_ref=current.source_ref,
                    scanner_identity=current.scanner_identity,
                    updated_at=datetime.now(UTC),
                    watermark_at=current.watermark_at,
                    processed_segments=current.processed_segments,
                    pending_detections=current.pending_detections[len(group) :],
                )
                current.write_atomic(self.settings.state_path)
                suppressed += 1
                continue
            candidate, occurred_at, post_seconds = self._candidate(
                manifest.camera_id,
                group,
            )
            evidence = self._extract(candidate, occurred_at, post_seconds)
            absolute_media = EventMedia(
                clip_path=str(Path(evidence.media.clip_path).resolve()),
                frame_paths=tuple(str(Path(path).resolve()) for path in evidence.media.frame_paths),
                coverage_gaps=evidence.media.coverage_gaps,
                selected_segments=evidence.media.selected_segments,
                version=evidence.media.version,
            )
            storage_root = self.settings.storage.active_media_root().resolve()
            assert self.queue is not None
            try:
                result = self.queue.enqueue(
                    candidate=candidate,
                    occurred_at=occurred_at,
                    source_kind="rtsp",
                    source_ref=manifest.source_ref,
                    media=absolute_media,
                    storage_root=storage_root,
                    max_active=self.settings.max_active_jobs,
                )
            except QueueFullError:
                queue_full += 1
            else:
                queued += int(not result.reused_existing)
                reused += int(result.reused_existing)
            emitted += 1
            current = CandidateScanState(
                camera_id=current.camera_id,
                source_ref=current.source_ref,
                scanner_identity=current.scanner_identity,
                updated_at=datetime.now(UTC),
                watermark_at=current.watermark_at,
                processed_segments=current.processed_segments,
                pending_detections=current.pending_detections[len(group) :],
            )
            current.write_atomic(self.settings.state_path)
        return current, ScanProgress(
            emitted_events=emitted,
            suppressed_candidate_groups=suppressed,
            queued_events=queued,
            reused_events=reused,
            queue_full_events=queue_full,
            pending_detections=len(current.pending_detections),
        )

    def _source_is_terminal(self, manifest: SegmentManifest) -> bool:
        health_path = self.settings.manifest_path.with_name("health.json")
        if not health_path.is_file():
            return False
        try:
            health = MediaSourceHealth.read(health_path)
        except MediaHealthError:
            return False
        return (
            health.camera_id == manifest.camera_id
            and health.source_ref == manifest.source_ref
            and health.state.terminal
        )

    def scan_once(self, *, segment_limit: int) -> ScanProgress:
        if segment_limit < 1:
            raise ValueError("segment_limit must be positive")
        if not self.settings.manifest_path.exists():
            return ScanProgress()
        manifest = SegmentManifest.read(self.settings.manifest_path)
        state = self._load_state(manifest)
        processed_names = {item.filename for item in state.processed_segments}
        available = sorted(
            (item for item in manifest.segments if item.filename not in processed_names),
            key=lambda item: (item.started_at, item.filename),
        )
        if not self.settings.emit_candidate_events:
            if state.watermark_at is not None:
                available = [item for item in available if item.ended_at > state.watermark_at]
            if len(available) > self.settings.presence_live_edge_segments:
                available = available[-self.settings.presence_live_edge_segments :]
                live_edge_floor = available[0].started_at
                state = CandidateScanState(
                    camera_id=state.camera_id,
                    source_ref=state.source_ref,
                    scanner_identity=state.scanner_identity,
                    updated_at=datetime.now(UTC),
                    watermark_at=(
                        max(state.watermark_at, live_edge_floor)
                        if state.watermark_at is not None
                        else live_edge_floor
                    ),
                    processed_segments=state.processed_segments,
                    pending_detections=(),
                )
                state.write_atomic(self.settings.state_path)
        scanned = detections = emitted = suppressed = queued = reused = queue_full = 0
        for segment in available[:segment_limit]:
            if state.watermark_at is not None and segment.ended_at <= state.watermark_at:
                raise CandidateWorkerError("late segment appeared behind the scan watermark")
            if (
                state.watermark_at is not None
                and segment.started_at < state.watermark_at - timedelta(seconds=0.25)
            ):
                raise CandidateWorkerError("segment overlaps the scan watermark")
            timed = self._scan_segment(manifest, segment)
            pending = (
                tuple(
                    sorted(
                        (*state.pending_detections, *timed),
                        key=lambda item: item.occurred_at,
                    )
                )
                if self.settings.emit_candidate_events
                else ()
            )
            if len(pending) > self.settings.max_pending_detections:
                raise CandidateWorkerError("candidate pending detection limit exceeded")
            state = CandidateScanState(
                camera_id=state.camera_id,
                source_ref=state.source_ref,
                scanner_identity=state.scanner_identity,
                updated_at=datetime.now(UTC),
                watermark_at=(
                    max(state.watermark_at, segment.ended_at)
                    if state.watermark_at is not None
                    else segment.ended_at
                ),
                processed_segments=(
                    *state.processed_segments,
                    SegmentFingerprint.from_segment(segment),
                ),
                pending_detections=pending,
            )
            state.write_atomic(self.settings.state_path)
            state, progress = self._emit_ready(manifest, state)
            self._health_event_progress(progress)
            scanned += 1
            detections += len(timed)
            emitted += progress.emitted_events
            suppressed += progress.suppressed_candidate_groups
            queued += progress.queued_events
            reused += progress.reused_events
            queue_full += progress.queue_full_events
        if len(available) <= segment_limit:
            state, progress = self._emit_ready(
                manifest,
                state,
                flush_pending=self._source_is_terminal(manifest),
            )
            self._health_event_progress(progress)
            emitted += progress.emitted_events
            suppressed += progress.suppressed_candidate_groups
            queued += progress.queued_events
            reused += progress.reused_events
            queue_full += progress.queue_full_events
        return ScanProgress(
            scanned_segments=scanned,
            detections=detections,
            emitted_events=emitted,
            suppressed_candidate_groups=suppressed,
            queued_events=queued,
            reused_events=reused,
            queue_full_events=queue_full,
            pending_detections=len(state.pending_detections),
        )

    def run(self) -> CandidateWorkerSummary:
        started = time.monotonic()
        last_progress = started
        scanned = detections = emitted = suppressed = queued = reused = queue_full = 0
        pending = 0
        reason = CandidateWorkerStopReason.RUNTIME_LIMIT
        try:
            if self.settings.health_path is not None and self.settings.manifest_path.is_file():
                self._ensure_health(SegmentManifest.read(self.settings.manifest_path).camera_id)
            while True:
                remaining = self.settings.max_segments_per_run - scanned
                if remaining <= 0:
                    reason = CandidateWorkerStopReason.SEGMENT_LIMIT
                    break
                progress = self.scan_once(segment_limit=remaining)
                scanned += progress.scanned_segments
                detections += progress.detections
                emitted += progress.emitted_events
                suppressed += progress.suppressed_candidate_groups
                queued += progress.queued_events
                reused += progress.reused_events
                queue_full += progress.queue_full_events
                pending = progress.pending_detections
                now = time.monotonic()
                if (
                    progress.scanned_segments
                    or progress.emitted_events
                    or progress.suppressed_candidate_groups
                ):
                    last_progress = now
                if scanned >= self.settings.max_segments_per_run:
                    reason = CandidateWorkerStopReason.SEGMENT_LIMIT
                    break
                if now - started >= self.settings.max_runtime_seconds:
                    reason = CandidateWorkerStopReason.RUNTIME_LIMIT
                    break
                if (
                    self.settings.idle_exit_seconds > 0
                    and now - last_progress >= self.settings.idle_exit_seconds
                ):
                    reason = CandidateWorkerStopReason.IDLE_TIMEOUT
                    break
                time.sleep(
                    min(
                        self.settings.poll_seconds,
                        max(0.0, self.settings.max_runtime_seconds - (now - started)),
                    )
                )
        except KeyboardInterrupt:
            reason = CandidateWorkerStopReason.INTERRUPTED
        except Exception:
            health = self._health
            if health is not None and health.state is not CandidateHealthState.FAILED:
                now = datetime.now(UTC)
                assert health.lifetime is not None
                self._persist_health(
                    replace(
                        health,
                        checked_at=now,
                        state=CandidateHealthState.FAILED,
                        worker_running=False,
                        last_error_code="candidate_worker_failed",
                        lifetime=health.lifetime.finish_session(
                            updated_at=now,
                            failed=True,
                        ),
                    )
                )
            raise
        self._health_stopped()
        return CandidateWorkerSummary(
            stop_reason=reason,
            scanned_segments=scanned,
            detections=detections,
            emitted_events=emitted,
            suppressed_candidate_groups=suppressed,
            queued_events=queued,
            reused_events=reused,
            queue_full_events=queue_full,
            pending_detections=pending,
            duration_seconds=round(time.monotonic() - started, 3),
            state_path=str(self.settings.state_path),
        )
