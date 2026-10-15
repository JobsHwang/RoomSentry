from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .domain import Detection, SignalEvent
from .object_tracker import (
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
    IoUObjectTracker,
    IoUTrackerConfig,
)
from .temporal_zone import (
    TemporalZoneAssembler,
    TemporalZoneConfig,
    TrackedObjectObservation,
)


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_detection_frame_batch(
    detections: tuple[Detection, ...],
    *,
    camera_id: str,
    detector_adapter: str,
    source_started_at: datetime,
) -> DetectionFrameBatch:
    """Convert bounded-file detector facts into the path-free tracker contract."""

    if not detections:
        raise ValueError("at least one detection is required for local tracking")
    if any(not isinstance(item, Detection) for item in detections):
        raise ValueError("local tracking accepts Detection values only")
    if (
        not isinstance(source_started_at, datetime)
        or source_started_at.tzinfo is None
        or source_started_at.utcoffset() is None
    ):
        raise ValueError("source_started_at must be timezone-aware")
    base_time = source_started_at.astimezone(UTC)

    grouped: dict[datetime, list[Detection]] = defaultdict(list)
    for detection in detections:
        occurred_at = base_time + timedelta(seconds=detection.timestamp_seconds)
        grouped[occurred_at].append(detection)

    frames: list[DetectionFrame] = []
    for occurred_at in sorted(grouped):
        frame_digest = _digest(
            {
                "version": 1,
                "camera_id": camera_id,
                "detector_adapter": detector_adapter,
                "occurred_at": occurred_at.isoformat(),
            }
        )
        frame_id = f"frame_{frame_digest[:24]}"
        ordered = sorted(
            grouped[occurred_at],
            key=lambda item: (
                item.label,
                item.bbox_xyxy,
                item.confidence,
                item.timestamp_seconds,
            ),
        )
        frame_detections: list[FrameDetection] = []
        for index, detection in enumerate(ordered):
            detection_digest = _digest(
                {
                    "version": 1,
                    "frame_id": frame_id,
                    "ordinal": index,
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "bbox_xyxy": list(detection.bbox_xyxy),
                }
            )
            frame_detections.append(
                FrameDetection(
                    detection_id=f"detection_{detection_digest[:24]}",
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox_xyxy=detection.bbox_xyxy,
                )
            )
        frames.append(
            DetectionFrame(
                frame_id=frame_id,
                occurred_at=occurred_at,
                detections=tuple(frame_detections),
            )
        )

    return DetectionFrameBatch(
        camera_id=camera_id,
        detector_adapter=detector_adapter,
        frames=tuple(frames),
    )


@dataclass(frozen=True, slots=True)
class LocalPerceptionResult:
    detection_frames: DetectionFrameBatch
    observations: tuple[TrackedObjectObservation, ...]
    signals: tuple[SignalEvent, ...]
    active_track_count: int


def run_local_perception(
    detections: tuple[Detection, ...],
    *,
    camera_id: str,
    detector_adapter: str,
    source_started_at: datetime,
    tracker_config: IoUTrackerConfig,
    zone_config: TemporalZoneConfig,
) -> LocalPerceptionResult:
    """Run the bounded one-file detection -> tracker -> zone bridge.

    This helper intentionally owns no service lifecycle and performs no identity
    inference. Durable streaming workers continue to use the existing tracker
    and temporal-zone state/outbox commands.
    """

    batch = build_detection_frame_batch(
        detections,
        camera_id=camera_id,
        detector_adapter=detector_adapter,
        source_started_at=source_started_at,
    )
    tracker = IoUObjectTracker(
        config=tracker_config,
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
    )
    observations: list[TrackedObjectObservation] = []
    for frame in batch.frames:
        observations.extend(tracker.process(frame))

    zone_assembler = TemporalZoneAssembler(config=zone_config)
    signals = zone_assembler.process_all(tuple(observations))
    return LocalPerceptionResult(
        detection_frames=batch,
        observations=tuple(observations),
        signals=signals,
        active_track_count=len(tracker.state.active_tracks),
    )
