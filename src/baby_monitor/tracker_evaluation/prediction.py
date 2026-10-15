"""TrackerPredictionBatch v1 and adapter exports over one fixed batch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baby_monitor.tracker_evaluation._shared import (
    MAX_TRACKER_EVAL_ASSIGNMENTS,
    TrackerEvaluationError,
    _canonical_sha256,
    _detection_keys,
    _exact_object,
    _read_json,
    _safe_id,
    _sha256,
    _write_new_json,
)

from ..motion_tracker import (
    MOTION_TRACKER_ADAPTER_ID,
    MOTION_TRACKER_REVISION,
    MotionTrackerConfig,
    build_motion_track_assignments,
)
from ..object_tracker import (
    TRACKER_ADAPTER_ID,
    DetectionFrameBatch,
    IoUObjectTracker,
    IoUTrackerConfig,
)


@dataclass(frozen=True, slots=True)
class TrackerPredictionAssignment:
    frame_id: str
    detection_id: str
    tracking_id: str

    def __post_init__(self) -> None:
        _safe_id(self.frame_id, "tracker prediction frame_id")
        _safe_id(self.detection_id, "tracker prediction detection_id")
        _safe_id(self.tracking_id, "tracker prediction tracking_id")

    @classmethod
    def from_dict(cls, value: Any) -> TrackerPredictionAssignment:
        data = _exact_object(
            value,
            {"frame_id", "detection_id", "tracking_id"},
            "tracker prediction assignment",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, str]:
        return {
            "frame_id": self.frame_id,
            "detection_id": self.detection_id,
            "tracking_id": self.tracking_id,
        }


@dataclass(frozen=True, slots=True)
class TrackerPredictionBatch:
    camera_id: str
    detector_adapter: str
    detection_batch_sha256: str
    tracker_adapter: str
    tracker_revision: str
    tracker_config_sha256: str
    runtime_id: str
    assignments: tuple[TrackerPredictionAssignment, ...]
    subject_identity_inference: bool = False
    paths_disclosed: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TrackerEvaluationError("unsupported tracker prediction version")
        for field in (
            "camera_id",
            "detector_adapter",
            "tracker_adapter",
            "tracker_revision",
            "runtime_id",
        ):
            _safe_id(getattr(self, field), f"tracker prediction {field}")
        _sha256(
            self.detection_batch_sha256,
            "tracker prediction detection batch sha256",
        )
        _sha256(self.tracker_config_sha256, "tracker prediction config sha256")
        if (
            not isinstance(self.assignments, tuple)
            or len(self.assignments) > MAX_TRACKER_EVAL_ASSIGNMENTS
            or any(not isinstance(item, TrackerPredictionAssignment) for item in self.assignments)
        ):
            raise TrackerEvaluationError("tracker prediction assignments are invalid")
        keys = tuple((item.frame_id, item.detection_id) for item in self.assignments)
        if len(set(keys)) != len(keys):
            raise TrackerEvaluationError("tracker prediction assignments are duplicated")
        frame_tracks = tuple((item.frame_id, item.tracking_id) for item in self.assignments)
        if len(set(frame_tracks)) != len(frame_tracks):
            raise TrackerEvaluationError("tracker prediction reuses a track within one frame")
        if self.subject_identity_inference is not False or self.paths_disclosed is not False:
            raise TrackerEvaluationError(
                "tracker predictions cannot infer identity or disclose paths"
            )

    @classmethod
    def from_dict(cls, value: Any) -> TrackerPredictionBatch:
        data = _exact_object(
            value,
            {
                "version",
                "camera_id",
                "detector_adapter",
                "detection_batch_sha256",
                "tracker_adapter",
                "tracker_revision",
                "tracker_config_sha256",
                "runtime_id",
                "assignments",
                "subject_identity_inference",
                "paths_disclosed",
            },
            "tracker prediction batch",
        )
        assignments = data["assignments"]
        if not isinstance(assignments, list):
            raise TrackerEvaluationError("tracker prediction assignments must be an array")
        return cls(
            **{
                **data,
                "assignments": tuple(
                    TrackerPredictionAssignment.from_dict(item) for item in assignments
                ),
            }
        )

    @classmethod
    def read(cls, path: str | Path) -> TrackerPredictionBatch:
        return cls.from_dict(_read_json(path, "tracker prediction batch"))

    def bind(self, batch: DetectionFrameBatch) -> None:
        if (
            self.camera_id != batch.camera_id
            or self.detector_adapter != batch.detector_adapter
            or self.detection_batch_sha256 != batch.fingerprint
            or tuple((item.frame_id, item.detection_id) for item in self.assignments)
            != _detection_keys(batch)
        ):
            raise TrackerEvaluationError(
                "tracker predictions belong to a different detection batch"
            )
        labels = {
            (frame.frame_id, detection.detection_id): detection.label
            for frame in batch.frames
            for detection in frame.detections
        }
        track_labels: dict[str, str] = {}
        for item in self.assignments:
            label = labels[(item.frame_id, item.detection_id)]
            previous = track_labels.setdefault(item.tracking_id, label)
            if previous != label:
                raise TrackerEvaluationError("tracker prediction changes a track label")

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "detector_adapter": self.detector_adapter,
            "detection_batch_sha256": self.detection_batch_sha256,
            "tracker_adapter": self.tracker_adapter,
            "tracker_revision": self.tracker_revision,
            "tracker_config_sha256": self.tracker_config_sha256,
            "runtime_id": self.runtime_id,
            "assignments": [item.to_dict() for item in self.assignments],
            "subject_identity_inference": self.subject_identity_inference,
            "paths_disclosed": self.paths_disclosed,
        }

    def write_new(self, path: str | Path) -> None:
        _write_new_json(path, self.to_dict(), "tracker prediction batch")

    def to_public_dict(self) -> dict[str, object]:
        return {
            "valid": True,
            "version": self.version,
            "detection_batch_sha256": self.detection_batch_sha256,
            "prediction_sha256": self.fingerprint,
            "tracker_adapter": self.tracker_adapter,
            "tracker_revision": self.tracker_revision,
            "tracker_config_sha256": self.tracker_config_sha256,
            "runtime_id": self.runtime_id,
            "assignments": len(self.assignments),
            "predicted_tracks": len({item.tracking_id for item in self.assignments}),
            "subject_identity_inference": False,
            "paths_disclosed": False,
        }


def build_iou_tracker_predictions(
    batch: DetectionFrameBatch,
    config: IoUTrackerConfig,
) -> TrackerPredictionBatch:
    if not isinstance(batch, DetectionFrameBatch) or not isinstance(config, IoUTrackerConfig):
        raise TrackerEvaluationError("IoU tracker evaluation inputs are invalid")
    tracker = IoUObjectTracker(
        config=config,
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
    )
    assignments: list[TrackerPredictionAssignment] = []
    for frame in batch.frames:
        observations = tracker.process(frame)
        if len(observations) != len(frame.detections):
            raise TrackerEvaluationError("IoU tracker did not return one observation per detection")
        for detection, observation in zip(
            frame.detections,
            observations,
            strict=True,
        ):
            if (
                observation.occurred_at != frame.occurred_at
                or observation.label != detection.label
                or observation.confidence != detection.confidence
                or observation.bbox_xyxy != detection.bbox_xyxy
            ):
                raise TrackerEvaluationError(
                    "IoU tracker observation order does not match detections"
                )
            assignments.append(
                TrackerPredictionAssignment(
                    frame_id=frame.frame_id,
                    detection_id=detection.detection_id,
                    tracking_id=observation.tracking_id,
                )
            )
        if observations:
            tracker.state = tracker.state.acknowledge(
                tuple(item.observation_id for item in observations)
            )
    result = TrackerPredictionBatch(
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
        detection_batch_sha256=batch.fingerprint,
        tracker_adapter=TRACKER_ADAPTER_ID,
        tracker_revision="greedy_iou_v1",
        tracker_config_sha256=config.fingerprint,
        runtime_id="python_stdlib",
        assignments=tuple(assignments),
    )
    result.bind(batch)
    return result


def build_motion_tracker_predictions(
    batch: DetectionFrameBatch,
    config: MotionTrackerConfig,
) -> TrackerPredictionBatch:
    assignments = build_motion_track_assignments(batch, config)
    result = TrackerPredictionBatch(
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
        detection_batch_sha256=batch.fingerprint,
        tracker_adapter=MOTION_TRACKER_ADAPTER_ID,
        tracker_revision=MOTION_TRACKER_REVISION,
        tracker_config_sha256=config.fingerprint,
        runtime_id="python_stdlib",
        assignments=tuple(
            TrackerPredictionAssignment(
                frame_id=item.frame_id,
                detection_id=item.detection_id,
                tracking_id=item.tracking_id,
            )
            for item in assignments
        ),
    )
    result.bind(batch)
    return result


def _prediction_links(
    batch: DetectionFrameBatch,
    prediction: TrackerPredictionBatch,
) -> set[tuple[str, str, str, str]]:
    frame_indices = {frame.frame_id: index for index, frame in enumerate(batch.frames)}
    by_track: dict[str, list[TrackerPredictionAssignment]] = {}
    for item in prediction.assignments:
        by_track.setdefault(item.tracking_id, []).append(item)
    links: set[tuple[str, str, str, str]] = set()
    for assignments in by_track.values():
        assignments.sort(key=lambda item: frame_indices[item.frame_id])
        for previous, current in zip(assignments, assignments[1:], strict=False):
            link = (
                previous.frame_id,
                previous.detection_id,
                current.frame_id,
                current.detection_id,
            )
            if link in links:
                raise TrackerEvaluationError(
                    "tracker prediction contains a duplicate association link"
                )
            links.add(link)
    return links
