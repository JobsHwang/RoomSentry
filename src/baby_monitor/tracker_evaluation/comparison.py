"""TrackerPredictionComparison v1: ground-truth-free link-set differences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baby_monitor.tracker_evaluation._shared import (
    _SAFE_ID,
    MAX_TRACKER_DISAGREEMENT_FRAMES,
    MAX_TRACKER_EVAL_ASSIGNMENTS,
    TrackerEvaluationError,
    _canonical_sha256,
    _exact_object,
    _integer,
    _read_json,
    _safe_id,
    _sha256,
    _write_new_json,
)
from baby_monitor.tracker_evaluation.prediction import (
    TrackerPredictionBatch,
    _prediction_links,
)

from ..object_tracker import (
    DetectionFrameBatch,
)


@dataclass(frozen=True, slots=True)
class TrackerPredictionComparison:
    detection_batch_sha256: str
    left_prediction_sha256: str
    right_prediction_sha256: str
    left_tracker_adapter: str
    left_tracker_revision: str
    left_tracker_config_sha256: str
    left_runtime_id: str
    right_tracker_adapter: str
    right_tracker_revision: str
    right_tracker_config_sha256: str
    right_runtime_id: str
    assignment_count: int
    left_predicted_track_count: int
    right_predicted_track_count: int
    left_link_count: int
    right_link_count: int
    shared_link_count: int
    left_only_link_count: int
    right_only_link_count: int
    continuous_disagreement_count: int
    recovery_disagreement_count: int
    disagreement_frame_count: int
    disagreement_frame_ids: tuple[str, ...]
    disagreement_frames_truncated: bool
    human_ground_truth_used: bool = False
    accuracy_evaluated: bool = False
    better_adapter_selected: bool = False
    subject_identity_inference: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TrackerEvaluationError("unsupported tracker prediction comparison version")
        for value, field in (
            (self.detection_batch_sha256, "detection batch sha256"),
            (self.left_prediction_sha256, "left prediction sha256"),
            (self.right_prediction_sha256, "right prediction sha256"),
            (self.left_tracker_config_sha256, "left tracker config sha256"),
            (self.right_tracker_config_sha256, "right tracker config sha256"),
        ):
            _sha256(value, f"tracker comparison {field}")
        for field in (
            "left_tracker_adapter",
            "left_tracker_revision",
            "left_runtime_id",
            "right_tracker_adapter",
            "right_tracker_revision",
            "right_runtime_id",
        ):
            _safe_id(getattr(self, field), f"tracker comparison {field}")
        for field in (
            "assignment_count",
            "left_predicted_track_count",
            "right_predicted_track_count",
            "left_link_count",
            "right_link_count",
            "shared_link_count",
            "left_only_link_count",
            "right_only_link_count",
            "continuous_disagreement_count",
            "recovery_disagreement_count",
            "disagreement_frame_count",
        ):
            _integer(
                getattr(self, field),
                f"tracker comparison {field}",
                minimum=0,
                maximum=MAX_TRACKER_EVAL_ASSIGNMENTS * 2,
            )
        if (
            self.left_predicted_track_count > self.assignment_count
            or self.right_predicted_track_count > self.assignment_count
            or self.left_link_count != self.assignment_count - self.left_predicted_track_count
            or self.right_link_count != self.assignment_count - self.right_predicted_track_count
        ):
            raise TrackerEvaluationError(
                "tracker comparison assignment and track counts are inconsistent"
            )
        if (
            self.shared_link_count + self.left_only_link_count != self.left_link_count
            or self.shared_link_count + self.right_only_link_count != self.right_link_count
            or self.continuous_disagreement_count + self.recovery_disagreement_count
            != self.left_only_link_count + self.right_only_link_count
        ):
            raise TrackerEvaluationError("tracker comparison link counts are inconsistent")
        if (
            not isinstance(self.disagreement_frame_ids, tuple)
            or len(self.disagreement_frame_ids) > MAX_TRACKER_DISAGREEMENT_FRAMES
            or any(
                _SAFE_ID.fullmatch(item) is None
                for item in self.disagreement_frame_ids
                if isinstance(item, str)
            )
            or any(not isinstance(item, str) for item in self.disagreement_frame_ids)
            or len(set(self.disagreement_frame_ids)) != len(self.disagreement_frame_ids)
        ):
            raise TrackerEvaluationError("tracker comparison disagreement frame IDs are invalid")
        if type(self.disagreement_frames_truncated) is not bool:
            raise TrackerEvaluationError("tracker comparison truncation flag must be boolean")
        if self.disagreement_frames_truncated:
            if len(
                self.disagreement_frame_ids
            ) != MAX_TRACKER_DISAGREEMENT_FRAMES or self.disagreement_frame_count <= len(
                self.disagreement_frame_ids
            ):
                raise TrackerEvaluationError(
                    "tracker comparison truncated frame list is inconsistent"
                )
        elif self.disagreement_frame_count != len(self.disagreement_frame_ids):
            raise TrackerEvaluationError(
                "tracker comparison disagreement frame count is inconsistent"
            )
        for field in (
            "human_ground_truth_used",
            "accuracy_evaluated",
            "better_adapter_selected",
            "subject_identity_inference",
            "source_media_included",
            "paths_disclosed",
        ):
            if getattr(self, field) is not False:
                raise TrackerEvaluationError(f"tracker comparison {field} must be false")

    @classmethod
    def from_dict(cls, value: Any) -> TrackerPredictionComparison:
        fields = {
            "version",
            "detection_batch_sha256",
            "left_prediction_sha256",
            "right_prediction_sha256",
            "left_tracker_adapter",
            "left_tracker_revision",
            "left_tracker_config_sha256",
            "left_runtime_id",
            "right_tracker_adapter",
            "right_tracker_revision",
            "right_tracker_config_sha256",
            "right_runtime_id",
            "assignment_count",
            "left_predicted_track_count",
            "right_predicted_track_count",
            "left_link_count",
            "right_link_count",
            "shared_link_count",
            "left_only_link_count",
            "right_only_link_count",
            "continuous_disagreement_count",
            "recovery_disagreement_count",
            "disagreement_frame_count",
            "disagreement_frame_ids",
            "disagreement_frames_truncated",
            "human_ground_truth_used",
            "accuracy_evaluated",
            "better_adapter_selected",
            "subject_identity_inference",
            "source_media_included",
            "paths_disclosed",
        }
        data = _exact_object(value, fields, "tracker prediction comparison")
        frame_ids = data["disagreement_frame_ids"]
        if not isinstance(frame_ids, list):
            raise TrackerEvaluationError(
                "tracker comparison disagreement_frame_ids must be an array"
            )
        return cls(
            **{
                **data,
                "disagreement_frame_ids": tuple(frame_ids),
            }
        )

    @classmethod
    def read(cls, path: str | Path) -> TrackerPredictionComparison:
        return cls.from_dict(_read_json(path, "tracker prediction comparison"))

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "detection_batch_sha256": self.detection_batch_sha256,
            "left_prediction_sha256": self.left_prediction_sha256,
            "right_prediction_sha256": self.right_prediction_sha256,
            "left_tracker_adapter": self.left_tracker_adapter,
            "left_tracker_revision": self.left_tracker_revision,
            "left_tracker_config_sha256": self.left_tracker_config_sha256,
            "left_runtime_id": self.left_runtime_id,
            "right_tracker_adapter": self.right_tracker_adapter,
            "right_tracker_revision": self.right_tracker_revision,
            "right_tracker_config_sha256": self.right_tracker_config_sha256,
            "right_runtime_id": self.right_runtime_id,
            "assignment_count": self.assignment_count,
            "left_predicted_track_count": self.left_predicted_track_count,
            "right_predicted_track_count": self.right_predicted_track_count,
            "left_link_count": self.left_link_count,
            "right_link_count": self.right_link_count,
            "shared_link_count": self.shared_link_count,
            "left_only_link_count": self.left_only_link_count,
            "right_only_link_count": self.right_only_link_count,
            "continuous_disagreement_count": self.continuous_disagreement_count,
            "recovery_disagreement_count": self.recovery_disagreement_count,
            "disagreement_frame_count": self.disagreement_frame_count,
            "disagreement_frame_ids": list(self.disagreement_frame_ids),
            "disagreement_frames_truncated": self.disagreement_frames_truncated,
            "human_ground_truth_used": self.human_ground_truth_used,
            "accuracy_evaluated": self.accuracy_evaluated,
            "better_adapter_selected": self.better_adapter_selected,
            "subject_identity_inference": self.subject_identity_inference,
            "source_media_included": self.source_media_included,
            "paths_disclosed": self.paths_disclosed,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())

    def write_new(self, path: str | Path) -> None:
        _write_new_json(path, self.to_dict(), "tracker prediction comparison")

    def to_public_dict(self) -> dict[str, object]:
        return {**self.to_dict(), "comparison_sha256": self.fingerprint}


def compare_tracker_predictions(
    batch: DetectionFrameBatch,
    left: TrackerPredictionBatch,
    right: TrackerPredictionBatch,
) -> TrackerPredictionComparison:
    if not isinstance(batch, DetectionFrameBatch):
        raise TrackerEvaluationError("tracker comparison batch is invalid")
    left.bind(batch)
    right.bind(batch)
    left_links = _prediction_links(batch, left)
    right_links = _prediction_links(batch, right)
    shared = left_links & right_links
    left_only = left_links - right_links
    right_only = right_links - left_links
    disagreements = left_only | right_only
    frame_indices = {frame.frame_id: index for index, frame in enumerate(batch.frames)}
    disagreement_frames = {frame_id for link in disagreements for frame_id in (link[0], link[2])}
    ordered_frames = tuple(
        frame.frame_id for frame in batch.frames if frame.frame_id in disagreement_frames
    )
    continuous_disagreements = sum(
        frame_indices[link[2]] - frame_indices[link[0]] == 1 for link in disagreements
    )
    assignment_count = len(left.assignments)
    left_tracks = len({item.tracking_id for item in left.assignments})
    right_tracks = len({item.tracking_id for item in right.assignments})
    result = TrackerPredictionComparison(
        detection_batch_sha256=batch.fingerprint,
        left_prediction_sha256=left.fingerprint,
        right_prediction_sha256=right.fingerprint,
        left_tracker_adapter=left.tracker_adapter,
        left_tracker_revision=left.tracker_revision,
        left_tracker_config_sha256=left.tracker_config_sha256,
        left_runtime_id=left.runtime_id,
        right_tracker_adapter=right.tracker_adapter,
        right_tracker_revision=right.tracker_revision,
        right_tracker_config_sha256=right.tracker_config_sha256,
        right_runtime_id=right.runtime_id,
        assignment_count=assignment_count,
        left_predicted_track_count=left_tracks,
        right_predicted_track_count=right_tracks,
        left_link_count=len(left_links),
        right_link_count=len(right_links),
        shared_link_count=len(shared),
        left_only_link_count=len(left_only),
        right_only_link_count=len(right_only),
        continuous_disagreement_count=continuous_disagreements,
        recovery_disagreement_count=len(disagreements) - continuous_disagreements,
        disagreement_frame_count=len(ordered_frames),
        disagreement_frame_ids=ordered_frames[:MAX_TRACKER_DISAGREEMENT_FRAMES],
        disagreement_frames_truncated=len(ordered_frames) > MAX_TRACKER_DISAGREEMENT_FRAMES,
    )
    return result
