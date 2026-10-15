"""TrackerAssociationReport v1 over approved human association labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baby_monitor.tracker_evaluation._shared import (
    MAX_TRACKER_EVAL_ASSIGNMENTS,
    TrackerEvaluationError,
    _canonical_sha256,
    _exact_object,
    _integer,
    _optional_ratio,
    _ratio,
    _read_json,
    _safe_id,
    _sha256,
    _write_new_json,
)
from baby_monitor.tracker_evaluation.prediction import (
    TrackerPredictionBatch,
)
from baby_monitor.tracker_evaluation.review import (
    _EVIDENCE_STATUSES,
    _GROUND_TRUTH_PROVENANCE,
    _REVIEW_STATES,
    TrackerAssociationReview,
)

from ..object_tracker import (
    DetectionFrameBatch,
)
from ..source_admission import ExternalSourceAdmission


@dataclass(frozen=True, slots=True)
class TrackerAssociationReport:
    detection_batch_sha256: str
    review_sha256: str
    prediction_sha256: str
    tracker_adapter: str
    tracker_revision: str
    tracker_config_sha256: str
    runtime_id: str
    evidence_status: str
    frame_count: int
    detection_count: int
    eligible_assignment_count: int
    duplicate_detection_count: int
    false_positive_detection_count: int
    unreviewable_detection_count: int
    ground_truth_subject_count: int
    subjects_with_links: int
    predicted_track_count: int
    fragmented_subject_count: int
    fragment_excess_count: int
    fragmentation_rate: float | None
    eligible_link_count: int
    id_switch_count: int
    id_switch_rate: float | None
    continuous_link_count: int
    continuous_id_switch_count: int
    recovery_link_count: int
    recovery_id_switch_count: int
    merged_predicted_track_count: int
    reviewer_count: int
    ground_truth_provenance: str = "local_human_review"
    scope: str = "association_on_detector_boxes"
    human_ground_truth: bool = True
    detection_accuracy_evaluated: bool = False
    missed_detections_evaluated: bool = False
    subject_identity_inference: bool = False
    age_role_inference: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TrackerEvaluationError("unsupported tracker report version")
        for value, field in (
            (self.detection_batch_sha256, "detection batch sha256"),
            (self.review_sha256, "review sha256"),
            (self.prediction_sha256, "prediction sha256"),
            (self.tracker_config_sha256, "tracker config sha256"),
        ):
            _sha256(value, f"tracker report {field}")
        for field in (
            "tracker_adapter",
            "tracker_revision",
            "runtime_id",
        ):
            _safe_id(getattr(self, field), f"tracker report {field}")
        if self.evidence_status not in _EVIDENCE_STATUSES:
            raise TrackerEvaluationError("tracker report evidence_status is invalid")
        if self.ground_truth_provenance not in _GROUND_TRUTH_PROVENANCE:
            raise TrackerEvaluationError("tracker report ground_truth_provenance is invalid")
        for field in (
            "frame_count",
            "detection_count",
            "eligible_assignment_count",
            "duplicate_detection_count",
            "false_positive_detection_count",
            "unreviewable_detection_count",
            "ground_truth_subject_count",
            "subjects_with_links",
            "predicted_track_count",
            "fragmented_subject_count",
            "fragment_excess_count",
            "eligible_link_count",
            "id_switch_count",
            "continuous_link_count",
            "continuous_id_switch_count",
            "recovery_link_count",
            "recovery_id_switch_count",
            "merged_predicted_track_count",
            "reviewer_count",
        ):
            minimum = 1 if field == "frame_count" else 0
            _integer(
                getattr(self, field),
                f"tracker report {field}",
                minimum=minimum,
                maximum=MAX_TRACKER_EVAL_ASSIGNMENTS,
            )
        if (
            self.eligible_assignment_count
            + self.duplicate_detection_count
            + self.false_positive_detection_count
            + self.unreviewable_detection_count
            != self.detection_count
        ):
            raise TrackerEvaluationError("tracker report review-state counts are inconsistent")
        if not (
            self.subjects_with_links
            <= self.ground_truth_subject_count
            <= self.eligible_assignment_count
        ):
            raise TrackerEvaluationError("tracker report subject counts are inconsistent")
        if self.predicted_track_count > self.eligible_assignment_count:
            raise TrackerEvaluationError("tracker report predicted track count is inconsistent")
        if self.merged_predicted_track_count > self.predicted_track_count:
            raise TrackerEvaluationError("tracker report merged track count is inconsistent")
        if self.fragmented_subject_count > self.subjects_with_links:
            raise TrackerEvaluationError("tracker report fragmented subject count is inconsistent")
        if self.fragment_excess_count > self.eligible_assignment_count:
            raise TrackerEvaluationError("tracker report fragment excess count is inconsistent")
        if self.id_switch_count > self.eligible_link_count:
            raise TrackerEvaluationError("tracker report id switch count is inconsistent")
        if (
            self.continuous_link_count + self.recovery_link_count != self.eligible_link_count
            or self.continuous_id_switch_count + self.recovery_id_switch_count
            != self.id_switch_count
            or self.continuous_id_switch_count > self.continuous_link_count
            or self.recovery_id_switch_count > self.recovery_link_count
        ):
            raise TrackerEvaluationError("tracker report link partitions are inconsistent")
        parsed_fragmentation = _optional_ratio(
            self.fragmentation_rate,
            "tracker report fragmentation_rate",
        )
        parsed_switch = _optional_ratio(
            self.id_switch_rate,
            "tracker report id_switch_rate",
        )
        if parsed_fragmentation != _ratio(
            self.fragmented_subject_count,
            self.subjects_with_links,
        ):
            raise TrackerEvaluationError("tracker report fragmentation_rate is inconsistent")
        if parsed_switch != _ratio(self.id_switch_count, self.eligible_link_count):
            raise TrackerEvaluationError("tracker report id_switch_rate is inconsistent")
        object.__setattr__(self, "fragmentation_rate", parsed_fragmentation)
        object.__setattr__(self, "id_switch_rate", parsed_switch)
        expected_status = (
            "no_eligible_assignments"
            if self.eligible_assignment_count == 0
            else (
                "single_appearance_only"
                if self.eligible_link_count == 0
                else "association_metrics_available"
            )
        )
        if self.evidence_status != expected_status:
            raise TrackerEvaluationError("tracker report evidence_status is inconsistent")
        if self.scope != "association_on_detector_boxes":
            raise TrackerEvaluationError("tracker report scope is invalid")
        if self.ground_truth_provenance == "local_human_review" and self.reviewer_count == 0:
            raise TrackerEvaluationError("local tracker report requires a reviewer")
        if self.human_ground_truth is not True:
            raise TrackerEvaluationError("tracker report requires human ground truth")
        for field in (
            "detection_accuracy_evaluated",
            "missed_detections_evaluated",
            "subject_identity_inference",
            "age_role_inference",
            "source_media_included",
            "paths_disclosed",
        ):
            if getattr(self, field) is not False:
                raise TrackerEvaluationError(f"tracker report {field} must be false")

    @classmethod
    def from_dict(cls, value: Any) -> TrackerAssociationReport:
        data = _exact_object(
            value,
            {
                "version",
                "detection_batch_sha256",
                "review_sha256",
                "prediction_sha256",
                "tracker_adapter",
                "tracker_revision",
                "tracker_config_sha256",
                "runtime_id",
                "evidence_status",
                "frame_count",
                "detection_count",
                "eligible_assignment_count",
                "duplicate_detection_count",
                "false_positive_detection_count",
                "unreviewable_detection_count",
                "ground_truth_subject_count",
                "subjects_with_links",
                "predicted_track_count",
                "fragmented_subject_count",
                "fragment_excess_count",
                "fragmentation_rate",
                "eligible_link_count",
                "id_switch_count",
                "id_switch_rate",
                "continuous_link_count",
                "continuous_id_switch_count",
                "recovery_link_count",
                "recovery_id_switch_count",
                "merged_predicted_track_count",
                "reviewer_count",
                "ground_truth_provenance",
                "scope",
                "human_ground_truth",
                "detection_accuracy_evaluated",
                "missed_detections_evaluated",
                "subject_identity_inference",
                "age_role_inference",
                "source_media_included",
                "paths_disclosed",
            },
            "tracker association report",
        )
        return cls(**data)

    @classmethod
    def read(cls, path: str | Path) -> TrackerAssociationReport:
        return cls.from_dict(_read_json(path, "tracker association report"))

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "version",
                "detection_batch_sha256",
                "review_sha256",
                "prediction_sha256",
                "tracker_adapter",
                "tracker_revision",
                "tracker_config_sha256",
                "runtime_id",
                "evidence_status",
                "frame_count",
                "detection_count",
                "eligible_assignment_count",
                "duplicate_detection_count",
                "false_positive_detection_count",
                "unreviewable_detection_count",
                "ground_truth_subject_count",
                "subjects_with_links",
                "predicted_track_count",
                "fragmented_subject_count",
                "fragment_excess_count",
                "fragmentation_rate",
                "eligible_link_count",
                "id_switch_count",
                "id_switch_rate",
                "continuous_link_count",
                "continuous_id_switch_count",
                "recovery_link_count",
                "recovery_id_switch_count",
                "merged_predicted_track_count",
                "reviewer_count",
                "ground_truth_provenance",
                "scope",
                "human_ground_truth",
                "detection_accuracy_evaluated",
                "missed_detections_evaluated",
                "subject_identity_inference",
                "age_role_inference",
                "source_media_included",
                "paths_disclosed",
            )
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())

    def write_new(self, path: str | Path) -> None:
        _write_new_json(path, self.to_dict(), "tracker association report")

    def to_public_dict(self) -> dict[str, object]:
        return {**self.to_dict(), "report_sha256": self.fingerprint}


def evaluate_tracker_association(
    batch: DetectionFrameBatch,
    review: TrackerAssociationReview,
    prediction: TrackerPredictionBatch,
    *,
    source_admission: ExternalSourceAdmission | None = None,
) -> TrackerAssociationReport:
    if not isinstance(batch, DetectionFrameBatch):
        raise TrackerEvaluationError("tracker evaluation batch is invalid")
    review.bind(batch)
    review.bind_source_admission(source_admission)
    prediction.bind(batch)
    if review.review_status != "approved" or not review.human_labels:
        raise TrackerEvaluationError("tracker evaluation requires approved human ground truth")

    frame_indices = {frame.frame_id: index for index, frame in enumerate(batch.frames)}
    predictions = {
        (item.frame_id, item.detection_id): item.tracking_id for item in prediction.assignments
    }
    state_counts = {
        state: sum(item.review_state == state for item in review.assignments)
        for state in _REVIEW_STATES
    }
    subject_observations: dict[str, list[tuple[int, str]]] = {}
    for item in review.assignments:
        if item.review_state != "assigned":
            continue
        assert item.subject_id is not None
        subject_observations.setdefault(item.subject_id, []).append(
            (
                frame_indices[item.frame_id],
                predictions[(item.frame_id, item.detection_id)],
            )
        )

    fragmented_subjects = 0
    fragment_excess = 0
    eligible_links = 0
    id_switches = 0
    continuous_links = 0
    continuous_switches = 0
    recovery_links = 0
    recovery_switches = 0
    subjects_with_links = 0
    track_subjects: dict[str, set[str]] = {}
    eligible_track_ids: set[str] = set()
    for subject_id, observations in subject_observations.items():
        observations.sort()
        track_ids = {tracking_id for _, tracking_id in observations}
        eligible_track_ids.update(track_ids)
        if len(observations) >= 2:
            subjects_with_links += 1
        if len(track_ids) > 1:
            fragmented_subjects += 1
            fragment_excess += len(track_ids) - 1
        for tracking_id in track_ids:
            track_subjects.setdefault(tracking_id, set()).add(subject_id)
        for previous, current in zip(
            observations,
            observations[1:],
            strict=False,
        ):
            eligible_links += 1
            switched = current[1] != previous[1]
            if switched:
                id_switches += 1
            if current[0] - previous[0] == 1:
                continuous_links += 1
                if switched:
                    continuous_switches += 1
            else:
                recovery_links += 1
                if switched:
                    recovery_switches += 1

    evidence_status = (
        "no_eligible_assignments"
        if not subject_observations
        else ("single_appearance_only" if eligible_links == 0 else "association_metrics_available")
    )
    return TrackerAssociationReport(
        detection_batch_sha256=batch.fingerprint,
        review_sha256=review.fingerprint,
        prediction_sha256=prediction.fingerprint,
        tracker_adapter=prediction.tracker_adapter,
        tracker_revision=prediction.tracker_revision,
        tracker_config_sha256=prediction.tracker_config_sha256,
        runtime_id=prediction.runtime_id,
        evidence_status=evidence_status,
        frame_count=len(batch.frames),
        detection_count=len(review.assignments),
        eligible_assignment_count=state_counts["assigned"],
        duplicate_detection_count=state_counts["duplicate"],
        false_positive_detection_count=state_counts["false_positive"],
        unreviewable_detection_count=state_counts["unreviewable"],
        ground_truth_subject_count=len(subject_observations),
        subjects_with_links=subjects_with_links,
        predicted_track_count=len(eligible_track_ids),
        fragmented_subject_count=fragmented_subjects,
        fragment_excess_count=fragment_excess,
        fragmentation_rate=_ratio(fragmented_subjects, subjects_with_links),
        eligible_link_count=eligible_links,
        id_switch_count=id_switches,
        id_switch_rate=_ratio(id_switches, eligible_links),
        continuous_link_count=continuous_links,
        continuous_id_switch_count=continuous_switches,
        recovery_link_count=recovery_links,
        recovery_id_switch_count=recovery_switches,
        merged_predicted_track_count=sum(len(subjects) > 1 for subjects in track_subjects.values()),
        reviewer_count=len(review.reviews),
        ground_truth_provenance=review.ground_truth_provenance,
    )
