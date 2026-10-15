"""TrackerAssociationReview v1: anonymous per-box human association labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from baby_monitor.tracker_evaluation._shared import (
    MAX_TRACKER_EVAL_ASSIGNMENTS,
    TrackerEvaluationError,
    _aware_timestamp,
    _canonical_sha256,
    _detection_keys,
    _exact_object,
    _optional_sha256,
    _read_json,
    _reviewer_id,
    _safe_id,
    _sha256,
    _subject_id,
    _write_new_json,
)

from ..object_tracker import (
    DetectionFrameBatch,
)
from ..source_admission import ExternalSourceAdmission

_REVIEW_STATES = {
    "assigned",
    "duplicate",
    "false_positive",
    "pending",
    "unreviewable",
}


_EVIDENCE_STATUSES = {
    "association_metrics_available",
    "no_eligible_assignments",
    "single_appearance_only",
}


_GROUND_TRUTH_PROVENANCE = {
    "admitted_publisher_human",
    "local_human_review",
}


@dataclass(frozen=True, slots=True)
class TrackerReviewer:
    reviewer_id: str
    reviewed_at: datetime
    verdict: str = "approve"

    def __post_init__(self) -> None:
        _reviewer_id(self.reviewer_id, "tracker reviewer_id")
        if not isinstance(self.reviewed_at, datetime):
            raise TrackerEvaluationError("tracker reviewed_at is invalid")
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise TrackerEvaluationError("tracker reviewed_at must include a timezone")
        object.__setattr__(self, "reviewed_at", self.reviewed_at.astimezone(UTC))
        if self.verdict != "approve":
            raise TrackerEvaluationError("tracker reviewer verdict must be approve")

    @classmethod
    def from_dict(cls, value: Any) -> TrackerReviewer:
        data = _exact_object(
            value,
            {"reviewer_id", "reviewed_at", "verdict"},
            "tracker reviewer",
        )
        return cls(
            reviewer_id=data["reviewer_id"],
            reviewed_at=_aware_timestamp(data["reviewed_at"], "reviewed_at"),
            verdict=data["verdict"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reviewer_id": self.reviewer_id,
            "reviewed_at": self.reviewed_at.isoformat(),
            "verdict": self.verdict,
        }


@dataclass(frozen=True, slots=True)
class TrackerReviewAssignment:
    frame_id: str
    detection_id: str
    review_state: str
    subject_id: str | None

    def __post_init__(self) -> None:
        _safe_id(self.frame_id, "tracker review frame_id")
        _safe_id(self.detection_id, "tracker review detection_id")
        if self.review_state not in _REVIEW_STATES:
            raise TrackerEvaluationError("tracker review_state is invalid")
        if self.review_state in {"assigned", "duplicate"}:
            _subject_id(self.subject_id, "tracker review subject_id")
        elif self.subject_id is not None:
            raise TrackerEvaluationError(
                "tracker review subject_id requires assigned or duplicate state"
            )

    @classmethod
    def from_dict(cls, value: Any) -> TrackerReviewAssignment:
        data = _exact_object(
            value,
            {"frame_id", "detection_id", "review_state", "subject_id"},
            "tracker review assignment",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "detection_id": self.detection_id,
            "review_state": self.review_state,
            "subject_id": self.subject_id,
        }


@dataclass(frozen=True, slots=True)
class TrackerAssociationReview:
    camera_id: str
    detector_adapter: str
    detection_batch_sha256: str
    source_admission_sha256: str | None
    review_status: str
    assignments: tuple[TrackerReviewAssignment, ...]
    reviews: tuple[TrackerReviewer, ...]
    human_labels: bool
    ground_truth_provenance: str = "local_human_review"
    scope: str = "association_on_detector_boxes"
    identity_semantics: str = "anonymous_within_review"
    model_generated_labels: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TrackerEvaluationError("unsupported tracker review version")
        _safe_id(self.camera_id, "tracker review camera_id")
        _safe_id(self.detector_adapter, "tracker review detector_adapter")
        _sha256(self.detection_batch_sha256, "tracker review detection batch sha256")
        _optional_sha256(
            self.source_admission_sha256,
            "tracker review source admission sha256",
        )
        if self.review_status not in {"approved", "pending"}:
            raise TrackerEvaluationError("tracker review_status is invalid")
        if (
            not isinstance(self.assignments, tuple)
            or len(self.assignments) > MAX_TRACKER_EVAL_ASSIGNMENTS
            or any(not isinstance(item, TrackerReviewAssignment) for item in self.assignments)
        ):
            raise TrackerEvaluationError("tracker review assignments are invalid")
        keys = tuple((item.frame_id, item.detection_id) for item in self.assignments)
        if len(set(keys)) != len(keys):
            raise TrackerEvaluationError("tracker review assignments are duplicated")
        if (
            not isinstance(self.reviews, tuple)
            or len(self.reviews) > 5
            or any(not isinstance(item, TrackerReviewer) for item in self.reviews)
            or len({item.reviewer_id for item in self.reviews}) != len(self.reviews)
        ):
            raise TrackerEvaluationError("tracker review reviewers are invalid")
        if type(self.human_labels) is not bool:
            raise TrackerEvaluationError("tracker review human_labels must be boolean")
        if self.ground_truth_provenance not in _GROUND_TRUTH_PROVENANCE:
            raise TrackerEvaluationError("tracker review ground_truth_provenance is invalid")
        if self.scope != "association_on_detector_boxes":
            raise TrackerEvaluationError("tracker review scope is invalid")
        if self.identity_semantics != "anonymous_within_review":
            raise TrackerEvaluationError("tracker review identity semantics are invalid")
        for field in (
            "model_generated_labels",
            "source_media_included",
            "paths_disclosed",
        ):
            if getattr(self, field) is not False:
                raise TrackerEvaluationError(f"tracker review {field} must be false")
        if self.review_status == "pending":
            if (
                self.human_labels
                or self.reviews
                or any(item.review_state != "pending" for item in self.assignments)
            ):
                raise TrackerEvaluationError(
                    "pending tracker review cannot contain completed labels"
                )
        elif not self.human_labels or any(
            item.review_state == "pending" for item in self.assignments
        ):
            raise TrackerEvaluationError("approved tracker review requires completed human labels")
        elif self.ground_truth_provenance == "local_human_review" and not self.reviews:
            raise TrackerEvaluationError("local tracker ground truth requires a reviewer")
        elif (
            self.ground_truth_provenance == "admitted_publisher_human"
            and self.source_admission_sha256 is None
        ):
            raise TrackerEvaluationError("publisher tracker ground truth requires source admission")

    @classmethod
    def pending(
        cls,
        batch: DetectionFrameBatch,
        *,
        source_admission_sha256: str | None = None,
    ) -> TrackerAssociationReview:
        if not isinstance(batch, DetectionFrameBatch):
            raise TrackerEvaluationError("tracker review input batch is invalid")
        return cls(
            camera_id=batch.camera_id,
            detector_adapter=batch.detector_adapter,
            detection_batch_sha256=batch.fingerprint,
            source_admission_sha256=_optional_sha256(
                source_admission_sha256,
                "tracker review source admission sha256",
            ),
            review_status="pending",
            assignments=tuple(
                TrackerReviewAssignment(
                    frame_id=frame.frame_id,
                    detection_id=detection.detection_id,
                    review_state="pending",
                    subject_id=None,
                )
                for frame in batch.frames
                for detection in frame.detections
            ),
            reviews=(),
            human_labels=False,
        )

    @classmethod
    def from_dict(cls, value: Any) -> TrackerAssociationReview:
        data = _exact_object(
            value,
            {
                "version",
                "camera_id",
                "detector_adapter",
                "detection_batch_sha256",
                "source_admission_sha256",
                "review_status",
                "assignments",
                "reviews",
                "human_labels",
                "ground_truth_provenance",
                "scope",
                "identity_semantics",
                "model_generated_labels",
                "source_media_included",
                "paths_disclosed",
            },
            "tracker association review",
        )
        assignments = data["assignments"]
        reviews = data["reviews"]
        if not isinstance(assignments, list) or not isinstance(reviews, list):
            raise TrackerEvaluationError("tracker review arrays are invalid")
        return cls(
            **{
                **data,
                "assignments": tuple(
                    TrackerReviewAssignment.from_dict(item) for item in assignments
                ),
                "reviews": tuple(TrackerReviewer.from_dict(item) for item in reviews),
            }
        )

    @classmethod
    def read(cls, path: str | Path) -> TrackerAssociationReview:
        return cls.from_dict(_read_json(path, "tracker association review"))

    def bind(self, batch: DetectionFrameBatch) -> None:
        if not isinstance(batch, DetectionFrameBatch):
            raise TrackerEvaluationError("tracker review batch is invalid")
        if (
            self.camera_id != batch.camera_id
            or self.detector_adapter != batch.detector_adapter
            or self.detection_batch_sha256 != batch.fingerprint
        ):
            raise TrackerEvaluationError("tracker review belongs to a different detection batch")
        expected = _detection_keys(batch)
        actual = tuple((item.frame_id, item.detection_id) for item in self.assignments)
        if actual != expected:
            raise TrackerEvaluationError(
                "tracker review must cover every detection in canonical order"
            )
        if self.review_status != "approved":
            return
        by_frame: dict[str, list[TrackerReviewAssignment]] = {}
        for assignment in self.assignments:
            by_frame.setdefault(assignment.frame_id, []).append(assignment)
        for assignments in by_frame.values():
            primaries = {item.subject_id for item in assignments if item.review_state == "assigned"}
            if len(primaries) != sum(item.review_state == "assigned" for item in assignments):
                raise TrackerEvaluationError(
                    "tracker review has multiple primary boxes for one subject in a frame"
                )
            if any(
                item.review_state == "duplicate" and item.subject_id not in primaries
                for item in assignments
            ):
                raise TrackerEvaluationError(
                    "tracker duplicate box requires one primary subject box in the frame"
                )

    def bind_source_admission(
        self,
        admission: ExternalSourceAdmission | None,
    ) -> None:
        if self.ground_truth_provenance != "admitted_publisher_human":
            return
        if admission is None:
            raise TrackerEvaluationError(
                "publisher tracker ground truth requires its source admission"
            )
        if (
            self.source_admission_sha256 != admission.fingerprint
            or admission.admission.status != "verified_local"
            or not admission.labels.ground_truth
            or not admission.labels.human_verified
            or admission.labels.origin not in {"human_expert", "human_nonexpert"}
        ):
            raise TrackerEvaluationError(
                "publisher tracker ground truth source admission is invalid"
            )

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "detector_adapter": self.detector_adapter,
            "detection_batch_sha256": self.detection_batch_sha256,
            "source_admission_sha256": self.source_admission_sha256,
            "review_status": self.review_status,
            "assignments": [item.to_dict() for item in self.assignments],
            "reviews": [item.to_dict() for item in self.reviews],
            "human_labels": self.human_labels,
            "ground_truth_provenance": self.ground_truth_provenance,
            "scope": self.scope,
            "identity_semantics": self.identity_semantics,
            "model_generated_labels": self.model_generated_labels,
            "source_media_included": self.source_media_included,
            "paths_disclosed": self.paths_disclosed,
        }

    def write_new(self, path: str | Path) -> None:
        _write_new_json(path, self.to_dict(), "tracker association review")

    def to_public_dict(self) -> dict[str, object]:
        counts = {
            state: sum(item.review_state == state for item in self.assignments)
            for state in sorted(_REVIEW_STATES)
        }
        return {
            "valid": True,
            "version": self.version,
            "review_status": self.review_status,
            "detection_batch_sha256": self.detection_batch_sha256,
            "review_sha256": self.fingerprint,
            "assignments": len(self.assignments),
            "assignment_states": counts,
            "reviewers": len(self.reviews),
            "human_labels": self.human_labels,
            "ground_truth_provenance": self.ground_truth_provenance,
            "model_generated_labels": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }
