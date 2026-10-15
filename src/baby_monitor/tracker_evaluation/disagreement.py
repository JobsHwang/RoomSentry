"""Fast human review of disagreeing links and its relative report."""

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
from baby_monitor.tracker_evaluation.comparison import (
    TrackerPredictionComparison,
    compare_tracker_predictions,
)
from baby_monitor.tracker_evaluation.prediction import (
    TrackerPredictionBatch,
    _prediction_links,
)
from baby_monitor.tracker_evaluation.review import (
    TrackerReviewer,
)

from ..object_tracker import (
    DetectionFrameBatch,
)

_DISAGREEMENT_REVIEW_STATES = {
    "pending",
    "same_subject",
    "different_subject",
    "unreviewable",
}


_DISAGREEMENT_EVIDENCE_STATUSES = {
    "relative_difference_available",
    "tie",
    "no_reviewable_links",
}


_DISAGREEMENT_PREFERENCES = {"left", "right", "tie", "insufficient_evidence"}


@dataclass(frozen=True, slots=True)
class TrackerDisagreementLinkReview:
    link_id: str
    previous_frame_id: str
    previous_detection_id: str
    current_frame_id: str
    current_detection_id: str
    proposed_by: str
    link_kind: str
    review_state: str

    def __post_init__(self) -> None:
        _safe_id(self.link_id, "tracker disagreement link_id")
        for field in (
            "previous_frame_id",
            "previous_detection_id",
            "current_frame_id",
            "current_detection_id",
        ):
            _safe_id(getattr(self, field), f"tracker disagreement {field}")
        if (
            self.previous_frame_id,
            self.previous_detection_id,
        ) == (
            self.current_frame_id,
            self.current_detection_id,
        ):
            raise TrackerEvaluationError("tracker disagreement link endpoints must differ")
        if self.proposed_by not in {"left", "right"}:
            raise TrackerEvaluationError("tracker disagreement proposed_by is invalid")
        if self.link_kind not in {"continuous", "recovery"}:
            raise TrackerEvaluationError("tracker disagreement link_kind is invalid")
        if self.review_state not in _DISAGREEMENT_REVIEW_STATES:
            raise TrackerEvaluationError("tracker disagreement review_state is invalid")

    @classmethod
    def from_dict(cls, value: Any) -> TrackerDisagreementLinkReview:
        data = _exact_object(
            value,
            {
                "link_id",
                "previous_frame_id",
                "previous_detection_id",
                "current_frame_id",
                "current_detection_id",
                "proposed_by",
                "link_kind",
                "review_state",
            },
            "tracker disagreement link review",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, str]:
        return {
            "link_id": self.link_id,
            "previous_frame_id": self.previous_frame_id,
            "previous_detection_id": self.previous_detection_id,
            "current_frame_id": self.current_frame_id,
            "current_detection_id": self.current_detection_id,
            "proposed_by": self.proposed_by,
            "link_kind": self.link_kind,
            "review_state": self.review_state,
        }

    @property
    def identity(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            self.link_id,
            self.previous_frame_id,
            self.previous_detection_id,
            self.current_frame_id,
            self.current_detection_id,
            self.proposed_by,
            self.link_kind,
        )


def _disagreement_link_specs(
    batch: DetectionFrameBatch,
    left: TrackerPredictionBatch,
    right: TrackerPredictionBatch,
    comparison: TrackerPredictionComparison,
) -> tuple[TrackerDisagreementLinkReview, ...]:
    computed = compare_tracker_predictions(batch, left, right)
    if computed != comparison:
        raise TrackerEvaluationError(
            "tracker disagreement comparison does not match the predictions"
        )
    left_only = _prediction_links(batch, left) - _prediction_links(batch, right)
    right_only = _prediction_links(batch, right) - _prediction_links(batch, left)
    frame_indices = {frame.frame_id: index for index, frame in enumerate(batch.frames)}
    raw = [(link, "left") for link in left_only] + [(link, "right") for link in right_only]
    raw.sort(
        key=lambda item: (
            frame_indices[item[0][2]],
            frame_indices[item[0][0]],
            item[0][1],
            item[0][3],
            item[1],
        )
    )
    return tuple(
        TrackerDisagreementLinkReview(
            link_id=(
                "link_"
                + _canonical_sha256(
                    {
                        "comparison_sha256": comparison.fingerprint,
                        "link": link,
                        "proposed_by": proposed_by,
                    }
                )[:24]
            ),
            previous_frame_id=link[0],
            previous_detection_id=link[1],
            current_frame_id=link[2],
            current_detection_id=link[3],
            proposed_by=proposed_by,
            link_kind=(
                "continuous" if frame_indices[link[2]] - frame_indices[link[0]] == 1 else "recovery"
            ),
            review_state="pending",
        )
        for link, proposed_by in raw
    )


@dataclass(frozen=True, slots=True)
class TrackerDisagreementReview:
    detection_batch_sha256: str
    comparison_sha256: str
    left_prediction_sha256: str
    right_prediction_sha256: str
    review_status: str
    links: tuple[TrackerDisagreementLinkReview, ...]
    reviews: tuple[TrackerReviewer, ...]
    human_labels: bool
    scope: str = "pairwise_disagreement_links"
    model_generated_labels: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TrackerEvaluationError("unsupported tracker disagreement review version")
        for value, field in (
            (self.detection_batch_sha256, "detection batch sha256"),
            (self.comparison_sha256, "comparison sha256"),
            (self.left_prediction_sha256, "left prediction sha256"),
            (self.right_prediction_sha256, "right prediction sha256"),
        ):
            _sha256(value, f"tracker disagreement review {field}")
        if self.review_status not in {"pending", "approved"}:
            raise TrackerEvaluationError("tracker disagreement review_status is invalid")
        if (
            not isinstance(self.links, tuple)
            or len(self.links) > MAX_TRACKER_EVAL_ASSIGNMENTS * 2
            or any(not isinstance(item, TrackerDisagreementLinkReview) for item in self.links)
            or len({item.link_id for item in self.links}) != len(self.links)
            or len(
                {
                    (
                        item.previous_frame_id,
                        item.previous_detection_id,
                        item.current_frame_id,
                        item.current_detection_id,
                    )
                    for item in self.links
                }
            )
            != len(self.links)
        ):
            raise TrackerEvaluationError("tracker disagreement review links are invalid")
        if (
            not isinstance(self.reviews, tuple)
            or len(self.reviews) > 5
            or any(not isinstance(item, TrackerReviewer) for item in self.reviews)
            or len({item.reviewer_id for item in self.reviews}) != len(self.reviews)
        ):
            raise TrackerEvaluationError("tracker disagreement reviewers are invalid")
        if type(self.human_labels) is not bool:
            raise TrackerEvaluationError("tracker disagreement human_labels must be boolean")
        if self.scope != "pairwise_disagreement_links":
            raise TrackerEvaluationError("tracker disagreement scope is invalid")
        for field in (
            "model_generated_labels",
            "source_media_included",
            "paths_disclosed",
        ):
            if getattr(self, field) is not False:
                raise TrackerEvaluationError(f"tracker disagreement review {field} must be false")
        if self.review_status == "pending":
            if (
                self.human_labels
                or self.reviews
                or any(item.review_state != "pending" for item in self.links)
            ):
                raise TrackerEvaluationError(
                    "pending tracker disagreement review cannot contain labels"
                )
        elif (
            not self.human_labels
            or not self.reviews
            or any(item.review_state == "pending" for item in self.links)
        ):
            raise TrackerEvaluationError(
                "approved tracker disagreement review requires completed human labels"
            )

    @classmethod
    def pending(
        cls,
        batch: DetectionFrameBatch,
        left: TrackerPredictionBatch,
        right: TrackerPredictionBatch,
        comparison: TrackerPredictionComparison,
    ) -> TrackerDisagreementReview:
        links = _disagreement_link_specs(batch, left, right, comparison)
        return cls(
            detection_batch_sha256=batch.fingerprint,
            comparison_sha256=comparison.fingerprint,
            left_prediction_sha256=left.fingerprint,
            right_prediction_sha256=right.fingerprint,
            review_status="pending",
            links=links,
            reviews=(),
            human_labels=False,
        )

    @classmethod
    def from_dict(cls, value: Any) -> TrackerDisagreementReview:
        data = _exact_object(
            value,
            {
                "version",
                "detection_batch_sha256",
                "comparison_sha256",
                "left_prediction_sha256",
                "right_prediction_sha256",
                "review_status",
                "links",
                "reviews",
                "human_labels",
                "scope",
                "model_generated_labels",
                "source_media_included",
                "paths_disclosed",
            },
            "tracker disagreement review",
        )
        links = data["links"]
        reviews = data["reviews"]
        if not isinstance(links, list) or not isinstance(reviews, list):
            raise TrackerEvaluationError("tracker disagreement review arrays are invalid")
        return cls(
            **{
                **data,
                "links": tuple(TrackerDisagreementLinkReview.from_dict(item) for item in links),
                "reviews": tuple(TrackerReviewer.from_dict(item) for item in reviews),
            }
        )

    @classmethod
    def read(cls, path: str | Path) -> TrackerDisagreementReview:
        return cls.from_dict(_read_json(path, "tracker disagreement review"))

    def bind(
        self,
        batch: DetectionFrameBatch,
        left: TrackerPredictionBatch,
        right: TrackerPredictionBatch,
        comparison: TrackerPredictionComparison,
    ) -> None:
        expected = _disagreement_link_specs(batch, left, right, comparison)
        if (
            self.detection_batch_sha256 != batch.fingerprint
            or self.comparison_sha256 != comparison.fingerprint
            or self.left_prediction_sha256 != left.fingerprint
            or self.right_prediction_sha256 != right.fingerprint
            or tuple(item.identity for item in self.links)
            != tuple(item.identity for item in expected)
        ):
            raise TrackerEvaluationError(
                "tracker disagreement review belongs to different predictions"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "detection_batch_sha256": self.detection_batch_sha256,
            "comparison_sha256": self.comparison_sha256,
            "left_prediction_sha256": self.left_prediction_sha256,
            "right_prediction_sha256": self.right_prediction_sha256,
            "review_status": self.review_status,
            "links": [item.to_dict() for item in self.links],
            "reviews": [item.to_dict() for item in self.reviews],
            "human_labels": self.human_labels,
            "scope": self.scope,
            "model_generated_labels": self.model_generated_labels,
            "source_media_included": self.source_media_included,
            "paths_disclosed": self.paths_disclosed,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())

    def write_new(self, path: str | Path) -> None:
        _write_new_json(path, self.to_dict(), "tracker disagreement review")

    def to_public_dict(self) -> dict[str, object]:
        states = {
            state: sum(item.review_state == state for item in self.links)
            for state in sorted(_DISAGREEMENT_REVIEW_STATES)
        }
        return {
            "valid": True,
            "version": self.version,
            "review_status": self.review_status,
            "detection_batch_sha256": self.detection_batch_sha256,
            "comparison_sha256": self.comparison_sha256,
            "review_sha256": self.fingerprint,
            "links": len(self.links),
            "link_states": states,
            "reviewers": len(self.reviews),
            "human_labels": self.human_labels,
            "model_generated_labels": False,
            "source_media_included": False,
            "paths_disclosed": False,
        }


@dataclass(frozen=True, slots=True)
class TrackerDisagreementReport:
    detection_batch_sha256: str
    comparison_sha256: str
    review_sha256: str
    left_prediction_sha256: str
    right_prediction_sha256: str
    left_tracker_adapter: str
    right_tracker_adapter: str
    evidence_status: str
    disagreement_link_count: int
    reviewed_link_count: int
    unreviewable_link_count: int
    same_subject_link_count: int
    different_subject_link_count: int
    left_exclusive_link_count: int
    right_exclusive_link_count: int
    left_disagreement_error_count: int
    right_disagreement_error_count: int
    left_disagreement_error_rate: float | None
    right_disagreement_error_rate: float | None
    relative_preference: str
    reviewer_count: int
    scope: str = "pairwise_disagreement_links"
    human_ground_truth: bool = True
    overall_accuracy_evaluated: bool = False
    shared_links_evaluated: bool = False
    global_adapter_selected: bool = False
    subject_identity_inference: bool = False
    source_media_included: bool = False
    paths_disclosed: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise TrackerEvaluationError("unsupported tracker disagreement report version")
        for value, field in (
            (self.detection_batch_sha256, "detection batch sha256"),
            (self.comparison_sha256, "comparison sha256"),
            (self.review_sha256, "review sha256"),
            (self.left_prediction_sha256, "left prediction sha256"),
            (self.right_prediction_sha256, "right prediction sha256"),
        ):
            _sha256(value, f"tracker disagreement report {field}")
        _safe_id(
            self.left_tracker_adapter,
            "tracker disagreement report left tracker adapter",
        )
        _safe_id(
            self.right_tracker_adapter,
            "tracker disagreement report right tracker adapter",
        )
        if self.evidence_status not in _DISAGREEMENT_EVIDENCE_STATUSES:
            raise TrackerEvaluationError("tracker disagreement report evidence_status is invalid")
        for field in (
            "disagreement_link_count",
            "reviewed_link_count",
            "unreviewable_link_count",
            "same_subject_link_count",
            "different_subject_link_count",
            "left_exclusive_link_count",
            "right_exclusive_link_count",
            "left_disagreement_error_count",
            "right_disagreement_error_count",
            "reviewer_count",
        ):
            _integer(
                getattr(self, field),
                f"tracker disagreement report {field}",
                minimum=1 if field == "reviewer_count" else 0,
                maximum=MAX_TRACKER_EVAL_ASSIGNMENTS * 2,
            )
        if (
            self.reviewed_link_count + self.unreviewable_link_count != self.disagreement_link_count
            or self.same_subject_link_count + self.different_subject_link_count
            != self.reviewed_link_count
            or self.left_exclusive_link_count + self.right_exclusive_link_count
            != self.disagreement_link_count
            or self.left_disagreement_error_count + self.right_disagreement_error_count
            != self.reviewed_link_count
        ):
            raise TrackerEvaluationError("tracker disagreement report counts are inconsistent")
        left_rate = _optional_ratio(
            self.left_disagreement_error_rate,
            "tracker disagreement report left error rate",
        )
        right_rate = _optional_ratio(
            self.right_disagreement_error_rate,
            "tracker disagreement report right error rate",
        )
        if left_rate != _ratio(
            self.left_disagreement_error_count, self.reviewed_link_count
        ) or right_rate != _ratio(self.right_disagreement_error_count, self.reviewed_link_count):
            raise TrackerEvaluationError("tracker disagreement report error rates are inconsistent")
        object.__setattr__(self, "left_disagreement_error_rate", left_rate)
        object.__setattr__(self, "right_disagreement_error_rate", right_rate)
        if self.reviewed_link_count == 0:
            expected_status = "no_reviewable_links"
            expected_preference = "insufficient_evidence"
        elif self.left_disagreement_error_count == self.right_disagreement_error_count:
            expected_status = "tie"
            expected_preference = "tie"
        elif self.left_disagreement_error_count < self.right_disagreement_error_count:
            expected_status = "relative_difference_available"
            expected_preference = "left"
        else:
            expected_status = "relative_difference_available"
            expected_preference = "right"
        if (
            self.evidence_status != expected_status
            or self.relative_preference not in _DISAGREEMENT_PREFERENCES
            or self.relative_preference != expected_preference
        ):
            raise TrackerEvaluationError("tracker disagreement report preference is inconsistent")
        if self.scope != "pairwise_disagreement_links":
            raise TrackerEvaluationError("tracker disagreement report scope is invalid")
        if self.human_ground_truth is not True:
            raise TrackerEvaluationError("tracker disagreement report requires human ground truth")
        for field in (
            "overall_accuracy_evaluated",
            "shared_links_evaluated",
            "global_adapter_selected",
            "subject_identity_inference",
            "source_media_included",
            "paths_disclosed",
        ):
            if getattr(self, field) is not False:
                raise TrackerEvaluationError(f"tracker disagreement report {field} must be false")

    @classmethod
    def from_dict(cls, value: Any) -> TrackerDisagreementReport:
        fields = {
            "version",
            "detection_batch_sha256",
            "comparison_sha256",
            "review_sha256",
            "left_prediction_sha256",
            "right_prediction_sha256",
            "left_tracker_adapter",
            "right_tracker_adapter",
            "evidence_status",
            "disagreement_link_count",
            "reviewed_link_count",
            "unreviewable_link_count",
            "same_subject_link_count",
            "different_subject_link_count",
            "left_exclusive_link_count",
            "right_exclusive_link_count",
            "left_disagreement_error_count",
            "right_disagreement_error_count",
            "left_disagreement_error_rate",
            "right_disagreement_error_rate",
            "relative_preference",
            "reviewer_count",
            "scope",
            "human_ground_truth",
            "overall_accuracy_evaluated",
            "shared_links_evaluated",
            "global_adapter_selected",
            "subject_identity_inference",
            "source_media_included",
            "paths_disclosed",
        }
        return cls(**_exact_object(value, fields, "tracker disagreement report"))

    @classmethod
    def read(cls, path: str | Path) -> TrackerDisagreementReport:
        return cls.from_dict(_read_json(path, "tracker disagreement report"))

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "version",
                "detection_batch_sha256",
                "comparison_sha256",
                "review_sha256",
                "left_prediction_sha256",
                "right_prediction_sha256",
                "left_tracker_adapter",
                "right_tracker_adapter",
                "evidence_status",
                "disagreement_link_count",
                "reviewed_link_count",
                "unreviewable_link_count",
                "same_subject_link_count",
                "different_subject_link_count",
                "left_exclusive_link_count",
                "right_exclusive_link_count",
                "left_disagreement_error_count",
                "right_disagreement_error_count",
                "left_disagreement_error_rate",
                "right_disagreement_error_rate",
                "relative_preference",
                "reviewer_count",
                "scope",
                "human_ground_truth",
                "overall_accuracy_evaluated",
                "shared_links_evaluated",
                "global_adapter_selected",
                "subject_identity_inference",
                "source_media_included",
                "paths_disclosed",
            )
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())

    def write_new(self, path: str | Path) -> None:
        _write_new_json(path, self.to_dict(), "tracker disagreement report")

    def to_public_dict(self) -> dict[str, object]:
        return {**self.to_dict(), "report_sha256": self.fingerprint}


def evaluate_tracker_disagreement(
    batch: DetectionFrameBatch,
    left: TrackerPredictionBatch,
    right: TrackerPredictionBatch,
    comparison: TrackerPredictionComparison,
    review: TrackerDisagreementReview,
) -> TrackerDisagreementReport:
    review.bind(batch, left, right, comparison)
    if review.review_status != "approved" or not review.human_labels:
        raise TrackerEvaluationError(
            "tracker disagreement evaluation requires approved human labels"
        )
    same_subject = sum(item.review_state == "same_subject" for item in review.links)
    different_subject = sum(item.review_state == "different_subject" for item in review.links)
    unreviewable = sum(item.review_state == "unreviewable" for item in review.links)
    left_errors = 0
    right_errors = 0
    for item in review.links:
        if item.review_state == "unreviewable":
            continue
        proposer_is_correct = item.review_state == "same_subject"
        if item.proposed_by == "left":
            left_errors += not proposer_is_correct
            right_errors += proposer_is_correct
        else:
            right_errors += not proposer_is_correct
            left_errors += proposer_is_correct
    reviewed = same_subject + different_subject
    if reviewed == 0:
        evidence_status = "no_reviewable_links"
        preference = "insufficient_evidence"
    elif left_errors == right_errors:
        evidence_status = "tie"
        preference = "tie"
    elif left_errors < right_errors:
        evidence_status = "relative_difference_available"
        preference = "left"
    else:
        evidence_status = "relative_difference_available"
        preference = "right"
    return TrackerDisagreementReport(
        detection_batch_sha256=batch.fingerprint,
        comparison_sha256=comparison.fingerprint,
        review_sha256=review.fingerprint,
        left_prediction_sha256=left.fingerprint,
        right_prediction_sha256=right.fingerprint,
        left_tracker_adapter=comparison.left_tracker_adapter,
        right_tracker_adapter=comparison.right_tracker_adapter,
        evidence_status=evidence_status,
        disagreement_link_count=len(review.links),
        reviewed_link_count=reviewed,
        unreviewable_link_count=unreviewable,
        same_subject_link_count=same_subject,
        different_subject_link_count=different_subject,
        left_exclusive_link_count=sum(item.proposed_by == "left" for item in review.links),
        right_exclusive_link_count=sum(item.proposed_by == "right" for item in review.links),
        left_disagreement_error_count=left_errors,
        right_disagreement_error_count=right_errors,
        left_disagreement_error_rate=_ratio(left_errors, reviewed),
        right_disagreement_error_rate=_ratio(right_errors, reviewed),
        relative_preference=preference,
        reviewer_count=len(review.reviews),
    )
