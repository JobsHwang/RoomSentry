"""Tracker evaluation contracts: review, prediction, comparison, disagreement, association.

Historically a single ``tracker_evaluation.py``; the public surface is
re-exported here so existing imports keep working.
"""

from __future__ import annotations

from baby_monitor.tracker_evaluation._shared import (  # noqa: F401  # noqa: F401
    MAX_TRACKER_DISAGREEMENT_FRAMES,
    MAX_TRACKER_EVAL_ASSIGNMENTS,
    MAX_TRACKER_EVAL_BYTES,
    TrackerEvaluationError,
)
from baby_monitor.tracker_evaluation.association import (  # noqa: F401
    TrackerAssociationReport,
    evaluate_tracker_association,
)
from baby_monitor.tracker_evaluation.comparison import (  # noqa: F401
    TrackerPredictionComparison,
    compare_tracker_predictions,
)
from baby_monitor.tracker_evaluation.disagreement import (  # noqa: F401
    TrackerDisagreementLinkReview,
    TrackerDisagreementReport,
    TrackerDisagreementReview,
    evaluate_tracker_disagreement,
)
from baby_monitor.tracker_evaluation.prediction import (  # noqa: F401
    TrackerPredictionAssignment,
    TrackerPredictionBatch,
    build_iou_tracker_predictions,
    build_motion_tracker_predictions,
)
from baby_monitor.tracker_evaluation.review import (  # noqa: F401
    TrackerAssociationReview,
    TrackerReviewAssignment,
    TrackerReviewer,
)

__all__ = [
    "MAX_TRACKER_DISAGREEMENT_FRAMES",
    "MAX_TRACKER_EVAL_ASSIGNMENTS",
    "MAX_TRACKER_EVAL_BYTES",
    "TrackerAssociationReport",
    "TrackerAssociationReview",
    "TrackerDisagreementLinkReview",
    "TrackerDisagreementReport",
    "TrackerDisagreementReview",
    "TrackerEvaluationError",
    "TrackerPredictionAssignment",
    "TrackerPredictionBatch",
    "TrackerPredictionComparison",
    "TrackerReviewAssignment",
    "TrackerReviewer",
    "build_iou_tracker_predictions",
    "build_motion_tracker_predictions",
    "compare_tracker_predictions",
    "evaluate_tracker_association",
    "evaluate_tracker_disagreement",
]
