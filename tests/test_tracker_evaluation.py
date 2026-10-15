from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.motion_tracker import MotionTrackerConfig
from baby_monitor.object_tracker import (
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
    IoUTrackerConfig,
)
from baby_monitor.tracker_evaluation import (
    TrackerAssociationReport,
    TrackerAssociationReview,
    TrackerEvaluationError,
    TrackerPredictionBatch,
    TrackerPredictionComparison,
    TrackerReviewAssignment,
    TrackerReviewer,
    build_iou_tracker_predictions,
    build_motion_tracker_predictions,
    compare_tracker_predictions,
    evaluate_tracker_association,
)


class TrackerEvaluationTests(unittest.TestCase):
    BASE = datetime(2026, 8, 8, 1, 2, 3, tzinfo=UTC)

    @staticmethod
    def _detection(
        detection_id: str,
        bbox: tuple[float, float, float, float],
        *,
        label: str = "person",
    ) -> FrameDetection:
        return FrameDetection(
            detection_id=detection_id,
            label=label,
            confidence=0.9,
            bbox_xyxy=bbox,
        )

    @classmethod
    def _frame(
        cls,
        index: int,
        *detections: FrameDetection,
    ) -> DetectionFrame:
        return DetectionFrame(
            frame_id=f"frame_{index:04d}",
            occurred_at=cls.BASE + timedelta(seconds=index),
            detections=tuple(detections),
        )

    @classmethod
    def _two_subject_batch(cls) -> DetectionFrameBatch:
        return DetectionFrameBatch(
            camera_id="synthetic_track_cam",
            detector_adapter="synthetic_detector",
            frames=(
                cls._frame(
                    0,
                    cls._detection("a0", (0.1, 0.1, 0.3, 0.8)),
                    cls._detection("b0", (0.7, 0.1, 0.9, 0.8)),
                ),
                cls._frame(
                    1,
                    cls._detection("a1", (0.12, 0.1, 0.32, 0.8)),
                    cls._detection("b1", (0.68, 0.1, 0.88, 0.8)),
                ),
                cls._frame(
                    2,
                    cls._detection("a2", (0.14, 0.1, 0.34, 0.8)),
                    cls._detection("b2", (0.66, 0.1, 0.86, 0.8)),
                ),
            ),
        )

    @classmethod
    def _approved_review(
        cls,
        batch: DetectionFrameBatch,
        *,
        states: dict[tuple[str, str], tuple[str, str | None]] | None = None,
    ) -> TrackerAssociationReview:
        assignments = []
        for frame in batch.frames:
            for detection in frame.detections:
                default_subject = (
                    "subject_001" if detection.detection_id.startswith("a") else "subject_002"
                )
                review_state, subject_id = (states or {}).get(
                    (frame.frame_id, detection.detection_id),
                    ("assigned", default_subject),
                )
                assignments.append(
                    TrackerReviewAssignment(
                        frame_id=frame.frame_id,
                        detection_id=detection.detection_id,
                        review_state=review_state,
                        subject_id=subject_id,
                    )
                )
        review = TrackerAssociationReview(
            camera_id=batch.camera_id,
            detector_adapter=batch.detector_adapter,
            detection_batch_sha256=batch.fingerprint,
            source_admission_sha256=None,
            review_status="approved",
            assignments=tuple(assignments),
            reviews=(
                TrackerReviewer(
                    reviewer_id="reviewer_01",
                    reviewed_at=cls.BASE,
                ),
            ),
            human_labels=True,
        )
        review.bind(batch)
        return review

    def test_iou_predictions_and_perfect_association_report(self) -> None:
        batch = self._two_subject_batch()
        review = self._approved_review(batch)
        prediction = build_iou_tracker_predictions(batch, IoUTrackerConfig())
        bounded_outbox_prediction = build_iou_tracker_predictions(
            batch,
            IoUTrackerConfig(max_pending_observations=2),
        )
        report = evaluate_tracker_association(batch, review, prediction)

        self.assertEqual(6, len(bounded_outbox_prediction.assignments))
        self.assertEqual("association_metrics_available", report.evidence_status)
        self.assertEqual(6, report.eligible_assignment_count)
        self.assertEqual(2, report.ground_truth_subject_count)
        self.assertEqual(2, report.predicted_track_count)
        self.assertEqual(4, report.eligible_link_count)
        self.assertEqual(4, report.continuous_link_count)
        self.assertEqual(0, report.recovery_link_count)
        self.assertEqual(0, report.id_switch_count)
        self.assertEqual(0.0, report.id_switch_rate)
        self.assertEqual(0, report.fragmented_subject_count)
        self.assertEqual(0.0, report.fragmentation_rate)
        self.assertEqual(0, report.merged_predicted_track_count)
        self.assertFalse(report.detection_accuracy_evaluated)
        self.assertFalse(report.missed_detections_evaluated)
        self.assertFalse(report.subject_identity_inference)

    def test_fragment_switch_and_merge_are_measured_separately(self) -> None:
        batch = self._two_subject_batch()
        review = self._approved_review(batch)
        prediction = build_iou_tracker_predictions(batch, IoUTrackerConfig())
        value = prediction.to_dict()
        assignments = value["assignments"]
        assert isinstance(assignments, list)
        assignments[2]["tracking_id"] = "track_fragment"  # type: ignore[index]
        assignments[4]["tracking_id"] = "track_a_new"  # type: ignore[index]
        assignments[5]["tracking_id"] = "track_fragment"  # type: ignore[index]
        modified = TrackerPredictionBatch.from_dict(value)
        report = evaluate_tracker_association(batch, review, modified)

        self.assertEqual(2, report.fragmented_subject_count)
        self.assertEqual(3, report.id_switch_count)
        self.assertEqual(1, report.merged_predicted_track_count)
        self.assertEqual(0, report.recovery_link_count)

    def test_recovery_switch_is_not_mixed_with_continuous_switch(self) -> None:
        batch = DetectionFrameBatch(
            camera_id="synthetic_track_cam",
            detector_adapter="synthetic_detector",
            frames=(
                self._frame(0, self._detection("a0", (0.1, 0.1, 0.3, 0.8))),
                self._frame(1),
                self._frame(2, self._detection("a2", (0.12, 0.1, 0.32, 0.8))),
            ),
        )
        review = self._approved_review(batch)
        prediction = build_iou_tracker_predictions(
            batch,
            IoUTrackerConfig(max_track_age_seconds=1.0),
        )
        report = evaluate_tracker_association(batch, review, prediction)
        self.assertEqual(1, report.eligible_link_count)
        self.assertEqual(0, report.continuous_link_count)
        self.assertEqual(1, report.recovery_link_count)
        self.assertEqual(1, report.recovery_id_switch_count)
        self.assertEqual(1, report.id_switch_count)

    def test_duplicate_false_positive_and_unreviewable_are_outside_association(self) -> None:
        batch = self._two_subject_batch()
        states = {
            ("frame_0000", "b0"): ("false_positive", None),
            ("frame_0001", "b1"): ("unreviewable", None),
            ("frame_0002", "b2"): ("duplicate", "subject_001"),
        }
        review = self._approved_review(batch, states=states)
        report = evaluate_tracker_association(
            batch,
            review,
            build_iou_tracker_predictions(batch, IoUTrackerConfig()),
        )
        self.assertEqual(3, report.eligible_assignment_count)
        self.assertEqual(1, report.duplicate_detection_count)
        self.assertEqual(1, report.false_positive_detection_count)
        self.assertEqual(1, report.unreviewable_detection_count)
        self.assertEqual(1, report.ground_truth_subject_count)

    def test_duplicate_requires_primary_and_subject_ids_are_anonymous(self) -> None:
        batch = self._two_subject_batch()
        with self.assertRaisesRegex(TrackerEvaluationError, "anonymous"):
            TrackerReviewAssignment(
                frame_id="frame_0000",
                detection_id="a0",
                review_state="assigned",
                subject_id="father",
            )
        with self.assertRaisesRegex(TrackerEvaluationError, "anonymous reviewer_"):
            TrackerReviewer("Alice", self.BASE)

        states = {
            ("frame_0000", "a0"): ("false_positive", None),
            ("frame_0000", "b0"): ("duplicate", "subject_001"),
        }
        with self.assertRaisesRegex(TrackerEvaluationError, "primary"):
            self._approved_review(batch, states=states)

    def test_pending_review_cannot_be_evaluated_or_claim_human_labels(self) -> None:
        batch = self._two_subject_batch()
        pending = TrackerAssociationReview.pending(
            batch,
            source_admission_sha256="a" * 64,
        )
        self.assertEqual("pending", pending.review_status)
        self.assertFalse(pending.human_labels)
        self.assertTrue(all(item.review_state == "pending" for item in pending.assignments))
        with self.assertRaisesRegex(TrackerEvaluationError, "approved human"):
            evaluate_tracker_association(
                batch,
                pending,
                build_iou_tracker_predictions(batch, IoUTrackerConfig()),
            )

        value = pending.to_dict()
        value["human_labels"] = True
        with self.assertRaisesRegex(TrackerEvaluationError, "pending"):
            TrackerAssociationReview.from_dict(value)

    def test_binding_rejects_different_batch_order_hash_and_missing_assignment(self) -> None:
        batch = self._two_subject_batch()
        review = self._approved_review(batch)
        prediction = build_iou_tracker_predictions(batch, IoUTrackerConfig())

        changed = DetectionFrameBatch(
            camera_id=batch.camera_id,
            detector_adapter=batch.detector_adapter,
            frames=(
                DetectionFrame(
                    frame_id=batch.frames[0].frame_id,
                    occurred_at=batch.frames[0].occurred_at,
                    detections=(
                        FrameDetection(
                            detection_id="a0",
                            label="person",
                            confidence=0.8,
                            bbox_xyxy=(0.1, 0.1, 0.3, 0.8),
                        ),
                        batch.frames[0].detections[1],
                    ),
                ),
                *batch.frames[1:],
            ),
        )
        with self.assertRaisesRegex(TrackerEvaluationError, "different detection batch"):
            review.bind(changed)
        with self.assertRaisesRegex(TrackerEvaluationError, "different detection batch"):
            prediction.bind(changed)

        value = prediction.to_dict()
        value["assignments"] = value["assignments"][:-1]  # type: ignore[index]
        incomplete = TrackerPredictionBatch.from_dict(value)
        with self.assertRaisesRegex(TrackerEvaluationError, "different detection batch"):
            incomplete.bind(batch)

    def test_prediction_rejects_same_frame_track_reuse_and_label_changes(self) -> None:
        batch = self._two_subject_batch()
        prediction = build_iou_tracker_predictions(batch, IoUTrackerConfig())
        reused = prediction.to_dict()
        assignments = reused["assignments"]
        assert isinstance(assignments, list)
        assignments[1]["tracking_id"] = assignments[0]["tracking_id"]  # type: ignore[index]
        with self.assertRaisesRegex(TrackerEvaluationError, "within one frame"):
            TrackerPredictionBatch.from_dict(reused)

        mixed_labels = DetectionFrameBatch(
            camera_id="synthetic_track_cam",
            detector_adapter="synthetic_detector",
            frames=(
                self._frame(
                    0,
                    self._detection("person0", (0.1, 0.1, 0.3, 0.8)),
                ),
                self._frame(
                    1,
                    self._detection(
                        "dog1",
                        (0.12, 0.1, 0.32, 0.8),
                        label="dog",
                    ),
                ),
            ),
        )
        mixed_prediction = build_iou_tracker_predictions(
            mixed_labels,
            IoUTrackerConfig(),
        )
        changed = mixed_prediction.to_dict()
        changed_assignments = changed["assignments"]
        assert isinstance(changed_assignments, list)
        changed_assignments[1]["tracking_id"] = changed_assignments[0][  # type: ignore[index]
            "tracking_id"
        ]
        invalid = TrackerPredictionBatch.from_dict(changed)
        with self.assertRaisesRegex(TrackerEvaluationError, "changes a track label"):
            invalid.bind(mixed_labels)

    def test_prediction_comparison_is_track_id_invariant_and_not_accuracy(self) -> None:
        batch = self._two_subject_batch()
        left = build_iou_tracker_predictions(batch, IoUTrackerConfig())
        renamed_value = left.to_dict()
        renamed_assignments = renamed_value["assignments"]
        assert isinstance(renamed_assignments, list)
        rename: dict[str, str] = {}
        for item in renamed_assignments:
            tracking_id = item["tracking_id"]  # type: ignore[index]
            item["tracking_id"] = rename.setdefault(  # type: ignore[index]
                tracking_id,
                f"renamed_{len(rename):02d}",
            )
        renamed = TrackerPredictionBatch.from_dict(renamed_value)
        same_partition = compare_tracker_predictions(batch, left, renamed)
        self.assertEqual(0, same_partition.left_only_link_count)
        self.assertEqual(0, same_partition.right_only_link_count)
        self.assertEqual(0, same_partition.disagreement_frame_count)

        changed_value = left.to_dict()
        changed_assignments = changed_value["assignments"]
        assert isinstance(changed_assignments, list)
        changed_assignments[2]["tracking_id"] = "track_new_a"  # type: ignore[index]
        changed_assignments[4]["tracking_id"] = "track_new_a"  # type: ignore[index]
        changed = TrackerPredictionBatch.from_dict(changed_value)
        comparison = compare_tracker_predictions(batch, left, changed)
        self.assertEqual(4, comparison.left_link_count)
        self.assertEqual(3, comparison.right_link_count)
        self.assertEqual(3, comparison.shared_link_count)
        self.assertEqual(1, comparison.left_only_link_count)
        self.assertEqual(0, comparison.right_only_link_count)
        self.assertEqual(1, comparison.continuous_disagreement_count)
        self.assertEqual(0, comparison.recovery_disagreement_count)
        self.assertEqual(
            ("frame_0000", "frame_0001"),
            comparison.disagreement_frame_ids,
        )
        self.assertFalse(comparison.human_ground_truth_used)
        self.assertFalse(comparison.accuracy_evaluated)
        self.assertFalse(comparison.better_adapter_selected)

    def test_prediction_comparison_separates_recovery_disagreement(self) -> None:
        batch = DetectionFrameBatch(
            camera_id="synthetic_track_cam",
            detector_adapter="synthetic_detector",
            frames=(
                self._frame(0, self._detection("a0", (0.1, 0.1, 0.3, 0.8))),
                self._frame(1),
                self._frame(2, self._detection("a2", (0.1, 0.1, 0.3, 0.8))),
            ),
        )
        left = build_iou_tracker_predictions(batch, IoUTrackerConfig())
        right_value = left.to_dict()
        right_assignments = right_value["assignments"]
        assert isinstance(right_assignments, list)
        right_assignments[1]["tracking_id"] = "track_after_gap"  # type: ignore[index]
        right = TrackerPredictionBatch.from_dict(right_value)
        comparison = compare_tracker_predictions(batch, left, right)
        self.assertEqual(0, comparison.continuous_disagreement_count)
        self.assertEqual(1, comparison.recovery_disagreement_count)

    def test_report_and_contracts_are_strict_and_match_public_schemas(self) -> None:
        batch = self._two_subject_batch()
        review = self._approved_review(batch)
        prediction = build_iou_tracker_predictions(batch, IoUTrackerConfig())
        report = evaluate_tracker_association(batch, review, prediction)
        comparison = compare_tracker_predictions(
            batch,
            prediction,
            build_motion_tracker_predictions(batch, MotionTrackerConfig()),
        )
        self.assertEqual(report, TrackerAssociationReport.from_dict(report.to_dict()))
        self.assertEqual(
            comparison,
            TrackerPredictionComparison.from_dict(comparison.to_dict()),
        )

        for schema_name, value in (
            ("tracker-association-review-v1.schema.json", review.to_dict()),
            ("tracker-prediction-batch-v1.schema.json", prediction.to_dict()),
            (
                "tracker-prediction-comparison-v1.schema.json",
                comparison.to_dict(),
            ),
            ("tracker-association-report-v1.schema.json", report.to_dict()),
        ):
            schema = json.loads(Path("schemas", schema_name).read_text(encoding="utf-8"))
            self.assertEqual(set(schema["required"]), set(value))

        tampered = report.to_dict()
        tampered["id_switch_rate"] = 0.5
        with self.assertRaisesRegex(TrackerEvaluationError, "id_switch_rate"):
            TrackerAssociationReport.from_dict(tampered)
        extra = review.to_dict()
        extra["subject_names"] = ["private"]
        with self.assertRaisesRegex(TrackerEvaluationError, "fields do not match"):
            TrackerAssociationReview.from_dict(extra)

    def test_public_review_prediction_and_report_fixtures_are_bound(self) -> None:
        batch = DetectionFrameBatch.read("tests/fixtures/valid_detection_frame_batch.json")
        review = TrackerAssociationReview.read(
            "tests/fixtures/valid_tracker_association_review.json"
        )
        predictions = TrackerPredictionBatch.read(
            "tests/fixtures/valid_tracker_prediction_batch.json"
        )
        motion_predictions = TrackerPredictionBatch.read(
            "tests/fixtures/valid_motion_tracker_prediction_batch.json"
        )
        comparison = TrackerPredictionComparison.read(
            "tests/fixtures/valid_tracker_prediction_comparison.json"
        )
        report = TrackerAssociationReport.read(
            "tests/fixtures/valid_tracker_association_report.json"
        )
        review.bind(batch)
        predictions.bind(batch)
        motion_predictions.bind(batch)
        self.assertEqual(
            motion_predictions,
            build_motion_tracker_predictions(
                batch,
                MotionTrackerConfig.read("config/motion-two-stage-tracker.example.json"),
            ),
        )
        self.assertEqual(
            comparison,
            compare_tracker_predictions(batch, predictions, motion_predictions),
        )
        self.assertEqual(
            report,
            evaluate_tracker_association(batch, review, predictions),
        )

    def test_cli_prepares_predicts_validates_and_scores_path_free(self) -> None:
        batch = self._two_subject_batch()
        review = self._approved_review(batch)
        with tempfile.TemporaryDirectory(prefix="PRIVATE_TRACK_EVAL_") as directory:
            root = Path(directory)
            frames = root / "frames.json"
            pending = root / "pending.json"
            approved = root / "approved.json"
            predictions = root / "predictions.json"
            report = root / "report.json"
            config = root / "tracker.json"
            frames.write_text(json.dumps(batch.to_dict()), encoding="utf-8")
            approved.write_text(json.dumps(review.to_dict()), encoding="utf-8")
            config.write_text(json.dumps(IoUTrackerConfig().to_dict()), encoding="utf-8")

            outputs = []
            commands = (
                [
                    "prepare-tracker-review",
                    "--frames",
                    str(frames),
                    "--source-admission-sha256",
                    "a" * 64,
                    "--output",
                    str(pending),
                ],
                [
                    "validate-tracker-review",
                    "--frames",
                    str(frames),
                    "--review",
                    str(approved),
                ],
                [
                    "track-iou-predict",
                    "--tracker-config",
                    str(config),
                    "--frames",
                    str(frames),
                    "--output",
                    str(predictions),
                ],
            )
            for command in commands:
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(0, main(command))
                outputs.append(output.getvalue())

            evaluate_output = io.StringIO()
            with redirect_stdout(evaluate_output):
                self.assertEqual(
                    0,
                    main(
                        [
                            "evaluate-tracker-association",
                            "--frames",
                            str(frames),
                            "--review",
                            str(approved),
                            "--predictions",
                            str(predictions),
                            "--output",
                            str(report),
                        ]
                    ),
                )
            outputs.append(evaluate_output.getvalue())

            validate_report = io.StringIO()
            with redirect_stdout(validate_report):
                self.assertEqual(
                    0,
                    main(["validate-tracker-association", str(report)]),
                )
            outputs.append(validate_report.getvalue())
            combined = "".join(outputs)
            self.assertNotIn(str(root), combined)
            self.assertNotIn("subject_001", combined)
            self.assertTrue(json.loads(evaluate_output.getvalue())["human_ground_truth"])

            errors = io.StringIO()
            with redirect_stderr(errors):
                self.assertEqual(2, main(commands[0]))
            self.assertNotIn(str(root), errors.getvalue())

            if os.name == "posix":
                link = root / "review-link.json"
                link.symlink_to(approved)
                with self.assertRaisesRegex(TrackerEvaluationError, "missing or unsafe"):
                    TrackerAssociationReview.read(link)

    def test_comparison_cli_is_path_free_exclusive_and_validates(self) -> None:
        batch = self._two_subject_batch()
        left = build_iou_tracker_predictions(batch, IoUTrackerConfig())
        right = build_motion_tracker_predictions(batch, MotionTrackerConfig())
        with tempfile.TemporaryDirectory(prefix="PRIVATE_TRACK_COMPARE_") as directory:
            root = Path(directory)
            frames = root / "frames.json"
            left_path = root / "left.json"
            right_path = root / "right.json"
            output = root / "comparison.json"
            frames.write_text(json.dumps(batch.to_dict()), encoding="utf-8")
            left_path.write_text(json.dumps(left.to_dict()), encoding="utf-8")
            right_path.write_text(json.dumps(right.to_dict()), encoding="utf-8")

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "compare-tracker-predictions",
                            "--frames",
                            str(frames),
                            "--left",
                            str(left_path),
                            "--right",
                            str(right_path),
                            "--output",
                            str(output),
                        ]
                    ),
                )
            self.assertNotIn(str(root), stdout.getvalue())
            self.assertFalse(json.loads(stdout.getvalue())["accuracy_evaluated"])
            self.assertEqual(0o600, output.stat().st_mode & 0o777)

            validate_stdout = io.StringIO()
            with redirect_stdout(validate_stdout):
                self.assertEqual(
                    0,
                    main(["validate-tracker-comparison", str(output)]),
                )
            self.assertNotIn(str(root), validate_stdout.getvalue())

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(
                    2,
                    main(
                        [
                            "compare-tracker-predictions",
                            "--frames",
                            str(frames),
                            "--left",
                            str(left_path),
                            "--right",
                            str(right_path),
                            "--output",
                            str(output),
                        ]
                    ),
                )
            self.assertNotIn(str(root), stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
