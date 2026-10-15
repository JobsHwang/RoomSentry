from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.object_tracker import (
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
)
from baby_monitor.tracker_evaluation import (
    TrackerDisagreementReport,
    TrackerDisagreementReview,
    TrackerEvaluationError,
    TrackerPredictionAssignment,
    TrackerPredictionBatch,
    TrackerPredictionComparison,
    TrackerReviewer,
    compare_tracker_predictions,
    evaluate_tracker_disagreement,
)


class TrackerDisagreementTests(unittest.TestCase):
    BASE = datetime(2026, 8, 10, 1, 2, 3, tzinfo=UTC)

    @classmethod
    def _batch(cls) -> DetectionFrameBatch:
        frames = []
        for index in range(2):
            frames.append(
                DetectionFrame(
                    frame_id=f"frame_{index:04d}",
                    occurred_at=cls.BASE + timedelta(seconds=index),
                    detections=(
                        FrameDetection(
                            detection_id=f"a{index}",
                            label="person",
                            confidence=0.9,
                            bbox_xyxy=(0.1, 0.1, 0.3, 0.8),
                        ),
                        FrameDetection(
                            detection_id=f"b{index}",
                            label="person",
                            confidence=0.9,
                            bbox_xyxy=(0.6, 0.1, 0.8, 0.8),
                        ),
                    ),
                )
            )
        return DetectionFrameBatch(
            camera_id="disagreement_cam",
            detector_adapter="synthetic_detector",
            frames=tuple(frames),
        )

    @staticmethod
    def _prediction(
        batch: DetectionFrameBatch,
        *,
        adapter: str,
        ids: tuple[str, str, str, str],
    ) -> TrackerPredictionBatch:
        keys = [
            (frame.frame_id, detection.detection_id)
            for frame in batch.frames
            for detection in frame.detections
        ]
        result = TrackerPredictionBatch(
            camera_id=batch.camera_id,
            detector_adapter=batch.detector_adapter,
            detection_batch_sha256=batch.fingerprint,
            tracker_adapter=adapter,
            tracker_revision="fixture_v1",
            tracker_config_sha256=("a" if adapter == "left_tracker" else "b") * 64,
            runtime_id="python_stdlib",
            assignments=tuple(
                TrackerPredictionAssignment(
                    frame_id=frame_id,
                    detection_id=detection_id,
                    tracking_id=tracking_id,
                )
                for (frame_id, detection_id), tracking_id in zip(
                    keys,
                    ids,
                    strict=True,
                )
            ),
        )
        result.bind(batch)
        return result

    @classmethod
    def _inputs(
        cls,
    ) -> tuple[
        DetectionFrameBatch,
        TrackerPredictionBatch,
        TrackerPredictionBatch,
    ]:
        batch = cls._batch()
        left = cls._prediction(
            batch,
            adapter="left_tracker",
            ids=("left_a", "left_b", "left_a", "left_b"),
        )
        right = cls._prediction(
            batch,
            adapter="right_tracker",
            ids=("right_a", "right_b", "right_b", "right_a"),
        )
        return batch, left, right

    @staticmethod
    def _approved(
        pending: TrackerDisagreementReview,
        *,
        all_unreviewable: bool = False,
        all_same: bool = False,
    ) -> TrackerDisagreementReview:
        value = pending.to_dict()
        links = value["links"]
        assert isinstance(links, list)
        for item in links:
            item["review_state"] = (  # type: ignore[index]
                "unreviewable"
                if all_unreviewable
                else (
                    "same_subject"
                    if all_same or item["proposed_by"] == "left"  # type: ignore[index]
                    else "different_subject"
                )
            )
        value["review_status"] = "approved"
        value["reviews"] = [
            TrackerReviewer(
                reviewer_id="reviewer_01",
                reviewed_at=datetime(2026, 8, 10, tzinfo=UTC),
            ).to_dict()
        ]
        value["human_labels"] = True
        return TrackerDisagreementReview.from_dict(value)

    def test_pending_review_is_exactly_bound_to_all_exclusive_links(self) -> None:
        batch, left, right = self._inputs()
        comparison = compare_tracker_predictions(batch, left, right)
        pending = TrackerDisagreementReview.pending(
            batch,
            left,
            right,
            comparison,
        )
        pending.bind(batch, left, right, comparison)

        self.assertEqual(4, len(pending.links))
        self.assertEqual(2, sum(item.proposed_by == "left" for item in pending.links))
        self.assertEqual(
            2,
            sum(item.proposed_by == "right" for item in pending.links),
        )
        self.assertTrue(all(item.link_kind == "continuous" for item in pending.links))
        self.assertTrue(all(item.review_state == "pending" for item in pending.links))
        self.assertFalse(pending.human_labels)
        self.assertFalse(pending.model_generated_labels)
        self.assertFalse(pending.paths_disclosed)

        schema = json.loads(
            Path("schemas/tracker-disagreement-review-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(pending.to_dict()))

    def test_human_link_judgments_produce_relative_not_overall_result(self) -> None:
        batch, left, right = self._inputs()
        comparison = compare_tracker_predictions(batch, left, right)
        review = self._approved(TrackerDisagreementReview.pending(batch, left, right, comparison))
        report = evaluate_tracker_disagreement(
            batch,
            left,
            right,
            comparison,
            review,
        )

        self.assertEqual("relative_difference_available", report.evidence_status)
        self.assertEqual(4, report.reviewed_link_count)
        self.assertEqual(0, report.left_disagreement_error_count)
        self.assertEqual(4, report.right_disagreement_error_count)
        self.assertEqual(0.0, report.left_disagreement_error_rate)
        self.assertEqual(1.0, report.right_disagreement_error_rate)
        self.assertEqual("left", report.relative_preference)
        self.assertFalse(report.overall_accuracy_evaluated)
        self.assertFalse(report.shared_links_evaluated)
        self.assertFalse(report.global_adapter_selected)
        self.assertFalse(report.subject_identity_inference)
        self.assertEqual(
            report,
            TrackerDisagreementReport.from_dict(report.to_dict()),
        )

        schema = json.loads(
            Path("schemas/tracker-disagreement-report-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))

    def test_tie_and_no_reviewable_links_are_explicit(self) -> None:
        batch, left, right = self._inputs()
        comparison = compare_tracker_predictions(batch, left, right)
        pending = TrackerDisagreementReview.pending(batch, left, right, comparison)

        tie = evaluate_tracker_disagreement(
            batch,
            left,
            right,
            comparison,
            self._approved(pending, all_same=True),
        )
        self.assertEqual("tie", tie.evidence_status)
        self.assertEqual("tie", tie.relative_preference)
        self.assertEqual(2, tie.left_disagreement_error_count)
        self.assertEqual(2, tie.right_disagreement_error_count)

        no_evidence = evaluate_tracker_disagreement(
            batch,
            left,
            right,
            comparison,
            self._approved(pending, all_unreviewable=True),
        )
        self.assertEqual("no_reviewable_links", no_evidence.evidence_status)
        self.assertEqual("insufficient_evidence", no_evidence.relative_preference)
        self.assertIsNone(no_evidence.left_disagreement_error_rate)
        self.assertIsNone(no_evidence.right_disagreement_error_rate)

    def test_pending_tampered_or_model_generated_review_fails_closed(self) -> None:
        batch, left, right = self._inputs()
        comparison = compare_tracker_predictions(batch, left, right)
        pending = TrackerDisagreementReview.pending(batch, left, right, comparison)
        with self.assertRaisesRegex(TrackerEvaluationError, "approved human"):
            evaluate_tracker_disagreement(
                batch,
                left,
                right,
                comparison,
                pending,
            )

        changed = pending.to_dict()
        changed_links = changed["links"]
        assert isinstance(changed_links, list)
        changed_links[0]["current_detection_id"] = "other"  # type: ignore[index]
        tampered = TrackerDisagreementReview.from_dict(changed)
        with self.assertRaisesRegex(TrackerEvaluationError, "different predictions"):
            tampered.bind(batch, left, right, comparison)

        generated = pending.to_dict()
        generated["model_generated_labels"] = True
        with self.assertRaisesRegex(TrackerEvaluationError, "must be false"):
            TrackerDisagreementReview.from_dict(generated)

    def test_strict_round_trip_private_exclusive_files(self) -> None:
        batch, left, right = self._inputs()
        comparison = compare_tracker_predictions(batch, left, right)
        pending = TrackerDisagreementReview.pending(batch, left, right, comparison)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "review.json")
            pending.write_new(path)
            self.assertEqual(pending, TrackerDisagreementReview.read(path))
            self.assertEqual(0o600, path.stat().st_mode & 0o777)
            with self.assertRaises(FileExistsError):
                pending.write_new(path)

    def test_cli_prepares_validates_and_scores_without_paths(self) -> None:
        batch, left, right = self._inputs()
        comparison = compare_tracker_predictions(batch, left, right)
        with tempfile.TemporaryDirectory(prefix="PRIVATE_LINK_CLI_") as directory:
            root = Path(directory)
            paths: dict[str, Path] = {}
            for name, value in (
                ("frames", batch.to_dict()),
                ("left", left.to_dict()),
                ("right", right.to_dict()),
                ("comparison", comparison.to_dict()),
            ):
                path = root / f"{name}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths[name] = path
            pending_path = root / "pending.json"
            approved_path = root / "approved.json"
            report_path = root / "report.json"

            shared = [
                "--frames",
                str(paths["frames"]),
                "--left",
                str(paths["left"]),
                "--right",
                str(paths["right"]),
                "--comparison",
                str(paths["comparison"]),
            ]
            outputs = []
            for command in (
                [
                    "prepare-tracker-disagreement-review",
                    *shared,
                    "--output",
                    str(pending_path),
                ],
                [
                    "validate-tracker-disagreement-review",
                    *shared,
                    "--review",
                    str(pending_path),
                ],
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    self.assertEqual(0, main(command))
                outputs.append(stdout.getvalue())

            pending = TrackerDisagreementReview.read(pending_path)
            approved_path.write_text(
                json.dumps(self._approved(pending).to_dict()),
                encoding="utf-8",
            )
            for command in (
                [
                    "evaluate-tracker-disagreement",
                    *shared,
                    "--review",
                    str(approved_path),
                    "--output",
                    str(report_path),
                ],
                ["validate-tracker-disagreement", str(report_path)],
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    self.assertEqual(0, main(command))
                outputs.append(stdout.getvalue())

            self.assertTrue(report_path.is_file())
            self.assertEqual(0o600, pending_path.stat().st_mode & 0o777)
            self.assertEqual(0o600, report_path.stat().st_mode & 0o777)
            self.assertTrue(all(str(root) not in output for output in outputs))

    def test_public_disagreement_fixtures_are_reproducibly_bound(self) -> None:
        batch = DetectionFrameBatch.read("tests/fixtures/valid_detection_frame_batch.json")
        left = TrackerPredictionBatch.read("tests/fixtures/valid_tracker_prediction_batch.json")
        right = TrackerPredictionBatch.read(
            "tests/fixtures/valid_tracker_disagreement_prediction_batch.json"
        )
        comparison = compare_tracker_predictions(batch, left, right)
        self.assertEqual(
            comparison,
            TrackerPredictionComparison.read(
                "tests/fixtures/valid_tracker_disagreement_comparison.json"
            ),
        )
        review = TrackerDisagreementReview.read(
            "tests/fixtures/valid_tracker_disagreement_review.json"
        )
        review.bind(batch, left, right, comparison)
        report = TrackerDisagreementReport.read(
            "tests/fixtures/valid_tracker_disagreement_report.json"
        )
        self.assertEqual(
            report,
            evaluate_tracker_disagreement(
                batch,
                left,
                right,
                comparison,
                review,
            ),
        )


if __name__ == "__main__":
    unittest.main()
