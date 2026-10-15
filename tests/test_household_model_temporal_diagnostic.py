from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.household_model_diagnostic import (
    HouseholdModelDiagnosticError,
    evaluate_household_model_presence_diagnostic,
)
from baby_monitor.household_model_temporal_diagnostic import (
    HOUSEHOLD_MODEL_TEMPORAL_SELECTION_RULE,
    HouseholdModelTemporalCameraPoint,
    HouseholdModelTemporalDiagnosticPoint,
    HouseholdModelTemporalDiagnosticPolicy,
    HouseholdModelTemporalDiagnosticReport,
    evaluate_household_model_temporal_diagnostic,
    select_household_model_temporal_working_point,
)
from tests.test_household_model_diagnostic import _admission, _review, _scan, _scans


class HouseholdModelTemporalDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.admissions = (_admission("camera_01"), _admission("camera_02"))
        self.review = _review(self.admissions)
        self.scans = _scans(self.admissions)

    def _report(self) -> HouseholdModelTemporalDiagnosticReport:
        return evaluate_household_model_temporal_diagnostic(
            self.review,
            media_admissions=self.admissions,
            detector_scans=self.scans,
            thresholds=(0.1, 0.2, 0.9),
            hold_seconds_grid=(0.0, 1.0, 2.0),
            bin_seconds=1.0,
        )

    def test_zero_hold_matches_same_bin_diagnostic_and_stratifies(self) -> None:
        temporal = self._report()
        same_bin = evaluate_household_model_presence_diagnostic(
            self.review,
            media_admissions=self.admissions,
            detector_scans=self.scans,
            thresholds=(0.1, 0.2, 0.9),
            bin_seconds=1.0,
        )
        zero_hold = tuple(point for point in temporal.points if point.hold_seconds == 0.0)
        for temporal_point, same_bin_point in zip(zero_hold, same_bin.points, strict=True):
            self.assertEqual(
                temporal_point.confidence_threshold, same_bin_point.confidence_threshold
            )
            for field in (
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
                "precision",
                "recall",
                "f1",
                "specificity",
                "partial_positive_count",
                "partial_detected_count",
                "partial_visibility_recall",
            ):
                self.assertEqual(getattr(temporal_point, field), getattr(same_bin_point, field))

        baseline = next(
            point
            for point in temporal.points
            if point.confidence_threshold == 0.2 and point.hold_seconds == 0.0
        )
        held = next(
            point
            for point in temporal.points
            if point.confidence_threshold == 0.2 and point.hold_seconds == 1.0
        )
        self.assertEqual(
            (2, 0, 0, 2),
            (
                baseline.true_positive,
                baseline.false_positive,
                baseline.false_negative,
                baseline.true_negative,
            ),
        )
        self.assertEqual(1, baseline.clear_positive_count)
        self.assertEqual(1, baseline.partial_positive_count)
        self.assertEqual(1.0, baseline.clear_visibility_recall)
        self.assertEqual(1.0, baseline.partial_visibility_recall)
        self.assertEqual(
            (2, 1, 0, 1),
            (
                held.true_positive,
                held.false_positive,
                held.false_negative,
                held.true_negative,
            ),
        )
        self.assertEqual(1.0, held.maximum_negative_camera_false_positive_rate)
        self.assertEqual(
            ("camera_01", "camera_02"), tuple(point.camera_id for point in held.camera_points)
        )

    def test_future_detection_never_satisfies_an_earlier_bin(self) -> None:
        first, second = self.admissions
        scans = (
            _scan(first, "camera_01", "yolo26m", ()),
            _scan(second, "camera_02", "yolo26m", ((1.2, 0.9),)),
        )
        report = evaluate_household_model_temporal_diagnostic(
            self.review,
            media_admissions=self.admissions,
            detector_scans=scans,
            thresholds=(0.1, 0.9),
            hold_seconds_grid=(0.0, 2.0),
            bin_seconds=1.0,
        )
        point = next(
            item
            for item in report.points
            if item.confidence_threshold == 0.9 and item.hold_seconds == 2.0
        )
        camera_02 = next(item for item in point.camera_points if item.camera_id == "camera_02")
        self.assertEqual(0, camera_02.true_positive)
        self.assertEqual(1, camera_02.false_negative)
        self.assertEqual(1, camera_02.false_positive)
        self.assertEqual(0, camera_02.true_negative)
        self.assertTrue(report.causal_temporal_aggregation)
        self.assertFalse(report.future_detections_used)
        self.assertFalse(report.frame_level_alignment_evaluated)

    def test_round_trip_schema_strictness_and_authority(self) -> None:
        report = self._report()
        restored = HouseholdModelTemporalDiagnosticReport.from_json(report.to_json())
        self.assertEqual(report, restored)
        self.assertEqual(9, len(report.points))
        schema = json.loads(
            Path("schemas/household-model-temporal-diagnostic-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))
        self.assertEqual(
            set(schema["$defs"]["cameraPoint"]["required"]),
            set(HouseholdModelTemporalCameraPoint.__dataclass_fields__),
        )
        self.assertEqual(
            set(schema["$defs"]["point"]["required"]),
            set(HouseholdModelTemporalDiagnosticPoint.__dataclass_fields__),
        )
        self.assertFalse(report.ground_truth)
        self.assertFalse(report.official_accuracy_claim)
        self.assertTrue(report.shadow_only)
        self.assertFalse(report.deployment_approved)
        self.assertFalse(report.household_configuration_changed)

        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(report, future_detections_used=True)
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(report, hold_seconds_grid=(1.0, 2.0))
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(report, sampling_strategy="unreviewed_strategy")
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelTemporalDiagnosticReport.from_json(
                report.to_json().replace('"version": 1,', '"version": 1, "version": 1,')
            )
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelTemporalDiagnosticReport.from_json(
                report.to_json().replace('"bin_seconds": 1.0', '"bin_seconds": NaN')
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            report.write_new(path)
            self.assertEqual(report, HouseholdModelTemporalDiagnosticReport.read(path))
            self.assertEqual(0o600, path.stat().st_mode & 0o777)
            with self.assertRaises(FileExistsError):
                report.write_new(path)

    def test_selection_is_constrained_to_a_shadow_candidate(self) -> None:
        report = self._report()
        policy = HouseholdModelTemporalDiagnosticPolicy(
            minimum_precision=1.0,
            minimum_recall=1.0,
            minimum_positive_camera_recall=1.0,
            minimum_clear_visibility_recall=1.0,
            minimum_partial_visibility_recall=1.0,
            maximum_negative_camera_false_positive_rate=0.0,
            maximum_hold_seconds=2.0,
            require_clear_visibility_samples=True,
        )
        selected, feasible_count = select_household_model_temporal_working_point(
            report,
            policy,
        )
        self.assertEqual(1, feasible_count)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(0.2, selected.confidence_threshold)
        self.assertEqual(0.0, selected.hold_seconds)
        self.assertIn("shorter_hold", HOUSEHOLD_MODEL_TEMPORAL_SELECTION_RULE)
        self.assertIn("minimum_positive_camera_recall", HOUSEHOLD_MODEL_TEMPORAL_SELECTION_RULE)

    def test_selection_rejects_camera_label_confounding_by_default(self) -> None:
        partitioned_review = replace(
            self.review,
            items=(self.review.items[0], self.review.items[4]),
        )
        report = evaluate_household_model_temporal_diagnostic(
            partitioned_review,
            media_admissions=self.admissions,
            detector_scans=self.scans,
            thresholds=(0.1, 0.2, 0.9),
            hold_seconds_grid=(0.0, 1.0),
            bin_seconds=1.0,
        )
        policy = HouseholdModelTemporalDiagnosticPolicy(
            minimum_precision=1.0,
            minimum_recall=1.0,
            minimum_positive_camera_recall=1.0,
            minimum_clear_visibility_recall=1.0,
            minimum_partial_visibility_recall=0.0,
            maximum_negative_camera_false_positive_rate=0.0,
            maximum_hold_seconds=1.0,
            require_partial_visibility_samples=False,
        )
        selected, feasible_count = select_household_model_temporal_working_point(
            report,
            policy,
        )
        self.assertIsNone(selected)
        self.assertEqual(0, feasible_count)

        diagnostic_selected, diagnostic_count = select_household_model_temporal_working_point(
            report,
            replace(policy, require_mixed_labels_per_camera=False),
        )
        self.assertIsNotNone(diagnostic_selected)
        self.assertGreater(diagnostic_count, 0)


class HouseholdModelTemporalDiagnosticCLITests(unittest.TestCase):
    def test_cli_round_trip_is_path_free_and_shadow_only(self) -> None:
        admissions = (_admission("camera_01"), _admission("camera_02"))
        review = _review(admissions)
        scans = _scans(admissions)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_path = root / "review.json"
            admission_paths = tuple(root / f"admission_{index}.json" for index in range(2))
            scan_paths = tuple(root / f"scan_{index}.json" for index in range(2))
            report_path = root / "temporal-report.json"
            review.write_new(review_path)
            for admission, path in zip(admissions, admission_paths, strict=True):
                path.write_text(f"{admission.to_json()}\n", encoding="utf-8")
            for scan, path in zip(scans, scan_paths, strict=True):
                scan.write_new(path)

            arguments = [
                "evaluate-household-model-temporal-diagnostic",
                "--review",
                str(review_path),
            ]
            for path in admission_paths:
                arguments.extend(("--media-admission", str(path)))
            for path in scan_paths:
                arguments.extend(("--detector-scan", str(path)))
            arguments.extend(
                (
                    "--threshold",
                    "0.1",
                    "--threshold",
                    "0.2",
                    "--threshold",
                    "0.9",
                    "--hold-seconds",
                    "0",
                    "--hold-seconds",
                    "1",
                    "--bin-seconds",
                    "1",
                    "--output",
                    str(report_path),
                )
            )
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(0, main(arguments))
            summary = json.loads(output.getvalue())
            self.assertNotIn(str(root), output.getvalue())
            report = HouseholdModelTemporalDiagnosticReport.read(report_path)
            self.assertEqual(report.fingerprint, summary["report_sha256"])
            self.assertTrue(summary["causal_temporal_aggregation"])
            self.assertFalse(summary["future_detections_used"])

            validation_output = StringIO()
            with redirect_stdout(validation_output):
                self.assertEqual(
                    0,
                    main(
                        [
                            "validate-household-model-temporal-diagnostic",
                            str(report_path),
                        ]
                    ),
                )
            self.assertNotIn(str(root), validation_output.getvalue())

            selection_output = StringIO()
            with redirect_stdout(selection_output):
                self.assertEqual(
                    0,
                    main(
                        [
                            "select-household-model-temporal-working-point",
                            str(report_path),
                            "--minimum-precision",
                            "1",
                            "--minimum-recall",
                            "1",
                            "--minimum-positive-camera-recall",
                            "1",
                            "--minimum-clear-visibility-recall",
                            "1",
                            "--minimum-partial-visibility-recall",
                            "1",
                            "--maximum-negative-camera-false-positive-rate",
                            "0",
                            "--maximum-hold-seconds",
                            "1",
                            "--require-clear-visibility-samples",
                        ]
                    ),
                )
            selection = json.loads(selection_output.getvalue())
            self.assertEqual(0.2, selection["selected_point"]["confidence_threshold"])
            self.assertEqual(0.0, selection["selected_point"]["hold_seconds"])
            self.assertTrue(selection["shadow_only"])
            self.assertFalse(selection["winner_selected"])
            self.assertFalse(selection["deployment_approved"])
            self.assertFalse(selection["household_configuration_changed"])
            self.assertNotIn(str(root), selection_output.getvalue())


if __name__ == "__main__":
    unittest.main()
