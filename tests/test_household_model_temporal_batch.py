from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.household_model_diagnostic import HouseholdModelDiagnosticError
from baby_monitor.household_model_temporal_batch import (
    HouseholdModelTemporalBatchBinding,
    HouseholdModelTemporalBatchDiagnosticReport,
    combine_household_model_temporal_diagnostic_batches,
    select_household_model_temporal_batch_working_point,
)
from baby_monitor.household_model_temporal_diagnostic import (
    HouseholdModelTemporalCameraPoint,
    HouseholdModelTemporalDiagnosticPoint,
    HouseholdModelTemporalDiagnosticPolicy,
    HouseholdModelTemporalDiagnosticReport,
    evaluate_household_model_temporal_diagnostic,
    select_household_model_temporal_working_point,
)
from baby_monitor.source_admission import ExternalSourceAdmission
from tests.test_household_model_diagnostic import _admission, _review, _scans


def _batch_admissions(batch_id: str) -> tuple[ExternalSourceAdmission, ...]:
    admissions = []
    for camera_id in ("camera_01", "camera_02"):
        value = _admission(camera_id).to_dict()
        content = f"private household {batch_id} {camera_id}".encode()
        value["source_id"] = f"household_{batch_id}_{camera_id}"
        value["canonical_url"] = f"https://example.invalid/{batch_id}/{camera_id}"
        value["source_revision"] = f"fixture_{batch_id}"
        value["access"]["expected_bytes"] = len(content)
        value["content"]["observed_bytes"] = len(content)
        value["content"]["sha256"] = hashlib.sha256(content).hexdigest()
        admissions.append(ExternalSourceAdmission.from_dict(value))
    return tuple(admissions)


def _complementary_report(
    batch_id: str,
    *,
    first_partition: bool,
) -> HouseholdModelTemporalDiagnosticReport:
    admissions = _batch_admissions(batch_id)
    review = _review(admissions)
    review = replace(
        review,
        review_id=f"household_model_review_{batch_id}",
        selection_manifest_sha256=hashlib.sha256(
            f"selection manifest {batch_id}".encode()
        ).hexdigest(),
        items=(
            (review.items[1], review.items[5])
            if first_partition
            else (review.items[0], review.items[4])
        ),
    )
    return evaluate_household_model_temporal_diagnostic(
        review,
        media_admissions=admissions,
        detector_scans=_scans(admissions),
        thresholds=(0.1, 0.2, 0.9),
        hold_seconds_grid=(0.0, 1.0),
        bin_seconds=1.0,
    )


def _reports() -> tuple[
    HouseholdModelTemporalDiagnosticReport,
    HouseholdModelTemporalDiagnosticReport,
]:
    return (
        _complementary_report("batch_a", first_partition=True),
        _complementary_report("batch_b", first_partition=False),
    )


def _strict_policy() -> HouseholdModelTemporalDiagnosticPolicy:
    return HouseholdModelTemporalDiagnosticPolicy(
        minimum_precision=1.0,
        minimum_recall=1.0,
        minimum_positive_camera_recall=1.0,
        minimum_clear_visibility_recall=1.0,
        minimum_partial_visibility_recall=1.0,
        maximum_negative_camera_false_positive_rate=0.0,
        maximum_hold_seconds=1.0,
        require_clear_visibility_samples=True,
    )


class HouseholdModelTemporalBatchTests(unittest.TestCase):
    def test_combines_complementary_strata_round_trips_and_matches_schema(self) -> None:
        first, second = _reports()
        report = combine_household_model_temporal_diagnostic_batches((second, first))

        self.assertEqual(2, len(report.batches))
        self.assertEqual(
            tuple(sorted((first.fingerprint, second.fingerprint))),
            tuple(binding.report_sha256 for binding in report.batches),
        )
        self.assertEqual(("camera_01", "camera_02"), report.camera_ids)
        self.assertEqual(4, report.reviewed_frame_count)
        self.assertEqual(4, report.evaluated_bin_count)
        self.assertEqual(2, report.positive_bin_count)
        self.assertEqual(2, report.negative_bin_count)
        self.assertEqual(1, report.clear_positive_bin_count)
        self.assertEqual(1, report.partial_positive_bin_count)
        self.assertEqual(2, report.mixed_label_camera_count)
        self.assertEqual(
            report, HouseholdModelTemporalBatchDiagnosticReport.from_json(report.to_json())
        )

        point = next(
            item
            for item in report.points
            if item.confidence_threshold == 0.2 and item.hold_seconds == 0.0
        )
        self.assertEqual(
            (2, 0, 0, 2),
            (
                point.true_positive,
                point.false_positive,
                point.false_negative,
                point.true_negative,
            ),
        )
        self.assertEqual(1.0, point.precision)
        self.assertEqual(1.0, point.recall)
        self.assertEqual(
            (1.0, 1.0),
            tuple(camera.recall for camera in point.camera_points),
        )
        self.assertEqual(
            (0.0, 0.0),
            tuple(camera.false_positive_rate for camera in point.camera_points),
        )

        schema = json.loads(
            Path(
                "schemas/household-model-temporal-batch-diagnostic-report-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))
        self.assertEqual(
            set(schema["$defs"]["batchBinding"]["required"]),
            set(HouseholdModelTemporalBatchBinding.__dataclass_fields__),
        )
        self.assertEqual(
            set(schema["$defs"]["cameraPoint"]["required"]),
            set(HouseholdModelTemporalCameraPoint.__dataclass_fields__),
        )
        self.assertEqual(
            set(schema["$defs"]["point"]["required"]),
            set(HouseholdModelTemporalDiagnosticPoint.__dataclass_fields__),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch-report.json"
            report.write_new(path)
            self.assertEqual(report, HouseholdModelTemporalBatchDiagnosticReport.read(path))
            self.assertEqual(0o600, path.stat().st_mode & 0o777)
            with self.assertRaises(FileExistsError):
                report.write_new(path)

    def test_cross_batch_mix_unlocks_only_a_shadow_candidate(self) -> None:
        reports = _reports()
        policy = _strict_policy()

        for report in reports:
            selected, feasible_count = select_household_model_temporal_working_point(
                report,
                policy,
            )
            self.assertIsNone(selected)
            self.assertEqual(0, feasible_count)

        combined = combine_household_model_temporal_diagnostic_batches(reports)
        selected, feasible_count = select_household_model_temporal_batch_working_point(
            combined,
            policy,
        )
        self.assertEqual(1, feasible_count)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(0.2, selected.confidence_threshold)
        self.assertEqual(0.0, selected.hold_seconds)
        self.assertTrue(combined.shadow_only)
        self.assertFalse(combined.ground_truth)
        self.assertFalse(combined.official_accuracy_claim)
        self.assertFalse(combined.deployment_approved)
        self.assertFalse(combined.household_configuration_changed)

    def test_rejects_overlap_mismatch_tampering_and_authority_inflation(self) -> None:
        first, second = _reports()
        combined = combine_household_model_temporal_diagnostic_batches((first, second))

        with self.assertRaises(HouseholdModelDiagnosticError):
            combine_household_model_temporal_diagnostic_batches((first, first))
        with self.assertRaises(HouseholdModelDiagnosticError):
            combine_household_model_temporal_diagnostic_batches(
                (first, replace(second, model_id="other_model"))
            )
        with self.assertRaises(HouseholdModelDiagnosticError):
            combine_household_model_temporal_diagnostic_batches(
                (
                    first,
                    replace(
                        second,
                        media_admission_sha256s=first.media_admission_sha256s,
                    ),
                )
            )
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(combined, ground_truth=True)
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(combined, input_reports_distinct=False)
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(combined, sampling_strategy="unreviewed_strategy")
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(combined, batches=tuple(reversed(combined.batches)))

        tampered = json.loads(combined.to_json())
        tampered["positive_bin_count"] += 1
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelTemporalBatchDiagnosticReport.from_dict(tampered)
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelTemporalBatchDiagnosticReport.from_json(
                combined.to_json().replace('"version": 1,', '"version": 1, "version": 1,')
            )
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelTemporalBatchDiagnosticReport.from_json(
                combined.to_json().replace('"bin_seconds": 1.0', '"bin_seconds": NaN')
            )


class HouseholdModelTemporalBatchCLITests(unittest.TestCase):
    def test_cli_combine_validate_and_select_are_path_free(self) -> None:
        reports = _reports()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_paths = tuple(root / f"temporal-{index}.json" for index in range(2))
            output_path = root / "batch.json"
            for report, path in zip(reports, input_paths, strict=True):
                report.write_new(path)

            combine_output = StringIO()
            with redirect_stdout(combine_output):
                self.assertEqual(
                    0,
                    main(
                        [
                            "combine-household-model-temporal-diagnostic-batches",
                            *(str(path) for path in input_paths),
                            "--output",
                            str(output_path),
                        ]
                    ),
                )
            summary = json.loads(combine_output.getvalue())
            report = HouseholdModelTemporalBatchDiagnosticReport.read(output_path)
            self.assertEqual(report.fingerprint, summary["report_sha256"])
            self.assertEqual(2, summary["batch_count"])
            self.assertEqual(2, summary["mixed_label_camera_count"])
            self.assertNotIn(str(root), combine_output.getvalue())

            validation_output = StringIO()
            with redirect_stdout(validation_output):
                self.assertEqual(
                    0,
                    main(
                        [
                            "validate-household-model-temporal-batch-diagnostic",
                            str(output_path),
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
                            "select-household-model-temporal-batch-working-point",
                            str(output_path),
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
            self.assertEqual(2, selection["input_batch_count"])
            self.assertEqual(2, selection["mixed_label_camera_count"])
            self.assertEqual(0.2, selection["selected_point"]["confidence_threshold"])
            self.assertEqual(0.0, selection["selected_point"]["hold_seconds"])
            self.assertTrue(selection["shadow_only"])
            self.assertFalse(selection["winner_selected"])
            self.assertFalse(selection["deployment_approved"])
            self.assertFalse(selection["household_configuration_changed"])
            self.assertNotIn(str(root), selection_output.getvalue())


if __name__ == "__main__":
    unittest.main()
