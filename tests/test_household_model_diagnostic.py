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
from baby_monitor.detector_protocol import AdmittedDetectorScan, DetectorScanResult
from baby_monitor.domain import Detection
from baby_monitor.household_model_diagnostic import (
    HOUSEHOLD_MODEL_DIAGNOSTIC_SELECTION_RULE,
    HouseholdModelDiagnosticError,
    HouseholdModelDiagnosticPolicy,
    HouseholdModelDiagnosticReport,
    HouseholdModelPresenceItem,
    HouseholdModelPresenceReview,
    HouseholdModelPresenceSource,
    compare_household_model_presence_diagnostics,
    evaluate_household_model_presence_diagnostic,
    select_household_model_diagnostic_working_point,
)
from baby_monitor.source_admission import ExternalSourceAdmission


def _admission(camera_id: str, *, purpose: str = "model_evaluation") -> ExternalSourceAdmission:
    content = f"private household {camera_id}".encode()
    return ExternalSourceAdmission.from_dict(
        {
            "version": 1,
            "source_id": f"household_{camera_id}",
            "asset_kind": "media_asset",
            "canonical_url": f"https://example.invalid/{camera_id}",
            "accessed_on": "2026-08-29",
            "source_revision": "fixture_v1",
            "upstream_name": "private household fixture",
            "license": {
                "name": "Private household media",
                "spdx_id": None,
                "terms_url": "https://example.invalid/terms",
                "commercial_use": "prohibited",
                "modification": "prohibited",
                "redistribution": "prohibited",
            },
            "access": {
                "gated": "no",
                "authentication_required": "no",
                "expected_bytes": len(content),
            },
            "content": {
                "observed_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            },
            "privacy": {
                "real_minors": "yes",
                "faces": "yes",
                "audio": "no",
                "device_or_account_ids": "no",
                "precise_timestamps": "no",
                "private_household_data": "yes",
            },
            "labels": {
                "origin": "none",
                "human_verified": False,
                "ground_truth": False,
            },
            "admission": {
                "tier": "C",
                "status": "verified_local",
                "purpose": purpose,
                "requires_user_approval": True,
                "user_approval_recorded": True,
                "git_allowed": False,
                "ci_allowed": False,
                "redistribution_allowed": False,
                "training_allowed": False,
                "reviewed_on": "2026-08-29",
                "blocker_codes": [],
                "restriction_codes": ["local_only", "private_household"],
            },
        }
    )


def _item(
    admission: ExternalSourceAdmission,
    camera_id: str,
    timestamp: float,
    visible_person: str,
    visibility: str,
    index: int,
) -> HouseholdModelPresenceItem:
    return HouseholdModelPresenceItem(
        review_item_id=f"review_{index:03d}",
        camera_id=camera_id,
        media_admission_sha256=admission.fingerprint,
        timestamp_seconds=timestamp,
        source_frame_sha256=hashlib.sha256(f"frame {index}".encode()).hexdigest(),
        visible_person=visible_person,
        visibility=visibility,
        label_confidence=0.96 if visibility == "partial" else 0.98,
    )


def _review(admissions: tuple[ExternalSourceAdmission, ...]) -> HouseholdModelPresenceReview:
    first, second = admissions
    return HouseholdModelPresenceReview(
        review_id="household_model_review_01",
        selection_manifest_sha256=hashlib.sha256(b"selection manifest").hexdigest(),
        review_model_id="codex_visual_reviewer",
        review_model_revision="session_fixture",
        review_media_transfer="selected_stills_remote",
        sampling_strategy="candidate_disagreement_and_control",
        sources=(
            HouseholdModelPresenceSource("camera_01", first.fingerprint, 4.0),
            HouseholdModelPresenceSource("camera_02", second.fingerprint, 4.0),
        ),
        items=(
            _item(first, "camera_01", 0.2, "absent", "clear", 1),
            _item(first, "camera_01", 1.2, "present", "partial", 2),
            _item(first, "camera_01", 1.6, "present", "partial", 3),
            _item(first, "camera_01", 2.2, "unknown", "unknown", 4),
            _item(second, "camera_02", 0.2, "present", "clear", 5),
            _item(second, "camera_02", 1.2, "absent", "clear", 6),
        ),
    )


def _scan(
    admission: ExternalSourceAdmission,
    camera_id: str,
    model_id: str,
    values: tuple[tuple[float, float], ...],
) -> AdmittedDetectorScan:
    detections = tuple(
        Detection(timestamp, "person", confidence, (0.1, 0.1, 0.4, 0.9))
        for timestamp, confidence in values
    )
    result = DetectorScanResult(
        camera_id=camera_id,
        detector_adapter="yolo_command",
        model_id=model_id,
        model_revision="fixture_v1",
        model_sha256=hashlib.sha256(model_id.encode()).hexdigest(),
        detections=detections,
    )
    return AdmittedDetectorScan(
        source_admission_sha256=admission.fingerprint,
        labels=("person",),
        confidence_threshold=0.1,
        requested_sample_fps=1.0,
        result=result,
    )


def _scans(
    admissions: tuple[ExternalSourceAdmission, ...],
    model_id: str = "yolo26m",
) -> tuple[AdmittedDetectorScan, ...]:
    first, second = admissions
    if model_id == "yolo26m":
        first_values = ((0.2, 0.15), (1.2, 0.9), (2.2, 0.2))
        second_values = ((0.2, 0.8),)
    else:
        first_values = ((0.2, 0.3), (1.2, 0.4), (2.2, 0.2))
        second_values = ()
    return (
        _scan(first, "camera_01", model_id, first_values),
        _scan(second, "camera_02", model_id, second_values),
    )


class HouseholdModelPresenceReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.admissions = (_admission("camera_01"), _admission("camera_02"))
        self.review = _review(self.admissions)

    def test_round_trip_schema_and_exclusive_write(self) -> None:
        restored = HouseholdModelPresenceReview.from_json(self.review.to_json())
        self.assertEqual(self.review, restored)
        self.assertEqual(3, self.review.present_frame_count)
        self.assertEqual(2, self.review.absent_frame_count)
        self.assertEqual(1, self.review.unknown_frame_count)
        schema = json.loads(
            Path("schemas/household-model-presence-review-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(self.review.to_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            self.review.write_new(path)
            self.assertEqual(self.review, HouseholdModelPresenceReview.read(path))
            with self.assertRaises(FileExistsError):
                self.review.write_new(path)

    def test_model_labels_cannot_claim_human_or_official_authority(self) -> None:
        for field, value in (
            ("model_generated_labels", False),
            ("human_verified", True),
            ("ground_truth", True),
            ("approved_for_official_evaluation", True),
            ("clips_shared", True),
            ("audio_shared", True),
            ("identity_inference", True),
            ("biometric_processing", True),
            ("training_allowed", True),
            ("redistribution_allowed", True),
            ("source_media_included", True),
            ("paths_disclosed", True),
        ):
            with (
                self.subTest(field=field),
                self.assertRaises(HouseholdModelDiagnosticError),
            ):
                replace(self.review, **{field: value})

    def test_items_fail_closed_on_visibility_order_and_binding(self) -> None:
        first_item = self.review.items[0]
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(first_item, visible_person="absent", visibility="partial")
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(first_item, visible_person="unknown", visibility="clear")
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(self.review, items=tuple(reversed(self.review.items)))
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(
                self.review,
                items=(
                    *self.review.items[:-1],
                    replace(
                        self.review.items[-1],
                        media_admission_sha256=hashlib.sha256(b"wrong").hexdigest(),
                    ),
                ),
            )

    def test_rejects_duplicate_keys_and_tampered_fingerprint(self) -> None:
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelPresenceReview.from_json('{"version": 1, "version": 1}')
        for field, value in (("sources", 1), ("items", {})):
            malformed = self.review.to_dict()
            malformed[field] = value
            with self.subTest(field=field), self.assertRaises(HouseholdModelDiagnosticError):
                HouseholdModelPresenceReview.from_dict(malformed)
        tampered = json.loads(self.review.to_json())
        tampered["items"][0]["visible_person"] = "present"
        tampered["items"][0]["visibility"] = "clear"
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelPresenceReview.from_dict(tampered)


class HouseholdModelDiagnosticEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.admissions = (_admission("camera_01"), _admission("camera_02"))
        self.review = _review(self.admissions)
        self.scans = _scans(self.admissions)

    def _report(self, *, scans: tuple[AdmittedDetectorScan, ...] | None = None):
        return evaluate_household_model_presence_diagnostic(
            self.review,
            media_admissions=self.admissions,
            detector_scans=self.scans if scans is None else scans,
            thresholds=(0.1, 0.2, 0.9),
            bin_seconds=1.0,
        )

    def test_metrics_unknown_bins_and_partial_visibility(self) -> None:
        report = self._report()
        self.assertEqual(6, report.reviewed_frame_count)
        self.assertEqual(4, report.evaluated_bin_count)
        self.assertEqual(2, report.positive_bin_count)
        self.assertEqual(2, report.negative_bin_count)
        self.assertEqual(1, report.unknown_bin_count)
        self.assertEqual(1, report.partial_positive_bin_count)
        low, selected, high = report.points
        self.assertEqual(
            (2, 1, 0, 1),
            (
                low.true_positive,
                low.false_positive,
                low.false_negative,
                low.true_negative,
            ),
        )
        self.assertEqual(0.666666666667, low.precision)
        self.assertEqual(1.0, low.recall)
        self.assertEqual(
            (2, 0, 0, 2),
            (
                selected.true_positive,
                selected.false_positive,
                selected.false_negative,
                selected.true_negative,
            ),
        )
        self.assertEqual(1.0, selected.f1)
        self.assertEqual(1.0, selected.partial_visibility_recall)
        self.assertEqual(
            (1, 0, 1, 2),
            (
                high.true_positive,
                high.false_positive,
                high.false_negative,
                high.true_negative,
            ),
        )

    def test_report_round_trip_schema_and_public_summary(self) -> None:
        report = self._report()
        restored = HouseholdModelDiagnosticReport.from_json(report.to_json())
        self.assertEqual(report, restored)
        schema = json.loads(
            Path("schemas/household-model-presence-diagnostic-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))
        public = report.to_public_dict()
        self.assertNotIn("camera_ids", public)
        self.assertNotIn("media_admission_sha256s", public)
        self.assertTrue(public["derived_without_model_rerun"])
        self.assertFalse(public["frame_level_alignment_evaluated"])
        self.assertFalse(public["candidate_trigger_coverage_evaluated"])
        self.assertFalse(public["winner_selected"])
        self.assertFalse(public["ground_truth"])
        tampered = json.loads(report.to_json())
        tampered["points"][0]["true_positive"] = 1
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelDiagnosticReport.from_dict(tampered)
        for field, value in (("points", 1), ("camera_ids", [1, "camera_02"])):
            malformed = report.to_dict()
            malformed[field] = value
            with self.subTest(field=field), self.assertRaises(HouseholdModelDiagnosticError):
                HouseholdModelDiagnosticReport.from_dict(malformed)

    def test_requires_exact_tier_c_admissions_and_one_model_config(self) -> None:
        wrong_admissions = (
            _admission("camera_01", purpose="research_only"),
            self.admissions[1],
        )
        with self.assertRaises(HouseholdModelDiagnosticError):
            evaluate_household_model_presence_diagnostic(
                self.review,
                media_admissions=wrong_admissions,
                detector_scans=self.scans,
                thresholds=(0.1, 0.2),
                bin_seconds=1.0,
            )
        mixed = (self.scans[0], _scans(self.admissions, "yolo11n")[1])
        with self.assertRaises(HouseholdModelDiagnosticError):
            self._report(scans=mixed)
        with self.assertRaises(HouseholdModelDiagnosticError):
            evaluate_household_model_presence_diagnostic(
                self.review,
                media_admissions=self.admissions,
                detector_scans=self.scans,
                thresholds=(0.2, 0.3),
                bin_seconds=1.0,
            )

    def test_conflicting_labels_in_one_bin_fail_closed(self) -> None:
        items = list(self.review.items)
        items[2] = replace(items[2], visible_person="absent", visibility="clear")
        conflicting = replace(self.review, items=tuple(items))
        with self.assertRaises(HouseholdModelDiagnosticError):
            evaluate_household_model_presence_diagnostic(
                conflicting,
                media_admissions=self.admissions,
                detector_scans=self.scans,
                thresholds=(0.1, 0.2),
                bin_seconds=1.0,
            )

    def test_selector_is_diagnostic_only_and_can_return_no_point(self) -> None:
        report = self._report()
        policy = HouseholdModelDiagnosticPolicy(
            minimum_precision=0.95,
            minimum_recall=0.95,
            minimum_partial_visibility_recall=0.95,
        )
        selected, feasible_count = select_household_model_diagnostic_working_point(
            report,
            policy,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(0.2, selected.confidence_threshold)
        self.assertEqual(1, feasible_count)
        impossible = replace(policy, minimum_recall=1.0, minimum_precision=1.0)
        weak_report = evaluate_household_model_presence_diagnostic(
            self.review,
            media_admissions=self.admissions,
            detector_scans=_scans(self.admissions, "yolo11n"),
            thresholds=(0.1, 0.2, 0.9),
            bin_seconds=1.0,
        )
        self.assertEqual(
            (None, 0),
            select_household_model_diagnostic_working_point(weak_report, impossible),
        )
        partial_miss = replace(
            report,
            points=(
                report.points[0],
                *(
                    replace(
                        point,
                        partial_detected_count=0,
                        partial_visibility_recall=0.0,
                    )
                    for point in report.points[1:]
                ),
            ),
        )
        allow_no_partial = replace(policy, require_partial_visibility_samples=False)
        self.assertEqual(
            (None, 0),
            select_household_model_diagnostic_working_point(
                partial_miss,
                allow_no_partial,
            ),
        )
        no_partial = replace(
            report,
            partial_positive_bin_count=0,
            points=tuple(
                replace(
                    point,
                    partial_positive_count=0,
                    partial_detected_count=0,
                    partial_visibility_recall=None,
                )
                for point in report.points
            ),
        )
        selected_without_partial, _count = select_household_model_diagnostic_working_point(
            no_partial,
            allow_no_partial,
        )
        self.assertIsNotNone(selected_without_partial)
        self.assertEqual(0.2, selected_without_partial.confidence_threshold)
        self.assertIn("recall_then_partial_recall", HOUSEHOLD_MODEL_DIAGNOSTIC_SELECTION_RULE)

    def test_comparison_requires_same_review_grid_and_distinct_models(self) -> None:
        first = self._report()
        second = evaluate_household_model_presence_diagnostic(
            self.review,
            media_admissions=self.admissions,
            detector_scans=_scans(self.admissions, "yolo11n"),
            thresholds=(0.1, 0.2, 0.9),
            bin_seconds=1.0,
        )
        comparison = compare_household_model_presence_diagnostics((first, second))
        self.assertFalse(comparison["winner_selected"])
        self.assertFalse(comparison["shadow_preference_selected"])
        self.assertFalse(comparison["ground_truth"])
        self.assertEqual(2, len(comparison["reports"]))
        with self.assertRaises(HouseholdModelDiagnosticError):
            compare_household_model_presence_diagnostics((first, first))
        different_grid = replace(second, bin_seconds=2.0)
        with self.assertRaises(HouseholdModelDiagnosticError):
            compare_household_model_presence_diagnostics((first, different_grid))


class HouseholdModelDiagnosticCLITests(unittest.TestCase):
    def test_cli_round_trip_is_path_free_and_never_selects_a_winner(self) -> None:
        admissions = (_admission("camera_01"), _admission("camera_02"))
        review = _review(admissions)
        scans = _scans(admissions)
        second_scans = _scans(admissions, "yolo11n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review_path = root / "review.json"
            admission_paths = tuple(root / f"admission_{index}.json" for index in range(2))
            scan_paths = tuple(root / f"scan_{index}.json" for index in range(2))
            second_scan_paths = tuple(root / f"second_scan_{index}.json" for index in range(2))
            first_report_path = root / "first-report.json"
            second_report_path = root / "second-report.json"
            review.write_new(review_path)
            for admission, path in zip(admissions, admission_paths, strict=True):
                path.write_text(f"{admission.to_json()}\n", encoding="utf-8")
            for scan, path in zip(scans, scan_paths, strict=True):
                scan.write_new(path)
            for scan, path in zip(second_scans, second_scan_paths, strict=True):
                scan.write_new(path)

            validate_output = StringIO()
            with redirect_stdout(validate_output):
                validate_code = main(["validate-household-model-presence-review", str(review_path)])
            self.assertEqual(0, validate_code)
            self.assertEqual(
                review.fingerprint,
                json.loads(validate_output.getvalue())["review_sha256"],
            )

            def evaluate(detector_paths: tuple[Path, ...], output_path: Path) -> dict[str, object]:
                arguments = [
                    "evaluate-household-model-presence-diagnostic",
                    "--review",
                    str(review_path),
                ]
                for path in admission_paths:
                    arguments.extend(("--media-admission", str(path)))
                for path in detector_paths:
                    arguments.extend(("--detector-scan", str(path)))
                arguments.extend(
                    (
                        "--threshold",
                        "0.1",
                        "--threshold",
                        "0.2",
                        "--threshold",
                        "0.9",
                        "--bin-seconds",
                        "1",
                        "--output",
                        str(output_path),
                    )
                )
                output = StringIO()
                with redirect_stdout(output):
                    self.assertEqual(0, main(arguments))
                self.assertNotIn(str(root), output.getvalue())
                return json.loads(output.getvalue())

            first_summary = evaluate(scan_paths, first_report_path)
            second_summary = evaluate(second_scan_paths, second_report_path)
            self.assertEqual(
                HouseholdModelDiagnosticReport.read(first_report_path).fingerprint,
                first_summary["report_sha256"],
            )

            selection_output = StringIO()
            with redirect_stdout(selection_output):
                selection_code = main(
                    [
                        "select-household-model-presence-working-point",
                        str(first_report_path),
                        "--minimum-precision",
                        "1",
                        "--minimum-recall",
                        "1",
                        "--minimum-partial-visibility-recall",
                        "1",
                    ]
                )
            selection = json.loads(selection_output.getvalue())
            self.assertEqual(0, selection_code)
            self.assertEqual(0.2, selection["selected_point"]["confidence_threshold"])
            self.assertFalse(selection["winner_selected"])
            self.assertFalse(selection["deployment_approved"])
            self.assertFalse(selection["frame_level_alignment_evaluated"])

            comparison_output = StringIO()
            with redirect_stdout(comparison_output):
                comparison_code = main(
                    [
                        "compare-household-model-presence-diagnostics",
                        str(first_report_path),
                        str(second_report_path),
                    ]
                )
            comparison = json.loads(comparison_output.getvalue())
            self.assertEqual(0, comparison_code)
            self.assertEqual(2, len(comparison["reports"]))
            self.assertFalse(comparison["winner_selected"])
            self.assertFalse(comparison["shadow_preference_selected"])
            self.assertNotIn(str(root), comparison_output.getvalue())
            self.assertNotEqual(first_summary["report_sha256"], second_summary["report_sha256"])


if __name__ == "__main__":
    unittest.main()
