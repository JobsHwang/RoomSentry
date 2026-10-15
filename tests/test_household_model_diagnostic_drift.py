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
    HouseholdModelDiagnosticDriftPolicy,
    HouseholdModelDiagnosticDriftReport,
    HouseholdModelDiagnosticError,
    HouseholdModelPresenceItem,
    HouseholdModelPresenceReview,
    HouseholdModelPresenceSource,
    compare_household_model_diagnostic_batches,
    evaluate_household_model_presence_diagnostic,
)
from baby_monitor.source_admission import ExternalSourceAdmission


def _admission(batch: str, camera_id: str) -> ExternalSourceAdmission:
    content = f"private household {batch} {camera_id}".encode()
    return ExternalSourceAdmission.from_dict(
        {
            "version": 1,
            "source_id": f"household_{batch}_{camera_id}",
            "asset_kind": "media_asset",
            "canonical_url": f"https://example.invalid/{batch}/{camera_id}",
            "accessed_on": "2026-08-29",
            "source_revision": f"{batch}_fixture_v1",
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
                "purpose": "model_evaluation",
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


def _review(
    batch: str,
    admissions: tuple[ExternalSourceAdmission, ExternalSourceAdmission],
) -> HouseholdModelPresenceReview:
    first, second = admissions

    def item(
        index: int,
        camera_id: str,
        admission: ExternalSourceAdmission,
        timestamp: float,
        visible_person: str,
        visibility: str,
    ) -> HouseholdModelPresenceItem:
        return HouseholdModelPresenceItem(
            review_item_id=f"{batch}_review_{index:03d}",
            camera_id=camera_id,
            media_admission_sha256=admission.fingerprint,
            timestamp_seconds=timestamp,
            source_frame_sha256=hashlib.sha256(f"{batch} frame {index}".encode()).hexdigest(),
            visible_person=visible_person,
            visibility=visibility,
            label_confidence=0.96 if visibility == "partial" else 0.98,
        )

    return HouseholdModelPresenceReview(
        review_id=f"household_model_review_{batch}",
        selection_manifest_sha256=hashlib.sha256(
            f"{batch} selection manifest".encode()
        ).hexdigest(),
        review_model_id="codex_visual_reviewer",
        review_model_revision="session_fixture",
        review_media_transfer="selected_stills_remote",
        sampling_strategy="candidate_disagreement_and_control",
        sources=(
            HouseholdModelPresenceSource("camera_01", first.fingerprint, 4.0),
            HouseholdModelPresenceSource("camera_02", second.fingerprint, 4.0),
        ),
        items=(
            item(1, "camera_01", first, 0.2, "absent", "clear"),
            item(2, "camera_01", first, 1.2, "present", "partial"),
            item(3, "camera_01", first, 1.6, "present", "partial"),
            item(4, "camera_01", first, 2.2, "unknown", "unknown"),
            item(5, "camera_02", second, 0.2, "present", "clear"),
            item(6, "camera_02", second, 1.2, "absent", "clear"),
        ),
    )


def _scan(
    admission: ExternalSourceAdmission,
    camera_id: str,
    values: tuple[tuple[float, float], ...],
) -> AdmittedDetectorScan:
    result = DetectorScanResult(
        camera_id=camera_id,
        detector_adapter="yolo_command",
        model_id="yolo26m",
        model_revision="fixture_v1",
        model_sha256=hashlib.sha256(b"yolo26m").hexdigest(),
        detections=tuple(
            Detection(timestamp, "person", confidence, (0.1, 0.1, 0.4, 0.9))
            for timestamp, confidence in values
        ),
    )
    return AdmittedDetectorScan(
        source_admission_sha256=admission.fingerprint,
        labels=("person",),
        confidence_threshold=0.1,
        requested_sample_fps=1.0,
        result=result,
    )


def _report(batch: str, *, weaker: bool):
    admissions = (_admission(batch, "camera_01"), _admission(batch, "camera_02"))
    review = _review(batch, admissions)
    if weaker:
        values = (((0.2, 0.3), (1.2, 0.4)), ())
    else:
        values = (((0.2, 0.15), (1.2, 0.9), (2.2, 0.2)), ((0.2, 0.8),))
    scans = tuple(
        _scan(admission, f"camera_{index:02d}", camera_values)
        for index, (admission, camera_values) in enumerate(
            zip(admissions, values, strict=True), start=1
        )
    )
    return evaluate_household_model_presence_diagnostic(
        review,
        media_admissions=admissions,
        detector_scans=scans,
        thresholds=(0.1, 0.2, 0.9),
        bin_seconds=1.0,
    )


class HouseholdModelDiagnosticDriftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = _report("baseline", weaker=False)
        self.current = _report("current", weaker=True)
        self.policy = HouseholdModelDiagnosticDriftPolicy(
            maximum_precision_drop=0.1,
            maximum_recall_drop=0.1,
            maximum_specificity_drop=0.1,
            maximum_partial_visibility_recall_drop=0.1,
        )

    def test_metrics_round_trip_schema_and_public_summary(self) -> None:
        report = compare_household_model_diagnostic_batches(
            self.baseline, self.current, self.policy
        )
        self.assertEqual(report, HouseholdModelDiagnosticDriftReport.from_json(report.to_json()))
        self.assertTrue(report.attention_required)
        self.assertEqual(3, report.attention_threshold_count)
        self.assertEqual(1, report.metric_availability_change_count)
        low, middle, high = report.points
        self.assertEqual(-0.166666666667, low.precision_delta)
        self.assertEqual(-0.5, low.recall_delta)
        self.assertTrue(low.precision_drop_exceeded)
        self.assertEqual(-0.5, middle.specificity_delta)
        self.assertTrue(middle.specificity_drop_exceeded)
        self.assertTrue(high.metric_availability_changed)
        self.assertEqual(-1.0, high.partial_visibility_recall_delta)

        schema = json.loads(
            Path("schemas/household-model-diagnostic-drift-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))
        public = report.to_public_dict()
        for hidden in (
            "camera_ids",
            "baseline_media_admission_sha256s",
            "current_media_admission_sha256s",
            "baseline_detector_scan_sha256s",
            "current_detector_scan_sha256s",
        ):
            self.assertNotIn(hidden, public)
        self.assertFalse(public["regression_confirmed"])
        self.assertFalse(public["winner_selected"])
        self.assertFalse(public["deployment_approved"])

    def test_rejects_reused_sources_mismatched_config_grid_and_authority(self) -> None:
        with self.assertRaises(HouseholdModelDiagnosticError):
            compare_household_model_diagnostic_batches(self.baseline, self.baseline, self.policy)
        reused_source = replace(
            self.current,
            media_admission_sha256s=self.baseline.media_admission_sha256s,
        )
        with self.assertRaises(HouseholdModelDiagnosticError):
            compare_household_model_diagnostic_batches(self.baseline, reused_source, self.policy)
        mismatched_model = replace(self.current, model_id="other_model")
        with self.assertRaises(HouseholdModelDiagnosticError):
            compare_household_model_diagnostic_batches(self.baseline, mismatched_model, self.policy)
        mismatched_grid = replace(self.current, bin_seconds=2.0)
        with self.assertRaises(HouseholdModelDiagnosticError):
            compare_household_model_diagnostic_batches(self.baseline, mismatched_grid, self.policy)
        report = compare_household_model_diagnostic_batches(
            self.baseline, self.current, self.policy
        )
        for field, value in (
            ("diagnostic_only", False),
            ("ground_truth", True),
            ("official_accuracy_claim", True),
            ("regression_confirmed", True),
            ("winner_selected", True),
            ("deployment_approved", True),
            ("household_configuration_changed", True),
            ("source_media_included", True),
            ("paths_disclosed", True),
        ):
            with (
                self.subTest(field=field),
                self.assertRaises(HouseholdModelDiagnosticError),
            ):
                replace(report, **{field: value})

    def test_rejects_tampering_wrong_shapes_duplicate_keys_and_nan(self) -> None:
        report = compare_household_model_diagnostic_batches(
            self.baseline, self.current, self.policy
        )
        tampered = json.loads(report.to_json())
        tampered["points"][0]["recall_delta"] = 0.0
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelDiagnosticDriftReport.from_dict(tampered)
        malformed = report.to_dict()
        malformed["points"] = {}
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelDiagnosticDriftReport.from_dict(malformed)
        for field, value in (
            ("camera_ids", [1, "camera_02"]),
            ("current_detector_scan_sha256s", [1, "a" * 64]),
        ):
            malformed = report.to_dict()
            malformed[field] = value
            with self.subTest(field=field), self.assertRaises(HouseholdModelDiagnosticError):
                HouseholdModelDiagnosticDriftReport.from_dict(malformed)
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelDiagnosticDriftReport.from_json('{"version":1,"version":1}')
        raw = report.to_json().replace(
            '"maximum_precision_drop": 0.1',
            '"maximum_precision_drop": NaN',
        )
        with self.assertRaises(HouseholdModelDiagnosticError):
            HouseholdModelDiagnosticDriftReport.from_json(raw)

    def test_cli_create_validate_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            current_path = root / "current.json"
            drift_path = root / "drift.json"
            self.baseline.write_new(baseline_path)
            self.current.write_new(current_path)
            output = StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "compare-household-model-diagnostic-batches",
                        "--baseline",
                        str(baseline_path),
                        "--current",
                        str(current_path),
                        "--maximum-precision-drop",
                        "0.1",
                        "--maximum-recall-drop",
                        "0.1",
                        "--maximum-specificity-drop",
                        "0.1",
                        "--maximum-partial-visibility-recall-drop",
                        "0.1",
                        "--output",
                        str(drift_path),
                    ]
                )
            self.assertEqual(0, code)
            self.assertNotIn(str(root), output.getvalue())
            created = HouseholdModelDiagnosticDriftReport.read(drift_path)
            self.assertEqual(
                created.fingerprint, json.loads(output.getvalue())["drift_report_sha256"]
            )

            validation_output = StringIO()
            with redirect_stdout(validation_output):
                validation_code = main(
                    ["validate-household-model-diagnostic-drift", str(drift_path)]
                )
            self.assertEqual(0, validation_code)
            self.assertNotIn(str(root), validation_output.getvalue())
            self.assertEqual(
                created.fingerprint,
                json.loads(validation_output.getvalue())["drift_report_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
