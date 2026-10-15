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
from baby_monitor.household_human_presence_audit import (
    HouseholdHumanFusionCameraPoint,
    HouseholdHumanModelAgreementCameraPoint,
    HouseholdHumanPoseFusionAuditReport,
    HouseholdHumanPresenceAudit,
    evaluate_household_human_pose_fusion_audit,
    import_household_human_presence_audit,
)
from baby_monitor.household_model_diagnostic import HouseholdModelDiagnosticError
from baby_monitor.household_pose_fusion_diagnostic import (
    combine_household_pose_fusion_diagnostic_batches,
    evaluate_household_pose_fusion_diagnostic,
)
from baby_monitor.household_pose_fusion_holdout import (
    attest_household_pose_fusion_temporal_holdout,
)
from tests.test_household_model_diagnostic import _admission, _review, _scans
from tests.test_household_pose_fusion_diagnostic import (
    _batch_admissions,
    _pose_scans,
    _profile,
    _report,
)


def _canonical_sha256(value: object, *, newline: bool = False) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if newline:
        payload += b"\n"
    return hashlib.sha256(payload).hexdigest()


def _evidence():
    admissions = (_admission("camera_01"), _admission("camera_02"))
    review = _review(admissions)
    detector_scans = _scans(admissions)
    pose_scans = _pose_scans(admissions)
    profile = _profile()
    selection, _audit_manifest, _result = _raw_inputs(
        review,
        detector_scans,
        pose_scans,
        profile,
    )
    review = replace(
        review,
        selection_manifest_sha256=selection["selection_manifest_sha256"],
    )
    holdout = evaluate_household_pose_fusion_diagnostic(
        review,
        media_admissions=admissions,
        detector_scans=detector_scans,
        pose_scans=pose_scans,
        profile=profile,
        bin_seconds=1.0,
    )
    selection_batch = combine_household_pose_fusion_diagnostic_batches(
        (
            _report(
                _batch_admissions("human_audit_selection_a"),
                review_id="human_audit_selection_a",
            ),
            _report(
                _batch_admissions("human_audit_selection_b"),
                review_id="human_audit_selection_b",
            ),
        )
    )
    attestation = attest_household_pose_fusion_temporal_holdout(
        selection_batch,
        holdout,
        confirm_selection_batch_used_for_profile_selection=True,
        confirm_profile_frozen_before_holdout_capture=True,
    )
    return admissions, review, detector_scans, pose_scans, profile, attestation


def _raw_inputs(review, detector_scans, pose_scans, profile):
    chosen = (review.items[0], review.items[1], review.items[4], review.items[5])
    detector_by_camera = {scan.result.camera_id: scan for scan in detector_scans}
    pose_by_camera = {scan.result.camera_id: scan for scan in pose_scans}
    selection_items = []
    for index, item in enumerate(chosen, start=1):
        selection_items.append(
            {
                "review_id": item.review_item_id,
                "camera_id": item.camera_id,
                "media_admission_sha256": item.media_admission_sha256,
                "detector_scan_sha256": detector_by_camera[item.camera_id].fingerprint,
                "pose_scan_sha256": pose_by_camera[item.camera_id].fingerprint,
                "timestamp_seconds": item.timestamp_seconds,
                "bin_index": int(item.timestamp_seconds // 1.0),
                "source_frame_sha256": item.source_frame_sha256,
                "frame_filename": f"fixture_frame_{index:02d}.jpg",
                "selection_source": "uniform_holdout",
                "detector_max_confidence_nearby": None,
                "pose_max_confidence_nearby": None,
                "sheet_id": "sheet_01",
                "sheet_panel_index": index - 1,
            }
        )
    selection = {
        "version": 1,
        "batch_id": "human_audit_fixture_batch",
        "profile_sha256": profile.fingerprint,
        "profile_frozen_before_capture": True,
        "source_count": 2,
        "frame_count": 4,
        "max_frames": 4,
        "bin_seconds": 1.0,
        "sampling_strategy": review.sampling_strategy,
        "model_generated_selection": True,
        "consent_recorded": True,
        "selected_stills_only": True,
        "clips_shared": False,
        "audio_shared": False,
        "identity_inference": False,
        "biometric_processing": False,
        "training_allowed": False,
        "redistribution_allowed": False,
        "human_verified": False,
        "ground_truth": False,
        "production_remote_vlm_changed": False,
        "items": selection_items,
    }
    selection["selection_manifest_sha256"] = _canonical_sha256(selection)

    binding_items = [
        {
            "review_item_id": f"sample_{index:02d}",
            "source_review_id": item["review_id"],
            "camera_id": item["camera_id"],
            "media_admission_sha256": item["media_admission_sha256"],
            "bin_index": item["bin_index"],
            "timestamp_seconds": item["timestamp_seconds"],
            "source_frame_sha256": item["source_frame_sha256"],
        }
        for index, item in enumerate(selection_items, start=1)
    ]
    audit_manifest = {
        "version": 1,
        "audit_type": "independent_human_frame_presence_audit",
        "audit_id": "human_audit_fixture_01",
        "created_at_utc": "2026-08-30T01:00:00Z",
        "source_selection_manifest_sha256": selection["selection_manifest_sha256"],
        "sampling": {
            "source": "uniform_holdout_only",
            "camera_count": 2,
            "frames_per_camera": 2,
            "frame_count": 4,
            "within_camera_strategy": "evenly_spaced_order_statistics",
            "presentation_order": "deterministic_blinded_shuffle",
        },
        "privacy": {
            "local_only": True,
            "network_requests_required": False,
            "clips_included": False,
            "audio_included": False,
            "identity_requested": False,
            "model_outputs_disclosed": False,
            "existing_model_labels_disclosed": False,
            "fusion_profile_disclosed": False,
        },
        "claims": {
            "human_review_completed": False,
            "human_verified": False,
            "ground_truth": False,
            "official_accuracy_claim": False,
            "deployment_change_authorized": False,
        },
        "expected_download_filename": "human_audit_fixture_01.completed.json",
        "items": binding_items,
    }
    audit_manifest["audit_manifest_sha256"] = _canonical_sha256(
        audit_manifest,
        newline=True,
    )
    labels = ("absent", "present", "present", "present")
    result = {
        "version": 1,
        "review_type": "independent_human_frame_presence_audit_result",
        "audit_id": audit_manifest["audit_id"],
        "completed_at_utc": "2026-08-30T01:01:00.000Z",
        "source_selection_manifest_sha256": selection["selection_manifest_sha256"],
        "audit_manifest_sha256": audit_manifest["audit_manifest_sha256"],
        "human_review_completed": True,
        "human_verified": True,
        "labels_created_without_model_outputs": True,
        "ground_truth": False,
        "official_accuracy_claim": False,
        "deployment_change_authorized": False,
        "items": [
            {**item, "visible_person": label}
            for item, label in zip(binding_items, labels, strict=True)
        ],
    }
    return selection, audit_manifest, result


def _write_raw_inputs(root: Path, selection, audit_manifest, result):
    selection_path = root / "selection.json"
    manifest_path = root / "audit-manifest.json"
    result_path = root / "result.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    manifest_path.write_text(json.dumps(audit_manifest), encoding="utf-8")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return selection_path, manifest_path, result_path


class HouseholdHumanPresenceAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        (
            self.admissions,
            self.review,
            self.detector_scans,
            self.pose_scans,
            self.profile,
            self.attestation,
        ) = _evidence()
        self.selection, self.audit_manifest, self.result = _raw_inputs(
            self.review,
            self.detector_scans,
            self.pose_scans,
            self.profile,
        )

    def _import(self, root: Path) -> HouseholdHumanPresenceAudit:
        selection_path, manifest_path, result_path = _write_raw_inputs(
            root,
            self.selection,
            self.audit_manifest,
            self.result,
        )
        return import_household_human_presence_audit(
            manifest_path,
            result_path,
            selection_path,
        )

    def test_import_round_trip_schema_authority_and_exclusive_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = self._import(root)
            self.assertEqual(4, len(audit.items))
            self.assertEqual(3, audit.present_frame_count)
            self.assertEqual(1, audit.absent_frame_count)
            self.assertEqual(0, audit.unknown_frame_count)
            self.assertTrue(audit.human_verified)
            self.assertFalse(audit.ground_truth)
            self.assertFalse(audit.independent_human_ground_truth)
            self.assertFalse(audit.official_accuracy_claim)
            self.assertEqual(audit, HouseholdHumanPresenceAudit.from_json(audit.to_json()))

            schema = json.loads(
                Path("schemas/household-human-presence-audit-v1.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(set(schema["required"]), set(audit.to_dict()))
            self.assertEqual(
                set(schema["$defs"]["item"]["required"]),
                set(audit.items[0].to_dict()),
            )

            output = root / "audit.json"
            audit.write_new(output)
            self.assertEqual(0o600, output.stat().st_mode & 0o777)
            with self.assertRaises(FileExistsError):
                audit.write_new(output)
            for field, value in (
                ("ground_truth", True),
                ("independent_human_ground_truth", True),
                ("official_accuracy_claim", True),
                ("deployment_approved", True),
                ("blindness_cryptographically_proven", True),
            ):
                with self.subTest(field=field), self.assertRaises(HouseholdModelDiagnosticError):
                    replace(audit, **{field: value})

    def test_import_rejects_binding_disclosure_and_nonuniform_selection_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_result = json.loads(json.dumps(self.result))
            bad_result["items"][0]["source_frame_sha256"] = hashlib.sha256(
                b"tampered frame"
            ).hexdigest()
            selection_path, manifest_path, result_path = _write_raw_inputs(
                root,
                self.selection,
                self.audit_manifest,
                bad_result,
            )
            with self.assertRaisesRegex(
                HouseholdModelDiagnosticError,
                "binding differs",
            ):
                import_household_human_presence_audit(
                    manifest_path,
                    result_path,
                    selection_path,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            disclosed = json.loads(json.dumps(self.audit_manifest))
            disclosed["privacy"]["model_outputs_disclosed"] = True
            unsigned = dict(disclosed)
            unsigned.pop("audit_manifest_sha256")
            disclosed["audit_manifest_sha256"] = _canonical_sha256(
                unsigned,
                newline=True,
            )
            result = json.loads(json.dumps(self.result))
            result["audit_manifest_sha256"] = disclosed["audit_manifest_sha256"]
            selection_path, manifest_path, result_path = _write_raw_inputs(
                root,
                self.selection,
                disclosed,
                result,
            )
            with self.assertRaisesRegex(HouseholdModelDiagnosticError, "must be false"):
                import_household_human_presence_audit(
                    manifest_path,
                    result_path,
                    selection_path,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = json.loads(json.dumps(self.selection))
            selection["items"][0]["selection_source"] = "pose_candidate"
            selection.pop("selection_manifest_sha256")
            selection["selection_manifest_sha256"] = _canonical_sha256(selection)
            manifest = json.loads(json.dumps(self.audit_manifest))
            manifest["source_selection_manifest_sha256"] = selection["selection_manifest_sha256"]
            manifest.pop("audit_manifest_sha256")
            manifest["audit_manifest_sha256"] = _canonical_sha256(manifest, newline=True)
            result = json.loads(json.dumps(self.result))
            result["source_selection_manifest_sha256"] = selection["selection_manifest_sha256"]
            result["audit_manifest_sha256"] = manifest["audit_manifest_sha256"]
            selection_path, manifest_path, result_path = _write_raw_inputs(
                root,
                selection,
                manifest,
                result,
            )
            with self.assertRaisesRegex(HouseholdModelDiagnosticError, "not a uniform"):
                import_household_human_presence_audit(
                    manifest_path,
                    result_path,
                    selection_path,
                )

    def test_evaluates_model_agreement_and_exact_frozen_fusion_without_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = self._import(Path(directory))
        report = evaluate_household_human_pose_fusion_audit(
            audit,
            model_review=self.review,
            temporal_holdout_attestation=self.attestation,
            media_admissions=self.admissions,
            detector_scans=self.detector_scans,
            pose_scans=self.pose_scans,
        )
        self.assertEqual(3, report.model_agreement.agreement_count)
        self.assertEqual(1, report.model_agreement.disagreement_count)
        self.assertEqual(0.75, report.model_agreement.agreement_rate)
        self.assertEqual(
            (2, 0, 1, 1),
            (
                report.fusion_metrics.true_positive,
                report.fusion_metrics.false_positive,
                report.fusion_metrics.false_negative,
                report.fusion_metrics.true_negative,
            ),
        )
        self.assertEqual(0.666666666667, report.fusion_metrics.recall)
        self.assertEqual(1.0, report.fusion_metrics.specificity)
        self.assertTrue(report.human_verified)
        self.assertFalse(report.ground_truth)
        self.assertFalse(report.independent_human_ground_truth)
        self.assertFalse(report.temporal_episode_ground_truth)
        self.assertFalse(report.official_accuracy_claim)
        self.assertFalse(report.deployment_approved)
        self.assertEqual(
            report,
            HouseholdHumanPoseFusionAuditReport.from_json(report.to_json()),
        )

        schema = json.loads(
            Path("schemas/household-human-pose-fusion-audit-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))
        self.assertEqual(
            set(schema["$defs"]["agreementCameraPoint"]["required"]),
            set(HouseholdHumanModelAgreementCameraPoint.__dataclass_fields__),
        )
        self.assertEqual(
            set(schema["$defs"]["fusionCameraPoint"]["required"]),
            set(HouseholdHumanFusionCameraPoint.__dataclass_fields__),
        )

        with self.assertRaisesRegex(HouseholdModelDiagnosticError, "scans"):
            evaluate_household_human_pose_fusion_audit(
                audit,
                model_review=self.review,
                temporal_holdout_attestation=self.attestation,
                media_admissions=self.admissions,
                detector_scans=(self.detector_scans[0], self.detector_scans[0]),
                pose_scans=self.pose_scans,
            )
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(report, official_accuracy_claim=True)

    def test_cli_import_evaluate_and_validate_are_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path, manifest_path, result_path = _write_raw_inputs(
                root,
                self.selection,
                self.audit_manifest,
                self.result,
            )
            audit_path = root / "audit.json"
            review_path = root / "model-review.json"
            attestation_path = root / "holdout-attestation.json"
            report_path = root / "report.json"
            self.review.write_new(review_path)
            self.attestation.write_new(attestation_path)
            admission_paths = []
            detector_paths = []
            pose_paths = []
            for index, admission in enumerate(self.admissions):
                admission_path = root / f"admission-{index}.json"
                detector_path = root / f"detector-{index}.json"
                pose_path = root / f"pose-{index}.json"
                admission_path.write_text(admission.to_json(), encoding="utf-8")
                self.detector_scans[index].write_new(detector_path)
                self.pose_scans[index].write_new(pose_path)
                admission_paths.append(admission_path)
                detector_paths.append(detector_path)
                pose_paths.append(pose_path)

            outputs = []
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "import-household-human-presence-audit",
                            "--audit-manifest",
                            str(manifest_path),
                            "--result",
                            str(result_path),
                            "--selection-manifest",
                            str(selection_path),
                            "--output",
                            str(audit_path),
                        ]
                    ),
                )
            outputs.append(stdout.getvalue())

            args = [
                "evaluate-household-human-pose-fusion-audit",
                "--audit",
                str(audit_path),
                "--model-review",
                str(review_path),
                "--temporal-holdout-attestation",
                str(attestation_path),
                "--output",
                str(report_path),
            ]
            for path in admission_paths:
                args.extend(("--media-admission", str(path)))
            for path in detector_paths:
                args.extend(("--detector-scan", str(path)))
            for path in pose_paths:
                args.extend(("--pose-scan", str(path)))
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(0, main(args))
            outputs.append(stdout.getvalue())

            for command, path in (
                ("validate-household-human-presence-audit", audit_path),
                ("validate-household-human-pose-fusion-audit", report_path),
            ):
                stdout = StringIO()
                with redirect_stdout(stdout):
                    self.assertEqual(0, main([command, str(path)]))
                outputs.append(stdout.getvalue())
            combined = "".join(outputs)
            self.assertNotIn(str(root), combined)
            self.assertNotIn("media_admission_sha256s", combined)
            self.assertIn('"human_verified": true', combined)
            self.assertIn('"ground_truth": false', combined)
            self.assertIn('"official_accuracy_claim": false', combined)
            self.assertIn('"paths_disclosed": false', combined)


if __name__ == "__main__":
    unittest.main()
