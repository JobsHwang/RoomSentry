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
from baby_monitor.household_pose_fusion_diagnostic import (
    HouseholdPoseFusionBatchBinding,
    HouseholdPoseFusionBatchDiagnosticReport,
    HouseholdPoseFusionCameraProfile,
    HouseholdPoseFusionDiagnosticReport,
    HouseholdPoseFusionMetrics,
    HouseholdPoseFusionProfile,
    combine_household_pose_fusion_diagnostic_batches,
    evaluate_household_pose_fusion_diagnostic,
)
from baby_monitor.pose_protocol import (
    COCO_PERSON_KEYPOINTS,
    AdmittedPoseScan,
    PoseFrame,
    PoseFrameBatch,
    PoseInstance,
    PoseKeypoint,
)
from baby_monitor.source_admission import ExternalSourceAdmission
from tests.test_household_model_diagnostic import _admission, _review, _scans


def _keypoints(usable_count: int) -> tuple[PoseKeypoint, ...]:
    return tuple(
        PoseKeypoint(
            name=name,
            x=0.5,
            y=0.5,
            confidence=0.9 if index < usable_count else 0.1,
            in_frame=True,
        )
        for index, name in enumerate(COCO_PERSON_KEYPOINTS)
    )


def _pose_scan(
    admission: ExternalSourceAdmission,
    camera_id: str,
    *,
    usable_keypoints: int,
) -> AdmittedPoseScan:
    batch = PoseFrameBatch(
        camera_id=camera_id,
        pose_adapter="fixture_pose",
        model_id="fixture_pose_model",
        model_revision="fixture_v1",
        model_sha256=hashlib.sha256(b"fixture pose model").hexdigest(),
        frames=(
            PoseFrame(
                frame_id="frame_0001",
                timestamp_seconds=0.2,
                poses=(
                    PoseInstance(
                        instance_id="pose_0001",
                        confidence=0.8,
                        bbox_xyxy=(0.1, 0.1, 0.5, 0.9),
                        keypoints=_keypoints(usable_keypoints),
                    ),
                ),
            ),
        ),
    )
    return AdmittedPoseScan(
        source_admission_sha256=admission.fingerprint,
        confidence_threshold=0.1,
        requested_sample_fps=1.0,
        requested_max_frames=10,
        image_size=640,
        rotation_degrees=0,
        result=batch,
    )


def _pose_scans(
    admissions: tuple[ExternalSourceAdmission, ...],
) -> tuple[AdmittedPoseScan, ...]:
    return (
        _pose_scan(admissions[0], "camera_01", usable_keypoints=3),
        _pose_scan(admissions[1], "camera_02", usable_keypoints=10),
    )


def _profile(*, camera_02_pose_hold: float = 0.0) -> HouseholdPoseFusionProfile:
    return HouseholdPoseFusionProfile(
        profile_id="fixture_camera_profile_v1",
        cameras=(
            HouseholdPoseFusionCameraProfile(
                camera_id="camera_01",
                detector_confidence_threshold=0.9,
                detector_hold_seconds=0.0,
                pose_enabled=True,
                pose_confidence_threshold=0.5,
                pose_hold_seconds=0.0,
                minimum_pose_bbox_area=0.05,
                minimum_pose_bbox_height=0.2,
                keypoint_confidence_threshold=0.5,
                minimum_keypoint_count=8,
            ),
            HouseholdPoseFusionCameraProfile(
                camera_id="camera_02",
                detector_confidence_threshold=0.9,
                detector_hold_seconds=0.0,
                pose_enabled=True,
                pose_confidence_threshold=0.5,
                pose_hold_seconds=camera_02_pose_hold,
                minimum_pose_bbox_area=0.05,
                minimum_pose_bbox_height=0.2,
                keypoint_confidence_threshold=0.5,
                minimum_keypoint_count=8,
            ),
        ),
    )


def _report(
    admissions: tuple[ExternalSourceAdmission, ...],
    *,
    review_id: str = "fixture",
    profile: HouseholdPoseFusionProfile | None = None,
) -> HouseholdPoseFusionDiagnosticReport:
    review = _review(admissions)
    if review_id != "fixture":
        review = replace(
            review,
            review_id=f"household_pose_fusion_{review_id}",
            selection_manifest_sha256=hashlib.sha256(f"selection {review_id}".encode()).hexdigest(),
            items=tuple(
                replace(
                    item,
                    review_item_id=f"{review_id}_{index:03d}",
                    source_frame_sha256=hashlib.sha256(
                        f"{review_id} frame {index}".encode()
                    ).hexdigest(),
                )
                for index, item in enumerate(review.items, start=1)
            ),
        )
    return evaluate_household_pose_fusion_diagnostic(
        review,
        media_admissions=admissions,
        detector_scans=_scans(admissions),
        pose_scans=_pose_scans(admissions),
        profile=_profile() if profile is None else profile,
        bin_seconds=1.0,
    )


def _batch_admissions(batch_id: str) -> tuple[ExternalSourceAdmission, ...]:
    result: list[ExternalSourceAdmission] = []
    for camera_id in ("camera_01", "camera_02"):
        payload = _admission(camera_id).to_dict()
        content = f"private household {batch_id} {camera_id}".encode()
        payload["source_id"] = f"household_{batch_id}_{camera_id}"
        payload["canonical_url"] = f"https://example.invalid/{batch_id}/{camera_id}"
        payload["source_revision"] = f"fixture_{batch_id}"
        payload["access"]["expected_bytes"] = len(content)
        payload["content"]["observed_bytes"] = len(content)
        payload["content"]["sha256"] = hashlib.sha256(content).hexdigest()
        result.append(ExternalSourceAdmission.from_dict(payload))
    return tuple(result)


class HouseholdPoseFusionDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.admissions = (_admission("camera_01"), _admission("camera_02"))
        self.review = _review(self.admissions)
        self.detector_scans = _scans(self.admissions)
        self.pose_scans = _pose_scans(self.admissions)
        self.profile = _profile()

    def _evaluate(
        self,
        *,
        review=None,
        pose_scans=None,
        profile=None,
    ) -> HouseholdPoseFusionDiagnosticReport:
        return evaluate_household_pose_fusion_diagnostic(
            self.review if review is None else review,
            media_admissions=self.admissions,
            detector_scans=self.detector_scans,
            pose_scans=self.pose_scans if pose_scans is None else pose_scans,
            profile=self.profile if profile is None else profile,
            bin_seconds=1.0,
        )

    def test_pose_rescues_detection_but_geometry_keypoint_gate_blocks_clutter(self) -> None:
        report = self._evaluate()

        self.assertEqual(6, report.reviewed_frame_count)
        self.assertEqual(4, report.evaluated_bin_count)
        self.assertEqual(2, report.positive_bin_count)
        self.assertEqual(2, report.negative_bin_count)
        self.assertEqual(1, report.unknown_bin_count)
        self.assertEqual(0, report.conflicting_bin_count)
        self.assertEqual(
            (2, 0, 0, 2),
            (
                report.metrics.true_positive,
                report.metrics.false_positive,
                report.metrics.false_negative,
                report.metrics.true_negative,
            ),
        )
        self.assertEqual(1.0, report.metrics.recall)
        self.assertEqual(1.0, report.metrics.specificity)
        self.assertFalse(report.deployment_approved)
        self.assertTrue(report.profile_selected_on_evaluation_data)
        self.assertFalse(report.posture_classification)

    def test_pose_hold_is_independent_and_conflicting_bins_are_explicitly_excluded(self) -> None:
        held = self._evaluate(profile=_profile(camera_02_pose_hold=1.0))
        self.assertEqual(1, held.metrics.false_positive)
        self.assertEqual(1, held.metrics.true_negative)

        items = list(self.review.items)
        items[2] = replace(items[2], visible_person="absent", visibility="clear")
        conflicting = self._evaluate(review=replace(self.review, items=tuple(items)))
        self.assertEqual(1, conflicting.conflicting_bin_count)
        self.assertEqual(3, conflicting.evaluated_bin_count)
        self.assertEqual(1, conflicting.positive_bin_count)
        self.assertTrue(conflicting.conflicting_bins_excluded)
        self.assertEqual(
            "exclude_present_absent_conflicts",
            conflicting.conflict_resolution_strategy,
        )

    def test_profile_scan_binding_round_trip_schemas_and_authority(self) -> None:
        report = self._evaluate()
        self.assertEqual(report, HouseholdPoseFusionDiagnosticReport.from_json(report.to_json()))
        self.assertEqual(self.profile, HouseholdPoseFusionProfile.from_json(self.profile.to_json()))

        profile_schema = json.loads(
            Path("schemas/household-pose-fusion-profile-v1.schema.json").read_text(encoding="utf-8")
        )
        report_schema = json.loads(
            Path("schemas/household-pose-fusion-diagnostic-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(profile_schema["required"]), set(self.profile.to_dict()))
        self.assertEqual(
            set(profile_schema["$defs"]["cameraProfile"]["required"]),
            set(HouseholdPoseFusionCameraProfile.__dataclass_fields__),
        )
        self.assertEqual(set(report_schema["required"]), set(report.to_dict()))
        self.assertEqual(
            set(report_schema["$defs"]["metrics"]["required"]),
            set(HouseholdPoseFusionMetrics.__dataclass_fields__),
        )

        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(report, deployment_approved=True)
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(report, pose_source_confidence_threshold=0.6)
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(self.profile, profile_selected_on_evaluation_data=False)
        with self.assertRaises(HouseholdModelDiagnosticError):
            self._evaluate(
                profile=replace(
                    self.profile,
                    cameras=(
                        replace(
                            self.profile.cameras[0],
                            pose_confidence_threshold=0.05,
                        ),
                        self.profile.cameras[1],
                    ),
                )
            )
        with self.assertRaises(HouseholdModelDiagnosticError):
            self._evaluate(
                pose_scans=(
                    self.pose_scans[0],
                    replace(
                        self.pose_scans[1],
                        source_admission_sha256=self.admissions[0].fingerprint,
                    ),
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            report_path = root / "report.json"
            self.profile.write_new(profile_path)
            report.write_new(report_path)
            self.assertEqual(0o600, profile_path.stat().st_mode & 0o777)
            self.assertEqual(0o600, report_path.stat().st_mode & 0o777)
            with self.assertRaises(FileExistsError):
                report.write_new(report_path)


class HouseholdPoseFusionBatchTests(unittest.TestCase):
    def _reports(self):
        first_admissions = _batch_admissions("batch_a")
        second_admissions = _batch_admissions("batch_b")
        return (
            _report(first_admissions, review_id="batch_a"),
            _report(second_admissions, review_id="batch_b"),
        )

    def test_combines_disjoint_reports_and_rejects_overlap(self) -> None:
        first, second = self._reports()
        combined = combine_household_pose_fusion_diagnostic_batches((second, first))

        self.assertEqual(2, len(combined.batches))
        self.assertEqual(8, combined.evaluated_bin_count)
        self.assertEqual(4, combined.positive_bin_count)
        self.assertEqual(4, combined.negative_bin_count)
        self.assertEqual(2, combined.unknown_bin_count)
        self.assertEqual(
            (4, 0, 0, 4),
            (
                combined.metrics.true_positive,
                combined.metrics.false_positive,
                combined.metrics.false_negative,
                combined.metrics.true_negative,
            ),
        )
        self.assertEqual(
            combined,
            HouseholdPoseFusionBatchDiagnosticReport.from_json(combined.to_json()),
        )
        schema = json.loads(
            Path("schemas/household-pose-fusion-batch-diagnostic-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(combined.to_dict()))
        self.assertEqual(
            set(schema["$defs"]["batchBinding"]["required"]),
            set(HouseholdPoseFusionBatchBinding.__dataclass_fields__),
        )
        with self.assertRaises(HouseholdModelDiagnosticError):
            combine_household_pose_fusion_diagnostic_batches((first, first))
        with self.assertRaises(HouseholdModelDiagnosticError):
            combine_household_pose_fusion_diagnostic_batches(
                (first, replace(second, profile=_profile(camera_02_pose_hold=1.0)))
            )
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(combined, official_accuracy_claim=True)
        with self.assertRaises(HouseholdModelDiagnosticError):
            replace(combined, detector_source_confidence_threshold=0.95)

    def test_cli_evaluate_validate_combine_is_path_free(self) -> None:
        first_admissions = _batch_admissions("cli_a")
        second_admissions = _batch_admissions("cli_b")
        first_review = _review(first_admissions)
        second_review = replace(
            _review(second_admissions),
            review_id="household_pose_fusion_cli_b",
            selection_manifest_sha256=hashlib.sha256(b"cli b selection").hexdigest(),
            items=tuple(
                replace(
                    item,
                    review_item_id=f"cli_b_{index:03d}",
                    source_frame_sha256=hashlib.sha256(f"cli b frame {index}".encode()).hexdigest(),
                )
                for index, item in enumerate(_review(second_admissions).items, start=1)
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            _profile().write_new(profile_path)
            report_paths: list[Path] = []
            outputs: list[str] = []
            for batch_index, (admissions, review) in enumerate(
                (
                    (first_admissions, first_review),
                    (second_admissions, second_review),
                )
            ):
                review_path = root / f"review-{batch_index}.json"
                review.write_new(review_path)
                admission_paths = []
                detector_paths = []
                pose_paths = []
                for source_index, admission in enumerate(admissions):
                    admission_path = root / f"admission-{batch_index}-{source_index}.json"
                    detector_path = root / f"detector-{batch_index}-{source_index}.json"
                    pose_path = root / f"pose-{batch_index}-{source_index}.json"
                    admission_path.write_text(admission.to_json(), encoding="utf-8")
                    _scans(admissions)[source_index].write_new(detector_path)
                    _pose_scans(admissions)[source_index].write_new(pose_path)
                    admission_paths.append(admission_path)
                    detector_paths.append(detector_path)
                    pose_paths.append(pose_path)
                report_path = root / f"report-{batch_index}.json"
                args = [
                    "evaluate-household-pose-fusion-diagnostic",
                    "--review",
                    str(review_path),
                    "--profile",
                    str(profile_path),
                    "--bin-seconds",
                    "1",
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
                report_paths.append(report_path)

            profile_stdout = StringIO()
            with redirect_stdout(profile_stdout):
                self.assertEqual(
                    0,
                    main(["validate-household-pose-fusion-profile", str(profile_path)]),
                )
            validate_stdout = StringIO()
            with redirect_stdout(validate_stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "validate-household-pose-fusion-diagnostic",
                            str(report_paths[0]),
                        ]
                    ),
                )
            batch_path = root / "batch.json"
            combine_stdout = StringIO()
            with redirect_stdout(combine_stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "combine-household-pose-fusion-diagnostic-batches",
                            *(str(path) for path in report_paths),
                            "--output",
                            str(batch_path),
                        ]
                    ),
                )
            batch_stdout = StringIO()
            with redirect_stdout(batch_stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "validate-household-pose-fusion-batch-diagnostic",
                            str(batch_path),
                        ]
                    ),
                )
            combined_output = "".join(
                [
                    *outputs,
                    profile_stdout.getvalue(),
                    validate_stdout.getvalue(),
                    combine_stdout.getvalue(),
                    batch_stdout.getvalue(),
                ]
            )
            self.assertNotIn(str(root), combined_output)
            self.assertFalse(json.loads(batch_stdout.getvalue())["deployment_approved"])


if __name__ == "__main__":
    unittest.main()
