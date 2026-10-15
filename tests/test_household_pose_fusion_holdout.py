from __future__ import annotations

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
    HouseholdPoseFusionBatchDiagnosticReport,
    HouseholdPoseFusionDiagnosticReport,
    combine_household_pose_fusion_diagnostic_batches,
)
from baby_monitor.household_pose_fusion_holdout import (
    HouseholdPoseFusionTemporalHoldoutAttestation,
    attest_household_pose_fusion_temporal_holdout,
)
from tests.test_household_pose_fusion_diagnostic import (
    _batch_admissions,
    _profile,
    _report,
)


def _evidence() -> tuple[
    HouseholdPoseFusionBatchDiagnosticReport,
    HouseholdPoseFusionDiagnosticReport,
]:
    first = _report(
        _batch_admissions("holdout_selection_a"),
        review_id="holdout_selection_a",
    )
    second = _report(
        _batch_admissions("holdout_selection_b"),
        review_id="holdout_selection_b",
    )
    holdout = _report(
        _batch_admissions("holdout_temporal_c"),
        review_id="holdout_temporal_c",
    )
    return (
        combine_household_pose_fusion_diagnostic_batches((first, second)),
        holdout,
    )


class HouseholdPoseFusionTemporalHoldoutTests(unittest.TestCase):
    def test_round_trip_schema_authority_and_exclusive_write(self) -> None:
        selection_batch, holdout = _evidence()
        attestation = attest_household_pose_fusion_temporal_holdout(
            selection_batch,
            holdout,
            confirm_selection_batch_used_for_profile_selection=True,
            confirm_profile_frozen_before_holdout_capture=True,
        )

        self.assertEqual(
            attestation,
            HouseholdPoseFusionTemporalHoldoutAttestation.from_json(attestation.to_json()),
        )
        self.assertTrue(attestation.temporal_holdout)
        self.assertTrue(attestation.profile_frozen_before_holdout_capture)
        self.assertFalse(attestation.profile_selected_on_holdout)
        self.assertFalse(attestation.capture_order_cryptographically_proven)
        self.assertFalse(attestation.independent_human_ground_truth)
        self.assertFalse(attestation.official_accuracy_claim)
        self.assertFalse(attestation.deployment_approved)

        schema = json.loads(
            Path(
                "schemas/household-pose-fusion-temporal-holdout-attestation-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(attestation.to_dict()))

        for field, value in (
            ("capture_order_cryptographically_proven", True),
            ("review_model_independence_established", True),
            ("human_verified", True),
            ("official_accuracy_claim", True),
            ("deployment_approved", True),
        ):
            with (
                self.subTest(field=field),
                self.assertRaises(HouseholdModelDiagnosticError),
            ):
                replace(attestation, **{field: value})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "holdout-attestation.json"
            attestation.write_new(path)
            self.assertEqual(0o600, path.stat().st_mode & 0o777)
            with self.assertRaises(FileExistsError):
                attestation.write_new(path)

    def test_requires_confirmations_and_rejects_overlap_or_profile_change(self) -> None:
        selection_batch, holdout = _evidence()
        with self.assertRaisesRegex(
            HouseholdModelDiagnosticError,
            "selection-batch confirmation",
        ):
            attest_household_pose_fusion_temporal_holdout(
                selection_batch,
                holdout,
                confirm_selection_batch_used_for_profile_selection=False,
                confirm_profile_frozen_before_holdout_capture=True,
            )
        with self.assertRaisesRegex(
            HouseholdModelDiagnosticError,
            "frozen-profile confirmation",
        ):
            attest_household_pose_fusion_temporal_holdout(
                selection_batch,
                holdout,
                confirm_selection_batch_used_for_profile_selection=True,
                confirm_profile_frozen_before_holdout_capture=False,
            )

        overlapping = _report(
            _batch_admissions("holdout_selection_a"),
            review_id="holdout_selection_a",
        )
        with self.assertRaisesRegex(
            HouseholdModelDiagnosticError,
            "reports must be distinct",
        ):
            attest_household_pose_fusion_temporal_holdout(
                selection_batch,
                overlapping,
                confirm_selection_batch_used_for_profile_selection=True,
                confirm_profile_frozen_before_holdout_capture=True,
            )

        changed_profile = _profile(camera_02_pose_hold=1.0)
        changed_holdout = _report(
            _batch_admissions("holdout_temporal_changed"),
            review_id="holdout_temporal_changed",
            profile=changed_profile,
        )
        with self.assertRaisesRegex(
            HouseholdModelDiagnosticError,
            "profile does not match",
        ):
            attest_household_pose_fusion_temporal_holdout(
                selection_batch,
                changed_holdout,
                confirm_selection_batch_used_for_profile_selection=True,
                confirm_profile_frozen_before_holdout_capture=True,
            )

    def test_cli_attest_and_validate_are_path_free(self) -> None:
        selection_batch, holdout = _evidence()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection_path = root / "selection.json"
            holdout_path = root / "holdout.json"
            output_path = root / "attestation.json"
            selection_batch.write_new(selection_path)
            holdout.write_new(holdout_path)

            created = StringIO()
            with redirect_stdout(created):
                self.assertEqual(
                    0,
                    main(
                        [
                            "attest-household-pose-fusion-temporal-holdout",
                            "--selection-batch",
                            str(selection_path),
                            "--holdout-report",
                            str(holdout_path),
                            "--confirm-selection-batch-used-for-profile-selection",
                            "--confirm-profile-frozen-before-holdout-capture",
                            "--output",
                            str(output_path),
                        ]
                    ),
                )
            validated = StringIO()
            with redirect_stdout(validated):
                self.assertEqual(
                    0,
                    main(
                        [
                            "validate-household-pose-fusion-temporal-holdout",
                            str(output_path),
                        ]
                    ),
                )

            combined_output = created.getvalue() + validated.getvalue()
            self.assertNotIn(str(root), combined_output)
            payload = json.loads(validated.getvalue())
            self.assertTrue(payload["temporal_holdout"])
            self.assertFalse(payload["profile_selected_on_holdout"])
            self.assertFalse(payload["official_accuracy_claim"])
            self.assertFalse(payload["paths_disclosed"])


if __name__ == "__main__":
    unittest.main()
