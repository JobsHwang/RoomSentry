from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from baby_monitor.frame_role_reference import (
    FrameRoleReference,
    FrameRoleReferenceError,
    diagnose_frame_role_claim,
)


class FrameRoleReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = FrameRoleReference(
            camera_id="cam_01",
            frame_sha256="a" * 64,
            frame_received_at="2026-10-03T16:49:51+08:00",
            reference_id="subject_01",
            role="adult",
            provenance="synthetic",
        )

    def diagnose(self, **changes: object) -> dict[str, object]:
        arguments = {
            "camera_id": "cam_01",
            "frame_sha256": "a" * 64,
            "reference_id": "subject_01",
            "claimed_role": "baby",
            "association_provenance": "assistant_weak",
        }
        return diagnose_frame_role_claim(self.reference, **(arguments | changes))

    def test_reference_round_trip_and_utc_normalization(self) -> None:
        self.assertEqual("2026-10-03T08:49:51+00:00", self.reference.frame_received_at)
        self.assertEqual(self.reference, FrameRoleReference.from_dict(self.reference.to_dict()))
        self.assertEqual(
            self.reference, FrameRoleReference.from_json(json.dumps(self.reference.to_dict()))
        )

    def test_reference_is_frozen(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.reference.role = "baby"

    def test_only_exact_scope_and_version_are_accepted(self) -> None:
        for changes in (
            {"version": True},
            {"version": 1.0},
            {"version": 2},
            {"scope": "whole_frame"},
            {"scope": "persistent_identity"},
        ):
            with self.subTest(changes=changes), self.assertRaises(FrameRoleReferenceError):
                replace(self.reference, **changes)

    def test_invalid_identifiers_hash_roles_and_provenance_rejected(self) -> None:
        for changes in (
            {"camera_id": "../../private"},
            {"camera_id": "cam_01\n"},
            {"frame_sha256": "A" * 64},
            {"frame_sha256": "a" * 63},
            {"frame_sha256": True},
            {"reference_id": "family_member"},
            {"reference_id": "subject_01/name"},
            {"role": "parent"},
            {"role": []},
            {"provenance": "assistant_weak"},
            {"provenance": "model_consensus"},
        ):
            with self.subTest(changes=changes), self.assertRaises(FrameRoleReferenceError):
                replace(self.reference, **changes)

    def test_naive_invalid_oversize_and_nonstring_times_rejected(self) -> None:
        for value in (
            "2026-10-03T16:49:51",
            "not_a_time",
            "2026-10-03T16:49:51+25:00",
            "x" * 65,
            True,
            None,
        ):
            with self.subTest(value=value), self.assertRaises(FrameRoleReferenceError):
                replace(self.reference, frame_received_at=value)

    def test_extra_and_missing_fields_rejected(self) -> None:
        for value in (
            self.reference.to_dict() | {"family_name": "synthetic"},
            {k: v for k, v in self.reference.to_dict().items() if k != "role"},
            [],
        ):
            with self.subTest(value=value), self.assertRaises(FrameRoleReferenceError):
                FrameRoleReference.from_dict(value)

    def test_duplicate_json_fields_rejected(self) -> None:
        value = json.dumps(self.reference.to_dict())
        duplicate = value[:-1] + ', "role": "baby"}'
        with self.assertRaises(FrameRoleReferenceError):
            FrameRoleReference.from_json(duplicate)

    def test_nonstandard_json_numbers_and_deep_or_oversize_json_rejected(self) -> None:
        for value in (
            '{"role": NaN}',
            '{"role": Infinity}',
            "[" * 2000 + "0" + "]" * 2000,
            " " * 8193,
            "\ud800",
            "not JSON",
            True,
        ):
            with (
                self.subTest(kind=type(value).__name__),
                self.assertRaises(FrameRoleReferenceError),
            ):
                FrameRoleReference.from_json(value)

    def test_adult_reference_contradicts_baby_claim_only_conditionally(self) -> None:
        diagnostic = self.diagnose()
        self.assertEqual("contradicts_reference", diagnostic["comparison"])
        self.assertEqual("adult", self.reference.role)
        self.assertIs(diagnostic["subject_association_human_verified"], False)
        self.assertIs(diagnostic["formal_accuracy_authorized"], False)

    def test_matching_age_role_is_consistent_not_deployment(self) -> None:
        diagnostic = self.diagnose(claimed_role="adult")
        self.assertEqual("consistent_with_reference", diagnostic["comparison"])
        self.assertIs(diagnostic["deployment_approved"], False)

    def test_person_and_unknown_are_abstention_not_wrong_age(self) -> None:
        for role in ("person", "unknown"):
            with self.subTest(role=role):
                self.assertEqual(
                    "age_role_abstention", self.diagnose(claimed_role=role)["comparison"]
                )

    def test_missing_reference_or_default_association_remains_unknown(self) -> None:
        for arguments in (
            {"reference_id": None},
            {"association_provenance": "unresolved"},
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(
                    "unresolved_subject_association", self.diagnose(**arguments)["comparison"]
                )
        value = diagnose_frame_role_claim(
            self.reference,
            camera_id="cam_01",
            frame_sha256="a" * 64,
            reference_id="subject_01",
            claimed_role="baby",
        )
        self.assertEqual("unresolved_subject_association", value["comparison"])

    def test_unknown_reference_never_establishes_age(self) -> None:
        value = diagnose_frame_role_claim(
            replace(self.reference, role="unknown"),
            camera_id="cam_01",
            frame_sha256="a" * 64,
            reference_id="subject_01",
            claimed_role="adult",
            association_provenance="assistant_weak",
        )
        self.assertEqual("unknown_reference_role", value["comparison"])

    def test_other_frame_camera_and_person_rejected(self) -> None:
        for changes in (
            {"camera_id": "cam_02"},
            {"frame_sha256": "b" * 64},
            {"reference_id": "subject_02"},
        ):
            with self.subTest(changes=changes), self.assertRaises(FrameRoleReferenceError):
                self.diagnose(**changes)

    def test_invalid_claims_and_promoted_associations_rejected(self) -> None:
        for changes in (
            {"claimed_role": "parent"},
            {"claimed_role": True},
            {"claimed_role": []},
            {"association_provenance": "human_verified"},
            {"association_provenance": "face_match"},
            {"reference_id": "family_member"},
        ):
            with self.subTest(changes=changes), self.assertRaises(FrameRoleReferenceError):
                self.diagnose(**changes)

    def test_human_provenance_is_declared_not_authenticated_association(self) -> None:
        value = diagnose_frame_role_claim(
            replace(self.reference, provenance="human_user_confirmation"),
            camera_id="cam_01",
            frame_sha256="a" * 64,
            reference_id="subject_01",
            claimed_role="person",
            association_provenance="assistant_weak",
        )
        self.assertIs(value["human_role_confirmation_declared"], True)
        self.assertIs(value["subject_association_human_verified"], False)
        self.assertIs(value["formal_accuracy_authorized"], False)

    def test_diagnostics_do_not_establish_baby_absence_location_or_state(self) -> None:
        for role in ("baby", "adult", "person", "unknown"):
            value = self.diagnose(claimed_role=role)
            with self.subTest(role=role):
                self.assertIs(value["offline_diagnostic_only"], True)
                self.assertIs(value["whole_frame_occupancy_established"], False)
                self.assertIs(value["persistent_identity_authorized"], False)
                self.assertIs(value["live_state_established"], False)
                self.assertEqual("unknown", value["baby_location"])
                self.assertEqual("unknown", value["baby_state"])

    def test_errors_do_not_echo_untrusted_values(self) -> None:
        private = "private_name_or_url"
        for operation in (
            lambda: replace(self.reference, camera_id=private + "/"),
            lambda: self.diagnose(claimed_role=private),
            lambda: FrameRoleReference.from_json(private),
        ):
            with self.assertRaises(FrameRoleReferenceError) as caught:
                operation()
            self.assertNotIn(private, str(caught.exception))

    def test_reference_contract_does_not_accept_dict_instead_of_validated_record(self) -> None:
        with self.assertRaises(FrameRoleReferenceError):
            diagnose_frame_role_claim(
                self.reference.to_dict(),
                camera_id="cam_01",
                frame_sha256="a" * 64,
                reference_id="subject_01",
                claimed_role="baby",
            )
