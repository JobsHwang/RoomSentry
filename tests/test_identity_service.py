from __future__ import annotations

import unittest

from baby_monitor.identity_service import (
    IdentityAuthorizationError,
    IdentitySettings,
    authorize_enrollment,
    authorize_profile_deletion,
)


class IdentitySettingsTests(unittest.TestCase):
    def test_defaults_fail_closed(self) -> None:
        settings = IdentitySettings()
        self.assertFalse(settings.enabled)
        self.assertFalse(settings.biometric_enrollment_consent)
        self.assertFalse(settings.enrollment_persistent_gate_satisfied)
        self.assertFalse(settings.event_matching_allowed)
        self.assertFalse(settings.external_identity_disclosure_allowed)

    def test_each_runtime_capability_requires_identity_enabled(self) -> None:
        settings = IdentitySettings(
            biometric_enrollment_consent=True,
            event_matching_enabled=True,
            external_identity_disclosure_enabled=True,
        )
        self.assertFalse(settings.enrollment_persistent_gate_satisfied)
        self.assertFalse(settings.event_matching_allowed)
        self.assertFalse(settings.external_identity_disclosure_allowed)

    def test_external_disclosure_also_requires_event_matching(self) -> None:
        settings = IdentitySettings(
            enabled=True,
            external_identity_disclosure_enabled=True,
        )
        self.assertFalse(settings.external_identity_disclosure_allowed)


class IdentityAuthorizationTests(unittest.TestCase):
    def test_enrollment_requires_both_persistent_gates_and_one_shot(self) -> None:
        cases = (
            (IdentitySettings(), True, "capability is disabled"),
            (
                IdentitySettings(enabled=True),
                True,
                "persistent biometric enrollment consent",
            ),
            (
                IdentitySettings(
                    enabled=True,
                    biometric_enrollment_consent=True,
                ),
                False,
                "one-shot biometric enrollment confirmation",
            ),
        )
        for settings, confirmation, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(
                    IdentityAuthorizationError,
                    message,
                ),
            ):
                authorize_enrollment(
                    settings,
                    one_shot_confirmation=confirmation,
                )

    def test_enrollment_receipt_is_media_free_and_non_mutating(self) -> None:
        receipt = authorize_enrollment(
            IdentitySettings(
                enabled=True,
                biometric_enrollment_consent=True,
            ),
            one_shot_confirmation=True,
        ).to_public_dict()
        self.assertTrue(receipt["authorized"])
        self.assertFalse(receipt["media_read"])
        self.assertFalse(receipt["event_history_read"])
        self.assertFalse(receipt["storage_mutated"])
        self.assertFalse(receipt["paths_disclosed"])

    def test_deletion_requires_only_target_and_independent_confirmation(self) -> None:
        with self.assertRaisesRegex(
            IdentityAuthorizationError,
            "one-shot identity profile deletion confirmation",
        ):
            authorize_profile_deletion(
                profile_id="profile_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                one_shot_confirmation=False,
            )

        receipt = authorize_profile_deletion(
            profile_id="profile_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            one_shot_confirmation=True,
        ).to_public_dict()
        self.assertEqual("profile_deletion", receipt["operation"])
        self.assertEqual(
            "profile_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            receipt["target_profile_id"],
        )
        self.assertFalse(receipt["storage_mutated"])

    def test_deletion_rejects_any_id_not_generated_by_the_store(self) -> None:
        for profile_id in ("../profile", "profile_01", "/tmp/profile", "宝宝"):
            with (
                self.subTest(profile_id=profile_id),
                self.assertRaisesRegex(
                    IdentityAuthorizationError,
                    "opaque profile_ id",
                ),
            ):
                authorize_profile_deletion(
                    profile_id=profile_id,
                    one_shot_confirmation=True,
                )
