from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.config import ConfigurationError, load_settings


class IdentityConfigTests(unittest.TestCase):
    def test_default_config_keeps_every_identity_capability_disabled(self) -> None:
        settings = load_settings("config/default.toml")
        self.assertFalse(settings.identity.enabled)
        self.assertFalse(settings.identity.biometric_enrollment_consent)
        self.assertFalse(settings.identity.event_matching_enabled)
        self.assertFalse(settings.identity.external_identity_disclosure_enabled)
        self.assertFalse(settings.identity.enrollment_persistent_gate_satisfied)
        self.assertFalse(settings.identity.event_matching_allowed)
        self.assertFalse(settings.identity.external_identity_disclosure_allowed)

    def test_identity_parser_accepts_only_explicit_boolean_opt_ins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "identity.toml"
            config.write_text(
                "\n".join(
                    (
                        "[identity]",
                        "enabled = true",
                        "biometric_enrollment_consent = true",
                        "event_matching_enabled = true",
                        "external_identity_disclosure_enabled = true",
                    )
                ),
                encoding="utf-8",
            )
            settings = load_settings(config)
        self.assertTrue(settings.identity.enrollment_persistent_gate_satisfied)
        self.assertTrue(settings.identity.event_matching_allowed)
        self.assertTrue(settings.identity.external_identity_disclosure_allowed)

    def test_identity_parser_rejects_ambiguous_consent_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "identity.toml"
            config.write_text(
                '[identity]\nbiometric_enrollment_consent = "maybe"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "true or false"):
                load_settings(config)


class IdentityCLITests(unittest.TestCase):
    def _identity_config(
        self,
        directory: str,
        *,
        enabled: bool,
        consent: bool,
    ) -> Path:
        config = Path(directory) / "identity.toml"
        config.write_text(
            "\n".join(
                (
                    "[identity]",
                    f"enabled = {str(enabled).lower()}",
                    f"biometric_enrollment_consent = {str(consent).lower()}",
                    "event_matching_enabled = false",
                    "external_identity_disclosure_enabled = false",
                )
            ),
            encoding="utf-8",
        )
        return config

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_status_is_path_free_non_mutating_and_reads_no_media(self) -> None:
        code, stdout, stderr = self._run(["identity-consent-status"])
        self.assertEqual(0, code)
        self.assertEqual("", stderr)
        payload = json.loads(stdout)
        self.assertFalse(payload["identity"]["enabled"])
        self.assertFalse(payload["identity"]["biometric_enrollment_consent"])
        self.assertFalse(payload["operation_performed"])
        self.assertTrue(payload["enrollment_command_available"])
        self.assertFalse(payload["media_read"])
        self.assertFalse(payload["event_history_read"])
        self.assertFalse(payload["storage_mutated"])
        self.assertFalse(payload["paths_disclosed"])

    def test_enrollment_rejects_each_missing_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            disabled = self._identity_config(
                directory,
                enabled=False,
                consent=True,
            )
            code, stdout, stderr = self._run(
                [
                    "identity-enrollment-preflight",
                    "--config",
                    str(disabled),
                    "--confirm-biometric-enrollment",
                ]
            )
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("identity capability is disabled", stderr)

            no_consent = self._identity_config(
                directory,
                enabled=True,
                consent=False,
            )
            code, stdout, stderr = self._run(
                [
                    "identity-enrollment-preflight",
                    "--config",
                    str(no_consent),
                    "--confirm-biometric-enrollment",
                ]
            )
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("persistent biometric enrollment consent", stderr)

            allowed = self._identity_config(
                directory,
                enabled=True,
                consent=True,
            )
            code, stdout, stderr = self._run(
                ["identity-enrollment-preflight", "--config", str(allowed)]
            )
            self.assertEqual(2, code)
            self.assertEqual("", stdout)
            self.assertIn("one-shot biometric enrollment confirmation", stderr)

    def test_enrollment_preflight_authorizes_without_media_or_store_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._identity_config(
                directory,
                enabled=True,
                consent=True,
            )
            code, stdout, stderr = self._run(
                [
                    "identity-enrollment-preflight",
                    "--config",
                    str(config),
                    "--confirm-biometric-enrollment",
                ]
            )
        self.assertEqual(0, code)
        self.assertEqual("", stderr)
        payload = json.loads(stdout)
        self.assertEqual("enrollment", payload["operation"])
        self.assertTrue(payload["authorized"])
        self.assertTrue(payload["persistent_identity_enabled"])
        self.assertTrue(payload["persistent_biometric_enrollment_consent"])
        self.assertTrue(payload["preflight_only"])
        self.assertFalse(payload["media_read"])
        self.assertFalse(payload["event_history_read"])
        self.assertFalse(payload["storage_mutated"])
        self.assertFalse(payload["paths_disclosed"])

    def test_deletion_confirmation_is_independent_of_enrollment_settings(self) -> None:
        code, stdout, stderr = self._run(
            [
                "identity-deletion-preflight",
                "--profile-id",
                "profile_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ]
        )
        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("one-shot identity profile deletion confirmation", stderr)

        code, stdout, stderr = self._run(
            [
                "identity-deletion-preflight",
                "--profile-id",
                "profile_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "--confirm-identity-profile-deletion",
            ]
        )
        self.assertEqual(0, code)
        self.assertEqual("", stderr)
        payload = json.loads(stdout)
        self.assertEqual("profile_deletion", payload["operation"])
        self.assertEqual(
            "profile_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            payload["target_profile_id"],
        )
        self.assertTrue(payload["preflight_only"])
        self.assertFalse(payload["storage_mutated"])


if __name__ == "__main__":
    unittest.main()
