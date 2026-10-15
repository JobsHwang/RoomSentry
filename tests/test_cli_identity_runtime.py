from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from baby_monitor.cli import build_parser, main
from baby_monitor.identity_domain import DeclaredKind
from baby_monitor.identity_enrollment import IdentityEnrollmentResult
from baby_monitor.identity_runtime import (
    IdentityRuntimeError,
    _build_command_enrollment_extractor,
    _require_runner_executable,
    _reviewed_model_reference,
    _systemd_isolated_runner_command,
    delete_local_identity_profile,
    enroll_local_identity,
    list_local_identity_profiles,
)
from baby_monitor.identity_service import (
    IdentityAuthorizationError,
    IdentitySettings,
)
from baby_monitor.identity_store import (
    IdentityDeletionSummary,
    IdentityProfileSummary,
)

_PROFILE_ID = "profile_" + "a" * 32


def _create_existing_store_markers(root: Path) -> None:
    for filename in ("identity.key", "identity.sqlite3"):
        path = root / filename
        path.touch(mode=0o600)
        path.chmod(0o600)


def _enrollment_result() -> IdentityEnrollmentResult:
    return IdentityEnrollmentResult(
        profile_id=_PROFILE_ID,
        template_ids=tuple(f"template_{index:032x}" for index in range(1, 4)),
        declared_kind=DeclaredKind.BABY,
        model_fingerprint="f" * 64,
        embedding_dimension=128,
        source_image_count=3,
    )


def _runtime_options(settings: IdentitySettings) -> dict[str, object]:
    return {
        "settings": settings,
        "one_shot_confirmation": True,
        "manifest_path": "/private/enrollment/manifest.json",
        "manifest_root": "/private/enrollment",
        "store_root": "/private/identity-store",
        "staging_parent": "/private/face-staging",
        "runner_executable": "/private/runtime/python",
        "detector_model_path": "/private/models/yunet.onnx",
        "detector_manifest_path": "/private/models/yunet.json",
        "embedding_model_path": "/private/models/sface.onnx",
        "embedding_manifest_path": "/private/models/sface.json",
    }


class IdentityRuntimeTests(unittest.TestCase):
    def test_enrollment_authorization_precedes_every_runtime_factory(self) -> None:
        extractor_factory = Mock()
        store_factory_builder = Mock()
        coordinator = Mock()
        options = _runtime_options(
            IdentitySettings(
                enabled=False,
                biometric_enrollment_consent=True,
            )
        )
        with self.assertRaisesRegex(
            IdentityAuthorizationError,
            "identity capability is disabled",
        ):
            enroll_local_identity(
                **options,
                _extractor_factory=extractor_factory,
                _store_factory_builder=store_factory_builder,
                _coordinator=coordinator,
            )
        extractor_factory.assert_not_called()
        store_factory_builder.assert_not_called()
        coordinator.assert_not_called()

    def test_enrollment_returns_path_vector_and_hash_free_summary(self) -> None:
        events: list[str] = []
        extractor = object()
        unopened_store = Mock()

        def build_extractor(**_kwargs: object) -> object:
            events.append("extractor")
            return extractor

        def build_store(_root: str | Path):
            events.append("store_factory")

            def open_store() -> object:
                events.append("store_opened")
                return unopened_store

            return open_store

        def coordinate(**kwargs: object) -> IdentityEnrollmentResult:
            events.append("coordinator")
            self.assertIs(extractor, kwargs["extractor"])
            self.assertTrue(callable(kwargs["store_factory"]))
            return _enrollment_result()

        payload = enroll_local_identity(
            **_runtime_options(
                IdentitySettings(
                    enabled=True,
                    biometric_enrollment_consent=True,
                )
            ),
            _extractor_factory=build_extractor,
            _store_factory_builder=build_store,
            _coordinator=coordinate,
        )

        self.assertEqual(["extractor", "store_factory", "coordinator"], events)
        self.assertEqual(_PROFILE_ID, payload["profile_id"])
        self.assertEqual(3, payload["template_count"])
        self.assertEqual("os_denied", payload["runner_network_status"])
        self.assertTrue(payload["runner_os_network_isolation_enforced"])
        self.assertEqual(
            "systemd_transient_user_unit",
            payload["runner_network_isolation_boundary"],
        )
        self.assertFalse(payload["coordinator_network_calls"])
        self.assertFalse(payload["embeddings_disclosed"])
        self.assertFalse(payload["model_hashes_disclosed"])
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("/private/", serialized)
        self.assertNotIn("f" * 64, serialized)
        self.assertNotIn("template_00000000000000000000000000000001", serialized)
        self.assertNotIn("embedding_dimension", serialized)

    def test_systemd_isolated_runner_argv_is_fixed_without_fallback(self) -> None:
        expected_command = _systemd_isolated_runner_command(
            "/opt/face-runtime/bin/python",
            _executable_admission=lambda path: str(path),
            _bus_admission=lambda: (
                Path("/run/user/1000"),
                Path("/run/user/1000/bus"),
            ),
            _runner_admission=lambda value: value,
        )
        self.assertEqual(
            (
                "/usr/bin/env",
                "XDG_RUNTIME_DIR=/run/user/1000",
                "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
                "/usr/bin/systemd-run",
                "--user",
                "--pipe",
                "--wait",
                "--collect",
                "--quiet",
                "--property=IPAddressDeny=any",
                "--property=RestrictAddressFamilies=AF_UNIX",
                "--property=MemoryMax=1610612736",
                "--property=NoNewPrivileges=yes",
                "--property=PrivateDevices=yes",
                "/opt/face-runtime/bin/python",
                "-m",
                "baby_monitor.local_runners.opencv_face",
            ),
            expected_command,
        )
        with self.assertRaises(IdentityRuntimeError):
            _systemd_isolated_runner_command(
                "/opt/face-runtime/bin/python",
                _executable_admission=lambda path: str(path),
                _bus_admission=Mock(side_effect=IdentityRuntimeError("enrollment_failed")),
                _runner_admission=lambda value: value,
            )

        detector = object()
        embedding = object()
        enrollment = object()
        adapter = object()
        with (
            patch(
                "baby_monitor.identity_runtime._systemd_isolated_runner_command",
                return_value=expected_command,
            ) as isolate,
            patch(
                "baby_monitor.identity_runtime._reviewed_model_reference",
                side_effect=(detector, embedding),
            ) as references,
            patch(
                "baby_monitor.identity_runtime.FaceEnrollmentCommandConfig",
                return_value=enrollment,
            ),
            patch(
                "baby_monitor.identity_runtime.CommandFaceEmbeddingAdapter",
                return_value=adapter,
            ) as adapter_type,
        ):
            actual = _build_command_enrollment_extractor(
                runner_executable="/opt/face-runtime/bin/python",
                staging_parent="/private/staging",
                detector_model_path="/private/models/yunet.onnx",
                detector_manifest_path="/private/models/yunet.json",
                embedding_model_path="/private/models/sface.onnx",
                embedding_manifest_path="/private/models/sface.json",
                timeout_seconds=120.0,
                detector_score_threshold=0.8,
                detector_nms_threshold=0.3,
                detector_top_k=100,
                min_face_size_pixels=40,
                min_eye_distance_pixels=12.0,
            )
        self.assertIs(adapter, actual)
        isolate.assert_called_once_with("/opt/face-runtime/bin/python")
        self.assertEqual(2, references.call_count)
        self.assertEqual(expected_command, adapter_type.call_args.kwargs["command"])

    def test_runner_provenance_accepts_venv_symlink_and_rejects_user_binary(self) -> None:
        # Do not derive the fixture from setup-python or another CI-managed
        # runtime. Such caches may intentionally be group-writable and must
        # remain rejected by the production admission boundary.
        system_executable = Path("/usr/bin/python3")
        if not system_executable.exists():
            self.skipTest("the target Ubuntu system Python is unavailable")
        admitted_system_executable = str(system_executable)
        self.assertEqual(
            admitted_system_executable,
            _require_runner_executable(admitted_system_executable),
        )

        with tempfile.TemporaryDirectory() as directory:
            replaceable_runner = Path(directory) / "python"
            replaceable_runner.symlink_to(system_executable)
            with self.assertRaises(IdentityRuntimeError):
                _require_runner_executable(str(replaceable_runner))

        with tempfile.TemporaryDirectory(dir=Path.home()) as directory:
            runner = Path(directory) / "python"
            runner.symlink_to(system_executable)
            self.assertEqual(str(runner), _require_runner_executable(str(runner)))

            executable = Path(directory) / "python"
            runner.unlink()
            executable.write_bytes(b"not-a-reviewed-runtime")
            executable.chmod(0o755)
            with self.assertRaises(IdentityRuntimeError):
                _require_runner_executable(str(executable))

    def test_model_reference_requires_review_before_file_verification(self) -> None:
        order = Mock()
        manifest = SimpleNamespace(
            role="detector",
            model_id="opencv.yunet.2023mar",
            upstream_revision="1" * 40,
        )
        manifest.require_reviewed = Mock(side_effect=lambda: order("reviewed"))
        manifest.verify_model = Mock(
            side_effect=lambda _path: (
                order("verified"),
                SimpleNamespace(sha256="a" * 64),
            )[1]
        )
        with (
            patch(
                "baby_monitor.identity_runtime.FaceModelManifest.read",
                return_value=manifest,
            ),
            patch(
                "baby_monitor.identity_runtime.FaceModelReference",
                return_value=object(),
            ),
        ):
            _reviewed_model_reference(
                role="detector",
                model_path="/private/models/yunet.onnx",
                manifest_path="/private/models/yunet.json",
            )
        self.assertEqual([call("reviewed"), call("verified")], order.call_args_list)

    def test_profiles_list_only_public_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "identity"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            _create_existing_store_markers(root)
            store = Mock()
            store.list_profiles.return_value = (
                IdentityProfileSummary(
                    profile_id=_PROFILE_ID,
                    declared_kind=DeclaredKind.BABY,
                    template_count=3,
                ),
            )
            payload = list_local_identity_profiles(
                store_root=root,
                _store_factory=Mock(return_value=store),
            )
        self.assertEqual(1, payload["profile_count"])
        self.assertEqual(3, payload["template_count"])
        self.assertEqual(_PROFILE_ID, payload["profiles"][0]["profile_id"])
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn("vector", serialized)
        self.assertNotIn("model_fingerprint", serialized)
        self.assertFalse(payload["embeddings_loaded"])

    def test_delete_authorizes_exact_profile_before_store_open(self) -> None:
        store_factory = Mock()
        with self.assertRaisesRegex(
            IdentityAuthorizationError,
            "one-shot identity profile deletion confirmation",
        ):
            delete_local_identity_profile(
                profile_id=_PROFILE_ID,
                one_shot_confirmation=False,
                store_root="/missing/private-store",
                _store_factory=store_factory,
            )
        store_factory.assert_not_called()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "identity"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            _create_existing_store_markers(root)
            store = Mock()
            store.delete_profile.return_value = IdentityDeletionSummary(
                deleted_profiles=1,
                deleted_templates=3,
                remaining_profiles=0,
                remaining_templates=0,
            )
            payload = delete_local_identity_profile(
                profile_id=_PROFILE_ID,
                one_shot_confirmation=True,
                store_root=root,
                _store_factory=Mock(return_value=store),
            )
        store.delete_profile.assert_called_once_with(_PROFILE_ID)
        self.assertEqual(_PROFILE_ID, payload["profile_id"])
        self.assertEqual(1, payload["deleted_profiles"])
        self.assertEqual(3, payload["deleted_templates"])

    def test_post_authorization_failures_are_fixed_and_path_free(self) -> None:
        secret_path = "/private/family/secret-child-photo.jpg"
        options = _runtime_options(
            IdentitySettings(
                enabled=True,
                biometric_enrollment_consent=True,
            )
        )
        with self.assertRaises(IdentityRuntimeError) as caught:
            enroll_local_identity(
                **options,
                _extractor_factory=Mock(side_effect=RuntimeError(secret_path)),
            )
        self.assertEqual("local identity enrollment failed safely", str(caught.exception))
        self.assertNotIn(secret_path, str(caught.exception))


class IdentityRuntimeCLITests(unittest.TestCase):
    @staticmethod
    def _config(directory: str, *, enabled: bool, consent: bool) -> Path:
        path = Path(directory) / "identity.toml"
        path.write_text(
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
        return path

    @staticmethod
    def _enroll_argv(config: Path) -> list[str]:
        return [
            "identity-enroll",
            "--config",
            str(config),
            "--manifest",
            "/private/enrollment/manifest.json",
            "--manifest-root",
            "/private/enrollment",
            "--store-root",
            "/private/identity",
            "--staging-parent",
            "/private/staging",
            "--runner-executable",
            "/opt/face-runtime/bin/python",
            "--detector-model",
            "/private/models/yunet.onnx",
            "--detector-manifest",
            "/private/models/yunet.json",
            "--embedding-model",
            "/private/models/sface.onnx",
            "--embedding-manifest",
            "/private/models/sface.json",
        ]

    @staticmethod
    def _run(argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_gate_rejects_before_runtime_is_called(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, enabled=False, consent=True)
            with patch("baby_monitor.cli_commands.identity.enroll_local_identity") as runtime:
                code, stdout, stderr = self._run(
                    [*self._enroll_argv(config), "--confirm-biometric-enrollment"]
                )
        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("identity capability is disabled", stderr)
        runtime.assert_not_called()

    def test_cli_enrollment_wires_only_fixed_runner_executable(self) -> None:
        public = {
            "status": "enrolled",
            "profile_id": _PROFILE_ID,
            "template_count": 3,
            "runner_network_status": "os_denied",
            "runner_os_network_isolation_enforced": True,
            "runner_network_isolation_boundary": "systemd_transient_user_unit",
            "coordinator_network_calls": False,
            "paths_disclosed": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, enabled=True, consent=True)
            with patch(
                "baby_monitor.cli_commands.identity.enroll_local_identity",
                return_value=public,
            ) as runtime:
                code, stdout, stderr = self._run(
                    [*self._enroll_argv(config), "--confirm-biometric-enrollment"]
                )
        self.assertEqual(0, code)
        self.assertEqual("", stderr)
        self.assertEqual(public, json.loads(stdout))
        self.assertEqual(
            "/opt/face-runtime/bin/python",
            runtime.call_args.kwargs["runner_executable"],
        )
        self.assertNotIn("runner_args", runtime.call_args.kwargs)

    def test_parser_rejects_relative_runner_and_arbitrary_runner_args(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(directory, enabled=True, consent=True)
            argv = self._enroll_argv(config)
            executable_index = argv.index("/opt/face-runtime/bin/python")
            relative = [*argv]
            relative[executable_index] = "python"
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(relative)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args([*argv, "--runner-arg=-c"])

    def test_cli_delete_gate_precedes_runtime(self) -> None:
        argv = [
            "identity-delete-profile",
            "--store-root",
            "/private/identity",
            "--profile-id",
            _PROFILE_ID,
        ]
        with patch("baby_monitor.cli_commands.identity.delete_local_identity_profile") as runtime:
            code, stdout, stderr = self._run(argv)
        self.assertEqual(2, code)
        self.assertEqual("", stdout)
        self.assertIn("one-shot identity profile deletion confirmation", stderr)
        runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
