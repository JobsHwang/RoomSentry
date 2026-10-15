from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from baby_monitor.cli import main
from baby_monitor.domain import SignalEvent
from baby_monitor.signal_dispatch import (
    BufferedSignalCommand,
    HASignalRoute,
    HASignalRouteMap,
    SignalDispatchAttempt,
    SignalDispatchError,
    _classify_child_failure,
    _default_runner,
    dispatch_one_signal,
    run_signal_dispatch_worker,
)
from baby_monitor.signal_outbox import SignalOutbox


def _signal(camera_id: str = "camera_01") -> SignalEvent:
    return SignalEvent.from_dict(
        {
            "version": 1,
            "signal_id": "ha_dispatch_fixture_01",
            "camera_id": camera_id,
            "occurred_at": "2026-08-18T08:00:00+00:00",
            "signal_type": "motion",
            "source_adapter": "home_assistant",
            "confidence": 0.5,
            "attributes": {"channel": "device"},
        }
    )


def _private_file(path: Path, text: str = "{}") -> Path:
    path.write_text(text, encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)
    return path


class SignalDispatchTests(unittest.TestCase):
    def test_child_stderr_is_only_mapped_to_fixed_failure_codes(self) -> None:
        cases = (
            (b"error: signal camera_id does not match the buffer manifest", 10),
            (b"error: Permission denied: /private/path", 9),
            (b"usage: baby-monitor [-h]", 12),
            (b"/usr/bin/python: No module named baby_monitor", 17),
            (b"Traceback (most recent call last):\nTypeError: private", 13),
            (b"Traceback (most recent call last):\nsqlite3.OperationalError: private", 14),
            (b"Traceback (most recent call last):\nRuntimeError: private", 16),
            (b"private unknown detail", 2),
        )
        for detail, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(expected, _classify_child_failure(2, detail))
        self.assertEqual(3, _classify_child_failure(3, b"private ignored"))

    def test_default_runner_uses_private_bounded_stderr_instead_of_logging(self) -> None:
        def fake_run(*_args: object, **kwargs: object) -> object:
            stderr = kwargs["stderr"]
            stderr.write(b"usage: baby-monitor [-h]\nprivate detail")
            return subprocess.CompletedProcess(args=("fixture",), returncode=2)

        with mock.patch("baby_monitor.signal_dispatch.subprocess.run", side_effect=fake_run):
            result = _default_runner(("/private/fixture",), 20.0, {"PATH": "/usr/bin:/bin"})

        self.assertEqual(12, result)

    def _fixture(
        self,
        root: Path,
        *,
        camera_id: str = "camera_01",
    ) -> tuple[SignalOutbox, HASignalRouteMap, BufferedSignalCommand, Path]:
        staging = root / "staging"
        staging.mkdir(mode=0o700)
        config = _private_file(root / "config.toml", "[fixture]\n")
        manifest = _private_file(root / "manifest.json")
        outbox = SignalOutbox(root / "signals.sqlite3")
        routes = HASignalRouteMap(
            routes=(HASignalRoute(camera_id=camera_id, manifest_path=manifest),)
        )
        command = BufferedSignalCommand(
            python_executable=Path(os.sys.executable).resolve(),
            config_path=config,
            staging_root=staging,
            timeout_seconds=20.0,
        )
        return outbox, routes, command, staging

    def test_success_stages_private_signal_then_acknowledges_and_scrubs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outbox, routes, command, staging = self._fixture(root)
            outbox.enqueue(_signal())
            observed: dict[str, object] = {}

            def runner(argv: object, timeout: float, env: dict[str, str]) -> int:
                arguments = tuple(argv)  # type: ignore[arg-type]
                signal_path = Path(arguments[arguments.index("--signal-json") + 1])
                observed["payload"] = json.loads(signal_path.read_text(encoding="utf-8"))
                observed["mode"] = stat.S_IMODE(signal_path.stat().st_mode)
                observed["timeout"] = timeout
                observed["env"] = env
                self.assertEqual(str(command.python_executable), arguments[0])
                self.assertEqual("enqueue-signal-event", arguments[3])
                self.assertNotIn("LD_LIBRARY_PATH", env)
                self.assertEqual("/usr/bin:/bin", env["PATH"])
                self.assertTrue(Path(env["PYTHONPATH"]).is_absolute())
                self.assertTrue((Path(env["PYTHONPATH"]) / "baby_monitor").is_dir())
                self.assertEqual(
                    {
                        "HOME",
                        "LANG",
                        "LC_ALL",
                        "PATH",
                        "PYTHONPATH",
                        "PYTHONUNBUFFERED",
                        "PYTHONDONTWRITEBYTECODE",
                    },
                    set(env),
                )
                return 0

            result = dispatch_one_signal(
                outbox=outbox,
                routes=routes,
                command=command,
                lease_seconds=30,
                runner=runner,
            )

            self.assertEqual("completed", result.status)
            self.assertEqual(_signal().to_dict(), observed["payload"])
            if os.name == "posix":
                self.assertEqual(0o600, observed["mode"])
            self.assertEqual([], list(staging.glob("signal-*.json")))
            snapshot = outbox.snapshot_read_only()
            self.assertEqual(1, snapshot.counts.completed)
            self.assertEqual(0, snapshot.counts.active)
            public = json.dumps(result.to_public_dict())
            self.assertNotIn(str(root), public)
            self.assertFalse(result.to_public_dict()["model_loaded"])

    def test_queue_full_retries_then_fixed_failure_stops_at_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outbox, routes, command, staging = self._fixture(root)
            outbox.enqueue(_signal())

            first = dispatch_one_signal(
                outbox=outbox,
                routes=routes,
                command=command,
                lease_seconds=30,
                max_attempts=2,
                retry_delay_seconds=0,
                runner=lambda _argv, _timeout, _env: 3,
            )
            self.assertEqual("retry_scheduled", first.status)
            self.assertEqual("analysis_queue_full", first.error_code)

            second = dispatch_one_signal(
                outbox=outbox,
                routes=routes,
                command=command,
                lease_seconds=30,
                max_attempts=2,
                retry_delay_seconds=0,
                runner=lambda _argv, _timeout, _env: 2,
            )
            self.assertEqual("failed", second.status)
            self.assertEqual("buffer_evidence_failed", second.error_code)
            self.assertEqual(1, outbox.snapshot_read_only().counts.failed)
            self.assertEqual([], list(staging.glob("signal-*.json")))

    def test_unavailable_buffer_window_has_a_distinct_fixed_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outbox, routes, command, _staging = self._fixture(root)
            outbox.enqueue(_signal())

            result = dispatch_one_signal(
                outbox=outbox,
                routes=routes,
                command=command,
                lease_seconds=30,
                max_attempts=1,
                retry_delay_seconds=0,
                runner=lambda _argv, _timeout, _env: 4,
            )

            self.assertEqual("failed", result.status)
            self.assertEqual("buffer_window_unavailable", result.error_code)

    def test_media_command_and_processing_failures_have_fixed_errors(self) -> None:
        for return_code, expected in (
            (5, "media_command_unavailable"),
            (6, "buffer_media_processing_failed"),
            (7, "buffer_config_invalid"),
            (8, "signal_staging_invalid"),
            (9, "buffer_storage_unavailable"),
            (10, "buffer_request_invalid"),
            (11, "buffer_runtime_failed"),
            (12, "buffer_command_invalid"),
            (13, "buffer_contract_type_error"),
            (14, "buffer_database_failed"),
            (15, "buffer_command_timeout"),
            (16, "buffer_child_exception"),
            (17, "buffer_module_unavailable"),
        ):
            with self.subTest(return_code=return_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                outbox, routes, command, _staging = self._fixture(root)
                outbox.enqueue(_signal())

                result = dispatch_one_signal(
                    outbox=outbox,
                    routes=routes,
                    command=command,
                    lease_seconds=30,
                    max_attempts=1,
                    retry_delay_seconds=0,
                    runner=lambda _argv, _timeout, _env, code=return_code: code,
                )

                self.assertEqual("failed", result.status)
                self.assertEqual(expected, result.error_code)

    def test_missing_route_is_recorded_without_starting_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outbox, routes, command, _staging = self._fixture(
                root,
                camera_id="other_camera",
            )
            outbox.enqueue(_signal())
            called = False

            def runner(_argv: object, _timeout: float, _env: dict[str, str]) -> int:
                nonlocal called
                called = True
                return 0

            result = dispatch_one_signal(
                outbox=outbox,
                routes=routes,
                command=command,
                lease_seconds=30,
                max_attempts=1,
                retry_delay_seconds=0,
                runner=runner,
            )

            self.assertFalse(called)
            self.assertEqual("failed", result.status)
            self.assertEqual("route_missing", result.error_code)

    def test_lease_must_outlive_timeout_before_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outbox, routes, command, _staging = self._fixture(root)
            outbox.enqueue(_signal())
            with self.assertRaisesRegex(SignalDispatchError, "outlive"):
                dispatch_one_signal(
                    outbox=outbox,
                    routes=routes,
                    command=command,
                    lease_seconds=29,
                )
            self.assertEqual(1, outbox.snapshot_read_only().counts.pending)

    def test_route_map_is_private_strict_and_manifest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _private_file(root / "manifest.json")
            route_map = _private_file(
                root / "routes.json",
                json.dumps(
                    {
                        "version": 1,
                        "routes": [
                            {
                                "camera_id": "camera_01",
                                "manifest_path": str(manifest),
                            }
                        ],
                    }
                ),
            )
            loaded = HASignalRouteMap.read(route_map)
            self.assertEqual(manifest, loaded.routes[0].manifest_path)

            route_map.write_text(
                '{"version":1,"version":1,"routes":[]}',
                encoding="utf-8",
            )
            if os.name == "posix":
                route_map.chmod(0o600)
            with self.assertRaisesRegex(SignalDispatchError, "invalid"):
                HASignalRouteMap.read(route_map)

    def test_v2_route_binds_signal_camera_to_media_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir(mode=0o700)
            config = _private_file(root / "config.toml", "[fixture]\n")
            manifest = _private_file(root / "manifest.json")
            route_map = _private_file(
                root / "routes.json",
                json.dumps(
                    {
                        "version": 2,
                        "routes": [
                            {
                                "signal_camera_id": "ha_camera_a",
                                "camera_id": "camera_01",
                                "manifest_path": str(manifest),
                            }
                        ],
                    }
                ),
            )
            outbox = SignalOutbox(root / "signals.sqlite3")
            outbox.enqueue(_signal(camera_id="ha_camera_a"))
            routes = HASignalRouteMap.read(route_map)
            command = BufferedSignalCommand(
                python_executable=Path(os.sys.executable).resolve(),
                config_path=config,
                staging_root=staging,
                timeout_seconds=20.0,
            )
            observed: dict[str, object] = {}

            def runner(argv: object, _timeout: float, _env: dict[str, str]) -> int:
                arguments = tuple(argv)  # type: ignore[arg-type]
                signal_path = Path(arguments[arguments.index("--signal-json") + 1])
                observed.update(json.loads(signal_path.read_text(encoding="utf-8")))
                return 0

            result = dispatch_one_signal(
                outbox=outbox,
                routes=routes,
                command=command,
                lease_seconds=30,
                runner=runner,
            )

            self.assertEqual("completed", result.status)
            self.assertEqual("camera_01", observed["camera_id"])
            self.assertEqual("camera_01", routes.routes[0].camera_id)
            self.assertEqual("ha_camera_a", routes.routes[0].signal_camera_id)
            self.assertEqual(1, outbox.snapshot_read_only().counts.completed)
            self.assertEqual([], list(staging.glob("signal-*.json")))

            with self.assertRaisesRegex(SignalDispatchError, "version fields"):
                HASignalRouteMap(
                    routes=(
                        HASignalRoute(
                            camera_id="camera_01",
                            manifest_path=manifest,
                            signal_camera_id="ha_camera_a",
                        ),
                    ),
                    version=1,
                )

    @unittest.skipUnless(os.name == "posix", "POSIX privacy modes are required")
    def test_route_map_and_staging_reject_weak_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _private_file(root / "manifest.json")
            route_map = _private_file(
                root / "routes.json",
                json.dumps(
                    {
                        "version": 1,
                        "routes": [
                            {
                                "camera_id": "camera_01",
                                "manifest_path": str(manifest),
                            }
                        ],
                    }
                ),
            )
            route_map.chmod(0o644)
            with self.assertRaisesRegex(SignalDispatchError, "permissions"):
                HASignalRouteMap.read(route_map)

            staging = root / "staging"
            staging.mkdir(mode=0o755)
            config = _private_file(root / "config.toml", "[fixture]\n")
            with self.assertRaisesRegex(SignalDispatchError, "permissions"):
                BufferedSignalCommand(
                    python_executable=Path(os.sys.executable).resolve(),
                    config_path=config,
                    staging_root=staging,
                )

    @unittest.skipUnless(os.name == "posix", "POSIX private roots are required")
    def test_route_map_and_staging_must_remain_outside_git(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            manifest = _private_file(root / "manifest.json")
            route_map = _private_file(
                root / "routes.json",
                json.dumps(
                    {
                        "version": 1,
                        "routes": [
                            {
                                "camera_id": "camera_01",
                                "manifest_path": str(manifest),
                            }
                        ],
                    }
                ),
            )
            with self.assertRaisesRegex(SignalDispatchError, "outside"):
                HASignalRouteMap.read(route_map)

            staging = root / "staging"
            staging.mkdir(mode=0o700)
            config = _private_file(root / "config.toml", "[fixture]\n")
            with self.assertRaisesRegex(SignalDispatchError, "outside"):
                BufferedSignalCommand(
                    python_executable=Path(os.sys.executable).resolve(),
                    config_path=config,
                    staging_root=staging,
                )

    def test_sensitive_command_and_route_reprs_hide_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _outbox, routes, command, _staging = self._fixture(root)
            rendered = repr((routes.routes[0], command))

            self.assertNotIn(str(root), rendered)

    def test_cli_wires_private_dispatch_and_returns_path_free_result(self) -> None:
        attempt = SignalDispatchAttempt(
            status="completed",
            outbox_id="outbox_fixture_01",
            error_code=None,
        )
        output = StringIO()
        with (
            mock.patch(
                "baby_monitor.cli_commands.home_assistant.HASignalRouteMap.read"
            ) as read_routes,
            mock.patch(
                "baby_monitor.cli_commands.home_assistant.BufferedSignalCommand"
            ) as build_command,
            mock.patch("baby_monitor.cli_commands.home_assistant.SignalOutbox") as build_outbox,
            mock.patch(
                "baby_monitor.cli_commands.home_assistant.dispatch_one_signal",
                return_value=attempt,
            ) as dispatch,
            redirect_stdout(output),
        ):
            exit_code = main(
                [
                    "ha-signal-dispatch-once",
                    "--outbox-database",
                    "/private/outbox.sqlite3",
                    "--route-map",
                    "/private/routes.json",
                    "--config",
                    "/private/config.toml",
                    "--staging-root",
                    "/private/staging",
                    "--analysis-database",
                    "/private/analysis.sqlite3",
                    "--output-root",
                    "/private/output",
                    "--lease-seconds",
                    "60",
                    "--command-timeout-seconds",
                    "20",
                ]
            )

        self.assertEqual(0, exit_code)
        read_routes.assert_called_once_with("/private/routes.json")
        build_outbox.assert_called_once_with("/private/outbox.sqlite3")
        build_command.assert_called_once()
        dispatch.assert_called_once()
        public = json.loads(output.getvalue())
        self.assertEqual("completed", public["status"])
        self.assertFalse(public["remote_used"])
        self.assertFalse(public["model_loaded"])
        self.assertNotIn("/private", output.getvalue())

    def test_cli_returns_nonzero_for_terminal_dispatch_failure(self) -> None:
        attempt = SignalDispatchAttempt(
            status="failed",
            outbox_id="outbox_fixture_01",
            error_code="route_missing",
        )
        with (
            mock.patch("baby_monitor.cli_commands.home_assistant.HASignalRouteMap.read"),
            mock.patch("baby_monitor.cli_commands.home_assistant.BufferedSignalCommand"),
            mock.patch("baby_monitor.cli_commands.home_assistant.SignalOutbox"),
            mock.patch(
                "baby_monitor.cli_commands.home_assistant.dispatch_one_signal", return_value=attempt
            ),
            redirect_stdout(StringIO()),
        ):
            exit_code = main(
                [
                    "ha-signal-dispatch-once",
                    "--outbox-database",
                    "/private/outbox.sqlite3",
                    "--route-map",
                    "/private/routes.json",
                    "--staging-root",
                    "/private/staging",
                ]
            )

        self.assertEqual(2, exit_code)

    def test_worker_processes_a_bounded_batch_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outbox, routes, command, _staging = self._fixture(root)
            outbox.enqueue(_signal())
            outbox.enqueue(
                SignalEvent.from_dict(
                    {
                        **_signal().to_dict(),
                        "signal_id": "ha_dispatch_fixture_02",
                        "occurred_at": "2026-08-18T08:00:01+00:00",
                    }
                )
            )

            summary = run_signal_dispatch_worker(
                outbox=outbox,
                routes=routes,
                command=command,
                lease_seconds=30,
                max_jobs=2,
                max_runtime_seconds=60,
                runner=lambda _argv, _timeout, _env: 0,
                sleeper=lambda _seconds: self.fail("worker unexpectedly slept"),
            )

            self.assertEqual(2, summary.processed)
            self.assertEqual(2, summary.completed)
            self.assertEqual("max_jobs", summary.stop_reason)
            self.assertFalse(summary.to_public_dict()["model_loaded"])
            self.assertEqual(2, outbox.snapshot_read_only().counts.completed)

    def test_worker_validates_bounds_before_touching_outbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outbox, routes, command, _staging = self._fixture(root)
            outbox.enqueue(_signal())

            with self.assertRaisesRegex(SignalDispatchError, "poll interval"):
                run_signal_dispatch_worker(
                    outbox=outbox,
                    routes=routes,
                    command=command,
                    lease_seconds=30,
                    max_runtime_seconds=60,
                    poll_seconds=0,
                )

            self.assertEqual(1, outbox.snapshot_read_only().counts.pending)


if __name__ == "__main__":
    unittest.main()
