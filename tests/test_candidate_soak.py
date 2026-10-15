from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.candidate_health import (
    CandidateHealthLifetime,
    CandidateHealthState,
    CandidateWorkerHealth,
)
from baby_monitor.candidate_soak import (
    CandidateSoakError,
    CandidateSoakLedger,
    finish_candidate_soak,
    inspect_candidate_soak,
    start_candidate_soak,
)
from baby_monitor.cli import main


class CandidateSoakTests(unittest.TestCase):
    NOW = datetime(2026, 8, 19, 9, 0, tzinfo=UTC)

    @staticmethod
    def _presence_health_init(path: Path) -> tuple[int, str, str]:
        output, error = io.StringIO(), io.StringIO()
        with (
            patch(
                "baby_monitor.cli_commands.candidate_runtime.load_settings", return_value=object()
            ),
            patch(
                "baby_monitor.cli_commands.candidate_runtime._candidate_detector",
                return_value=(object(), "yolo_command_v1_fixture"),
            ),
            patch(
                "baby_monitor.cli_commands.candidate_runtime._candidate_scanner_identity",
                return_value="candidate_v1_presence_fixture",
            ),
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            code = main(
                [
                    "candidate-presence-health-init",
                    "--config",
                    "ignored.toml",
                    "--camera-id",
                    "camera_01",
                    "--health",
                    str(path),
                    "--detector",
                    "yolo_command",
                ]
            )
        return code, output.getvalue(), error.getvalue()

    def test_presence_health_init_reuses_stopped_v2_v3_v4_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            path = root / "presence-health.json"
            baseline = self._health(
                path,
                camera_id="camera_01",
                now=self.NOW,
                scanner_identity="candidate_v1_presence_fixture",
            )
            for version in (2, 3, 4):
                with self.subTest(version=version):
                    health = replace(baseline, version=version)
                    health.write_atomic_private(path)
                    original = path.read_bytes()
                    code, output, error = self._presence_health_init(path)
                    self.assertEqual(0, code, error)
                    self.assertFalse(json.loads(output)["created"])
                    self.assertEqual(original, path.read_bytes())
                    self.assertEqual(health, CandidateWorkerHealth.read_private(path))
                    self.assertNotIn(str(root), output + error)

    def test_presence_health_init_rejects_legacy_identity_mismatch_and_unsafe_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            path = root / "presence-health.json"
            baseline = self._health(
                path,
                camera_id="camera_01",
                now=self.NOW,
                scanner_identity="candidate_v1_presence_fixture",
            )
            cases = [replace(baseline, version=1, lifetime=None)]
            for version in (2, 3, 4):
                health = replace(baseline, version=version)
                cases.extend(
                    [
                        replace(health, camera_id="camera_02"),
                        replace(health, scanner_identity="different_scanner"),
                        replace(health, detector_identity="different_detector"),
                        replace(health, state=CandidateHealthState.STARTING, worker_running=True),
                        replace(health, state=CandidateHealthState.SCANNING, worker_running=True),
                        replace(health, state=CandidateHealthState.IDLE, worker_running=True),
                        replace(
                            health,
                            state=CandidateHealthState.FAILED,
                            worker_running=False,
                            last_error_code="candidate_scan_failed",
                        ),
                    ]
                )
            for health in cases:
                with self.subTest(
                    version=health.version,
                    state=health.state,
                    camera=health.camera_id,
                    scanner=health.scanner_identity,
                    detector=health.detector_identity,
                ):
                    health.write_atomic_private(path)
                    original = path.read_bytes()
                    code, output, error = self._presence_health_init(path)
                    self.assertEqual(2, code)
                    self.assertIn("reusable stopped baseline", error)
                    self.assertEqual(original, path.read_bytes())
                    self.assertNotIn(str(root), output + error)

    @staticmethod
    def _health(
        path: Path,
        *,
        camera_id: str,
        now: datetime,
        lifetime: CandidateHealthLifetime | None = None,
        running: bool = False,
        scanner_identity: str = "candidate_v1_fixture",
    ) -> CandidateWorkerHealth:
        current_lifetime = lifetime or CandidateHealthLifetime.begin_session(
            None,
            updated_at=now,
        ).finish_session(updated_at=now, failed=False)
        health = CandidateWorkerHealth(
            version=2,
            camera_id=camera_id,
            scanner_identity=scanner_identity,
            detector_identity="yolo_command_v1_fixture",
            checked_at=now,
            state=CandidateHealthState.IDLE if running else CandidateHealthState.STOPPED,
            worker_running=running,
            lifetime=current_lifetime,
        )
        health.write_atomic_private(path)
        return health

    @staticmethod
    def _run_lifetime(
        previous: CandidateWorkerHealth,
        *,
        now: datetime,
        successes: int,
        failures: int = 0,
    ) -> CandidateHealthLifetime:
        lifetime = CandidateHealthLifetime.begin_session(previous, updated_at=now)
        for index in range(successes):
            lifetime = lifetime.record_scan(
                updated_at=now + timedelta(seconds=index + 1),
                succeeded=True,
                duration_seconds=4.0 + index,
                execution_lock_wait_seconds=0.5 + index,
            )
        for index in range(failures):
            lifetime = lifetime.record_scan(
                updated_at=now + timedelta(seconds=successes + index + 1),
                succeeded=False,
                duration_seconds=8.0,
                execution_lock_wait_seconds=2.0,
            )
        return lifetime.finish_session(
            updated_at=now + timedelta(seconds=successes + failures + 2),
            failed=bool(failures),
        )

    def test_private_ledger_tracks_one_completed_run_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            ledger_path = root / "candidate-soak-ledger.json"
            health_paths = (root / "camera_01.json", root / "camera_02.json")
            baseline = tuple(
                self._health(path, camera_id=f"camera_0{index}", now=self.NOW)
                for index, path in enumerate(health_paths, start=1)
            )

            started = start_candidate_soak(
                ledger_path=ledger_path,
                health_paths=health_paths,
                started_at=self.NOW,
                expected_duration_seconds=60,
            )
            self.assertEqual("running", started["latest"]["state"])
            self.assertEqual(0o600, ledger_path.stat().st_mode & 0o777)
            self.assertNotIn(str(root), json.dumps(started))

            for index, (path, previous) in enumerate(
                zip(health_paths, baseline, strict=True),
                start=1,
            ):
                lifetime = self._run_lifetime(
                    previous,
                    now=self.NOW + timedelta(seconds=1),
                    successes=index + 1,
                )
                self._health(
                    path,
                    camera_id=previous.camera_id,
                    now=self.NOW + timedelta(seconds=65),
                    lifetime=lifetime,
                )

            finished = finish_candidate_soak(
                ledger_path=ledger_path,
                health_paths=health_paths,
                ended_at=self.NOW + timedelta(seconds=65),
            )
            latest = finished["latest"]
            schema = json.loads(
                Path("schemas/candidate-soak-status-v1.schema.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(schema["required"]), set(finished))
            self.assertEqual("completed", latest["state"])
            self.assertEqual(5, latest["scans_succeeded"])
            self.assertEqual(0, latest["scans_failed"])
            self.assertFalse(finished["attention_required"])
            self.assertFalse(finished["media_read"])

            repeated = finish_candidate_soak(
                ledger_path=ledger_path,
                health_paths=health_paths,
                ended_at=self.NOW + timedelta(seconds=70),
            )
            self.assertEqual(finished["latest"]["state"], repeated["latest"]["state"])
            self.assertEqual(1, CandidateSoakLedger.read_private(ledger_path).sessions.__len__())

            tampered = json.loads(ledger_path.read_text(encoding="utf-8"))
            tampered["sessions"][-1]["attention_required"] = True
            ledger_path.write_text(json.dumps(tampered), encoding="utf-8")
            ledger_path.chmod(0o600)
            with self.assertRaisesRegex(CandidateSoakError, "attention state"):
                CandidateSoakLedger.read_private(ledger_path)

    def test_early_stop_and_failure_are_visible_without_private_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            ledger_path = root / "ledger.json"
            health_path = root / "health.json"
            previous = self._health(health_path, camera_id="camera_01", now=self.NOW)
            start_candidate_soak(
                ledger_path=ledger_path,
                health_paths=(health_path,),
                started_at=self.NOW,
                expected_duration_seconds=86_400,
            )
            lifetime = self._run_lifetime(
                previous,
                now=self.NOW + timedelta(seconds=1),
                successes=1,
                failures=1,
            )
            self._health(
                health_path,
                camera_id="camera_01",
                now=self.NOW + timedelta(minutes=1),
                lifetime=lifetime,
            )

            result = finish_candidate_soak(
                ledger_path=ledger_path,
                health_paths=(health_path,),
                ended_at=self.NOW + timedelta(minutes=1),
            )

            self.assertEqual("stopped_early", result["latest"]["state"])
            self.assertEqual(1, result["latest"]["scans_failed"])
            self.assertEqual(1, result["latest"]["sessions_failed"])
            self.assertTrue(result["attention_required"])
            self.assertNotIn(str(root), json.dumps(result))

    def test_restart_marks_unfinished_session_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            ledger_path = root / "ledger.json"
            health_path = root / "health.json"
            self._health(health_path, camera_id="camera_01", now=self.NOW)
            start_candidate_soak(
                ledger_path=ledger_path,
                health_paths=(health_path,),
                started_at=self.NOW,
                expected_duration_seconds=60,
            )
            long_running = inspect_candidate_soak(
                ledger_path=ledger_path,
                checked_at=self.NOW + timedelta(days=8),
            )
            self.assertGreater(long_running["latest"]["elapsed_seconds"], 604_800)
            start_candidate_soak(
                ledger_path=ledger_path,
                health_paths=(health_path,),
                started_at=self.NOW + timedelta(minutes=2),
                expected_duration_seconds=60,
            )

            ledger = CandidateSoakLedger.read_private(ledger_path)
            self.assertEqual(2, len(ledger.sessions))
            self.assertEqual("interrupted", ledger.sessions[0].state)
            self.assertTrue(ledger.sessions[0].attention_required)
            self.assertEqual("running", ledger.sessions[1].state)

    def test_missing_corrupt_running_or_identity_change_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            ledger_path = root / "ledger.json"
            health_path = root / "health.json"
            missing = inspect_candidate_soak(ledger_path=ledger_path, checked_at=self.NOW)
            self.assertEqual("not_initialized", missing["status"])
            self.assertFalse(ledger_path.exists())

            self._health(
                health_path,
                camera_id="camera_01",
                now=self.NOW,
                running=True,
            )
            with self.assertRaisesRegex(CandidateSoakError, "must be stopped"):
                start_candidate_soak(
                    ledger_path=ledger_path,
                    health_paths=(health_path,),
                    started_at=self.NOW,
                    expected_duration_seconds=60,
                )

            previous = self._health(health_path, camera_id="camera_01", now=self.NOW)
            start_candidate_soak(
                ledger_path=ledger_path,
                health_paths=(health_path,),
                started_at=self.NOW,
                expected_duration_seconds=60,
            )
            lifetime = self._run_lifetime(
                previous,
                now=self.NOW + timedelta(seconds=1),
                successes=1,
            )
            self._health(
                health_path,
                camera_id="camera_01",
                now=self.NOW + timedelta(seconds=65),
                lifetime=lifetime,
                scanner_identity="changed_scanner",
            )
            with self.assertRaisesRegex(CandidateSoakError, "identity changed"):
                finish_candidate_soak(
                    ledger_path=ledger_path,
                    health_paths=(health_path,),
                    ended_at=self.NOW + timedelta(seconds=65),
                )

            ledger_path.write_text("{}", encoding="utf-8")
            ledger_path.chmod(0o600)
            unavailable = inspect_candidate_soak(
                ledger_path=ledger_path,
                checked_at=self.NOW,
            )
            self.assertEqual("unavailable", unavailable["status"])
            self.assertTrue(unavailable["attention_required"])

    def test_cli_start_finish_and_status_are_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            ledger_path = root / "ledger.json"
            health_path = root / "health.json"
            self._health(health_path, camera_id="camera_01", now=datetime.now(UTC))

            output = io.StringIO()
            with redirect_stdout(output):
                start_code = main(
                    [
                        "candidate-soak-start",
                        "--ledger",
                        str(ledger_path),
                        "--health",
                        str(health_path),
                        "--expected-duration-seconds",
                        "60",
                    ]
                )
            self.assertEqual(0, start_code)
            self.assertEqual("running", json.loads(output.getvalue())["latest"]["state"])
            self.assertNotIn(str(root), output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                finish_code = main(
                    [
                        "candidate-soak-finish",
                        "--ledger",
                        str(ledger_path),
                        "--health",
                        str(health_path),
                    ]
                )
            self.assertEqual(0, finish_code)
            self.assertEqual(
                "stopped_early",
                json.loads(output.getvalue())["latest"]["state"],
            )
            self.assertNotIn(str(root), output.getvalue())

            output = io.StringIO()
            with redirect_stdout(output):
                status_code = main(["candidate-soak-status", "--ledger", str(ledger_path)])
            self.assertEqual(1, status_code)
            self.assertTrue(json.loads(output.getvalue())["attention_required"])

    def test_presence_health_init_creates_and_reuses_only_stopped_v2_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            health_path = root / "presence-health.json"
            argv = [
                "candidate-presence-health-init",
                "--config",
                "ignored.toml",
                "--camera-id",
                "camera_01",
                "--health",
                str(health_path),
                "--detector",
                "yolo_command",
            ]
            with (
                patch(
                    "baby_monitor.cli_commands.candidate_runtime.load_settings",
                    return_value=object(),
                ),
                patch(
                    "baby_monitor.cli_commands.candidate_runtime._candidate_detector",
                    return_value=(object(), "yolo_command_v1_fixture"),
                ),
                patch(
                    "baby_monitor.cli_commands.candidate_runtime._candidate_scanner_identity",
                    return_value="candidate_v1_presence_fixture",
                ),
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(0, main(argv))
                created = json.loads(output.getvalue())
                self.assertTrue(created["created"])
                self.assertFalse(created["media_read"])
                self.assertFalse(created["analysis_queue_touched"])
                self.assertNotIn(str(root), output.getvalue())

                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(0, main(argv))
                self.assertFalse(json.loads(output.getvalue())["created"])

                current = CandidateWorkerHealth.read_private(health_path)
                CandidateWorkerHealth(
                    version=2,
                    camera_id=current.camera_id,
                    scanner_identity=current.scanner_identity,
                    detector_identity=current.detector_identity,
                    checked_at=current.checked_at,
                    state=CandidateHealthState.IDLE,
                    worker_running=True,
                    lifetime=CandidateHealthLifetime.begin_session(
                        current,
                        updated_at=current.checked_at,
                    ),
                ).write_atomic_private(health_path)
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    self.assertEqual(2, main(argv))
                self.assertIn("reusable stopped baseline", stderr.getvalue())
                self.assertNotIn(str(root), stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
