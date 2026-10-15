from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from baby_monitor.cli import main
from baby_monitor.home_dashboard import HomeDashboardApplication
from baby_monitor.room_presence import (
    StreamPresenceBatch,
    StreamPresenceSample,
    StreamPresenceSampleState,
)
from tests.test_room_registry import fixture_registry


class RoomCLITests(unittest.TestCase):
    NOW = datetime(2026, 8, 20, 5, 0, tzinfo=UTC)

    @staticmethod
    def _write_private(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        path.chmod(0o600)

    def test_registry_status_and_presence_report_are_path_free(self) -> None:
        registry = fixture_registry()
        batch = StreamPresenceBatch(
            samples=(
                StreamPresenceSample(
                    sample_id="sample_1",
                    camera_id="camera_01",
                    observed_at=self.NOW,
                    state=StreamPresenceSampleState.PERSON_DETECTED,
                    scanner_id="candidate_v1_fixture",
                    detector_id="yolo_command_v1_fixture",
                    source_fingerprint="1" * 64,
                ),
                StreamPresenceSample(
                    sample_id="sample_2",
                    camera_id="camera_01",
                    observed_at=self.NOW + timedelta(seconds=10),
                    state=StreamPresenceSampleState.PERSON_DETECTED,
                    scanner_id="candidate_v1_fixture",
                    detector_id="yolo_command_v1_fixture",
                    source_fingerprint="2" * 64,
                ),
            )
        )
        with tempfile.TemporaryDirectory(prefix="baby-monitor-room-cli-") as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            registry_path = root / "registry.json"
            samples_path = root / "samples.json"
            self._write_private(registry_path, registry.to_dict())
            self._write_private(samples_path, batch.to_dict())

            status_stdout = io.StringIO()
            with redirect_stdout(status_stdout):
                result = main(["room-registry-status", "--registry", str(registry_path)])
            self.assertEqual(0, result)
            status = json.loads(status_stdout.getvalue())
            self.assertEqual(
                (3, 3, 4),
                (
                    status["room_count"],
                    status["physical_camera_count"],
                    status["logical_stream_count"],
                ),
            )
            self.assertNotIn(str(root), status_stdout.getvalue())
            self.assertFalse(status["identity_inference"])

            report_stdout = io.StringIO()
            with redirect_stdout(report_stdout):
                result = main(
                    [
                        "room-presence-evaluate",
                        "--registry",
                        str(registry_path),
                        "--samples",
                        str(samples_path),
                        "--checked-at",
                        (self.NOW + timedelta(seconds=20)).isoformat(),
                    ]
                )
            self.assertEqual(0, result)
            report = json.loads(report_stdout.getvalue())
            self.assertEqual("person_present", report["rooms"][0]["status"])
            self.assertFalse(report["identity_inference"])
            self.assertFalse(report["risk_assessment"])
            self.assertFalse(report["notification_authority"])
            self.assertNotIn(str(root), report_stdout.getvalue())
            self.assertNotIn("camera_01", report_stdout.getvalue())

    def test_cli_rejects_weak_private_registry_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-room-cli-") as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            registry_path = root / "registry.json"
            samples_path = root / "samples.json"
            self._write_private(registry_path, fixture_registry().to_dict())
            self._write_private(samples_path, StreamPresenceBatch(samples=()).to_dict())
            registry_path.chmod(0o644)

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = main(
                    [
                        "room-presence-evaluate",
                        "--registry",
                        str(registry_path),
                        "--samples",
                        str(samples_path),
                    ]
                )
            self.assertEqual(2, result)
            self.assertEqual("", stdout.getvalue())
            self.assertIn("permissions are unsafe", stderr.getvalue())
            self.assertNotIn(str(root), stderr.getvalue())

    def test_home_dashboard_status_provider_reads_room_facts_without_media(self) -> None:
        registry = fixture_registry()
        observed_at = datetime.now(UTC) - timedelta(seconds=15)
        batch = StreamPresenceBatch(
            samples=(
                StreamPresenceSample(
                    sample_id="dashboard_1",
                    camera_id="camera_01",
                    observed_at=observed_at,
                    state=StreamPresenceSampleState.PERSON_DETECTED,
                    scanner_id="candidate_v1_fixture",
                    detector_id="yolo_command_v1_fixture",
                    source_fingerprint="1" * 64,
                ),
                StreamPresenceSample(
                    sample_id="dashboard_2",
                    camera_id="camera_01",
                    observed_at=observed_at + timedelta(seconds=10),
                    state=StreamPresenceSampleState.PERSON_DETECTED,
                    scanner_id="candidate_v1_fixture",
                    detector_id="yolo_command_v1_fixture",
                    source_fingerprint="2" * 64,
                ),
            )
        )
        with tempfile.TemporaryDirectory(prefix="baby-monitor-room-dashboard-") as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            registry_path = root / "registry.json"
            samples_path = root / "samples.json"
            self._write_private(registry_path, registry.to_dict())
            self._write_private(samples_path, batch.to_dict())
            captured: list[dict[str, object]] = []

            def serve(application: object, **kwargs: object) -> dict[str, object]:
                self.assertIsInstance(application, HomeDashboardApplication)
                assert isinstance(application, HomeDashboardApplication)
                captured.append(application.status_provider())
                return {
                    "version": 1,
                    "status": "stopped",
                    "requests": 0,
                    "bind": "loopback",
                    "remote_used": False,
                    "media_read": False,
                    "paths_disclosed": False,
                }

            stdout = io.StringIO()
            with (
                mock.patch(
                    "baby_monitor.cli_commands.home_dashboard.control_home_stack",
                    return_value={"version": 1, "manual_start_only": True},
                ),
                mock.patch(
                    "baby_monitor.cli_commands.home_dashboard.serve_home_dashboard",
                    side_effect=serve,
                ),
                redirect_stdout(stdout),
            ):
                result = main(
                    [
                        "home-dashboard",
                        "--room-registry",
                        str(registry_path),
                        "--room-presence-samples",
                        str(samples_path),
                        "--max-runtime-seconds",
                        "60",
                    ]
                )

            self.assertEqual(0, result)
            self.assertEqual(1, len(captured))
            room_status = captured[0]["room_presence"]
            self.assertIsInstance(room_status, dict)
            assert isinstance(room_status, dict)
            self.assertEqual("available", room_status["status"])
            self.assertEqual("person_present", room_status["rooms"][0]["status"])
            self.assertFalse(room_status["media_read"])
            self.assertFalse(room_status["identity_inference"])
            self.assertNotIn(str(root), json.dumps(captured, ensure_ascii=False))
            self.assertNotIn("camera_01", json.dumps(room_status, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
