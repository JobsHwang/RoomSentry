from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from baby_monitor.cli import main
from baby_monitor.systemd_deployment import (
    DeploymentProfileError,
    RuntimeDeploymentProfile,
    render_systemd_user_units,
)


class SystemdDeploymentTests(unittest.TestCase):
    @staticmethod
    def _fixture(
        root: Path,
        *,
        confirm: bool = True,
        candidate_detector: str | None = None,
        auto_file: bool = False,
        trial_run_seconds: int | None = None,
        deferred_reports: bool | None = None,
        visual_shadow: bool = False,
        pose_presence_shadow: bool = False,
    ) -> tuple[Path, Path, dict[str, object]]:
        project = root / "project"
        python = project / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("fixture", encoding="utf-8")
        python.chmod(0o700)
        private = root / "private"
        private.mkdir(mode=0o700)
        config = private / "runtime.toml"
        config_lines = ["[buffer]", "eviction_enabled = true"]
        if candidate_detector == "yolo_command":
            config_lines.extend(
                [
                    "[detector]",
                    'backend = "yolo_command"',
                    f'yolo_runner_command = ["{python.as_posix()}", "{{request}}"]',
                    *(
                        [f'yolo_execution_lock_path = "{(private / "gpu.lock").as_posix()}"']
                        if pose_presence_shadow
                        else []
                    ),
                ]
            )
        if auto_file:
            event_policy = private / "event-policy.json"
            shutil.copyfile(
                Path("config/household-event-policy.example.json"),
                event_policy,
            )
            config_lines.extend(
                [
                    "[notification]",
                    "enabled = true",
                    "auto_file_enabled = true",
                    'auto_file_sink_id = "shadow_local"',
                    f'event_policy_path = "{event_policy.as_posix()}"',
                ]
            )
        if visual_shadow:
            config_lines.extend(
                [
                    "[visual_predicate_shadow]",
                    "enabled = true",
                    'backend = "conservative"',
                ]
            )
        config.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
        database_parent = root / "state"
        database_parent.mkdir()
        output = root / "spool"
        output.mkdir()
        buffer_root = root / "buffer"
        buffer_root.mkdir()
        units = root / "units"
        units.mkdir()
        payload: dict[str, object] = {
            "version": 1,
            "deployment_id": "home_lab",
            "project_root": str(project),
            "config_path": str(config),
            "database_path": str(database_parent / "events.sqlite3"),
            "output_root": str(output),
            "buffer_run_seconds": 3600,
            "confirm_ephemeral_buffer_eviction": confirm,
            "cameras": [
                {
                    "camera_id": "c500_fixed",
                    "rtsp_uri": "rtsp://127.0.0.1:8554/c500/fixed",
                    "buffer_root": str(buffer_root),
                }
            ],
        }
        if candidate_detector is not None:
            payload["candidate_detector"] = candidate_detector
        if trial_run_seconds is not None:
            payload["trial_run_seconds"] = trial_run_seconds
        if deferred_reports is not None:
            payload["deferred_reports"] = deferred_reports
        if pose_presence_shadow:
            artifacts = {
                "profile_path": private / "pose-profile.json",
                "temporal_attestation_path": private / "pose-attestation.json",
                "human_audit_report_path": private / "pose-audit.json",
                "detector_manifest_path": private / "detector-manifest.json",
                "detector_weights_path": private / "detector-weights.pt",
                "pose_manifest_path": private / "pose-manifest.json",
                "pose_weights_path": private / "pose-weights.pt",
            }
            for artifact in artifacts.values():
                artifact.write_bytes(b"fixture")
            payload["pose_presence_shadow"] = {
                **{key: str(path) for key, path in artifacts.items()},
                "detector_runner_executable": str(python),
                "detector_device": "cpu",
                "pose_runner_executable": str(python),
                "pose_device": "cpu",
                "pose_timeout_seconds": 120,
                "sample_interval_seconds": 300,
            }
        profile = private / "deployment.json"
        profile.write_text(json.dumps(payload), encoding="utf-8")
        return profile, units, payload

    def test_rendered_units_are_hardened_bounded_and_never_auto_started(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(root)
            profile = RuntimeDeploymentProfile.read(profile_path)
            rendered = render_systemd_user_units(profile, units_directory)

            self.assertEqual(9, len(rendered.unit_files))
            self.assertTrue(rendered.confirm_ephemeral_buffer_eviction)
            public = rendered.to_public_dict()
            self.assertFalse(public["installed"])
            self.assertFalse(public["enabled"])
            self.assertFalse(public["started"])
            self.assertFalse(public["contains_credentials"])
            self.assertFalse(public["allows_remote_network"])
            self.assertEqual("scene", public["candidate_detector"])
            self.assertIsNone(public["trial_run_seconds"])
            self.assertFalse(public["bounded_trial"])
            self.assertTrue(public["continuous_buffer_ready"])
            self.assertFalse(public["requires_user_eviction_confirmation"])
            self.assertNotIn(str(root), json.dumps(public))

            buffer_unit = (
                units_directory / "baby-monitor-home_lab-c500_fixed-buffer.service"
            ).read_text(encoding="utf-8")
            candidate_unit = (
                units_directory / "baby-monitor-home_lab-c500_fixed-candidate.service"
            ).read_text(encoding="utf-8")
            analysis_unit = (units_directory / "baby-monitor-home_lab-analysis.service").read_text(
                encoding="utf-8"
            )
            init_unit = (units_directory / "baby-monitor-home_lab-init.service").read_text(
                encoding="utf-8"
            )
            report_unit = (
                units_directory / "baby-monitor-home_lab-daily-report.service"
            ).read_text(encoding="utf-8")
            target_unit = (units_directory / "baby-monitor-home_lab.target").read_text(
                encoding="utf-8"
            )

            self.assertIn("IPAddressAllow=localhost", buffer_unit)
            self.assertIn("PartOf=baby-monitor-home_lab.target", buffer_unit)
            self.assertIn("PartOf=baby-monitor-home_lab.target", candidate_unit)
            self.assertIn("PartOf=baby-monitor-home_lab.target", analysis_unit)
            self.assertIn("PartOf=baby-monitor-home_lab.target", init_unit)
            self.assertIn("--confirm-ephemeral-eviction", buffer_unit)
            self.assertIn("Restart=always", buffer_unit)
            self.assertIn("MemoryMax=1G", buffer_unit)
            self.assertIn('--detector" "scene', candidate_unit)
            self.assertIn('--image-size" "640', candidate_unit)
            self.assertIn('--rotation-degrees" "0', candidate_unit)
            self.assertIn('--idle-exit-seconds" "0', candidate_unit)
            self.assertIn('--max-runtime-seconds" "3600', candidate_unit)
            self.assertIn('--presence" "', candidate_unit)
            self.assertIn("stream-presence-batch.json", candidate_unit)
            self.assertIn("MemoryMax=12G", analysis_unit)
            self.assertIn("PrivateTmp=yes", analysis_unit)
            self.assertIn("RuntimeDirectory=baby-monitor-vlm-execution", analysis_unit)
            self.assertIn("RuntimeDirectoryMode=0700", analysis_unit)
            self.assertIn("RuntimeDirectoryPreserve=yes", analysis_unit)
            self.assertIn("IPAddressDeny=any", analysis_unit)
            self.assertNotIn("IPAddressAllow=", analysis_unit)
            self.assertIn("init.service", analysis_unit)
            self.assertIn('init-db" "--config', init_unit)
            self.assertIn("RemainAfterExit=yes", init_unit)
            self.assertIn("IPAddressDeny=any", init_unit)
            self.assertIn('--day-offset" "-1', report_unit)
            self.assertIn("daily-report.timer", target_unit)
            self.assertIn("health.timer", target_unit)

            combined = "\n".join(
                (units_directory / name).read_text(encoding="utf-8") for name in rendered.unit_files
            )
            self.assertNotIn("/bin/sh", combined)
            self.assertNotIn("bash -c", combined)
            self.assertNotIn("password", combined.lower())
            self.assertNotIn("token", combined.lower())
            if os.name == "posix":
                self.assertTrue(
                    all(
                        (units_directory / name).stat().st_mode & 0o777 == 0o600
                        for name in (*rendered.unit_files, rendered.manifest_file)
                    )
                )

            manifest = json.loads(
                (units_directory / rendered.manifest_file).read_text(encoding="utf-8")
            )
            for item in manifest["unit_files"]:
                content = (units_directory / item["name"]).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])

    def test_deferred_reports_default_and_explicit_false_keep_identical_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, payload = self._fixture(root)
            profile = RuntimeDeploymentProfile.read(profile_path)
            self.assertFalse(profile.deferred_reports)
            default = render_systemd_user_units(profile, units_directory)
            explicit_directory = root / "explicit-units"
            explicit_directory.mkdir()
            payload["deferred_reports"] = False
            profile_path.write_text(json.dumps(payload), encoding="utf-8")
            explicit = render_systemd_user_units(
                RuntimeDeploymentProfile.read(profile_path), explicit_directory
            )
            self.assertEqual(default.to_public_dict(), explicit.to_public_dict())
            self.assertNotIn("deferred_reports", default.to_public_dict())
            self.assertNotIn("baby-monitor-home_lab-report-refresh.service", default.unit_files)
            for name in (*default.unit_files, default.manifest_file):
                with self.subTest(name=name):
                    self.assertEqual(
                        (units_directory / name).read_bytes(),
                        (explicit_directory / name).read_bytes(),
                    )
            analysis = (units_directory / "baby-monitor-home_lab-analysis.service").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("--defer-reports", analysis)

    def test_deferred_report_worker_is_bounded_private_and_part_of_trial_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(
                root, deferred_reports=True, trial_run_seconds=86_400
            )
            profile = RuntimeDeploymentProfile.read(profile_path)
            rendered = render_systemd_user_units(profile, units_directory)
            refresh_name = "baby-monitor-home_lab-report-refresh.service"
            refresh = (units_directory / refresh_name).read_text(encoding="utf-8")
            analysis = (units_directory / "baby-monitor-home_lab-analysis.service").read_text(
                encoding="utf-8"
            )
            daily = (units_directory / "baby-monitor-home_lab-daily-report.service").read_text(
                encoding="utf-8"
            )
            target = (units_directory / "baby-monitor-home_lab.target").read_text(encoding="utf-8")
            stop = (units_directory / "baby-monitor-home_lab-trial-stop.service").read_text(
                encoding="utf-8"
            )
            self.assertEqual(13, len(rendered.unit_files))
            self.assertIn('"--defer-reports"', analysis)
            self.assertIn("After=baby-monitor-home_lab-init.service", refresh)
            self.assertIn("Requires=baby-monitor-home_lab-init.service", refresh)
            self.assertIn("PartOf=baby-monitor-home_lab.target", refresh)
            self.assertIn("WantedBy=baby-monitor-home_lab.target", refresh)
            self.assertIn(refresh_name, target)
            self.assertIn("Conflicts=baby-monitor-home_lab.target", stop)
            self.assertIn("After=baby-monitor-home_lab.target", stop)
            for argument in (
                '"report-refresh-worker"',
                '"--max-jobs" "0"',
                '"--idle-exit-seconds" "0"',
                '"--max-runtime-seconds" "3600"',
                '"--poll-seconds" "5"',
            ):
                self.assertIn(argument, refresh)
            self.assertIn(f'"--config" "{profile.config_path}"', refresh)
            self.assertIn(f'"--database" "{profile.database_path}"', refresh)
            for directive in (
                "MemoryMax=1G",
                "IPAddressDeny=any",
                "PrivateDevices=yes",
                "NoNewPrivileges=yes",
                "ProtectSystem=strict",
                "UMask=0077",
                "KillMode=control-group",
                "TimeoutStopSec=30s",
                "Restart=always",
            ):
                self.assertIn(directive, refresh)
            self.assertNotIn("IPAddressAllow=", refresh)
            self.assertNotIn("gpu", refresh.lower())
            self.assertEqual(
                {str(profile.database_path.parent), str(profile.output_root)},
                {
                    line.removeprefix("ReadWritePaths=")
                    for line in refresh.splitlines()
                    if line.startswith("ReadWritePaths=")
                },
            )
            self.assertIn('"report"', daily)
            self.assertIn('"--day-offset" "-1"', daily)
            self.assertNotIn("report-refresh", daily)
            self.assertTrue(rendered.to_public_dict()["deferred_reports"])
            self.assertFalse(rendered.to_public_dict()["started"])
            manifest = json.loads(
                (units_directory / rendered.manifest_file).read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["deferred_reports"])
            item = next(item for item in manifest["unit_files"] if item["name"] == refresh_name)
            self.assertEqual(
                hashlib.sha256((units_directory / refresh_name).read_bytes()).hexdigest(),
                item["sha256"],
            )
            if os.name == "posix":
                self.assertEqual(0o600, (units_directory / refresh_name).stat().st_mode & 0o777)

    def test_deferred_reports_requires_a_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, _units_directory, payload = self._fixture(root)
            for invalid in (None, 0, 1, "true", [], {}):
                with self.subTest(value=invalid):
                    payload["deferred_reports"] = invalid
                    profile_path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(DeploymentProfileError, "deferred_reports"):
                        RuntimeDeploymentProfile.read(profile_path)

    def test_enabled_shadow_worker_is_isolated_and_not_rendered_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(
                root,
                visual_shadow=True,
            )
            rendered = render_systemd_user_units(
                RuntimeDeploymentProfile.read(profile_path),
                units_directory,
            )
            self.assertEqual(10, len(rendered.unit_files))
            shadow = (
                units_directory / "baby-monitor-home_lab-visual-predicate-shadow.service"
            ).read_text(encoding="utf-8")
            target = (units_directory / "baby-monitor-home_lab.target").read_text(encoding="utf-8")
            self.assertIn(
                "After=baby-monitor-home_lab-init.service baby-monitor-home_lab-analysis.service",
                shadow,
            )
            self.assertIn("Requires=baby-monitor-home_lab-init.service", shadow)
            self.assertIn("PartOf=baby-monitor-home_lab.target", shadow)
            self.assertIn('"visual-predicate-shadow-worker"', shadow)
            self.assertIn('"--max-jobs" "32"', shadow)
            self.assertIn('"--idle-exit-seconds" "0"', shadow)
            self.assertIn("MemoryMax=12G", shadow)
            self.assertIn("IPAddressDeny=any", shadow)
            self.assertNotIn("IPAddressAllow=", shadow)
            self.assertIn("Restart=always", shadow)
            self.assertIn("visual-predicate-shadow.service", target)
            public = rendered.to_public_dict()
            self.assertFalse(public["enabled"])
            self.assertFalse(public["started"])
            self.assertFalse(public["allows_remote_network"])

    def test_shadow_deployment_rejects_a_second_vlm_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(
                root,
                visual_shadow=True,
            )
            profile = RuntimeDeploymentProfile.read(profile_path)
            with profile.config_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "\n[local_vlm]\n"
                    'backend = "command"\n'
                    'name = "second_model"\n'
                    'command = ["/bin/false", "{request}"]\n'
                )
            with self.assertRaisesRegex(
                DeploymentProfileError,
                "requires all other VLM backends disabled",
            ):
                render_systemd_user_units(profile, units_directory)
            self.assertEqual([], list(units_directory.iterdir()))

    def test_yolo_candidate_and_local_notification_worker_are_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(
                root,
                candidate_detector="yolo_command",
                auto_file=True,
            )
            rendered = render_systemd_user_units(
                RuntimeDeploymentProfile.read(profile_path),
                units_directory,
            )
            self.assertEqual(10, len(rendered.unit_files))
            self.assertEqual("yolo_command", rendered.candidate_detector)
            candidate = (
                units_directory / "baby-monitor-home_lab-c500_fixed-candidate.service"
            ).read_text(encoding="utf-8")
            notification = (
                units_directory / "baby-monitor-home_lab-notification.service"
            ).read_text(encoding="utf-8")
            target = (units_directory / "baby-monitor-home_lab.target").read_text(encoding="utf-8")
            self.assertIn('--detector" "yolo_command', candidate)
            self.assertIn("MemoryMax=4G", candidate)
            self.assertIn('notification-worker" "--config', notification)
            self.assertIn('--sink" "file', notification)
            self.assertIn('--sink-id" "shadow_local', notification)
            self.assertIn("IPAddressDeny=any", notification)
            self.assertNotIn("IPAddressAllow=", notification)
            self.assertIn("notification.service", target)

    def test_pose_presence_shadow_is_rendered_but_never_started(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(
                root,
                candidate_detector="yolo_command",
                pose_presence_shadow=True,
            )
            rendered = render_systemd_user_units(
                RuntimeDeploymentProfile.read(profile_path),
                units_directory,
            )
            candidate = (
                units_directory / "baby-monitor-home_lab-c500_fixed-candidate.service"
            ).read_text(encoding="utf-8")
            public = rendered.to_public_dict()

            self.assertTrue(public["pose_presence_shadow"])
            self.assertFalse(public["installed"])
            self.assertFalse(public["enabled"])
            self.assertFalse(public["started"])
            self.assertIn("MemoryMax=8G", candidate)
            self.assertIn('"--pose-presence-shadow-profile"', candidate)
            self.assertIn('"--pose-presence-shadow-attestation"', candidate)
            self.assertIn('"--pose-presence-shadow-audit"', candidate)
            self.assertIn('"--pose-presence-shadow-output"', candidate)
            self.assertIn("pose-presence-shadow-batch.json", candidate)
            self.assertIn('"--pose-shadow-detector-manifest"', candidate)
            self.assertIn('"--pose-shadow-detector-weights"', candidate)
            self.assertIn('"--pose-shadow-detector-runner-executable"', candidate)
            self.assertIn('"--pose-shadow-detector-device" "cpu"', candidate)
            self.assertIn('"--pose-shadow-manifest"', candidate)
            self.assertIn('"--pose-shadow-weights"', candidate)
            self.assertIn('"--pose-shadow-runner-executable"', candidate)
            self.assertIn('"--pose-shadow-device" "cpu"', candidate)
            self.assertIn('"--pose-shadow-timeout-seconds" "120.0"', candidate)
            self.assertIn('"--pose-shadow-min-interval-seconds" "300.0"', candidate)
            self.assertIn("ReadWritePaths=", candidate)
            self.assertNotIn("IPAddressAllow=", candidate)

            manifest = json.loads(
                (units_directory / rendered.manifest_file).read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["pose_presence_shadow"])

    def test_pose_presence_shadow_requires_yolo_and_private_shared_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, payload = self._fixture(
                root,
                candidate_detector="yolo_command",
                pose_presence_shadow=True,
            )
            changed = json.loads(json.dumps(payload))
            changed["candidate_detector"] = "scene"
            profile_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(
                DeploymentProfileError,
                "requires candidate_detector=yolo_command",
            ):
                render_systemd_user_units(
                    RuntimeDeploymentProfile.read(profile_path),
                    units_directory,
                )

            changed = json.loads(json.dumps(payload))
            changed["pose_presence_shadow"]["sample_interval_seconds"] = 29
            profile_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(DeploymentProfileError, "sample interval"):
                RuntimeDeploymentProfile.read(profile_path)

    def test_trial_timer_stops_target_after_exact_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(
                root,
                candidate_detector="yolo_command",
                pose_presence_shadow=True,
                trial_run_seconds=86_400,
            )
            rendered = render_systemd_user_units(
                RuntimeDeploymentProfile.read(profile_path),
                units_directory,
            )
            self.assertEqual(12, len(rendered.unit_files))
            self.assertEqual(86_400, rendered.trial_run_seconds)
            public = rendered.to_public_dict()
            self.assertTrue(public["bounded_trial"])
            self.assertEqual(86_400, public["trial_run_seconds"])

            stop_service = (units_directory / "baby-monitor-home_lab-trial-stop.service").read_text(
                encoding="utf-8"
            )
            baseline_service = (
                units_directory / "baby-monitor-home_lab-trial-baseline.service"
            ).read_text(encoding="utf-8")
            stop_timer = (units_directory / "baby-monitor-home_lab-trial-stop.timer").read_text(
                encoding="utf-8"
            )
            target = (units_directory / "baby-monitor-home_lab.target").read_text(encoding="utf-8")
            self.assertIn("Conflicts=baby-monitor-home_lab.target", stop_service)
            self.assertIn("After=baby-monitor-home_lab.target", stop_service)
            self.assertIn('"runtime-trial-finish"', stop_service)
            self.assertIn("IPAddressDeny=any", stop_service)
            self.assertIn('"runtime-trial-start"', baseline_service)
            self.assertIn('"--health-wait-seconds" "30"', baseline_service)
            self.assertIn('"--shadow-expected"', baseline_service)
            self.assertIn("RemainAfterExit=yes", baseline_service)
            self.assertIn("Before=baby-monitor-home_lab.target", baseline_service)
            self.assertIn(
                "Requires=baby-monitor-home_lab-c500_fixed-candidate.service", baseline_service
            )
            self.assertIn("OnActiveSec=86400s", stop_timer)
            self.assertIn("AccuracySec=1s", stop_timer)
            self.assertIn("PartOf=baby-monitor-home_lab.target", stop_timer)
            self.assertIn("Requires=baby-monitor-home_lab-trial-baseline.service", stop_timer)
            self.assertIn("trial-stop.timer", target)
            self.assertIn("Requires=baby-monitor-home_lab-trial-baseline.service", target)

            manifest = json.loads(
                (units_directory / rendered.manifest_file).read_text(encoding="utf-8")
            )
            self.assertEqual(86_400, manifest["trial_run_seconds"])

    def test_trial_bound_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, _units_directory, payload = self._fixture(root)
            for invalid in (False, 3_599, 604_801):
                with self.subTest(invalid=invalid):
                    changed = json.loads(json.dumps(payload))
                    changed["trial_run_seconds"] = invalid
                    profile_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(
                        DeploymentProfileError,
                        "trial_run_seconds",
                    ):
                        RuntimeDeploymentProfile.read(profile_path)

    def test_camera_transform_is_validated_and_rendered_per_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, payload = self._fixture(root)
            camera = payload["cameras"][0]
            assert isinstance(camera, dict)
            camera["image_size"] = 960
            camera["rotation_degrees"] = 90
            profile_path.write_text(json.dumps(payload), encoding="utf-8")

            profile = RuntimeDeploymentProfile.read(profile_path)
            self.assertEqual(960, profile.cameras[0].image_size)
            self.assertEqual(90, profile.cameras[0].rotation_degrees)
            render_systemd_user_units(profile, units_directory)
            candidate = (
                units_directory / "baby-monitor-home_lab-c500_fixed-candidate.service"
            ).read_text(encoding="utf-8")
            self.assertIn('--image-size" "960', candidate)
            self.assertIn('--rotation-degrees" "90', candidate)

    def test_camera_transform_rejects_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, _units_directory, payload = self._fixture(root)
            for field, value in (
                ("image_size", 641),
                ("rotation_degrees", 45),
            ):
                with self.subTest(field=field, value=value):
                    changed = json.loads(json.dumps(payload))
                    changed["cameras"][0][field] = value
                    profile_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(
                        DeploymentProfileError,
                        field,
                    ):
                        RuntimeDeploymentProfile.read(profile_path)

    def test_yolo_candidate_requires_matching_detector_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(root)
            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            payload["candidate_detector"] = "yolo_command"
            profile_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                DeploymentProfileError,
                "requires detector.backend",
            ):
                render_systemd_user_units(
                    RuntimeDeploymentProfile.read(profile_path),
                    units_directory,
                )

    @unittest.skipUnless(os.name == "posix", "venv symlink semantics are POSIX-specific")
    def test_standard_venv_python_symlink_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, _units_directory, payload = self._fixture(root)
            python = Path(str(payload["project_root"])) / ".venv/bin/python"
            target = python.parent / "python3"
            python.unlink()
            target.write_text("fixture", encoding="utf-8")
            target.chmod(0o700)
            python.symlink_to("python3")
            profile = RuntimeDeploymentProfile.read(profile_path)
            self.assertTrue(profile.python_executable.is_symlink())

    def test_renderer_is_exclusive_and_false_confirmation_omits_delete_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(root, confirm=False)
            profile = RuntimeDeploymentProfile.read(profile_path)
            rendered = render_systemd_user_units(profile, units_directory)
            buffer_path = units_directory / "baby-monitor-home_lab-c500_fixed-buffer.service"
            before = buffer_path.read_bytes()
            self.assertNotIn("--confirm-ephemeral-eviction", before.decode("utf-8"))
            with self.assertRaisesRegex(FileExistsError, "never overwrites"):
                render_systemd_user_units(profile, units_directory)
            self.assertEqual(before, buffer_path.read_bytes())
            self.assertFalse(rendered.confirm_ephemeral_buffer_eviction)
            self.assertFalse(rendered.to_public_dict()["continuous_buffer_ready"])

    def test_confirmation_requires_persistent_buffer_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, payload = self._fixture(root)
            config_path = Path(str(payload["config_path"]))
            config_path.write_text(
                "[buffer]\neviction_enabled = false\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DeploymentProfileError,
                "persistent buffer.eviction_enabled",
            ):
                render_systemd_user_units(
                    RuntimeDeploymentProfile.read(profile_path),
                    units_directory,
                )

    def test_partial_render_failure_cleans_only_files_from_that_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(root)
            existing = units_directory / "unrelated.keep"
            existing.write_text("owner data", encoding="utf-8")
            original_open = Path.open
            created_calls = 0

            def flaky_open(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
                nonlocal created_calls
                mode = str(args[0]) if args else str(kwargs.get("mode", "r"))
                if path.parent == units_directory and "x" in mode:
                    created_calls += 1
                    if created_calls == 2:
                        raise OSError("synthetic render failure")
                return original_open(path, *args, **kwargs)

            with (
                patch.object(Path, "open", flaky_open),
                self.assertRaisesRegex(OSError, "synthetic render failure"),
            ):
                render_systemd_user_units(
                    RuntimeDeploymentProfile.read(profile_path),
                    units_directory,
                )
            self.assertEqual("owner data", existing.read_text(encoding="utf-8"))
            self.assertEqual([existing], list(units_directory.iterdir()))

    @unittest.skipUnless(
        os.name == "posix" and shutil.which("systemd-analyze") is not None,
        "systemd-analyze is unavailable",
    )
    def test_rendered_units_pass_systemd_analyze_verify(self) -> None:
        for deferred_reports in (False, True):
            with self.subTest(deferred_reports=deferred_reports):
                self._verify_rendered_units(deferred_reports=deferred_reports)

    def _verify_rendered_units(self, *, deferred_reports: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "unicode 路径"
            root.mkdir()
            profile_path, units_directory, _payload = self._fixture(
                root, deferred_reports=deferred_reports
            )
            rendered = render_systemd_user_units(
                RuntimeDeploymentProfile.read(profile_path),
                units_directory,
            )
            completed = subprocess.run(
                [
                    str(shutil.which("systemd-analyze")),
                    "verify",
                    *(str(units_directory / name) for name in rendered.unit_files),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
                timeout=15,
            )
            self.assertEqual(
                0,
                completed.returncode,
                "generated unit syntax failed",
            )

    def test_profile_rejects_credentials_non_loopback_query_and_extra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, _units_directory, payload = self._fixture(root)
            # release-audit: allow=fixture-credential
            cases = (
                "rtsp://user:secret@127.0.0.1:8554/camera",
                "rtsp://192.0.2.10:8554/camera",
                "rtsp://127.0.0.1:8554/camera?private=value",
            )
            for index, uri in enumerate(cases):
                with self.subTest(uri_index=index):
                    changed = json.loads(json.dumps(payload))
                    changed["cameras"][0]["rtsp_uri"] = uri
                    profile_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(DeploymentProfileError):
                        RuntimeDeploymentProfile.read(profile_path)
            changed = json.loads(json.dumps(payload))
            changed["unexpected"] = True
            profile_path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(DeploymentProfileError, "fields"):
                RuntimeDeploymentProfile.read(profile_path)

    def test_cli_summary_is_path_free_and_report_supports_previous_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path, units_directory, _payload = self._fixture(root)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "systemd-render",
                        "--profile",
                        str(profile_path),
                        "--output-dir",
                        str(units_directory),
                    ]
                )
            self.assertEqual(0, code)
            summary = json.loads(output.getvalue())
            self.assertEqual(9, summary["unit_count"])
            self.assertNotIn(str(root), output.getvalue())

            report_output = io.StringIO()
            expected = datetime.now(ZoneInfo("Asia/Shanghai")).date() - timedelta(days=1)
            with redirect_stdout(report_output):
                report_code = main(
                    [
                        "report",
                        "--database",
                        str(root / "state" / "report.sqlite3"),
                        "--output-root",
                        str(root / "reports"),
                        "--day-offset",
                        "-1",
                    ]
                )
            self.assertEqual(0, report_code)
            self.assertEqual(expected.isoformat(), json.loads(report_output.getvalue())["date"])

    def test_public_schema_required_fields_match_profile_contract(self) -> None:
        schema = json.loads(
            Path("schemas/runtime-deployment-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {"type": "boolean", "default": False}, schema["properties"]["deferred_reports"]
        )
        self.assertEqual(
            {
                "version",
                "deployment_id",
                "project_root",
                "config_path",
                "database_path",
                "output_root",
                "buffer_run_seconds",
                "confirm_ephemeral_buffer_eviction",
                "cameras",
            },
            set(schema["required"]),
        )


if __name__ == "__main__":
    unittest.main()
