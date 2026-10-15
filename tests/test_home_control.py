from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from baby_monitor.analysis_queue import AnalysisQueue
from baby_monitor.candidate_health import (
    CandidateHealthLifetime,
    CandidateHealthState,
    CandidateSamplingHealth,
    CandidateWorkerHealth,
)
from baby_monitor.cli import main
from baby_monitor.database import EventDatabase
from baby_monitor.home_control import (
    CommandResult,
    HomeControlError,
    control_home_stack,
    home_stack_status,
)
from baby_monitor.home_daily_report import generate_home_daily_report
from baby_monitor.home_dashboard import render_home_dashboard
from baby_monitor.notification import NotificationOutbox
from baby_monitor.signal_outbox import SignalOutbox
from baby_monitor.visual_predicate_shadow_queue import (
    VisualPredicateShadowAttemptQueue,
)


class FakeSystemctl:
    def __init__(self) -> None:
        self.active = {
            "xiaomi-stream-gateway.service": False,
            "baby-monitor-ha-signals.service": False,
            "baby-monitor-home_shadow-camera_01-buffer.service": False,
            "baby-monitor-home_shadow-camera_02-buffer.service": False,
            "baby-monitor-home_shadow-camera_03-buffer.service": False,
            "baby-monitor-home_shadow-camera_04-buffer.service": False,
            "baby-monitor-ha-signal-dispatch.service": False,
            "baby-monitor-home_shadow.target": False,
            "baby-monitor-candidate-soak.target": False,
            "baby-monitor-mage-shadow-once.service": False,
            "baby-monitor-dashboard.service": False,
        }
        self.enabled = {
            "xiaomi-stream-gateway.service": False,
            "baby-monitor-ha-signals.service": False,
            "baby-monitor-home_shadow-camera_01-buffer.service": False,
            "baby-monitor-home_shadow-camera_02-buffer.service": False,
            "baby-monitor-home_shadow-camera_03-buffer.service": False,
            "baby-monitor-home_shadow-camera_04-buffer.service": False,
            "baby-monitor-ha-signal-dispatch.service": False,
            "baby-monitor-home_shadow.target": False,
            "baby-monitor-candidate-soak.target": False,
            "baby-monitor-mage-shadow-once.service": False,
            "baby-monitor-dashboard.service": False,
        }
        self.calls: list[tuple[str, ...]] = []

    def activate_base(self) -> None:
        for unit in (
            "xiaomi-stream-gateway.service",
            "baby-monitor-ha-signals.service",
            "baby-monitor-home_shadow-camera_01-buffer.service",
            "baby-monitor-home_shadow-camera_02-buffer.service",
            "baby-monitor-home_shadow-camera_03-buffer.service",
            "baby-monitor-home_shadow-camera_04-buffer.service",
            "baby-monitor-ha-signal-dispatch.service",
        ):
            self.active[unit] = True

    def __call__(self, argv: tuple[str, ...], timeout_seconds: float) -> CommandResult:
        del timeout_seconds
        self.calls.append(argv)
        if "show" in argv:
            unit = argv[argv.index("show") + 1]
            active = self.active[unit]
            substate = "active" if unit.endswith(".target") and active else "running"
            if not active:
                substate = "dead"
            unit_file_state = "enabled" if self.enabled[unit] else "disabled"
            if unit == "baby-monitor-mage-shadow-once.service":
                unit_file_state = "static"
            return CommandResult(
                0,
                "\n".join(
                    (
                        f"Id={unit}",
                        "LoadState=loaded",
                        f"ActiveState={'active' if active else 'inactive'}",
                        f"SubState={substate}",
                        f"UnitFileState={unit_file_state}",
                    )
                )
                + "\n",
                "",
            )

        verb = "start" if "start" in argv else "stop"
        start_index = argv.index(verb) + 1
        for unit in argv[start_index:]:
            self.active[unit] = verb == "start"
            if verb == "start" and unit == "baby-monitor-candidate-soak-stop.service":
                self.active["baby-monitor-candidate-soak.target"] = False
        return CommandResult(0, "", "")


class HomeControlTests(unittest.TestCase):
    def test_manual_dispatch_unit_cannot_be_enabled_and_denies_ip_network(self) -> None:
        unit = (
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "manual"
            / "baby-monitor-ha-signal-dispatch.service"
        ).read_text(encoding="utf-8")

        self.assertNotIn("\n[Install]\n", unit)
        self.assertIn("IPAddressDeny=any", unit)
        self.assertIn("RestrictAddressFamilies=AF_UNIX", unit)
        self.assertIn("ha-signal-dispatch-worker", unit)
        self.assertNotIn("visual-predicate", unit)
        self.assertNotIn("analysis-worker", unit)

        dashboard_unit = (
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "manual"
            / "baby-monitor-dashboard.service"
        ).read_text(encoding="utf-8")
        self.assertNotIn("\n[Install]\n", dashboard_unit)
        self.assertIn("IPAddressDeny=any", dashboard_unit)
        self.assertIn("IPAddressAllow=localhost", dashboard_unit)
        self.assertIn("home-dashboard", dashboard_unit)
        self.assertIn(
            "--config %h/projects/baby-monitor/config/home-shadow.local.toml", dashboard_unit
        )
        self.assertIn("--analysis-database", dashboard_unit)
        self.assertEqual(4, dashboard_unit.count("--candidate-health "))
        self.assertEqual(1, dashboard_unit.count("--room-registry "))
        self.assertEqual(4, dashboard_unit.count("--room-presence-samples "))
        self.assertEqual(8, dashboard_unit.count("--room-presence-health "))
        self.assertIn("stream-presence-batch.json", dashboard_unit)
        self.assertEqual(1, dashboard_unit.count("ReadWritePaths="))
        self.assertIn(
            "ReadWritePaths=%h/projects/baby-monitor/data/shadow/state",
            dashboard_unit,
        )
        self.assertNotIn(
            "ReadWritePaths=%h/projects/baby-monitor/data/shadow/spool",
            dashboard_unit,
        )

        mage_unit = (
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "manual"
            / "baby-monitor-mage-shadow-once.service"
        ).read_text(encoding="utf-8")
        self.assertNotIn("\n[Install]\n", mage_unit)
        self.assertIn("IPAddressDeny=any", mage_unit)
        self.assertIn("RestrictAddressFamilies=AF_UNIX", mage_unit)
        self.assertIn("visual-predicate-shadow-worker", mage_unit)
        self.assertIn("--max-jobs 4", mage_unit)
        self.assertIn("--max-runtime-seconds 600", mage_unit)
        self.assertNotIn("--frame", mage_unit)
        self.assertNotIn("visual-predicate-shadow-run", mage_unit)
        self.assertNotIn("analysis-worker", mage_unit)

        soak_target = (
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "manual"
            / "baby-monitor-candidate-soak.target"
        ).read_text(encoding="utf-8")
        soak_timer = (
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "manual"
            / "baby-monitor-candidate-soak-stop.timer"
        ).read_text(encoding="utf-8")
        soak_baseline = (
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "manual"
            / "baby-monitor-candidate-soak-baseline.service"
        ).read_text(encoding="utf-8")
        soak_stop = (
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "manual"
            / "baby-monitor-candidate-soak-stop.service"
        ).read_text(encoding="utf-8")
        soak_worker = (
            Path(__file__).resolve().parents[1]
            / "deploy"
            / "manual"
            / "baby-monitor-candidate-soak@.service"
        ).read_text(encoding="utf-8")
        self.assertNotIn("\n[Install]\n", soak_target)
        self.assertNotIn("\n[Install]\n", soak_timer)
        self.assertIn("baby-monitor-candidate-soak-baseline.service", soak_target)
        self.assertIn("baby-monitor-candidate-soak-stop.timer", soak_target)
        self.assertEqual(4, soak_target.count("baby-monitor-candidate-soak@camera_"))
        self.assertEqual(4, soak_target.count("-buffer.service"))
        self.assertNotIn("Wants=baby-monitor-home_shadow-analysis.service", soak_target)
        self.assertNotIn("Wants=baby-monitor-home_shadow-notification.service", soak_target)
        self.assertNotIn("ha-signals.service", soak_target)
        self.assertNotIn("mage-shadow", soak_target.split("Conflicts=")[0])
        self.assertNotIn("Conflicts=baby-monitor-home_shadow.target", soak_target)
        self.assertIn("Conflicts=baby-monitor-home_shadow-analysis.service", soak_target)
        self.assertIn("baby-monitor-home_shadow-notification.service", soak_target)
        self.assertIn("OnActiveSec=86400s", soak_timer)
        self.assertNotIn("PartOf=baby-monitor-candidate-soak.target", soak_timer)
        self.assertIn("candidate-soak-start", soak_baseline)
        self.assertEqual(4, soak_baseline.count("candidate-presence-health-init"))
        self.assertIn("--expected-duration-seconds 86400", soak_baseline)
        self.assertEqual(8, soak_baseline.count("--health "))
        self.assertIn("IPAddressDeny=any", soak_baseline)
        self.assertNotIn("\n[Install]\n", soak_baseline)
        self.assertNotIn("\n[Install]\n", soak_worker)
        self.assertIn("--presence-only", soak_worker)
        self.assertIn("--presence ", soak_worker)
        self.assertIn("Environment=BABY_MONITOR_YOLO_IMAGE_SIZE=640", soak_worker)
        self.assertIn("Environment=BABY_MONITOR_ROTATION_DEGREES=0", soak_worker)
        self.assertIn(
            "EnvironmentFile=-%h/.config/roomsentry/streams/%i.env",
            soak_worker,
        )
        self.assertIn("--rotation-degrees ${BABY_MONITOR_ROTATION_DEGREES}", soak_worker)
        self.assertIn("IPAddressDeny=any", soak_worker)
        self.assertIn("RestrictAddressFamilies=AF_UNIX", soak_worker)
        self.assertNotIn("data/shadow/spool", soak_worker)
        self.assertNotIn("analysis-worker", soak_worker)
        self.assertIn("IPAddressDeny=any", soak_stop)
        self.assertIn("ExecStart=-%h/projects/baby-monitor/.venv/bin/python", soak_stop)
        self.assertIn("candidate-soak-finish", soak_stop)
        self.assertIn("baby-monitor-candidate-soak.target", soak_stop)
        self.assertIn(
            "ExecStart=/usr/bin/systemctl --user stop",
            soak_stop,
        )
        self.assertIn("baby-monitor-home_shadow-camera_01-buffer.service", soak_stop)
        self.assertIn("xiaomi-stream-gateway.service", soak_stop)
        self.assertNotIn("Conflicts=baby-monitor-candidate-soak.target", soak_stop)
        self.assertEqual(8, soak_stop.count("baby-monitor-candidate-soak@camera_"))

        launcher = (
            Path(__file__).resolve().parents[1] / "scripts" / "BabyMonitor-Home.cmd"
        ).read_text(encoding="utf-8")
        self.assertIn("-Action dashboard-start", launcher)
        self.assertNotIn("-Action start", launcher)

    def test_status_reports_base_and_experimental_components_separately(self) -> None:
        runner = FakeSystemctl()
        runner.activate_base()

        result = home_stack_status(runner=runner)

        self.assertTrue(result["base_operational"])
        self.assertFalse(result["base_autostart_enabled"])
        self.assertTrue(result["manual_start_only"])
        self.assertFalse(result["shadow_active"])
        self.assertFalse(result["int8_shadow_active"])
        self.assertEqual("available", result["dashboard"])
        self.assertEqual("not_configured", result["daily_digest"]["status"])
        self.assertFalse(result["daily_digest"]["video_available"])
        self.assertEqual(result["ha_signal_outbox"]["status"], "not_configured")
        self.assertEqual(result["analysis_queue"]["status"], "not_configured")
        self.assertEqual(result["candidate_health"]["status"], "not_configured")
        self.assertEqual(result["candidate_soak"]["status"], "not_configured")
        self.assertFalse(result["paths_disclosed"])
        self.assertFalse(result["secrets_disclosed"])

    def test_candidate_health_is_private_read_only_and_manual_stop_is_not_failure(self) -> None:
        runner = FakeSystemctl()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            now = datetime.now(UTC)
            paths: list[Path] = []
            for camera_id in ("camera_01", "camera_02"):
                path = root / f"{camera_id}.json"
                CandidateWorkerHealth(
                    camera_id=camera_id,
                    scanner_identity="candidate_v1_fixture",
                    detector_identity="yolo_command_v1_fixture",
                    checked_at=now,
                    state=CandidateHealthState.IDLE,
                    worker_running=True,
                ).write_atomic_private(path)
                paths.append(path)

            active = home_stack_status(
                runner=runner,
                candidate_health_paths=paths,
            )
            candidate = active["candidate_health"]
            self.assertEqual("available", candidate["status"])
            self.assertTrue(candidate["operational"])
            self.assertEqual(2, len(candidate["cameras"]))
            self.assertFalse(candidate["media_read"])
            self.assertNotIn(str(root), json.dumps(active))

            CandidateWorkerHealth(
                camera_id="camera_02",
                scanner_identity="candidate_v1_fixture",
                detector_identity="yolo_command_v1_fixture",
                checked_at=datetime.now(UTC),
                state=CandidateHealthState.STOPPED,
                worker_running=False,
            ).write_atomic_private(paths[1])
            stopped = home_stack_status(
                runner=runner,
                candidate_health_paths=paths,
            )["candidate_health"]
            self.assertFalse(stopped["operational"])
            self.assertFalse(stopped["attention_required"])

            partial = home_stack_status(
                runner=runner,
                candidate_health_paths=(*paths, root / "missing.json"),
            )["candidate_health"]
            self.assertEqual("unavailable", partial["status"])
            self.assertTrue(partial["attention_required"])

    def test_candidate_sampling_evidence_survives_read_only_home_status(self) -> None:
        runner = FakeSystemctl()
        runner.activate_base()
        now = datetime.now(UTC)
        sampling = CandidateSamplingHealth(
            sampled_frame_count=30,
            zero_detection_sample_count=30,
            decoded_frame_count=1202,
            timestamp_fallback_frame_count=0,
            max_adjacent_container_pts_gap_seconds=3.0539,
            container_pts_regression_count=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            path = root / "candidate-health.json"
            CandidateWorkerHealth(
                version=4,
                camera_id="camera_01",
                scanner_identity="candidate_v1_fixture",
                detector_identity="yolo_command_v1_fixture",
                checked_at=now,
                state=CandidateHealthState.IDLE,
                worker_running=True,
                scans_started=1,
                scans_succeeded=1,
                segments_scanned=1,
                last_scan_started_at=now,
                last_scan_completed_at=now,
                last_scan_duration_seconds=1,
                lifetime=CandidateHealthLifetime.begin_session(None, updated_at=now).record_scan(
                    updated_at=now,
                    succeeded=True,
                    duration_seconds=1,
                    execution_lock_wait_seconds=0,
                ),
                last_sampling=sampling,
            ).write_atomic_private(path)
            before = path.read_bytes(), path.stat().st_mtime_ns
            result = home_stack_status(runner=runner, candidate_health_paths=[path])
            candidate = result["candidate_health"]
            self.assertEqual(candidate["cameras"][0]["health"]["last_sampling"], sampling.to_dict())
            self.assertTrue(candidate["operational"])
            self.assertFalse(candidate["attention_required"])
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
            self.assertEqual([path], list(root.iterdir()))
            self.assertFalse(candidate["media_read"])
            self.assertFalse(candidate["notification_authority"])
            self.assertNotIn(str(root), json.dumps(result))
            rendered = render_home_dashboard(result).decode("utf-8")
            self.assertIn("实际模型调用 30 次 · 零框 30 次 · 解码 1202 帧", rendered)
            self.assertIn("最大相邻源帧 PTS 间隔：3.054 秒", rendered)
            self.assertIn("历史记录，非实时", rendered)
            self.assertNotIn(str(root), rendered)
            self.assertNotIn("camera_01", rendered)

    def test_private_ha_outbox_health_is_aggregated_without_its_path(self) -> None:
        runner = FakeSystemctl()
        runner.activate_base()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "signals.sqlite3"
            SignalOutbox(database).initialize()

            result = home_stack_status(
                runner=runner,
                ha_outbox_database=database,
            )

            self.assertTrue(result["base_operational"])
            self.assertEqual("available", result["ha_signal_outbox"]["status"])
            self.assertEqual(0, result["ha_signal_outbox"]["active"])
            self.assertFalse(result["ha_signal_outbox"]["payloads_disclosed"])
            self.assertNotIn(str(database), json.dumps(result))

    def test_missing_configured_ha_outbox_degrades_base_without_creating_it(self) -> None:
        runner = FakeSystemctl()
        runner.activate_base()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "missing.sqlite3"
            result = home_stack_status(
                runner=runner,
                ha_outbox_database=database,
            )

            self.assertFalse(result["base_operational"])
            self.assertEqual("unavailable", result["ha_signal_outbox"]["status"])
            self.assertTrue(result["ha_signal_outbox"]["attention_required"])
            self.assertFalse(database.exists())

    def test_analysis_queue_health_is_read_only_path_free_and_operational(self) -> None:
        runner = FakeSystemctl()
        runner.activate_base()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            AnalysisQueue(database).initialize()

            result = home_stack_status(runner=runner, analysis_database=database)

            self.assertTrue(result["base_operational"])
            self.assertEqual("available", result["analysis_queue"]["status"])
            self.assertEqual(0, result["analysis_queue"]["completed"])
            self.assertFalse(result["analysis_queue"]["media_read"])
            self.assertEqual(
                "not_initialized",
                result["visual_predicate_shadow"]["status"],
            )
            self.assertFalse(result["visual_predicate_shadow"]["media_read"])
            self.assertNotIn(str(database), json.dumps(result))

    def test_initialized_shadow_status_is_read_only_and_path_free(self) -> None:
        runner = FakeSystemctl()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            AnalysisQueue(database).initialize()
            VisualPredicateShadowAttemptQueue(database).initialize()

            result = home_stack_status(runner=runner, analysis_database=database)

            shadow = result["visual_predicate_shadow"]
            self.assertEqual("available", shadow["status"])
            self.assertEqual(0, shadow["completed"])
            self.assertEqual([], shadow["recent_answers"])
            self.assertEqual("atomic_visual_review_v1", shadow["review_policy_version"])
            self.assertEqual(0, shadow["recent_assessed_count"])
            self.assertEqual(0, shadow["recent_review_recommended"])
            self.assertEqual(0, shadow["recent_unknown"])
            self.assertEqual(0, shadow["recent_no_rule_match"])
            self.assertFalse(shadow["risk_inference"])
            self.assertFalse(shadow["notification_decision"])
            self.assertFalse(shadow["media_read"])
            self.assertNotIn(str(database), json.dumps(result))

    def test_notification_queue_status_is_read_only_path_free_and_external_off(self) -> None:
        runner = FakeSystemctl()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "events.sqlite3"
            AnalysisQueue(database).initialize()
            NotificationOutbox(database).enqueue_local_test(
                test_id="b" * 32,
                sink_kind="file",
                sink_id="dashboard_test",
                max_active=4,
            )

            result = home_stack_status(
                runner=runner,
                analysis_database=database,
                notification_enabled=True,
                external_notification_delivery_enabled=False,
            )

            notification = result["notification_queue"]
            self.assertEqual("available", notification["status"])
            self.assertEqual(1, notification["pending"])
            self.assertTrue(notification["enabled"])
            self.assertTrue(notification["local_test_available"])
            self.assertFalse(notification["external_delivery_enabled"])
            self.assertFalse(notification["media_included"])
            self.assertFalse(notification["paths_disclosed"])
            self.assertNotIn(str(database), json.dumps(result))

    def test_daily_report_status_is_private_and_does_not_generate_on_read(self) -> None:
        runner = FakeSystemctl()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            database = root / "events.sqlite3"
            EventDatabase(database).initialize()
            database.chmod(0o600)
            output = root / "daily-reports"

            missing = home_stack_status(
                runner=runner,
                analysis_database=database,
                daily_report_output=output,
            )
            self.assertEqual("not_generated", missing["daily_report"]["status"])
            self.assertFalse(output.exists())

            generate_home_daily_report(
                database_path=database,
                output_root=output,
                now=datetime.now(UTC),
            )
            available = home_stack_status(
                runner=runner,
                analysis_database=database,
                daily_report_output=output,
            )
            self.assertEqual("available", available["daily_report"]["status"])
            self.assertTrue(available["daily_report"]["available"])
            self.assertFalse(available["daily_report"]["media_read"])
            self.assertNotIn(str(root), json.dumps(available))

    def test_missing_analysis_queue_degrades_without_creating_it(self) -> None:
        runner = FakeSystemctl()
        runner.activate_base()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "missing.sqlite3"

            result = home_stack_status(runner=runner, analysis_database=database)

            self.assertFalse(result["base_operational"])
            self.assertEqual("unavailable", result["analysis_queue"]["status"])
            self.assertFalse(database.exists())

    def test_start_only_starts_the_fixed_base_units(self) -> None:
        runner = FakeSystemctl()

        result = control_home_stack(
            "start",
            runner=runner,
            confirm_buffer_eviction=True,
        )

        action_call = runner.calls[0]
        self.assertIn("xiaomi-stream-gateway.service", action_call)
        self.assertIn("baby-monitor-ha-signals.service", action_call)
        self.assertIn("baby-monitor-home_shadow-camera_01-buffer.service", action_call)
        self.assertIn("baby-monitor-home_shadow-camera_02-buffer.service", action_call)
        self.assertIn("baby-monitor-home_shadow-camera_03-buffer.service", action_call)
        self.assertIn("baby-monitor-home_shadow-camera_04-buffer.service", action_call)
        self.assertIn("baby-monitor-ha-signal-dispatch.service", action_call)
        self.assertNotIn("baby-monitor-home_shadow.target", action_call)
        self.assertNotIn("enable", action_call)
        self.assertTrue(result["base_operational"])
        self.assertFalse(result["persistent_enablement_changed"])

    def test_shadow_start_starts_base_then_bounded_target(self) -> None:
        runner = FakeSystemctl()

        result = control_home_stack(
            "shadow-start",
            runner=runner,
            confirm_buffer_eviction=True,
        )

        action_calls = [call for call in runner.calls if "start" in call]
        self.assertEqual(len(action_calls), 3)
        self.assertIn("baby-monitor-candidate-soak-stop.service", action_calls[0])
        self.assertIn("baby-monitor-home_shadow.target", action_calls[2])
        self.assertTrue(result["base_operational"])
        self.assertTrue(result["shadow_active"])

    def test_int8_once_is_explicit_and_non_blocking(self) -> None:
        runner = FakeSystemctl()

        result = control_home_stack(
            "int8-shadow-once",
            runner=runner,
            confirm_buffer_eviction=True,
        )

        action_calls = [call for call in runner.calls if "start" in call]
        self.assertEqual(len(action_calls), 3)
        self.assertIn("baby-monitor-candidate-soak-stop.service", action_calls[0])
        self.assertIn("--no-block", action_calls[2])
        self.assertIn("baby-monitor-mage-shadow-once.service", action_calls[2])
        self.assertTrue(result["int8_shadow_active"])

    def test_candidate_soak_starts_only_dedicated_target_and_stops_via_fixed_unit(self) -> None:
        runner = FakeSystemctl()
        for unit in runner.active:
            runner.active[unit] = True

        started = control_home_stack(
            "soak-start",
            runner=runner,
            confirm_buffer_eviction=True,
        )

        action_calls = [call for call in runner.calls if "start" in call]
        self.assertEqual(1, len(action_calls))
        self.assertIn("baby-monitor-candidate-soak.target", action_calls[0])
        self.assertNotIn("baby-monitor-home_shadow.target", action_calls[0])
        self.assertNotIn("baby-monitor-mage-shadow-once.service", action_calls[0])
        self.assertTrue(started["candidate_soak_active"])
        self.assertFalse(started["shadow_active"])
        self.assertFalse(started["base_operational"])
        stop_call = next(call for call in runner.calls if "stop" in call)
        self.assertIn("baby-monitor-ha-signals.service", stop_call)
        self.assertIn("baby-monitor-ha-signal-dispatch.service", stop_call)

        runner.calls.clear()
        stopped = control_home_stack("soak-stop", runner=runner)
        action_calls = [call for call in runner.calls if "start" in call]
        self.assertEqual(1, len(action_calls))
        self.assertIn("baby-monitor-candidate-soak-stop.service", action_calls[0])
        self.assertFalse(stopped["candidate_soak_active"])

    def test_stop_stops_all_fixed_units_without_disabling_autostart(self) -> None:
        runner = FakeSystemctl()
        for unit in runner.active:
            runner.active[unit] = True

        result = control_home_stack("stop", runner=runner)

        self.assertIn("baby-monitor-candidate-soak-stop.service", runner.calls[0])
        action_call = next(call for call in runner.calls if "stop" in call)
        self.assertIn("stop", action_call)
        self.assertNotIn("disable", action_call)
        self.assertFalse(result["base_operational"])
        self.assertFalse(result["shadow_active"])
        self.assertFalse(result["base_autostart_enabled"])
        self.assertTrue(result["manual_start_only"])

    def test_dashboard_start_is_manual_and_does_not_start_monitoring(self) -> None:
        runner = FakeSystemctl()

        result = control_home_stack("dashboard-start", runner=runner)

        action_calls = [call for call in runner.calls if "start" in call]
        self.assertEqual(1, len(action_calls))
        self.assertIn("baby-monitor-dashboard.service", action_calls[0])
        self.assertNotIn("xiaomi-stream-gateway.service", action_calls[0])
        self.assertFalse(result["base_operational"])
        dashboard = next(
            item for item in result["components"] if item["component"] == "local_dashboard"
        )
        self.assertTrue(dashboard["active"])
        self.assertFalse(dashboard["enabled"])

    def test_manual_buffer_start_requires_one_shot_confirmation(self) -> None:
        runner = FakeSystemctl()
        for action in ("start", "shadow-start", "soak-start", "int8-shadow-once"):
            with (
                self.subTest(action=action),
                self.assertRaisesRegex(HomeControlError, "one-shot confirmation"),
            ):
                control_home_stack(action, runner=runner)
        self.assertEqual([], runner.calls)

    def test_malformed_status_is_redacted_as_unavailable(self) -> None:
        def malformed(_argv: tuple[str, ...], _timeout: float) -> CommandResult:
            return CommandResult(0, "Id=wrong.service\nSECRET=do-not-copy\n", "")

        result = home_stack_status(runner=malformed)

        self.assertFalse(result["base_operational"])
        self.assertTrue(all(item["state"] == "unavailable" for item in result["components"]))
        self.assertNotIn("do-not-copy", json.dumps(result))

    def test_action_failure_never_copies_stderr(self) -> None:
        def failed(_argv: tuple[str, ...], _timeout: float) -> CommandResult:
            return CommandResult(1, "", "private argument must not escape")

        with self.assertRaisesRegex(HomeControlError, "home control action failed") as raised:
            control_home_stack(
                "start",
                runner=failed,
                confirm_buffer_eviction=True,
            )
        self.assertNotIn("private argument", str(raised.exception))

    def test_invalid_action_fails_closed(self) -> None:
        with self.assertRaisesRegex(HomeControlError, "action is invalid"):
            control_home_stack("arbitrary-unit-start", runner=FakeSystemctl())

    def test_cli_prints_path_free_status(self) -> None:
        public_status = {
            "version": 1,
            "action": "status",
            "action_accepted": True,
            "paths_disclosed": False,
            "secrets_disclosed": False,
        }
        output = io.StringIO()
        with (
            mock.patch(
                "baby_monitor.cli_commands.home_dashboard.control_home_stack",
                return_value=public_status,
            ),
            redirect_stdout(output),
        ):
            exit_code = main(["home-control", "status"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue()), public_status)


if __name__ == "__main__":
    unittest.main()
