from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baby_monitor.config import ConfigurationError, load_settings


class ConfigTests(unittest.TestCase):
    def test_defaults_keep_remote_and_deletion_disabled(self) -> None:
        settings = load_settings("config/default.toml")
        self.assertFalse(settings.remote_vlm.uploads_allowed)
        self.assertFalse(settings.notification.enabled)
        self.assertFalse(settings.notification.external_delivery_enabled)
        self.assertFalse(settings.notification.auto_file_enabled)
        self.assertEqual("shadow_local", settings.notification.auto_file_sink_id)
        self.assertIsNone(settings.notification.event_policy_path)
        self.assertEqual(64, settings.notification.max_pending)
        self.assertEqual(3600, settings.notification.max_pending_age_seconds)
        self.assertEqual(300, settings.notification.lease_seconds)
        self.assertEqual(30, settings.notification.retry_delay_seconds)
        self.assertEqual(5, settings.notification.max_attempts)
        self.assertEqual(3, settings.remote_vlm.max_frames)
        self.assertEqual(5_242_880, settings.remote_vlm.max_upload_bytes)
        self.assertEqual(262_144, settings.remote_vlm.max_response_bytes)
        self.assertEqual(20, settings.remote_vlm.max_requests_per_day)
        self.assertEqual(60, settings.remote_vlm.min_interval_seconds)
        self.assertFalse(settings.storage.retention_scan_enabled)
        self.assertFalse(settings.storage.retention_delete_enabled)
        self.assertFalse(settings.buffer.eviction_enabled)
        self.assertIsNone(settings.detector.yolo_model_path)
        self.assertIsNone(settings.detector.yolo_manifest_path)
        self.assertEqual((), settings.detector.yolo_runner_command)
        self.assertEqual(600.0, settings.detector.yolo_runner_timeout_seconds)
        self.assertIsNone(settings.detector.yolo_execution_lock_path)
        self.assertEqual(
            600.0,
            settings.detector.yolo_execution_lock_timeout_seconds,
        )
        self.assertEqual(640, settings.detector.image_size)
        self.assertEqual(0, settings.detector.rotation_degrees)
        self.assertEqual(("person", "cat", "dog"), settings.detector.labels)
        self.assertEqual(10.0, settings.media_health.stale_after_seconds)
        self.assertEqual(5.0, settings.media_health.future_tolerance_seconds)
        self.assertEqual(32, settings.candidate_worker.max_segments_per_run)
        self.assertEqual(30.0, settings.candidate_worker.idle_exit_seconds)
        self.assertEqual(3600.0, settings.candidate_worker.max_runtime_seconds)
        self.assertEqual(30.0, settings.candidate_worker.max_event_span_seconds)
        self.assertEqual(0.35, settings.candidate_worker.candidate_start_confidence)
        self.assertEqual(1000, settings.candidate_worker.max_pending_detections)
        self.assertFalse(settings.analysis_worker.gpu_guard_enabled)
        self.assertEqual(32, settings.analysis_worker.max_jobs_per_run)
        self.assertEqual(30.0, settings.analysis_worker.idle_exit_seconds)
        self.assertEqual(3600.0, settings.analysis_worker.max_runtime_seconds)
        self.assertEqual(64, settings.archive_queue.max_pending)
        self.assertEqual(3600, settings.archive_queue.lease_seconds)
        self.assertEqual(32, settings.archive_worker.max_jobs_per_run)
        self.assertEqual(30.0, settings.archive_worker.idle_exit_seconds)
        self.assertEqual(3600.0, settings.archive_worker.max_runtime_seconds)
        self.assertGreaterEqual(settings.media.keyframe_count, 3)
        self.assertLessEqual(settings.media.keyframe_count, 8)
        self.assertEqual("mock", settings.local_vlm.backend)
        self.assertEqual("", settings.local_vlm.revision)
        self.assertEqual((), settings.local_vlm.command)
        self.assertEqual("0", settings.local_vlm.gpu_device)
        self.assertEqual(600, settings.review_policy.signal_frequency_window_seconds)
        self.assertEqual(3, settings.review_policy.signal_frequency_secondary_threshold)
        self.assertEqual(("fall_hint",), settings.review_policy.priority_signal_types)
        self.assertFalse(settings.secondary_vlm.enabled)
        self.assertEqual("", settings.secondary_vlm.revision)
        self.assertEqual((), settings.secondary_vlm.command)
        self.assertEqual("0", settings.secondary_vlm.gpu_device)

    def test_vlm_gpu_devices_must_be_numeric_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                '[local_vlm]\ngpu_device = "cuda:0"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "local VLM GPU device"):
                load_settings(config)

    def test_remote_requires_both_flags(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BABY_MONITOR_REMOTE_ENABLED": "true",
                "BABY_MONITOR_REMOTE_UPLOAD_CONSENT": "false",
            },
            clear=False,
        ):
            settings = load_settings("config/default.toml")
        self.assertFalse(settings.remote_vlm.uploads_allowed)

    def test_remote_frame_and_request_budgets_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                "[remote_vlm]\nmax_frames = 4\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "at most 3 frames"):
                load_settings(config)

    def test_notification_backlog_age_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                "[notification]\nmax_pending_age_seconds = 604801\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "must be <= 604800"):
                load_settings(config)

    def test_notification_auto_file_requires_enabled_notifications(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                "[notification]\nauto_file_enabled = true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "requires.*enabled"):
                load_settings(config)

    def test_notification_auto_file_sink_id_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                '[notification]\nauto_file_sink_id = "Family Room"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "sink id"):
                load_settings(config)

    def test_notification_auto_file_requires_an_event_policy_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                "[notification]\nenabled = true\nauto_file_enabled = true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "event_policy_path"):
                load_settings(config)

    def test_invalid_keyframe_count_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text("[media]\nkeyframe_count = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "between 3 and 8"):
                load_settings(config)

    def test_invalid_media_health_threshold_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                "[media_health]\nstale_after_seconds = 604801\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "stale threshold"):
                load_settings(config)

    def test_invalid_scene_threshold_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text("[detector]\nscene_threshold = 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "scene threshold"):
                load_settings(config)

    def test_yolo_sampling_device_and_labels_are_bounded(self) -> None:
        cases = (
            ("sample_fps = 10.01", "sample FPS"),
            ("image_size = 641", "image size"),
            ("rotation_degrees = 45", "rotation degrees"),
            ('device = "0,1"', "YOLO device"),
            ("labels = []", "non-empty and unique"),
            ('labels = ["person", "person"]', "non-empty and unique"),
        )
        for body, message in cases:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "bad.toml"
                config.write_text(f"[detector]\n{body}\n", encoding="utf-8")
                with self.assertRaisesRegex(ConfigurationError, message):
                    load_settings(config)

    def test_yolo_command_requires_runner_and_loads_bounded_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_command = root / "missing-command.toml"
            missing_command.write_text(
                '[detector]\nbackend = "yolo_command"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "runner_command"):
                load_settings(missing_command)

            command_config = root / "command.toml"
            command_config.write_text(
                "\n".join(
                    [
                        "[detector]",
                        'backend = "yolo_command"',
                        'yolo_runner_command = ["/usr/bin/python3", "-m", "runner", "{request}"]',
                        "yolo_runner_timeout_seconds = 12",
                        f'yolo_execution_lock_path = "{root / "yolo.lock"}"',
                        "yolo_execution_lock_timeout_seconds = 13",
                    ]
                ),
                encoding="utf-8",
            )
            settings = load_settings(command_config)
            self.assertEqual(
                ("/usr/bin/python3", "-m", "runner", "{request}"),
                settings.detector.yolo_runner_command,
            )
            self.assertEqual(12.0, settings.detector.yolo_runner_timeout_seconds)
            self.assertEqual(
                root / "yolo.lock",
                settings.detector.yolo_execution_lock_path,
            )
            self.assertEqual(
                13.0,
                settings.detector.yolo_execution_lock_timeout_seconds,
            )

    def test_yolo_execution_lock_configuration_is_bounded(self) -> None:
        cases = (
            ('yolo_execution_lock_path = "relative.lock"', "path must be absolute"),
            (
                "yolo_execution_lock_timeout_seconds = 3601",
                "lock timeout must be between",
            ),
        )
        for body, message in cases:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "bad.toml"
                config.write_text(f"[detector]\n{body}\n", encoding="utf-8")
                with self.assertRaisesRegex(ConfigurationError, message):
                    load_settings(config)

    def test_candidate_start_confidence_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                "[candidate_worker]\ncandidate_start_confidence = 1.1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "between 0 and 1"):
                load_settings(config)

    def test_candidate_evidence_window_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                "\n".join(
                    [
                        "[media]",
                        "pre_event_seconds = 150",
                        "post_event_seconds = 150",
                        "[candidate_worker]",
                        "max_event_span_seconds = 1",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "evidence window"):
                load_settings(config)

    def test_command_vlm_requires_explicit_local_executable_and_request_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "command.toml"
            executable = Path(sys.executable).as_posix()
            config.write_text(
                "\n".join(
                    [
                        "[local_vlm]",
                        'backend = "command"',
                        'name = "fixture_runner"',
                        f'command = ["{executable}", "-c", "print(1)", "{{request}}"]',
                        "timeout_seconds = 12",
                    ]
                ),
                encoding="utf-8",
            )
            settings = load_settings(config)
        self.assertEqual("command", settings.local_vlm.backend)
        self.assertEqual("fixture_runner", settings.local_vlm.name)
        self.assertEqual(12, settings.local_vlm.timeout_seconds)
        self.assertEqual("{request}", settings.local_vlm.command[-1])

    def test_command_vlm_rejects_missing_request_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            executable = Path(sys.executable).as_posix()
            config.write_text(
                "\n".join(
                    [
                        "[local_vlm]",
                        'backend = "command"',
                        'name = "fixture_runner"',
                        f'command = ["{executable}", "-c", "print(1)"]',
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "exactly once"):
                load_settings(config)

    def test_vlm_revision_must_be_a_path_free_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                '[local_vlm]\nrevision = "/private/model/path"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "path-free revision"):
                load_settings(config)

    def test_unknown_local_vlm_backend_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text('[local_vlm]\nbackend = "mystery"\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "mock or command"):
                load_settings(config)

    def test_analysis_lease_must_outlive_command_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            executable = Path(sys.executable).as_posix()
            config.write_text(
                "\n".join(
                    [
                        "[analysis_queue]",
                        "lease_seconds = 10",
                        "[local_vlm]",
                        'backend = "command"',
                        'name = "fixture_runner"',
                        f'command = ["{executable}", "-c", "print(1)", "{{request}}"]',
                        "timeout_seconds = 10",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "lease must be longer"):
                load_settings(config)

    def test_analysis_worker_requires_a_stopping_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                "\n".join(
                    [
                        "[analysis_worker]",
                        "max_jobs_per_run = 0",
                        "idle_exit_seconds = 0",
                        "max_runtime_seconds = 0",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "stopping bound"):
                load_settings(config)

    def test_archive_worker_requires_a_stopping_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                "\n".join(
                    [
                        "[archive_worker]",
                        "max_jobs_per_run = 0",
                        "idle_exit_seconds = 0",
                        "max_runtime_seconds = 0",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "stopping bound"):
                load_settings(config)

    def test_invalid_review_policy_signal_type_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            config.write_text(
                '[review_policy]\npriority_signal_types = ["secret_vendor_alarm"]\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "priority signal types"):
                load_settings(config)

    def test_zone_dwell_can_be_a_review_policy_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "zone-dwell.toml"
            config.write_text(
                '[review_policy]\npriority_signal_types = ["zone_dwell"]\n',
                encoding="utf-8",
            )
            self.assertEqual(
                ("zone_dwell",),
                load_settings(config).review_policy.priority_signal_types,
            )

    def test_sequential_command_timeouts_must_fit_analysis_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.toml"
            executable = Path(sys.executable).as_posix()
            config.write_text(
                "\n".join(
                    [
                        "[analysis_queue]",
                        "lease_seconds = 30",
                        "[local_vlm]",
                        'backend = "command"',
                        'name = "primary"',
                        f'command = ["{executable}", "{{request}}"]',
                        "timeout_seconds = 20",
                        "[secondary_vlm]",
                        "enabled = true",
                        'backend = "command"',
                        'name = "secondary"',
                        f'command = ["{executable}", "{{request}}"]',
                        "timeout_seconds = 20",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "sequential review"):
                load_settings(config)


if __name__ == "__main__":
    unittest.main()
