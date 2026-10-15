from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from baby_monitor.analysis_queue import AnalysisQueue
from baby_monitor.cli import main
from baby_monitor.database import EventDatabase
from baby_monitor.domain import EventMedia, EvidenceGap
from baby_monitor.event_policy import HouseholdEventPolicy, PolicyEventRule
from baby_monitor.gpu_guard import GPUCapacity
from baby_monitor.notification import NotificationOutbox
from baby_monitor.ring_buffer import (
    BufferedEvidenceResult,
    BufferWindowUnavailableError,
    RingBufferError,
    SegmentManifest,
)


class AnalysisQueueCLITests(unittest.TestCase):
    @staticmethod
    def _evidence(root: Path, name: str) -> BufferedEvidenceResult:
        media_root = root / name
        media_root.mkdir(parents=True)
        clip = media_root / "event.mp4"
        clip.write_bytes(b"clip")
        frames = []
        for index in range(3):
            frame = media_root / f"frame_{index}.jpg"
            frame.write_bytes(b"frame")
            frames.append(str(frame))
        return BufferedEvidenceResult(
            media=EventMedia(str(clip), tuple(frames), selected_segments=4),
            requested_start="2026-08-03T01:01:58+00:00",
            requested_end="2026-08-03T01:02:10+00:00",
            covered_start="2026-08-03T01:01:58+00:00",
            covered_end="2026-08-03T01:02:10+00:00",
            complete_coverage=True,
            coverage_gaps=(),
            selected_segments=4,
            reused_existing=False,
        )

    @staticmethod
    def _write_enabled_local_policy(path: Path) -> None:
        HouseholdEventPolicy(
            package_id="analysis_cli_test",
            timezone="Asia/Shanghai",
            rules=(
                PolicyEventRule(
                    rule_id="unknown_local",
                    event_types=frozenset({"unknown"}),
                    risk_levels=frozenset({"none"}),
                    review_statuses=frozenset({"final", "pending_review"}),
                    sink_ids=("shadow_local",),
                    cooldown_seconds=300.0,
                ),
            ),
            enabled=True,
        ).write_new(path)

    @staticmethod
    def _write_enabled_nonlocal_policy(path: Path) -> None:
        HouseholdEventPolicy(
            package_id="analysis_cli_nonlocal_test",
            timezone="Asia/Shanghai",
            rules=(
                PolicyEventRule(
                    rule_id="unknown_external",
                    event_types=frozenset({"unknown"}),
                    risk_levels=frozenset({"none"}),
                    review_statuses=frozenset({"final", "pending_review"}),
                    sink_ids=("external_alert",),
                    local_only=False,
                ),
            ),
            enabled=True,
        ).write_new(path)

    def test_enqueue_restart_consume_report_and_idle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-analysis-cli-") as directory:
            root = Path(directory).resolve()
            occurred_at = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)
            manifest_path = root / "buffer/manifest.json"
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=occurred_at,
                updated_at=occurred_at,
            ).write_atomic(manifest_path)
            output_root = root / "output"
            output_root.mkdir()
            policy_path = root / "event-policy.json"
            self._write_enabled_local_policy(policy_path)
            config = root / "shadow.toml"
            config.write_text(
                "\n".join(
                    [
                        "[notification]",
                        "enabled = true",
                        "auto_file_enabled = true",
                        'auto_file_sink_id = "shadow_local"',
                        f'event_policy_path = "{policy_path.as_posix()}"',
                    ]
                ),
                encoding="utf-8",
            )
            clip = output_root / "event.mp4"
            clip.write_bytes(b"clip")
            frames = []
            for index in range(3):
                frame = output_root / f"frame_{index}.jpg"
                frame.write_bytes(b"frame")
                frames.append(str(frame))
            evidence = BufferedEvidenceResult(
                media=EventMedia(str(clip), tuple(frames), selected_segments=4),
                requested_start="2026-08-03T01:01:58+00:00",
                requested_end="2026-08-03T01:02:10+00:00",
                covered_start="2026-08-03T01:01:58+00:00",
                covered_end="2026-08-03T01:02:10+00:00",
                complete_coverage=True,
                coverage_gaps=(),
                selected_segments=4,
                reused_existing=False,
            )
            database = root / "events.sqlite3"
            enqueue_args = [
                "enqueue-buffer-event",
                "--config",
                str(config),
                "--manifest",
                str(manifest_path),
                "--trigger-at",
                occurred_at.isoformat(),
                "--event-id",
                "event_queue_01",
                "--database",
                str(database),
                "--output-root",
                str(output_root),
            ]
            with patch(
                "baby_monitor.cli_support.extract_buffered_evidence",
                return_value=evidence,
            ):
                enqueue_stdout = StringIO()
                with redirect_stdout(enqueue_stdout):
                    enqueue_code = main(enqueue_args)

            enqueued = json.loads(enqueue_stdout.getvalue())
            self.assertEqual(0, enqueue_code)
            self.assertTrue(enqueued["queued"])
            self.assertEqual(1, enqueued["queue"]["pending"])
            self.assertEqual([], EventDatabase(database).list_recent())

            run_stdout = StringIO()
            with redirect_stdout(run_stdout):
                run_code = main(
                    [
                        "analysis-run-once",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                    ]
                )
            processed = json.loads(run_stdout.getvalue())
            self.assertEqual(0, run_code)
            self.assertTrue(processed["processed"])
            self.assertEqual("completed", processed["status"])
            self.assertEqual(1, processed["queue"]["completed"])
            self.assertEqual(1, len(EventDatabase(database).list_recent()))
            self.assertEqual("route", processed["event_policy_decision"]["action"])
            self.assertEqual("routed", processed["event_policy_decision"]["reason"])
            self.assertFalse(processed["event_policy_decision"]["notification_delivery_authority"])
            self.assertEqual("pending", processed["notification_enqueue"]["status"])
            self.assertEqual(
                "shadow_local",
                processed["notification_enqueue"]["sink_id"],
            )
            self.assertTrue(Path(processed["report"]["markdown"]).is_file())
            self.assertTrue(Path(processed["report"]["html"]).is_file())

            second_evidence = self._evidence(root, "second-event")
            second_args = [
                *enqueue_args,
            ]
            second_args[second_args.index("--trigger-at") + 1] = (
                occurred_at + timedelta(seconds=60)
            ).isoformat()
            second_args[second_args.index("--event-id") + 1] = "event_queue_02"
            with (
                patch(
                    "baby_monitor.cli_support.extract_buffered_evidence",
                    return_value=second_evidence,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(0, main(second_args))

            suppressed_stdout = StringIO()
            with redirect_stdout(suppressed_stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "analysis-run-once",
                            "--config",
                            str(config),
                            "--database",
                            str(database),
                        ]
                    ),
                )
            suppressed = json.loads(suppressed_stdout.getvalue())
            self.assertEqual("suppress", suppressed["event_policy_decision"]["action"])
            self.assertEqual("cooldown", suppressed["event_policy_decision"]["reason"])
            self.assertEqual("suppressed", suppressed["notification_enqueue"]["status"])
            notification_counts = NotificationOutbox(database).snapshot_read_only().counts
            self.assertEqual(1, notification_counts.pending)

            nonlocal_policy_path = root / "event-policy-nonlocal.json"
            self._write_enabled_nonlocal_policy(nonlocal_policy_path)
            config.write_text(
                "\n".join(
                    [
                        "[notification]",
                        "enabled = true",
                        "auto_file_enabled = true",
                        'auto_file_sink_id = "shadow_local"',
                        f'event_policy_path = "{nonlocal_policy_path.as_posix()}"',
                    ]
                ),
                encoding="utf-8",
            )
            third_evidence = self._evidence(root, "third-event")
            third_args = [*enqueue_args]
            third_args[third_args.index("--trigger-at") + 1] = (
                occurred_at + timedelta(seconds=600)
            ).isoformat()
            third_args[third_args.index("--event-id") + 1] = "event_queue_03"
            with (
                patch(
                    "baby_monitor.cli_support.extract_buffered_evidence",
                    return_value=third_evidence,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(0, main(third_args))

            nonlocal_stdout = StringIO()
            with redirect_stdout(nonlocal_stdout):
                self.assertEqual(
                    0,
                    main(
                        [
                            "analysis-run-once",
                            "--config",
                            str(config),
                            "--database",
                            str(database),
                        ]
                    ),
                )
            nonlocal_result = json.loads(nonlocal_stdout.getvalue())
            self.assertEqual("route", nonlocal_result["event_policy_decision"]["action"])
            self.assertFalse(nonlocal_result["event_policy_decision"]["local_only"])
            self.assertEqual("not_enqueued", nonlocal_result["notification_enqueue"]["status"])
            self.assertEqual(
                "external_delivery_not_authorized",
                nonlocal_result["notification_enqueue"]["error_code"],
            )
            self.assertEqual(
                1,
                NotificationOutbox(database).snapshot_read_only().counts.pending,
            )

            idle_stdout = StringIO()
            with redirect_stdout(idle_stdout):
                idle_code = main(
                    [
                        "analysis-run-once",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                    ]
                )
            idle = json.loads(idle_stdout.getvalue())
            self.assertEqual(0, idle_code)
            self.assertEqual("idle", idle["status"])
            self.assertFalse(idle["processed"])

    def test_queue_full_returns_structured_result_and_preserves_second_media(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-queue-full-") as directory:
            root = Path(directory).resolve()
            occurred_at = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)
            manifest = root / "buffer/manifest.json"
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=occurred_at,
                updated_at=occurred_at,
            ).write_atomic(manifest)
            config = root / "queue.toml"
            config.write_text("[analysis_queue]\nmax_pending = 1\n", encoding="utf-8")
            database = root / "events.sqlite3"
            first_evidence = self._evidence(root, "first")
            second_evidence = self._evidence(root, "second")

            def enqueue(event_id: str, evidence: BufferedEvidenceResult) -> tuple[int, dict]:
                arguments = [
                    "enqueue-buffer-event",
                    "--config",
                    str(config),
                    "--manifest",
                    str(manifest),
                    "--trigger-at",
                    occurred_at.isoformat(),
                    "--event-id",
                    event_id,
                    "--database",
                    str(database),
                    "--output-root",
                    str(root),
                ]
                with patch(
                    "baby_monitor.cli_support.extract_buffered_evidence",
                    return_value=evidence,
                ):
                    stdout = StringIO()
                    with redirect_stdout(stdout):
                        code = main(arguments)
                return code, json.loads(stdout.getvalue())

            first_code, first = enqueue("event_01", first_evidence)
            second_code, second = enqueue("event_02", second_evidence)
            self.assertEqual(0, first_code)
            self.assertTrue(first["queued"])
            self.assertEqual(3, second_code)
            self.assertFalse(second["queued"])
            self.assertEqual("queue_full", second["reason"])
            self.assertTrue(second["evidence_preserved"])
            self.assertTrue(Path(second_evidence.media.clip_path).is_file())
            self.assertTrue(all(Path(path).is_file() for path in second_evidence.media.frame_paths))

    def test_enqueue_preserves_coverage_metadata_when_paths_become_absolute(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-queue-gap-") as directory:
            root = Path(directory).resolve()
            occurred_at = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)
            manifest = root / "buffer/manifest.json"
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=occurred_at,
                updated_at=occurred_at,
            ).write_atomic(manifest)
            base = self._evidence(root, "gapped")
            gap = EvidenceGap(
                started_at=occurred_at - timedelta(seconds=5),
                ended_at=occurred_at - timedelta(seconds=1),
                reason="segment_missing",
            )
            evidence = replace(
                base,
                media=replace(
                    base.media,
                    coverage_gaps=(gap,),
                    selected_segments=1,
                ),
                complete_coverage=False,
                coverage_gaps=(gap,),
                selected_segments=1,
            )
            database = root / "events.sqlite3"
            with (
                patch(
                    "baby_monitor.cli_support.extract_buffered_evidence",
                    return_value=evidence,
                ),
                redirect_stdout(StringIO()),
            ):
                code = main(
                    [
                        "enqueue-buffer-event",
                        "--manifest",
                        str(manifest),
                        "--trigger-at",
                        occurred_at.isoformat(),
                        "--event-id",
                        "event_gap",
                        "--database",
                        str(database),
                        "--output-root",
                        str(root),
                    ]
                )
            self.assertEqual(0, code)
            claimed = AnalysisQueue(database).claim_next(
                lease_seconds=10,
                max_attempts=3,
            )
            assert claimed is not None
            self.assertEqual((gap,), claimed.media.coverage_gaps)
            self.assertEqual(1, claimed.media.selected_segments)
            self.assertFalse(claimed.media.complete_coverage)

    def test_worker_failure_is_bounded_and_manual_retry_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-queue-fail-") as directory:
            root = Path(directory).resolve()
            occurred_at = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)
            manifest = root / "buffer/manifest.json"
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=occurred_at,
                updated_at=occurred_at,
            ).write_atomic(manifest)
            runner = root / "fail_runner.py"
            runner.write_text("raise SystemExit(7)\n", encoding="utf-8")
            config = root / "worker.toml"
            config.write_text(
                "\n".join(
                    [
                        "[analysis_queue]",
                        "retry_delay_seconds = 0",
                        "max_attempts = 1",
                        "[local_vlm]",
                        'backend = "command"',
                        'name = "fixture_failure"',
                        "command = ["
                        f'"{Path(sys.executable).as_posix()}", '
                        f'"{runner.as_posix()}", "{{request}}"]',
                        "timeout_seconds = 5",
                    ]
                ),
                encoding="utf-8",
            )
            evidence = self._evidence(root, "evidence")
            database = root / "events.sqlite3"
            arguments = [
                "enqueue-buffer-event",
                "--config",
                str(config),
                "--manifest",
                str(manifest),
                "--trigger-at",
                occurred_at.isoformat(),
                "--event-id",
                "event_failure",
                "--database",
                str(database),
                "--output-root",
                str(root),
            ]
            with (
                patch(
                    "baby_monitor.cli_support.extract_buffered_evidence",
                    return_value=evidence,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(0, main(arguments))

            run_stdout = StringIO()
            with redirect_stdout(run_stdout):
                run_code = main(
                    [
                        "analysis-run-once",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                    ]
                )
            failed = json.loads(run_stdout.getvalue())
            self.assertEqual(2, run_code)
            self.assertEqual("failed", failed["status"])
            self.assertEqual("VLMError", failed["error_code"])
            self.assertTrue(failed["evidence_preserved"])
            self.assertEqual([], EventDatabase(database).list_recent())

            retry_stdout = StringIO()
            with redirect_stdout(retry_stdout):
                retry_code = main(
                    [
                        "analysis-retry",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                        "--event-id",
                        "event_failure",
                    ]
                )
            retried = json.loads(retry_stdout.getvalue())
            self.assertEqual(0, retry_code)
            self.assertEqual("pending", retried["status"])
            self.assertEqual(0, retried["attempt_count"])

            worker_stdout = StringIO()
            with redirect_stdout(worker_stdout):
                worker_code = main(
                    [
                        "analysis-worker",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                        "--max-jobs",
                        "1",
                        "--idle-exit-seconds",
                        "1",
                        "--max-runtime-seconds",
                        "10",
                        "--poll-seconds",
                        "0.01",
                    ]
                )
            worker_result = json.loads(worker_stdout.getvalue())
            self.assertEqual(2, worker_code)
            self.assertEqual(1, worker_result["worker"]["failed_jobs"])
            self.assertEqual(1, worker_result["worker"]["queue"]["failed"])

    def test_bounded_worker_consumes_multiple_jobs_sequentially(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-worker-cli-") as directory:
            root = Path(directory).resolve()
            occurred_at = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)
            manifest = root / "buffer/manifest.json"
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=occurred_at,
                updated_at=occurred_at,
            ).write_atomic(manifest)
            database = root / "events.sqlite3"
            for event_id in ("worker_event_01", "worker_event_02"):
                evidence = self._evidence(root, event_id)
                with (
                    patch(
                        "baby_monitor.cli_support.extract_buffered_evidence",
                        return_value=evidence,
                    ),
                    redirect_stdout(StringIO()),
                ):
                    self.assertEqual(
                        0,
                        main(
                            [
                                "enqueue-buffer-event",
                                "--manifest",
                                str(manifest),
                                "--trigger-at",
                                occurred_at.isoformat(),
                                "--event-id",
                                event_id,
                                "--database",
                                str(database),
                                "--output-root",
                                str(root),
                            ]
                        ),
                    )

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "analysis-worker",
                        "--database",
                        str(database),
                        "--max-jobs",
                        "2",
                        "--idle-exit-seconds",
                        "5",
                        "--max-runtime-seconds",
                        "30",
                        "--poll-seconds",
                        "0.01",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(0, code)
            self.assertEqual("stopped", payload["status"])
            self.assertEqual("max_jobs", payload["worker"]["stop_reason"])
            self.assertEqual(2, payload["worker"]["attempted_jobs"])
            self.assertEqual(2, payload["worker"]["completed_jobs"])
            self.assertEqual(2, payload["worker"]["queue"]["completed"])
            self.assertEqual(2, len(EventDatabase(database).list_recent()))

    def test_gpu_guard_defers_before_claiming_job(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-gpu-guard-") as directory:
            root = Path(directory).resolve()
            occurred_at = datetime(2026, 8, 3, 1, 2, 3, tzinfo=UTC)
            manifest = root / "buffer/manifest.json"
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=occurred_at,
                updated_at=occurred_at,
            ).write_atomic(manifest)
            config = root / "guard.toml"
            config.write_text(
                "[analysis_worker]\ngpu_guard_enabled = true\n",
                encoding="utf-8",
            )
            evidence = self._evidence(root, "evidence")
            database = root / "events.sqlite3"
            with (
                patch(
                    "baby_monitor.cli_support.extract_buffered_evidence",
                    return_value=evidence,
                ),
                redirect_stdout(StringIO()),
            ):
                enqueue_code = main(
                    [
                        "enqueue-buffer-event",
                        "--config",
                        str(config),
                        "--manifest",
                        str(manifest),
                        "--trigger-at",
                        occurred_at.isoformat(),
                        "--event-id",
                        "event_guard",
                        "--database",
                        str(database),
                        "--output-root",
                        str(root),
                    ]
                )
            self.assertEqual(0, enqueue_code)
            busy = GPUCapacity(False, "gpu_busy", 0, 12_282, 2_000, 10_282, 80)
            stdout = StringIO()
            with (
                patch("baby_monitor.cli_commands.analysis.probe_gpu_capacity", return_value=busy),
                redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "analysis-run-once",
                        "--config",
                        str(config),
                        "--database",
                        str(database),
                    ]
                )
            payload = json.loads(stdout.getvalue())
            self.assertEqual(0, code)
            self.assertEqual("deferred", payload["status"])
            self.assertEqual("gpu_busy", payload["reason"])
            self.assertEqual(1, payload["queue"]["pending"])
            self.assertEqual([], EventDatabase(database).list_recent())

    def test_signal_with_unavailable_buffer_window_returns_a_fixed_reason(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-signal-window-") as directory:
            root = Path(directory).resolve()
            signal_path = root / "signal.json"
            signal_value = json.loads(
                Path("tests/fixtures/valid_signal_event.json").read_text(encoding="utf-8")
            )
            signal_path.write_text(json.dumps(signal_value), encoding="utf-8")
            occurred_at = datetime.fromisoformat(signal_value["occurred_at"])
            manifest = root / "buffer/manifest.json"
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=occurred_at,
                updated_at=occurred_at,
            ).write_atomic(manifest)
            stdout = StringIO()

            with (
                patch(
                    "baby_monitor.cli_support.extract_buffered_evidence",
                    side_effect=BufferWindowUnavailableError("private detail"),
                ),
                redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "enqueue-signal-event",
                        "--signal-json",
                        str(signal_path),
                        "--manifest",
                        str(manifest),
                    ]
                )

            payload = json.loads(stdout.getvalue())
            self.assertEqual(4, code)
            self.assertEqual("buffer_window_unavailable", payload["reason"])
            self.assertFalse(payload["media_paths_disclosed"])
            self.assertNotIn("private detail", stdout.getvalue())
            self.assertNotIn(str(root), stdout.getvalue())

    def test_signal_media_failures_return_only_fixed_reasons(self) -> None:
        for failure, expected_code, expected_reason in (
            (FileNotFoundError("private command"), 5, "media_command_unavailable"),
            (BufferWindowUnavailableError("private media"), 4, "buffer_window_unavailable"),
            (RingBufferError("private media"), 6, "buffer_media_processing_failed"),
        ):
            with (
                self.subTest(expected_reason=expected_reason),
                tempfile.TemporaryDirectory(prefix="baby-monitor-signal-failure-") as directory,
            ):
                root = Path(directory).resolve()
                signal_path = root / "signal.json"
                signal_value = json.loads(
                    Path("tests/fixtures/valid_signal_event.json").read_text(encoding="utf-8")
                )
                signal_path.write_text(json.dumps(signal_value), encoding="utf-8")
                occurred_at = datetime.fromisoformat(signal_value["occurred_at"])
                manifest = root / "buffer/manifest.json"
                SegmentManifest(
                    camera_id="cam_01",
                    source_ref="rtsp://127.0.0.1:8554/test",
                    created_at=occurred_at,
                    updated_at=occurred_at,
                ).write_atomic(manifest)
                stdout = StringIO()

                with (
                    patch(
                        "baby_monitor.cli_support.extract_buffered_evidence", side_effect=failure
                    ),
                    redirect_stdout(stdout),
                ):
                    code = main(
                        [
                            "enqueue-signal-event",
                            "--signal-json",
                            str(signal_path),
                            "--manifest",
                            str(manifest),
                        ]
                    )

                payload = json.loads(stdout.getvalue())
                self.assertEqual(expected_code, code)
                self.assertEqual(expected_reason, payload["reason"])
                self.assertNotIn("private", stdout.getvalue())
                self.assertNotIn(str(root), stdout.getvalue())

    def test_signal_preflight_failures_return_only_fixed_reasons(self) -> None:
        cases = (
            ("baby_monitor.cli_commands.signal_events.load_settings", 7, "buffer_config_invalid"),
            ("baby_monitor.cli_commands.signal_events._one_signal", 8, "signal_staging_invalid"),
        )
        for target, expected_code, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                stdout = StringIO()
                with (
                    patch(target, side_effect=RuntimeError("private detail")),
                    redirect_stdout(stdout),
                ):
                    code = main(
                        [
                            "enqueue-signal-event",
                            "--signal-json",
                            "/private/signal.json",
                            "--manifest",
                            "/private/manifest.json",
                        ]
                    )

                payload = json.loads(stdout.getvalue())
                self.assertEqual(expected_code, code)
                self.assertEqual(expected_reason, payload["reason"])
                self.assertNotIn("private detail", stdout.getvalue())
                self.assertNotIn("/private", stdout.getvalue())

    def test_strict_signal_triggers_buffer_evidence_queue_and_mock_review(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-signal-cli-") as directory:
            root = Path(directory).resolve()
            signal_path = root / "signal.json"
            signal_value = json.loads(
                Path("tests/fixtures/valid_signal_event.json").read_text(encoding="utf-8")
            )
            signal_path.write_text(json.dumps(signal_value), encoding="utf-8")
            occurred_at = datetime.fromisoformat(signal_value["occurred_at"])
            manifest = root / "buffer/manifest.json"
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=occurred_at,
                updated_at=occurred_at,
            ).write_atomic(manifest)
            evidence = self._evidence(root, "evidence")
            database = root / "events.sqlite3"
            arguments = [
                "enqueue-signal-event",
                "--signal-json",
                str(signal_path),
                "--manifest",
                str(manifest),
                "--database",
                str(database),
                "--output-root",
                str(root),
            ]

            payloads = []
            with patch(
                "baby_monitor.cli_support.extract_buffered_evidence",
                return_value=evidence,
            ):
                for _attempt in range(2):
                    stdout = StringIO()
                    with redirect_stdout(stdout):
                        self.assertEqual(0, main(arguments))
                    payloads.append(json.loads(stdout.getvalue()))
            self.assertTrue(payloads[0]["queued"])
            self.assertFalse(payloads[0]["reused_existing"])
            self.assertTrue(payloads[1]["reused_existing"])
            self.assertEqual(payloads[0]["event_id"], payloads[1]["event_id"])
            self.assertEqual(1, payloads[0]["signal_count"])
            self.assertEqual(["hardware_cry"], payloads[0]["signal_types"])

            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    0,
                    main(["analysis-run-once", "--database", str(database)]),
                )
            processed = json.loads(stdout.getvalue())
            self.assertEqual("completed", processed["status"])
            self.assertEqual("unknown", processed["result"]["event_type"])
            self.assertTrue(
                any("hardware_cry" in fact for fact in processed["result"]["observations"])
            )
            stored = EventDatabase(database).list_recent()
            self.assertEqual(1, len(stored))
            self.assertEqual(
                "hardware_cry",
                stored[0].candidate["signals"][0]["signal_type"],
            )


if __name__ == "__main__":
    unittest.main()
