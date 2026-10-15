from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from baby_monitor.adapters import vlm_execution as gate
from baby_monitor.cli import main
from baby_monitor.database import EventDatabase
from baby_monitor.domain import EventMedia
from baby_monitor.ring_buffer import BufferedEvidenceResult, SegmentManifest


class ProcessBufferEventCLITests(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic command runners must not contend with a real resident VLM.
        temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-buffer-cli-gate-")
        self.addCleanup(temporary.cleanup)
        execution_root = Path(temporary.name)
        execution_root.chmod(0o700)
        replacement = patch.object(gate, "_execution_root", return_value=execution_root)
        replacement.start()
        self.addCleanup(replacement.stop)

    def test_retry_reuses_one_sqlite_event_and_keeps_remote_disabled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-buffer-cli-") as directory:
            root = Path(directory)
            manifest_path = root / "buffer/manifest.json"
            occurred_at = datetime(2026, 8, 2, 1, 2, 3, tzinfo=UTC)
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=occurred_at,
                updated_at=occurred_at,
            ).write_atomic(manifest_path)
            media = EventMedia(
                clip_path=str(root / "output/event.mp4"),
                frame_paths=tuple(str(root / f"output/frame_{index}.jpg") for index in range(3)),
                selected_segments=4,
            )
            evidence = BufferedEvidenceResult(
                media=media,
                requested_start="2026-08-02T01:01:58+00:00",
                requested_end="2026-08-02T01:02:10+00:00",
                covered_start="2026-08-02T01:01:58+00:00",
                covered_end="2026-08-02T01:02:10+00:00",
                complete_coverage=True,
                coverage_gaps=(),
                selected_segments=4,
                reused_existing=False,
            )
            database_path = root / "events.sqlite3"
            output_root = root / "output"
            arguments = [
                "process-buffer-event",
                "--manifest",
                str(manifest_path),
                "--trigger-at",
                occurred_at.isoformat(),
                "--event-id",
                "event_01",
                "--database",
                str(database_path),
                "--output-root",
                str(output_root),
            ]
            with patch(
                "baby_monitor.cli_commands.media_pipeline.extract_buffered_evidence",
                return_value=evidence,
            ):
                first_stdout = StringIO()
                with redirect_stdout(first_stdout):
                    first_code = main(arguments)
                second_stdout = StringIO()
                with redirect_stdout(second_stdout):
                    second_code = main(arguments)

            first = json.loads(first_stdout.getvalue())
            second = json.loads(second_stdout.getvalue())
            self.assertEqual(0, first_code)
            self.assertEqual(0, second_code)
            self.assertFalse(first["reused_existing"])
            self.assertTrue(second["reused_existing"])
            self.assertFalse(first["remote_reviewed"])
            self.assertTrue(first["pending_remote_review"])
            self.assertEqual("unknown", first["result"]["event_type"])
            self.assertEqual(1, len(first["result"]["observations"]))
            self.assertIn("未验证连续覆盖", first["result"]["observations"][0])
            self.assertEqual(1, len(EventDatabase(database_path).list_recent()))

    def test_command_vlm_backend_persists_strict_result(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-command-cli-") as directory:
            root = Path(directory)
            manifest_path = root / "buffer/manifest.json"
            occurred_at = datetime(2026, 8, 2, 2, 3, 4, tzinfo=UTC)
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=occurred_at,
                updated_at=occurred_at,
            ).write_atomic(manifest_path)
            media = EventMedia(
                clip_path=str(root / "output/event.mp4"),
                frame_paths=tuple(str(root / f"output/frame_{index}.jpg") for index in range(3)),
                selected_segments=4,
            )
            evidence = BufferedEvidenceResult(
                media=media,
                requested_start="2026-08-02T02:02:59+00:00",
                requested_end="2026-08-02T02:03:11+00:00",
                covered_start="2026-08-02T02:02:59+00:00",
                covered_end="2026-08-02T02:03:11+00:00",
                complete_coverage=True,
                coverage_gaps=(),
                selected_segments=4,
                reused_existing=False,
            )
            runner_path = root / "fixture_runner.py"
            runner_path.write_text(
                """
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    request = json.load(handle)
print(json.dumps({
    "event_type": "unknown",
    "subjects": [],
    "observations": [],
    "inferences": [],
    "risk_level": "none",
    "confidence": 0.8,
    "needs_remote_review": False,
    "summary_zh": "fixture command runner",
    "evidence_frames": [request["frame_paths"][0]],
}))
""",
                encoding="utf-8",
            )
            config_path = root / "command.toml"
            executable = Path(sys.executable).as_posix()
            runner = runner_path.as_posix()
            config_path.write_text(
                "\n".join(
                    [
                        "[local_vlm]",
                        'backend = "command"',
                        'name = "fixture_command"',
                        f'command = ["{executable}", "{runner}", "{{request}}"]',
                        "timeout_seconds = 5",
                    ]
                ),
                encoding="utf-8",
            )
            database_path = root / "events.sqlite3"
            arguments = [
                "process-buffer-event",
                "--config",
                str(config_path),
                "--manifest",
                str(manifest_path),
                "--trigger-at",
                occurred_at.isoformat(),
                "--event-id",
                "event_command_01",
                "--database",
                str(database_path),
                "--output-root",
                str(root / "output"),
            ]
            with patch(
                "baby_monitor.cli_commands.media_pipeline.extract_buffered_evidence",
                return_value=evidence,
            ):
                stdout = StringIO()
                with redirect_stdout(stdout):
                    code = main(arguments)

            payload = json.loads(stdout.getvalue())
            self.assertEqual(0, code)
            self.assertEqual("fixture_command", payload["local_vlm"])
            self.assertEqual(0.8, payload["result"]["confidence"])
            self.assertFalse(payload["pending_remote_review"])
            events = EventDatabase(database_path).list_recent()
            self.assertEqual(1, len(events))
            self.assertIsNotNone(events[0].local_metrics)
            assert events[0].local_metrics is not None
            self.assertTrue(events[0].local_metrics.json_compliant)
            self.assertGreaterEqual(events[0].local_metrics.duration_ms, 0)
            if Path("/usr/bin/time").is_file():
                self.assertIsNotNone(events[0].local_metrics.process_rss_peak_kib)


if __name__ == "__main__":
    unittest.main()
