from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import chdir, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from baby_monitor.adapters import vlm_execution as gate
from baby_monitor.cli import main
from baby_monitor.database import EventDatabase
from baby_monitor.object_tracker import DetectionFrameBatch


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg and ffprobe are required",
)
class ProcessFileE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        # Keep the real command-adapter gate, but use a private test-only root.
        temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-file-cli-gate-")
        self.addCleanup(temporary.cleanup)
        execution_root = Path(temporary.name)
        execution_root.chmod(0o700)
        replacement = patch.object(gate, "_execution_root", return_value=execution_root)
        replacement.start()
        self.addCleanup(replacement.stop)

    @staticmethod
    def _generate_two_scene_video(path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=red:size=160x90:rate=10:d=2",
                "-f",
                "lavfi",
                "-i",
                "color=blue:size=160x90:rate=10:d=2",
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )

    @staticmethod
    def _generate_three_scene_video(path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=black:size=160x90:rate=10:d=1",
                "-f",
                "lavfi",
                "-i",
                "color=white:size=160x90:rate=10:d=1",
                "-f",
                "lavfi",
                "-i",
                "color=black:size=160x90:rate=10:d=1",
                "-filter_complex",
                "[0:v][1:v][2:v]concat=n=3:v=1:a=0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )

    def test_scene_detector_runs_real_file_to_sqlite_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "input.mp4"
            database_path = root / "events.sqlite3"
            output_root = root / "output"
            config_path = root / "test.toml"
            self._generate_two_scene_video(source_path)
            config_path.write_text(
                """
[general]
timezone = "Asia/Shanghai"

[detector]
backend = "scene"
candidate_gap_seconds = 1.0
scene_threshold = 0.2
scene_max_detections = 20

[media]
pre_event_seconds = 0.5
post_event_seconds = 0.5
keyframe_count = 3
ffmpeg = "ffmpeg"
ffprobe = "ffprobe"

[local_vlm]
backend = "mock"
confidence_threshold = 0.65
""".strip()
                + "\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "process-file",
                        "--config",
                        str(config_path),
                        "--source",
                        str(source_path),
                        "--database",
                        str(database_path),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(0, exit_code, stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            self.assertEqual("scene", payload["detector"])
            self.assertEqual("mock", payload["local_vlm"])
            self.assertEqual(1, payload["detections"])
            self.assertEqual(1, payload["candidates"])
            self.assertEqual(1, len(payload["events"]))
            event = payload["events"][0]
            self.assertEqual("movement", event["result"]["event_type"])
            self.assertEqual("mock", event["local_provenance"]["adapter_name"])
            self.assertTrue(event["local_metrics"]["json_compliant"])
            self.assertGreaterEqual(event["local_metrics"]["duration_ms"], 0)
            self.assertEqual("scene_change", event["candidate"]["detections"][0]["label"])
            self.assertTrue(Path(event["clip_path"]).is_file())
            self.assertTrue(
                all(Path(path).is_file() for path in event["result"]["evidence_frames"])
            )
            self.assertEqual(1, payload["report"]["event_count"])
            self.assertTrue(Path(payload["report"]["markdown"]).is_file())
            self.assertTrue(Path(payload["report"]["html"]).is_file())

            stored = EventDatabase(database_path).list_recent(limit=10)
            self.assertEqual(1, len(stored))
            self.assertEqual(event["event_id"], stored[0].event_id)

            list_stdout = StringIO()
            with redirect_stdout(list_stdout):
                list_code = main(["list-events", "--database", str(database_path), "--limit", "10"])
            listed = json.loads(list_stdout.getvalue())
            self.assertEqual(0, list_code)
            self.assertEqual("mock", listed[0]["local_provenance"]["adapter_name"])
            self.assertTrue(listed[0]["local_metrics"]["json_compliant"])

    def test_relative_output_root_gives_command_vlm_absolute_media_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "input.mp4"
            database_path = root / "events.sqlite3"
            config_path = root / "command.toml"
            runner_path = root / "fixture_runner.py"
            self._generate_two_scene_video(source_path)
            runner_path.write_text(
                """
import json
import sys
from pathlib import Path

with open(sys.argv[1], encoding="utf-8") as handle:
    request = json.load(handle)
frame_paths = request["frame_paths"]
if not frame_paths or not all(Path(value).is_absolute() for value in frame_paths):
    raise SystemExit(3)
print(json.dumps({
    "event_type": "unknown",
    "subjects": [],
    "observations": ["Only synthetic color frames are visible."],
    "inferences": [],
    "risk_level": "none",
    "confidence": 0.0,
    "needs_remote_review": True,
    "summary_zh": "合成色块不足以判断宝宝事件。",
    "evidence_frames": [frame_paths[0]],
}))
""".strip()
                + "\n",
                encoding="utf-8",
            )
            executable = Path(sys.executable).resolve().as_posix()
            runner = runner_path.resolve().as_posix()
            config_path.write_text(
                "\n".join(
                    [
                        "[detector]",
                        'backend = "scene"',
                        "candidate_gap_seconds = 1.0",
                        "scene_threshold = 0.2",
                        "scene_max_detections = 20",
                        "",
                        "[media]",
                        "pre_event_seconds = 0.5",
                        "post_event_seconds = 0.5",
                        "keyframe_count = 3",
                        'ffmpeg = "ffmpeg"',
                        'ffprobe = "ffprobe"',
                        "",
                        "[local_vlm]",
                        'backend = "command"',
                        'name = "absolute_path_fixture"',
                        f'command = ["{executable}", "{runner}", "{{request}}"]',
                        "timeout_seconds = 5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            stdout = StringIO()
            stderr = StringIO()
            with chdir(root), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "process-file",
                        "--config",
                        str(config_path),
                        "--source",
                        str(source_path),
                        "--database",
                        str(database_path),
                        "--output-root",
                        "relative-output",
                    ]
                )

            self.assertEqual(0, exit_code, stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            event = payload["events"][0]
            self.assertTrue(Path(event["clip_path"]).is_absolute())
            self.assertTrue(
                all(Path(path).is_absolute() for path in event["result"]["evidence_frames"])
            )
            self.assertTrue(Path(payload["report"]["markdown"]).is_absolute())
            self.assertTrue(Path(payload["report"]["html"]).is_absolute())

    def test_scene_frames_track_through_zone_into_event_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "synthetic.mp4"
            database_path = root / "events.sqlite3"
            output_root = root / "output"
            config_path = root / "test.toml"
            tracker_path = root / "tracker.json"
            zone_path = root / "zones.json"
            detection_frames_path = root / "detection-frames.json"
            self._generate_three_scene_video(source_path)
            config_path.write_text(
                """
[general]
timezone = "Asia/Shanghai"

[detector]
backend = "scene"
candidate_gap_seconds = 1.5
scene_threshold = 0.2
scene_max_detections = 20

[media]
pre_event_seconds = 0.25
post_event_seconds = 0.25
keyframe_count = 3
ffmpeg = "ffmpeg"
ffprobe = "ffprobe"

[local_vlm]
backend = "mock"
confidence_threshold = 0.65
""".strip()
                + "\n",
                encoding="utf-8",
            )
            tracker_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "iou_threshold": 0.3,
                        "max_track_age_seconds": 2.0,
                        "max_active_tracks": 16,
                        "max_processed_frames": 100,
                        "max_pending_observations": 100,
                    }
                ),
                encoding="utf-8",
            )
            zone_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "track_expiry_seconds": 30.0,
                        "max_processed_observations": 100,
                        "zones": [
                            {
                                "zone_id": "synthetic_full_frame",
                                "labels": ["scene_change"],
                                "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                                "dwell_seconds": 0.5,
                                "exit_margin": 0.0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main(
                    [
                        "process-file",
                        "--config",
                        str(config_path),
                        "--source",
                        str(source_path),
                        "--database",
                        str(database_path),
                        "--output-root",
                        str(output_root),
                        "--tracker-config",
                        str(tracker_path),
                        "--zone-config",
                        str(zone_path),
                        "--detection-frames-output",
                        str(detection_frames_path),
                    ]
                )

            self.assertEqual(0, exit_code, stderr.getvalue())
            payload = json.loads(stdout.getvalue())
            perception = payload["local_perception"]
            self.assertTrue(perception["enabled"])
            self.assertEqual(2, perception["detection_frames"])
            self.assertEqual(2, perception["frame_detections"])
            self.assertEqual(2, perception["tracked_observations"])
            self.assertEqual(1, perception["active_tracks"])
            self.assertEqual(2, perception["temporal_signals"])
            self.assertEqual(["zone_dwell", "zone_enter"], perception["signal_types"])
            self.assertFalse(perception["subject_identity_inference"])
            self.assertFalse(perception["paths_disclosed"])
            self.assertTrue(perception["detection_frames_saved"])
            saved_batch = DetectionFrameBatch.read(detection_frames_path)
            self.assertEqual(2, len(saved_batch.frames))
            self.assertEqual(0o600, detection_frames_path.stat().st_mode & 0o777)

            self.assertEqual(1, payload["candidates"])
            event = payload["events"][0]
            signals = event["candidate"]["signals"]
            self.assertEqual(
                ["zone_enter", "zone_dwell"],
                [signal["signal_type"] for signal in signals],
            )
            self.assertTrue(
                all(signal["attributes"]["object_label"] == "scene_change" for signal in signals)
            )
            tracking_ids = {signal["attributes"]["tracking_id"] for signal in signals}
            self.assertEqual(1, len(tracking_ids))
            self.assertTrue(next(iter(tracking_ids)).startswith("track_"))
            self.assertEqual([], event["result"]["inferences"])
            self.assertEqual("none", event["result"]["risk_level"])
            self.assertTrue(
                any("temporal_zone_v1" in fact for fact in event["result"]["observations"])
            )
            self.assertTrue(Path(event["clip_path"]).is_file())
            self.assertTrue(Path(payload["report"]["markdown"]).is_file())
            self.assertTrue(Path(payload["report"]["html"]).is_file())

            stored = EventDatabase(database_path).list_recent(limit=10)
            self.assertEqual(1, len(stored))
            self.assertEqual(2, len(stored[0].candidate["signals"]))

    def test_tracking_requires_both_configs_before_database_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "synthetic.mp4"
            database_path = root / "events.sqlite3"
            self._generate_two_scene_video(source_path)

            stderr = StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "process-file",
                        "--source",
                        str(source_path),
                        "--database",
                        str(database_path),
                        "--tracker-config",
                        str(root / "tracker.json"),
                    ]
                )

            self.assertEqual(2, exit_code)
            self.assertIn("must be provided together", stderr.getvalue())
            self.assertFalse(database_path.exists())

            output_only_error = StringIO()
            with redirect_stderr(output_only_error):
                output_only_code = main(
                    [
                        "process-file",
                        "--source",
                        str(source_path),
                        "--database",
                        str(database_path),
                        "--detection-frames-output",
                        str(root / "frames.json"),
                    ]
                )
            self.assertEqual(2, output_only_code)
            self.assertIn("requires --tracker-config", output_only_error.getvalue())
            self.assertFalse(database_path.exists())


if __name__ == "__main__":
    unittest.main()
