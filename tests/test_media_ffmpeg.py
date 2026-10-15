from __future__ import annotations

import json
import shutil
import stat
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.media import FFmpegMediaProcessor, FFprobeMediaInspector, MediaError
from baby_monitor.config import StorageSettings
from baby_monitor.domain import CandidateEvent
from baby_monitor.storage import StorageResolver


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class FFmpegMediaIntegrationTests(unittest.TestCase):
    def test_synthetic_video_produces_playable_clip_and_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "synthetic-input.mp4"
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
                    "testsrc=size=160x90:rate=10",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=16000",
                    "-t",
                    "5",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-c:a",
                    "aac",
                    str(source_path),
                ],
                check=True,
            )

            media = FFmpegMediaProcessor(keyframe_count=5).prepare(
                source=LocalFileCamera("cam_01", source_path).source(),
                candidate=CandidateEvent("event_01", "cam_01", 1.0, 3.0, ()),
                occurred_at=datetime(2026, 8, 1, tzinfo=UTC),
                storage=StorageResolver(
                    StorageSettings(
                        local_root=root / "output",
                        database=root / "events.sqlite3",
                    )
                ),
            )

            clip = Path(media.clip_path)
            self.assertTrue(clip.is_file())
            self.assertEqual(0o600, stat.S_IMODE(clip.stat().st_mode))
            self.assertEqual(5, len(media.frame_paths))
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in media.frame_paths))
            self.assertTrue(
                all(stat.S_IMODE(Path(path).stat().st_mode) == 0o600 for path in media.frame_paths)
            )

            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(clip),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            duration = float(json.loads(probe.stdout)["format"]["duration"])
            self.assertGreater(duration, 1.5)
            self.assertLess(duration, 2.5)

            metadata = FFprobeMediaInspector().probe(
                LocalFileCamera("cam_01", source_path).source()
            )
            self.assertAlmostEqual(5.0, metadata.duration_seconds, places=1)
            self.assertEqual("h264", metadata.video_codec)
            self.assertEqual((160, 90), (metadata.width, metadata.height))
            self.assertAlmostEqual(10.0, metadata.average_fps or 0.0, places=1)
            self.assertTrue(metadata.has_audio)

    def test_window_past_eof_is_clamped_and_no_audio_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "short-silent.mp4"
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
                    "testsrc=size=160x90:rate=10",
                    "-t",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:v",
                    "libx264",
                    str(source_path),
                ],
                check=True,
            )
            processor = FFmpegMediaProcessor(keyframe_count=3)
            source = LocalFileCamera("cam_02", source_path).source()
            self.assertFalse(processor.probe(source).has_audio)
            media = processor.prepare(
                source=source,
                candidate=CandidateEvent("event_02", "cam_02", 0.5, 5.0, ()),
                occurred_at=datetime(2026, 8, 1, tzinfo=UTC),
                storage=StorageResolver(
                    StorageSettings(
                        local_root=root / "output",
                        database=root / "events.sqlite3",
                    )
                ),
            )
            self.assertTrue(Path(media.clip_path).is_file())
            self.assertEqual(3, len(media.frame_paths))
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in media.frame_paths))

    def test_damaged_input_fails_before_media_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "damaged.mp4"
            source_path.write_bytes(b"not a video")
            output_root = root / "output"
            with self.assertRaises(MediaError):
                FFmpegMediaProcessor(keyframe_count=3).prepare(
                    source=LocalFileCamera("cam_03", source_path).source(),
                    candidate=CandidateEvent("event_03", "cam_03", 0.0, 1.0, ()),
                    occurred_at=datetime(2026, 8, 1, tzinfo=UTC),
                    storage=StorageResolver(
                        StorageSettings(
                            local_root=output_root,
                            database=root / "events.sqlite3",
                        )
                    ),
                )
            self.assertFalse((output_root / "events").exists())


if __name__ == "__main__":
    unittest.main()
