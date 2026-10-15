from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.detector import DetectorError, FFmpegSceneDetector


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class FFmpegSceneDetectorTests(unittest.TestCase):
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

    def test_real_scene_change_produces_non_semantic_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "two-scenes.mp4"
            self._generate_two_scene_video(source_path)
            detections = FFmpegSceneDetector(threshold=0.2).scan(
                LocalFileCamera("cam_01", source_path).source()
            )

            self.assertEqual(1, len(detections))
            self.assertEqual("scene_change", detections[0].label)
            self.assertAlmostEqual(2.0, detections[0].timestamp_seconds, places=1)
            self.assertGreater(detections[0].confidence, 0.2)
            self.assertEqual((0.0, 0.0, 1.0, 1.0), detections[0].bbox_xyxy)

    def test_damaged_input_fails_without_exposing_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "private-name.mp4"
            source_path.write_bytes(b"not a video")
            with self.assertRaises(DetectorError) as raised:
                FFmpegSceneDetector().scan(LocalFileCamera("cam_01", source_path).source())
            self.assertNotIn(source_path.name, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
