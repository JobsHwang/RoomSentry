from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from baby_monitor.cli import main
from baby_monitor.rtsp_soak import RTSPSoakResult


class RTSPSoakCLITests(unittest.TestCase):
    def _result(self) -> RTSPSoakResult:
        return RTSPSoakResult(
            ok=True,
            stream="rtsp://127.0.0.1:8554/test",
            started_at="2026-08-01T00:00:00+00:00",
            requested_seconds=3,
            elapsed_seconds=3.1,
            frames=30,
            media_seconds=2.9,
            speed=1.0,
            ffmpeg_peak_rss_mb=64.0,
            ffmpeg_exit_code=0,
            progress_end=True,
            error_type=None,
        )

    def test_output_is_created_once_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-soak-cli-") as directory:
            output = Path(directory) / "nested/result.json"
            arguments = [
                "rtsp-soak",
                "--uri",
                "rtsp://127.0.0.1:8554/test",
                "--seconds",
                "3",
                "--output",
                str(output),
            ]
            with patch(
                "baby_monitor.cli_commands.media_pipeline.run_rtsp_soak",
                return_value=self._result(),
            ):
                with redirect_stdout(StringIO()):
                    first_code = main(arguments)
                original = json.loads(output.read_text(encoding="utf-8"))
                error = StringIO()
                with redirect_stderr(error):
                    second_code = main(arguments)
            self.assertEqual(0, first_code)
            self.assertEqual(2, second_code)
            self.assertEqual(30, original["frames"])
            self.assertIn("File exists", error.getvalue())


if __name__ == "__main__":
    unittest.main()
