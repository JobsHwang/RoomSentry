from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from baby_monitor.cli import main


class RemoteCLITests(unittest.TestCase):
    def test_remote_preview_is_path_free_and_sends_no_request(self) -> None:
        with tempfile.TemporaryDirectory(prefix="PRIVATE_PREVIEW_ROOT_") as directory:
            root = Path(directory)
            config = root / "remote.toml"
            config.write_text(
                "\n".join(
                    [
                        "[remote_vlm]",
                        'base_url = "http://127.0.0.1:18081/v1"',
                        'model = "local-fixture"',
                        "max_frames = 3",
                    ]
                ),
                encoding="utf-8",
            )
            frames = []
            for index in range(5):
                frame = root / f"private_{index}.jpg"
                frame.write_bytes(f"fixture-{index}".encode("ascii"))
                frames.append(frame)
            argv = ["remote-preview", "--config", str(config)]
            for frame in frames:
                argv.extend(["--frame", str(frame)])
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(argv)

            self.assertEqual(0, code)
            payload = json.loads(output.getvalue())
            self.assertFalse(payload["network_request_sent"])
            self.assertEqual(3, payload["preview"]["selected_frame_count"])
            self.assertEqual(2, payload["preview"]["omitted_frame_count"])
            self.assertFalse(payload["preview"]["includes_local_paths"])
            self.assertNotIn(str(root), output.getvalue())

    def test_remote_audit_status_does_not_create_a_missing_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "missing" / "events.sqlite3"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(["remote-audit-status", "--database", str(database)])
            self.assertEqual(2, code)
            self.assertFalse(database.exists())
            self.assertEqual("", stdout.getvalue())
            self.assertIn("does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
