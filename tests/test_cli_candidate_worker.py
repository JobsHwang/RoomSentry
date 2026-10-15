from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from baby_monitor.candidate_health import CandidateHealthState, CandidateWorkerHealth
from baby_monitor.cli import main
from tests import test_candidate_worker as candidate_fixtures


class CandidateWorkerCLITests(unittest.TestCase):
    @staticmethod
    def _config(path: Path, root: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "[storage]",
                    f'local_root = "{(root / "media").as_posix()}"',
                    f'database = "{(root / "events.sqlite3").as_posix()}"',
                    "[detector]",
                    'backend = "mock"',
                    "candidate_gap_seconds = 0.1",
                    "[media]",
                    "pre_event_seconds = 0.5",
                    "post_event_seconds = 0.5",
                    "keyframe_count = 3",
                    "[candidate_worker]",
                    "max_segments_per_run = 2",
                    "idle_exit_seconds = 0",
                    "max_runtime_seconds = 5",
                    "poll_seconds = 0.05",
                    "max_event_span_seconds = 10",
                    "candidate_start_confidence = 0.35",
                    "max_pending_detections = 20",
                    "detector_timeout_seconds = 5",
                ]
            ),
            encoding="utf-8",
        )

    def test_mock_requires_explicit_flag_then_runs_and_restarts_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _segments = candidate_fixtures.CandidateWorkerTests._manifest(
                root / "buffer",
                2,
            )
            manifest.parent.chmod(0o700)
            config = root / "config.toml"
            state = root / "candidate-state.json"
            database = root / "events.sqlite3"
            self._config(config, root)
            base_args = [
                "candidate-worker",
                "--config",
                str(config),
                "--manifest",
                str(manifest),
                "--state",
                str(state),
                "--database",
                str(database),
                "--detector",
                "mock",
                "--max-segments",
                "2",
            ]

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                denied = main(base_args)
            self.assertEqual(2, denied)
            self.assertIn("--allow-fixture-detector", stderr.getvalue())
            self.assertFalse(state.exists())

            output = io.StringIO()
            with (
                patch(
                    "baby_monitor.candidate_worker.extract_buffered_evidence",
                    side_effect=candidate_fixtures.CandidateWorkerTests._fake_extract,
                ),
                redirect_stdout(output),
            ):
                code = main([*base_args, "--allow-fixture-detector"])
            payload = json.loads(output.getvalue())
            self.assertEqual(0, code)
            self.assertEqual("mock", payload["detector"])
            self.assertTrue(payload["detector_identity"].startswith("mock_v1_"))
            self.assertTrue(payload["scanner_identity"].startswith("candidate_v1_"))
            self.assertEqual(2, payload["worker"]["scanned_segments"])
            self.assertEqual(2, payload["worker"]["emitted_events"])
            self.assertEqual(0, payload["worker"]["suppressed_candidate_groups"])
            self.assertEqual(0.35, payload["candidate_start_confidence"])
            self.assertEqual(2, payload["queue"]["pending"])
            health_path = manifest.with_name("candidate-health.json")
            health = CandidateWorkerHealth.read_private(health_path)
            self.assertEqual(CandidateHealthState.STOPPED, health.state)
            self.assertEqual(2, health.segments_scanned)
            self.assertEqual(0o600, health_path.stat().st_mode & 0o777)

            output = io.StringIO()
            with (
                patch(
                    "baby_monitor.candidate_worker.extract_buffered_evidence",
                    side_effect=candidate_fixtures.CandidateWorkerTests._fake_extract,
                ),
                redirect_stdout(output),
            ):
                restart_code = main(
                    [
                        *base_args,
                        "--allow-fixture-detector",
                        "--idle-exit-seconds",
                        "0.05",
                        "--max-runtime-seconds",
                        "1",
                    ]
                )
            restarted = json.loads(output.getvalue())
            self.assertEqual(0, restart_code)
            self.assertEqual("idle_timeout", restarted["worker"]["stop_reason"])
            self.assertEqual(0, restarted["worker"]["scanned_segments"])
            self.assertEqual(2, restarted["queue"]["pending"])

            config.write_text(
                config.read_text(encoding="utf-8").replace(
                    "candidate_gap_seconds = 0.1",
                    "candidate_gap_seconds = 0.2",
                ),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                changed_policy = main(
                    [
                        *base_args,
                        "--allow-fixture-detector",
                        "--idle-exit-seconds",
                        "0.05",
                        "--max-runtime-seconds",
                        "1",
                    ]
                )
            self.assertEqual(2, changed_policy)
            self.assertIn("identity does not match", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
