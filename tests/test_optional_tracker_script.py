from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from baby_monitor.local_runners.tracker_replay import TrackerReplayConfig, replay_tracker
from scripts.household_debug_common import write_new
from scripts.replay_optional_trackers import main, run
from tests import test_tracker_replay as fixtures


class OptionalTrackerScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.batch = self.root / "batch.local.json"
        self.config = self.root / "config.local.json"
        self.output = self.root / "output"
        self.fixture = fixtures.TrackerReplayTests()
        write_new(self.batch, self.fixture.batch().to_dict())
        write_new(self.config, asdict(TrackerReplayConfig("bytetrack", 384, 288)))

    def replay(self, batch, config):
        factory, _, _ = self.fixture.factory()
        return replay_tracker(batch, config, _backend_factory=factory)

    def test_success_is_bound_and_private_and_contains_no_source_path(self):
        with patch("scripts.replay_optional_trackers.replay_tracker", side_effect=self.replay):
            summary = run(self.batch, self.config, self.output)
        self.assertEqual("completed_offline_replay", summary["status"])
        self.assertEqual(2, summary["assignments"])
        self.assertFalse(summary["deployment_approved"])
        self.assertEqual(0o700, self.output.stat().st_mode & 0o777)
        for file in self.output.iterdir():
            self.assertEqual(0o600, file.stat().st_mode & 0o777)
            self.assertNotIn(str(self.root), file.read_text())
        protocol = json.loads((self.output / "protocol.local.json").read_text())
        self.assertTrue(protocol["inputs_unchanged_before_after"])
        self.assertFalse(protocol["same_uid_concurrent_mutation_prevented"])
        self.assertFalse(protocol["image_or_video_input"])
        self.assertEqual("injected_stdlib_test_backend", protocol["runtime_id"])

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir(mode=0o700)
        with self.assertRaises(ValueError):
            run(self.batch, self.config, self.output)

    def test_shared_or_symlink_inputs_fail_before_backend(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink):
                path = self.batch
                if symlink:
                    path = self.root / "symlink"
                    path.symlink_to(self.batch)
                    self.batch.chmod(0o600)
                else:
                    self.batch.chmod(0o644)
                with patch("scripts.replay_optional_trackers.replay_tracker") as runner:
                    with self.assertRaises(ValueError):
                        run(path, self.config, self.output)
                    runner.assert_not_called()

    def test_configuration_must_freeze_all_parameters(self):
        incomplete = self.root / "incomplete.json"
        write_new(incomplete, {"algorithm": "bytetrack", "canvas_width": 384, "canvas_height": 288})
        with patch("scripts.replay_optional_trackers.replay_tracker") as runner:
            with self.assertRaises(ValueError):
                run(self.batch, incomplete, self.output)
            runner.assert_not_called()

    def test_code_change_fails_without_publishing_outputs(self):
        with (
            patch("scripts.replay_optional_trackers.replay_tracker", side_effect=self.replay),
            patch(
                "scripts.replay_optional_trackers.code_hashes", side_effect=[{"a": "1"}, {"a": "2"}]
            ),
            self.assertRaises(ValueError),
        ):
            run(self.batch, self.config, self.output)
        self.assertFalse(self.output.exists())

    def test_changed_inputs_fail_without_publishing_outputs(self):
        from scripts.replay_optional_trackers import read_private_json_with_sha

        def read(path):
            value, sha = read_private_json_with_sha(path)
            if path == self.batch and self.read_count:
                return value, "0" * 64
            if path == self.batch:
                self.read_count += 1
            return value, sha

        self.read_count = 0
        with (
            patch("scripts.replay_optional_trackers.replay_tracker", side_effect=self.replay),
            patch("scripts.replay_optional_trackers.read_private_json_with_sha", side_effect=read),
            self.assertRaises(ValueError),
        ):
            run(self.batch, self.config, self.output)
        self.assertFalse(self.output.exists())

    def test_cli_errors_are_path_free(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(
                [
                    "--batch",
                    str(self.root / "private-missing"),
                    "--config",
                    str(self.config),
                    "--output",
                    str(self.output),
                ]
            )
        self.assertEqual(1, result)
        self.assertEqual({"status": "offline_tracker_replay_failed"}, json.loads(output.getvalue()))
        self.assertNotIn(str(self.root), output.getvalue())


if __name__ == "__main__":
    unittest.main()
