from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from baby_monitor.cli import main
from baby_monitor.config import Settings
from baby_monitor.pipeline import process_prepared_event
from tests import test_pipeline_evidence_binding as binding_fixtures


class LocalReviewCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.arguments = binding_fixtures.PipelineEvidenceBindingTests()._arguments(self.root)
        process_prepared_event(**self.arguments)
        self.database = self.arguments["database"].path
        original = Settings()
        self.settings = replace(
            original,
            storage=replace(original.storage, local_root=self.root),
            secondary_vlm=replace(original.secondary_vlm, enabled=True),
            analysis_worker=replace(original.analysis_worker, gpu_guard_enabled=False),
        )

    def _call(self, command, *extra):
        output, errors = io.StringIO(), io.StringIO()
        with (
            patch(
                "baby_monitor.cli_commands.local_review.load_settings", return_value=self.settings
            ),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            code = main([command, "--database", str(self.database), *extra])
        return code, output.getvalue(), errors.getvalue()

    def test_retry_and_status_are_idempotent_and_path_free(self):
        options = ("--event-id", self.arguments["candidate"].event_id, "--request-id", "retry_01")
        first, output, errors = self._call("local-review-retry", *options)
        self.assertEqual(0, first, errors)
        self.assertEqual("succeeded", json.loads(output)["status"])
        self.assertEqual("queued", json.loads(output)["report_refresh"]["status"])
        self.assertFalse(json.loads(output)["reports_refreshed"])
        with patch(
            "baby_monitor.cli_commands.local_review._secondary_adapter",
            side_effect=AssertionError("idempotent request must not construct a model"),
        ):
            code, replay, errors = self._call("local-review-retry", *options)
        self.assertEqual(0, code, errors)
        self.assertTrue(json.loads(replay)["reused_existing"])
        code, status, errors = self._call("local-review-status")
        self.assertEqual(0, code, errors)
        self.assertEqual(1, json.loads(status)["returned"])
        self.assertEqual("queued", json.loads(status)["attempts"][0]["report_refresh"]["status"])
        for text in (output, replay, status, errors):
            self.assertNotIn(str(self.root), text)
            self.assertNotIn("synthetic.mp4", text)

    def test_disabled_secondary_does_not_silently_enable_model(self):
        self.settings = replace(
            self.settings, secondary_vlm=replace(self.settings.secondary_vlm, enabled=False)
        )
        with patch(
            "baby_monitor.cli_commands.local_review._secondary_adapter",
            side_effect=AssertionError("model must not be constructed"),
        ):
            code, output, errors = self._call(
                "local-review-retry", "--event-id", "binding_event_fixture", "--request-id", "r1"
            )
        self.assertEqual(2, code)
        self.assertEqual("", output)
        self.assertIn("secondary_vlm.enabled", errors)

    def test_changed_media_failure_and_same_request_never_retry(self):
        Path(self.arguments["media"].clip_path).write_bytes(b"changed synthetic asset")
        options = ("--event-id", "binding_event_fixture", "--request-id", "r1")
        with patch(
            "baby_monitor.cli_commands.local_review._secondary_adapter",
            side_effect=AssertionError("changed evidence must not construct a model"),
        ):
            code, output, errors = self._call("local-review-retry", *options)
            repeated_code, repeated, _ = self._call("local-review-retry", *options)
        self.assertEqual(1, code, errors)
        self.assertEqual("evidence_changed", json.loads(output)["error_code"])
        self.assertEqual(1, repeated_code)
        self.assertTrue(json.loads(repeated)["reused_existing"])

    def test_status_on_existing_database_does_not_create_review_schema(self):
        import sqlite3
        from contextlib import closing

        before = self.database.read_bytes()
        code, output, errors = self._call("local-review-status")
        self.assertEqual(0, code, errors)
        self.assertEqual([], json.loads(output)["attempts"])
        self.assertEqual(before, self.database.read_bytes())
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='local_review_attempts'"
                ).fetchone()
            )

    def test_gpu_guard_refuses_before_adapter_construction(self):
        from baby_monitor.gpu_guard import GPUCapacity

        self.settings = replace(
            self.settings,
            analysis_worker=replace(self.settings.analysis_worker, gpu_guard_enabled=True),
        )
        with (
            patch("baby_monitor.cli_commands.local_review.probe_gpu_capacity") as probe,
            patch(
                "baby_monitor.cli_commands.local_review._secondary_adapter",
                side_effect=AssertionError("GPU guard refused"),
            ),
        ):
            probe.return_value = unittest.mock.Mock(spec=GPUCapacity, allowed=False)
            code, output, errors = self._call(
                "local-review-retry", "--event-id", "binding_event_fixture", "--request-id", "r1"
            )
        self.assertEqual(1, code, errors)
        self.assertEqual("model_failed", json.loads(output)["error_code"])

    def test_unexpected_factory_error_is_not_printed_by_cli(self):
        with patch(
            "baby_monitor.cli_commands.local_review._secondary_adapter",
            side_effect=RuntimeError("PRIVATE_SYNTHETIC_DIAGNOSTIC " + str(self.root)),
        ):
            code, output, errors = self._call(
                "local-review-retry", "--event-id", "binding_event_fixture", "--request-id", "r1"
            )
        self.assertEqual(2, code)
        self.assertEqual("", output)
        self.assertNotIn("PRIVATE_SYNTHETIC_DIAGNOSTIC", errors)
        self.assertNotIn(str(self.root), errors)
        self.assertIn("failed unexpectedly", errors)

    def test_report_handoff_failure_recovers_without_repeating_secondary(self):
        from baby_monitor.report_queue import ReportQueueError, ReportRefreshQueue

        options = ("--event-id", "binding_event_fixture", "--request-id", "r1")
        with patch.object(
            ReportRefreshQueue, "request", side_effect=ReportQueueError("PRIVATE_REPORT_DIAGNOSTIC")
        ):
            code, output, errors = self._call("local-review-retry", *options)
        receipt = json.loads(output)
        self.assertEqual(0, code, errors)
        self.assertEqual("succeeded", receipt["status"])
        self.assertEqual("pending", receipt["report_refresh"]["status"])
        self.assertEqual(1, receipt["report_dispatch"]["errors_count"])
        self.assertNotIn("PRIVATE_REPORT_DIAGNOSTIC", output + errors)
        with patch(
            "baby_monitor.cli_commands.local_review._secondary_adapter",
            side_effect=AssertionError("completed review must not run again"),
        ):
            code, output, errors = self._call("local-review-retry", *options)
        self.assertEqual(0, code, errors)
        self.assertTrue(json.loads(output)["reused_existing"])
        self.assertEqual("queued", json.loads(output)["report_refresh"]["status"])


if __name__ == "__main__":
    unittest.main()
