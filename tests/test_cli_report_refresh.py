from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from baby_monitor.cli import build_parser, main
from baby_monitor.database import EventDatabase
from baby_monitor.report_queue import ReportRefreshQueue


class ReportRefreshCLITests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-report-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.database = self.root / "events.sqlite3"
        self.config = self.root / "fixture.toml"
        self.config.write_text('[general]\ntimezone = "Asia/Shanghai"\n', encoding="utf-8")
        self.output_root = self.root / "output"
        self.output_root.mkdir(mode=0o700)

    def _run(self, command: str, *arguments: str):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(
                [
                    command,
                    "--config",
                    str(self.config),
                    "--database",
                    str(self.database),
                    *arguments,
                ]
            )
        self.assertNotIn(str(self.root), stdout.getvalue())
        self.assertNotIn(str(self.root), stderr.getvalue())
        payload = json.loads(stdout.getvalue()) if stdout.getvalue() else None
        return code, payload, stderr.getvalue()

    def _request(self, request_id="manual_fixture"):
        return self._run(
            "report-refresh-request",
            "--date",
            "2026-08-03",
            "--output-root",
            str(self.output_root),
            "--request-id",
            request_id,
            "--debounce-seconds",
            "0",
        )

    def test_read_only_status_does_not_create_missing_or_migrate_legacy_database(self) -> None:
        code, payload, error = self._run("report-refresh-status")
        self.assertEqual((0, ""), (code, error))
        self.assertFalse(payload["table_present"])
        self.assertFalse(self.database.exists())
        EventDatabase(self.database).initialize()
        before = self.database.read_bytes()
        code, payload, error = self._run("report-refresh-status")
        self.assertEqual((0, ""), (code, error))
        self.assertFalse(payload["table_present"])
        self.assertEqual(before, self.database.read_bytes())

    def test_request_is_idempotent_and_run_once_publishes_without_private_paths(self) -> None:
        code, first, error = self._request()
        self.assertEqual((0, ""), (code, error))
        self.assertEqual("recorded", first["status"])
        self.assertFalse((self.output_root / "reports").exists())
        self.assertEqual(1, self._request()[1]["report"]["requested_revision"])
        code, result, error = self._run("report-refresh-run-once")
        self.assertEqual((0, ""), (code, error))
        self.assertEqual("completed", result["status"])
        self.assertEqual(0, result["event_count"])
        self.assertTrue(result["processed"])
        self.assertEqual(1, result["rendered_revision"])
        self.assertEqual(1, len(list(self.output_root.rglob("*.md"))))
        self.assertEqual(1, len(list(self.output_root.rglob("*.html"))))
        self.assertEqual("idle", self._run("report-refresh-run-once")[1]["status"])
        self.assertEqual(1, self._run("report-refresh-status")[1]["counts"]["completed"])

    def test_failure_and_explicit_retry_do_not_repeat_analysis(self) -> None:
        _code, requested, _error = self._request()
        job_id = requested["report"]["job_id"]
        with patch(
            "baby_monitor.report_service.DailyReportGenerator.generate",
            side_effect=OSError("private-render-diagnostics"),
        ):
            code, failed, error = self._run("report-refresh-run-once", "--max-attempts", "1")
        self.assertEqual(2, code)
        self.assertEqual("", error)
        self.assertEqual("failed", failed["status"])
        self.assertEqual("OSError", failed["error_code"])
        self.assertNotIn("private-render-diagnostics", repr(failed))
        code, retried, error = self._run("report-refresh-retry", "--job-id", job_id)
        self.assertEqual((0, ""), (code, error))
        self.assertEqual(0, retried["report"]["attempt_count"])
        code, rendered, error = self._run("report-refresh-worker", "--max-jobs", "1")
        self.assertEqual((0, ""), (code, error))
        self.assertEqual(1, rendered["worker"]["completed_jobs"])
        self.assertEqual(job_id, rendered["worker"]["last_job_id"])
        self.assertNotIn("last_event_id", rendered["worker"])
        self.assertEqual([], EventDatabase(self.database).list_recent())

    def test_idle_worker_is_bounded_and_unbounded_worker_is_rejected(self) -> None:
        code, result, error = self._run(
            "report-refresh-worker",
            "--idle-exit-seconds",
            "0.02",
            "--poll-seconds",
            "0.01",
        )
        self.assertEqual((0, ""), (code, error))
        self.assertEqual("idle_timeout", result["worker"]["stop_reason"])
        code, result, error = self._run(
            "report-refresh-worker",
            "--max-jobs",
            "0",
            "--idle-exit-seconds",
            "0",
            "--max-runtime-seconds",
            "0",
        )
        self.assertEqual(2, code)
        self.assertIsNone(result)
        self.assertIn("stopping bound", error)

    def test_invalid_destination_and_request_identifier_fail_before_database_creation(self) -> None:
        code, result, error = self._request("not a valid identifier")
        self.assertEqual(2, code)
        self.assertIsNone(result)
        self.assertIn("identifier is invalid", error)
        self.assertFalse(self.database.exists())
        self.assertFalse(ReportRefreshQueue(self.database).snapshot_read_only()["table_present"])

    def test_analysis_deferred_mode_is_explicit_and_defaults_remain_synchronous(self) -> None:
        parser = build_parser()
        for command in ("analysis-run-once", "analysis-worker"):
            with self.subTest(command=command):
                self.assertFalse(parser.parse_args([command]).defer_reports)
                self.assertTrue(parser.parse_args([command, "--defer-reports"]).defer_reports)


if __name__ == "__main__":
    unittest.main()
