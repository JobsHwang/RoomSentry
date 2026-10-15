from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
from pathlib import Path
from unittest import mock

from baby_monitor.cli import main
from baby_monitor.daily_digest import DailyDigestConfig, plan_daily_digest
from baby_monitor.daily_digest_export import DailyDigestExportReceipt
from baby_monitor.daily_digest_source import DailyDigestAdmission
from baby_monitor.daily_digest_store import DailyDigestWriteReceipt


class DailyDigestCLITests(unittest.TestCase):
    def test_plan_requires_confirmation_before_media_admission(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch(
                "baby_monitor.cli_commands.home_dashboard.load_daily_digest_candidates"
            ) as load,
            redirect_stderr(stderr),
        ):
            code = main(["daily-digest-plan", "--confirmation", "NO"])
        self.assertEqual(2, code)
        self.assertFalse(load.called)
        self.assertIn("PLAN confirmation", stderr.getvalue())

    def test_plan_public_output_is_path_free_and_does_not_claim_video(self) -> None:
        admission = DailyDigestAdmission(
            candidates=(),
            local_date="2026-08-19",
            source_event_count=0,
            media_files_hashed=0,
            media_bytes_hashed=0,
            coverage_gap_count=0,
        )
        receipt = DailyDigestWriteReceipt(
            local_date="2026-08-19",
            source_event_count=0,
            selected_event_count=0,
            selected_duration_seconds=0.0,
            target_met=False,
            reused_existing=False,
        )
        output = io.StringIO()
        with (
            mock.patch(
                "baby_monitor.cli_commands.home_dashboard.load_daily_digest_candidates",
                return_value=admission,
            ),
            mock.patch(
                "baby_monitor.cli_commands.home_dashboard.write_daily_digest_plan",
                return_value=receipt,
            ),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "daily-digest-plan",
                    "--date",
                    "2026-08-19",
                    "--confirmation",
                    "PLAN",
                ]
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(0, code)
        self.assertEqual("stored", payload["status"])
        self.assertTrue(payload["facts_only"])
        self.assertFalse(payload["media_content_decoded"])
        self.assertFalse(payload["video_generated"])
        self.assertFalse(payload["notification_decision"])
        self.assertFalse(payload["event_ids_disclosed"])
        self.assertFalse(payload["camera_ids_disclosed"])

    def test_status_is_read_only_and_does_not_create_plan_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            database = root / "events.sqlite3"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE marker (value INTEGER NOT NULL)")
            connection.commit()
            connection.close()
            database.chmod(0o600)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "daily-digest-plan-status",
                        "--database",
                        str(database),
                    ]
                )
            payload = json.loads(output.getvalue())
            self.assertEqual(0, code)
            self.assertEqual("not_generated", payload["status"])
            self.assertFalse((root / "daily-digests").exists())
            self.assertNotIn(str(root), output.getvalue())

    def test_export_requires_confirmation_before_plan_or_media_read(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch(
                "baby_monitor.cli_commands.home_dashboard.load_latest_daily_digest_plan"
            ) as load_plan,
            mock.patch(
                "baby_monitor.cli_commands.home_dashboard.load_daily_digest_candidates"
            ) as load_media,
            redirect_stderr(stderr),
        ):
            code = main(["daily-digest-export", "--confirmation", "NO"])
        self.assertEqual(2, code)
        self.assertFalse(load_plan.called)
        self.assertFalse(load_media.called)
        self.assertIn("EXPORT confirmation", stderr.getvalue())

    def test_export_public_output_is_path_free_and_explicit_about_decode(self) -> None:
        plan = plan_daily_digest(
            (),
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
            config=DailyDigestConfig(target_duration_seconds=2.0),
        )
        admission = DailyDigestAdmission(
            candidates=(),
            local_date="2026-08-19",
            source_event_count=0,
            media_files_hashed=0,
            media_bytes_hashed=0,
            coverage_gap_count=0,
        )
        receipt = DailyDigestExportReceipt(
            local_date="2026-08-19",
            selected_event_count=1,
            selected_duration_seconds=2.0,
            output_duration_seconds=2.0,
            output_bytes=1234,
            reused_existing=False,
        )
        output = io.StringIO()
        with (
            mock.patch(
                "baby_monitor.cli_commands.home_dashboard.load_latest_daily_digest_plan",
                return_value=plan,
            ),
            mock.patch(
                "baby_monitor.cli_commands.home_dashboard.load_daily_digest_candidates",
                return_value=admission,
            ),
            mock.patch(
                "baby_monitor.cli_commands.home_dashboard.export_daily_digest", return_value=receipt
            ),
            redirect_stdout(output),
        ):
            code = main(
                [
                    "daily-digest-export",
                    "--date",
                    "2026-08-19",
                    "--confirmation",
                    "EXPORT",
                ]
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(0, code)
        self.assertTrue(payload["media_content_decoded"])
        self.assertTrue(payload["video_generated"])
        self.assertFalse(payload["external_delivery"])
        self.assertFalse(payload["paths_disclosed"])
        self.assertFalse(payload["event_ids_disclosed"])


if __name__ == "__main__":
    unittest.main()
