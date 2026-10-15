from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from baby_monitor.cli import main

_SCHEMA = """
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    occurred_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    confidence REAL NOT NULL,
    needs_remote_review INTEGER NOT NULL
)
"""


class HomeDailyReportCLITests(unittest.TestCase):
    def test_cli_generates_path_free_aggregate_without_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            database = root / "events.sqlite3"
            connection = sqlite3.connect(database)
            connection.execute(_SCHEMA)
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "PRIVATE_CLI_EVENT",
                    "2026-08-19T01:00:00+00:00",
                    "movement",
                    "none",
                    0.6,
                    0,
                ),
            )
            connection.commit()
            connection.close()
            database.chmod(0o600)

            output = io.StringIO()
            with (
                mock.patch(
                    "baby_monitor.home_daily_report.datetime",
                    wraps=datetime,
                ) as clock,
                redirect_stdout(output),
            ):
                clock.now.return_value = datetime(2026, 8, 19, 12, tzinfo=UTC)
                code = main(["home-daily-report", "--database", str(database)])

            payload = json.loads(output.getvalue())
            self.assertEqual(0, code)
            self.assertEqual("generated", payload["status"])
            self.assertEqual(1, payload["event_count"])
            self.assertFalse(payload["media_read"])
            self.assertFalse(payload["paths_disclosed"])
            self.assertFalse(payload["notification_decision"])
            self.assertNotIn(str(root), output.getvalue())
            reports = list((root / "daily-reports").iterdir())
            self.assertEqual(2, len(reports))


if __name__ == "__main__":
    unittest.main()
