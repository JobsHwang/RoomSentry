from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from baby_monitor.database import SCHEMA_SQL, EventDatabase


class EventDatabaseMigrationTests(unittest.TestCase):
    def test_initialize_adds_decision_column_to_pre_decision_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            old_schema = (
                SCHEMA_SQL.replace("    secondary_result_json TEXT,\n", "")
                .replace("    decision_json TEXT,\n", "")
                .replace("    evidence_json TEXT,\n", "")
                .replace("    local_provenance_json TEXT,\n", "")
                .replace("    secondary_provenance_json TEXT,\n", "")
                .replace("    remote_provenance_json TEXT,\n", "")
                .replace("    local_metrics_json TEXT,\n", "")
                .replace("    secondary_metrics_json TEXT,\n", "")
                .replace("    remote_metrics_json TEXT,\n", "")
            )
            connection = sqlite3.connect(path)
            try:
                connection.executescript(old_schema)
                connection.commit()
            finally:
                connection.close()

            EventDatabase(path).initialize()
            connection = sqlite3.connect(path)
            try:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(events)").fetchall()
                }
                presence_tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
            finally:
                connection.close()
            self.assertIn("decision_json", columns)
            self.assertIn("secondary_result_json", columns)
            self.assertIn("evidence_json", columns)
            self.assertIn("local_provenance_json", columns)
            self.assertIn("secondary_provenance_json", columns)
            self.assertIn("remote_provenance_json", columns)
            self.assertIn("local_metrics_json", columns)
            self.assertIn("secondary_metrics_json", columns)
            self.assertIn("remote_metrics_json", columns)
            self.assertIn("presence_facts", presence_tables)


if __name__ == "__main__":
    unittest.main()
