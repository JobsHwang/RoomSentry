from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.home_notification import (
    HomeNotificationError,
    run_home_local_notification_test,
)
from baby_monitor.notification import NotificationMessage


class HomeNotificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o700)
        self.state = self.root / "state"
        self.state.mkdir(mode=0o700)
        self.database = self.state / "events.sqlite3"
        sqlite3.connect(self.database).close()
        self.database.chmod(0o600)
        self.target = self.state / "local-notifications"

    def _run(self):
        return run_home_local_notification_test(
            database_path=self.database,
            target_root=self.target,
            max_pending=4,
            lease_seconds=30,
            retry_delay_seconds=0,
            max_attempts=2,
            test_id="a" * 32,
            now=datetime(2026, 8, 19, 2, 0, tzinfo=UTC),
        )

    def test_local_test_is_persistent_path_free_and_not_an_event_decision(self) -> None:
        receipt = self._run()
        public = receipt.to_public_dict()
        self.assertEqual("delivered", public["status"])
        self.assertTrue(public["test_only"])
        self.assertTrue(public["local_file_written"])
        self.assertFalse(public["external_delivery_used"])
        self.assertFalse(public["event_authority"])
        self.assertFalse(public["risk_authority"])
        self.assertFalse(public["notification_rule_decision"])
        self.assertNotIn(str(self.root), json.dumps(public))

        artifacts = list(self.target.glob("*.json"))
        self.assertEqual(1, len(artifacts))
        self.assertEqual(0o700, self.target.stat().st_mode & 0o777)
        self.assertEqual(0o600, artifacts[0].stat().st_mode & 0o777)
        message = NotificationMessage.from_json(artifacts[0].read_text(encoding="utf-8"))
        self.assertEqual("none", message.risk_level.value)
        self.assertEqual("unknown", message.event_type.value)
        self.assertEqual(("manual_local_notification_test",), message.observations)
        self.assertIn("不代表真实监控事件或风险", message.summary_zh)

        connection = sqlite3.connect(self.database)
        row = connection.execute(
            "SELECT status, sink_kind, sink_id FROM notification_jobs"
        ).fetchone()
        connection.close()
        self.assertEqual(("delivered", "file", "dashboard_test"), row)

    def test_repeat_is_idempotent_and_creates_no_second_artifact(self) -> None:
        first = self._run()
        second = self._run()
        self.assertEqual(first, second)
        self.assertEqual(1, len(list(self.target.glob("*.json"))))

    def test_weak_database_or_target_and_outside_target_fail_closed(self) -> None:
        self.database.chmod(0o644)
        with self.assertRaisesRegex(HomeNotificationError, "database is unavailable"):
            self._run()
        self.database.chmod(0o600)

        self.target.mkdir(mode=0o755)
        with self.assertRaisesRegex(HomeNotificationError, "target is unavailable"):
            self._run()
        self.target.chmod(0o700)

        with self.assertRaisesRegex(HomeNotificationError, "target is unavailable"):
            run_home_local_notification_test(
                database_path=self.database,
                target_root=self.root / "outside",
                max_pending=4,
                lease_seconds=30,
                retry_delay_seconds=0,
                max_attempts=2,
            )


if __name__ == "__main__":
    unittest.main()
