from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path

from baby_monitor.daily_digest import plan_daily_digest
from baby_monitor.daily_digest_store import (
    DailyDigestStoreError,
    inspect_daily_digest_plans,
    load_daily_digest_plan,
    load_latest_daily_digest_plan,
    write_daily_digest_plan,
)


class DailyDigestStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o700)
        self.database = self.root / "events.sqlite3"
        connection = sqlite3.connect(self.database)
        connection.execute("CREATE TABLE marker (value INTEGER NOT NULL)")
        connection.commit()
        connection.close()
        self.database.chmod(0o600)
        self.output = self.root / "daily-digests"
        self.plan = plan_daily_digest(
            (),
            local_date=date(2026, 8, 19),
            timezone_name="Asia/Shanghai",
        )

    def _write(self):
        return write_daily_digest_plan(
            self.plan,
            database_path=self.database,
            output_root=self.output,
        )

    def test_plan_is_atomic_private_idempotent_and_public_receipt_is_path_free(self) -> None:
        first = self._write()
        second = self._write()
        self.assertFalse(first.reused_existing)
        self.assertTrue(second.reused_existing)
        self.assertEqual(0o700, self.output.stat().st_mode & 0o777)
        files = list(self.output.iterdir())
        self.assertEqual(1, len(files))
        self.assertEqual(0o600, files[0].stat().st_mode & 0o777)
        public = second.to_public_dict()
        self.assertFalse(public["media_read"])
        self.assertFalse(public["video_generated"])
        self.assertNotIn(str(self.root), json.dumps(public))

        status = inspect_daily_digest_plans(
            database_path=self.database,
            output_root=self.output,
        )
        self.assertEqual("available", status["status"])
        self.assertEqual(1, status["plan_count"])
        self.assertEqual(0, status["latest_selected_event_count"])
        self.assertNotIn(str(self.root), json.dumps(status))
        loaded = load_latest_daily_digest_plan(
            database_path=self.database,
            output_root=self.output,
            local_date="2026-08-19",
        )
        self.assertEqual(self.plan, loaded)
        self.assertEqual(
            self.plan,
            load_daily_digest_plan(
                database_path=self.database,
                output_root=self.output,
                local_date=self.plan.local_date,
                digest_id=self.plan.digest_id,
            ),
        )
        with self.assertRaisesRegex(DailyDigestStoreError, "plan store failed"):
            load_latest_daily_digest_plan(
                database_path=self.database,
                output_root=self.output,
                local_date="2026-08-18",
            )

    def test_status_does_not_create_output(self) -> None:
        status = inspect_daily_digest_plans(
            database_path=self.database,
            output_root=self.output,
        )
        self.assertEqual("not_generated", status["status"])
        self.assertFalse(self.output.exists())

    def test_tampered_existing_plan_and_unknown_file_fail_closed(self) -> None:
        self._write()
        path = next(self.output.iterdir())
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        path.chmod(0o600)
        with self.assertRaisesRegex(DailyDigestStoreError, "plan store failed"):
            self._write()
        self.assertEqual(
            "unavailable",
            inspect_daily_digest_plans(
                database_path=self.database,
                output_root=self.output,
            )["status"],
        )

        path.unlink()
        unknown = self.output / "unknown.json"
        unknown.write_text("{}", encoding="utf-8")
        unknown.chmod(0o600)
        self.assertEqual(
            "unavailable",
            inspect_daily_digest_plans(
                database_path=self.database,
                output_root=self.output,
            )["status"],
        )

    def test_outside_weak_and_symlink_targets_are_rejected(self) -> None:
        with self.assertRaisesRegex(DailyDigestStoreError, "plan store failed"):
            write_daily_digest_plan(
                self.plan,
                database_path=self.database,
                output_root=self.root / "outside",
            )

        self.output.mkdir(mode=0o755)
        with self.assertRaisesRegex(DailyDigestStoreError, "plan store failed"):
            self._write()
        self.output.chmod(0o700)
        self.output.rmdir()

        target = self.root / "target"
        target.mkdir(mode=0o700)
        self.output.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(DailyDigestStoreError, "plan store failed"):
            self._write()


if __name__ == "__main__":
    unittest.main()
