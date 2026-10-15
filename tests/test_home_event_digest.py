from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from baby_monitor.domain import DecisionRecord, FinalSource, ReviewTier, RiskLevel
from baby_monitor.home_event_digest import (
    HomeEventDigestError,
    read_home_event_digest,
)

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


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(_SCHEMA)
    return connection


class HomeEventDigestTests(unittest.TestCase):
    def _decision_database(self, entries):
        temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-home-decision-")
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "events.sqlite3"
        connection = _database(path)
        connection.execute("ALTER TABLE events ADD COLUMN decision_json TEXT")
        connection.execute("ALTER TABLE events ADD COLUMN summary TEXT")
        for index, (legacy_remote, decision_json) in enumerate(entries):
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"private_event_{index}",
                    f"2026-08-18T01:{index:02d}:00+00:00",
                    "unknown",
                    "none",
                    0.5,
                    legacy_remote,
                    decision_json,
                    "private summary must not be read",
                ),
            )
        connection.commit()
        connection.close()
        path.chmod(0o600)
        return path

    @staticmethod
    def _decision(*, tier=ReviewTier.LOCAL_SECONDARY, source=FinalSource.LOCAL_PRIMARY):
        return DecisionRecord(
            risk_level=RiskLevel.NONE,
            review_tier=tier,
            review_reasons=("uncertain_scene",),
            contradictions=(),
            remote_allowed=tier is ReviewTier.REMOTE,
            remote_used=source is FinalSource.REMOTE,
            final_source=source,
        ).to_json()

    def test_local_pending_is_visible_and_completed_secondary_is_not_pending(self):
        path = self._decision_database(
            (
                (0, self._decision()),
                (0, self._decision(source=FinalSource.LOCAL_SECONDARY)),
            )
        )
        before = path.read_bytes(), path.stat().st_mtime_ns
        result = read_home_event_digest(
            path, now=datetime(2026, 8, 18, 3, tzinfo=UTC), recent_day_only=True
        )
        self.assertEqual([False, True], [item.needs_review for item in result.recent])
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
        public = json.dumps(result.to_public_dict())
        self.assertNotIn("uncertain_scene", public)
        self.assertNotIn("private summary", public)
        self.assertNotIn("private_event", public)

    def test_remote_decision_and_legacy_flags_are_combined_without_clearing_old_flags(self):
        path = self._decision_database(
            (
                (0, self._decision(tier=ReviewTier.REMOTE)),
                (0, self._decision(tier=ReviewTier.REMOTE, source=FinalSource.REMOTE)),
                (1, self._decision(source=FinalSource.LOCAL_SECONDARY)),
                (1, None),
                (0, None),
            )
        )
        result = read_home_event_digest(path, now=datetime(2026, 8, 18, 3, tzinfo=UTC))
        self.assertEqual(
            [False, True, True, False, True], [item.needs_review for item in result.recent]
        )

    def test_invalid_decision_json_fails_with_path_free_error_even_if_legacy_flag_true(self):
        valid = self._decision()
        duplicate = valid.replace('"version": 1', '"version": 1, "version": 1', 1)
        for payload in (
            "private/path",
            "{}",
            duplicate,
            "",
            b"private_binary",
            "[" * 3000,
            valid + "\x00garbage",
        ):
            with self.subTest(kind=type(payload)):
                path = self._decision_database(((1, payload),))
                with self.assertRaisesRegex(HomeEventDigestError, "decision is invalid") as error:
                    read_home_event_digest(path, now=datetime(2026, 8, 18, 3, tzinfo=UTC))
                self.assertNotIn("private", str(error.exception))
                self.assertNotIn(str(path), str(error.exception))
                self.assertTrue(
                    error.exception.__suppress_context__ or error.exception.__context__ is None
                )

    def test_oversized_decision_is_rejected_before_parser_and_bounded_in_sql(self):
        # Valid JSON plus whitespace is still too large. Non-ASCII text is
        # measured in UTF-8 bytes, not merely its smaller character count.
        for payload in (self._decision() + " " * (64 * 1024), "中" * 22000):
            with self.subTest(non_ascii=payload.startswith("中")):
                path = self._decision_database(((0, payload),))
                with (
                    patch("baby_monitor.home_event_digest.DecisionRecord.from_json") as parse,
                    self.assertRaisesRegex(HomeEventDigestError, "decision is invalid"),
                ):
                    read_home_event_digest(path, now=datetime(2026, 8, 18, 3, tzinfo=UTC))
                parse.assert_not_called()

    def test_only_recent_limit_decisions_are_parsed_and_summaries_are_never_read(self):
        path = self._decision_database(
            ((0, "old invalid private decision"),) + ((0, self._decision()),) * 16
        )
        connect = sqlite3.connect
        original_parser = DecisionRecord.from_json

        def restricted_connection(*args, **kwargs):
            connection = connect(*args, **kwargs)
            connection.set_authorizer(
                lambda action, _table, column, _database, _trigger: (
                    sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_READ and column in {"summary", "event_id"}
                    else sqlite3.SQLITE_OK
                )
            )
            return connection

        with (
            patch(
                "baby_monitor.home_event_digest.sqlite3.connect", side_effect=restricted_connection
            ),
            patch(
                "baby_monitor.home_event_digest.DecisionRecord.from_json", wraps=original_parser
            ) as parse,
        ):
            result = read_home_event_digest(
                path, now=datetime(2026, 8, 18, 3, tzinfo=UTC), recent_limit=16
            )
        self.assertEqual(17, result.today_total)
        self.assertEqual(16, len(result.recent))
        self.assertEqual(16, parse.call_count)
        self.assertTrue(all(item.needs_review for item in result.recent))

    def test_daily_counts_and_recent_timeline_are_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            connection = _database(path)
            connection.executemany(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ("private_event_1", "2026-08-17T16:30:00+00:00", "unknown", "none", 0.4, 1),
                    ("private_event_2", "2026-08-18T01:00:00+00:00", "movement", "low", 0.7, 0),
                    ("private_event_3", "2026-08-16T01:00:00+00:00", "crying", "none", 0.5, 0),
                ),
            )
            connection.commit()
            connection.close()
            path.chmod(0o600)

            digest = read_home_event_digest(
                path,
                now=datetime(2026, 8, 18, 3, tzinfo=UTC),
            ).to_public_dict()

            self.assertEqual("2026-08-18", digest["local_date"])
            self.assertEqual(2, digest["today_total"])
            self.assertEqual({"movement": 1, "unknown": 1}, digest["event_type_counts"])
            self.assertEqual("08-18 09:00:00", digest["recent"][0]["local_time"])
            self.assertFalse(digest["media_read"])
            rendered = json.dumps(digest)
            self.assertNotIn("private_event", rendered)
            self.assertNotIn(str(path), rendered)

    def test_daily_report_scope_excludes_older_recent_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            connection = _database(path)
            connection.executemany(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    (
                        "older_event",
                        "2026-08-18T01:00:00+00:00",
                        "movement",
                        "none",
                        0.4,
                        0,
                    ),
                    (
                        "today_event",
                        "2026-08-19T01:00:00+00:00",
                        "unknown",
                        "none",
                        0.5,
                        1,
                    ),
                ),
            )
            connection.commit()
            connection.close()
            path.chmod(0o600)

            digest = read_home_event_digest(
                path,
                now=datetime(2026, 8, 19, 12, tzinfo=UTC),
                recent_day_only=True,
            )

            self.assertEqual(1, digest.today_total)
            self.assertEqual(1, len(digest.recent))
            self.assertTrue(digest.recent[0].local_time.startswith("08-19 "))

    def test_missing_or_invalid_database_fails_without_creation_or_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.sqlite3"
            with self.assertRaisesRegex(HomeEventDigestError, "unavailable"):
                read_home_event_digest(missing)
            self.assertFalse(missing.exists())

            invalid = root / "invalid.sqlite3"
            connection = _database(invalid)
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                ("event", "2026-08-18T01:00:00+00:00", "invented", "none", 0.5, 0),
            )
            connection.commit()
            connection.close()
            invalid.chmod(0o600)
            with self.assertRaisesRegex(HomeEventDigestError, "invalid"):
                read_home_event_digest(invalid, now=datetime(2026, 8, 18, 3, tzinfo=UTC))

    @unittest.skipUnless(os.name == "posix", "POSIX symlink policy required")
    def test_symlink_database_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "events.sqlite3"
            _database(target).close()
            target.chmod(0o600)
            link = root / "link.sqlite3"
            link.symlink_to(target)
            with self.assertRaisesRegex(HomeEventDigestError, "unavailable"):
                read_home_event_digest(link)

    @unittest.skipUnless(os.name == "posix", "POSIX private mode policy required")
    def test_group_or_world_accessible_database_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.sqlite3"
            _database(path).close()
            path.chmod(0o640)
            with self.assertRaisesRegex(HomeEventDigestError, "unavailable"):
                read_home_event_digest(path)


if __name__ == "__main__":
    unittest.main()
