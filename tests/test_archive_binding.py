from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

from baby_monitor.archive import ArchiveError
from baby_monitor.archive_binding import (
    ArchiveBindingError,
    build_event_archive_manifest,
    build_report_archive_manifest,
)
from baby_monitor.archive_queue import ArchiveQueue
from baby_monitor.cli import main
from baby_monitor.database import EventDatabase


class ArchiveBindingTests(unittest.TestCase):
    @staticmethod
    def _smoke_spool(root: Path) -> tuple[Path, Path, str]:
        source = root / "spool"
        with redirect_stdout(io.StringIO()):
            code = main(["smoke-demo", "--output-root", str(source)])
        if code != 0:
            raise AssertionError("smoke fixture failed")
        database_path = source / "events.sqlite3"
        event = EventDatabase(database_path).list_recent(limit=1)[0]
        return source, database_path, event.event_id

    def test_event_binding_selects_only_recorded_media_and_is_stable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-bind-") as directory:
            root = Path(directory).resolve()
            source, database_path, event_id = self._smoke_spool(root)
            database = EventDatabase(database_path)
            first = build_event_archive_manifest(
                database=database,
                event_id=event_id,
                source_root=source,
            )
            second = build_event_archive_manifest(
                database=database,
                event_id=event_id,
                source_root=source,
            )
            self.assertEqual(first, second)
            self.assertEqual(4, len(first.items))
            self.assertTrue(all(item.relative_path.startswith("events/") for item in first.items))
            self.assertFalse(any("sqlite" in item.relative_path for item in first.items))
            self.assertFalse(any("reports/" in item.relative_path for item in first.items))

    def test_event_binding_fails_closed_for_missing_event_or_wrong_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-bind-") as directory:
            root = Path(directory).resolve()
            source, database_path, event_id = self._smoke_spool(root)
            database = EventDatabase(database_path)
            with self.assertRaisesRegex(ArchiveBindingError, "event is missing"):
                build_event_archive_manifest(
                    database=database,
                    event_id="missing_event",
                    source_root=source,
                )
            other_root = root / "other"
            other_root.mkdir()
            with self.assertRaisesRegex(ArchiveError, "outside the source root"):
                build_event_archive_manifest(
                    database=database,
                    event_id=event_id,
                    source_root=other_root,
                )

    def test_report_binding_is_content_versioned_and_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-bind-") as directory:
            root = Path(directory).resolve()
            source, _, _ = self._smoke_spool(root)
            report_date = date.today()
            report_directory = (
                source
                / "reports"
                / report_date.strftime("%Y")
                / report_date.strftime("%m")
                / report_date.strftime("%d")
            )
            if not report_directory.is_dir():
                available = next((source / "reports").rglob("*_daily_report.md"))
                report_date = date.fromisoformat(available.name[:10])
            first = build_report_archive_manifest(
                report_date=report_date,
                source_root=source,
            )
            self.assertEqual(2, len(first.items))
            self.assertEqual(
                {".html", ".md"},
                {Path(item.relative_path).suffix for item in first.items},
            )
            markdown_relative = next(
                item.relative_path for item in first.items if item.relative_path.endswith(".md")
            )
            markdown = source.joinpath(*markdown_relative.split("/"))
            markdown.write_text(markdown.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            changed = build_report_archive_manifest(
                report_date=report_date,
                source_root=source,
            )
            self.assertNotEqual(first.archive_id, changed.archive_id)
            self.assertEqual(first.created_at, changed.created_at)

    def test_binding_clis_enqueue_idempotently_without_disclosing_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="baby-monitor-archive-bind-") as directory:
            root = Path(directory).resolve()
            source, database_path, event_id = self._smoke_spool(root)
            event_output = io.StringIO()
            event_args = [
                "archive-enqueue-event",
                "--event-id",
                event_id,
                "--source-root",
                str(source),
                "--database",
                str(database_path),
            ]
            with redirect_stdout(event_output):
                first_code = main(event_args)
            first = json.loads(event_output.getvalue())
            self.assertEqual(0, first_code)
            self.assertEqual("event_media", first["binding_kind"])
            self.assertFalse(first["reused_existing"])
            self.assertNotIn(str(root), event_output.getvalue())

            retry_output = io.StringIO()
            with redirect_stdout(retry_output):
                retry_code = main(event_args)
            retry = json.loads(retry_output.getvalue())
            self.assertEqual(0, retry_code)
            self.assertTrue(retry["reused_existing"])
            self.assertEqual(1, ArchiveQueue(database_path).snapshot().counts.pending)

            report_file = next((source / "reports").rglob("*_daily_report.md"))
            report_date = report_file.name[:10]
            report_output = io.StringIO()
            with redirect_stdout(report_output):
                report_code = main(
                    [
                        "archive-enqueue-report",
                        "--date",
                        report_date,
                        "--source-root",
                        str(source),
                        "--database",
                        str(database_path),
                    ]
                )
            report = json.loads(report_output.getvalue())
            self.assertEqual(0, report_code)
            self.assertEqual("daily_report", report["binding_kind"])
            self.assertEqual(report_date, report["report_date"])
            self.assertNotIn(str(root), report_output.getvalue())
            self.assertEqual(2, ArchiveQueue(database_path).snapshot().counts.pending)


if __name__ == "__main__":
    unittest.main()
