from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

from baby_monitor.config import StorageSettings
from baby_monitor.database import EventDatabase
from baby_monitor.report_lock import ReportLockError, report_publication_lock
from baby_monitor.reporting import DailyReportGenerator
from baby_monitor.storage import StorageResolver


class ReportPublicationLockTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-report-lock-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.database = EventDatabase(self.root / "events.sqlite3")
        self.database.initialize()
        self.storage = StorageResolver(
            StorageSettings(local_root=self.root / "output", database=self.database.path)
        )
        self.day = date(2026, 8, 3)
        self.generator = DailyReportGenerator(
            database=self.database, storage=self.storage, timezone_name="Asia/Shanghai"
        )

    def test_cross_process_contention_is_bounded_and_precedes_database_reads(self) -> None:
        markdown, _html = self.storage.report_paths(self.day)
        script = """
import sys
from pathlib import Path
from baby_monitor.report_lock import report_publication_lock
with report_publication_lock(Path(sys.argv[1])):
    print('locked', flush=True)
    sys.stdin.readline()
"""
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(markdown.parent)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert child.stdout is not None
            # Bound the readiness read independently of the child's lock lifetime.
            import select

            ready, _, _ = select.select([child.stdout], [], [], 5)
            self.assertTrue(ready, "report lock child did not become ready")
            self.assertEqual("locked", child.stdout.readline().strip())
            with patch.object(self.database, "fetch_between") as fetch:
                with self.assertRaisesRegex(ReportLockError, "busy"):
                    self.generator.generate(self.day, lock_timeout_seconds=0)
                fetch.assert_not_called()
            child.communicate("release\n", timeout=5)
            self.assertEqual(0, child.returncode)
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
        self.generator.generate(self.day, lock_timeout_seconds=0)
        lock = markdown.parent / ".daily-report.lock"
        self.assertEqual(0o600, stat.S_IMODE(lock.stat().st_mode))

    def test_later_writer_reads_only_after_earlier_publication_finishes(self) -> None:
        first_rendering = threading.Event()
        release_first = threading.Event()
        second_read = threading.Event()
        failures: list[BaseException] = []
        second_database = EventDatabase(self.database.path)
        second = DailyReportGenerator(
            database=second_database, storage=self.storage, timezone_name="Asia/Shanghai"
        )

        def first_markdown(*args, **kwargs):
            first_rendering.set()
            if not release_first.wait(5):
                raise AssertionError("first writer was not released")
            return "older-report"

        original_fetch = second_database.fetch_between

        def later_fetch(*args, **kwargs):
            second_read.set()
            return original_fetch(*args, **kwargs)

        def run(generator):
            try:
                generator.generate(self.day)
            except BaseException as exc:
                failures.append(exc)

        with (
            patch.object(self.generator, "_markdown", side_effect=first_markdown),
            patch.object(self.generator, "_html", return_value="older-html"),
            patch.object(second_database, "fetch_between", side_effect=later_fetch),
            patch.object(second, "_markdown", return_value="newer-report"),
            patch.object(second, "_html", return_value="newer-html"),
        ):
            first_thread = threading.Thread(target=run, args=(self.generator,), daemon=True)
            second_thread = threading.Thread(target=run, args=(second,), daemon=True)
            first_thread.start()
            try:
                self.assertTrue(first_rendering.wait(5))
                second_thread.start()
                self.assertFalse(second_read.wait(0.15))
            finally:
                release_first.set()
                first_thread.join(5)
                if second_thread.ident is not None:
                    second_thread.join(5)
            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
        self.assertEqual([], failures)
        self.assertTrue(second_read.is_set())
        markdown, html = self.storage.report_paths(self.day)
        self.assertEqual("newer-report", markdown.read_text())
        self.assertEqual("newer-html", html.read_text())

    def test_claim_guard_rejects_before_read_and_again_before_publication(self) -> None:
        artifacts = self.generator.generate(self.day)
        original = (artifacts.markdown_path.read_bytes(), artifacts.html_path.read_bytes())
        with patch.object(self.database, "fetch_between") as fetch:
            with self.assertRaisesRegex(RuntimeError, "stale"):
                self.generator.generate(
                    self.day, claim_guard=Mock(side_effect=RuntimeError("stale"))
                )
            fetch.assert_not_called()
        guard = Mock(side_effect=[None, RuntimeError("stale")])
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.generator.generate(self.day, claim_guard=guard)
        self.assertEqual(2, guard.call_count)
        self.assertEqual(original[0], artifacts.markdown_path.read_bytes())
        self.assertEqual(original[1], artifacts.html_path.read_bytes())

    def test_lock_rejects_links_weak_mode_and_invalid_timeouts(self) -> None:
        markdown, _html = self.storage.report_paths(self.day)
        lock = markdown.parent / ".daily-report.lock"
        outside = self.root / "outside"
        outside.write_bytes(b"unchanged")
        lock.symlink_to(outside)
        with self.assertRaises(ReportLockError):
            self.generator.generate(self.day)
        self.assertEqual(b"unchanged", outside.read_bytes())
        lock.unlink()
        os.link(outside, lock)
        with self.assertRaises(ReportLockError):
            self.generator.generate(self.day)
        lock.unlink()
        lock.write_bytes(b"")
        lock.chmod(0o666)
        with self.assertRaises(ReportLockError):
            self.generator.generate(self.day)
        for timeout in (-1, float("nan"), float("inf"), 61, True):
            with (
                self.subTest(timeout=timeout),
                self.assertRaises(ValueError),
                report_publication_lock(markdown.parent, timeout_seconds=timeout),
            ):
                self.fail("invalid timeout acquired a lock")

    def test_render_failure_keeps_both_previous_outputs(self) -> None:
        artifacts = self.generator.generate(self.day)
        before = (artifacts.markdown_path.read_bytes(), artifacts.html_path.read_bytes())
        with (
            patch.object(self.generator, "_markdown", return_value="unpublished-markdown"),
            patch.object(self.generator, "_html", side_effect=RuntimeError("render failed")),
            self.assertRaisesRegex(RuntimeError, "render failed"),
        ):
            self.generator.generate(self.day)
        self.assertEqual(before[0], artifacts.markdown_path.read_bytes())
        self.assertEqual(before[1], artifacts.html_path.read_bytes())
        self.assertFalse(list(artifacts.markdown_path.parent.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
