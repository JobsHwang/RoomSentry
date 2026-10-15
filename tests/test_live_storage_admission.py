from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from baby_monitor import live_monitor
from baby_monitor.adapters.camera import RTSPCamera

MIB = 1024**2
GIB = 1024**3


def entry(size=100, mode=0o100600):
    result = Mock()
    result.name = "child"
    result.stat.return_value = SimpleNamespace(st_size=size, st_mode=mode)
    return result


class Scan:
    def __init__(self, values):
        self.values = iter(values)
        self.closed = False
        self.next_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        close = getattr(self.values, "close", None)
        if close is not None:
            close()

    def __iter__(self):
        return self

    def __next__(self):
        self.next_calls += 1
        return next(self.values)


class LiveStorageSummaryTests(unittest.TestCase):
    def scan(self, entries=(), *, free=10 * GIB, limit=64 * MIB, max_entries=5000):
        scans = [Scan(entries), Scan(())]
        with tempfile.TemporaryDirectory(prefix="private-synthetic-root-") as directory:
            root = Path(directory)
            (root / "child").mkdir()
            with (
                patch.object(live_monitor.os, "scandir", side_effect=scans) as walk,
                patch.object(
                    live_monitor.shutil, "disk_usage", return_value=SimpleNamespace(free=free)
                ) as disk,
            ):
                result = live_monitor._review_storage_summary(root, limit, max_entries)
        self.assertTrue(scans[0].closed)
        for current in scans[: walk.call_count]:
            self.assertTrue(current.closed)
        disk.assert_called_once()
        return result

    def test_single_scan_reports_bytes_entries_and_preserved_default_reserve(self):
        result = self.scan((entry(mode=0o040700), entry(300), entry(200)))
        self.assertTrue(result["available"])
        self.assertEqual("available", result["reason"])
        self.assertEqual(500, result["used_bytes"])
        self.assertEqual(3, result["entries"])
        self.assertTrue(result["scan_complete"])
        self.assertEqual(64 * MIB - 500, result["remaining_budget_bytes"])
        self.assertEqual(4997, result["remaining_entries"])
        self.assertEqual(12 * MIB, result["reserve_bytes"])
        self.assertEqual(2 * GIB, result["minimum_free_disk_bytes"])
        self.assertEqual(10 * GIB, result["free_disk_bytes"])
        self.assertIn("checked_at", result)
        self.assertNotIn("private-synthetic-root", json.dumps(result))

    def test_capacity_reserve_is_strict_at_exact_boundary(self):
        for limit in (64 * MIB, 4 * GIB, 64 * GIB):
            for difference, expected in (
                (-1, "available"),
                (0, "capacity_limit"),
                (1, "capacity_limit"),
            ):
                with self.subTest(limit=limit, difference=difference):
                    result = self.scan((entry(limit - 12 * MIB + difference),), limit=limit)
                    self.assertEqual(expected, result["reason"])
                    self.assertEqual(expected == "available", result["available"])
                    self.assertTrue(result["scan_complete"])

    def test_remaining_budget_never_negative_when_existing_data_exceeds_limit(self):
        result = self.scan((entry(80 * MIB),))
        self.assertEqual("capacity_limit", result["reason"])
        self.assertEqual(0, result["remaining_budget_bytes"])

    def test_disk_free_threshold_remains_two_gib(self):
        for limit in (64 * MIB, 4 * GIB, 64 * GIB):
            for free, reason in ((2 * GIB, "available"), (2 * GIB - 1, "disk_space_low")):
                with self.subTest(limit=limit, free=free):
                    result = self.scan(free=free, limit=limit)
                    self.assertEqual(reason, result["reason"])
                    self.assertEqual(free, result["free_disk_bytes"])

    def test_entry_limit_is_bounded_and_partial_totals_are_marked(self):
        values = [entry(100), entry(100), entry(100), entry(100)]
        result = self.scan(values, max_entries=3)
        self.assertEqual("entry_limit", result["reason"])
        self.assertFalse(result["scan_complete"])
        self.assertEqual(3, result["entries"])
        self.assertEqual(200, result["used_bytes"])
        self.assertEqual(0, result["remaining_entries"])
        values[2].stat.assert_not_called()
        values[3].stat.assert_not_called()

    def test_disappearing_atomic_file_is_counted_without_becoming_io_failure(self):
        disappearing = Mock()
        disappearing.stat.side_effect = FileNotFoundError("private-temporary-name")
        result = self.scan((disappearing, entry(150)))
        self.assertTrue(result["available"])
        self.assertTrue(result["scan_complete"])
        self.assertEqual(2, result["entries"])
        self.assertEqual(150, result["used_bytes"])

    def test_symlink_has_fixed_reason_and_does_not_follow_target(self):
        link = entry(mode=0o120777)
        next_entry = entry()
        result = self.scan((link, next_entry))
        self.assertEqual("unsafe_symlink", result["reason"])
        self.assertFalse(result["scan_complete"])
        next_entry.stat.assert_not_called()

    def test_metadata_and_walk_errors_have_no_exception_text(self):
        bad = Mock()
        bad.stat.side_effect = PermissionError("private-directory-secret")
        result = self.scan((entry(42), bad))
        self.assertEqual("io_error", result["reason"])
        self.assertEqual(42, result["used_bytes"])
        self.assertFalse(result["scan_complete"])
        self.assertNotIn("private-directory-secret", json.dumps(result))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(live_monitor.os, "scandir", side_effect=OSError("private-walk-secret")),
            patch.object(
                live_monitor.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * GIB)
            ),
        ):
            result = live_monitor._review_storage_summary(Path(directory), 64 * MIB)
        self.assertEqual("io_error", result["reason"])
        self.assertFalse(result["scan_complete"])
        self.assertNotIn("private-walk-secret", json.dumps(result))

    def test_disk_query_failure_is_unknown_free_space_not_zero(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(live_monitor.shutil, "disk_usage", side_effect=OSError("private-disk")),
        ):
            result = live_monitor._review_storage_summary(Path(directory), 64 * MIB)
        self.assertEqual("io_error", result["reason"])
        self.assertIsNone(result["free_disk_bytes"])
        self.assertTrue(result["scan_complete"])
        self.assertNotIn("private-disk", json.dumps(result))

    def test_explicit_budgets_allow_previous_and_new_bounded_maxima(self):
        for limit, entries in ((2 * GIB, 10000), (4 * GIB, 20000), (64 * GIB, 200000)):
            with self.subTest(limit=limit, entries=entries):
                result = self.scan(limit=limit, max_entries=entries)
                self.assertTrue(result["available"])
                self.assertEqual(limit, result["limit_bytes"])
                self.assertEqual(entries, result["limit_entries"])

    def test_limits_reject_boolean_float_string_and_out_of_range_before_scanning(self):
        class IntegerSubclass(int):
            pass

        for field, values in (
            (
                "bytes",
                (
                    True,
                    False,
                    None,
                    "67108864",
                    float(4 * GIB),
                    IntegerSubclass(4 * GIB),
                    64 * MIB - 1,
                    64 * GIB + 1,
                ),
            ),
            (
                "entries",
                (True, False, None, "20000", 20000.0, IntegerSubclass(20000), 0, 200001),
            ),
        ):
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    patch.object(live_monitor.os, "scandir") as walk,
                ):
                    with self.assertRaises(ValueError):
                        live_monitor._review_storage_summary(
                            Path("/private"),
                            value if field == "bytes" else 64 * MIB,
                            value if field == "entries" else 5000,
                        )
                    walk.assert_not_called()

    def test_boolean_compatibility_wrapper_uses_one_summary_scan(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(live_monitor.os, "scandir", return_value=Scan(())) as walk,
            patch.object(
                live_monitor.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * GIB)
            ) as disk,
        ):
            self.assertIs(True, live_monitor._review_storage_available(Path(directory), 64 * MIB))
        walk.assert_called_once()
        disk.assert_called_once()

    def test_scandir_permission_failure_in_real_child_directory_fails_closed(self):
        original_scan = os.scandir
        scans, descriptors = [], []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "unreadable-child"
            child.mkdir()
            child_inode = child.stat().st_ino

            def scan(descriptor):
                descriptors.append(descriptor)
                if os.fstat(descriptor).st_ino == child_inode:
                    raise PermissionError("private-child-name")
                iterator = Scan(original_scan(descriptor))
                scans.append(iterator)
                return iterator

            with patch.object(live_monitor.os, "scandir", side_effect=scan):
                result = live_monitor._review_storage_summary(root, 64 * MIB)
            self.assertEqual("io_error", result["reason"])
            self.assertFalse(result["available"])
            self.assertFalse(result["scan_complete"])
            self.assertEqual(1, result["entries"])
            self.assertNotIn("private-child-name", json.dumps(result))
            self.assertTrue(all(iterator.closed for iterator in scans))
            self.assertEqual(2, len(descriptors))
            for descriptor in descriptors:
                with self.assertRaises(OSError):
                    os.fstat(descriptor)

    def test_scandir_iteration_error_is_visible_and_closes_iterator(self):
        original_scan = os.scandir

        class InterruptedScan(Scan):
            def __next__(self):
                if self.next_calls:
                    raise OSError("private-iterator-failure")
                return super().__next__()

        scans = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one").write_bytes(b"synthetic")
            (root / "two").write_bytes(b"synthetic")

            def scan(descriptor):
                iterator = InterruptedScan(original_scan(descriptor))
                scans.append(iterator)
                return iterator

            with patch.object(live_monitor.os, "scandir", side_effect=scan):
                result = live_monitor._review_storage_summary(root, 64 * MIB)
        self.assertEqual("io_error", result["reason"])
        self.assertFalse(result["scan_complete"])
        self.assertEqual(1, result["entries"])
        self.assertEqual(len(b"synthetic"), result["used_bytes"])
        self.assertTrue(scans[0].closed)
        self.assertNotIn("private-iterator-failure", json.dumps(result))

    def test_entry_budget_does_not_prefetch_and_closes_active_iterators(self):
        generated = []

        def many_entries():
            for number in range(1000000):
                generated.append(number)
                yield entry()

        iterator = Scan(many_entries())
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(live_monitor.os, "scandir", return_value=iterator) as scan:
                result = live_monitor._review_storage_summary(Path(directory), 64 * MIB, 3)
            self.assertEqual("entry_limit", result["reason"])
            self.assertFalse(result["scan_complete"])
            self.assertEqual([0, 1, 2], generated)
            self.assertEqual(3, iterator.next_calls)
            self.assertTrue(iterator.closed)
            with self.assertRaises(OSError):
                os.fstat(scan.call_args.args[0])

    def test_maximum_entry_budget_still_streams_without_prefetch_or_retained_items(self):
        generated = 0
        closed = False
        metadata = SimpleNamespace(st_size=100, st_mode=0o100600)
        item = SimpleNamespace(name="fixture", stat=lambda **_kwargs: metadata)

        def many_entries():
            nonlocal generated, closed
            try:
                for _ in range(1000000):
                    generated += 1
                    yield item
            finally:
                closed = True

        iterator = Scan(many_entries())
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(live_monitor.os, "scandir", return_value=iterator) as scan:
                result = live_monitor._review_storage_summary(Path(directory), 64 * GIB, 200000)
            self.assertEqual("entry_limit", result["reason"])
            self.assertFalse(result["available"])
            self.assertFalse(result["scan_complete"])
            self.assertEqual(200000, result["entries"])
            self.assertEqual(199999 * 100, result["used_bytes"])
            self.assertEqual(0, result["remaining_entries"])
            self.assertEqual(200000, generated)
            self.assertEqual(200000, iterator.next_calls)
            self.assertTrue(iterator.closed)
            self.assertTrue(closed)
            with self.assertRaises(OSError):
                os.fstat(scan.call_args.args[0])

    def test_real_directory_symlink_is_rejected_without_traversing_target(self):
        original_scan = os.scandir
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            target = Path(directory) / "private-target"
            target.mkdir()
            (target / "large").write_bytes(b"synthetic")
            (root / "link").symlink_to(target, target_is_directory=True)
            with patch.object(live_monitor.os, "scandir", wraps=original_scan) as scan:
                result = live_monitor._review_storage_summary(root, 64 * MIB)
            self.assertEqual("unsafe_symlink", result["reason"])
            self.assertEqual(0, result["used_bytes"])
            scan.assert_called_once()

    def test_directory_changed_to_symlink_before_open_is_not_followed(self):
        original_open = os.open
        original_scan = os.scandir
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            child = root / "child"
            child.mkdir()
            target = Path(directory) / "private-target"
            target.mkdir()

            def open_directory(path, *args, **kwargs):
                if path == "child":
                    child.rmdir()
                    child.symlink_to(target, target_is_directory=True)
                return original_open(path, *args, **kwargs)

            with (
                patch.object(live_monitor.os, "open", side_effect=open_directory),
                patch.object(live_monitor.os, "scandir", wraps=original_scan) as scan,
            ):
                result = live_monitor._review_storage_summary(root, 64 * MIB)
            self.assertEqual("io_error", result["reason"])
            self.assertFalse(result["scan_complete"])
            scan.assert_called_once()


@unittest.skipUnless(os.name == "posix", "live monitor requires POSIX")
class LiveStorageMonitorTests(unittest.TestCase):
    def run_monitor(self, summary, *, memory_available=False, recover=False, **limits):
        class Stop:
            stopped = False
            waits = 0

            def is_set(self):
                return self.stopped

            def set(self):
                self.stopped = True

            def wait(self, _timeout):
                self.waits += 1
                if not recover or self.waits > 1:
                    self.set()

        class Thread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

            def is_alive(self):
                return False

        class Source:
            def __init__(self, source, **_kwargs):
                self.source = source

            def start(self):
                pass

            def close(self):
                pass

            def snapshot(self):
                return {"phase": "fixture"}

        stop = Stop()
        queue = Mock(condition=MagicMock())
        queue.snapshot.return_value = {}
        queue.take_live.side_effect = lambda: stop.set()
        admission = Mock(return_value=memory_available)
        detector = Mock(identity={"fixture": True})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            with (
                patch.object(live_monitor, "FFmpegLiveSource", Source),
                patch.object(live_monitor, "EventDatabase"),
                patch.object(live_monitor, "ReviewMailbox", return_value=queue),
                patch.object(live_monitor.threading, "Thread", Thread),
                patch.object(live_monitor.threading, "Event", side_effect=(stop, Stop())),
                patch.object(live_monitor, "_review_storage_summary", side_effect=summary) as scan,
            ):
                live_monitor.run_live_monitor(
                    streams=((RTSPCamera("cam1", "rtsp://127.0.0.1/fixture").source(), 16, 16),),
                    detector=detector,
                    local_vlm=object(),
                    root=root,
                    review_admission=admission,
                    **limits,
                )
            status = json.loads((root / "status.json").read_text())
            self.assertEqual([], list((root / "frames").iterdir()))
            self.assertEqual([], list((root / "records").iterdir()))
        return status, scan, admission

    def test_defaults_are_unchanged_and_storage_deferral_is_explicit(self):
        defaults = inspect.signature(live_monitor.run_live_monitor).parameters
        self.assertEqual(512 * MIB, defaults["max_storage_bytes"].default)
        self.assertEqual(5000, defaults["max_storage_entries"].default)
        for scanner in (
            live_monitor._review_storage_summary,
            live_monitor._review_storage_available,
        ):
            self.assertEqual(
                5000, inspect.signature(scanner).parameters["max_storage_entries"].default
            )
        report = {"available": False, "reason": "capacity_limit", "used_bytes": 510 * MIB}
        status, scan, admission = self.run_monitor([report])
        self.assertEqual("storage_deferred", status["review_phase"])
        self.assertEqual(report, status["review_storage"])
        self.assertEqual(512 * MIB, status["review_storage_limit_bytes"])
        self.assertEqual(5000, status["review_storage_limit_entries"])
        self.assertEqual((512 * MIB, 5000), scan.call_args.args[1:])
        self.assertEqual(1, scan.call_count)
        admission.assert_not_called()

    def test_memory_deferral_keeps_successful_storage_diagnostics(self):
        report = {"available": True, "reason": "available", "used_bytes": 100}
        status, scan, admission = self.run_monitor([report])
        self.assertEqual("memory_deferred", status["review_phase"])
        self.assertEqual(report, status["review_storage"])
        self.assertEqual(1, scan.call_count)
        admission.assert_called_once_with()

    def test_explicit_larger_budgets_are_reported_and_sent_to_scan(self):
        status, scan, _ = self.run_monitor(
            [{"available": False, "reason": "entry_limit"}],
            max_storage_bytes=64 * GIB,
            max_storage_entries=200000,
        )
        self.assertEqual(64 * GIB, status["review_storage_limit_bytes"])
        self.assertEqual(200000, status["review_storage_limit_entries"])
        self.assertEqual((64 * GIB, 200000), scan.call_args.args[1:])

    def test_recovered_storage_does_not_leave_stale_deferred_phase(self):
        available = {"available": True, "reason": "available"}
        status, scan, admission = self.run_monitor(
            [{"available": False, "reason": "capacity_limit"}, available],
            memory_available=True,
            recover=True,
        )
        self.assertEqual("waiting", status["review_phase"])
        self.assertEqual(available, status["review_storage"])
        self.assertEqual(2, scan.call_count)
        admission.assert_called_once_with()

    def test_invalid_monitor_budget_does_not_start_detector_or_create_outputs(self):
        for limits in (
            {"max_storage_bytes": True},
            {"max_storage_bytes": float(512 * MIB)},
            {"max_storage_bytes": "4294967296"},
            {"max_storage_bytes": 64 * GIB + 1},
            {"max_storage_entries": False},
            {"max_storage_entries": 5000.0},
            {"max_storage_entries": "20000"},
            {"max_storage_entries": 200001},
        ):
            with self.subTest(limits=limits), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                root.chmod(0o700)
                detector = Mock()
                with self.assertRaises(ValueError):
                    live_monitor.run_live_monitor(
                        streams=(
                            (RTSPCamera("cam1", "rtsp://127.0.0.1/fixture").source(), 16, 16),
                        ),
                        detector=detector,
                        local_vlm=object(),
                        root=root,
                        review_admission=lambda: True,
                        **limits,
                    )
                detector.start.assert_not_called()
                self.assertEqual([], list(root.iterdir()))


if __name__ == "__main__":
    unittest.main()
