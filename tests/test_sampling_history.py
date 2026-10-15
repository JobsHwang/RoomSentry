from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from baby_monitor.sampling_history import (
    SamplingEvidenceStore,
    SamplingHistoryError,
    read_sampling_day_summary,
)


def ledger(*, count: int = 1, fallback: bool = False) -> DetectorSamplingLedger:
    return DetectorSamplingLedger(
        native_fps=25,
        native_fps_source="reported",
        requested_sample_fps=1,
        frame_step=25,
        decoded_frame_count=51,
        samples=(
            DetectorSampleFrame(0, 0, "container_pts", 0, 0),
            DetectorSampleFrame(25, 1, "nominal_fps" if fallback else "container_pts", 0, count),
            DetectorSampleFrame(50, 2, "container_pts", count, 0),
        ),
        timestamp_fallback_frame_count=int(fallback),
        max_adjacent_container_pts_gap_seconds=0.125,
        container_pts_regression_count=1,
    )


class SamplingHistoryTests(unittest.TestCase):
    BASE = datetime(2026, 9, 12, 2, tzinfo=UTC)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-sampling-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "events.sqlite3"
        self.store = SamplingEvidenceStore(self.path)
        self.store.initialize()

    def record(self, **changes):
        values = dict(
            camera_id="camera_01",
            scanner_id="detector_v1",
            source_fingerprint="a" * 64,
            started_at=self.BASE,
            ended_at=self.BASE + timedelta(seconds=3),
            ledger=ledger(),
        )
        return self.store.record(**(values | changes))

    def summary(self, local_date=date(2026, 9, 12), timezone_name="UTC"):
        return read_sampling_day_summary(
            self.path, local_date=local_date, timezone_name=timezone_name
        )

    def test_zero_box_samples_are_counted_not_presence_or_duration(self):
        self.record(ledger=ledger(fallback=True))
        summary = self.summary()
        self.assertEqual(3, summary.sampled_frame_count)
        self.assertEqual(2, summary.zero_detection_sample_count)
        self.assertEqual(1, summary.timestamp_fallback_sample_count)
        self.assertEqual(1, summary.recorded_scan_count)
        self.assertEqual(0, summary.unknown_scan_count)
        self.assertEqual(0.125, summary.max_adjacent_container_pts_gap_seconds)
        self.assertEqual(1, summary.container_pts_regression_count)
        self.assertFalse(summary.source_exhaustion_verified)
        self.assertIsNone(summary.continuous_observation_seconds)

    def test_unknown_scan_has_no_synthesized_samples(self):
        self.record(ledger=None)
        summary = self.summary()
        self.assertEqual(1, summary.unknown_scan_count)
        self.assertEqual(0, summary.recorded_scan_count)
        self.assertEqual(0, summary.sampled_frame_count)
        self.assertIsNone(summary.max_adjacent_container_pts_gap_seconds)

    def test_worker_compatible_leading_punctuation_identifiers(self):
        for prefix in ("_", "-", "."):
            with self.subTest(prefix=prefix):
                first = self.record(camera_id=f"{prefix}camera", scanner_id=f"{prefix}scanner")
                replay = self.record(camera_id=f"{prefix}camera", scanner_id=f"{prefix}scanner")
                self.assertFalse(first.reused_existing)
                self.assertTrue(replay.reused_existing)
        self.assertEqual(3, self.summary().recorded_scan_count)
        self.assertEqual(9, self.summary().sampled_frame_count)

    def test_idempotent_replay_and_conflicting_ledger_roll_back(self):
        first, second = self.record(), self.record()
        self.assertEqual(first.record_id, second.record_id)
        self.assertFalse(first.reused_existing)
        self.assertTrue(second.reused_existing)
        with self.assertRaisesRegex(SamplingHistoryError, "conflicts"):
            self.record(ledger=ledger(count=2))
        with self.assertRaisesRegex(SamplingHistoryError, "conflicts"):
            self.record(ledger=None)
        self.assertEqual(3, self.summary().sampled_frame_count)
        self.assertTrue(self.record().reused_existing)

    def test_partial_insert_failure_rolls_back_header_and_samples(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("""CREATE TRIGGER reject_second BEFORE INSERT ON sampling_frames
                WHEN NEW.decoded_frame_index = 25 BEGIN SELECT RAISE(ABORT, 'private path'); END""")
        with self.assertRaisesRegex(SamplingHistoryError, "could not be persisted") as error:
            self.record()
        self.assertNotIn("private", str(error.exception))
        self.assertEqual(0, self.summary().recorded_scan_count)
        self.assertEqual(0, self.summary().sampled_frame_count)

    def test_cross_midnight_points_split_and_scan_diagnostics_overlap(self):
        start = datetime(2026, 9, 11, 23, 59, 59, tzinfo=UTC)
        self.record(started_at=start, ended_at=start + timedelta(seconds=3))
        previous, current = self.summary(date(2026, 9, 11)), self.summary()
        self.assertEqual((1, 2), (previous.sampled_frame_count, current.sampled_frame_count))
        self.assertEqual((1, 1), (previous.recorded_scan_count, current.recorded_scan_count))
        self.assertEqual(
            (1, 1),
            (previous.container_pts_regression_count, current.container_pts_regression_count),
        )

    def test_day_boundary_uses_local_timezone(self):
        start = datetime(2026, 9, 11, 15, 59, 59, tzinfo=UTC)
        self.record(started_at=start, ended_at=start + timedelta(seconds=3))
        self.assertEqual(2, self.summary(timezone_name="Asia/Shanghai").sampled_frame_count)

    def test_dst_23_and_25_hour_days_use_real_utc_boundaries(self):
        zone = ZoneInfo("America/New_York")
        for local_day in (date(2026, 3, 8), date(2026, 11, 1)):
            with self.subTest(local_day=local_day):
                beginning = datetime.combine(local_day, datetime.min.time(), tzinfo=zone)
                following = beginning + timedelta(days=1)
                self.record(started_at=beginning, ended_at=beginning + timedelta(seconds=3))
                self.record(started_at=following - timedelta(seconds=3), ended_at=following)
                self.record(started_at=following, ended_at=following + timedelta(seconds=3))
                result = self.summary(local_day, "America/New_York")
                self.assertEqual(6, result.sampled_frame_count)
                self.assertEqual(2, result.recorded_scan_count)

    def test_missing_and_old_database_read_does_not_create_or_migrate(self):
        missing = self.path.parent / "absent.sqlite3"
        self.assertIsNone(
            read_sampling_day_summary(missing, local_date=date(2026, 9, 12), timezone_name="UTC")
        )
        self.assertFalse(missing.exists())
        old = self.path.parent / "old.sqlite3"
        with sqlite3.connect(old) as connection:
            connection.execute("CREATE TABLE events (event_id TEXT)")
        before = old.read_bytes(), old.stat().st_mtime_ns
        self.assertIsNone(
            read_sampling_day_summary(old, local_date=date(2026, 9, 12), timezone_name="UTC")
        )
        self.assertEqual(before, (old.read_bytes(), old.stat().st_mtime_ns))

    def test_initialized_empty_day_differs_from_unavailable(self):
        result = self.summary()
        self.assertIsNotNone(result)
        self.assertEqual(0, result.sampled_frame_count)
        self.assertEqual(0, result.unknown_scan_count)

    def test_partial_or_wrong_schema_fails_closed(self):
        for ddl in (
            "DROP TABLE sampling_frames",
            "ALTER TABLE sampling_scans ADD COLUMN unexpected TEXT",
        ):
            with self.subTest(ddl=ddl):
                target = self.path.parent / str(abs(hash(ddl)))
                store = SamplingEvidenceStore(target)
                store.initialize()
                with sqlite3.connect(target) as connection:
                    connection.execute(ddl)
                with self.assertRaisesRegex(SamplingHistoryError, "schema"):
                    read_sampling_day_summary(
                        target, local_date=date(2026, 9, 12), timezone_name="UTC"
                    )

    def test_invalid_rows_and_missing_frame_fail_closed(self):
        self.record()
        for sql in (
            "UPDATE sampling_scans SET camera_id = '/private/path'",
            "UPDATE sampling_scans SET sampled_frame_count = 4",
            "UPDATE sampling_frames SET detection_count = 100001 WHERE decoded_frame_index = 0",
            "DELETE FROM sampling_frames WHERE decoded_frame_index = 25",
        ):
            with self.subTest(sql=sql):
                with sqlite3.connect(self.path) as connection:
                    connection.execute("SAVEPOINT corruption")
                    connection.execute(sql)
                    connection.commit()
                with self.assertRaisesRegex(SamplingHistoryError, "row is invalid"):
                    self.summary()
                with sqlite3.connect(self.path) as connection:
                    connection.execute("DELETE FROM sampling_frames")
                    connection.execute("DELETE FROM sampling_scans")
                self.record()

    def test_bounds_and_path_free_input_errors(self):
        for changes in (
            {"camera_id": "/private/path"},
            {"scanner_id": "bad id"},
            {"source_fingerprint": "wrong"},
            {"started_at": datetime(2026, 9, 12)},
            {"ended_at": self.BASE},
            {"ended_at": self.BASE + timedelta(seconds=1.749)},
            {"ledger": {}},
        ):
            with self.subTest(changes=changes), self.assertRaises(SamplingHistoryError):
                self.record(**changes)
        self.record(ended_at=self.BASE + timedelta(seconds=1.75))
        with self.assertRaises(SamplingHistoryError) as error:
            SamplingEvidenceStore("secret\x00.sqlite3")
        self.assertNotIn("secret", str(error.exception))

    def test_ledger_size_bound_and_corrupt_ledger_revalidated(self):
        with (
            patch("baby_monitor.sampling_history.MAX_LEDGER_BYTES", 10),
            self.assertRaisesRegex(SamplingHistoryError, "size limit"),
        ):
            self.record()
        corrupted = ledger()
        object.__setattr__(corrupted, "decoded_frame_count", False)
        with self.assertRaisesRegex(SamplingHistoryError, "ledger is invalid"):
            self.record(ledger=corrupted)

    def test_long_valid_buffer_interval_is_not_rejected_or_omitted(self):
        self.record(ended_at=self.BASE + timedelta(seconds=601))
        self.assertEqual(3, self.summary().sampled_frame_count)
        self.assertEqual(1, self.summary().recorded_scan_count)
        # BufferSegment permits longer intervals too. A fixed 600-second query
        # lookback would silently omit this overlapping unknown scan header.
        start = datetime(2026, 9, 11, 23, tzinfo=UTC)
        self.record(started_at=start, ended_at=start + timedelta(hours=2), ledger=None)
        self.assertEqual(1, self.summary().unknown_scan_count)

    def test_replay_checks_saved_frame_details(self):
        self.record()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE sampling_frames SET detection_count = 3 WHERE decoded_frame_index = 25"
            )
        with self.assertRaisesRegex(SamplingHistoryError, "conflicts"):
            self.record()

    def test_invalid_dates_timezones_and_database_are_safe(self):
        for day, zone in (("2026-09-12", "UTC"), (date(2026, 9, 12), "/secret"), (date.max, "UTC")):
            with self.subTest(day=day, zone=zone), self.assertRaises(SamplingHistoryError):
                self.summary(day, zone)
        with self.assertRaises(SamplingHistoryError) as error:
            read_sampling_day_summary(
                self.path.parent, local_date=date(2026, 9, 12), timezone_name="UTC"
            )
        self.assertNotIn(str(self.path.parent), str(error.exception))

    def test_summary_rejects_inconsistent_counts(self):
        summary = self.summary()
        with self.assertRaises(SamplingHistoryError):
            replace(summary, zero_detection_sample_count=1)
        with self.assertRaises(ValueError):
            replace(summary, source_exhaustion_verified=True)


if __name__ == "__main__":
    unittest.main()
