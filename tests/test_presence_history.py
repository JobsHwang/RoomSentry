from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from baby_monitor.presence_history import (
    PresenceDaySummary,
    PresenceFactMode,
    PresenceFactStore,
    PresenceHistoryError,
    PresenceStreamDaySummary,
    read_presence_day_summary,
)
from baby_monitor.room_presence import (
    StreamPresenceSample,
    StreamPresenceSampleState,
)
from scripts.benchmark_presence_summary import benchmark as benchmark_presence_summary


class PresenceHistoryTests(unittest.TestCase):
    BASE = datetime(2026, 8, 19, 1, tzinfo=UTC)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-presence-history-")
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "events.sqlite3"
        self.store = PresenceFactStore(self.database)
        self.store.initialize()

    def _sample(
        self,
        sample_id: str,
        *,
        camera_id: str = "cam_01",
        offset_seconds: float = 0.0,
        state: StreamPresenceSampleState = StreamPresenceSampleState.CLEAR,
    ) -> StreamPresenceSample:
        gap = state is StreamPresenceSampleState.COVERAGE_GAP
        return StreamPresenceSample(
            sample_id=sample_id,
            camera_id=camera_id,
            observed_at=self.BASE + timedelta(seconds=offset_seconds),
            state=state,
            scanner_id="scanner_v1",
            detector_id=None if gap else "detector_v1",
            source_fingerprint=("a" if camera_id == "cam_01" else "b") * 64,
            reason_code="worker_not_running" if gap else None,
        )

    def test_record_is_idempotent_and_conflicts_fail_closed(self) -> None:
        sample = self._sample("sample_01")
        first = self.store.record(sample, recorded_at=self.BASE)
        second = self.store.record(sample, recorded_at=self.BASE + timedelta(seconds=1))
        self.assertFalse(first.reused_existing)
        self.assertIs(PresenceFactMode.PRIMARY, first.mode)
        self.assertTrue(second.reused_existing)
        self.assertEqual(
            (sample,),
            self.store.fetch_between(
                self.BASE - timedelta(seconds=1),
                self.BASE + timedelta(seconds=1),
            ),
        )

        conflicting = replace(sample, camera_id="cam_02")
        with self.assertRaisesRegex(PresenceHistoryError, "conflicts"):
            self.store.record(conflicting)

    def test_day_summary_counts_facts_and_includes_day_boundary_gaps(self) -> None:
        samples = (
            self._sample("sample_01", offset_seconds=0, state=StreamPresenceSampleState.CLEAR),
            self._sample(
                "sample_02",
                offset_seconds=20,
                state=StreamPresenceSampleState.PERSON_DETECTED,
            ),
            self._sample(
                "sample_03",
                camera_id="cam_02",
                offset_seconds=40,
                state=StreamPresenceSampleState.COVERAGE_GAP,
            ),
        )
        for sample in samples:
            self.store.record(sample)

        summary = read_presence_day_summary(
            self.database,
            local_date=date(2026, 8, 19),
            timezone_name="Asia/Shanghai",
        )
        assert summary is not None
        self.assertEqual(3, summary.sample_count)
        self.assertEqual(1, summary.person_detected_count)
        self.assertEqual(1, summary.clear_count)
        self.assertEqual(1, summary.coverage_gap_count)
        self.assertEqual(("cam_01", "cam_02"), tuple(item.camera_id for item in summary.streams))
        self.assertGreater(summary.maximum_observation_gap_seconds or 0, 14 * 3600)
        self.assertFalse(summary.media_read)
        self.assertFalse(summary.subject_identity_inference)
        self.assertFalse(summary.risk_assessment)
        self.assertFalse(summary.notification_authority)

    def test_missing_table_is_distinct_from_an_empty_day(self) -> None:
        empty = read_presence_day_summary(
            self.database,
            local_date=date(2026, 8, 20),
            timezone_name="Asia/Shanghai",
        )
        assert empty is not None
        self.assertEqual((), empty.streams)

        legacy = self.database.with_name("legacy.sqlite3")
        connection = sqlite3.connect(legacy)
        connection.execute("CREATE TABLE events (event_id TEXT PRIMARY KEY)")
        connection.commit()
        connection.close()
        original_database = legacy.read_bytes()
        self.assertIsNone(
            read_presence_day_summary(
                legacy,
                local_date=date(2026, 8, 20),
                timezone_name="Asia/Shanghai",
            )
        )
        self.assertEqual(original_database, legacy.read_bytes())

    def test_day_summary_preserves_interleaved_facts_and_separates_modes(self) -> None:
        start = datetime(2026, 8, 19, tzinfo=UTC)
        end = start + timedelta(days=1)
        clear = StreamPresenceSampleState.CLEAR
        person = StreamPresenceSampleState.PERSON_DETECTED
        gap = StreamPresenceSampleState.COVERAGE_GAP
        # Insertion order differs from observation order; camera_01 has two
        # distinct observations at the same timestamp, which both count.
        facts = (
            ("cam_01", 86340, gap),
            ("cam_02", 72000, person),
            ("cam_01", 21600, person),
            ("cam_03", 7200, gap),
            ("cam_01", 0, clear),
            ("cam_02", 75600, person),
            ("cam_03", 3600, clear),
            ("cam_01", 21600, clear),
        )
        for index, (camera_id, seconds, state) in enumerate(facts):
            self.store.record(
                replace(
                    self._sample(f"primary_{index}", camera_id=camera_id, state=state),
                    observed_at=start + timedelta(seconds=seconds),
                )
            )
        for index, (camera_id, state) in enumerate((("cam_01", gap), ("cam_04", person))):
            self.store.record(
                replace(
                    self._sample(f"shadow_{index}", camera_id=camera_id, state=state),
                    observed_at=start + timedelta(hours=12),
                ),
                mode=PresenceFactMode.SHADOW,
            )

        primary = read_presence_day_summary(
            self.database, local_date=start.date(), timezone_name="UTC"
        )
        self.assertEqual(
            PresenceDaySummary(
                mode=PresenceFactMode.PRIMARY,
                local_date=start.date(),
                timezone_name="UTC",
                start_at=start,
                end_at=end,
                streams=(
                    PresenceStreamDaySummary(
                        "cam_01", 4, 1, 2, 1, start, end - timedelta(seconds=60), 64740
                    ),
                    PresenceStreamDaySummary(
                        "cam_02",
                        2,
                        2,
                        0,
                        0,
                        start + timedelta(hours=20),
                        start + timedelta(hours=21),
                        72000,
                    ),
                    PresenceStreamDaySummary(
                        "cam_03",
                        2,
                        0,
                        1,
                        1,
                        start + timedelta(hours=1),
                        start + timedelta(hours=2),
                        79200,
                    ),
                ),
            ),
            primary,
        )
        shadow_time = start + timedelta(hours=12)
        shadow = read_presence_day_summary(
            self.database,
            local_date=start.date(),
            timezone_name="UTC",
            mode=PresenceFactMode.SHADOW,
        )
        self.assertEqual(
            PresenceDaySummary(
                mode=PresenceFactMode.SHADOW,
                local_date=start.date(),
                timezone_name="UTC",
                start_at=start,
                end_at=end,
                streams=(
                    PresenceStreamDaySummary("cam_01", 1, 0, 0, 1, shadow_time, shadow_time, 43200),
                    PresenceStreamDaySummary("cam_04", 1, 1, 0, 0, shadow_time, shadow_time, 43200),
                ),
            ),
            shadow,
        )
        assert primary is not None
        self.assertEqual(
            (8, 3, 3, 2, 79200.0),
            (
                primary.sample_count,
                primary.person_detected_count,
                primary.clear_count,
                primary.coverage_gap_count,
                primary.maximum_observation_gap_seconds,
            ),
        )

    def test_day_summary_uses_dst_day_boundaries_and_excludes_adjacent_days(self) -> None:
        cases = (
            (date(2026, 3, 8), datetime(2026, 3, 8, 5, tzinfo=UTC), 23),
            (date(2026, 11, 1), datetime(2026, 11, 1, 4, tzinfo=UTC), 25),
        )
        for local_date, start, hours in cases:
            with self.subTest(local_date=local_date):
                end = start + timedelta(hours=hours)
                timestamps = (
                    start - timedelta(microseconds=1),
                    start,
                    start + timedelta(hours=1.5),
                    start + timedelta(hours=2.5),
                    end - timedelta(hours=1),
                    end,
                )
                # On the fall-back day, the two middle observations both have
                # local time 01:30, but are different UTC instants.
                for index, observed_at in enumerate(reversed(timestamps)):
                    self.store.record(
                        replace(self._sample(f"dst_{hours}_{index}"), observed_at=observed_at)
                    )
                summary = read_presence_day_summary(
                    self.database, local_date=local_date, timezone_name="America/New_York"
                )
                self.assertEqual(
                    PresenceDaySummary(
                        mode=PresenceFactMode.PRIMARY,
                        local_date=local_date,
                        timezone_name="America/New_York",
                        start_at=start,
                        end_at=end,
                        streams=(
                            PresenceStreamDaySummary(
                                "cam_01",
                                4,
                                0,
                                4,
                                0,
                                start,
                                end - timedelta(hours=1),
                                (hours - 3.5) * 3600,
                            ),
                        ),
                    ),
                    summary,
                )

    def test_day_summary_rejects_invalid_tail_after_many_valid_rows(self) -> None:
        connection = sqlite3.connect(self.database)
        try:
            connection.executemany(
                "INSERT INTO presence_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        f"sample_bulk_{index}",
                        "primary",
                        "cam_01",
                        (self.BASE + timedelta(seconds=index)).isoformat(),
                        "clear",
                        "scanner_v1",
                        "detector_v1",
                        "a" * 64 if index < 4096 else "invalid_fingerprint",
                        None,
                        self.BASE.isoformat(),
                    )
                    for index in range(4097)
                ),
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(PresenceHistoryError, "row is invalid"):
            read_presence_day_summary(
                self.database,
                local_date=date(2026, 8, 19),
                timezone_name="Asia/Shanghai",
            )

    def test_day_summary_memory_stays_bounded_as_fact_count_grows(self) -> None:
        small = benchmark_presence_summary(samples=2000, cameras=4)
        large = benchmark_presence_summary(samples=20000, cameras=4)
        for result, samples, counts in (
            (small, 2000, (667, 667, 666)),
            (large, 20000, (6667, 6667, 6666)),
        ):
            with self.subTest(samples=samples):
                self.assertEqual(samples, result["samples"])
                self.assertEqual(4, result["cameras"])
                self.assertEqual(counts, result["state_counts"])
        small_peak = small["python_peak_bytes"]
        large_peak = large["python_peak_bytes"]
        assert isinstance(small_peak, int)
        assert isinstance(large_peak, int)
        # This bounds Python query/aggregation allocations, not SQLite native
        # memory or RSS. Leave headroom for runtime and allocator variation.
        self.assertLessEqual(large_peak - small_peak, 256 * 1024)
        self.assertLessEqual(large_peak, 512 * 1024)

    def test_invalid_stored_fact_is_rejected(self) -> None:
        connection = sqlite3.connect(self.database)
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "INSERT INTO presence_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "sample_bad",
                "primary",
                "cam_01",
                self.BASE.isoformat(),
                "invalid",
                "scanner_v1",
                "detector_v1",
                "a" * 64,
                None,
                self.BASE.isoformat(),
            ),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(PresenceHistoryError, "row is invalid"):
            read_presence_day_summary(
                self.database,
                local_date=date(2026, 8, 19),
                timezone_name="Asia/Shanghai",
            )


if __name__ == "__main__":
    unittest.main()
