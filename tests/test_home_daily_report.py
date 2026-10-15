from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from baby_monitor.home_daily_report import (
    HomeDailyReportError,
    generate_home_daily_report,
    inspect_home_daily_report,
    read_home_daily_report_html,
)
from baby_monitor.presence_history import (
    PRESENCE_FACT_SCHEMA_SQL,
    PresenceFactMode,
    PresenceFactStore,
)
from baby_monitor.room_presence import StreamPresenceSample, StreamPresenceSampleState
from baby_monitor.sampling_history import SamplingEvidenceStore, SamplingHistoryError

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


class HomeDailyReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o700)
        self.state = self.root / "state"
        self.state.mkdir(mode=0o700)
        self.database = self.state / "events.sqlite3"
        connection = sqlite3.connect(self.database)
        connection.execute(_SCHEMA)
        connection.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
            (
                (
                    "PRIVATE_TODAY_EVENT",
                    "2026-08-19T01:00:00+00:00",
                    "movement",
                    "low",
                    0.75,
                    0,
                ),
                (
                    "PRIVATE_OLDER_EVENT",
                    "2026-08-18T01:00:00+00:00",
                    "crying",
                    "medium",
                    0.65,
                    1,
                ),
            ),
        )
        connection.commit()
        connection.close()
        self.database.chmod(0o600)
        self.output = self.state / "daily-reports"
        self.now = datetime(2026, 8, 19, 12, tzinfo=UTC)

    def _generate(self):
        return generate_home_daily_report(
            database_path=self.database,
            output_root=self.output,
            now=self.now,
        )

    def _formats(self) -> tuple[str, str]:
        return tuple(
            (self.output / f"2026-08-19_daily_report.{extension}").read_text(encoding="utf-8")
            for extension in ("md", "html")
        )

    def _record_sampling(self, *, all_empty: bool = False, unknown: bool = False) -> None:
        store = SamplingEvidenceStore(self.database)
        store.initialize()
        ledger = None
        if not unknown:
            ledger = DetectorSamplingLedger(
                native_fps=25,
                native_fps_source="reported",
                requested_sample_fps=1,
                frame_step=25,
                decoded_frame_count=51,
                samples=(
                    DetectorSampleFrame(0, 0, "container_pts", 0, 0),
                    DetectorSampleFrame(25, 1, "nominal_fps", 0, 0 if all_empty else 1),
                    DetectorSampleFrame(50, 2, "container_pts", 0 if all_empty else 1, 0),
                ),
                timestamp_fallback_frame_count=1,
                max_adjacent_container_pts_gap_seconds=0.125,
                container_pts_regression_count=1,
            )
        store.record(
            camera_id="private_camera",
            scanner_id="private_scanner",
            source_fingerprint=("b" if unknown else "a") * 64,
            started_at=self.now,
            ended_at=self.now + timedelta(seconds=3),
            ledger=ledger,
        )

    def test_old_database_sampling_unknown_and_no_report_migration(self) -> None:
        before = self.database.read_bytes(), self.database.stat().st_mtime_ns
        self._generate()
        self.assertEqual(before, (self.database.read_bytes(), self.database.stat().st_mtime_ns))
        for rendered in self._formats():
            self.assertIn("实际检查记录", rendered)
            self.assertIn("实际检查历史表尚不可用", rendered)
            self.assertIn("调用次数未知", rendered)
            self.assertNotIn("实际模型调用（已记录）：0", rendered)

    def test_empty_sampling_day_is_zero_not_missing_table(self) -> None:
        SamplingEvidenceStore(self.database).initialize()
        before = self.database.read_bytes(), self.database.stat().st_mtime_ns
        self._generate()
        self.assertEqual(before, (self.database.read_bytes(), self.database.stat().st_mtime_ns))
        for rendered in self._formats():
            self.assertIn("已记录扫描：0", rendered)
            self.assertIn("实际模型调用（已记录）：0", rendered)
            self.assertIn("当天没有已落库扫描", rendered)
            self.assertIn("无可计算记录", rendered)
            self.assertNotIn("实际检查历史表尚不可用", rendered)

    def test_sampling_counts_match_in_markdown_and_html_with_uncertainty(self) -> None:
        self._record_sampling()
        self._record_sampling(unknown=True)
        receipt = self._generate()
        self.assertEqual(1, receipt.event_count)
        for rendered in self._formats():
            for expected in (
                "已记录扫描：2",
                "实际模型调用（已记录）：3",
                "其中零框调用：2",
                "缺少实际采样证据的扫描：1",
                "使用回退时间戳的采样：1",
                "源容器 PTS 回退次数：1",
                "相邻源帧最大容器 PTS 间隔：0.125 秒",
                "不是确认丢失时长",
                "不覆盖扫描之间的空档",
                "不能当作当天完整调用总数",
                "并非连续监护",
            ):
                self.assertIn(expected, rendered)
            self.assertNotIn("private_camera", rendered)
            self.assertNotIn("private_scanner", rendered)
            self.assertNotIn("a" * 64, rendered)
            self.assertNotIn(str(self.database), rendered)

    def test_all_zero_box_calls_do_not_mean_absent_or_safe(self) -> None:
        self._record_sampling(all_empty=True)
        self._generate()
        for rendered in self._formats():
            self.assertIn("实际模型调用（已记录）：3", rendered)
            self.assertIn("其中零框调用：3", rendered)
            self.assertIn("不表示画面无人、宝宝不在场或环境安全", rendered)
            self.assertIn("不是人数、独立画面数或连续观察时长", rendered)

    def test_unknown_only_scan_is_not_backfilled_as_model_call(self) -> None:
        self._record_sampling(unknown=True)
        self._generate()
        for rendered in self._formats():
            self.assertIn("已记录扫描：1", rendered)
            self.assertIn("实际模型调用（已记录）：0", rendered)
            self.assertIn("缺少实际采样证据的扫描：1", rendered)
            self.assertNotIn("当天没有已落库扫描", rendered)

    def test_sampling_failure_is_path_free_and_does_not_publish(self) -> None:
        with (
            patch(
                "baby_monitor.home_daily_report.read_sampling_day_summary",
                side_effect=SamplingHistoryError(f"private failure {self.database}"),
            ),
            self.assertRaisesRegex(HomeDailyReportError, "database is unavailable") as caught,
        ):
            self._generate()
        self.assertNotIn(str(self.database), str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertFalse(self.output.exists())

    def test_sampling_html_escapes_all_rendered_lines(self) -> None:
        with patch(
            "baby_monitor.home_daily_report._sampling_markdown_lines",
            return_value=['- <script>alert("synthetic")</script> &'],
        ):
            self._generate()
        _markdown, html = self._formats()
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;alert(&quot;synthetic&quot;)&lt;/script&gt; &amp;", html)

    def test_sampling_failure_preserves_previous_report_pair(self) -> None:
        self._generate()
        before = self._formats()
        with (
            patch(
                "baby_monitor.home_daily_report.read_sampling_day_summary",
                side_effect=SamplingHistoryError("invalid sampling history"),
            ),
            self.assertRaises(HomeDailyReportError),
        ):
            self._generate()
        self.assertEqual(before, self._formats())
        self.assertEqual(2, len(list(self.output.iterdir())))

    def test_invalid_sampling_schema_fails_closed_without_migration(self) -> None:
        connection = sqlite3.connect(self.database)
        connection.execute("CREATE TABLE sampling_scans (unexpected TEXT)")
        connection.commit()
        connection.close()
        before = self.database.read_bytes(), self.database.stat().st_mtime_ns
        with self.assertRaisesRegex(HomeDailyReportError, "database is unavailable"):
            self._generate()
        self.assertEqual(before, (self.database.read_bytes(), self.database.stat().st_mtime_ns))
        self.assertFalse(self.output.exists())

    def test_report_is_private_aggregate_and_path_free(self) -> None:
        receipt = self._generate()
        public = receipt.to_public_dict()
        self.assertEqual("2026-08-19", public["local_date"])
        self.assertEqual(1, public["event_count"])
        self.assertFalse(public["media_read"])
        self.assertFalse(public["risk_authority"])
        self.assertFalse(public["notification_decision"])
        self.assertNotIn(str(self.root), json.dumps(public))

        files = sorted(self.output.iterdir())
        self.assertEqual(2, len(files))
        self.assertEqual(0o700, self.output.stat().st_mode & 0o777)
        self.assertTrue(all((path.stat().st_mode & 0o777) == 0o600 for path in files))
        combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
        self.assertIn("2026-08-19", combined)
        self.assertIn("活动分类", combined)
        self.assertIn("75%", combined)
        self.assertNotIn("PRIVATE_", combined)
        self.assertNotIn(str(self.root), combined)
        self.assertNotIn("<img", combined)

        html = read_home_daily_report_html(
            database_path=self.database,
            output_root=self.output,
            now=self.now,
        )
        self.assertTrue(html.startswith(b"<!doctype html>"))
        status = inspect_home_daily_report(
            database_path=self.database,
            output_root=self.output,
            now=self.now,
        )
        self.assertEqual("available", status["status"])
        self.assertTrue(status["available"])
        self.assertEqual(1, status["event_count"])

    def test_regeneration_updates_managed_pair_without_extra_files(self) -> None:
        self._generate()
        connection = sqlite3.connect(self.database)
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
            (
                "PRIVATE_SECOND_EVENT",
                "2026-08-19T02:00:00+00:00",
                "unknown",
                "none",
                0.5,
                1,
            ),
        )
        connection.commit()
        connection.close()
        second = self._generate()
        self.assertEqual(2, second.event_count)
        self.assertEqual(2, len(list(self.output.iterdir())))
        self.assertIn(
            "2 条事件",
            read_home_daily_report_html(
                database_path=self.database,
                output_root=self.output,
                now=self.now,
            ).decode("utf-8"),
        )

    def test_primary_and_shadow_presence_are_reported_separately(self) -> None:
        connection = sqlite3.connect(self.database)
        connection.executescript(PRESENCE_FACT_SCHEMA_SQL)
        connection.commit()
        connection.close()
        store = PresenceFactStore(self.database)
        observed_at = datetime(2026, 8, 19, 2, tzinfo=UTC)
        primary = StreamPresenceSample(
            sample_id="primary_sample_01",
            camera_id="cam_01",
            observed_at=observed_at,
            state=StreamPresenceSampleState.CLEAR,
            scanner_id="primary_scanner",
            detector_id="primary_detector",
            source_fingerprint="a" * 64,
        )
        shadow = StreamPresenceSample(
            sample_id="shadow_sample_01",
            camera_id="cam_01",
            observed_at=observed_at,
            state=StreamPresenceSampleState.PERSON_DETECTED,
            scanner_id="pose_shadow",
            detector_id="pose_fusion_detector",
            source_fingerprint="a" * 64,
        )
        store.record(primary)
        store.record(shadow, mode=PresenceFactMode.SHADOW)

        receipt = self._generate()
        self.assertEqual(1, receipt.presence_sample_count)
        self.assertEqual(1, receipt.shadow_presence_sample_count)
        self.assertTrue(receipt.presence_history_available)
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted(self.output.iterdir())
        )
        self.assertIn("已落库 presence 样本：1", combined)
        self.assertIn("shadow 样本：1", combined)
        self.assertIn("shadow 与主 presence 分开计数", combined)
        self.assertIn("不是官方准确率或部署批准", combined)
        for path in self.output.iterdir():
            rendered = path.read_text(encoding="utf-8")
            self.assertIn("业务事实条数，不是模型调用次数", rendered)
            self.assertIn("缺口评估边界，不是成功检查时间", rendered)
            self.assertIn("历史记录未自动迁移", rendered)

    def test_status_does_not_create_output_and_unsafe_targets_fail_closed(self) -> None:
        status = inspect_home_daily_report(
            database_path=self.database,
            output_root=self.output,
            now=self.now,
        )
        self.assertEqual("not_generated", status["status"])
        self.assertFalse(self.output.exists())

        with self.assertRaisesRegex(HomeDailyReportError, "output is unavailable"):
            generate_home_daily_report(
                database_path=self.database,
                output_root=self.root / "outside",
                now=self.now,
            )

        self.output.mkdir(mode=0o755)
        with self.assertRaisesRegex(HomeDailyReportError, "output is unavailable"):
            self._generate()

    @unittest.skipUnless(os.name == "posix", "POSIX symlink policy required")
    def test_symlink_report_is_rejected(self) -> None:
        self.output.mkdir(mode=0o700)
        outside = self.root / "outside.html"
        outside.write_text("outside", encoding="utf-8")
        outside.chmod(0o600)
        (_markdown, html_path) = (
            self.output / "2026-08-19_daily_report.md",
            self.output / "2026-08-19_daily_report.html",
        )
        html_path.symlink_to(outside)
        with self.assertRaisesRegex(HomeDailyReportError, "unavailable"):
            self._generate()


if __name__ == "__main__":
    unittest.main()
