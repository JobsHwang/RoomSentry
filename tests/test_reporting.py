from __future__ import annotations

import os
import stat
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor.adapters.vlm import VLMAdapter
from baby_monitor.config import StorageSettings
from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    EventMedia,
    EventType,
    EvidenceGap,
    RecognitionProvenance,
    RecognitionRunMetrics,
    RiskLevel,
    VLMEvent,
)
from baby_monitor.home_dashboard.copy import _DAILY_DIGEST_REASON_LABELS
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy
from baby_monitor.presence_history import PresenceFactMode, PresenceFactStore
from baby_monitor.recognition_attempt import (
    RecognitionAttemptAuditStore,
    RecognitionAttemptStatus,
    RecognitionStage,
)
from baby_monitor.reporting import DailyReportGenerator
from baby_monitor.reporting_rendering import (
    _coverage_label,
    _coverage_overview_lines,
    _coverage_status_label,
)
from baby_monitor.room_presence import StreamPresenceSample, StreamPresenceSampleState
from baby_monitor.storage import StorageResolver
from tests.test_pipeline import StaticResultVLM


class ReportFixtureVLM(VLMAdapter):
    name = "report_fixture"

    @property
    def provenance(self) -> RecognitionProvenance:
        return RecognitionProvenance(
            adapter_name=self.name,
            model_revision="fixture_revision_abc",
        )

    def analyze(self, request):  # type: ignore[no-untyped-def]
        return VLMEvent(
            event_type=EventType.SLEEPING,
            subjects=(),
            observations=(
                "<script>alert(1)</script>",
                "![external](https://example.invalid/private)",
            ),
            inferences=("宝宝可能睡着。\n## injected heading",),
            risk_level=RiskLevel.MEDIUM,
            confidence=0.72,
            needs_remote_review=False,
            summary_zh='<img src="https://example.invalid/pixel"> 模型摘要',
            evidence_frames=tuple(request.media.frame_paths),
        )


class DailyReportTests(unittest.TestCase):
    def test_local_pending_and_completed_secondary_have_distinct_report_states(self) -> None:
        for use_secondary in (False, True):
            with self.subTest(secondary=use_secondary), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                database = EventDatabase(root / "events.sqlite3")
                order = []
                adapters = {
                    name: StaticResultVLM(
                        name=name, event_type=EventType.UNKNOWN, confidence=0.9, call_order=order
                    )
                    for name in ("primary", "secondary")
                }
                process_prepared_event(
                    candidate=CandidateEvent("coverage_review", "cam_01", 0.0, 1.0, ()),
                    occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
                    source_kind="fixture",
                    source_ref="synthetic-review-fixture",
                    media=EventMedia("", (str(root / "synthetic-frame.jpg"),)),
                    local_vlm=adapters["primary"],
                    secondary_vlm=adapters["secondary"] if use_secondary else None,
                    database=database,
                    policy=ReviewPolicy(),
                )
                report = DailyReportGenerator(
                    database=database,
                    storage=StorageResolver(
                        StorageSettings(local_root=root / "reports", database=database.path)
                    ),
                    timezone_name="Asia/Shanghai",
                ).generate(date(2026, 8, 3))
                for path in (report.markdown_path, report.html_path):
                    content = path.read_text(encoding="utf-8")
                    self.assertIn("未验证连续覆盖", content)
                    self.assertIn(f"本地待复核：{'否' if use_secondary else '是'}", content)
                    self.assertNotIn("远端复核 待复核", content)
                    if use_secondary:
                        self.assertIn("未记录到 medium/high 风险或待复核事件", content)
                        self.assertNotIn("本地待复核 是", content)
                    else:
                        self.assertIn("本地待复核 是 · 远端复核 否", content)
                        self.assertNotIn("未记录到 medium/high 风险或待复核事件", content)

    def test_legacy_empty_gap_media_is_unverified_not_complete_in_report_labels(self) -> None:
        media = EventMedia.from_dict(
            {
                "version": 1,
                "clip_path": "",
                "frame_paths": [],
                "coverage_gaps": [],
                "selected_segments": 0,
            }
        )
        self.assertTrue(media.complete_coverage)  # Legacy shape is not certification.
        event = SimpleNamespace(evidence=media)
        self.assertEqual("未验证连续覆盖", _coverage_label(event))
        self.assertEqual("未验证连续覆盖", _coverage_status_label(event))

    def test_coverage_overview_counts_events_without_promoting_unknown_to_missing(self) -> None:
        started = datetime(2026, 8, 3, tzinfo=UTC)
        gap = EvidenceGap(started, started + timedelta(seconds=1), "source_disconnect")
        unknown = SimpleNamespace(evidence=EventMedia("", ()))
        recorded = SimpleNamespace(evidence=EventMedia("", (), coverage_gaps=(gap, gap)))
        lines = _coverage_overview_lines([unknown, unknown, recorded])
        self.assertIn("未验证连续覆盖事件：2", lines)
        self.assertIn("已记录证据缺口事件：1", lines)
        self.assertIn("覆盖未验证也不等于已确认缺失", "".join(lines))
        self.assertEqual("已记录缺口", _coverage_status_label(recorded))
        self.assertEqual(
            "已记录缺口（source_disconnect, source_disconnect）", _coverage_label(recorded)
        )

    def test_historical_digest_coverage_score_label_does_not_certify_continuity(self) -> None:
        self.assertEqual("连续覆盖加分（未验证）", _DAILY_DIGEST_REASON_LABELS["coverage_complete"])

    def test_markdown_and_html_share_safety_coverage_and_review_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = EventDatabase(root / "events.sqlite3")
            storage = StorageResolver(
                StorageSettings(local_root=root / "storage", database=database.path)
            )
            started = datetime(2026, 8, 3, 1, 0, tzinfo=UTC)
            with patch("baby_monitor.pipeline.datetime", wraps=datetime) as clock:
                for index in range(2):
                    occurred_at = started + timedelta(minutes=20 * index)
                    clock.now.return_value = occurred_at
                    frame = root / "media" / f"frame {index + 1:02d}.jpg"
                    frame.parent.mkdir(parents=True, exist_ok=True)
                    frame.write_bytes(b"synthetic-fixture")
                    clip = root / "media" / f"clip_{index + 1:02d}.mp4"
                    clip.write_bytes(b"synthetic-fixture")
                    gaps = ()
                    if index == 0:
                        gaps = (
                            EvidenceGap(
                                started_at=occurred_at - timedelta(seconds=2),
                                ended_at=occurred_at - timedelta(seconds=1),
                                reason="source_disconnect",
                            ),
                        )
                    process_prepared_event(
                        candidate=CandidateEvent(
                            event_id=f"report_event_{index + 1:02d}",
                            camera_id="cam_01",
                            start_seconds=0.0,
                            end_seconds=1.0,
                            detections=(),
                        ),
                        occurred_at=occurred_at,
                        source_kind="fixture",
                        source_ref="synthetic-report-fixture",
                        media=EventMedia(
                            clip_path=str(clip),
                            frame_paths=(str(frame),),
                            coverage_gaps=gaps,
                        ),
                        local_vlm=ReportFixtureVLM(),
                        database=database,
                        policy=ReviewPolicy(),
                        remote_uploads_allowed=False,
                    )

            failed_started = started + timedelta(minutes=50)
            audit = RecognitionAttemptAuditStore(database.path)
            failed_id = audit.start(
                event_id="report_failed_attempt",
                stage=RecognitionStage.LOCAL_PRIMARY,
                provenance=RecognitionProvenance(adapter_name="report_fixture"),
                started_at=failed_started,
            )
            audit.complete(
                failed_id,
                status=RecognitionAttemptStatus.FAILED,
                completed_at=failed_started + timedelta(seconds=1),
                metrics=RecognitionRunMetrics(
                    duration_ms=1_000,
                    json_compliant=False,
                ),
                error_code="invalid_structured_output",
            )
            presence_store = PresenceFactStore(database.path)
            presence_store.record(
                StreamPresenceSample(
                    sample_id="primary_report_sample",
                    camera_id="cam_01",
                    observed_at=started,
                    state=StreamPresenceSampleState.CLEAR,
                    scanner_id="primary_scanner",
                    detector_id="primary_detector",
                    source_fingerprint="a" * 64,
                )
            )
            presence_store.record(
                StreamPresenceSample(
                    sample_id="shadow_report_sample",
                    camera_id="cam_01",
                    observed_at=started,
                    state=StreamPresenceSampleState.PERSON_DETECTED,
                    scanner_id="pose_shadow",
                    detector_id="pose_fusion_detector",
                    source_fingerprint="a" * 64,
                ),
                mode=PresenceFactMode.SHADOW,
            )

            artifacts = DailyReportGenerator(
                database=database,
                storage=storage,
                timezone_name="Asia/Shanghai",
            ).generate(date(2026, 8, 3))
            markdown = artifacts.markdown_path.read_text(encoding="utf-8")
            rendered_html = artifacts.html_path.read_text(encoding="utf-8")

            self.assertEqual(2, artifacts.event_count)
            for section in (
                "概览",
                "数据覆盖",
                "识别质量与运行覆盖",
                "睡眠区间（模型推测）",
                "异常与待复核",
            ):
                self.assertIn(section, markdown)
                self.assertIn(section, rendered_html)
            self.assertIn("严格 JSON 合规率：2/3（66.7%）", markdown)
            self.assertIn("严格 JSON 合规率：2/3（66.7%）", rendered_html)
            self.assertIn("成功：2；失败：1", markdown)
            self.assertIn("成功：2；失败：1", rendered_html)
            self.assertIn("09:00:00–09:20:00", markdown)
            self.assertIn("09:00:00–09:20:00", rendered_html)
            self.assertIn("source\\_disconnect", markdown)
            self.assertIn("source_disconnect", rendered_html)
            for content in (markdown, rendered_html):
                self.assertIn("未验证连续覆盖事件：1", content)
                self.assertIn("已记录证据缺口事件：1", content)
                self.assertIn("覆盖未验证也不等于已确认缺失", content)
                self.assertIn("证据覆盖：未验证连续覆盖", content)
                self.assertIn("证据覆盖：已记录缺口（", content)
                self.assertNotIn("证据覆盖不完整事件", content)
                self.assertNotIn("证据覆盖：完整", content)
            self.assertIn("| 未验证连续覆盖 |", markdown)
            self.assertIn("| 已记录缺口 |", markdown)
            self.assertIn("<td>未验证连续覆盖</td>", rendered_html)
            self.assertIn("<td>已记录缺口</td>", rendered_html)
            self.assertNotIn("| 完整 |", markdown)
            self.assertNotIn("<td>完整</td>", rendered_html)
            self.assertIn("待复核", markdown)
            self.assertIn("待复核", rendered_html)
            self.assertIn("local_primary", markdown)
            self.assertIn("local_primary", rendered_html)
            self.assertIn("0.72", markdown)
            self.assertIn("0.72", rendered_html)
            self.assertIn("report_fixture @ fixture_revision_abc", markdown)
            self.assertIn("report_fixture @ fixture_revision_abc", rendered_html)
            self.assertIn("frame%2001.jpg", markdown)
            self.assertIn('src="', rendered_html)
            self.assertIn("frame%2001.jpg", rendered_html)
            self.assertIn("已落库 presence 样本：1", markdown)
            self.assertIn("已落库 presence 样本：1", rendered_html)
            self.assertIn("Detector + Pose shadow（实验旁路）", markdown)
            self.assertIn("Detector + Pose shadow（实验旁路）", rendered_html)
            self.assertIn("实验旁路样本：1", markdown)
            self.assertIn("实验旁路样本：1", rendered_html)
            self.assertIn("与主 presence 分开计数", markdown)
            self.assertIn("与主 presence 分开计数", rendered_html)

            self.assertNotIn("<script>", markdown)
            self.assertNotIn("<script>", rendered_html)
            self.assertIn("&lt;script&gt;", markdown)
            self.assertIn("&lt;script&gt;", rendered_html)
            self.assertNotIn("![external](", markdown)
            self.assertNotIn("\n## injected heading", markdown)

    def test_empty_day_never_claims_safety_or_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = EventDatabase(root / "events.sqlite3")
            database.initialize()
            artifacts = DailyReportGenerator(
                database=database,
                storage=StorageResolver(
                    StorageSettings(local_root=root / "storage", database=database.path)
                ),
                timezone_name="Asia/Shanghai",
            ).generate(date(2026, 8, 3))
            markdown = artifacts.markdown_path.read_text(encoding="utf-8")
            rendered_html = artifacts.html_path.read_text(encoding="utf-8")

            self.assertEqual(0, artifacts.event_count)
            for content in (markdown, rendered_html):
                self.assertIn("未验证连续覆盖事件：0", content)
                self.assertIn("已记录证据缺口事件：0", content)
                self.assertIn("不能证明全天覆盖完整", content)
                self.assertIn("不代表环境安全", content)
                self.assertIn("不能据此断言没有睡眠", content)
                self.assertIn("无法计算", content)

    def test_detailed_reports_are_atomic_private_files_independent_of_umask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = EventDatabase(root / "events.sqlite3")
            database.initialize()
            generator = DailyReportGenerator(
                database=database,
                storage=StorageResolver(
                    StorageSettings(local_root=root / "storage", database=database.path)
                ),
                timezone_name="Asia/Shanghai",
            )

            previous_umask = os.umask(0)
            try:
                artifacts = generator.generate(date(2026, 8, 3))
                artifacts.markdown_path.chmod(0o644)
                artifacts.html_path.chmod(0o644)
                artifacts = generator.generate(date(2026, 8, 3))
            finally:
                os.umask(previous_umask)

            for path in (artifacts.markdown_path, artifacts.html_path):
                self.assertEqual(0o600, stat.S_IMODE(path.stat().st_mode))
                self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(
                [],
                list(artifacts.markdown_path.parent.glob(".*.tmp")),
            )

            outside = root / "outside.html"
            outside.write_text("do-not-overwrite", encoding="utf-8")
            artifacts.html_path.unlink()
            artifacts.html_path.symlink_to(outside)
            artifacts = generator.generate(date(2026, 8, 3))

            self.assertEqual("do-not-overwrite", outside.read_text(encoding="utf-8"))
            self.assertFalse(artifacts.html_path.is_symlink())
            self.assertEqual(0o600, stat.S_IMODE(artifacts.html_path.stat().st_mode))


if __name__ == "__main__":
    unittest.main()
