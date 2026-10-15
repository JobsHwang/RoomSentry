from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from baby_monitor.config import StorageSettings
from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    EventMedia,
    EventType,
    EvidenceGap,
    ReviewTier,
    RiskLevel,
)
from baby_monitor.home_event_digest import read_home_event_digest
from baby_monitor.notification import NotificationMessage, ReviewStatus
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy, record_secondary_result
from baby_monitor.reporting import DailyReportGenerator
from baby_monitor.ring_buffer import BufferedEvidenceResult
from baby_monitor.storage import StorageResolver
from tests.test_pipeline import StaticResultVLM
from tests.test_policy import _result


class EvidenceCoverageStatusTests(unittest.TestCase):
    def test_buffer_cli_result_keeps_legacy_flag_but_exposes_unknown_authority(self):
        media = EventMedia("clip.mp4", ("frame.jpg",), selected_segments=1)
        result = BufferedEvidenceResult(
            media=media,
            requested_start="2026-09-12T00:00:00+00:00",
            requested_end="2026-09-12T00:00:02+00:00",
            covered_start="2026-09-12T00:00:00+00:00",
            covered_end="2026-09-12T00:00:02+00:00",
            complete_coverage=True,
            coverage_gaps=(),
            selected_segments=1,
            reused_existing=False,
        ).to_dict()
        self.assertTrue(result["complete_coverage"])
        self.assertEqual("unverified", result["coverage_status"])
        self.assertFalse(result["continuous_coverage_verified"])

    def test_old_gap_free_media_round_trips_without_certifying_continuity(self):
        media = EventMedia("clip.mp4", ("frame.jpg",), selected_segments=3)
        payload = media.to_dict()
        self.assertTrue(media.complete_coverage)  # Legacy absence-of-gaps flag.
        self.assertEqual("unverified", media.coverage_status)
        self.assertNotIn("coverage_status", payload)
        self.assertEqual(payload, EventMedia.from_dict(payload).to_dict())
        self.assertEqual("unverified", EventMedia.from_dict(payload).coverage_status)
        self.assertEqual((), media.coverage_gaps)

    def test_recorded_gap_remains_distinct_from_unverified_coverage(self):
        start = datetime(2026, 9, 12, tzinfo=UTC)
        gap = EvidenceGap(start, start + timedelta(seconds=3), "segment_missing")
        media = EventMedia("clip.mp4", ("frame.jpg",), coverage_gaps=(gap,))
        self.assertFalse(media.complete_coverage)
        self.assertEqual("gaps_recorded", media.coverage_status)
        self.assertEqual((gap,), EventMedia.from_dict(media.to_dict()).coverage_gaps)

    def test_unverified_coverage_requests_only_local_review_and_keeps_risk(self):
        for remote_allowed in (False, True):
            with self.subTest(remote_allowed=remote_allowed):
                result = _result(confidence=0.99, risk_level=RiskLevel.NONE)
                decision = ReviewPolicy().decide(result, remote_allowed=remote_allowed)
                self.assertEqual(ReviewTier.LOCAL_SECONDARY, decision.review_tier)
                self.assertEqual(("evidence_coverage_unverified",), decision.review_reasons)
                self.assertFalse(decision.needs_remote_review)
                self.assertTrue(decision.needs_local_review)
                self.assertTrue(decision.needs_review)
                self.assertFalse(decision.remote_used)
                self.assertEqual(RiskLevel.NONE, decision.risk_level)
                self.assertEqual(0.99, result.confidence)

    def test_known_gap_uses_gap_reason_not_unknown_or_invented_missing_interval(self):
        decision = ReviewPolicy().decide(_result(), evidence_complete=False)
        self.assertEqual(("evidence_coverage_gap",), decision.review_reasons)
        self.assertEqual(ReviewTier.LOCAL_SECONDARY, decision.review_tier)

    def test_second_recognizer_agreement_cannot_certify_continuity(self):
        result = _result()
        decision = record_secondary_result(
            ReviewPolicy().decide(result), primary=result, secondary=result, final=result
        )
        self.assertIn("evidence_coverage_unverified", decision.review_reasons)
        self.assertFalse(decision.needs_remote_review)
        self.assertFalse(decision.needs_local_review)
        self.assertFalse(decision.needs_review)

    def test_existing_remote_reason_is_not_downgraded_by_unknown_coverage(self):
        decision = ReviewPolicy().decide(_result(confidence=0.1), remote_allowed=False)
        self.assertEqual(ReviewTier.REMOTE, decision.review_tier)
        self.assertIn("low_confidence", decision.review_reasons)
        self.assertIn("evidence_coverage_unverified", decision.review_reasons)
        self.assertFalse(decision.remote_allowed)

    def test_legacy_coverage_flag_still_requires_a_boolean(self):
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReviewPolicy().decide(_result(), evidence_complete=value)

    def test_unverified_media_survives_pipeline_storage_report_and_idempotent_replay(self):
        for use_secondary in (False, True):
            with self.subTest(secondary=use_secondary), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                database = EventDatabase(root / "events.sqlite3")
                order = []
                adapters = {
                    name: StaticResultVLM(
                        name=name, event_type=EventType.UNKNOWN, confidence=0.9, call_order=order
                    )
                    for name in ("primary", "secondary", "remote")
                }
                media = EventMedia(str(root / "clip.mp4"), (str(root / "frame.jpg"),))
                payload = media.to_dict()
                kwargs = dict(
                    candidate=CandidateEvent("coverage_unknown", "cam_01", 0.0, 1.0, ()),
                    occurred_at=datetime(2026, 9, 12, tzinfo=UTC),
                    source_kind="file",
                    source_ref="fixture.mp4",
                    media=media,
                    local_vlm=adapters["primary"],
                    secondary_vlm=adapters["secondary"] if use_secondary else None,
                    remote_vlm=adapters["remote"],
                    remote_uploads_allowed=True,
                    database=database,
                    policy=ReviewPolicy(),
                )
                result = process_prepared_event(**kwargs)
                self.assertEqual(use_secondary, result.secondary_reviewed)
                self.assertEqual(["primary", "secondary"] if use_secondary else ["primary"], order)
                self.assertEqual(0, adapters["remote"].calls)
                stored = database.get_event("coverage_unknown")
                self.assertIsNotNone(stored)
                self.assertEqual(payload, stored.evidence.to_dict())
                self.assertEqual("unverified", stored.evidence.coverage_status)
                self.assertEqual(("evidence_coverage_unverified",), stored.decision.review_reasons)
                self.assertEqual(RiskLevel.NONE, stored.result.risk_level)
                self.assertFalse(stored.remote_used)
                self.assertEqual(not use_secondary, stored.decision.needs_local_review)
                self.assertEqual(not use_secondary, stored.decision.needs_review)
                database.path.chmod(0o600)
                digest = read_home_event_digest(
                    database.path, now=datetime(2026, 9, 12, 1, tzinfo=UTC)
                )
                self.assertEqual(not use_secondary, digest.recent[0].needs_review)
                notification = NotificationMessage.from_event(
                    stored, sink_kind="stdout", sink_id="test"
                )
                self.assertEqual(
                    ReviewStatus.FINAL if use_secondary else ReviewStatus.PENDING_REVIEW,
                    notification.review_status,
                )
                self.assertTrue(process_prepared_event(**kwargs).reused_existing)
                self.assertEqual(1, adapters["primary"].calls)
                self.assertEqual(int(use_secondary), adapters["secondary"].calls)
                report = DailyReportGenerator(
                    database=database,
                    storage=StorageResolver(
                        StorageSettings(local_root=root, database=database.path)
                    ),
                    timezone_name="Asia/Shanghai",
                ).generate(date(2026, 9, 12))
                for path in (report.markdown_path, report.html_path):
                    content = path.read_text(encoding="utf-8")
                    self.assertIn("未验证连续覆盖", content)
                    self.assertNotIn("证据覆盖：完整", content)


if __name__ == "__main__":
    unittest.main()
