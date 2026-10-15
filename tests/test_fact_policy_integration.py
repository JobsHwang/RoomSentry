from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.adapters.vlm import VLMAdapter
from baby_monitor.config import StorageSettings
from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    Detection,
    EventMedia,
    EventType,
    RiskLevel,
    Subject,
    SubjectKind,
    VLMEvent,
)
from baby_monitor.notification import (
    FileNotificationSink,
    NotificationMessage,
    NotificationOutbox,
    ReviewStatus,
    deliver_notification_once,
)
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy
from baby_monitor.reporting import DailyReportGenerator
from baby_monitor.storage import StorageResolver

_CLAIM = "宝宝正在哭闹"


class _SummaryOnlyClaimVLM(VLMAdapter):
    name = "synthetic_summary_only"

    def __init__(self, *, event_type: EventType = EventType.UNKNOWN, claim: str = _CLAIM):
        self.event_type = event_type
        self.claim = claim
        self.calls = 0

    def analyze(self, request):
        self.calls += 1
        return VLMEvent(
            event_type=self.event_type,
            subjects=(Subject(SubjectKind.PERSON, 0.9, request.media.frame_paths),),
            observations=("画面中可见人员。",),
            inferences=(),
            risk_level=RiskLevel.NONE,
            confidence=0.97,
            needs_remote_review=False,
            summary_zh=self.claim,
            evidence_frames=request.media.frame_paths,
        )


class FactPolicyIntegrationTests(unittest.TestCase):
    NOW = datetime(2026, 9, 12, 8, 0, tzinfo=UTC)

    def _prepare(self, root: Path, *, secondary=False, event_type=EventType.UNKNOWN, claim=_CLAIM):
        frame, clip = root / "synthetic.jpg", root / "synthetic.mp4"
        frame.write_bytes(b"synthetic frame fixture")
        clip.write_bytes(b"synthetic clip fixture")
        database = EventDatabase(root / "events.sqlite3")
        candidate = CandidateEvent(
            "summary_boundary_fixture",
            "camera_01",
            0.0,
            1.0,
            (Detection(0.5, "person", 0.9, (0.1, 0.1, 0.9, 0.9)),),
        )
        primary = _SummaryOnlyClaimVLM(event_type=event_type, claim=claim)
        second = _SummaryOnlyClaimVLM(event_type=event_type, claim=claim) if secondary else None
        arguments = {
            "candidate": candidate,
            "occurred_at": self.NOW,
            "source_kind": "fixture",
            "source_ref": "synthetic_local_source",
            "media": EventMedia(str(clip), (str(frame),)),
            "local_vlm": primary,
            "secondary_vlm": second,
            "database": database,
            "policy": ReviewPolicy(),
            "remote_uploads_allowed": False,
        }
        summary = process_prepared_event(**arguments)
        # Reopen the persisted event through a new database instance, rather
        # than asserting only the policy transformation's in-memory result.
        stored = EventDatabase(database.path).get_event(candidate.event_id)
        assert stored is not None
        return database, stored, summary, arguments

    def _assert_bounded_result(self, result):
        self.assertNotIn(_CLAIM, result.summary_zh)
        self.assertFalse(any(_CLAIM in text for text in result.observations))
        self.assertTrue(any(_CLAIM in text and "模型推测" in text for text in result.inferences))
        self.assertLessEqual(result.confidence, 0.49)
        self.assertTrue(result.needs_remote_review)
        self.assertTrue(all(subject.kind is SubjectKind.PERSON for subject in result.subjects))

    def test_summary_only_identity_and_action_claim_is_bounded_before_persistence(self):
        for event_type in (EventType.UNKNOWN, EventType.CRYING):
            with self.subTest(event_type=event_type), tempfile.TemporaryDirectory() as directory:
                _, stored, summary, _ = self._prepare(Path(directory), event_type=event_type)
                self._assert_bounded_result(stored.local_result)
                self._assert_bounded_result(stored.result)
                self.assertEqual(EventType.UNKNOWN, stored.result.event_type)
                self.assertFalse(stored.remote_used)
                self.assertFalse(summary.remote_reviewed)

    def test_age_synonym_summary_is_bounded_at_both_stages_before_persistence(self):
        for claim in ("婴幼儿在床边。", "成年人在画面左侧。"):
            with self.subTest(claim=claim), tempfile.TemporaryDirectory() as directory:
                _, stored, summary, _ = self._prepare(Path(directory), secondary=True, claim=claim)
                self.assertTrue(summary.secondary_reviewed)
                for result in (stored.local_result, stored.secondary_result, stored.result):
                    self.assertNotIn(claim, result.summary_zh)
                    self.assertTrue(
                        any(claim in text and "模型推测" in text for text in result.inferences)
                    )
                    self.assertLessEqual(result.confidence, 0.49)
                    self.assertTrue(result.needs_remote_review)
                    self.assertEqual(SubjectKind.PERSON, result.subjects[0].kind)
                self.assertFalse(stored.remote_used)
                self.assertFalse(summary.remote_reviewed)

    def test_secondary_stage_and_idempotent_reuse_do_not_restore_unbounded_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            database, stored, summary, arguments = self._prepare(Path(directory), secondary=True)
            self.assertTrue(summary.secondary_reviewed)
            self.assertIsNotNone(stored.secondary_result)
            self._assert_bounded_result(stored.local_result)
            self._assert_bounded_result(stored.secondary_result)
            self._assert_bounded_result(stored.result)
            repeated = process_prepared_event(**arguments)
            self.assertTrue(repeated.reused_existing)
            self.assertEqual(1, arguments["local_vlm"].calls)
            self.assertEqual(1, arguments["secondary_vlm"].calls)
            self._assert_bounded_result(database.get_event(stored.event_id).result)

    def test_daily_markdown_and_html_keep_original_only_under_inferences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, stored, _, _ = self._prepare(root)
            report = DailyReportGenerator(
                database=database,
                storage=StorageResolver(StorageSettings(local_root=root, database=database.path)),
                timezone_name="Asia/Shanghai",
            ).generate(self.NOW.date())
            markdown = report.markdown_path.read_text(encoding="utf-8")
            html = report.html_path.read_text(encoding="utf-8")
            self.assertEqual(1, report.event_count)
            self.assertIn(stored.event_id, html)
            self.assertIn(_CLAIM, markdown)
            self.assertIn(_CLAIM, html)
            without_markdown_inferences = re.sub(
                r"\*\*模型推测（inferences）\*\*.*?\*\*证据帧\*\*", "", markdown, flags=re.DOTALL
            )
            without_html_inferences = re.sub(
                r"<h4>模型推测</h4><ul>.*?</ul>", "", html, flags=re.DOTALL
            )
            self.assertNotIn(_CLAIM, without_markdown_inferences)
            self.assertNotIn(_CLAIM, without_html_inferences)

    def test_persisted_file_notification_uses_bounded_summary_and_pending_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database, stored, _, _ = self._prepare(root)
            notifications = root / "notifications"
            notifications.mkdir(mode=0o700)
            outbox = NotificationOutbox(database.path)
            queued = outbox.enqueue(
                event=stored, sink_kind="file", sink_id="fixture_local", max_active=4, now=self.NOW
            )
            delivered = deliver_notification_once(
                outbox=outbox,
                sink=FileNotificationSink(notifications),
                sink_id="fixture_local",
                lease_seconds=30,
                retry_delay_seconds=0,
                max_attempts=1,
                now=self.NOW,
                notification_id=queued.job.notification_id,
            )
            self.assertEqual("delivered", delivered.status)
            message = NotificationMessage.from_json(
                (notifications / f"{queued.job.notification_id}.json").read_text(encoding="utf-8")
            )
            self.assertEqual(ReviewStatus.PENDING_REVIEW, message.review_status)
            self.assertLessEqual(message.confidence, 0.49)
            self.assertNotIn(_CLAIM, message.summary_zh)
            self.assertTrue(
                any(_CLAIM in text and "模型推测" in text for text in message.inferences)
            )
            public_without_inferences = {
                key: value for key, value in message.to_dict().items() if key != "inferences"
            }
            self.assertNotIn(_CLAIM, json.dumps(public_without_inferences, ensure_ascii=False))
            self.assertNotIn(str(root), message.to_json())


if __name__ == "__main__":
    unittest.main()
