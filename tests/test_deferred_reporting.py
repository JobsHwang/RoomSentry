from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from baby_monitor.adapters.vlm import MockVLMAdapter
from baby_monitor.analysis_queue import AnalysisQueue
from baby_monitor.analysis_service import AnalysisDependencies, execute_analysis_run_once
from baby_monitor.analysis_worker import WorkerAttemptState
from baby_monitor.config import Settings, StorageSettings
from baby_monitor.database import EventDatabase
from baby_monitor.domain import CandidateEvent, EventMedia
from baby_monitor.event_policy import HouseholdEventPolicy, PolicyEventRule
from baby_monitor.notification import NotificationOutbox
from baby_monitor.policy import ReviewPolicy
from baby_monitor.recognition_attempt import RecognitionAttemptAuditStore
from baby_monitor.report_queue import ReportClaimLostError, ReportRefreshQueue
from baby_monitor.report_service import execute_report_refresh_once
from baby_monitor.reporting import DailyReportGenerator, _atomic_write_private_report
from baby_monitor.storage import StorageResolver


@unittest.skipUnless(os.name == "posix", "private report publication requires POSIX locking")
class DeferredReportingTests(unittest.TestCase):
    EVENT_TIME = datetime(2026, 8, 3, 1, tzinfo=UTC)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-deferred-report-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.database = self.root / "events.sqlite3"
        self.settings = Settings(
            storage=StorageSettings(local_root=self.root, database=self.database)
        )
        self.settings = replace(
            self.settings,
            analysis_queue=replace(self.settings.analysis_queue, retry_delay_seconds=0),
        )
        self.adapter = MockVLMAdapter(name="deferred_fixture")
        analyze_patch = patch.object(self.adapter, "analyze", wraps=self.adapter.analyze)
        self.analyze = analyze_patch.start()
        self.addCleanup(analyze_patch.stop)
        self.local_factory = Mock(return_value=self.adapter)
        self.dependencies = AnalysisDependencies(
            local_adapter_factory=self.local_factory,
            secondary_adapter_factory=Mock(return_value=None),
            remote_adapter_factory=Mock(return_value=None),
            review_policy_factory=Mock(return_value=ReviewPolicy()),
            probe_gpu_capacity=Mock(side_effect=AssertionError("unexpected GPU probe")),
        )
        self.clock = datetime.now(UTC) + timedelta(minutes=2)
        self.queue = ReportRefreshQueue(self.database)

    def _enqueue(self, event_id: str, *, occurred_at: datetime | None = None) -> None:
        clip = self.root / f"{event_id}.mp4"
        frame = self.root / f"{event_id}.jpg"
        clip.write_bytes(b"synthetic-clip-not-decoded")
        frame.write_bytes(b"synthetic-frame-not-decoded")
        AnalysisQueue(self.database).enqueue(
            candidate=CandidateEvent(event_id, "camera_01", 0.0, 1.0, ()),
            occurred_at=occurred_at or self.EVENT_TIME,
            source_kind="fixture",
            source_ref="synthetic-deferred-report",
            media=EventMedia(str(clip), (str(frame),)),
            storage_root=self.root,
            max_active=10,
        )

    def _analyze(self, policy: HouseholdEventPolicy | None = None):
        return execute_analysis_run_once(
            self.settings,
            self.database,
            policy,
            dependencies=self.dependencies,
            defer_reports=True,
        )

    def _job(self, event_id: str, *, occurred_at: datetime | None = None):
        return self.queue.request(
            request_id="event:" + hashlib.sha256(event_id.encode("utf-8")).hexdigest(),
            local_date=(occurred_at or self.EVENT_TIME)
            .astimezone(ZoneInfo("Asia/Shanghai"))
            .date(),
            timezone_name="Asia/Shanghai",
            storage_root=self.root,
        )

    def _paths(self, local_date: date | None = None) -> tuple[Path, Path]:
        report_date = local_date or self.EVENT_TIME.astimezone(ZoneInfo("Asia/Shanghai")).date()
        directory = self.root / "reports" / report_date.strftime("%Y/%m/%d")
        stem = f"{report_date.isoformat()}_daily_report"
        return directory / f"{stem}.md", directory / f"{stem}.html"

    def _assert_analysis_completed(self, *event_ids: str) -> None:
        for event_id in event_ids:
            with self.subTest(event_id=event_id):
                job = AnalysisQueue(self.database).get_read_only(event_id)
                assert job is not None
                self.assertEqual("completed", job.status)

    def test_same_day_events_finish_and_enqueue_notifications_before_rendering(self) -> None:
        self.settings = replace(
            self.settings,
            notification=replace(self.settings.notification, enabled=True, auto_file_enabled=True),
        )
        policy = HouseholdEventPolicy(
            package_id="deferred_fixture",
            timezone="Asia/Shanghai",
            enabled=True,
            rules=(
                PolicyEventRule(
                    rule_id="local_unknown",
                    event_types=frozenset({"unknown"}),
                    risk_levels=frozenset({"none"}),
                    review_statuses=frozenset({"final", "pending_review"}),
                    sink_ids=("shadow_local",),
                ),
            ),
        )
        for index, event_id in enumerate(("first", "second")):
            self._enqueue(event_id, occurred_at=self.EVENT_TIME + timedelta(minutes=index))
        with patch(
            "baby_monitor.analysis_service.DailyReportGenerator.generate",
            side_effect=AssertionError("analysis must not render a deferred report"),
        ) as render:
            for _ in range(2):
                code, payload, attempt = self._analyze(policy)
                self.assertEqual(0, code)
                self.assertEqual(WorkerAttemptState.COMPLETED, attempt.state)
                self.assertEqual("queued", payload["report"]["mode"])
                self.assertEqual("pending", payload["notification_enqueue"]["status"])
                self.assertNotIn("markdown", payload["report"])
            render.assert_not_called()
        self._assert_analysis_completed("first", "second")
        self.assertEqual(2, NotificationOutbox(self.database).snapshot_read_only().counts.pending)
        self.assertFalse(any(path.exists() for path in self._paths()))
        job = self._job("first")
        self.assertEqual(2, job.requested_revision)
        self.assertEqual(self._job("second").job_id, job.job_id)

        code, payload, attempt = execute_report_refresh_once(self.database, now=self.clock)

        self.assertEqual(0, code)
        self.assertEqual(WorkerAttemptState.COMPLETED, attempt.state)
        self.assertEqual("completed", payload["status"])
        self.assertEqual(2, payload["event_count"])
        self.assertEqual(2, self.analyze.call_count)
        self.assertEqual(2, NotificationOutbox(self.database).snapshot_read_only().counts.pending)
        self.assertTrue(all(path.is_file() for path in self._paths()))

    def test_failed_request_retries_handoff_without_repeating_recognition(self) -> None:
        self._enqueue("handoff_failure")
        with patch(
            "baby_monitor.analysis_service.ReportRefreshQueue.request",
            side_effect=OSError("synthetic-private-request-failure"),
        ):
            code, payload, attempt = self._analyze()
        self.assertEqual(2, code)
        self.assertEqual(WorkerAttemptState.RETRY_PENDING, attempt.state)
        self.assertNotIn("synthetic-private-request-failure", repr(payload))
        before = EventDatabase(self.database).get_event("handoff_failure")
        self.assertIsNotNone(before)

        code, payload, attempt = self._analyze()

        self.assertEqual(0, code)
        self.assertEqual(WorkerAttemptState.COMPLETED, attempt.state)
        self.assertTrue(payload["reused_existing"])
        self.assertEqual(1, self._job("handoff_failure").requested_revision)
        self.assertEqual(1, self.analyze.call_count)
        self.assertEqual(1, len(RecognitionAttemptAuditStore(self.database).recent()))
        self.assertEqual(before, EventDatabase(self.database).get_event("handoff_failure"))

    def test_committed_request_then_failure_is_idempotent_on_analysis_retry(self) -> None:
        self._enqueue("committed_handoff")
        original_request = ReportRefreshQueue.request

        def committed_then_failed(queue, **kwargs):
            original_request(queue, **kwargs)
            raise OSError("synthetic-handoff-response-failure")

        with patch.object(
            ReportRefreshQueue, "request", autospec=True, side_effect=committed_then_failed
        ):
            code, _payload, attempt = self._analyze()
        self.assertEqual(2, code)
        self.assertEqual(WorkerAttemptState.RETRY_PENDING, attempt.state)
        requested = self._job("committed_handoff")
        self.assertEqual(1, requested.requested_revision)

        code, payload, _attempt = self._analyze()

        self.assertEqual(0, code)
        self.assertTrue(payload["reused_existing"])
        self.assertEqual(1, self._job("committed_handoff").requested_revision)
        self.assertEqual(1, self.analyze.call_count)
        self._assert_analysis_completed("committed_handoff")

    def test_render_failure_retries_only_report_and_preserves_completed_analysis(self) -> None:
        self._enqueue("render_failure")
        self.assertEqual(0, self._analyze()[0])
        before = EventDatabase(self.database).get_event("render_failure")
        with patch.object(
            DailyReportGenerator, "_markdown", side_effect=OSError("synthetic-render-failure")
        ):
            code, payload, attempt = execute_report_refresh_once(
                self.database, now=self.clock, retry_delay_seconds=0
            )
        self.assertEqual(2, code)
        self.assertEqual(WorkerAttemptState.RETRY_PENDING, attempt.state)
        self.assertEqual("pending", payload["status"])
        self.assertEqual("OSError", payload["error_code"])
        self.assertNotIn("synthetic-render-failure", repr(payload))
        self.assertFalse(any(path.exists() for path in self._paths()))
        self._assert_analysis_completed("render_failure")

        code, payload, _attempt = execute_report_refresh_once(
            self.database, now=self.clock + timedelta(seconds=1)
        )

        self.assertEqual(0, code)
        self.assertEqual("completed", payload["status"])
        self.assertEqual(before, EventDatabase(self.database).get_event("render_failure"))
        self.assertEqual(1, self.analyze.call_count)
        self.assertEqual(1, self.local_factory.call_count)

    def test_dirty_during_render_stays_pending_and_next_refresh_includes_new_event(self) -> None:
        self._enqueue("before_refresh")
        self.assertEqual(0, self._analyze()[0])
        original_markdown = DailyReportGenerator._markdown

        def new_event_during_render(generator, *args, **kwargs):
            self._enqueue("during_refresh", occurred_at=self.EVENT_TIME + timedelta(minutes=1))
            self.assertEqual(0, self._analyze()[0])
            return original_markdown(generator, *args, **kwargs)

        with patch.object(DailyReportGenerator, "_markdown", new=new_event_during_render):
            code, payload, _attempt = execute_report_refresh_once(self.database, now=self.clock)
        self.assertEqual(0, code)
        self.assertEqual("pending", payload["status"])
        self.assertEqual(1, payload["rendered_revision"])
        self.assertEqual(1, payload["event_count"])
        self.assertNotIn("during_refresh", self._paths()[0].read_text(encoding="utf-8"))
        pending = self._job("before_refresh")
        self.assertEqual(2, pending.requested_revision)
        self.assertEqual(1, pending.completed_revision)

        code, payload, _attempt = execute_report_refresh_once(
            self.database, now=self.clock + timedelta(minutes=1)
        )

        self.assertEqual(0, code)
        self.assertEqual("completed", payload["status"])
        self.assertEqual(2, payload["rendered_revision"])
        self.assertEqual(2, payload["event_count"])
        self.assertIn("during_refresh", self._paths()[0].read_text(encoding="utf-8"))
        self._assert_analysis_completed("before_refresh", "during_refresh")
        self.assertEqual(2, self.analyze.call_count)

    def test_refresh_uses_bound_output_root_and_local_day_after_settings_change(self) -> None:
        occurred_at = datetime(2026, 8, 3, 16, 1, tzinfo=UTC)
        self._enqueue("local_midnight", occurred_at=occurred_at)
        self.assertEqual(0, self._analyze()[0])
        job = self._job("local_midnight", occurred_at=occurred_at)
        self.assertEqual(date(2026, 8, 4), job.local_date)
        self.assertEqual("Asia/Shanghai", job.timezone_name)
        self.assertEqual(self.root, Path(job.storage_root))
        different_root = self.root / "changed-root"
        different_root.mkdir(mode=0o700)
        self.settings = replace(
            self.settings,
            timezone="UTC",
            storage=replace(self.settings.storage, local_root=different_root),
        )

        code, payload, _attempt = execute_report_refresh_once(self.database, now=self.clock)

        self.assertEqual(0, code)
        self.assertEqual(1, payload["event_count"])
        self.assertTrue(all(path.is_file() for path in self._paths(date(2026, 8, 4))))
        self.assertFalse(any(path.exists() for path in self._paths(date(2026, 8, 3))))
        self.assertFalse((different_root / "reports").exists())
        self.assertNotIn(str(self.root), repr(payload))

    def test_reclaimed_old_job_guard_cannot_overwrite_newly_completed_report(self) -> None:
        self._enqueue("old_claim")
        self.assertEqual(0, self._analyze()[0])
        old = self.queue.claim_next(lease_seconds=10, max_attempts=3, now=self.clock)
        assert old is not None
        self._enqueue("new_claim", occurred_at=self.EVENT_TIME + timedelta(minutes=1))
        self.assertEqual(0, self._analyze()[0])
        code, payload, _attempt = execute_report_refresh_once(
            self.database, now=self.clock + timedelta(seconds=11)
        )
        self.assertEqual(0, code)
        self.assertEqual(2, payload["event_count"])
        before = tuple(path.read_bytes() for path in self._paths())
        generator = DailyReportGenerator(
            database=EventDatabase(self.database),
            storage=StorageResolver(self.settings.storage),
            timezone_name=old.timezone_name,
        )

        with self.assertRaises(ReportClaimLostError):
            generator.generate(
                old.local_date, claim_guard=lambda: self.queue.assert_claim_current(old)
            )

        self.assertEqual(before, tuple(path.read_bytes() for path in self._paths()))
        completed = self.queue.get_read_only(old.job_id)
        assert completed is not None
        self.assertEqual("completed", completed.status)
        self.assertEqual(2, completed.completed_revision)
        self._assert_analysis_completed("old_claim", "new_claim")

    def test_html_publish_failure_does_not_complete_job_or_repeat_analysis(self) -> None:
        self._enqueue("html_failure")
        self.assertEqual(0, self._analyze()[0])
        before = EventDatabase(self.database).get_event("html_failure")

        def fail_html(path, content):
            if path.suffix == ".html":
                raise OSError("synthetic-html-publish-failure")
            _atomic_write_private_report(path, content)

        with patch("baby_monitor.reporting._atomic_write_private_report", side_effect=fail_html):
            code, payload, attempt = execute_report_refresh_once(
                self.database, now=self.clock, retry_delay_seconds=0
            )
        self.assertEqual(2, code)
        self.assertEqual("pending", payload["status"])
        self.assertEqual(WorkerAttemptState.RETRY_PENDING, attempt.state)
        self.assertTrue(self._paths()[0].is_file())
        self.assertFalse(self._paths()[1].exists())
        self.assertEqual(0, self._job("html_failure").completed_revision)
        self._assert_analysis_completed("html_failure")

        code, payload, _attempt = execute_report_refresh_once(
            self.database, now=self.clock + timedelta(seconds=1)
        )

        self.assertEqual(0, code)
        self.assertEqual("completed", payload["status"])
        self.assertTrue(all(path.is_file() for path in self._paths()))
        self.assertEqual(before, EventDatabase(self.database).get_event("html_failure"))
        self.assertEqual(1, self.analyze.call_count)


if __name__ == "__main__":
    unittest.main()
