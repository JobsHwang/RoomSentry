from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from baby_monitor.adapters.vlm import VLMError
from baby_monitor.cli import main
from baby_monitor.config import Settings, StorageSettings
from baby_monitor.local_review import LocalReviewError, LocalReviewStore
from baby_monitor.local_review_reports import dispatch_local_review_reports
from baby_monitor.local_review_service import retry_local_review
from baby_monitor.policy import ReviewPolicy
from baby_monitor.recognition_attempt import RecognitionAttemptAuditStore
from baby_monitor.report_queue import ReportQueueError, ReportRefreshQueue
from baby_monitor.report_service import execute_report_refresh_once
from baby_monitor.reporting import DailyReportGenerator
from baby_monitor.storage import StorageResolver
from tests import test_local_review_service as review_fixtures


class LocalReviewReportIntegrationTests(unittest.TestCase):
    REQUEST_ID = "report_retry_01"

    def setUp(self):
        self.fixture = review_fixtures.LocalReviewServiceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.database = self.fixture.database
        self.event_id = self.fixture.event_id
        self.report_root = self.root / "report_target"
        self.report_root.mkdir(mode=0o700)
        self.adapter = review_fixtures._LocalStaticVLM()
        self.factory = Mock(return_value=self.adapter)
        self.store = LocalReviewStore(self.database.path)
        self.queue = ReportRefreshQueue(self.database.path)
        self.clock = datetime.now(UTC) + timedelta(minutes=5)

    def _seed_old_report(self, *, occurred_at=None):
        if occurred_at is not None:
            self.fixture.arguments["occurred_at"] = occurred_at
        self.fixture._seed()
        self.day = (
            self.fixture.arguments["occurred_at"].astimezone(ZoneInfo("Asia/Shanghai")).date()
        )
        self.generator = DailyReportGenerator(
            database=self.database,
            storage=StorageResolver(
                StorageSettings(local_root=self.report_root, database=self.database.path)
            ),
            timezone_name="Asia/Shanghai",
        )
        self.report = self.generator.generate(self.day)
        self.old_markdown = self.report.markdown_path.read_bytes()
        self.old_html = self.report.html_path.read_bytes()
        self.assertIn("本地待复核：是", self.old_markdown.decode())

    def _retry(self, *, target=True, factory=None, **overrides):
        kwargs = dict(
            database_path=self.database.path,
            event_id=self.event_id,
            request_id=self.REQUEST_ID,
            secondary_adapter_factory=factory or self.factory,
            policy=ReviewPolicy(),
            lease_seconds=600,
        )
        if target:
            kwargs.update(report_storage_root=self.report_root, report_timezone="Asia/Shanghai")
        kwargs.update(overrides)
        return retry_local_review(**kwargs)

    def _run_worker(self, *, now=None):
        with (
            patch.object(
                review_fixtures._LocalStaticVLM,
                "analyze",
                side_effect=AssertionError("report worker must not run models"),
            ),
            patch(
                "baby_monitor.notification.NotificationOutbox.enqueue",
                side_effect=AssertionError("report worker must not notify"),
            ),
        ):
            return execute_report_refresh_once(
                self.database.path, now=now or self.clock, retry_delay_seconds=0
            )

    def _assert_original_report(self):
        self.assertEqual(self.old_markdown, self.report.markdown_path.read_bytes())
        self.assertEqual(self.old_html, self.report.html_path.read_bytes())

    def _assert_refreshed_report(self):
        for path in (self.report.markdown_path, self.report.html_path):
            content = path.read_text(encoding="utf-8")
            self.assertIn("本地待复核：否", content)
            self.assertIn("local_secondary", content)
            self.assertIn("未验证连续覆盖", content)
            self.assertNotIn("证据覆盖：完整", content)
        self.assertEqual(1, self.adapter.calls)
        self.assertEqual(1, self.fixture.arguments["local_vlm"].calls)
        self.assertEqual(0, self.fixture.arguments["remote_vlm"].calls)
        self.assertEqual(2, len(RecognitionAttemptAuditStore(self.database.path).recent()))

    def test_success_stores_pending_intent_then_worker_refreshes_without_new_models(self):
        self._seed_old_report()
        with (
            patch.object(
                ReportRefreshQueue,
                "request",
                side_effect=AssertionError("service must not enqueue"),
            ),
            patch.object(
                DailyReportGenerator,
                "generate",
                side_effect=AssertionError("service must not render"),
            ),
        ):
            receipt = self._retry()
        self.assertEqual("succeeded", receipt["status"])
        self.assertEqual("pending", receipt["report_refresh"]["status"])
        self.assertFalse(receipt["reports_refreshed"])
        self.assertFalse(self.database.get_event(self.event_id).decision.needs_local_review)
        self.assertEqual("pending", self.store.report_refresh_state(self.REQUEST_ID)["status"])
        self._assert_original_report()
        self.assertEqual(0, self.queue.snapshot_read_only()["counts"]["total"])

        code, payload, _ = self._run_worker()

        self.assertEqual(0, code)
        self.assertEqual("completed", payload["status"])
        self.assertEqual("queued", self.store.report_refresh_state(self.REQUEST_ID)["status"])
        self.assertNotIn(str(self.root), json.dumps(payload))
        self._assert_refreshed_report()

    def test_failed_dispatch_remains_pending_and_later_worker_recovers_without_model_retry(self):
        self._seed_old_report()
        self._retry()
        with patch.object(
            ReportRefreshQueue, "request", side_effect=ReportQueueError("PRIVATE_ENQUEUE_FAILURE")
        ):
            dispatch = dispatch_local_review_reports(self.database.path, request_id=self.REQUEST_ID)
        self.assertEqual(1, dispatch["pending"])
        self.assertEqual(1, dispatch["errors_count"])
        self.assertNotIn("PRIVATE_ENQUEUE_FAILURE", json.dumps(dispatch))
        self.assertEqual("pending", self.store.report_refresh_state(self.REQUEST_ID)["status"])
        self._assert_original_report()
        replay = self._retry(factory=Mock(side_effect=AssertionError("duplicate model")))
        self.assertTrue(replay["reused_existing"])
        self.assertEqual("succeeded", replay["status"])
        self.assertEqual("pending", replay["report_refresh"]["status"])
        self.assertEqual(0, self._run_worker()[0])
        self._assert_refreshed_report()

    def test_report_render_failure_retries_only_report_and_keeps_review_success(self):
        self._seed_old_report()
        self._retry()
        reviewed = self.database.get_event(self.event_id)
        with patch.object(
            DailyReportGenerator, "_markdown", side_effect=OSError("PRIVATE_RENDER_FAILURE")
        ):
            code, payload, _ = self._run_worker()
        self.assertEqual(2, code)
        self.assertEqual("pending", payload["status"])
        self.assertNotIn("PRIVATE_RENDER_FAILURE", json.dumps(payload))
        self.assertEqual(reviewed, self.database.get_event(self.event_id))
        self._assert_original_report()
        self.assertEqual(0, self._run_worker(now=self.clock + timedelta(seconds=1))[0])
        self._assert_refreshed_report()

    def test_pending_intent_with_no_queue_job_is_deferred_not_idle(self):
        self._seed_old_report()
        self._retry()
        with patch.object(
            ReportRefreshQueue, "request", side_effect=ReportQueueError("PRIVATE_ENQUEUE_FAILURE")
        ):
            code, payload, attempt = self._run_worker()
        self.assertEqual(0, code)
        self.assertEqual("deferred", payload["status"])
        self.assertEqual("deferred", attempt.state)
        self.assertEqual("local_review_report_pending", payload["reason"])
        self.assertEqual(1, payload["report_intents"]["pending"])
        self.assertFalse(payload["processed"])
        self.assertNotIn("PRIVATE_ENQUEUE_FAILURE", json.dumps(payload))
        self.assertEqual(0, self.queue.snapshot_read_only()["counts"]["total"])
        self._assert_original_report()
        self.assertEqual(0, self._run_worker()[0])
        self._assert_refreshed_report()

    def test_invalid_worker_parameters_do_not_dispatch_or_write_intent(self):
        self._seed_old_report()
        self._retry()
        before = self.database.path.read_bytes()
        for changes in (
            {"lease_seconds": 0},
            {"lease_seconds": True},
            {"max_attempts": 0},
            {"max_attempts": 101},
            {"retry_delay_seconds": -1},
            {"retry_delay_seconds": True},
        ):
            with (
                self.subTest(changes=changes),
                patch(
                    "baby_monitor.report_service.dispatch_local_review_reports",
                    side_effect=AssertionError("invalid parameters reached dispatch"),
                ),
                self.assertRaises(ValueError),
            ):
                execute_report_refresh_once(self.database.path, **changes)
        self.assertEqual(before, self.database.path.read_bytes())
        self.assertEqual("pending", self.store.report_refresh_state(self.REQUEST_ID)["status"])

    def test_unavailable_refresh_status_does_not_relabel_successful_review(self):
        self._seed_old_report()
        with patch.object(
            LocalReviewStore,
            "report_refresh_state",
            side_effect=LocalReviewError("database_unavailable"),
        ):
            receipt = self._retry()
        self.assertEqual("succeeded", receipt["status"])
        self.assertEqual("unknown", receipt["report_refresh"]["status"])
        self.assertFalse(self.database.get_event(self.event_id).decision.needs_local_review)
        self.assertEqual("pending", self.store.report_refresh_state(self.REQUEST_ID)["status"])
        self._assert_original_report()
        self.assertEqual(0, self._run_worker()[0])
        self._assert_refreshed_report()

    def test_running_old_revision_remains_dirty_until_followup_worker_catches_up(self):
        self._seed_old_report()
        current = datetime.now(UTC)
        first = self.queue.request(
            request_id="old_primary_report",
            local_date=self.day,
            timezone_name="Asia/Shanghai",
            storage_root=self.report_root,
            debounce_seconds=0,
            now=current,
        )
        original_markdown = DailyReportGenerator._markdown

        def review_while_old_snapshot_is_rendering(generator, *args, **kwargs):
            self.assertEqual("running", self.queue.get_read_only(first.job_id).status)
            # The generator already loaded primary-only event objects. Commit
            # the review now, then let this old snapshot finish publishing.
            self._retry()
            dispatch_local_review_reports(self.database.path, request_id=self.REQUEST_ID)
            dirty = self.queue.get_read_only(first.job_id)
            self.assertEqual(2, dirty.requested_revision)
            self.assertEqual(1, dirty.claimed_revision)
            return original_markdown(generator, *args, **kwargs)

        with patch.object(
            DailyReportGenerator,
            "_markdown",
            autospec=True,
            side_effect=review_while_old_snapshot_is_rendering,
        ):
            code, old_payload, _ = execute_report_refresh_once(self.database.path, now=current)
        self.assertEqual(0, code)
        self.assertEqual(1, old_payload["rendered_revision"])
        completed = self.queue.get_read_only(first.job_id)
        self.assertEqual("pending", completed.status)
        self.assertEqual(1, completed.completed_revision)
        self._assert_original_report()

        code, payload, _ = self._run_worker()

        self.assertEqual(0, code)
        self.assertEqual(2, payload["rendered_revision"])
        self.assertEqual(2, self.queue.get_read_only(first.job_id).completed_revision)
        self._assert_refreshed_report()

    def test_legacy_no_target_does_not_request_or_refresh_old_report(self):
        self._seed_old_report()
        receipt = self._retry(target=False)
        self.assertEqual("succeeded", receipt["status"])
        self.assertEqual("not_requested", receipt["report_refresh"]["status"])
        self.assertEqual("idle", self._run_worker()[1]["status"])
        self._assert_original_report()
        self.assertEqual(1, self.adapter.calls)

    def test_failed_review_does_not_enqueue_or_refresh_report(self):
        self._seed_old_report()
        failed_adapter = review_fixtures._LocalStaticVLM(error=VLMError("synthetic model failure"))
        receipt = self._retry(factory=Mock(return_value=failed_adapter))
        self.assertEqual("failed", receipt["status"])
        self.assertNotEqual("pending", receipt["report_refresh"]["status"])
        self.assertEqual(0, dispatch_local_review_reports(self.database.path)["enqueued"])
        self.assertEqual("idle", self._run_worker()[1]["status"])
        self._assert_original_report()
        self.assertTrue(self.database.get_event(self.event_id).decision.needs_local_review)

    def test_shanghai_date_is_bound_to_event_and_replay_cannot_redirect(self):
        self._seed_old_report(occurred_at=datetime(2026, 9, 12, 23, 59, tzinfo=UTC))
        self.assertEqual("2026-09-13", self.day.isoformat())
        self._retry()
        other_root = self.root / "other_target"
        other_root.mkdir(mode=0o700)
        for changes in ({"report_storage_root": other_root}, {"report_timezone": "UTC"}):
            with self.subTest(changes=changes), self.assertRaises(LocalReviewError):
                self._retry(factory=Mock(side_effect=AssertionError("redirect model")), **changes)
        dispatch_local_review_reports(self.database.path, request_id=self.REQUEST_ID)
        state = self.store.report_refresh_state(self.REQUEST_ID)
        job = self.queue.get_read_only(state["job_id"])
        self.assertEqual(self.day, job.local_date)
        self.assertEqual("Asia/Shanghai", job.timezone_name)
        self.assertEqual(self.report_root, job.storage_root)
        self.assertEqual(0, self._run_worker()[0])
        self._assert_refreshed_report()
        self.assertEqual([], list(other_root.iterdir()))

    def test_cli_default_target_dispatch_failure_is_pending_then_same_request_re_dispatches(self):
        self._seed_old_report()
        settings = Settings(
            storage=StorageSettings(local_root=self.report_root, database=self.database.path)
        )
        settings = replace(
            settings,
            secondary_vlm=replace(settings.secondary_vlm, enabled=True),
            analysis_worker=replace(settings.analysis_worker, gpu_guard_enabled=False),
        )

        def cli_call():
            output, errors = io.StringIO(), io.StringIO()
            with (
                patch(
                    "baby_monitor.cli_commands.local_review.load_settings", return_value=settings
                ),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                code = main(
                    [
                        "local-review-retry",
                        "--database",
                        str(self.database.path),
                        "--event-id",
                        self.event_id,
                        "--request-id",
                        self.REQUEST_ID,
                    ]
                )
            self.assertEqual(0, code, errors.getvalue())
            self.assertNotIn(str(self.root), output.getvalue() + errors.getvalue())
            return json.loads(output.getvalue())

        with (
            patch(
                "baby_monitor.cli_commands.local_review._secondary_adapter",
                return_value=self.adapter,
            ),
            patch.object(
                ReportRefreshQueue,
                "request",
                side_effect=ReportQueueError("PRIVATE_ENQUEUE_FAILURE"),
            ),
        ):
            first = cli_call()
        self.assertEqual("succeeded", first["status"])
        self.assertEqual("pending", first["report_refresh"]["status"])
        self._assert_original_report()
        with patch(
            "baby_monitor.cli_commands.local_review._secondary_adapter",
            side_effect=AssertionError("repeat model"),
        ):
            repeated = cli_call()
        self.assertTrue(repeated["reused_existing"])
        self.assertEqual("queued", repeated["report_refresh"]["status"])
        self._assert_original_report()
        self.assertEqual(0, self._run_worker()[0])
        self._assert_refreshed_report()


if __name__ == "__main__":
    unittest.main()
