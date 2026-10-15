from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from baby_monitor.adapters.vlm import MockVLMAdapter
from baby_monitor.analysis_queue import AnalysisQueue
from baby_monitor.analysis_service import AnalysisDependencies, execute_analysis_run_once
from baby_monitor.analysis_worker import WorkerAttemptState
from baby_monitor.config import Settings, StorageSettings
from baby_monitor.database import EventDatabase
from baby_monitor.domain import CandidateEvent, EventMedia
from baby_monitor.event_policy import EventPolicyError, HouseholdEventPolicy, PolicyEventRule
from baby_monitor.gpu_guard import GPUCapacity
from baby_monitor.notification import NotificationOutbox
from baby_monitor.policy import ReviewPolicy
from baby_monitor.recognition_attempt import RecognitionAttemptAuditStore


class AnalysisServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="baby-monitor-analysis-service-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.database = self.root / "events.sqlite3"
        self.settings = Settings(
            storage=StorageSettings(local_root=self.root, database=self.database)
        )
        self.adapter = MockVLMAdapter(name="service_fixture")
        self.local_factory = Mock(return_value=self.adapter)
        self.secondary_factory = Mock(return_value=None)
        self.remote_factory = Mock(return_value=None)
        self.policy_factory = Mock(return_value=ReviewPolicy())
        self.gpu_probe = Mock(
            return_value=GPUCapacity(False, "gpu_busy", 0, 12000, 10000, 2000, 95)
        )
        self.dependencies = AnalysisDependencies(
            local_adapter_factory=self.local_factory,
            secondary_adapter_factory=self.secondary_factory,
            remote_adapter_factory=self.remote_factory,
            review_policy_factory=self.policy_factory,
            probe_gpu_capacity=self.gpu_probe,
        )

    def _enqueue(self) -> None:
        clip = self.root / "event.mp4"
        frame = self.root / "frame.jpg"
        clip.write_bytes(b"synthetic-clip")
        frame.write_bytes(b"synthetic-frame")
        AnalysisQueue(self.database).enqueue(
            candidate=CandidateEvent(
                event_id="service_event",
                camera_id="camera_01",
                start_seconds=0.0,
                end_seconds=1.0,
                detections=(),
            ),
            occurred_at=datetime(2026, 8, 3, 1, tzinfo=UTC),
            source_kind="fixture",
            source_ref="synthetic-analysis-service",
            media=EventMedia(clip_path=str(clip), frame_paths=(str(frame),)),
            storage_root=self.root,
            max_active=10,
        )

    @staticmethod
    def _event_policy() -> HouseholdEventPolicy:
        return HouseholdEventPolicy(
            package_id="service_fixture",
            timezone="Asia/Shanghai",
            rules=(
                PolicyEventRule(
                    rule_id="local_unknown",
                    event_types=frozenset({"unknown"}),
                    risk_levels=frozenset({"none"}),
                    review_statuses=frozenset({"final", "pending_review"}),
                    sink_ids=("shadow_local",),
                ),
            ),
            enabled=True,
        )

    def _run(self, event_policy: HouseholdEventPolicy | None = None):
        return execute_analysis_run_once(
            self.settings,
            self.database,
            event_policy,
            dependencies=self.dependencies,
        )

    def _assert_no_model_factories(self) -> None:
        self.local_factory.assert_not_called()
        self.secondary_factory.assert_not_called()
        self.remote_factory.assert_not_called()
        self.policy_factory.assert_not_called()

    def _assert_policy_preflight_rejects(
        self,
        event_policy: HouseholdEventPolicy | None,
        message: str,
    ) -> None:
        self._enqueue()
        self.settings = replace(
            self.settings,
            notification=replace(self.settings.notification, auto_file_enabled=True),
            analysis_worker=replace(self.settings.analysis_worker, gpu_guard_enabled=True),
        )
        before = set(self.root.rglob("*"))

        with self.assertRaisesRegex(EventPolicyError, message):
            self._run(event_policy)

        queued = AnalysisQueue(self.database).get_read_only("service_event")
        assert queued is not None
        self.assertEqual("pending", queued.status.value)
        self.assertEqual(0, queued.attempt_count)
        self.assertEqual(before, set(self.root.rglob("*")))
        self._assert_no_model_factories()
        self.gpu_probe.assert_not_called()

    def test_missing_local_notification_policy_is_rejected_before_claim(self) -> None:
        self._assert_policy_preflight_rejects(None, "require an event policy")

    def test_mismatched_local_notification_sink_is_rejected_before_claim(self) -> None:
        policy = self._event_policy()
        mismatched = replace(
            policy,
            rules=(replace(policy.rules[0], sink_ids=("another_local_sink",)),),
        )
        self._assert_policy_preflight_rejects(mismatched, "must match the configured")

    def test_embedded_host_completes_report_and_local_route_with_one_adapter_instance(self) -> None:
        self._enqueue()
        self.settings = replace(
            self.settings,
            notification=replace(self.settings.notification, auto_file_enabled=True),
        )
        output = StringIO()
        with redirect_stdout(output):
            code, payload, attempt = self._run(self._event_policy())

        self.assertEqual("", output.getvalue())
        self.assertEqual(0, code)
        self.assertEqual(WorkerAttemptState.COMPLETED, attempt.state)
        self.assertEqual("service_fixture", payload["local_vlm"])
        self.assertEqual("completed", payload["status"])
        self.assertEqual("route", payload["event_policy_decision"]["action"])
        self.assertEqual("pending", payload["notification_enqueue"]["status"])
        self.assertTrue(Path(payload["report"]["markdown"]).is_file())
        self.assertTrue(Path(payload["report"]["html"]).is_file())
        self.assertEqual(1, NotificationOutbox(self.database).snapshot_read_only().counts.pending)
        self.assertEqual(1, len(EventDatabase(self.database).list_recent()))
        self.local_factory.assert_called_once_with()
        self.secondary_factory.assert_called_once_with()
        self.remote_factory.assert_called_once_with(self.database)
        self.policy_factory.assert_called_once_with()
        self.gpu_probe.assert_not_called()

    def test_gpu_busy_defers_before_claim_or_adapter_construction(self) -> None:
        self._enqueue()
        self.settings = replace(
            self.settings,
            analysis_worker=replace(self.settings.analysis_worker, gpu_guard_enabled=True),
        )

        code, payload, attempt = self._run()

        self.assertEqual(0, code)
        self.assertEqual(WorkerAttemptState.DEFERRED, attempt.state)
        self.assertEqual("gpu_busy", payload["reason"])
        queued = AnalysisQueue(self.database).get_read_only("service_event")
        assert queued is not None
        self.assertEqual("pending", queued.status.value)
        self.assertEqual(0, queued.attempt_count)
        self.gpu_probe.assert_called_once_with()
        self._assert_no_model_factories()

    def test_idle_does_not_construct_adapters(self) -> None:
        code, payload, attempt = self._run()

        self.assertEqual(0, code)
        self.assertEqual("idle", payload["status"])
        self.assertEqual(WorkerAttemptState.IDLE, attempt.state)
        self._assert_no_model_factories()
        self.gpu_probe.assert_not_called()

    def test_report_failure_retries_outputs_without_repeating_recognition(self) -> None:
        self._enqueue()
        self.settings = replace(
            self.settings,
            analysis_queue=replace(self.settings.analysis_queue, retry_delay_seconds=0),
            notification=replace(self.settings.notification, auto_file_enabled=True),
        )
        event_policy = self._event_policy()
        with (
            patch(
                "baby_monitor.analysis_service.DailyReportGenerator.generate",
                side_effect=OSError("synthetic-output-failure"),
            ),
            patch(
                "baby_monitor.analysis_service.EventPolicyDecisionStore.decide_and_record"
            ) as route,
        ):
            failed_code, failed_payload, failed_attempt = self._run(event_policy)
        self.assertEqual(2, failed_code)
        self.assertEqual("OSError", failed_payload["error_code"])
        self.assertEqual(WorkerAttemptState.RETRY_PENDING, failed_attempt.state)
        self.assertTrue(failed_payload["evidence_preserved"])
        self.assertEqual(1, len(EventDatabase(self.database).list_recent()))
        route.assert_not_called()

        code, payload, attempt = self._run(event_policy)

        self.assertEqual(0, code)
        self.assertEqual(WorkerAttemptState.COMPLETED, attempt.state)
        self.assertTrue(payload["reused_existing"])
        self.assertEqual(2, payload["attempt_count"])
        self.assertEqual(1, len(RecognitionAttemptAuditStore(self.database).recent()))
        self.assertEqual(1, NotificationOutbox(self.database).snapshot_read_only().counts.pending)
        self.assertEqual(2, self.local_factory.call_count)
        self.assertTrue((self.root / "event.mp4").is_file())
        self.assertTrue((self.root / "frame.jpg").is_file())

    def test_factory_failure_uses_existing_bounded_failure_path(self) -> None:
        self._enqueue()
        self.settings = replace(
            self.settings,
            analysis_queue=replace(self.settings.analysis_queue, max_attempts=1),
        )
        self.local_factory.side_effect = RuntimeError("private factory diagnostics")

        code, payload, attempt = self._run()

        self.assertEqual(2, code)
        self.assertEqual(WorkerAttemptState.FAILED, attempt.state)
        self.assertEqual("RuntimeError", payload["error_code"])
        self.assertNotIn("private factory diagnostics", repr(payload))
        self.assertEqual([], EventDatabase(self.database).list_recent())
        self.secondary_factory.assert_not_called()
        self.remote_factory.assert_not_called()
        self.policy_factory.assert_not_called()
        self.assertTrue((self.root / "event.mp4").is_file())


if __name__ == "__main__":
    unittest.main()
