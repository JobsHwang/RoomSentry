from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from baby_monitor.adapters.vlm import VLMAdapter, VLMAnalysis, VLMError
from baby_monitor.config import StorageSettings
from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    EventMedia,
    EventType,
    FinalSource,
    RecognitionRunMetrics,
    ReviewTier,
    RiskLevel,
    VLMEvent,
)
from baby_monitor.home_daily_report import generate_home_daily_report
from baby_monitor.home_event_digest import read_home_event_digest
from baby_monitor.notification import NotificationMessage, ReviewStatus
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy
from baby_monitor.recognition_attempt import (
    RecognitionAttemptAuditStore,
    RecognitionAttemptStatus,
    RecognitionStage,
)
from baby_monitor.reporting import DailyReportGenerator
from baby_monitor.storage import StorageResolver

_FAILURE_TEXT = "synthetic secondary diagnostic must not become event facts"


class _StaticVLM(VLMAdapter):
    def __init__(self, name, *, confidence=0.9):
        self.name = name
        self.confidence = confidence
        self.calls = 0
        self.metrics = RecognitionRunMetrics(
            duration_ms=12,
            json_compliant=True,
            process_rss_peak_kib=123,
            gpu_device_memory_peak_mib=45,
        )

    def analyze(self, request):
        self.calls += 1
        return VLMEvent(
            event_type=EventType.UNKNOWN,
            subjects=(),
            observations=("画面中可见浅色矩形。",),
            inferences=(),
            risk_level=RiskLevel.NONE,
            confidence=self.confidence,
            needs_remote_review=False,
            summary_zh="画面中可见浅色矩形。",
            evidence_frames=request.media.frame_paths,
        )

    def analyze_with_metrics(self, request):
        return VLMAnalysis(result=self.analyze(request), metrics=self.metrics)


class _FailingVLM(VLMAdapter):
    name = "synthetic_secondary_failure"

    def __init__(self, error=None):
        self.calls = 0
        self.metrics = RecognitionRunMetrics(
            duration_ms=25,
            json_compliant=False,
            process_rss_peak_kib=456,
            gpu_device_memory_peak_mib=78,
        )
        self.error = (
            error
            if error is not None
            else VLMError(_FAILURE_TEXT, code="invalid_structured_output", metrics=self.metrics)
        )

    def analyze(self, request):
        self.calls += 1
        raise self.error


class SecondaryReviewFailureTests(unittest.TestCase):
    NOW = datetime(2026, 9, 12, 8, 0, tzinfo=UTC)

    def _arguments(self, root, *, remote_allowed=False, primary_confidence=0.9):
        root.chmod(0o700)
        frame, clip = root / "synthetic.jpg", root / "synthetic.mp4"
        frame.write_bytes(b"synthetic frame placeholder")
        clip.write_bytes(b"synthetic clip placeholder")
        database = EventDatabase(root / "events.sqlite3")
        return {
            "candidate": CandidateEvent("secondary_failure_fixture", "cam_01", 0.0, 1.0, ()),
            "occurred_at": self.NOW,
            "source_kind": "fixture",
            "source_ref": "synthetic_secondary_review_source",
            "media": EventMedia(str(clip), (str(frame),)),
            "local_vlm": _StaticVLM("synthetic_primary", confidence=primary_confidence),
            "secondary_vlm": _FailingVLM(),
            "remote_vlm": _StaticVLM("synthetic_remote"),
            "remote_uploads_allowed": remote_allowed,
            "database": database,
            "policy": ReviewPolicy(),
        }

    def _stored(self, arguments):
        # Check persisted decoding, not only the in-memory processing summary.
        stored = EventDatabase(arguments["database"].path).get_event(
            arguments["candidate"].event_id
        )
        self.assertIsNotNone(stored)
        return stored

    def _assert_secondary_absent(self, stored):
        self.assertIsNone(stored.secondary_result)
        self.assertIsNone(stored.secondary_provenance)
        self.assertIsNone(stored.secondary_metrics)
        self.assertIn("evidence_coverage_unverified", stored.decision.review_reasons)
        self.assertIn("local_secondary_review_failed", stored.decision.review_reasons)
        self.assertNotIn(_FAILURE_TEXT, stored.result.to_json())
        self.assertNotIn(_FAILURE_TEXT, stored.local_result.to_json())

    def test_secondary_failure_preserves_primary_without_creating_remote_escalation(self):
        for allowed in (False, True):
            with self.subTest(remote_allowed=allowed), tempfile.TemporaryDirectory() as directory:
                args = self._arguments(Path(directory), remote_allowed=allowed)
                summary = process_prepared_event(**args)
                stored = self._stored(args)
                self._assert_secondary_absent(stored)
                self.assertEqual(0.9, stored.local_result.confidence)
                self.assertEqual(stored.local_result, stored.result)
                self.assertEqual(args["local_vlm"].metrics, stored.local_metrics)
                self.assertEqual(args["local_vlm"].provenance, stored.local_provenance)
                self.assertEqual(ReviewTier.LOCAL_SECONDARY, stored.decision.review_tier)
                self.assertEqual(FinalSource.LOCAL_PRIMARY, stored.decision.final_source)
                self.assertTrue(stored.decision.needs_local_review)
                self.assertFalse(stored.decision.needs_remote_review)
                self.assertFalse(summary.secondary_reviewed)
                self.assertFalse(summary.remote_reviewed)
                self.assertFalse(summary.pending_remote_review)
                self.assertEqual(0, args["remote_vlm"].calls)
                self.assertFalse(stored.remote_used)
                self.assertIsNone(stored.remote_result)
                self.assertIsNone(stored.remote_metrics)
                self.assertIsNone(stored.remote_provenance)

    def test_repeated_event_reuses_primary_and_failed_secondary_attempt_without_rerunning(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory), remote_allowed=True)
            process_prepared_event(**args)
            audit = RecognitionAttemptAuditStore(args["database"].path)
            original = audit.recent()
            repeated = process_prepared_event(**args)
            self.assertTrue(repeated.reused_existing)
            self.assertFalse(repeated.secondary_reviewed)
            self.assertEqual(1, args["local_vlm"].calls)
            self.assertEqual(1, args["secondary_vlm"].calls)
            self.assertEqual(0, args["remote_vlm"].calls)
            self.assertEqual(original, audit.recent())
            self.assertEqual(2, len(original))
            by_stage = {attempt.stage: attempt for attempt in original}
            primary = by_stage[RecognitionStage.LOCAL_PRIMARY]
            secondary = by_stage[RecognitionStage.LOCAL_SECONDARY]
            self.assertEqual(RecognitionAttemptStatus.SUCCEEDED, primary.status)
            self.assertEqual(args["local_vlm"].metrics, primary.metrics)
            self.assertEqual(RecognitionAttemptStatus.FAILED, secondary.status)
            self.assertEqual("invalid_structured_output", secondary.error_code)
            self.assertEqual(args["secondary_vlm"].metrics, secondary.metrics)
            self.assertNotIn(
                _FAILURE_TEXT, json.dumps([item.to_public_dict() for item in original])
            )

    def test_reports_notification_and_home_digest_remain_pending_after_secondary_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._arguments(root)
            process_prepared_event(**args)
            stored = self._stored(args)
            database = args["database"]
            database.path.chmod(0o600)
            message = NotificationMessage.from_event(stored, sink_kind="file", sink_id="fixture")
            self.assertEqual(ReviewStatus.PENDING_REVIEW, message.review_status)
            self.assertEqual(ReviewTier.LOCAL_SECONDARY, message.review_tier)
            self.assertIn("local_secondary_review_failed", message.review_reasons)
            report = DailyReportGenerator(
                database=database,
                storage=StorageResolver(
                    StorageSettings(local_root=root / "storage", database=database.path)
                ),
                timezone_name="UTC",
            ).generate(self.NOW.date())
            for path in (report.markdown_path, report.html_path):
                content = path.read_text(encoding="utf-8")
                self.assertIn("待复核", content)
                self.assertIn("local_secondary_review_failed", content)
                self.assertNotIn(_FAILURE_TEXT, content)
            digest = read_home_event_digest(database.path, timezone_name="UTC", now=self.NOW)
            self.assertEqual(1, digest.today_total)
            self.assertEqual(1, len(digest.recent))
            self.assertTrue(digest.recent[0].needs_review)
            output = root / "daily-reports"
            receipt = generate_home_daily_report(
                database_path=database.path, output_root=output, timezone_name="UTC", now=self.NOW
            )
            self.assertEqual(1, receipt.event_count)
            for extension in ("md", "html"):
                content = (output / f"2026-09-12_daily_report.{extension}").read_text(
                    encoding="utf-8"
                )
                self.assertIn("待复核", content)
                self.assertNotIn(_FAILURE_TEXT, content)
            self.assertNotIn(_FAILURE_TEXT, message.to_json())

    def test_secondary_failure_keeps_primary_remote_reasons_subject_to_existing_consent(self):
        for allowed in (False, True):
            with self.subTest(remote_allowed=allowed), tempfile.TemporaryDirectory() as directory:
                args = self._arguments(
                    Path(directory), remote_allowed=allowed, primary_confidence=0.4
                )
                summary = process_prepared_event(**args)
                stored = self._stored(args)
                self._assert_secondary_absent(stored)
                self.assertIn("low_confidence", stored.decision.review_reasons)
                self.assertEqual(ReviewTier.REMOTE, stored.decision.review_tier)
                self.assertEqual(int(allowed), args["remote_vlm"].calls)
                self.assertEqual(allowed, stored.remote_used)
                self.assertEqual(allowed, summary.remote_reviewed)
                self.assertEqual(not allowed, summary.pending_remote_review)
                if allowed:
                    self.assertEqual(FinalSource.REMOTE, stored.decision.final_source)
                    self.assertEqual(0.9, stored.remote_result.confidence)
                else:
                    self.assertEqual(FinalSource.LOCAL_PRIMARY, stored.decision.final_source)
                    self.assertIsNone(stored.remote_result)
                    self.assertEqual(0.4, stored.result.confidence)

    def test_primary_failure_still_raises_without_an_event_or_other_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory), remote_allowed=True)
            failing = _FailingVLM()
            args["local_vlm"] = failing
            with self.assertRaises(VLMError) as raised:
                process_prepared_event(**args)
            self.assertIs(failing.error, raised.exception)
            self.assertIsNone(args["database"].get_event(args["candidate"].event_id))
            self.assertEqual(0, args["secondary_vlm"].calls)
            self.assertEqual(0, args["remote_vlm"].calls)
            attempts = RecognitionAttemptAuditStore(args["database"].path).recent()
            self.assertEqual(1, len(attempts))
            self.assertEqual(RecognitionStage.LOCAL_PRIMARY, attempts[0].stage)
            self.assertEqual(RecognitionAttemptStatus.FAILED, attempts[0].status)

    def test_secondary_interrupts_and_unexpected_exceptions_are_not_swallowed(self):
        for error in (
            KeyboardInterrupt(),
            SystemExit(7),
            RuntimeError("synthetic unexpected error"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                args = self._arguments(Path(directory), remote_allowed=True)
                args["secondary_vlm"] = _FailingVLM(error)
                with self.assertRaises(type(error)) as raised:
                    process_prepared_event(**args)
                self.assertIs(error, raised.exception)
                self.assertIsNone(args["database"].get_event(args["candidate"].event_id))
                self.assertEqual(1, args["local_vlm"].calls)
                self.assertEqual(0, args["remote_vlm"].calls)

    @staticmethod
    def _secondary_failed_audit_unavailable():
        complete = RecognitionAttemptAuditStore.complete

        def fail_only_failed_completion(store, attempt_id, **kwargs):
            if kwargs["status"] is RecognitionAttemptStatus.FAILED:
                raise OSError("synthetic secondary audit unavailable")
            return complete(store, attempt_id, **kwargs)

        return patch.object(
            RecognitionAttemptAuditStore, "complete", new=fail_only_failed_completion
        )

    def test_secondary_interrupt_or_unexpected_error_survives_simultaneous_audit_failure(self):
        for error in (
            KeyboardInterrupt(),
            SystemExit(7),
            RuntimeError("synthetic unexpected error"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                tempfile.TemporaryDirectory() as directory,
                self._secondary_failed_audit_unavailable(),
            ):
                args = self._arguments(Path(directory), remote_allowed=True)
                args["secondary_vlm"] = _FailingVLM(error)
                with self.assertRaises(type(error)) as raised:
                    process_prepared_event(**args)
                self.assertIs(error, raised.exception)
                self.assertIsNone(args["database"].get_event(args["candidate"].event_id))
                self.assertEqual(0, args["remote_vlm"].calls)
                attempts = RecognitionAttemptAuditStore(args["database"].path).recent()
                self.assertEqual(2, len(attempts))
                by_stage = {item.stage: item for item in attempts}
                self.assertEqual(
                    RecognitionAttemptStatus.SUCCEEDED,
                    by_stage[RecognitionStage.LOCAL_PRIMARY].status,
                )
                self.assertEqual(
                    RecognitionAttemptStatus.RUNNING,
                    by_stage[RecognitionStage.LOCAL_SECONDARY].status,
                )

    def test_secondary_vlm_and_audit_failure_preserve_primary_without_fabricating_success(self):
        with tempfile.TemporaryDirectory() as directory, self._secondary_failed_audit_unavailable():
            args = self._arguments(Path(directory), remote_allowed=True)
            summary = process_prepared_event(**args)
            stored = self._stored(args)
            self._assert_secondary_absent(stored)
            self.assertEqual(stored.local_result, stored.result)
            self.assertEqual(args["local_vlm"].metrics, stored.local_metrics)
            self.assertTrue(stored.decision.needs_local_review)
            self.assertFalse(summary.secondary_reviewed)
            self.assertEqual(0, args["remote_vlm"].calls)
            attempts = RecognitionAttemptAuditStore(args["database"].path).recent()
            self.assertEqual(2, len(attempts))
            by_stage = {item.stage: item for item in attempts}
            self.assertEqual(
                RecognitionAttemptStatus.SUCCEEDED, by_stage[RecognitionStage.LOCAL_PRIMARY].status
            )
            # An unavailable completion write is not a successful secondary run.
            secondary = by_stage[RecognitionStage.LOCAL_SECONDARY]
            self.assertEqual(RecognitionAttemptStatus.RUNNING, secondary.status)
            self.assertIsNone(secondary.metrics)
            self.assertIsNone(secondary.error_code)


if __name__ == "__main__":
    unittest.main()
