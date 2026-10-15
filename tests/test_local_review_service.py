from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

from baby_monitor.adapters.vlm import MockVLMAdapter, VLMError
from baby_monitor.config import StorageSettings
from baby_monitor.domain import FinalSource, ReviewTier, RiskLevel, SignalType
from baby_monitor.evidence_binding import EvidenceBinding
from baby_monitor.local_review_service import LocalReviewError, retry_local_review
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy
from baby_monitor.recognition_attempt import (
    RecognitionAttemptAuditStore,
    RecognitionAttemptStatus,
    RecognitionStage,
)
from baby_monitor.reporting import DailyReportGenerator
from baby_monitor.storage import StorageResolver
from tests import test_pipeline_evidence_binding as binding_fixtures
from tests.test_policy import _signal
from tests.test_secondary_review_failure import _StaticVLM


class _LocalStaticVLM(_StaticVLM, MockVLMAdapter):
    """Deterministic local adapter admitted through the explicit mock contract."""

    def __init__(self, *, confidence=0.9, error=None, mutate=None, risk_level=None):
        super().__init__("synthetic_retry_secondary", confidence=confidence)
        self.error = error
        self.mutate = mutate
        self.risk_level = risk_level

    def analyze(self, request):
        result = super().analyze(request)
        if self.error is not None:
            raise self.error
        if self.mutate is not None:
            self.mutate()
        if self.risk_level is not None:
            result = replace(result, risk_level=self.risk_level)
        return result


class LocalReviewServiceTests(unittest.TestCase):
    FIRST_REQUEST = "1" * 32
    SECOND_REQUEST = "2" * 32

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.arguments = binding_fixtures.PipelineEvidenceBindingTests()._arguments(self.root)
        self.database = self.arguments["database"]
        self.event_id = self.arguments["candidate"].event_id

    def _seed(self):
        process_prepared_event(**self.arguments)
        self.original = self.database.get_event(self.event_id)
        self.assertIsNotNone(self.original)
        return self.original

    def _retry(self, factory, *, request_id=None, policy=None):
        return retry_local_review(
            database_path=self.database.path,
            event_id=self.event_id,
            request_id=request_id or self.FIRST_REQUEST,
            secondary_adapter_factory=factory,
            policy=policy or ReviewPolicy(),
            lease_seconds=600,
        )

    def _assert_path_free(self, receipt):
        serialized = json.dumps(receipt, ensure_ascii=False)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("synthetic.mp4", serialized)
        self.assertNotIn("synthetic_01.jpg", serialized)
        self.assertNotIn("PRIVATE_FAILURE_DETAIL", serialized)

    def _assert_primary_preserved(self, stored):
        for field in (
            "local_result",
            "local_provenance",
            "local_metrics",
            "occurred_at",
            "created_at",
            "candidate",
            "evidence",
            "camera_id",
            "source_kind",
            "source_ref",
        ):
            self.assertEqual(getattr(self.original, field), getattr(stored, field), field)
        self.assertEqual(1, self.arguments["local_vlm"].calls)
        self.assertEqual(0, self.arguments["remote_vlm"].calls)
        self.assertFalse(stored.remote_used)

    def test_success_runs_only_secondary_preserves_primary_and_same_event_binding(self):
        self._seed()
        binding = self.database.get_evidence_binding(self.event_id)
        adapter = _LocalStaticVLM()
        factory = Mock(return_value=adapter)
        with (
            patch.object(
                self.database.__class__,
                "count_recent_signal_events",
                side_effect=AssertionError("recount"),
            ),
            patch(
                "baby_monitor.notification.NotificationOutbox.enqueue",
                side_effect=AssertionError("notification"),
            ),
            patch.object(
                DailyReportGenerator, "generate", side_effect=AssertionError("report refresh")
            ),
        ):
            receipt = self._retry(factory)
        self.assertEqual("succeeded", receipt["status"])
        self.assertFalse(receipt["reused_existing"])
        self.assertEqual(self.FIRST_REQUEST, receipt["request_id"])
        self.assertEqual(self.event_id, receipt["event_id"])
        self.assertFalse(receipt["needs_review"])
        self._assert_path_free(receipt)
        factory.assert_called_once_with()
        self.assertEqual(1, adapter.calls)
        stored = self.database.get_event(self.event_id)
        self._assert_primary_preserved(stored)
        self.assertEqual(FinalSource.LOCAL_SECONDARY, stored.decision.final_source)
        self.assertEqual(stored.secondary_result, stored.result)
        self.assertEqual(adapter.metrics, stored.secondary_metrics)
        self.assertEqual(adapter.provenance, stored.secondary_provenance)
        self.assertEqual("unverified", stored.evidence.coverage_status)
        self.assertIn("evidence_coverage_unverified", stored.decision.review_reasons)
        self.assertEqual(binding, self.database.get_evidence_binding(self.event_id))
        attempts = RecognitionAttemptAuditStore(self.database.path).recent()
        self.assertEqual(2, len(attempts))
        self.assertEqual(
            1, sum(item.stage is RecognitionStage.LOCAL_SECONDARY for item in attempts)
        )
        self.assertTrue(all(item.status is RecognitionAttemptStatus.SUCCEEDED for item in attempts))

    def test_success_replay_does_not_run_factory_or_reverify_current_media(self):
        self._seed()
        first = self._retry(Mock(return_value=_LocalStaticVLM()))
        factory = Mock(side_effect=AssertionError("factory must remain lazy"))
        with patch.object(EvidenceBinding, "verify", side_effect=AssertionError("reverify")):
            replay = self._retry(factory)
        self.assertEqual(first["status"], replay["status"])
        self.assertTrue(replay["reused_existing"])
        factory.assert_not_called()

    def test_new_request_after_success_is_rejected_before_factory(self):
        self._seed()
        self._retry(Mock(return_value=_LocalStaticVLM()))
        factory = Mock(side_effect=AssertionError("already reviewed"))
        with self.assertRaises(LocalReviewError):
            self._retry(factory, request_id=self.SECOND_REQUEST)
        factory.assert_not_called()

    def test_legacy_unbound_event_is_ineligible_even_if_media_now_safe(self):
        clip = Path(self.arguments["media"].clip_path)
        clip.chmod(0o644)
        self._seed()
        self.assertIsNone(self.database.get_evidence_binding(self.event_id))
        clip.chmod(0o600)
        factory = Mock(side_effect=AssertionError("legacy event reached factory"))
        with self.assertRaises(LocalReviewError):
            self._retry(factory)
        factory.assert_not_called()
        self.assertIsNone(self.database.get_evidence_binding(self.event_id))

    def test_original_remote_pending_and_reason_survive_agreeing_secondary(self):
        self.arguments["local_vlm"] = _StaticVLM("synthetic_primary", confidence=0.4)
        self._seed()
        receipt = self._retry(Mock(return_value=_LocalStaticVLM()))
        self.assertEqual("succeeded", receipt["status"])
        self.assertTrue(receipt["needs_review"])
        stored = self.database.get_event(self.event_id)
        self.assertEqual(ReviewTier.REMOTE, stored.decision.review_tier)
        self.assertIn("low_confidence", stored.decision.review_reasons)
        self.assertTrue(stored.decision.needs_remote_review)
        self._assert_primary_preserved(stored)

    def test_secondary_low_confidence_only_requests_remote_without_calling_it(self):
        self._seed()
        receipt = self._retry(Mock(return_value=_LocalStaticVLM(confidence=0.3)))
        self.assertEqual("succeeded", receipt["status"])
        self.assertTrue(receipt["needs_review"])
        stored = self.database.get_event(self.event_id)
        self.assertEqual(ReviewTier.REMOTE, stored.decision.review_tier)
        self.assertIn("low_confidence", stored.decision.review_reasons)
        self._assert_primary_preserved(stored)

    def test_secondary_semantic_disagreement_is_pending_not_remote_execution(self):
        self._seed()
        receipt = self._retry(Mock(return_value=_LocalStaticVLM(risk_level=RiskLevel.LOW)))
        self.assertEqual("succeeded", receipt["status"])
        stored = self.database.get_event(self.event_id)
        self.assertEqual(ReviewTier.REMOTE, stored.decision.review_tier)
        self.assertIn("contradictory_results", stored.decision.review_reasons)
        self._assert_primary_preserved(stored)

    def test_policy_reuses_stored_signals_and_frequency_without_recounting(self):
        signal = _signal(SignalType.MOTION)
        self.arguments["candidate"] = replace(self.arguments["candidate"], signals=(signal,))
        with patch.object(self.database.__class__, "count_recent_signal_events", return_value=4):
            self._seed()
        policy = Mock(spec=ReviewPolicy, wraps=ReviewPolicy())
        with patch.object(
            self.database.__class__,
            "count_recent_signal_events",
            side_effect=AssertionError("recount"),
        ):
            self._retry(Mock(return_value=_LocalStaticVLM()), policy=policy)
        self.assertEqual(5, policy.decide.call_args.kwargs["signal_frequency_count"])
        self.assertEqual((signal,), policy.decide.call_args.kwargs["signals"])

    def test_same_running_request_is_reused_without_second_model_call(self):
        self._seed()
        nested = []
        adapter = _LocalStaticVLM()
        forbidden = Mock(side_effect=AssertionError("duplicate factory"))

        def factory():
            with patch.object(EvidenceBinding, "verify", side_effect=AssertionError("reverify")):
                nested.append(self._retry(forbidden))
            return adapter

        result = self._retry(factory)
        self.assertEqual("succeeded", result["status"])
        self.assertEqual("running", nested[0]["status"])
        self.assertTrue(nested[0]["reused_existing"])
        self.assertEqual(1, adapter.calls)
        forbidden.assert_not_called()

    def test_controlled_failure_keeps_pending_primary_and_new_request_can_recover(self):
        self._seed()
        adapter = _LocalStaticVLM(
            error=VLMError("PRIVATE_FAILURE_DETAIL", code="invalid_structured_output")
        )
        failed = self._retry(Mock(return_value=adapter))
        self.assertEqual("failed", failed["status"])
        self.assertTrue(failed["needs_review"])
        self.assertEqual("model_failed", failed["error_code"])
        self._assert_path_free(failed)
        stored = self.database.get_event(self.event_id)
        self._assert_primary_preserved(stored)
        self.assertEqual(self.original.decision, stored.decision)
        self.assertIsNone(stored.secondary_result)
        self.assertIsNone(stored.secondary_metrics)
        self.assertIsNone(stored.secondary_provenance)
        forbidden = Mock(side_effect=AssertionError("failed request replay"))
        replay = self._retry(forbidden)
        self.assertEqual("failed", replay["status"])
        self.assertTrue(replay["reused_existing"])
        forbidden.assert_not_called()
        recovered = self._retry(
            Mock(return_value=_LocalStaticVLM()), request_id=self.SECOND_REQUEST
        )
        self.assertEqual("succeeded", recovered["status"])
        self.assertFalse(recovered["needs_review"])

    def test_changed_media_before_retry_prevents_factory_and_preserves_event(self):
        self._seed()
        clip = Path(self.arguments["media"].clip_path)
        clip.write_bytes(b"X" * clip.stat().st_size)
        factory = Mock(side_effect=AssertionError("changed evidence reached factory"))
        receipt = self._retry(factory)
        self.assertEqual("failed", receipt["status"])
        self.assertEqual("evidence_changed", receipt["error_code"])
        self._assert_path_free(receipt)
        factory.assert_not_called()
        self.assertEqual(self.original, self.database.get_event(self.event_id))

    def test_changed_media_during_secondary_rejects_result_without_updating_event(self):
        self._seed()
        frame = Path(self.arguments["media"].frame_paths[0])
        adapter = _LocalStaticVLM(mutate=lambda: frame.write_bytes(b"X" * frame.stat().st_size))
        receipt = self._retry(Mock(return_value=adapter))
        self.assertEqual("failed", receipt["status"])
        self.assertEqual("evidence_changed", receipt["error_code"])
        self._assert_path_free(receipt)
        self.assertEqual(1, adapter.calls)
        self.assertEqual(self.original, self.database.get_event(self.event_id))

    def test_adapter_outside_local_allowlist_never_runs(self):
        self._seed()
        untrusted = _StaticVLM("synthetic_untrusted_remote_adapter")
        receipt = self._retry(Mock(return_value=untrusted))
        self.assertEqual("failed", receipt["status"])
        self.assertEqual(0, untrusted.calls)
        self.assertEqual(self.original, self.database.get_event(self.event_id))
        self._assert_path_free(receipt)

    def test_unconfigured_secondary_is_a_controlled_failure_without_losing_primary(self):
        self._seed()
        receipt = self._retry(Mock(return_value=None))
        self.assertEqual("failed", receipt["status"])
        self.assertEqual(self.original, self.database.get_event(self.event_id))
        self._assert_path_free(receipt)

    def test_unexpected_failure_and_process_interrupts_propagate_after_receipt_recording(self):
        self._seed()
        for index, kind in enumerate((RuntimeError, KeyboardInterrupt, SystemExit), start=3):
            with self.subTest(kind=kind.__name__):
                original_error = kind("PRIVATE_FAILURE_DETAIL")
                adapter = _LocalStaticVLM(error=original_error)
                request_id = str(index) * 32
                with self.assertRaises(kind) as raised:
                    self._retry(Mock(return_value=adapter), request_id=request_id)
                self.assertIs(original_error, raised.exception)
                forbidden = Mock(side_effect=AssertionError("unexpected failure replay"))
                replay = self._retry(forbidden, request_id=request_id)
                self.assertEqual("failed", replay["status"])
                self.assertEqual("review_failed", replay["error_code"])
                self._assert_path_free(replay)
                forbidden.assert_not_called()
                self.assertEqual(self.original, self.database.get_event(self.event_id))

    def test_explicit_report_regeneration_reflects_completed_local_review(self):
        self._seed()
        generator = DailyReportGenerator(
            database=self.database,
            storage=StorageResolver(
                StorageSettings(local_root=self.root / "report", database=self.database.path)
            ),
            timezone_name="Asia/Shanghai",
        )
        report = generator.generate(date(2026, 9, 12))
        before = report.markdown_path.read_bytes()
        self.assertIn("本地待复核：是", before.decode())
        self._retry(Mock(return_value=_LocalStaticVLM()))
        self.assertEqual(before, report.markdown_path.read_bytes())
        refreshed = generator.generate(date(2026, 9, 12))
        for path in (refreshed.markdown_path, refreshed.html_path):
            content = path.read_text(encoding="utf-8")
            self.assertIn("本地待复核：否", content)
            self.assertIn("未验证连续覆盖", content)

    def test_interruption_survives_recognition_audit_failure(self):
        self._seed()
        for index, kind in enumerate((KeyboardInterrupt, SystemExit), start=5):
            with self.subTest(kind=kind.__name__):
                original = kind("PRIVATE_FAILURE_DETAIL")
                adapter = _LocalStaticVLM(error=original)
                request_id = str(index) * 32
                with (
                    patch.object(
                        RecognitionAttemptAuditStore,
                        "complete",
                        side_effect=OSError("synthetic audit failure"),
                    ),
                    self.assertRaises(kind) as raised,
                ):
                    self._retry(Mock(return_value=adapter), request_id=request_id)
                self.assertIs(original, raised.exception)
                replay = self._retry(
                    Mock(side_effect=AssertionError("retry")), request_id=request_id
                )
                self.assertEqual("review_failed", replay["error_code"])
                self.assertEqual(self.original, self.database.get_event(self.event_id))

    def test_original_interruption_survives_failure_to_persist_request_failure(self):
        self._seed()
        original = KeyboardInterrupt("PRIVATE_FAILURE_DETAIL")
        with (
            patch(
                "baby_monitor.local_review_service.LocalReviewStore.fail",
                side_effect=OSError("synthetic request audit failure"),
            ),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            self._retry(Mock(return_value=_LocalStaticVLM(error=original)))
        self.assertIs(original, raised.exception)
        self.assertEqual(self.original, self.database.get_event(self.event_id))

    def test_controlled_failure_does_not_manufacture_terminal_receipt_when_store_fail_fails(self):
        self._seed()
        error = LocalReviewError("database_unavailable")
        with (
            patch("baby_monitor.local_review_service.LocalReviewStore.fail", side_effect=error),
            self.assertRaises(LocalReviewError) as raised,
        ):
            self._retry(
                Mock(return_value=_LocalStaticVLM(error=VLMError("PRIVATE_FAILURE_DETAIL")))
            )
        self.assertIs(error, raised.exception)
        self.assertEqual(self.original, self.database.get_event(self.event_id))


if __name__ == "__main__":
    unittest.main()
