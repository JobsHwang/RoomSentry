"""Local-only layer-two admission tests; no installed model or GPU required."""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.adapters.vlm import CommandVLMAdapter, VLMAdapter, VLMError
from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    Detection,
    EventMedia,
    EventType,
    FinalSource,
    RiskLevel,
    VLMEvent,
)
from baby_monitor.local_runners.bonsai_27b import MODEL_REVISION, PROMPT_VERSION
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy
from baby_monitor.recognition_attempt import RecognitionAttemptAuditStore, RecognitionAttemptStatus


class LocalFixture(VLMAdapter):
    name = "bonsai_contract_fixture"

    def __init__(self, *, confidence=0.6, risk=RiskLevel.NONE, failure=False):
        self.confidence, self.risk, self.failure = confidence, risk, failure
        self.calls = 0

    def analyze(self, request):
        self.calls += 1
        if self.failure:
            raise VLMError("bounded fixture failure")
        return VLMEvent(
            event_type=EventType.UNKNOWN,
            subjects=(),
            observations=("可见人物。",),
            inferences=(),
            risk_level=self.risk,
            confidence=self.confidence,
            needs_remote_review=False,
            summary_zh="身份与状态未知。",
            evidence_frames=request.media.frame_paths,
        )


class BonsaiPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        frame, clip = root / "frame.jpg", root / "event.mp4"
        frame.write_bytes(b"synthetic contract bytes, not media")
        clip.write_bytes(b"synthetic contract bytes, not video")
        self.database = EventDatabase(root / "events.sqlite3")
        self.candidate = CandidateEvent(
            "local_two_layer_fixture",
            "public_cam",
            0,
            1,
            (Detection(0, "person", 0.9, (0.1, 0.1, 0.5, 0.5)),),
        )
        self.remote = LocalFixture(failure=True)
        self.kwargs = dict(
            candidate=self.candidate,
            occurred_at=datetime(2026, 9, 29, tzinfo=UTC),
            source_kind="fixture",
            source_ref="fixture",
            media=EventMedia(str(clip), (str(frame),)),
            database=self.database,
            policy=ReviewPolicy(),
            secondary_vlm=None,
            remote_vlm=self.remote,
            remote_uploads_allowed=False,
        )

    def test_low_confidence_false_review_flag_stays_pending_without_cloud(self):
        local = LocalFixture()
        result = process_prepared_event(local_vlm=local, **self.kwargs)
        stored = self.database.get_event(self.candidate.event_id)
        self.assertFalse(stored.local_result.needs_remote_review)
        self.assertTrue(stored.result.needs_remote_review)
        self.assertTrue(result.pending_remote_review)
        self.assertIn("low_confidence", stored.decision.review_reasons)
        self.assertEqual(0, self.remote.calls)
        self.assertFalse(stored.remote_used)
        self.assertFalse(stored.decision.remote_allowed)
        self.assertEqual(EventType.UNKNOWN, stored.result.event_type)
        self.assertEqual(FinalSource.LOCAL_PRIMARY, stored.decision.final_source)
        self.assertIsNone(stored.secondary_result)
        self.assertTrue(process_prepared_event(local_vlm=local, **self.kwargs).reused_existing)
        self.assertEqual(1, local.calls)

    def test_high_confidence_unknown_is_not_accepted_as_complete_coverage(self):
        process_prepared_event(local_vlm=LocalFixture(confidence=0.95), **self.kwargs)
        stored = self.database.get_event(self.candidate.event_id)
        self.assertEqual(EventType.UNKNOWN, stored.result.event_type)
        self.assertTrue(stored.decision.needs_review)
        self.assertIn("evidence_coverage_unverified", stored.decision.review_reasons)
        self.assertEqual(0, self.remote.calls)

    def test_elevated_risk_does_not_override_disabled_remote(self):
        result = process_prepared_event(
            local_vlm=LocalFixture(confidence=0.95, risk=RiskLevel.HIGH), **self.kwargs
        )
        self.assertTrue(result.pending_remote_review)
        self.assertEqual(0, self.remote.calls)
        stored = self.database.get_event(self.candidate.event_id)
        self.assertIn("elevated_risk", stored.decision.review_reasons)
        self.assertFalse(stored.remote_used)

    def test_runner_failure_records_failed_attempt_not_normal_event(self):
        with self.assertRaises(VLMError):
            process_prepared_event(local_vlm=LocalFixture(failure=True), **self.kwargs)
        self.assertIsNone(self.database.get_event(self.candidate.event_id))
        attempts = RecognitionAttemptAuditStore(self.database.path).recent()
        self.assertEqual(1, len(attempts))
        self.assertEqual(RecognitionAttemptStatus.FAILED, attempts[0].status)
        self.assertEqual(0, self.remote.calls)

    def test_explicit_bonsai_prompt_revision_is_auditable(self):
        adapter = CommandVLMAdapter(
            name="bonsai_2_27b",
            command=("/fixture/python", "{request}"),
            model_revision=MODEL_REVISION,
            prompt_revision=PROMPT_VERSION,
        )
        self.assertEqual(PROMPT_VERSION, adapter.provenance.prompt_revision)
        self.assertEqual(MODEL_REVISION, adapter.provenance.model_revision)


if __name__ == "__main__":
    unittest.main()
