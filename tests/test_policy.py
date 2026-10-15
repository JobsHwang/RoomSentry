from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.domain import (
    DecisionRecord,
    EventType,
    FinalSource,
    ReviewTier,
    RiskLevel,
    SchemaError,
    SignalEvent,
    SignalType,
    VLMEvent,
)
from baby_monitor.policy import ReviewPolicy, reconcile_remote, record_remote_result


def _result(
    *,
    event_type: EventType = EventType.UNKNOWN,
    risk_level: RiskLevel = RiskLevel.NONE,
    confidence: float = 0.9,
    needs_remote_review: bool = False,
) -> VLMEvent:
    return VLMEvent(
        event_type=event_type,
        subjects=(),
        observations=("系统记录到测试事实。",),
        inferences=(),
        risk_level=risk_level,
        confidence=confidence,
        needs_remote_review=needs_remote_review,
        summary_zh="测试结果。",
        evidence_frames=("frame_01.jpg",),
    )


def _signal(
    signal_type: SignalType,
    *,
    severity_hint: str | None = None,
) -> SignalEvent:
    attributes = () if severity_hint is None else (("severity_hint", severity_hint),)
    return SignalEvent(
        signal_id=f"signal_{signal_type.value}",
        camera_id="cam_01",
        occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
        signal_type=signal_type,
        source_adapter="fixture",
        confidence=0.9,
        attributes=attributes,
    )


class DecisionPolicyTests(unittest.TestCase):
    def test_public_fixture_round_trips_and_schema_enums_match(self) -> None:
        fixture = Path("tests/fixtures/valid_decision_record.json").read_text(encoding="utf-8")
        decision = DecisionRecord.from_json(fixture)
        self.assertEqual(ReviewTier.LOCAL_SECONDARY, decision.review_tier)
        self.assertEqual(decision, DecisionRecord.from_dict(decision.to_dict()))

        schema = json.loads(
            Path("schemas/decision-record-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {value.value for value in ReviewTier},
            set(schema["properties"]["review_tier"]["enum"]),
        )
        self.assertFalse(schema["additionalProperties"])

    def test_extra_fields_and_remote_invariant_fail_closed(self) -> None:
        value = json.loads(
            Path("tests/fixtures/valid_decision_record.json").read_text(encoding="utf-8")
        )
        value["private_note"] = "not allowed"
        with self.assertRaisesRegex(SchemaError, "extra"):
            DecisionRecord.from_dict(value)
        value.pop("private_note")
        value["remote_used"] = True
        value["final_source"] = "remote"
        with self.assertRaisesRegex(SchemaError, "not allowed"):
            DecisionRecord.from_dict(value)

    def test_frequency_and_priority_raise_review_but_never_rewrite_risk(self) -> None:
        result = _result(risk_level=RiskLevel.LOW)
        motion = _signal(SignalType.MOTION)
        policy = ReviewPolicy(
            confidence_threshold=0.65,
            signal_frequency_secondary_threshold=3,
        )
        first = policy.decide(result, signals=(motion,), signal_frequency_count=1)
        repeated = policy.decide(result, signals=(motion,), signal_frequency_count=3)
        high_hint = policy.decide(
            result,
            signals=(_signal(SignalType.MOTION, severity_hint="high"),),
            signal_frequency_count=1,
        )

        self.assertEqual(ReviewTier.LOCAL_SECONDARY, first.review_tier)
        self.assertIn("evidence_coverage_unverified", first.review_reasons)
        self.assertEqual(ReviewTier.LOCAL_SECONDARY, repeated.review_tier)
        self.assertEqual(ReviewTier.LOCAL_SECONDARY, high_hint.review_tier)
        self.assertEqual(RiskLevel.LOW, repeated.risk_level)
        self.assertFalse(repeated.needs_remote_review)
        self.assertEqual((SignalType.MOTION,), repeated.frequent_signal_types)
        self.assertEqual(("系统记录到测试事实。",), result.observations)

    def test_low_confidence_requests_remote_without_overriding_consent(self) -> None:
        decision = ReviewPolicy().decide(
            _result(confidence=0.2),
            remote_allowed=False,
        )
        self.assertEqual(ReviewTier.REMOTE, decision.review_tier)
        self.assertTrue(decision.needs_remote_review)
        self.assertFalse(decision.remote_allowed)
        self.assertFalse(decision.remote_used)
        self.assertEqual(FinalSource.LOCAL_PRIMARY, decision.final_source)

    def test_remote_result_records_disagreement_and_final_source(self) -> None:
        local = _result(event_type=EventType.AWAKE)
        remote = _result(event_type=EventType.SLEEPING)
        final = reconcile_remote(local, remote)
        initial = ReviewPolicy().decide(
            local.require_remote_review(),
            remote_allowed=True,
        )
        decision = record_remote_result(
            initial,
            local=local,
            remote=remote,
            final=final,
        )
        self.assertTrue(decision.remote_used)
        self.assertFalse(final.needs_remote_review)
        self.assertEqual(FinalSource.REMOTE, decision.final_source)
        self.assertIn("event_type_disagreement", decision.contradictions)
        self.assertIn("contradictory_results", decision.review_reasons)


if __name__ == "__main__":
    unittest.main()
