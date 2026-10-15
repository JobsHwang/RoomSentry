from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime

from baby_monitor.domain import (
    CandidateEvent,
    Detection,
    EventType,
    RiskLevel,
    SignalEvent,
    SignalType,
    Subject,
    SubjectKind,
    VLMEvent,
)
from baby_monitor.fact_policy import (
    enforce_subject_identity_boundary,
    enforce_temporal_fact_boundary,
)


class SummaryFactBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = CandidateEvent(
            "synthetic",
            "cam_01",
            0.0,
            1.0,
            (Detection(0.5, "person", 0.9, (0.1, 0.1, 0.9, 0.9)),),
        )
        self.result = VLMEvent(
            event_type=EventType.UNKNOWN,
            subjects=(),
            observations=(),
            inferences=(),
            risk_level=RiskLevel.MEDIUM,
            confidence=0.95,
            needs_remote_review=False,
            summary_zh="",
            evidence_frames=(),
        )

    def test_summary_only_identity_claims_are_inferences(self) -> None:
        for text in ("宝宝在画面中", "婴儿在画面中", "成人在画面中", "An ADULT beside a baby"):
            with self.subTest(text=text):
                original = replace(self.result, summary_zh=text)
                bounded = enforce_subject_identity_boundary(original, self.candidate)
                self.assertNotEqual(text, bounded.summary_zh)
                self.assertEqual((f"身份证据不足，模型推测：{text}",), bounded.inferences)
                self.assertEqual(0.49, bounded.confidence)
                self.assertTrue(bounded.needs_remote_review)
                self.assertEqual(RiskLevel.MEDIUM, bounded.risk_level)
                self.assertEqual(text, original.summary_zh)

    def test_summary_only_temporal_claim_downgrades_event_type(self) -> None:
        for event_type, text in (
            (EventType.MOVEMENT, "person is walking"),
            (EventType.CRYING, "正在哭闹"),
        ):
            with self.subTest(event_type=event_type):
                original = replace(self.result, event_type=event_type, summary_zh=text)
                bounded = enforce_temporal_fact_boundary(original, self.candidate)
                self.assertEqual(EventType.UNKNOWN, bounded.event_type)
                self.assertNotEqual(text, bounded.summary_zh)
                self.assertEqual((f"时序证据不足，模型推测：{text}",), bounded.inferences)
                self.assertTrue(bounded.needs_remote_review)
                self.assertEqual(0.49, bounded.confidence)

    def test_hardware_signal_preserves_type_not_model_prose(self) -> None:
        for event_type, signal_type, text in (
            (EventType.MOVEMENT, SignalType.MOTION, "person is moving"),
            (EventType.CRYING, SignalType.HARDWARE_CRY, "正在哭闹"),
        ):
            with self.subTest(signal_type=signal_type):
                signal = SignalEvent(
                    "signal_01",
                    "cam_01",
                    datetime(2026, 9, 12, tzinfo=UTC),
                    signal_type,
                    "synthetic",
                    0.9,
                )
                candidate = replace(self.candidate, signals=(signal,))
                bounded = enforce_temporal_fact_boundary(
                    replace(self.result, event_type=event_type, summary_zh=text),
                    candidate,
                )
                self.assertEqual(event_type, bounded.event_type)
                self.assertNotEqual(text, bounded.summary_zh)
                self.assertEqual(0.49, bounded.confidence)

    def test_disclaimer_does_not_exempt_other_claims(self) -> None:
        for text in (
            "宝宝在画面中；这不是身份结论。",
            "宝宝在画面中；这不是宝宝身份结论。",
            "这不是宝宝身份结论；成人在画面中。",
            "不是身份结论：宝宝在画面中。",
            "不能证明；宝宝在画面中，这是结论。",
            "A baby is visible; this is not an identity conclusion.",
            "This is not a baby identity conclusion; an adult is visible.",
            "This is not a conclusion. A baby is visible.",
            "Cannot determine, a baby is visible: identity conclusion.",
        ):
            for field in ("summary_zh", "observations"):
                with self.subTest(text=text, field=field):
                    value = (text,) if field == "observations" else text
                    bounded = enforce_subject_identity_boundary(
                        replace(self.result, **{field: value}),
                        self.candidate,
                    )
                    self.assertEqual((), bounded.observations)
                    self.assertNotEqual(text, bounded.summary_zh)
                    self.assertIn(f"身份证据不足，模型推测：{text}", bounded.inferences)

    def test_local_identity_disclaimers_remain_unchanged(self) -> None:
        for text in (
            "系统记录到 person；这不是宝宝/成人身份结论。",
            "This is not a baby/adult identity conclusion.",
            "不能确定宝宝身份。并非成人身份结论。",
        ):
            with self.subTest(text=text):
                original = replace(self.result, observations=(text,), summary_zh=text)
                self.assertIs(original, enforce_subject_identity_boundary(original, self.candidate))

    def test_disclaimer_span_stops_at_bilingual_clause_boundaries(self) -> None:
        for separator in "。.;；,，:：!！?？\r\n":
            for prefix, assertion in (
                ("不能确定其他信息", "宝宝身份已确认。"),
                ("This is not a risk assessment", "baby identity is confirmed."),
            ):
                for field in ("summary_zh", "observations"):
                    with self.subTest(separator=repr(separator), prefix=prefix, field=field):
                        text = prefix + separator + assertion
                        value = (text,) if field == "observations" else text
                        bounded = enforce_subject_identity_boundary(
                            replace(self.result, **{field: value}),
                            self.candidate,
                        )
                        self.assertEqual((), bounded.observations)
                        self.assertNotEqual(text, bounded.summary_zh)
                        self.assertIn(f"身份证据不足，模型推测：{text}", bounded.inferences)

    def test_explicit_identity_label_only_exempts_matching_kind(self) -> None:
        candidate = replace(
            self.candidate, detections=(Detection(0.5, "adult", 0.9, (0.1, 0.1, 0.9, 0.9)),)
        )
        original = replace(self.result, summary_zh="an adult is visible")
        self.assertIs(original, enforce_subject_identity_boundary(original, candidate))
        bounded = enforce_subject_identity_boundary(
            replace(original, summary_zh="an adult beside a baby"),
            candidate,
        )
        self.assertEqual(0.49, bounded.confidence)

    def test_combined_filters_are_idempotent_and_preserve_original_once(self) -> None:
        text = "宝宝正在哭闹"
        original = replace(self.result, summary_zh=text, observations=(text,))
        bounded = enforce_subject_identity_boundary(
            enforce_temporal_fact_boundary(original, self.candidate),
            self.candidate,
        )
        repeated = enforce_subject_identity_boundary(
            enforce_temporal_fact_boundary(bounded, self.candidate),
            self.candidate,
        )
        self.assertIs(bounded, repeated)
        self.assertEqual(1, sum(text in inference for inference in bounded.inferences))
        self.assertEqual((), bounded.observations)
        self.assertNotIn(text, bounded.summary_zh)

    def test_low_confidence_is_not_inflated(self) -> None:
        for policy, text in (
            (enforce_temporal_fact_boundary, "person is walking"),
            (enforce_subject_identity_boundary, "宝宝在画面中"),
        ):
            with self.subTest(policy=policy.__name__):
                bounded = policy(
                    replace(self.result, summary_zh=text, confidence=0.2), self.candidate
                )
                self.assertEqual(0.2, bounded.confidence)

    def test_safe_summary_and_empty_summary_preserve_object(self) -> None:
        for text in ("", "画面中可见 person。", "A chair is visible"):
            with self.subTest(text=text):
                original = replace(self.result, summary_zh=text)
                self.assertIs(original, enforce_temporal_fact_boundary(original, self.candidate))
                self.assertIs(original, enforce_subject_identity_boundary(original, self.candidate))

    def test_identity_filter_is_idempotent_and_deduplicates(self) -> None:
        text = "宝宝在画面中"
        inference = f"身份证据不足，模型推测：{text}"
        original = replace(
            self.result, summary_zh=text, observations=(text,), inferences=(inference,)
        )
        bounded = enforce_subject_identity_boundary(original, self.candidate)
        self.assertEqual((inference,), bounded.inferences)
        self.assertIs(bounded, enforce_subject_identity_boundary(bounded, self.candidate))

    def test_age_synonyms_are_bounded_even_when_structured_kind_is_person(self) -> None:
        # Synthetic wording: public fixtures never copy private model records.
        for text in ("婴幼儿在床边。", "成年人在画面左侧。", "成年人旁有婴幼儿。"):
            for field in ("observations", "summary_zh"):
                with self.subTest(text=text, field=field):
                    original = replace(
                        self.result,
                        subjects=(Subject(SubjectKind.PERSON, 0.9, ()),),
                        **{field: (text,) if field == "observations" else text},
                    )
                    bounded = enforce_subject_identity_boundary(original, self.candidate)
                    self.assertEqual((), bounded.observations)
                    self.assertNotEqual(text, bounded.summary_zh)
                    self.assertIn(f"身份证据不足，模型推测：{text}", bounded.inferences)
                    self.assertEqual(original.subjects, bounded.subjects)
                    self.assertEqual(0.49, bounded.confidence)
                    self.assertTrue(bounded.needs_remote_review)
                    self.assertIs(
                        bounded, enforce_subject_identity_boundary(bounded, self.candidate)
                    )

    def test_synonyms_require_the_matching_explicit_role(self) -> None:
        for label, matching, unsupported in (
            ("adult", "成年人在画面中。", "成年人旁有婴幼儿。"),
            ("baby", "婴幼儿在画面中。", "婴幼儿旁有成年人。"),
        ):
            with self.subTest(label=label):
                candidate = replace(
                    self.candidate,
                    detections=(Detection(0.5, label, 0.9, (0.1, 0.1, 0.9, 0.9)),),
                )
                original = replace(self.result, observations=(matching,), summary_zh=matching)
                self.assertIs(original, enforce_subject_identity_boundary(original, candidate))
                bounded = enforce_subject_identity_boundary(
                    replace(original, observations=(unsupported,), summary_zh=unsupported),
                    candidate,
                )
                self.assertEqual((), bounded.observations)
                self.assertEqual(0.49, bounded.confidence)

    def test_synonym_disclaimer_does_not_exempt_next_clause(self) -> None:
        disclaimer = "这不是婴幼儿/成年人身份结论。"
        original = replace(self.result, observations=(disclaimer,), summary_zh=disclaimer)
        self.assertIs(original, enforce_subject_identity_boundary(original, self.candidate))
        for assertion in ("婴幼儿在床边。", "成年人在画面左侧。"):
            text = disclaimer + assertion
            bounded = enforce_subject_identity_boundary(
                replace(original, observations=(text,), summary_zh=text), self.candidate
            )
            self.assertEqual((), bounded.observations)
            self.assertNotEqual(text, bounded.summary_zh)

    def test_synonym_filter_does_not_inflate_low_confidence(self) -> None:
        for text in ("婴幼儿在床边。", "成年人在画面左侧。"):
            bounded = enforce_subject_identity_boundary(
                replace(self.result, summary_zh=text, confidence=0.2), self.candidate
            )
            self.assertEqual(0.2, bounded.confidence)

    def test_generic_person_text_is_still_visible_not_a_role_claim(self) -> None:
        text = "画面左侧可见一名person，其年龄无法确定。"
        original = replace(self.result, observations=(text,), summary_zh=text)
        self.assertIs(original, enforce_subject_identity_boundary(original, self.candidate))
