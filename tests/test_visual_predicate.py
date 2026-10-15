from __future__ import annotations

import json
import unittest

from baby_monitor.domain import SchemaError
from baby_monitor.visual_predicate import (
    VisualPredicate,
    VisualPredicateAnswer,
    VisualPredicateResult,
)


class VisualPredicateTests(unittest.TestCase):
    revision = "1" * 40

    def _result(self, **changes: object) -> dict[str, object]:
        value: dict[str, object] = {
            "version": 1,
            "event_id": "event_01",
            "predicate": "support_contact_visible",
            "answer": "yes",
            "confidence": 0.8,
            "evidence_frames": ["frame_02"],
            "model_id": "microsoft.Phi-3.5-vision-instruct",
            "model_revision": self.revision,
            "prompt_version": "visual_predicate_v1",
            "subject_identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
        }
        value.update(changes)
        return value

    def test_accepts_path_free_evidenced_atomic_result(self) -> None:
        result = VisualPredicateResult.from_dict(
            self._result(),
            allowed_frame_ids=("frame_01", "frame_02", "frame_03"),
        )
        self.assertEqual(result.predicate, VisualPredicate.SUPPORT_CONTACT_VISIBLE)
        self.assertEqual(result.answer, VisualPredicateAnswer.YES)
        self.assertFalse(result.risk_inference)
        self.assertFalse(result.notification_decision)
        self.assertEqual(result, VisualPredicateResult.from_json(result.to_json()))

    def test_unknown_requires_no_evidence_and_low_confidence(self) -> None:
        accepted = VisualPredicateResult.from_dict(
            self._result(answer="unknown", confidence=0.4, evidence_frames=[]),
        )
        self.assertEqual(accepted.answer, VisualPredicateAnswer.UNKNOWN)
        for changes in (
            {"answer": "unknown", "confidence": 0.6, "evidence_frames": []},
            {"answer": "unknown", "confidence": 0.4, "evidence_frames": ["frame_01"]},
        ):
            with self.assertRaises(SchemaError):
                VisualPredicateResult.from_dict(self._result(**changes))

    def test_rejects_wrong_answer_family_and_unadmitted_frame(self) -> None:
        with self.assertRaises(SchemaError):
            VisualPredicateResult.from_dict(self._result(answer="clear"))
        with self.assertRaisesRegex(SchemaError, "unadmitted"):
            VisualPredicateResult.from_dict(
                self._result(evidence_frames=["frame_99"]),
                allowed_frame_ids=("frame_01", "frame_02"),
            )

    def test_rejects_identity_risk_notification_or_extra_fields(self) -> None:
        for changes in (
            {"subject_identity_inference": True},
            {"risk_inference": True},
            {"notification_decision": True},
        ):
            with self.assertRaisesRegex(SchemaError, "inference boundary"):
                VisualPredicateResult.from_dict(self._result(**changes))
        value = self._result()
        value["summary"] = "invented"
        with self.assertRaisesRegex(SchemaError, "fields"):
            VisualPredicateResult.from_dict(value)

    def test_rejects_duplicate_json_and_nonfinite_confidence(self) -> None:
        raw = json.dumps(self._result()).replace('"version": 1', '"version": 1, "version": 1')
        with self.assertRaisesRegex(SchemaError, "duplicate"):
            VisualPredicateResult.from_json(raw)
        with self.assertRaises(SchemaError):
            VisualPredicateResult.from_dict(self._result(confidence=float("nan")))


if __name__ == "__main__":
    unittest.main()
