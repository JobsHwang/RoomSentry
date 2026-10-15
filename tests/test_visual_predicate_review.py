from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.visual_predicate import (
    VisualPredicate,
    VisualPredicateAnswer,
    VisualPredicateResult,
)
from baby_monitor.visual_predicate_review import (
    VisualPredicateReviewError,
    VisualPredicateReviewReason,
    VisualPredicateReviewState,
    assess_visual_predicate_review,
    build_visual_predicate_review_report,
)
from baby_monitor.visual_predicate_shadow import (
    VisualPredicateShadowGate,
    VisualPredicateShadowRun,
    VisualPredicateShadowStore,
    VisualPredicateShadowTrigger,
)
from baby_monitor.visual_predicate_shadow_queue import (
    VisualPredicateShadowAttempt,
    VisualPredicateShadowAttemptQueue,
    VisualPredicateShadowAttemptStatus,
    VisualPredicateShadowObservation,
)


class VisualPredicateReviewTests(unittest.TestCase):
    NOW = datetime(2026, 8, 19, 12, tzinfo=UTC)
    REVISION = "5c78cab61938e73859b63724d9bf5cb88c477eaa"

    @classmethod
    def _gate(
        cls,
        predicates: tuple[VisualPredicate, ...],
        *,
        event_id: str = "event_01",
    ) -> VisualPredicateShadowGate:
        return VisualPredicateShadowGate(
            event_id=event_id,
            selected=True,
            triggers=(VisualPredicateShadowTrigger.PERIODIC_SAMPLE,),
            predicates=predicates,
            person_detector_id=None,
            person_detection_count=0,
            hardware_signal_adapters=(),
            hardware_signal_count=0,
        )

    @classmethod
    def _result(
        cls,
        predicate: VisualPredicate,
        answer: VisualPredicateAnswer,
        *,
        event_id: str = "event_01",
    ) -> VisualPredicateResult:
        unknown = answer is VisualPredicateAnswer.UNKNOWN
        return VisualPredicateResult(
            event_id=event_id,
            predicate=predicate,
            answer=answer,
            confidence=0.5 if unknown else 0.9,
            evidence_frames=() if unknown else ("frame_01",),
            model_id="microsoft.Mage-VL",
            model_revision=cls.REVISION,
            prompt_version="atomic_shadow_v1",
        )

    @classmethod
    def _completed(
        cls,
        results: tuple[VisualPredicateResult, ...],
        *,
        event_id: str = "event_01",
    ) -> VisualPredicateShadowObservation:
        gate = cls._gate(tuple(item.predicate for item in results), event_id=event_id)
        attempt = VisualPredicateShadowAttempt(
            gate=gate,
            status=VisualPredicateShadowAttemptStatus.COMPLETED,
            attempt_count=1,
            available_at=cls.NOW,
            lease_until=None,
            last_error_code=None,
            result_binding_sha256="a" * 64,
            created_at=cls.NOW,
            updated_at=cls.NOW,
            completed_at=cls.NOW,
        )
        return VisualPredicateShadowObservation(
            attempt=attempt,
            results=results,
            recorded_at=cls.NOW,
        )

    @classmethod
    def _pending(cls, *, event_id: str = "event_pending") -> VisualPredicateShadowObservation:
        gate = cls._gate((VisualPredicate.FACE_VISIBILITY,), event_id=event_id)
        return VisualPredicateShadowObservation(
            attempt=VisualPredicateShadowAttempt(
                gate=gate,
                status=VisualPredicateShadowAttemptStatus.PENDING,
                attempt_count=0,
                available_at=cls.NOW,
                lease_until=None,
                last_error_code=None,
                result_binding_sha256=None,
                created_at=cls.NOW,
                updated_at=cls.NOW,
                completed_at=None,
            )
        )

    def test_clear_and_no_are_only_no_rule_match_not_a_safety_claim(self) -> None:
        observation = self._completed(
            (
                self._result(
                    VisualPredicate.FACE_VISIBILITY,
                    VisualPredicateAnswer.CLEAR,
                ),
                self._result(
                    VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
                    VisualPredicateAnswer.NO,
                ),
            )
        )

        recommendation = assess_visual_predicate_review(observation)

        self.assertIs(VisualPredicateReviewState.NO_RULE_MATCH, recommendation.state)
        self.assertEqual((), recommendation.reasons)
        self.assertFalse(recommendation.manual_review_recommended)
        public = recommendation.to_dict()
        self.assertFalse(public["model_confidence_calibrated"])
        self.assertFalse(public["risk_level_assigned"])
        self.assertFalse(public["notification_enqueued"])

    def test_visible_review_reasons_take_precedence_over_unknown(self) -> None:
        observation = self._completed(
            (
                self._result(
                    VisualPredicate.FACE_VISIBILITY,
                    VisualPredicateAnswer.PARTIALLY_OBSCURED,
                ),
                self._result(
                    VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
                    VisualPredicateAnswer.UNKNOWN,
                ),
            )
        )

        recommendation = assess_visual_predicate_review(observation)

        self.assertIs(VisualPredicateReviewState.REVIEW_RECOMMENDED, recommendation.state)
        self.assertEqual(
            (
                VisualPredicateReviewReason.FACE_PARTIALLY_OBSCURED,
                VisualPredicateReviewReason.COVERING_NEAR_FACE_UNKNOWN,
            ),
            recommendation.reasons,
        )
        self.assertTrue(recommendation.manual_review_recommended)

    def test_missing_or_unknown_required_predicate_remains_unknown(self) -> None:
        observation = self._completed(
            (
                self._result(
                    VisualPredicate.FACE_VISIBILITY,
                    VisualPredicateAnswer.UNKNOWN,
                ),
            )
        )

        recommendation = assess_visual_predicate_review(observation)

        self.assertIs(VisualPredicateReviewState.UNKNOWN, recommendation.state)
        self.assertEqual(
            (
                VisualPredicateReviewReason.FACE_VISIBILITY_UNKNOWN,
                VisualPredicateReviewReason.REQUIRED_COVERING_NEAR_FACE_MISSING,
            ),
            recommendation.reasons,
        )
        self.assertTrue(recommendation.manual_review_recommended)

    def test_noncompleted_or_unbound_observation_fails_closed(self) -> None:
        with self.assertRaisesRegex(VisualPredicateReviewError, "bound completed"):
            assess_visual_predicate_review(self._pending())

    def test_report_counts_completed_and_unassessed_without_media(self) -> None:
        review = self._completed(
            (
                self._result(
                    VisualPredicate.FACE_VISIBILITY,
                    VisualPredicateAnswer.NOT_VISIBLE,
                ),
                self._result(
                    VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
                    VisualPredicateAnswer.YES,
                ),
            )
        )
        report = build_visual_predicate_review_report((self._pending(), review))
        payload = report.to_dict()

        self.assertEqual(2, payload["observed_attempt_count"])
        self.assertEqual(1, payload["assessed_completed_count"])
        self.assertEqual(1, payload["review_recommended_count"])
        self.assertEqual(1, payload["unassessed_counts"]["pending"])
        self.assertFalse(payload["media_read"])
        self.assertFalse(payload["risk_inference"])
        self.assertFalse(payload["notification_decision"])
        schema = json.loads(
            Path("schemas/visual-predicate-review-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(payload))

    def test_cli_reads_digest_bound_results_without_media_or_database_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database = root / "events.sqlite3"
            sqlite3.connect(database).close()
            results = (
                self._result(
                    VisualPredicate.FACE_VISIBILITY,
                    VisualPredicateAnswer.NOT_VISIBLE,
                ),
                self._result(
                    VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
                    VisualPredicateAnswer.NO,
                ),
            )
            gate = self._gate(tuple(item.predicate for item in results))
            queue = VisualPredicateShadowAttemptQueue(database)
            queue.enqueue(gate, max_active=4, now=self.NOW)
            claimed = queue.claim_next(lease_seconds=60, max_attempts=2, now=self.NOW)
            self.assertIsNotNone(claimed)
            VisualPredicateShadowStore(database).persist(
                VisualPredicateShadowRun(
                    gate=gate,
                    results=results,
                    recorded_at=self.NOW,
                )
            )
            queue.mark_completed(gate.event_id, now=self.NOW)
            before = database.stat().st_mtime_ns

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "visual-predicate-review-report",
                        "--database",
                        str(database),
                        "--limit",
                        "10",
                    ]
                )

            payload = json.loads(output.getvalue())
            self.assertEqual(0, code)
            self.assertEqual(1, payload["review_recommended_count"])
            self.assertEqual("review_recommended", payload["assessments"][0]["state"])
            self.assertFalse(payload["risk_level_assigned"])
            self.assertFalse(payload["notification_enqueued"])
            self.assertFalse(payload["media_read"])
            self.assertNotIn(str(root), output.getvalue())
            self.assertEqual(before, database.stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
