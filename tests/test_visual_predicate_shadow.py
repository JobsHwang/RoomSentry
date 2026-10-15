from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.adapters.visual_predicate import (
    VisualPredicateAdapter,
    VisualPredicateFrame,
    VisualPredicateRequest,
)
from baby_monitor.domain import (
    CandidateEvent,
    Detection,
    SignalEvent,
    SignalType,
)
from baby_monitor.visual_predicate import (
    VisualPredicate,
    VisualPredicateAnswer,
    VisualPredicateResult,
)
from baby_monitor.visual_predicate_shadow import (
    VisualPredicateShadowError,
    VisualPredicateShadowRun,
    VisualPredicateShadowStore,
    VisualPredicateShadowTrigger,
    run_visual_predicate_shadow,
    select_visual_predicate_shadow,
)

_REVISION = "5c78cab61938e73859b63724d9bf5cb88c477eaa"


class _RecordingAdapter(VisualPredicateAdapter):
    def __init__(
        self,
        *,
        fail_on: VisualPredicate | None = None,
        covering_answer: VisualPredicateAnswer = VisualPredicateAnswer.YES,
    ):
        self.requests: list[VisualPredicateRequest] = []
        self.fail_on = fail_on
        self.covering_answer = covering_answer

    def assess(self, request: VisualPredicateRequest) -> VisualPredicateResult:
        self.requests.append(request)
        if request.predicate is self.fail_on:
            raise RuntimeError("private adapter detail")
        answer = (
            VisualPredicateAnswer.CLEAR
            if request.predicate is VisualPredicate.FACE_VISIBILITY
            else self.covering_answer
        )
        return VisualPredicateResult(
            event_id=request.event_id,
            predicate=request.predicate,
            answer=answer,
            confidence=0.8,
            evidence_frames=(request.frames[0].frame_id,),
            model_id="microsoft.Mage-VL",
            model_revision=_REVISION,
            prompt_version="atomic_shadow_v1",
        )


class _UnboundAdapter(VisualPredicateAdapter):
    def assess(self, request: VisualPredicateRequest) -> VisualPredicateResult:
        return VisualPredicateResult(
            event_id=request.event_id,
            predicate=request.predicate,
            answer=VisualPredicateAnswer.CLEAR,
            confidence=0.8,
            evidence_frames=("unadmitted_frame",),
            model_id="microsoft.Mage-VL",
            model_revision=_REVISION,
            prompt_version="atomic_shadow_v1",
        )


class VisualPredicateShadowTests(unittest.TestCase):
    @staticmethod
    def _candidate(
        *,
        person_confidence: float | None = 0.8,
        with_signal: bool = True,
        signal_source_adapter: str = "home_assistant",
    ) -> CandidateEvent:
        detections = (
            (
                Detection(
                    timestamp_seconds=1.0,
                    label="person",
                    confidence=person_confidence,
                    bbox_xyxy=(0.1, 0.1, 0.8, 0.9),
                ),
            )
            if person_confidence is not None
            else ()
        )
        signals = (
            (
                SignalEvent(
                    signal_id="signal_01",
                    camera_id="camera_01",
                    occurred_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
                    signal_type=SignalType.MOTION,
                    source_adapter=signal_source_adapter,
                    confidence=1.0,
                ),
            )
            if with_signal
            else ()
        )
        return CandidateEvent(
            event_id="event_01",
            camera_id="camera_01",
            start_seconds=0.0,
            end_seconds=2.0,
            detections=detections,
            signals=signals,
        )

    @staticmethod
    def _frames() -> tuple[VisualPredicateFrame, ...]:
        return (
            VisualPredicateFrame(
                frame_id="frame_01",
                sha256="a" * 64,
                media_type="image/jpeg",
                path="/private/not-read.jpg",
            ),
        )

    @staticmethod
    def _database(root: Path) -> Path:
        path = root / "events.sqlite3"
        sqlite3.connect(path).close()
        return path

    def test_gate_unions_yolo_hardware_and_periodic_facts(self) -> None:
        predicates = (
            VisualPredicate.FACE_VISIBILITY,
            VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
        )
        gate = select_visual_predicate_shadow(
            self._candidate(),
            predicates=predicates,
            person_detector_id="yolo11n",
            periodic_sample_due=True,
            min_person_confidence=0.5,
        )
        self.assertTrue(gate.selected)
        self.assertEqual(
            (
                VisualPredicateShadowTrigger.PERSON_DETECTION,
                VisualPredicateShadowTrigger.HARDWARE_SIGNAL,
                VisualPredicateShadowTrigger.PERIODIC_SAMPLE,
            ),
            gate.triggers,
        )
        self.assertEqual(predicates, gate.predicates)
        self.assertEqual("yolo11n", gate.person_detector_id)
        self.assertEqual(1, gate.person_detection_count)
        self.assertEqual(("home_assistant",), gate.hardware_signal_adapters)
        self.assertEqual(1, gate.hardware_signal_count)
        payload = gate.to_dict()
        self.assertFalse(payload["subject_identity_inference"])
        self.assertFalse(payload["risk_inference"])
        self.assertFalse(payload["notification_decision"])

    def test_public_schema_required_fields_match_canonical_run(self) -> None:
        gate = select_visual_predicate_shadow(
            self._candidate(),
            predicates=(VisualPredicate.FACE_VISIBILITY,),
            person_detector_id="yolo11n",
        )
        frame = self._frames()[0]
        result = _RecordingAdapter().assess(
            VisualPredicateRequest(
                event_id=gate.event_id,
                predicate=VisualPredicate.FACE_VISIBILITY,
                frames=(frame,),
            )
        )
        run = VisualPredicateShadowRun(
            gate=gate,
            results=(result,),
            recorded_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
        )
        schema_path = (
            Path(__file__).parents[1] / "schemas" / "visual-predicate-shadow-run-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(set(schema["required"]), set(run.to_dict()))
        self.assertEqual(set(schema["properties"]["gate"]["required"]), set(gate.to_dict()))

    def test_gate_skips_without_qualifying_fact_or_predicate(self) -> None:
        no_fact = select_visual_predicate_shadow(
            self._candidate(person_confidence=0.2, with_signal=False),
            predicates=(VisualPredicate.FACE_VISIBILITY,),
            person_detector_id="yolo11n",
            min_person_confidence=0.5,
        )
        self.assertFalse(no_fact.selected)
        self.assertEqual((), no_fact.triggers)

        disabled = select_visual_predicate_shadow(
            self._candidate(),
            predicates=(),
            person_detector_id="yolo11n",
            periodic_sample_due=True,
        )
        self.assertFalse(disabled.selected)
        self.assertNotEqual((), disabled.triggers)

    def test_unprovenanced_detection_and_unadmitted_signal_do_not_trigger(self) -> None:
        gate = select_visual_predicate_shadow(
            self._candidate(signal_source_adapter="temporal_zone"),
            predicates=(VisualPredicate.FACE_VISIBILITY,),
        )
        self.assertFalse(gate.selected)
        self.assertEqual((), gate.triggers)
        self.assertEqual(0, gate.person_detection_count)
        self.assertEqual(0, gate.hardware_signal_count)

        detector_bound = select_visual_predicate_shadow(
            self._candidate(with_signal=False),
            predicates=(VisualPredicate.FACE_VISIBILITY,),
            person_detector_id="yolo11n",
        )
        self.assertEqual((VisualPredicateShadowTrigger.PERSON_DETECTION,), detector_bound.triggers)

    def test_all_predicates_finish_before_one_atomic_persist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(Path(directory))
            gate = select_visual_predicate_shadow(
                self._candidate(),
                predicates=(
                    VisualPredicate.FACE_VISIBILITY,
                    VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
                ),
            )
            adapter = _RecordingAdapter()
            factory_calls = 0

            def store_factory() -> VisualPredicateShadowStore:
                nonlocal factory_calls
                factory_calls += 1
                return VisualPredicateShadowStore(database)

            summary = run_visual_predicate_shadow(
                gate=gate,
                frames=self._frames(),
                adapter=adapter,
                store_factory=store_factory,
                recorded_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
            )
            self.assertEqual(1, factory_calls)
            self.assertTrue(summary.persisted)
            self.assertFalse(summary.reused_existing)
            self.assertEqual(gate.predicates, tuple(item.predicate for item in summary.results))

            connection = sqlite3.connect(database)
            row = connection.execute(
                "SELECT binding_json, recorded_at FROM visual_predicate_shadow_runs"
            ).fetchone()
            connection.close()
            self.assertIsNotNone(row)
            assert row is not None
            binding = json.loads(row[0])
            self.assertEqual("event_01", binding["gate"]["event_id"])
            self.assertNotIn("/private/not-read.jpg", row[0])
            self.assertFalse(binding["risk_inference"])
            self.assertFalse(binding["notification_decision"])
            self.assertEqual("2026-08-13T12:00:00+00:00", row[1])

            repeated = run_visual_predicate_shadow(
                gate=gate,
                frames=self._frames(),
                adapter=_RecordingAdapter(),
                store_factory=store_factory,
                recorded_at=datetime(2026, 8, 13, 13, 0, tzinfo=UTC),
            )
            self.assertTrue(repeated.reused_existing)

    def test_model_failure_never_opens_store_or_persists_partial_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(Path(directory))
            gate = select_visual_predicate_shadow(
                self._candidate(),
                predicates=(
                    VisualPredicate.FACE_VISIBILITY,
                    VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
                ),
            )
            factory_calls = 0

            def store_factory() -> VisualPredicateShadowStore:
                nonlocal factory_calls
                factory_calls += 1
                return VisualPredicateShadowStore(database)

            with self.assertRaisesRegex(
                VisualPredicateShadowError,
                "^visual predicate shadow assessment failed$",
            ):
                run_visual_predicate_shadow(
                    gate=gate,
                    frames=self._frames(),
                    adapter=_RecordingAdapter(fail_on=VisualPredicate.COVERING_NEAR_FACE_VISIBLE),
                    store_factory=store_factory,
                )
            self.assertEqual(0, factory_calls)
            connection = sqlite3.connect(database)
            table_count = connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'visual_predicate_shadow_runs'
                """
            ).fetchone()[0]
            connection.close()
            self.assertEqual(0, table_count)

    def test_conflicting_retry_fails_closed_and_preserves_first_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(Path(directory))
            gate = select_visual_predicate_shadow(
                self._candidate(),
                predicates=(VisualPredicate.COVERING_NEAR_FACE_VISIBLE,),
            )

            def factory() -> VisualPredicateShadowStore:
                return VisualPredicateShadowStore(database)

            first = run_visual_predicate_shadow(
                gate=gate,
                frames=self._frames(),
                adapter=_RecordingAdapter(covering_answer=VisualPredicateAnswer.YES),
                store_factory=factory,
            )
            self.assertFalse(first.reused_existing)
            with self.assertRaisesRegex(VisualPredicateShadowError, "conflicts"):
                run_visual_predicate_shadow(
                    gate=gate,
                    frames=self._frames(),
                    adapter=_RecordingAdapter(covering_answer=VisualPredicateAnswer.NO),
                    store_factory=factory,
                )
            connection = sqlite3.connect(database)
            binding = json.loads(
                connection.execute(
                    "SELECT binding_json FROM visual_predicate_shadow_runs"
                ).fetchone()[0]
            )
            connection.close()
            self.assertEqual("yes", binding["results"][0]["answer"])

    def test_store_never_recreates_a_database_removed_after_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(Path(directory))
            store = VisualPredicateShadowStore(database)
            gate = select_visual_predicate_shadow(
                self._candidate(),
                predicates=(VisualPredicate.FACE_VISIBILITY,),
            )
            frame = self._frames()[0]
            result = _RecordingAdapter().assess(
                VisualPredicateRequest(
                    event_id=gate.event_id,
                    predicate=VisualPredicate.FACE_VISIBILITY,
                    frames=(frame,),
                )
            )
            run = VisualPredicateShadowRun(
                gate=gate,
                results=(result,),
                recorded_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
            )
            database.unlink()

            with self.assertRaisesRegex(
                VisualPredicateShadowError,
                "^visual predicate shadow database is unavailable$",
            ):
                store.persist(run)
            self.assertFalse(database.exists())

    def test_unselected_gate_never_calls_adapter_or_store(self) -> None:
        gate = select_visual_predicate_shadow(
            self._candidate(person_confidence=None, with_signal=False),
            predicates=(VisualPredicate.FACE_VISIBILITY,),
        )
        adapter = _RecordingAdapter()
        factory_calls = 0

        def store_factory() -> VisualPredicateShadowStore:
            nonlocal factory_calls
            factory_calls += 1
            raise AssertionError("store must remain unopened")

        summary = run_visual_predicate_shadow(
            gate=gate,
            frames=self._frames(),
            adapter=adapter,
            store_factory=store_factory,
        )
        self.assertFalse(summary.persisted)
        self.assertEqual([], adapter.requests)
        self.assertEqual(0, factory_calls)

    def test_store_factory_error_is_fixed_and_path_free(self) -> None:
        gate = select_visual_predicate_shadow(
            self._candidate(),
            predicates=(VisualPredicate.FACE_VISIBILITY,),
        )

        def store_factory() -> VisualPredicateShadowStore:
            raise OSError("/private/family/path")

        with self.assertRaisesRegex(
            VisualPredicateShadowError,
            "^visual predicate shadow database is unavailable$",
        ):
            run_visual_predicate_shadow(
                gate=gate,
                frames=self._frames(),
                adapter=_RecordingAdapter(),
                store_factory=store_factory,
            )

    def test_gate_rejects_unsafe_event_id(self) -> None:
        candidate = self._candidate()
        object.__setattr__(candidate, "event_id", "../private-event")
        with self.assertRaises(VisualPredicateShadowError):
            select_visual_predicate_shadow(
                candidate,
                predicates=(VisualPredicate.FACE_VISIBILITY,),
            )

    def test_unbound_adapter_result_never_opens_store(self) -> None:
        gate = select_visual_predicate_shadow(
            self._candidate(),
            predicates=(VisualPredicate.FACE_VISIBILITY,),
        )
        factory_calls = 0

        def store_factory() -> VisualPredicateShadowStore:
            nonlocal factory_calls
            factory_calls += 1
            raise AssertionError("store must remain unopened")

        with self.assertRaisesRegex(
            VisualPredicateShadowError,
            "^visual predicate shadow assessment failed$",
        ):
            run_visual_predicate_shadow(
                gate=gate,
                frames=self._frames(),
                adapter=_UnboundAdapter(),
                store_factory=store_factory,
            )
        self.assertEqual(0, factory_calls)


if __name__ == "__main__":
    unittest.main()
