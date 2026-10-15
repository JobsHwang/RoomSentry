from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.detector import StaticDetector
from baby_monitor.adapters.media import FixtureMediaProcessor
from baby_monitor.adapters.vlm import MockVLMAdapter, VLMAdapter, VLMError
from baby_monitor.config import StorageSettings
from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    Detection,
    EventMedia,
    EventType,
    EvidenceGap,
    FinalSource,
    ReviewTier,
    RiskLevel,
    SignalEvent,
    SignalType,
    Subject,
    SubjectKind,
    VLMEvent,
)
from baby_monitor.pipeline import (
    PipelineRunner,
    enforce_subject_identity_boundary,
    enforce_temporal_fact_boundary,
    process_prepared_event,
)
from baby_monitor.policy import ReviewPolicy
from baby_monitor.recognition_attempt import (
    RecognitionAttemptAuditStore,
    RecognitionAttemptStatus,
    RecognitionStage,
)
from baby_monitor.reporting import DailyReportGenerator
from baby_monitor.storage import StorageResolver


class StaticResultVLM(VLMAdapter):
    def __init__(
        self,
        *,
        name: str,
        event_type: EventType,
        confidence: float,
        call_order: list[str] | None = None,
    ):
        self.name = name
        self.event_type = event_type
        self.confidence = confidence
        self.call_order = call_order
        self.calls = 0

    def analyze(self, request):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.call_order is not None:
            self.call_order.append(self.name)
        return VLMEvent(
            event_type=self.event_type,
            subjects=(),
            observations=(f"{self.name} 记录到测试事实。",),
            inferences=(),
            risk_level=RiskLevel.NONE,
            confidence=self.confidence,
            needs_remote_review=False,
            summary_zh=f"{self.name} 测试结果。",
            evidence_frames=tuple(request.media.frame_paths),
        )


class PipelineIntegrationTests(unittest.TestCase):
    def test_temporal_action_claims_are_downgraded_without_motion_facts(self) -> None:
        candidate = CandidateEvent("static_frames", "cam_01", 0.0, 3.0, ())
        model_result = VLMEvent(
            event_type=EventType.MOVEMENT,
            subjects=(),
            observations=(
                "a blue bus is present on the street",
                "people are walking around the bus",
            ),
            inferences=(),
            risk_level=RiskLevel.NONE,
            confidence=0.91,
            needs_remote_review=False,
            summary_zh="",
            evidence_frames=("frame.jpg",),
        )

        bounded = enforce_temporal_fact_boundary(model_result, candidate)

        self.assertEqual(EventType.UNKNOWN, bounded.event_type)
        self.assertEqual(("a blue bus is present on the street",), bounded.observations)
        self.assertEqual(
            ("时序证据不足，模型推测：people are walking around the bus",),
            bounded.inferences,
        )
        self.assertEqual(0.49, bounded.confidence)
        self.assertTrue(bounded.needs_remote_review)

    def test_generic_person_cannot_be_promoted_to_adult_or_baby(self) -> None:
        candidate = CandidateEvent(
            "generic_person",
            "cam_01",
            0.0,
            1.0,
            (Detection(0.5, "person", 0.9, (0.1, 0.1, 0.9, 0.9)),),
        )
        model_result = VLMEvent(
            event_type=EventType.ADULT_INTERACTION,
            subjects=(
                Subject(SubjectKind.ADULT, 0.92, ("frame.jpg",)),
                Subject(SubjectKind.BABY, 0.87, ("frame.jpg",)),
            ),
            observations=("an adult is standing beside a baby", "a bus is visible"),
            inferences=(),
            risk_level=RiskLevel.MEDIUM,
            confidence=0.93,
            needs_remote_review=False,
            summary_zh="",
            evidence_frames=("frame.jpg",),
        )

        bounded = enforce_subject_identity_boundary(model_result, candidate)

        self.assertEqual(EventType.UNKNOWN, bounded.event_type)
        self.assertEqual(
            (SubjectKind.PERSON, SubjectKind.PERSON),
            tuple(subject.kind for subject in bounded.subjects),
        )
        self.assertEqual((0.49, 0.49), tuple(s.confidence for s in bounded.subjects))
        self.assertEqual(("a bus is visible",), bounded.observations)
        self.assertIn(
            "身份证据不足，模型推测：模型将主体标记为 adult。",
            bounded.inferences,
        )
        self.assertIn(
            "身份证据不足，模型推测：模型将主体标记为 baby。",
            bounded.inferences,
        )
        self.assertIn(
            "身份证据不足，模型推测：an adult is standing beside a baby",
            bounded.inferences,
        )
        self.assertIn(
            "身份证据不足，模型推测：事件被标记为 adult_interaction。",
            bounded.inferences,
        )
        self.assertEqual(RiskLevel.MEDIUM, bounded.risk_level)
        self.assertEqual(0.49, bounded.confidence)
        self.assertTrue(bounded.needs_remote_review)

    def test_identity_claim_without_human_candidate_fact_becomes_unknown(self) -> None:
        candidate = CandidateEvent("no_human_fact", "cam_01", 0.0, 1.0, ())
        model_result = VLMEvent(
            event_type=EventType.UNKNOWN,
            subjects=(Subject(SubjectKind.ADULT, 0.8, ("frame.jpg",)),),
            observations=(),
            inferences=(),
            risk_level=RiskLevel.NONE,
            confidence=0.8,
            needs_remote_review=False,
            summary_zh="",
            evidence_frames=("frame.jpg",),
        )

        bounded = enforce_subject_identity_boundary(model_result, candidate)

        self.assertEqual(SubjectKind.UNKNOWN, bounded.subjects[0].kind)
        self.assertEqual(0.49, bounded.subjects[0].confidence)
        self.assertTrue(bounded.needs_remote_review)

    def test_identity_disclaimer_remains_an_observation(self) -> None:
        candidate = CandidateEvent(
            "generic_person_disclaimer",
            "cam_01",
            0.0,
            1.0,
            (Detection(0.5, "person", 0.9, (0.1, 0.1, 0.9, 0.9)),),
        )
        observation = "系统记录到 person；这不是宝宝/成人身份结论。"
        model_result = VLMEvent(
            event_type=EventType.UNKNOWN,
            subjects=(Subject(SubjectKind.PERSON, 0.8, ("frame.jpg",)),),
            observations=(observation,),
            inferences=(),
            risk_level=RiskLevel.NONE,
            confidence=0.8,
            needs_remote_review=False,
            summary_zh="",
            evidence_frames=("frame.jpg",),
        )

        bounded = enforce_subject_identity_boundary(model_result, candidate)

        self.assertIs(model_result, bounded)
        self.assertEqual((observation,), bounded.observations)

    def test_explicit_reviewed_identity_label_is_preserved(self) -> None:
        candidate = CandidateEvent(
            "explicit_adult",
            "cam_01",
            0.0,
            1.0,
            (Detection(0.5, "adult", 0.9, (0.1, 0.1, 0.9, 0.9)),),
        )
        model_result = VLMEvent(
            event_type=EventType.ADULT_INTERACTION,
            subjects=(Subject(SubjectKind.ADULT, 0.88, ("frame.jpg",)),),
            observations=("an adult is visible",),
            inferences=(),
            risk_level=RiskLevel.LOW,
            confidence=0.88,
            needs_remote_review=False,
            summary_zh="",
            evidence_frames=("frame.jpg",),
        )

        bounded = enforce_subject_identity_boundary(model_result, candidate)

        self.assertIs(model_result, bounded)

    def test_remote_failure_persists_pending_local_result(self) -> None:
        class FailingRemote(VLMAdapter):
            name = "failing_remote"

            def analyze(self, request):  # type: ignore[no-untyped-def]
                raise VLMError("sanitized remote fixture failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame.jpg"
            frame.write_bytes(b"fixture")
            clip = root / "event.mp4"
            clip.write_bytes(b"fixture")
            database = EventDatabase(root / "events.sqlite3")
            candidate = CandidateEvent("remote_failure", "cam_01", 0.0, 1.0, ())
            summary = process_prepared_event(
                candidate=candidate,
                occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
                source_kind="fixture",
                source_ref="fixture",
                media=EventMedia(str(clip), (str(frame),)),
                local_vlm=MockVLMAdapter(),
                remote_vlm=FailingRemote(),
                remote_uploads_allowed=True,
                database=database,
                policy=ReviewPolicy(),
            )

            self.assertFalse(summary.remote_reviewed)
            self.assertTrue(summary.pending_remote_review)
            stored = database.get_event(candidate.event_id)
            assert stored is not None
            self.assertIsNone(stored.remote_result)
            self.assertFalse(stored.remote_used)
            self.assertTrue(stored.result.needs_remote_review)
            self.assertEqual(stored.local_result.event_type, stored.result.event_type)
            self.assertEqual(FinalSource.LOCAL_PRIMARY, stored.decision.final_source)
            self.assertTrue(stored.decision.remote_allowed)
            self.assertIn("remote_review_failed", stored.decision.review_reasons)
            attempts = RecognitionAttemptAuditStore(database.path).recent()
            self.assertEqual(
                [RecognitionStage.REMOTE, RecognitionStage.LOCAL_PRIMARY],
                [attempt.stage for attempt in attempts],
            )
            self.assertEqual(RecognitionAttemptStatus.FAILED, attempts[0].status)
            self.assertEqual("vlm_error", attempts[0].error_code)

    def test_coverage_gap_escalates_review_without_inventing_risk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            started = datetime(2026, 8, 3, tzinfo=UTC)
            gap = EvidenceGap(
                started_at=started,
                ended_at=started + timedelta(seconds=2),
                reason="segment_missing",
            )
            media = EventMedia(
                clip_path=str(root / "event.mp4"),
                frame_paths=(str(root / "frame.jpg"),),
                coverage_gaps=(gap,),
                selected_segments=1,
            )
            database = EventDatabase(root / "events.sqlite3")
            candidate = CandidateEvent("coverage_gap", "cam_01", 0.0, 1.0, ())
            summary = process_prepared_event(
                candidate=candidate,
                occurred_at=started,
                source_kind="rtsp",
                source_ref="rtsp://127.0.0.1:8554/test",
                media=media,
                local_vlm=MockVLMAdapter(),
                database=database,
                policy=ReviewPolicy(confidence_threshold=0.0),
            )
            event = database.get_event(candidate.event_id)
            assert event is not None
            self.assertFalse(summary.pending_remote_review)
            self.assertEqual(ReviewTier.LOCAL_SECONDARY, event.decision.review_tier)
            self.assertIn("evidence_coverage_gap", event.decision.review_reasons)
            self.assertEqual(RiskLevel.NONE, event.result.risk_level)
            self.assertEqual(EventType.UNKNOWN, event.result.event_type)
            self.assertEqual(media, event.evidence)
            self.assertTrue(any("覆盖缺口" in value for value in event.result.observations))

            report = DailyReportGenerator(
                database=database,
                storage=StorageResolver(StorageSettings(local_root=root, database=database.path)),
                timezone_name="Asia/Shanghai",
            ).generate(date(2026, 8, 3))
            markdown = report.markdown_path.read_text(encoding="utf-8")
            self.assertIn("已记录证据缺口事件：1", markdown)
            self.assertIn(r"segment\_missing", markdown)

    def test_secondary_runs_sequentially_and_can_resolve_low_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            call_order: list[str] = []
            primary = StaticResultVLM(
                name="primary",
                event_type=EventType.UNKNOWN,
                confidence=0.2,
                call_order=call_order,
            )
            secondary = StaticResultVLM(
                name="secondary",
                event_type=EventType.UNKNOWN,
                confidence=0.9,
                call_order=call_order,
            )
            database = EventDatabase(root / "events.sqlite3")
            candidate = CandidateEvent("secondary_ok", "cam_01", 0.0, 1.0, ())
            media = EventMedia(str(root / "event.mp4"), (str(root / "frame.jpg"),))
            summary = process_prepared_event(
                candidate=candidate,
                occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
                source_kind="file",
                source_ref="fixture.mp4",
                media=media,
                local_vlm=primary,
                secondary_vlm=secondary,
                database=database,
                policy=ReviewPolicy(),
            )
            event = database.get_event(candidate.event_id)
            assert event is not None

            self.assertEqual(1, primary.calls)
            self.assertEqual(1, secondary.calls)
            self.assertEqual(["primary", "secondary"], call_order)
            self.assertTrue(summary.secondary_reviewed)
            self.assertFalse(summary.pending_remote_review)
            self.assertEqual(secondary.confidence, event.result.confidence)
            self.assertIsNotNone(event.secondary_result)
            self.assertEqual(ReviewTier.LOCAL_SECONDARY, event.decision.review_tier)
            self.assertEqual(FinalSource.LOCAL_SECONDARY, event.decision.final_source)
            self.assertFalse(event.decision.remote_used)
            self.assertIn("confidence_gap_ge_0.4", event.decision.contradictions)
            self.assertNotIn("contradictory_results", event.decision.review_reasons)
            attempts = RecognitionAttemptAuditStore(database.path).recent()
            self.assertEqual(2, len(attempts))
            self.assertEqual(
                {RecognitionStage.LOCAL_PRIMARY, RecognitionStage.LOCAL_SECONDARY},
                {attempt.stage for attempt in attempts},
            )
            self.assertTrue(
                all(attempt.status is RecognitionAttemptStatus.SUCCEEDED for attempt in attempts)
            )

    def test_secondary_disagreement_is_recorded_without_remote_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = StaticResultVLM(
                name="primary",
                event_type=EventType.AWAKE,
                confidence=0.2,
            )
            secondary = StaticResultVLM(
                name="secondary",
                event_type=EventType.SLEEPING,
                confidence=0.9,
            )
            database = EventDatabase(root / "events.sqlite3")
            candidate = CandidateEvent("secondary_conflict", "cam_01", 0.0, 1.0, ())
            media = EventMedia(str(root / "event.mp4"), (str(root / "frame.jpg"),))
            summary = process_prepared_event(
                candidate=candidate,
                occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
                source_kind="file",
                source_ref="fixture.mp4",
                media=media,
                local_vlm=primary,
                secondary_vlm=secondary,
                database=database,
                policy=ReviewPolicy(),
                remote_uploads_allowed=False,
            )
            event = database.get_event(candidate.event_id)
            assert event is not None

            self.assertTrue(summary.secondary_reviewed)
            self.assertTrue(summary.pending_remote_review)
            self.assertEqual(ReviewTier.REMOTE, event.decision.review_tier)
            self.assertEqual(FinalSource.LOCAL_SECONDARY, event.decision.final_source)
            self.assertIn("event_type_disagreement", event.decision.contradictions)
            self.assertFalse(event.decision.remote_allowed)
            self.assertFalse(event.remote_used)

    def test_repeated_signals_raise_review_tier_without_rewriting_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = EventDatabase(root / "events.sqlite3")
            policy = ReviewPolicy(
                confidence_threshold=0.0,
                signal_frequency_window_seconds=600,
                signal_frequency_secondary_threshold=3,
            )
            started = datetime(2026, 8, 3, tzinfo=UTC)
            stored = []
            for index in range(3):
                occurred_at = started + timedelta(minutes=index)
                signal = SignalEvent(
                    signal_id=f"motion_{index}",
                    camera_id="cam_01",
                    occurred_at=occurred_at,
                    signal_type=SignalType.MOTION,
                    source_adapter="fixture",
                    confidence=0.9,
                )
                candidate = CandidateEvent(
                    event_id=f"signal_event_{index}",
                    camera_id="cam_01",
                    start_seconds=0.0,
                    end_seconds=1.0,
                    detections=(),
                    signals=(signal,),
                )
                media = EventMedia(
                    clip_path=str(root / f"event_{index}.mp4"),
                    frame_paths=(str(root / f"frame_{index}.jpg"),),
                )
                summary = process_prepared_event(
                    candidate=candidate,
                    occurred_at=occurred_at,
                    source_kind="rtsp",
                    source_ref="rtsp://127.0.0.1:8554/test",
                    media=media,
                    local_vlm=MockVLMAdapter(),
                    database=database,
                    policy=policy,
                )
                self.assertFalse(summary.pending_remote_review)
                event = database.get_event(candidate.event_id)
                assert event is not None
                stored.append(event)

            self.assertEqual(ReviewTier.LOCAL_SECONDARY, stored[0].decision.review_tier)
            self.assertIn("evidence_coverage_unverified", stored[0].decision.review_reasons)
            self.assertNotIn("repeated_signal_window", stored[0].decision.review_reasons)
            self.assertEqual(1, stored[0].decision.signal_frequency_count)
            self.assertEqual(ReviewTier.LOCAL_SECONDARY, stored[1].decision.review_tier)
            self.assertNotIn("repeated_signal_window", stored[1].decision.review_reasons)
            self.assertEqual(2, stored[1].decision.signal_frequency_count)
            self.assertEqual(ReviewTier.LOCAL_SECONDARY, stored[2].decision.review_tier)
            self.assertEqual(3, stored[2].decision.signal_frequency_count)
            self.assertEqual((SignalType.MOTION,), stored[2].decision.frequent_signal_types)
            self.assertEqual(RiskLevel.NONE, stored[2].result.risk_level)
            self.assertEqual(EventType.UNKNOWN, stored[2].result.event_type)
            self.assertTrue(any("motion" in value for value in stored[2].result.observations))
            self.assertFalse(stored[2].remote_used)

            storage = StorageResolver(StorageSettings(local_root=root, database=database.path))
            report = DailyReportGenerator(
                database=database,
                storage=storage,
                timezone_name="Asia/Shanghai",
            ).generate(date(2026, 8, 3))
            markdown = report.markdown_path.read_text(encoding="utf-8")
            html = report.html_path.read_text(encoding="utf-8")
            self.assertIn("local_secondary", markdown)
            self.assertIn("repeated_signal_window", markdown)
            self.assertIn("决策与复核", html)

    def test_prepared_event_rejects_result_without_evidence_binding(self) -> None:
        class MissingEvidenceVLM(MockVLMAdapter):
            def analyze(self, request):  # type: ignore[no-untyped-def]
                result = super().analyze(request).to_dict()
                result["evidence_frames"] = []
                for subject in result["subjects"]:
                    subject["evidence_frames"] = []
                return VLMEvent.from_dict(result)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = CandidateEvent(
                event_id="missing_evidence",
                camera_id="cam_01",
                start_seconds=0.0,
                end_seconds=1.0,
                detections=(Detection(0.5, "person", 0.8, (0, 0, 1, 1)),),
            )
            media = EventMedia(
                clip_path=str(root / "event.mp4"),
                frame_paths=(str(root / "frame_01.jpg"),),
            )
            with self.assertRaisesRegex(VLMError, "must cite"):
                process_prepared_event(
                    candidate=candidate,
                    occurred_at=datetime(2026, 8, 2, tzinfo=UTC),
                    source_kind="file",
                    source_ref="fixture.mp4",
                    media=media,
                    local_vlm=MissingEvidenceVLM(),
                    database=EventDatabase(root / "events.sqlite3"),
                    policy=ReviewPolicy(),
                )

    def test_prepared_event_canonicalizes_unique_evidence_basenames(self) -> None:
        class BasenameEvidenceVLM(MockVLMAdapter):
            def analyze(self, request):  # type: ignore[no-untyped-def]
                result = super().analyze(request).to_dict()
                result["evidence_frames"] = [Path(path).name for path in result["evidence_frames"]]
                for subject in result["subjects"]:
                    subject["evidence_frames"] = [
                        Path(path).name for path in subject["evidence_frames"]
                    ]
                return VLMEvent.from_dict(result)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = tuple(str(root / f"frame_{index}.jpg") for index in range(2))
            for frame in frames:
                Path(frame).touch()
            candidate = CandidateEvent(
                event_id="basename_evidence",
                camera_id="cam_01",
                start_seconds=0.0,
                end_seconds=1.0,
                detections=(Detection(0.5, "person", 0.8, (0, 0, 1, 1)),),
            )
            database = EventDatabase(root / "events.sqlite3")
            process_prepared_event(
                candidate=candidate,
                occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
                source_kind="file",
                source_ref="fixture.mp4",
                media=EventMedia(str(root / "event.mp4"), frames),
                local_vlm=BasenameEvidenceVLM(),
                database=database,
                policy=ReviewPolicy(),
            )

            event = database.get_event(candidate.event_id)
            assert event is not None
            self.assertEqual(frames, event.local_result.evidence_frames)
            self.assertEqual(frames, event.result.evidence_frames)
            self.assertEqual((frames[0],), event.result.subjects[0].evidence_frames)
            self.assertTrue(all(Path(path).is_file() for path in event.result.evidence_frames))

    def test_prepared_event_rejects_ambiguous_evidence_basename(self) -> None:
        class BasenameEvidenceVLM(MockVLMAdapter):
            def analyze(self, request):  # type: ignore[no-untyped-def]
                result = super().analyze(request).to_dict()
                result["evidence_frames"] = ["frame.jpg"]
                return VLMEvent.from_dict(result)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = CandidateEvent("ambiguous_evidence", "cam_01", 0.0, 1.0, ())
            media = EventMedia(
                str(root / "event.mp4"),
                (str(root / "one" / "frame.jpg"), str(root / "two" / "frame.jpg")),
            )
            with self.assertRaisesRegex(VLMError, "ambiguous"):
                process_prepared_event(
                    candidate=candidate,
                    occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
                    source_kind="file",
                    source_ref="fixture.mp4",
                    media=media,
                    local_vlm=BasenameEvidenceVLM(),
                    database=EventDatabase(root / "events.sqlite3"),
                    policy=ReviewPolicy(),
                )

    def test_report_resolves_legacy_basename_through_event_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "media" / "frame.jpg"
            frame.parent.mkdir()
            frame.touch()
            database = EventDatabase(root / "events.sqlite3")
            candidate = CandidateEvent("legacy_basename", "cam_01", 0.0, 1.0, ())
            media = EventMedia(str(root / "media" / "event.mp4"), (str(frame),))
            process_prepared_event(
                candidate=candidate,
                occurred_at=datetime(2026, 8, 3, tzinfo=UTC),
                source_kind="file",
                source_ref="fixture.mp4",
                media=media,
                local_vlm=MockVLMAdapter(),
                database=database,
                policy=ReviewPolicy(),
            )
            event = database.get_event(candidate.event_id)
            assert event is not None
            legacy_result = event.result.to_dict()
            legacy_result["evidence_frames"] = [frame.name]
            with closing(sqlite3.connect(database.path)) as connection, connection:
                connection.execute(
                    "UPDATE events SET result_json = ?, evidence_frames_json = ? "
                    "WHERE event_id = ?",
                    (
                        json.dumps(legacy_result, ensure_ascii=False),
                        json.dumps([frame.name]),
                        candidate.event_id,
                    ),
                )

            report = DailyReportGenerator(
                database=database,
                storage=StorageResolver(
                    StorageSettings(
                        local_root=root / "storage",
                        database=database.path,
                    )
                ),
                timezone_name="Asia/Shanghai",
            ).generate(date(2026, 8, 3))
            expected = os.path.relpath(frame, report.markdown_path.parent).replace(os.sep, "/")
            markdown = report.markdown_path.read_text(encoding="utf-8")
            html = report.html_path.read_text(encoding="utf-8")
            self.assertIn(f"]({expected})", markdown)
            self.assertIn(f'src="{expected}"', html)
            self.assertNotIn("../../../../../../../../frame.jpg", markdown)

    def test_fixture_pipeline_persists_json_and_generates_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "fixture.placeholder"
            source_path.write_text("not a real video", encoding="utf-8")
            database_path = root / "events.sqlite3"
            storage_settings = StorageSettings(local_root=root, database=database_path)
            database = EventDatabase(database_path)
            runner = PipelineRunner(
                camera=LocalFileCamera("nursery_cam_01", source_path),
                detector=StaticDetector(
                    [
                        Detection(2.0, "person", 0.80, (0.0, 0.0, 0.5, 1.0)),
                        Detection(3.0, "person", 0.85, (0.05, 0.0, 0.55, 1.0)),
                    ]
                ),
                media=FixtureMediaProcessor(frame_count=3),
                local_vlm=MockVLMAdapter(),
                database=database,
                storage=StorageResolver(storage_settings),
                policy=ReviewPolicy(confidence_threshold=0.65),
                candidate_gap_seconds=3.0,
                pre_event_seconds=5.0,
                post_event_seconds=7.0,
                remote_vlm=None,
                remote_uploads_allowed=False,
            )
            source_started_at = datetime(2026, 8, 1, 1, 0, tzinfo=UTC)
            summary = runner.run(source_started_at=source_started_at)
            self.assertEqual(2, summary.detections)
            self.assertEqual(1, summary.candidates)
            self.assertEqual(0, summary.remote_reviews)
            self.assertEqual(1, summary.pending_remote_reviews)

            events = database.fetch_between(
                source_started_at - timedelta(minutes=1),
                source_started_at + timedelta(minutes=1),
            )
            self.assertEqual(1, len(events))
            event = events[0]
            self.assertEqual(EventType.MOVEMENT, event.result.event_type)
            self.assertFalse(event.local_result.needs_remote_review)
            self.assertTrue(event.result.needs_remote_review)
            self.assertIsNone(event.remote_result)
            self.assertTrue(Path(event.clip_path).is_file())
            self.assertEqual(3, len(event.result.evidence_frames))
            self.assertTrue(all(Path(path).is_file() for path in event.result.evidence_frames))

            artifacts = DailyReportGenerator(
                database=database,
                storage=StorageResolver(storage_settings),
                timezone_name="Asia/Shanghai",
            ).generate(date(2026, 8, 1))
            markdown = artifacts.markdown_path.read_text(encoding="utf-8")
            html = artifacts.html_path.read_text(encoding="utf-8")
            self.assertEqual(1, artifacts.event_count)
            self.assertIn("记录中的可核实事实", markdown)
            self.assertIn("模型推测", markdown)
            self.assertIn("不能替代成人照护", html)

    def test_prepared_event_retry_reuses_database_without_rerunning_vlm(self) -> None:
        class CountingMockVLM(MockVLMAdapter):
            def __init__(self) -> None:
                self.calls = 0

            def analyze(self, request):  # type: ignore[no-untyped-def]
                self.calls += 1
                return super().analyze(request)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = EventDatabase(root / "events.sqlite3")
            candidate = CandidateEvent(
                event_id="buffer_event_01",
                camera_id="cam_01",
                start_seconds=0.0,
                end_seconds=12.0,
                detections=(),
            )
            media = EventMedia(
                clip_path=str(root / "event.mp4"),
                frame_paths=tuple(str(root / f"frame_{index}.jpg") for index in range(3)),
            )
            occurred_at = datetime(2026, 8, 2, 1, 2, 3, tzinfo=UTC)
            vlm = CountingMockVLM()
            kwargs = {
                "candidate": candidate,
                "occurred_at": occurred_at,
                "source_kind": "rtsp",
                "source_ref": "rtsp://127.0.0.1:8554/test",
                "media": media,
                "local_vlm": vlm,
                "database": database,
                "policy": ReviewPolicy(confidence_threshold=0.65),
            }
            first = process_prepared_event(**kwargs)
            with closing(sqlite3.connect(database.path)) as connection, connection:
                connection.execute(
                    "UPDATE events SET evidence_json = NULL WHERE event_id = ?",
                    (candidate.event_id,),
                )
            second = process_prepared_event(**kwargs)

            self.assertFalse(first.reused_existing)
            self.assertTrue(second.reused_existing)
            self.assertEqual(1, vlm.calls)
            self.assertEqual(1, len(database.list_recent()))
            legacy = database.get_event(candidate.event_id)
            assert legacy is not None
            self.assertFalse(legacy.evidence_metadata_present)

            with self.assertRaisesRegex(ValueError, "different immutable"):
                process_prepared_event(
                    **{
                        **kwargs,
                        "candidate": CandidateEvent(
                            event_id="buffer_event_01",
                            camera_id="different_camera",
                            start_seconds=0.0,
                            end_seconds=12.0,
                            detections=(),
                        ),
                    }
                )


if __name__ == "__main__":
    unittest.main()
