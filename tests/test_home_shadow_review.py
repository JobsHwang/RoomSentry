from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from baby_monitor.adapters.visual_predicate import (
    ConservativeVisualPredicateAdapter,
    VisualPredicateFrame,
)
from baby_monitor.analysis_queue import AnalysisQueue
from baby_monitor.cli import main
from baby_monitor.domain import CandidateEvent, Detection, EventMedia
from baby_monitor.home_shadow_review import (
    HomeShadowReviewError,
    HomeShadowReviewRepository,
)
from baby_monitor.visual_predicate import (
    VisualPredicate,
    VisualPredicateAnswer,
    VisualPredicateResult,
)
from baby_monitor.visual_predicate_shadow import (
    VisualPredicateShadowRun,
    VisualPredicateShadowStore,
    run_visual_predicate_shadow,
    select_visual_predicate_shadow,
)
from baby_monitor.visual_predicate_shadow_queue import (
    VisualPredicateShadowAttemptQueue,
)

_REVISION = "5c78cab61938e73859b63724d9bf5cb88c477eaa"
_SECRET = bytes(range(32))


class HomeShadowReviewRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o700)
        self.state = self.root / "state"
        self.state.mkdir(mode=0o700)
        self.media_root = self.root / "spool"
        self.media_root.mkdir(mode=0o700)
        event_root = self.media_root / "opaque-event"
        event_root.mkdir(mode=0o700)
        self.frame = event_root / "frame.jpg"
        self.frame.write_bytes(b"private-frame-payload")
        self.frame.chmod(0o600)
        self.clip = event_root / "clip.mp4"
        self.clip.write_bytes(b"private-clip-payload")
        self.clip.chmod(0o600)
        self.database = self.state / "events.sqlite3"
        candidate = CandidateEvent(
            event_id="event_private_01",
            camera_id="camera_private_01",
            start_seconds=0.0,
            end_seconds=2.0,
            detections=(
                Detection(
                    timestamp_seconds=1.0,
                    label="person",
                    confidence=0.8,
                    bbox_xyxy=(0.1, 0.1, 0.8, 0.9),
                ),
            ),
            detector_id="yolo11n.sha256_fixture",
        )
        AnalysisQueue(self.database).enqueue(
            candidate=candidate,
            occurred_at=datetime(2026, 8, 19, tzinfo=UTC),
            source_kind="file",
            source_ref="private-fixture.mp4",
            media=EventMedia(str(self.clip), (str(self.frame),)),
            storage_root=self.media_root,
            max_active=2,
        )
        gate = select_visual_predicate_shadow(
            candidate,
            predicates=(
                VisualPredicate.FACE_VISIBILITY,
                VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
            ),
            person_detector_id=candidate.detector_id,
        )
        attempts = VisualPredicateShadowAttemptQueue(self.database)
        attempts.enqueue(gate, max_active=2, now=datetime(2026, 8, 19, tzinfo=UTC))
        claimed = attempts.claim_next(
            lease_seconds=30,
            max_attempts=2,
            now=datetime(2026, 8, 19, tzinfo=UTC),
        )
        assert claimed is not None
        run_visual_predicate_shadow(
            gate=gate,
            frames=(
                VisualPredicateFrame(
                    frame_id="frame_01",
                    sha256=hashlib.sha256(self.frame.read_bytes()).hexdigest(),
                    media_type="image/jpeg",
                    path=str(self.frame),
                ),
            ),
            adapter=ConservativeVisualPredicateAdapter(
                model_id="microsoft.Mage-VL",
                model_revision=_REVISION,
                prompt_version="atomic_shadow_v1",
            ),
            store_factory=lambda: VisualPredicateShadowStore(self.database),
            recorded_at=datetime(2026, 8, 19, 0, 0, 2, tzinfo=UTC),
        )
        attempts.mark_completed(
            claimed.event_id,
            now=datetime(2026, 8, 19, 0, 0, 3, tzinfo=UTC),
        )
        self.database.chmod(0o600)
        self.repository = HomeShadowReviewRepository(
            self.database,
            token_secret=_SECRET,
        )

    def test_public_items_use_opaque_stable_tokens_and_hide_private_identifiers(self) -> None:
        items = self.repository.list_items(limit=6)
        self.assertEqual(1, len(items))
        public = items[0].to_public_dict()
        serialized = json.dumps(public, sort_keys=True)

        self.assertRegex(public["review_token"], r"^[A-Za-z0-9_-]{43}$")
        self.assertEqual(
            public["review_token"],
            HomeShadowReviewRepository(
                self.database,
                token_secret=_SECRET,
            )
            .list_items(limit=1)[0]
            .review_token,
        )
        self.assertEqual(1, public["frame_count"])
        self.assertTrue(public["clip_available"])
        self.assertEqual("unknown", public["review_state"])
        self.assertEqual(
            ["face_visibility_unknown", "covering_near_face_unknown"],
            public["review_reasons"],
        )
        self.assertTrue(public["manual_review_recommended"])
        self.assertFalse(public["paths_disclosed"])
        self.assertFalse(public["event_identity_disclosed"])
        self.assertFalse(public["risk_inference"])
        self.assertFalse(public["notification_decision"])
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("event_private_01", serialized)
        self.assertNotIn("camera_private_01", serialized)
        self.assertNotIn(str(self.database), repr(self.repository))

    def test_review_recommendations_sort_a_older_item_before_newer_unknown(self) -> None:
        event_root = self.media_root / "opaque-review-event"
        event_root.mkdir(mode=0o700)
        frame = event_root / "frame.jpg"
        frame.write_bytes(b"review-frame")
        frame.chmod(0o600)
        clip = event_root / "clip.mp4"
        clip.write_bytes(b"review-clip")
        clip.chmod(0o600)
        candidate = CandidateEvent(
            event_id="event_review_old",
            camera_id="camera_private_01",
            start_seconds=0.0,
            end_seconds=2.0,
            detections=(
                Detection(
                    timestamp_seconds=1.0,
                    label="person",
                    confidence=0.8,
                    bbox_xyxy=(0.1, 0.1, 0.8, 0.9),
                ),
            ),
            detector_id="yolo11n.sha256_fixture",
        )
        old_time = datetime(2026, 8, 18, tzinfo=UTC)
        AnalysisQueue(self.database).enqueue(
            candidate=candidate,
            occurred_at=old_time,
            source_kind="file",
            source_ref="private-review-fixture.mp4",
            media=EventMedia(str(clip), (str(frame),)),
            storage_root=self.media_root,
            max_active=4,
        )
        gate = select_visual_predicate_shadow(
            candidate,
            predicates=(
                VisualPredicate.FACE_VISIBILITY,
                VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
            ),
            person_detector_id=candidate.detector_id,
        )
        attempts = VisualPredicateShadowAttemptQueue(self.database)
        attempts.enqueue(gate, max_active=4, now=old_time)
        claimed = attempts.claim_next(lease_seconds=30, max_attempts=2, now=old_time)
        self.assertIsNotNone(claimed)
        results = (
            VisualPredicateResult(
                event_id=candidate.event_id,
                predicate=VisualPredicate.FACE_VISIBILITY,
                answer=VisualPredicateAnswer.NOT_VISIBLE,
                confidence=0.5,
                evidence_frames=("frame_old",),
                model_id="microsoft.Mage-VL",
                model_revision=_REVISION,
                prompt_version="atomic_shadow_v1",
            ),
            VisualPredicateResult(
                event_id=candidate.event_id,
                predicate=VisualPredicate.COVERING_NEAR_FACE_VISIBLE,
                answer=VisualPredicateAnswer.NO,
                confidence=0.5,
                evidence_frames=("frame_old",),
                model_id="microsoft.Mage-VL",
                model_revision=_REVISION,
                prompt_version="atomic_shadow_v1",
            ),
        )
        VisualPredicateShadowStore(self.database).persist(
            VisualPredicateShadowRun(
                gate=gate,
                results=results,
                recorded_at=old_time,
            )
        )
        attempts.mark_completed(candidate.event_id, now=old_time)

        items = self.repository.list_items(limit=2)

        self.assertEqual(2, len(items))
        self.assertEqual("review_recommended", items[0].review_state)
        self.assertEqual("unknown", items[1].review_state)
        self.assertLess(items[0].recorded_at, items[1].recorded_at)

    def test_media_requires_valid_token_index_private_mode_and_local_root(self) -> None:
        token = self.repository.list_items(limit=1)[0].review_token
        frame = self.repository.read_media(token, media_kind="frame", index=0)
        clip = self.repository.read_media(token, media_kind="clip", index=0)
        self.assertEqual(b"private-frame-payload", frame.payload)
        self.assertEqual("image/jpeg", frame.content_type)
        self.assertEqual(b"private-clip-payload", clip.payload)
        self.assertNotIn("private-frame-payload", repr(frame))

        tampered = ("A" if token[0] != "A" else "B") + token[1:]
        with self.assertRaisesRegex(HomeShadowReviewError, "token is invalid"):
            self.repository.read_media(tampered, media_kind="frame", index=0)
        with self.assertRaisesRegex(HomeShadowReviewError, "request is invalid"):
            self.repository.read_media(token, media_kind="frame", index=1)

        self.frame.chmod(0o644)
        with self.assertRaisesRegex(HomeShadowReviewError, "media is unavailable"):
            self.repository.read_media(token, media_kind="frame", index=0)
        self.frame.chmod(0o600)

        outside = self.root / "outside.jpg"
        outside.write_bytes(b"outside")
        outside.chmod(0o600)
        with self.assertRaisesRegex(HomeShadowReviewError, "media is unavailable"):
            HomeShadowReviewRepository._read_private_media(
                str(outside),
                storage_root=str(self.media_root),
                media_kind="frame",
            )

        connection = sqlite3.connect(self.database)
        connection.execute(
            "UPDATE analysis_jobs SET frame_paths_json = ? WHERE event_id = ?",
            ("not-json", "event_private_01"),
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(
            HomeShadowReviewError,
            "^shadow review database is unavailable$",
        ):
            self.repository.list_items(limit=1)

    def test_labels_are_append_only_latest_wins_and_remain_shadow_only(self) -> None:
        token = self.repository.list_items(limit=1)[0].review_token
        first = self.repository.record_label(
            token,
            predicate="face_visibility",
            human_answer="clear",
            reviewed_at=datetime(2026, 8, 19, 1, 0, tzinfo=UTC),
        )
        second = self.repository.record_label(
            token,
            predicate="face_visibility",
            human_answer="unknown",
            reviewed_at=datetime(2026, 8, 19, 1, 0, tzinfo=UTC),
        )
        public = second.to_public_dict()
        self.assertFalse(public["risk_inference"])
        self.assertFalse(public["notification_decision"])
        self.assertFalse(public["paths_disclosed"])
        self.assertEqual("clear", first.human_answer)
        self.assertEqual("unknown", second.human_answer)

        answers = {
            answer.predicate: answer for answer in self.repository.list_items(limit=1)[0].answers
        }
        self.assertEqual("unknown", answers["face_visibility"].human_answer)
        connection = sqlite3.connect(self.database)
        count = connection.execute(
            "SELECT COUNT(*) FROM visual_predicate_human_reviews"
        ).fetchone()[0]
        analysis_status = connection.execute(
            "SELECT status FROM analysis_jobs WHERE event_id = ?",
            ("event_private_01",),
        ).fetchone()[0]
        connection.close()
        self.assertEqual(2, count)
        self.assertEqual("pending", analysis_status)

        with self.assertRaisesRegex(HomeShadowReviewError, "label is invalid"):
            self.repository.record_label(
                token,
                predicate="covering_near_face_visible",
                human_answer="clear",
            )
        with self.assertRaisesRegex(HomeShadowReviewError, "label is invalid"):
            self.repository.record_label(
                token,
                predicate="face_visibility",
                human_answer="clear",
                reviewed_at=datetime(2026, 8, 19),
            )

    def test_calibration_report_separates_models_and_latest_human_labels(self) -> None:
        initial = self.repository.calibration_report(limit=100)
        self.assertEqual(1, initial["observation_count"])
        self.assertEqual(2, initial["answer_count"])
        self.assertEqual(0, initial["labeled_answer_count"])
        self.assertIsNone(initial["exact_agreement_rate"])
        self.assertFalse(initial["exact_answer_agreement_evaluated"])

        token = self.repository.list_items(limit=1)[0].review_token
        self.repository.record_label(
            token,
            predicate="face_visibility",
            human_answer="clear",
        )
        self.repository.record_label(
            token,
            predicate="covering_near_face_visible",
            human_answer="unknown",
        )
        report = self.repository.calibration_report(limit=100)
        self.assertEqual(1, report["labeled_observation_count"])
        self.assertEqual(2, report["labeled_answer_count"])
        self.assertEqual(0, report["unlabeled_answer_count"])
        self.assertEqual(1, report["exact_agreement_count"])
        self.assertEqual(0.5, report["exact_agreement_rate"])
        self.assertTrue(report["exact_answer_agreement_evaluated"])
        self.assertFalse(report["risk_accuracy_evaluated"])
        self.assertFalse(report["media_read"])
        self.assertFalse(report["paths_disclosed"])
        groups = {item["predicate"]: item for item in report["groups"]}
        self.assertEqual(
            [
                {
                    "model_answer": "unknown",
                    "human_answer": "clear",
                    "count": 1,
                }
            ],
            groups["face_visibility"]["confusion"],
        )
        self.assertEqual(
            [
                {
                    "model_answer": "unknown",
                    "human_answer": "unknown",
                    "count": 1,
                }
            ],
            groups["covering_near_face_visible"]["confusion"],
        )
        self.assertEqual(1, groups["covering_near_face_visible"]["human_unknown_count"])
        self.assertEqual(1, groups["face_visibility"]["model_unknown_count"])

        schema = json.loads(
            Path("schemas/visual-predicate-human-calibration-report-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(set(schema["required"]), set(report))
        self.assertEqual(set(schema["$defs"]["group"]["required"]), set(report["groups"][0]))

        stdout = StringIO()
        with redirect_stdout(stdout):
            code = main(
                [
                    "visual-predicate-human-calibration-report",
                    "--database",
                    str(self.database),
                    "--limit",
                    "100",
                ]
            )
        self.assertEqual(0, code)
        self.assertEqual(report, json.loads(stdout.getvalue()))
        self.assertNotIn(str(self.root), stdout.getvalue())
        self.assertNotIn("event_private_01", stdout.getvalue())
        connection = sqlite3.connect(self.database)
        connection.execute(
            """
            UPDATE visual_predicate_human_reviews
            SET human_answer = 'clear'
            WHERE predicate = 'covering_near_face_visible'
            """
        )
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(HomeShadowReviewError, "label is invalid"):
            self.repository.calibration_report(limit=100)
        for invalid_limit in (0, 1_001, True):
            with (
                self.subTest(limit=invalid_limit),
                self.assertRaisesRegex(ValueError, "between 1 and 1000"),
            ):
                self.repository.calibration_report(limit=invalid_limit)

    def test_database_and_parent_permissions_fail_closed_without_creation(self) -> None:
        self.database.chmod(0o644)
        with self.assertRaisesRegex(HomeShadowReviewError, "database is unavailable"):
            HomeShadowReviewRepository(self.database, token_secret=_SECRET)
        self.database.chmod(0o600)
        self.state.chmod(0o755)
        try:
            with self.assertRaisesRegex(HomeShadowReviewError, "database is unavailable"):
                HomeShadowReviewRepository(self.database, token_secret=_SECRET)
        finally:
            self.state.chmod(0o700)
        missing = self.state / "missing.sqlite3"
        with self.assertRaisesRegex(HomeShadowReviewError, "database is unavailable"):
            HomeShadowReviewRepository(missing, token_secret=_SECRET)
        self.assertFalse(missing.exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink is unavailable")
    def test_media_symlink_swap_fails_closed(self) -> None:
        token = self.repository.list_items(limit=1)[0].review_token
        original = self.frame.with_suffix(".original")
        self.frame.rename(original)
        self.frame.symlink_to(original)
        with self.assertRaisesRegex(HomeShadowReviewError, "media is unavailable"):
            self.repository.read_media(token, media_kind="frame", index=0)


if __name__ == "__main__":
    unittest.main()
