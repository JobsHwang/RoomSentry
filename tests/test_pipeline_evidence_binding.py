from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from baby_monitor.adapters.vlm import VLMError
from baby_monitor.database import EventDatabase
from baby_monitor.domain import CandidateEvent, EventMedia
from baby_monitor.evidence_binding import EvidenceBinding, capture_evidence_binding
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy
from baby_monitor.recognition_attempt import (
    RecognitionAttemptAuditStore,
    RecognitionAttemptStatus,
    RecognitionStage,
)
from tests.test_secondary_review_failure import _FailingVLM, _StaticVLM


class _MutatingVLM(_StaticVLM):
    def __init__(self, name, target):
        super().__init__(name)
        self.target = target

    def analyze(self, request):
        result = super().analyze(request)
        # Same-size mutation distinguishes byte binding from mere path/size checks.
        self.target.write_bytes(b"X" * self.target.stat().st_size)
        return result


class PipelineEvidenceBindingTests(unittest.TestCase):
    NOW = datetime(2026, 9, 12, 8, 0, tzinfo=UTC)

    def _arguments(self, root):
        root.chmod(0o700)
        clip = root / "synthetic.mp4"
        frames = (root / "synthetic_01.jpg", root / "synthetic_02.jpg")
        for index, path in enumerate((clip, *frames)):
            path.write_bytes(f"synthetic private evidence asset {index}".encode())
            path.chmod(0o600)
        return {
            "candidate": CandidateEvent("binding_event_fixture", "cam_01", 0.0, 1.0, ()),
            "occurred_at": self.NOW,
            "source_kind": "fixture",
            "source_ref": "synthetic_bound_source",
            "media": EventMedia(str(clip), tuple(str(path) for path in frames)),
            "local_vlm": _StaticVLM("synthetic_primary"),
            "secondary_vlm": None,
            "remote_vlm": _StaticVLM("synthetic_remote"),
            "remote_uploads_allowed": False,
            "database": EventDatabase(root / "events.sqlite3"),
            "policy": ReviewPolicy(),
        }

    def _binding(self, arguments):
        return EventDatabase(arguments["database"].path).get_evidence_binding(
            arguments["candidate"].event_id
        )

    def _event(self, arguments):
        return EventDatabase(arguments["database"].path).get_event(arguments["candidate"].event_id)

    def test_success_persists_ordered_evidence_binding_with_event_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory))
            expected = capture_evidence_binding(args["media"])
            summary = process_prepared_event(**args)
            event, binding = self._event(args), self._binding(args)
            self.assertIsNotNone(event)
            self.assertIsInstance(binding, EvidenceBinding)
            self.assertEqual(expected, binding)
            self.assertIsNone(binding.verify(args["media"]))
            self.assertEqual(args["local_vlm"].metrics, event.local_metrics)
            self.assertEqual(
                ("clip", "frame", "frame"), tuple(asset.role for asset in binding.assets)
            )
            self.assertEqual((0, 0, 1), tuple(asset.index for asset in binding.assets))
            for asset, path in zip(
                binding.assets, (args["media"].clip_path, *args["media"].frame_paths), strict=True
            ):
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), asset.sha256)
            self.assertNotIn(str(directory), binding.to_json())
            self.assertIn("evidence_coverage_unverified", event.decision.review_reasons)
            self.assertFalse(summary.reused_existing)

    def test_secondary_failure_still_saves_successful_primary_and_verified_endpoint_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory))
            args["secondary_vlm"] = _FailingVLM()
            expected = capture_evidence_binding(args["media"])
            summary = process_prepared_event(**args)
            event, binding = self._event(args), self._binding(args)
            self.assertEqual(expected, binding)
            binding.verify(args["media"])
            self.assertEqual(event.local_result, event.result)
            self.assertEqual(args["local_vlm"].metrics, event.local_metrics)
            self.assertIsNone(event.secondary_result)
            self.assertIsNone(event.secondary_metrics)
            self.assertIn("local_secondary_review_failed", event.decision.review_reasons)
            self.assertFalse(summary.secondary_reviewed)

    def _assert_changed_evidence_rejected(self, args):
        with self.assertRaises(VLMError) as raised:
            process_prepared_event(**args)
        self.assertEqual("evidence_binding_changed", raised.exception.code)
        self.assertNotIn(args["media"].clip_path, str(raised.exception))
        for frame in args["media"].frame_paths:
            self.assertNotIn(frame, str(raised.exception))
        self.assertIsNone(self._event(args))
        self.assertIsNone(self._binding(args))

    def test_primary_mutation_of_clip_or_either_frame_rejects_before_event_insert(self):
        for index in range(3):
            with self.subTest(asset_index=index), tempfile.TemporaryDirectory() as directory:
                args = self._arguments(Path(directory))
                paths = (args["media"].clip_path, *args["media"].frame_paths)
                args["local_vlm"] = _MutatingVLM("synthetic_mutating_primary", Path(paths[index]))
                self._assert_changed_evidence_rejected(args)
                self.assertEqual(1, args["local_vlm"].calls)
                attempts = RecognitionAttemptAuditStore(args["database"].path).recent()
                self.assertEqual(1, len(attempts))
                self.assertEqual(RecognitionStage.LOCAL_PRIMARY, attempts[0].stage)
                # Model output succeeded, but its event cannot bind changed evidence.
                self.assertEqual(RecognitionAttemptStatus.SUCCEEDED, attempts[0].status)

    def test_mutation_in_secondary_or_remote_is_checked_after_all_model_stages(self):
        for stage in ("secondary", "remote"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                args = self._arguments(Path(directory))
                mutating = _MutatingVLM(
                    "synthetic_mutating_review", Path(args["media"].frame_paths[0])
                )
                if stage == "secondary":
                    args["secondary_vlm"] = mutating
                else:
                    args["local_vlm"] = _StaticVLM("synthetic_primary", confidence=0.4)
                    args["remote_vlm"] = mutating
                    args["remote_uploads_allowed"] = True
                self._assert_changed_evidence_rejected(args)
                self.assertEqual(1, mutating.calls)
                self.assertEqual(1, args["local_vlm"].calls)

    def test_unsafe_or_missing_media_remain_compatible_but_never_receive_a_binding(self):
        for kind in ("unsafe_clip", "unsafe_frame", "missing_clip", "missing_frame"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                args = self._arguments(Path(directory))
                target = Path(
                    args["media"].clip_path
                    if kind.endswith("clip")
                    else args["media"].frame_paths[0]
                )
                if kind.startswith("unsafe"):
                    target.chmod(0o644)
                else:
                    target.unlink()
                summary = process_prepared_event(**args)
                self.assertFalse(summary.reused_existing)
                self.assertIsNotNone(self._event(args))
                self.assertIsNone(self._binding(args))
                self.assertEqual(1, args["local_vlm"].calls)

    def test_existing_bound_event_reuse_does_not_reread_or_reverify_deleted_media(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory))
            process_prepared_event(**args)
            original = self._binding(args)
            for path in (args["media"].clip_path, *args["media"].frame_paths):
                Path(path).unlink()
            with (
                patch(
                    "baby_monitor.pipeline.capture_evidence_binding",
                    side_effect=AssertionError("no recapture"),
                ),
                patch.object(
                    EvidenceBinding, "verify", side_effect=AssertionError("no verification")
                ),
            ):
                replay = process_prepared_event(**args)
            self.assertTrue(replay.reused_existing)
            self.assertEqual(original, self._binding(args))
            self.assertEqual(1, args["local_vlm"].calls)

    def test_existing_unbound_event_is_not_backfilled_when_media_become_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._arguments(Path(directory))
            clip = Path(args["media"].clip_path)
            clip.chmod(0o644)
            process_prepared_event(**args)
            self.assertIsNone(self._binding(args))
            clip.chmod(0o600)
            self.assertIsInstance(capture_evidence_binding(args["media"]), EvidenceBinding)
            with patch(
                "baby_monitor.pipeline.capture_evidence_binding",
                side_effect=AssertionError("no backfill"),
            ):
                replay = process_prepared_event(**args)
            self.assertTrue(replay.reused_existing)
            self.assertIsNone(self._binding(args))
            self.assertEqual(1, args["local_vlm"].calls)


if __name__ == "__main__":
    unittest.main()
