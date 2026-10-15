from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from baby_monitor import live_monitor
from baby_monitor.adapters.camera import RTSPCamera
from baby_monitor.domain import Detection
from baby_monitor.live_confirmation_shadow import (
    ConfirmationShadowRequest,
    LiveConfirmationShadow,
    summarize_shadow_probe,
)
from baby_monitor.live_detection import LiveObservation, ReviewCandidate
from baby_monitor.local_runners.subject_probe import PROMPT_VERSION
from tests.test_live_detection import frame

A = Detection(0.0, "person", 0.8, (0.1, 0.1, 0.4, 0.6))
B = Detection(0.0, "person", 0.8, (0.7, 0.1, 0.95, 0.6))
BASE_UTC = datetime(2026, 10, 12, tzinfo=UTC)


def observed(sequence=0, now=10.0, detections=(A,), generation=0):
    source = replace(
        frame(sequence=sequence, now=now),
        generation=generation,
        received_at=BASE_UTC + timedelta(seconds=now),
    )
    return LiveObservation(source, detections, now)


def probe_result(people=None):
    return {
        "version": 1,
        "prompt_revision": PROMPT_VERSION,
        "semantics": "unverified_model_claims",
        "frame_id": "fixture.png",
        "input_sha256": "a" * 64,
        "coordinate_space": "submitted_image_thousandths",
        "claims": {"people": people or [], "uncertainties": []},
    }


class ConfirmationShadowTests(unittest.TestCase):
    def tracker(self):
        return LiveConfirmationShadow(("cam1",), minimum_frame_confidence=0.4)

    def observe(self, tracker, obs, *, actual=False, view="rotation_180"):
        tracker.observe(obs, detector_view=view, actual_alternate_requested=actual)

    def request(self, tracker, now=11.0):
        return tracker.take(now, BASE_UTC + timedelta(seconds=now))

    def confirmed(self):
        tracker = self.tracker()
        self.observe(tracker, observed())
        self.observe(tracker, observed(1, 11.0))
        return tracker, self.request(tracker)

    def test_stable_union_control_is_enqueued_and_original_detections_unchanged(self):
        tracker, request = self.confirmed()
        row = request.comparison
        self.assertTrue(row["whole_requested"])
        self.assertTrue(row["filtered_whole_requested"])
        self.assertEqual((0,), row["per_candidate"]["review_requested_indices"])
        self.assertEqual((A,), request.observation.detections)
        self.assertEqual(request.observation.frame.sha256, row["raw_bgr_sha256"])
        self.assertEqual(1, tracker.summary()["counts"]["probe_queued"])

    def test_strong_flicker_new_only_difference_is_preserved(self):
        tracker = self.tracker()
        for i in range(7):
            self.observe(tracker, observed(i, 10 + i / 5, (A, B) if i % 2 else (A,)))
        request = self.request(tracker, 11.2)
        self.assertFalse(request.comparison["whole_requested"])
        self.assertFalse(request.comparison["filtered_whole_requested"])
        self.assertEqual((0,), request.comparison["per_candidate"]["review_requested_indices"])
        self.assertEqual(1, tracker.summary()["cameras"]["cam1"]["request_matrix_01"])

    def test_coarse_same_cell_old_only_case_is_sampled_not_hidden(self):
        tracker = self.tracker()
        for i in range(7):
            box = replace(A, bbox_xyxy=(0.1, 0.1, 0.11, 0.11) if i % 2 else (0.13, 0.1, 0.14, 0.11))
            self.observe(tracker, observed(i, 10 + i / 5, (box,)))
        request = self.request(tracker, 11.2)
        self.assertTrue(request.comparison["whole_requested"])
        self.assertFalse(request.comparison["per_candidate"]["review_requested_indices"])
        self.assertEqual(1, tracker.summary()["cameras"]["cam1"]["request_matrix_10"])

    def test_checked_zero_frames_count_but_unchecked_skips_are_not_fabricated(self):
        tracker = self.tracker()
        self.observe(tracker, observed())
        self.observe(tracker, observed(1, 10.5, ()))
        self.observe(tracker, observed(2, 11.0))
        report = tracker.snapshot()
        self.assertEqual(3, report["counts"]["checked_frames"])
        self.assertEqual(1, report["cameras"]["cam1"]["zero_frames"])
        self.assertEqual([], report["checks"][1]["raw_boxes"])
        self.assertFalse(report["checks"][2]["per_candidate"]["chains"][0]["confirmed"])
        self.assertIsNone(self.request(tracker))

    def test_both_auxiliary_channels_have_separate_production_denominators(self):
        tracker = self.tracker()
        self.observe(tracker, observed(), actual=None)
        self.observe(tracker, observed(1, 11.0), actual=True)
        counts = tracker.summary()["cameras"]["cam1"]
        self.assertEqual(1, counts["nonempty_auxiliary_checks"])
        self.assertEqual(1, counts["production_alternate_checks"])
        self.assertEqual(1, counts["production_alternate_requests"])

    def test_scale_change_cannot_continue_chain_or_bypass_probe_cooldown(self):
        tracker = self.tracker()
        self.observe(tracker, observed())
        self.observe(tracker, observed(1, 11.0), view="rotation_180_scale_fallback")
        self.assertIsNone(self.request(tracker))
        self.observe(tracker, observed(2, 12.0), view="rotation_180_scale_fallback")
        self.assertIsNotNone(self.request(tracker, 12.0))
        self.observe(tracker, observed(3, 13.0))
        self.observe(tracker, observed(4, 14.0))
        self.assertIsNone(self.request(tracker, 14.0))
        self.assertEqual(1, tracker.summary()["counts"]["probe_cooldown_rejected"])

    def test_pending_first_evidence_is_not_overwritten(self):
        tracker = self.tracker()
        self.observe(tracker, observed())
        self.observe(tracker, observed(1, 11.0))
        self.observe(tracker, observed(2, 12.0, ()))
        self.observe(tracker, observed(3, 42.0, (B,)))
        self.observe(tracker, observed(4, 43.0, (B,)))
        request = self.request(tracker, 43.0)
        self.assertEqual(1, request.observation.frame.sequence)
        self.assertEqual(1, tracker.summary()["counts"]["pending_slot_rejected"])
        self.assertEqual(1, tracker.summary()["capacity_per_camera"])

    def test_dual_age_rejects_expired_or_future_clocks_and_boundary_keeps(self):
        for now, wall in ((131.001, 131.001), (12.0, 131.001), (10.0, 12.0)):
            tracker, _ = self.confirmed()
            # Prepare another independent tracker whose first request is still pending.
            tracker = self.tracker()
            self.observe(tracker, observed())
            self.observe(tracker, observed(1, 11.0))
            self.assertIsNone(tracker.take(now, BASE_UTC + timedelta(seconds=wall)))
            self.assertEqual(1, tracker.summary()["counts"]["probe_expired_or_clock_invalid"])
        tracker = self.tracker()
        self.observe(tracker, observed())
        self.observe(tracker, observed(1, 11.0))
        self.assertIsNotNone(self.request(tracker, 131.0))

    def test_generation_discards_pending_and_stale_or_invalid_checks_never_enqueue(self):
        tracker = self.tracker()
        self.observe(tracker, observed())
        self.observe(tracker, observed(1, 11.0))
        self.observe(tracker, observed(0, 12.0, generation=1))
        self.assertIsNone(self.request(tracker, 12.0))
        self.observe(tracker, observed(99, 50.0, generation=0))
        self.observe(tracker, observed(0, 12.0, generation=1))
        self.assertIsNone(self.request(tracker, 50.0))
        self.assertEqual(2, tracker.summary()["counts"]["rejected_checks"])

    def test_suspend_resume_and_failure_clear_pending_without_erasing_counts(self):
        for reason in ("suspension", "resume", "telemetry_failed"):
            tracker = self.tracker()
            self.observe(tracker, observed())
            self.observe(tracker, observed(1, 11.0))
            tracker.gap(reason=reason)
            self.assertIsNone(self.request(tracker))
            self.assertEqual(2, tracker.summary()["counts"]["checked_frames"])
            self.observe(tracker, observed(2, 12.0))
            self.assertEqual(
                2 if reason == "telemetry_failed" else 3,
                tracker.summary()["counts"]["checked_frames"],
            )

    def test_history_bounded_and_snapshot_mutation_cannot_change_state(self):
        tracker = self.tracker()
        for i in range(160):
            self.observe(tracker, observed(i, 10 + i / 5, ()))
        snapshot = tracker.snapshot()
        self.assertEqual(128, len(snapshot["checks"]))
        self.assertEqual(32, snapshot["counts"]["check_history_evicted"])
        snapshot["checks"][0]["raw_boxes"].append("malicious")
        self.assertEqual([], tracker.snapshot()["checks"][0]["raw_boxes"])
        tracker, request = self.confirmed()
        for _ in range(40):
            tracker.complete(request, status="analysis_failed", summary={"error_code": "fixed"})
        self.assertEqual(32, len(tracker.snapshot()["reviews"]))
        self.assertEqual(8, tracker.summary()["counts"]["review_history_evicted"])

    def test_bad_configuration_and_observations_rejected(self):
        for minimum in (True, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                LiveConfirmationShadow(("cam1",), minimum_frame_confidence=minimum)
        tracker = self.tracker()
        with self.assertRaises(ValueError):
            self.observe(tracker, observed(), actual="yes")
        with self.assertRaises(ValueError):
            self.observe(tracker, observed(), view="untrusted_view")
        with self.assertRaises(ValueError):
            tracker.gap(reason="anything")

    def test_model_summary_keeps_null_heads_and_empty_claims_unresolved(self):
        _, request = self.confirmed()
        people = [
            {"head_center": [200, 200], "age_group": "unknown", "visual_evidence": "fixture"},
            {"head_center": None, "age_group": "unknown", "visual_evidence": "fixture"},
        ]
        summary = summarize_shadow_probe(
            request, probe_result(people), frame_id="fixture.png", png_sha256="a" * 64
        )
        self.assertEqual(2, summary["claimed_people"])
        self.assertEqual(1, summary["unlocated_head_claims"])
        self.assertEqual("unique_head_claim_support", summary["head_support"][0]["status"])
        self.assertFalse(summary["person_truth_available"])
        self.assertFalse(summary["unique_head_support_means_unique_identity"])
        empty = summarize_shadow_probe(
            request, probe_result(), frame_id="fixture.png", png_sha256="a" * 64
        )
        self.assertEqual("no_person_claim", empty["model_person_presence_claim"])
        self.assertFalse(empty["no_head_support_means_false_detection"])

    def test_model_summary_rejects_wrong_image_binding_and_preserves_overlap_ambiguity(self):
        _, request = self.confirmed()
        result = probe_result(
            [{"head_center": [200, 200], "age_group": "unknown", "visual_evidence": "fixture"}]
        )
        with self.assertRaises(ValueError):
            summarize_shadow_probe(request, result, frame_id="other.png", png_sha256="a" * 64)
        with self.assertRaises(ValueError):
            summarize_shadow_probe(request, result, frame_id="fixture.png", png_sha256="b" * 64)
        comparison = copy.deepcopy(request.comparison)
        comparison["raw_boxes"] *= 2
        duplicated = replace(request, comparison=comparison)
        summary = summarize_shadow_probe(
            duplicated, result, frame_id="fixture.png", png_sha256="a" * 64
        )
        self.assertEqual(
            ["ambiguous_head_claim_support"] * 2, [r["status"] for r in summary["head_support"]]
        )

    def test_regular_zero_native_probe_can_flag_model_disagreement_without_auxiliary_guess(self):
        tracker = self.tracker()
        result = probe_result(
            [{"head_center": None, "age_group": "unknown", "visual_evidence": "fixture"}]
        )
        report = tracker.record_subject_probe(
            observed(detections=()), result, frame_id="fixture.png", png_sha256="a" * 64
        )
        self.assertEqual("native_0_model_1", report["native_model_category"])
        self.assertFalse(report["auxiliary_check_same_frame"])
        self.assertIsNone(report["auxiliary_candidate_count"])
        self.assertFalse(report["model_agreement"]["accuracy_verified"])

    def test_regular_probe_joins_only_exact_sequence_generation_and_raw_pixel_hash(self):
        tracker = self.tracker()
        obs = observed()
        self.observe(tracker, obs)
        exact = tracker.record_subject_probe(
            obs, probe_result(), frame_id="fixture.png", png_sha256="a" * 64
        )
        self.assertTrue(exact["auxiliary_check_same_frame"])
        for changed in (
            replace(obs, frame=replace(obs.frame, sequence=1)),
            replace(obs, frame=replace(obs.frame, generation=1)),
            replace(obs, frame=replace(obs.frame, pixels=bytes([30, 20, 10]) * 256)),
        ):
            report = tracker.record_subject_probe(
                changed, probe_result(), frame_id="fixture.png", png_sha256="a" * 64
            )
            self.assertFalse(report["auxiliary_check_same_frame"])

    def test_regular_probe_history_bounded_and_binding_error_not_counted(self):
        tracker = self.tracker()
        for i in range(40):
            tracker.record_subject_probe(
                observed(i), probe_result(), frame_id="fixture.png", png_sha256="a" * 64
            )
        self.assertEqual(32, len(tracker.snapshot()["subject_probes"]))
        self.assertEqual(8, tracker.summary()["counts"]["subject_probe_history_evicted"])
        with self.assertRaises(ValueError):
            tracker.record_subject_probe(
                observed(), probe_result(), frame_id="wrong.png", png_sha256="a" * 64
            )
        self.assertEqual(40, tracker.summary()["counts"]["subject_probe_checks"])


class ConfirmationShadowMonitorTests(unittest.TestCase):
    def test_real_review_thread_uses_probe_without_event_writeback_or_live_claim_projection(self):
        self.run_review_case()

    def test_failed_shadow_probe_does_not_increment_production_failures(self):
        self.run_review_case(fail=True)

    def test_shadow_empty_claim_does_not_erase_completed_regular_subject_claims(self):
        self.run_review_case(regular_first=True)

    def test_busy_regular_queue_gets_four_slots_before_shadow_then_resumes(self):
        self.run_review_case(regular_first=True, regular_count=6)

    def run_review_case(self, *, fail=False, regular_first=False, regular_count=None):
        if regular_count is None:
            regular_count = int(regular_first)
        source_frame = frame(sequence=5, now=time.monotonic())
        obs = LiveObservation(source_frame, (A,), time.monotonic())
        comparison = {
            "probe_id": "fixture",
            "detector_view": "rotation_180",
            "raw_boxes": [{"confidence": 0.8, "bbox_xyxy": list(A.bbox_xyxy)}],
        }
        request = ConfirmationShadowRequest(obs, comparison)
        shadow = Mock()
        shadow.take.side_effect = lambda *_: requests.pop(0) if requests else None
        shadow.summary.return_value = {"enabled": True}
        shadow.snapshot.return_value = {"enabled": True, "checks": []}
        requests = [request]
        regular_queue = Mock()
        regular_queue.snapshot.return_value = {}
        regular_queue.condition = threading.Condition()
        regular_items = [
            ReviewCandidate(obs, "subject_current_scene", 0, obs.completed_monotonic)
        ] * regular_count
        regular_queue.take_live.side_effect = lambda: (
            regular_items.pop(0) if regular_items else None
        )
        subject_claim_inputs = []
        original_projection = live_monitor._subject_projection

        def projection(attempt, completed, claims, **context):
            subject_claim_inputs.append(copy.deepcopy(claims))
            return original_projection(attempt, completed, claims, **context)

        class Mailbox:
            def __init__(self, *args, **kwargs):
                self.condition = threading.Condition()

            def wait_for_batch(self):
                pass

            def take_live(self):
                return ()

            def snapshot(self, _now):
                return {"cam1": {}}

        class Source:
            def __init__(self, source, *args, **kwargs):
                self.source = source

            def start(self):
                pass

            def snapshot(self):
                return {"phase": "receiving", "generation": 0}

            def close(self):
                pass

        def probe(request):
            if fail:
                raise RuntimeError("fixture failed probe")
            path = Path(request.media.frame_paths[0])
            result = probe_result(
                [{"head_center": [200, 200], "age_group": "adult", "visual_evidence": "fixture"}]
                if regular_first and vlm.probe_subjects.call_count == 1
                else []
            )
            result.update(
                frame_id=path.name, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            return result

        vlm = Mock(last_diagnostics=None, probe_subjects=Mock(side_effect=probe))
        detector = Mock(identity={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            with (
                patch.object(live_monitor, "FrameMailbox", Mailbox),
                patch.object(live_monitor, "FFmpegLiveSource", Source),
                patch.object(live_monitor, "LiveConfirmationShadow", return_value=shadow),
                patch.object(live_monitor, "process_prepared_event") as writeback,
                patch.object(live_monitor, "EventDatabase"),
                patch.object(live_monitor, "ReviewMailbox", return_value=regular_queue),
                patch.object(live_monitor, "_subject_projection", side_effect=projection),
            ):
                live_monitor.run_live_monitor(
                    streams=((RTSPCamera("cam1", "rtsp://127.0.0.1/fixture").source(), 16, 16),),
                    detector=detector,
                    local_vlm=vlm,
                    root=root,
                    review_admission=lambda: True,
                    max_runtime_seconds=0.3,
                    confirmation_shadow=True,
                    review_task="subjects" if regular_first else "event",
                )
                writeback.assert_not_called()
            records = [json.loads(p.read_text()) for p in (root / "records").iterdir()]
            self.assertEqual(regular_count + 1, len(records))
            ordered_tasks = [
                r["review_task"] for r in sorted(records, key=lambda r: r["selected_at"])
            ]
            if regular_count:
                before_shadow = min(4, regular_count)
                self.assertEqual(
                    ["subjects"] * before_shadow
                    + ["confirmation_shadow"]
                    + ["subjects"] * (regular_count - before_shadow),
                    ordered_tasks,
                )
            shadow_record = next(r for r in records if r["review_task"] == "confirmation_shadow")
            expected_status = "analysis_failed" if fail else "analyzed"
            self.assertEqual(expected_status, shadow_record["status"])
            self.assertFalse(shadow_record["production_writeback"])
            self.assertEqual(source_frame.sha256, shadow_record["raw_bgr_sha256"])
            self.assertEqual(1 if regular_first else 0, len(list((root / "reviews").iterdir())))
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(0, status["review_successes"])
            self.assertEqual(regular_count, status["subject_probe_successes"])
            self.assertEqual(0, status["review_failures"])
            self.assertEqual(0, status["subject_probe_failures"])
            self.assertIsNone(status["cameras"]["cam1"]["subject_claims"])
        self.assertEqual(regular_count + 1, vlm.probe_subjects.call_count)
        if regular_first and regular_count == 1:
            self.assertEqual(
                [{"age_claim": "adult", "head_center": [200, 200]}], subject_claim_inputs[-1]
            )
        vlm.analyze.assert_not_called()
        self.assertEqual(expected_status, shadow.complete.call_args.kwargs["status"])

    def test_configuration_requires_explicit_boolean_and_probe_capability(self):
        for enabled in ("true", 1, True):
            with self.assertRaises(ValueError):
                live_monitor.run_live_monitor(
                    streams=(),
                    detector=object(),
                    local_vlm=object(),
                    root=Path("/unused"),
                    review_admission=lambda: True,
                    confirmation_shadow=enabled,
                )
