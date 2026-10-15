from __future__ import annotations

import json
import os
import signal
import tempfile
import threading
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

from baby_monitor import live_monitor
from baby_monitor.adapters.camera import RTSPCamera
from baby_monitor.live_detection import LiveFrame, ReviewCandidate, ReviewMailbox
from baby_monitor.live_review_control import LiveReviewSuspension, process_start_ticks
from tests.test_live_detection import observation
from tests.test_live_monitor_diagnostics import SubjectVLM


class ReviewSuspensionTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.control = LiveReviewSuspension(timeout_seconds=5, clock=lambda: self.now)

    def test_delivery_is_not_unload_acknowledgement(self):
        self.control.request_suspend()
        self.assertTrue(self.control.requested)
        self.assertTrue(self.control.blocked)
        self.assertFalse(self.control.snapshot()["resident_unloaded"])
        self.assertEqual(0, self.control.suspension_count)

    def test_acknowledgement_and_resume_have_a_new_evidence_cutoff(self):
        self.control.request_suspend()
        self.control.acknowledge_unloaded()
        self.control.request_resume()
        self.assertTrue(self.control.blocked)
        self.now = 12.0
        self.assertTrue(self.control.release())
        self.assertFalse(self.control.blocked)
        self.assertEqual(12, self.control.resumed_after)
        self.assertEqual(1, self.control.suspension_count)

    def test_timeout_does_not_clear_unload_until_review_thread_resumes(self):
        self.control.request_suspend()
        self.control.acknowledge_unloaded()
        self.now = 15.0
        self.assertFalse(self.control.requested)
        self.assertTrue(self.control.blocked)
        self.assertEqual(0, self.control.snapshot()["remaining_seconds"])
        self.assertTrue(self.control.release())

    def test_renewed_signal_during_resume_flush_cannot_crash_or_release(self):
        self.control.request_suspend()
        self.control.acknowledge_unloaded()
        self.control.request_resume()
        self.control.request_suspend()
        self.assertFalse(self.control.release())
        self.assertTrue(self.control.unloaded)
        self.assertEqual(1, self.control.suspension_count)

    def test_duplicate_unload_and_unacknowledged_resume_are_rejected(self):
        with self.assertRaises(ValueError):
            self.control.release()
        self.control.acknowledge_unloaded()
        with self.assertRaises(ValueError):
            self.control.acknowledge_unloaded()

    def test_timeout_is_bounded_and_strict(self):
        for value in (None, True, False, 0, 601, 5.0, "5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LiveReviewSuspension(timeout_seconds=value)
        with self.assertRaises(ValueError):
            LiveReviewSuspension(clock=None)

    def test_private_queue_discard_preserves_source_generation_and_counters(self):
        queue = ReviewMailbox(("cam1",))
        row = observation(now=10)
        queue.put(ReviewCandidate(row, "change", 2, 10))
        queue.discard_pending_for_suspend()
        self.assertEqual(1, queue.snapshot()["control_discarded_candidates"])
        self.assertEqual(0, queue.snapshot()["pending_candidates"])
        self.assertEqual({"cam1": 0}, queue.generations)
        self.assertEqual(0, queue.snapshot()["generation_discarded_candidates"])
        queue.put(ReviewCandidate(row, "change", 2, 11))
        self.assertIsNotNone(queue.take(11))

    def test_pause_hides_even_young_completed_claims(self):
        result = live_monitor._subject_projection(
            dict(
                status="analyzed",
                frame_received_at="fixture",
                received_monotonic=10,
                frame_generation=0,
                completed_at="done",
            ),
            None,
            [{"age_claim": "baby", "head_center": [100, 200]}],
            now=11,
            source_phase="receiving",
            source_generation=0,
            monitor_phase="local_review_suspended",
        )
        self.assertIsNone(result["subject_claims"])
        self.assertEqual("local_review_suspended", result["subject_evidence"]["status"])

    def test_process_start_parser_accepts_parentheses_in_comm(self):
        payload = b"42 (synthetic ) name) S " + b"0 " * 18 + b"123 0 0\n"
        with patch.object(Path, "open", mock_open(read_data=payload)):
            self.assertEqual(123, process_start_ticks(42))

    def test_process_identity_rejects_invalid_pid_and_oversized_stat(self):
        for value in (True, 0, -1, 42.0, "42", 2**31):
            with self.assertRaises(ValueError):
                process_start_ticks(value)
        for value in (b"broken", b"x" * 4097):
            with (
                patch.object(Path, "open", mock_open(read_data=value)),
                self.assertRaises(ValueError),
            ):
                process_start_ticks(42)


@unittest.skipUnless(os.name == "posix", "live monitor requires POSIX")
class ReviewSuspensionMonitorTests(unittest.TestCase):
    def exercise(self, *, fail_close=False):
        """Real monitor threads; synthetic frames/model, no devices or GPU calls."""
        control = LiveReviewSuspension()
        snapshots = []
        predict_counts = []

        class Source:
            def __init__(self, source, *, mailbox, **_kwargs):
                self.source, self.mailbox = source, mailbox
                self.stop = threading.Event()
                self.sequence = 0

            def start(self):
                self.thread = threading.Thread(target=self.feed, daemon=True)
                self.thread.start()

            def feed(self):
                while not self.stop.is_set():
                    self.mailbox.put(
                        LiveFrame(
                            self.source.camera_id,
                            self.sequence,
                            16,
                            16,
                            datetime.now(UTC),
                            time.monotonic(),
                            b"\x00" * (16 * 16 * 3),
                        )
                    )
                    self.sequence += 1
                    self.stop.wait(0.025)

            def snapshot(self):
                return {"phase": "receiving", "generation": 0}

            def close(self):
                self.stop.set()
                self.thread.join(timeout=1)

        class VLM(SubjectVLM):
            active = False
            close_at = None
            close_calls = 0
            failed = False
            paused_detections = None

            def probe_subjects(self, request):
                self.active = True
                if not control.suspension_count:
                    control.request_suspend()
                    prior = len(predict_counts)
                    time.sleep(0.15)
                    self.paused_detections = len(predict_counts) - prior
                try:
                    return super().probe_subjects(request)
                finally:
                    self.active = False

            def close(self):
                if self.active:
                    raise AssertionError("must drain in-flight request")
                self.close_calls += 1
                if fail_close and not self.failed:
                    self.failed = True
                    raise RuntimeError("synthetic-private-close-error")
                self.close_at = time.monotonic()

        adapter = VLM()
        detector = Mock(spec=("identity", "start", "close", "predict"), identity={"fixture": True})
        resumed = []

        def predict(frames):
            predict_counts.append(time.monotonic())
            if (
                control.unloaded
                and adapter.close_at is not None
                and time.monotonic() - adapter.close_at >= 1.1
            ):
                resumed.append(len(predict_counts))
                control.request_resume()
            return tuple(() for _ in frames)

        detector.predict.side_effect = predict
        original_write = live_monitor._write_private

        def write(path, value):
            if path.name == "status.json":
                snapshots.append(json.loads(json.dumps(value)))
            original_write(path, value)

        handlers = {s: signal.getsignal(s) for s in (signal.SIGUSR1, signal.SIGUSR2)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            with (
                patch.object(live_monitor, "LiveReviewSuspension", return_value=control),
                patch.object(live_monitor, "FFmpegLiveSource", Source),
                patch.object(live_monitor, "_write_private", side_effect=write),
                patch.object(
                    live_monitor, "_review_storage_summary", return_value={"available": True}
                ),
            ):
                arguments = dict(
                    streams=tuple(
                        (RTSPCamera(c, "rtsp://127.0.0.1/fixture").source(), 16, 16)
                        for c in ("a", "b", "c", "d")
                    ),
                    detector=detector,
                    local_vlm=adapter,
                    root=root,
                    review_admission=lambda: True,
                    review_task="subjects",
                    allow_review_suspend_signals=True,
                    max_runtime_seconds=2.5,
                )
                if fail_close:
                    with self.assertRaisesRegex(RuntimeError, "local review worker failed"):
                        live_monitor.run_live_monitor(**arguments)
                else:
                    live_monitor.run_live_monitor(**arguments)
        for signum, previous in handlers.items():
            self.assertIs(signal.getsignal(signum), previous)
        self.assertGreater(adapter.paused_detections, 0)
        self.assertTrue(detector.predict.called)
        self.assertEqual(1, detector.start.call_count)
        self.assertEqual(1, detector.close.call_count)
        return control, snapshots, adapter, resumed

    def test_drains_unloads_and_resumes_while_detection_continues(self):
        control, snapshots, adapter, resumed = self.exercise()
        paused = [
            s
            for s in snapshots
            if s.get("review_phase") == "suspended" and s["phase"] == "detecting"
        ]
        self.assertTrue(paused)
        for snapshot in paused:
            self.assertTrue(snapshot["review_control"]["resident_unloaded"])
            for camera in snapshot["cameras"].values():
                self.assertIsNone(camera["subject_claims"])
                self.assertEqual("local_review_suspended", camera["subject_evidence"]["status"])
                self.assertGreater(camera["inferred_frames"], 0)
        self.assertEqual(1, control.suspension_count)
        self.assertTrue(resumed)
        self.assertFalse(control.blocked)
        self.assertGreater(snapshots[-1]["review_suspended_frames"], 0)
        self.assertGreater(snapshots[-1]["subject_probe_successes"], 1)
        self.assertEqual(0, snapshots[-1]["subject_probe_failures"])
        self.assertEqual(2, adapter.close_calls)  # pause boundary, final cleanup

    def test_close_failure_is_never_a_model_release_acknowledgement(self):
        control, snapshots, _adapter, _resumed = self.exercise(fail_close=True)
        self.assertFalse(control.unloaded)
        self.assertEqual(0, control.suspension_count)
        self.assertFalse(any(s.get("review_phase") == "suspended" for s in snapshots))

    def test_suspend_is_explicit_and_requires_main_thread_and_close_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for option in (None, 1, "true", True):
                detector = Mock()
                with self.assertRaises(ValueError):
                    live_monitor.run_live_monitor(
                        streams=(),
                        detector=detector,
                        local_vlm=object(),
                        root=root,
                        review_admission=lambda: True,
                        allow_review_suspend_signals=option,
                    )
                detector.start.assert_not_called()
            self.assertEqual([], list(root.iterdir()))


if __name__ == "__main__":
    unittest.main()
