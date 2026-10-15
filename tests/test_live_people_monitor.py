"""Real detector/review threads and evidence binding, using synthetic pixels."""

import hashlib
import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor import live_monitor
from baby_monitor.adapters.camera import RTSPCamera
from baby_monitor.domain import Detection
from baby_monitor.live_detection import LiveFrame
from baby_monitor.local_runners import subject_state_probe
from tests.test_live_people import BOX, claims


@unittest.skipUnless(os.name == "posix", "live monitor requires POSIX")
class PeopleMonitorIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Healthy-thread tests must not acquire random wall-clock fault inputs
        # from WSL time synchronization. Explicit dual-clock fault tests remain
        # separate; production still uses the real, conservative two-clock gate.
        utc_start, monotonic_start = datetime.now(UTC), time.monotonic()

        class FixtureDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                value = utc_start + timedelta(seconds=time.monotonic() - monotonic_start)
                return value.replace(tzinfo=None) if tz is None else value.astimezone(tz)

        for module in (sys.modules[__name__], live_monitor):
            clock_patch = patch.object(module, "datetime", FixtureDateTime)
            clock_patch.start()
            self.addCleanup(clock_patch.stop)

    def test_completed_empty_review_invalidates_redundant_inflight_queue_slot(self):
        class Source:
            def __init__(self, source, *, mailbox, **kwargs):
                self.source = source
                self.camera, self.mailbox = source.camera_id, mailbox
                self.stop = threading.Event()
                self.thread = threading.Thread(target=self.feed)

            def feed(self):
                sequence = 0
                while not self.stop.is_set():
                    self.mailbox.put(
                        LiveFrame(
                            self.camera,
                            sequence,
                            4,
                            4,
                            datetime.now(UTC),
                            time.monotonic(),
                            bytes([sequence % 256]) * 48,
                        )
                    )
                    sequence += 1
                    self.stop.wait(0.05)

            def start(self):
                self.thread.start()

            def close(self):
                self.stop.set()
                self.thread.join(1)

            def snapshot(self):
                return {"phase": "receiving", "generation": 0}

        class Detector:
            identity = {"fixture": True}

            def start(self):
                pass

            def close(self):
                pass

            def predict(self, frames):
                return tuple(() for _ in frames)

        class VLM:
            calls = 0

            def probe_subjects(self, request):
                raise AssertionError("unexpected legacy task")

            def probe_subject_state(self, request):
                self.calls += 1
                time.sleep(0.45)  # Detection queues another frame during this call.
                frame = Path(request.media.frame_paths[0])
                return {
                    "version": 2,
                    "prompt_revision": subject_state_probe.PROMPT_VERSION,
                    "semantics": "unverified_model_claims",
                    "frame_id": frame.name,
                    "input_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    "coordinate_space": "submitted_image_thousandths",
                    "claims": {"people": [], "uncertainties": []},
                }

        vlm = VLM()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            discarded_before_stop = threading.Event()
            write_private = live_monitor._write_private

            def stop_after_discard_status(path, value):
                write_private(path, value)
                if (
                    path == root / "status.json"
                    and value["phase"] == "detecting"
                    and value["subject_probe_successes"] >= 1
                    and value["subject_not_due_discards"] > 0
                    and value["cameras"]["a"]["inferred_frames"] > 2
                ):
                    discarded_before_stop.set()
                    handler = signal.getsignal(signal.SIGTERM)
                    self.assertTrue(callable(handler))
                    handler(signal.SIGTERM, None)

            with (
                patch.object(live_monitor, "FFmpegLiveSource", Source),
                patch.object(live_monitor, "_write_private", stop_after_discard_status),
                patch.object(Detector, "start", lambda _self: time.sleep(0.7)),
            ):
                live_monitor.run_live_monitor(
                    streams=((RTSPCamera("a", "rtsp://127.0.0.1/synthetic").source(), 4, 4),),
                    detector=Detector(),
                    local_vlm=vlm,
                    root=root,
                    review_admission=lambda: True,
                    max_runtime_seconds=10.0,  # Wait for discard; startup is not coverage.
                    sample_fps=5,
                    review_task="subjects",
                    people_tracking=True,
                )
            self.assertTrue(discarded_before_stop.is_set(), "discard watchdog expired")
            status = json.loads((root / "status.json").read_text())
            self.assertGreater(status["cameras"]["a"]["inferred_frames"], 2)
            self.assertEqual(1, vlm.calls)
            self.assertEqual(1, len(list((root / "records").glob("*.json"))))
            record = json.loads(next((root / "records").glob("*.json")).read_text())
            self.assertNotIn("people_trace", record)
            self.assertFalse(status["people_review_trace_enabled"])
            self.assertFalse(status["people_transition_capture_enabled"])
            self.assertNotIn("transition_capture", status)
            self.assertFalse((root / "transition-captures").exists())
            self.assertGreater(status["subject_not_due_discards"], 0)

    def test_actual_loop_binds_roles_preserves_native_boxes_and_expires_on_stop(self):
        class Source:
            def __init__(self, source, *, mailbox, **kwargs):
                self.source, self.mailbox = source, mailbox
                self.stop = threading.Event()
                self.thread = threading.Thread(target=self.feed)

            def feed(self):
                sequence = 0
                while not self.stop.is_set():
                    self.mailbox.put(
                        LiveFrame(
                            self.source.camera_id,
                            sequence,
                            4,
                            4,
                            datetime.now(UTC),
                            time.monotonic(),
                            bytes([sequence % 256]) * 48,
                        )
                    )
                    sequence += 1
                    self.stop.wait(0.05)

            def start(self):
                self.thread.start()

            def close(self):
                self.stop.set()
                self.thread.join(1)

            def snapshot(self):
                return {"phase": "stopped" if self.stop.is_set() else "receiving", "generation": 0}

        class Detector:
            identity = {"fixture": True}

            def start(self):
                pass

            def close(self):
                pass

            def predict(self, frames):
                return tuple((Detection(0.0, "person", 0.8, BOX),) for f in frames)

        class VLM:
            execution_mode = "synthetic"

            def probe_subjects(self, request):
                raise AssertionError("the state task must use its explicit contract")

            def probe_subject_state(self, request):
                frame = Path(request.media.frame_paths[0])
                return {
                    "version": 2,
                    "prompt_revision": subject_state_probe.PROMPT_VERSION,
                    "semantics": "unverified_model_claims",
                    "frame_id": frame.name,
                    "input_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    "coordinate_space": "submitted_image_thousandths",
                    "claims": claims(),
                }

        registry = SimpleNamespace(
            room_for_stream=lambda c: SimpleNamespace(
                to_dict=lambda: {"room_id": "room_a", "display_name": "Room A"}
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            confirmed_before_stop = threading.Event()
            write_private = live_monitor._write_private

            def stop_after_confirmed_status(path, value):
                write_private(path, value)
                if (
                    path == root / "status.json"
                    and value["phase"] == "detecting"
                    and value["subject_probe_successes"] >= 2
                    and value["baby_supervision"]["current"] is not None
                ):
                    # Exercise the monitor's ordinary stop callback only after
                    # the intended observable state exists. A fixed 1.6-second
                    # run includes startup and can expire before two spaced
                    # reviews under load. No process-wide signal is sent.
                    confirmed_before_stop.set()
                    handler = signal.getsignal(signal.SIGTERM)
                    self.assertTrue(callable(handler))
                    handler(signal.SIGTERM, None)

            with (
                patch.object(live_monitor, "FFmpegLiveSource", Source),
                patch.object(live_monitor, "_write_private", stop_after_confirmed_status),
                # Explicit startup latency makes the old fixed-duration race
                # reproducible; role spacing and production clocks stay intact.
                patch.object(Detector, "start", lambda _self: time.sleep(0.7)),
            ):
                live_monitor.run_live_monitor(
                    streams=((RTSPCamera("a", "rtsp://127.0.0.1/synthetic").source(), 4, 4),),
                    detector=Detector(),
                    local_vlm=VLM(),
                    root=root,
                    review_admission=lambda: True,
                    max_runtime_seconds=10.0,  # Watchdog, not the success condition.
                    sample_fps=5,
                    review_task="subjects",
                    people_tracking=True,
                    people_review_trace=True,
                    room_registry=registry,
                )
            self.assertTrue(confirmed_before_stop.is_set(), "confirmation watchdog expired")
            records = [json.loads(p.read_text()) for p in (root / "records").glob("*.json")]
            self.assertGreaterEqual(len(records), 2)
            self.assertTrue(all(r["status"] == "analyzed" for r in records))
            self.assertTrue(all(r["people_binding"]["bound_people"] == 1 for r in records))
            for record in records:
                trace = record["people_trace"]
                self.assertEqual("bound_interval", trace["status"])
                self.assertEqual(record["raw_bgr_sha256"], trace["review_raw_bgr_sha256"])
                self.assertEqual(record["frame_sequence"], trace["frames"][0]["sequence"])
                self.assertEqual(
                    record["review_boxes"],
                    [
                        {k: row[k] for k in ("confidence", "bbox_xyxy")}
                        for row in trace["frames"][0]["detections"]
                    ],
                )
                self.assertFalse(trace["media_included"])
                self.assertIn("association_audit", trace["frames"][0])
                self.assertFalse(trace["frames"][0]["association_audit"]["accuracy_measured"])
            status = json.loads((root / "status.json").read_text())
            self.assertEqual("stopped", status["phase"])
            self.assertEqual(1, status["cameras"]["a"]["last_person_count"])
            self.assertIsNone(status["baby_supervision"]["current"])
            self.assertIsNotNone(status["baby_supervision"]["last_seen"])
            self.assertFalse(status["production_writeback"])
            self.assertFalse(status["baby_supervision"]["sleep_established"])
            for p in (root / "records").glob("*.json"):
                self.assertEqual(0, p.stat().st_mode & 0o077)

    def test_trace_cannot_be_enabled_without_explicit_tracking(self):
        for value, tracking in ((True, False), (1, True), ("yes", True)):
            with self.assertRaisesRegex(ValueError, "trace requires"):
                live_monitor.run_live_monitor(
                    streams=(),
                    detector=None,
                    local_vlm=None,
                    root=Path("/missing"),
                    review_admission=lambda: True,
                    people_tracking=tracking,
                    people_review_trace=value,
                )

    def test_transition_capture_requires_explicit_trace(self):
        for value, trace in ((True, False), (1, True), ("yes", True)):
            with self.assertRaisesRegex(ValueError, "transition capture requires"):
                live_monitor.run_live_monitor(
                    streams=(),
                    detector=None,
                    local_vlm=None,
                    root=Path("/missing"),
                    review_admission=lambda: True,
                    people_transition_capture=value,
                    people_review_trace=trace,
                )

    def test_transition_capture_binds_actual_inputs_without_adding_model_calls(self):
        from tests.test_live_transition_capture import decoded_bgr

        class Source:
            def __init__(self, source, *, mailbox, **kwargs):
                self.source = source
                self.camera, self.mailbox = source.camera_id, mailbox
                self.stop = threading.Event()
                self.thread = threading.Thread(target=self.feed)

            def feed(self):
                sequence = 0
                while not self.stop.is_set():
                    self.mailbox.put(
                        LiveFrame(
                            self.camera,
                            sequence,
                            4,
                            4,
                            datetime.now(UTC),
                            time.monotonic(),
                            bytes([sequence % 256]) * 48,
                        )
                    )
                    sequence += 1
                    self.stop.wait(0.05)

            def start(self):
                self.thread.start()

            def close(self):
                self.stop.set()
                self.thread.join(1)

            def snapshot(self):
                return {"phase": "receiving", "generation": 0}

        class Detector:
            identity = {"fixture": True}

            def start(self):
                pass

            def close(self):
                pass

            def predict(self, frames):
                return tuple(
                    () if f.sequence % 4 == 1 else (Detection(0.0, "person", 0.8, BOX),)
                    for f in frames
                )

        class VLM:
            calls = 0

            def probe_subjects(self, request):
                raise AssertionError("unexpected legacy task")

            def probe_subject_state(self, request):
                self.calls += 1
                frame = Path(request.media.frame_paths[0])
                return {
                    "version": 2,
                    "prompt_revision": subject_state_probe.PROMPT_VERSION,
                    "semantics": "unverified_model_claims",
                    "frame_id": frame.name,
                    "input_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    "coordinate_space": "submitted_image_thousandths",
                    "claims": {"people": [], "uncertainties": []},
                }

        vlm = VLM()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            with patch.object(live_monitor, "FFmpegLiveSource", Source):
                live_monitor.run_live_monitor(
                    streams=((RTSPCamera("a", "rtsp://127.0.0.1/synthetic").source(), 4, 4),),
                    detector=Detector(),
                    local_vlm=vlm,
                    root=root,
                    review_admission=lambda: True,
                    max_runtime_seconds=0.8,
                    sample_fps=None,
                    review_task="subjects",
                    people_tracking=True,
                    people_review_trace=True,
                    people_transition_capture=True,
                )
            status = json.loads((root / "status.json").read_text())
            capture = status["transition_capture"]
            self.assertEqual(0, status["detection_failures"])
            self.assertGreater(status["cameras"]["a"]["inferred_frames"], 4)
            self.assertEqual(1, capture["counts"]["completed"])
            self.assertEqual(0, capture["model_calls_added"])
            self.assertFalse(capture["worker_running"])
            self.assertNotIn("unbound_or_oversized_inputs", capture["counts"])
            self.assertEqual(vlm.calls, status["subject_probe_successes"])
            receipts = list((root / "transition-captures").glob("*/receipt.json"))
            self.assertEqual(1, len(receipts))
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(status["session_started_at"], receipt["session_started_at"])
            self.assertEqual([0, 1, 2], [f["input"]["sequence"] for f in receipt["frames"]])
            self.assertEqual([1, 0, 1], [len(f["input"]["detections"]) for f in receipt["frames"]])
            for frame in receipt["frames"]:
                pixels = decoded_bgr((receipts[0].parent / frame["filename"]).read_bytes())
                self.assertEqual(bytes([frame["input"]["sequence"]]) * 48, pixels)
                self.assertEqual(frame["input"]["raw_sha256"], hashlib.sha256(pixels).hexdigest())
            self.assertIsNone(status["baby_supervision"]["current"])
            self.assertFalse(status["production_writeback"])
