"""Exact synthetic input evidence; never household/model labels as truth."""

import copy
import hashlib
import json
import os
import struct
import tempfile
import threading
import unittest
import zlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from baby_monitor.live_people import LivePeople
from baby_monitor.live_transition_capture import (
    COOLDOWN_SECONDS,
    MAX_CAPTURE_ATTEMPTS,
    REASON_LIMITS,
    CaptureInput,
    TransitionBundle,
    TransitionCaptureWorker,
    TransitionSelector,
    write_transition_bundle,
)
from tests.test_live_people import BOX, OTHER, observation


def decoded_bgr(raw):
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    offset, data, width, height = 8, bytearray(), None, None
    while offset < len(raw):
        size = struct.unpack(">I", raw[offset : offset + 4])[0]
        kind = raw[offset + 4 : offset + 8]
        payload = raw[offset + 8 : offset + 8 + size]
        assert (
            zlib.crc32(kind + payload)
            == struct.unpack(">I", raw[offset + 8 + size : offset + 12 + size])[0]
        )
        if kind == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        elif kind == b"IDAT":
            data.extend(payload)
        offset += size + 12
    expanded = zlib.decompress(data)
    rows = []
    for i in range(height):
        start = i * (width * 3 + 1)
        assert expanded[start] == 0
        rows.append(expanded[start + 1 : start + 1 + width * 3])
    rgb = b"".join(rows)
    bgr = bytearray(len(rgb))
    bgr[0::3], bgr[1::3], bgr[2::3] = rgb[2::3], rgb[1::3], rgb[0::3]
    return bytes(bgr)


class TransitionCaptureTests(unittest.TestCase):
    def setUp(self):
        self.people = LivePeople(("a", "b"), auxiliary=True, association_diagnostics=True)

    def item(self, sequence, *, elapsed, boxes=(BOX,), camera="a", generation=0, view="original"):
        obs = observation(
            sequence, elapsed=elapsed, boxes=boxes, camera=camera, generation=generation
        )
        if view == "original":
            self.people.observe(obs)
        else:
            self.people.observe_auxiliary(obs, view=view)
        trace = self.people.review_trace(obs, view=view)
        return CaptureInput.bind(obs, trace, view)

    def bundle(self):
        return TransitionBundle(
            "positive_to_zero",
            self.item(0, elapsed=0),
            self.item(1, elapsed=0.1, boxes=()),
            self.item(2, elapsed=0.2),
            "observed_followup",
        )

    def test_one_two_input_zero_run_has_exact_observed_followup(self):
        selector = TransitionSelector(("a",))
        items = [
            self.item(i, elapsed=i * 0.1, boxes=() if i in (1, 2) else (BOX,)) for i in range(4)
        ]
        outputs = [selector.observe(item) for item in items]
        self.assertEqual([(), (), ()], outputs[:3])
        bundle = outputs[3][0]
        self.assertEqual([0, 1, 3], [i.observation.frame.sequence for i in bundle.inputs])
        self.assertEqual("observed_followup", bundle.end_reason)
        self.assertEqual(1, selector.attempts)
        self.assertEqual((), bundle.at.observation.detections)
        # A restored anonymous ID is not inherited role evidence.
        self.assertEqual(
            "detection_gap", next(iter(bundle.after.metadata["detections"][0]["epoch_causes"]))
        )

    def test_two_to_one_and_dormant_competition_are_separate_diagnostics(self):
        selector = TransitionSelector(("a",))
        first = self.item(0, elapsed=0, boxes=(BOX, OTHER))
        second = self.item(1, elapsed=0.1)
        third = self.item(2, elapsed=0.2)
        for item in (first, second):
            self.assertEqual((), selector.observe(item))
        self.assertEqual("two_to_one", selector.observe(third)[0].reason)
        # This seed describes competing edges, not an adjudicated wrong person.
        selector = TransitionSelector(("a",))
        metadata = copy.deepcopy(third.metadata)
        metadata["association_audit"]["detections"][0].update(
            selected_source="dormant_retained_track", eligible_previous_input_track_count=1
        )
        altered = CaptureInput(third.observation, metadata)
        selector.observe(second)
        selector.observe(altered)
        self.assertEqual("dormant_with_eligible_recent_edge", selector.finish()[0].reason)

    def test_deadline_gap_view_switch_and_stop_are_censored_not_recovery(self):
        for mode in ("deadline", "generation", "clock", "view", "stop"):
            with self.subTest(mode=mode):
                self.people = LivePeople(("a", "b"), auxiliary=True, association_diagnostics=True)
                selector = TransitionSelector(("a",))
                view = "rotation_180" if mode == "view" else "original"
                selector.observe(self.item(0, elapsed=0, view=view))
                selector.observe(self.item(1, elapsed=0.1, boxes=(), view=view))
                if mode == "stop":
                    bundle = selector.finish()[0]
                else:
                    follow = self.item(
                        2,
                        elapsed=2.5 if mode == "clock" else 1.2 if mode == "deadline" else 0.2,
                        generation=1 if mode == "generation" else 0,
                        view="rotation_180_scale_fallback" if mode == "view" else view,
                    )
                    bundle = selector.observe(follow)[0]
                self.assertIsNone(bundle.after)
                self.assertNotEqual("observed_followup", bundle.end_reason)

    def test_duplicate_old_input_does_not_replace_pending_or_previous(self):
        selector = TransitionSelector(("a",))
        before = self.item(0, elapsed=0)
        at = self.item(1, elapsed=0.1, boxes=())
        selector.observe(before)
        selector.observe(at)
        prior = (dict(selector.previous), dict(selector.pending))
        self.assertEqual((), selector.observe(before))
        self.assertEqual((), selector.observe(at))
        self.assertEqual(prior, (selector.previous, selector.pending))

    def test_camera_native_auxiliary_isolation_and_pause_discards(self):
        selector = TransitionSelector(("a", "b"))
        for camera, view in (("a", "original"), ("a", "rotation_180"), ("b", "original")):
            selector.observe(self.item(0, elapsed=0, camera=camera, view=view))
            selector.observe(self.item(1, elapsed=0.1, boxes=(), camera=camera, view=view))
        self.assertEqual(3, len(selector.pending))
        selector.gap()
        self.assertEqual([], selector.finish())
        self.assertEqual(3, selector.counts["pending_discarded_on_pause"])

    def test_cooldown_and_hard_attempt_limit(self):
        selector = TransitionSelector(("a",))
        selector.observe(self.item(0, elapsed=0))
        selector.observe(self.item(1, elapsed=0.1, boxes=()))
        selector.observe(self.item(2, elapsed=0.2))
        selector.observe(self.item(3, elapsed=0.3, boxes=()))
        self.assertEqual(1, selector.attempts)
        self.assertEqual(1, selector.counts["cooldown_rejections"])
        selector.attempts = MAX_CAPTURE_ATTEMPTS
        selector.cooldowns.clear()
        selector.observe(self.item(4, elapsed=0.4))
        selector.observe(self.item(5, elapsed=0.5, boxes=()))
        self.assertEqual(1, selector.counts["attempt_limit_rejections"])
        self.assertEqual(15.0, COOLDOWN_SECONDS)

    def test_sha_original_boxes_view_and_mutable_bytes_are_rejected(self):
        obs = observation(0, elapsed=0)
        self.people.observe(obs)
        trace = self.people.review_trace(obs)
        for field, value in (("raw_sha256", "0" * 64), ("view", "rotation_180"), ("sequence", 99)):
            altered = copy.deepcopy(trace)
            altered["frames"][0][field] = value
            with self.assertRaises(ValueError):
                CaptureInput.bind(obs, altered, "original")
        with self.assertRaises(ValueError):
            CaptureInput.bind(replace(obs, detections=()), trace, "original")
        with self.assertRaises(ValueError):
            CaptureInput.bind(
                replace(obs, frame=replace(obs.frame, pixels=bytearray(obs.frame.pixels))),
                trace,
                "original",
            )
        item = CaptureInput.bind(obs, trace, "original")
        trace["frames"][0]["detections"].clear()
        self.assertEqual(1, len(item.metadata["detections"]))

    def test_dormant_cooldown_does_not_hide_zero_transition(self):
        selector = TransitionSelector(("a",))
        before = self.item(0, elapsed=0)
        competition = self.item(1, elapsed=0.1)
        metadata = copy.deepcopy(competition.metadata)
        metadata["association_audit"]["detections"][0].update(
            selected_source="dormant_retained_track", eligible_previous_input_track_count=1
        )
        selector.observe(before)
        selector.observe(CaptureInput(competition.observation, metadata))
        zero = self.item(2, elapsed=0.2, boxes=())
        self.assertEqual("dormant_with_eligible_recent_edge", selector.observe(zero)[0].reason)
        self.assertEqual(2, selector.attempts)
        restored = self.item(3, elapsed=0.3)
        self.assertEqual("positive_to_zero", selector.observe(restored)[0].reason)
        self.assertEqual(1, selector.attempts_by_reason["positive_to_zero"])

    def test_reason_quota_leaves_zero_capacity_and_survives_pause(self):
        self.assertEqual(MAX_CAPTURE_ATTEMPTS, sum(REASON_LIMITS.values()))
        selector = TransitionSelector(("a",))
        selector.attempts = REASON_LIMITS["two_to_one"]
        selector.attempts_by_reason["two_to_one"] = REASON_LIMITS["two_to_one"]
        selector.observe(self.item(0, elapsed=0, boxes=(BOX, OTHER)))
        selector.observe(self.item(1, elapsed=0.1))
        self.assertEqual(1, selector.counts["reason_limit_rejections_two_to_one"])
        self.assertFalse(selector.pending)
        selector.observe(self.item(2, elapsed=0.2, boxes=()))
        self.assertEqual(REASON_LIMITS["two_to_one"] + 1, selector.attempts)
        selector.gap()
        self.assertEqual(REASON_LIMITS["two_to_one"], selector.attempts_by_reason["two_to_one"])
        self.assertEqual(1, selector.attempts_by_reason["positive_to_zero"])
        self.assertFalse(selector.pending)

    @unittest.skipUnless(os.name == "posix", "private local evidence requires POSIX")
    def test_private_png_roundtrip_hash_and_receipt_last(self):
        bundle = self.bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            size = write_transition_bundle(root, bundle, "2026-10-12T00:00:00+00:00")
            event = next(root.iterdir())
            receipt = json.loads((event / "receipt.json").read_text())
            self.assertEqual(3, len(receipt["frames"]))
            self.assertFalse(receipt["human_ground_truth"])
            self.assertFalse(receipt["production_writeback"])
            self.assertFalse(receipt["role_inheritance"])
            self.assertEqual(0, receipt["model_calls_added"])
            self.assertEqual(size, sum(f["size_bytes"] for f in receipt["frames"]))
            for entry, item in zip(receipt["frames"], bundle.inputs, strict=True):
                png = (event / entry["filename"]).read_bytes()
                self.assertEqual(hashlib.sha256(png).hexdigest(), entry["png_sha256"])
                self.assertEqual(item.observation.frame.pixels, decoded_bgr(png))
                self.assertEqual(0o600, (event / entry["filename"]).stat().st_mode & 0o777)
            self.assertEqual(0o700, event.stat().st_mode & 0o777)
            self.assertEqual(0o600, (event / "receipt.json").stat().st_mode & 0o777)
            with (
                patch(
                    "baby_monitor.live_transition_capture.uuid.uuid4",
                    return_value=type("ID", (), {"hex": receipt["event_id"]})(),
                ),
                self.assertRaises(FileExistsError),
            ):
                write_transition_bundle(root, bundle, "same")

    @unittest.skipUnless(os.name == "posix", "private local evidence requires POSIX")
    def test_unsafe_output_and_failed_write_never_publish_receipt(self):
        bundle = self.bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o755)
            with self.assertRaises(ValueError):
                write_transition_bundle(root, bundle, "session")
            root.chmod(0o700)
            with (
                patch(
                    "baby_monitor.live_transition_capture.encode_bgr_png",
                    side_effect=ValueError("synthetic"),
                ),
                self.assertRaises(ValueError),
            ):
                write_transition_bundle(root, bundle, "session")
            self.assertFalse(list(root.glob("*/receipt.json")))
            link = root / "link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(OSError):
                write_transition_bundle(link, bundle, "session")

    def test_nonblocking_queue_limit_pause_discard_and_worker_storage_failure(self):
        bundle = self.bundle()
        worker = TransitionCaptureWorker(
            ("a",), root=Path("/unused"), session_started_at="session", admission=lambda: False
        )
        for _ in range(3):
            worker._submit(bundle)
        self.assertEqual(2, worker.queue.qsize())
        self.assertEqual(1, worker.counts["queue_full_rejections"])
        worker.gap()
        self.assertEqual(0, worker.queue.qsize())
        self.assertEqual(2, worker.counts["queued_discarded_on_pause"])
        worker._submit(bundle)
        worker.start()
        worker.close()
        self.assertEqual(1, worker.counts["storage_deferred"])
        self.assertEqual(0, worker.counts["completed"])
        failing = TransitionCaptureWorker(
            ("a",),
            root=Path("/unused"),
            session_started_at="session",
            admission=lambda: True,
            writer=lambda *_: (_ for _ in ()).throw(OSError("private-error")),
        )
        failing._submit(bundle)
        failing.start()
        failing.close()
        self.assertTrue(failing.snapshot()["disabled_after_error"])
        self.assertEqual(1, failing.counts["write_failures"])
        self.assertNotIn("private-error", json.dumps(failing.snapshot()))

    def test_writer_thread_is_separate_and_submissions_do_not_wait_for_io(self):
        bundle = self.bundle()
        entered, release = threading.Event(), threading.Event()
        caller = threading.get_ident()
        threads = []

        def writer(*_):
            threads.append(threading.get_ident())
            entered.set()
            release.wait(2)
            return 99

        worker = TransitionCaptureWorker(
            ("a",),
            root=Path("/unused"),
            session_started_at="session",
            admission=lambda: True,
            writer=writer,
        )
        worker.start()
        worker._submit(bundle)
        self.assertTrue(entered.wait(1))
        worker._submit(bundle)
        self.assertEqual(1, worker.queue.qsize())
        release.set()
        worker.close()
        self.assertEqual(2, worker.counts["completed"])
        self.assertEqual(198, worker.counts["png_bytes_written"])
        self.assertTrue(all(t != caller for t in threads))
