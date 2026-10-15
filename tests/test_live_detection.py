from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import threading
import time
import unittest
import zlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor.adapters.camera import RTSPCamera
from baby_monitor.adapters.live_detector import PersistentYoloDetector
from baby_monitor.adapters.live_media import FFmpegLiveSource
from baby_monitor.domain import Detection
from baby_monitor.live_detection import (
    FrameMailbox,
    LiveFrame,
    LiveObservation,
    MotionBurstSettings,
    ReviewCandidate,
    ReviewMailbox,
    ReviewTriggerPolicy,
    encode_bgr_png,
)
from baby_monitor.live_detector_protocol import MAX_HEADER, receive_packet, send_packet
from baby_monitor.live_monitor import run_live_monitor


def frame(camera="cam1", sequence=0, now=10.0, pixels=None):
    return LiveFrame(
        camera,
        sequence,
        16,
        16,
        datetime.now(UTC),
        now,
        pixels if pixels is not None else bytes([10, 20, 30]) * 256,
    )


def observation(camera="cam1", sequence=0, now=10.0, count=0):
    detections = tuple(Detection(0.0, "person", 0.8, (0.1, 0.1, 0.4, 0.6)) for _ in range(count))
    return LiveObservation(frame(camera, sequence, now), detections, now)


class LiveDetectionTests(unittest.TestCase):
    def test_all_four_cameras_are_fairly_batched_without_skipping(self):
        mailbox = FrameMailbox(("a", "b", "c", "d"))
        for sequence in range(5):
            for camera in mailbox.frames:
                mailbox.put(frame(camera, sequence))
        for sequence in range(5):
            batch = mailbox.take(10.0)
            self.assertEqual([item.camera_id for item in batch], ["a", "b", "c", "d"])
            self.assertEqual([item.sequence for item in batch], [sequence] * 4)
        self.assertTrue(all(item["discarded_frames"] == 0 for item in mailbox.snapshot().values()))

    def test_backpressure_and_stale_frames_are_counted(self):
        mailbox = FrameMailbox(("cam1",), capacity=2, max_age_seconds=0.5)
        for sequence in range(3):
            mailbox.put(frame(sequence=sequence, now=float(sequence)))
        self.assertEqual([], list(mailbox.take(3.0)))
        self.assertEqual(3, mailbox.snapshot()["cam1"]["discarded_frames"])

    def test_twenty_fps_source_selects_five_fps_without_queueing_skipped_frames(self):
        mailbox = FrameMailbox(("cam1",), sample_fps=5)
        selected = []
        for sequence in range(20):
            now = 10 + sequence / 20
            mailbox.put(frame(sequence=sequence, now=now))
            selected.extend(item.sequence for item in mailbox.take(now))
        self.assertEqual([0, 4, 8, 12, 16], selected)
        status = mailbox.snapshot()["cam1"]
        self.assertEqual(20, status["decoded_frames"])
        self.assertEqual(15, status["sampling_skipped_frames"])
        self.assertEqual(5, status["eligible_frames"])
        self.assertEqual(5, status["dispatched_frames"])
        self.assertEqual(0, status["discarded_frames"])
        self.assertEqual(0, status["queued_frames"])

    def test_four_cameras_have_independent_sampling_phases(self):
        mailbox = FrameMailbox(("a", "b", "c", "d"), sample_fps=5)
        counts = dict.fromkeys(mailbox.frames, 0)
        for sequence in range(20):
            for index, camera in enumerate(mailbox.frames):
                now = 10 + sequence / 20 + index * 0.005
                mailbox.put(frame(camera, sequence, now))
                for selected in mailbox.take(now):
                    counts[selected.camera_id] += 1
        self.assertEqual(dict.fromkeys(counts, 5), counts)
        self.assertTrue(
            all(value["sampling_skipped_frames"] == 15 for value in mailbox.snapshot().values())
        )

    def test_sampler_does_not_drift_to_four_fps_with_jitter(self):
        mailbox = FrameMailbox(("cam1",), sample_fps=5)
        selected = []
        for sequence in range(100):
            now = 10 + sequence / 20 + (0.001 if sequence % 4 == 0 else 0)
            mailbox.put(frame(sequence=sequence, now=now))
            selected.extend(mailbox.take(now))
        self.assertEqual(25, len(selected))

    def test_sampled_overload_keeps_latest_eligible_and_is_not_hidden_as_sampling(self):
        mailbox = FrameMailbox(("cam1",), sample_fps=5)
        for sequence in range(5):
            mailbox.put(frame(sequence=sequence, now=10 + sequence / 20))
        selected = mailbox.take(10.2)
        self.assertEqual([4], [item.sequence for item in selected])
        status = mailbox.snapshot()["cam1"]
        self.assertEqual(3, status["sampling_skipped_frames"])
        self.assertEqual(1, status["discarded_frames"])
        self.assertEqual(1, status["overload_discarded_frames"])
        self.assertEqual(0, status["stale_discarded_frames"])
        self.assertEqual(
            status["decoded_frames"], status["sampling_skipped_frames"] + status["eligible_frames"]
        )
        self.assertEqual(
            status["eligible_frames"],
            status["dispatched_frames"] + status["discarded_frames"] + status["queued_frames"],
        )

    def test_sampled_stale_frame_is_counted_separately(self):
        mailbox = FrameMailbox(("cam1",), sample_fps=5, max_age_seconds=0.5)
        mailbox.put(frame())
        self.assertEqual((), mailbox.take(11.0))
        status = mailbox.snapshot()["cam1"]
        self.assertEqual(1, status["stale_discarded_frames"])
        self.assertEqual(0, status["overload_discarded_frames"])
        self.assertEqual(0, status["sampling_skipped_frames"])

    def test_source_gap_does_not_cause_sampling_catchup_burst(self):
        mailbox = FrameMailbox(("cam1",), sample_fps=5)
        for sequence, now in enumerate((10.0, 100.0, 100.001, 100.002, 100.003)):
            mailbox.put(frame(sequence=sequence, now=now))
            mailbox.take(now)
        status = mailbox.snapshot()["cam1"]
        self.assertEqual(2, status["eligible_frames"])
        self.assertEqual(3, status["sampling_skipped_frames"])

    def test_reconnected_generation_is_selected_without_waiting_for_old_deadline(self):
        mailbox = FrameMailbox(("cam1",), sample_fps=5)
        mailbox.put(frame())
        mailbox.take(10.0)
        mailbox.put(replace(frame(sequence=1, now=10.01), generation=1))
        self.assertEqual([1], [item.sequence for item in mailbox.take(10.01)])

    def test_sampling_rate_rejects_nonfinite_bool_and_unbounded_values(self):
        for rate in (True, "5", 0, 0.49, 31, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                FrameMailbox(("cam1",), sample_fps=rate)

    def test_people_change_does_not_wait_for_fixed_cooldown(self):
        policy = ReviewTriggerPolicy()
        self.assertIsNotNone(policy.observe(observation()))
        self.assertIsNone(policy.observe(observation(sequence=1, now=10.1, count=1)))
        candidate = policy.observe(observation(sequence=2, now=10.2, count=1))
        self.assertEqual("person_count_changed", candidate.reason)
        self.assertIsNone(policy.observe(observation(sequence=3, now=10.3, count=1)))

    def test_missing_person_has_supplemental_review(self):
        policy = ReviewTriggerPolicy(patrol_seconds=10)
        policy.observe(observation())
        self.assertIsNone(policy.observe(observation(sequence=1, now=19.0)))
        self.assertEqual(
            "supplemental_missed_detection_review",
            policy.observe(observation(sequence=2, now=20.0)).reason,
        )

    def test_review_queue_coalesces_and_does_not_starve_other_camera(self):
        queue = ReviewMailbox(("a", "b"))
        queue.put(ReviewCandidate(observation("a"), "change", 2, 10.0))
        queue.put(ReviewCandidate(observation("b"), "baseline", 0, 10.0))
        queue.put(ReviewCandidate(observation("a", sequence=2), "change", 2, 11.0))
        queue.put(ReviewCandidate(observation("a", sequence=3), "change", 2, 12.0))
        self.assertEqual(0, queue.take().observation.frame.sequence)
        queue.put(ReviewCandidate(observation("a", sequence=3), "change", 2, 12.0))
        self.assertEqual("b", queue.take().observation.frame.camera_id)
        self.assertEqual(1, queue.coalesced)

    def test_waiting_review_is_refreshed_with_current_bound_frame(self):
        queue = ReviewMailbox(("cam1",))
        queue.put(ReviewCandidate(observation(), "baseline", 0, 10.0))
        queue.refresh(observation(sequence=20, now=11.0, count=1))
        candidate = queue.take()
        self.assertEqual(20, candidate.observation.frame.sequence)
        self.assertEqual(1, len(candidate.observation.detections))
        self.assertEqual(10.0, candidate.queued_monotonic)

    def test_busy_pair_does_not_starve_quiet_camera_review(self):
        queue = ReviewMailbox(("a", "b", "c", "d"))
        for camera in ("a", "b", "c", "d"):
            priority = 2 if camera in ("a", "b") else 0
            queue.put(ReviewCandidate(observation(camera), "baseline", priority, 10.0))
        selected = []
        for index in range(4):
            candidate = queue.take()
            camera = candidate.observation.frame.camera_id
            selected.append(camera)
            if camera in ("a", "b"):
                queue.put(ReviewCandidate(observation(camera), "change", 2, 11.0 + index))
        self.assertEqual({"a", "b", "c", "d"}, set(selected))

    def test_precision_is_explicit_and_cuda_only(self):
        detector = PersistentYoloDetector(
            command=("fixture",), model_path="fixture", manifest_path="fixture", half_precision=True
        )
        self.assertTrue(detector.settings["half_precision"])
        for device, precision in (("cpu", True), ("0", "yes")):
            with self.assertRaises(ValueError):
                PersistentYoloDetector(
                    command=("fixture",),
                    model_path="fixture",
                    manifest_path="fixture",
                    device=device,
                    half_precision=precision,
                )

    def test_rectangle_height_is_explicit_and_stride_aligned(self):
        detector = PersistentYoloDetector(
            command=("fixture",), model_path="fixture", manifest_path="fixture", image_height=384
        )
        self.assertEqual(384, detector.settings["image_height"])
        for height in (True, 319, 385, 672):
            with self.assertRaises(ValueError):
                PersistentYoloDetector(
                    command=("fixture",),
                    model_path="fixture",
                    manifest_path="fixture",
                    image_height=height,
                )

    def test_lossless_png_keeps_original_rgb_pixels(self):
        payload = encode_bgr_png(frame())
        shape, rgb, filters = self._reconstruct_png(payload)
        self.assertEqual((16, 16), shape)
        self.assertEqual(bytes([30, 20, 10]) * 256, rgb)
        self.assertEqual([1] * 16, filters)

    def test_fast_png_preserves_pixels_and_uses_unfiltered_scanlines(self):
        for width, height in ((1, 5), (2, 2), (17, 3), (4096, 1), (1, 4096)):
            with self.subTest(width=width, height=height):
                pixels = bytes((index * 127) % 256 for index in range(width * height * 3))
                source = replace(frame(), width=width, height=height, pixels=pixels)
                digest = source.sha256
                payload = encode_bgr_png(source, profile="fast")
                shape, rgb, filters = self._reconstruct_png(payload)
                expected = b"".join(
                    pixels[index : index + 3][::-1] for index in range(0, len(pixels), 3)
                )
                self.assertEqual((width, height), shape)
                self.assertEqual(expected, rgb)
                self.assertEqual([0] * height, filters)
                self.assertEqual(digest, source.sha256)
                self.assertEqual(payload, encode_bgr_png(source, profile="fast"))

    def test_unknown_png_profile_is_rejected(self):
        for profile in (None, True, 1, "jpeg", [], {}):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                encode_bgr_png(frame(), profile=profile)

    def _reconstruct_png(self, payload):
        """Independent PNG parser and reverse filter, not the encoder's code path."""
        self.assertEqual(b"\x89PNG\r\n\x1a\n", payload[:8])
        offset, compressed, shape, chunks = 8, bytearray(), None, []
        while offset < len(payload):
            size = struct.unpack(">I", payload[offset : offset + 4])[0]
            kind = payload[offset + 4 : offset + 8]
            data = payload[offset + 8 : offset + 8 + size]
            crc = struct.unpack(">I", payload[offset + 8 + size : offset + 12 + size])[0]
            self.assertEqual(zlib.crc32(kind + data), crc)
            chunks.append(kind)
            if kind == b"IHDR":
                self.assertEqual(13, size)
                width, height, *format_fields = struct.unpack(">IIBBBBB", data)
                self.assertEqual([8, 2, 0, 0, 0], format_fields)
                shape = width, height
            elif kind == b"IDAT":
                compressed.extend(data)
            elif kind == b"IEND":
                self.assertEqual(0, size)
            offset += 12 + size
        self.assertEqual(len(payload), offset)
        self.assertEqual([b"IHDR", b"IDAT", b"IEND"], chunks)
        width, height = shape
        stride = width * 3
        scanlines = zlib.decompress(compressed)
        self.assertEqual((stride + 1) * height, len(scanlines))
        rgb, filters = bytearray(), []
        for start in range(0, len(scanlines), stride + 1):
            kind = scanlines[start]
            filters.append(kind)
            row = bytearray(scanlines[start + 1 : start + stride + 1])
            self.assertIn(kind, (0, 1))
            if kind == 1:
                for index in range(3, stride):
                    row[index] = (row[index] + row[index - 3]) & 255
            rgb.extend(row)
        return shape, bytes(rgb), filters

    def test_png_sub_filter_preserves_nonuniform_pixels_and_wraparound(self):
        for width, height in ((1, 1), (1, 5), (2, 2), (17, 3), (4096, 1), (1, 4096)):
            with self.subTest(width=width, height=height):
                pixels = bytes((index * 127) % 256 for index in range(width * height * 3))
                source = replace(frame(), width=width, height=height, pixels=pixels)
                digest = source.sha256
                payload = encode_bgr_png(source)
                shape, rgb, filters = self._reconstruct_png(payload)
                expected = b"".join(
                    pixels[index : index + 3][::-1] for index in range(0, len(pixels), 3)
                )
                self.assertEqual((width, height), shape)
                self.assertEqual(expected, rgb)
                self.assertEqual([1] * height, filters)
                self.assertEqual(digest, source.sha256)
                self.assertEqual(payload, encode_bgr_png(source))

    def test_png_sub_filter_resets_at_each_row_and_preserves_channel_order(self):
        pixels = bytes([255, 0, 1, 0, 255, 254, 10, 20, 30, 255, 254, 253])
        source = replace(frame(), width=2, height=2, pixels=pixels)
        payload = encode_bgr_png(source)
        _, rgb, _ = self._reconstruct_png(payload)
        self.assertEqual(bytes([1, 0, 255, 254, 255, 0, 30, 20, 10, 253, 254, 255]), rgb)

    def test_png_parser_checks_chunk_crc(self):
        payload = bytearray(encode_bgr_png(frame()))
        payload[-1] ^= 1
        with self.assertRaises(AssertionError):
            self._reconstruct_png(bytes(payload))

    def test_protocol_preserves_payload_and_refuses_unbounded_input(self):
        sender, receiver = socket.socketpair()
        self.addCleanup(sender.close)
        self.addCleanup(receiver.close)
        send_packet(sender, {"type": "fixture"}, b"synthetic")
        self.assertEqual(({"type": "fixture"}, b"synthetic"), receive_packet(receiver))
        sender.sendall(struct.pack(">II", MAX_HEADER + 1, 0))
        with self.assertRaises(ValueError):
            receive_packet(receiver)

    def test_live_media_rejects_credentials_before_process_launch(self):
        mailbox = FrameMailbox(("cam1",))
        with self.assertRaises(ValueError):
            FFmpegLiveSource(
                # release-audit: allow=fixture-credential
                RTSPCamera("cam1", "rtsp://fixture:fixture@127.0.0.1/feed").source(),
                width=16,
                height=16,
                mailbox=mailbox,
            )

    def test_detection_keeps_running_during_slow_local_review(self):
        self._verify_monitor_concurrency(sample_fps=None)

    def test_detection_keeps_running_while_png_encoder_is_busy(self):
        self._verify_monitor_concurrency(sample_fps=None, slow_encoder=True)

    def test_sampled_detection_keeps_running_during_slow_local_review(self):
        self._verify_monitor_concurrency(sample_fps=5)

    def test_motion_sampling_keeps_running_during_slow_local_review(self):
        self._verify_monitor_concurrency(
            sample_fps=5, motion_burst=MotionBurstSettings(hold_seconds=0.1)
        )

    def test_two_slot_motion_detection_keeps_running_during_slow_local_review(self):
        self._verify_monitor_concurrency(
            sample_fps=5, motion_burst=MotionBurstSettings(hold_seconds=0.1, pending_capacity=2)
        )

    def test_alternate_hints_reach_local_review_without_changing_native_count(self):
        self._verify_monitor_concurrency(sample_fps=5, alternate_hints=True)

    def _verify_monitor_concurrency(
        self, *, sample_fps, motion_burst=None, alternate_hints=False, slow_encoder=False
    ):
        previous_umask = os.umask(0o022)
        self.addCleanup(os.umask, previous_umask)

        class Source:
            def __init__(self, source, *, width, height, mailbox):
                self.source, self.mailbox = source, mailbox
                self.stop = threading.Event()
                self.thread = threading.Thread(target=self.feed)

            def feed(self):
                sequence = 0
                while not self.stop.is_set():
                    pixels = None
                    if motion_burst is not None and 1 <= sequence <= 10:
                        pixels = bytes([250]) * (128 * 3) + bytes([10, 20, 30]) * 128
                    self.mailbox.put(
                        frame(self.source.camera_id, sequence, time.monotonic(), pixels)
                    )
                    sequence += 1
                    self.stop.wait(0.005)

            def start(self):
                self.thread.start()

            def snapshot(self):
                return {"phase": "receiving", "decoder_restarts": 0}

            def close(self):
                self.stop.set()
                self.thread.join()

        class Detector:
            identity = {"model_loads": 1, "network_enabled": False}
            counts = 0

            def start(self):
                pass

            def predict(self, frames):
                self.counts += len(frames)
                return tuple(() for _ in frames)

            def close(self):
                pass

            def review_hints_for(self, current_frame):
                if alternate_hints:
                    return (Detection(0.0, "person", 0.8, (0.1, 0.1, 0.4, 0.6)),)
                return None

        detector = Detector()
        counts_during_review = []
        counts_during_encoding = []

        def encode_on_review_thread(source):
            self.assertEqual("local-review", threading.current_thread().name)
            start = detector.counts
            result = encode_bgr_png(source)
            if slow_encoder:
                time.sleep(0.06)
                counts_during_encoding.append(detector.counts - start)
            return result

        def slow_review(**kwargs):
            start = detector.counts
            time.sleep(0.06 if sample_fps is None else 0.3)
            counts_during_review.append(detector.counts - start)
            self.assertFalse(kwargs["remote_uploads_allowed"])
            self.assertIsNone(kwargs["remote_vlm"])
            return SimpleNamespace(review_tier=SimpleNamespace(value="local_primary"))

        stored = SimpleNamespace(
            remote_used=False,
            result=SimpleNamespace(
                event_type=SimpleNamespace(value="unknown"),
                to_dict=lambda: {"event_type": "unknown"},
            ),
            decision=SimpleNamespace(to_dict=lambda: {"remote_used": False}),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            streams = tuple(
                (RTSPCamera(camera, "rtsp://127.0.0.1/feed").source(), 16, 16)
                for camera in ("a", "b", "c", "d")
            )
            with (
                patch("baby_monitor.live_monitor.FFmpegLiveSource", Source),
                patch("baby_monitor.live_monitor.process_prepared_event", side_effect=slow_review),
                patch(
                    "baby_monitor.live_monitor.encode_bgr_png", side_effect=encode_on_review_thread
                ),
                patch("baby_monitor.live_monitor.EventDatabase") as database,
            ):
                database.return_value.get_event.return_value = stored
                run_live_monitor(
                    streams=streams,
                    detector=detector,
                    local_vlm=object(),
                    root=root,
                    review_admission=lambda: True,
                    # Alternate checks now require distinct frames spanning >=1 second.
                    max_runtime_seconds=2.0
                    if alternate_hints
                    else 0.3
                    if sample_fps is None
                    else 0.8,
                    sample_fps=sample_fps,
                    motion_burst=motion_burst,
                    batch_coalesce_seconds=0.01 if motion_burst else 0,
                )
            status = json.loads((root / "status.json").read_text())
            records = [json.loads(path.read_text()) for path in (root / "records").glob("*.json")]
            self.assertTrue(records)
            for record in records:
                self.assertIn(record["evidence_role"], ("trigger_frame", "latest_scene"))
                self.assertGreaterEqual(record["queue_wait_seconds"], 0)
                self.assertGreaterEqual(record["evidence_age_at_selection_seconds"], 0)
                self.assertGreaterEqual(
                    record["evidence_age_at_completion_seconds"],
                    record["evidence_age_at_selection_seconds"],
                )
            if alternate_hints:
                self.assertTrue(
                    any(record["detector_view"] == "rotation_180" for record in records)
                )
                self.assertTrue(
                    all(row["last_person_count"] == 0 for row in status["cameras"].values())
                )
                self.assertTrue(
                    all(
                        row["alternate_view_review_triggers"] > 0
                        for row in status["cameras"].values()
                    )
                )
            self.assertEqual(0o022, os.umask(0o022))
            self.assertEqual(0, (root / "debug-events.sqlite3").stat().st_mode & 0o077)
            self.assertTrue(
                all(path.stat().st_mode & 0o077 == 0 for path in (root / "frames").glob("*.png"))
            )
        self.assertTrue(counts_during_review)
        if slow_encoder:
            self.assertTrue(any(count > 4 for count in counts_during_encoding))
        self.assertTrue(
            all(count > (4 if sample_fps is None else 0) for count in counts_during_review[:-1])
        )
        self.assertEqual(4, len(status["cameras"]))
        self.assertEqual(status["inference_batches"], sum(status["inference_batch_sizes"].values()))
        if sample_fps is None:
            self.assertTrue(
                all(camera["inferred_frames"] > 20 for camera in status["cameras"].values())
            )
        else:
            self.assertEqual(5, status["sampling"]["target_fps_per_camera"])
            self.assertEqual(motion_burst is not None, status["sampling"]["adaptive_burst_enabled"])
            for camera in status["cameras"].values():
                self.assertTrue(
                    2 <= camera["inferred_frames"] <= (12 if motion_burst or alternate_hints else 6)
                )
                if motion_burst is not None:
                    self.assertGreater(camera["motion_burst_episodes"], 0)
                self.assertGreater(camera["sampling_skipped_frames"], 0)
                self.assertEqual(
                    camera["decoded_frames"],
                    camera["sampling_skipped_frames"] + camera["eligible_frames"],
                )
        self.assertGreater(status["review_successes"], 0)


if __name__ == "__main__":
    unittest.main()
