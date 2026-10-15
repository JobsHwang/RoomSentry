from __future__ import annotations

import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor.adapters.camera import RTSPCamera
from baby_monitor.adapters.live_media import FFmpegLiveSource, is_uniform_gray_bgr
from baby_monitor.domain import Detection
from baby_monitor.live_detection import FrameMailbox, LiveObservation, MotionBurstSettings
from baby_monitor.live_monitor import _subject_projection, run_live_monitor
from baby_monitor.live_people import LivePeople
from tests.test_live_people import BASE, BOX, claims

GRAY = bytes([130]) * (16 * 16 * 3)
VISIBLE = bytes([130, 130, 129]) * (16 * 16)


class LiveImageQualityTests(unittest.TestCase):
    def source(self, camera="a", *, enabled=True, mailbox=None):
        mailbox = mailbox if mailbox is not None else FrameMailbox((camera,))
        source = FFmpegLiveSource(
            RTSPCamera(camera, "rtsp://127.0.0.1:8554/fixture").source(),
            width=16,
            height=16,
            mailbox=mailbox,
            reject_uniform_frames=enabled,
        )
        return source, mailbox

    def admit(self, source, payload, elapsed):
        return source._admit_payload(payload, BASE + timedelta(seconds=elapsed), float(elapsed))

    def test_exact_gray_black_white_are_information_free_not_person_absence(self):
        for value in (0, 130, 255):
            self.assertTrue(is_uniform_gray_bgr(bytes([value]) * 768))
        self.assertFalse(is_uniform_gray_bgr(VISIBLE))
        self.assertFalse(is_uniform_gray_bgr(bytes([0, 0, 255]) * 256))

    def test_unsampled_single_byte_difference_must_not_be_rejected(self):
        for offset in (0, 1, 2, 11, 384, 767):
            changed = bytearray(GRAY)
            changed[offset] = 129
            self.assertFalse(is_uniform_gray_bgr(bytes(changed)))

    def test_malformed_payloads_reject_without_mutating_source(self):
        source, mailbox = self.source()
        for payload in (b"", b"a", GRAY[:-1]):
            with self.assertRaises(ValueError):
                self.admit(source, payload, 0)
        with self.assertRaises(ValueError):
            source._admit_payload(GRAY, BASE.replace(tzinfo=None), 0)
        with self.assertRaises(ValueError):
            source._admit_payload(GRAY, BASE, float("nan"))
        for payload in (b"", b"x", bytearray(GRAY), None):
            with self.assertRaises(ValueError):
                is_uniform_gray_bgr(payload)
        self.assertEqual(0, source.sequence)
        self.assertEqual(0, source.decoded_payloads)
        self.assertEqual((), mailbox.take(0))

    def test_default_off_preserves_uniform_input_and_legacy_snapshot(self):
        source, mailbox = self.source(enabled=False)
        self.assertTrue(self.admit(source, GRAY, 0))
        self.assertEqual(GRAY, mailbox.take(0)[0].pixels)
        self.assertEqual(0, source.generation)
        self.assertNotIn("image_quality", source.snapshot())

    def test_guard_requires_explicit_boolean_before_starting_resources(self):
        for value in (1, None, "true"):
            with self.assertRaises(ValueError):
                self.source(enabled=value)
            with self.assertRaises(ValueError):
                run_live_monitor(
                    reject_uniform_frames=value,
                    streams=(),
                    detector=None,
                    local_vlm=None,
                    root=None,
                    review_admission=None,
                )

    def test_unusable_input_has_no_zero_detection_or_mailbox_admission(self):
        source, mailbox = self.source()
        self.assertFalse(self.admit(source, GRAY, 0))
        self.assertEqual((), mailbox.take(0))
        self.assertEqual(0, mailbox.snapshot()["a"]["decoded_frames"])
        state = source.snapshot()
        self.assertEqual("image_unavailable", state["phase"])
        self.assertEqual(0, state["decoder_restarts"])
        self.assertFalse(state["image_quality"]["absence_of_people_established"])
        self.assertEqual(1, state["image_quality"]["decoded_payloads"])

    def test_pending_old_frames_purged_other_camera_kept(self):
        mailbox = FrameMailbox(("a", "b"))
        a, _ = self.source("a", mailbox=mailbox)
        b, _ = self.source("b", mailbox=mailbox)
        self.admit(a, VISIBLE, 0)
        self.admit(b, VISIBLE, 0)
        self.admit(a, VISIBLE, 0.1)
        self.admit(a, GRAY, 0.2)
        batch = mailbox.take(0.2)
        self.assertEqual(["b"], [f.camera_id for f in batch])
        row = mailbox.snapshot()["a"]
        self.assertEqual(2, row["source_unavailable_discarded_frames"])
        self.assertEqual(2, row["discarded_frames"])
        self.assertEqual(0, row["stale_discarded_frames"])
        self.assertEqual(0, row["overload_discarded_frames"])

    def test_repeated_gray_has_one_boundary_recovery_new_generation(self):
        source, mailbox = self.source()
        for elapsed in (0, 0.1, 0.2):
            self.assertFalse(self.admit(source, GRAY, elapsed))
        self.assertTrue(self.admit(source, VISIBLE, 0.3))
        frame = mailbox.take(0.3)[0]
        self.assertEqual((3, 1), (frame.sequence, frame.generation))
        self.assertEqual("receiving", source.snapshot()["phase"])
        self.assertIsNone(source.snapshot()["image_quality"]["reason"])
        self.admit(source, GRAY, 0.4)
        self.assertEqual(2, source.generation)
        quality = source.snapshot()["image_quality"]
        self.assertEqual(4, quality["unusable_frames"])
        self.assertEqual(2, quality["quality_generation_boundaries"])
        self.assertEqual(0, source.failures)

    def test_scheduled_recovery_resets_deadline_but_preserves_motion_counters(self):
        mailbox = FrameMailbox(("a",), sample_fps=5, motion_burst=MotionBurstSettings())
        source, _ = self.source(mailbox=mailbox)
        self.admit(source, VISIBLE, 0)
        state = mailbox.motion.state["a"]
        state.trigger_frames, state.episodes, state.burst_until = 4, 2, 3
        self.admit(source, GRAY, 0.01)
        self.admit(source, VISIBLE, 0.02)
        frame = mailbox.take(0.02)[0]
        self.assertEqual(2, frame.sequence)
        row = mailbox.snapshot(0.02)["a"]
        self.assertFalse(row["sampling_burst_active"])
        self.assertEqual(4, row["motion_trigger_frames"])
        self.assertEqual(2, row["motion_burst_episodes"])

    def test_purge_failure_stays_unavailable_and_cannot_admit_bad_pixels(self):
        source, mailbox = self.source()
        with (
            patch.object(mailbox, "discard_unavailable_source", side_effect=RuntimeError),
            self.assertRaises(RuntimeError),
        ):
            self.admit(source, GRAY, 0)
        self.assertEqual("image_unavailable", source.snapshot()["phase"])
        self.assertEqual((), mailbox.take(0))

    def test_real_decode_loop_applies_guard_and_closes_partial_input(self):
        source, mailbox = self.source()
        payloads = iter((GRAY, GRAY, VISIBLE, b"partial"))

        def read(size):
            self.assertEqual(768, size)
            payload = next(payloads)
            if len(payload) != size:
                source.stop.set()
            return payload

        closed = []
        process = SimpleNamespace(
            stdout=SimpleNamespace(read=read, close=lambda: closed.append(True)),
            poll=lambda: 0,
            wait=lambda timeout: 0,
        )
        with patch("baby_monitor.adapters.live_media.subprocess.Popen", return_value=process):
            source._run()
        self.assertEqual([True], closed)
        self.assertEqual("stopped", source.snapshot()["phase"])
        self.assertEqual(0, source.failures)
        self.assertEqual(3, source.decoded_payloads)
        self.assertEqual(2, source.unusable_frames)
        batch = mailbox.take(10**12)
        self.assertEqual((), batch)  # expired input, not an immortal retained result
        self.assertEqual(1, mailbox.snapshot()["a"]["decoded_frames"])

    def test_completed_subject_claim_hidden_during_gap_and_after_recovery(self):
        source, _ = self.source()
        attempt = {
            "status": "analyzed",
            "frame_received_at": BASE.isoformat(),
            "received_monotonic": 0.0,
            "frame_generation": 0,
            "completed_at": BASE.isoformat(),
        }
        value = claims()["people"]
        self.admit(source, GRAY, 0.1)
        for elapsed, phase in ((0.1, "image_unavailable"), (0.2, "receiving")):
            if phase == "receiving":
                self.admit(source, VISIBLE, elapsed)
            context = source.snapshot()
            projected = _subject_projection(
                attempt,
                None,
                value,
                now=elapsed,
                now_utc=BASE + timedelta(seconds=elapsed),
                source_phase=context["phase"],
                source_generation=context["generation"],
                monitor_phase="detecting",
            )
            self.assertIsNone(projected["subject_claims"])

    def test_inflight_old_review_cannot_restore_roles_across_quality_generation(self):
        source, mailbox = self.source()
        people = LivePeople(("a",), auxiliary=True)
        old = None
        for elapsed in (0, 1):
            self.admit(source, VISIBLE, elapsed)
            frame = mailbox.take(elapsed)[0]
            old = LiveObservation(frame, (Detection(0, "person", 0.8, BOX),), float(elapsed))
            people.observe(old)
            people.apply_review(
                old,
                claims(),
                evidence_ref="synthetic.png",
                now=float(elapsed),
                now_utc=BASE + timedelta(seconds=elapsed),
            )
        self.assertIsNotNone(
            people.project(
                now=1,
                now_utc=BASE + timedelta(seconds=1),
                sources={"a": source.snapshot()},
                monitor_phase="detecting",
            )["current"]
        )
        self.admit(source, GRAY, 1.1)
        self.assertIsNone(
            people.project(
                now=1.1,
                now_utc=BASE + timedelta(seconds=1.1),
                sources={"a": source.snapshot()},
                monitor_phase="detecting",
            )["current"]
        )
        self.admit(source, VISIBLE, 1.2)
        frame = mailbox.take(1.2)[0]
        people.observe(LiveObservation(frame, (Detection(0, "person", 0.8, BOX),), 1.2))
        rejected = people.apply_review(
            old,
            claims(),
            evidence_ref="synthetic.png",
            now=1.2,
            now_utc=BASE + timedelta(seconds=1.2),
        )
        self.assertEqual("stale_or_unbound_review", rejected["status"])
        self.assertIsNone(
            people.project(
                now=1.2,
                now_utc=BASE + timedelta(seconds=1.2),
                sources={"a": source.snapshot()},
                monitor_phase="detecting",
            )["current"]
        )


if __name__ == "__main__":
    unittest.main()
