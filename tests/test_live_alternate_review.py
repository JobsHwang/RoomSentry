from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from baby_monitor.adapters.live_detector import PersistentYoloDetector
from baby_monitor.live_detection import AlternateViewReviewPolicy, ReviewMailbox
from baby_monitor.local_runners import live_yolo
from tests.test_live_detection import frame, observation


class LiveAlternateReviewTests(unittest.TestCase):
    def test_weak_frames_never_accumulate_trigger_votes(self):
        policy = AlternateViewReviewPolicy(minimum_frame_confidence=0.4)
        for i in range(4):
            obs = observation(sequence=i, now=10 + i, count=1)
            weak = replace(
                obs, detections=tuple(replace(d, confidence=0.385) for d in obs.detections)
            )
            self.assertFalse(policy.frame_eligible(weak.detections))
            self.assertIsNone(policy.observe(weak))

    def test_strong_frame_keeps_weaker_second_candidate(self):
        policy = AlternateViewReviewPolicy(minimum_frame_confidence=0.4)
        obs = observation(count=2)
        pair = (
            replace(obs.detections[0], confidence=0.53),
            replace(obs.detections[1], confidence=0.36),
        )
        obs = replace(obs, detections=pair)
        self.assertIsNone(policy.observe(obs))
        second = replace(
            obs, frame=replace(obs.frame, sequence=1, received_monotonic=11), completed_monotonic=11
        )
        result = policy.observe(second)
        self.assertEqual(pair, result.observation.detections)
        self.assertEqual(0.36, result.observation.detections[1].confidence)

    def test_weak_gap_breaks_two_check_confirmation(self):
        policy = AlternateViewReviewPolicy(minimum_frame_confidence=0.4)
        strong = observation(count=1)
        policy.observe(strong)
        weak = replace(
            observation(sequence=1, now=11, count=1),
            detections=(replace(strong.detections[0], confidence=0.39),),
        )
        self.assertIsNone(policy.observe(weak))
        self.assertIsNone(policy.observe(observation(sequence=2, now=12, count=1)))
        self.assertIsNotNone(policy.observe(observation(sequence=3, now=13, count=1)))

    def test_frame_gate_boundary_and_invalid_configuration(self):
        policy = AlternateViewReviewPolicy(minimum_frame_confidence=0.4)
        detection = replace(observation(count=1).detections[0], confidence=0.4)
        self.assertTrue(policy.frame_eligible((detection,)))
        self.assertFalse(policy.frame_eligible(()))
        for value in (True, -0.1, 1.1, float("nan"), float("inf"), "0.4"):
            with self.assertRaises(ValueError):
                AlternateViewReviewPolicy(minimum_frame_confidence=value)

    def test_two_checks_trigger_once_and_keep_view_when_queue_coalesces(self):
        policy = AlternateViewReviewPolicy()
        queue = ReviewMailbox(("cam1",))
        self.assertIsNone(policy.observe(observation(now=10, count=1)))
        candidate = policy.observe(observation(sequence=1, now=11, count=1))
        self.assertEqual("alternate_view_person_hint", candidate.reason)
        self.assertEqual("rotation_180", candidate.evidence_view)
        queue.put(candidate)
        queue.refresh(observation(sequence=2, now=12, count=0))
        self.assertEqual("rotation_180", queue.take().evidence_view)
        self.assertIsNone(policy.observe(observation(sequence=3, now=13, count=1)))

    def test_gap_and_empty_checks_cannot_combine_into_confirmation(self):
        for middle in (
            observation(sequence=1, now=11, count=0),
            observation(sequence=1, now=15, count=1),
        ):
            policy = AlternateViewReviewPolicy()
            policy.observe(observation(now=10, count=1))
            self.assertIsNone(policy.observe(middle))
            if not middle.detections:
                self.assertIsNone(policy.observe(observation(sequence=2, now=12, count=1)))

    def test_hint_retrigger_is_bounded_and_generation_is_independent(self):
        policy = AlternateViewReviewPolicy()
        policy.observe(observation(now=10, count=1))
        self.assertIsNotNone(policy.observe(observation(sequence=1, now=11, count=1)))
        policy.observe(observation(sequence=2, now=12, count=0))
        policy.observe(observation(sequence=3, now=13, count=1))
        self.assertIsNone(policy.observe(observation(sequence=4, now=14, count=1)))
        fresh = observation(sequence=5, now=15, count=1)
        fresh = replace(fresh, frame=replace(fresh.frame, generation=1))
        self.assertIsNone(policy.observe(fresh))

    def test_protocol_keeps_hints_out_of_native_detections_and_binds_frame(self):
        detector = PersistentYoloDetector(
            command=("fixture",),
            model_path="fixture",
            manifest_path="fixture",
            review_rotation_180=True,
        )
        detector.channel = Mock()
        source = frame()
        row = {
            "camera_id": source.camera_id,
            "sequence": source.sequence,
            "sha256": source.sha256,
            "detections": [],
            "review_checked": True,
            "review_detections": [{"confidence": 0.8, "bbox_xyxy": [0.1, 0.2, 0.5, 0.8]}],
        }
        with (
            patch("baby_monitor.adapters.live_detector.send_packet"),
            patch(
                "baby_monitor.adapters.live_detector.receive_packet",
                return_value=({"type": "predictions", "frames": [row]}, b""),
            ),
        ):
            self.assertEqual(((),), detector.predict((source,)))
            self.assertEqual(1, len(detector.review_hints_for(source)))
            self.assertIsNone(detector.review_hints_for(replace(source, sequence=1)))
            row["review_checked"] = False
            detector.predict((source,))
            self.assertIsNone(detector.review_hints_for(source))

    def test_disabled_client_rejects_unsolicited_hint(self):
        detector = PersistentYoloDetector(
            command=("fixture",), model_path="fixture", manifest_path="fixture"
        )
        detector.channel = Mock()
        source = frame()
        row = {
            "camera_id": source.camera_id,
            "sequence": source.sequence,
            "sha256": source.sha256,
            "detections": [],
            "review_checked": True,
            "review_detections": [],
        }
        with (
            patch("baby_monitor.adapters.live_detector.send_packet"),
            patch(
                "baby_monitor.adapters.live_detector.receive_packet",
                return_value=({"type": "predictions", "frames": [row]}, b""),
            ),
            self.assertRaises(RuntimeError),
        ):
            detector.predict((source,))

    def test_runner_maps_180_boxes_and_limits_extra_checks_to_one_per_second(self):
        self.runner_case()

    def test_empty_views_run_fallback_and_map_coordinates(self):
        self.runner_case(fallback=True)

    def test_nonempty_alternate_view_is_not_replaced_by_fallback(self):
        self.runner_case(fallback=True, alternate_positive=True)

    def test_fallback_requires_explicit_admitted_configuration(self):
        for settings in (
            {"review_rotation_180": False, "image_height": 384},
            {"review_rotation_180": True, "image_height": 640},
            {"review_rotation_180": True, "image_height": 384, "image_size": 960},
        ):
            with self.assertRaises(ValueError):
                PersistentYoloDetector(
                    command=("fixture",),
                    model_path="fixture",
                    manifest_path="fixture",
                    review_empty_scale_fallback=True,
                    **settings,
                )

    def test_fallback_marker_is_bound_and_opt_in(self):
        source = frame()
        row = {
            "camera_id": source.camera_id,
            "sequence": source.sequence,
            "sha256": source.sha256,
            "detections": [],
            "review_checked": True,
            "review_detections": [],
            "review_fallback_checked": True,
        }
        for enabled in (False, True):
            detector = PersistentYoloDetector(
                command=("fixture",),
                model_path="fixture",
                manifest_path="fixture",
                image_height=384,
                review_rotation_180=True,
                review_empty_scale_fallback=enabled,
            )
            detector.channel = Mock()
            with (
                patch("baby_monitor.adapters.live_detector.send_packet"),
                patch(
                    "baby_monitor.adapters.live_detector.receive_packet",
                    return_value=({"type": "predictions", "frames": [row]}, b""),
                ),
            ):
                if not enabled:
                    with self.assertRaises(RuntimeError):
                        detector.predict((source,))
                else:
                    detector.predict((source,))
                    self.assertTrue(detector.review_fallback_for(source))
                    self.assertFalse(detector.review_fallback_for(replace(source, generation=1)))
                    detector.close()
                    self.assertFalse(detector.review_fallback_for(source))

    def runner_case(self, *, fallback=False, alternate_positive=False):
        class FakeImage:
            def reshape(self, *args):
                return self

            def __getitem__(self, key):
                return self

        class Value:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

            def tolist(self):
                return self.value

        empty = SimpleNamespace(boxes=None)
        positive = SimpleNamespace(
            boxes=[SimpleNamespace(conf=Value(0.8), xyxyn=[Value([0.1, 0.2, 0.4, 0.6])])]
        )
        calls = []

        def predict(images, **kwargs):
            calls.append(len(images))
            positive_call = (10 if alternate_positive else 11) if fallback else 6
            if fallback and len(calls) == 11:
                self.assertEqual((576, 960), kwargs["imgsz"])
            return [positive if len(calls) == positive_call else empty for _ in images]

        model = SimpleNamespace(predict=predict)
        admission = SimpleNamespace(
            model_path="fixture",
            device="cpu",
            image_size=640,
            confidence=0.35,
            model_sha256="fixture",
            _validate_loaded_model=lambda _: None,
        )
        source = frame()
        entry = {
            "camera_id": source.camera_id,
            "sequence": source.sequence,
            "width": source.width,
            "height": source.height,
            "sha256": source.sha256,
            "byte_count": len(source.pixels),
        }
        request = {
            "type": "initialize",
            "version": 1,
            "manifest_path": "fixture",
            "model_path": "fixture",
            "device": "cpu",
            "confidence": 0.35,
            "image_size": 640,
            "image_height": 384,
            "review_rotation_180": True,
            "review_empty_scale_fallback": fallback,
        }
        messages = [
            (request, b""),
            ({"type": "predict", "frames": [entry]}, source.pixels),
            ({"type": "predict", "frames": [entry]}, source.pixels),
            ({"type": "stop"}, b""),
        ]
        if fallback:
            messages.pop(2)
        receipts = iter(messages)
        clock = [9.9]

        def receive(_):
            request, payload = next(receipts)
            if request["type"] == "predict":
                clock[0] += 0.1
            return request, payload

        fake_np = SimpleNamespace(
            zeros=lambda *a, **k: FakeImage(),
            frombuffer=lambda *a, **k: FakeImage(),
            ascontiguousarray=lambda a: a,
            uint8=object(),
        )
        with (
            patch.dict(
                "sys.modules",
                {
                    "numpy": fake_np,
                    "torch": SimpleNamespace(set_num_threads=lambda _: None),
                    "ultralytics": SimpleNamespace(YOLO=lambda _: model),
                },
            ),
            patch.object(live_yolo.socket, "if_nameindex", return_value=[(1, "lo")]),
            patch.object(
                live_yolo.YoloModelManifest,
                "read",
                return_value=SimpleNamespace(labels=["person"], model_id="fixture"),
            ),
            patch.object(live_yolo, "UltralyticsYoloDetector", return_value=admission),
            patch.object(live_yolo, "receive_packet", side_effect=receive),
            patch.object(live_yolo, "send_packet") as sent,
            patch.object(live_yolo.time, "monotonic", side_effect=lambda: clock[0]),
        ):
            live_yolo.run(Mock())
        first = sent.call_args_list[1].args[1]["frames"][0]
        self.assertEqual([], first["detections"])
        self.assertEqual([0.6, 0.4, 0.9, 0.8], first["review_detections"][0]["bbox_xyxy"])
        self.assertTrue(first["review_checked"])
        if fallback:
            self.assertEqual(not alternate_positive, first.get("review_fallback_checked", False))
            self.assertEqual([1, 2, 3, 4] * 2 + [1] * (2 if alternate_positive else 3), calls)
        else:
            second = sent.call_args_list[2].args[1]["frames"][0]
            self.assertFalse(second["review_checked"])
            self.assertEqual([1, 2, 3, 4, 1, 1, 1], calls)


if __name__ == "__main__":
    unittest.main()
