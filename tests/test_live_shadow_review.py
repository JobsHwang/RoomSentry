from __future__ import annotations

import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from baby_monitor.adapters.live_detector import PersistentYoloDetector
from baby_monitor.local_runners import live_yolo
from tests.test_live_detection import frame

BOX = {"confidence": 0.8, "bbox_xyxy": [0.1, 0.2, 0.4, 0.6]}


def detector(*, shadow=False, rotation=True):
    return PersistentYoloDetector(
        command=("fixture",),
        model_path="fixture",
        manifest_path="fixture",
        review_rotation_180=rotation,
        review_nonempty_shadow=shadow,
    )


def row(source, *, native=True, checked=True, reason=None, hints=None):
    result = {
        "camera_id": source.camera_id,
        "sequence": source.sequence,
        "sha256": source.sha256,
        "detections": [copy.deepcopy(BOX)] if native else [],
        "review_checked": False,
        "shadow_checked": checked,
        "shadow_skip_reason": reason,
    }
    if checked:
        result["shadow_detections"] = copy.deepcopy([BOX] if hints is None else hints)
    return result


def predict(client, sources, rows):
    client.channel = Mock()
    with (
        patch("baby_monitor.adapters.live_detector.send_packet"),
        patch(
            "baby_monitor.adapters.live_detector.receive_packet",
            return_value=({"type": "predictions", "frames": rows}, b""),
        ),
    ):
        return client.predict(tuple(sources))


def start(client, ready):
    parent, child = Mock(), Mock()
    child.fileno.return_value = 14
    with (
        patch(
            "baby_monitor.adapters.live_detector.socket.socketpair", return_value=(parent, child)
        ),
        patch("baby_monitor.adapters.live_detector.subprocess.Popen"),
        patch("baby_monitor.adapters.live_detector.send_packet"),
        patch("baby_monitor.adapters.live_detector.receive_packet", return_value=(ready, b"")),
    ):
        client.start()


class LiveShadowClientTests(unittest.TestCase):
    def test_option_is_strict_bool_and_requires_rotation(self):
        for value in (None, 0, 1, "true", [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                detector(shadow=value)
        with self.assertRaises(ValueError):
            detector(shadow=True, rotation=False)
        self.assertFalse(detector().settings["review_nonempty_shadow"])

    def test_legacy_ready_and_predictions_only_work_for_opt_out(self):
        client = detector()
        start(client, {"type": "ready"})
        source = frame()
        response = row(source)
        for field in ("shadow_checked", "shadow_skip_reason", "shadow_detections"):
            del response[field]
        self.assertEqual(1, len(predict(client, [source], [response])[0]))
        self.assertIsNone(client.shadow_hints_for(source))
        self.assertIsNone(client.shadow_status_for(source))
        with self.assertRaises(RuntimeError):
            predict(detector(shadow=True), [source], [response])

    def test_ready_must_acknowledge_requested_mode_and_rotation(self):
        for ready in (
            {"type": "ready"},
            {"type": "ready", "review_nonempty_shadow": False},
            {"type": "ready", "review_nonempty_shadow": 1, "review_rotation_180": True},
            {"type": "ready", "review_nonempty_shadow": True},
            {"type": "ready", "review_nonempty_shadow": True, "review_rotation_180": False},
        ):
            with self.subTest(ready=ready), self.assertRaises(RuntimeError):
                start(detector(shadow=True), ready)
        ready = {"type": "ready", "review_nonempty_shadow": True, "review_rotation_180": True}
        client = detector(shadow=True)
        start(client, ready)
        self.assertEqual(ready, client.identity)
        with self.assertRaises(RuntimeError):
            start(detector(), ready)

    def test_native_output_and_shadow_channel_are_separate(self):
        client, source = detector(shadow=True), frame()
        output = predict(client, [source], [row(source)])
        self.assertEqual(1, len(output[0]))
        self.assertEqual(tuple(BOX["bbox_xyxy"]), output[0][0].bbox_xyxy)
        self.assertEqual(1, len(client.shadow_hints_for(source)))
        self.assertEqual("checked", client.shadow_status_for(source))
        self.assertIsNone(client.review_hints_for(source))

    def test_checked_empty_is_distinct_from_rate_limited(self):
        client, source = detector(shadow=True), frame()
        predict(client, [source], [row(source, hints=[])])
        self.assertEqual((), client.shadow_hints_for(source))
        self.assertEqual("checked", client.shadow_status_for(source))
        predict(client, [source], [row(source, checked=False, reason="rate_limited")])
        self.assertIsNone(client.shadow_hints_for(source))
        self.assertEqual("rate_limited", client.shadow_status_for(source))

    def test_empty_native_uses_only_ordinary_review_channel(self):
        client, source = detector(shadow=True), frame()
        response = row(source, native=False, checked=False, reason="native_empty")
        response.update(review_checked=True, review_detections=[copy.deepcopy(BOX)])
        self.assertEqual(((),), predict(client, [source], [response]))
        self.assertEqual(1, len(client.review_hints_for(source)))
        self.assertIsNone(client.shadow_hints_for(source))
        self.assertEqual("native_empty", client.shadow_status_for(source))

    def test_disabled_shadow_is_explicit_without_hint_payload(self):
        client, source = detector(), frame()
        predict(client, [source], [row(source, checked=False, reason="disabled")])
        self.assertEqual("disabled", client.shadow_status_for(source))
        self.assertIsNone(client.shadow_hints_for(source))

    def test_shadow_rejects_unauthorized_or_inconsistent_payloads(self):
        source = frame()
        invalid = [
            row(source, native=False),
            row(source, reason="rate_limited"),
            row(source, checked=False, reason="disabled"),
            row(source, checked=False, reason="native_empty"),
            row(source, checked=False, reason=None),
            {**row(source), "shadow_checked": 1},
            {**row(source), "shadow_detections": None},
            {**row(source), "shadow_detections": [BOX] * 101},
            {**row(source, checked=False, reason="rate_limited"), "shadow_detections": []},
        ]
        for missing in ("shadow_checked", "shadow_skip_reason", "shadow_detections"):
            response = row(source)
            del response[missing]
            invalid.append(response)
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                predict(detector(shadow=True), [source], [response])
        with self.assertRaises(RuntimeError):
            predict(detector(), [source], [row(source)])

    def test_nonempty_native_cannot_smuggle_hints_into_ordinary_review(self):
        source = frame()
        for checked in (True, False):
            response = row(source)
            response.update(review_checked=checked, review_detections=[copy.deepcopy(BOX)])
            with self.subTest(checked=checked), self.assertRaises(RuntimeError):
                predict(detector(shadow=True), [source], [response])

    def test_all_cached_channels_bind_generation_sequence_camera_and_sha(self):
        client, source = detector(shadow=True), frame()
        predict(client, [source], [row(source)])
        alternatives = (
            replace(source, generation=1),
            replace(source, sequence=1),
            replace(source, camera_id="other"),
            replace(source, pixels=bytes([40, 50, 60]) * 256),
        )
        for other in alternatives:
            self.assertIsNone(client.shadow_hints_for(other))
            self.assertIsNone(client.shadow_status_for(other))
        response = row(source, native=False, checked=False, reason="native_empty")
        response.update(review_checked=True, review_detections=[copy.deepcopy(BOX)])
        predict(client, [source], [response])
        for other in alternatives:
            self.assertIsNone(client.review_hints_for(other))
        self.assertIsNone(client.shadow_hints_for(source))
        client.close()
        self.assertIsNone(client.review_hints_for(source))
        self.assertIsNone(client.shadow_status_for(source))

    def test_bad_frame_binding_and_late_bad_row_publish_no_partial_cache(self):
        sources = [frame(), frame("cam2")]
        for field, value in (("camera_id", "wrong"), ("sequence", 42), ("sha256", "wrong")):
            client = detector(shadow=True)
            predict(client, [sources[0]], [row(sources[0])])
            responses = [row(source) for source in sources]
            responses[1][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                predict(client, sources, responses)
            for source in sources:
                self.assertIsNone(client.shadow_hints_for(source))
                self.assertIsNone(client.shadow_status_for(source))

    def test_duplicate_camera_batch_is_rejected(self):
        source = frame()
        with self.assertRaises(RuntimeError):
            predict(detector(shadow=True), [source, replace(source, sequence=1)], [])


class FakeImage:
    def __init__(self, *, rotated=False):
        self.rotated = rotated

    def reshape(self, *args):
        return self

    def __getitem__(self, key):
        if key != (slice(None, None, -1), slice(None, None, -1)):
            raise AssertionError("unexpected image transform")
        return FakeImage(rotated=True)


def prediction(present):
    if not present:
        return SimpleNamespace(boxes=None)
    return SimpleNamespace(
        boxes=[
            SimpleNamespace(
                conf=SimpleNamespace(item=lambda: BOX["confidence"]),
                xyxyn=[SimpleNamespace(tolist=lambda: BOX["bbox_xyxy"])],
            )
        ]
    )


def run_runner(
    batches, predictions, times, *, settings=None, prediction_delays=None, weak_trace=False
):
    """Run only protocol/control logic with synthetic bytes and fake model modules."""
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
        **(settings or {}),
    }
    messages = [(request, b"")]
    for sources in batches:
        entries = [
            {
                "camera_id": source.camera_id,
                "sequence": source.sequence,
                "width": source.width,
                "height": source.height,
                "sha256": source.sha256,
                "byte_count": len(source.pixels),
            }
            for source in sources
        ]
        messages.append(
            (
                {
                    "type": "predict",
                    "frames": entries,
                    **({"weak_trace": weak_trace} if weak_trace else {}),
                },
                b"".join(s.pixels for s in sources),
            )
        )
    messages.append(({"type": "stop"}, b""))
    calls = []
    answers = iter(predictions)
    clock = [0.0]
    delays = iter(prediction_delays or [0.0] * len(predictions))
    receipts = iter(messages)
    batch_times = iter(times)

    def receive(_):
        message, payload = next(receipts)
        if message["type"] == "predict":
            clock[0] = next(batch_times, clock[0])
        return message, payload

    def model_predict(images, **kwargs):
        calls.append({"rotations": [image.rotated for image in images], **kwargs})
        if len(calls) <= (8 if request.get("review_empty_scale_fallback") else 4):
            return [prediction(False) for _ in images]
        answer = next(answers)
        clock[0] += next(delays)
        return [prediction(present) for present in answer]

    admission = SimpleNamespace(
        model_path="fixture",
        device="cpu",
        image_size=640,
        confidence=0.35,
        model_sha256="fixture",
        _validate_loaded_model=lambda _: None,
    )
    fake_np = SimpleNamespace(
        zeros=lambda *a, **k: FakeImage(),
        frombuffer=lambda *a, **k: FakeImage(),
        ascontiguousarray=lambda image: image,
        uint8=object(),
    )
    model_factory = Mock(return_value=SimpleNamespace(predict=model_predict))
    with (
        patch.dict(
            "sys.modules",
            {
                "numpy": fake_np,
                "torch": SimpleNamespace(set_num_threads=lambda _: None),
                "ultralytics": SimpleNamespace(YOLO=model_factory),
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
    if next(answers, None) is not None:
        raise AssertionError("planned model call not made")
    return [call.args[1] for call in sent.call_args_list], calls, model_factory.call_count


class LiveShadowRunnerTests(unittest.TestCase):
    def test_default_skips_nonempty_but_keeps_empty_frame_review(self):
        messages, calls, loads = run_runner(
            [[frame()], [frame(sequence=1)]],
            [(True,), (False,), (True,)],
            [10, 11],
        )
        self.assertEqual(1, loads)
        self.assertFalse(messages[0]["review_nonempty_shadow"])
        first, second = [message["frames"][0] for message in messages[1:]]
        self.assertFalse(first["review_checked"])
        self.assertFalse(first["shadow_checked"])
        self.assertEqual("disabled", first["shadow_skip_reason"])
        self.assertTrue(second["review_checked"])
        self.assertEqual("disabled", second["shadow_skip_reason"])
        self.assertEqual([[False], [False], [True]], [c["rotations"] for c in calls[4:]])

    def test_opt_in_mixed_batch_has_separate_channels_and_inverse_mapping(self):
        sources = [frame(), frame("cam2")]
        messages, calls, loads = run_runner(
            [sources],
            [(True, False), (True, True)],
            [10],
            settings={"review_nonempty_shadow": True},
        )
        self.assertEqual(1, loads)
        self.assertTrue(messages[0]["review_nonempty_shadow"])
        self.assertEqual(1.0, messages[0]["review_interval_seconds"])
        first, second = messages[1]["frames"]
        self.assertTrue(first["shadow_checked"])
        self.assertIsNone(first["shadow_skip_reason"])
        self.assertFalse(first["review_checked"])
        self.assertNotIn("review_detections", first)
        self.assertEqual([0.6, 0.4, 0.9, 0.8], first["shadow_detections"][0]["bbox_xyxy"])
        self.assertEqual([BOX], first["detections"])
        self.assertTrue(second["review_checked"])
        self.assertFalse(second["shadow_checked"])
        self.assertEqual("native_empty", second["shadow_skip_reason"])
        self.assertNotIn("shadow_detections", second)
        self.assertEqual([False, False], calls[-2]["rotations"])
        self.assertEqual([True, True], calls[-1]["rotations"])
        # Feed the real runner's protocol output through the client parser.
        client = detector(shadow=True)
        self.assertEqual(
            [1, 0], [len(items) for items in predict(client, sources, [first, second])]
        )
        self.assertIsNone(client.review_hints_for(sources[0]))
        self.assertIsNone(client.shadow_hints_for(sources[1]))

    def test_empty_and_nonempty_checks_share_one_camera_rate_limit(self):
        messages, calls, _ = run_runner(
            [[frame(sequence=index)] for index in range(5)],
            [(True,), (True,), (False,), (False,), (True,), (True,), (True,), (False,)],
            [10, 10.1, 11, 11.1, 12],
            settings={"review_nonempty_shadow": True},
        )
        rows = [message["frames"][0] for message in messages[1:]]
        self.assertEqual([True, False, False, False, True], [r["shadow_checked"] for r in rows])
        self.assertEqual([False, False, True, False, False], [r["review_checked"] for r in rows])
        self.assertEqual("rate_limited", rows[3]["shadow_skip_reason"])
        self.assertEqual([], rows[-1]["shadow_detections"])
        self.assertEqual(5, sum(not call["rotations"][0] for call in calls[4:]))
        self.assertEqual(3, sum(call["rotations"][0] for call in calls[4:]))

    def test_rate_limits_are_per_camera(self):
        messages, _, _ = run_runner(
            [[frame()], [frame("cam2")]],
            [(True,), (False,), (True,), (True,)],
            [10, 10.1],
            settings={"review_nonempty_shadow": True},
        )
        self.assertTrue(all(message["frames"][0]["shadow_checked"] for message in messages[1:]))

    def test_runner_rejects_invalid_options_before_model_load(self):
        for settings in (
            {"review_nonempty_shadow": None},
            {"review_nonempty_shadow": 1},
            {"review_nonempty_shadow": "true"},
            {"review_nonempty_shadow": True, "review_rotation_180": False},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                run_runner([], [], [], settings=settings)

    def test_runner_rejects_duplicate_camera_within_a_batch(self):
        with self.assertRaises(ValueError):
            run_runner(
                [[frame(), frame(sequence=1)]],
                [],
                [],
                settings={"review_nonempty_shadow": True},
            )


if __name__ == "__main__":
    unittest.main()
