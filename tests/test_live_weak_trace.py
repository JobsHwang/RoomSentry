from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.adapters.live_detector import PersistentYoloDetector
from baby_monitor.domain import Detection
from baby_monitor.live_weak_trace import BoundedWeakTrace
from baby_monitor.tracker_evaluation.weak_continuation import compare_weak_continuation
from tests.test_live_detection import frame
from tests.test_live_shadow_review import BOX, predict, row, run_runner, start


def diagnostic(sources, boxes=None):
    return {
        "version": 1,
        "status": "completed",
        "primary_ms": 1.5,
        "weak_ms": 2.5,
        "frames": [
            {
                "camera_id": source.camera_id,
                "sequence": source.sequence,
                "sha256": source.sha256,
                "detections": copy.deepcopy([BOX] if boxes is None else boxes),
            }
            for source in sources
        ],
    }


def replay_row(sequence, seconds, *, native=None, low=None, generation=0):
    source = frame(sequence=sequence)
    native = [copy.deepcopy(BOX)] if native is None else native
    low = copy.deepcopy(native) if low is None else low
    return {
        "camera_id": source.camera_id,
        "sequence": sequence,
        "generation": generation,
        "received_at": (source.received_at + timedelta(seconds=seconds)).isoformat(),
        "received_monotonic": seconds,
        "native": native,
        "low": low,
        "strong_output_equal": [item for item in low if item["confidence"] >= 0.35] == native,
    }


@unittest.skipUnless(os.name == "posix", "private POSIX debug sink")
class WeakTraceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.root.chmod(0o700)
        self.sinks = []

    def tearDown(self):
        for sink in self.sinks:
            sink.close()
        self.directory.cleanup()

    def sink(self, **kwargs):
        sink = BoundedWeakTrace(self.root / f"trace_{len(self.sinks)}.jsonl", **kwargs)
        self.sinks.append(sink)
        return sink

    def test_private_exclusive_file_and_no_pixels(self):
        sink, source = self.sink(), frame()
        self.assertTrue(sink.due(1))
        sink.record(
            (source,),
            ((Detection(0, "person", 0.8, tuple(BOX["bbox_xyxy"])),),),
            diagnostic([source]),
        )
        sink.close()
        path = self.root / "trace_0.jsonl"
        self.assertEqual(0o600, path.stat().st_mode & 0o777)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertTrue(rows[1]["frames"][0]["strong_output_equal"])
        self.assertNotIn("pixels", path.read_text())
        self.assertEqual(1, rows[-1]["frames"])
        with self.assertRaises(FileExistsError):
            BoundedWeakTrace(path)

    def test_rejects_unsafe_parent_links_and_bounds(self):
        self.root.chmod(0o755)
        with self.assertRaises(ValueError):
            self.sink()
        self.root.chmod(0o700)
        link = self.root / "link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            BoundedWeakTrace(link / "trace.jsonl")
        for kwargs in (
            {"seconds": True},
            {"seconds": float("nan")},
            {"seconds": 181},
            {"max_frames": 0},
            {"max_frames": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sink(**kwargs)

    def test_fixed_quota_does_not_split_batch_or_renew(self):
        sink, source = self.sink(max_frames=1), frame()
        self.assertTrue(sink.due(1))
        sink.record((source,), ((),), diagnostic([source], []))
        self.assertFalse(sink.due(1))
        self.assertEqual("frame_limit", sink.status)
        self.assertFalse(sink.due(1))

    def test_duration_uses_conservative_dual_clock(self):
        sink = self.sink(seconds=1)
        with patch("baby_monitor.live_weak_trace.time.monotonic", return_value=10):
            self.assertTrue(sink.due(1))
        with patch("baby_monitor.live_weak_trace.time.monotonic", return_value=11):
            self.assertFalse(sink.due(1))
        self.assertEqual("duration_limit", sink.status)

    def test_clock_regression_stops_capture(self):
        sink = self.sink()
        with patch("baby_monitor.live_weak_trace.time.monotonic", return_value=10):
            self.assertTrue(sink.due(1))
        with patch("baby_monitor.live_weak_trace.time.monotonic", return_value=9):
            self.assertFalse(sink.due(1))
        self.assertEqual("clock_invalid", sink.status)

    def test_binding_and_invalid_number_rejected_atomically(self):
        sink, source = self.sink(), frame()
        native = ((),)
        bad = diagnostic([source], [])
        for key, value in (("camera_id", "other"), ("sequence", 2), ("sha256", "wrong")):
            edited = copy.deepcopy(bad)
            edited["frames"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                sink.record((source,), native, edited)
        bad["weak_ms"] = float("nan")
        with self.assertRaises(ValueError):
            sink.record((source,), native, bad)
        self.assertEqual(0, sink.frames)

    def test_weak_failure_is_not_zero_box_success(self):
        sink, source = self.sink(), frame()
        value = {**diagnostic([], []), "status": "inference_failed"}
        sink.record((source,), ((),), value)
        self.assertEqual(1, sink.failed_batches)
        sink.close()
        saved = json.loads((self.root / "trace_0.jsonl").read_text().splitlines()[1])
        self.assertIsNone(saved["frames"][0]["low"])

    def test_native_return_survives_bad_capture_and_does_not_request_again(self):
        sink, source = self.sink(), frame()
        client = PersistentYoloDetector(
            command=("fixture",),
            model_path="fixture",
            manifest_path="fixture",
            weak_continuation_trace=sink,
        )
        response = row(source, checked=False, reason="disabled")
        # Missing extra response is a rejected diagnostic, never a missing person.
        output = predict(client, [source], [response])
        self.assertEqual(1, len(output[0]))
        self.assertEqual("capture_failed", sink.status)
        self.assertFalse(sink.due(1))

    def test_requested_trace_requires_ack_and_native_threshold(self):
        sink = self.sink()
        client = PersistentYoloDetector(
            command=("fixture",),
            model_path="fixture",
            manifest_path="fixture",
            weak_continuation_trace=sink,
        )
        with self.assertRaises(RuntimeError):
            start(client, {"type": "ready"})
        with self.assertRaises(ValueError):
            PersistentYoloDetector(
                command=("fixture",),
                model_path="fixture",
                manifest_path="fixture",
                confidence=0.2,
                weak_continuation_trace=sink,
            )

    def test_trace_write_failure_closes_file_and_preserves_native(self):
        sink, source = self.sink(), frame()
        client = PersistentYoloDetector(
            command=("fixture",),
            model_path="fixture",
            manifest_path="fixture",
            weak_continuation_trace=sink,
        )
        with patch.object(sink, "due", side_effect=OSError):
            self.assertEqual(
                1,
                len(predict(client, [source], [row(source, checked=False, reason="disabled")])[0]),
            )
        self.assertIsNone(sink.fd)


class WeakRunnerTests(unittest.TestCase):
    def test_extra_inference_same_model_original_view_native_unchanged(self):
        messages, calls, loads = run_runner(
            [[frame()]],
            [(True,), (False,)],
            [10],
            settings={"review_rotation_180": False},
            weak_trace=True,
        )
        self.assertEqual(1, loads)
        self.assertEqual([0.35, 0.20], [call["conf"] for call in calls[4:]])
        self.assertEqual([[False], [False]], [call["rotations"] for call in calls[4:]])
        self.assertEqual([BOX], messages[1]["frames"][0]["detections"])
        self.assertEqual([], messages[1]["weak_trace"]["frames"][0]["detections"])

    def test_invalid_request_fails_without_low_inference(self):
        with self.assertRaises(ValueError):
            run_runner([[frame()]], [], [10], weak_trace="yes")


class WeakReplayTests(unittest.TestCase):
    def test_weak_only_cannot_seed_or_propagate_role(self):
        weak = {**BOX, "confidence": 0.25}
        report = compare_weak_continuation(
            [replay_row(0, 1, native=[], low=[weak]), replay_row(1, 1.2)]
        )
        for values in report["frames"]:
            for arm in values["arms"].values():
                self.assertEqual([], arm["strong_links"])
                self.assertEqual(0, arm["weak_continuations"])
        self.assertFalse(report["role_propagation"])
        self.assertFalse(report["live_assignments_reproduced"])

    def test_unique_weak_can_bridge_strong_seed_but_not_renew_forever(self):
        weak = {**BOX, "confidence": 0.25}
        rows = [replay_row(0, 1), replay_row(1, 1.2, native=[], low=[weak]), replay_row(2, 1.4)]
        report = compare_weak_continuation(rows)
        self.assertEqual(1, report["frames"][1]["arms"]["controlled"]["weak_continuations"])
        self.assertEqual(1, len(report["frames"][2]["arms"]["controlled"]["strong_links"]))
        rows = [
            replay_row(0, 1),
            *[replay_row(i, 1 + i * 0.5, native=[], low=[weak]) for i in range(1, 7)],
        ]
        report = compare_weak_continuation(rows)
        self.assertEqual(0, report["frames"][-1]["arms"]["weak"]["weak_continuations"])

    def test_duplicate_weak_rejected_and_raw_strong_denominator_unchanged(self):
        weak = {**BOX, "confidence": 0.25}
        report = compare_weak_continuation(
            [replay_row(0, 1), replay_row(1, 1.2, native=[], low=[weak, weak])]
        )
        self.assertEqual(1, report["frames"][1]["arms"]["weak"]["weak_continuations"])
        self.assertEqual(2, report["frames"][1]["arms"]["controlled"]["duplicate_weak_rejected"])
        self.assertEqual(0, report["frames"][1]["native_count"])

    def test_gap_and_generation_clear_strong_seed(self):
        weak = {**BOX, "confidence": 0.25}
        for seconds, generation in ((4, 0), (1.2, 1)):
            report = compare_weak_continuation(
                [
                    replay_row(0, 1),
                    replay_row(1, seconds, native=[], low=[weak], generation=generation),
                ]
            )
            self.assertTrue(report["frames"][1]["reset"])
            self.assertEqual(0, report["frames"][1]["arms"]["controlled"]["weak_continuations"])

    def test_mismatch_kept_as_censored_frame_resets_all_arms(self):
        weak = {**BOX, "confidence": 0.25}
        bad = replay_row(1, 1.2, low=[weak])
        report = compare_weak_continuation([replay_row(0, 1), bad, replay_row(2, 1.4)])
        self.assertTrue(report["frames"][1]["rejected"])
        self.assertEqual(1, report["frames"][1]["native_count"])
        self.assertEqual([], report["frames"][2]["arms"]["controlled"]["strong_links"])

    def test_failed_capture_is_causal_break_not_zero_box(self):
        bad = replay_row(1, 1.2)
        bad.update(low=None, strong_output_equal=None)
        report = compare_weak_continuation([replay_row(0, 1), bad, replay_row(2, 1.4)])
        self.assertTrue(report["frames"][1]["rejected"])

    def test_prefix_independence_and_json_round_trip(self):
        rows = [replay_row(i, 1 + i * 0.2) for i in range(3)]
        full = compare_weak_continuation(rows)
        self.assertEqual(full["frames"][:2], compare_weak_continuation(rows[:2])["frames"])
        self.assertEqual(full, json.loads(json.dumps(full)))

    def test_reordering_and_forged_equality_rejected(self):
        with self.assertRaises(ValueError):
            compare_weak_continuation([replay_row(1, 1), replay_row(0, 1.2)])
        bad = replay_row(0, 1)
        bad["strong_output_equal"] = False
        with self.assertRaises(ValueError):
            compare_weak_continuation([bad])


if __name__ == "__main__":
    unittest.main()
