from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from baby_monitor import live_monitor
from baby_monitor.adapters.camera import RTSPCamera
from baby_monitor.domain import Detection
from baby_monitor.live_detection import LiveObservation
from tests.test_live_detection import frame, observation


class ShadowMonitorTests(unittest.TestCase):
    def test_checked_pair_is_bound_and_does_not_change_primary_observation(self):
        state = live_monitor._shadow_state()
        original = observation(count=1)
        hints = original.detections * 2
        live_monitor._record_shadow_observation(state, original, "checked", hints)
        self.assertEqual(1, state["checks"])
        self.assertEqual(1, state["positive_frames"])
        self.assertEqual(1, state["count_disagreement_frames"])
        self.assertEqual(original.frame.sha256, state["last_pair"]["raw_bgr_sha256"])
        self.assertEqual(original.frame.sequence, state["last_pair"]["sequence"])
        self.assertEqual(1, len(original.detections))
        self.assertEqual(1, len(state["last_pair"]["native_boxes"]))
        self.assertEqual(2, len(state["last_pair"]["shadow_boxes"]))

    def test_checked_empty_is_not_skipped_and_latest_pair_is_bounded(self):
        state = live_monitor._shadow_state()
        for sequence in range(20):
            obs = replace(observation(sequence=sequence, count=1), frame=frame(sequence=sequence))
            live_monitor._record_shadow_observation(state, obs, "checked", ())
        self.assertEqual(20, state["checks"])
        self.assertEqual(0, state["positive_frames"])
        self.assertEqual(20, state["count_disagreement_frames"])
        self.assertEqual(19, state["last_pair"]["sequence"])
        self.assertEqual([], state["last_pair"]["shadow_boxes"])
        self.assertEqual(0, sum(state["skipped"].values()))

    def test_skip_reasons_are_separate_and_preserve_last_pair_timestamp(self):
        state = live_monitor._shadow_state()
        obs = observation(count=1)
        live_monitor._record_shadow_observation(state, obs, "checked", ())
        pair = state["last_pair"]
        for reason in ("disabled", "native_empty", "rate_limited", None):
            live_monitor._record_shadow_observation(state, obs, reason, None)
        self.assertEqual(dict.fromkeys(state["skipped"], 1), state["skipped"])
        self.assertIs(pair, state["last_pair"])
        self.assertEqual(1, state["checks"])

    def test_invalid_status_or_mixed_channels_cannot_be_recorded(self):
        for status, hints, count in (
            ("untrusted", None, 1),
            ("disabled", (), 1),
            ("checked", None, 1),
            ("checked", (), 0),
            ("checked", (object(),), 1),
        ):
            with self.subTest(status=status, count=count), self.assertRaises(RuntimeError):
                live_monitor._record_shadow_observation(
                    live_monitor._shadow_state(), observation(count=count), status, hints
                )

    def test_monitor_never_enqueues_shadow_or_counts_extra_boxes(self):
        native = (Detection(0.0, "person", 0.8, (0.1, 0.1, 0.4, 0.6)),)
        frames = [frame(sequence=i, now=time.monotonic()) for i in range(4)]

        class Mailbox:
            def __init__(self, *args, **kwargs):
                self.index = 0

            def wait_for_batch(self):
                pass

            def take_live(self):
                value = frames[self.index]
                self.index += 1
                return (value,)

            def snapshot(self, _now):
                return {"cam1": {}}

        class Source:
            def __init__(self, source, **kwargs):
                self.source = source

            def start(self):
                pass

            def snapshot(self):
                return {}

            def close(self):
                pass

        class Detector:
            identity = {"review_nonempty_shadow": True}

            def start(self):
                pass

            def predict(self, values):
                return (native,)

            def shadow_status_for(self, value):
                return "checked"

            def shadow_hints_for(self, value):
                return native * 2

            def close(self):
                pass

        class Thread:
            def __init__(self, **kwargs):
                pass

            def start(self):
                pass

            def is_alive(self):
                return False

        trigger = Mock()
        trigger.observe.return_value = None
        queue = Mock()
        queue.snapshot.return_value = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            with (
                patch.object(live_monitor, "FrameMailbox", Mailbox),
                patch.object(live_monitor, "FFmpegLiveSource", Source),
                patch.object(live_monitor, "EventDatabase"),
                patch.object(live_monitor.threading, "Thread", Thread),
                patch.object(live_monitor, "ReviewTriggerPolicy", return_value=trigger),
                patch.object(live_monitor, "ReviewMailbox", return_value=queue),
                patch.object(live_monitor, "AlternateViewReviewPolicy") as alternate,
                patch.object(live_monitor.threading, "Event") as event,
            ):
                stop = Mock()
                stop.is_set.side_effect = [False] * 4 + [True]
                event.side_effect = [stop, Mock(is_set=lambda: False)]
                live_monitor.run_live_monitor(
                    streams=((RTSPCamera("cam1", "rtsp://127.0.0.1/fixture").source(), 16, 16),),
                    detector=Detector(),
                    local_vlm=object(),
                    root=root,
                    review_admission=lambda: False,
                )
                alternate.return_value.observe.assert_not_called()
            state = json.loads((root / "status.json").read_text())
            self.assertEqual([], list((root / "frames").iterdir()))
            self.assertEqual([], list((root / "records").iterdir()))
        queue.put.assert_not_called()
        self.assertEqual(4, queue.refresh.call_count)
        for call in trigger.observe.call_args_list:
            self.assertIsInstance(call.args[0], LiveObservation)
            self.assertEqual(native, call.args[0].detections)
        camera = state["cameras"]["cam1"]
        self.assertEqual(1, camera["last_person_count"])
        self.assertEqual(4, camera["nonempty_shadow"]["checks"])
        self.assertEqual(4, camera["nonempty_shadow"]["count_disagreement_frames"])
        self.assertEqual(0, camera["alternate_view_review_triggers"])
        self.assertEqual("unknown", camera["baby_state"])
        self.assertEqual("unknown", camera["baby_location"])


if __name__ == "__main__":
    unittest.main()
