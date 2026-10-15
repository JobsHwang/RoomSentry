from __future__ import annotations

import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baby_monitor import live_monitor
from baby_monitor.adapters.camera import RTSPCamera


@unittest.skipUnless(os.name == "posix", "live monitor requires POSIX")
class LiveCleanupTests(unittest.TestCase):
    def run_monitor(self, failures=(), *, publish_failure=False, join_timeout=False):
        calls = []
        lock_descriptors = []
        original_open = os.open
        original_write = live_monitor._write_private
        previous_handlers = {
            item: signal.getsignal(item) for item in (signal.SIGTERM, signal.SIGINT)
        }

        def record(name):
            calls.append(name)
            if name in failures:
                raise RuntimeError("private-source-or-model-information")

        class Source:
            def __init__(self, source, **_kwargs):
                self.source = source

            def start(self):
                pass

            def snapshot(self):
                return {"phase": "fixture"}

            def close(self):
                record("source_" + self.source.camera_id)

        class Detector:
            identity = {"fixture": True}

            def start(self):
                pass

            def close(self):
                record("detector")

        class ReviewThread:
            alive = False

            def __init__(self, **_kwargs):
                pass

            def start(self):
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, *, timeout):
                self_timeout.assertEqual(220, timeout)
                record("review_join")
                self.alive = join_timeout

        class VLM:
            def close(self):
                record("vlm")

        self_timeout = self

        def open_file(path, *args, **kwargs):
            descriptor = original_open(path, *args, **kwargs)
            if Path(path).name == "worker.lock":
                lock_descriptors.append(descriptor)
            return descriptor

        def write_status(path, value):
            if path.name == "status.json" and value["phase"] in ("stopped", "cleanup_failed"):
                record("publish")
                if publish_failure:
                    raise OSError("private-storage-information")
            return original_write(path, value)

        error = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            streams = tuple(
                (RTSPCamera(camera, "rtsp://127.0.0.1/fixture").source(), 16, 16)
                for camera in ("a", "b")
            )
            with (
                patch.object(live_monitor, "FFmpegLiveSource", Source),
                patch.object(live_monitor, "EventDatabase"),
                patch.object(live_monitor.threading, "Thread", ReviewThread),
                patch.object(live_monitor.os, "open", side_effect=open_file),
                patch.object(live_monitor, "_write_private", side_effect=write_status),
            ):
                try:
                    live_monitor.run_live_monitor(
                        streams=streams,
                        detector=Detector(),
                        local_vlm=VLM(),
                        root=root,
                        review_admission=lambda: False,
                        max_runtime_seconds=0,
                    )
                except RuntimeError as caught:
                    error = str(caught)
            # These run before opening another file can reuse a leaked lock FD.
            self.assertEqual(1, len(lock_descriptors))
            with self.assertRaises(OSError):
                os.fstat(lock_descriptors[0])
            for signum, handler in previous_handlers.items():
                self.assertEqual(handler, signal.getsignal(signum))
            status = json.loads((root / "status.json").read_text())
        self.assertEqual(
            ["source_a", "source_b", "detector", "review_join", "vlm", "publish"], calls
        )
        if error is not None:
            self.assertNotIn("private-", error)
        return status, error

    def test_clean_shutdown_still_reports_stopped(self):
        status, error = self.run_monitor()
        self.assertIsNone(error)
        self.assertEqual("stopped", status["phase"])
        self.assertEqual([], status["cleanup_error_codes"])

    def test_source_or_detector_failure_does_not_skip_review_cleanup(self):
        for failures, expected in (
            (("source_a",), ["source_close_failed"]),
            (("detector",), ["detector_close_failed"]),
            (
                ("source_a", "source_b", "detector"),
                ["source_close_failed", "detector_close_failed"],
            ),
        ):
            with self.subTest(failures=failures):
                status, error = self.run_monitor(failures)
                self.assertEqual("cleanup_failed", status["phase"])
                self.assertEqual(expected, status["cleanup_error_codes"])
                self.assertEqual("live monitor cleanup failed: " + ",".join(expected), error)

    def test_review_join_failure_does_not_skip_vlm_cleanup(self):
        status, error = self.run_monitor(("review_join",))
        self.assertEqual("cleanup_failed", status["phase"])
        self.assertEqual(["review_join_failed"], status["cleanup_error_codes"])
        self.assertIn("review_join_failed", error)

    def test_review_join_timeout_is_not_success(self):
        status, error = self.run_monitor(join_timeout=True)
        self.assertEqual("cleanup_failed", status["phase"])
        self.assertEqual(["review_join_timeout"], status["cleanup_error_codes"])
        self.assertIn("review_join_timeout", error)

    def test_vlm_close_failure_still_publishes_and_releases_process_resources(self):
        status, error = self.run_monitor(("vlm",))
        self.assertEqual("cleanup_failed", status["phase"])
        self.assertEqual(["vlm_close_failed"], status["cleanup_error_codes"])
        self.assertIn("vlm_close_failed", error)

    def test_publish_failure_still_restores_handlers_and_closes_worker_lock(self):
        status, error = self.run_monitor(("source_a",), publish_failure=True)
        self.assertNotEqual("stopped", status["phase"])
        self.assertEqual(
            "live monitor cleanup failed: source_close_failed,status_publish_failed", error
        )


if __name__ == "__main__":
    unittest.main()
