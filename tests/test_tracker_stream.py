from __future__ import annotations

import math
import runpy
import sys
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from baby_monitor.local_runners.tracker_replay import (
    TrackerReplayConfig,
    TrackerReplayError,
    replay_tracker,
)
from baby_monitor.local_runners.tracker_stream import (
    IncrementalTrackerExperiment,
    TrackerStreamConfig,
)
from baby_monitor.object_tracker import DetectionFrame, DetectionFrameBatch, FrameDetection


class TrackerStreamTests(unittest.TestCase):
    def frame(self, index=0, seconds=0.0, *, empty=False, confidence=0.9, label="person"):
        return DetectionFrame(
            f"f{index}",
            datetime(2026, 10, 10, tzinfo=UTC) + timedelta(seconds=seconds),
            () if empty else (FrameDetection("d0", label, confidence, (0.1, 0.1, 0.3, 0.6)),),
        )

    def factory(self, transform=None, tentative=False):
        calls, instances = [], []

        class Backend:
            provenance = {"synthetic": True}
            tentative_indices = ()

            def update(self, detections):
                calls.append(detections)
                self.tentative_indices = (
                    (0,) if tentative and len(calls) == 1 and detections else ()
                )
                rows = [
                    [10.0, 10.0, 20.0, 40.0, i + 1, d.confidence, 0, i]
                    for i, d in enumerate(detections)
                ]
                return transform(rows) if transform else rows

        def create(config):
            backend = Backend()
            instances.append(backend)
            return backend

        return create, calls, instances

    def stream(self, factory, **kwargs):
        defaults = dict(
            camera_id="synthetic_cam",
            detector_adapter="synthetic_detector",
            label="person",
            session_id="0" * 32,
            config=TrackerStreamConfig(TrackerReplayConfig("bytetrack", 384, 288)),
        )
        defaults.update(kwargs)
        return IncrementalTrackerExperiment(**defaults, _backend_factory=factory)

    def test_incremental_prefix_does_not_depend_on_future_frames(self):
        factory, calls, _ = self.factory()
        with self.stream(factory) as stream:
            first = stream.update(self.frame(), generation=0, elapsed_seconds=0)
            second = stream.update(self.frame(1, 0.2), generation=0, elapsed_seconds=0.2)
            frozen = first.to_dict()
            stream.update(self.frame(2, 0.4), generation=0, elapsed_seconds=0.4)
            self.assertEqual(frozen, first.to_dict())
            self.assertEqual(first.assignments[0].tracking_id, second.assignments[0].tracking_id)
        with self.stream(factory) as stream:
            self.assertEqual(first, stream.update(self.frame(), generation=0, elapsed_seconds=0))
        self.assertEqual(4, len(calls))

    def test_empty_input_updates_backend_but_never_emits_predicted_person(self):
        factory, calls, _ = self.factory()
        with self.stream(factory) as stream:
            first = stream.update(self.frame(), generation=0, elapsed_seconds=0)
            empty = stream.update(self.frame(1, 0.2, empty=True), generation=0, elapsed_seconds=0.2)
            last = stream.update(self.frame(2, 0.4), generation=0, elapsed_seconds=0.4)
            self.assertEqual((), empty.assignments)
            self.assertEqual((), empty.phases)
            self.assertEqual(first.assignments[0].tracking_id, last.assignments[0].tracking_id)
            self.assertEqual(3, last.backend_update_count)
        self.assertEqual((), calls[1])

    def test_duplicate_retry_returns_exact_result_without_advancing_native(self):
        factory, calls, _ = self.factory()
        with self.stream(factory) as stream:
            first = stream.update(self.frame(), generation=0, elapsed_seconds=0)
            stream.update(self.frame(1, 0.2), generation=0, elapsed_seconds=0.2)
            self.assertIs(first, stream.update(self.frame(), generation=0, elapsed_seconds=0))
            for frame, elapsed in ((self.frame(confidence=0.8), 0), (self.frame(), 0.1)):
                with self.assertRaisesRegex(TrackerReplayError, "conflicting"):
                    stream.update(frame, generation=0, elapsed_seconds=elapsed)
            self.assertEqual(2, len(calls))

    def test_evicted_duplicate_cannot_reenter_with_new_time(self):
        factory, calls, _ = self.factory()
        config = TrackerStreamConfig(
            TrackerReplayConfig("bytetrack", 384, 288), duplicate_cache_frames=1
        )
        with self.stream(factory, config=config) as stream:
            stream.update(self.frame(), generation=0, elapsed_seconds=0)
            stream.update(self.frame(1, 0.2), generation=0, elapsed_seconds=0.2)
            with self.assertRaisesRegex(TrackerReplayError, "outside retry cache"):
                stream.update(self.frame(seconds=0.4), generation=0, elapsed_seconds=0.4)
        self.assertEqual(2, len(calls))

    def test_actual_elapsed_expires_export_even_if_empty_updates_keep_native_alive(self):
        factory, _, instances = self.factory()
        config = TrackerStreamConfig(
            TrackerReplayConfig("bytetrack", 384, 288), max_export_age_seconds=0.5
        )
        with self.stream(factory, config=config) as stream:
            first = stream.update(self.frame(), generation=0, elapsed_seconds=0)
            stream.update(self.frame(1, 0.2, empty=True), generation=0, elapsed_seconds=0.2)
            expired = stream.update(
                self.frame(2, 0.6, empty=True), generation=0, elapsed_seconds=0.6
            )
            last = stream.update(self.frame(3, 0.8), generation=0, elapsed_seconds=0.8)
            self.assertEqual(1, expired.expired_export_count)
            self.assertEqual(0, last.segment)
            self.assertNotEqual(first.assignments[0].tracking_id, last.assignments[0].tracking_id)
        self.assertEqual(1, len(instances))
        self.assertFalse(config.to_dict()["native_history_expired_by_export_age"])

    def test_generation_and_either_clock_gap_reset_namespace(self):
        for generation, elapsed, utc, reason in (
            (1, 0.2, -0.2, "source_generation_changed"),
            (0, 2, 0.2, "elapsed_input_gap"),
            (0, 0.2, 2, "utc_input_gap"),
        ):
            factory, _, instances = self.factory()
            with self.subTest(reason=reason), self.stream(factory) as stream:
                first = stream.update(self.frame(), generation=0, elapsed_seconds=0)
                last = stream.update(
                    self.frame(1, utc), generation=generation, elapsed_seconds=elapsed
                )
                self.assertEqual(reason, last.reset_reason)
                self.assertEqual(1, last.segment)
                self.assertNotEqual(
                    first.assignments[0].tracking_id, last.assignments[0].tracking_id
                )
            self.assertEqual(2, len(instances))

    def test_either_clock_can_expire_exports_without_a_single_input_gap(self):
        for elapsed, utc in ((0.1, 0.8), (0.8, 0.1)):
            factory, calls, instances = self.factory()
            with self.subTest(elapsed=elapsed, utc=utc), self.stream(factory) as stream:
                first = stream.update(self.frame(), generation=0, elapsed_seconds=0)
                stream.update(self.frame(1, utc, empty=True), generation=0, elapsed_seconds=elapsed)
                expired = stream.update(
                    self.frame(2, utc * 2, empty=True), generation=0, elapsed_seconds=elapsed * 2
                )
                last = stream.update(
                    self.frame(3, utc * 3), generation=0, elapsed_seconds=elapsed * 3
                )
                self.assertEqual(1, expired.expired_export_count)
                self.assertEqual(0, last.segment)
                self.assertNotEqual(
                    first.assignments[0].tracking_id, last.assignments[0].tracking_id
                )
            self.assertEqual(4, len(calls))
            self.assertEqual(1, len(instances))

    def test_old_generation_retry_cannot_return_historical_presence(self):
        factory, calls, _ = self.factory()
        with self.stream(factory) as stream:
            stream.update(self.frame(), generation=0, elapsed_seconds=0)
            stream.update(self.frame(1, 0.2), generation=1, elapsed_seconds=0.2)
            with self.assertRaisesRegex(TrackerReplayError, "stale incremental source generation"):
                stream.update(self.frame(), generation=0, elapsed_seconds=0)
            self.assertEqual(2, len(calls))

    def test_bad_or_stale_inputs_do_not_mutate_backend(self):
        factory, calls, _ = self.factory()
        with self.stream(factory) as stream:
            stream.update(self.frame(), generation=1, elapsed_seconds=1)
            bad = (
                (self.frame(1, 0.2), 0, 1.2),
                (self.frame(1, 0.2), 1, 1),
                (self.frame(1, 0), 1, 1.2),
                (self.frame(1, 0.2, label="head"), 1, 1.2),
                (self.frame(1, 0.2), True, 1.2),
                (self.frame(1, 0.2), 1, math.nan),
                (self.frame(1, 0.2), 1, 10**10000),
            )
            for frame, generation, elapsed in bad:
                with (
                    self.subTest(generation=generation, value_type=type(elapsed).__name__),
                    self.assertRaises(TrackerReplayError),
                ):
                    stream.update(frame, generation=generation, elapsed_seconds=elapsed)
            result = stream.update(self.frame(1, 0.2), generation=1, elapsed_seconds=1.2)
            self.assertEqual(2, result.backend_update_count)
        self.assertEqual(2, len(calls))

    def test_malformed_output_fail_stops_and_releases_native_ownership(self):
        factory, _, _ = self.factory(lambda rows: [[0] * 8])
        with self.stream(factory) as stream:
            with self.assertRaisesRegex(TrackerReplayError, "session closed"):
                stream.update(self.frame(), generation=0, elapsed_seconds=0)
            self.assertEqual("injected_stdlib_test_backend", stream.runtime_id)
            self.assertEqual(1, stream.attempted_update_count)
            with self.assertRaisesRegex(TrackerReplayError, "closed"):
                stream.update(self.frame(1, 0.2), generation=0, elapsed_seconds=0.2)
        good, _, _ = self.factory()
        with self.stream(good) as stream:
            stream.update(self.frame(), generation=0, elapsed_seconds=0)

    def test_constructor_failure_is_closed_without_update_claim_or_path_leak(self):
        def fail(config):
            raise RuntimeError("private/path/not/for/logging")

        with self.stream(fail) as stream:
            with self.assertRaisesRegex(
                TrackerReplayError, "^incremental backend failed; session closed$"
            ):
                stream.update(self.frame(), generation=0, elapsed_seconds=0)
            self.assertEqual("no_backend_invoked", stream.runtime_id)
            self.assertEqual(0, stream.attempted_update_count)
        good, _, _ = self.factory()
        with self.stream(good):
            pass

    def test_runtime_change_between_segments_fails_closed(self):
        factory, _, instances = self.factory()
        with self.stream(factory) as stream:
            stream.update(self.frame(), generation=0, elapsed_seconds=0)
            instances[0].provenance["synthetic"] = False
            with self.assertRaisesRegex(TrackerReplayError, "session closed"):
                stream.update(self.frame(1, 2), generation=0, elapsed_seconds=2)

    def test_runtime_provenance_copy_and_no_uninvoked_runtime_claim(self):
        factory, _, _ = self.factory()
        with self.stream(factory) as stream:
            self.assertEqual("no_backend_invoked", stream.runtime_id)
            stream.update(self.frame(), generation=0, elapsed_seconds=0)
            value = stream.runtime_provenance
            value["synthetic"] = False
            self.assertEqual({"synthetic": True}, stream.runtime_provenance)
            self.assertEqual("injected_stdlib_test_backend", stream.runtime_id)

    def test_exclusive_native_ownership_refuses_second_stream_and_replay(self):
        factory, _, _ = self.factory()
        with self.stream(factory):
            with self.assertRaisesRegex(TrackerReplayError, "already in use"):
                self.stream(factory)
            with self.assertRaisesRegex(TrackerReplayError, "already in use"):
                replay_tracker(
                    DetectionFrameBatch("synthetic_cam", "synthetic_detector", (self.frame(),)),
                    TrackerReplayConfig("bytetrack", 384, 288),
                    _backend_factory=factory,
                )
        with self.stream(factory):
            pass

    def test_unreturned_singletons_do_not_associate_or_disappear(self):
        factory, _, _ = self.factory(lambda rows: [])
        with self.stream(factory) as stream:
            first = stream.update(self.frame(), generation=0, elapsed_seconds=0)
            last = stream.update(self.frame(1, 0.2), generation=0, elapsed_seconds=0.2)
            self.assertEqual(("unassociated_singleton",), first.phases)
            self.assertNotEqual(first.assignments[0].tracking_id, last.assignments[0].tracking_id)
            first.bind(self.frame())
            with self.assertRaises(TrackerReplayError):
                first.bind(self.frame(confidence=0.8))

    def test_tentative_phase_is_explicit_without_backfilling(self):
        factory, _, _ = self.factory(tentative=True)
        config = TrackerStreamConfig(
            TrackerReplayConfig("bytetrack", 384, 288, emit_tentative_assignments=True)
        )
        with self.stream(factory, config=config) as stream:
            first = stream.update(self.frame(), generation=0, elapsed_seconds=0)
            frozen = first.to_dict()
            last = stream.update(self.frame(1, 0.2), generation=0, elapsed_seconds=0.2)
            self.assertEqual(("native_tentative",), first.phases)
            self.assertEqual(("native_emitted",), last.phases)
            self.assertEqual(first.assignments[0].tracking_id, last.assignments[0].tracking_id)
            self.assertEqual(frozen, first.to_dict())
            self.assertFalse(last.to_dict()["identity_or_role_propagation"])

    def test_namespace_changes_with_camera_config_or_session(self):
        factory, _, _ = self.factory()
        results = []
        for changes in (
            {},
            {"camera_id": "other_cam"},
            {"session_id": "1" * 32},
            {"config": TrackerStreamConfig(TrackerReplayConfig("bytetrack", 640, 288))},
        ):
            with self.stream(factory, **changes) as stream:
                results.append(
                    stream.update(self.frame(), generation=0, elapsed_seconds=0)
                    .assignments[0]
                    .tracking_id
                )
        self.assertEqual(4, len(set(results)))

    def test_invalid_configs_and_session_names_refused_before_lock(self):
        config = TrackerStreamConfig(TrackerReplayConfig("bytetrack", 384, 288))
        for change in (
            {"max_export_age_seconds": 0},
            {"max_export_age_seconds": math.inf},
            {"duplicate_cache_frames": True},
            {"duplicate_cache_frames": 257},
            {"native": None},
        ):
            with self.assertRaises(TrackerReplayError):
                replace(config, **change)
        factory, _, _ = self.factory()
        for change in (
            {"session_id": "father"},
            {"camera_id": "../private"},
            {"label": "person\n"},
            {"config": None},
        ):
            with self.assertRaises(TrackerReplayError):
                self.stream(factory, **change)

    def test_bound_refusal_and_close_are_explicit(self):
        factory, calls, _ = self.factory()
        with self.stream(factory) as stream:
            stream._frames = 5000
            with self.assertRaisesRegex(TrackerReplayError, "bound reached"):
                stream.update(self.frame(), generation=0, elapsed_seconds=0)
            self.assertEqual([], calls)
        stream.close()
        with self.assertRaisesRegex(TrackerReplayError, "closed"):
            stream.update(self.frame(), generation=0, elapsed_seconds=0)

    def test_module_remains_stdlib_only_until_optional_backend_is_called(self):
        original = __import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "ultralytics", "torch", "cv2", "lap"}:
                raise AssertionError("optional dependency imported")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            runpy.run_path(sys.modules["baby_monitor.local_runners.tracker_stream"].__file__)


if __name__ == "__main__":
    unittest.main()
