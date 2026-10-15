from __future__ import annotations

import math
import runpy
import sys
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor.local_runners.tracker_replay import (
    TrackerReplayConfig,
    TrackerReplayError,
    _current_tentative_rows,
    make_backend,
    replay_tracker,
)
from baby_monitor.object_tracker import DetectionFrame, DetectionFrameBatch, FrameDetection


class TrackerReplayTests(unittest.TestCase):
    BASE = datetime(2026, 10, 10, tzinfo=UTC)

    def batch(self, labels=(("person",), (), ("person",)), times=(0, 0.2, 0.4)):
        return DetectionFrameBatch(
            camera_id="synthetic_cam",
            detector_adapter="synthetic_detector",
            frames=tuple(
                DetectionFrame(
                    frame_id=f"f{index}",
                    occurred_at=self.BASE + timedelta(seconds=seconds),
                    detections=tuple(
                        FrameDetection(
                            detection_id=f"d{column}",
                            label=label,
                            confidence=0.9,
                            bbox_xyxy=(0.1, 0.1, 0.3, 0.6),
                        )
                        for column, label in enumerate(frame_labels)
                    ),
                )
                for index, (frame_labels, seconds) in enumerate(zip(labels, times, strict=True))
            ),
        )

    @staticmethod
    def config(**kwargs):
        return TrackerReplayConfig("bytetrack", 384, 288, **kwargs)

    @staticmethod
    def row(index, native_id=1, confidence=0.9):
        return [10.0, 10.0, 20.0, 40.0, native_id, confidence, 0, index]

    def factory(self, transform=None):
        calls = []
        instances = []
        owner = self

        class Backend:
            def update(self, detections):
                calls.append(tuple(d.label for d in detections))
                rows = [owner.row(i, i + 1, d.confidence) for i, d in enumerate(detections)]
                return transform(rows, detections) if transform else rows

        def create(config):
            backend = Backend()
            instances.append(backend)
            return backend

        return create, calls, instances

    def test_empty_frames_update_native_without_losing_original_order(self):
        batch = self.batch()
        original = batch.fingerprint
        factory, calls, _ = self.factory()
        prediction, audit = replay_tracker(batch, self.config(), _backend_factory=factory)
        prediction.bind(batch)
        self.assertEqual([("person",), (), ("person",)], calls)
        self.assertEqual(["f0", "f2"], [a.frame_id for a in prediction.assignments])
        self.assertEqual(
            prediction.assignments[0].tracking_id, prediction.assignments[1].tracking_id
        )
        self.assertEqual(original, batch.fingerprint)
        self.assertEqual(3, audit["backend_update_count"])
        self.assertEqual(1, audit["empty_frame_count"])
        self.assertFalse(audit["kalman_elapsed_dt_used"])
        self.assertFalse(audit["source_exposure_continuity_established"])
        self.assertFalse(audit["identity_or_age_evaluated"])
        self.assertEqual("injected_stdlib_test_backend", prediction.runtime_id)

    def test_all_empty_batch_does_not_import_or_construct_backend(self):
        batch = self.batch(labels=((), (), ()))
        factory, calls, instances = self.factory()
        prediction, audit = replay_tracker(batch, self.config(), _backend_factory=factory)
        self.assertEqual((), prediction.assignments)
        self.assertEqual([], calls)
        self.assertEqual([], instances)
        self.assertEqual(0, audit["backend_update_count"])
        self.assertFalse(audit["backend_invoked"])
        self.assertFalse(audit["dependencies_verified"])
        self.assertEqual("no_backend_invoked", prediction.runtime_id)

    def test_native_unreturned_are_unique_singletons_not_backend_detections(self):
        factory, calls, _ = self.factory(lambda rows, detections: [])
        prediction, audit = replay_tracker(self.batch(), self.config(), _backend_factory=factory)
        self.assertEqual(2, len({a.tracking_id for a in prediction.assignments}))
        self.assertEqual(2, audit["unassociated_singleton_count"])
        self.assertEqual(0, audit["native_emitted_assignment_count"])
        self.assertTrue(all("unassociated_" in a.tracking_id for a in prediction.assignments))
        self.assertEqual([("person",), (), ("person",)], calls)

    def test_low_confidence_unreturned_remains_in_prediction(self):
        batch = self.batch()
        first = batch.frames[0]
        batch = replace(
            batch,
            frames=(
                replace(first, detections=(replace(first.detections[0], confidence=0.05),)),
                *batch.frames[1:],
            ),
        )
        factory, _, _ = self.factory(lambda rows, detections: [r for r in rows if r[5] >= 0.1])
        prediction, audit = replay_tracker(batch, self.config(), _backend_factory=factory)
        self.assertEqual(2, len(prediction.assignments))
        self.assertEqual(1, audit["unassociated_singleton_count"])

    def test_identical_boxes_and_out_of_order_native_rows_use_explicit_index(self):
        batch = self.batch(labels=(("person", "person"),), times=(0,))
        factory, _, _ = self.factory(lambda rows, _: list(reversed(rows)))
        prediction, _ = replay_tracker(batch, self.config(), _backend_factory=factory)
        self.assertEqual(["d0", "d1"], [a.detection_id for a in prediction.assignments])
        self.assertTrue(prediction.assignments[0].tracking_id.endswith("_n1"))
        self.assertTrue(prediction.assignments[1].tracking_id.endswith("_n2"))

    def test_labels_processed_sequentially_and_have_separate_id_namespaces(self):
        batch = self.batch(labels=(("person", "head"), (), ("person", "head")))
        factory, calls, instances = self.factory()
        prediction, audit = replay_tracker(batch, self.config(), _backend_factory=factory)
        self.assertEqual([("head",), (), ("head",), ("person",), (), ("person",)], calls)
        self.assertEqual(2, len(instances))
        self.assertEqual(6, audit["backend_update_count"])
        ids = [a.tracking_id for a in prediction.assignments]
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(ids[:2], ids[2:])

    def test_elapsed_gap_resets_namespaced_segment_including_empty_frames(self):
        factory, _, instances = self.factory()
        batch = self.batch(times=(0, 2, 2.2))
        prediction, audit = replay_tracker(batch, self.config(), _backend_factory=factory)
        self.assertEqual(2, len(instances))
        self.assertNotEqual(
            prediction.assignments[0].tracking_id, prediction.assignments[1].tracking_id
        )
        self.assertEqual(2, audit["label_segment_count"])
        self.assertEqual(2, audit["maximum_input_gap_seconds"])

    def test_camera_and_config_scope_prevent_cross_batch_identity(self):
        factory, _, _ = self.factory()
        batch = self.batch()
        first, _ = replay_tracker(batch, self.config(), _backend_factory=factory)
        second, _ = replay_tracker(
            replace(batch, camera_id="other_cam"), self.config(), _backend_factory=factory
        )
        third, _ = replay_tracker(
            batch, replace(self.config(), canvas_width=640), _backend_factory=factory
        )
        self.assertNotEqual(first.assignments[0].tracking_id, second.assignments[0].tracking_id)
        self.assertNotEqual(first.assignments[0].tracking_id, third.assignments[0].tracking_id)

    def test_repeat_is_deterministic(self):
        factory, _, _ = self.factory(lambda rows, _: rows[:1])
        batch = self.batch(labels=(("person", "person"), (), ("person", "person")))
        first = replay_tracker(batch, self.config(), _backend_factory=factory)
        second = replay_tracker(batch, self.config(), _backend_factory=factory)
        self.assertEqual(first, second)

    def test_malformed_backend_outputs_fail_closed(self):
        bad_rows = [
            None,
            (),
            [self.row(0), self.row(0)],
            [[0] * 7],
            [self.row(-1)],
            [self.row(1)],
            [self.row(0.5)],
            [self.row(0, native_id=0)],
            [self.row(0, native_id=1.5)],
            [self.row(0, confidence=0.8)],
        ]
        for position, value in ((0, math.nan), (0, 10**10000), (2, 5), (4, True), (6, 1)):
            row = self.row(0)
            row[position] = value
            bad_rows.append([row])
        batch = self.batch(labels=(("person",),), times=(0,))
        for bad in bad_rows:
            with self.subTest(output_type=type(bad).__name__):
                factory, _, _ = self.factory(lambda rows, detections, value=bad: value)
                with self.assertRaises(TrackerReplayError):
                    replay_tracker(batch, self.config(), _backend_factory=factory)

    def test_duplicate_native_ids_for_two_detections_fail_closed(self):
        factory, _, _ = self.factory(lambda rows, _: [self.row(0), self.row(1)])
        with self.assertRaises(TrackerReplayError):
            replay_tracker(
                self.batch(labels=(("person", "person"),), times=(0,)),
                self.config(),
                _backend_factory=factory,
            )

    def test_invalid_configs(self):
        changes = [
            {"algorithm": "deepocsort"},
            {"canvas_width": 0},
            {"canvas_height": True},
            {"track_low_thresh": 0.25},
            {"new_track_thresh": 0.2},
            {"track_high_thresh": math.nan},
            {"match_thresh": math.inf},
            {"match_thresh": 10**10000},
            {"fuse_score": 1},
            {"track_buffer": 0},
            {"max_input_gap_seconds": 0},
            {"max_input_gap_seconds": 10**10000},
            {"emit_tentative_assignments": 1},
        ]
        for change in changes:
            with self.subTest(fields=list(change)), self.assertRaises(TrackerReplayError):
                replace(self.config(), **change)

    def test_optional_backend_refuses_nonisolated_network_before_import(self):
        with (
            patch("socket.if_nameindex", return_value=[(1, "lo"), (2, "eth0")]),
            self.assertRaisesRegex(TrackerReplayError, "isolated offline"),
        ):
            make_backend(self.config())

    def test_global_native_counter_is_locked_for_entire_replay(self):
        from baby_monitor.local_runners.tracker_replay import _REPLAY_LOCK

        observed = []
        factory, _, _ = self.factory(lambda rows, _: observed.append(_REPLAY_LOCK.locked()) or rows)
        replay_tracker(self.batch(), self.config(), _backend_factory=factory)
        self.assertEqual([True, True, True], observed)
        self.assertFalse(_REPLAY_LOCK.locked())

    def test_tentative_selector_excludes_lost_stale_and_activated_tracks(self):
        def track(activated=False, state=1, frame=3):
            return SimpleNamespace(
                is_activated=activated, state=state, frame_id=frame, result=self.row(0)
            )

        tracker = SimpleNamespace(
            frame_id=3,
            tracked_stracks=[
                track(),
                track(activated=True),
                track(state=2),
                track(frame=2),
            ],
        )
        self.assertEqual([self.row(0)], _current_tentative_rows(tracker, 1))

    def test_tentative_opt_in_preserves_native_id_with_explicit_phase(self):
        owner = self

        class Backend:
            tentative_indices = ()
            calls = 0

            def update(self, detections):
                self.calls += 1
                self.tentative_indices = (0,) if self.calls == 1 else ()
                return [owner.row(0)] if detections else []

        prediction, audit = replay_tracker(
            self.batch(),
            self.config(emit_tentative_assignments=True),
            _backend_factory=lambda _: Backend(),
        )
        self.assertEqual(
            prediction.assignments[0].tracking_id, prediction.assignments[1].tracking_id
        )
        self.assertEqual(1, audit["native_tentative_assignment_count"])
        self.assertEqual(1, audit["native_emitted_assignment_count"])
        self.assertEqual(0, audit["unassociated_singleton_count"])
        self.assertEqual(
            ["native_tentative", "native_emitted"],
            [a["status"] for a in audit["assignment_provenance"]],
        )
        self.assertFalse(audit["deployment_approved"])

    def test_tentative_metadata_cannot_enable_itself_without_configuration(self):
        owner = self

        class Backend:
            tentative_indices = (0,)

            def update(self, detections):
                return [owner.row(0)]

        with self.assertRaises(TrackerReplayError):
            replay_tracker(self.batch(), self.config(), _backend_factory=lambda _: Backend())

    def test_invalid_tentative_metadata_fails_closed(self):
        owner = self
        batch = self.batch(labels=(("person", "person"),), times=(0,))
        for values in ([0], (0, 0), (1,), (True,), (-1,), (2,)):

            class Backend:
                tentative_indices = values

                def update(self, detections):
                    return [owner.row(0)]

            with (
                self.subTest(metadata_type=type(values).__name__),
                self.assertRaises(TrackerReplayError),
            ):
                replay_tracker(
                    batch,
                    self.config(emit_tentative_assignments=True),
                    _backend_factory=lambda _, backend_type=Backend: backend_type(),
                )

    def test_module_import_does_not_require_optional_vision_dependencies(self):
        forbidden = {"numpy", "ultralytics", "torch", "cv2", "lap"}
        original = __import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in forbidden:
                raise AssertionError("optional dependency imported by stdlib module")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            runpy.run_path(sys.modules["baby_monitor.local_runners.tracker_replay"].__file__)


if __name__ == "__main__":
    unittest.main()
