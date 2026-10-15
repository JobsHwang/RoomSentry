from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from scripts import compare_yolo_pts_sampling as runner
from scripts.pts_sampling_diagnostic import plan_pts_sampling
from tests import test_prediction_head_runner as fixtures


class _Capture:
    def __init__(self, pts, *, fps=2.0, frames=None, opened=True):
        self.pts = pts
        self.fps = fps
        self.frames = frames if frames is not None else [SimpleNamespace(size=1) for _ in pts]
        self.opened = opened
        self.index = 0
        self.release = Mock()

    def isOpened(self):
        return self.opened

    def get(self, property_id):
        return self.fps if property_id == 1 else self.pts[self.index - 1] * 1000

    def read(self):
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame


def _cv2(*captures):
    module = ModuleType("cv2")
    module.CAP_PROP_FPS = 1
    module.CAP_PROP_POS_MSEC = 2
    module.VideoCapture = Mock(side_effect=captures)
    return module


def _row(index, timestamp, confidences=()):
    return {
        "decoded_frame_index": index,
        "timestamp_seconds": timestamp,
        "boxes": [
            {"confidence": confidence, "bbox_xyxy": [0.1, 0.2, 0.5, 0.8]}
            for confidence in confidences
        ],
    }


class PtsSamplingDecodeTests(unittest.TestCase):
    def test_first_pass_records_every_real_pts_including_vfr_and_releases_capture(self):
        pts = (0.0, 0.1, 0.6, 2.8)
        capture = _Capture(pts)
        self.assertEqual(
            (2.0, pts), runner.decode_timeline(Path("synthetic.mp4"), cv2_module=_cv2(capture))
        )
        self.assertEqual(4, capture.index)
        capture.release.assert_called_once()

    def test_first_pass_rejects_unopened_empty_invalid_frames_and_frame_limit(self):
        cases = (
            _Capture((0.0,), opened=False),
            _Capture(()),
            _Capture((0.0,), frames=[None]),
            _Capture((0.0,), frames=[SimpleNamespace(size=0)]),
            _Capture((0.0, 0.5, 1.0)),
        )
        for capture in cases:
            with (
                self.subTest(capture=capture),
                patch.object(runner, "MAX_FRAMES", 2),
                self.assertRaises(ValueError),
            ):
                runner.decode_timeline(Path("synthetic.mp4"), cv2_module=_cv2(capture))
            capture.release.assert_called_once()

    def test_first_pass_rejects_unknown_nonfinite_nonmonotonic_clock_and_bad_fps(self):
        cases = (
            _Capture((0.0, float("nan"))),
            _Capture((0.0, float("inf"))),
            _Capture((-0.1, 0.0)),
            _Capture((0.5, 0.4)),
            _Capture((0.0, 0.0)),
            _Capture((0.0,), fps=0),
            _Capture((0.0,), fps=float("nan")),
        )
        for capture in cases:
            with self.subTest(pts=capture.pts, fps=capture.fps), self.assertRaises(ValueError):
                runner.decode_timeline(Path("synthetic.mp4"), cv2_module=_cv2(capture))
            capture.release.assert_called_once()

    def test_union_predicts_each_selected_frame_once_with_fixed_person_only_kwargs(self):
        pts = (0.0, 0.5, 2.2, 3.0)
        capture = _Capture(pts)
        model = SimpleNamespace(predict=Mock(return_value=[fixtures._result()]))
        with patch.object(runner, "head_state", return_value={}) as head:
            rows = runner.predict_union(
                Path("synthetic.mp4"),
                cv2_module=_cv2(capture),
                model=model,
                reference_pts=pts,
                selected_indices=(0, 2),
            )
        self.assertEqual([0, 2], [row["decoded_frame_index"] for row in rows])
        self.assertEqual([0.0, 2.2], [row["timestamp_seconds"] for row in rows])
        self.assertEqual(2, model.predict.call_count)
        self.assertEqual(2, head.call_count)
        for call, index in zip(model.predict.call_args_list, (0, 2), strict=True):
            self.assertEqual(
                {
                    "source": capture.frames[index],
                    "conf": 0.1,
                    "classes": [0],
                    "imgsz": 640,
                    "device": "0",
                    "end2end": True,
                    "iou": 0.7,
                    "max_det": 300,
                    "agnostic_nms": False,
                    "compile": False,
                    "verbose": False,
                    "save": False,
                    "save_txt": False,
                    "save_crop": False,
                    "show": False,
                },
                call.kwargs,
            )
        self.assertEqual(4, capture.index)
        capture.release.assert_called_once()

    def test_invalid_union_is_rejected_before_open_or_predict(self):
        for selection in ((), (1, 0), (0, 0), (True,), (-1,), (2,), tuple(range(257))):
            module, model = _cv2(), Mock()
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                runner.predict_union(
                    Path("synthetic.mp4"),
                    cv2_module=module,
                    model=model,
                    reference_pts=(0.0, 0.5),
                    selected_indices=selection,
                )
            module.VideoCapture.assert_not_called()
            model.predict.assert_not_called()

    def test_replay_checks_unselected_frames_and_rejects_clock_or_count_changes(self):
        cases = (
            _Capture((0.0, 0.6, 1.0)),  # Changed frame is not selected.
            _Capture((0.0, float("nan"), 1.0)),
            _Capture((0.0, 0.5)),
            _Capture((0.0, 0.5, 1.0, 1.5)),
            _Capture((0.0, 0.5, 1.0), frames=[SimpleNamespace(size=1), None]),
            _Capture((0.0, 0.5, 1.0), opened=False),
        )
        for capture in cases:
            with (
                self.subTest(pts=capture.pts, opened=capture.opened),
                patch.object(runner, "head_state", return_value={}),
                self.assertRaises(ValueError),
            ):
                runner.predict_union(
                    Path("synthetic.mp4"),
                    cv2_module=_cv2(capture),
                    model=SimpleNamespace(predict=Mock(return_value=[fixtures._result()])),
                    reference_pts=(0.0, 0.5, 1.0),
                    selected_indices=(0,),
                )
            capture.release.assert_called_once()

    def test_prediction_cardinality_and_head_failure_release_capture(self):
        for predictions in (None, [], [fixtures._result(), fixtures._result()]):
            capture = _Capture((0.0,))
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                runner.predict_union(
                    Path("synthetic.mp4"),
                    cv2_module=_cv2(capture),
                    model=SimpleNamespace(predict=Mock(return_value=predictions)),
                    reference_pts=(0.0,),
                    selected_indices=(0,),
                )
            capture.release.assert_called_once()
        capture = _Capture((0.0,))
        with (
            patch.object(runner, "head_state", side_effect=ValueError("head changed")),
            self.assertRaisesRegex(ValueError, "head changed"),
        ):
            runner.predict_union(
                Path("synthetic.mp4"),
                cv2_module=_cv2(capture),
                model=SimpleNamespace(predict=Mock(return_value=[fixtures._result()])),
                reference_pts=(0.0,),
                selected_indices=(0,),
            )
        capture.release.assert_called_once()


class PtsSamplingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan_pts_sampling(
            (0.0, 0.5, 1.0, 2.2, 2.4, 3.0, 4.1, 4.8), native_fps=2, sample_fps=0.5
        )
        self.rows = [_row(0, 0.0), _row(3, 2.2, (0.35,)), _row(4, 2.4, (0.1,)), _row(6, 4.1)]

    def test_summary_preserves_each_schedule_denominator_and_shared_raw_predictions(self):
        original = copy.deepcopy(self.rows)
        summary = runner.summarize_schedule(self.plan, self.rows, threshold=0.35)
        self.assertEqual(4, summary["unique_prediction_count"])
        self.assertEqual(2, summary["branches"]["decoded_stride"]["sample_count"])
        self.assertEqual(2, summary["branches"]["decoded_stride"]["zero_person_box_samples"])
        self.assertEqual(0, summary["branches"]["decoded_stride"]["person_box_count"])
        self.assertEqual(3, summary["branches"]["source_pts_deadline"]["sample_count"])
        self.assertEqual(2, summary["branches"]["source_pts_deadline"]["zero_person_box_samples"])
        self.assertEqual(1, summary["branches"]["source_pts_deadline"]["person_box_count"])
        self.assertTrue(summary["shared_frame_results_identical"])
        for flag in ("accuracy_evaluated", "ground_truth", "deployment_approved"):
            self.assertIs(False, summary[flag])
        self.assertEqual(original, self.rows)

    def test_summary_rejects_missing_extra_duplicate_and_plan_unbound_rows(self):
        cases = (
            self.rows[:-1],
            self.rows + [_row(7, 4.8)],
            self.rows + [self.rows[0]],
            [{**self.rows[0], "timestamp_seconds": 0.1}, *self.rows[1:]],
        )
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                runner.summarize_schedule(self.plan, rows, threshold=0.35)

    def test_summary_rejects_invalid_thresholds_boxes_and_row_types(self):
        for threshold in (True, None, "0.35", float("nan"), 0.099, 1.1):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                runner.summarize_schedule(self.plan, self.rows, threshold=threshold)
        bad_rows = (
            {"decoded_frame_index": True},
            {"timestamp_seconds": float("nan")},
            {"timestamp_seconds": True},
            {"boxes": [{"confidence": 0.09, "bbox_xyxy": [0, 0, 1, 1]}]},
            {"boxes": [{"confidence": 0.5, "bbox_xyxy": [0, 0, 0, 1]}]},
            {"boxes": [{"confidence": True, "bbox_xyxy": [0, 0, 1, 1]}]},
        )
        for change in bad_rows:
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.summarize_schedule(
                    self.plan, [{**self.rows[0], **change}, *self.rows[1:]], threshold=0.35
                )


class PtsSamplingWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root.chmod(0o700)
        self.output = self.root / "output"
        self.output.mkdir(mode=0o700)
        self.source, self.admission_path, self.baseline_path = (
            self.root / name for name in ("synthetic.mp4", "admission.json", "scan.json")
        )
        for path in (self.source, self.admission_path, self.baseline_path):
            path.write_bytes(b"synthetic test bytes only")
            path.chmod(0o600)
        self.pts = (0.0, 0.5, 1.0, 2.2, 2.4, 3.0, 4.1, 4.8)
        self.request_path = self.root / "request.json"
        self.request = {
            "version": 1,
            "camera_id": "camera_03",
            "source": str(self.source),
            "admission": str(self.admission_path),
            "admission_sha256": "a" * 64,
            "baseline_scan": str(self.baseline_path),
            "baseline_scan_sha256": hashlib.sha256(self.baseline_path.read_bytes()).hexdigest(),
            "weight": str(self.root / "fake.pt"),
            "weight_sha256": "b" * 64,
            "parent_net_namespace": "net:[parent-fixture]",
            "output_root": str(self.output),
        }
        self.admission = Mock(fingerprint="a" * 64)
        self.manifest = Mock(sha256="b" * 64)
        self.package = self.root / "fake_package"
        for relative in (
            "engine/predictor.py",
            "nn/tasks.py",
            "nn/modules/head.py",
            "nn/autobackend.py",
            "nn/backends/pytorch.py",
            "utils/nms.py",
        ):
            path = self.package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic runtime source")
        ledger = DetectorSamplingLedger(
            native_fps=2,
            native_fps_source="reported",
            requested_sample_fps=0.5,
            frame_step=4,
            decoded_frame_count=8,
            samples=tuple(
                DetectorSampleFrame(i, self.pts[i], "container_pts", 0, 0) for i in (0, 4)
            ),
            timestamp_fallback_frame_count=0,
            max_adjacent_container_pts_gap_seconds=1.2,
            container_pts_regression_count=0,
        )
        self.baseline = SimpleNamespace(
            source_admission_sha256="a" * 64,
            version=2,
            requested_sample_fps=0.5,
            result=SimpleNamespace(camera_id="camera_03", sampling=ledger),
        )

    def _save_request(self):
        self.request_path.write_text(json.dumps(self.request), encoding="utf-8")
        self.request_path.chmod(0o600)

    def _stack(self):
        stack = ExitStack()
        stack.enter_context(patch.object(runner.os, "readlink", return_value="net:[child-fixture]"))
        stack.enter_context(patch.object(runner.socket, "if_nameindex", return_value=[(1, "lo")]))
        stack.enter_context(
            patch.object(runner.importlib.metadata, "version", return_value=runner.RUNTIME_VERSION)
        )
        stack.enter_context(
            patch.object(
                runner.importlib.metadata,
                "distribution",
                return_value=SimpleNamespace(locate_file=lambda _name: self.package),
            )
        )
        stack.enter_context(
            patch.object(runner.ExternalSourceAdmission, "read", return_value=self.admission)
        )
        stack.enter_context(
            patch.object(runner.AdmittedDetectorScan, "read", return_value=self.baseline)
        )
        stack.enter_context(
            patch.object(runner.YoloModelManifest, "read", return_value=self.manifest)
        )
        return stack

    def _runtime(self, *, replay_pts=None, fresh=True):
        captures = (_Capture(self.pts), _Capture(self.pts if replay_pts is None else replay_pts))
        cv2 = _cv2(*captures)
        ultralytics = ModuleType("ultralytics")
        head = type("Detect", (), {})()
        model = SimpleNamespace(
            task="detect",
            model=SimpleNamespace(model=[head]),
            predictor=None if fresh else object(),
            names={0: "person"},
            predict=Mock(return_value=[fixtures._result()]),
        )

        def construct(_weight):
            self.assertTrue((self.output / "sampling-plan.json").exists())
            self.assertFalse((self.output / "result.json").exists())
            captures[0].release.assert_called_once()
            return model

        ultralytics.YOLO = Mock(side_effect=construct)
        return {"cv2": cv2, "ultralytics": ultralytics}, captures, model

    def test_worker_freezes_plan_before_fresh_model_and_predicts_union_only_once(self):
        self._save_request()
        modules, captures, model = self._runtime()
        with (
            self._stack(),
            patch.dict(sys.modules, modules),
            patch.object(runner, "head_state", return_value={"effective": True}),
        ):
            runner.worker(self.request_path)
        modules["ultralytics"].YOLO.assert_called_once_with(self.request["weight"])
        self.assertEqual(4, model.predict.call_count)
        for capture in captures:
            capture.release.assert_called_once()
        self.assertEqual(2, self.admission.verify_asset.call_count)
        self.assertEqual(2, self.manifest.verify_weight.call_count)
        frozen = json.loads((self.output / "sampling-plan.json").read_text())
        result = json.loads((self.output / "result.json").read_text())
        self.assertTrue(frozen["plan_frozen_before_inference"])
        self.assertEqual(list(self.pts), frozen["reference_pts_seconds"])
        self.assertEqual(4, frozen["union_prediction_count"])
        self.assertFalse(frozen["plan"]["equal_stride_ledger_claimed"])
        self.assertEqual([0, 3, 4, 6], [row["decoded_frame_index"] for row in result["rows"]])
        self.assertEqual(frozen["plan"]["report_sha256"], result["plan_sha256"])
        self.assertFalse(result["source_exhaustion_verified"])
        self.assertFalse(result["new_pixels_uploaded"])
        self.assertTrue(result["network_namespace_isolated"])
        self.assertEqual(6, len(result["runtime_code_sha256s"]))
        self.assertFalse(result["summary"]["accuracy_evaluated"])
        self.assertEqual(0o600, (self.output / "result.json").stat().st_mode & 0o777)

    def test_replay_mismatch_leaves_frozen_plan_but_no_final_result(self):
        self._save_request()
        modules, captures, _ = self._runtime(replay_pts=(0.0, 0.6, *self.pts[2:]))
        with (
            self._stack(),
            patch.dict(sys.modules, modules),
            patch.object(runner, "head_state", return_value={}),
            self.assertRaisesRegex(ValueError, "timestamp changed"),
        ):
            runner.worker(self.request_path)
        self.assertTrue((self.output / "sampling-plan.json").exists())
        self.assertFalse((self.output / "result.json").exists())
        for capture in captures:
            capture.release.assert_called_once()

    def test_nonfresh_model_is_rejected_without_any_prediction(self):
        self._save_request()
        modules, _, model = self._runtime(fresh=False)
        with self._stack(), patch.dict(sys.modules, modules), self.assertRaises(ValueError):
            runner.worker(self.request_path)
        model.predict.assert_not_called()
        self.assertFalse((self.output / "result.json").exists())

    def test_invalid_requests_and_network_isolation_fail_before_source_or_weight_reads(self):
        for change in ({"version": True}, {"camera_id": "camera_01"}, {"unexpected": 1}):
            original = self.request.copy()
            self.request.update(change)
            self._save_request()
            with self.subTest(change=change), self._stack(), self.assertRaises(ValueError):
                runner.worker(self.request_path)
            self.request = original
        for parent, interfaces in (
            ("net:[child-fixture]", [(1, "lo")]),
            ("net:[parent-fixture]", [(1, "lo"), (2, "eth0")]),
        ):
            self.request["parent_net_namespace"] = parent
            self._save_request()
            with (
                self.subTest(parent=parent, interfaces=interfaces),
                self._stack(),
                patch.object(runner.socket, "if_nameindex", return_value=interfaces),
                self.assertRaises(ValueError),
            ):
                runner.worker(self.request_path)
        self.admission.verify_asset.assert_not_called()
        self.manifest.verify_weight.assert_not_called()
        self.assertEqual([], list(self.output.iterdir()))

    def test_baseline_v1_or_hash_mismatch_cannot_reach_model_construction(self):
        self._save_request()
        for field, invalid in (("version", 1), ("source_admission_sha256", "f" * 64)):
            original = getattr(self.baseline, field)
            setattr(self.baseline, field, invalid)
            with self.subTest(field=field), self._stack(), self.assertRaises(ValueError):
                runner.worker(self.request_path)
            setattr(self.baseline, field, original)
        self.baseline_path.write_bytes(b"changed baseline bytes")
        with self._stack(), self.assertRaises(ValueError):
            runner.worker(self.request_path)
        self.manifest.verify_weight.assert_not_called()
        self.assertEqual([], list(self.output.iterdir()))

    def test_source_or_weight_binding_failure_prevents_optional_model_initialization(self):
        modules, _, model = self._runtime()
        for failure in ("source", "weight"):
            self._save_request()
            self.admission.verify_asset.side_effect = (
                ValueError("source bytes changed") if failure == "source" else None
            )
            self.manifest.sha256 = "f" * 64 if failure == "weight" else "b" * 64
            with (
                self.subTest(failure=failure),
                self._stack(),
                patch.dict(sys.modules, modules),
                self.assertRaises(ValueError),
            ):
                runner.worker(self.request_path)
            modules["ultralytics"].YOLO.assert_not_called()
            model.predict.assert_not_called()
            self.assertEqual([], list(self.output.iterdir()))

    def test_fallback_baseline_or_union_limit_prevents_plan_and_model_creation(self):
        self._save_request()
        original = self.baseline.result.sampling
        for failure in ("fallback", "union_limit"):
            self.baseline.result.sampling = (
                replace(original, timestamp_fallback_frame_count=1)
                if failure == "fallback"
                else original
            )
            modules, captures, model = self._runtime()
            with (
                self.subTest(failure=failure),
                self._stack(),
                patch.dict(sys.modules, modules),
                patch.object(runner, "MAX_UNION_SAMPLES", 3 if failure == "union_limit" else 256),
                self.assertRaises(ValueError),
            ):
                runner.worker(self.request_path)
            captures[0].release.assert_called_once()
            modules["ultralytics"].YOLO.assert_not_called()
            model.predict.assert_not_called()
            self.assertEqual([], list(self.output.iterdir()))


class PtsSamplingParentTests(unittest.TestCase):
    def _compare(self, root, *, tamper=False):
        root.chmod(0o700)
        source = root / "source"
        source.mkdir(mode=0o700)
        for subdir in ("admissions", "sampling-ledger-v2"):
            (source / subdir).mkdir(mode=0o700)
        for camera in runner.CAMERAS:
            for path in (
                source / f"fixture_{camera}_opaque.mp4",
                source / "admissions" / f"{camera}.source-admission.json",
                source / "sampling-ledger-v2" / f"{camera}.scan.json",
            ):
                path.write_bytes(b"synthetic parent input")
                path.chmod(0o600)
        admission = Mock(fingerprint="a" * 64)
        manifest = Mock(sha256="b" * 64)
        completed = []

        def child(arguments, *, environment):
            # A call completes and its artifacts are checked before another source starts.
            self.assertEqual(runner.ISOLATION, arguments[: len(runner.ISOLATION)])
            self.assertEqual(str(root / "fake-python"), arguments[len(runner.ISOLATION)])
            self.assertEqual(str(runner.REPO / "src"), environment["PYTHONPATH"])
            self.assertEqual("false", environment["YOLO_AUTOINSTALL"])
            self.assertEqual("false", environment["YOLO_VERBOSE"])
            self.assertEqual("1", environment["OMP_NUM_THREADS"])
            self.assertEqual("1", environment["MKL_NUM_THREADS"])
            self.assertEqual(
                ("-m", "scripts.compare_yolo_pts_sampling", "worker"), arguments[-4:-1]
            )
            request_path = Path(arguments[-1])
            request, request_sha = runner.read_private_json_with_sha(request_path)
            self.assertEqual(runner.CAMERAS[len(completed)], request["camera_id"])
            self.assertEqual("net:[parent-fixture]", request["parent_net_namespace"])
            output = request_path.parent
            pts = (0.0, 0.5, 1.0, 2.2)
            plan = plan_pts_sampling(pts, native_fps=2, sample_fps=0.5)
            rows = [_row(0, 0.0), _row(3, 2.2)]
            summary = runner.summarize_schedule(plan, rows, threshold=0.35)
            if tamper:
                summary["branches"]["decoded_stride"]["sample_count"] += 1
            runner.write_new(
                output / "sampling-plan.json",
                {
                    "request_sha256": request_sha,
                    "reference_pts_seconds": pts,
                    "plan": plan,
                    "runner_sha256": runner._sha(Path(runner.__file__)),
                    "debug_code_sha256s": runner._diagnostic_code_sha256s(),
                    "planner_sha256": runner._sha(
                        runner.REPO / "scripts/pts_sampling_diagnostic.py"
                    ),
                },
            )
            runner.write_new(
                output / "result.json",
                {
                    "request_sha256": request_sha,
                    "plan_sha256": plan["report_sha256"],
                    "rows": rows,
                    "summary": summary,
                },
            )
            completed.append(request["camera_id"])
            return 1.0

        with (
            patch.object(
                runner, "runtime_versions", return_value={"ultralytics": runner.RUNTIME_VERSION}
            ),
            patch.object(runner.YoloModelManifest, "read", return_value=manifest),
            patch.object(runner.ExternalSourceAdmission, "read", return_value=admission),
            patch.object(runner.os, "readlink", return_value="net:[parent-fixture]"),
            patch.object(runner, "run_child", side_effect=child),
        ):
            report = runner.compare(
                source_root=source,
                output_root=root / "output",
                vision_python=root / "fake-python",
                weights=root / "fake-weights",
            )
        self.assertEqual(list(runner.CAMERAS), completed)
        self.assertEqual(4, admission.verify_asset.call_count)
        return report

    def test_parent_uses_separate_isolated_children_and_recomputes_each_schedule_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._compare(root)
            self.assertEqual(list(runner.CAMERAS), [row["camera_id"] for row in report["rows"]])
            for flag in (
                "accuracy_evaluated",
                "deployment_approved",
                "new_pixels_uploaded",
                "production_configuration_changed",
            ):
                self.assertFalse(report[flag])
            self.assertEqual(report, json.loads((root / "output/comparison.json").read_text()))

    def test_parent_rejects_changed_child_denominator_without_final_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "binding is invalid"):
                self._compare(root, tamper=True)
            self.assertFalse((root / "output/comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
