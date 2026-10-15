from __future__ import annotations

import builtins
import hashlib
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from scripts import compare_yolo_prediction_heads as runner


class _Tensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def _result(classes=(0,), confidences=(0.5,), boxes=None):
    return SimpleNamespace(
        boxes=SimpleNamespace(
            cls=_Tensor(list(classes)),
            conf=_Tensor(list(confidences)),
            xyxyn=_Tensor([[0.1, 0.2, 0.4, 0.8]] if boxes is None else boxes),
        )
    )


def _head_model(flag, *, task="detect"):
    head = SimpleNamespace(end2end=flag, cv2=object(), cv3=object(), cv4=object())
    wrapped_head = SimpleNamespace(end2end=flag)
    wrapped = SimpleNamespace(end2end=flag, model=[wrapped_head])
    predictor = SimpleNamespace(
        args=SimpleNamespace(end2end=flag),
        model=SimpleNamespace(end2end=flag, backend=SimpleNamespace(end2end=flag), model=wrapped),
    )
    return SimpleNamespace(
        task=task, predictor=predictor, model=SimpleNamespace(end2end=flag, model=[head])
    )


class PredictionHeadRunnerTests(unittest.TestCase):
    def test_head_state_requires_all_seven_effective_flags(self):
        for expected in (False, True):
            valid = runner.head_state(_head_model(expected), expected)
            self.assertEqual(8, len(valid))
            self.assertTrue(valid["one_to_many_heads_present"])
            paths = (
                lambda m: m.predictor.args,
                lambda m: m.predictor.model,
                lambda m: m.predictor.model.backend,
                lambda m: m.predictor.model.model,
                lambda m: m.predictor.model.model.model[-1],
                lambda m: m.model,
                lambda m: m.model.model[-1],
            )
            for index, target in enumerate(paths):
                for invalid in (not expected, int(expected), None):
                    model = _head_model(expected)
                    target(model).end2end = invalid
                    with (
                        self.subTest(expected=expected, index=index, invalid=invalid),
                        self.assertRaises(ValueError),
                    ):
                        runner.head_state(model, expected)

    def test_one_to_many_requires_retained_detector_and_pose_heads(self):
        for task, key in (("detect", "cv2"), ("detect", "cv3"), ("pose", "cv4")):
            model = _head_model(False, task=task)
            setattr(model.model.model[-1], key, None)
            with self.subTest(task=task, key=key), self.assertRaises(ValueError):
                runner.head_state(model, False)
        fused = _head_model(True)
        fused.model.model[-1].cv2 = None
        fused.model.model[-1].cv3 = None
        self.assertFalse(runner.head_state(fused, True)["one_to_many_heads_present"])

    def test_box_conversion_accepts_empty_and_bounded_person_boxes(self):
        self.assertEqual([], runner.boxes_from_result(SimpleNamespace(boxes=None)))
        self.assertEqual([], runner.boxes_from_result(_result((), (), [])))
        self.assertEqual(
            [{"confidence": 0.5, "bbox_xyxy": [0.1, 0.2, 0.4, 0.8]}],
            runner.boxes_from_result(_result()),
        )
        boxes = [[0.0, 0.0, 1.0, 1.0] for _ in range(300)]
        self.assertEqual(300, len(runner.boxes_from_result(_result([0] * 300, [0.1] * 300, boxes))))

    def test_box_conversion_rejects_inconsistent_arrays_and_more_than_300(self):
        for result in (
            _result((0, 0)),
            _result((0,), ()),
            _result([0] * 301, [0.5] * 301, [[0, 0, 1, 1]] * 301),
        ):
            with self.assertRaises(ValueError):
                runner.boxes_from_result(result)

    def test_box_conversion_rejects_nonperson_and_bool_classes(self):
        for value in (1, -1, 0.5, False, None, "0", float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.boxes_from_result(_result((value,)))

    def test_box_conversion_rejects_invalid_confidence_types_and_bounds(self):
        for value in (True, None, "0.5", float("nan"), float("inf"), 10**1000, 0.099, 1.01):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.boxes_from_result(_result(confidences=(value,)))

    def test_box_conversion_rejects_invalid_bbox_types_bounds_and_area(self):
        boxes = (
            [0, 0, 0, 1],
            [0, 1, 1, 1],
            [-0.1, 0, 1, 1],
            [0, 0, 1.1, 1],
            [0, 0, float("inf"), 1],
            [0, 0, float("nan"), 1],
            [False, 0, 1, 1],
            [None, 0, 1, 1],
            ["0", 0, 1, 1],
            [0, 0, 1],
            (0, 0, 1, 1),
        )
        for bbox in boxes:
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                runner.boxes_from_result(_result(boxes=[bbox]))

    def test_box_conversion_rejects_non_list_tensor_payloads(self):
        for attribute in ("cls", "conf", "xyxyn"):
            for value in (None, (), "invalid"):
                result = _result()
                setattr(result.boxes, attribute, _Tensor(value))
                with self.subTest(attribute=attribute, value=value), self.assertRaises(ValueError):
                    runner.boxes_from_result(result)

    def _process(self):
        process = MagicMock()
        process.pid = 424242
        process.__enter__.return_value = process
        return process

    def test_child_success_has_isolated_process_group_and_discarded_logs(self):
        process = self._process()
        process.wait.return_value = 0
        environment = {"fixture": "safe"}
        with (
            patch.object(runner.subprocess, "Popen", return_value=process) as popen,
            patch.object(runner.time, "monotonic", side_effect=(10.0, 13.0)),
        ):
            self.assertEqual(3.0, runner.run_child(("fake-runtime",), environment=environment))
        popen.assert_called_once_with(
            ("fake-runtime",),
            cwd=runner.REPO,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        process.wait.assert_called_once_with(timeout=120)

    def test_failed_child_does_not_expose_runtime_logs(self):
        process = self._process()
        process.wait.return_value = 3
        with (
            patch.object(runner.subprocess, "Popen", return_value=process),
            self.assertRaisesRegex(ValueError, "no runtime logs disclosed"),
        ):
            runner.run_child(("fake-runtime", "private_argument"), environment={})

    def test_timeout_and_interrupt_kill_only_child_group_then_wait(self):
        for error in (subprocess.TimeoutExpired("fake", 120), KeyboardInterrupt()):
            process = self._process()
            process.wait.side_effect = [error, 0]
            with (
                patch.object(runner.subprocess, "Popen", return_value=process),
                patch.object(runner.os, "killpg") as kill,
                self.assertRaises(type(error)),
            ):
                runner.run_child(("fake-runtime",), environment={})
            kill.assert_called_once_with(process.pid, signal.SIGKILL)
            self.assertEqual([call(timeout=120), call()], process.wait.call_args_list)

    def test_exited_child_race_still_waits_and_preserves_original_timeout(self):
        process = self._process()
        error = subprocess.TimeoutExpired("fake", 120)
        process.wait.side_effect = [error, 0]
        with (
            patch.object(runner.subprocess, "Popen", return_value=process),
            patch.object(runner.os, "killpg", side_effect=ProcessLookupError),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            runner.run_child(("fake-runtime",), environment={})
        self.assertEqual([call(timeout=120), call()], process.wait.call_args_list)


class PredictionHeadWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root.chmod(0o700)
        self.frame = self.root / "synthetic.jpg"
        self.frame.write_bytes(b"synthetic bytes, not an actual photograph")
        self.frame.chmod(0o600)
        self.request_path = self.root / "request.json"
        self.request = {
            "version": 1,
            "model_id": "yolo26m",
            "weight_path": str(self.root / "fake.pt"),
            "weight_sha256": "a" * 64,
            "end2end": False,
            "frames": [
                {
                    "review_id": "frame_01",
                    "frame_sha256": hashlib.sha256(self.frame.read_bytes()).hexdigest(),
                    "path": str(self.frame),
                }
            ],
            "parent_net_namespace": "net:[parent-fixture]",
            "output": str(self.root / "result.json"),
            "device": "cpu",
            "labels_source_sha256": "b" * 64,
        }

    def _save_request(self):
        self.request_path.write_text(json.dumps(self.request), encoding="utf-8")
        self.request_path.chmod(0o600)

    def test_invalid_request_and_same_namespace_fail_before_weights_or_optional_imports(self):
        invalids = (
            {"version": True},
            {"version": 2},
            {"end2end": 0},
            {"device": "invalid"},
            {"model_id": "unknown"},
            {"frames": []},
            {"frames": [{}] * 33},
            {"parent_net_namespace": "net:[child-fixture]"},
        )
        original_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "torchvision", "ultralytics"}:
                raise AssertionError("must not import optional model runtimes")
            return original_import(name, *args, **kwargs)

        for change in invalids:
            original = self.request.copy()
            self.request.update(change)
            self._save_request()
            with (
                self.subTest(change=change),
                patch.object(runner.os, "readlink", return_value="net:[child-fixture]"),
                patch.object(
                    runner.importlib.metadata, "version", return_value=runner.RUNTIME_VERSION
                ),
                patch.object(
                    runner.YoloModelManifest,
                    "read",
                    side_effect=AssertionError("must not read weights"),
                ),
                patch.object(builtins, "__import__", side_effect=guard),
                self.assertRaises(ValueError),
            ):
                runner.worker(self.request_path)
            self.request = original

    def _fake_runtime(self, *, fresh=True, missing_branch=None):
        package = self.root / "fake_package"
        package.mkdir()
        for relative in (
            "engine/predictor.py",
            "nn/tasks.py",
            "nn/modules/head.py",
            "nn/autobackend.py",
            "nn/backends/pytorch.py",
            "utils/nms.py",
        ):
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic reviewed source")
        torch, torchvision, ultralytics = (
            ModuleType(name) for name in ("torch", "torchvision", "ultralytics")
        )
        torch.set_num_threads = Mock()
        torchvision.__version__ = "fixture"
        ultralytics.__file__ = str(package / "__init__.py")
        head = type("Detect", (), {})()
        for key in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            setattr(head, key, object())
        if missing_branch:
            setattr(head, missing_branch, None)
        head.end2end = True
        model = SimpleNamespace(
            predictor=None if fresh else object(),
            names={0: "person"},
            task="detect",
            model=SimpleNamespace(end2end=True, model=[head]),
        )
        callbacks = []
        model.add_callback = Mock(side_effect=lambda _name, callback: callbacks.append(callback))

        def predict(**kwargs):
            expected = kwargs["end2end"]
            head.end2end = expected
            model.predictor = SimpleNamespace(
                args=SimpleNamespace(end2end=expected),
                model=SimpleNamespace(
                    end2end=expected, backend=SimpleNamespace(end2end=expected), model=model.model
                ),
            )
            for callback in callbacks:
                callback(model.predictor)
            return [_result()]

        model.predict = Mock(side_effect=predict)
        ultralytics.YOLO = Mock(return_value=model)
        return {"torch": torch, "torchvision": torchvision, "ultralytics": ultralytics}, model

    def _worker_stack(self, modules):
        stack = ExitStack()
        stack.enter_context(patch.object(runner.socket, "if_nameindex", return_value=[(1, "lo")]))
        stack.enter_context(patch.object(runner.os, "readlink", return_value="net:[child-fixture]"))
        stack.enter_context(
            patch.object(runner.importlib.metadata, "version", return_value=runner.RUNTIME_VERSION)
        )
        manifest = Mock(sha256="a" * 64)
        stack.enter_context(patch.object(runner.YoloModelManifest, "read", return_value=manifest))
        stack.enter_context(patch.dict(sys.modules, modules))
        return stack, manifest

    def test_non_loopback_interface_blocks_spoofed_namespace_before_weights(self):
        self._save_request()
        with (
            patch.object(runner.os, "readlink", return_value="net:[different-fixture]"),
            patch.object(runner.socket, "if_nameindex", return_value=[(1, "lo"), (2, "eth0")]),
            patch.object(
                runner.importlib.metadata,
                "version",
                side_effect=AssertionError("runtime not reached"),
            ),
            patch.object(
                runner.YoloModelManifest, "read", side_effect=AssertionError("weights not reached")
            ),
            self.assertRaisesRegex(ValueError, "isolated inputs"),
        ):
            runner.worker(self.request_path)

    def test_fake_yolo_branch_is_set_before_first_prediction_and_output_bound(self):
        self._save_request()
        modules, model = self._fake_runtime()
        stack, manifest = self._worker_stack(modules)
        with stack:
            runner.worker(self.request_path)
        modules["ultralytics"].YOLO.assert_called_once_with(self.request["weight_path"])
        self.assertEqual(2, manifest.verify_weight.call_count)
        model.predict.assert_called_once()
        options = model.predict.call_args.kwargs
        self.assertIs(False, options["end2end"])
        self.assertEqual([0], options["classes"])
        self.assertEqual(300, options["max_det"])
        self.assertFalse(options["save"])
        output = json.loads((self.root / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(self.request["labels_source_sha256"], output["labels_source_sha256"])
        self.assertTrue(output["fresh_model_instance"])
        self.assertTrue(output["network_namespace_isolated"])
        self.assertEqual(6, len(output["runtime_code_sha256s"]))
        self.assertFalse(output["end2end"])
        self.assertEqual(
            self.request["frames"][0]["frame_sha256"], output["rows"][0]["frame_sha256"]
        )

    def test_reused_fake_model_is_rejected_before_predict(self):
        self._save_request()
        modules, model = self._fake_runtime(fresh=False)
        stack, _ = self._worker_stack(modules)
        with stack, self.assertRaisesRegex(ValueError, "fresh"):
            runner.worker(self.request_path)
        model.predict.assert_not_called()
        self.assertFalse((self.root / "result.json").exists())

    def test_missing_fake_checkpoint_branch_is_rejected_before_predict(self):
        self._save_request()
        modules, model = self._fake_runtime(missing_branch="cv2")
        stack, _ = self._worker_stack(modules)
        with stack, self.assertRaisesRegex(ValueError, "preserve both"):
            runner.worker(self.request_path)
        model.predict.assert_not_called()
        self.assertFalse((self.root / "result.json").exists())

    def test_reviewed_weight_and_frame_hash_mismatch_prevent_fake_model_construction(self):
        self._save_request()
        modules, model = self._fake_runtime()
        for change in ("weight", "frame"):
            with self.subTest(change=change):
                original = self.request["weight_sha256"]
                if change == "weight":
                    self.request["weight_sha256"] = "f" * 64
                    self._save_request()
                else:
                    self.frame.write_bytes(b"changed synthetic input")
                stack, _ = self._worker_stack(modules)
                with stack, self.assertRaises(ValueError):
                    runner.worker(self.request_path)
                modules["ultralytics"].YOLO.assert_not_called()
                model.predict.assert_not_called()
                self.request["weight_sha256"] = original
                self._save_request()
        self.assertFalse((self.root / "result.json").exists())


class PredictionHeadParentTests(unittest.TestCase):
    def test_parent_child_environment_uses_checkout_source_and_disables_autoinstall(self):
        class StopBeforeChild(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.chmod(0o700)
            source = root / "source"
            source.mkdir(mode=0o700)
            (source / "fixture_camera_01_opaque.mp4").write_bytes(b"synthetic video placeholder")
            item = {
                "review_id": "frame_01",
                "camera_id": "camera_01",
                "timestamp_seconds": 0.0,
                "source_frame_sha256": "c" * 64,
                "frame_filename": "frame_01.jpg",
                "media_admission_sha256": "d" * 64,
            }
            selection = {"items": [item]}
            labels = {
                "model_generated_labels": True,
                "human_verified": False,
                "ground_truth": False,
                "items": [{**item, "visible_person": "unknown"}],
            }
            manifest = Mock(sha256="e" * 64)
            admission = Mock(fingerprint="d" * 64)
            with (
                patch.object(
                    runner,
                    "read_private_json_with_sha",
                    side_effect=[(selection, "a" * 64), (labels, "b" * 64)],
                ),
                patch.object(runner, "verify_frame"),
                patch.object(runner, "private_file"),
                patch.object(runner.ExternalSourceAdmission, "read", return_value=admission),
                patch.object(
                    runner, "runtime_versions", return_value={"ultralytics": runner.RUNTIME_VERSION}
                ),
                patch.object(runner.YoloModelManifest, "read", return_value=manifest),
                patch.object(runner.YoloPoseModelManifest, "read", return_value=manifest),
                patch.object(runner.os, "readlink", return_value="net:[parent-fixture]"),
                patch.dict(runner.os.environ, {"PYTHONPATH": "/fixture-stale-pythonpath"}),
                patch.object(runner, "run_child", side_effect=StopBeforeChild) as child,
                self.assertRaises(StopBeforeChild),
            ):
                runner.compare(
                    source_root=source,
                    output_root=root / "output",
                    vision_python=root / "fake-python",
                    weights=root / "fake-weights",
                    device="cpu",
                )
            child.assert_called_once()
            arguments = child.call_args.args[0]
            environment = child.call_args.kwargs["environment"]
            self.assertEqual(str(runner.REPO / "src"), environment["PYTHONPATH"])
            self.assertEqual("false", environment["YOLO_AUTOINSTALL"])
            self.assertEqual("false", environment["YOLO_VERBOSE"])
            self.assertEqual(runner.ISOLATION, arguments[: len(runner.ISOLATION)])
            self.assertEqual(str(root / "fake-python"), arguments[len(runner.ISOLATION)])


if __name__ == "__main__":
    unittest.main()
