from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import compare_yolo_sampling_rates as runner
from scripts.pts_sampling_diagnostic import plan_pts_sampling
from tests.test_household_ground_truth import _media_admission


def _write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    path.chmod(0o600)


def _rows(plan, reference):
    return [
        {"decoded_frame_index": index, "timestamp_seconds": reference[index], "boxes": []}
        for index in plan["union_decoded_frame_indices"]
    ]


class SamplingRateSummaryTests(unittest.TestCase):
    def setUp(self):
        self.reference = (0.0, 0.5, 1.0, 2.2, 2.4, 3.0, 4.1, 6.8)
        self.plan = runner.plan_sampling_rates(
            self.reference, native_fps=2, source_duration_seconds=8
        )
        self.rows = _rows(self.plan, self.reference)
        self.seconds = [4.0, *([0.2] * (len(self.rows) - 1))]

    def test_zero_box_union_preserves_each_rate_denominator_and_separates_cold_timing(self):
        summary = runner.summarize_rates(self.plan, self.reference, self.rows, self.seconds)
        self.assertEqual(len(self.rows), summary["unique_prediction_count"])
        self.assertEqual(4.0, summary["first_prediction_wall_seconds"])
        self.assertEqual(len(self.rows) - 1, summary["subsequent_prediction_count"])
        self.assertEqual(0.2, summary["subsequent_prediction_median_seconds"])
        self.assertEqual(0.2, summary["subsequent_prediction_p95_nearest_rank_seconds"])
        self.assertAlmostEqual(sum(self.seconds), summary["sum_prediction_wall_seconds"])
        for name, branch in self.plan["branches"].items():
            actual = summary["branches"][name]
            count = len(branch["selection"]["decoded_frame_indices"])
            self.assertEqual(count, actual["sample_count"])
            self.assertEqual(count, actual["zero_person_box_samples"])
            self.assertEqual(0, actual["person_box_count"])
            self.assertEqual(branch["sample_fps"], actual["requested_sample_fps"])
        for field in (
            "independent_per_rate_throughput_measured",
            "live_latency_evaluated",
            "accuracy_evaluated",
            "ground_truth",
            "deployment_approved",
        ):
            self.assertIs(False, summary[field])
        self.assertTrue(summary["timing_includes_gpu_synchronization"])
        self.assertTrue(summary["shared_frame_results_identical"])

    def test_one_call_has_no_subsequent_statistics(self):
        reference = (0.0,)
        plan = runner.plan_sampling_rates(reference, native_fps=2, source_duration_seconds=8)
        summary = runner.summarize_rates(plan, reference, _rows(plan, reference), [0.0])
        self.assertEqual(0, summary["subsequent_prediction_count"])
        self.assertIsNone(summary["subsequent_prediction_median_seconds"])
        self.assertIsNone(summary["subsequent_prediction_p95_nearest_rank_seconds"])

    def test_subsequent_p95_is_nearest_rank_and_excludes_first_prediction(self):
        seconds = [100.0, 0.7, 0.1, 0.6, 0.2, 0.5, 0.3, 0.4]
        self.assertEqual(8, len(self.rows))
        summary = runner.summarize_rates(self.plan, self.reference, self.rows, seconds)
        self.assertEqual(0.4, summary["subsequent_prediction_median_seconds"])
        self.assertEqual(0.7, summary["subsequent_prediction_p95_nearest_rank_seconds"])

    def test_missing_duplicate_extra_or_reordered_union_rows_are_rejected(self):
        variants = (
            self.rows[:-1],
            self.rows + [self.rows[0]],
            self.rows + [{"decoded_frame_index": 8, "timestamp_seconds": 7.0, "boxes": []}],
            list(reversed(self.rows)),
        )
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                runner.summarize_rates(self.plan, self.reference, rows, [0.2] * len(rows))

    def test_actual_timestamps_must_be_bound_to_reference_and_branch_plan(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["timestamp_seconds"] = 0.01
        with self.assertRaises(ValueError):
            runner.summarize_rates(self.plan, self.reference, rows, self.seconds)
        plan = copy.deepcopy(self.plan)
        next(iter(plan["branches"].values()))["selection"]["pts_seconds"][0] = 0.01
        with self.assertRaises(ValueError):
            runner.summarize_rates(plan, self.reference, self.rows, self.seconds)

    def test_timing_requires_finite_nonboolean_bounded_values_and_exact_list_count(self):
        variants = [(), None, [], self.seconds[:-1], self.seconds + [0.2]]
        variants.extend(
            [invalid, *self.seconds[1:]]
            for invalid in (True, None, "0.2", float("nan"), float("inf"), -0.1, 120.1, 10**1000)
        )
        for seconds in variants:
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                runner.summarize_rates(self.plan, self.reference, self.rows, seconds)


class SamplingRateTimedModelTests(unittest.TestCase):
    def test_synchronization_surrounds_prediction_and_delegates_attributes(self):
        events = []
        prediction = object()
        model = SimpleNamespace(
            task="detect",
            predict=Mock(side_effect=lambda **kwargs: events.append("predict") or prediction),
        )
        ticks = iter((10.0, 11.5, 20.0, 20.25))

        def clock():
            events.append("clock")
            return next(ticks)

        timed = runner.TimedModel(model, lambda: events.append("sync"), clock=clock)
        self.assertEqual("detect", timed.task)
        self.assertIs(model, timed.wrapped)
        for _ in range(2):
            self.assertIs(prediction, timed.predict(source="synthetic", end2end=True))
        self.assertEqual(["sync", "clock", "predict", "sync", "clock"] * 2, events)
        self.assertEqual([1.5, 0.25], timed.seconds)
        self.assertEqual({"source": "synthetic", "end2end": True}, model.predict.call_args.kwargs)

    def test_failed_prediction_never_fabricates_a_successful_timing_sample(self):
        failure = RuntimeError("synthetic inference failure")
        model, synchronize, clock = Mock(), Mock(), Mock(return_value=10)
        model.predict.side_effect = failure
        timed = runner.TimedModel(model, synchronize, clock=clock)
        with self.assertRaises(RuntimeError) as raised:
            timed.predict(source="synthetic")
        self.assertIs(failure, raised.exception)
        self.assertEqual([], timed.seconds)
        synchronize.assert_called_once()
        clock.assert_called_once()


class SamplingRateInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root.chmod(0o700)
        self.source = self.root / "synthetic.mp4"
        self.source.write_bytes(b"private household fixture clip sampling_rate")
        self.source.chmod(0o600)
        self.admission = _media_admission("sampling_rate")
        self.admission_path = self.root / "admission.json"
        _write(self.admission_path, self.admission.to_dict())
        self.reference = (0.0, 0.5, 1.0, 2.2, 2.4, 3.0, 4.1, 6.8)
        self.frozen_path = self.root / "prior-plan.json"
        self.frozen = {
            "reference_pts_seconds": self.reference,
            "source_admission_sha256": self.admission.fingerprint,
            "debug_code_sha256s": runner._diagnostic_code_sha256s(),
            "plan": plan_pts_sampling(self.reference, native_fps=2, sample_fps=0.5),
        }
        _write(self.frozen_path, self.frozen)
        self.request = {
            "version": 1,
            "camera_id": "camera_03",
            "source": str(self.source),
            "admission": str(self.admission_path),
            "admission_sha256": self.admission.fingerprint,
            "reference_artifact": str(self.frozen_path),
            "reference_artifact_sha256": runner._sha(self.frozen_path),
            "source_duration_seconds": 8.0,
            "weight": str(self.root / "fake.pt"),
            "weight_sha256": "a" * 64,
            "parent_net_namespace": "net:[parent-fixture]",
            "output_root": str(self.root / "output"),
            "code_sha256s": runner.code_hashes(),
        }
        self.manifest = Mock(sha256="a" * 64)

    def test_load_inputs_verifies_real_synthetic_media_and_plan_before_weight_binding(self):
        with patch.object(runner.YoloModelManifest, "read", return_value=self.manifest):
            source, admission, manifest, reference, plan = runner.load_inputs(self.request)
        self.assertEqual(self.source, source)
        self.assertEqual(self.admission, admission)
        self.assertIs(self.manifest, manifest)
        self.assertEqual(self.reference, reference)
        self.assertEqual([0.5, 1.0, 2.0, 4.0], [b["sample_fps"] for b in plan["branches"].values()])
        self.manifest.verify_weight.assert_called_once_with(Path(self.request["weight"]))

    def test_request_shape_version_camera_and_code_fail_before_admission(self):
        for request in (
            None,
            [],
            {},
            {**self.request, "version": True},
            {**self.request, "version": 2},
            {**self.request, "camera_id": "camera_01"},
            {**self.request, "extra": 1},
            {**self.request, "code_sha256s": {}},
        ):
            with (
                self.subTest(request=request),
                patch.object(runner.ExternalSourceAdmission, "read") as read,
                self.assertRaises(ValueError),
            ):
                runner.load_inputs(request)
            read.assert_not_called()

    def test_source_admission_reference_hash_and_weight_changes_are_rejected(self):
        for change in (
            {"admission_sha256": "f" * 64},
            {"reference_artifact_sha256": "f" * 64},
            {"weight_sha256": "f" * 64},
            {"source_duration_seconds": True},
        ):
            with (
                self.subTest(change=change),
                patch.object(runner.YoloModelManifest, "read", return_value=self.manifest),
                self.assertRaises(ValueError),
            ):
                runner.load_inputs({**self.request, **change})
        self.source.write_bytes(b"changed source")
        with self.assertRaises(ValueError):
            runner.load_inputs(self.request)

    def test_rehashed_reference_still_requires_source_admission_and_reviewed_code(self):
        for change in (
            {"source_admission_sha256": "f" * 64},
            {"debug_code_sha256s": {}},
        ):
            _write(self.frozen_path, {**self.frozen, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.load_inputs(
                    {**self.request, "reference_artifact_sha256": runner._sha(self.frozen_path)}
                )

    def test_worker_namespace_and_runtime_guards_fail_before_loading_inputs(self):
        request_path = self.root / "request.json"
        _write(request_path, self.request)
        for namespace, interfaces, version in (
            ("net:[parent-fixture]", [(1, "lo")], runner.RUNTIME_VERSION),
            ("net:[child-fixture]", [(1, "lo"), (2, "eth0")], runner.RUNTIME_VERSION),
            ("net:[child-fixture]", [(1, "lo")], "unreviewed"),
        ):
            with (
                self.subTest(namespace=namespace, interfaces=interfaces, version=version),
                patch.object(runner.os, "readlink", return_value=namespace),
                patch.object(runner.socket, "if_nameindex", return_value=interfaces),
                patch.object(runner.importlib.metadata, "version", return_value=version),
                patch.object(runner, "load_inputs") as load,
                self.assertRaises(ValueError),
            ):
                runner.worker(request_path)
            load.assert_not_called()
        self.assertFalse(Path(self.request["output_root"]).exists())


class SamplingRateParentTests(unittest.TestCase):
    def _compare(self, root, *, tamper=None):
        root.chmod(0o700)
        source, experiment = root / "source", root / "experiment"
        source.mkdir(mode=0o700)
        experiment.mkdir(mode=0o700)
        audit_rows = []
        for camera in runner.CAMERAS:
            directory = experiment / camera
            directory.mkdir(mode=0o700)
            _write(
                directory / "request.json",
                {
                    "source": str(source / f"{camera}.mp4"),
                    "admission": str(source / "admission.json"),
                },
            )
            audit_rows.append(
                {
                    "camera_id": camera,
                    "source_admission_sha256": "a" * 64,
                    "frozen_plan_sha256": "b" * 64,
                    "opportunity": {"source_duration_seconds": 8.0},
                }
            )
        audit = {"report_sha256": "c" * 64, "rows": audit_rows}
        reference = (0.0, 0.5, 1.0, 2.2, 2.4, 3.0, 4.1, 6.8)
        plan = runner.plan_sampling_rates(reference, native_fps=2, source_duration_seconds=8)
        rows = _rows(plan, reference)
        seconds = [4.0, *([0.2] * (len(rows) - 1))]
        manifest = Mock(sha256="d" * 64)
        completed = []

        def child(arguments, *, environment):
            self.assertEqual(runner.ISOLATION, arguments[: len(runner.ISOLATION)])
            self.assertEqual(str(root / "fake-python"), arguments[len(runner.ISOLATION)])
            self.assertEqual(str(runner.REPO / "src"), environment["PYTHONPATH"])
            self.assertEqual("false", environment["YOLO_AUTOINSTALL"])
            self.assertEqual("false", environment["YOLO_VERBOSE"])
            request_path = Path(arguments[-1])
            request, request_sha = runner._object_json(request_path)
            self.assertEqual(runner.CAMERAS[len(completed)], request["camera_id"])
            frozen = {
                "request_sha256": request_sha,
                "plan": plan,
                "plan_frozen_before_inference": True,
                "code_sha256s": runner.code_hashes(),
            }
            result = {
                "request_sha256": request_sha,
                "plan_sha256": plan["report_sha256"],
                "rows": rows,
                "prediction_wall_seconds": seconds,
                "decode_and_union_replay_wall_seconds": 6.0,
                "summary": runner.summarize_rates(plan, reference, rows, seconds),
                "network_namespace_isolated": True,
                "media_bytes_verified_before_and_after": True,
                "full_reference_decode_count_and_pts_matched": True,
                "source_exhaustion_verified": False,
                "new_pixels_uploaded": False,
            }
            if tamper == "request_hash":
                result["request_sha256"] = "f" * 64
            elif tamper == "frozen_plan":
                frozen["plan_frozen_before_inference"] = False
            elif tamper == "code_hash":
                frozen["code_sha256s"] = {}
            elif tamper == "summary":
                result["summary"]["first_prediction_wall_seconds"] = 0.0
            elif tamper == "timing":
                result["prediction_wall_seconds"] = [True, *seconds[1:]]
            elif tamper == "replay_timing":
                result["decode_and_union_replay_wall_seconds"] = True
            elif tamper == "replay_none":
                result["decode_and_union_replay_wall_seconds"] = None
            elif tamper == "replay_huge":
                result["decode_and_union_replay_wall_seconds"] = 10**1000
            elif tamper == "replay_negative":
                result["decode_and_union_replay_wall_seconds"] = -0.1
            elif tamper == "prediction_sum_exceeds_replay":
                result["decode_and_union_replay_wall_seconds"] = 1.0
            elif tamper == "source_exhaustion":
                result["source_exhaustion_verified"] = True
            elif tamper == "rows":
                result["rows"] = rows[:-1]
            _write(request_path.parent / "sampling-plan.json", frozen)
            _write(request_path.parent / "result.json", result)
            completed.append(request["camera_id"])
            return 1.0

        with (
            patch.object(runner, "analyze", return_value=audit) as admission_audit,
            patch.object(
                runner, "runtime_versions", return_value={"ultralytics": runner.RUNTIME_VERSION}
            ),
            patch.object(runner.YoloModelManifest, "read", return_value=manifest),
            patch.object(runner.os, "readlink", return_value="net:[parent-fixture]"),
            patch.object(
                runner, "load_inputs", return_value=(None, None, None, reference, plan)
            ) as load,
            patch.object(runner, "run_child", side_effect=child),
        ):
            report = runner.compare(
                source_root=source,
                experiment_root=experiment,
                output_root=root / "output",
                vision_python=root / "fake-python",
                weights=root / "fake-weights",
            )
        admission_audit.assert_called_once()
        self.assertEqual(4, load.call_count)
        self.assertEqual(list(runner.CAMERAS), completed)
        return report

    def test_parent_revalidates_before_and_after_sequential_children_and_binds_final_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._compare(root)
            self.assertEqual(list(runner.CAMERAS), [row["camera_id"] for row in report["rows"]])
            self.assertEqual("c" * 64, report["prior_execution_audit_sha256"])
            self.assertEqual(
                runner._hash(
                    {key: value for key, value in report.items() if key != "report_sha256"}
                ),
                report["report_sha256"],
            )
            self.assertFalse(report["event_recall_evaluated"])
            self.assertFalse(report["production_configuration_changed"])
            self.assertEqual(0o600, (root / "output/comparison.json").stat().st_mode & 0o777)

    def test_parent_rejects_child_binding_row_and_timing_changes_without_final_report(self):
        for tamper in (
            "request_hash",
            "frozen_plan",
            "code_hash",
            "summary",
            "timing",
            "replay_timing",
            "replay_none",
            "replay_huge",
            "replay_negative",
            "prediction_sum_exceeds_replay",
            "source_exhaustion",
            "rows",
        ):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    self._compare(root, tamper=tamper)
                self.assertFalse((root / "output/comparison.json").exists())


if __name__ == "__main__":
    unittest.main()
