from __future__ import annotations

import builtins
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import analyze_pts_sampling_opportunity as reader
from scripts.compare_yolo_pts_sampling import summarize_schedule
from scripts.pts_sampling_diagnostic import _hash, plan_pts_sampling
from tests.test_household_ground_truth import _media_admission


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    path.chmod(0o600)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SamplingOpportunityReaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root.chmod(0o700)
        self.source = self.root / "source"
        self.experiment = self.root / "experiment"
        self.output = self.root / "output"
        self.source.mkdir(mode=0o700)
        self.experiment.mkdir(mode=0o700)
        (self.source / "admissions").mkdir(mode=0o700)
        self.metadata_path = self.source / "capture-verification.json"
        self.comparison_path = self.experiment / "comparison.json"
        self.experiment_plan_path = self.experiment / "plan.json"
        self.reference = (0.0, 0.5, 1.0, 2.2, 2.4, 3.0, 4.1, 6.8)
        metadata_rows, comparison_rows = [], []
        code_hashes = reader._diagnostic_code_sha256s()
        for camera in reader.CAMERAS:
            directory = self.experiment / camera
            directory.mkdir(mode=0o700)
            admission = _media_admission(camera)
            media = self.source / f"fixture_{camera}_opaque.mp4"
            media.write_bytes(f"private household fixture clip {camera}".encode())
            media.chmod(0o600)
            admission_path = self.source / "admissions" / f"{camera}.source-admission.json"
            _write(admission_path, admission.to_dict())
            request_path = directory / "request.json"
            _write(
                request_path,
                {
                    "version": 1,
                    "camera_id": camera,
                    "source": str(media),
                    "admission": str(admission_path),
                    "admission_sha256": admission.fingerprint,
                    "baseline_scan": str(self.source / f"{camera}.scan.json"),
                    "baseline_scan_sha256": "b" * 64,
                    "weight": str(self.root / "fake.pt"),
                    "weight_sha256": "c" * 64,
                    "parent_net_namespace": "net:[fixture]",
                    "output_root": str(directory),
                },
            )
            plan = plan_pts_sampling(self.reference, native_fps=2, sample_fps=0.5)
            indices = sorted(
                set(plan["decoded_stride"]["decoded_frame_indices"])
                | set(plan["source_pts_deadline"]["decoded_frame_indices"])
            )
            # A real row with no boxes remains execution evidence, unlike an absent row.
            rows = [
                {"decoded_frame_index": i, "timestamp_seconds": self.reference[i], "boxes": []}
                for i in indices
            ]
            summary = summarize_schedule(plan, rows, threshold=reader.REPORT_THRESHOLD)
            _write(
                directory / "sampling-plan.json",
                {
                    "request_sha256": _sha(request_path),
                    "source_admission_sha256": admission.fingerprint,
                    "reference_pts_seconds": self.reference,
                    "plan": plan,
                    "debug_code_sha256s": code_hashes,
                    "plan_frozen_before_inference": True,
                    "union_prediction_count": len(rows),
                },
            )
            result_path = directory / "result.json"
            _write(
                result_path,
                {
                    "request_sha256": _sha(request_path),
                    "plan_sha256": plan["report_sha256"],
                    "rows": rows,
                    "summary": summary,
                    "source_exhaustion_verified": False,
                    "network_namespace_isolated": True,
                    "media_bytes_verified_before_and_after": True,
                },
            )
            metadata_rows.append(
                {
                    "camera_id": camera,
                    "source_admission_sha256": admission.fingerprint,
                    "sha256": _sha(media),
                    "bytes": media.stat().st_size,
                    "duration_seconds": 8.0,
                }
            )
            comparison_rows.append(
                {"camera_id": camera, "result_sha256": _sha(result_path), **summary}
            )
        _write(self.metadata_path, {"rows": metadata_rows})
        _write(
            self.comparison_path,
            {"rows": comparison_rows, "accuracy_evaluated": False, "deployment_approved": False},
        )
        _write(
            self.experiment_plan_path,
            {"cameras": list(reader.CAMERAS), "debug_code_sha256s": code_hashes},
        )

    def _analyze(self):
        return reader.analyze(
            source_root=self.source, experiment_root=self.experiment, output_root=self.output
        )

    def _assert_rejected(self):
        with self.assertRaises(ValueError):
            self._analyze()
        self.assertFalse(self.output.exists())

    def _result_path(self):
        return self.experiment / reader.CAMERAS[0] / "result.json"

    def _frozen_path(self):
        return self.experiment / reader.CAMERAS[0] / "sampling-plan.json"

    def _rebind_result_hash(self):
        comparison = _read(self.comparison_path)
        comparison["rows"][0]["result_sha256"] = _sha(self._result_path())
        _write(self.comparison_path, comparison)

    def test_reads_two_executed_sources_without_decode_model_or_network_and_writes_private_report(
        self,
    ):
        original_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"cv2", "torch", "torchvision", "ultralytics"}:
                raise AssertionError("reader must not load optional inference runtimes")
            return original_import(name, *args, **kwargs)

        inputs_before = {
            path: path.read_bytes()
            for directory in (self.source, self.experiment)
            for path in directory.rglob("*")
            if path.is_file()
        }
        with (
            patch.object(builtins, "__import__", side_effect=guard),
            patch("subprocess.run", side_effect=AssertionError("no subprocess")),
            patch("subprocess.Popen", side_effect=AssertionError("no subprocess")),
            patch("socket.socket", side_effect=AssertionError("no network")),
        ):
            report = self._analyze()
        self.assertEqual(list(reader.CAMERAS), [row["camera_id"] for row in report["rows"]])
        self.assertEqual(0o700, self.output.stat().st_mode & 0o777)
        self.assertEqual(0o600, (self.output / "report.json").stat().st_mode & 0o777)
        self.assertEqual(json.loads(json.dumps(report)), _read(self.output / "report.json"))
        self.assertEqual(
            _hash({key: value for key, value in report.items() if key != "report_sha256"}),
            report["report_sha256"],
        )
        self.assertEqual(_sha(self.metadata_path), report["capture_verification_sha256"])
        self.assertEqual(_sha(self.comparison_path), report["comparison_sha256"])
        self.assertEqual(_sha(self.experiment_plan_path), report["experiment_plan_sha256"])
        self.assertTrue(report["recorded_execution_union_verified"])
        self.assertTrue(report["media_bytes_verified"])
        self.assertTrue(report["duration_from_prior_probe_not_reprobed"])
        for flag in (
            "source_exhaustion_verified",
            "actual_event_labels_used",
            "accuracy_evaluated",
            "event_recall_evaluated",
            "model_loaded",
            "video_decoded",
            "new_pixels_uploaded",
            "deployment_approved",
            "production_configuration_changed",
        ):
            self.assertIs(False, report[flag])
        for row in report["rows"]:
            opportunity = row["opportunity"]
            self.assertEqual(8, opportunity["reference_frame_count"])
            self.assertFalse(opportunity["equal_selected_sample_counts"])
            self.assertEqual(5, len(opportunity["rows"]))
            self.assertFalse(opportunity["event_recall_evaluated"])
            self.assertFalse(opportunity["ground_truth"])
            for window in opportunity["rows"]:
                self.assertEqual(
                    8.0 - window["hypothetical_event_duration_seconds"],
                    window["possible_start_position_measure_seconds"],
                )
                self.assertEqual(2, window["branches"]["decoded_stride"]["sample_count"])
                self.assertEqual(4, window["branches"]["source_pts_deadline"]["sample_count"])
        self.assertNotIn(str(self.root), json.dumps(report))
        self.assertEqual(inputs_before, {path: path.read_bytes() for path in inputs_before})

    def test_existing_output_is_not_overwritten_or_reverified(self):
        self._analyze()
        original = (self.output / "report.json").read_bytes()
        with (
            patch.object(
                reader.ExternalSourceAdmission, "read", side_effect=AssertionError("no read")
            ),
            self.assertRaisesRegex(ValueError, "already exist"),
        ):
            self._analyze()
        self.assertEqual(original, (self.output / "report.json").read_bytes())

    def test_missing_zero_box_row_or_entire_execution_cannot_be_substituted_by_plan(self):
        original = _read(self._result_path())
        for rows in ([], original["rows"][:-1]):
            with self.subTest(rows=len(rows)):
                changed = {**original, "rows": rows}
                _write(self._result_path(), changed)
                self._rebind_result_hash()
                self._assert_rejected()

    def test_changed_source_bytes_are_rejected_against_real_admission(self):
        source = next(self.source.glob(f"*_{reader.CAMERAS[0]}_*.mp4"))
        original = source.read_bytes()
        source.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        self._assert_rejected()

    def test_result_byte_hash_change_is_rejected_even_when_prediction_semantics_unchanged(self):
        path = self._result_path()
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        self._assert_rejected()

    def test_request_change_breaks_both_recorded_request_hashes(self):
        path = self.experiment / reader.CAMERAS[0] / "request.json"
        request = _read(path)
        request["weight_sha256"] = "f" * 64
        _write(path, request)
        self._assert_rejected()

    def test_frozen_plan_reference_clock_code_and_admission_bindings_are_checked(self):
        original = _read(self._frozen_path())
        changes = (
            {"request_sha256": "f" * 64},
            {"source_admission_sha256": "f" * 64},
            {"debug_code_sha256s": {}},
            {"reference_pts_seconds": [0.0, 0.6, *self.reference[2:]]},
            {"plan": {**original["plan"], "native_fps": 3.0}},
        )
        for change in changes:
            with self.subTest(change=change):
                _write(self._frozen_path(), {**original, **change})
                self._assert_rejected()

    def test_result_summary_plan_and_execution_flags_are_recomputed_after_hash_rebinding(self):
        original = _read(self._result_path())
        summary = copy.deepcopy(original["summary"])
        summary["branches"]["decoded_stride"]["sample_count"] += 1
        for change in (
            {"summary": summary},
            {"plan_sha256": "f" * 64},
            {"source_exhaustion_verified": True},
            {"network_namespace_isolated": False},
            {"media_bytes_verified_before_and_after": False},
        ):
            with self.subTest(change=change):
                _write(self._result_path(), {**original, **change})
                self._rebind_result_hash()
                self._assert_rejected()

    def test_prior_duration_metadata_requires_matching_source_sha_size_and_admission(self):
        original = _read(self.metadata_path)
        for change in (
            {"sha256": "f" * 64},
            {"source_admission_sha256": "f" * 64},
            {"bytes": original["rows"][0]["bytes"] + 1},
            {"bytes": True},
        ):
            with self.subTest(change=change):
                altered = copy.deepcopy(original)
                altered["rows"][0].update(change)
                _write(self.metadata_path, altered)
                self._assert_rejected()

    def test_invalid_duration_or_duration_not_beyond_last_reference_frame_is_rejected(self):
        original = _read(self.metadata_path)
        for duration in (True, "8", 0, 5.0, 6.8):
            with self.subTest(duration=duration):
                altered = copy.deepcopy(original)
                altered["rows"][0]["duration_seconds"] = duration
                _write(self.metadata_path, altered)
                self._assert_rejected()

    def test_valid_prior_duration_change_is_a_new_hashed_input_not_historical_verification(self):
        first = self._analyze()
        metadata = _read(self.metadata_path)
        metadata["rows"][0]["duration_seconds"] = 9.0
        _write(self.metadata_path, metadata)
        self.output = self.root / "second-output"
        second = self._analyze()
        self.assertNotEqual(
            first["capture_verification_sha256"], second["capture_verification_sha256"]
        )
        self.assertNotEqual(first["report_sha256"], second["report_sha256"])
        self.assertNotEqual(
            first["rows"][0]["opportunity"]["input_sha256"],
            second["rows"][0]["opportunity"]["input_sha256"],
        )
        self.assertEqual(first["rows"][1], second["rows"][1])
        self.assertTrue(second["duration_from_prior_probe_not_reprobed"])
        self.assertFalse(second["source_exhaustion_verified"])
        self.assertFalse(second["event_recall_evaluated"])

    def test_duplicate_camera_rows_in_metadata_or_comparison_are_rejected(self):
        for path in (self.metadata_path, self.comparison_path):
            original = _read(path)
            altered = copy.deepcopy(original)
            altered["rows"][1] = copy.deepcopy(altered["rows"][0])
            with self.subTest(path=path.name):
                _write(path, altered)
                self._assert_rejected()
            _write(path, original)

    def test_changed_experiment_camera_or_code_identity_is_rejected(self):
        original = _read(self.experiment_plan_path)
        for change in ({"cameras": list(reversed(reader.CAMERAS))}, {"debug_code_sha256s": {}}):
            with self.subTest(change=change):
                _write(self.experiment_plan_path, {**original, **change})
                self._assert_rejected()


if __name__ == "__main__":
    unittest.main()
