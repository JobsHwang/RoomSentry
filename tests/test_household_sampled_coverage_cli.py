from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from tests.test_household_ground_truth import _annotation, _media_admission, _scan


class HouseholdSampledCoverageCLITests(unittest.TestCase):
    REFERENCE = b"opaque alignment evidence must not appear in output\xff"

    @classmethod
    def _inputs(cls, root: Path, *, pending: bool = False) -> list[str]:
        admission = _media_admission("sampled_cli")
        annotation = _annotation(admission, review_state="pending" if pending else "approved")
        base = _scan(admission)
        ledger = DetectorSamplingLedger(
            native_fps=25.0,
            native_fps_source="reported",
            requested_sample_fps=5.0,
            frame_step=5,
            decoded_frame_count=15,
            samples=(
                DetectorSampleFrame(0, 0.1, "container_pts", 0, 0),
                DetectorSampleFrame(5, 2.1, "container_pts", 0, 0),
                DetectorSampleFrame(10, 3.8, "container_pts", 0, 0),
            ),
            timestamp_fallback_frame_count=0,
            max_adjacent_container_pts_gap_seconds=2.0,
            container_pts_regression_count=0,
        )
        scan = replace(
            base, version=2, result=replace(base.result, version=2, detections=(), sampling=ledger)
        )
        annotation.write_new(root / "annotation.json")
        (root / "admission.json").write_text(admission.to_json(), encoding="utf-8")
        (root / "media.bin").write_bytes(b"private household fixture clip sampled_cli")
        scan.write_new(root / "scan.json")
        (root / "reference.bin").write_bytes(cls.REFERENCE)
        binding = {
            "version": 1,
            "annotation_sha256": annotation.fingerprint,
            "media_admission_sha256": admission.fingerprint,
            "reference_sha256": hashlib.sha256(cls.REFERENCE).hexdigest(),
            "source_pts_at_annotation_zero_seconds": 0.0,
            "alignment_confirmed": True,
        }
        (root / "binding.json").write_text(json.dumps(binding), encoding="utf-8")
        return [
            "evaluate-household-sampled-coverage",
            "--annotation",
            str(root / "annotation.json"),
            "--detector-scan",
            str(root / "scan.json"),
            "--media-admission",
            str(root / "admission.json"),
            "--media-asset",
            str(root / "media.bin"),
            "--timeline-binding",
            str(root / "binding.json"),
            "--timeline-reference",
            str(root / "reference.bin"),
            "--confidence-threshold",
            "0.35",
            "--output",
            str(root / "report.json"),
        ]

    @staticmethod
    def _run(argv: list[str]) -> tuple[int, str, str]:
        output, error = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main(argv)
        return code, output.getvalue(), error.getvalue()

    def _assert_rejected(self, root: Path, argv: list[str]) -> None:
        code, output, error = self._run(argv)
        self.assertEqual(2, code)
        self.assertEqual("", output)
        self.assertFalse((root / "report.json").exists())
        self.assertNotIn(str(root), error)
        self.assertNotIn("opaque alignment evidence", error)

    def test_actual_pts_cli_writes_hashed_path_free_report_without_parsing_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self._inputs(root)
            code, output, error = self._run(argv)
            self.assertEqual(0, code, error)
            report = json.loads(output)
            self.assertEqual(report, json.loads((root / "report.json").read_text(encoding="utf-8")))
            self.assertEqual(64, len(report["report_sha256"]))
            unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
            self.assertEqual(
                hashlib.sha256(
                    json.dumps(
                        unsigned,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
                report["report_sha256"],
            )
            self.assertEqual(hashlib.sha256(self.REFERENCE).hexdigest(), report["reference_sha256"])
            self.assertEqual(3, report["actual_sample_count"])
            self.assertEqual(3, report["zero_box_sample_count"])
            self.assertEqual(2, report["positive_sample_count"])
            self.assertEqual(1, report["negative_sample_count"])
            self.assertEqual(2, report["positive_interval_count"])
            self.assertEqual(2, report["observed_positive_without_detection_count"])
            self.assertEqual(0, report["unobserved_positive_interval_count"])
            self.assertFalse(report["reference_semantics_verified"])
            self.assertTrue(report["alignment_asserted_by_input"])
            for field in (
                "candidate_grouping_evaluated",
                "baby_identity_evaluated",
                "box_localization_evaluated",
                "independent_person_coverage_evaluated",
                "danger_recognition_evaluated",
                "deployment_approved",
            ):
                self.assertFalse(report[field])
            self.assertFalse(report["paths_disclosed"])
            self.assertFalse(report["source_exhaustion_verified"])
            self.assertNotIn(str(root), output + error)
            self.assertNotIn("opaque alignment evidence", output + error)

    def test_cli_rejects_changed_reference_bytes_and_each_wrong_binding_hash(self) -> None:
        for field in (
            "reference_bytes",
            "reference_sha256",
            "annotation_sha256",
            "media_admission_sha256",
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = self._inputs(root)
                if field == "reference_bytes":
                    (root / "reference.bin").write_bytes(b"changed reference")
                else:
                    path = root / "binding.json"
                    binding = json.loads(path.read_text(encoding="utf-8"))
                    binding[field] = "f" * 64
                    path.write_text(json.dumps(binding), encoding="utf-8")
                self._assert_rejected(root, argv)

    def test_cli_rejects_unconfirmed_incomplete_or_extra_binding_fields(self) -> None:
        for change in ("false", "numeric_true", "missing_offset", "extra"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = self._inputs(root)
                path = root / "binding.json"
                binding = json.loads(path.read_text(encoding="utf-8"))
                if change == "false":
                    binding["alignment_confirmed"] = False
                elif change == "numeric_true":
                    binding["alignment_confirmed"] = 1
                elif change == "missing_offset":
                    del binding["source_pts_at_annotation_zero_seconds"]
                else:
                    binding["private_note"] = "do not disclose"
                path.write_text(json.dumps(binding), encoding="utf-8")
                self._assert_rejected(root, argv)

    def test_cli_rejects_duplicate_nonfinite_non_json_and_non_utf8_binding(self) -> None:
        invalid = (
            b'{"version":1,"version":1}',
            b'{"source_pts_at_annotation_zero_seconds":NaN}',
            b'{"source_pts_at_annotation_zero_seconds":Infinity}',
            b"[]",
            b"not json",
            b"\xff",
        )
        for raw in invalid:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = self._inputs(root)
                (root / "binding.json").write_bytes(raw)
                self._assert_rejected(root, argv)

    def test_cli_rejects_pending_annotation_and_modified_media(self) -> None:
        for pending in (False, True):
            with self.subTest(pending=pending), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                argv = self._inputs(root, pending=pending)
                if not pending:
                    (root / "media.bin").write_bytes(b"modified media")
                self._assert_rejected(root, argv)

    def test_cli_input_size_limits_and_nonregular_inputs_fail_without_paths(self) -> None:
        cases = (
            ("binding.json", 64 * 1024 + 1),
            ("reference.bin", 8 * 1024 * 1024 + 1),
            ("binding.json", "directory"),
            ("reference.bin", "directory"),
            ("binding.json", "symlink"),
            ("reference.bin", "symlink"),
            ("binding.json", "missing"),
            ("reference.bin", "missing"),
        )
        for filename, change in cases:
            with (
                self.subTest(filename=filename, change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                argv = self._inputs(root)
                path = root / filename
                if isinstance(change, int):
                    path.write_bytes(b"x" * change)
                else:
                    path.unlink()
                    if change == "directory":
                        path.mkdir()
                    elif change == "symlink":
                        path.symlink_to(root / "media.bin")
                self._assert_rejected(root, argv)

    def test_cli_output_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self._inputs(root)
            path = root / "report.json"
            path.write_bytes(b"existing report fixture")
            code, output, error = self._run(argv)
            self.assertEqual(2, code)
            self.assertEqual("", output)
            self.assertEqual(b"existing report fixture", path.read_bytes())
            self.assertNotIn(str(root), error)

    def test_cli_requires_all_explicit_binding_and_source_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = self._inputs(root)
            for option in (
                "--timeline-binding",
                "--timeline-reference",
                "--media-asset",
                "--confidence-threshold",
            ):
                missing = list(argv)
                index = missing.index(option)
                del missing[index : index + 2]
                with (
                    self.subTest(option=option),
                    redirect_stderr(StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    main(missing)
                self.assertEqual(2, error.exception.code)
                self.assertFalse((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
