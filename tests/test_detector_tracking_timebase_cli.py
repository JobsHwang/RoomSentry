from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from baby_monitor.cli import main
from baby_monitor.detector_protocol import AdmittedDetectorScan
from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from baby_monitor.object_tracker import DetectionFrameBatch
from tests import test_detector_evaluation as fixtures


class DetectorTrackingTimebaseCLITests(unittest.TestCase):
    ANCHOR = datetime(2026, 9, 12, tzinfo=UTC)

    @staticmethod
    def _actual_scan(*, timestamps=(0.125, 1.5, 4.57), fallback=False) -> AdmittedDetectorScan:
        *_, scan = fixtures.DetectorEvaluationTests._inputs()
        detections = (
            replace(scan.result.detections[0], timestamp_seconds=timestamps[0]),
            replace(scan.result.detections[-1], timestamp_seconds=timestamps[2]),
        )
        ledger = DetectorSamplingLedger(
            native_fps=2.0,
            native_fps_source="reported",
            requested_sample_fps=1.0,
            frame_step=2,
            decoded_frame_count=5,
            samples=(
                DetectorSampleFrame(0, timestamps[0], "container_pts", 0, 1),
                DetectorSampleFrame(
                    2, timestamps[1], "nominal_fps" if fallback else "container_pts", 1, 0
                ),
                DetectorSampleFrame(4, timestamps[2], "container_pts", 1, 1),
            ),
            timestamp_fallback_frame_count=int(fallback),
            max_adjacent_container_pts_gap_seconds=3.07,
            container_pts_regression_count=0,
        )
        return replace(
            scan,
            version=2,
            result=replace(scan.result, version=2, detections=detections, sampling=ledger),
        )

    @classmethod
    def _arguments(cls, root: Path) -> list[str]:
        return [
            "prepare-detector-tracking-batch",
            "--media-admission",
            str(root / "admission.json"),
            "--media-asset",
            str(root / "fixture.mpg"),
            "--detector-scan",
            str(root / "scan.json"),
            "--timeline-start",
            cls.ANCHOR.isoformat(),
            "--output",
            str(root / "frames.json"),
        ]

    @staticmethod
    def _write_inputs(root: Path, scan: AdmittedDetectorScan) -> None:
        *_, admission, _ = fixtures.DetectorEvaluationTests._inputs()
        (root / "fixture.mpg").write_bytes(b"fixture video")
        (root / "admission.json").write_text(admission.to_json(), encoding="utf-8")
        scan.write_new(root / "scan.json")

    @staticmethod
    def _run(argv: list[str]) -> tuple[int, str, str]:
        output, error = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main(argv)
        return code, output.getvalue(), error.getvalue()

    def test_actual_pts_cli_preserves_observed_times_and_zero_box_sample_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = self._actual_scan()
            self._write_inputs(root, scan)
            code, output, error = self._run(self._arguments(root) + ["--timebase", "actual_pts"])
            self.assertEqual(0, code, error)
            summary = json.loads(output)
            self.assertEqual("actual_pts", summary["timebase"])
            self.assertTrue(summary["actual_samples_only"])
            self.assertFalse(summary["source_exhaustion_verified"])
            self.assertFalse(summary["accuracy_evaluated"])
            self.assertTrue(summary["timeline_is_ordering_anchor"])
            self.assertEqual(
                (3, 1, 2),
                (summary["frame_count"], summary["empty_frame_count"], summary["detection_count"]),
            )
            batch = DetectionFrameBatch.read(root / "frames.json")
            self.assertEqual(
                (0.125, 1.5, 4.57),
                tuple((frame.occurred_at - self.ANCHOR).total_seconds() for frame in batch.frames),
            )
            self.assertEqual((1, 0, 1), tuple(len(frame.detections) for frame in batch.frames))
            self.assertEqual(scan.fingerprint, summary["detector_scan_sha256"])
            self.assertNotIn(str(root), output + error)
            self.assertFalse(summary["paths_disclosed"])
            self.assertFalse(summary["network_used"])

    def test_default_and_explicit_nominal_keep_v1_output_shape(self) -> None:
        *_, scan = fixtures.DetectorEvaluationTests._inputs()
        expected_keys = {
            "valid",
            "version",
            "detector_scan_sha256",
            "source_admission_sha256",
            "detection_batch_sha256",
            "camera_id",
            "detector_adapter",
            "frame_count",
            "frames_with_detections",
            "empty_frame_count",
            "detection_count",
            "timeline_is_ordering_anchor",
            "subject_identity_inference",
            "source_media_included",
            "paths_disclosed",
            "network_used",
        }
        results = []
        for mode in ([], ["--timebase", "nominal"]):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._write_inputs(root, scan)
                code, output, error = self._run(
                    self._arguments(root)
                    + mode
                    + ["--source-fps", "2", "--source-frame-count", "5"]
                )
                self.assertEqual(0, code, error)
                summary = json.loads(output)
                self.assertEqual(expected_keys, set(summary))
                results.append(summary)
        self.assertEqual(results[0], results[1])

    def test_invalid_timebase_parameter_combinations_fail_before_reading_assets(self) -> None:
        combinations = (
            [],
            ["--source-fps", "2"],
            ["--source-frame-count", "5"],
            ["--timebase", "nominal"],
            ["--timebase", "actual_pts", "--source-fps", "2"],
            ["--timebase", "actual_pts", "--source-frame-count", "5"],
            ["--timebase", "actual_pts", "--source-fps", "2", "--source-frame-count", "5"],
        )
        for extra in combinations:
            with (
                self.subTest(extra=extra),
                patch(
                    "baby_monitor.cli_commands.detection_eval.ExternalSourceAdmission.read",
                    side_effect=AssertionError("must not read admission"),
                ),
                patch(
                    "baby_monitor.cli_commands.detection_eval.ExternalSourceAdmission.verify_asset",
                    side_effect=AssertionError("must not read media"),
                ),
                patch(
                    "baby_monitor.cli_commands.detection_eval.AdmittedDetectorScan.read",
                    side_effect=AssertionError("must not read scan"),
                ),
            ):
                code, output, error = self._run(self._arguments(Path("/not-read")) + extra)
                self.assertEqual(2, code)
                self.assertEqual("", output)
                self.assertIn("timebase", error)
                self.assertNotIn("/not-read", error)

    def test_actual_pts_rejects_v1_fallback_duplicate_and_datetime_collapsed_pts(self) -> None:
        *_, legacy = fixtures.DetectorEvaluationTests._inputs()
        cases = (
            legacy,
            self._actual_scan(fallback=True),
            self._actual_scan(timestamps=(0.125, 0.125, 4.57)),
            self._actual_scan(timestamps=(0.125, 0.12500001, 4.57)),
        )
        for index, scan in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._write_inputs(root, scan)
                code, output, error = self._run(
                    self._arguments(root) + ["--timebase", "actual_pts"]
                )
                self.assertEqual(2, code)
                self.assertEqual("", output)
                self.assertFalse((root / "frames.json").exists())
                self.assertNotIn(str(root), error)

    def test_default_nominal_still_rejects_variable_pts_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_inputs(root, self._actual_scan())
            code, output, error = self._run(
                self._arguments(root) + ["--source-fps", "2", "--source-frame-count", "5"]
            )
            self.assertEqual(2, code)
            self.assertEqual("", output)
            self.assertFalse((root / "frames.json").exists())
            self.assertNotIn(str(root), error)

    def test_actual_pts_still_verifies_media_bytes_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_inputs(root, self._actual_scan())
            (root / "fixture.mpg").write_bytes(b"changed fixture bytes")
            code, output, error = self._run(self._arguments(root) + ["--timebase", "actual_pts"])
            self.assertEqual(2, code)
            self.assertEqual("", output)
            self.assertFalse((root / "frames.json").exists())
            self.assertNotIn(str(root), error)

    def test_actual_pts_all_zero_vfr_samples_remain_three_distinct_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scan = self._actual_scan()
            ledger = scan.result.sampling
            assert ledger is not None
            scan = replace(
                scan,
                result=replace(
                    scan.result,
                    detections=(),
                    sampling=replace(
                        ledger,
                        samples=tuple(
                            replace(sample, detection_start=0, detection_count=0)
                            for sample in ledger.samples
                        ),
                    ),
                ),
            )
            self._write_inputs(root, scan)
            code, output, error = self._run(self._arguments(root) + ["--timebase", "actual_pts"])
            self.assertEqual(0, code, error)
            summary = json.loads(output)
            self.assertEqual(3, summary["frame_count"])
            self.assertEqual(3, summary["empty_frame_count"])
            self.assertEqual(0, summary["detection_count"])
            batch = DetectionFrameBatch.read(root / "frames.json")
            self.assertEqual(
                (0.125, 1.5, 4.57),
                tuple((frame.occurred_at - self.ANCHOR).total_seconds() for frame in batch.frames),
            )

    def test_actual_pts_output_is_exclusive_and_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_inputs(root, self._actual_scan())
            path = root / "frames.json"
            original = b"existing user fixture"
            path.write_bytes(original)
            code, output, error = self._run(self._arguments(root) + ["--timebase", "actual_pts"])
            self.assertEqual(2, code)
            self.assertEqual("", output)
            self.assertEqual(original, path.read_bytes())
            self.assertIn("exists", error)


if __name__ == "__main__":
    unittest.main()
