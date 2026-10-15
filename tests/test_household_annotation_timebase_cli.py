from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.detector_protocol import AdmittedDetectorScan
from baby_monitor.domain import Detection
from baby_monitor.household_ground_truth import HouseholdEpisodeAnnotation, HouseholdEpisodeRange
from tests import test_household_sampled_coverage_cli as sampled_fixtures


class HouseholdAnnotationTimebaseCLITests(unittest.TestCase):
    @staticmethod
    def _run(argv: list[str]) -> tuple[int, str, str]:
        output, error = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            code = main(argv)
        return code, output.getvalue(), error.getvalue()

    @staticmethod
    def _v2_inputs(
        root: Path, *, pending: bool = False
    ) -> tuple[list[str], HouseholdEpisodeAnnotation]:
        argv = sampled_fixtures.HouseholdSampledCoverageCLITests._inputs(root, pending=pending)
        path = root / "annotation.json"
        annotation = replace(
            HouseholdEpisodeAnnotation.read(path),
            version=2,
            timebase="elapsed_source_pts",
            source_fps=None,
            source_frame_count=None,
            source_duration_seconds=7.0,
            ranges=(
                HouseholdEpisodeRange(0.0, 2.0, "empty_room"),
                HouseholdEpisodeRange(2.0, 4.0, "person_present"),
                HouseholdEpisodeRange(5.0, 7.0, "danger_event"),
            ),
        )
        path.write_text(annotation.to_json(), encoding="utf-8")
        scan_path = root / "scan.json"
        scan = AdmittedDetectorScan.read(scan_path)
        ledger = scan.result.sampling
        assert ledger is not None
        # The zero-point sidecar is explicitly 1 s, so these correspond to
        # elapsed times 0.1, 2.1, 6.2 s, including beyond the old 4 s grid.
        samples = tuple(
            replace(
                sample, timestamp_seconds=pts, detection_start=0, detection_count=int(index == 2)
            )
            for index, (sample, pts) in enumerate(zip(ledger.samples, (1.1, 3.1, 7.2), strict=True))
        )
        scan = replace(
            scan,
            result=replace(
                scan.result,
                detections=(Detection(7.2, "person", 0.9, (0.1, 0.1, 0.3, 0.8)),),
                sampling=replace(ledger, samples=samples),
            ),
        )
        scan_path.write_text(scan.to_json(), encoding="utf-8")
        binding_path = root / "binding.json"
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        binding["annotation_sha256"] = annotation.fingerprint
        binding["source_pts_at_annotation_zero_seconds"] = 1.0
        binding_path.write_text(json.dumps(binding), encoding="utf-8")
        return argv, annotation

    def test_validate_elapsed_pts_v2_preserves_independent_duration_and_exact_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, annotation = self._v2_inputs(root)
            code, output, error = self._run(
                ["validate-household-episode-annotation", str(root / "annotation.json")]
            )
            self.assertEqual(0, code, error)
            payload = json.loads(output)
            self.assertEqual(annotation.to_dict(), payload)
            self.assertEqual(2, payload["version"])
            self.assertEqual("elapsed_source_pts", payload["timebase"])
            self.assertEqual(7.0, payload["source_duration_seconds"])
            self.assertNotIn("source_fps", payload)
            self.assertNotIn("source_frame_count", payload)
            self.assertEqual(annotation.fingerprint, payload["annotation_sha256"])
            self.assertNotIn(str(root), output + error)

    def test_actual_pts_cli_evaluates_late_v2_interval_using_explicit_nonzero_origin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, annotation = self._v2_inputs(root)
            code, output, error = self._run(argv)
            self.assertEqual(0, code, error)
            report = json.loads(output)
            self.assertEqual(annotation.fingerprint, report["annotation_sha256"])
            self.assertEqual(1.0, report["source_pts_at_annotation_zero_seconds"])
            self.assertEqual(3, report["actual_sample_count"])
            self.assertEqual(0, report["outside_source_sample_count"])
            self.assertEqual(2, report["positive_interval_count"])
            self.assertEqual(1, report["detected_positive_interval_count"])
            self.assertEqual(1, report["observed_positive_without_detection_count"])
            self.assertEqual(0, report["unobserved_positive_interval_count"])
            late = report["intervals"][2]
            self.assertEqual("person_detected", late["state"])
            self.assertEqual(1, late["actual_sample_count"])
            self.assertAlmostEqual(1.2, late["first_person_detection_delay_seconds"])
            self.assertFalse(report["danger_recognition_evaluated"])
            self.assertFalse(report["source_exhaustion_verified"])
            self.assertEqual(report, json.loads((root / "report.json").read_text(encoding="utf-8")))
            self.assertNotIn(str(root), output + error)

    def test_nominal_cli_rejects_v2_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._v2_inputs(root)
            code, output, error = self._run(
                [
                    "evaluate-household-trigger-coverage",
                    "--annotation",
                    str(root / "annotation.json"),
                    "--media-admission",
                    str(root / "admission.json"),
                    "--media-asset",
                    str(root / "media.bin"),
                    "--detector-scan",
                    str(root / "scan.json"),
                    "--start-threshold",
                    "0.35",
                    "--start-threshold",
                    "0.55",
                    "--candidate-gap-seconds",
                    "0.5",
                    "--max-event-span-seconds",
                    "30",
                    "--output",
                    str(root / "report.json"),
                ]
            )
            self.assertEqual(2, code)
            self.assertEqual("", output)
            self.assertFalse((root / "report.json").exists())
            self.assertNotIn(str(root), error)

    def test_pending_v2_validation_does_not_approve_or_enable_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, annotation = self._v2_inputs(root, pending=True)
            path = root / "annotation.json"
            original = path.read_bytes()
            code, output, error = self._run(["validate-household-episode-annotation", str(path)])
            self.assertEqual(0, code, error)
            payload = json.loads(output)
            self.assertEqual("pending", payload["review_state"])
            self.assertFalse(payload["human_verified"])
            self.assertEqual([], payload["reviewers"])
            self.assertEqual(annotation.fingerprint, payload["annotation_sha256"])
            code, output, error = self._run(argv)
            self.assertEqual(2, code)
            self.assertEqual("", output)
            self.assertFalse((root / "report.json").exists())
            self.assertEqual(original, path.read_bytes())
            self.assertNotIn(str(root), error)

    def test_v1_validation_keeps_original_fields_and_hash_and_actual_cli_compatibility(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = sampled_fixtures.HouseholdSampledCoverageCLITests._inputs(root)
            path = root / "annotation.json"
            before = json.loads(path.read_text(encoding="utf-8"))
            code, output, error = self._run(["validate-household-episode-annotation", str(path)])
            self.assertEqual(0, code, error)
            self.assertEqual(before, json.loads(output))
            self.assertNotIn("timebase", before)
            self.assertEqual(25.0, before["source_fps"])
            self.assertEqual(100, before["source_frame_count"])
            code, output, error = self._run(argv)
            self.assertEqual(0, code, error)
            self.assertEqual(before["annotation_sha256"], json.loads(output)["annotation_sha256"])

    def test_v2_cli_validation_rejects_missing_timebase_and_nominal_fields(self) -> None:
        for change in ("missing_timebase", "source_fps", "source_frame_count"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._v2_inputs(root)
                path = root / "annotation.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                if change == "missing_timebase":
                    del payload["timebase"]
                else:
                    payload[change] = None
                path.write_text(json.dumps(payload), encoding="utf-8")
                code, output, error = self._run(
                    ["validate-household-episode-annotation", str(path)]
                )
                self.assertEqual(2, code)
                self.assertEqual("", output)
                self.assertNotIn(str(root), error)


if __name__ == "__main__":
    unittest.main()
