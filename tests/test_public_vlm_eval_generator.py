from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from baby_monitor.ab_eval import load_dataset


class PublicVLMEvalGeneratorTests(unittest.TestCase):
    def test_generates_fourteen_synthetic_cases_with_verified_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "evaluation"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_public_vlm_eval.py",
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual([2], summary["case_versions"])
            self.assertEqual(14, summary["cases"])
            self.assertEqual(30, summary["frames"])
            self.assertFalse(summary["private_media_used"])
            self.assertFalse(summary["paths_disclosed"])
            self.assertNotIn(str(root), completed.stdout)
            cases = load_dataset(output / "dataset.jsonl")
            self.assertEqual(14, len(cases))
            self.assertTrue(
                all(
                    json.loads(line)["version"] == 2
                    for line in (output / "dataset.jsonl").read_text().splitlines()
                )
            )
            self.assertEqual(30, sum(len(case.frames) for case in cases))
            self.assertEqual(
                "movement",
                next(
                    case.expected.event_type.value
                    for case in cases
                    if case.case_id == "position_change_sequence"
                ),
            )
            movement = next(case for case in cases if case.case_id == "position_change_sequence")
            self.assertEqual(1, len(movement.expected.required_observation_groups))
            self.assertEqual(1, len(movement.expected.required_inference_groups))
            added = {
                case.case_id: case
                for case in cases
                if case.case_id
                in {
                    "appearance_sequence",
                    "count_change_sequence",
                    "size_change_sequence",
                    "occlusion_progression_sequence",
                }
            }
            self.assertEqual(4, len(added))
            self.assertTrue(all(len(case.frames) == 3 for case in added.values()))
            self.assertTrue(all(not case.expected.allowed_subjects for case in added.values()))
            self.assertTrue(
                all(len(case.expected.required_observation_groups) == 1 for case in added.values())
            )

    def test_optional_public_image_adds_identity_and_static_sequence_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_image = root / "public.ppm"
            public_image.write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")
            output = root / "evaluation"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_public_vlm_eval.py",
                    "--output-dir",
                    str(output),
                    "--public-image",
                    str(public_image),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(16, summary["cases"])
            self.assertEqual(34, summary["frames"])
            cases = load_dataset(output / "dataset.jsonl")
            bus = next(case for case in cases if case.case_id == "public_bus_single")
            self.assertEqual(
                ("person",),
                tuple(item.value for item in bus.expected.allowed_subjects),
            )
            static = next(case for case in cases if case.case_id == "public_bus_static_sequence")
            self.assertEqual(3, len(static.frames))

    def test_refuses_to_overwrite_an_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/generate_public_vlm_eval.py",
                    "--output-dir",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, completed.returncode)
            self.assertEqual("preserve", marker.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
