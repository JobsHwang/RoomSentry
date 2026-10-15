from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from baby_monitor.ab_eval import load_dataset


def _fake_ffmpeg(root: Path) -> Path:
    executable = root / "fake_ffmpeg.py"
    executable.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

count = int(sys.argv[sys.argv.index('-frames:v') + 1])
start = int(sys.argv[sys.argv.index('-start_number') + 1])
pattern = sys.argv[-1]
for index in range(start, start + count):
    pathlib.Path(pattern % index).write_bytes(b'bounded synthetic jpeg fixture')
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


class HouseholdVLMEvalPreparationTests(unittest.TestCase):
    def test_prepares_ten_pending_cases_without_persisting_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "household-review"
            executable = _fake_ffmpeg(root)
            sources = (
                "rtsp://127.0.0.1:8554/cameras/opaque-a",
                "rtsp://127.0.0.1:8554/cameras/opaque-b",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_household_vlm_eval.py",
                    "--output-dir",
                    str(output),
                    "--source",
                    sources[0],
                    "--source",
                    sources[1],
                    "--ffmpeg",
                    str(executable),
                    "--confirm-private-media",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(10, summary["cases"])
            self.assertEqual(30, summary["frames"])
            self.assertEqual("pending", summary["review_status"])
            self.assertTrue(summary["private_media_used"])
            self.assertFalse(summary["paths_disclosed"])
            self.assertFalse(summary["uploads_performed"])
            self.assertFalse(summary["ready_for_ab"])

            review_path = output / "review.pending.jsonl"
            raw = review_path.read_text(encoding="utf-8")
            self.assertNotIn(sources[0], raw)
            self.assertNotIn(sources[1], raw)
            self.assertNotIn(str(root), completed.stdout)
            cases = [json.loads(line) for line in raw.splitlines()]
            self.assertEqual(10, len(cases))
            self.assertTrue(all(case["review_status"] == "pending" for case in cases))
            self.assertTrue(all(case["expected"] is None for case in cases))
            self.assertTrue(all(case["reviews"] == [] for case in cases))
            self.assertEqual(30, sum(len(case["frames"]) for case in cases))
            self.assertEqual(0o700, output.stat().st_mode & 0o777)
            self.assertEqual(0o600, review_path.stat().st_mode & 0o777)
            with self.assertRaises(ValueError):
                load_dataset(review_path)

    def test_requires_explicit_consent_and_loopback_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = _fake_ffmpeg(root)
            base = [
                sys.executable,
                "scripts/prepare_household_vlm_eval.py",
                "--output-dir",
                str(root / "review"),
                "--source",
                "rtsp://127.0.0.1:8554/cameras/a",
                "--source",
                "rtsp://127.0.0.1:8554/cameras/b",
                "--ffmpeg",
                str(executable),
            ]
            missing_consent = subprocess.run(base, check=False, capture_output=True, text=True)
            self.assertNotEqual(0, missing_consent.returncode)
            self.assertFalse((root / "review").exists())

            remote = base.copy()
            remote[remote.index("rtsp://127.0.0.1:8554/cameras/a")] = (
                "rtsp://192.0.2.10:8554/cameras/a"
            )
            remote.append("--confirm-private-media")
            rejected = subprocess.run(remote, check=False, capture_output=True, text=True)
            self.assertNotEqual(0, rejected.returncode)
            self.assertFalse((root / "review").exists())

    def test_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "review"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            executable = _fake_ffmpeg(root)
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_household_vlm_eval.py",
                    "--output-dir",
                    str(output),
                    "--source",
                    "rtsp://127.0.0.1:8554/cameras/a",
                    "--source",
                    "rtsp://127.0.0.1:8554/cameras/b",
                    "--ffmpeg",
                    str(executable),
                    "--confirm-private-media",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, completed.returncode)
            self.assertEqual("preserve", marker.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
