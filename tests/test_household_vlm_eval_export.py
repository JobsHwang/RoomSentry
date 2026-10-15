from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from baby_monitor.ab_eval import load_dataset


def _completed_review(root: Path, *, safety: bool = False) -> Path:
    frames = root / "frames"
    frames.mkdir()
    cases = []
    for index in range(1, 11):
        frame = frames / f"f_{index:03d}.jpg"
        frame.write_bytes(f"synthetic-{index}".encode())
        expected = {
            "event_type": "possible_risk" if safety and index == 1 else "unknown",
            "risk_level": "medium" if safety and index == 1 else "none",
            "allowed_subjects": [],
            "allowed_observations": ["visible synthetic fixture"],
            "allowed_inferences": [],
            "required_observation_groups": [["visible synthetic fixture"]],
            "required_inference_groups": [],
        }
        cases.append(
            {
                "version": 1,
                "review_status": "approved",
                "case_id": f"household_{index:03d}",
                "frames": [
                    {
                        "frame_id": "f01",
                        "path": frame.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    }
                ],
                "question": "只标注可见事实。",
                "expected": expected,
                "reviews": [
                    {
                        "reviewer_id": "local_reviewer_1",
                        "verdict": "approve",
                        "reviewed_at": "2026-08-07T00:00:00Z",
                    }
                ],
            }
        )
    path = root / "review.completed.jsonl"
    path.write_text(
        "".join(json.dumps(case, separators=(",", ":")) + "\n" for case in cases),
        encoding="utf-8",
    )
    return path


class HouseholdVLMEvalExportTests(unittest.TestCase):
    def test_exports_fully_approved_review_as_v2_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = _completed_review(root)
            output = root / "dataset.reviewed.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/export_household_vlm_eval.py",
                    "--review",
                    str(review),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(10, summary["cases"])
            self.assertEqual(10, summary["frames"])
            self.assertTrue(summary["all_cases_approved"])
            self.assertTrue(summary["hashes_verified"])
            self.assertTrue(summary["ready_for_ab"])
            self.assertFalse(summary["paths_disclosed"])
            dataset = load_dataset(output)
            self.assertEqual(10, len(dataset))
            self.assertTrue(all(case.schema_version == 2 for case in dataset))

    def test_rejects_safety_case_without_second_reviewer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = _completed_review(root, safety=True)
            output = root / "dataset.reviewed.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/export_household_vlm_eval.py",
                    "--review",
                    str(review),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, completed.returncode)
            self.assertFalse(output.exists())

    def test_rejects_pending_second_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = _completed_review(root)
            raw = review.read_text(encoding="utf-8").replace(
                '"review_status":"approved"',
                '"review_status":"needs_second_review"',
                1,
            )
            review.write_text(raw, encoding="utf-8")
            output = root / "dataset.reviewed.jsonl"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/export_household_vlm_eval.py",
                    "--review",
                    str(review),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, completed.returncode)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
