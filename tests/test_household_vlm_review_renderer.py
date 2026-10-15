from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _pending_review(root: Path) -> Path:
    frames = root / "frames"
    frames.mkdir()
    lines = []
    for case_index in range(1, 11):
        path = frames / f"s01_{case_index:04d}.jpg"
        path.write_bytes(f"synthetic-{case_index}".encode())
        import hashlib

        lines.append(
            {
                "version": 1,
                "review_status": "pending",
                "case_id": f"household_{case_index:03d}",
                "frames": [
                    {
                        "frame_id": "f01",
                        "path": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                ],
                "question": "只标注可见事实。",
                "expected": None,
                "reviews": [],
            }
        )
    manifest = root / "review.pending.jsonl"
    manifest.write_text(
        "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines),
        encoding="utf-8",
    )
    return manifest


class HouseholdVLMReviewRendererTests(unittest.TestCase):
    def test_renders_offline_page_with_no_embedded_pixels_or_remote_urls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _pending_review(root)
            output = root / "review.local.html"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/render_household_vlm_review.py",
                    "--manifest",
                    str(manifest),
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
            self.assertFalse(summary["uploads_performed"])
            self.assertFalse(summary["pixels_disclosed"])
            self.assertFalse(summary["paths_disclosed"])
            page = output.read_text(encoding="utf-8")
            self.assertIn("connect-src 'none'", page)
            self.assertIn('src="frames/s01_0001.jpg"', page)
            self.assertIn("needs_second_review", page)
            self.assertNotIn("https://", page)
            self.assertNotIn("data:image", page)
            self.assertNotIn(str(root), page)

    def test_refuses_to_overwrite_review_page(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _pending_review(root)
            output = root / "review.local.html"
            output.write_text("preserve", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/render_household_vlm_review.py",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, completed.returncode)
            self.assertEqual("preserve", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
