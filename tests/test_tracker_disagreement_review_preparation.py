from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.object_tracker import (
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
)
from baby_monitor.tracker_evaluation import (
    TrackerDisagreementReview,
    TrackerPredictionAssignment,
    TrackerPredictionBatch,
    compare_tracker_predictions,
)


class TrackerDisagreementReviewPreparationTests(unittest.TestCase):
    @staticmethod
    def _batch() -> DetectionFrameBatch:
        base = datetime(2026, 8, 10, 2, 3, 4, tzinfo=UTC)
        return DetectionFrameBatch(
            camera_id="quick_review_cam",
            detector_adapter="synthetic_detector",
            frames=tuple(
                DetectionFrame(
                    frame_id=f"frame_{index:04d}",
                    occurred_at=base + timedelta(seconds=index),
                    detections=(
                        FrameDetection(
                            detection_id=f"a{index}",
                            label="person",
                            confidence=0.9,
                            bbox_xyxy=(0.1, 0.1, 0.3, 0.8),
                        ),
                        FrameDetection(
                            detection_id=f"b{index}",
                            label="person",
                            confidence=0.9,
                            bbox_xyxy=(0.6, 0.1, 0.8, 0.8),
                        ),
                    ),
                )
                for index in range(2)
            ),
        )

    @staticmethod
    def _prediction(
        batch: DetectionFrameBatch,
        *,
        adapter: str,
        ids: tuple[str, str, str, str],
    ) -> TrackerPredictionBatch:
        keys = [
            (frame.frame_id, detection.detection_id)
            for frame in batch.frames
            for detection in frame.detections
        ]
        return TrackerPredictionBatch(
            camera_id=batch.camera_id,
            detector_adapter=batch.detector_adapter,
            detection_batch_sha256=batch.fingerprint,
            tracker_adapter=adapter,
            tracker_revision="fixture_v1",
            tracker_config_sha256=("a" if adapter == "left_tracker" else "b") * 64,
            runtime_id="python_stdlib",
            assignments=tuple(
                TrackerPredictionAssignment(
                    frame_id=frame_id,
                    detection_id=detection_id,
                    tracking_id=tracking_id,
                )
                for (frame_id, detection_id), tracking_id in zip(
                    keys,
                    ids,
                    strict=True,
                )
            ),
        )

    def _inputs(self, root: Path) -> tuple[list[str], Path]:
        batch = self._batch()
        left = self._prediction(
            batch,
            adapter="left_tracker",
            ids=("left_a", "left_b", "left_a", "left_b"),
        )
        right = self._prediction(
            batch,
            adapter="right_tracker",
            ids=("right_a", "right_b", "right_b", "right_a"),
        )
        comparison = compare_tracker_predictions(batch, left, right)
        source_root = root / "source_review"
        frame_root = source_root / "frames"
        frame_root.mkdir(parents=True)
        media_items = []
        for frame in batch.frames:
            content = f"bounded jpeg fixture {frame.frame_id}".encode()
            path = frame_root / f"{frame.frame_id}.jpg"
            path.write_bytes(content)
            media_items.append(
                {
                    "frame_id": frame.frame_id,
                    "path": f"frames/{frame.frame_id}.jpg",
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        media = source_root / "frame-media.local.json"
        media.write_text(
            json.dumps(
                {
                    "version": 1,
                    "detection_batch_sha256": batch.fingerprint,
                    "frames": media_items,
                    "private_relative_paths_included": True,
                    "source_path_included": False,
                    "uploads_performed": False,
                }
            ),
            encoding="utf-8",
        )
        paths = []
        for name, value in (
            ("frames.json", batch.to_dict()),
            ("left.json", left.to_dict()),
            ("right.json", right.to_dict()),
            ("comparison.json", comparison.to_dict()),
        ):
            path = root / name
            path.write_text(json.dumps(value), encoding="utf-8")
            paths.append(str(path))
        return paths, media

    def test_prepares_minimal_private_six_choice_style_review(self) -> None:
        with tempfile.TemporaryDirectory(prefix="PRIVATE_LINK_REVIEW_") as directory:
            root = Path(directory)
            inputs, media = self._inputs(root)
            output = root / "output"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_tracker_disagreement_review.py",
                    "--frames",
                    inputs[0],
                    "--left",
                    inputs[1],
                    "--right",
                    inputs[2],
                    "--comparison",
                    inputs[3],
                    "--media-manifest",
                    str(media),
                    "--output-dir",
                    str(output),
                    "--confirm-private-media",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(4, summary["questions"])
            self.assertEqual(2, summary["unique_frames"])
            self.assertFalse(summary["human_labels"])
            self.assertFalse(summary["model_generated_labels"])
            self.assertFalse(summary["overall_accuracy_evaluated"])
            self.assertFalse(summary["global_adapter_selected"])
            self.assertFalse(summary["paths_disclosed"])
            self.assertNotIn(str(root), completed.stdout)

            batch = DetectionFrameBatch.read(inputs[0])
            left = TrackerPredictionBatch.read(inputs[1])
            right = TrackerPredictionBatch.read(inputs[2])
            comparison = compare_tracker_predictions(batch, left, right)
            review = TrackerDisagreementReview.read(output / "review.pending.json")
            review.bind(batch, left, right, comparison)
            self.assertEqual(0o700, output.stat().st_mode & 0o777)
            self.assertEqual(
                0o600,
                (output / "review.pending.json").stat().st_mode & 0o777,
            )

            page = (output / "review.local.html").read_text(encoding="utf-8")
            self.assertEqual(4, page.count('class="question"'))
            self.assertEqual(8, page.count('class="crop"'))
            self.assertEqual(8, page.count('class="source-fallback"'))
            self.assertIn("只需回答 4 个", page)
            self.assertIn("是，同一个人", page)
            self.assertIn("不是，不同的人", page)
            self.assertIn("任一框含多人 / 重叠 / 看不清（本题不计分）", page)
            self.assertIn("每个黄框都只对应一名可辨认的人时才判断", page)
            self.assertIn("不用填写任何内容", page)
            self.assertNotIn('id="reviewer"', page)
            self.assertNotIn("匿名 reviewer：", page)
            self.assertIn("String.fromCharCode(10)", page)
            self.assertNotIn("JSON.stringify(completed, null, 2) + '\n'", page)
            self.assertIn("connect-src 'none'", page)
            self.assertNotIn("http://", page)
            self.assertNotIn("https://", page)
            self.assertNotIn("left_tracker", page)
            self.assertNotIn("right_tracker", page)
            self.assertNotIn(str(root), page)

            copied = json.loads((output / "frame-media.local.json").read_text(encoding="utf-8"))
            self.assertEqual(2, len(copied["frames"]))
            for item in copied["frames"]:
                path = output / item["path"]
                self.assertEqual(item["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(0o600, path.stat().st_mode & 0o777)

    def test_requires_confirmation_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, media = self._inputs(root)
            output = root / "output"
            base = [
                sys.executable,
                "scripts/prepare_tracker_disagreement_review.py",
                "--frames",
                inputs[0],
                "--left",
                inputs[1],
                "--right",
                inputs[2],
                "--comparison",
                inputs[3],
                "--media-manifest",
                str(media),
                "--output-dir",
                str(output),
            ]
            missing_confirmation = subprocess.run(
                base,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, missing_confirmation.returncode)
            self.assertFalse(output.exists())

            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            exists = subprocess.run(
                [*base, "--confirm-private-media"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, exists.returncode)
            self.assertEqual("preserve", marker.read_text(encoding="utf-8"))

    def test_manifest_mismatch_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, media = self._inputs(root)
            value = json.loads(media.read_text(encoding="utf-8"))
            value["detection_batch_sha256"] = "a" * 64
            media.write_text(json.dumps(value), encoding="utf-8")
            output = root / "output"
            failed = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_tracker_disagreement_review.py",
                    "--frames",
                    inputs[0],
                    "--left",
                    inputs[1],
                    "--right",
                    inputs[2],
                    "--comparison",
                    inputs[3],
                    "--media-manifest",
                    str(media),
                    "--output-dir",
                    str(output),
                    "--confirm-private-media",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, failed.returncode)
            self.assertIn("provenance is invalid", failed.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
