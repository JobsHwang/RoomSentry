from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from baby_monitor.motion_tracker import MotionTrackerConfig
from baby_monitor.object_tracker import (
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
    IoUTrackerConfig,
)
from baby_monitor.tracker_evaluation import (
    TrackerAssociationReview,
    TrackerPredictionBatch,
    build_iou_tracker_predictions,
    build_motion_tracker_predictions,
    compare_tracker_predictions,
)


def _fake_ffmpeg(root: Path, *, fail: bool = False) -> Path:
    executable = root / ("failing_ffmpeg.py" if fail else "fake_ffmpeg.py")
    executable.write_text(
        (
            "#!/usr/bin/env python3\nimport sys\nsys.exit(2)\n"
            if fail
            else """#!/usr/bin/env python3
import pathlib
import sys
pathlib.Path(sys.argv[-1]).write_bytes(b'bounded synthetic jpeg fixture')
"""
        ),
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


def _batch() -> DetectionFrameBatch:
    from datetime import UTC, datetime, timedelta

    base = datetime(2026, 8, 8, 1, 2, 3, tzinfo=UTC)
    return DetectionFrameBatch(
        camera_id="synthetic_review_cam",
        detector_adapter="synthetic_detector",
        frames=tuple(
            DetectionFrame(
                frame_id=f"frame_{index:04d}",
                occurred_at=base + timedelta(seconds=index * 0.5),
                detections=(
                    FrameDetection(
                        detection_id=f"detection_{index:04d}",
                        label="person",
                        confidence=0.9,
                        bbox_xyxy=(0.1 + index * 0.01, 0.1, 0.4, 0.8),
                    ),
                ),
            )
            for index in range(2)
        ),
    )


def _admission(source: Path) -> dict[str, object]:
    content = source.read_bytes()
    return {
        "version": 1,
        "source_id": "synthetic_review_media",
        "asset_kind": "media_asset",
        "canonical_url": "https://example.com/synthetic-review-media",
        "accessed_on": "2026-08-08",
        "source_revision": "fixture_v1",
        "upstream_name": "Synthetic tracker review fixture",
        "license": {
            "name": "Apache License 2.0",
            "spdx_id": "Apache-2.0",
            "terms_url": "https://www.apache.org/licenses/LICENSE-2.0",
            "commercial_use": "allowed",
            "modification": "allowed",
            "redistribution": "allowed",
        },
        "access": {
            "gated": "no",
            "authentication_required": "no",
            "expected_bytes": len(content),
        },
        "content": {
            "observed_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        "privacy": {
            "real_minors": "no",
            "faces": "not_applicable",
            "audio": "no",
            "device_or_account_ids": "no",
            "precise_timestamps": "no",
            "private_household_data": "no",
        },
        "labels": {
            "origin": "none",
            "human_verified": False,
            "ground_truth": False,
        },
        "admission": {
            "tier": "A",
            "status": "verified_local",
            "purpose": "adapter_development",
            "requires_user_approval": False,
            "user_approval_recorded": False,
            "git_allowed": False,
            "ci_allowed": False,
            "redistribution_allowed": False,
            "training_allowed": False,
            "reviewed_on": "2026-08-08",
            "blocker_codes": [],
            "restriction_codes": [],
        },
    }


class TrackerReviewPreparationTests(unittest.TestCase):
    def test_prepares_private_offline_review_with_integrity_binding(self) -> None:
        with tempfile.TemporaryDirectory(prefix="PRIVATE_TRACK_REVIEW_") as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"synthetic local video")
            frames = root / "detections.json"
            frames.write_text(json.dumps(_batch().to_dict()), encoding="utf-8")
            admission = root / "admission.json"
            admission.write_text(json.dumps(_admission(source)), encoding="utf-8")
            left = build_iou_tracker_predictions(_batch(), IoUTrackerConfig())
            right_value = build_motion_tracker_predictions(
                _batch(),
                MotionTrackerConfig(),
            ).to_dict()
            right_assignments = right_value["assignments"]
            assert isinstance(right_assignments, list)
            right_assignments[1]["tracking_id"] = "track_review_difference"  # type: ignore[index]
            right = TrackerPredictionBatch.from_dict(right_value)
            comparison = root / "comparison.json"
            compare_tracker_predictions(_batch(), left, right).write_new(comparison)
            output = root / "review"
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_tracker_review.py",
                    "--frames",
                    str(frames),
                    "--source",
                    str(source),
                    "--source-admission",
                    str(admission),
                    "--prediction-comparison",
                    str(comparison),
                    "--output-dir",
                    str(output),
                    "--ffmpeg",
                    str(_fake_ffmpeg(root)),
                    "--first-frame-offset-seconds",
                    "0",
                    "--confirm-private-media",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(2, summary["frames"])
            self.assertEqual(2, summary["detections"])
            self.assertEqual("pending", summary["review_status"])
            self.assertFalse(summary["human_labels"])
            self.assertFalse(summary["model_generated_labels"])
            self.assertFalse(summary["uploads_performed"])
            self.assertTrue(summary["prediction_comparison_used"])
            self.assertEqual(2, summary["disagreement_frames"])
            self.assertFalse(summary["accuracy_evaluated"])
            self.assertFalse(summary["better_adapter_selected"])
            self.assertFalse(summary["paths_disclosed"])
            self.assertNotIn(str(root), completed.stdout)

            review = TrackerAssociationReview.read(output / "review.pending.json")
            review.bind(_batch())
            self.assertEqual(summary["review_sha256"], review.fingerprint)
            self.assertEqual(0o700, output.stat().st_mode & 0o777)
            self.assertEqual(
                0o600,
                (output / "review.pending.json").stat().st_mode & 0o777,
            )
            media = json.loads((output / "frame-media.local.json").read_text(encoding="utf-8"))
            self.assertEqual(_batch().fingerprint, media["detection_batch_sha256"])
            self.assertEqual(2, len(media["frames"]))
            for item in media["frames"]:
                path = output / item["path"]
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    item["sha256"],
                )

            page = (output / "review.local.html").read_text(encoding="utf-8")
            self.assertIn("connect-src 'none'", page)
            self.assertIn('src="frames/frame_0000.jpg"', page)
            self.assertIn("association_on_detector_boxes", page)
            self.assertIn("subject_001", page)
            self.assertEqual(2, page.count('class="frame disagreement"'))
            self.assertIn("不表示任一 adapter 正确或错误", page)
            self.assertNotIn("https://", page)
            self.assertNotIn("data:image", page)
            self.assertNotIn(str(root), page)
            self.assertTrue((output / "PRIVATE_MEDIA_DO_NOT_COMMIT").is_file())

    def test_requires_confirmation_and_never_overwrites_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"synthetic local video")
            frames = root / "detections.json"
            frames.write_text(json.dumps(_batch().to_dict()), encoding="utf-8")
            admission = root / "admission.json"
            admission.write_text(json.dumps(_admission(source)), encoding="utf-8")
            output = root / "review"
            base = [
                sys.executable,
                "scripts/prepare_tracker_review.py",
                "--frames",
                str(frames),
                "--source",
                str(source),
                "--source-admission",
                str(admission),
                "--output-dir",
                str(output),
                "--ffmpeg",
                str(_fake_ffmpeg(root)),
                "--first-frame-offset-seconds",
                "0",
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

    def test_rejects_prediction_comparison_for_a_different_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"synthetic local video")
            frames = root / "detections.json"
            frames.write_text(json.dumps(_batch().to_dict()), encoding="utf-8")
            admission = root / "admission.json"
            admission.write_text(json.dumps(_admission(source)), encoding="utf-8")
            left = build_iou_tracker_predictions(_batch(), IoUTrackerConfig())
            right = build_motion_tracker_predictions(_batch(), MotionTrackerConfig())
            comparison_value = compare_tracker_predictions(
                _batch(),
                left,
                right,
            ).to_dict()
            comparison_value["detection_batch_sha256"] = "a" * 64
            comparison = root / "comparison.json"
            comparison.write_text(json.dumps(comparison_value), encoding="utf-8")
            output = root / "review"

            failed = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_tracker_review.py",
                    "--frames",
                    str(frames),
                    "--source",
                    str(source),
                    "--source-admission",
                    str(admission),
                    "--prediction-comparison",
                    str(comparison),
                    "--output-dir",
                    str(output),
                    "--ffmpeg",
                    str(_fake_ffmpeg(root)),
                    "--first-frame-offset-seconds",
                    "0",
                    "--confirm-private-media",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(0, failed.returncode)
            self.assertIn("different batch", failed.stderr)
            self.assertFalse(output.exists())

    def test_extraction_failure_removes_only_new_incomplete_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"synthetic local video")
            frames = root / "detections.json"
            frames.write_text(json.dumps(_batch().to_dict()), encoding="utf-8")
            admission = root / "admission.json"
            admission.write_text(json.dumps(_admission(source)), encoding="utf-8")
            output = root / "review"
            unrelated = root / "unrelated.txt"
            unrelated.write_text("preserve", encoding="utf-8")
            failed = subprocess.run(
                [
                    sys.executable,
                    "scripts/prepare_tracker_review.py",
                    "--frames",
                    str(frames),
                    "--source",
                    str(source),
                    "--source-admission",
                    str(admission),
                    "--output-dir",
                    str(output),
                    "--ffmpeg",
                    str(_fake_ffmpeg(root, fail=True)),
                    "--first-frame-offset-seconds",
                    "0",
                    "--confirm-private-media",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, failed.returncode)
            self.assertFalse(output.exists())
            self.assertEqual("preserve", unrelated.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
