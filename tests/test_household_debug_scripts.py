from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from baby_monitor.domain.candidate import Detection
from baby_monitor.pose_protocol import COCO_PERSON_KEYPOINTS, PoseFrame, PoseInstance, PoseKeypoint

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "scripts"), *sys.path]):
    from scripts import compare_yolo_reviewed_stills as stills
    from scripts import replay_household_holdout as replay


class HouseholdDebugScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir(mode=0o700)
        self.review = self.source / "visual-review"
        self.review.mkdir(mode=0o700)
        (self.review / "frames").mkdir(mode=0o700)
        (self.source / "admissions").mkdir(mode=0o700)
        self.output = self.root / "new-output"
        self.labels = self.root / "labels.json"
        self.weights = self.root / "weights"
        self.weights.mkdir(mode=0o700)
        self.interpreter = self.root / "fixture-python"
        self.interpreter.write_bytes(b"not-an-executable")
        self.interpreter.chmod(0o700)
        self.items = []
        self.selected = []
        self.add_frame("frame_01")
        self.persist_selection()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(stills, "VARIANTS", (("yolo11n", 640, False),)))
        self.stack.enter_context(
            patch.object(stills.shutil, "which", return_value="/usr/bin/unshare")
        )
        self.stack.enter_context(patch.object(stills.subprocess, "run"))
        self.stack.enter_context(
            patch.object(stills, "runtime_versions", return_value={"ultralytics": "fixture"})
        )
        admission = Mock(fingerprint="a" * 64)
        self.stack.enter_context(
            patch.object(stills.ExternalSourceAdmission, "read", return_value=admission)
        )
        manifest = Mock(sha256="b" * 64)
        self.stack.enter_context(
            patch.object(stills.YoloModelManifest, "read", return_value=manifest)
        )
        self.scan = Mock(return_value=[])
        self.detector = self.stack.enter_context(
            patch.object(
                stills, "CommandYoloDetector", return_value=SimpleNamespace(scan=self.scan)
            )
        )

    def add_frame(self, review_id: str, *, camera: str = "cam_01") -> Path:
        content = ("synthetic-still-" + review_id).encode()
        frame = self.review / "frames" / (review_id + ".jpg")
        frame.write_bytes(content)
        frame.chmod(0o600)
        video = self.source / ("20260912T000000Z_" + camera + "_fixture.mp4")
        video.write_bytes(b"synthetic-video")
        video.chmod(0o600)
        admission_file = self.source / "admissions" / (camera + ".source-admission.json")
        admission_file.write_text("{}", encoding="utf-8")
        admission_file.chmod(0o600)
        item = {
            "review_id": review_id,
            "camera_id": camera,
            "timestamp_seconds": float(len(self.items)),
            "source_frame_sha256": hashlib.sha256(content).hexdigest(),
            "visible_person": "unknown",
            "visible_person_count": None,
        }
        self.items.append(item)
        self.selected.append(
            {
                **item,
                "frame_filename": frame.name,
                "media_admission_sha256": "a" * 64,
            }
        )
        return frame

    def persist_selection(self) -> None:
        self.labels.write_text(
            json.dumps(
                {
                    "version": 1,
                    "model_generated_labels": True,
                    "human_verified": False,
                    "ground_truth": False,
                    "items": self.items,
                }
            ),
            encoding="utf-8",
        )
        self.labels.chmod(0o600)
        selection = self.review / "selection-manifest.json"
        selection.write_text(json.dumps({"items": self.selected}), encoding="utf-8")
        selection.chmod(0o600)

    def compare(self):
        return stills.compare(
            source_root=self.source,
            labels_path=self.labels,
            output_root=self.output,
            vision_python=self.interpreter,
            weights=self.weights,
            device="cpu",
        )

    def test_different_stills_of_same_camera_do_not_collide(self) -> None:
        self.add_frame("frame_02")
        self.persist_selection()
        result = self.compare()
        self.assertEqual(2, len(result["rows"]))
        self.assertEqual(2, self.scan.call_count)
        outputs = [p for p in self.output.glob("*.json") if not p.name.startswith("comparison-")]
        self.assertEqual(2, len(outputs))

    def assert_box_row(self, result, *, model: str, expected: list[dict]) -> None:
        row = result["rows"][0]
        self.assertEqual(expected, row["objects"])
        self.assertEqual(self.items[0]["review_id"], row["review_id"])
        self.assertEqual(self.items[0]["timestamp_seconds"], row["source_timestamp_seconds"])
        self.assertEqual(self.items[0]["source_frame_sha256"], row["source_frame_sha256"])
        self.assertEqual("normalized_selected_still_xyxy", row["bbox_coordinate_space"])
        self.assertFalse(result["plan"]["bbox_reference_matching_performed"])
        self.assertEqual(2, row["counts_at_thresholds"]["0.1"])
        self.assertEqual(1, row["counts_at_thresholds"]["0.5"])
        saved = json.loads((self.output / f"{model}-640-frame_01.json").read_text())
        self.assertEqual(row, saved)
        combined = json.loads((self.output / "comparison-result.json").read_text())
        self.assertEqual(expected, combined["rows"][0]["objects"])

    def test_detector_boxes_stay_bound_to_confidence_and_selected_frame(self) -> None:
        self.scan.return_value = [
            Detection(0.0, "person", 0.2, (0.03125, 0.125, 0.375, 0.75)),
            Detection(0.0, "person", 0.8, (0.5625, 0.25, 0.9375, 1.0)),
        ]
        self.assert_box_row(
            self.compare(),
            model="yolo11n",
            expected=[
                {"confidence": 0.2, "bbox_xyxy": [0.03125, 0.125, 0.375, 0.75]},
                {"confidence": 0.8, "bbox_xyxy": [0.5625, 0.25, 0.9375, 1.0]},
            ],
        )

    def test_pose_boxes_stay_bound_to_confidence_and_keypoint_count(self) -> None:
        poses = tuple(
            PoseInstance(
                instance_id=f"instance_{index}",
                confidence=confidence,
                bbox_xyxy=box,
                keypoints=tuple(
                    PoseKeypoint(name, 0.5, 0.5, 0.9 if number < index else 0.05, True)
                    for number, name in enumerate(COCO_PERSON_KEYPOINTS)
                ),
            )
            for index, confidence, box in (
                (1, 0.2, (0.03125, 0.125, 0.375, 0.75)),
                (2, 0.8, (0.5625, 0.25, 0.9375, 1.0)),
            )
        )
        with (
            patch.object(stills, "VARIANTS", (("yolo26m-pose", 640, True),)),
            patch.object(stills.YoloPoseModelManifest, "read", return_value=Mock(sha256="b" * 64)),
            patch.object(stills, "CommandYoloPoseEstimator") as estimator,
        ):
            estimator.return_value.scan.return_value = SimpleNamespace(
                frames=(PoseFrame("frame_0", 0.0, poses),)
            )
            self.assert_box_row(
                self.compare(),
                model="yolo26m-pose",
                expected=[
                    {
                        "confidence": 0.2,
                        "bbox_xyxy": [0.03125, 0.125, 0.375, 0.75],
                        "keypoints_at_010": 1,
                    },
                    {
                        "confidence": 0.8,
                        "bbox_xyxy": [0.5625, 0.25, 0.9375, 1.0],
                        "keypoints_at_010": 2,
                    },
                ],
            )
        self.detector.assert_not_called()

    def test_box_evidence_rejects_nonfinite_out_of_range_and_degenerate_boxes(self) -> None:
        for box in (
            None,
            (0.0, 0.0, 1.0),
            (False, 0.0, 1.0, 1.0),
            ("0", 0.0, 1.0, 1.0),
            (float("nan"), 0.0, 1.0, 1.0),
            (0.0, 0.0, float("inf"), 1.0),
            (-0.01, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.01, 1.0),
            (0.0, 0.0, 10**1000, 1.0),
            (0.8, 0.0, 0.2, 1.0),
            (0.0, 0.5, 1.0, 0.5),
        ):
            with self.subTest(box=box), self.assertRaises(ValueError):
                stills.box_evidence(box)

    def test_invalid_adapter_box_cannot_publish_comparison_result(self) -> None:
        self.scan.return_value = [SimpleNamespace(confidence=0.5, bbox_xyxy=(0.8, 0.0, 0.2, 1.0))]
        with self.assertRaises(ValueError):
            self.compare()
        self.assertFalse((self.output / "comparison-result.json").exists())
        self.assertFalse((self.output / "yolo11n-640-frame_01.json").exists())

    def test_invalid_pose_box_cannot_publish_comparison_result(self) -> None:
        with (
            patch.object(stills, "VARIANTS", (("yolo26m-pose", 640, True),)),
            patch.object(stills.YoloPoseModelManifest, "read", return_value=Mock(sha256="b" * 64)),
            patch.object(stills, "CommandYoloPoseEstimator") as estimator,
        ):
            estimator.return_value.scan.return_value = SimpleNamespace(
                frames=(
                    SimpleNamespace(
                        poses=(
                            SimpleNamespace(
                                confidence=0.5,
                                bbox_xyxy=(0.0, 0.0, float("nan"), 1.0),
                                keypoints=(),
                            ),
                        )
                    ),
                )
            )
            with self.assertRaises(ValueError):
                self.compare()
        self.assertFalse((self.output / "comparison-result.json").exists())
        self.assertFalse((self.output / "yolo26m-pose-640-frame_01.json").exists())

    def test_weak_label_unknown_count_stays_null(self) -> None:
        result = self.compare()
        self.assertIsNone(result["rows"][0]["reference_count"])
        self.assertFalse(result["plan"]["sample_is_independent_holdout"])
        self.assertFalse(result["plan"]["human_verified"])
        self.assertFalse(result["plan"]["official_accuracy_claim"])
        self.assertFalse(result["plan"]["deployment_approved"])

    def test_changed_frame_during_inference_cannot_publish_old_hash(self) -> None:
        frame = self.review / "frames" / "frame_01.jpg"

        def mutate(_source):
            frame.write_bytes(b"changed-synthetic-still")
            return []

        self.scan.side_effect = mutate
        with self.assertRaises(ValueError):
            self.compare()
        self.assertFalse((self.output / "comparison-result.json").exists())

    def test_mismatched_frame_hash_fails_before_model_construction(self) -> None:
        frame = self.review / "frames" / "frame_01.jpg"
        frame.write_bytes(b"changed-before-run")
        with self.assertRaises(ValueError):
            self.compare()
        self.detector.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_label_hash_must_describe_the_labels_actually_consumed(self) -> None:
        original_hash = hashlib.sha256(self.labels.read_bytes()).hexdigest()

        def mutate_after_parse(_path):
            changed = json.loads(self.labels.read_text(encoding="utf-8"))
            changed["items"][0]["visible_person"] = "present"
            changed["items"][0]["visible_person_count"] = 1
            self.labels.write_text(json.dumps(changed), encoding="utf-8")
            return Mock(sha256="b" * 64)

        with patch.object(stills.YoloModelManifest, "read", side_effect=mutate_after_parse):
            try:
                result = self.compare()
            except ValueError:
                # Either reject changed inputs, or retain the exact digest of
                # the immutable labels used. Never tag old labels with new bytes.
                self.assertFalse((self.output / "comparison-result.json").exists())
            else:
                self.assertEqual(original_hash, result["plan"]["label_file_sha256"])

    @unittest.skipUnless(os.name == "posix", "POSIX private input policy")
    def test_world_readable_labels_are_rejected_before_models(self) -> None:
        self.labels.chmod(0o644)
        with self.assertRaises(ValueError):
            self.compare()
        self.detector.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX private input policy")
    def test_nonprivate_source_directory_is_rejected_before_models(self) -> None:
        self.source.chmod(0o755)
        with self.assertRaises(ValueError):
            self.compare()
        self.detector.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX private input policy")
    def test_replay_nonprivate_source_is_rejected_before_old_evidence_is_read(self) -> None:
        self.source.chmod(0o755)
        with patch.object(replay.HouseholdPoseFusionTemporalHoldoutAttestation, "read") as read:
            with self.assertRaises(ValueError):
                replay.replay(
                    source_root=self.source,
                    output_root=self.output,
                    vision_python=self.interpreter,
                    weights=self.weights,
                    device="cpu",
                )
            read.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX private input policy")
    def test_nonprivate_selection_manifest_is_rejected_before_models(self) -> None:
        (self.review / "selection-manifest.json").chmod(0o644)
        with self.assertRaises(ValueError):
            self.compare()
        self.detector.assert_not_called()

    def test_unknown_reference_cannot_claim_zero_people(self) -> None:
        self.items[0]["visible_person_count"] = 0
        self.persist_selection()
        with self.assertRaises(ValueError):
            self.compare()
        self.detector.assert_not_called()

    def test_reference_ids_reject_path_traversal_before_models(self) -> None:
        for field in ("camera_id", "review_id"):
            with self.subTest(field=field):
                original = self.items[0][field]
                self.items[0][field] = "../outside"
                self.selected[0][field] = "../outside"
                self.persist_selection()
                with self.assertRaises(ValueError):
                    self.compare()
                self.detector.assert_not_called()
                self.items[0][field] = original
                self.selected[0][field] = original

    @unittest.skipUnless(os.name == "posix", "POSIX symlink policy")
    def test_output_symlink_ancestor_is_rejected_before_models(self) -> None:
        link = self.root / "parent-link"
        link.symlink_to(self.root, target_is_directory=True)
        self.output = link / "new-output"
        with self.assertRaises(ValueError):
            self.compare()
        self.detector.assert_not_called()

    def test_exclusive_writes_preserve_existing_artifacts(self) -> None:
        for module in (stills, replay):
            with self.subTest(module=module.__name__):
                path = self.root / (module.__name__.split(".")[-1] + ".json")
                module.write_new(path, {"original": True})
                original = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    module.write_new(path, {"original": False})
                self.assertEqual(original, path.read_bytes())
                if os.name == "posix":
                    self.assertEqual(0o600, path.stat().st_mode & 0o777)

    def test_confusion_excludes_unknown_without_claiming_true_negatives(self) -> None:
        self.assertEqual(
            {"tp": 1, "fn": 1, "fp": 1, "tn": 1, "unknown_excluded": 2},
            replay.confusion(
                [
                    ("present", True),
                    ("present", False),
                    ("absent", True),
                    ("absent", False),
                    ("unknown", True),
                    ("unknown", False),
                ]
            ),
        )
