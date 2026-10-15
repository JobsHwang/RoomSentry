from __future__ import annotations

import io
import json
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from baby_monitor.cli import main
from baby_monitor.pose_protocol import (
    COCO_PERSON_KEYPOINTS,
    PoseFrame,
    PoseFrameBatch,
    PoseInstance,
    PoseKeypoint,
)
from baby_monitor.pose_quality import (
    PoseQualityConfig,
    PoseQualityError,
    PoseQualityReport,
    evaluate_pose_quality,
)


def _pose(
    instance_id: str,
    bbox: tuple[float, float, float, float],
    *,
    high_confidence_points: int = 17,
) -> PoseInstance:
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    keypoints = []
    for index, name in enumerate(COCO_PERSON_KEYPOINTS):
        column = index % 4
        row = index // 4
        keypoints.append(
            PoseKeypoint(
                name=name,
                x=left + width * (0.2 + 0.15 * column),
                y=top + height * (0.15 + 0.16 * row),
                confidence=0.9 if index < high_confidence_points else 0.2,
                in_frame=True,
            )
        )
    return PoseInstance(
        instance_id=instance_id,
        confidence=0.9,
        bbox_xyxy=bbox,
        keypoints=tuple(keypoints),
    )


def _batch(*frames: PoseFrame) -> PoseFrameBatch:
    return PoseFrameBatch(
        camera_id="public_pose_cam",
        pose_adapter="fixture_pose_v1",
        model_id="fixture_pose",
        model_revision="fixture-v1",
        model_sha256="a" * 64,
        frames=frames,
    )


class PoseQualityTests(unittest.TestCase):
    def test_static_pair_has_full_coverage_and_zero_displacement(self) -> None:
        first = _pose("p0001", (0.1, 0.1, 0.5, 0.9))
        second = _pose("p0001", (0.1, 0.1, 0.5, 0.9))
        report = evaluate_pose_quality(
            _batch(
                PoseFrame(frame_id="f000001", timestamp_seconds=0.0, poses=(first,)),
                PoseFrame(frame_id="f000002", timestamp_seconds=1.0, poses=(second,)),
            )
        )

        self.assertEqual("temporal_measurements_available", report.summary.evidence_status)
        self.assertEqual(2, report.summary.usable_pose_count)
        self.assertEqual(1.0, report.summary.usable_pose_rate)
        self.assertEqual(1, report.summary.matched_pose_pairs)
        self.assertEqual(1, report.summary.comparable_pose_pairs)
        self.assertEqual(0.0, report.summary.image_displacement_median)
        self.assertEqual(0.0, report.summary.shape_displacement_median)
        self.assertEqual(0, report.summary.image_change_pairs)
        self.assertEqual(0, report.summary.shape_change_pairs)
        self.assertTrue(report.anonymous_pairing_only)
        self.assertFalse(report.subject_identity_inference)
        self.assertFalse(report.posture_classification)
        self.assertFalse(report.risk_inference)

    def test_translation_changes_image_coordinates_but_preserves_bbox_relative_shape(self) -> None:
        first = _pose("p0001", (0.1, 0.1, 0.5, 0.9))
        translated = _pose("p0001", (0.2, 0.1, 0.6, 0.9))
        report = evaluate_pose_quality(
            _batch(
                PoseFrame(frame_id="f000001", timestamp_seconds=0.0, poses=(first,)),
                PoseFrame(frame_id="f000002", timestamp_seconds=1.0, poses=(translated,)),
            ),
            PoseQualityConfig(
                image_change_threshold=0.05,
                shape_change_threshold=0.001,
            ),
        )

        self.assertEqual(1, report.summary.image_change_pairs)
        self.assertEqual(0, report.summary.shape_change_pairs)
        self.assertGreater(report.summary.image_displacement_median or 0.0, 0.05)
        self.assertAlmostEqual(0.0, report.summary.shape_displacement_median or 0.0)

    def test_occlusion_like_confidence_loss_is_insufficient_not_a_posture_claim(self) -> None:
        first = _pose("p0001", (0.1, 0.1, 0.5, 0.9))
        low_coverage = _pose(
            "p0001",
            (0.1, 0.1, 0.5, 0.9),
            high_confidence_points=4,
        )
        report = evaluate_pose_quality(
            _batch(
                PoseFrame(frame_id="f000001", timestamp_seconds=0.0, poses=(first,)),
                PoseFrame(
                    frame_id="f000002",
                    timestamp_seconds=1.0,
                    poses=(low_coverage,),
                ),
            )
        )

        self.assertEqual("insufficient_shared_keypoints", report.summary.evidence_status)
        self.assertEqual(1, report.summary.usable_pose_count)
        self.assertEqual(1, report.summary.matched_pose_pairs)
        self.assertEqual(0, report.summary.comparable_pose_pairs)
        self.assertEqual(1, report.summary.insufficient_shared_keypoint_pairs)
        self.assertIsNone(report.summary.image_displacement_median)
        self.assertIsNone(report.summary.shape_displacement_median)
        self.assertFalse(report.posture_classification)
        self.assertFalse(report.risk_inference)

    def test_empty_pose_frames_produce_explicit_no_evidence(self) -> None:
        report = evaluate_pose_quality(
            _batch(
                PoseFrame(frame_id="f000001", timestamp_seconds=0.0, poses=()),
                PoseFrame(frame_id="f000002", timestamp_seconds=1.0, poses=()),
            )
        )

        self.assertEqual("no_pose_evidence", report.summary.evidence_status)
        self.assertEqual(0, report.summary.pose_count)
        self.assertIsNone(report.summary.usable_pose_rate)
        self.assertIsNone(report.summary.temporal_comparability_rate)

    def test_report_is_strict_round_trips_and_matches_public_schema(self) -> None:
        pose = _pose("p0001", (0.1, 0.1, 0.5, 0.9))
        report = evaluate_pose_quality(
            _batch(PoseFrame(frame_id="f000001", timestamp_seconds=0.0, poses=(pose,)))
        )
        round_trip = PoseQualityReport.from_json(report.to_json())
        self.assertEqual(report, round_trip)
        with self.assertRaisesRegex(PoseQualityError, "duplicate keys"):
            PoseQualityReport.from_json('{"version":1,"version":1}')
        with self.assertRaisesRegex(PoseQualityError, "non-standard number"):
            PoseQualityReport.from_json('{"version":NaN}')

        schema = json.loads(
            Path("schemas/pose-quality-report-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(report.to_dict()))
        fixture = PoseQualityReport.read("tests/fixtures/valid_pose_quality_report.json")
        self.assertEqual("no_pose_evidence", fixture.summary.evidence_status)

        payload = report.to_dict()
        payload["posture_classification"] = True
        with self.assertRaisesRegex(PoseQualityError, "must be false"):
            PoseQualityReport.from_dict(payload)

        payload = report.to_dict()
        payload["unexpected"] = True
        with self.assertRaisesRegex(PoseQualityError, "fields do not match"):
            PoseQualityReport.from_dict(payload)

        payload = report.to_dict()
        payload["summary"]["pose_count"] = 2
        with self.assertRaises(PoseQualityError):
            PoseQualityReport.from_dict(payload)

    def test_config_rejects_boolean_and_out_of_range_thresholds(self) -> None:
        with self.assertRaises(PoseQualityError):
            PoseQualityConfig(keypoint_confidence_threshold=True)
        with self.assertRaises(PoseQualityError):
            PoseQualityConfig(min_shared_keypoints=18)
        with self.assertRaises(PoseQualityError):
            PoseQualityConfig(min_pair_bbox_iou=0.0)
        with self.assertRaises(PoseQualityError):
            PoseQualityConfig(image_change_threshold=-0.1)

    def test_report_rejects_impossible_usable_pose_aggregate(self) -> None:
        pose = _pose("p0001", (0.1, 0.1, 0.5, 0.9))
        report = evaluate_pose_quality(
            _batch(PoseFrame(frame_id="f000001", timestamp_seconds=0.0, poses=(pose,)))
        )
        payload = report.to_dict()
        payload["frames"][0]["high_confidence_keypoint_count"] = 0

        with self.assertRaisesRegex(PoseQualityError, "keypoint coverage"):
            PoseQualityReport.from_dict(payload)

    def test_report_rejects_change_count_that_contradicts_metrics(self) -> None:
        pose = _pose("p0001", (0.1, 0.1, 0.5, 0.9))
        report = evaluate_pose_quality(
            _batch(
                PoseFrame(frame_id="f000001", timestamp_seconds=0.0, poses=(pose,)),
                PoseFrame(frame_id="f000002", timestamp_seconds=1.0, poses=(pose,)),
            )
        )
        payload = report.to_dict()
        payload["transitions"][0]["image_change_pairs"] = 1
        payload["summary"]["image_change_pairs"] = 1

        with self.assertRaisesRegex(PoseQualityError, "change count"):
            PoseQualityReport.from_dict(payload)

    def test_cli_writes_once_and_validates_without_disclosing_paths(self) -> None:
        pose = _pose("p0001", (0.1, 0.1, 0.5, 0.9))
        batch = _batch(
            PoseFrame(frame_id="f000001", timestamp_seconds=0.0, poses=(pose,)),
            PoseFrame(frame_id="f000002", timestamp_seconds=1.0, poses=(pose,)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch_path = root / "pose-frames.json"
            report_path = root / "pose-quality.json"
            batch_path.write_text(batch.to_json(), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "evaluate-pose-quality",
                        "--pose-frames",
                        str(batch_path),
                        "--output",
                        str(report_path),
                    ]
                )
            self.assertEqual(0, code)
            self.assertTrue(report_path.is_file())
            self.assertEqual(0o600, stat.S_IMODE(report_path.stat().st_mode))
            self.assertNotIn(str(root), output.getvalue())
            self.assertFalse(json.loads(output.getvalue())["paths_disclosed"])

            validated_output = io.StringIO()
            with redirect_stdout(validated_output):
                validate_code = main(["validate-pose-quality", str(report_path)])
            self.assertEqual(0, validate_code)
            self.assertNotIn(str(root), validated_output.getvalue())

            report_link = root / "pose-quality-link.json"
            report_link.symlink_to(report_path)
            with self.assertRaisesRegex(PoseQualityError, "missing or unsafe"):
                PoseQualityReport.read(report_link)

            errors = io.StringIO()
            with redirect_stderr(errors):
                duplicate_code = main(
                    [
                        "evaluate-pose-quality",
                        "--pose-frames",
                        str(batch_path),
                        "--output",
                        str(report_path),
                    ]
                )
            self.assertEqual(2, duplicate_code)
            self.assertNotIn(str(root), errors.getvalue())


if __name__ == "__main__":
    unittest.main()
