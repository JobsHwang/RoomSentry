from __future__ import annotations

import copy
import unittest

from scripts import yolo_input_diagnostic as diagnostic


def fixture():
    frames = [
        {
            "frame_id": f"frame_{i}",
            "frame_sha256": str(i + 1) * 64,
            "pixels_sha256_by_rotation": {"0": "a" * 64, "180": "b" * 64},
            "reference_presence": label,
        }
        for i, label in enumerate(("present", "absent", "unknown"))
    ]
    rows = [
        {
            "frame_id": frame["frame_id"],
            "frame_sha256": frame["frame_sha256"],
            "arm": arm,
            "rotation": rotation,
            "boxes": [],
            "elapsed_ms": 1.0,
            "input_shape": [1, 3, *shape],
            "input_dtype": dtype,
            "tensor_sha256": "c" * 64,
            "pixels_sha256": frame["pixels_sha256_by_rotation"][str(rotation)],
            "head_end2end": True,
        }
        for frame in frames
        for arm, shape, dtype in diagnostic.ARMS
        for rotation in diagnostic.ROTATIONS
    ]
    return frames, rows


class YoloInputDiagnosticTests(unittest.TestCase):
    def test_explicit_empty_matrix_keeps_references_and_unknown(self):
        frames, rows = fixture()
        result = diagnostic.score(frames, rows, [0.1, 0.35])
        self.assertEqual(24, result["cells"])
        self.assertEqual(16, len(result["summaries"]))
        for row in result["summaries"]:
            self.assertEqual({"tp": 0, "fn": 1, "fp": 0, "tn": 1}, row["reference_presence"])
            self.assertEqual(1, row["unknown_reference_frames"])
        for field in (
            "baby_accuracy_verified",
            "formal_box_accuracy",
            "deployment_approved",
            "independent_holdout",
        ):
            self.assertIs(result[field], False)

    def test_sweep_and_precision_differences_are_candidate_counts_only(self):
        frames, rows = fixture()
        rows[0]["boxes"] = [{"confidence": 0.2, "bbox_xyxy": [0, 0, 1, 1]}]
        result = diagnostic.score(frames, rows, [0.1, 0.35])
        self.assertEqual(1, result["summaries"][0]["reference_presence"]["tp"])
        self.assertEqual(1, result["summaries"][1]["reference_presence"]["fn"])
        self.assertEqual(
            [{"frame_id": "frame_0", "fp16_boxes": 1, "fp32_boxes": 0}],
            result["precision_candidate_count_changes"][0]["changed_frames"],
        )
        self.assertEqual([], result["precision_candidate_count_changes"][1]["changed_frames"])

    def test_missing_extra_duplicate_cells_fail(self):
        frames, rows = fixture()
        for invalid in (rows[:-1], rows + [rows[0]], [rows[0], *rows[1:-1], rows[0]]):
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                diagnostic.score(frames, invalid, [0.1])

    def test_invalid_cell_binding_shape_dtype_and_head_fail(self):
        frames, rows = fixture()
        invalid = {
            "frame_id": ("other", None),
            "frame_sha256": ("d" * 64, "A" * 64),
            "pixels_sha256": ("d" * 64,),
            "arm": ("fp64_640", None),
            "rotation": (True, 90, "0"),
            "input_shape": ([1, 3, 640, 384], [True, 3, 384, 640], (1, 3, 384, 640)),
            "input_dtype": ("float32", None),
            "tensor_sha256": ("bad",),
            "head_end2end": (False, 1),
            "elapsed_ms": (-1, True, float("nan"), 120_001),
        }
        for field, values in invalid.items():
            for value in values:
                changed = copy.deepcopy(rows)
                changed[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    diagnostic.score(frames, changed, [0.1])

    def test_bad_boxes_fail(self):
        frames, rows = fixture()
        for confidence, xy in (
            (True, [0, 0, 1, 1]),
            (0.09, [0, 0, 1, 1]),
            (float("inf"), [0, 0, 1, 1]),
            (0.2, [0, 0, 0, 1]),
            (0.2, [0, 0, True, 1]),
            (0.2, [-1, 0, 1, 1]),
        ):
            changed = copy.deepcopy(rows)
            changed[0]["boxes"] = [{"confidence": confidence, "bbox_xyxy": xy}]
            with self.subTest(confidence=confidence, xy=xy), self.assertRaises(ValueError):
                diagnostic.score(frames, changed, [0.1])

    def test_saturation_is_explicit_without_trimming(self):
        frames, rows = fixture()
        rows[0]["boxes"] = [{"confidence": 0.5, "bbox_xyxy": [0, 0, 1, 1]} for _ in range(100)]
        report = diagnostic.score(frames, rows, [0.1])
        self.assertEqual(["frame_0"], report["summaries"][0]["saturated_source_frames"])
        rows[0]["boxes"].append(rows[0]["boxes"][0])
        with self.assertRaises(ValueError):
            diagnostic.score(frames, rows, [0.1])

    def test_thresholds_require_source_lower_bound_and_unique_order(self):
        frames, rows = fixture()
        for invalid in ([], [0.35], [0.1, 0.1], [0.35, 0.1], [True], [0.1, float("nan")], (0.1,)):
            with self.subTest(thresholds=invalid), self.assertRaises(ValueError):
                diagnostic.score(frames, rows, invalid)

    def test_frame_hash_duplicates_bad_labels_and_pixels_fail(self):
        frames, rows = fixture()
        for field, value in (
            ("frame_sha256", frames[1]["frame_sha256"]),
            ("reference_presence", True),
            ("reference_presence", "yes"),
            ("pixels_sha256_by_rotation", {"0": "a" * 64}),
            ("frame_id", "../frame"),
        ):
            changed = copy.deepcopy(frames)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnostic.score(changed, rows, [0.1])

    def test_inputs_are_unchanged_and_record_order_does_not_change_counts(self):
        frames, rows = fixture()
        before = copy.deepcopy((frames, rows))
        result = diagnostic.score(frames, rows, [0.1])
        reversed_result = diagnostic.score(frames, rows[::-1], [0.1])
        self.assertEqual(result["summaries"], reversed_result["summaries"])
        self.assertEqual(before, (frames, rows))


if __name__ == "__main__":
    unittest.main()
