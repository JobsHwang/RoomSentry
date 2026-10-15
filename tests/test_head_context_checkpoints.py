from __future__ import annotations

import copy
import unittest

from scripts.head_context_checkpoints import complete_checkpoint_rows, make_case_checkpoint


def fixture(index=0):
    case = {
        "frame": f"synthetic_{index:02}.png",
        "camera_id": "camera_04",
        "png_sha256": "a" * 64,
        "raw_bgr_sha256": "b" * 64,
    }
    keys = [(None, None, rot) for rot in (0, 180)] + [
        (a, side, rot) for a in (0, 1) for side in (192, 288, 384) for rot in (0, 180)
    ]
    rows = [
        {
            "frame": case["frame"],
            "camera_id": case["camera_id"],
            "frame_sha256": case["png_sha256"],
            "raw_bgr_sha256": case["raw_bgr_sha256"],
            "anchor_index": a,
            "side_pixels": side,
            "rotation": rot,
            "pixels_sha256": "c" * 64,
            "tensor_sha256": "d" * 64,
            "boxes": [],
        }
        for a, side, rot in keys
    ]
    return case, rows


class HeadContextCheckpointTests(unittest.TestCase):
    def test_complete_source_is_detached_and_not_complete_comparison(self):
        case, rows = fixture()
        checkpoint = make_case_checkpoint(0, case, rows, protocol_sha256="e" * 64)
        rows[0]["boxes"].append({"synthetic": True})
        self.assertEqual([], checkpoint["rows"][0]["boxes"])
        self.assertIs(False, checkpoint["complete_matrix"])
        self.assertIs(False, checkpoint["accuracy_measured"])

    def test_complete_matrix_reassembles_in_frozen_source_order(self):
        cases, checkpoints = [], []
        for index in range(20):
            case, rows = fixture(index)
            cases.append(case)
            checkpoints.append(make_case_checkpoint(index, case, rows, protocol_sha256="e" * 64))
        restored = complete_checkpoint_rows(cases, checkpoints[::-1], protocol_sha256="e" * 64)
        self.assertEqual(280, len(restored))
        self.assertEqual("synthetic_00.png", restored[0]["frame"])
        restored[0]["boxes"].append({"synthetic": True})
        self.assertEqual([], checkpoints[0]["rows"][0]["boxes"])

    def test_partial_duplicate_and_undeclared_calls_fail(self):
        case, rows = fixture()
        for bad in (rows[:-1], rows + [rows[0]], [rows[0]] + rows[:-1]):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                make_case_checkpoint(0, case, bad, protocol_sha256="e" * 64)
        bad = copy.deepcopy(rows)
        bad[-1]["side_pixels"] = 960
        with self.assertRaises(ValueError):
            make_case_checkpoint(0, case, bad, protocol_sha256="e" * 64)

    def test_source_bindings_and_matrix_types_fail_closed(self):
        case, rows = fixture()
        for key, value in (
            ("frame", "another.png"),
            ("camera_id", "camera_03"),
            ("frame_sha256", "f" * 64),
            ("raw_bgr_sha256", "f" * 64),
            ("rotation", False),
            ("anchor_index", False),
            ("side_pixels", 192.0),
            ("pixels_sha256", "bad"),
            ("tensor_sha256", "D" * 64),
        ):
            bad = copy.deepcopy(rows)
            bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                make_case_checkpoint(0, case, bad, protocol_sha256="e" * 64)

    def test_frame_paths_indices_nonfinite_and_oversize_rejected(self):
        case, rows = fixture()
        for frame in ("../private.png", "x\\private.png", "/tmp/private.png", "bad.jpg"):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                make_case_checkpoint(0, {**case, "frame": frame}, rows, protocol_sha256="e" * 64)
        for index in (True, -1, 20, 1.0):
            with self.subTest(index=index), self.assertRaises(ValueError):
                make_case_checkpoint(index, case, rows, protocol_sha256="e" * 64)
        for value in (float("nan"), "x" * 1024**2):
            bad = copy.deepcopy(rows)
            bad[0]["extra"] = value
            with self.assertRaises(ValueError):
                make_case_checkpoint(0, case, bad, protocol_sha256="e" * 64)

    def test_partial_cohort_mixed_protocol_tampering_and_false_authority_rejected(self):
        cases, checkpoints = [], []
        for index in range(20):
            case, rows = fixture(index)
            cases.append(case)
            checkpoints.append(make_case_checkpoint(index, case, rows, protocol_sha256="e" * 64))
        with self.assertRaises(ValueError):
            complete_checkpoint_rows(cases, checkpoints[:18], protocol_sha256="e" * 64)
        for key, value in (
            ("protocol_sha256", "f" * 64),
            ("case_index", 1),
            ("case_index", False),
            ("version", True),
            ("rows_sha256", "f" * 64),
            ("complete_matrix", True),
            ("accuracy_measured", 0),
            ("production_writeback", True),
            ("extra", False),
        ):
            bad = copy.deepcopy(checkpoints)
            bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                complete_checkpoint_rows(cases, bad, protocol_sha256="e" * 64)


if __name__ == "__main__":
    unittest.main()
