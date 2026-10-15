from __future__ import annotations

import unittest
from types import SimpleNamespace

from baby_monitor.frame_transform import (
    inverse_normalized_box,
    inverse_normalized_point,
    rotate_frame,
    validate_image_size,
    validate_rotation_degrees,
)


class FrameTransformTests(unittest.TestCase):
    def test_inverse_points_cover_all_supported_clockwise_rotations(self) -> None:
        self.assertEqual(
            (0.2, 0.3),
            inverse_normalized_point(0.2, 0.3, rotation_degrees=0),
        )
        self.assertEqual(
            (0.3, 0.8),
            inverse_normalized_point(0.2, 0.3, rotation_degrees=90),
        )
        self.assertEqual(
            (0.8, 0.7),
            inverse_normalized_point(0.2, 0.3, rotation_degrees=180),
        )
        self.assertEqual(
            (0.7, 0.2),
            inverse_normalized_point(0.2, 0.3, rotation_degrees=270),
        )

    def test_inverse_box_returns_original_normalized_coordinate_space(self) -> None:
        self.assertEqual(
            (0.2, 0.6, 0.8, 0.9),
            inverse_normalized_box(
                (0.1, 0.2, 0.4, 0.8),
                rotation_degrees=90,
            ),
        )

    def test_rotation_calls_only_the_explicit_opencv_transform(self) -> None:
        calls: list[tuple[object, int]] = []
        cv2 = SimpleNamespace(
            ROTATE_90_CLOCKWISE=7,
            rotate=lambda frame, constant: calls.append((frame, constant)) or "rotated",
        )

        self.assertEqual(
            "rotated",
            rotate_frame("original", cv2_module=cv2, rotation_degrees=90),
        )
        self.assertEqual([("original", 7)], calls)
        self.assertEqual(
            "original",
            rotate_frame("original", cv2_module=SimpleNamespace(), rotation_degrees=0),
        )

    def test_transform_settings_fail_closed(self) -> None:
        for invalid in (319, 641, 1312, True):
            with self.subTest(image_size=invalid), self.assertRaises(ValueError):
                validate_image_size(invalid)
        for invalid in (-90, 45, 360, False):
            with self.subTest(rotation=invalid), self.assertRaises(ValueError):
                validate_rotation_degrees(invalid)


if __name__ == "__main__":
    unittest.main()
