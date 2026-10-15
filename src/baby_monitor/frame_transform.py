from __future__ import annotations

import math
from typing import Any

SUPPORTED_ROTATIONS = (0, 90, 180, 270)
MIN_IMAGE_SIZE = 320
MAX_IMAGE_SIZE = 1280


def validate_rotation_degrees(value: object) -> int:
    if type(value) is not int or value not in SUPPORTED_ROTATIONS:
        raise ValueError("rotation_degrees must be one of 0, 90, 180, or 270")
    return value


def validate_image_size(value: object) -> int:
    if type(value) is not int or not MIN_IMAGE_SIZE <= value <= MAX_IMAGE_SIZE or value % 32 != 0:
        raise ValueError("image_size must be a multiple of 32 between 320 and 1280")
    return value


def rotate_frame(
    frame: object,
    *,
    cv2_module: Any,
    rotation_degrees: int,
) -> object:
    rotation = validate_rotation_degrees(rotation_degrees)
    if rotation == 0:
        return frame
    constant_name = {
        90: "ROTATE_90_CLOCKWISE",
        180: "ROTATE_180",
        270: "ROTATE_90_COUNTERCLOCKWISE",
    }[rotation]
    rotate = getattr(cv2_module, "rotate", None)
    constant = getattr(cv2_module, constant_name, None)
    if not callable(rotate) or constant is None:
        raise RuntimeError("OpenCV runtime does not support the configured rotation")
    return rotate(frame, constant)


def inverse_normalized_point(
    x: float,
    y: float,
    *,
    rotation_degrees: int,
) -> tuple[float, float]:
    rotation = validate_rotation_degrees(rotation_degrees)
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (x, y)):
        raise ValueError("normalized point is invalid")
    if rotation == 0:
        original_x, original_y = x, y
    elif rotation == 90:
        original_x, original_y = y, 1.0 - x
    elif rotation == 180:
        original_x, original_y = 1.0 - x, 1.0 - y
    else:
        original_x, original_y = 1.0 - y, x
    return (
        min(1.0, max(0.0, original_x)),
        min(1.0, max(0.0, original_y)),
    )


def inverse_normalized_box(
    value: object,
    *,
    rotation_degrees: int,
) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("normalized box must contain four coordinates")
    try:
        left, top, right, bottom = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("normalized box contains a non-number") from exc
    if not (
        all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in (left, top, right, bottom))
        and left < right
        and top < bottom
    ):
        raise ValueError("normalized box is invalid")
    points = tuple(
        inverse_normalized_point(x, y, rotation_degrees=rotation_degrees)
        for x, y in (
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        )
    )
    xs = tuple(point[0] for point in points)
    ys = tuple(point[1] for point in points)
    return min(xs), min(ys), max(xs), max(ys)
