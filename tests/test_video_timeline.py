from __future__ import annotations

import unittest
from unittest.mock import Mock

from baby_monitor.video_timeline import decoded_frame_timestamp_seconds


class VideoTimelineTests(unittest.TestCase):
    def test_valid_container_pts_takes_priority_over_nominal_fps(self) -> None:
        for index, milliseconds in ((0, 0), (5, 500), (5, 2300)):
            with self.subTest(index=index, milliseconds=milliseconds):
                capture = Mock()
                capture.get.return_value = milliseconds
                self.assertEqual(
                    decoded_frame_timestamp_seconds(
                        capture,
                        position_property=0,
                        frame_index=index,
                        native_fps=5,
                    ),
                    milliseconds / 1000,
                )
                capture.get.assert_called_once_with(0)

    def test_unavailable_or_invalid_pts_falls_back_to_nominal_fps(self) -> None:
        for milliseconds in (0, -1, float("nan"), float("inf"), None, "invalid"):
            with self.subTest(milliseconds=milliseconds):
                capture = Mock()
                capture.get.return_value = milliseconds
                self.assertEqual(
                    decoded_frame_timestamp_seconds(
                        capture,
                        position_property=0,
                        frame_index=5,
                        native_fps=5,
                    ),
                    1,
                )

    def test_missing_property_does_not_query_capture(self) -> None:
        capture = Mock()
        self.assertEqual(
            decoded_frame_timestamp_seconds(
                capture,
                position_property=None,
                frame_index=5,
                native_fps=5,
            ),
            1,
        )
        capture.get.assert_not_called()

    def test_invalid_capture_property_result_falls_back(self) -> None:
        for error in (TypeError, ValueError):
            with self.subTest(error=error):
                capture = Mock()
                capture.get.side_effect = error
                self.assertEqual(
                    decoded_frame_timestamp_seconds(
                        capture,
                        position_property=0,
                        frame_index=5,
                        native_fps=5,
                    ),
                    1,
                )

    def test_backend_runtime_failure_is_not_silently_masked(self) -> None:
        capture = Mock()
        capture.get.side_effect = RuntimeError
        with self.assertRaises(RuntimeError):
            decoded_frame_timestamp_seconds(
                capture,
                position_property=0,
                frame_index=5,
                native_fps=5,
            )


if __name__ == "__main__":
    unittest.main()
