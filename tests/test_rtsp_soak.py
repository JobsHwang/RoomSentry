from __future__ import annotations

import unittest

from baby_monitor.rtsp_soak import (
    RTSPSoakSettings,
    _completion_budget_seconds,
    _parse_progress_int,
)


class RTSPSoakSettingsTests(unittest.TestCase):
    def test_command_is_loopback_tcp_decode_to_null(self) -> None:
        settings = RTSPSoakSettings(
            uri="rtsp://127.0.0.1:8554/xiaomi/main",
            duration_seconds=7200,
        )
        command = settings.command()
        self.assertEqual("tcp", command[command.index("-rtsp_transport") + 1])
        self.assertEqual("7200", command[command.index("-t") + 1])
        self.assertEqual("pipe:1", command[command.index("-progress") + 1])
        self.assertEqual("null", command[command.index("-f") + 1])
        self.assertEqual(settings.uri, command[command.index("-i") + 1])

    def test_non_loopback_is_rejected_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            RTSPSoakSettings(uri="rtsp://192.0.2.10:8554/camera", duration_seconds=1)

    def test_credentials_and_query_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "credentials"):
            # release-audit: allow=fixture-credential
            RTSPSoakSettings(uri="rtsp://user:pass@127.0.0.1:8554/camera")
        with self.assertRaisesRegex(ValueError, "query"):
            RTSPSoakSettings(uri="rtsp://127.0.0.1:8554/camera?token=hidden")

    def test_duration_and_timeouts_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            RTSPSoakSettings(uri="rtsp://127.0.0.1:8554/camera", duration_seconds=0)
        with self.assertRaisesRegex(ValueError, "startup_timeout_seconds"):
            RTSPSoakSettings(
                uri="rtsp://127.0.0.1:8554/camera",
                startup_timeout_seconds=0,
            )

    def test_initial_ffmpeg_na_progress_is_treated_as_zero(self) -> None:
        self.assertEqual(0, _parse_progress_int("N/A"))
        self.assertEqual(0, _parse_progress_int(""))
        self.assertEqual(42, _parse_progress_int("42"))

    def test_long_soak_budget_allows_five_percent_clock_drift(self) -> None:
        settings = RTSPSoakSettings(
            uri="rtsp://127.0.0.1:8554/xiaomi/main",
            duration_seconds=7200,
            startup_timeout_seconds=30,
            completion_grace_seconds=60,
        )
        self.assertEqual(7590, _completion_budget_seconds(settings))


if __name__ == "__main__":
    unittest.main()
