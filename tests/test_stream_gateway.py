from __future__ import annotations

import unittest

from baby_monitor.adapters.stream_gateway import (
    AnnexBAccessUnit,
    AnnexBRTSPPublisher,
    RTSPPublisherSettings,
    StreamPublisherError,
    VideoCodec,
)


class StreamGatewayContractTests(unittest.TestCase):
    def test_publisher_is_loopback_copy_only(self) -> None:
        settings = RTSPPublisherSettings(
            target_uri="rtsp://127.0.0.1:8554/cameras/cam_01/0",
            codec=VideoCodec.H265,
            input_fps=15.0,
        )
        command = AnnexBRTSPPublisher(settings).command()
        self.assertIn("hevc", command)
        self.assertEqual(
            "1",
            command[command.index("-use_wallclock_as_timestamps") + 1],
        )
        self.assertEqual("8192", command[command.index("-probesize") + 1])
        self.assertEqual("100000", command[command.index("-analyzeduration") + 1])
        self.assertEqual("copy", command[command.index("-c:v") + 1])
        self.assertEqual("tcp", command[command.index("-rtsp_transport") + 1])
        self.assertEqual(settings.target_uri, command[-1])

    def test_non_loopback_is_rejected_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            RTSPPublisherSettings(
                target_uri="rtsp://192.0.2.10:8554/cam",
                codec=VideoCodec.H264,
            )

    def test_credentials_are_rejected_from_process_command(self) -> None:
        with self.assertRaisesRegex(ValueError, "process lists"):
            # release-audit: allow=fixture-credential
            RTSPPublisherSettings(
                target_uri="rtsp://user:pass@127.0.0.1:8554/cam",
                codec=VideoCodec.H264,
            )

    def test_codec_switch_fails_closed(self) -> None:
        publisher = AnnexBRTSPPublisher(
            RTSPPublisherSettings(
                target_uri="rtsp://127.0.0.1:8554/cam",
                codec=VideoCodec.H264,
            )
        )
        unit = AnnexBAccessUnit(
            camera_id="cam_01",
            channel=0,
            codec=VideoCodec.H265,
            timestamp_ms=1,
            sequence=1,
            data=b"\x00\x00\x00\x01\x40\x01",
        )
        with self.assertRaisesRegex(StreamPublisherError, "codec changed"):
            publisher.write(unit)


if __name__ == "__main__":
    unittest.main()
