from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.config import StorageSettings
from baby_monitor.domain import EventMedia, EvidenceGap
from baby_monitor.ring_buffer import (
    BufferSegment,
    RingBufferError,
    RTSPSegmentBuffer,
    SegmentBufferSettings,
    SegmentManifest,
    _coverage_gaps,
    extract_buffered_evidence,
)
from baby_monitor.storage import StorageResolver


class RingBufferContractTests(unittest.TestCase):
    def test_camera_id_must_be_a_safe_non_personal_identifier(self) -> None:
        with self.assertRaisesRegex(ValueError, "safe non-personal"):
            SegmentBufferSettings(
                uri="rtsp://127.0.0.1:8554/test/camera",
                camera_id="baby/name",
                buffer_root=Path("./data/test-buffer"),
            )

    def test_settings_are_loopback_copy_only_and_bounded(self) -> None:
        settings = SegmentBufferSettings(
            uri="rtsp://127.0.0.1:8554/camera/main",
            camera_id="cam_01",
            buffer_root=Path("buffer"),
        )
        command = settings.command("0123456789ab", 30)
        self.assertEqual("tcp", command[command.index("-rtsp_transport") + 1])
        self.assertEqual("5000000", command[command.index("-timeout") + 1])
        self.assertEqual("copy", command[command.index("-c:v") + 1])
        self.assertEqual("segment", command[command.index("-f") + 1])
        self.assertNotIn("-strftime", command)
        self.assertEqual("0", command[command.index("-segment_start_number") + 1])
        self.assertRegex(command[-1], r"_[0-9a-f]{12}_%06d\.mkv$")
        self.assertEqual("csv", command[command.index("-segment_list_type") + 1])
        self.assertEqual("+live", command[command.index("-segment_list_flags") + 1])
        self.assertEqual("rtsp://127.0.0.1:8554/camera/main", settings.audit_ref)

        with self.assertRaisesRegex(ValueError, "credentials"):
            # release-audit: allow=fixture-credential
            SegmentBufferSettings(
                uri="rtsp://user:pass@127.0.0.1:8554/camera",
                camera_id="cam_01",
                buffer_root=Path("buffer"),
            )
        with self.assertRaisesRegex(ValueError, "loopback"):
            SegmentBufferSettings(
                uri="rtsp://192.0.2.10:8554/camera",
                camera_id="cam_01",
                buffer_root=Path("buffer"),
            )

    def test_eviction_requires_both_persistent_and_one_shot_flags(self) -> None:
        with self.assertRaisesRegex(ValueError, "persistent opt-in"):
            SegmentBufferSettings(
                uri="rtsp://127.0.0.1:8554/camera",
                camera_id="cam_01",
                buffer_root=Path("buffer"),
                eviction_enabled=True,
                eviction_confirmed=False,
            )
        settings = SegmentBufferSettings(
            uri="rtsp://127.0.0.1:8554/camera",
            camera_id="cam_01",
            buffer_root=Path("buffer"),
            eviction_enabled=True,
            eviction_confirmed=True,
        )
        self.assertTrue(settings.eviction_enabled)

    def test_manifest_rejects_unsafe_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            now = datetime(2026, 8, 2, tzinfo=UTC)
            manifest = SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/camera",
                created_at=now,
                updated_at=now,
                segments=(
                    BufferSegment(
                        filename="segment_20260802T000000Z_0123456789ab.mkv",
                        started_at=now,
                        ended_at=now + timedelta(seconds=2),
                        duration_seconds=2,
                        size_bytes=10,
                        session_id="0123456789ab",
                    ),
                ),
            )
            value = manifest.to_dict()
            value["segments"][0]["filename"] = "../outside.mkv"  # type: ignore[index]
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(RingBufferError):
                SegmentManifest.read(manifest_path)

    def test_coverage_gap_is_explicit(self) -> None:
        start = datetime(2026, 8, 2, tzinfo=UTC)
        segments = [
            BufferSegment(
                "segment_20260802T000000Z_0123456789ab.mkv",
                start,
                start + timedelta(seconds=2),
                2,
                10,
                "0123456789ab",
            ),
            BufferSegment(
                "segment_20260802T000004Z_0123456789ab.mkv",
                start + timedelta(seconds=4),
                start + timedelta(seconds=6),
                2,
                10,
                "0123456789ab",
            ),
        ]
        gaps = _coverage_gaps(segments, start, start + timedelta(seconds=6))
        self.assertEqual(1, len(gaps))
        self.assertEqual("segment_missing", gaps[0].reason)
        self.assertEqual(start + timedelta(seconds=2), gaps[0].started_at)
        media = EventMedia(
            clip_path="event.mp4",
            frame_paths=("frame_01.jpg",),
            coverage_gaps=(gaps[0],),
            selected_segments=2,
        )
        self.assertFalse(media.complete_coverage)
        self.assertEqual(media, EventMedia.from_dict(media.to_dict()))
        schema = json.loads(Path("schemas/event-media-v1.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(1, media.to_dict()["version"])
        self.assertEqual(1, schema["properties"]["version"]["const"])
        self.assertFalse(schema["additionalProperties"])
        with self.assertRaisesRegex(ValueError, "positive duration"):
            EvidenceGap(start, start, "segment_missing")

    def test_owned_temp_segments_are_evicted_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SegmentBufferSettings(
                uri="rtsp://127.0.0.1:8554/camera",
                camera_id="cam_01",
                buffer_root=Path(directory),
                max_segments=3,
                eviction_enabled=True,
                eviction_confirmed=True,
            )
            worker = RTSPSegmentBuffer(settings)
            start = datetime(2026, 8, 2, tzinfo=UTC)
            for index in range(4):
                timestamp = start + timedelta(seconds=index * 2)
                filename = f"segment_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_0123456789ab.mkv"
                path = settings.buffer_directory / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"owned-test-segment")
                worker._known[filename] = BufferSegment(  # noqa: SLF001
                    filename,
                    timestamp,
                    timestamp + timedelta(seconds=2),
                    2,
                    path.stat().st_size,
                    "0123456789ab",
                )
            worker._evict_if_allowed()  # noqa: SLF001
            self.assertEqual(3, len(worker._known))  # noqa: SLF001
            self.assertEqual(1, worker._evicted)  # noqa: SLF001

    def test_refresh_reports_progress_when_eviction_keeps_buffer_full(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SegmentBufferSettings(
                uri="rtsp://127.0.0.1:8554/camera",
                camera_id="cam_01",
                buffer_root=Path(directory),
                max_segments=3,
                eviction_enabled=True,
                eviction_confirmed=True,
            )
            settings.buffer_directory.mkdir(parents=True)
            worker = RTSPSegmentBuffer(settings)
            old_started_at = datetime(2026, 8, 2, tzinfo=UTC)
            old_paths: list[Path] = []
            for index in range(3):
                started_at = old_started_at + timedelta(seconds=index * 2)
                old_filename = (
                    f"segment_{started_at.strftime('%Y%m%dT%H%M%SZ')}_aaaaaaaaaaaa_{index:06d}.mkv"
                )
                old_path = settings.buffer_directory / old_filename
                old_path.write_bytes(b"old-owned-test-segment")
                old_paths.append(old_path)
                worker._known[old_filename] = BufferSegment(  # noqa: SLF001
                    old_filename,
                    started_at,
                    started_at + timedelta(seconds=2),
                    2,
                    old_path.stat().st_size,
                    "aaaaaaaaaaaa",
                )

            session_id = "bbbbbbbbbbbb"
            new_filename = "segment_20260802T000002Z_bbbbbbbbbbbb_000000.mkv"
            (settings.buffer_directory / new_filename).write_bytes(b"new-owned-test-segment")
            (settings.buffer_directory / f"segments_{session_id}.csv").write_text(
                f"{new_filename},0,2\n",
                encoding="utf-8",
            )

            self.assertTrue(worker._refresh(session_id))  # noqa: SLF001
            self.assertEqual(3, len(worker._known))  # noqa: SLF001
            self.assertIn(new_filename, worker._known)  # noqa: SLF001
            self.assertFalse(old_paths[0].exists())
            self.assertFalse(worker._refresh(session_id))  # noqa: SLF001

    def test_non_evicting_buffer_reports_owned_file_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SegmentBufferSettings(
                uri="rtsp://127.0.0.1:8554/camera",
                camera_id="cam_01",
                buffer_root=Path(directory),
                max_segments=3,
            )
            settings.buffer_directory.mkdir(parents=True)
            start = datetime(2026, 8, 2, tzinfo=UTC)
            for index in range(3):
                timestamp = start + timedelta(seconds=index * 2)
                path = settings.buffer_directory / (
                    f"segment_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_0123456789ab_{index:06d}.mkv"
                )
                path.write_bytes(b"owned-test-segment")
            worker = RTSPSegmentBuffer(settings)
            self.assertTrue(worker._capacity_reached())  # noqa: SLF001
            self.assertEqual(3, len(tuple(settings.buffer_directory.glob("*.mkv"))))


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg and ffprobe are required",
)
class BufferedEvidenceIntegrationTests(unittest.TestCase):
    def test_segments_are_assembled_into_idempotent_clip_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segment_root = root / "buffer" / "cam_01"
            segment_root.mkdir(parents=True)
            raw_pattern = root / "raw_%02d.mkv"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=160x90:rate=10",
                    "-t",
                    "6",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-g",
                    "10",
                    "-sc_threshold",
                    "0",
                    "-f",
                    "segment",
                    "-segment_time",
                    "2",
                    "-reset_timestamps",
                    "1",
                    str(raw_pattern),
                ],
                check=True,
            )
            raw_segments = sorted(root.glob("raw_*.mkv"))
            self.assertGreaterEqual(len(raw_segments), 3)
            base = datetime(2026, 8, 2, tzinfo=UTC)
            session_id = "0123456789ab"
            records: list[BufferSegment] = []
            cursor = base
            for raw in raw_segments:
                duration = float(
                    subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "default=nw=1:nk=1",
                            str(raw),
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout.strip()
                )
                filename = f"segment_{cursor.strftime('%Y%m%dT%H%M%SZ')}_{session_id}.mkv"
                destination = segment_root / filename
                raw.replace(destination)
                records.append(
                    BufferSegment(
                        filename,
                        cursor,
                        cursor + timedelta(seconds=duration),
                        duration,
                        destination.stat().st_size,
                        session_id,
                    )
                )
                cursor += timedelta(seconds=duration)
            manifest_path = segment_root / "manifest.json"
            SegmentManifest(
                camera_id="cam_01",
                source_ref="rtsp://127.0.0.1:8554/test",
                created_at=base,
                updated_at=cursor,
                segments=tuple(records),
                gaps=(),
            ).write_atomic(manifest_path)
            storage = StorageResolver(
                StorageSettings(
                    local_root=root / "output",
                    database=root / "events.sqlite3",
                )
            )
            kwargs = {
                "manifest_path": manifest_path,
                "event_time": base + timedelta(seconds=3),
                "event_id": "event_01",
                "pre_event_seconds": 2.0,
                "post_event_seconds": 2.0,
                "keyframe_count": 5,
                "storage": storage,
            }
            result = extract_buffered_evidence(**kwargs)
            self.assertTrue(result.complete_coverage)
            self.assertFalse(result.reused_existing)
            self.assertGreaterEqual(result.selected_segments, 2)
            self.assertTrue(Path(result.media.clip_path).is_file())
            self.assertEqual(5, len(result.media.frame_paths))
            self.assertTrue(all(Path(path).stat().st_size > 0 for path in result.media.frame_paths))
            clip_duration = float(
                subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=nw=1:nk=1",
                        result.media.clip_path,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
            self.assertGreater(clip_duration, 3.5)
            self.assertLess(clip_duration, 4.5)

            replay = extract_buffered_evidence(**kwargs)
            self.assertTrue(replay.reused_existing)
            self.assertEqual(result.media, replay.media)


if __name__ == "__main__":
    unittest.main()
