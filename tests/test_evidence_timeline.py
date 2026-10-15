from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from baby_monitor.adapters.camera import LocalFileCamera
from baby_monitor.adapters.media import FFmpegMediaProcessor
from baby_monitor.config import StorageSettings
from baby_monitor.domain import CandidateEvent
from baby_monitor.ring_buffer import BufferSegment, SegmentManifest, extract_buffered_evidence
from baby_monitor.storage import StorageResolver


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class EvidenceTimelineIntegrationTests(unittest.TestCase):
    BASE = datetime(2026, 9, 12, tzinfo=UTC)

    @staticmethod
    def _generate(source, *, irregular=False):
        video_filter = "select=not(between(t\\,2\\,4.999))"
        if irregular:
            video_filter += ",settb=1/1000000,setpts=PTS+mod(N\\,3)*17003"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=10:duration=6",
                "-vf",
                video_filter,
                "-fps_mode",
                "passthrough",
                "-enc_time_base",
                "1:1000000",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=30,
        )

    @staticmethod
    def _pts(source):
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(source),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return tuple(
            float(frame["best_effort_timestamp_time"])
            for frame in json.loads(probe.stdout)["frames"]
        )

    @staticmethod
    def _storage(root):
        return StorageResolver(
            StorageSettings(local_root=root / "output", database=root / "events.sqlite3")
        )

    def _assert_timeline_preserved(self, source, before_sha, reference, media, *, tolerance=0.001):
        self.assertEqual(before_sha, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(3, len(media.frame_paths))
        for frame in media.frame_paths:
            path = Path(frame)
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(0o600, path.stat().st_mode & 0o777)
            self.assertEqual(b"\xff\xd8", path.read_bytes()[:2])
        clip = Path(media.clip_path)
        self.assertEqual(0o600, clip.stat().st_mode & 0o777)
        actual = self._pts(clip)
        source_gaps = tuple(
            right - left for left, right in zip(reference, reference[1:], strict=False)
        )
        output_gaps = tuple(right - left for left, right in zip(actual, actual[1:], strict=False))
        diagnostic = (
            f"source_frames={len(reference)}, output_frames={len(actual)}, "
            f"source_max_gap={max(source_gaps):.6f}s, output_max_gap={max(output_gaps):.6f}s"
        )
        self.assertEqual(len(reference), len(actual), diagnostic)
        # Codec/container timestamp origins may shift; compare adjacent differences.
        for index, (expected, observed) in enumerate(zip(source_gaps, output_gaps, strict=True)):
            self.assertAlmostEqual(
                expected, observed, delta=tolerance, msg=f"gap {index}: {diagnostic}"
            )

    def _buffered_media(self, source, root, *, start=0.0):
        manifest = source.parent / "manifest.json"
        SegmentManifest(
            camera_id="cam_01",
            source_ref="rtsp://127.0.0.1:8554/synthetic",
            created_at=self.BASE,
            updated_at=self.BASE + timedelta(seconds=6),
            segments=(
                BufferSegment(
                    filename=source.name,
                    started_at=self.BASE,
                    ended_at=self.BASE + timedelta(seconds=6),
                    duration_seconds=6.0,
                    size_bytes=source.stat().st_size,
                    session_id="0123456789ab",
                ),
            ),
        ).write_atomic(manifest)
        midpoint = (start + 6.0) / 2
        return extract_buffered_evidence(
            manifest_path=manifest,
            event_time=self.BASE + timedelta(seconds=midpoint),
            event_id="timeline_buffer",
            pre_event_seconds=midpoint - start,
            post_event_seconds=6.0 - midpoint,
            keyframe_count=3,
            storage=self._storage(root),
        )

    def test_local_file_full_window_does_not_fill_internal_source_pts_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.mp4"
            self._generate(source)
            before_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            reference = self._pts(source)
            self.assertEqual(30, len(reference))
            self.assertAlmostEqual(3.1, reference[20] - reference[19], delta=0.001)
            media = FFmpegMediaProcessor(keyframe_count=3).prepare(
                source=LocalFileCamera("cam_01", source).source(),
                candidate=CandidateEvent("timeline_local", "cam_01", 0.0, 6.0, ()),
                occurred_at=self.BASE,
                storage=self._storage(root),
            )
            self._assert_timeline_preserved(source, before_sha, reference, media)

    def test_buffered_full_single_segment_does_not_fill_internal_source_pts_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer = root / "buffer"
            buffer.mkdir()
            session = "0123456789ab"
            source = buffer / f"segment_20260912T000000Z_{session}.mkv"
            self._generate(source)
            before_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            reference = self._pts(source)
            self.assertEqual(30, len(reference))
            self.assertAlmostEqual(3.1, reference[20] - reference[19], delta=0.001)
            result = self._buffered_media(source, root)
            self.assertEqual(1, result.selected_segments)
            # Manifest-window coverage does not certify frames inside the interval.
            self.assertTrue(result.complete_coverage)
            self._assert_timeline_preserved(source, before_sha, reference, result.media)

    def test_window_start_inside_gap_does_not_generate_leading_duplicate_frames(self):
        for buffered in (False, True):
            with self.subTest(buffered=buffered), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / (
                    "segment_20260912T000000Z_0123456789ab.mkv" if buffered else "synthetic.mp4"
                )
                self._generate(source)
                before_sha = hashlib.sha256(source.read_bytes()).hexdigest()
                all_pts = self._pts(source)
                selected = tuple(pts for pts in all_pts if 3.0 <= pts < 6.0)
                self.assertEqual(10, len(selected))
                self.assertEqual(5.0, selected[0])
                if buffered:
                    media = self._buffered_media(source, root, start=3.0).media
                else:
                    media = FFmpegMediaProcessor(keyframe_count=3).prepare(
                        source=LocalFileCamera("cam_01", source).source(),
                        candidate=CandidateEvent("timeline_local", "cam_01", 3.0, 6.0, ()),
                        occurred_at=self.BASE,
                        storage=self._storage(root),
                    )
                self._assert_timeline_preserved(source, before_sha, selected, media)

    def test_irregular_vfr_adjacent_pts_survive_both_transcode_paths(self):
        for buffered in (False, True):
            with self.subTest(buffered=buffered), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / (
                    "segment_20260912T000000Z_0123456789ab.mkv" if buffered else "synthetic.mp4"
                )
                self._generate(source, irregular=True)
                before_sha = hashlib.sha256(source.read_bytes()).hexdigest()
                reference = self._pts(source)
                self.assertEqual(30, len(reference))
                self.assertGreater(reference[1] - reference[0], 0.11)
                self.assertLess(reference[3] - reference[2], 0.07)
                self.assertGreater(reference[20] - reference[19], 3.0)
                if buffered:
                    media = self._buffered_media(source, root).media
                else:
                    media = FFmpegMediaProcessor(keyframe_count=3).prepare(
                        source=LocalFileCamera("cam_01", source).source(),
                        candidate=CandidateEvent("timeline_local", "cam_01", 0.0, 6.0, ()),
                        occurred_at=self.BASE,
                        storage=self._storage(root),
                    )
                self._assert_timeline_preserved(
                    source, before_sha, reference, media, tolerance=0.00001
                )


if __name__ == "__main__":
    unittest.main()
