from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path

from baby_monitor.adapters.vlm import MockVLMAdapter
from baby_monitor.daily_digest_source import (
    DailyDigestSourceError,
    latest_daily_digest_local_date,
    load_daily_digest_candidates,
)
from baby_monitor.database import EventDatabase
from baby_monitor.domain import (
    CandidateEvent,
    Detection,
    EventMedia,
    SignalEvent,
    SignalType,
)
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy


class DailyDigestSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.root.chmod(0o700)
        self.state = self.root / "state"
        self.state.mkdir(mode=0o700)
        self.storage = self.root / "storage"
        self.storage.mkdir(mode=0o700)
        media_directory = self.storage
        for component in ("events", "2026", "08", "19"):
            media_directory = media_directory / component
            media_directory.mkdir(mode=0o700)
        self.clip = media_directory / "PRIVATE_EVENT_CLIP.mp4"
        self.clip.write_bytes(b"synthetic-private-media-fixture")
        self.clip.chmod(0o600)
        self.frame = media_directory / "PRIVATE_EVENT_FRAME.jpg"
        self.frame.write_bytes(b"synthetic-frame-fixture")
        self.frame.chmod(0o600)
        self.database_path = self.state / "events.sqlite3"
        self.database = EventDatabase(self.database_path)
        occurred_at = datetime(2026, 8, 19, 1, tzinfo=UTC)
        candidate = CandidateEvent(
            event_id="private_event_01",
            camera_id="camera_01",
            start_seconds=0.0,
            end_seconds=1.0,
            detections=(Detection(0.5, "person", 0.82, (0.1, 0.1, 0.8, 0.9)),),
            signals=(
                SignalEvent(
                    signal_id="private_signal_01",
                    camera_id="camera_01",
                    occurred_at=occurred_at,
                    signal_type=SignalType.MOTION,
                    source_adapter="ha_local",
                    confidence=0.73,
                ),
            ),
            detector_id="fixture_detector",
        )
        process_prepared_event(
            candidate=candidate,
            occurred_at=occurred_at,
            source_kind="fixture",
            source_ref="private_source_ref",
            media=EventMedia(str(self.clip), (str(self.frame),)),
            local_vlm=MockVLMAdapter(),
            database=self.database,
            policy=ReviewPolicy(),
        )
        self.database_path.chmod(0o600)

    def _probe(self, descriptor: int) -> float:
        os.lseek(descriptor, 0, os.SEEK_SET)
        self.assertEqual(self.clip.read_bytes(), os.read(descriptor, 1024))
        return 12.0

    def _load(self, **overrides: object):
        arguments = {
            "database_path": self.database_path,
            "storage_root": self.storage,
            "local_date": date(2026, 8, 19),
            "timezone_name": "UTC",
            "duration_probe": self._probe,
        }
        arguments.update(overrides)
        return load_daily_digest_candidates(**arguments)  # type: ignore[arg-type]

    def test_read_only_admission_hashes_private_clip_without_decoding(self) -> None:
        before_mtime = self.database_path.stat().st_mtime_ns
        admission = self._load()
        public = admission.to_public_dict()
        self.assertEqual(1, admission.source_event_count)
        self.assertEqual(1, admission.media_files_hashed)
        self.assertEqual(self.clip.stat().st_size, admission.media_bytes_hashed)
        self.assertTrue(public["database_read_only"])
        self.assertTrue(public["source_hashes_verified"])
        self.assertTrue(public["media_read"])
        self.assertFalse(public["media_content_decoded"])
        self.assertFalse(public["fact_offsets_admitted"])
        self.assertFalse(public["video_generated"])
        self.assertEqual(before_mtime, self.database_path.stat().st_mtime_ns)
        rendered = json.dumps(public)
        self.assertNotIn(str(self.root), rendered)
        self.assertNotIn("private_event", rendered)

        candidate = admission.candidates[0]
        self.assertEqual("private_event_01", candidate.event_id)
        self.assertEqual(0.82, candidate.person_confidence)
        self.assertEqual(0.73, candidate.motion_score)
        self.assertTrue(candidate.hardware_signal)
        self.assertEqual(12.0, candidate.media_duration_seconds)
        self.assertEqual(64, len(candidate.source_binding_sha256))
        self.assertEqual(1, len(admission.media_sources))
        self.assertEqual(self.clip, admission.media_sources[0].clip_path)
        self.assertNotIn(str(self.root), repr(admission))

    def test_source_binding_changes_when_exact_clip_bytes_change(self) -> None:
        first = self._load().candidates[0].source_binding_sha256
        self.clip.write_bytes(b"changed-private-media-fixture")
        self.clip.chmod(0o600)
        second = self._load().candidates[0].source_binding_sha256
        self.assertNotEqual(first, second)

    @unittest.skipUnless(
        Path("/usr/bin/ffmpeg").is_file() and Path("/usr/bin/ffprobe").is_file(),
        "FFmpeg tools are required",
    )
    def test_real_ffprobe_reads_inherited_fd_for_synthetic_clip(self) -> None:
        subprocess.run(
            (
                "/usr/bin/ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x90:r=5",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(self.clip),
            ),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        self.clip.chmod(0o600)
        admission = self._load(duration_probe=None)
        self.assertEqual(1, admission.source_event_count)
        self.assertGreaterEqual(admission.candidates[0].media_duration_seconds, 0.9)
        self.assertLessEqual(admission.candidates[0].media_duration_seconds, 1.1)

    def test_empty_day_reads_no_media_and_returns_no_candidates(self) -> None:
        admission = self._load(local_date=date(2026, 8, 20))
        self.assertEqual((), admission.candidates)
        self.assertEqual(0, admission.media_bytes_hashed)
        self.assertFalse(admission.to_public_dict()["media_read"])
        self.assertEqual(
            date(2026, 8, 19),
            latest_daily_digest_local_date(
                database_path=self.database_path,
                timezone_name="UTC",
            ),
        )

    def test_outside_root_weak_mode_and_symlink_fail_closed(self) -> None:
        other = self.root / "other-storage"
        other.mkdir(mode=0o700)
        with self.assertRaisesRegex(DailyDigestSourceError, "source admission failed"):
            self._load(storage_root=other)

        self.clip.chmod(0o644)
        with self.assertRaisesRegex(DailyDigestSourceError, "source admission failed"):
            self._load()
        self.clip.chmod(0o600)

        target = self.clip.with_suffix(".target.mp4")
        self.clip.replace(target)
        self.clip.symlink_to(target)
        with self.assertRaisesRegex(DailyDigestSourceError, "source admission failed"):
            self._load()

    def test_invalid_duration_and_tampered_candidate_fail_closed(self) -> None:
        with self.assertRaisesRegex(DailyDigestSourceError, "source admission failed"):
            self._load(duration_probe=lambda _descriptor: float("nan"))

        connection = self.database._connect()  # test-only corruption fixture
        with connection:
            connection.execute(
                "UPDATE events SET candidate_json = ?",
                ('{"event_id":"wrong"}',),
            )
        connection.close()
        self.database_path.chmod(0o600)
        with self.assertRaisesRegex(DailyDigestSourceError, "source admission failed"):
            self._load()


if __name__ == "__main__":
    unittest.main()
