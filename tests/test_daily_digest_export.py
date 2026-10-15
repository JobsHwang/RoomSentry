from __future__ import annotations

import json
import stat
import subprocess
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

from baby_monitor.adapters.vlm import MockVLMAdapter
from baby_monitor.daily_digest import DailyDigestConfig, plan_daily_digest
from baby_monitor.daily_digest_export import (
    DailyDigestCatalog,
    DailyDigestExportError,
    export_daily_digest,
    inspect_daily_digest_exports,
    read_latest_daily_digest_video,
)
from baby_monitor.daily_digest_source import load_daily_digest_candidates
from baby_monitor.daily_digest_store import write_daily_digest_plan
from baby_monitor.database import EventDatabase
from baby_monitor.domain import CandidateEvent, Detection, EventMedia
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy


@unittest.skipUnless(
    Path("/usr/bin/ffmpeg").is_file() and Path("/usr/bin/ffprobe").is_file(),
    "FFmpeg tools are required",
)
class DailyDigestExportTests(unittest.TestCase):
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
        self.clip = media_directory / "private_source.mp4"
        subprocess.run(
            (
                "/usr/bin/ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=160x90:rate=10",
                "-t",
                "3",
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
            timeout=20,
        )
        self.clip.chmod(0o600)
        frame = media_directory / "private_frame.jpg"
        frame.write_bytes(b"synthetic-frame")
        frame.chmod(0o600)
        self.database_path = self.state / "events.sqlite3"
        database = EventDatabase(self.database_path)
        occurred_at = datetime(2026, 8, 19, 1, tzinfo=UTC)
        candidate = CandidateEvent(
            event_id="event_01",
            camera_id="camera_01",
            start_seconds=0.0,
            end_seconds=3.0,
            detections=(Detection(1.0, "person", 0.8, (0.1, 0.1, 0.8, 0.9)),),
            detector_id="fixture_detector",
        )
        process_prepared_event(
            candidate=candidate,
            occurred_at=occurred_at,
            source_kind="fixture",
            source_ref="fixture",
            media=EventMedia(str(self.clip), (str(frame),)),
            local_vlm=MockVLMAdapter(),
            database=database,
            policy=ReviewPolicy(),
        )
        self.database_path.chmod(0o600)
        self.admission = load_daily_digest_candidates(
            database_path=self.database_path,
            storage_root=self.storage,
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
        )
        self.plan = plan_daily_digest(
            self.admission.candidates,
            local_date=date(2026, 8, 19),
            timezone_name="UTC",
            config=DailyDigestConfig(target_duration_seconds=2.0),
        )
        self.output_root = self.state / "daily-digest-exports"

    def test_real_export_is_private_playable_and_idempotent(self) -> None:
        before_database_mtime = self.database_path.stat().st_mtime_ns
        receipt = export_daily_digest(
            self.plan,
            self.admission,
            database_path=self.database_path,
            output_root=self.output_root,
            storage_root=self.storage,
        )
        self.assertFalse(receipt.reused_existing)
        self.assertEqual(1, receipt.selected_event_count)
        self.assertAlmostEqual(2.0, receipt.output_duration_seconds, delta=0.3)
        self.assertEqual(before_database_mtime, self.database_path.stat().st_mtime_ns)
        bundles = list(self.output_root.iterdir())
        self.assertEqual(1, len(bundles))
        self.assertEqual(0o700, stat.S_IMODE(bundles[0].stat().st_mode))
        video = bundles[0] / "digest.mp4"
        private_receipt = bundles[0] / "receipt.json"
        self.assertEqual(0o600, stat.S_IMODE(video.stat().st_mode))
        self.assertEqual(0o600, stat.S_IMODE(private_receipt.stat().st_mode))
        schema = json.loads(
            (
                Path(__file__).parents[1] / "schemas/daily-digest-export-receipt-v1.schema.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(json.loads(private_receipt.read_text())))
        probe = subprocess.run(
            (
                "/usr/bin/ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertAlmostEqual(2.0, float(probe.stdout), delta=0.3)
        public = receipt.to_public_dict()
        self.assertTrue(public["media_content_decoded"])
        self.assertTrue(public["video_generated"])
        rendered = json.dumps(public)
        self.assertNotIn(str(self.root), rendered)
        self.assertNotIn("event_01", rendered)

        reused = export_daily_digest(
            self.plan,
            self.admission,
            database_path=self.database_path,
            output_root=self.output_root,
            storage_root=self.storage,
        )
        self.assertTrue(reused.reused_existing)
        self.assertEqual(1, len(list(self.output_root.iterdir())))

        status = inspect_daily_digest_exports(
            database_path=self.database_path,
            output_root=self.output_root,
        )
        self.assertTrue(status["video_available"])
        self.assertFalse(status["media_read"])
        self.assertNotIn(str(self.root), json.dumps(status))
        self.assertEqual(
            video.read_bytes(),
            read_latest_daily_digest_video(
                database_path=self.database_path,
                output_root=self.output_root,
            ),
        )
        catalog = DailyDigestCatalog(
            database_path=self.database_path,
            plan_root=self.state / "daily-digests",
            export_root=self.output_root,
            token_secret=b"catalog-secret" * 4,
        )
        # The catalog requires the exact private plan that the export receipt binds.
        with self.assertRaisesRegex(DailyDigestExportError, "daily digest export failed"):
            catalog.list_public()
        write_daily_digest_plan(
            self.plan,
            database_path=self.database_path,
            output_root=self.state / "daily-digests",
        )
        entries = catalog.list_public()
        self.assertEqual(1, len(entries))
        entry = entries[0]
        self.assertEqual(43, len(entry["review_token"]))
        self.assertEqual(1, entry["selected_event_count"])
        self.assertEqual(1, len(entry["items"]))
        self.assertIn("event_recorded", entry["items"][0]["selection_reasons"])
        rendered_catalog = json.dumps(entries)
        self.assertNotIn(str(self.root), rendered_catalog)
        self.assertNotIn("event_01", rendered_catalog)
        self.assertNotIn("camera_01", rendered_catalog)
        self.assertNotIn(self.plan.digest_id, rendered_catalog)
        self.assertEqual(video.read_bytes(), catalog.read_video(entry["review_token"]))
        with self.assertRaisesRegex(DailyDigestExportError, "daily digest export failed"):
            catalog.read_video("A" * 43)
        video.write_bytes(b"x" * video.stat().st_size)
        video.chmod(0o600)
        with self.assertRaisesRegex(DailyDigestExportError, "daily digest export failed"):
            catalog.read_video(entry["review_token"])

    def test_changed_source_and_incomplete_bundle_fail_closed(self) -> None:
        outside = self.root / "outside.mp4"
        shutil_source = self.clip.read_bytes()
        outside.write_bytes(shutil_source)
        outside.chmod(0o600)
        forged = replace(
            self.admission,
            media_sources=(replace(self.admission.media_sources[0], clip_path=outside),),
        )
        with self.assertRaisesRegex(DailyDigestExportError, "daily digest export failed"):
            export_daily_digest(
                self.plan,
                forged,
                database_path=self.database_path,
                output_root=self.output_root,
                storage_root=self.storage,
            )

        self.clip.write_bytes(b"changed-after-admission")
        self.clip.chmod(0o600)
        with self.assertRaisesRegex(DailyDigestExportError, "daily digest export failed"):
            export_daily_digest(
                self.plan,
                self.admission,
                database_path=self.database_path,
                output_root=self.output_root,
                storage_root=self.storage,
            )
        self.assertFalse(self.output_root.exists() and any(self.output_root.iterdir()))

        self.output_root.mkdir(mode=0o700, exist_ok=True)
        incomplete = self.output_root / f"{self.plan.local_date}_{self.plan.digest_id}"
        incomplete.mkdir(mode=0o700)
        status = inspect_daily_digest_exports(
            database_path=self.database_path,
            output_root=self.output_root,
        )
        self.assertTrue(status["attention_required"])
        self.assertFalse(status["video_available"])

    def test_duplicate_key_receipt_is_rejected(self) -> None:
        export_daily_digest(
            self.plan,
            self.admission,
            database_path=self.database_path,
            output_root=self.output_root,
            storage_root=self.storage,
        )
        receipt_path = next(self.output_root.iterdir()) / "receipt.json"
        receipt_path.write_text('{"version":1,"version":1}\n', encoding="utf-8")
        receipt_path.chmod(0o600)
        status = inspect_daily_digest_exports(
            database_path=self.database_path,
            output_root=self.output_root,
        )
        self.assertEqual("unavailable", status["status"])
        with self.assertRaisesRegex(DailyDigestExportError, "daily digest export failed"):
            export_daily_digest(
                self.plan,
                self.admission,
                database_path=self.database_path,
                output_root=self.output_root,
                storage_root=self.storage,
            )


if __name__ == "__main__":
    unittest.main()
