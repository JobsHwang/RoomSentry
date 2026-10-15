from __future__ import annotations

import stat
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from baby_monitor.config import StorageSettings
from baby_monitor.storage import StorageResolver


class StorageTests(unittest.TestCase):
    def test_missing_nas_falls_back_to_local(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            missing_nas = root / "not-mounted"
            resolver = StorageResolver(
                StorageSettings(
                    local_root=local,
                    nas_root=missing_nas,
                    prefer_nas=True,
                )
            )
            allocated = resolver.allocate_media(
                occurred_at=datetime(2026, 8, 1, 0, 0, tzinfo=UTC),
                camera_id="nursery cam/01",
                event_id="event:123",
                frame_count=5,
            )
            self.assertTrue(allocated.clip_path.is_relative_to(local))
            self.assertIn("nursery_cam_01", allocated.clip_path.name)
            self.assertIn("event_123", allocated.clip_path.name)
            self.assertFalse(missing_nas.exists())
            current = allocated.clip_path.parent
            while current != local.parent:
                self.assertEqual(0o700, stat.S_IMODE(current.stat().st_mode))
                current = current.parent

    def test_existing_weak_directories_are_tightened_and_symlinks_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / "local"
            events = local / "events"
            events.mkdir(parents=True)
            events.chmod(0o755)
            resolver = StorageResolver(StorageSettings(local_root=local))
            allocated = resolver.allocate_media(
                occurred_at=datetime(2026, 8, 1, 0, 0, tzinfo=UTC),
                camera_id="cam_01",
                event_id="event_01",
                frame_count=1,
            )
            self.assertEqual(0o700, stat.S_IMODE(events.stat().st_mode))
            self.assertEqual(0o700, stat.S_IMODE(allocated.clip_path.parent.stat().st_mode))

            unsafe_root = root / "unsafe"
            unsafe_root.mkdir()
            (unsafe_root / "events").symlink_to(events, target_is_directory=True)
            unsafe = StorageResolver(StorageSettings(local_root=unsafe_root))
            with self.assertRaisesRegex(ValueError, "private storage"):
                unsafe.allocate_media(
                    occurred_at=datetime(2026, 8, 1, 0, 0, tzinfo=UTC),
                    camera_id="cam_01",
                    event_id="event_01",
                    frame_count=1,
                )


if __name__ == "__main__":
    unittest.main()
