from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from baby_monitor.room_registry import (
    LogicalStreamDefinition,
    PhysicalCameraDefinition,
    RoomDefinition,
    RoomRegistry,
    RoomRegistryError,
)


def fixture_registry() -> RoomRegistry:
    return RoomRegistry(
        rooms=(
            RoomDefinition(room_id="room_a", display_name="房间 A"),
            RoomDefinition(room_id="room_b", display_name="房间 B"),
            RoomDefinition(room_id="room_c", display_name="房间 C"),
        ),
        physical_cameras=(
            PhysicalCameraDefinition(
                physical_camera_id="physical_01",
                display_name="双通道摄像头",
            ),
            PhysicalCameraDefinition(
                physical_camera_id="physical_02",
                display_name="摄像头 2",
            ),
            PhysicalCameraDefinition(
                physical_camera_id="physical_03",
                display_name="摄像头 3",
            ),
        ),
        logical_streams=(
            LogicalStreamDefinition(
                camera_id="camera_01",
                physical_camera_id="physical_01",
                room_id="room_a",
                display_name="视角 1",
            ),
            LogicalStreamDefinition(
                camera_id="camera_02",
                physical_camera_id="physical_01",
                room_id="room_b",
                display_name="视角 2",
            ),
            LogicalStreamDefinition(
                camera_id="camera_03",
                physical_camera_id="physical_02",
                room_id="room_b",
                display_name="主视角",
            ),
            LogicalStreamDefinition(
                camera_id="camera_04",
                physical_camera_id="physical_03",
                room_id="room_c",
                display_name="主视角",
            ),
        ),
    )


class RoomRegistryTests(unittest.TestCase):
    def test_one_physical_camera_can_cover_two_rooms_without_becoming_two_devices(self) -> None:
        registry = fixture_registry()

        self.assertEqual("room_a", registry.room_for_stream("camera_01").room_id)
        self.assertEqual("room_b", registry.room_for_stream("camera_02").room_id)
        self.assertEqual(
            ("camera_02", "camera_03"),
            tuple(item.camera_id for item in registry.streams_for_room("room_b")),
        )
        self.assertEqual(registry, RoomRegistry.from_dict(registry.to_dict()))
        self.assertEqual(2, registry.version)
        self.assertEqual(64, len(registry.fingerprint))

        summary = registry.summary()
        self.assertEqual(3, summary["room_count"])
        self.assertEqual(3, summary["physical_camera_count"])
        self.assertEqual(4, summary["logical_stream_count"])
        self.assertEqual(1, summary["multi_stream_physical_camera_count"])
        self.assertEqual(1, summary["multi_room_physical_camera_count"])
        self.assertEqual(3, len(summary["rooms"]))
        self.assertEqual(2, summary["rooms"][1]["physical_camera_count"])
        self.assertFalse(summary["media_read"])
        self.assertFalse(summary["identity_inference"])

    def test_v1_registry_migrates_to_canonical_stream_level_rooms(self) -> None:
        legacy = {
            "version": 1,
            "rooms": [item.to_dict() for item in fixture_registry().rooms],
            "physical_cameras": [
                {
                    "physical_camera_id": "physical_01",
                    "room_id": "room_a",
                    "display_name": "双通道摄像头",
                },
                {
                    "physical_camera_id": "physical_02",
                    "room_id": "room_b",
                    "display_name": "摄像头 2",
                },
                {
                    "physical_camera_id": "physical_03",
                    "room_id": "room_c",
                    "display_name": "摄像头 3",
                },
            ],
            "logical_streams": [
                {
                    "camera_id": "camera_01",
                    "physical_camera_id": "physical_01",
                    "display_name": "视角 1",
                },
                {
                    "camera_id": "camera_02",
                    "physical_camera_id": "physical_01",
                    "display_name": "视角 2",
                },
                {
                    "camera_id": "camera_03",
                    "physical_camera_id": "physical_02",
                    "display_name": "主视角",
                },
                {
                    "camera_id": "camera_04",
                    "physical_camera_id": "physical_03",
                    "display_name": "主视角",
                },
            ],
        }

        migrated = RoomRegistry.from_dict(legacy)

        self.assertEqual(2, migrated.version)
        self.assertEqual("room_a", migrated.room_for_stream("camera_01").room_id)
        self.assertEqual("room_a", migrated.room_for_stream("camera_02").room_id)
        self.assertNotIn("room_id", migrated.to_dict()["physical_cameras"][0])
        self.assertEqual("room_a", migrated.to_dict()["logical_streams"][1]["room_id"])

    def test_duplicate_or_orphan_topology_fails_closed(self) -> None:
        payload = fixture_registry().to_dict()
        payload["logical_streams"][1]["camera_id"] = "camera_01"  # type: ignore[index]
        with self.assertRaisesRegex(RoomRegistryError, "camera_id values must be unique"):
            RoomRegistry.from_dict(payload)

        payload = fixture_registry().to_dict()
        payload["logical_streams"][0]["physical_camera_id"] = "missing"  # type: ignore[index]
        with self.assertRaisesRegex(RoomRegistryError, "unknown physical camera"):
            RoomRegistry.from_dict(payload)

        payload = fixture_registry().to_dict()
        payload["logical_streams"][0]["room_id"] = "missing"  # type: ignore[index]
        with self.assertRaisesRegex(RoomRegistryError, "unknown room"):
            RoomRegistry.from_dict(payload)

        payload = fixture_registry().to_dict()
        payload["physical_cameras"] = payload["physical_cameras"][:-1]  # type: ignore[index]
        with self.assertRaisesRegex(RoomRegistryError, "unknown physical camera"):
            RoomRegistry.from_dict(payload)

        payload = fixture_registry().to_dict()
        payload["logical_streams"] = payload["logical_streams"][:-1]  # type: ignore[index]
        with self.assertRaisesRegex(RoomRegistryError, "every physical camera"):
            RoomRegistry.from_dict(payload)

        payload = fixture_registry().to_dict()
        payload["rooms"] = payload["rooms"][:-1]  # type: ignore[index]
        with self.assertRaisesRegex(RoomRegistryError, "unknown room"):
            RoomRegistry.from_dict(payload)

    def test_labels_and_exact_contract_are_strict(self) -> None:
        payload = fixture_registry().to_dict()
        payload["rooms"][0]["display_name"] = " 房间 A"  # type: ignore[index]
        with self.assertRaisesRegex(RoomRegistryError, "display_name"):
            RoomRegistry.from_dict(payload)

        payload = fixture_registry().to_dict()
        payload["rooms"][0]["display_name"] = "房间\u200bA"  # type: ignore[index]
        with self.assertRaisesRegex(RoomRegistryError, "display_name"):
            RoomRegistry.from_dict(payload)

        payload = fixture_registry().to_dict()
        payload["unexpected"] = True
        with self.assertRaisesRegex(RoomRegistryError, "fields do not match"):
            RoomRegistry.from_dict(payload)

        schema = json.loads(
            Path("schemas/room-registry-v2.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(schema["required"]), set(fixture_registry().to_dict()))
        example = json.loads(Path("config/room-registry.example.json").read_text(encoding="utf-8"))
        self.assertEqual(
            (3, 3, 4),
            (
                len(example["rooms"]),
                len(example["physical_cameras"]),
                len(example["logical_streams"]),
            ),
        )
        self.assertEqual(fixture_registry(), RoomRegistry.from_dict(example))

    @unittest.skipUnless(os.name == "posix", "POSIX permission contract")
    def test_private_reader_rejects_weak_permissions_symlink_and_hardlink(self) -> None:
        registry = fixture_registry()
        with tempfile.TemporaryDirectory(prefix="baby-monitor-rooms-") as directory:
            root = Path(directory).resolve()
            root.chmod(0o700)
            path = root / "rooms.json"
            path.write_text(json.dumps(registry.to_dict(), ensure_ascii=False), encoding="utf-8")
            path.chmod(0o600)

            self.assertEqual(registry, RoomRegistry.read_private(path))

            path.chmod(0o644)
            with self.assertRaisesRegex(RoomRegistryError, "permissions are unsafe"):
                RoomRegistry.read_private(path)
            path.chmod(0o600)

            link = root / "rooms-link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(RoomRegistryError, "missing or unsafe"):
                RoomRegistry.read_private(link)

            hardlink = root / "rooms-hardlink.json"
            os.link(path, hardlink)
            with self.assertRaisesRegex(RoomRegistryError, "missing or unsafe"):
                RoomRegistry.read_private(path)


if __name__ == "__main__":
    unittest.main()
