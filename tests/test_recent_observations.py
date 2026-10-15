from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from baby_monitor.cli import build_parser
from baby_monitor.cli_commands.recent_observations import (
    command_recent_observations,
    render_recent_observations_text,
)
from baby_monitor.recent_observations import read_recent_observations
from baby_monitor.room_registry import (
    LogicalStreamDefinition,
    PhysicalCameraDefinition,
    RoomDefinition,
    RoomRegistry,
)


class RecentObservationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "events.sqlite3"
        self.now = datetime(2026, 9, 16, 0, 0, tzinfo=UTC)
        self.registry = RoomRegistry(
            (RoomDefinition("room_01", "测试房间"),),
            (PhysicalCameraDefinition("physical_01", "测试设备"),),
            (
                LogicalStreamDefinition("camera_01", "physical_01", "room_01", "视角一"),
                LogicalStreamDefinition("camera_02", "physical_01", "room_01", "视角二"),
            ),
        )
        self.payload = json.loads(
            (Path(__file__).parent / "fixtures/valid_vlm_event.json").read_text()
        )
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "CREATE TABLE events (event_id TEXT, camera_id TEXT, occurred_at TEXT, "
                "result_json TEXT, source_ref TEXT)"
            )
        self.database.chmod(0o600)

    def insert(self, *, camera="camera_01", age=30, payload=None, event_id="event_01"):
        with sqlite3.connect(self.database) as connection:
            connection.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                (
                    event_id,
                    camera,
                    (self.now - timedelta(seconds=age)).isoformat(),
                    json.dumps(self.payload) if payload is None else payload,
                    "private-source-must-not-be-read",
                ),
            )

    def read(self, **kwargs):
        return read_recent_observations(self.database, self.registry, now=self.now, **kwargs)

    def test_recent_description_not_live_baby_state(self):
        self.insert()
        result = self.read()
        entry = result["entries"][0]
        self.assertEqual("recent_record", entry["status"])
        self.assertEqual(30, entry["age_seconds"])
        self.assertEqual("测试房间", entry["mapped_room"])
        self.assertEqual(self.payload["observations"], entry["model_reported"]["observations"])
        self.assertEqual(self.payload["inferences"], entry["model_reported"]["inferences"])
        self.assertEqual("unknown", result["baby_location"])
        self.assertEqual("unknown", result["baby_state"])
        self.assertFalse(result["live_state_established"])
        self.assertNotIn("private-source", json.dumps(result))
        self.assertNotIn("frame_02", json.dumps(result))

    def test_empty_database_preserves_both_views(self):
        self.assertEqual(["no_record", "no_record"], [e["status"] for e in self.read()["entries"]])

    def test_stale_and_boundary(self):
        self.insert(age=120)
        self.insert(camera="camera_02", age=121)
        self.assertEqual(
            ["recent_record", "stale_record"], [e["status"] for e in self.read()["entries"]]
        )

    def test_latest_per_camera_not_global_latest(self):
        self.insert(age=300, event_id="old")
        self.insert(age=10, event_id="new")
        self.insert(camera="camera_02", age=20)
        self.insert(camera="unmapped", age=0)
        self.assertEqual([10, 20], [e["age_seconds"] for e in self.read()["entries"]])

    def test_invalid_latest_does_not_fallback_to_old(self):
        self.insert(age=300, event_id="old")
        self.insert(age=0, payload="bad", event_id="new")
        self.assertEqual("invalid_record", self.read()["entries"][0]["status"])

    def test_future_not_current(self):
        self.insert(age=-1)
        self.assertEqual("invalid_record", self.read()["entries"][0]["status"])

    def test_invalid_payloads(self):
        for payload in ("{}", "null", "x" * 65537, '{"x":1,"x":2}', "{", b"blob"):
            with self.subTest(payload_type=type(payload)):
                with sqlite3.connect(self.database) as connection:
                    connection.execute("DELETE FROM events")
                self.insert(payload=payload)
                self.assertEqual("invalid_record", self.read()["entries"][0]["status"])

    def test_read_only(self):
        self.insert()
        original = self.database.read_bytes()
        self.read()
        self.assertEqual(original, self.database.read_bytes())

    def test_missing_does_not_create_database(self):
        path = Path(self.temp.name) / "absent.sqlite3"
        with self.assertRaises(ValueError):
            read_recent_observations(path, self.registry)
        self.assertFalse(path.exists())

    def test_invalid_window(self):
        for value in (True, 0, 3601, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read(recent_seconds=value)

    def test_cli_parser(self):
        args = build_parser().parse_args(
            ["recent-observations", "--database", "db", "--registry", "rooms"]
        )
        self.assertEqual(120, args.recent_seconds)
        self.assertEqual("json", args.format)

    def test_text_does_not_promote_sleeping_label_to_current_state(self):
        self.insert()
        output = render_recent_observations_text(self.read())
        self.assertIn("宝宝当前状态：无法确认", output)
        self.assertIn("近期历史记录，仍不是实时状态", output)
        self.assertIn("当时的模型推测（不是事实）", output)
        self.assertIn("未独立验证", output)
        self.assertNotIn("宝宝当前状态：睡", output)
        self.assertIn("未判断未知是否由遮挡或断流造成", output)

    def test_text_distinguishes_no_record_invalid_and_stale(self):
        self.assertIn("暂无记录", render_recent_observations_text(self.read()))
        self.insert(age=121)
        self.insert(camera="camera_02", payload="bad")
        output = render_recent_observations_text(self.read())
        self.assertIn("记录已过期，不能代表现在", output)
        self.assertIn("最新记录不可用", output)

    def test_text_removes_terminal_and_bidi_control_characters(self):
        self.insert()
        payload = self.read()
        payload["entries"][0]["model_reported"]["observations"] = [
            "可见\x1b[2J\r\n宝宝当前状态：睡着\u202e"
        ]
        output = render_recent_observations_text(payload)
        for control in ("\x1b", "\r", "\u202e"):
            self.assertNotIn(control, output)
        self.assertNotIn("\n宝宝当前状态：睡着", output)

    def test_cli_text_and_json_are_read_only_views_of_same_result(self):
        self.insert()
        payload = self.read()
        for format_name in ("json", "text"):
            args = build_parser().parse_args(
                [
                    "recent-observations",
                    "--database",
                    "db",
                    "--registry",
                    "rooms",
                    "--format",
                    format_name,
                ]
            )
            output = io.StringIO()
            with (
                patch(
                    "baby_monitor.cli_commands.recent_observations.RoomRegistry.read_private",
                    return_value=self.registry,
                ),
                patch(
                    "baby_monitor.cli_commands.recent_observations.read_recent_observations",
                    return_value=payload,
                ),
                redirect_stdout(output),
            ):
                self.assertEqual(0, command_recent_observations(args))
            if format_name == "json":
                self.assertEqual(payload, json.loads(output.getvalue()))
            else:
                self.assertEqual(render_recent_observations_text(payload) + "\n", output.getvalue())


if __name__ == "__main__":
    unittest.main()
