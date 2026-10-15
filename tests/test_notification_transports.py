from __future__ import annotations

import io
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from baby_monitor.adapters.vlm import MockVLMAdapter
from baby_monitor.cli import main
from baby_monitor.database import EventDatabase, StoredEvent
from baby_monitor.domain import CandidateEvent, EventMedia
from baby_monitor.notification import NotificationMessage, NotificationOutbox
from baby_monitor.notification_transports import (
    SMTPNotificationProfile,
    SMTPNotificationSink,
    WebhookNotificationProfile,
    WebhookNotificationSink,
    load_notification_transport_profile,
)
from baby_monitor.pipeline import process_prepared_event
from baby_monitor.policy import ReviewPolicy


def _store_event(root: Path, event_id: str) -> tuple[EventDatabase, StoredEvent]:
    frame = root / f"PRIVATE_FRAME_{event_id}.jpg"
    clip = root / f"PRIVATE_CLIP_{event_id}.mp4"
    frame.write_bytes(b"fixture")
    clip.write_bytes(b"fixture")
    database = EventDatabase(root / "events.sqlite3")
    process_prepared_event(
        candidate=CandidateEvent(event_id, "camera_01", 0.0, 1.0, ()),
        occurred_at=datetime(2026, 8, 11, tzinfo=UTC),
        source_kind="fixture",
        source_ref="PRIVATE_SOURCE_REF_DO_NOT_SEND",
        media=EventMedia(str(clip), (str(frame),)),
        local_vlm=MockVLMAdapter(),
        database=database,
        policy=ReviewPolicy(confidence_threshold=0.0),
    )
    event = database.get_event(event_id)
    assert event is not None
    return database, event


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)


class _FakeSMTPClient:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def send_message(self, message: Any) -> None:
        self.messages.append(message)


class _FakeResponse:
    status = 204

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def read(self, size: int) -> bytes:
        del size
        return b""


class _FakeOpener:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def open(self, request: Any, *, timeout: int) -> _FakeResponse:
        self.requests.append((request, timeout))
        return _FakeResponse()


class NotificationTransportTests(unittest.TestCase):
    def test_private_profiles_preview_security_without_disclosing_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            smtp_path = root / "smtp.json"
            _write_private_json(
                smtp_path,
                {
                    "version": 1,
                    "kind": "smtp",
                    "sink_id": "family_email",
                    "host": "smtp.example.invalid",
                    "port": 465,
                    "security": "tls",
                    "username": "sender@example.invalid",
                    "password": "PRIVATE_SMTP_PASSWORD",
                    "from_address": "sender@example.invalid",
                    "to_addresses": ["recipient@example.invalid"],
                    "timeout_seconds": 15,
                },
            )
            profile = load_notification_transport_profile(smtp_path)
            self.assertIsInstance(profile, SMTPNotificationProfile)
            serialized_preview = json.dumps(profile.to_preview())
            self.assertNotIn("PRIVATE_SMTP_PASSWORD", serialized_preview)
            self.assertNotIn("recipient@", serialized_preview)
            self.assertEqual(1, profile.to_preview()["recipient_count"])
            self.assertFalse(profile.to_preview()["network_used"])

            webhook_path = root / "webhook.json"
            _write_private_json(
                webhook_path,
                {
                    "version": 1,
                    "kind": "webhook",
                    "sink_id": "family_chat",
                    "url": "https://hooks.example.invalid/PRIVATE_WEBHOOK_TOKEN",
                    "headers": {"Authorization": "Bearer PRIVATE_HEADER_TOKEN"},
                    "timeout_seconds": 15,
                },
            )
            webhook = load_notification_transport_profile(webhook_path)
            self.assertIsInstance(webhook, WebhookNotificationProfile)
            serialized_preview = json.dumps(webhook.to_preview())
            self.assertNotIn("PRIVATE_WEBHOOK_TOKEN", serialized_preview)
            self.assertNotIn("PRIVATE_HEADER_TOKEN", serialized_preview)
            self.assertEqual("hooks.example.invalid", webhook.to_preview()["target_host"])
            self.assertFalse(webhook.to_preview()["redirects_allowed"])

            if os.name != "nt":
                webhook_path.chmod(0o644)
                with self.assertRaisesRegex(ValueError, "group or others"):
                    load_notification_transport_profile(webhook_path)

    def test_public_plaintext_targets_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            SMTPNotificationProfile(
                sink_id="family_email",
                host="smtp.example.invalid",
                port=25,
                security="loopback_plain",
                username="",
                credential="",
                from_address="sender@example.invalid",
                to_addresses=("recipient@example.invalid",),
                timeout_seconds=15,
            )
        with self.assertRaisesRegex(ValueError, "loopback"):
            WebhookNotificationProfile(
                sink_id="family_chat",
                url="http://hooks.example.invalid/private",
                headers=(),
                timeout_seconds=15,
            )

    def test_smtp_and_webhook_sinks_send_path_free_messages_through_fakes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="PRIVATE_NOTIFY_TRANSPORT_") as directory:
            root = Path(directory)
            _, event = _store_event(root, "transport_event")

            smtp_message = NotificationMessage.from_event(
                event,
                sink_kind="smtp",
                sink_id="family_email",
            )
            smtp_profile = SMTPNotificationProfile(
                sink_id="family_email",
                host="smtp.example.invalid",
                port=465,
                security="tls",
                username="sender@example.invalid",
                credential="test",
                from_address="sender@example.invalid",
                to_addresses=("recipient@example.invalid",),
                timeout_seconds=15,
            )
            fake_client = _FakeSMTPClient()

            @contextmanager
            def fake_factory(profile: SMTPNotificationProfile) -> Any:
                self.assertEqual(smtp_profile, profile)
                yield fake_client

            receipt = SMTPNotificationSink(
                smtp_profile,
                client_factory=fake_factory,
            ).deliver(smtp_message)
            self.assertEqual("smtp", receipt.sink_kind)
            self.assertEqual(1, len(fake_client.messages))
            email_text = fake_client.messages[0].as_string()
            self.assertNotIn(str(root), email_text)
            self.assertNotIn("PRIVATE_SOURCE_REF_DO_NOT_SEND", email_text)
            self.assertNotIn("PRIVATE_SMTP_PASSWORD", email_text)

            webhook_message = NotificationMessage.from_event(
                event,
                sink_kind="webhook",
                sink_id="family_chat",
            )
            webhook_profile = WebhookNotificationProfile(
                sink_id="family_chat",
                url="https://hooks.example.invalid/PRIVATE_WEBHOOK_TOKEN",
                headers=(("Authorization", "Bearer PRIVATE_HEADER_TOKEN"),),
                timeout_seconds=15,
            )
            opener = _FakeOpener()
            receipt = WebhookNotificationSink(webhook_profile, opener=opener).deliver(
                webhook_message
            )
            self.assertEqual("webhook", receipt.sink_kind)
            request, timeout = opener.requests[0]
            self.assertEqual(15, timeout)
            body = request.data.decode("utf-8")
            self.assertNotIn(str(root), body)
            self.assertNotIn("PRIVATE_SOURCE_REF_DO_NOT_SEND", body)
            self.assertNotIn("PRIVATE_HEADER_TOKEN", body)
            self.assertEqual(
                webhook_message.notification_id,
                request.get_header("Idempotency-key"),
            )

    def test_old_local_outbox_schema_migrates_and_accepts_webhook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database, event = _store_event(root, "migration_event")
            local_message = NotificationMessage.from_event(
                event,
                sink_kind="stdout",
                sink_id="local_stdout",
            )
            queued_at = datetime(2026, 8, 11, tzinfo=UTC).isoformat()
            with sqlite3.connect(database.path) as connection, connection:
                connection.executescript(
                    """
                    CREATE TABLE notification_jobs (
                        notification_id TEXT PRIMARY KEY,
                        event_id TEXT NOT NULL,
                        sink_kind TEXT NOT NULL CHECK (sink_kind IN ('stdout', 'file')),
                        sink_id TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('pending', 'running', 'delivered', 'failed')
                        ),
                        attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
                        available_at TEXT NOT NULL,
                        lease_until TEXT,
                        last_error_code TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        delivered_at TEXT,
                        UNIQUE (event_id, sink_kind, sink_id)
                    );
                    CREATE INDEX idx_notification_jobs_status_available
                        ON notification_jobs (
                            sink_kind, sink_id, status, available_at, created_at
                        );
                    """
                )
                connection.execute(
                    """
                    INSERT INTO notification_jobs (
                        notification_id, event_id, sink_kind, sink_id, payload_json,
                        status, attempt_count, available_at, lease_until, last_error_code,
                        created_at, updated_at, delivered_at
                    ) VALUES (?, ?, 'stdout', 'local_stdout', ?, 'pending', 0, ?, NULL,
                              NULL, ?, ?, NULL)
                    """,
                    (
                        local_message.notification_id,
                        event.event_id,
                        local_message.to_json(indent=None),
                        queued_at,
                        queued_at,
                        queued_at,
                    ),
                )
            queued = NotificationOutbox(database.path).enqueue(
                event=event,
                sink_kind="webhook",
                sink_id="family_chat",
                max_active=4,
            )
            self.assertEqual("webhook", queued.job.sink_kind)
            with sqlite3.connect(database.path) as connection:
                schema = connection.execute(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'notification_jobs'"
                ).fetchone()[0]
                rows = connection.execute(
                    "SELECT sink_kind, sink_id FROM notification_jobs ORDER BY sink_kind"
                ).fetchall()
            self.assertIn("'smtp'", schema)
            self.assertIn("'webhook'", schema)
            self.assertEqual(
                [("stdout", "local_stdout"), ("webhook", "family_chat")],
                rows,
            )

    def test_cli_external_switch_and_redacted_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            database, event = _store_event(root, "cli_external_event")
            profile_path = root / "webhook.json"
            _write_private_json(
                profile_path,
                {
                    "version": 1,
                    "kind": "webhook",
                    "sink_id": "family_chat",
                    "url": "https://hooks.example.invalid/PRIVATE_WEBHOOK_TOKEN",
                    "headers": {"Authorization": "Bearer PRIVATE_HEADER_TOKEN"},
                    "timeout_seconds": 15,
                },
            )
            preview_output = io.StringIO()
            with redirect_stdout(preview_output):
                code = main(["notification-transport-preview", "--profile", str(profile_path)])
            self.assertEqual(0, code)
            self.assertFalse(json.loads(preview_output.getvalue())["network_used"])
            self.assertNotIn("PRIVATE_WEBHOOK_TOKEN", preview_output.getvalue())
            self.assertNotIn("PRIVATE_HEADER_TOKEN", preview_output.getvalue())

            config = root / "notification.toml"
            config.write_text("[notification]\nenabled = true\n", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "notification-enqueue-event",
                        "--config",
                        str(config),
                        "--database",
                        str(database.path),
                        "--event-id",
                        event.event_id,
                        "--sink",
                        "webhook",
                        "--sink-id",
                        "family_chat",
                    ]
                )
            self.assertEqual(2, code)
            self.assertIn("external notification delivery is disabled", stderr.getvalue())

            config.write_text(
                "[notification]\nenabled = true\nexternal_delivery_enabled = true\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(
                    [
                        "notification-enqueue-event",
                        "--config",
                        str(config),
                        "--database",
                        str(database.path),
                        "--event-id",
                        event.event_id,
                        "--sink",
                        "webhook",
                        "--sink-id",
                        "family_chat",
                    ]
                )
            self.assertEqual(0, code)
            self.assertEqual("webhook", json.loads(output.getvalue())["sink_kind"])


if __name__ == "__main__":
    unittest.main()
