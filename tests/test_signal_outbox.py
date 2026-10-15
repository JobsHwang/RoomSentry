from __future__ import annotations

import json
import os
import sqlite3
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from baby_monitor.adapters.home_assistant_live import HomeAssistantWatchResult
from baby_monitor.cli import main
from baby_monitor.domain import SignalEvent
from baby_monitor.signal_outbox import (
    SignalOutbox,
    SignalOutboxError,
    SignalOutboxFullError,
    SignalOutboxStatus,
)


def _signal(signal_id: str = "ha_fixture_01", *, confidence: float = 0.75) -> SignalEvent:
    return SignalEvent.from_dict(
        {
            "version": 1,
            "signal_id": signal_id,
            "camera_id": "cam_01",
            "occurred_at": "2026-08-18T01:02:03+00:00",
            "signal_type": "motion",
            "source_adapter": "home_assistant",
            "confidence": confidence,
            "attributes": {"channel": "device"},
        }
    )


class SignalOutboxTests(unittest.TestCase):
    def test_restart_idempotence_and_conflicting_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.sqlite3"
            first = SignalOutbox(path).enqueue(_signal())
            second = SignalOutbox(path).enqueue(_signal())

            self.assertFalse(first.reused_existing)
            self.assertTrue(second.reused_existing)
            self.assertEqual(first.outbox_id, second.outbox_id)
            self.assertEqual(SignalOutboxStatus.PENDING, second.status)
            with self.assertRaisesRegex(SignalOutboxError, "immutable"):
                SignalOutbox(path).enqueue(_signal(confidence=0.9))

            snapshot = SignalOutbox(path).snapshot_read_only()
            self.assertEqual(1, snapshot.counts.pending)
            self.assertEqual(1, snapshot.ready_pending)
            self.assertFalse(snapshot.to_public_dict()["paths_disclosed"])
            if os.name == "posix":
                self.assertEqual(0o600, stat.S_IMODE(path.stat().st_mode))

    def test_bounded_queue_rejects_new_signal_but_reuses_existing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outbox = SignalOutbox(Path(directory) / "signals.sqlite3")
            first = outbox.enqueue(_signal(), max_active=1)
            reused = outbox.enqueue(_signal(), max_active=1)
            self.assertEqual(first.outbox_id, reused.outbox_id)
            with self.assertRaisesRegex(SignalOutboxFullError, "full"):
                outbox.enqueue(_signal("ha_fixture_02"), max_active=1)
            self.assertEqual(1, outbox.snapshot_read_only().counts.active)

    def test_claim_ack_scrubs_payload_and_receipt_is_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.sqlite3"
            outbox = SignalOutbox(path)
            result = outbox.enqueue(_signal())
            claim = outbox.claim_next(lease_seconds=30)
            self.assertIsNotNone(claim)
            assert claim is not None
            self.assertEqual(result.outbox_id, claim.outbox_id)
            self.assertEqual(_signal(), claim.signal)
            public = json.dumps(claim.to_public_dict(), ensure_ascii=False)
            self.assertNotIn(str(path), public)
            self.assertNotIn("binary_sensor", public)

            with self.assertRaisesRegex(SignalOutboxError, "stale"):
                outbox.acknowledge(claim.outbox_id, "wrong-token")
            outbox.acknowledge(claim.outbox_id, claim.lease_token)
            with self.assertRaisesRegex(SignalOutboxError, "stale"):
                outbox.acknowledge(claim.outbox_id, claim.lease_token)

            snapshot = outbox.snapshot_read_only()
            self.assertEqual(1, snapshot.counts.completed)
            connection = sqlite3.connect(path)
            try:
                row = connection.execute(
                    "SELECT signal_json, lease_token FROM signal_outbox WHERE outbox_id = ?",
                    (claim.outbox_id,),
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual((None, None), row)

    def test_failed_attempt_retries_then_stops_at_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outbox = SignalOutbox(Path(directory) / "signals.sqlite3")
            outbox.enqueue(_signal())
            first = outbox.claim_next(lease_seconds=30)
            assert first is not None
            status = outbox.record_failure(
                first.outbox_id,
                first.lease_token,
                error_code="downstream_busy",
                retry_delay_seconds=0,
                max_attempts=2,
            )
            self.assertEqual(SignalOutboxStatus.PENDING, status)

            second = outbox.claim_next(lease_seconds=30, max_attempts=2)
            assert second is not None
            status = outbox.record_failure(
                second.outbox_id,
                second.lease_token,
                error_code="downstream_busy",
                retry_delay_seconds=0,
                max_attempts=2,
            )
            self.assertEqual(SignalOutboxStatus.FAILED, status)
            self.assertEqual(1, outbox.snapshot_read_only().counts.failed)
            outbox.retry_failed(second.outbox_id)
            self.assertEqual(1, outbox.snapshot_read_only().counts.pending)

    def test_expired_lease_recovers_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.sqlite3"
            start = datetime(2026, 8, 18, 1, tzinfo=UTC)
            outbox = SignalOutbox(path)
            outbox.enqueue(_signal(), now=start)
            first = outbox.claim_next(lease_seconds=10, now=start)
            assert first is not None

            recovered = SignalOutbox(path).claim_next(
                lease_seconds=10,
                max_attempts=2,
                now=start + timedelta(seconds=11),
            )
            assert recovered is not None
            self.assertEqual(first.outbox_id, recovered.outbox_id)
            self.assertNotEqual(first.lease_token, recovered.lease_token)
            self.assertEqual(2, recovered.attempt_count)

    def test_read_only_snapshot_does_not_create_missing_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.sqlite3"
            with self.assertRaisesRegex(SignalOutboxError, "missing or invalid"):
                SignalOutbox(path).snapshot_read_only()
            self.assertFalse(path.exists())

    def test_receipt_mutations_do_not_create_a_missing_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.sqlite3"
            outbox = SignalOutbox(path)
            operations = (
                lambda: outbox.acknowledge("sig_missing", "0" * 32),
                lambda: outbox.record_failure(
                    "sig_missing",
                    "0" * 32,
                    error_code="fixture_failure",
                ),
                lambda: outbox.retry_failed("sig_missing"),
            )
            for operation in operations:
                with (
                    self.subTest(operation=operation),
                    self.assertRaisesRegex(SignalOutboxError, "missing or invalid"),
                ):
                    operation()
                self.assertFalse(path.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX mode checks are required")
    def test_weak_permissions_and_hardlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "signals.sqlite3"
            SignalOutbox(path).initialize()
            path.chmod(0o640)
            with self.assertRaisesRegex(SignalOutboxError, "permissions"):
                SignalOutbox(path).enqueue(_signal())
            path.chmod(0o600)

            link = root / "signals-hardlink.sqlite3"
            os.link(path, link)
            with self.assertRaisesRegex(SignalOutboxError, "unsafe"):
                SignalOutbox(path).enqueue(_signal())

    def test_symlink_database_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.sqlite3"
            SignalOutbox(target).initialize()
            link = root / "link.sqlite3"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlink is unavailable: {type(exc).__name__}")
            with self.assertRaisesRegex(SignalOutboxError, "unsafe"):
                SignalOutbox(link).enqueue(_signal())

    def test_watch_cli_persists_before_return_and_reuses_after_restart(self) -> None:
        signal = _signal()

        def fake_watch(*_args: object, **kwargs: object) -> HomeAssistantWatchResult:
            callback = kwargs.get("on_signal")
            self.assertTrue(callable(callback))
            callback(signal)  # type: ignore[operator]
            return HomeAssistantWatchResult(
                signals=(signal,),
                connection_attempts=1,
                successful_connections=1,
                reconnect_count=0,
                idle_timeout_count=0,
                invalid_message_count=0,
                received_message_count=1,
                stop_reason="signal_limit",
                elapsed_seconds=0.1,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.sqlite3"
            outputs: list[dict[str, object]] = []
            for _index in range(2):
                stdout = StringIO()
                with (
                    patch(
                        "baby_monitor.cli_commands.home_assistant.watch_home_assistant_signals",
                        side_effect=fake_watch,
                    ),
                    redirect_stdout(stdout),
                ):
                    code = main(
                        [
                            "watch-ha-signals",
                            "--url",
                            "http://127.0.0.1:8123",
                            "--token-file",
                            "unused-private-token-path",
                            "--mapping",
                            "config/home-assistant-signals.example.json",
                            "--outbox-database",
                            str(path),
                        ]
                    )
                self.assertEqual(0, code)
                self.assertNotIn(str(path), stdout.getvalue())
                self.assertNotIn("unused-private-token-path", stdout.getvalue())
                outputs.append(json.loads(stdout.getvalue()))

            self.assertEqual(1, outputs[0]["outbox_enqueued"])
            self.assertEqual(0, outputs[0]["outbox_reused"])
            self.assertEqual(0, outputs[1]["outbox_enqueued"])
            self.assertEqual(1, outputs[1]["outbox_reused"])
            self.assertEqual(1, SignalOutbox(path).snapshot_read_only().counts.pending)

    def test_watch_cli_can_filter_outbox_to_locally_routable_camera_ids(self) -> None:
        signal = _signal()

        def fake_watch(*_args: object, **kwargs: object) -> HomeAssistantWatchResult:
            callback = kwargs.get("on_signal")
            self.assertTrue(callable(callback))
            callback(signal)  # type: ignore[operator]
            return HomeAssistantWatchResult(
                signals=(signal,),
                connection_attempts=1,
                successful_connections=1,
                reconnect_count=0,
                idle_timeout_count=0,
                invalid_message_count=0,
                received_message_count=1,
                stop_reason="signal_limit",
                elapsed_seconds=0.1,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.sqlite3"
            stdout = StringIO()
            with (
                patch(
                    "baby_monitor.cli_commands.home_assistant.watch_home_assistant_signals",
                    side_effect=fake_watch,
                ),
                redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "watch-ha-signals",
                        "--url",
                        "http://127.0.0.1:8123",
                        "--token-file",
                        "unused-private-token-path",
                        "--mapping",
                        "config/home-assistant-signals.example.json",
                        "--outbox-database",
                        str(path),
                        "--outbox-camera-id",
                        "different_camera",
                    ]
                )

            self.assertEqual(0, code)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(0, payload["outbox_enqueued"])
            self.assertEqual(0, payload["outbox_reused"])
            self.assertEqual(1, payload["outbox_filtered"])
            self.assertFalse(path.exists())
            self.assertNotIn(str(path), stdout.getvalue())

    def test_claim_ack_and_status_cli_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.sqlite3"
            SignalOutbox(path).enqueue(_signal())
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "ha-signal-outbox-claim",
                        "--database",
                        str(path),
                        "--lease-seconds",
                        "30",
                    ]
                )
            self.assertEqual(0, code)
            claimed = json.loads(stdout.getvalue())
            self.assertEqual("claimed", claimed["status"])
            self.assertNotIn(str(path), stdout.getvalue())

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "ha-signal-outbox-ack",
                        "--database",
                        str(path),
                        "--outbox-id",
                        claimed["outbox_id"],
                        "--lease-token",
                        claimed["lease_token"],
                    ]
                )
            self.assertEqual(0, code)
            self.assertTrue(json.loads(stdout.getvalue())["payload_scrubbed"])

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["ha-signal-outbox-status", "--database", str(path)])
            self.assertEqual(0, code)
            status_payload = json.loads(stdout.getvalue())
            self.assertEqual(1, status_payload["completed"])
            self.assertFalse(status_payload["payloads_disclosed"])
            self.assertNotIn(str(path), stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
