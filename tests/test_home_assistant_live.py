from __future__ import annotations

import json
import os
import socket
import stat
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

from baby_monitor.adapters.home_assistant import HomeAssistantSignalMap
from baby_monitor.adapters.home_assistant_live import (
    HomeAssistantEndpoint,
    HomeAssistantLiveError,
    HomeAssistantLiveSession,
    HomeAssistantLiveTimeout,
    HomeAssistantOAuthCredentials,
    HomeAssistantWatchResult,
    HomeAssistantWatchSettings,
    _access_value_from_file,
    _read_token_file,
    _WebSocket,
    begin_home_assistant_authorization,
    begin_home_assistant_token_capture,
    discover_home_assistant_entities,
    listen_home_assistant_signals,
    watch_home_assistant_signals,
    write_home_assistant_discovery,
    write_home_assistant_oauth_credentials,
    write_home_assistant_token,
)
from baby_monitor.cli import main
from baby_monitor.domain import SignalEvent, SignalType


def _mapping_value() -> dict[str, object]:
    return json.loads(
        Path("config/home-assistant-signals.example.json").read_text(encoding="utf-8")
    )


def _event(
    entity_id: str,
    new_state: str,
    *,
    old_state: str = "off",
    event_type: str | None = None,
) -> dict[str, object]:
    attributes: dict[str, object] = {"friendly_name": "Private camera label"}
    if event_type is not None:
        attributes["event_type"] = event_type
    return {
        "id": 2,
        "type": "event",
        "event": {
            "event_type": "state_changed",
            "data": {
                "entity_id": entity_id,
                "old_state": {
                    "entity_id": entity_id,
                    "state": old_state,
                    "attributes": {},
                },
                "new_state": {
                    "entity_id": entity_id,
                    "state": new_state,
                    "attributes": attributes,
                },
            },
            "origin": "LOCAL",
            "time_fired": "2026-08-10T01:02:03+00:00",
            "context": {
                "id": "private_context",
                "parent_id": None,
                "user_id": "private_user",
            },
        },
    }


class _FakeSession:
    def __init__(
        self,
        *,
        states: tuple[dict[str, object], ...] = (),
        events: tuple[dict[str, object] | Exception, ...] = (),
    ):
        self._states = states
        self._events = list(events)
        self.closed = False
        self.subscribed = False

    def states(self) -> tuple[dict[str, object], ...]:
        return self._states

    def subscribe_state_changed(self) -> int:
        self.subscribed = True
        return 2

    def receive_event(self, *, timeout_seconds: float) -> dict[str, object]:
        if timeout_seconds <= 0:
            raise AssertionError("receive timeout must be positive")
        if not self._events:
            raise HomeAssistantLiveTimeout("fixture receive window ended")
        value = self._events.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self) -> None:
        self.closed = True


class _FakeWebSocket:
    def __init__(self, messages: tuple[dict[str, object], ...]):
        self.messages = list(messages)
        self.sent: list[dict[str, object]] = []
        self.closed = False

    def receive_json(self) -> dict[str, object]:
        return self.messages.pop(0)

    def send_json(self, value: dict[str, object]) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


class HomeAssistantLiveTests(unittest.TestCase):
    def test_endpoint_is_lan_only_and_credential_free(self) -> None:
        endpoint = HomeAssistantEndpoint.parse("http://127.0.0.1:8123")
        self.assertEqual("127.0.0.1", endpoint.host)
        self.assertEqual(8123, endpoint.port)
        self.assertEqual("/api/websocket", endpoint.websocket_path)
        self.assertFalse(endpoint.tls)

        # release-audit: allow=fixture-credential
        for value in (
            "http://user:password@127.0.0.1:8123",
            "http://127.0.0.1:8123/api/states",
            "http://8.8.8.8:8123",
        ):
            with self.subTest(value=value), self.assertRaises(HomeAssistantLiveError):
                HomeAssistantEndpoint.parse(value)

    def test_token_file_requires_private_mode_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = root / "ha-access"
            # release-audit: allow=fixture-credential
            write_home_assistant_token(token, "fixture_access_value_1234567890")
            self.assertEqual(31, len(_read_token_file(token)))
            if os.name == "posix":
                self.assertEqual(0o600, stat.S_IMODE(token.stat().st_mode))
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                # release-audit: allow=fixture-credential
                write_home_assistant_token(token, "fixture_access_value_1234567890")

            if os.name == "posix":
                token.chmod(0o644)
                with self.assertRaisesRegex(HomeAssistantLiveError, "group"):
                    _read_token_file(token)
                token.chmod(0o600)

            link = root / "ha-access-link"
            try:
                link.symlink_to(token)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {type(exc).__name__}")
            with self.assertRaisesRegex(HomeAssistantLiveError, "missing or unsafe"):
                _read_token_file(link)

    def test_store_token_cli_uses_hidden_input_and_never_prints_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ha-access"
            stdout = StringIO()
            with (
                # release-audit: allow=fixture-credential
                patch(
                    "baby_monitor.cli_commands.home_assistant.getpass.getpass",
                    return_value="fixture_access_value_1234567890",
                ),
                redirect_stdout(stdout),
            ):
                code = main(["store-ha-token", "--output", str(output)])
            self.assertEqual(0, code)
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["token_file_written"])
            self.assertFalse(result["token_disclosed"])
            self.assertNotIn("fixture_access", stdout.getvalue())
            self.assertNotIn(str(output), stdout.getvalue())
            self.assertEqual(31, len(_read_token_file(output)))

    def test_loopback_capture_stores_token_without_echoing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ha-access"
            capture = begin_home_assistant_token_capture(output)
            # release-audit: allow=fixture-credential
            fixture_access = "fixture_access_value_1234567890"
            request_errors: list[Exception] = []
            response_bodies: list[str] = []

            def submit_access_value() -> None:
                try:
                    payload = urlencode({"access_value": fixture_access}).encode("ascii")
                    request = Request(
                        capture.capture_url + "capture",
                        data=payload,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        method="POST",
                    )
                    with urlopen(request, timeout=5) as response:  # noqa: S310
                        self.assertEqual(200, response.status)
                        response_bodies.append(response.read().decode("utf-8"))
                except Exception as exc:  # pragma: no cover - reported below
                    request_errors.append(exc)

            submitter = threading.Thread(target=submit_access_value)
            submitter.start()
            capture.complete(timeout_seconds=5)
            submitter.join(timeout=5)

            self.assertFalse(submitter.is_alive())
            self.assertEqual([], request_errors)
            self.assertEqual(1, len(response_bodies))
            self.assertNotIn(fixture_access, response_bodies[0])
            self.assertEqual(len(fixture_access), len(_read_token_file(output)))
            if os.name == "posix":
                self.assertEqual(0o600, stat.S_IMODE(output.stat().st_mode))

    def test_browser_oauth_callback_stores_private_refreshable_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ha-oauth"
            authorization = begin_home_assistant_authorization(
                "http://127.0.0.1:8123",
                output,
            )
            state = parse_qs(urlsplit(authorization.authorization_url).query)["state"][0]
            callback_query = urlencode({"code": "fixture_authorization_code", "state": state})
            callback_url = f"{authorization.redirect_uri}?{callback_query}"
            callback_errors: list[Exception] = []

            def visit_callback() -> None:
                try:
                    with urlopen(callback_url, timeout=5) as response:  # noqa: S310
                        self.assertEqual(200, response.status)
                        self.assertNotIn(
                            "fixture_access",
                            response.read().decode("utf-8"),
                        )
                except Exception as exc:  # pragma: no cover - reported below
                    callback_errors.append(exc)

            visitor = threading.Thread(target=visit_callback)
            visitor.start()
            with patch(
                "baby_monitor.adapters.home_assistant_live._post_home_assistant_form",
                return_value={
                    # release-audit: allow=fixture-credential
                    "access_token": "fixture_access_value_1234567890",
                    # release-audit: allow=fixture-credential
                    "refresh_token": "fixture_refresh_value_1234567890",
                    "expires_in": 1800,
                    "token_type": "Bearer",
                },
            ):
                credentials = authorization.complete(timeout_seconds=5)
            visitor.join(timeout=5)
            self.assertFalse(visitor.is_alive())
            self.assertEqual([], callback_errors)
            self.assertTrue(output.is_file())
            if os.name == "posix":
                self.assertEqual(0o600, stat.S_IMODE(output.stat().st_mode))
            self.assertEqual("http://127.0.0.1:8123", credentials.base_url)
            self.assertGreater(credentials.expires_at, time.time())

    def test_oauth_credential_file_refreshes_without_disclosing_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ha-oauth"
            credentials = HomeAssistantOAuthCredentials(
                base_url="http://127.0.0.1:8123",
                client_id="http://127.0.0.1:43210",
                # release-audit: allow=fixture-credential
                access_token="fixture_expired_value_1234567890",
                # release-audit: allow=fixture-credential
                refresh_token="fixture_refresh_value_1234567890",
                expires_at=time.time() - 1,
            )
            write_home_assistant_oauth_credentials(output, credentials)
            with patch(
                "baby_monitor.adapters.home_assistant_live._post_home_assistant_form",
                return_value={
                    # release-audit: allow=fixture-credential
                    "access_token": "fixture_refreshed_value_1234567890",
                    "expires_in": 1800,
                    "token_type": "Bearer",
                },
            ) as post_form:
                access_value = _access_value_from_file(
                    "http://127.0.0.1:8123",
                    output,
                    timeout_seconds=5,
                )
            self.assertGreaterEqual(len(access_value), 16)
            self.assertEqual(1, post_form.call_count)
            if os.name == "posix":
                self.assertEqual(0o600, stat.S_IMODE(output.stat().st_mode))

    def test_websocket_receives_unmasked_json_and_masks_client_frames(self) -> None:
        client, server = socket.socketpair()
        try:
            websocket = _WebSocket(client)
            server_payload = b'{"type":"auth_required"}'
            server.sendall(bytes((0x81, len(server_payload))) + server_payload)
            self.assertEqual("auth_required", websocket.receive_json()["type"])

            websocket.send_json({"id": 1, "type": "get_states"})
            first, second = server.recv(2)
            self.assertEqual(0x81, first)
            self.assertTrue(second & 0x80)
            length = second & 0x7F
            mask = server.recv(4)
            masked = server.recv(length)
            decoded = bytes(value ^ mask[index % 4] for index, value in enumerate(masked))
            self.assertEqual(
                {"id": 1, "type": "get_states"},
                json.loads(decoded.decode("utf-8")),
            )
        finally:
            client.close()
            server.close()

    def test_live_session_authenticates_from_private_file_without_error_echo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "ha-access"
            # release-audit: allow=fixture-credential
            write_home_assistant_token(token_path, "fixture_access_value_1234567890")
            websocket = _FakeWebSocket(
                (
                    {"type": "auth_required", "ha_version": "fixture"},
                    {"type": "auth_ok", "ha_version": "fixture"},
                )
            )
            with patch(
                "baby_monitor.adapters.home_assistant_live._open_websocket",
                return_value=websocket,
            ):
                session = HomeAssistantLiveSession.open(
                    "http://127.0.0.1:8123",
                    token_path,
                )
            self.assertEqual("auth", websocket.sent[0]["type"])
            self.assertEqual(31, len(str(websocket.sent[0]["access_token"])))
            session.close()
            self.assertTrue(websocket.closed)

            rejected = _FakeWebSocket(
                (
                    {"type": "auth_required", "ha_version": "fixture"},
                    {"type": "auth_invalid", "message": "private server reason"},
                )
            )
            with (
                patch(
                    "baby_monitor.adapters.home_assistant_live._open_websocket",
                    return_value=rejected,
                ),
                self.assertRaisesRegex(HomeAssistantLiveError, "authentication failed") as raised,
            ):
                HomeAssistantLiveSession.open(
                    "http://127.0.0.1:8123",
                    token_path,
                )
            self.assertNotIn("private server reason", str(raised.exception))
            self.assertNotIn("fixture_access", str(raised.exception))

    def test_discovery_is_private_bounded_and_exclusive(self) -> None:
        states = (
            {
                "entity_id": "binary_sensor.private_camera_motion",
                "state": "on",
                "attributes": {
                    "device_class": "motion",
                    "friendly_name": "Private room",
                },
            },
            {
                "entity_id": "event.private_camera_activity",
                "state": "private timestamp",
                "attributes": {
                    "event_types": ["motion", "cry"],
                    "friendly_name": "Private room",
                },
            },
            {
                "entity_id": "light.private_room",
                "state": "on",
                "attributes": {},
            },
        )
        session = _FakeSession(states=states)
        with patch(
            "baby_monitor.adapters.home_assistant_live.HomeAssistantLiveSession.open",
            return_value=session,
        ):
            candidates = discover_home_assistant_entities(
                "http://127.0.0.1:8123",
                "unused-private-token-path",
            )
        self.assertTrue(session.closed)
        self.assertEqual(2, len(candidates))
        self.assertEqual("motion", candidates[0].suggested_signal_type)
        self.assertEqual(("cry", "motion"), candidates[1].event_types)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "discovery.json"
            write_home_assistant_discovery(output, candidates)
            if os.name == "posix":
                self.assertEqual(0o600, stat.S_IMODE(output.stat().st_mode))
            value = json.loads(output.read_text(encoding="utf-8"))
            serialized = json.dumps(value, ensure_ascii=False)
            self.assertNotIn("Private room", serialized)
            self.assertNotIn("private timestamp", serialized)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_home_assistant_discovery(output, candidates)

    def test_live_listener_emits_only_mapped_path_free_signals(self) -> None:
        signal_map = HomeAssistantSignalMap.from_dict(_mapping_value())
        session = _FakeSession(
            events=(
                _event("binary_sensor.camera_01_motion", "on"),
                _event("binary_sensor.unmapped_private", "on"),
            )
        )
        with patch(
            "baby_monitor.adapters.home_assistant_live.HomeAssistantLiveSession.open",
            return_value=session,
        ):
            signals = listen_home_assistant_signals(
                "http://127.0.0.1:8123",
                "unused-private-token-path",
                signal_map,
                duration_seconds=10.0,
            )
        self.assertTrue(session.subscribed)
        self.assertTrue(session.closed)
        self.assertEqual(1, len(signals))
        self.assertEqual(SignalType.MOTION, signals[0].signal_type)
        payload = json.dumps(signals[0].to_dict(), ensure_ascii=False)
        self.assertNotIn("binary_sensor", payload)
        self.assertNotIn("private", payload)

    def test_watch_reconnects_and_deduplicates_without_error_disclosure(self) -> None:
        signal_map = HomeAssistantSignalMap.from_dict(_mapping_value())
        motion = _event("binary_sensor.camera_01_motion", "on")
        cry = _event(
            "event.camera_01_cry",
            "2026-08-10T01:02:03+00:00",
            old_state="2026-08-09T01:02:03+00:00",
            event_type="cry",
        )
        first = _FakeSession(
            events=(
                motion,
                HomeAssistantLiveError("PRIVATE disconnect detail"),
            )
        )
        second = _FakeSession(events=(motion, cry))
        with patch(
            "baby_monitor.adapters.home_assistant_live.HomeAssistantLiveSession.open",
            side_effect=(first, second),
        ):
            result = watch_home_assistant_signals(
                "http://127.0.0.1:8123",
                "unused-private-token-path",
                signal_map,
                settings=HomeAssistantWatchSettings(
                    max_runtime_seconds=10.0,
                    poll_seconds=1.0,
                    reconnect_initial_seconds=0.0,
                    reconnect_max_seconds=0.0,
                    max_connection_attempts=2,
                    max_messages=10,
                    max_signals=2,
                ),
            )

        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertEqual(2, result.connection_attempts)
        self.assertEqual(2, result.successful_connections)
        self.assertEqual(1, result.reconnect_count)
        self.assertEqual(3, result.received_message_count)
        self.assertEqual(2, len(result.signals))
        self.assertEqual("signal_limit", result.stop_reason)
        serialized = json.dumps(result.to_dict(), ensure_ascii=False)
        self.assertNotIn("PRIVATE", serialized)
        self.assertNotIn("binary_sensor", serialized)
        self.assertNotIn("event.camera", serialized)
        self.assertNotIn("unused-private", serialized)

    def test_watch_settings_are_bounded(self) -> None:
        with self.assertRaisesRegex(HomeAssistantLiveError, "runtime"):
            HomeAssistantWatchSettings(max_runtime_seconds=float("inf"))
        with self.assertRaisesRegex(HomeAssistantLiveError, "delay range"):
            HomeAssistantWatchSettings(
                reconnect_initial_seconds=10.0,
                reconnect_max_seconds=1.0,
            )
        with self.assertRaisesRegex(HomeAssistantLiveError, "signal limit"):
            HomeAssistantWatchSettings(max_signals=1_001)

    def test_watch_cli_streams_each_signal_without_repeating_payload(self) -> None:
        signal = SignalEvent.from_dict(
            {
                "version": 1,
                "signal_id": "fixture_stream_signal",
                "camera_id": "cam_01",
                "occurred_at": "2026-08-10T01:02:03+00:00",
                "signal_type": "motion",
                "source_adapter": "home_assistant",
                "confidence": 0.5,
                "attributes": {"channel": "device"},
            }
        )

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
                    "--stream-json-lines",
                ]
            )

        self.assertEqual(0, code)
        lines = stdout.getvalue().splitlines()
        streamed = json.loads(lines[0])
        summary = json.loads("\n".join(lines[1:]))
        self.assertEqual("signal", streamed["record_type"])
        self.assertEqual("fixture_stream_signal", streamed["signal"]["signal_id"])
        self.assertEqual(1, summary["signals_streamed"])
        self.assertEqual([], summary["signals"])
        self.assertEqual(1, stdout.getvalue().count("fixture_stream_signal"))
        self.assertNotIn("unused-private-token-path", stdout.getvalue())

    def test_cli_discovery_requires_confirmation_and_hides_entity_ids(self) -> None:
        states = (
            {
                "entity_id": "binary_sensor.private_camera_motion",
                "state": "on",
                "attributes": {"device_class": "motion"},
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "discovery.json"
            stderr = StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "discover-ha-signals",
                        "--url",
                        "http://127.0.0.1:8123",
                        "--token-file",
                        str(root / "unused"),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(2, code)
            self.assertIn("confirm-private-output", stderr.getvalue())
            self.assertFalse(output.exists())

            session = _FakeSession(states=states)
            stdout = StringIO()
            with (
                patch(
                    "baby_monitor.adapters.home_assistant_live.HomeAssistantLiveSession.open",
                    return_value=session,
                ),
                redirect_stdout(stdout),
            ):
                code = main(
                    [
                        "discover-ha-signals",
                        "--url",
                        "http://127.0.0.1:8123",
                        "--token-file",
                        str(root / "unused"),
                        "--output",
                        str(output),
                        "--confirm-private-output",
                    ]
                )
            self.assertEqual(0, code)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(1, summary["candidate_entity_count"])
            self.assertFalse(summary["entity_ids_disclosed"])
            self.assertFalse(summary["token_disclosed"])
            self.assertNotIn("private_camera", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
