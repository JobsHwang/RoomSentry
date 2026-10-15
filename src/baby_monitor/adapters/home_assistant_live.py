from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import json
import os
import secrets
import socket
import ssl
import stat
import struct
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

from baby_monitor.adapters.home_assistant import (
    HomeAssistantSignalMap,
    normalize_home_assistant_event,
)
from baby_monitor.adapters.signal import deduplicate_signals
from baby_monitor.domain import SignalEvent

MAX_HA_MESSAGE_BYTES = 2 * 1024 * 1024
MAX_HA_HTTP_HEADER_BYTES = 32 * 1024
MAX_HA_STATES = 20_000
MAX_HA_DISCOVERY_BYTES = 2 * 1024 * 1024
MAX_HA_CREDENTIAL_BYTES = 16 * 1024
MAX_HA_TOKEN_RESPONSE_BYTES = 64 * 1024
_DISCOVERY_DOMAINS = {"binary_sensor", "event"}
_MOTION_DEVICE_CLASSES = {"motion", "occupancy", "presence"}
_SOUND_DEVICE_CLASSES = {"sound"}


class HomeAssistantLiveError(RuntimeError):
    """A live HA connection failed without exposing private server data."""


class HomeAssistantLiveTimeout(HomeAssistantLiveError):
    """A bounded live HA receive window ended without another message."""


def _strict_object(raw: str, *, context: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HomeAssistantLiveError(f"{context} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                HomeAssistantLiveError(f"{context} contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise HomeAssistantLiveError(f"{context} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise HomeAssistantLiveError(f"{context} must be an object")
    return value


def _private_addresses(host: str, port: int) -> tuple[str, ...]:
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise HomeAssistantLiveError("HA host could not be resolved") from exc
    addresses = tuple(sorted({record[4][0] for record in records}))
    if not addresses:
        raise HomeAssistantLiveError("HA host did not resolve to an address")
    for address in addresses:
        parsed = ipaddress.ip_address(address)
        if not (parsed.is_private or parsed.is_loopback or parsed.is_link_local):
            raise HomeAssistantLiveError("HA live connector is restricted to the LAN")
    return addresses


@dataclass(frozen=True, slots=True)
class HomeAssistantEndpoint:
    host: str
    port: int
    websocket_path: str
    tls: bool

    @classmethod
    def parse(cls, value: str) -> HomeAssistantEndpoint:
        if not isinstance(value, str) or not value or len(value) > 2048:
            raise HomeAssistantLiveError("HA URL is invalid")
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/", "/api/websocket"}
        ):
            raise HomeAssistantLiveError("HA URL must be a credential-free HTTP(S) origin")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as exc:
            raise HomeAssistantLiveError("HA URL port is invalid") from exc
        _private_addresses(parsed.hostname, port)
        return cls(
            host=parsed.hostname,
            port=port,
            websocket_path="/api/websocket",
            tls=parsed.scheme == "https",
        )

    @property
    def host_header(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        default_port = 443 if self.tls else 80
        return host if self.port == default_port else f"{host}:{self.port}"

    @property
    def origin(self) -> str:
        scheme = "https" if self.tls else "http"
        return f"{scheme}://{self.host_header}"


def _safe_private_source(path: str | Path, *, max_bytes: int, label: str) -> Path:
    source = Path(path).expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        metadata = source.stat()
        if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise HomeAssistantLiveError(f"{label} must not be readable by group or other users")
        if not 1 <= metadata.st_size <= max_bytes:
            raise HomeAssistantLiveError(f"{label} size is unsafe")
    except HomeAssistantLiveError:
        raise
    except OSError as exc:
        raise HomeAssistantLiveError(f"{label} is missing or unsafe") from exc
    return source


def _safe_private_destination(path: str | Path, *, label: str) -> Path:
    target = Path(path).expanduser()
    if target.is_symlink() or not target.parent.is_dir() or target.parent.is_symlink():
        raise HomeAssistantLiveError(f"{label} destination is unsafe")
    return target


def _write_private_exclusive(path: str | Path, payload: bytes, *, label: str) -> None:
    target = _safe_private_destination(path, label=label)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError:
        raise FileExistsError(f"{label} output already exists") from None
    except OSError as exc:
        raise HomeAssistantLiveError(f"{label} output could not be created") from exc
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        with suppress(OSError):
            target.unlink()
        raise HomeAssistantLiveError(f"{label} output could not be written") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _replace_private_file(path: str | Path, payload: bytes, *, label: str) -> None:
    target = _safe_private_source(
        path,
        max_bytes=MAX_HA_CREDENTIAL_BYTES,
        label=label,
    )
    temporary = target.with_name(f".{target.name}.tmp-{secrets.token_hex(8)}")
    _write_private_exclusive(temporary, payload, label=label)
    try:
        os.replace(temporary, target)
        if os.name == "posix":
            target.chmod(0o600)
    except OSError as exc:
        with suppress(OSError):
            temporary.unlink()
        raise HomeAssistantLiveError(f"{label} could not be refreshed") from exc


def _read_token_file(path: str | Path) -> str:
    source = _safe_private_source(path, max_bytes=4096, label="HA token file")
    try:
        value = source.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise HomeAssistantLiveError("HA token file is missing or unsafe") from exc
    token = value.strip()
    _validate_token_value(token)
    return token


def _validate_token_value(token: str) -> None:
    if (
        not 16 <= len(token) <= 4096
        or any(character.isspace() for character in token)
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in token)
    ):
        raise HomeAssistantLiveError("HA token file content is invalid")


def write_home_assistant_token(path: str | Path, token: str) -> None:
    _validate_token_value(token)
    _write_private_exclusive(
        path,
        f"{token}\n".encode("ascii"),
        label="HA token",
    )


@dataclass(frozen=True, slots=True)
class HomeAssistantOAuthCredentials:
    base_url: str
    client_id: str
    access_token: str
    refresh_token: str
    expires_at: float
    token_type: str = "Bearer"
    kind: str = "home_assistant_oauth"
    version: int = 1

    def __post_init__(self) -> None:
        endpoint = HomeAssistantEndpoint.parse(self.base_url)
        if self.base_url != endpoint.origin:
            raise HomeAssistantLiveError("HA OAuth base URL is not canonical")
        _validate_loopback_client_id(self.client_id)
        _validate_token_value(self.access_token)
        _validate_token_value(self.refresh_token)
        if (
            type(self.expires_at) not in {int, float}
            or not 0 < float(self.expires_at) < 100_000_000_000
        ):
            raise HomeAssistantLiveError("HA OAuth expiry is invalid")
        if self.token_type != "Bearer":
            raise HomeAssistantLiveError("HA OAuth token type is invalid")
        if self.kind != "home_assistant_oauth" or self.version != 1:
            raise HomeAssistantLiveError("HA OAuth credential version is invalid")

    @classmethod
    def from_dict(cls, value: object) -> HomeAssistantOAuthCredentials:
        fields = {
            "version",
            "kind",
            "base_url",
            "client_id",
            "access_token",
            "refresh_token",
            "expires_at",
            "token_type",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise HomeAssistantLiveError("HA OAuth credential fields are invalid")
        return cls(
            version=value["version"],
            kind=value["kind"],
            base_url=value["base_url"],
            client_id=value["client_id"],
            access_token=value["access_token"],
            refresh_token=value["refresh_token"],
            expires_at=value["expires_at"],
            token_type=value["token_type"],
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "kind": self.kind,
            "base_url": self.base_url,
            "client_id": self.client_id,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": float(self.expires_at),
            "token_type": self.token_type,
        }


def _validate_loopback_client_id(value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise HomeAssistantLiveError("HA OAuth client ID is invalid")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise HomeAssistantLiveError("HA OAuth client ID is invalid") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1"}
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise HomeAssistantLiveError("HA OAuth client ID must be a loopback HTTP origin")


def _credential_payload(credentials: HomeAssistantOAuthCredentials) -> bytes:
    payload = (
        json.dumps(
            credentials.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    if len(payload) > MAX_HA_CREDENTIAL_BYTES:
        raise HomeAssistantLiveError("HA OAuth credential output is too large")
    return payload


def write_home_assistant_oauth_credentials(
    path: str | Path,
    credentials: HomeAssistantOAuthCredentials,
) -> None:
    _write_private_exclusive(
        path,
        _credential_payload(credentials),
        label="HA OAuth credentials",
    )


def _read_home_assistant_oauth_credentials(
    path: str | Path,
) -> HomeAssistantOAuthCredentials:
    source = _safe_private_source(
        path,
        max_bytes=MAX_HA_CREDENTIAL_BYTES,
        label="HA OAuth credential file",
    )
    try:
        raw = source.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise HomeAssistantLiveError("HA OAuth credential file is unreadable") from exc
    return HomeAssistantOAuthCredentials.from_dict(
        _strict_object(raw, context="HA OAuth credential file")
    )


def _post_home_assistant_form(
    endpoint: HomeAssistantEndpoint,
    path: str,
    fields: dict[str, str],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    if path not in {"/auth/token", "/auth/revoke"}:
        raise HomeAssistantLiveError("HA authentication endpoint is invalid")
    _private_addresses(endpoint.host, endpoint.port)
    body = urlencode(fields).encode("ascii")
    connection_type: type[http.client.HTTPConnection]
    connection_type = http.client.HTTPSConnection if endpoint.tls else http.client.HTTPConnection
    connection: http.client.HTTPConnection | None = None
    try:
        connection = connection_type(
            endpoint.host,
            endpoint.port,
            timeout=timeout_seconds,
        )
        connection.request(
            "POST",
            path,
            body=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        payload = response.read(MAX_HA_TOKEN_RESPONSE_BYTES + 1)
        status = response.status
    except (OSError, http.client.HTTPException) as exc:
        raise HomeAssistantLiveError("HA token request failed") from exc
    finally:
        if connection is not None:
            with suppress(Exception):
                connection.close()
    if len(payload) > MAX_HA_TOKEN_RESPONSE_BYTES:
        raise HomeAssistantLiveError("HA token response is too large")
    if status != 200:
        raise HomeAssistantLiveError("HA token request was rejected")
    try:
        raw = payload.decode("utf-8")
    except UnicodeError as exc:
        raise HomeAssistantLiveError("HA token response is invalid") from exc
    return _strict_object(raw, context="HA token response")


def _credentials_from_token_response(
    value: dict[str, Any],
    *,
    endpoint: HomeAssistantEndpoint,
    client_id: str,
    existing_refresh_token: str | None = None,
) -> HomeAssistantOAuthCredentials:
    access_token = value.get("access_token")
    refresh_token = value.get("refresh_token", existing_refresh_token)
    expires_in = value.get("expires_in")
    token_type = value.get("token_type")
    if (
        not isinstance(access_token, str)
        or not isinstance(refresh_token, str)
        or type(expires_in) not in {int, float}
        or not 1 <= float(expires_in) <= 86_400
        or token_type != "Bearer"
    ):
        raise HomeAssistantLiveError("HA token response fields are invalid")
    return HomeAssistantOAuthCredentials(
        base_url=endpoint.origin,
        client_id=client_id,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + float(expires_in),
        token_type=token_type,
    )


def _access_value_from_file(
    base_url: str,
    path: str | Path,
    *,
    timeout_seconds: float,
) -> str:
    source = _safe_private_source(
        path,
        max_bytes=MAX_HA_CREDENTIAL_BYTES,
        label="HA credential file",
    )
    try:
        prefix = source.read_bytes()[:1]
    except OSError as exc:
        raise HomeAssistantLiveError("HA credential file is unreadable") from exc
    if prefix != b"{":
        return _read_token_file(source)
    credentials = _read_home_assistant_oauth_credentials(source)
    endpoint = HomeAssistantEndpoint.parse(base_url)
    if credentials.base_url != endpoint.origin:
        raise HomeAssistantLiveError("HA OAuth credentials belong to another origin")
    if credentials.expires_at > time.time() + 60:
        return credentials.access_token
    response = _post_home_assistant_form(
        endpoint,
        "/auth/token",
        {
            "grant_type": "refresh_token",
            "refresh_token": credentials.refresh_token,
            "client_id": credentials.client_id,
        },
        timeout_seconds=timeout_seconds,
    )
    refreshed = _credentials_from_token_response(
        response,
        endpoint=endpoint,
        client_id=credentials.client_id,
        existing_refresh_token=credentials.refresh_token,
    )
    _replace_private_file(
        source,
        _credential_payload(refreshed),
        label="HA OAuth credential file",
    )
    return refreshed.access_token


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    server_version = "BabyMonitorHAAuth/1"
    sys_version = ""

    def do_GET(self) -> None:  # noqa: N802
        server = self.server
        if not isinstance(server, _OAuthCallbackServer):
            self.send_error(500)
            return
        if self.client_address[0] not in {"127.0.0.1", "::1"} or len(self.path) > 8192:
            self.send_error(403)
            return
        parsed = urlsplit(self.path)
        try:
            values = parse_qs(
                parsed.query,
                keep_blank_values=True,
                strict_parsing=True,
            )
        except ValueError:
            values = {}
        state_values = values.get("state", [])
        code_values = values.get("code", [])
        valid = (
            parsed.path == "/auth/callback"
            and set(values) == {"code", "state"}
            and len(state_values) == 1
            and len(code_values) == 1
            and secrets.compare_digest(state_values[0], server.expected_state)
            and 1 <= len(code_values[0]) <= 8192
        )
        if not valid:
            server.callback_error = "HA authorization callback was invalid"
            self.send_error(400)
            return
        server.authorization_code = code_values[0]
        body = (
            "<!doctype html><meta charset=utf-8>"
            "<title>Home Assistant authorized</title>"
            "<h1>Home Assistant 已授权</h1>"
            "<p>凭据只保存到本机；现在可以关闭此页面。</p>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


class _OAuthCallbackServer(HTTPServer):
    allow_reuse_address = False

    def __init__(self, port: int, expected_state: str):
        super().__init__(("127.0.0.1", port), _OAuthCallbackHandler)
        self.expected_state = expected_state
        self.authorization_code: str | None = None
        self.callback_error: str | None = None


@dataclass(slots=True)
class HomeAssistantAuthorization:
    endpoint: HomeAssistantEndpoint
    credential_output: Path
    server: _OAuthCallbackServer
    client_id: str
    redirect_uri: str
    authorization_url: str

    def complete(self, *, timeout_seconds: float) -> HomeAssistantOAuthCredentials:
        if not 5 <= timeout_seconds <= 600:
            raise HomeAssistantLiveError("HA authorization timeout is invalid")
        deadline = time.monotonic() + timeout_seconds
        self.server.timeout = 0.5
        try:
            while (
                self.server.authorization_code is None
                and self.server.callback_error is None
                and time.monotonic() < deadline
            ):
                self.server.handle_request()
            if self.server.callback_error is not None:
                raise HomeAssistantLiveError(self.server.callback_error)
            code = self.server.authorization_code
            if code is None:
                raise HomeAssistantLiveTimeout("HA browser authorization timed out")
            response = _post_home_assistant_form(
                self.endpoint,
                "/auth/token",
                {
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": self.client_id,
                },
                timeout_seconds=min(timeout_seconds, 30.0),
            )
            code = ""
            credentials = _credentials_from_token_response(
                response,
                endpoint=self.endpoint,
                client_id=self.client_id,
            )
            write_home_assistant_oauth_credentials(
                self.credential_output,
                credentials,
            )
            return credentials
        finally:
            self.server.server_close()


def begin_home_assistant_authorization(
    base_url: str,
    credential_output: str | Path,
    *,
    callback_port: int = 0,
) -> HomeAssistantAuthorization:
    if type(callback_port) is not int or not 0 <= callback_port <= 65535:
        raise HomeAssistantLiveError("HA OAuth callback port is invalid")
    endpoint = HomeAssistantEndpoint.parse(base_url)
    output = _safe_private_destination(
        credential_output,
        label="HA OAuth credentials",
    )
    if output.exists():
        raise FileExistsError("HA OAuth credentials output already exists")
    state_value = secrets.token_urlsafe(32)
    server = _OAuthCallbackServer(callback_port, state_value)
    port = server.server_address[1]
    client_id = f"http://127.0.0.1:{port}"
    redirect_uri = f"{client_id}/auth/callback"
    authorization_query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state_value,
        }
    )
    authorization_url = f"{endpoint.origin}/auth/authorize?{authorization_query}"
    return HomeAssistantAuthorization(
        endpoint=endpoint,
        credential_output=output,
        server=server,
        client_id=client_id,
        redirect_uri=redirect_uri,
        authorization_url=authorization_url,
    )


def write_home_assistant_authorization_request(
    path: str | Path,
    authorization: HomeAssistantAuthorization,
) -> None:
    payload = (
        json.dumps(
            {
                "version": 1,
                "authorization_url": authorization.authorization_url,
                "callback_loopback_only": True,
                "token_disclosed": False,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    _write_private_exclusive(
        path,
        payload,
        label="HA authorization request",
    )


class _TokenCaptureHandler(BaseHTTPRequestHandler):
    server_version = "BabyMonitorHATokenCapture/1"
    sys_version = ""

    def do_GET(self) -> None:  # noqa: N802
        server = self.server
        if not isinstance(server, _TokenCaptureServer):
            self.send_error(500)
            return
        if self.client_address[0] not in {"127.0.0.1", "::1"} or self.path != "/":
            self.send_error(404)
            return
        body = (
            "<!doctype html><meta charset=utf-8>"
            "<title>Store Home Assistant access</title>"
            "<h1>仅保存到本机</h1>"
            "<form method=post action=/capture autocomplete=off>"
            "<label>HA access value "
            "<input type=password name=access_value required minlength=16 "
            "maxlength=4096 autofocus></label>"
            "<button type=submit>安全保存</button></form>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'none'; form-action 'self'")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        server = self.server
        if not isinstance(server, _TokenCaptureServer):
            self.send_error(500)
            return
        raw_length = self.headers.get("Content-Length")
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        try:
            length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            length = -1
        if (
            self.client_address[0] not in {"127.0.0.1", "::1"}
            or self.path != "/capture"
            or content_type != "application/x-www-form-urlencoded"
            or not 1 <= length <= 8192
        ):
            server.capture_error = "HA access capture request was invalid"
            self.send_error(400)
            return
        payload = self.rfile.read(length)
        try:
            values = parse_qs(
                payload.decode("ascii"),
                keep_blank_values=True,
                strict_parsing=True,
            )
        except (UnicodeError, ValueError):
            values = {}
        access_values = values.get("access_value", [])
        if set(values) != {"access_value"} or len(access_values) != 1:
            server.capture_error = "HA access capture request was invalid"
            self.send_error(400)
            return
        access_value = access_values[0]
        try:
            write_home_assistant_token(server.output, access_value)
            access_value = ""
        except Exception:
            access_value = ""
            server.capture_error = "HA access value could not be stored"
            self.send_error(500)
            return
        server.captured = True
        body = (
            "<!doctype html><meta charset=utf-8>"
            "<title>Home Assistant access stored</title>"
            "<h1>已安全保存</h1><p>可以关闭此页面。</p>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'none'")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


class _TokenCaptureServer(HTTPServer):
    allow_reuse_address = False

    def __init__(self, output: Path, port: int):
        super().__init__(("127.0.0.1", port), _TokenCaptureHandler)
        self.output = output
        self.captured = False
        self.capture_error: str | None = None


@dataclass(slots=True)
class HomeAssistantTokenCapture:
    output: Path
    server: _TokenCaptureServer
    capture_url: str

    def complete(self, *, timeout_seconds: float) -> None:
        if not 5 <= timeout_seconds <= 600:
            raise HomeAssistantLiveError("HA access capture timeout is invalid")
        deadline = time.monotonic() + timeout_seconds
        self.server.timeout = 0.5
        try:
            while (
                not self.server.captured
                and self.server.capture_error is None
                and time.monotonic() < deadline
            ):
                self.server.handle_request()
            if self.server.capture_error is not None:
                raise HomeAssistantLiveError(self.server.capture_error)
            if not self.server.captured:
                raise HomeAssistantLiveTimeout("HA access capture timed out")
        finally:
            self.server.server_close()


def begin_home_assistant_token_capture(
    output: str | Path,
    *,
    callback_port: int = 0,
) -> HomeAssistantTokenCapture:
    if type(callback_port) is not int or not 0 <= callback_port <= 65535:
        raise HomeAssistantLiveError("HA access capture port is invalid")
    target = _safe_private_destination(output, label="HA token")
    if target.exists():
        raise FileExistsError("HA token output already exists")
    server = _TokenCaptureServer(target, callback_port)
    port = server.server_address[1]
    return HomeAssistantTokenCapture(
        output=target,
        server=server,
        capture_url=f"http://127.0.0.1:{port}/",
    )


def write_home_assistant_token_capture_request(
    path: str | Path,
    capture: HomeAssistantTokenCapture,
) -> None:
    payload = (
        json.dumps(
            {
                "version": 1,
                "capture_url": capture.capture_url,
                "loopback_only": True,
                "token_disclosed": False,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    _write_private_exclusive(
        path,
        payload,
        label="HA token capture request",
    )


def _receive_http_upgrade(
    connection: socket.socket,
) -> tuple[bytes, bytes]:
    received = bytearray()
    while b"\r\n\r\n" not in received:
        if len(received) >= MAX_HA_HTTP_HEADER_BYTES:
            raise HomeAssistantLiveError("HA WebSocket upgrade headers are too large")
        try:
            block = connection.recv(4096)
        except TimeoutError as exc:
            raise HomeAssistantLiveTimeout("HA WebSocket upgrade timed out") from exc
        except OSError as exc:
            raise HomeAssistantLiveError("HA WebSocket upgrade failed") from exc
        if not block:
            raise HomeAssistantLiveError("HA WebSocket upgrade closed early")
        received.extend(block)
    boundary = received.index(b"\r\n\r\n") + 4
    return bytes(received[:boundary]), bytes(received[boundary:])


class _WebSocket:
    def __init__(self, connection: socket.socket, initial: bytes = b""):
        self.connection = connection
        self.buffer = bytearray(initial)

    def set_timeout(self, seconds: float) -> None:
        self.connection.settimeout(seconds)

    def _receive_exact(self, size: int) -> bytes:
        while len(self.buffer) < size:
            try:
                block = self.connection.recv(min(65536, size - len(self.buffer)))
            except TimeoutError as exc:
                raise HomeAssistantLiveTimeout("HA live receive window ended") from exc
            except OSError as exc:
                raise HomeAssistantLiveError("HA WebSocket receive failed") from exc
            if not block:
                raise HomeAssistantLiveError("HA WebSocket connection closed")
            self.buffer.extend(block)
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def _receive_frame(self) -> tuple[bool, int, bytes]:
        first, second = self._receive_exact(2)
        if first & 0x70:
            raise HomeAssistantLiveError("HA WebSocket used unsupported extensions")
        final = bool(first & 0x80)
        opcode = first & 0x0F
        if second & 0x80:
            raise HomeAssistantLiveError("HA server sent a masked WebSocket frame")
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._receive_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._receive_exact(8))[0]
        if length > MAX_HA_MESSAGE_BYTES:
            raise HomeAssistantLiveError("HA WebSocket message is too large")
        return final, opcode, self._receive_exact(length)

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        if len(payload) > MAX_HA_MESSAGE_BYTES:
            raise HomeAssistantLiveError("HA WebSocket message is too large")
        first = bytes((0x80 | opcode,))
        if len(payload) < 126:
            header = bytes((0x80 | len(payload),))
        elif len(payload) <= 0xFFFF:
            header = bytes((0x80 | 126,)) + struct.pack("!H", len(payload))
        else:
            header = bytes((0x80 | 127,)) + struct.pack("!Q", len(payload))
        mask = secrets.token_bytes(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        try:
            self.connection.sendall(first + header + mask + masked)
        except OSError as exc:
            raise HomeAssistantLiveError("HA WebSocket send failed") from exc

    def send_json(self, value: dict[str, object]) -> None:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self._send_frame(0x1, payload)

    def receive_json(self) -> dict[str, Any]:
        fragments: list[bytes] = []
        started = False
        while True:
            final, opcode, payload = self._receive_frame()
            if opcode == 0x8:
                raise HomeAssistantLiveError("HA WebSocket connection closed")
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode == 0x1 and not started:
                started = True
            elif opcode != 0x0 or not started:
                raise HomeAssistantLiveError("HA WebSocket frame sequence is invalid")
            fragments.append(payload)
            if sum(len(item) for item in fragments) > MAX_HA_MESSAGE_BYTES:
                raise HomeAssistantLiveError("HA WebSocket message is too large")
            if final:
                break
        try:
            raw = b"".join(fragments).decode("utf-8")
        except UnicodeError as exc:
            raise HomeAssistantLiveError("HA WebSocket text is invalid UTF-8") from exc
        return _strict_object(raw, context="HA WebSocket message")

    def close(self) -> None:
        with suppress(HomeAssistantLiveError):
            self._send_frame(0x8, b"")
        with suppress(OSError):
            self.connection.close()


def _open_websocket(endpoint: HomeAssistantEndpoint, timeout: float) -> _WebSocket:
    try:
        connection = socket.create_connection(
            (endpoint.host, endpoint.port),
            timeout=timeout,
        )
        if endpoint.tls:
            context = ssl.create_default_context()
            connection = context.wrap_socket(connection, server_hostname=endpoint.host)
        connection.settimeout(timeout)
    except OSError as exc:
        raise HomeAssistantLiveError("HA WebSocket connection failed") from exc

    key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
    request = (
        f"GET {endpoint.websocket_path} HTTP/1.1\r\n"
        f"Host: {endpoint.host_header}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode("ascii")
    try:
        connection.sendall(request)
        headers, initial = _receive_http_upgrade(connection)
        lines = headers.decode("iso-8859-1").split("\r\n")
        if not lines or not lines[0].startswith("HTTP/1.1 101 "):
            raise HomeAssistantLiveError("HA WebSocket upgrade was rejected")
        parsed_headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            if ":" not in line:
                raise HomeAssistantLiveError("HA WebSocket upgrade response is invalid")
            name, value = line.split(":", 1)
            lowered = name.strip().lower()
            if lowered in parsed_headers:
                raise HomeAssistantLiveError(
                    "HA WebSocket upgrade response contains duplicate headers"
                )
            parsed_headers[lowered] = value.strip()
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if (
            parsed_headers.get("upgrade", "").lower() != "websocket"
            or "upgrade"
            not in {
                item.strip().lower() for item in parsed_headers.get("connection", "").split(",")
            }
            or not secrets.compare_digest(
                parsed_headers.get("sec-websocket-accept", ""),
                expected,
            )
        ):
            raise HomeAssistantLiveError("HA WebSocket upgrade response is invalid")
        return _WebSocket(connection, initial)
    except Exception:
        with suppress(OSError):
            connection.close()
        raise


class HomeAssistantLiveSession:
    def __init__(self, websocket: _WebSocket):
        self.websocket = websocket
        self.next_request_id = 1

    @classmethod
    def open(
        cls,
        base_url: str,
        token_file: str | Path,
        *,
        timeout_seconds: float = 10.0,
    ) -> HomeAssistantLiveSession:
        if not 0.1 <= timeout_seconds <= 120.0:
            raise HomeAssistantLiveError("HA connection timeout is invalid")
        endpoint = HomeAssistantEndpoint.parse(base_url)
        websocket = _open_websocket(endpoint, timeout_seconds)
        try:
            required = websocket.receive_json()
            if required.get("type") != "auth_required":
                raise HomeAssistantLiveError("HA authentication handshake is invalid")
            access_value = _access_value_from_file(
                base_url,
                token_file,
                timeout_seconds=timeout_seconds,
            )
            websocket.send_json({"type": "auth", "access_token": access_value})
            access_value = ""
            response = websocket.receive_json()
            if response.get("type") != "auth_ok":
                raise HomeAssistantLiveError("HA authentication failed")
            return cls(websocket)
        except Exception:
            websocket.close()
            raise

    def request(self, request_type: str) -> object:
        request_id = self.next_request_id
        self.next_request_id += 1
        self.websocket.send_json({"id": request_id, "type": request_type})
        for _ in range(100):
            response = self.websocket.receive_json()
            if response.get("type") != "result" or response.get("id") != request_id:
                continue
            if response.get("success") is not True:
                raise HomeAssistantLiveError("HA WebSocket command failed")
            return response.get("result")
        raise HomeAssistantLiveError("HA WebSocket command result was not received")

    def states(self) -> tuple[dict[str, Any], ...]:
        result = self.request("get_states")
        if not isinstance(result, list) or len(result) > MAX_HA_STATES:
            raise HomeAssistantLiveError("HA state inventory is invalid or too large")
        states: list[dict[str, Any]] = []
        for value in result:
            if not isinstance(value, dict):
                raise HomeAssistantLiveError("HA state inventory is invalid")
            states.append(value)
        return tuple(states)

    def subscribe_state_changed(self) -> int:
        request_id = self.next_request_id
        self.next_request_id += 1
        self.websocket.send_json(
            {
                "id": request_id,
                "type": "subscribe_events",
                "event_type": "state_changed",
            }
        )
        for _ in range(100):
            response = self.websocket.receive_json()
            if response.get("type") != "result" or response.get("id") != request_id:
                continue
            if response.get("success") is not True:
                raise HomeAssistantLiveError("HA event subscription failed")
            return request_id
        raise HomeAssistantLiveError("HA event subscription was not acknowledged")

    def receive_event(self, *, timeout_seconds: float) -> dict[str, Any]:
        self.websocket.set_timeout(timeout_seconds)
        for _ in range(100):
            message = self.websocket.receive_json()
            if message.get("type") == "event":
                return message
        raise HomeAssistantLiveError("HA live message window contained no event")

    def close(self) -> None:
        self.websocket.close()


@dataclass(frozen=True, slots=True)
class HomeAssistantEntityCandidate:
    entity_id: str
    domain: str
    device_class: str | None
    event_types: tuple[str, ...]
    suggested_signal_type: str | None
    suggested_trigger: str

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "domain": self.domain,
            "device_class": self.device_class,
            "event_types": list(self.event_types),
            "suggested_signal_type": self.suggested_signal_type,
            "suggested_trigger": self.suggested_trigger,
        }


def _safe_event_types(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > 64:
        return ()
    result = tuple(
        item
        for item in value
        if isinstance(item, str)
        and 1 <= len(item) <= 80
        and all(0x20 < ord(character) < 0x7F for character in item)
    )
    return tuple(sorted(set(result)))


def _suggest_signal_type(
    *,
    device_class: str | None,
    event_types: tuple[str, ...],
) -> str | None:
    if device_class in _MOTION_DEVICE_CLASSES:
        return "motion"
    if device_class in _SOUND_DEVICE_CLASSES:
        return "sound_peak"
    lowered = " ".join(event_types).lower()
    if "cry" in lowered:
        return "hardware_cry"
    if "person" in lowered or "human" in lowered:
        return "person"
    if "motion" in lowered:
        return "motion"
    if "sound" in lowered or "noise" in lowered:
        return "sound_peak"
    return None


def discover_home_assistant_entities(
    base_url: str,
    token_file: str | Path,
    *,
    timeout_seconds: float = 10.0,
) -> tuple[HomeAssistantEntityCandidate, ...]:
    session = HomeAssistantLiveSession.open(
        base_url,
        token_file,
        timeout_seconds=timeout_seconds,
    )
    try:
        states = session.states()
    finally:
        session.close()
    candidates: list[HomeAssistantEntityCandidate] = []
    for state_value in states:
        entity_id = state_value.get("entity_id")
        if not isinstance(entity_id, str) or entity_id.count(".") != 1 or len(entity_id) > 255:
            continue
        domain = entity_id.split(".", 1)[0]
        if domain not in _DISCOVERY_DOMAINS:
            continue
        attributes = state_value.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        raw_device_class = attributes.get("device_class")
        device_class = (
            raw_device_class
            if isinstance(raw_device_class, str) and len(raw_device_class) <= 80
            else None
        )
        event_types = _safe_event_types(attributes.get("event_types"))
        candidates.append(
            HomeAssistantEntityCandidate(
                entity_id=entity_id,
                domain=domain,
                device_class=device_class,
                event_types=event_types,
                suggested_signal_type=_suggest_signal_type(
                    device_class=device_class,
                    event_types=event_types,
                ),
                suggested_trigger=("state_change" if domain == "event" else "rising_edge"),
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.entity_id))


def write_home_assistant_discovery(
    path: str | Path,
    candidates: tuple[HomeAssistantEntityCandidate, ...],
) -> None:
    target = Path(path).expanduser()
    if target.is_symlink() or not target.parent.is_dir() or target.parent.is_symlink():
        raise HomeAssistantLiveError("HA discovery destination is unsafe")
    payload = (
        json.dumps(
            {
                "version": 1,
                "private_entity_inventory": True,
                "entities": [item.to_dict() for item in candidates],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_HA_DISCOVERY_BYTES:
        raise HomeAssistantLiveError("HA discovery output is too large")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError:
        raise FileExistsError("HA discovery output already exists") from None
    except OSError as exc:
        raise HomeAssistantLiveError("HA discovery output could not be created") from exc
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        with suppress(OSError):
            target.unlink()
        raise HomeAssistantLiveError("HA discovery output could not be written") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def listen_home_assistant_signals(
    base_url: str,
    token_file: str | Path,
    signal_map: HomeAssistantSignalMap,
    *,
    duration_seconds: float,
    max_messages: int = 1_000,
    max_signals: int = 100,
    timeout_seconds: float = 10.0,
) -> tuple[SignalEvent, ...]:
    if not 0.1 <= duration_seconds <= 3600.0:
        raise HomeAssistantLiveError("HA listen duration is invalid")
    if type(max_messages) is not int or not 1 <= max_messages <= 10_000:
        raise HomeAssistantLiveError("HA max_messages is invalid")
    if type(max_signals) is not int or not 1 <= max_signals <= 1_000:
        raise HomeAssistantLiveError("HA max_signals is invalid")
    mappings = signal_map.by_entity_id()
    session = HomeAssistantLiveSession.open(
        base_url,
        token_file,
        timeout_seconds=timeout_seconds,
    )
    signals: list[SignalEvent] = []
    deadline = time.monotonic() + duration_seconds
    try:
        session.subscribe_state_changed()
        for _ in range(max_messages):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or len(signals) >= max_signals:
                break
            try:
                message = session.receive_event(timeout_seconds=remaining)
            except HomeAssistantLiveTimeout:
                break
            signal = normalize_home_assistant_event(message, mappings=mappings)
            if signal is not None:
                signals.append(signal)
    finally:
        session.close()
    return deduplicate_signals(signals)


@dataclass(frozen=True, slots=True)
class HomeAssistantWatchSettings:
    max_runtime_seconds: float = 3600.0
    poll_seconds: float = 15.0
    reconnect_initial_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    max_connection_attempts: int = 100
    max_messages: int = 100_000
    max_signals: int = 1_000

    def __post_init__(self) -> None:
        numeric_ranges = (
            (self.max_runtime_seconds, 0.1, 604_800.0, "runtime"),
            (self.poll_seconds, 0.1, 60.0, "poll interval"),
            (
                self.reconnect_initial_seconds,
                0.0,
                60.0,
                "initial reconnect delay",
            ),
            (
                self.reconnect_max_seconds,
                0.0,
                300.0,
                "maximum reconnect delay",
            ),
        )
        for value, minimum, maximum, label in numeric_ranges:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not minimum <= float(value) <= maximum
            ):
                raise HomeAssistantLiveError(f"HA watch {label} is invalid")
        if self.reconnect_max_seconds < self.reconnect_initial_seconds:
            raise HomeAssistantLiveError("HA watch reconnect delay range is invalid")
        integer_ranges = (
            (self.max_connection_attempts, 1, 10_000, "connection attempts"),
            (self.max_messages, 1, 1_000_000, "message limit"),
            (self.max_signals, 1, 1_000, "signal limit"),
        )
        for value, minimum, maximum, label in integer_ranges:
            if type(value) is not int or not minimum <= value <= maximum:
                raise HomeAssistantLiveError(f"HA watch {label} is invalid")


@dataclass(frozen=True, slots=True)
class HomeAssistantWatchResult:
    signals: tuple[SignalEvent, ...]
    connection_attempts: int
    successful_connections: int
    reconnect_count: int
    idle_timeout_count: int
    invalid_message_count: int
    received_message_count: int
    stop_reason: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "source_adapter": "home_assistant_live",
            "emitted_count": len(self.signals),
            "connection_attempts": self.connection_attempts,
            "successful_connections": self.successful_connections,
            "reconnect_count": self.reconnect_count,
            "idle_timeout_count": self.idle_timeout_count,
            "invalid_message_count": self.invalid_message_count,
            "received_message_count": self.received_message_count,
            "stop_reason": self.stop_reason,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "entity_ids_disclosed": False,
            "token_disclosed": False,
            "paths_disclosed": False,
            "network_used": True,
            "signals": [signal.to_dict() for signal in self.signals],
        }


def watch_home_assistant_signals(
    base_url: str,
    token_file: str | Path,
    signal_map: HomeAssistantSignalMap,
    *,
    settings: HomeAssistantWatchSettings,
    timeout_seconds: float = 10.0,
    on_signal: Callable[[SignalEvent], None] | None = None,
) -> HomeAssistantWatchResult:
    if not isinstance(settings, HomeAssistantWatchSettings):
        raise HomeAssistantLiveError("HA watch settings are invalid")
    if on_signal is not None and not callable(on_signal):
        raise HomeAssistantLiveError("HA watch signal callback is invalid")
    mappings = signal_map.by_entity_id()
    started = time.monotonic()
    deadline = started + settings.max_runtime_seconds
    signals_by_id: dict[str, SignalEvent] = {}
    connection_attempts = 0
    successful_connections = 0
    reconnect_count = 0
    idle_timeout_count = 0
    invalid_message_count = 0
    received_message_count = 0
    retry_delay = settings.reconnect_initial_seconds
    stop_reason = "connection_attempt_limit"

    while connection_attempts < settings.max_connection_attempts:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop_reason = "runtime_limit"
            break
        if received_message_count >= settings.max_messages:
            stop_reason = "message_limit"
            break
        if len(signals_by_id) >= settings.max_signals:
            stop_reason = "signal_limit"
            break

        connection_attempts += 1
        session: HomeAssistantLiveSession | None = None
        connection_failed = False
        try:
            session = HomeAssistantLiveSession.open(
                base_url,
                token_file,
                timeout_seconds=timeout_seconds,
            )
            successful_connections += 1
            session.subscribe_state_changed()
            retry_delay = settings.reconnect_initial_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stop_reason = "runtime_limit"
                    break
                if received_message_count >= settings.max_messages:
                    stop_reason = "message_limit"
                    break
                if len(signals_by_id) >= settings.max_signals:
                    stop_reason = "signal_limit"
                    break
                try:
                    message = session.receive_event(
                        timeout_seconds=min(settings.poll_seconds, remaining)
                    )
                except HomeAssistantLiveTimeout:
                    idle_timeout_count += 1
                    continue
                received_message_count += 1
                try:
                    signal = normalize_home_assistant_event(
                        message,
                        mappings=mappings,
                    )
                except (TypeError, ValueError):
                    invalid_message_count += 1
                    continue
                if signal is None or signal.signal_id in signals_by_id:
                    continue
                signals_by_id[signal.signal_id] = signal
                if on_signal is not None:
                    try:
                        on_signal(signal)
                    except HomeAssistantLiveError as exc:
                        raise RuntimeError("HA watch signal callback failed") from exc
            if stop_reason != "connection_attempt_limit":
                break
        except HomeAssistantLiveError:
            connection_failed = True
        finally:
            if session is not None:
                session.close()

        if not connection_failed:
            continue
        remaining = deadline - time.monotonic()
        if connection_attempts >= settings.max_connection_attempts:
            stop_reason = "connection_attempt_limit"
            break
        reconnect_count += 1
        if retry_delay > 0 and remaining > 0:
            time.sleep(min(retry_delay, remaining))
        retry_delay = min(
            settings.reconnect_max_seconds,
            max(settings.reconnect_initial_seconds, retry_delay * 2),
        )

    elapsed_seconds = max(0.0, time.monotonic() - started)
    return HomeAssistantWatchResult(
        signals=tuple(signals_by_id.values()),
        connection_attempts=connection_attempts,
        successful_connections=successful_connections,
        reconnect_count=reconnect_count,
        idle_timeout_count=idle_timeout_count,
        invalid_message_count=invalid_message_count,
        received_message_count=received_message_count,
        stop_reason=stop_reason,
        elapsed_seconds=elapsed_seconds,
    )
