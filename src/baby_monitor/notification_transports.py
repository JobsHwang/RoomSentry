from __future__ import annotations

import ipaddress
import json
import os
import re
import smtplib
import ssl
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from .notification import (
    NotificationDeliveryError,
    NotificationDeliveryReceipt,
    NotificationMessage,
)

_SAFE_SINK_ID = re.compile(r"[a-z0-9_.-]{1,64}")
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,80}")
_FORBIDDEN_HEADERS = frozenset(
    {
        "content-length",
        "content-type",
        "host",
        "idempotency-key",
        "transfer-encoding",
        "x-baby-monitor-notification-id",
    }
)
_MAX_PROFILE_BYTES = 16_384
_MAX_WEBHOOK_RESPONSE_BYTES = 65_536


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("notification transport profile has duplicate fields")
        result[key] = value
    return result


def _strict_fields(data: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(data) != expected:
        raise ValueError(f"{context} fields do not match the v1 contract")


def _safe_sink_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_SINK_ID.fullmatch(value) is None:
        raise ValueError("notification transport sink_id is invalid")
    return value


def _bounded_string(
    value: object,
    *,
    field: str,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"notification transport {field} is invalid")
    if not allow_empty and not value:
        raise ValueError(f"notification transport {field} is invalid")
    if any(character in value for character in ("\r", "\n", "\x00")):
        raise ValueError(f"notification transport {field} is invalid")
    return value


def _positive_int(value: object, *, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"notification transport {field} is invalid")
    return value


def _mailbox(value: object, *, field: str) -> str:
    address = _bounded_string(value, field=field, maximum=254)
    if (
        address.count("@") != 1
        or address.startswith("@")
        or address.endswith("@")
        or any(character.isspace() for character in address)
        or any(character in address for character in "<>,;")
    ):
        raise ValueError(f"notification transport {field} is invalid")
    return address


def _literal_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _read_private_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path).expanduser()
    if not profile_path.is_absolute():
        raise ValueError("notification transport profile path must be absolute")
    try:
        metadata = profile_path.lstat()
    except OSError:
        raise ValueError("notification transport profile is unavailable") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("notification transport profile must be a regular non-symlink file")
    if metadata.st_size > _MAX_PROFILE_BYTES:
        raise ValueError("notification transport profile is too large")
    if os.name != "nt" and metadata.st_mode & 0o077:
        raise ValueError("notification transport profile must not be accessible by group or others")
    try:
        raw = profile_path.read_text(encoding="utf-8")
        data = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("notification transport profile is not strict JSON") from None
    if not isinstance(data, dict):
        raise ValueError("notification transport profile must be an object")
    return data


@dataclass(frozen=True, slots=True)
class SMTPNotificationProfile:
    sink_id: str
    host: str
    port: int
    security: str
    username: str
    credential: str = field(repr=False)
    from_address: str
    to_addresses: tuple[str, ...]
    timeout_seconds: int
    version: int = 1
    kind: str = "smtp"

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1 or self.kind != "smtp":
            raise ValueError("unsupported SMTP notification profile")
        _safe_sink_id(self.sink_id)
        host = _bounded_string(self.host, field="host", maximum=253)
        _positive_int(self.port, field="port", minimum=1, maximum=65_535)
        if self.security not in {"tls", "starttls", "loopback_plain"}:
            raise ValueError("notification transport SMTP security is invalid")
        if self.security == "loopback_plain" and not _literal_loopback(host):
            raise ValueError("plain SMTP is restricted to a literal loopback IP")
        username = _bounded_string(
            self.username,
            field="username",
            maximum=512,
            allow_empty=True,
        )
        smtp_credential = _bounded_string(
            self.credential,
            field="password",
            maximum=4_096,
            allow_empty=True,
        )
        if bool(username) != bool(smtp_credential):
            raise ValueError("SMTP username and password must be configured together")
        _mailbox(self.from_address, field="from_address")
        if not isinstance(self.to_addresses, tuple) or not 1 <= len(self.to_addresses) <= 5:
            raise ValueError("SMTP notification recipient count is invalid")
        if len(set(self.to_addresses)) != len(self.to_addresses):
            raise ValueError("SMTP notification recipients must be unique")
        for address in self.to_addresses:
            _mailbox(address, field="to_address")
        _positive_int(
            self.timeout_seconds,
            field="timeout_seconds",
            minimum=1,
            maximum=60,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SMTPNotificationProfile:
        _strict_fields(
            data,
            {
                "version",
                "kind",
                "sink_id",
                "host",
                "port",
                "security",
                "username",
                "password",
                "from_address",
                "to_addresses",
                "timeout_seconds",
            },
            "SMTP notification profile",
        )
        recipients = data["to_addresses"]
        if not isinstance(recipients, list):
            raise ValueError("SMTP notification recipients must be an array")
        return cls(
            version=data["version"],
            kind=data["kind"],
            sink_id=data["sink_id"],
            host=data["host"],
            port=data["port"],
            security=data["security"],
            username=data["username"],
            credential=data["password"],
            from_address=data["from_address"],
            to_addresses=tuple(recipients),
            timeout_seconds=data["timeout_seconds"],
        )

    def to_preview(self) -> dict[str, object]:
        return {
            "version": 1,
            "kind": self.kind,
            "sink_id": self.sink_id,
            "transport_security": self.security,
            "target_host": self.host,
            "recipient_count": len(self.to_addresses),
            "credentials_configured": bool(self.username),
            "attachments_allowed": False,
            "media_allowed": False,
            "paths_allowed": False,
            "network_used": False,
            "external_delivery_required": True,
        }


@dataclass(frozen=True, slots=True)
class WebhookNotificationProfile:
    sink_id: str
    url: str
    headers: tuple[tuple[str, str], ...]
    timeout_seconds: int
    version: int = 1
    kind: str = "webhook"

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1 or self.kind != "webhook":
            raise ValueError("unsupported webhook notification profile")
        _safe_sink_id(self.sink_id)
        url = _bounded_string(self.url, field="url", maximum=4_096)
        try:
            parsed = urlsplit(url)
            parsed_port = parsed.port
        except ValueError:
            raise ValueError("notification webhook URL is invalid") from None
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or parsed_port == 0
        ):
            raise ValueError("notification webhook URL is invalid")
        if parsed.scheme == "http" and not _literal_loopback(parsed.hostname):
            raise ValueError("HTTP webhook is restricted to a literal loopback IP")
        if not isinstance(self.headers, tuple) or len(self.headers) > 16:
            raise ValueError("notification webhook has too many headers")
        seen: set[str] = set()
        for item in self.headers:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("notification webhook header is invalid")
            name, value = item
            if not isinstance(name, str):
                raise ValueError("notification webhook header name is invalid")
            lowered = name.lower()
            if (
                _HEADER_NAME.fullmatch(name) is None
                or lowered in _FORBIDDEN_HEADERS
                or lowered in seen
            ):
                raise ValueError("notification webhook header name is invalid")
            _bounded_string(value, field="header value", maximum=4_096)
            seen.add(lowered)
        _positive_int(
            self.timeout_seconds,
            field="timeout_seconds",
            minimum=1,
            maximum=60,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WebhookNotificationProfile:
        _strict_fields(
            data,
            {"version", "kind", "sink_id", "url", "headers", "timeout_seconds"},
            "webhook notification profile",
        )
        headers = data["headers"]
        if not isinstance(headers, dict):
            raise ValueError("notification webhook headers must be an object")
        return cls(
            version=data["version"],
            kind=data["kind"],
            sink_id=data["sink_id"],
            url=data["url"],
            headers=tuple(headers.items()),
            timeout_seconds=data["timeout_seconds"],
        )

    def to_preview(self) -> dict[str, object]:
        parsed = urlsplit(self.url)
        return {
            "version": 1,
            "kind": self.kind,
            "sink_id": self.sink_id,
            "transport_security": parsed.scheme,
            "target_host": parsed.hostname,
            "custom_header_count": len(self.headers),
            "credentials_configured": bool(parsed.query or self.headers),
            "redirects_allowed": False,
            "proxy_inheritance_allowed": False,
            "media_allowed": False,
            "paths_allowed": False,
            "network_used": False,
            "external_delivery_required": True,
        }


NotificationTransportProfile = SMTPNotificationProfile | WebhookNotificationProfile


def load_notification_transport_profile(
    path: str | Path,
) -> NotificationTransportProfile:
    data = _read_private_profile(path)
    kind = data.get("kind")
    if kind == "smtp":
        return SMTPNotificationProfile.from_dict(data)
    if kind == "webhook":
        return WebhookNotificationProfile.from_dict(data)
    raise ValueError("notification transport profile kind is unsupported")


def _notification_text(message: NotificationMessage) -> str:
    lines = [
        f"通知 ID: {message.notification_id}",
        f"事件 ID: {message.event_id}",
        f"摄像头: {message.camera_id}",
        f"时间: {message.occurred_at.isoformat()}",
        f"状态: {message.review_status.value}",
        f"事件类型: {message.event_type.value}",
        f"风险: {message.risk_level.value}",
        f"置信度: {message.confidence:.3f}",
        "",
        "摘要:",
        message.summary_zh or "（无）",
        "",
        "检测事实:",
    ]
    lines.extend(f"- {item}" for item in message.observations)
    if not message.observations:
        lines.append("- （无）")
    lines.extend(("", "推测:"))
    lines.extend(f"- {item}" for item in message.inferences)
    if not message.inferences:
        lines.append("- （无）")
    lines.extend(("", "本通知不含图片、视频、音频或本机路径。"))
    return "\n".join(lines) + "\n"


@contextmanager
def _smtp_client(profile: SMTPNotificationProfile) -> Iterator[Any]:
    client: Any | None = None
    try:
        if profile.security == "tls":
            client = smtplib.SMTP_SSL(
                profile.host,
                profile.port,
                timeout=profile.timeout_seconds,
                context=ssl.create_default_context(),
            )
        else:
            client = smtplib.SMTP(
                profile.host,
                profile.port,
                timeout=profile.timeout_seconds,
            )
            if profile.security == "starttls":
                client.ehlo()
                client.starttls(context=ssl.create_default_context())
                client.ehlo()
        if profile.username:
            client.login(profile.username, profile.credential)
        yield client
    finally:
        if client is not None:
            with suppress(Exception):
                client.quit()


class SMTPNotificationSink:
    kind = "smtp"

    def __init__(
        self,
        profile: SMTPNotificationProfile,
        *,
        client_factory: Callable[[SMTPNotificationProfile], Any] = _smtp_client,
    ):
        self.profile = profile
        self.client_factory = client_factory

    def deliver(self, message: NotificationMessage) -> NotificationDeliveryReceipt:
        email = EmailMessage()
        email["From"] = self.profile.from_address
        email["To"] = ", ".join(self.profile.to_addresses)
        email["Subject"] = f"[baby-monitor] {message.risk_level.value} / {message.event_type.value}"
        email["X-Baby-Monitor-Notification-ID"] = message.notification_id
        email.set_content(_notification_text(message))
        try:
            with self.client_factory(self.profile) as client:
                client.send_message(email)
        except smtplib.SMTPAuthenticationError:
            raise NotificationDeliveryError("smtp_auth_error") from None
        except smtplib.SMTPRecipientsRefused:
            raise NotificationDeliveryError("smtp_recipient_rejected") from None
        except (OSError, smtplib.SMTPException, ssl.SSLError):
            raise NotificationDeliveryError("smtp_transport_error") from None
        return NotificationDeliveryReceipt(self.kind, reused_existing=False)


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del req, fp, code, msg, headers, newurl
        return None


class WebhookNotificationSink:
    kind = "webhook"

    def __init__(self, profile: WebhookNotificationProfile, *, opener: Any | None = None):
        self.profile = profile
        self.opener = opener or build_opener(ProxyHandler({}), _NoRedirectHandler())

    def deliver(self, message: NotificationMessage) -> NotificationDeliveryReceipt:
        payload = json.dumps(
            {"version": 1, "message": message.to_dict()},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = dict(self.profile.headers)
        headers.update(
            {
                "Content-Type": "application/json; charset=utf-8",
                "Idempotency-Key": message.notification_id,
                "X-Baby-Monitor-Notification-ID": message.notification_id,
            }
        )
        request = Request(
            self.profile.url,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with self.opener.open(
                request,
                timeout=self.profile.timeout_seconds,
            ) as response:
                status_code = getattr(response, "status", None)
                if status_code is None:
                    status_code = response.getcode()
                if not 200 <= int(status_code) < 300:
                    raise NotificationDeliveryError("webhook_http_error")
                response_body = response.read(_MAX_WEBHOOK_RESPONSE_BYTES + 1)
                if len(response_body) > _MAX_WEBHOOK_RESPONSE_BYTES:
                    raise NotificationDeliveryError("webhook_response_too_large")
        except NotificationDeliveryError:
            raise
        except HTTPError:
            raise NotificationDeliveryError("webhook_http_error") from None
        except (OSError, TimeoutError, URLError, ValueError):
            raise NotificationDeliveryError("webhook_transport_error") from None
        return NotificationDeliveryReceipt(self.kind, reused_existing=False)
