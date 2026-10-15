"""Loopback HTTP server and request handler wiring for the dashboard."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from baby_monitor.home_dashboard.application import (
    HomeDashboardApplication,
    HomeDashboardError,
)
from baby_monitor.home_dashboard.copy import (
    _ACTIONS,
    _BYTE_RANGE,
    _MAX_FORM_BYTES,
    _REVIEW_ALLOWED_ANSWERS,
    _REVIEW_TOKEN,
)
from baby_monitor.home_dashboard.rendering import (
    _STYLE,
    render_daily_digest_video,
    render_home_controls,
    render_home_dashboard,
    render_home_review,
)


class HomeDashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        server_address: tuple[str, int],
        application: HomeDashboardApplication,
    ) -> None:
        self.application = application
        self.request_count = 0
        self.media_request_count = 0
        super().__init__(server_address, HomeDashboardRequestHandler)

    @property
    def canonical_origin(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class HomeDashboardRequestHandler(BaseHTTPRequestHandler):
    server: HomeDashboardServer
    server_version = "BabyMonitorDashboard/1"
    sys_version = ""

    def log_message(self, _format: str, *args: object) -> None:
        return

    def _send(
        self,
        status: HTTPStatus,
        payload: bytes,
        *,
        content_type: str,
        set_control_session: bool = False,
        location: str | None = None,
        accept_ranges: bool = False,
        content_range: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'self'; img-src 'self'; "
            "media-src 'self'; form-action 'self'; base-uri 'none'; "
            "frame-ancestors 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        if set_control_session:
            self.send_header(
                "Set-Cookie",
                f"bm_control={self.server.application.control_session_token}; "
                "HttpOnly; SameSite=Strict; Path=/",
            )
        if location is not None:
            self.send_header("Location", location)
        if accept_ranges:
            self.send_header("Accept-Ranges", "bytes")
        if content_range is not None:
            self.send_header("Content-Range", content_range)
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _requested_media_range(self, size: int) -> tuple[int, int] | None:
        supplied = self.headers.get("Range")
        if supplied is None:
            return None
        if not isinstance(supplied, str) or len(supplied) > 80:
            raise HomeDashboardError("home dashboard media range is invalid")
        matched = _BYTE_RANGE.fullmatch(supplied)
        if matched is None or not any(matched.groups()):
            raise HomeDashboardError("home dashboard media range is invalid")
        start_text, end_text = matched.groups()
        if len(start_text) > 18 or len(end_text) > 18:
            raise HomeDashboardError("home dashboard media range is invalid")
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise HomeDashboardError("home dashboard media range is invalid")
            return max(0, size - suffix), size - 1
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
        if start >= size or end < start:
            raise HomeDashboardError("home dashboard media range is invalid")
        return start, min(end, size - 1)

    def _valid_host(self) -> bool:
        expected = self.server.canonical_origin.removeprefix("http://")
        return secrets.compare_digest(self.headers.get("Host", ""), expected)

    def _valid_control_session(self) -> bool:
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except CookieError:
            return False
        supplied = cookie.get("bm_control")
        return supplied is not None and secrets.compare_digest(
            supplied.value,
            self.server.application.control_session_token,
        )

    def do_GET(self) -> None:  # noqa: N802
        self.server.request_count += 1
        if not self._valid_host():
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        path = urlsplit(self.path)
        if path.query or path.fragment:
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        if path.path == "/style.css":
            self._send(HTTPStatus.OK, _STYLE, content_type="text/css; charset=utf-8")
            return
        if path.path == "/daily-report":
            if not self.server.application.daily_report_enabled:
                self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
                return
            provider = self.server.application.daily_report_provider
            assert provider is not None
            try:
                payload = provider()
            except Exception:
                self._send(
                    HTTPStatus.NOT_FOUND,
                    b"report unavailable",
                    content_type="text/plain",
                )
                return
            if (
                not isinstance(payload, bytes)
                or not 0 < len(payload) <= 256 * 1024
                or not payload.startswith(b"<!doctype html>")
            ):
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    b"report unavailable",
                    content_type="text/plain",
                )
                return
            self._send(
                HTTPStatus.OK,
                payload,
                content_type="text/html; charset=utf-8",
            )
            return
        if path.path.startswith("/daily-digest-video/"):
            if not self.server.application.daily_digest_enabled:
                self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
                return
            review_token = path.path.removeprefix("/daily-digest-video/")
            if _REVIEW_TOKEN.fullmatch(review_token) is None:
                self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
                return
            if not self._valid_control_session():
                self._send(HTTPStatus.FORBIDDEN, b"forbidden", content_type="text/plain")
                return
            provider = self.server.application.daily_digest_video_provider
            assert provider is not None
            try:
                payload = provider(review_token)
            except Exception:
                self._send(
                    HTTPStatus.NOT_FOUND,
                    b"digest unavailable",
                    content_type="text/plain",
                )
                return
            if not isinstance(payload, bytes) or not 0 < len(payload) <= 64 * 1024 * 1024:
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    b"digest unavailable",
                    content_type="text/plain",
                )
                return
            try:
                requested_range = self._requested_media_range(len(payload))
            except HomeDashboardError:
                self._send(
                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    b"range unavailable",
                    content_type="text/plain",
                    accept_ranges=True,
                    content_range=f"bytes */{len(payload)}",
                )
                return
            self.server.media_request_count += 1
            if requested_range is None:
                self._send(
                    HTTPStatus.OK,
                    payload,
                    content_type="video/mp4",
                    accept_ranges=True,
                )
                return
            start, end = requested_range
            self._send(
                HTTPStatus.PARTIAL_CONTENT,
                payload[start : end + 1],
                content_type="video/mp4",
                accept_ranges=True,
                content_range=f"bytes {start}-{end}/{len(payload)}",
            )
            return
        if path.path.startswith("/review-media/"):
            if not self.server.application.review_enabled:
                self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
                return
            if not self._valid_control_session():
                self._send(HTTPStatus.FORBIDDEN, b"forbidden", content_type="text/plain")
                return
            parts = path.path.split("/")
            if (
                len(parts) != 5
                or parts[:2] != ["", "review-media"]
                or _REVIEW_TOKEN.fullmatch(parts[2]) is None
                or parts[3] not in {"frame", "clip"}
                or not parts[4].isascii()
                or not parts[4].isdigit()
                or len(parts[4]) > 2
            ):
                self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
                return
            provider = self.server.application.review_media_provider
            assert provider is not None
            try:
                content_type, payload = provider(parts[2], parts[3], int(parts[4]))
            except Exception:
                self._send(
                    HTTPStatus.NOT_FOUND,
                    b"media unavailable",
                    content_type="text/plain",
                )
                return
            if (
                content_type
                not in {
                    "image/jpeg",
                    "image/png",
                    "video/mp4",
                    "video/x-matroska",
                }
                or not isinstance(payload, bytes)
                or not 0 < len(payload) <= 64 * 1024 * 1024
            ):
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    b"media unavailable",
                    content_type="text/plain",
                )
                return
            try:
                requested_range = self._requested_media_range(len(payload))
            except HomeDashboardError:
                self._send(
                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    b"range unavailable",
                    content_type="text/plain",
                    accept_ranges=True,
                    content_range=f"bytes */{len(payload)}",
                )
                return
            self.server.media_request_count += 1
            if requested_range is None:
                self._send(
                    HTTPStatus.OK,
                    payload,
                    content_type=content_type,
                    accept_ranges=True,
                )
                return
            start, end = requested_range
            self._send(
                HTTPStatus.PARTIAL_CONTENT,
                payload[start : end + 1],
                content_type=content_type,
                accept_ranges=True,
                content_range=f"bytes {start}-{end}/{len(payload)}",
            )
            return
        daily_digest_token: str | None = None
        is_daily_digest_page = path.path == "/daily-digest"
        if path.path.startswith("/daily-digest/"):
            daily_digest_token = path.path.removeprefix("/daily-digest/")
            if _REVIEW_TOKEN.fullmatch(daily_digest_token) is None:
                self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
                return
            is_daily_digest_page = True
        if path.path not in {"/", "/control", "/review"} and not is_daily_digest_page:
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        if path.path == "/review" and not self.server.application.review_enabled:
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        if is_daily_digest_page and not self.server.application.daily_digest_enabled:
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        try:
            if path.path == "/":
                page = render_home_dashboard(self.server.application.status_provider())
                set_control_session = False
            elif path.path == "/control":
                page = render_home_controls(
                    self.server.application.control_session_token,
                    self.server.application.issue_control_nonces(),
                    notification_test_enabled=(self.server.application.notification_test_enabled),
                    daily_report_enabled=(self.server.application.daily_report_enabled),
                    daily_digest_enabled=(self.server.application.daily_digest_generation_enabled),
                )
                set_control_session = True
            elif is_daily_digest_page:
                provider = self.server.application.daily_digest_catalog_provider
                assert provider is not None
                entries = provider(14)
                if daily_digest_token is not None and not any(
                    isinstance(entry, dict)
                    and isinstance(entry.get("review_token"), str)
                    and secrets.compare_digest(entry["review_token"], daily_digest_token)
                    for entry in entries
                ):
                    self._send(
                        HTTPStatus.NOT_FOUND,
                        b"not found",
                        content_type="text/plain",
                    )
                    return
                page = render_daily_digest_video(
                    entries,
                    selected_token=daily_digest_token,
                )
                set_control_session = True
            else:
                provider = self.server.application.review_provider
                assert provider is not None
                calibration_provider = self.server.application.review_calibration_provider
                items = provider(6)
                calibration = None if calibration_provider is None else calibration_provider()
                page = render_home_review(
                    items,
                    self.server.application.control_session_token,
                    self.server.application.issue_review_nonces(items),
                    calibration=calibration,
                )
                set_control_session = True
        except Exception:
            self._send(
                HTTPStatus.SERVICE_UNAVAILABLE,
                b"status unavailable",
                content_type="text/plain; charset=utf-8",
            )
            return
        self._send(
            HTTPStatus.OK,
            page,
            content_type="text/html; charset=utf-8",
            set_control_session=set_control_session,
        )

    def do_POST(self) -> None:  # noqa: N802
        self.server.request_count += 1
        path = urlsplit(self.path)
        allowed_path = path.path in {"/action", "/review-label"}
        if (
            not self._valid_host()
            or not allowed_path
            or path.query
            or path.fragment
            or path.path == "/review-label"
            and not self.server.application.review_enabled
        ):
            self._send(HTTPStatus.NOT_FOUND, b"not found", content_type="text/plain")
            return
        if (
            not secrets.compare_digest(
                self.headers.get("Origin", ""),
                self.server.canonical_origin,
            )
            or not self._valid_control_session()
        ):
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", content_type="text/plain")
            return
        if self.headers.get_content_type() != "application/x-www-form-urlencoded":
            self._send(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                b"form required",
                content_type="text/plain",
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= _MAX_FORM_BYTES:
            self._send(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                b"invalid form",
                content_type="text/plain",
            )
            return
        max_fields = 4 if path.path == "/action" else 5
        try:
            values = parse_qs(
                self.rfile.read(length).decode("ascii"),
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=max_fields,
            )
        except (UnicodeError, ValueError):
            self._send(HTTPStatus.BAD_REQUEST, b"invalid form", content_type="text/plain")
            return
        if any(len(value) != 1 for value in values.values()):
            self._send(HTTPStatus.BAD_REQUEST, b"invalid form", content_type="text/plain")
            return
        if path.path == "/review-label":
            expected_fields = {
                "csrf",
                "nonce",
                "review_token",
                "predicate",
                "human_answer",
            }
            if set(values) != expected_fields:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    b"invalid form",
                    content_type="text/plain",
                )
                return
            csrf = values["csrf"][0]
            nonce = values["nonce"][0]
            review_token = values["review_token"][0]
            predicate = values["predicate"][0]
            human_answer = values["human_answer"][0]
            if (
                not secrets.compare_digest(
                    csrf,
                    self.server.application.control_session_token,
                )
                or _REVIEW_TOKEN.fullmatch(review_token) is None
                or predicate not in _REVIEW_ALLOWED_ANSWERS
                or human_answer not in _REVIEW_ALLOWED_ANSWERS[predicate]
            ):
                self._send(HTTPStatus.BAD_REQUEST, b"invalid form", content_type="text/plain")
                return
            if not self.server.application.consume_review_nonce(
                review_token,
                predicate,
                nonce,
            ):
                self._send(HTTPStatus.CONFLICT, b"nonce consumed", content_type="text/plain")
                return
            recorder = self.server.application.review_label_recorder
            assert recorder is not None
            try:
                recorder(review_token, predicate, human_answer)
            except Exception:
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    b"review failed",
                    content_type="text/plain",
                )
                return
            self._send(
                HTTPStatus.SEE_OTHER,
                b"",
                content_type="text/plain",
                location="/review",
            )
            return
        if set(values) != {"csrf", "nonce", "action", "confirmation"}:
            self._send(HTTPStatus.BAD_REQUEST, b"invalid form", content_type="text/plain")
            return
        csrf = values["csrf"][0]
        nonce = values["nonce"][0]
        action = values["action"][0]
        confirmation = values["confirmation"][0]
        expected = _ACTIONS.get(action)
        if (
            expected is None
            or not secrets.compare_digest(
                csrf,
                self.server.application.control_session_token,
            )
            or (
                action == "notification-test"
                and not self.server.application.notification_test_enabled
            )
            or action == "daily-report"
            and not self.server.application.daily_report_enabled
            or action == "daily-digest"
            and not self.server.application.daily_digest_generation_enabled
        ):
            self._send(HTTPStatus.FORBIDDEN, b"forbidden", content_type="text/plain")
            return
        if not self.server.application.consume_control_nonce(action, nonce):
            self._send(HTTPStatus.CONFLICT, b"nonce consumed", content_type="text/plain")
            return
        if not secrets.compare_digest(confirmation.upper(), expected):
            self._send(
                HTTPStatus.BAD_REQUEST,
                b"confirmation required",
                content_type="text/plain",
            )
            return
        try:
            if action == "notification-test":
                executor = self.server.application.notification_test_executor
                assert executor is not None
                executor()
            elif action == "daily-report":
                executor = self.server.application.daily_report_executor
                assert executor is not None
                executor()
            elif action == "daily-digest":
                executor = self.server.application.daily_digest_executor
                assert executor is not None
                executor()
            else:
                self.server.application.action_executor(action, bool(expected))
        except Exception:
            self._send(
                HTTPStatus.SERVICE_UNAVAILABLE,
                b"action failed",
                content_type="text/plain",
            )
            return
        self._send(
            HTTPStatus.SEE_OTHER,
            b"",
            content_type="text/plain",
            location=(
                "/"
                if action == "notification-test"
                else "/daily-report"
                if action == "daily-report"
                else "/daily-digest"
                if action == "daily-digest"
                else "/control"
            ),
        )


def make_home_dashboard_server(
    application: HomeDashboardApplication,
    *,
    port: int,
) -> HomeDashboardServer:
    if type(port) is not int or not 0 <= port <= 65_535:
        raise HomeDashboardError("home dashboard port is invalid")
    try:
        return HomeDashboardServer(("127.0.0.1", port), application)
    except OSError as exc:
        raise HomeDashboardError("home dashboard could not bind loopback") from exc


def serve_home_dashboard(
    application: HomeDashboardApplication,
    *,
    port: int = 8765,
    max_runtime_seconds: float = 28_800.0,
    ready: Callable[[int], None] | None = None,
) -> dict[str, object]:
    if type(port) is not int or not 1_024 <= port <= 65_535:
        raise HomeDashboardError("home dashboard port is invalid")
    if not isinstance(max_runtime_seconds, (int, float)) or not (
        60.0 <= float(max_runtime_seconds) <= 86_400.0
    ):
        raise HomeDashboardError("home dashboard max runtime is invalid")
    if ready is not None and not callable(ready):
        raise HomeDashboardError("home dashboard ready callback is invalid")
    server = make_home_dashboard_server(application, port=port)
    server.timeout = 0.5
    started_at = time.monotonic()
    try:
        if ready is not None:
            ready(server.server_address[1])
        while time.monotonic() - started_at < float(max_runtime_seconds):
            server.handle_request()
    finally:
        server.server_close()
    return {
        "version": 1,
        "status": "stopped",
        "requests": server.request_count,
        "media_reads": server.media_request_count,
        "bind": "loopback",
        "remote_used": False,
        "media_read": server.media_request_count > 0,
        "paths_disclosed": False,
    }
