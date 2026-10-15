from __future__ import annotations

import ipaddress
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .domain import EventType, RiskLevel, VLMEvent

MAX_MOCK_REQUEST_BYTES = 8_500_000


class RemoteMockServer(HTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        capture_payload: bool = False,
    ):
        host = server_address[0]
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError("remote mock server may listen only on an IP loopback address")
        self.capture_payload = capture_payload
        self.last_request_payload: dict[str, Any] | None = None
        self.last_request_summary: dict[str, object] | None = None
        self.request_count = 0
        super().__init__(server_address, RemoteMockRequestHandler)


class RemoteMockRequestHandler(BaseHTTPRequestHandler):
    server: RemoteMockServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format_string: str, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _bad_request(self, code: str) -> None:
        self._send_json(
            400,
            {
                "error": {
                    "type": "invalid_request_error",
                    "code": code,
                    "message": "local fixture rejected the request",
                }
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(404, {"error": {"code": "not_found"}})
            return
        authorization = self.headers.get("Authorization", "")
        if not authorization.startswith("Bearer ") or len(authorization) <= len("Bearer "):
            self._send_json(401, {"error": {"code": "missing_fixture_authorization"}})
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._bad_request("invalid_content_length")
            return
        if not 1 <= content_length <= MAX_MOCK_REQUEST_BYTES:
            self._bad_request("request_size_out_of_bounds")
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise TypeError
            messages = payload["messages"]
            user_content = messages[-1]["content"]
            if not isinstance(messages, list) or not isinstance(user_content, list):
                raise TypeError
            image_items = [
                item
                for item in user_content
                if isinstance(item, dict) and item.get("type") == "image_url"
            ]
            if not 1 <= len(image_items) <= 3:
                raise ValueError
            if any(
                not isinstance(item.get("image_url"), dict)
                or not str(item["image_url"].get("url", "")).startswith("data:image/")
                for item in image_items
            ):
                raise ValueError
            response_format = payload.get("response_format")
            has_strict_schema = (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                and isinstance(response_format.get("json_schema"), dict)
                and response_format["json_schema"].get("strict") is True
            )
            if not has_strict_schema:
                raise ValueError
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ):
            self._bad_request("invalid_openai_compatible_payload")
            return

        self.server.request_count += 1
        self.server.last_request_summary = {
            "model": str(payload.get("model", "")),
            "image_count": len(image_items),
            "strict_json_schema": True,
            "request_bytes": content_length,
        }
        if self.server.capture_payload:
            self.server.last_request_payload = payload

        event = VLMEvent(
            event_type=EventType.UNKNOWN,
            subjects=(),
            observations=("本地模拟服务收到结构化复核请求。",),
            inferences=(),
            risk_level=RiskLevel.NONE,
            confidence=0.0,
            needs_remote_review=True,
            summary_zh="本地模拟响应，不代表视觉判断。",
            evidence_frames=("frame_001",),
        )
        self._send_json(
            200,
            {
                "id": "local_remote_mock",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": event.to_json(indent=None),
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )


def serve_remote_mock(*, host: str, port: int, max_requests: int) -> None:
    if type(port) is not int or not 0 <= port <= 65_535:
        raise ValueError("remote mock port must be between 0 and 65535")
    if type(max_requests) is not int or not 1 <= max_requests <= 1_000:
        raise ValueError("remote mock max_requests must be between 1 and 1000")
    with RemoteMockServer((host, port)) as server:
        actual_host, actual_port = server.server_address[:2]
        print(
            json.dumps(
                {
                    "status": "ready",
                    "base_url": f"http://{actual_host}:{actual_port}/v1",
                    "loopback_only": True,
                    "max_requests": max_requests,
                    "persists_request_media": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for _ in range(max_requests):
            server.handle_request()
