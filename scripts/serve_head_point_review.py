"""Serve a frozen head-point page on loopback, without write/media-list routes."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from scripts.prepare_head_point_review import read_bundle


class BoundedServer(HTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(5)
        return connection, address


def handler_type(page: bytes):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, _format, *_args):
            pass

        def allowed(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origins = {f"http://{host}" for host in hosts}
            host_values = self.headers.get_all("Host", [])
            origin_values = self.headers.get_all("Origin", [])
            return (
                len(host_values) == 1
                and host_values[0] in hosts
                and (not origin_values or (len(origin_values) == 1 and origin_values[0] in origins))
                and self.headers.get("Sec-Fetch-Site") != "cross-site"
            )

        def reply(self, status, body, content_type="text/plain; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            with contextlib.suppress(OSError):
                self.wfile.write(body)

        def do_GET(self):
            self.connection.settimeout(5)
            if not self.allowed():
                self.reply(403, b"review request rejected\n")
            elif self.path == "/review":
                self.reply(200, page, "text/html; charset=utf-8")
            elif self.path == "/health":
                self.reply(200, b"local frozen review ready\n")
            else:
                self.reply(404, b"not found\n")

        def do_POST(self):
            self.reply(405, b"no review submission route\n")

    return Handler


def serve(root: Path, *, port=0, seconds=3600):
    if (
        type(port) is not int
        or not 0 <= port <= 65535
        or type(seconds) is not int
        or not 1 <= seconds <= 86400
    ):
        raise ValueError("invalid local review server limits")
    manifest, package, page = read_bundle(root)
    with BoundedServer(("127.0.0.1", port), handler_type(page)) as server:
        server.timeout = 1
        print(
            json.dumps(
                {
                    "url": f"http://127.0.0.1:{server.server_port}/review",
                    "item_count": len(manifest["items"]),
                    "synthetic": package["synthetic"],
                    "expires_after_seconds": seconds,
                }
            ),
            flush=True,
        )
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            server.handle_request()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        serve(args.review_root, port=args.port, seconds=args.seconds)
        return 0
    except (ValueError, OSError):
        print(json.dumps({"ok": False, "error_code": "head_point_review_unavailable"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
