"""Length-bounded messages over an inherited local socket, including raw frames."""

from __future__ import annotations

import json
import socket
import struct

MAX_HEADER = 256 * 1024
MAX_PAYLOAD = 20 * 1024 * 1024


def _exact(channel: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = channel.recv(min(remaining, 1024 * 1024))
        if not chunk:
            raise RuntimeError("live detector channel closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(channel: socket.socket, header: dict[str, object], payload: bytes = b"") -> None:
    encoded = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    if not 0 < len(encoded) <= MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ValueError("live detector packet exceeds bounds")
    channel.sendall(struct.pack(">II", len(encoded), len(payload)))
    channel.sendall(encoded)
    if payload:
        channel.sendall(payload)


def receive_packet(channel: socket.socket) -> tuple[dict[str, object], bytes]:
    header_size, payload_size = struct.unpack(">II", _exact(channel, 8))
    if not 0 < header_size <= MAX_HEADER or payload_size > MAX_PAYLOAD:
        raise ValueError("live detector packet exceeds bounds")
    value = json.loads(
        _exact(channel, header_size),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    if not isinstance(value, dict):
        raise ValueError("invalid live detector header")
    return value, _exact(channel, payload_size)
