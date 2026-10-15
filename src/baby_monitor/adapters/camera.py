from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True, slots=True)
class CameraSource:
    camera_id: str
    kind: str
    read_uri: str
    audit_ref: str


def redact_uri(uri: str) -> str:
    """Remove credentials and query/fragment values before persistence or logs."""
    try:
        parts = urlsplit(uri)
    except ValueError:
        return "<invalid-uri>"
    if not parts.scheme:
        return str(Path(uri))
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


class CameraAdapter(ABC):
    @abstractmethod
    def source(self) -> CameraSource:
        raise NotImplementedError


class LocalFileCamera(CameraAdapter):
    def __init__(self, camera_id: str, path: str | Path):
        self.camera_id = camera_id
        self.path = Path(path).expanduser()

    def source(self) -> CameraSource:
        if not self.path.is_file():
            raise FileNotFoundError(f"input video does not exist: {self.path}")
        resolved = self.path.resolve()
        return CameraSource(
            camera_id=self.camera_id,
            kind="file",
            read_uri=str(resolved),
            audit_ref=str(resolved),
        )


class RTSPCamera(CameraAdapter):
    """Generic RTSP source; credentials are retained only in memory."""

    def __init__(self, camera_id: str, uri: str):
        if not uri.lower().startswith(("rtsp://", "rtsps://")):
            raise ValueError("RTSP source must use rtsp:// or rtsps://")
        self.camera_id = camera_id
        self.uri = uri

    def source(self) -> CameraSource:
        return CameraSource(
            camera_id=self.camera_id,
            kind="rtsp",
            read_uri=self.uri,
            audit_ref=redact_uri(self.uri),
        )


class ExternalBridgeCamera(CameraAdapter):
    """Vendor-neutral seam for a future Xiaomi/open-source bridge adapter."""

    def __init__(self, camera_id: str, upstream: CameraAdapter, bridge_name: str):
        self.camera_id = camera_id
        self.upstream = upstream
        self.bridge_name = bridge_name

    def source(self) -> CameraSource:
        source = self.upstream.source()
        return CameraSource(
            camera_id=self.camera_id,
            kind=f"bridge:{self.bridge_name}",
            read_uri=source.read_uri,
            audit_ref=source.audit_ref,
        )
