from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, time
from pathlib import Path

from .archive import ArchiveError, ArchiveManifest, build_archive_manifest
from .database import EventDatabase


class ArchiveBindingError(ArchiveError):
    pass


def _opaque_id(kind: str, identity: bytes) -> str:
    digest = hashlib.sha256(identity).hexdigest()[:32]
    return f"archive_{kind}_{digest}"


def build_event_archive_manifest(
    *,
    database: EventDatabase,
    event_id: str,
    source_root: str | Path,
) -> ArchiveManifest:
    if not isinstance(event_id, str) or not event_id:
        raise ArchiveBindingError("event_id cannot be empty")
    event = database.get_event(event_id)
    if event is None:
        raise ArchiveBindingError("event is missing from the local database")
    files = (event.evidence.clip_path, *event.evidence.frame_paths)
    if len(set(files)) != len(files):
        raise ArchiveBindingError("event media contains duplicate file references")
    return build_archive_manifest(
        archive_id=_opaque_id("event", event_id.encode("utf-8")),
        source_root=source_root,
        files=files,
        created_at=event.created_at,
    )


def build_report_archive_manifest(
    *,
    report_date: date,
    source_root: str | Path,
) -> ArchiveManifest:
    if not isinstance(report_date, date) or isinstance(report_date, datetime):
        raise ArchiveBindingError("report_date must be a date")
    root = Path(source_root).expanduser()
    report_directory = (
        root
        / "reports"
        / report_date.strftime("%Y")
        / report_date.strftime("%m")
        / report_date.strftime("%d")
    )
    stem = report_date.isoformat() + "_daily_report"
    provisional = build_archive_manifest(
        archive_id="archive_report_provisional",
        source_root=root,
        files=(report_directory / f"{stem}.md", report_directory / f"{stem}.html"),
        created_at=datetime.combine(report_date, time.min, tzinfo=UTC),
    )
    identity = bytearray(report_date.isoformat().encode("ascii"))
    for item in provisional.items:
        identity.extend(b"\0")
        identity.extend(item.relative_path.encode("utf-8"))
        identity.extend(b"\0")
        identity.extend(str(item.size_bytes).encode("ascii"))
        identity.extend(b"\0")
        identity.extend(item.sha256.encode("ascii"))
    return ArchiveManifest(
        archive_id=_opaque_id("report", bytes(identity)),
        created_at=provisional.created_at,
        items=provisional.items,
    )
