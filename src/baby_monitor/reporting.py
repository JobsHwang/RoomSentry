from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .database import EventDatabase, StoredEvent
from .presence_history import PresenceDaySummary, PresenceFactMode, read_presence_day_summary
from .recognition_attempt import RecognitionAttempt, RecognitionAttemptAuditStore
from .report_lock import report_publication_lock
from .reporting_rendering import render_html, render_markdown
from .storage import StorageResolver, secure_media_file


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    markdown_path: Path
    html_path: Path
    event_count: int


def _atomic_write_private_report(path: Path, content: str) -> None:
    """Replace one detailed report through a private, durable staging file."""

    payload = content.encode("utf-8")
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        secure_media_file(path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(OSError):
            temporary.unlink(missing_ok=True)


def _local_day_bounds(report_date: date, timezone_name: str) -> tuple[datetime, datetime]:
    timezone = ZoneInfo(timezone_name)
    start = datetime.combine(report_date, time.min, tzinfo=timezone)
    return start, start + timedelta(days=1)


class DailyReportGenerator:
    def __init__(
        self,
        *,
        database: EventDatabase,
        storage: StorageResolver,
        timezone_name: str,
    ):
        self.database = database
        self.storage = storage
        self.timezone_name = timezone_name

    def generate(
        self,
        report_date: date,
        *,
        claim_guard: Callable[[], None] | None = None,
        lock_timeout_seconds: float = 30.0,
    ) -> ReportArtifacts:
        """Read and publish under one per-destination lock, including sync hosts."""

        if type(report_date) is not date:
            raise ValueError("report_date must be a date")
        _local_day_bounds(report_date, self.timezone_name)
        markdown_path, html_path = self.storage.report_paths(report_date)
        with report_publication_lock(markdown_path.parent, timeout_seconds=lock_timeout_seconds):
            if claim_guard is not None:
                claim_guard()
            return self._generate_locked(
                report_date,
                markdown_path=markdown_path,
                html_path=html_path,
                claim_guard=claim_guard,
            )

    def _generate_locked(
        self,
        report_date: date,
        *,
        markdown_path: Path,
        html_path: Path,
        claim_guard: Callable[[], None] | None,
    ) -> ReportArtifacts:
        start, end = _local_day_bounds(report_date, self.timezone_name)
        events = self.database.fetch_between(start, end)
        attempts = RecognitionAttemptAuditStore(self.database.path).between(start, end)
        presence = read_presence_day_summary(
            self.database.path,
            local_date=report_date,
            timezone_name=self.timezone_name,
        )
        shadow_presence = read_presence_day_summary(
            self.database.path,
            local_date=report_date,
            timezone_name=self.timezone_name,
            mode=PresenceFactMode.SHADOW,
        )
        markdown = self._markdown(
            report_date,
            events,
            attempts,
            presence,
            shadow_presence,
            markdown_path.parent,
        )
        rendered_html = self._html(
            report_date,
            events,
            attempts,
            presence,
            shadow_presence,
            html_path.parent,
        )
        if claim_guard is not None:
            claim_guard()
        _atomic_write_private_report(markdown_path, markdown)
        _atomic_write_private_report(html_path, rendered_html)
        return ReportArtifacts(
            markdown_path=markdown_path,
            html_path=html_path,
            event_count=len(events),
        )

    def _markdown(
        self,
        report_date: date,
        events: list[StoredEvent],
        attempts: tuple[RecognitionAttempt, ...],
        presence: PresenceDaySummary | None,
        shadow_presence: PresenceDaySummary | None,
        report_dir: Path,
    ) -> str:
        return render_markdown(
            report_date,
            events,
            attempts,
            presence,
            shadow_presence,
            report_dir,
            timezone_name=self.timezone_name,
        )

    def _html(
        self,
        report_date: date,
        events: list[StoredEvent],
        attempts: tuple[RecognitionAttempt, ...],
        presence: PresenceDaySummary | None,
        shadow_presence: PresenceDaySummary | None,
        report_dir: Path,
    ) -> str:
        return render_html(
            report_date,
            events,
            attempts,
            presence,
            shadow_presence,
            report_dir,
            timezone_name=self.timezone_name,
        )
