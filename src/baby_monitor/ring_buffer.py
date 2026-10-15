from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from baby_monitor.adapters.camera import CameraSource, redact_uri
from baby_monitor.adapters.media import FFprobeMediaInspector, MediaError
from baby_monitor.domain import EventMedia, EvidenceGap
from baby_monitor.media_health import MediaSourceHealth, MediaSourceState
from baby_monitor.storage import StorageResolver, safe_component, secure_media_file


class RingBufferError(RuntimeError):
    pass


class BufferWindowUnavailableError(RingBufferError):
    """The requested time window has no currently retained segment overlap."""


_SEGMENT_NAME = re.compile(
    r"^segment_(?P<started>\d{8}T\d{6}Z)_(?P<session>[0-9a-f]{12})"
    r"(?:_(?P<index>\d{6}))?\.mkv$"
)


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("manifest timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _validate_rtsp_uri(uri: str, *, allow_non_loopback: bool) -> str:
    try:
        parts = urlsplit(uri)
    except ValueError as exc:
        raise ValueError("invalid RTSP buffer URI") from exc
    if parts.scheme.lower() not in {"rtsp", "rtsps"}:
        raise ValueError("buffer URI must use rtsp:// or rtsps://")
    if not parts.hostname or not parts.path.strip("/"):
        raise ValueError("buffer URI must include a host and stream path")
    if parts.username is not None or parts.password is not None:
        raise ValueError("RTSP credentials are disabled because process arguments are visible")
    if parts.query or parts.fragment:
        raise ValueError("RTSP buffer URI cannot contain query or fragment values")
    is_loopback = parts.hostname.lower() == "localhost" or parts.hostname in {
        "127.0.0.1",
        "::1",
    }
    if not allow_non_loopback and not is_loopback:
        raise ValueError("RTSP buffer URI must be loopback unless explicitly allowed")
    return redact_uri(uri)


@dataclass(frozen=True, slots=True)
class BufferSegment:
    filename: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    size_bytes: int
    session_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "started_at": _iso_utc(self.started_at),
            "ended_at": _iso_utc(self.ended_at),
            "duration_seconds": self.duration_seconds,
            "size_bytes": self.size_bytes,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> BufferSegment:
        if not isinstance(value, dict):
            raise ValueError("manifest segment must be an object")
        filename = str(value.get("filename", ""))
        if Path(filename).name != filename or _SEGMENT_NAME.fullmatch(filename) is None:
            raise ValueError("manifest segment filename is unsafe")
        started_at = _parse_utc(str(value.get("started_at", "")))
        ended_at = _parse_utc(str(value.get("ended_at", "")))
        duration = float(value.get("duration_seconds", 0))
        size = int(value.get("size_bytes", 0))
        session_id = str(value.get("session_id", ""))
        if ended_at <= started_at or not math.isfinite(duration) or duration <= 0 or size <= 0:
            raise ValueError("manifest segment metadata is invalid")
        if not re.fullmatch(r"[0-9a-f]{12}", session_id):
            raise ValueError("manifest session id is invalid")
        return cls(filename, started_at, ended_at, duration, size, session_id)


CoverageGap = EvidenceGap


@dataclass(frozen=True, slots=True)
class SegmentManifest:
    camera_id: str
    source_ref: str
    created_at: datetime
    updated_at: datetime
    segments: tuple[BufferSegment, ...] = ()
    gaps: tuple[CoverageGap, ...] = ()
    evicted_segments: int = 0
    version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "camera_id": self.camera_id,
            "source_ref": self.source_ref,
            "created_at": _iso_utc(self.created_at),
            "updated_at": _iso_utc(self.updated_at),
            "segments": [item.to_dict() for item in self.segments],
            "gaps": [item.to_dict() for item in self.gaps],
            "evicted_segments": self.evicted_segments,
        }

    @classmethod
    def read(cls, path: str | Path) -> SegmentManifest:
        manifest_path = Path(path).expanduser()
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RingBufferError("buffer manifest is missing or invalid") from exc
        if not isinstance(value, dict) or value.get("version") != 1:
            raise RingBufferError("unsupported buffer manifest version")
        try:
            camera_id = str(value["camera_id"])
            source_ref = str(value["source_ref"])
            created_at = _parse_utc(str(value["created_at"]))
            updated_at = _parse_utc(str(value["updated_at"]))
            segments = tuple(BufferSegment.from_dict(item) for item in value["segments"])
            gaps = tuple(CoverageGap.from_dict(item) for item in value["gaps"])
            evicted = int(value.get("evicted_segments", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise RingBufferError("buffer manifest has invalid fields") from exc
        if not camera_id or not source_ref or evicted < 0:
            raise RingBufferError("buffer manifest has invalid identity fields")
        return cls(
            camera_id=camera_id,
            source_ref=source_ref,
            created_at=created_at,
            updated_at=updated_at,
            segments=segments,
            gaps=gaps,
            evicted_segments=evicted,
        )

    def write_atomic(self, path: str | Path) -> None:
        manifest_path = Path(path).expanduser()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()


@dataclass(frozen=True, slots=True)
class SegmentBufferSettings:
    uri: str
    camera_id: str
    buffer_root: Path
    segment_seconds: float = 2.0
    max_segments: int = 30
    run_seconds: float = 30.0
    reconnect_initial_seconds: float = 0.5
    reconnect_max_seconds: float = 8.0
    poll_interval_seconds: float = 0.25
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    allow_non_loopback: bool = False
    eviction_enabled: bool = False
    eviction_confirmed: bool = False

    def __post_init__(self) -> None:
        _validate_rtsp_uri(self.uri, allow_non_loopback=self.allow_non_loopback)
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.camera_id) is None:
            raise ValueError("camera_id must be a safe non-personal identifier")
        if not 1.0 <= self.segment_seconds <= 60.0:
            raise ValueError("segment_seconds must be between 1 and 60")
        if not 3 <= self.max_segments <= 10_000:
            raise ValueError("max_segments must be between 3 and 10000")
        if not 3.0 <= self.run_seconds <= 86_400.0:
            raise ValueError("run_seconds must be between 3 and 86400")
        if not 0.1 <= self.reconnect_initial_seconds <= 60.0:
            raise ValueError("initial reconnect delay must be between 0.1 and 60 seconds")
        if not self.reconnect_initial_seconds <= self.reconnect_max_seconds <= 300.0:
            raise ValueError("maximum reconnect delay is invalid")
        if not 0.05 <= self.poll_interval_seconds <= 5.0:
            raise ValueError("poll interval must be between 0.05 and 5 seconds")
        if self.eviction_enabled and not self.eviction_confirmed:
            raise ValueError("buffer eviction requires persistent opt-in and one-shot confirmation")

    @property
    def audit_ref(self) -> str:
        return _validate_rtsp_uri(self.uri, allow_non_loopback=self.allow_non_loopback)

    @property
    def buffer_directory(self) -> Path:
        camera = safe_component(self.camera_id, fallback="camera")
        return self.buffer_root.expanduser() / camera

    @property
    def manifest_path(self) -> Path:
        return self.buffer_directory / "manifest.json"

    @property
    def health_path(self) -> Path:
        return self.buffer_directory / "health.json"

    def command(self, session_id: str, remaining_seconds: float) -> list[str]:
        session_started = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = self.buffer_directory / (f"segment_{session_started}_{session_id}_%06d.mkv")
        segment_list = self.buffer_directory / f"segments_{session_id}.csv"
        return [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-timeout",
            "5000000",
            "-i",
            self.uri,
            "-t",
            f"{remaining_seconds:.3f}",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-f",
            "segment",
            "-segment_time",
            f"{self.segment_seconds:g}",
            "-reset_timestamps",
            "1",
            "-segment_start_number",
            "0",
            "-segment_list",
            str(segment_list),
            "-segment_list_type",
            "csv",
            "-segment_list_flags",
            "+live",
            str(output),
        ]


@dataclass(frozen=True, slots=True)
class SegmentBufferResult:
    ok: bool
    source_ref: str
    manifest_path: str
    health_path: str
    started_at: str
    ended_at: str
    requested_seconds: float
    segments: int
    coverage_seconds: float
    gaps: int
    reconnects: int
    evicted_segments: int
    capacity_exceeded: bool
    ffmpeg_exit_code: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RTSPSegmentBuffer:
    def __init__(self, settings: SegmentBufferSettings):
        self.settings = settings
        self._known: dict[str, BufferSegment] = {}
        self._gaps: list[CoverageGap] = []
        self._created_at = datetime.now(UTC)
        self._evicted = 0
        self._run_started_at: datetime | None = None
        self._reconnects = 0
        self._last_exit: int | None = None
        if settings.manifest_path.exists():
            existing = SegmentManifest.read(settings.manifest_path)
            if (
                existing.camera_id != settings.camera_id
                or existing.source_ref != settings.audit_ref
            ):
                raise RingBufferError("existing manifest belongs to a different source")
            self._known = {item.filename: item for item in existing.segments}
            self._gaps = list(existing.gaps)
            self._created_at = existing.created_at
            self._evicted = existing.evicted_segments

    def _write_health(
        self,
        state: MediaSourceState,
        *,
        ended_at: datetime | None = None,
        last_error_code: str | None = None,
    ) -> None:
        if self._run_started_at is None:
            raise RingBufferError("buffer health session has not started")
        observed_at = datetime.now(UTC)
        health = MediaSourceHealth(
            camera_id=self.settings.camera_id,
            source_ref=self.settings.audit_ref,
            state=state,
            observed_at=observed_at,
            session_started_at=self._run_started_at,
            session_ended_at=ended_at,
            segments=len(self._known),
            coverage_seconds=round(sum(item.duration_seconds for item in self._known.values()), 3),
            gap_count=len(self._gaps),
            reconnect_count=self._reconnects,
            capacity_reached=state is MediaSourceState.CAPACITY_REACHED,
            ffmpeg_exit_code=self._last_exit,
            last_error_code=last_error_code,
        )
        health.write_atomic(self.settings.health_path)

    def _manifest(self) -> SegmentManifest:
        return SegmentManifest(
            camera_id=self.settings.camera_id,
            source_ref=self.settings.audit_ref,
            created_at=self._created_at,
            updated_at=datetime.now(UTC),
            segments=tuple(sorted(self._known.values(), key=lambda item: item.started_at)),
            gaps=tuple(self._gaps),
            evicted_segments=self._evicted,
        )

    def _refresh(self, session_id: str) -> bool:
        list_path = self.settings.buffer_directory / f"segments_{session_id}.csv"
        try:
            with list_path.open(newline="", encoding="utf-8") as handle:
                rows = tuple(csv.reader(handle))
        except OSError:
            rows = ()
        parsed: list[tuple[str, float, float, datetime]] = []
        for row in rows:
            if len(row) != 3:
                continue
            filename = row[0]
            match = _SEGMENT_NAME.fullmatch(filename)
            if match is None or match.group("session") != session_id:
                continue
            try:
                media_start = float(row[1])
                media_end = float(row[2])
                wall_name = datetime.strptime(match.group("started"), "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=UTC
                )
            except ValueError:
                continue
            if (
                not math.isfinite(media_start)
                or not math.isfinite(media_end)
                or media_start < 0
                or media_end <= media_start
            ):
                continue
            parsed.append((filename, media_start, media_end, wall_name))
        admitted_segment = False
        if parsed:
            anchor = parsed[0][3] - timedelta(seconds=parsed[0][1])
            for filename, media_start, media_end, _wall_name in parsed:
                if filename in self._known:
                    continue
                path = self.settings.buffer_directory / filename
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if path.is_symlink() or size <= 0:
                    continue
                duration = media_end - media_start
                self._known[filename] = BufferSegment(
                    filename=filename,
                    started_at=anchor + timedelta(seconds=media_start),
                    ended_at=anchor + timedelta(seconds=media_end),
                    duration_seconds=duration,
                    size_bytes=size,
                    session_id=session_id,
                )
                admitted_segment = True
        self._evict_if_allowed()
        self._manifest().write_atomic(self.settings.manifest_path)
        return admitted_segment

    def _evict_if_allowed(self) -> None:
        if not self.settings.eviction_enabled:
            return
        ordered = sorted(self._known.values(), key=lambda item: item.started_at)
        while len(ordered) > self.settings.max_segments:
            segment = ordered.pop(0)
            path = self.settings.buffer_directory / segment.filename
            resolved_root = self.settings.buffer_directory.resolve()
            resolved = path.resolve()
            if path.is_symlink() or not resolved.is_relative_to(resolved_root):
                raise RingBufferError("refusing to evict an unsafe buffer path")
            try:
                path.unlink()
            except OSError as exc:
                raise RingBufferError("could not evict an owned buffer segment") from exc
            self._known.pop(segment.filename, None)
            self._evicted += 1

    def _capacity_reached(self) -> bool:
        if self.settings.eviction_enabled:
            return False
        try:
            paths = self.settings.buffer_directory.iterdir()
        except OSError:
            return False
        owned_segments = 0
        for path in paths:
            if (
                _SEGMENT_NAME.fullmatch(path.name) is not None
                and not path.is_symlink()
                and path.is_file()
            ):
                owned_segments += 1
                if owned_segments >= self.settings.max_segments:
                    return True
        return False

    @staticmethod
    def _stop(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def run(self) -> SegmentBufferResult:
        self.settings.buffer_directory.mkdir(parents=True, exist_ok=True)
        self._run_started_at = datetime.now(UTC)
        self._reconnects = 0
        self._last_exit = None
        self._write_health(MediaSourceState.STARTING)
        try:
            return self._run_session()
        except BaseException as exc:
            with suppress(Exception):
                self._write_health(
                    MediaSourceState.FAILED,
                    ended_at=datetime.now(UTC),
                    last_error_code=type(exc).__name__,
                )
            raise

    def _run_session(self) -> SegmentBufferResult:
        command = self.settings.ffmpeg
        if shutil.which(command) is None and not Path(command).is_file():
            raise FileNotFoundError(f"required media command is missing: {command}")
        started_wall = self._run_started_at
        if started_wall is None:
            raise RingBufferError("buffer health session has not started")
        started = time.monotonic()
        deadline = started + self.settings.run_seconds
        backoff = self.settings.reconnect_initial_seconds
        capacity_reached = False

        with tempfile.TemporaryFile(mode="w+b") as error_log:
            while time.monotonic() < deadline:
                if self._capacity_reached():
                    capacity_reached = True
                    break
                session_id = uuid.uuid4().hex[:12]
                remaining = max(0.1, deadline - time.monotonic())
                session_produced_segment = False
                environment = os.environ.copy()
                environment.update({"TZ": "UTC", "LC_ALL": "C"})
                process = subprocess.Popen(
                    self.settings.command(session_id, remaining),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=error_log,
                    env=environment,
                )
                self._write_health(MediaSourceState.STARTING)
                try:
                    while process.poll() is None:
                        session_produced_segment = (
                            self._refresh(session_id) or session_produced_segment
                        )
                        self._write_health(
                            MediaSourceState.RUNNING
                            if session_produced_segment
                            else MediaSourceState.STARTING
                        )
                        if self._capacity_reached():
                            capacity_reached = True
                            self._stop(process)
                            break
                        if time.monotonic() >= deadline:
                            self._stop(process)
                            break
                        time.sleep(self.settings.poll_interval_seconds)
                except BaseException:
                    self._stop(process)
                    raise
                self._last_exit = process.wait(timeout=5)
                session_produced_segment = self._refresh(session_id) or session_produced_segment
                if capacity_reached:
                    break
                if time.monotonic() >= deadline - self.settings.poll_interval_seconds:
                    break

                self._reconnects += 1
                self._write_health(
                    MediaSourceState.DEGRADED,
                    last_error_code="stream_ended",
                )
                gap_start = datetime.now(UTC)
                sleep_seconds = min(backoff, max(0.0, deadline - time.monotonic()))
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                gap_end = datetime.now(UTC)
                if gap_end > gap_start:
                    self._gaps.append(CoverageGap(gap_start, gap_end, "rtsp_reconnect"))
                if session_produced_segment:
                    backoff = self.settings.reconnect_initial_seconds
                else:
                    backoff = min(self.settings.reconnect_max_seconds, backoff * 2)
                self._manifest().write_atomic(self.settings.manifest_path)
                self._write_health(
                    MediaSourceState.DEGRADED,
                    last_error_code="rtsp_reconnect",
                )

        manifest = self._manifest()
        manifest.write_atomic(self.settings.manifest_path)
        coverage = sum(item.duration_seconds for item in manifest.segments)
        capacity_exceeded = capacity_reached or len(manifest.segments) >= self.settings.max_segments
        ended_at = datetime.now(UTC)
        final_state = (
            MediaSourceState.CAPACITY_REACHED
            if capacity_exceeded
            else (MediaSourceState.STOPPED if manifest.segments else MediaSourceState.FAILED)
        )
        self._write_health(
            final_state,
            ended_at=ended_at,
            last_error_code=("no_segments" if not manifest.segments else None),
        )
        return SegmentBufferResult(
            ok=bool(manifest.segments),
            source_ref=self.settings.audit_ref,
            manifest_path=str(self.settings.manifest_path),
            health_path=str(self.settings.health_path),
            started_at=_iso_utc(started_wall),
            ended_at=_iso_utc(ended_at),
            requested_seconds=self.settings.run_seconds,
            segments=len(manifest.segments),
            coverage_seconds=round(coverage, 3),
            gaps=len(manifest.gaps),
            reconnects=self._reconnects,
            evicted_segments=manifest.evicted_segments,
            capacity_exceeded=capacity_exceeded,
            ffmpeg_exit_code=self._last_exit,
        )


@dataclass(frozen=True, slots=True)
class BufferedEvidenceResult:
    media: EventMedia
    requested_start: str
    requested_end: str
    covered_start: str
    covered_end: str
    complete_coverage: bool
    coverage_gaps: tuple[CoverageGap, ...]
    selected_segments: int
    reused_existing: bool

    def __post_init__(self) -> None:
        if self.complete_coverage != self.media.complete_coverage:
            raise ValueError("buffered evidence coverage state is inconsistent")
        if self.coverage_gaps != self.media.coverage_gaps:
            raise ValueError("buffered evidence gaps are inconsistent")
        if self.selected_segments != self.media.selected_segments:
            raise ValueError("buffered evidence segment count is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_path": self.media.clip_path,
            "frame_paths": list(self.media.frame_paths),
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "covered_start": self.covered_start,
            "covered_end": self.covered_end,
            "complete_coverage": self.complete_coverage,
            "coverage_status": self.media.coverage_status,
            "continuous_coverage_verified": False,
            "coverage_gaps": [item.to_dict() for item in self.coverage_gaps],
            "selected_segments": self.selected_segments,
            "reused_existing": self.reused_existing,
        }


def _coverage_gaps(
    segments: list[BufferSegment],
    start: datetime,
    end: datetime,
    *,
    tolerance_seconds: float = 0.25,
) -> tuple[CoverageGap, ...]:
    gaps: list[CoverageGap] = []
    cursor = start
    tolerance = timedelta(seconds=tolerance_seconds)
    for segment in segments:
        segment_start = max(start, segment.started_at)
        segment_end = min(end, segment.ended_at)
        if segment_end <= start or segment_start >= end:
            continue
        if segment_start > cursor + tolerance:
            gaps.append(CoverageGap(cursor, segment_start, "segment_missing"))
        cursor = max(cursor, segment_end)
    if cursor < end - tolerance:
        gaps.append(CoverageGap(cursor, end, "segment_missing"))
    return tuple(gaps)


def _concat_line(path: Path) -> str:
    escaped = str(path).replace("'", "'\\''")
    return f"file '{escaped}'\n"


def extract_buffered_evidence(
    *,
    manifest_path: str | Path,
    event_time: datetime,
    event_id: str,
    pre_event_seconds: float,
    post_event_seconds: float,
    keyframe_count: int,
    storage: StorageResolver,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> BufferedEvidenceResult:
    if event_time.tzinfo is None:
        raise ValueError("event_time must be timezone-aware")
    if not event_id.strip():
        raise ValueError("event_id cannot be empty")
    if not 0 <= pre_event_seconds <= 300 or not 0 <= post_event_seconds <= 300:
        raise ValueError("event window values must be between 0 and 300 seconds")
    if pre_event_seconds + post_event_seconds <= 0:
        raise ValueError("event window cannot be empty")
    if not 3 <= keyframe_count <= 8:
        raise ValueError("keyframe_count must be between 3 and 8")
    for command in (ffmpeg, ffprobe):
        if shutil.which(command) is None and not Path(command).is_file():
            raise FileNotFoundError(f"required media command is missing: {command}")

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = SegmentManifest.read(manifest_file)
    root = manifest_file.parent.resolve()
    requested_start = event_time.astimezone(UTC) - timedelta(seconds=pre_event_seconds)
    requested_end = event_time.astimezone(UTC) + timedelta(seconds=post_event_seconds)
    selected = [
        item
        for item in manifest.segments
        if item.ended_at > requested_start and item.started_at < requested_end
    ]
    selected.sort(key=lambda item: item.started_at)
    if not selected:
        raise BufferWindowUnavailableError("buffer has no segments for the requested event window")

    paths: list[Path] = []
    for segment in selected:
        path = root / segment.filename
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
            raise RingBufferError("manifest references an unsafe or missing segment")
        paths.append(resolved)

    gaps = _coverage_gaps(selected, requested_start, requested_end)
    covered_start = max(requested_start, selected[0].started_at)
    covered_end = min(requested_end, max(item.ended_at for item in selected))
    if covered_end <= covered_start:
        raise BufferWindowUnavailableError("buffer coverage does not overlap the event window")

    allocated = storage.allocate_media(
        occurred_at=event_time,
        camera_id=manifest.camera_id,
        event_id=event_id,
        frame_count=keyframe_count,
    )
    all_outputs = (allocated.clip_path, *allocated.frame_paths)
    if all(path.is_file() and path.stat().st_size > 0 for path in all_outputs):
        try:
            for path in all_outputs:
                secure_media_file(path)
        except ValueError as exc:
            raise RingBufferError("existing buffered evidence is unsafe") from exc
        return BufferedEvidenceResult(
            media=EventMedia(
                clip_path=str(allocated.clip_path),
                frame_paths=tuple(str(path) for path in allocated.frame_paths),
                coverage_gaps=gaps,
                selected_segments=len(selected),
            ),
            requested_start=_iso_utc(requested_start),
            requested_end=_iso_utc(requested_end),
            covered_start=_iso_utc(covered_start),
            covered_end=_iso_utc(covered_end),
            complete_coverage=not gaps,
            coverage_gaps=gaps,
            selected_segments=len(selected),
            reused_existing=True,
        )
    if any(path.exists() for path in all_outputs):
        raise RingBufferError("partial evidence already exists; refusing to overwrite")

    with tempfile.TemporaryDirectory(prefix="baby-monitor-buffer-concat-") as directory:
        concat_path = Path(directory) / "segments.txt"
        concat_path.write_text("".join(_concat_line(path) for path in paths), encoding="utf-8")
        offset = max(0.0, (covered_start - selected[0].started_at).total_seconds())
        duration = (covered_end - covered_start).total_seconds()
        completed = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-ss",
                f"{offset:.3f}",
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "libx264",
                # Preserve in-segment PTS gaps; this does not restore wallclock
                # gaps between files concatenated from different segments.
                "-fps_mode:v",
                "passthrough",
                "-enc_time_base:v",
                "1:1000000",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "2",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                "-n",
                str(allocated.clip_path),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=180,
            check=False,
        )
    if (
        completed.returncode != 0
        or not allocated.clip_path.is_file()
        or allocated.clip_path.stat().st_size == 0
    ):
        raise RingBufferError("FFmpeg could not assemble buffered evidence")
    try:
        secure_media_file(allocated.clip_path)
    except ValueError as exc:
        raise RingBufferError("assembled buffered evidence is unsafe") from exc

    clip_source = CameraSource(
        camera_id=manifest.camera_id,
        kind="file",
        read_uri=str(allocated.clip_path),
        audit_ref=str(allocated.clip_path),
    )
    try:
        metadata = FFprobeMediaInspector(ffprobe=ffprobe).probe(clip_source)
    except MediaError as exc:
        raise RingBufferError("assembled evidence is not readable") from exc
    frame_interval = 1.0 / metadata.average_fps if metadata.average_fps else 0.1
    end_guard = min(max(0.05, frame_interval + 0.001), metadata.duration_seconds / 2.0)
    last_timestamp = max(0.0, metadata.duration_seconds - end_guard)
    spacing = last_timestamp / (keyframe_count - 1)
    for index, output in enumerate(allocated.frame_paths):
        timestamp = spacing * index
        frame = subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(allocated.clip_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-n",
                str(output),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if frame.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            raise RingBufferError("FFmpeg could not extract a buffered evidence frame")
        try:
            secure_media_file(output)
        except ValueError as exc:
            raise RingBufferError("buffered evidence frame is unsafe") from exc

    return BufferedEvidenceResult(
        media=EventMedia(
            clip_path=str(allocated.clip_path),
            frame_paths=tuple(str(path) for path in allocated.frame_paths),
            coverage_gaps=gaps,
            selected_segments=len(selected),
        ),
        requested_start=_iso_utc(requested_start),
        requested_end=_iso_utc(requested_end),
        covered_start=_iso_utc(covered_start),
        covered_end=_iso_utc(covered_end),
        complete_coverage=not gaps,
        coverage_gaps=gaps,
        selected_segments=len(selected),
        reused_existing=False,
    )
