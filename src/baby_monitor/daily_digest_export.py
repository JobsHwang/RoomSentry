from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import signal
import stat
import subprocess
import tempfile
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from zoneinfo import ZoneInfo

from .daily_digest import DailyDigestItem, DailyDigestPlan
from .daily_digest_source import DailyDigestAdmission, DailyDigestMediaSource
from .daily_digest_store import DailyDigestStoreError, load_daily_digest_plan
from .storage import secure_media_file

_ERROR = "daily digest export failed"
_MAX_EXPORTS = 400
_MAX_OUTPUT_BYTES = 512 * 1024 * 1024
_MAX_DASHBOARD_VIDEO_BYTES = 64 * 1024 * 1024
_BUNDLE_NAME = re.compile(r"\d{4}-\d{2}-\d{2}_[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RECEIPT_KEYS = {
    "version",
    "digest_id",
    "local_date",
    "plan_sha256",
    "video_sha256",
    "output_bytes",
    "output_duration_seconds",
    "selected_event_count",
    "selected_duration_seconds",
    "facts_only",
    "identity_inference",
    "risk_inference",
    "notification_decision",
    "external_delivery",
    "video_generated",
}


class DailyDigestExportError(RuntimeError):
    """Fixed, path-free export error."""


@dataclass(frozen=True, slots=True)
class DailyDigestExportReceipt:
    local_date: str
    selected_event_count: int
    selected_duration_seconds: float
    output_duration_seconds: float
    output_bytes: int
    reused_existing: bool

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": "exported",
            "local_date": self.local_date,
            "selected_event_count": self.selected_event_count,
            "selected_duration_seconds": self.selected_duration_seconds,
            "output_duration_seconds": self.output_duration_seconds,
            "output_bytes": self.output_bytes,
            "reused_existing": self.reused_existing,
            "facts_only": True,
            "media_read": True,
            "media_content_decoded": True,
            "video_generated": True,
            "paths_disclosed": False,
            "event_ids_disclosed": False,
            "camera_ids_disclosed": False,
            "identity_inference": False,
            "risk_inference": False,
            "notification_decision": False,
            "external_delivery": False,
        }


@dataclass(frozen=True, slots=True)
class _DailyDigestCatalogRecord:
    review_token: str
    plan: DailyDigestPlan = dataclass_field(repr=False)
    bundle: Path = dataclass_field(repr=False)
    receipt: dict[str, object] = dataclass_field(repr=False)
    bundle_mtime_ns: int


class DailyDigestCatalog:
    def __init__(
        self,
        *,
        database_path: str | Path,
        plan_root: str | Path,
        export_root: str | Path,
        token_secret: bytes,
    ):
        if not isinstance(token_secret, bytes) or len(token_secret) < 32:
            raise DailyDigestExportError(_ERROR)
        self._database_path = Path(database_path).expanduser()
        self._plan_root = Path(plan_root).expanduser()
        self._export_root = Path(export_root).expanduser()
        self._token_secret = token_secret

    def _token(self, local_date: str, digest_id: str) -> str:
        binding = f"daily-digest-v1\0{local_date}\0{digest_id}".encode("ascii")
        return (
            urlsafe_b64encode(hmac.new(self._token_secret, binding, hashlib.sha256).digest())
            .rstrip(b"=")
            .decode("ascii")
        )

    def _records(self) -> tuple[_DailyDigestCatalogRecord, ...]:
        root = _private_root(self._database_path, self._export_root, create=False)
        if root is None:
            return ()
        try:
            bundles = sorted(root.iterdir())
            if len(bundles) > _MAX_EXPORTS:
                raise DailyDigestExportError(_ERROR)
            records: list[_DailyDigestCatalogRecord] = []
            for bundle in bundles:
                bundle_metadata = bundle.lstat()
                matched = _BUNDLE_NAME.fullmatch(bundle.name)
                if (
                    matched is None
                    or bundle.is_symlink()
                    or not stat.S_ISDIR(bundle_metadata.st_mode)
                    or bundle_metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(bundle_metadata.st_mode) != 0o700
                ):
                    raise DailyDigestExportError(_ERROR)
                local_date = bundle.name[:10]
                digest_id = bundle.name[11:]
                raw, receipt = _read_private_receipt(bundle / "receipt.json")
                video_metadata = (bundle / "digest.mp4").lstat()
                plan = load_daily_digest_plan(
                    database_path=self._database_path,
                    output_root=self._plan_root,
                    local_date=local_date,
                    digest_id=digest_id,
                )
                expected_plan_sha = hashlib.sha256(
                    (plan.to_json(indent=2) + "\n").encode("utf-8")
                ).hexdigest()
                if (
                    raw != json.dumps(receipt, sort_keys=True, indent=2) + "\n"
                    or receipt["local_date"] != local_date
                    or receipt["digest_id"] != digest_id
                    or receipt["plan_sha256"] != expected_plan_sha
                    or receipt["selected_event_count"] != plan.selected_event_count
                    or not math.isclose(
                        float(receipt["selected_duration_seconds"]),
                        plan.selected_duration_seconds,
                        abs_tol=1e-6,
                    )
                    or (bundle / "digest.mp4").is_symlink()
                    or not stat.S_ISREG(video_metadata.st_mode)
                    or video_metadata.st_uid != os.geteuid()
                    or video_metadata.st_nlink != 1
                    or stat.S_IMODE(video_metadata.st_mode) != 0o600
                    or video_metadata.st_size != receipt["output_bytes"]
                ):
                    raise DailyDigestExportError(_ERROR)
                records.append(
                    _DailyDigestCatalogRecord(
                        review_token=self._token(local_date, digest_id),
                        plan=plan,
                        bundle=bundle,
                        receipt=receipt,
                        bundle_mtime_ns=bundle_metadata.st_mtime_ns,
                    )
                )
        except (OSError, DailyDigestStoreError, DailyDigestExportError):
            raise DailyDigestExportError(_ERROR) from None
        latest_by_date: dict[str, _DailyDigestCatalogRecord] = {}
        for record in records:
            existing = latest_by_date.get(record.plan.local_date)
            if existing is None or (record.bundle_mtime_ns, record.plan.digest_id) > (
                existing.bundle_mtime_ns,
                existing.plan.digest_id,
            ):
                latest_by_date[record.plan.local_date] = record
        return tuple(
            sorted(
                latest_by_date.values(),
                key=lambda item: (item.plan.local_date, item.plan.digest_id),
                reverse=True,
            )
        )

    def list_public(self, *, limit: int = 14) -> tuple[dict[str, object], ...]:
        if type(limit) is not int or not 1 <= limit <= 30:
            raise DailyDigestExportError(_ERROR)
        public: list[dict[str, object]] = []
        for record in self._records()[:limit]:
            timezone = ZoneInfo(record.plan.timezone)
            public.append(
                {
                    "review_token": record.review_token,
                    "local_date": record.plan.local_date,
                    "selected_event_count": record.plan.selected_event_count,
                    "selected_duration_seconds": record.plan.selected_duration_seconds,
                    "output_duration_seconds": float(record.receipt["output_duration_seconds"]),
                    "target_met": record.plan.target_met,
                    "selected_camera_count": len(record.plan.coverage.selected_by_camera),
                    "selected_hours": [
                        int(hour) for hour, _count in record.plan.coverage.selected_by_hour
                    ],
                    "gap_reason_counts": dict(record.plan.coverage.gap_reason_counts),
                    "items": [
                        {
                            "rank": item.rank,
                            "local_time": item.occurred_at.astimezone(timezone).strftime(
                                "%H:%M:%S"
                            ),
                            "duration_seconds": item.duration_seconds,
                            "selection_score": item.selection_score,
                            "selection_reasons": list(item.selection_reasons),
                        }
                        for item in record.plan.items
                    ],
                    "facts_only": True,
                    "paths_disclosed": False,
                    "event_ids_disclosed": False,
                    "camera_ids_disclosed": False,
                    "identity_inference": False,
                    "risk_inference": False,
                    "notification_decision": False,
                }
            )
        return tuple(public)

    def read_video(self, review_token: str) -> bytes:
        if (
            not isinstance(review_token, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{43}", review_token) is None
        ):
            raise DailyDigestExportError(_ERROR)
        for record in self._records():
            if hmac.compare_digest(record.review_token, review_token):
                return _read_bound_video(record.bundle, record.receipt)
        raise DailyDigestExportError(_ERROR)


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _parse_receipt(raw: str) -> dict[str, object]:
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise DailyDigestExportError(_ERROR) from None
    if (
        not isinstance(payload, dict)
        or set(payload) != _RECEIPT_KEYS
        or payload["version"] != 1
        or type(payload["version"]) is not int
        or not isinstance(payload["digest_id"], str)
        or re.fullmatch(r"[0-9a-f]{32}", payload["digest_id"]) is None
        or not isinstance(payload["local_date"], str)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", payload["local_date"]) is None
        or not isinstance(payload["plan_sha256"], str)
        or _SHA256.fullmatch(payload["plan_sha256"]) is None
        or not isinstance(payload["video_sha256"], str)
        or _SHA256.fullmatch(payload["video_sha256"]) is None
        or type(payload["output_bytes"]) is not int
        or not 1 <= payload["output_bytes"] <= _MAX_OUTPUT_BYTES
        or type(payload["selected_event_count"]) is not int
        or not 1 <= payload["selected_event_count"] <= 60
    ):
        raise DailyDigestExportError(_ERROR)
    for key, minimum, maximum in (
        ("output_duration_seconds", 0.1, 600.0),
        ("selected_duration_seconds", 0.1, 600.0),
    ):
        value = payload[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not minimum <= float(value) <= maximum
        ):
            raise DailyDigestExportError(_ERROR)
    if any(
        type(payload[key]) is not bool or payload[key] is not expected
        for key, expected in (
            ("facts_only", True),
            ("identity_inference", False),
            ("risk_inference", False),
            ("notification_decision", False),
            ("external_delivery", False),
            ("video_generated", True),
        )
    ):
        raise DailyDigestExportError(_ERROR)
    return payload


def _read_private_receipt(path: Path) -> tuple[str, dict[str, object]]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or not 0 < before.st_size <= 4096
            ):
                raise DailyDigestExportError(_ERROR)
            raw_bytes = os.read(descriptor, 4097)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise DailyDigestExportError(_ERROR) from None
    try:
        path_metadata = path.lstat()
        raw = raw_bytes.decode("utf-8")
    except (OSError, UnicodeError):
        raise DailyDigestExportError(_ERROR) from None
    if (
        len(raw_bytes) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (path_metadata.st_dev, path_metadata.st_ino, path_metadata.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
    ):
        raise DailyDigestExportError(_ERROR)
    return raw, _parse_receipt(raw)


def _trusted_tool(path: str | Path) -> Path:
    candidate = Path(path)
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise DailyDigestExportError(_ERROR) from None
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
        or not os.access(candidate, os.X_OK)
        or resolved != candidate
    ):
        raise DailyDigestExportError(_ERROR)
    return resolved


def _private_root(
    database_path: str | Path, output_root: str | Path, *, create: bool
) -> Path | None:
    database = Path(database_path).expanduser()
    candidate = Path(output_root).expanduser()
    expected = database.parent / "daily-digest-exports"
    try:
        database_metadata = database.lstat()
        parent_metadata = database.parent.lstat()
        database_resolved = database.resolve(strict=True)
        parent_resolved = database.parent.resolve(strict=True)
    except OSError:
        raise DailyDigestExportError(_ERROR) from None
    if (
        not database.is_absolute()
        or database.is_symlink()
        or not stat.S_ISREG(database_metadata.st_mode)
        or database_metadata.st_nlink != 1
        or database_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(database_metadata.st_mode) != 0o600
        or database_resolved != database
        or database.parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        or parent_resolved != database.parent
        or candidate != expected
    ):
        raise DailyDigestExportError(_ERROR)
    if not candidate.exists() and not create:
        return None
    try:
        if create:
            candidate.mkdir(mode=0o700, exist_ok=True)
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise DailyDigestExportError(_ERROR) from None
    if (
        candidate.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or resolved != candidate
    ):
        raise DailyDigestExportError(_ERROR)
    return resolved


def _admit_storage_root(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError:
        raise DailyDigestExportError(_ERROR) from None
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or resolved != candidate
    ):
        raise DailyDigestExportError(_ERROR)
    return resolved


def _source_is_within_private_root(source: DailyDigestMediaSource, root: Path) -> bool:
    try:
        relative = source.clip_path.relative_to(root)
    except ValueError:
        return False
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError:
            return False
        if (
            current.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            return False
    return True


def _run(command: tuple[str, ...], *, timeout_seconds: float) -> None:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            start_new_session=True,
            umask=0o077,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise DailyDigestExportError(_ERROR) from None
    except (OSError, subprocess.SubprocessError):
        raise DailyDigestExportError(_ERROR) from None
    if return_code != 0:
        raise DailyDigestExportError(_ERROR)


def _copy_verified(source: DailyDigestMediaSource, destination: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        source_fd = os.open(source.clip_path, flags)
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size != source.clip_bytes
        ):
            raise DailyDigestExportError(_ERROR)
        destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            if copied > source.clip_bytes:
                raise DailyDigestExportError(_ERROR)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        destination_metadata = os.fstat(destination_fd)
    except (OSError, ValueError):
        raise DailyDigestExportError(_ERROR) from None
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)
    if (
        digest.hexdigest() != source.clip_sha256
        or copied != source.clip_bytes
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or not stat.S_ISREG(destination_metadata.st_mode)
        or stat.S_IMODE(destination_metadata.st_mode) != 0o600
        or destination_metadata.st_size != source.clip_bytes
    ):
        raise DailyDigestExportError(_ERROR)


def _hash_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or not 0 < metadata.st_size <= _MAX_OUTPUT_BYTES
            ):
                raise DailyDigestExportError(_ERROR)
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise DailyDigestExportError(_ERROR) from None
    try:
        path_metadata = path.lstat()
    except OSError:
        raise DailyDigestExportError(_ERROR) from None
    if (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or (path_metadata.st_dev, path_metadata.st_ino, path_metadata.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise DailyDigestExportError(_ERROR)
    return digest.hexdigest(), metadata.st_size


def _probe_output(path: Path, ffprobe: Path) -> float:
    try:
        completed = subprocess.run(
            (
                str(ffprobe),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        raise DailyDigestExportError(_ERROR) from None
    if completed.returncode != 0 or len(completed.stdout) > 128 or completed.stderr:
        raise DailyDigestExportError(_ERROR)
    try:
        duration = float(completed.stdout.decode("ascii").strip())
    except (UnicodeError, ValueError):
        raise DailyDigestExportError(_ERROR) from None
    if not math.isfinite(duration) or not 0.1 <= duration <= 600.0:
        raise DailyDigestExportError(_ERROR)
    return round(duration, 3)


def _private_receipt(
    *,
    plan: DailyDigestPlan,
    video_sha256: str,
    output_bytes: int,
    output_duration_seconds: float,
) -> dict[str, object]:
    plan_sha256 = hashlib.sha256((plan.to_json(indent=2) + "\n").encode("utf-8")).hexdigest()
    return {
        "version": 1,
        "digest_id": plan.digest_id,
        "local_date": plan.local_date,
        "plan_sha256": plan_sha256,
        "video_sha256": video_sha256,
        "output_bytes": output_bytes,
        "output_duration_seconds": output_duration_seconds,
        "selected_event_count": plan.selected_event_count,
        "selected_duration_seconds": plan.selected_duration_seconds,
        "facts_only": True,
        "identity_inference": False,
        "risk_inference": False,
        "notification_decision": False,
        "external_delivery": False,
        "video_generated": True,
    }


def _read_receipt(bundle: Path, plan: DailyDigestPlan) -> DailyDigestExportReceipt:
    receipt_path = bundle / "receipt.json"
    video_path = bundle / "digest.mp4"
    try:
        bundle_metadata = bundle.lstat()
        receipt_metadata = receipt_path.lstat()
        video_metadata = video_path.lstat()
        if (
            bundle.is_symlink()
            or not stat.S_ISDIR(bundle_metadata.st_mode)
            or bundle_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(bundle_metadata.st_mode) != 0o700
            or receipt_path.is_symlink()
            or not stat.S_ISREG(receipt_metadata.st_mode)
            or receipt_metadata.st_nlink != 1
            or receipt_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(receipt_metadata.st_mode) != 0o600
            or not 0 < receipt_metadata.st_size <= 4096
            or video_path.is_symlink()
            or not stat.S_ISREG(video_metadata.st_mode)
            or video_metadata.st_nlink != 1
            or video_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(video_metadata.st_mode) != 0o600
        ):
            raise ValueError
        raw, payload = _read_private_receipt(receipt_path)
    except (OSError, ValueError, DailyDigestExportError):
        raise DailyDigestExportError(_ERROR) from None
    video_sha256, output_bytes = _hash_file(video_path)
    expected = _private_receipt(
        plan=plan,
        video_sha256=video_sha256,
        output_bytes=output_bytes,
        output_duration_seconds=float(payload["output_duration_seconds"]),
    )
    if payload != expected or raw != json.dumps(expected, sort_keys=True, indent=2) + "\n":
        raise DailyDigestExportError(_ERROR)
    return DailyDigestExportReceipt(
        local_date=plan.local_date,
        selected_event_count=plan.selected_event_count,
        selected_duration_seconds=plan.selected_duration_seconds,
        output_duration_seconds=float(expected["output_duration_seconds"]),
        output_bytes=output_bytes,
        reused_existing=True,
    )


def export_daily_digest(
    plan: DailyDigestPlan,
    admission: DailyDigestAdmission,
    *,
    database_path: str | Path,
    output_root: str | Path,
    storage_root: str | Path,
    ffmpeg_path: str | Path = "/usr/bin/ffmpeg",
    ffprobe_path: str | Path = "/usr/bin/ffprobe",
) -> DailyDigestExportReceipt:
    if not isinstance(plan, DailyDigestPlan) or not isinstance(admission, DailyDigestAdmission):
        raise DailyDigestExportError(_ERROR)
    if plan.selected_event_count == 0:
        raise DailyDigestExportError(_ERROR)
    root = _private_root(database_path, output_root, create=True)
    assert root is not None
    admitted_storage_root = _admit_storage_root(storage_root)
    ffmpeg = _trusted_tool(ffmpeg_path)
    ffprobe = _trusted_tool(ffprobe_path)
    candidate_map = {candidate.event_id: candidate for candidate in admission.candidates}
    source_map = {source.event_id: source for source in admission.media_sources}
    if (
        admission.local_date != plan.local_date
        or admission.source_event_count != plan.source_event_count
        or admission.source_event_count != len(admission.candidates)
        or admission.source_event_count != len(admission.media_sources)
        or len(candidate_map) != len(admission.candidates)
        or len(source_map) != len(admission.media_sources)
        or set(candidate_map) != set(source_map)
    ):
        raise DailyDigestExportError(_ERROR)
    selected: list[tuple[DailyDigestItem, DailyDigestMediaSource]] = []
    for item in plan.items:
        candidate = candidate_map.get(item.event_id)
        source = source_map.get(item.event_id)
        if (
            candidate is None
            or source is None
            or item.camera_id != candidate.camera_id
            or item.camera_id != source.camera_id
            or item.occurred_at != candidate.occurred_at
            or item.source_binding_sha256 != candidate.source_binding_sha256
            or item.clip_end_seconds > source.duration_seconds + 1e-6
            or not _source_is_within_private_root(source, admitted_storage_root)
        ):
            raise DailyDigestExportError(_ERROR)
        selected.append((item, source))
    bundle = root / f"{plan.local_date}_{plan.digest_id}"
    if bundle.exists() or bundle.is_symlink():
        return _read_receipt(bundle, plan)

    with tempfile.TemporaryDirectory(prefix=".digest-export-", dir=root) as temporary:
        staging = Path(temporary)
        staging.chmod(0o700)
        segment_names: list[str] = []
        for index, (item, source) in enumerate(selected, start=1):
            staged_source = staging / f"source_{index:03d}.mp4"
            segment = staging / f"segment_{index:03d}.mp4"
            _copy_verified(source, staged_source)
            duration = round(item.clip_end_seconds - item.clip_start_seconds, 3)
            _run(
                (
                    str(ffmpeg),
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{item.clip_start_seconds:.3f}",
                    "-i",
                    str(staged_source),
                    "-t",
                    f"{duration:.3f}",
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,fps=15",
                    "-c:v",
                    "libx264",
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
                    "-movflags",
                    "+faststart",
                    "-n",
                    str(segment),
                ),
                timeout_seconds=120,
            )
            try:
                secure_media_file(segment)
            except ValueError as exc:
                raise DailyDigestExportError(_ERROR) from exc
            segment_names.append(segment.name)
        concat_path = staging / "segments.txt"
        concat_payload = "".join(f"file '{name}'\n" for name in segment_names).encode("ascii")
        try:
            descriptor = os.open(concat_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(concat_payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            raise DailyDigestExportError(_ERROR) from None
        final_staged = staging / "digest.mp4"
        _run(
            (
                str(ffmpeg),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "1",
                "-i",
                str(concat_path),
                "-c",
                "copy",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-movflags",
                "+faststart",
                "-n",
                str(final_staged),
            ),
            timeout_seconds=120,
        )
        try:
            secure_media_file(final_staged)
        except ValueError as exc:
            raise DailyDigestExportError(_ERROR) from exc
        video_sha256, output_bytes = _hash_file(final_staged)
        output_duration = _probe_output(final_staged, ffprobe)
        if abs(output_duration - plan.selected_duration_seconds) > max(
            1.5, plan.selected_duration_seconds * 0.1
        ):
            raise DailyDigestExportError(_ERROR)
        receipt_payload = _private_receipt(
            plan=plan,
            video_sha256=video_sha256,
            output_bytes=output_bytes,
            output_duration_seconds=output_duration,
        )
        receipt_staged = staging / "receipt.json"
        try:
            descriptor = os.open(receipt_staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(
                    (json.dumps(receipt_payload, sort_keys=True, indent=2) + "\n").encode()
                )
                handle.flush()
                os.fsync(handle.fileno())
            bundle.mkdir(mode=0o700)
            os.link(final_staged, bundle / "digest.mp4", follow_symlinks=False)
            os.link(receipt_staged, bundle / "receipt.json", follow_symlinks=False)
            root_fd = os.open(root, os.O_RDONLY)
            bundle_fd = os.open(bundle, os.O_RDONLY)
            try:
                os.fsync(bundle_fd)
                os.fsync(root_fd)
            finally:
                os.close(bundle_fd)
                os.close(root_fd)
        except OSError:
            raise DailyDigestExportError(_ERROR) from None
    result = _read_receipt(bundle, plan)
    return DailyDigestExportReceipt(
        local_date=result.local_date,
        selected_event_count=result.selected_event_count,
        selected_duration_seconds=result.selected_duration_seconds,
        output_duration_seconds=result.output_duration_seconds,
        output_bytes=result.output_bytes,
        reused_existing=False,
    )


def inspect_daily_digest_exports(
    *, database_path: str | Path, output_root: str | Path
) -> dict[str, object]:
    common = {
        "version": 1,
        "paths_disclosed": False,
        "event_ids_disclosed": False,
        "camera_ids_disclosed": False,
        "media_read": False,
        "media_content_decoded": False,
        "identity_inference": False,
        "risk_inference": False,
        "notification_decision": False,
        "external_delivery": False,
    }
    try:
        root = _private_root(database_path, output_root, create=False)
        if root is None:
            return {
                **common,
                "status": "not_generated",
                "export_count": 0,
                "video_available": False,
                "operational": True,
                "attention_required": False,
            }
        bundles = sorted(root.iterdir())
        if len(bundles) > _MAX_EXPORTS:
            raise DailyDigestExportError(_ERROR)
        summaries: list[tuple[str, int, float, int]] = []
        for bundle in bundles:
            bundle_metadata = bundle.lstat()
            if (
                bundle.is_symlink()
                or not stat.S_ISDIR(bundle_metadata.st_mode)
                or bundle_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(bundle_metadata.st_mode) != 0o700
                or _BUNDLE_NAME.fullmatch(bundle.name) is None
            ):
                raise DailyDigestExportError(_ERROR)
            receipt_path = bundle / "receipt.json"
            video_path = bundle / "digest.mp4"
            receipt_metadata = receipt_path.lstat()
            video_metadata = video_path.lstat()
            if (
                receipt_path.is_symlink()
                or video_path.is_symlink()
                or not stat.S_ISREG(receipt_metadata.st_mode)
                or not stat.S_ISREG(video_metadata.st_mode)
                or receipt_metadata.st_uid != os.geteuid()
                or video_metadata.st_uid != os.geteuid()
                or receipt_metadata.st_nlink != 1
                or video_metadata.st_nlink != 1
                or stat.S_IMODE(receipt_metadata.st_mode) != 0o600
                or stat.S_IMODE(video_metadata.st_mode) != 0o600
            ):
                raise DailyDigestExportError(_ERROR)
            raw, payload = _read_private_receipt(receipt_path)
            if (
                raw != json.dumps(payload, sort_keys=True, indent=2) + "\n"
                or payload.get("local_date") != bundle.name[:10]
                or payload.get("digest_id") != bundle.name[11:]
                or payload.get("video_generated") is not True
                or payload.get("external_delivery") is not False
                or payload.get("output_bytes") != video_metadata.st_size
            ):
                raise DailyDigestExportError(_ERROR)
            summaries.append(
                (
                    payload["local_date"],
                    payload["selected_event_count"],
                    float(payload["output_duration_seconds"]),
                    video_metadata.st_size,
                )
            )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DailyDigestExportError):
        return {
            **common,
            "status": "unavailable",
            "export_count": 0,
            "video_available": False,
            "operational": False,
            "attention_required": True,
        }
    if not summaries:
        return {
            **common,
            "status": "not_generated",
            "export_count": 0,
            "video_available": False,
            "operational": True,
            "attention_required": False,
        }
    latest = max(summaries)
    return {
        **common,
        "status": "available",
        "export_count": len(summaries),
        "video_available": True,
        "latest_local_date": latest[0],
        "latest_selected_event_count": latest[1],
        "latest_output_duration_seconds": latest[2],
        "latest_output_bytes": latest[3],
        "operational": True,
        "attention_required": False,
    }


def _read_bound_video(bundle: Path, receipt: dict[str, object]) -> bytes:
    video = bundle / "digest.mp4"
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(video, flags)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or not 0 < before.st_size <= _MAX_DASHBOARD_VIDEO_BYTES
                or receipt["output_bytes"] != before.st_size
            ):
                raise DailyDigestExportError(_ERROR)
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            remaining = before.st_size
            while remaining:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    raise DailyDigestExportError(_ERROR)
                chunks.append(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_metadata = video.lstat()
    except (OSError, DailyDigestExportError):
        raise DailyDigestExportError(_ERROR) from None
    if (
        digest.hexdigest() != receipt["video_sha256"]
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or (path_metadata.st_dev, path_metadata.st_ino, path_metadata.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
    ):
        raise DailyDigestExportError(_ERROR)
    return b"".join(chunks)


def read_latest_daily_digest_video(*, database_path: str | Path, output_root: str | Path) -> bytes:
    root = _private_root(database_path, output_root, create=False)
    if root is None:
        raise DailyDigestExportError(_ERROR)
    try:
        bundles = sorted(root.iterdir())
        if not bundles or len(bundles) > _MAX_EXPORTS:
            raise DailyDigestExportError(_ERROR)
        candidates: list[tuple[str, str, Path, dict[str, object]]] = []
        for bundle in bundles:
            metadata = bundle.lstat()
            if (
                bundle.is_symlink()
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or _BUNDLE_NAME.fullmatch(bundle.name) is None
            ):
                raise DailyDigestExportError(_ERROR)
            raw, receipt = _read_private_receipt(bundle / "receipt.json")
            if (
                raw != json.dumps(receipt, sort_keys=True, indent=2) + "\n"
                or receipt["local_date"] != bundle.name[:10]
                or receipt["digest_id"] != bundle.name[11:]
            ):
                raise DailyDigestExportError(_ERROR)
            candidates.append((bundle.name[:10], bundle.name[11:], bundle, receipt))
        _date, _digest, bundle, receipt = max(candidates)
    except (OSError, DailyDigestExportError):
        raise DailyDigestExportError(_ERROR) from None
    return _read_bound_video(bundle, receipt)
