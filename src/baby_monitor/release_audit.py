from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


class ReleaseAuditError(RuntimeError):
    pass


_FORBIDDEN_DIRECTORIES = {
    ".venv",
    "artifacts",
    "data",
    "models",
    "private",
    "reports",
    "secrets",
    "venv",
    "weights",
}
_FORBIDDEN_EXTENSIONS = {
    ".3gp",
    ".aac",
    ".avi",
    ".bin",
    ".flac",
    ".gif",
    ".gguf",
    ".jpeg",
    ".jpg",
    ".key",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".onnx",
    ".pem",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
    ".webm",
    ".webp",
}
_ALLOWED_SPECIAL_PATHS = {".env.example", "data/.gitkeep"}
_SAFE_GIT_MODES = {"100644", "100755"}
_SAFE_PATH = re.compile(r"[A-Za-z0-9_.@+/-]{1,240}")
_PRIVATE_KEY = re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_CREDENTIAL_URL = re.compile(
    rb"[a-z][a-z0-9+.-]{1,20}://[^\s/:@]{1,128}:"
    rb"(?P<password>[^\s/@]{1,128})@",
    re.I,
)
_PROVIDER_TOKEN = re.compile(
    rb"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    rb"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16})"
)
_PASSWORD_ASSIGNMENT = re.compile(r"(?i)\b(?:password|passwd)\b\s*[:=]\s*[\"']?([^\s\"'#,;]+)")
# Concrete user-specific home directories leak the operator's identity; docs
# must use "~" or an explicit "<placeholder>" form instead.
_MACHINE_HOME_PATH = re.compile(
    r"(?:/home/|/Users/|[A-Za-z]:[\\/]+Users[\\/]+|/mnt/[A-Za-z]/Users/|"
    r"[\\/]{2,}(?:wsl\$|wsl\.localhost)[\\/]+[^\\/\s]+[\\/]+home[\\/]+)"
    r"[A-Za-z0-9_.-]{1,64}(?:[\\/]|$)",
    re.I,
)
_FIXTURE_CREDENTIAL_MARKER = b"release-audit: allow=fixture-credential"
_FIXTURE_HOME_PATH_MARKER = "release-audit: allow=fixture-home-path"
_PLACEHOLDERS = {
    "changeme",
    "dummy",
    "example",
    "none",
    "null",
    "password",
    "placeholder",
    "redacted",
    "replace-me",
    "secret",
    "test",
    "your-password",
}


@dataclass(frozen=True, slots=True)
class ReleaseViolation:
    code: str
    path: str
    line: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReleaseAuditReport:
    tracked_files: int
    scanned_text_files: int
    total_bytes: int
    max_file_bytes: int
    violations: tuple[ReleaseViolation, ...]
    source: str = "git_index"
    untracked_files: int = 0
    version: int = 1

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "status": "passed" if self.passed else "failed",
            "source": self.source,
            "tracked_files": self.tracked_files,
            "scanned_text_files": self.scanned_text_files,
            "total_bytes": self.total_bytes,
            "max_file_bytes": self.max_file_bytes,
            "violations": [item.to_dict() for item in self.violations],
            "content_disclosed": False,
        }
        if self.source == "worktree":
            payload["untracked_files"] = self.untracked_files
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _git(root: Path, *args: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReleaseAuditError("Git index is unavailable") from exc
    if completed.returncode != 0:
        raise ReleaseAuditError("Git index is unavailable")
    return completed.stdout


def _tracked_entries(root: Path) -> tuple[tuple[str, str, str, str], ...]:
    raw = _git(root, "ls-files", "--stage", "-z")
    entries: list[tuple[str, str, str, str]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            metadata, path_bytes = item.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split(" ", 2)
            path = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            raise ReleaseAuditError("Git index contains an invalid entry") from None
        entries.append((mode, object_id, stage, path))
    return tuple(entries)


def _untracked_paths(root: Path) -> tuple[str, ...]:
    raw = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        try:
            paths.append(item.decode("utf-8"))
        except UnicodeDecodeError:
            raise ReleaseAuditError("Git worktree contains an invalid path") from None
    return tuple(paths)


def _safe_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        _SAFE_PATH.fullmatch(path) is not None
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and path == candidate.as_posix()
    )


def _forbidden_path(path: str) -> bool:
    if path in _ALLOWED_SPECIAL_PATHS:
        return False
    candidate = PurePosixPath(path)
    lowered_parts = tuple(part.lower() for part in candidate.parts)
    if any(part in _FORBIDDEN_DIRECTORIES for part in lowered_parts[:-1]):
        return True
    name = candidate.name.lower()
    return (
        name == ".env"
        or (name.startswith(".env.") and name != ".env.example")
        or name in {"credentials", "credentials.json", "local.toml", "secrets.toml"}
    )


def _line_number(content: bytes, offset: int) -> int:
    return content.count(b"\n", 0, offset) + 1


def _credential_is_marked_fixture(path: str, content: bytes, offset: int) -> bool:
    if not path.startswith("tests/"):
        return False
    line_start = content.rfind(b"\n", 0, offset) + 1
    context_start = line_start
    for _ in range(3):
        context_start = content.rfind(b"\n", 0, max(0, context_start - 1)) + 1
    line_end = content.find(b"\n", offset)
    if line_end < 0:
        line_end = len(content)
    return _FIXTURE_CREDENTIAL_MARKER in content[context_start:line_end]


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        normalized in _PLACEHOLDERS
        or normalized.startswith("${")
        or (normalized.startswith("<") and normalized.endswith(">"))
        or (normalized.startswith("{") and normalized.endswith("}"))
    )


def _content_violations(path: str, content: bytes) -> tuple[ReleaseViolation, ...]:
    violations: list[ReleaseViolation] = []
    for code, pattern in (("private_key", _PRIVATE_KEY), ("provider_token", _PROVIDER_TOKEN)):
        for match in pattern.finditer(content):
            violations.append(ReleaseViolation(code, path, _line_number(content, match.start())))
    for match in _CREDENTIAL_URL.finditer(content):
        if _credential_is_marked_fixture(path, content, match.start()):
            continue
        violations.append(
            ReleaseViolation(
                "credential_url",
                path,
                _line_number(content, match.start()),
            )
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return (*violations, ReleaseViolation("binary_file", path))
    if "\x00" in text:
        return (*violations, ReleaseViolation("binary_file", path))
    text_lines = text.splitlines()
    for line_number, line in enumerate(text_lines, start=1):
        for _match in _MACHINE_HOME_PATH.finditer(line):
            if path.startswith("tests/") and _FIXTURE_HOME_PATH_MARKER in "\n".join(
                text_lines[max(0, line_number - 4) : line_number]
            ):
                continue
            violations.append(ReleaseViolation("machine_home_path", path, line_number))
        for match in _PASSWORD_ASSIGNMENT.finditer(line):
            if not _is_placeholder(match.group(1)):
                violations.append(ReleaseViolation("password_assignment", path, line_number))
    return tuple(violations)


def _audit_entries(
    repository: Path,
    entries: tuple[tuple[str, str, str, str], ...],
    *,
    max_file_bytes: int,
) -> ReleaseAuditReport:
    if type(max_file_bytes) is not int or not 1 <= max_file_bytes <= 1024**3:
        raise ReleaseAuditError("max_file_bytes must be between 1 and 1073741824")
    violations: list[ReleaseViolation] = []
    total_bytes = 0
    scanned_text_files = 0
    seen_paths: set[str] = set()
    for mode, object_id, stage, path in entries:
        if path in seen_paths or stage != "0":
            violations.append(ReleaseViolation("unsupported_git_mode", path))
            continue
        seen_paths.add(path)
        if not _safe_path(path):
            violations.append(ReleaseViolation("unsafe_path", "<invalid>"))
            continue
        if mode not in _SAFE_GIT_MODES:
            violations.append(ReleaseViolation("unsupported_git_mode", path))
            continue
        try:
            size = int(_git(repository, "cat-file", "-s", object_id).strip())
        except ValueError:
            raise ReleaseAuditError("Git index contains an invalid blob") from None
        if size < 0:
            raise ReleaseAuditError("Git index contains an invalid blob")
        total_bytes += size

        forbidden_path = _forbidden_path(path)
        forbidden_extension = PurePosixPath(path).suffix.lower() in _FORBIDDEN_EXTENSIONS
        if forbidden_path:
            violations.append(ReleaseViolation("forbidden_path", path))
        if forbidden_extension:
            violations.append(ReleaseViolation("forbidden_extension", path))
        if size > max_file_bytes:
            violations.append(ReleaseViolation("file_too_large", path))
        if forbidden_path or forbidden_extension or size > max_file_bytes:
            continue

        content = _git(repository, "cat-file", "blob", object_id)
        if len(content) != size:
            raise ReleaseAuditError("Git index contains an invalid blob")
        content_findings = _content_violations(path, content)
        if not any(item.code == "binary_file" for item in content_findings):
            scanned_text_files += 1
        violations.extend(content_findings)

    return ReleaseAuditReport(
        tracked_files=len(entries),
        scanned_text_files=scanned_text_files,
        total_bytes=total_bytes,
        max_file_bytes=max_file_bytes,
        violations=tuple(
            sorted(violations, key=lambda item: (item.path, item.line or 0, item.code))
        ),
    )


def _read_regular_worktree_file(
    repository: Path,
    path: str,
) -> tuple[int, bytes] | str:
    candidate = repository.joinpath(*PurePosixPath(path).parts)
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
    except OSError:
        return "missing_worktree_file"
    if resolved_parent != repository and repository not in resolved_parent.parents:
        return "unsafe_path"
    try:
        initial = candidate.lstat()
    except OSError:
        return "missing_worktree_file"
    if not stat.S_ISREG(initial.st_mode):
        return "unsupported_worktree_type"

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError:
        return "worktree_file_changed"
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino, opened.st_size) != (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
        ):
            return "worktree_file_changed"
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                return "worktree_file_changed"
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            return "worktree_file_changed"
        final = os.fstat(descriptor)
        if (final.st_dev, final.st_ino, final.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            return "worktree_file_changed"
        return opened.st_size, b"".join(chunks)
    finally:
        os.close(descriptor)


def _audit_worktree(
    repository: Path,
    *,
    max_file_bytes: int,
) -> ReleaseAuditReport:
    tracked_entries = _tracked_entries(repository)
    untracked_paths = _untracked_paths(repository)
    violations: list[ReleaseViolation] = []
    tracked_modes: dict[str, str] = {}
    for mode, _object_id, stage, path in tracked_entries:
        if path in tracked_modes or stage != "0":
            violations.append(ReleaseViolation("unsupported_git_mode", path))
            continue
        tracked_modes[path] = mode

    paths = (*tracked_modes, *untracked_paths)
    total_bytes = 0
    scanned_text_files = 0
    for path in paths:
        if not _safe_path(path):
            violations.append(ReleaseViolation("unsafe_path", "<invalid>"))
            continue
        mode = tracked_modes.get(path)
        if mode is not None and mode not in _SAFE_GIT_MODES:
            violations.append(ReleaseViolation("unsupported_git_mode", path))
            continue

        forbidden_path = _forbidden_path(path)
        forbidden_extension = PurePosixPath(path).suffix.lower() in _FORBIDDEN_EXTENSIONS
        if forbidden_path:
            violations.append(ReleaseViolation("forbidden_path", path))
        if forbidden_extension:
            violations.append(ReleaseViolation("forbidden_extension", path))
        if forbidden_path or forbidden_extension:
            continue

        result = _read_regular_worktree_file(repository, path)
        if isinstance(result, str):
            violations.append(ReleaseViolation(result, path))
            continue
        size, content = result
        total_bytes += size
        if size > max_file_bytes:
            violations.append(ReleaseViolation("file_too_large", path))
            continue
        content_findings = _content_violations(path, content)
        if not any(item.code == "binary_file" for item in content_findings):
            scanned_text_files += 1
        violations.extend(content_findings)

    return ReleaseAuditReport(
        tracked_files=len(tracked_entries),
        untracked_files=len(untracked_paths),
        scanned_text_files=scanned_text_files,
        total_bytes=total_bytes,
        max_file_bytes=max_file_bytes,
        violations=tuple(
            sorted(violations, key=lambda item: (item.path, item.line or 0, item.code))
        ),
        source="worktree",
    )


def audit_release(
    root: str | Path,
    *,
    max_file_bytes: int = 10 * 1024 * 1024,
    source: str = "git_index",
) -> ReleaseAuditReport:
    if type(max_file_bytes) is not int or not 1 <= max_file_bytes <= 1024**3:
        raise ReleaseAuditError("max_file_bytes must be between 1 and 1073741824")
    repository = Path(root).expanduser().resolve()
    if not repository.is_dir():
        raise ReleaseAuditError("repository root is missing or invalid")
    if source == "worktree":
        return _audit_worktree(repository, max_file_bytes=max_file_bytes)
    if source != "git_index":
        raise ReleaseAuditError("release audit source is invalid")
    return _audit_entries(
        repository,
        _tracked_entries(repository),
        max_file_bytes=max_file_bytes,
    )
