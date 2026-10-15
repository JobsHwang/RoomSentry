"""Private, standard-library artifact helpers for opt-in household debug scripts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def safe_id(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None:
        raise ValueError("debug artifact identifier is invalid")
    if value in {".", ".."}:
        raise ValueError("debug artifact identifier is invalid")
    return value


def _local_external(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("debug artifact must use an absolute canonical path")
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("debug artifact symlinks are forbidden")
    if path.resolve().is_relative_to(REPOSITORY):
        raise ValueError("household artifacts must remain outside the repository")


def private_directory(path: Path) -> None:
    _local_external(path)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("debug input directory must be private and owned")


def private_file(path: Path, *, limit: int = 8 * 1024 * 1024) -> None:
    _local_external(path)
    info = path.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
        or info.st_nlink != 1
        or not 0 < info.st_size <= limit
    ):
        raise ValueError("debug input file must be bounded, private, regular and owned")
    private_directory(path.parent)


def new_output_directory(path: Path) -> None:
    _local_external(path)
    if path.exists():
        raise ValueError("debug output must not already exist")
    parent = path.parent.stat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or parent.st_mode & 0o022:
        raise ValueError("debug output parent must be owned and not shared-writable")


def read_private_json_with_sha(path: Path) -> tuple[object, str]:
    private_file(path)
    raw = path.read_bytes()
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("debug JSON exceeds size limit")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate debug JSON fields")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("non-finite debug JSON is forbidden")

    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid_constant
    )
    return value, hashlib.sha256(raw).hexdigest()


def read_private_json(path: Path) -> object:
    return read_private_json_with_sha(path)[0]


def runtime_versions(vision_python: Path, isolation: tuple[str, ...]) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json; "
        "print(json.dumps({p:m.version(p) for p in "
        "('ultralytics','torch','opencv-python','numpy')}))"
    )
    completed = subprocess.run(
        (*isolation, str(vision_python), "-c", code),
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return json.loads(completed.stdout)


def write_new(path: Path, value: object) -> None:
    _local_external(path)
    private_directory(path.parent)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(payload) > 8 * 1024 * 1024:
        raise ValueError("debug artifact output exceeds size limit")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
