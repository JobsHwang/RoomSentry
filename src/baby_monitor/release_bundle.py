from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import uuid
from contextlib import suppress
from pathlib import Path

from .release_audit import _git, _tracked_entries
from .release_sbom import _generate_source_sbom


class ReleaseBundleError(RuntimeError):
    pass


_SAFE_RELEASE_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}")


def _archive_info(name: str, content: bytes, *, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_bundle(root: str | Path, output: str | Path) -> dict[str, object]:
    target = Path(output).expanduser()
    if target.name in {"", ".", ".."} or not target.name.endswith(".tar.gz"):
        raise ReleaseBundleError("release output must use the .tar.gz suffix")
    if target.exists() or target.is_symlink():
        raise FileExistsError("release output already exists")
    if not target.parent.is_dir():
        raise ReleaseBundleError("release output parent must already exist")

    repository = Path(root).expanduser().resolve()
    entries = tuple(sorted(_tracked_entries(repository), key=lambda item: item[3]))
    sbom = _generate_source_sbom(repository, entries)
    metadata = sbom.get("metadata")
    if not isinstance(metadata, dict):
        raise ReleaseBundleError("generated release metadata is invalid")
    component = metadata.get("component")
    if not isinstance(component, dict):
        raise ReleaseBundleError("generated release metadata is invalid")
    name = component.get("name")
    version = component.get("version")
    if (
        not isinstance(name, str)
        or _SAFE_RELEASE_NAME.fullmatch(name) is None
        or not isinstance(version, str)
        or _SAFE_VERSION.fullmatch(version) is None
    ):
        raise ReleaseBundleError("generated release identity is invalid")
    prefix = f"{name}-{version}/"

    if any(stage != "0" for _, _, stage, _ in entries):
        raise ReleaseBundleError("Git index contains an unsupported staged entry")
    sbom_payload = (json.dumps(sbom, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    partial = target.parent / f".{target.name}.partial-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as raw:
            descriptor = -1
            with (
                gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=9,
                    fileobj=raw,
                    mtime=0,
                ) as compressed,
                tarfile.open(
                    mode="w",
                    fileobj=compressed,
                    format=tarfile.PAX_FORMAT,
                ) as archive,
            ):
                for mode, object_id, _, path in entries:
                    content = _git(repository, "cat-file", "blob", object_id)
                    archive.addfile(
                        _archive_info(
                            prefix + path,
                            content,
                            mode=0o755 if mode == "100755" else 0o644,
                        ),
                        io.BytesIO(content),
                    )
                archive.addfile(
                    _archive_info(prefix + "SBOM.cdx.json", sbom_payload, mode=0o644),
                    io.BytesIO(sbom_payload),
                )
            raw.flush()
            os.fsync(raw.fileno())
        os.link(partial, target)
    except FileExistsError as exc:
        raise FileExistsError("release output already exists") from exc
    except OSError as exc:
        raise ReleaseBundleError("release output could not be written") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(OSError):
            partial.unlink(missing_ok=True)

    return {
        "version": 1,
        "status": "written",
        "format": "tar+gzip",
        "project_version": version,
        "staged_source_files": len(entries),
        "archive_files": len(entries) + 1,
        "includes_source_sbom": True,
        "bytes": target.stat().st_size,
        "sha256": _sha256_file(target),
        "path_disclosed": False,
        "content_disclosed": False,
    }
