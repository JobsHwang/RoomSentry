from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from . import __version__
from .release_audit import ReleaseAuditError, _audit_entries, _git, _tracked_entries


class ReleaseSBOMError(RuntimeError):
    pass


_REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_PROJECT_NAME = "baby-monitor"
_PROJECT_LICENSE = "Apache-2.0"
_ROOT_REF_PREFIX = "pkg:pypi/baby-monitor@"


def _property(name: str, value: object) -> dict[str, str]:
    return {"name": name, "value": str(value).lower() if isinstance(value, bool) else str(value)}


def _object_content(root: Path, object_id: str) -> bytes:
    return _git(root, "cat-file", "blob", object_id)


def _load_pyproject(root: Path, object_id: str | None) -> dict[str, Any]:
    if object_id is None:
        raise ReleaseSBOMError("Git index does not contain pyproject.toml")
    try:
        import tomllib

        document = tomllib.loads(_object_content(root, object_id).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, ReleaseAuditError) as exc:
        raise ReleaseSBOMError("staged pyproject.toml is invalid") from exc
    if not isinstance(document, dict):
        raise ReleaseSBOMError("staged pyproject.toml is invalid")
    return document


def _requirements(value: object, *, group: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ReleaseSBOMError(f"staged {group} requirements are invalid")
    requirements: list[str] = []
    for item in value:
        if not 1 <= len(item) <= 512 or any(ord(char) < 32 for char in item):
            raise ReleaseSBOMError(f"staged {group} requirements are invalid")
        match = _REQUIREMENT_NAME.match(item)
        if match is None:
            raise ReleaseSBOMError(f"staged {group} requirements are invalid")
        requirements.append(item)
    return tuple(requirements)


def _dependency_component(requirement: str, *, group: str) -> dict[str, object]:
    match = _REQUIREMENT_NAME.match(requirement)
    assert match is not None
    name = match.group(0)
    normalized = re.sub(r"[-_.]+", "-", name).lower()
    scope = "required" if group == "runtime" else "optional"
    if group in {"build", "dev"}:
        scope = "excluded"
    return {
        "type": "library",
        "bom-ref": f"urn:baby-monitor:declared:{group}:{normalized}",
        "name": name,
        "scope": scope,
        "properties": [
            _property("baby-monitor:dependency-group", group),
            _property("baby-monitor:declared-requirement", requirement),
            _property("baby-monitor:resolved-version", False),
            _property("baby-monitor:bundled", False),
        ],
    }


def _declared_dependencies(pyproject: dict[str, Any]) -> tuple[list[dict[str, object]], list[str]]:
    project = pyproject.get("project")
    build_system = pyproject.get("build-system")
    if not isinstance(project, dict) or not isinstance(build_system, dict):
        raise ReleaseSBOMError("staged project metadata is invalid")

    grouped: list[tuple[str, str]] = []
    for item in _requirements(project.get("dependencies", []), group="runtime"):
        grouped.append(("runtime", item))
    for item in _requirements(build_system.get("requires", []), group="build"):
        grouped.append(("build", item))

    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ReleaseSBOMError("staged optional dependency metadata is invalid")
    for group, values in sorted(optional.items()):
        if not isinstance(group, str) or not _REQUIREMENT_NAME.fullmatch(group):
            raise ReleaseSBOMError("staged optional dependency metadata is invalid")
        dependency_group = "dev" if group == "dev" else f"optional-{group}"
        for item in _requirements(values, group=dependency_group):
            grouped.append((dependency_group, item))

    components = [_dependency_component(item, group=group) for group, item in grouped]
    refs = [str(component["bom-ref"]) for component in components]
    if len(refs) != len(set(refs)):
        raise ReleaseSBOMError("staged dependency metadata contains duplicate package names")
    components.sort(key=lambda item: str(item["bom-ref"]))
    runtime_refs = sorted(
        str(component["bom-ref"]) for component in components if component["scope"] == "required"
    )
    return components, runtime_refs


def _source_file_component(path: str, content: bytes, *, mode: str) -> dict[str, object]:
    path_digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return {
        "type": "file",
        "bom-ref": f"urn:baby-monitor:source-file:{path_digest}",
        "name": path,
        "hashes": [{"alg": "SHA-256", "content": hashlib.sha256(content).hexdigest()}],
        "properties": [_property("baby-monitor:git-mode", mode)],
    }


def _generate_source_sbom(
    repository: Path,
    entries: tuple[tuple[str, str, str, str], ...],
) -> dict[str, object]:
    report = _audit_entries(repository, entries, max_file_bytes=10 * 1024 * 1024)
    if not report.passed:
        raise ReleaseSBOMError("Git index must pass release-audit before SBOM generation")

    objects = {path: object_id for _, object_id, stage, path in entries if stage == "0"}
    paths = set(objects)
    if any(
        Path(path).name == "bom.json" or path.endswith(".cdx.json") or path.endswith(".cdx.xml")
        for path in paths
    ):
        raise ReleaseSBOMError("Git index contains a generated SBOM self-reference")
    pyproject = _load_pyproject(repository, objects.get("pyproject.toml"))
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise ReleaseSBOMError("staged project metadata is invalid")
    name = project.get("name")
    version = project.get("version")
    description = project.get("description")
    license_metadata = project.get("license")
    if name != _PROJECT_NAME or version != __version__:
        raise ReleaseSBOMError("staged project identity does not match the running source")
    if not isinstance(description, str) or not description:
        raise ReleaseSBOMError("staged project description is invalid")
    if license_metadata != {"file": "LICENSE"} or "LICENSE" not in paths:
        raise ReleaseSBOMError("staged project license metadata is invalid")

    file_components: list[dict[str, object]] = []
    for mode, object_id, stage, path in entries:
        if stage != "0":
            raise ReleaseSBOMError("Git index contains an unsupported staged entry")
        file_components.append(
            _source_file_component(path, _object_content(repository, object_id), mode=mode)
        )
    file_components.sort(key=lambda item: str(item["name"]))

    dependency_components, runtime_refs = _declared_dependencies(pyproject)
    root_ref = f"{_ROOT_REF_PREFIX}{version}"
    root_component: dict[str, object] = {
        "type": "application",
        "bom-ref": root_ref,
        "name": name,
        "version": version,
        "description": description,
        "scope": "required",
        "licenses": [{"expression": _PROJECT_LICENSE, "acknowledgement": "declared"}],
        "purl": root_ref,
        "components": file_components,
        "properties": [
            _property("baby-monitor:source", "git-index"),
            _property("baby-monitor:tracked-files", report.tracked_files),
            _property("baby-monitor:source-bytes", report.total_bytes),
            _property("baby-monitor:private-runtime-data-included", False),
            _property("baby-monitor:model-weights-included", False),
        ],
    }
    return {
        "$schema": "https://cyclonedx.org/schema/bom-1.6.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "lifecycles": [{"phase": "pre-build"}],
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "baby-monitor release-sbom",
                        "version": __version__,
                    }
                ]
            },
            "component": root_component,
            "properties": [
                _property("baby-monitor:generated-from", "git-index-only"),
                _property("baby-monitor:content-disclosed", False),
            ],
        },
        "components": dependency_components,
        "dependencies": [{"ref": root_ref, "dependsOn": runtime_refs}],
    }


def generate_source_sbom(root: str | Path) -> dict[str, object]:
    repository = Path(root).expanduser().resolve()
    if not repository.is_dir():
        raise ReleaseSBOMError("repository root is missing or invalid")
    return _generate_source_sbom(repository, _tracked_entries(repository))


def source_sbom_json(root: str | Path, *, indent: int | None = 2) -> str:
    return json.dumps(
        generate_source_sbom(root),
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
    )


def write_source_sbom(root: str | Path, output: str | Path) -> dict[str, object]:
    target = Path(output).expanduser()
    if target.name in {"", ".", ".."} or not target.name.endswith(".cdx.json"):
        raise ReleaseSBOMError("SBOM output must use the .cdx.json suffix")
    if target.exists() or target.is_symlink():
        raise FileExistsError("SBOM output already exists")
    if not target.parent.is_dir():
        raise ReleaseSBOMError("SBOM output parent must already exist")

    document = generate_source_sbom(root)
    payload = (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    partial = target.parent / f".{target.name}.partial-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(partial, target)
    except FileExistsError as exc:
        raise FileExistsError("SBOM output already exists") from exc
    except OSError as exc:
        raise ReleaseSBOMError("SBOM output could not be written") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(OSError):
            partial.unlink(missing_ok=True)

    metadata = document.get("metadata")
    declared_components = document.get("components")
    if not isinstance(metadata, dict) or not isinstance(declared_components, list):
        raise ReleaseSBOMError("generated SBOM summary is invalid")
    root_component = metadata.get("component")
    if not isinstance(root_component, dict):
        raise ReleaseSBOMError("generated SBOM summary is invalid")
    source_files = root_component.get("components")
    if not isinstance(source_files, list):
        raise ReleaseSBOMError("generated SBOM summary is invalid")
    return {
        "version": 1,
        "status": "written",
        "format": "CycloneDX",
        "spec_version": "1.6",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_files": len(source_files),
        "declared_components": len(declared_components),
        "path_disclosed": False,
        "content_disclosed": False,
    }
