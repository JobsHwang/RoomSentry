from __future__ import annotations

import json
import math
import os
import secrets
import sqlite3
import stat
import struct
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .identity_domain import DeclaredKind
from .identity_matching import IdentityTemplate


class IdentityStoreError(RuntimeError):
    """The private identity store could not complete a safe local operation."""


class _Aead(Protocol):
    def encrypt(self, nonce: bytes, data: bytes, associated_data: bytes) -> bytes: ...

    def decrypt(self, nonce: bytes, data: bytes, associated_data: bytes) -> bytes: ...


def _load_aesgcm() -> Any:
    """Load the optional crypto backend without making module import depend on it."""

    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except (ImportError, ModuleNotFoundError):
        raise IdentityStoreError("identity encryption backend is unavailable") from None
    return AESGCM


_SCHEMA_VERSION = 2
_AAD_SCHEMA = "baby-monitor.identity-template.v1"
_KEY_BYTES = 32
_NONCE_BYTES = 12
_MAX_EMBEDDING_DIMENSION = 4096
_PROFILE_KINDS = frozenset(
    {
        DeclaredKind.BABY,
        DeclaredKind.ADULT,
        DeclaredKind.PERSON,
    }
)
_PROFILE_PREFIX = "profile_"
_TEMPLATE_PREFIX = "template_"
_HEX_DIGITS = frozenset("0123456789abcdef")
_MAX_REVISION = (1 << 63) - 1
_REVISION_TRIGGER_NAMES = frozenset(
    {
        "identity_profiles_revision_after_insert",
        "identity_profiles_revision_after_update",
        "identity_profiles_revision_after_delete",
        "identity_templates_revision_after_insert",
        "identity_templates_revision_after_update",
        "identity_templates_revision_after_delete",
    }
)
_REVISION_TRIGGER_SQL = f"""
CREATE TRIGGER identity_profiles_revision_after_insert
AFTER INSERT ON profiles
BEGIN
    UPDATE store_state
    SET revision = revision + 1
    WHERE singleton = 1 AND revision < {_MAX_REVISION};
    SELECT CASE WHEN changes() = 1
        THEN NULL ELSE RAISE(ABORT, 'identity store revision is invalid') END;
END;
CREATE TRIGGER identity_profiles_revision_after_update
AFTER UPDATE ON profiles
BEGIN
    UPDATE store_state
    SET revision = revision + 1
    WHERE singleton = 1 AND revision < {_MAX_REVISION};
    SELECT CASE WHEN changes() = 1
        THEN NULL ELSE RAISE(ABORT, 'identity store revision is invalid') END;
END;
CREATE TRIGGER identity_profiles_revision_after_delete
AFTER DELETE ON profiles
BEGIN
    UPDATE store_state
    SET revision = revision + 1
    WHERE singleton = 1 AND revision < {_MAX_REVISION};
    SELECT CASE WHEN changes() = 1
        THEN NULL ELSE RAISE(ABORT, 'identity store revision is invalid') END;
END;
CREATE TRIGGER identity_templates_revision_after_insert
AFTER INSERT ON templates
BEGIN
    UPDATE store_state
    SET revision = revision + 1
    WHERE singleton = 1 AND revision < {_MAX_REVISION};
    SELECT CASE WHEN changes() = 1
        THEN NULL ELSE RAISE(ABORT, 'identity store revision is invalid') END;
END;
CREATE TRIGGER identity_templates_revision_after_update
AFTER UPDATE ON templates
BEGIN
    UPDATE store_state
    SET revision = revision + 1
    WHERE singleton = 1 AND revision < {_MAX_REVISION};
    SELECT CASE WHEN changes() = 1
        THEN NULL ELSE RAISE(ABORT, 'identity store revision is invalid') END;
END;
CREATE TRIGGER identity_templates_revision_after_delete
AFTER DELETE ON templates
BEGIN
    UPDATE store_state
    SET revision = revision + 1
    WHERE singleton = 1 AND revision < {_MAX_REVISION};
    SELECT CASE WHEN changes() = 1
        THEN NULL ELSE RAISE(ABORT, 'identity store revision is invalid') END;
END;
"""


@dataclass(frozen=True, slots=True)
class IdentityStoreSummary:
    profile_count: int
    template_count: int
    schema_version: int = _SCHEMA_VERSION
    embeddings_encrypted: bool = True
    metadata_encrypted: bool = False
    raw_images_stored: bool = False

    def to_public_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_count": self.profile_count,
            "template_count": self.template_count,
            "embeddings_encrypted": self.embeddings_encrypted,
            "metadata_encrypted": self.metadata_encrypted,
            "raw_images_stored": self.raw_images_stored,
        }


@dataclass(frozen=True, slots=True)
class IdentityProfileSummary:
    profile_id: str
    declared_kind: DeclaredKind
    template_count: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "declared_kind": self.declared_kind.value,
            "template_count": self.template_count,
        }


@dataclass(frozen=True, slots=True)
class IdentityTemplateSummary:
    template_id: str
    profile_id: str
    model_fingerprint: str
    embedding_dimension: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "template_id": self.template_id,
            "profile_id": self.profile_id,
            "model_fingerprint": self.model_fingerprint,
            "embedding_dimension": self.embedding_dimension,
        }


@dataclass(frozen=True, slots=True)
class IdentityDeletionSummary:
    deleted_profiles: int
    deleted_templates: int
    remaining_profiles: int
    remaining_templates: int
    committed: bool = True
    checkpoint_completed: bool = True
    cleanup_pending: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.committed) is not bool
            or type(self.checkpoint_completed) is not bool
            or type(self.cleanup_pending) is not bool
            or not self.committed
            or self.checkpoint_completed is self.cleanup_pending
        ):
            raise IdentityStoreError("identity deletion status is invalid")

    def to_public_dict(self) -> dict[str, int | bool]:
        return {
            "deleted_profiles": self.deleted_profiles,
            "deleted_templates": self.deleted_templates,
            "remaining_profiles": self.remaining_profiles,
            "remaining_templates": self.remaining_templates,
            "committed": self.committed,
            "checkpoint_completed": self.checkpoint_completed,
            "cleanup_pending": self.cleanup_pending,
        }


@dataclass(frozen=True, slots=True)
class IdentityCleanupSummary:
    checkpoint_completed: bool
    cleanup_pending: bool

    def __post_init__(self) -> None:
        if (
            type(self.checkpoint_completed) is not bool
            or type(self.cleanup_pending) is not bool
            or self.checkpoint_completed is self.cleanup_pending
        ):
            raise IdentityStoreError("identity cleanup status is invalid")

    def to_public_dict(self) -> dict[str, bool]:
        return {
            "checkpoint_completed": self.checkpoint_completed,
            "cleanup_pending": self.cleanup_pending,
        }


@dataclass(frozen=True, slots=True)
class IdentityEnrollmentSummary:
    """Opaque result for one atomic profile-and-template enrollment."""

    profile_id: str
    template_ids: tuple[str, ...]
    declared_kind: DeclaredKind
    model_fingerprint: str
    embedding_dimension: int

    def to_public_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "template_ids": list(self.template_ids),
            "declared_kind": self.declared_kind.value,
            "model_fingerprint": self.model_fingerprint,
            "embedding_dimension": self.embedding_dimension,
        }


@dataclass(frozen=True, slots=True)
class InMemoryIdentityTemplate(IdentityTemplate):
    """A matcher-compatible decrypted template whose repr hides biometric data."""

    embedding: tuple[float, ...] = field(repr=False)


class IdentityTemplateSnapshot(tuple[InMemoryIdentityTemplate, ...]):
    """Matcher-compatible tuple that fails closed after any store mutation.

    The private validator is intentionally retained by the snapshot so normal tuple
    operations used by the matcher re-check the persisted store revision. Callers
    must not strip this guard by copying templates into a plain tuple.
    """

    def __new__(
        cls,
        templates: Sequence[InMemoryIdentityTemplate],
        *,
        revision: int,
        model_fingerprint: str,
        embedding_dimension: int,
        validator: Callable[[int], None],
    ) -> IdentityTemplateSnapshot:
        instance = super().__new__(cls, templates)
        object.__setattr__(instance, "_revision", revision)
        object.__setattr__(instance, "_model_fingerprint", model_fingerprint)
        object.__setattr__(instance, "_embedding_dimension", embedding_dimension)
        object.__setattr__(instance, "_validator", validator)
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("identity template snapshots are immutable")

    @property
    def revision(self) -> int:
        return int(self._revision)

    @property
    def model_fingerprint(self) -> str:
        return str(self._model_fingerprint)

    @property
    def embedding_dimension(self) -> int:
        return int(self._embedding_dimension)

    def assert_current(self) -> None:
        self._validator(self._revision)

    def __bool__(self) -> bool:
        self.assert_current()
        return tuple.__len__(self) != 0

    def __len__(self) -> int:
        self.assert_current()
        return tuple.__len__(self)

    def __iter__(self) -> Iterator[InMemoryIdentityTemplate]:
        self.assert_current()
        for index in range(tuple.__len__(self)):
            self.assert_current()
            yield tuple.__getitem__(self, index)
        self.assert_current()

    def __getitem__(
        self,
        index: int | slice,
    ) -> InMemoryIdentityTemplate | IdentityTemplateSnapshot:
        self.assert_current()
        item = tuple.__getitem__(self, index)
        if isinstance(index, slice):
            return type(self)(
                item,
                revision=self._revision,
                model_fingerprint=self._model_fingerprint,
                embedding_dimension=self._embedding_dimension,
                validator=self._validator,
            )
        return item

    def __contains__(self, value: object) -> bool:
        return any(item == value for item in self)

    def count(self, value: object) -> int:
        return sum(item == value for item in self)

    def index(self, value: object, start: int = 0, stop: int | None = None) -> int:
        self.assert_current()
        try:
            if stop is None:
                result = tuple.index(self, value, start)
            else:
                result = tuple.index(self, value, start, stop)
        except ValueError:
            self.assert_current()
            raise
        self.assert_current()
        return result

    def __add__(self, value: object) -> IdentityTemplateSnapshot:
        raise IdentityStoreError("identity template snapshot cannot be transformed")

    def __mul__(self, value: object) -> IdentityTemplateSnapshot:
        raise IdentityStoreError("identity template snapshot cannot be transformed")

    def __rmul__(self, value: object) -> IdentityTemplateSnapshot:
        raise IdentityStoreError("identity template snapshot cannot be transformed")

    def __repr__(self) -> str:
        return (
            "IdentityTemplateSnapshot("
            f"template_count={tuple.__len__(self)}, revision={self._revision})"
        )


def _absolute_private_root(value: str | Path) -> Path:
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise IdentityStoreError("identity store requires a private POSIX root")
    try:
        source = Path(value)
    except (TypeError, ValueError):
        raise IdentityStoreError("identity store path is invalid") from None
    if not source.is_absolute():
        raise IdentityStoreError("identity store path must be absolute")
    try:
        absolute = Path(os.path.abspath(os.fspath(source)))
        home_root = Path.home().resolve(strict=True)
        project_root = Path(__file__).resolve(strict=True).parents[2]
    except (OSError, TypeError, ValueError):
        raise IdentityStoreError("identity store path is invalid") from None
    lowered = tuple(part.lower() for part in absolute.parts)
    if len(lowered) >= 3 and lowered[:3] == ("/", "mnt", "c"):
        raise IdentityStoreError("identity store cannot use a Windows-mounted path")
    if absolute in {Path("/"), home_root, project_root}:
        raise IdentityStoreError("identity store requires a dedicated private root")
    return absolute


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise IdentityStoreError("identity store path could not be inspected") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise IdentityStoreError("identity store path cannot contain symlinks")


def _validate_owner_and_mode(metadata: os.stat_result, expected_mode: int, kind: str) -> None:
    if metadata.st_uid != os.geteuid():
        raise IdentityStoreError(f"identity store {kind} owner is unsafe")
    if stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise IdentityStoreError(f"identity store {kind} permissions are unsafe")


def _prepare_root(path: Path, *, create: bool) -> None:
    _reject_symlink_components(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if not create:
            raise IdentityStoreError("identity store directory does not exist") from None
        try:
            parent_metadata = path.parent.lstat()
            if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(parent_metadata.st_mode):
                raise OSError
            path.mkdir(mode=0o700, parents=False, exist_ok=False)
            path.chmod(0o700)
            metadata = path.lstat()
        except (FileExistsError, OSError):
            raise IdentityStoreError("identity store directory could not be created") from None
    except OSError:
        raise IdentityStoreError("identity store directory could not be inspected") from None
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise IdentityStoreError("identity store directory is unsafe")
    _validate_owner_and_mode(metadata, 0o700, "directory")


def _file_metadata(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise IdentityStoreError("identity store private file could not be inspected") from None


def _validate_private_file(path: Path, kind: str) -> os.stat_result:
    metadata = _file_metadata(path)
    if metadata is None or not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise IdentityStoreError(f"identity store {kind} file is unsafe")
    if metadata.st_nlink != 1:
        raise IdentityStoreError(f"identity store {kind} file has unsafe links")
    _validate_owner_and_mode(metadata, 0o600, kind)
    return metadata


def _metadata_signature(
    metadata: os.stat_result,
    *,
    include_content_metadata: bool,
) -> tuple[int, ...]:
    signature = (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_uid),
        int(metadata.st_nlink),
    )
    if include_content_metadata:
        return signature + (int(metadata.st_size), int(metadata.st_mtime_ns))
    return signature


def _read_only_layout_snapshot(
    root: Path,
    key_path: Path,
    database_path: Path,
    wal_path: Path,
    shm_path: Path,
) -> tuple[bool, tuple[object, ...]]:
    """Validate and snapshot a complete store without opening SQLite.

    A read-only SQLite client can update lock bytes in an existing shared-memory
    sidecar. Its inode and permissions remain part of the snapshot, while its size
    and mtime are intentionally excluded. All other entries are compared exactly.
    """

    _prepare_root(root, create=False)
    key_metadata = _file_metadata(key_path)
    database_metadata = _file_metadata(database_path)
    if (key_metadata is None) != (database_metadata is None):
        raise IdentityStoreError("identity store files are incomplete")
    if key_metadata is None:
        raise IdentityStoreError("identity store files do not exist")
    _validate_private_file(key_path, "key")
    _validate_private_file(database_path, "database")

    wal_metadata = _file_metadata(wal_path)
    shm_metadata = _file_metadata(shm_path)
    if (wal_metadata is None) != (shm_metadata is None):
        raise IdentityStoreError("identity store database sidecars are incomplete")
    sidecars_present = wal_metadata is not None
    if sidecars_present:
        _validate_private_file(wal_path, "database sidecar")
        _validate_private_file(shm_path, "database sidecar")

    try:
        root_metadata = root.lstat()
        entries: list[tuple[str, tuple[int, ...]]] = []
        for entry in root.iterdir():
            metadata = entry.lstat()
            entries.append(
                (
                    entry.name,
                    _metadata_signature(
                        metadata,
                        include_content_metadata=not (
                            sidecars_present and entry.name == shm_path.name
                        ),
                    ),
                )
            )
    except OSError:
        raise IdentityStoreError("identity store layout could not be inspected") from None
    layout = (
        _metadata_signature(root_metadata, include_content_metadata=True),
        tuple(sorted(entries)),
    )
    return sidecars_present, layout


def _create_private_file(path: Path, payload: bytes = b"") -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except (FileExistsError, OSError):
        raise IdentityStoreError("identity store private file could not be created") from None
    _validate_private_file(path, "private")


def _read_key(path: Path) -> bytes:
    _validate_private_file(path, "key")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            _validate_owner_and_mode(metadata, 0o600, "key")
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise IdentityStoreError("identity store key file is unsafe")
            key = os.read(descriptor, _KEY_BYTES + 1)
        finally:
            os.close(descriptor)
    except IdentityStoreError:
        raise
    except OSError:
        raise IdentityStoreError("identity store key could not be read") from None
    if len(key) != _KEY_BYTES:
        raise IdentityStoreError("identity store key material is invalid")
    return key


def _identifier(value: object, prefix: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(prefix)
        or len(value) != len(prefix) + 32
        or any(character not in _HEX_DIGITS for character in value[len(prefix) :])
    ):
        raise IdentityStoreError(f"identity store {field_name} is invalid")
    return value


def _fingerprint(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise IdentityStoreError("identity store model fingerprint is invalid")
    return value


def _dimension(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_EMBEDDING_DIMENSION:
        raise IdentityStoreError("identity store embedding dimension is invalid")
    return value


def _encode_embedding(embedding: Sequence[float]) -> tuple[bytes, int]:
    if isinstance(embedding, (str, bytes, bytearray, memoryview)):
        raise IdentityStoreError("identity store embedding is invalid")
    try:
        values = tuple(embedding)
    except TypeError:
        raise IdentityStoreError("identity store embedding is invalid") from None
    dimension = _dimension(len(values))
    parsed: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise IdentityStoreError("identity store embedding contains an invalid value")
        number = float(item)
        if not math.isfinite(number):
            raise IdentityStoreError("identity store embedding contains a nonfinite value")
        parsed.append(number)
    norm = math.hypot(*parsed)
    if not math.isfinite(norm) or norm == 0.0:
        raise IdentityStoreError("identity store embedding norm is invalid")
    normalized = tuple(item / norm for item in parsed)
    try:
        first_pass = struct.pack(f"<{dimension}f", *normalized)
        quantized = struct.unpack(f"<{dimension}f", first_pass)
    except (OverflowError, struct.error):
        raise IdentityStoreError("identity store embedding could not be encoded") from None
    quantized_norm = math.hypot(*quantized)
    if not math.isfinite(quantized_norm) or quantized_norm == 0.0:
        raise IdentityStoreError("identity store embedding norm is invalid")
    try:
        payload = struct.pack(
            f"<{dimension}f",
            *(item / quantized_norm for item in quantized),
        )
    except (OverflowError, struct.error):
        raise IdentityStoreError("identity store embedding could not be encoded") from None
    return payload, dimension


def _decode_embedding(payload: bytes, dimension: int) -> tuple[float, ...]:
    dimension = _dimension(dimension)
    if not isinstance(payload, bytes) or len(payload) != dimension * 4:
        raise IdentityStoreError("identity template decryption failed")
    try:
        values = struct.unpack(f"<{dimension}f", payload)
    except struct.error:
        raise IdentityStoreError("identity template decryption failed") from None
    if any(not math.isfinite(item) for item in values):
        raise IdentityStoreError("identity template decryption failed")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise IdentityStoreError("identity template decryption failed")
    return tuple(item / norm for item in values)


def _associated_data(
    *,
    profile_id: str,
    template_id: str,
    declared_kind: DeclaredKind,
    model_fingerprint: str,
    embedding_dimension: int,
) -> bytes:
    return json.dumps(
        {
            "declared_kind": declared_kind.value,
            "embedding_dimension": embedding_dimension,
            "model_fingerprint": model_fingerprint,
            "profile_id": profile_id,
            "schema": _AAD_SCHEMA,
            "template_id": template_id,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


class IdentityTemplateStore:
    """A local-only, per-row authenticated store for face embedding templates."""

    def __init__(self, root: str | Path) -> None:
        self._initialize(root, create=True, allow_migration=True)

    @classmethod
    def open_existing(cls, root: str | Path) -> IdentityTemplateStore:
        """Open a complete current store without creating files or migrating schema."""

        # Prove the store is already a valid v2 layout through the zero-key,
        # zero-write reader before any mutable SQLite connection is possible.
        cls.list_profiles_read_only(root)
        store = cls.__new__(cls)
        store._initialize(root, create=False, allow_migration=False)
        return store

    def _initialize(
        self,
        root: str | Path,
        *,
        create: bool,
        allow_migration: bool,
    ) -> None:
        aesgcm_type = _load_aesgcm()
        self._root = _absolute_private_root(root)
        self._key_path = self._root / "identity.key"
        self._database_path = self._root / "identity.sqlite3"
        self._wal_path = self._root / "identity.sqlite3-wal"
        self._shm_path = self._root / "identity.sqlite3-shm"
        _prepare_root(self._root, create=create)

        key_metadata = _file_metadata(self._key_path)
        database_metadata = _file_metadata(self._database_path)
        if (key_metadata is None) != (database_metadata is None):
            raise IdentityStoreError("identity store files are incomplete")
        if key_metadata is None:
            if not create:
                raise IdentityStoreError("identity store files do not exist")
            if (
                _file_metadata(self._wal_path) is not None
                or _file_metadata(self._shm_path) is not None
            ):
                raise IdentityStoreError("identity store files are incomplete")
            _create_private_file(self._key_path, os.urandom(_KEY_BYTES))
            _create_private_file(self._database_path)
        else:
            _validate_private_file(self._key_path, "key")
            _validate_private_file(self._database_path, "database")
            self._validate_sidecars()

        key = _read_key(self._key_path)
        try:
            self._aead: _Aead = aesgcm_type(key)
        except Exception:
            raise IdentityStoreError(
                "identity encryption backend could not be initialized"
            ) from None
        self._initialize_schema(allow_migration=allow_migration)

    def _validate_sidecars(self) -> None:
        for path in (self._wal_path, self._shm_path):
            if _file_metadata(path) is not None:
                _validate_private_file(path, "database sidecar")

    def _assert_files_safe(self) -> None:
        _reject_symlink_components(self._root)
        try:
            metadata = self._root.lstat()
        except OSError:
            raise IdentityStoreError("identity store directory could not be inspected") from None
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise IdentityStoreError("identity store directory is unsafe")
        _validate_owner_and_mode(metadata, 0o700, "directory")
        _validate_private_file(self._key_path, "key")
        _validate_private_file(self._database_path, "database")
        self._validate_sidecars()

    def _connect(self) -> sqlite3.Connection:
        self._assert_files_safe()
        connection: sqlite3.Connection | None = None
        try:
            database_uri = f"{self._database_path.as_uri()}?mode=rw"
            connection = sqlite3.connect(database_uri, uri=True, timeout=5.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA secure_delete = ON")
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()
            foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()
            secure_delete = connection.execute("PRAGMA secure_delete").fetchone()
            if (
                mode is None
                or str(mode[0]).lower() != "wal"
                or foreign_keys is None
                or foreign_keys[0] != 1
                or secure_delete is None
                or secure_delete[0] != 1
            ):
                raise IdentityStoreError("identity store database safety settings are unavailable")
            self._assert_files_safe()
            return connection
        except IdentityStoreError:
            if connection is not None:
                connection.close()
            raise
        except (OSError, sqlite3.Error):
            if connection is not None:
                connection.close()
            raise IdentityStoreError("identity store database could not be opened") from None

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        except IdentityStoreError:
            raise
        except sqlite3.Error:
            raise IdentityStoreError("identity store database operation failed") from None
        finally:
            connection.close()
            self._assert_files_safe()

    def _initialize_schema(self, *, allow_migration: bool) -> None:
        with self._connection() as connection:
            version_row = connection.execute("PRAGMA user_version").fetchone()
            version = int(version_row[0]) if version_row is not None else -1
            if version == 0:
                if not allow_migration:
                    raise IdentityStoreError("identity store schema version is unsupported")
                tables = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                triggers = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                ).fetchall()
                if tables or triggers:
                    raise IdentityStoreError("identity store schema is invalid")
                connection.executescript(
                    f"""
                    BEGIN IMMEDIATE;
                    CREATE TABLE profiles (
                        profile_id TEXT PRIMARY KEY,
                        declared_kind TEXT NOT NULL
                            CHECK (declared_kind IN ('baby', 'adult', 'person'))
                    );
                    CREATE TABLE templates (
                        template_id TEXT PRIMARY KEY,
                        profile_id TEXT NOT NULL,
                        model_fingerprint TEXT NOT NULL,
                        embedding_dimension INTEGER NOT NULL
                            CHECK (embedding_dimension BETWEEN 1 AND 4096),
                        nonce BLOB NOT NULL UNIQUE CHECK (length(nonce) = 12),
                        ciphertext BLOB NOT NULL CHECK (length(ciphertext) >= 20),
                        FOREIGN KEY (profile_id) REFERENCES profiles(profile_id)
                            ON DELETE CASCADE
                    );
                    CREATE TABLE store_state (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        revision INTEGER NOT NULL
                            CHECK (revision BETWEEN 0 AND {_MAX_REVISION})
                    );
                    INSERT INTO store_state (singleton, revision) VALUES (1, 0);
                    CREATE INDEX templates_profile_idx ON templates(profile_id);
                    CREATE INDEX templates_model_idx
                        ON templates(model_fingerprint, embedding_dimension);
                    {_REVISION_TRIGGER_SQL}
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )
            elif version == 1:
                if not allow_migration:
                    raise IdentityStoreError("identity store schema version is unsupported")
                self._validate_schema(connection, legacy=True)
                try:
                    connection.executescript(
                        f"""
                        BEGIN IMMEDIATE;
                        CREATE TABLE store_state (
                            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                            revision INTEGER NOT NULL
                                CHECK (revision BETWEEN 0 AND {_MAX_REVISION})
                        );
                        INSERT INTO store_state (singleton, revision) VALUES (1, 0);
                        {_REVISION_TRIGGER_SQL}
                        PRAGMA user_version = 2;
                        COMMIT;
                        """
                    )
                except sqlite3.Error:
                    if connection.in_transaction:
                        connection.rollback()
                    raise IdentityStoreError("identity store schema migration failed") from None
            elif version != _SCHEMA_VERSION:
                raise IdentityStoreError("identity store schema version is unsupported")
            self._validate_schema(connection)

    @staticmethod
    def _validate_schema(
        connection: sqlite3.Connection,
        *,
        legacy: bool = False,
    ) -> None:
        expected = {
            "profiles": {"profile_id", "declared_kind"},
            "templates": {
                "template_id",
                "profile_id",
                "model_fingerprint",
                "embedding_dimension",
                "nonce",
                "ciphertext",
            },
        }
        if not legacy:
            expected["store_state"] = {"singleton", "revision"}
        actual_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        if actual_tables != set(expected):
            raise IdentityStoreError("identity store schema is invalid")
        for table, columns in expected.items():
            actual = {
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if actual != columns:
                raise IdentityStoreError("identity store schema is invalid")
        actual_triggers = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        expected_triggers = set() if legacy else set(_REVISION_TRIGGER_NAMES)
        if actual_triggers != expected_triggers:
            raise IdentityStoreError("identity store schema is invalid")
        foreign_keys = connection.execute("PRAGMA foreign_key_list(templates)").fetchall()
        if not any(
            row[2] == "profiles"
            and row[3] == "profile_id"
            and row[4] == "profile_id"
            and str(row[6]).upper() == "CASCADE"
            for row in foreign_keys
        ):
            raise IdentityStoreError("identity store schema is invalid")
        inconsistent_model = connection.execute(
            """
            SELECT 1
            FROM templates
            GROUP BY model_fingerprint
            HAVING count(DISTINCT embedding_dimension) != 1
            LIMIT 1
            """
        ).fetchone()
        if inconsistent_model is not None:
            raise IdentityStoreError("identity store model metadata is inconsistent")
        if not legacy:
            IdentityTemplateStore._read_revision(connection)

    @staticmethod
    def _read_revision(connection: sqlite3.Connection) -> int:
        rows = connection.execute("SELECT singleton, revision FROM store_state").fetchall()
        if (
            len(rows) != 1
            or type(rows[0][0]) is not int
            or int(rows[0][0]) != 1
            or type(rows[0][1]) is not int
            or not 0 <= int(rows[0][1]) <= _MAX_REVISION
        ):
            raise IdentityStoreError("identity store revision is invalid")
        return int(rows[0][1])

    @staticmethod
    def _assert_revision_advanced(
        connection: sqlite3.Connection,
        previous_revision: int,
    ) -> None:
        if IdentityTemplateStore._read_revision(connection) <= previous_revision:
            raise IdentityStoreError("identity store revision did not advance")

    def _assert_snapshot_revision(self, expected_revision: int) -> None:
        with self._connection() as connection:
            current_revision = self._read_revision(connection)
        if current_revision != expected_revision:
            raise IdentityStoreError("identity template snapshot is stale")

    @staticmethod
    def _validated_kind(declared_kind: DeclaredKind) -> DeclaredKind:
        if not isinstance(declared_kind, DeclaredKind) or declared_kind not in _PROFILE_KINDS:
            raise IdentityStoreError("identity store declared kind is invalid")
        return declared_kind

    @staticmethod
    def _model_dimensions(
        connection: sqlite3.Connection,
        model_fingerprint: str,
    ) -> set[int]:
        try:
            dimensions = {
                _dimension(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT embedding_dimension FROM templates "
                    "WHERE model_fingerprint = ?",
                    (model_fingerprint,),
                ).fetchall()
            }
        except (IdentityStoreError, TypeError, ValueError):
            raise IdentityStoreError("identity store model metadata is inconsistent") from None
        if len(dimensions) > 1:
            raise IdentityStoreError("identity store model metadata is inconsistent")
        return dimensions

    def _insert_encrypted_template(
        self,
        connection: sqlite3.Connection,
        *,
        profile_id: str,
        declared_kind: DeclaredKind,
        template_id: str,
        model_fingerprint: str,
        plaintext: bytes,
        embedding_dimension: int,
    ) -> None:
        nonce = os.urandom(_NONCE_BYTES)
        aad = _associated_data(
            profile_id=profile_id,
            template_id=template_id,
            declared_kind=declared_kind,
            model_fingerprint=model_fingerprint,
            embedding_dimension=embedding_dimension,
        )
        try:
            ciphertext = self._aead.encrypt(nonce, plaintext, aad)
        except Exception:
            raise IdentityStoreError("identity template encryption failed") from None
        if (
            not isinstance(ciphertext, bytes)
            or len(nonce) != _NONCE_BYTES
            or len(ciphertext) < len(plaintext) + 16
        ):
            raise IdentityStoreError("identity template encryption failed")
        connection.execute(
            """
            INSERT INTO templates (
                template_id,
                profile_id,
                model_fingerprint,
                embedding_dimension,
                nonce,
                ciphertext
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                template_id,
                profile_id,
                model_fingerprint,
                embedding_dimension,
                nonce,
                ciphertext,
            ),
        )

    def create_profile(self, declared_kind: DeclaredKind) -> str:
        declared_kind = self._validated_kind(declared_kind)
        profile_id = f"{_PROFILE_PREFIX}{secrets.token_hex(16)}"
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                revision = self._read_revision(connection)
                connection.execute(
                    "INSERT INTO profiles (profile_id, declared_kind) VALUES (?, ?)",
                    (profile_id, declared_kind.value),
                )
                self._assert_revision_advanced(connection, revision)
                connection.commit()
            except IdentityStoreError:
                connection.rollback()
                raise
            except sqlite3.IntegrityError:
                connection.rollback()
                raise IdentityStoreError("identity store transaction failed") from None
        return profile_id

    def add_template(
        self,
        *,
        profile_id: str,
        model_fingerprint: str,
        embedding: Sequence[float],
    ) -> str:
        profile_id = _identifier(profile_id, _PROFILE_PREFIX, "profile ID")
        model_fingerprint = _fingerprint(model_fingerprint)
        plaintext, embedding_dimension = _encode_embedding(embedding)
        template_id = f"{_TEMPLATE_PREFIX}{secrets.token_hex(16)}"
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                revision = self._read_revision(connection)
                profile = connection.execute(
                    "SELECT declared_kind FROM profiles WHERE profile_id = ?",
                    (profile_id,),
                ).fetchone()
                if profile is None:
                    raise IdentityStoreError("identity profile does not exist")
                try:
                    declared_kind = DeclaredKind(str(profile[0]))
                except ValueError:
                    raise IdentityStoreError("identity store profile metadata is invalid") from None
                if declared_kind not in _PROFILE_KINDS:
                    raise IdentityStoreError("identity store profile metadata is invalid")
                dimensions = self._model_dimensions(connection, model_fingerprint)
                if dimensions and dimensions != {embedding_dimension}:
                    raise IdentityStoreError("identity store model metadata is inconsistent")
                self._insert_encrypted_template(
                    connection,
                    profile_id=profile_id,
                    declared_kind=declared_kind,
                    template_id=template_id,
                    model_fingerprint=model_fingerprint,
                    plaintext=plaintext,
                    embedding_dimension=embedding_dimension,
                )
                self._assert_revision_advanced(connection, revision)
                connection.commit()
            except IdentityStoreError:
                connection.rollback()
                raise
            except sqlite3.IntegrityError:
                connection.rollback()
                raise IdentityStoreError("identity store transaction failed") from None
        return template_id

    def enroll_profile(
        self,
        *,
        declared_kind: DeclaredKind,
        model_fingerprint: str,
        embeddings: Sequence[Sequence[float]],
    ) -> IdentityEnrollmentSummary:
        """Create one profile and all of its encrypted templates in one transaction."""

        declared_kind = self._validated_kind(declared_kind)
        model_fingerprint = _fingerprint(model_fingerprint)
        if isinstance(embeddings, (str, bytes, bytearray, memoryview)):
            raise IdentityStoreError("identity store embedding batch is invalid")
        try:
            encoded = tuple(_encode_embedding(embedding) for embedding in embeddings)
        except TypeError:
            raise IdentityStoreError("identity store embedding batch is invalid") from None
        if not encoded:
            raise IdentityStoreError("identity store embedding batch is empty")
        embedding_dimensions = {item[1] for item in encoded}
        if len(embedding_dimensions) != 1:
            raise IdentityStoreError("identity store model metadata is inconsistent")
        embedding_dimension = next(iter(embedding_dimensions))
        profile_id = f"{_PROFILE_PREFIX}{secrets.token_hex(16)}"
        template_ids = tuple(f"{_TEMPLATE_PREFIX}{secrets.token_hex(16)}" for _item in encoded)

        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                revision = self._read_revision(connection)
                dimensions = self._model_dimensions(connection, model_fingerprint)
                if dimensions and dimensions != {embedding_dimension}:
                    raise IdentityStoreError("identity store model metadata is inconsistent")
                connection.execute(
                    "INSERT INTO profiles (profile_id, declared_kind) VALUES (?, ?)",
                    (profile_id, declared_kind.value),
                )
                for template_id, (plaintext, dimension) in zip(
                    template_ids,
                    encoded,
                    strict=True,
                ):
                    self._insert_encrypted_template(
                        connection,
                        profile_id=profile_id,
                        declared_kind=declared_kind,
                        template_id=template_id,
                        model_fingerprint=model_fingerprint,
                        plaintext=plaintext,
                        embedding_dimension=dimension,
                    )
                self._assert_revision_advanced(connection, revision)
                connection.commit()
            except IdentityStoreError:
                connection.rollback()
                raise
            except sqlite3.IntegrityError:
                connection.rollback()
                raise IdentityStoreError("identity store enrollment failed") from None
        return IdentityEnrollmentSummary(
            profile_id=profile_id,
            template_ids=template_ids,
            declared_kind=declared_kind,
            model_fingerprint=model_fingerprint,
            embedding_dimension=embedding_dimension,
        )

    def summary(self) -> IdentityStoreSummary:
        with self._connection() as connection:
            profile_count = int(connection.execute("SELECT count(*) FROM profiles").fetchone()[0])
            template_count = int(connection.execute("SELECT count(*) FROM templates").fetchone()[0])
        return IdentityStoreSummary(
            profile_count=profile_count,
            template_count=template_count,
        )

    def list_profiles(self) -> tuple[IdentityProfileSummary, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT p.profile_id, p.declared_kind, count(t.template_id) AS template_count
                FROM profiles AS p
                LEFT JOIN templates AS t ON t.profile_id = p.profile_id
                GROUP BY p.profile_id, p.declared_kind
                ORDER BY p.profile_id
                """
            ).fetchall()
        summaries: list[IdentityProfileSummary] = []
        for row in rows:
            try:
                profile_id = _identifier(str(row[0]), _PROFILE_PREFIX, "profile ID")
                kind = DeclaredKind(str(row[1]))
                template_count = int(row[2])
                if template_count < 0:
                    raise ValueError
            except (IdentityStoreError, TypeError, ValueError):
                raise IdentityStoreError("identity store profile metadata is invalid") from None
            summaries.append(
                IdentityProfileSummary(
                    profile_id=profile_id,
                    declared_kind=kind,
                    template_count=template_count,
                )
            )
        return tuple(summaries)

    @classmethod
    def list_profiles_read_only(
        cls,
        root: str | Path,
    ) -> tuple[IdentityProfileSummary, ...]:
        """Read opaque profile metadata without keys, migrations, or new files.

        A clean database with no WAL sidecars can be opened as immutable.  When
        WAL state exists, the ordinary read-only URI is required so committed
        rows in the WAL remain visible; the pre-existing shared-memory file may
        receive SQLite lock-byte updates, but no store entry may be created,
        removed, replaced, or have its protected metadata changed.
        """

        private_root = _absolute_private_root(root)
        key_path = private_root / "identity.key"
        database_path = private_root / "identity.sqlite3"
        wal_path = private_root / "identity.sqlite3-wal"
        shm_path = private_root / "identity.sqlite3-shm"
        sidecars_present, before = _read_only_layout_snapshot(
            private_root,
            key_path,
            database_path,
            wal_path,
            shm_path,
        )
        connection: sqlite3.Connection | None = None
        try:
            suffix = "?mode=ro" if sidecars_present else "?mode=ro&immutable=1"
            connection = sqlite3.connect(
                f"{database_path.as_uri()}{suffix}",
                uri=True,
                timeout=5.0,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            query_only = connection.execute("PRAGMA query_only").fetchone()
            version_row = connection.execute("PRAGMA user_version").fetchone()
            if (
                query_only is None
                or int(query_only[0]) != 1
                or version_row is None
                or int(version_row[0]) != _SCHEMA_VERSION
            ):
                raise IdentityStoreError("identity store read-only mode is unavailable")
            cls._validate_schema(connection)
            rows = connection.execute(
                """
                SELECT p.profile_id, p.declared_kind, count(t.template_id) AS template_count
                FROM profiles AS p
                LEFT JOIN templates AS t ON t.profile_id = p.profile_id
                GROUP BY p.profile_id, p.declared_kind
                ORDER BY p.profile_id
                """
            ).fetchall()
        except IdentityStoreError:
            raise
        except (OSError, sqlite3.Error, TypeError, ValueError):
            raise IdentityStoreError("identity store could not be read without mutation") from None
        finally:
            if connection is not None:
                connection.close()

        after_sidecars, after = _read_only_layout_snapshot(
            private_root,
            key_path,
            database_path,
            wal_path,
            shm_path,
        )
        if after_sidecars != sidecars_present or after != before:
            raise IdentityStoreError("identity store changed during read-only access")

        summaries: list[IdentityProfileSummary] = []
        for row in rows:
            try:
                profile_id = _identifier(str(row[0]), _PROFILE_PREFIX, "profile ID")
                kind = DeclaredKind(str(row[1]))
                template_count = int(row[2])
                if kind not in _PROFILE_KINDS or template_count < 0:
                    raise ValueError
            except (IdentityStoreError, TypeError, ValueError):
                raise IdentityStoreError("identity store profile metadata is invalid") from None
            summaries.append(
                IdentityProfileSummary(
                    profile_id=profile_id,
                    declared_kind=kind,
                    template_count=template_count,
                )
            )
        return tuple(summaries)

    def list_templates(
        self,
        profile_id: str | None = None,
    ) -> tuple[IdentityTemplateSummary, ...]:
        parameters: tuple[object, ...] = ()
        where = ""
        if profile_id is not None:
            profile_id = _identifier(profile_id, _PROFILE_PREFIX, "profile ID")
            where = " WHERE profile_id = ?"
            parameters = (profile_id,)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT template_id, profile_id, model_fingerprint, embedding_dimension
                FROM templates
                """
                + where
                + " ORDER BY template_id",
                parameters,
            ).fetchall()
        summaries: list[IdentityTemplateSummary] = []
        for row in rows:
            try:
                summary = IdentityTemplateSummary(
                    template_id=_identifier(str(row[0]), _TEMPLATE_PREFIX, "template ID"),
                    profile_id=_identifier(str(row[1]), _PROFILE_PREFIX, "profile ID"),
                    model_fingerprint=_fingerprint(str(row[2])),
                    embedding_dimension=_dimension(int(row[3])),
                )
            except (IdentityStoreError, TypeError, ValueError):
                raise IdentityStoreError("identity store template metadata is invalid") from None
            summaries.append(summary)
        return tuple(summaries)

    def load_templates_for_matching(
        self,
        *,
        model_fingerprint: str,
        embedding_dimension: int,
    ) -> IdentityTemplateSnapshot:
        model_fingerprint = _fingerprint(model_fingerprint)
        embedding_dimension = _dimension(embedding_dimension)
        with self._connection() as connection:
            connection.execute("BEGIN")
            revision = self._read_revision(connection)
            dimensions = self._model_dimensions(connection, model_fingerprint)
            if dimensions and dimensions != {embedding_dimension}:
                raise IdentityStoreError("identity store model metadata is inconsistent")
            rows = connection.execute(
                """
                SELECT
                    t.template_id,
                    t.profile_id,
                    p.declared_kind,
                    t.nonce,
                    t.ciphertext
                FROM templates AS t
                JOIN profiles AS p ON p.profile_id = t.profile_id
                WHERE t.model_fingerprint = ? AND t.embedding_dimension = ?
                ORDER BY t.template_id
                """,
                (model_fingerprint, embedding_dimension),
            ).fetchall()
            connection.commit()
        templates: list[InMemoryIdentityTemplate] = []
        for row in rows:
            try:
                template_id = _identifier(str(row[0]), _TEMPLATE_PREFIX, "template ID")
                profile_id = _identifier(str(row[1]), _PROFILE_PREFIX, "profile ID")
                declared_kind = DeclaredKind(str(row[2]))
                if declared_kind not in _PROFILE_KINDS:
                    raise ValueError
                if not isinstance(row[3], bytes) or not isinstance(row[4], bytes):
                    raise TypeError
                nonce = bytes(row[3])
                ciphertext = bytes(row[4])
                if len(nonce) != _NONCE_BYTES or len(ciphertext) < embedding_dimension * 4 + 16:
                    raise ValueError
            except (IdentityStoreError, TypeError, ValueError):
                raise IdentityStoreError("identity template decryption failed") from None
            aad = _associated_data(
                profile_id=profile_id,
                template_id=template_id,
                declared_kind=declared_kind,
                model_fingerprint=model_fingerprint,
                embedding_dimension=embedding_dimension,
            )
            try:
                plaintext = self._aead.decrypt(nonce, ciphertext, aad)
            except Exception:
                raise IdentityStoreError("identity template decryption failed") from None
            embedding = _decode_embedding(plaintext, embedding_dimension)
            templates.append(
                InMemoryIdentityTemplate(
                    template_id=template_id,
                    profile_id=profile_id,
                    declared_kind=declared_kind,
                    model_fingerprint=model_fingerprint,
                    embedding=embedding,
                )
            )
        return IdentityTemplateSnapshot(
            templates,
            revision=revision,
            model_fingerprint=model_fingerprint,
            embedding_dimension=embedding_dimension,
            validator=self._assert_snapshot_revision,
        )

    @staticmethod
    def _checkpoint(connection: sqlite3.Connection) -> None:
        try:
            result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result is None or int(result[0]) != 0:
                raise IdentityStoreError("identity store deletion checkpoint failed")
        except IdentityStoreError:
            raise
        except (sqlite3.Error, TypeError, ValueError):
            raise IdentityStoreError("identity store deletion checkpoint failed") from None

    def delete_template(self, template_id: str) -> IdentityDeletionSummary:
        template_id = _identifier(template_id, _TEMPLATE_PREFIX, "template ID")
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                revision = self._read_revision(connection)
                cursor = connection.execute(
                    "DELETE FROM templates WHERE template_id = ?",
                    (template_id,),
                )
                if cursor.rowcount != 1:
                    raise IdentityStoreError("identity template does not exist")
                self._assert_revision_advanced(connection, revision)
                profile_count = int(
                    connection.execute("SELECT count(*) FROM profiles").fetchone()[0]
                )
                template_count = int(
                    connection.execute("SELECT count(*) FROM templates").fetchone()[0]
                )
                connection.commit()
            except IdentityStoreError:
                connection.rollback()
                raise
            except sqlite3.IntegrityError:
                connection.rollback()
                raise IdentityStoreError("identity store deletion failed") from None
            try:
                self._checkpoint(connection)
                checkpoint_completed = True
            except (IdentityStoreError, sqlite3.Error):
                checkpoint_completed = False
        return IdentityDeletionSummary(
            deleted_profiles=0,
            deleted_templates=1,
            remaining_profiles=profile_count,
            remaining_templates=template_count,
            checkpoint_completed=checkpoint_completed,
            cleanup_pending=not checkpoint_completed,
        )

    def delete_profile(self, profile_id: str) -> IdentityDeletionSummary:
        profile_id = _identifier(profile_id, _PROFILE_PREFIX, "profile ID")
        with self._connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                revision = self._read_revision(connection)
                template_row = connection.execute(
                    "SELECT count(*) FROM templates WHERE profile_id = ?",
                    (profile_id,),
                ).fetchone()
                deleted_templates = int(template_row[0])
                cursor = connection.execute(
                    "DELETE FROM profiles WHERE profile_id = ?",
                    (profile_id,),
                )
                if cursor.rowcount != 1:
                    raise IdentityStoreError("identity profile does not exist")
                self._assert_revision_advanced(connection, revision)
                profile_count = int(
                    connection.execute("SELECT count(*) FROM profiles").fetchone()[0]
                )
                template_count = int(
                    connection.execute("SELECT count(*) FROM templates").fetchone()[0]
                )
                connection.commit()
            except IdentityStoreError:
                connection.rollback()
                raise
            except sqlite3.IntegrityError:
                connection.rollback()
                raise IdentityStoreError("identity store deletion failed") from None
            try:
                self._checkpoint(connection)
                checkpoint_completed = True
            except (IdentityStoreError, sqlite3.Error):
                checkpoint_completed = False
        return IdentityDeletionSummary(
            deleted_profiles=1,
            deleted_templates=deleted_templates,
            remaining_profiles=profile_count,
            remaining_templates=template_count,
            checkpoint_completed=checkpoint_completed,
            cleanup_pending=not checkpoint_completed,
        )

    def retry_cleanup(self) -> IdentityCleanupSummary:
        """Retry the post-deletion WAL truncation without changing identity rows."""

        with self._connection() as connection:
            try:
                self._checkpoint(connection)
                checkpoint_completed = True
            except (IdentityStoreError, sqlite3.Error):
                checkpoint_completed = False
        return IdentityCleanupSummary(
            checkpoint_completed=checkpoint_completed,
            cleanup_pending=not checkpoint_completed,
        )
