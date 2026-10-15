from __future__ import annotations

import hashlib
import hmac
import math
import sqlite3
import stat
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baby_monitor.identity_domain import DeclaredKind
from baby_monitor.identity_matching import IdentityTemplate
from baby_monitor.identity_store import (
    IdentityStoreError,
    IdentityTemplateStore,
    InMemoryIdentityTemplate,
)


class _FakeAESGCM:
    """Authenticated test double; it is deliberately not a production fallback."""

    plaintexts: list[bytes] = []

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError
        self._key = key

    def _stream(self, nonce: bytes, aad: bytes, length: int) -> bytes:
        blocks: list[bytes] = []
        counter = 0
        while sum(map(len, blocks)) < length:
            blocks.append(
                hmac.new(
                    self._key,
                    b"stream\x00" + nonce + aad + counter.to_bytes(4, "big"),
                    hashlib.sha256,
                ).digest()
            )
            counter += 1
        return b"".join(blocks)[:length]

    def encrypt(self, nonce: bytes, data: bytes, associated_data: bytes) -> bytes:
        type(self).plaintexts.append(data)
        stream = self._stream(nonce, associated_data, len(data))
        ciphertext = bytes(left ^ right for left, right in zip(data, stream, strict=True))
        tag = hmac.new(
            self._key,
            b"tag\x00" + nonce + associated_data + ciphertext,
            hashlib.sha256,
        ).digest()[:16]
        return ciphertext + tag

    def decrypt(self, nonce: bytes, data: bytes, associated_data: bytes) -> bytes:
        if len(data) < 16:
            raise ValueError
        ciphertext, tag = data[:-16], data[-16:]
        expected = hmac.new(
            self._key,
            b"tag\x00" + nonce + associated_data + ciphertext,
            hashlib.sha256,
        ).digest()[:16]
        if not hmac.compare_digest(tag, expected):
            raise ValueError
        stream = self._stream(nonce, associated_data, len(ciphertext))
        return bytes(left ^ right for left, right in zip(ciphertext, stream, strict=True))


class IdentityTemplateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAESGCM.plaintexts.clear()
        patcher = mock.patch(
            "baby_monitor.identity_store._load_aesgcm",
            return_value=_FakeAESGCM,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _new_store(self, parent: str) -> tuple[Path, IdentityTemplateStore]:
        root = Path(parent) / "private-identity"
        return root, IdentityTemplateStore(root)

    def test_crud_normalizes_encrypts_and_returns_matcher_templates_without_repr_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, store = self._new_store(directory)
            baby_id = store.create_profile(DeclaredKind.BABY)
            adult_id = store.create_profile(DeclaredKind.ADULT)
            model = "a" * 64

            baby_template = store.add_template(
                profile_id=baby_id,
                model_fingerprint=model,
                embedding=(3.0, 4.0),
            )
            adult_template = store.add_template(
                profile_id=adult_id,
                model_fingerprint=model,
                embedding=(0.0, 8.0),
            )
            store.add_template(
                profile_id=baby_id,
                model_fingerprint="b" * 64,
                embedding=(1.0, 0.0, 0.0),
            )

            self.assertTrue(baby_id.startswith("profile_"))
            self.assertTrue(baby_template.startswith("template_"))
            self.assertNotEqual(baby_template, adult_template)
            self.assertEqual(
                {
                    "schema_version": 2,
                    "profile_count": 2,
                    "template_count": 3,
                    "embeddings_encrypted": True,
                    "metadata_encrypted": False,
                    "raw_images_stored": False,
                },
                store.summary().to_public_dict(),
            )
            profiles = {item.profile_id: item for item in store.list_profiles()}
            self.assertEqual(2, profiles[baby_id].template_count)
            self.assertEqual(DeclaredKind.BABY, profiles[baby_id].declared_kind)
            self.assertEqual(1, profiles[adult_id].template_count)
            summaries = store.list_templates(baby_id)
            self.assertEqual(2, len(summaries))
            self.assertEqual({baby_id}, {item.profile_id for item in summaries})

            loaded = store.load_templates_for_matching(
                model_fingerprint=model,
                embedding_dimension=2,
            )
            self.assertEqual(2, len(loaded))
            self.assertTrue(all(isinstance(item, IdentityTemplate) for item in loaded))
            self.assertTrue(all(isinstance(item, InMemoryIdentityTemplate) for item in loaded))
            self.assertTrue(all(math.isclose(math.hypot(*item.embedding), 1.0) for item in loaded))
            by_profile = {item.profile_id: item for item in loaded}
            self.assertAlmostEqual(0.6, by_profile[baby_id].embedding[0], places=6)
            self.assertAlmostEqual(0.8, by_profile[baby_id].embedding[1], places=6)
            self.assertNotIn("embedding", repr(by_profile[baby_id]))
            self.assertFalse(hasattr(by_profile[baby_id], "to_dict"))
            self.assertFalse(hasattr(by_profile[baby_id], "to_json"))

            first_vector = struct.unpack("<2f", _FakeAESGCM.plaintexts[0])
            self.assertAlmostEqual(0.6, first_vector[0], places=6)
            self.assertAlmostEqual(0.8, first_vector[1], places=6)
            with sqlite3.connect(root / "identity.sqlite3") as connection:
                rows = connection.execute("SELECT nonce, ciphertext FROM templates").fetchall()
            self.assertEqual(3, len(rows))
            self.assertEqual(3, len({bytes(row[0]) for row in rows}))
            self.assertTrue(all(len(bytes(row[0])) == 12 for row in rows))
            self.assertTrue(all(bytes(row[1]) not in _FakeAESGCM.plaintexts for row in rows))

    def test_optional_crypto_backend_is_lazy_and_fails_before_creating_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "identity"
            with (
                mock.patch(
                    "baby_monitor.identity_store._load_aesgcm",
                    side_effect=IdentityStoreError("identity encryption backend is unavailable"),
                ),
                self.assertRaisesRegex(IdentityStoreError, "backend is unavailable"),
            ):
                IdentityTemplateStore(root)
            self.assertFalse(root.exists())

    def test_files_permissions_schema_and_public_results_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, store = self._new_store(directory)
            profile_id = store.create_profile(DeclaredKind.PERSON)
            store.add_template(
                profile_id=profile_id,
                model_fingerprint="c" * 64,
                embedding=(1.0, 2.0),
            )

            self.assertEqual(0o700, stat.S_IMODE(root.stat().st_mode))
            self.assertEqual(0o600, stat.S_IMODE((root / "identity.key").stat().st_mode))
            self.assertEqual(
                0o600,
                stat.S_IMODE((root / "identity.sqlite3").stat().st_mode),
            )
            for sidecar in (root / "identity.sqlite3-wal", root / "identity.sqlite3-shm"):
                if sidecar.exists():
                    self.assertEqual(0o600, stat.S_IMODE(sidecar.stat().st_mode))

            with sqlite3.connect(root / "identity.sqlite3") as connection:
                tables = {
                    str(row[0]): str(row[1])
                    for row in connection.execute(
                        "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
                    )
                }
                foreign_keys = connection.execute("PRAGMA foreign_key_list(templates)").fetchall()
                journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
            self.assertEqual({"profiles", "templates", "store_state"}, set(tables))
            schema_text = " ".join(tables.values()).lower()
            self.assertNotIn("image", schema_text)
            self.assertNotIn("name", schema_text)
            self.assertNotIn("path", schema_text)
            self.assertTrue(any(str(row[6]).upper() == "CASCADE" for row in foreign_keys))
            self.assertEqual("wal", journal_mode.lower())

            public_text = repr(store.summary().to_public_dict())
            public_text += repr([item.to_public_dict() for item in store.list_profiles()])
            public_text += repr([item.to_public_dict() for item in store.list_templates()])
            self.assertNotIn(directory, public_text)
            self.assertNotIn("identity.sqlite3", public_text)

    def test_atomic_enrollment_and_v1_migration_preserve_private_templates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, store = self._new_store(directory)
            model = "7" * 64
            enrollment = store.enroll_profile(
                declared_kind=DeclaredKind.BABY,
                model_fingerprint=model,
                embeddings=((3.0, 4.0), (4.0, 3.0)),
            )
            self.assertEqual(2, len(enrollment.template_ids))
            self.assertEqual(2, store.summary().template_count)
            self.assertNotIn("(3.0, 4.0)", repr(enrollment))
            self.assertNotIn("(4.0, 3.0)", repr(enrollment))
            self.assertNotIn(directory, repr(enrollment.to_public_dict()))

            before = store.summary().to_public_dict()
            with self.assertRaisesRegex(IdentityStoreError, "model metadata"):
                store.enroll_profile(
                    declared_kind=DeclaredKind.ADULT,
                    model_fingerprint=model,
                    embeddings=((1.0, 0.0, 0.0),),
                )
            with self.assertRaisesRegex(IdentityStoreError, "batch is empty"):
                store.enroll_profile(
                    declared_kind=DeclaredKind.ADULT,
                    model_fingerprint="6" * 64,
                    embeddings=(),
                )
            self.assertEqual(before, store.summary().to_public_dict())

            database = root / "identity.sqlite3"
            with sqlite3.connect(database) as connection:
                trigger_names = tuple(
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                    )
                )
                for trigger_name in trigger_names:
                    connection.execute(f'DROP TRIGGER "{trigger_name}"')
                connection.execute("DROP TABLE store_state")
                connection.execute("PRAGMA user_version = 1")

            migrated = IdentityTemplateStore(root)
            self.assertEqual(2, migrated.summary().schema_version)
            loaded = migrated.load_templates_for_matching(
                model_fingerprint=model,
                embedding_dimension=2,
            )
            self.assertEqual(set(enrollment.template_ids), {item.template_id for item in loaded})
            with sqlite3.connect(database) as connection:
                self.assertEqual(2, int(connection.execute("PRAGMA user_version").fetchone()[0]))
                state = connection.execute("SELECT singleton, revision FROM store_state").fetchall()
                triggers = connection.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type = 'trigger'"
                ).fetchone()
            self.assertEqual([(1, 0)], state)
            self.assertEqual(6, int(triggers[0]))

    def test_invalid_kinds_fingerprints_and_embeddings_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _root, store = self._new_store(directory)
            with self.assertRaisesRegex(IdentityStoreError, "declared kind"):
                store.create_profile(DeclaredKind.UNKNOWN)
            with self.assertRaisesRegex(IdentityStoreError, "declared kind"):
                store.create_profile("baby")  # type: ignore[arg-type]
            profile_id = store.create_profile(DeclaredKind.BABY)

            invalid_embeddings = (
                (),
                (0.0, 0.0),
                (float("nan"), 1.0),
                (float("inf"), 1.0),
                (True, 1.0),
                tuple(1.0 for _ in range(4097)),
            )
            for embedding in invalid_embeddings:
                with (
                    self.subTest(embedding_length=len(embedding)),
                    self.assertRaises(IdentityStoreError),
                ):
                    store.add_template(
                        profile_id=profile_id,
                        model_fingerprint="d" * 64,
                        embedding=embedding,
                    )
            with self.assertRaisesRegex(IdentityStoreError, "fingerprint"):
                store.add_template(
                    profile_id=profile_id,
                    model_fingerprint="D" * 64,
                    embedding=(1.0,),
                )
            with self.assertRaisesRegex(IdentityStoreError, "profile ID"):
                store.add_template(
                    profile_id="baby-name",
                    model_fingerprint="d" * 64,
                    embedding=(1.0,),
                )
            self.assertEqual(0, store.summary().template_count)

    def test_rejects_windows_mount_symlinks_and_insecure_permissions(self) -> None:
        with self.assertRaisesRegex(IdentityStoreError, "Windows-mounted"):
            IdentityTemplateStore("/mnt/c/private-identity-test")

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "real"
            target.mkdir(mode=0o700)
            alias = parent / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(IdentityStoreError, "symlinks"):
                IdentityTemplateStore(alias)

        with tempfile.TemporaryDirectory() as directory:
            root, store = self._new_store(directory)
            root.chmod(0o750)
            try:
                with self.assertRaisesRegex(IdentityStoreError, "permissions"):
                    store.summary()
                with self.assertRaisesRegex(IdentityStoreError, "permissions"):
                    IdentityTemplateStore(root)
            finally:
                root.chmod(0o700)

            database = root / "identity.sqlite3"
            database.chmod(0o640)
            try:
                with self.assertRaisesRegex(IdentityStoreError, "permissions"):
                    store.summary()
            finally:
                database.chmod(0o600)

    def test_rejects_symlinked_or_incomplete_private_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _store = self._new_store(directory)
            key = root / "identity.key"
            backup = root / "key-backup"
            key.rename(backup)
            key.symlink_to(backup)
            with self.assertRaisesRegex(IdentityStoreError, "unsafe"):
                IdentityTemplateStore(root)

        with tempfile.TemporaryDirectory() as directory:
            root, _store = self._new_store(directory)
            (root / "identity.key").unlink()
            with self.assertRaisesRegex(IdentityStoreError, "incomplete"):
                IdentityTemplateStore(root)

    def test_wrong_key_ciphertext_or_aad_tampering_share_one_safe_error(self) -> None:
        def populate(directory: str) -> tuple[Path, str]:
            root, store = self._new_store(directory)
            profile_id = store.create_profile(DeclaredKind.BABY)
            store.add_template(
                profile_id=profile_id,
                model_fingerprint="e" * 64,
                embedding=(1.0, 0.0),
            )
            store.add_template(
                profile_id=profile_id,
                model_fingerprint="e" * 64,
                embedding=(0.0, 1.0),
            )
            return root, profile_id

        expected = "identity template decryption failed"
        with tempfile.TemporaryDirectory() as directory:
            root, _profile_id = populate(directory)
            (root / "identity.key").write_bytes(b"z" * 32)
            reopened = IdentityTemplateStore(root)
            with self.assertRaisesRegex(IdentityStoreError, f"^{expected}$"):
                reopened.load_templates_for_matching(
                    model_fingerprint="e" * 64,
                    embedding_dimension=2,
                )

        with tempfile.TemporaryDirectory() as directory:
            root, _profile_id = populate(directory)
            database = root / "identity.sqlite3"
            with sqlite3.connect(database) as connection:
                row = connection.execute(
                    "SELECT template_id, ciphertext FROM templates ORDER BY template_id LIMIT 1"
                ).fetchone()
                ciphertext = bytearray(row[1])
                ciphertext[0] ^= 0x01
                connection.execute(
                    "UPDATE templates SET ciphertext = ? WHERE template_id = ?",
                    (bytes(ciphertext), row[0]),
                )
            reopened = IdentityTemplateStore(root)
            with self.assertRaisesRegex(IdentityStoreError, f"^{expected}$"):
                reopened.load_templates_for_matching(
                    model_fingerprint="e" * 64,
                    embedding_dimension=2,
                )

        with tempfile.TemporaryDirectory() as directory:
            root, _profile_id = populate(directory)
            database = root / "identity.sqlite3"
            with sqlite3.connect(database) as connection:
                rows = connection.execute(
                    "SELECT template_id, nonce, ciphertext FROM templates ORDER BY template_id"
                ).fetchall()
                connection.execute(
                    "UPDATE templates SET ciphertext = ? WHERE template_id = ?",
                    (rows[1][2], rows[0][0]),
                )
            reopened = IdentityTemplateStore(root)
            with self.assertRaisesRegex(IdentityStoreError, f"^{expected}$"):
                reopened.load_templates_for_matching(
                    model_fingerprint="e" * 64,
                    embedding_dimension=2,
                )

    def test_delete_cascades_securely_checkpoints_and_survives_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, store = self._new_store(directory)
            first_profile = store.create_profile(DeclaredKind.BABY)
            second_profile = store.create_profile(DeclaredKind.ADULT)
            first_template = store.add_template(
                profile_id=first_profile,
                model_fingerprint="f" * 64,
                embedding=(1.0, 0.0),
            )
            store.add_template(
                profile_id=first_profile,
                model_fingerprint="f" * 64,
                embedding=(0.8, 0.2),
            )
            second_template = store.add_template(
                profile_id=second_profile,
                model_fingerprint="f" * 64,
                embedding=(0.0, 1.0),
            )

            deleted = store.delete_profile(first_profile)
            self.assertEqual(
                {
                    "deleted_profiles": 1,
                    "deleted_templates": 2,
                    "remaining_profiles": 1,
                    "remaining_templates": 1,
                    "committed": True,
                    "checkpoint_completed": True,
                    "cleanup_pending": False,
                },
                deleted.to_public_dict(),
            )
            self.assertFalse(
                any(item.template_id == first_template for item in store.list_templates())
            )
            wal = root / "identity.sqlite3-wal"
            self.assertTrue(not wal.exists() or wal.stat().st_size == 0)

            reopened = IdentityTemplateStore(root)
            loaded = reopened.load_templates_for_matching(
                model_fingerprint="f" * 64,
                embedding_dimension=2,
            )
            self.assertEqual((second_template,), tuple(item.template_id for item in loaded))
            deleted_template = reopened.delete_template(second_template)
            self.assertEqual(0, deleted_template.deleted_profiles)
            self.assertEqual(1, deleted_template.deleted_templates)
            self.assertEqual(1, deleted_template.remaining_profiles)
            self.assertEqual(0, deleted_template.remaining_templates)
            with self.assertRaisesRegex(IdentityStoreError, "does not exist"):
                reopened.delete_profile(first_profile)

    def test_read_only_listing_sees_committed_wal_without_crypto_or_layout_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, store = self._new_store(directory)
            first_profile = store.create_profile(DeclaredKind.BABY)
            database = root / "identity.sqlite3"
            writer = sqlite3.connect(database)
            self.addCleanup(writer.close)
            writer.execute("PRAGMA journal_mode = WAL")
            second_profile = "profile_" + "2" * 32
            writer.execute(
                "INSERT INTO profiles (profile_id, declared_kind) VALUES (?, ?)",
                (second_profile, DeclaredKind.ADULT.value),
            )
            writer.commit()
            self.assertTrue((root / "identity.sqlite3-wal").exists())
            self.assertTrue((root / "identity.sqlite3-shm").exists())

            with mock.patch(
                "baby_monitor.identity_store._load_aesgcm",
                side_effect=AssertionError("read-only listing loaded crypto"),
            ):
                profiles = IdentityTemplateStore.list_profiles_read_only(root)

            self.assertEqual(
                {
                    first_profile: DeclaredKind.BABY,
                    second_profile: DeclaredKind.ADULT,
                },
                {profile.profile_id: profile.declared_kind for profile in profiles},
            )
            writer.close()

    def test_read_only_listing_never_initializes_an_empty_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private-identity"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            with (
                mock.patch(
                    "baby_monitor.identity_store._load_aesgcm",
                    side_effect=AssertionError("read-only listing loaded crypto"),
                ),
                self.assertRaisesRegex(IdentityStoreError, "files do not exist"),
            ):
                IdentityTemplateStore.list_profiles_read_only(root)
            self.assertEqual((), tuple(root.iterdir()))

    def test_open_existing_rejects_uninitialized_schema_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private-identity"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            key = root / "identity.key"
            database = root / "identity.sqlite3"
            key.write_bytes(b"k" * 32)
            database.touch()
            key.chmod(0o600)
            database.chmod(0o600)
            before = {
                path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in root.iterdir()
            }

            with self.assertRaisesRegex(IdentityStoreError, "read-only|schema"):
                IdentityTemplateStore.open_existing(root)

            after = {
                path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in root.iterdir()
            }
            self.assertEqual(before, after)
            self.assertFalse((root / "identity.sqlite3-wal").exists())
            self.assertFalse((root / "identity.sqlite3-shm").exists())

    def test_committed_deletion_reports_pending_checkpoint_and_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _root, store = self._new_store(directory)
            profile_id = store.create_profile(DeclaredKind.BABY)
            store.add_template(
                profile_id=profile_id,
                model_fingerprint="a" * 64,
                embedding=(1.0, 0.0),
            )

            with mock.patch.object(
                store,
                "_checkpoint",
                side_effect=sqlite3.OperationalError("simulated checkpoint failure"),
            ):
                deleted = store.delete_profile(profile_id)

            self.assertTrue(deleted.committed)
            self.assertFalse(deleted.checkpoint_completed)
            self.assertTrue(deleted.cleanup_pending)
            self.assertEqual(0, store.summary().profile_count)
            cleanup = store.retry_cleanup()
            self.assertTrue(cleanup.checkpoint_completed)
            self.assertFalse(cleanup.cleanup_pending)


if __name__ == "__main__":
    unittest.main()
