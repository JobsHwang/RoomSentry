from __future__ import annotations

import importlib.util
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baby_monitor.identity_domain import DeclaredKind
from baby_monitor.identity_matching import IdentityMatchPolicy, assess_identities
from baby_monitor.identity_store import IdentityStoreError, IdentityTemplateStore


@unittest.skipUnless(
    importlib.util.find_spec("cryptography") is not None,
    "optional identity crypto runtime is not installed",
)
class IdentityTemplateStoreRealCryptoTests(unittest.TestCase):
    def test_real_aesgcm_round_trip_tamper_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private-identity"
            store = IdentityTemplateStore(root)
            profile_id = store.create_profile(DeclaredKind.BABY)
            model_fingerprint = "9" * 64
            template_id = store.add_template(
                profile_id=profile_id,
                model_fingerprint=model_fingerprint,
                embedding=(3.0, 4.0),
            )

            plaintext = struct.pack("<2f", 0.6, 0.8)
            with sqlite3.connect(root / "identity.sqlite3") as connection:
                nonce, ciphertext = connection.execute(
                    "SELECT nonce, ciphertext FROM templates WHERE template_id = ?",
                    (template_id,),
                ).fetchone()
            self.assertEqual(12, len(bytes(nonce)))
            self.assertNotIn(plaintext, bytes(ciphertext))

            reopened = IdentityTemplateStore(root)
            loaded = reopened.load_templates_for_matching(
                model_fingerprint=model_fingerprint,
                embedding_dimension=2,
            )
            self.assertEqual((template_id,), tuple(item.template_id for item in loaded))
            self.assertAlmostEqual(0.6, loaded[0].embedding[0], places=6)
            self.assertAlmostEqual(0.8, loaded[0].embedding[1], places=6)

            with sqlite3.connect(root / "identity.sqlite3") as connection:
                damaged = bytearray(ciphertext)
                damaged[-1] ^= 1
                connection.execute(
                    "UPDATE templates SET ciphertext = ? WHERE template_id = ?",
                    (bytes(damaged), template_id),
                )
            tampered = IdentityTemplateStore(root)
            with self.assertRaisesRegex(
                IdentityStoreError,
                "^identity template decryption failed$",
            ):
                tampered.load_templates_for_matching(
                    model_fingerprint=model_fingerprint,
                    embedding_dimension=2,
                )

            deleted = tampered.delete_profile(profile_id)
            self.assertEqual(1, deleted.deleted_profiles)
            self.assertEqual(1, deleted.deleted_templates)

    def test_fingerprint_dimension_conflict_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = IdentityTemplateStore(Path(directory) / "private-identity")
            profile_id = store.create_profile(DeclaredKind.PERSON)
            model_fingerprint = "8" * 64
            store.add_template(
                profile_id=profile_id,
                model_fingerprint=model_fingerprint,
                embedding=(1.0, 0.0),
            )
            with self.assertRaisesRegex(IdentityStoreError, "model metadata"):
                store.add_template(
                    profile_id=profile_id,
                    model_fingerprint=model_fingerprint,
                    embedding=(1.0, 0.0, 0.0),
                )
            with self.assertRaisesRegex(IdentityStoreError, "model metadata"):
                store.load_templates_for_matching(
                    model_fingerprint=model_fingerprint,
                    embedding_dimension=3,
                )

    def test_real_aesgcm_atomic_enrollment_rolls_back_after_late_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private-identity"
            store = IdentityTemplateStore(root)
            with (
                mock.patch(
                    "baby_monitor.identity_store.secrets.token_hex",
                    side_effect=("1" * 32, "2" * 32, "2" * 32),
                ),
                self.assertRaisesRegex(IdentityStoreError, "enrollment failed"),
            ):
                store.enroll_profile(
                    declared_kind=DeclaredKind.BABY,
                    model_fingerprint="7" * 64,
                    embeddings=((1.0, 0.0), (0.0, 1.0)),
                )

            self.assertEqual(0, store.summary().profile_count)
            self.assertEqual(0, store.summary().template_count)
            with sqlite3.connect(root / "identity.sqlite3") as connection:
                profile_count = int(
                    connection.execute("SELECT count(*) FROM profiles").fetchone()[0]
                )
                template_count = int(
                    connection.execute("SELECT count(*) FROM templates").fetchone()[0]
                )
                revision = int(
                    connection.execute(
                        "SELECT revision FROM store_state WHERE singleton = 1"
                    ).fetchone()[0]
                )
            self.assertEqual((0, 0, 0), (profile_count, template_count, revision))

    def test_real_aesgcm_template_snapshots_fail_closed_after_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = IdentityTemplateStore(Path(directory) / "private-identity")
            model_fingerprint = "6" * 64
            enrollment = store.enroll_profile(
                declared_kind=DeclaredKind.BABY,
                model_fingerprint=model_fingerprint,
                embeddings=((1.0, 0.0), (0.8, 0.2)),
            )
            snapshot = store.load_templates_for_matching(
                model_fingerprint=model_fingerprint,
                embedding_dimension=2,
            )
            snapshot.assert_current()
            self.assertEqual(2, len(snapshot))
            self.assertNotIn("embedding", repr(snapshot))

            store.add_template(
                profile_id=enrollment.profile_id,
                model_fingerprint=model_fingerprint,
                embedding=(0.6, 0.4),
            )
            with self.assertRaisesRegex(IdentityStoreError, "snapshot is stale"):
                snapshot.assert_current()
            with self.assertRaisesRegex(IdentityStoreError, "snapshot is stale"):
                len(snapshot)

            refreshed = store.load_templates_for_matching(
                model_fingerprint=model_fingerprint,
                embedding_dimension=2,
            )
            self.assertEqual(3, len(refreshed))
            store.delete_profile(enrollment.profile_id)
            with self.assertRaisesRegex(IdentityStoreError, "snapshot is stale"):
                next(iter(refreshed))
            policy = IdentityMatchPolicy(
                model_fingerprint=model_fingerprint,
                embedding_dimension=2,
                match_threshold=0.5,
                runner_up_margin_threshold=0.1,
                matcher_revision="matcher_v1",
                threshold_revision="threshold_v1",
            )
            with self.assertRaisesRegex(IdentityStoreError, "snapshot is stale"):
                assess_identities(
                    assessment_id="assessment_1",
                    event_id="event_1",
                    camera_id="camera_1",
                    observations=(),
                    embeddings=(),
                    templates=refreshed,
                    policy=policy,
                )


if __name__ == "__main__":
    unittest.main()
