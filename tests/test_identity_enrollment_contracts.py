from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from baby_monitor import identity_enrollment as enrollment
from baby_monitor import identity_enrollment_contracts as contracts
from baby_monitor.adapters import identity as adapter


class EnrollmentContractTests(unittest.TestCase):
    def test_legacy_and_adapter_imports_share_the_same_contract_objects(self) -> None:
        self.assertIs(enrollment.EnrollmentImage, contracts.EnrollmentImage)
        self.assertIs(adapter.EnrollmentImage, contracts.EnrollmentImage)
        self.assertIs(
            enrollment.ValidatedEnrollmentEmbedding, contracts.ValidatedEnrollmentEmbedding
        )
        self.assertIs(adapter.ValidatedEnrollmentEmbedding, contracts.ValidatedEnrollmentEmbedding)
        self.assertIs(enrollment.IdentityEnrollmentError, contracts.IdentityEnrollmentError)

    def test_adapter_import_does_not_load_enrollment_or_store_coordination(self) -> None:
        script = (
            "import baby_monitor.adapters.identity\n"
            "import sys\n"
            "for name in ('baby_monitor.identity_enrollment', "
            "'baby_monitor.identity_service', 'baby_monitor.identity_store'):\n"
            "    assert name not in sys.modules, name\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_foreign_token_cannot_mint_a_binding_proof_through_either_import(self) -> None:
        for proof_type in (
            enrollment.ValidatedEnrollmentEmbedding,
            contracts.ValidatedEnrollmentEmbedding,
        ):
            with (
                self.subTest(proof_type=proof_type),
                self.assertRaises(enrollment.IdentityEnrollmentError),
            ):
                proof_type(
                    object(),
                    frame_ref="frame_01",
                    source_sha256="a" * 64,
                    model_fingerprint="b" * 64,
                    embedding=(1.0,) + (0.0,) * 127,
                )

    @unittest.skipUnless(os.name == "posix", "POSIX source admission required")
    def test_image_path_admission_stays_strict_and_uses_the_shared_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "synthetic.jpg"
            payload = b"synthetic-enrollment-image"
            source.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            image = contracts.EnrollmentImage("frame_01", str(source), digest)
            self.assertEqual(str(source), image.source_path)
            self.assertNotIn(str(source), repr(image))

            linked = root / "linked.jpg"
            linked.symlink_to(source)
            unsafe_paths = (
                str(linked),
                str(root / "nested" / ".." / source.name),
                source.name,
                str(root),
                str(root / "missing.jpg"),
            )
            for source_path in unsafe_paths:
                with (
                    self.subTest(source_path=source_path),
                    self.assertRaises(enrollment.IdentityEnrollmentError),
                ):
                    contracts.EnrollmentImage("frame_01", source_path, digest)


if __name__ == "__main__":
    unittest.main()
