from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from baby_monitor.identity_model import FaceModelError, FaceModelManifest


def _payload(content: bytes, *, filename: str = "fixture.onnx") -> dict[str, object]:
    return {
        "version": 1,
        "backend": "opencv_dnn",
        "model_format": "ONNX",
        "role": "embedding",
        "model_id": "fixture.face.embedding",
        "upstream_revision": "1" * 40,
        "filename": filename,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "source_url": "https://example.invalid/models/fixture.onnx",
        "license_spdx": "LicenseRef-Fixture-Only",
        "license_url": "https://example.invalid/licenses/fixture",
        "input_width": 112,
        "input_height": 112,
        "input_color_space": "BGR",
        "landmark_count": 5,
        "embedding_dimension": 128,
        "preprocessing_revision": "fixture_bgr_align_v1",
        "unsafe_pickle": False,
        "automatic_download": False,
        "identity_decision_authority": False,
    }


class FaceModelTests(unittest.TestCase):
    def test_reviewed_opencv_manifests_match_schema_and_pins(self) -> None:
        schema = json.loads(
            Path("schemas/face-model-manifest-v1.schema.json").read_text(encoding="utf-8")
        )
        paths = (
            Path("config/opencv-yunet-2023mar.reviewed.example.json"),
            Path("config/opencv-sface-2021dec.reviewed.example.json"),
        )
        manifests = [FaceModelManifest.read(path) for path in paths]
        for path, manifest in zip(paths, manifests, strict=True):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(schema["required"]), set(payload))
            self.assertEqual(payload, manifest.to_dict())
            self.assertEqual(manifest.canonical_sha256(), manifest.require_reviewed())
        self.assertEqual("detector", manifests[0].role)

        self.assertEqual("MIT", manifests[0].license_spdx)
        self.assertIsNone(manifests[0].embedding_dimension)
        self.assertEqual("embedding", manifests[1].role)
        self.assertEqual("Apache-2.0", manifests[1].license_spdx)
        self.assertEqual(128, manifests[1].embedding_dimension)

    def test_arbitrary_self_consistent_manifest_is_not_reviewed(self) -> None:
        manifest = FaceModelManifest.from_dict(_payload(b"self-consistent-fixture"))
        with self.assertRaisesRegex(FaceModelError, "built-in reviewed allowlist"):
            manifest.require_reviewed()

    def test_manifest_verifies_model_and_public_output_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"synthetic-onnx-fixture"
            model = root / "fixture.onnx"
            model.write_bytes(content)
            manifest = FaceModelManifest.from_dict(_payload(content))
            public = manifest.verify_model(model).to_public_dict()

            self.assertTrue(public["valid"])
            self.assertFalse(public["paths_disclosed"])
            self.assertFalse(public["model_loaded"])
            self.assertFalse(public["network_used"])
            self.assertFalse(public["automatic_download"])
            self.assertFalse(public["identity_decision_authority"])
            self.assertNotIn(str(root), json.dumps(public))

    def test_manifest_rejects_unknown_fields_paths_and_unpinned_revision(self) -> None:
        payload = _payload(b"fixture")
        payload["extra"] = True
        with self.assertRaisesRegex(FaceModelError, "fields do not match"):
            FaceModelManifest.from_dict(payload)

        payload = _payload(b"fixture", filename="../fixture.onnx")
        with self.assertRaisesRegex(FaceModelError, "filename"):
            FaceModelManifest.from_dict(payload)

        payload = _payload(b"fixture")
        payload["upstream_revision"] = "main"
        with self.assertRaisesRegex(FaceModelError, "not pinned"):
            FaceModelManifest.from_dict(payload)

    def test_manifest_rejects_credentials_or_implicit_authority(self) -> None:
        payload = _payload(b"fixture")
        # release-audit: allow=fixture-credential
        payload["source_url"] = "https://user:secret@example.invalid/model.onnx"
        with self.assertRaisesRegex(FaceModelError, "credential-free"):
            FaceModelManifest.from_dict(payload)

        for field in (
            "unsafe_pickle",
            "automatic_download",
            "identity_decision_authority",
        ):
            payload = _payload(b"fixture")
            payload[field] = True
            with self.assertRaisesRegex(FaceModelError, f"{field} must be false"):
                FaceModelManifest.from_dict(payload)

    def test_role_specific_embedding_dimension_fails_closed(self) -> None:
        payload = _payload(b"fixture")
        payload["role"] = "detector"
        with self.assertRaisesRegex(FaceModelError, "cannot declare"):
            FaceModelManifest.from_dict(payload)

        payload = _payload(b"fixture")
        payload["embedding_dimension"] = None
        with self.assertRaisesRegex(FaceModelError, "dimension is invalid"):
            FaceModelManifest.from_dict(payload)

    def test_manifest_rejects_size_and_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"synthetic-onnx-fixture"
            model = root / "fixture.onnx"
            model.write_bytes(content)

            payload = _payload(content)
            payload["size_bytes"] = len(content) + 1
            with self.assertRaisesRegex(FaceModelError, "size does not match"):
                FaceModelManifest.from_dict(payload).verify_model(model)

            payload = _payload(content)
            payload["sha256"] = "0" * 64
            with self.assertRaisesRegex(FaceModelError, "sha256 does not match"):
                FaceModelManifest.from_dict(payload).verify_model(model)

    @unittest.skipUnless(os.name == "posix", "symlink policy is verified on POSIX/WSL")
    def test_manifest_rejects_symlink_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"synthetic-onnx-fixture"
            target = root / "target.onnx"
            target.write_bytes(content)
            link = root / "fixture.onnx"
            link.symlink_to(target)
            with self.assertRaisesRegex(FaceModelError, "missing or unsafe"):
                FaceModelManifest.from_dict(_payload(content)).verify_model(link)


if __name__ == "__main__":
    unittest.main()
