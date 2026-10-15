from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from baby_monitor.identity_domain import DeclaredKind, FaceQualityState
from baby_monitor.identity_matching import FaceEmbedding, IdentityTemplate
from baby_monitor.identity_protocol import (
    FACE_LANDMARK_NAMES,
    FaceEmbeddingObservation,
    FaceImageInput,
    FaceLandmark,
    FaceModelReference,
)


class IdentitySensitiveReprTests(unittest.TestCase):
    def test_embeddings_are_hidden_from_default_repr(self) -> None:
        fingerprint = "7" * 64
        sensitive = 0.123456789
        query = FaceEmbedding("face_01", fingerprint, (sensitive, 1.0))
        template = IdentityTemplate(
            template_id="template_01",
            profile_id="profile_01",
            declared_kind=DeclaredKind.BABY,
            model_fingerprint=fingerprint,
            embedding=(sensitive, 1.0),
        )
        observation = FaceEmbeddingObservation(
            face_id="face_01",
            frame_ref="frame_01",
            bbox_xyxy=(0.1, 0.1, 0.9, 0.9),
            detector_confidence=0.99,
            landmarks=tuple(
                FaceLandmark(name=name, x=0.5, y=0.5, in_frame=True) for name in FACE_LANDMARK_NAMES
            ),
            quality_state=FaceQualityState.ACCEPTED,
            rejection_codes=(),
            embedding=(1.0,) + (0.0,) * 127,
        )

        self.assertNotIn(str(sensitive), repr(query))
        self.assertNotIn(str(sensitive), repr(template))
        self.assertNotIn("embedding=", repr(query))
        self.assertNotIn("embedding=", repr(template))
        self.assertNotIn("embedding=", repr(observation))

    def test_private_paths_are_hidden_from_default_repr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "private-family-source.jpg"
            model = root / "private-face-model.onnx"
            manifest = root / "private-face-model.json"
            source.write_bytes(b"source")
            model.write_bytes(b"model")
            manifest.write_text("{}", encoding="utf-8")

            image = FaceImageInput(
                frame_ref="frame_01",
                source_path=str(source),
                source_sha256="1" * 64,
            )
            reference = FaceModelReference(
                role="detector",
                model_path=str(model),
                manifest_path=str(manifest),
                model_id="fixture.detector",
                upstream_revision="2" * 40,
                model_sha256="3" * 64,
            )

            self.assertNotIn(str(source), repr(image))
            self.assertNotIn(str(model), repr(reference))
            self.assertNotIn(str(manifest), repr(reference))


if __name__ == "__main__":
    unittest.main()
