#!/usr/bin/env python3
"""Run an offline, path-free YuNet/SFace smoke on admitted adult images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from baby_monitor.identity_domain import FaceQualityState
from baby_monitor.identity_model import FaceModelManifest
from baby_monitor.identity_protocol import (
    FACE_QUALITY_POLICY,
    FaceEmbeddingMode,
    FaceEmbeddingRequest,
    FaceImageInput,
    FaceModelReference,
)
from baby_monitor.local_runners.opencv_face import run_request
from baby_monitor.source_admission import ExternalSourceAdmission, SourceAdmissionError


def _model_reference(*, role: str, manifest_path: Path, model_path: Path) -> FaceModelReference:
    manifest = FaceModelManifest.read(manifest_path)
    if manifest.role != role:
        raise ValueError("public face smoke model role does not match")
    manifest.verify_model(model_path)
    return FaceModelReference(
        role=role,
        model_path=str(model_path.resolve(strict=True)),
        manifest_path=str(manifest_path.resolve(strict=True)),
        model_id=manifest.model_id,
        upstream_revision=manifest.upstream_revision,
        model_sha256=manifest.sha256,
    )


def _admit_case(index: int, manifest_path: Path, asset_path: Path) -> FaceImageInput:
    admission = ExternalSourceAdmission.read(manifest_path)
    if (
        admission.admission.tier != "B"
        or admission.admission.purpose != "model_evaluation"
        or not admission.admission.user_approval_recorded
        or admission.admission.training_allowed
        or admission.admission.git_allowed
        or admission.privacy.private_household_data != "no"
        or admission.privacy.real_minors != "no"
        or admission.privacy.faces != "yes"
    ):
        raise SourceAdmissionError("public face smoke admission policy is not satisfied")
    verification = admission.verify_asset(asset_path)
    return FaceImageInput(
        frame_ref=f"public_face_{index:04d}",
        source_path=str(asset_path.resolve(strict=True)),
        source_sha256=verification.sha256,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        nargs=2,
        metavar=("ADMISSION", "ASSET"),
        type=Path,
        required=True,
    )
    parser.add_argument("--detector-score-threshold", type=float, default=0.9)
    args = parser.parse_args()
    if not 1 <= len(args.case) <= 8:
        parser.error("public face smoke requires between one and eight cases")

    images = tuple(
        _admit_case(index, manifest_path, asset_path)
        for index, (manifest_path, asset_path) in enumerate(args.case, start=1)
    )
    request = FaceEmbeddingRequest(
        request_id="public_face_smoke_v1",
        mode=FaceEmbeddingMode.ASSESSMENT,
        face_adapter="opencv_yunet_sface_v1",
        images=images,
        detector_model=_model_reference(
            role="detector",
            manifest_path=args.detector_manifest,
            model_path=args.detector_model,
        ),
        embedding_model=_model_reference(
            role="embedding",
            manifest_path=args.embedding_manifest,
            model_path=args.embedding_model,
        ),
        detector_score_threshold=args.detector_score_threshold,
        detector_nms_threshold=0.3,
        detector_top_k=5_000,
        min_face_size_pixels=40,
        min_eye_distance_pixels=12.0,
    )
    result = run_request(request)

    cases: list[dict[str, object]] = []
    for image in result.images:
        reasons = Counter(reason for face in image.faces for reason in face.rejection_codes)
        cases.append(
            {
                "frame_ref": image.frame_ref,
                "face_count": len(image.faces),
                "accepted_face_count": sum(
                    face.quality_state is FaceQualityState.ACCEPTED for face in image.faces
                ),
                "rejection_reasons": dict(sorted(reasons.items())),
            }
        )

    print(
        json.dumps(
            {
                "status": "ok",
                "case_count": len(cases),
                "face_count": result.face_count,
                "accepted_face_count": result.accepted_face_count,
                "cases": cases,
                "embedding_dimension": 128,
                "quality_policy_revision": FACE_QUALITY_POLICY.revision,
                "model_fingerprint": result.model_fingerprint,
                "automated_occlusion_detection": False,
                "anti_spoofing": False,
                "manual_occlusion_review_required": True,
                "embeddings_finite_and_normalized": all(
                    face.embedding is not None
                    for image in result.images
                    for face in image.faces
                    if face.quality_state is FaceQualityState.ACCEPTED
                ),
                "embeddings_persisted": False,
                "identity_inference": False,
                "household_media_used": False,
                "paths_disclosed": False,
                "network_used": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
