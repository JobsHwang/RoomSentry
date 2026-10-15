from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data/verification/public/pexels_7918208_tracker_review_v4"
FRAME_INDEX = SOURCE_ROOT / "frame-media.local.json"
DETECTIONS = ROOT / "data/verification/public/pexels_7918208_detection_frames.json"
OUTPUT = ROOT / "data/verification/mage-vl-predicate-public-ab-20260812"

_DYNAMIC_INDEXES = (0, 6, 12, 18)
_FINAL_INDEX = 18
_FIRST_INDEX = 0
_SOURCE_ADMISSION_SHA256 = "d44881f969e2b168c0226e21abc5f751ed7b594c650cc147b10e4c63b57fc7b3"
_DETECTOR_BATCH_SHA256 = "ab6cb3bfa41d737d040e3a822e33639cf9081cc5d66f662771486b25ababfee3"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_private(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _frame_records() -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    frame_index = json.loads(FRAME_INDEX.read_text(encoding="utf-8"))
    detections = json.loads(DETECTIONS.read_text(encoding="utf-8"))
    media_by_id = {item["frame_id"]: item for item in frame_index["frames"]}
    records: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for frame in detections["frames"]:
        frame_id = frame["frame_id"]
        media = media_by_id[frame_id]
        path = SOURCE_ROOT / media["path"]
        payload = path.read_bytes()
        if _sha256(payload) != media["sha256"]:
            raise RuntimeError("public frame hash mismatch")
        records.append((frame, media, path))
    if len(records) != 19:
        raise RuntimeError("public detector sequence changed")
    return records


def _expanded_crop(
    bbox: list[float],
    width: int,
    height: int,
    *,
    padding: float = 0.15,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    left = max(0.0, x1 - box_width * padding)
    top = max(0.0, y1 - box_height * padding)
    right = min(1.0, x2 + box_width * padding)
    bottom = min(1.0, y2 + box_height * padding)
    pixels = (
        int(left * width),
        int(top * height),
        max(int(right * width), int(left * width) + 1),
        max(int(bottom * height), int(top * height) + 1),
    )
    if not (0 <= pixels[0] < pixels[2] <= width and 0 <= pixels[1] < pixels[3] <= height):
        raise RuntimeError("public detector crop is invalid")
    return pixels


def _render_frame(
    source: Path,
    destination: Path,
    *,
    bbox: list[float] | None,
) -> dict[str, object]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required in the offline preparation runtime") from exc
    with Image.open(source) as image:
        image.load()
        converted = image.convert("RGB")
        if bbox is not None:
            cropped = converted.crop(_expanded_crop(bbox, *converted.size))
            converted.close()
            converted = cropped
        try:
            converted.thumbnail((512, 512), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            converted.save(buffer, format="JPEG", quality=92, optimize=True, progressive=False)
            payload = buffer.getvalue()
        finally:
            converted.close()
    _write_private(destination, payload)
    return {
        "path": destination.name,
        "sha256": _sha256(payload),
        "media_type": "image/jpeg",
        "bytes": len(payload),
    }


def _top_person(frame: dict[str, Any]) -> dict[str, Any]:
    candidates = [item for item in frame["detections"] if item["label"] == "person"]
    if not candidates:
        raise RuntimeError("selected public frame has no YOLO person candidate")
    return max(candidates, key=lambda item: (item["confidence"], item["detection_id"]))


def _variant(
    records: list[tuple[dict[str, Any], dict[str, Any], Path]],
    indexes: tuple[int, ...],
    *,
    case_id: str,
    include_crops: bool,
    duplicate_final: bool = False,
) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for time_index, record_index in enumerate(indexes):
        detection_frame, _media, source = records[record_index]
        if duplicate_final:
            source = records[_FINAL_INDEX][2]
            detection_frame = records[_FINAL_INDEX][0]
        full_name = f"{case_id}_t{time_index:02d}_full.jpg"
        full = _render_frame(source, OUTPUT / full_name, bbox=None)
        full["frame_id"] = f"t{time_index:02d}_full"
        frames.append(full)
        if include_crops:
            candidate = _top_person(detection_frame)
            crop_name = f"{case_id}_t{time_index:02d}_crop00.jpg"
            crop = _render_frame(source, OUTPUT / crop_name, bbox=candidate["bbox_xyxy"])
            crop.update(
                {
                    "frame_id": f"t{time_index:02d}_crop00",
                    "detector_model_id": "Ultralytics.YOLO11n.COCO",
                    "detector_model_revision": "5599de6f3b0ef95f98f37530ed0f3bf149c10b37",
                    "detector_confidence": candidate["confidence"],
                }
            )
            frames.append(crop)
    return frames


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("output already exists; it is never overwritten")
    OUTPUT.mkdir(parents=True, mode=0o700)
    OUTPUT.chmod(0o700)
    try:
        records = _frame_records()
        cases = []
        definitions = (
            (
                "dynamic_posture_yes",
                VisualDefinition("posture_transition_visible", "yes"),
                _DYNAMIC_INDEXES,
                False,
            ),
            (
                "static_posture_no",
                VisualDefinition("posture_transition_visible", "no"),
                (_FINAL_INDEX,) * 4,
                True,
            ),
            (
                "face_clear",
                VisualDefinition("face_visibility", "clear"),
                (_FINAL_INDEX,),
                False,
            ),
            (
                "yolo_false_positive_face_not_visible",
                VisualDefinition("face_visibility", "not_visible"),
                (_FIRST_INDEX,),
                False,
            ),
        )
        for case_id, definition, indexes, duplicate_final in definitions:
            cases.append(
                {
                    "case_id": case_id,
                    "predicate": definition.predicate,
                    "expected_answer": definition.expected_answer,
                    "human_label_scope": "directly_visible_public_evidence",
                    "identity_ground_truth": False,
                    "variants": {
                        "full": _variant(
                            records,
                            indexes,
                            case_id=f"{case_id}_full",
                            include_crops=False,
                            duplicate_final=duplicate_final,
                        ),
                        "full_yolo_crop": _variant(
                            records,
                            indexes,
                            case_id=f"{case_id}_crop",
                            include_crops=True,
                            duplicate_final=duplicate_final,
                        ),
                    },
                }
            )
        dataset = {
            "version": 1,
            "source": {
                "kind": "public_stock_video_frames",
                "asset_id": "pexels_7918208",
                "source_admission_sha256": _SOURCE_ADMISSION_SHA256,
                "detector_batch_sha256": _DETECTOR_BATCH_SHA256,
                "household_media": False,
                "uploads_performed": False,
            },
            "cases": cases,
        }
        payload = (
            json.dumps(
                dataset,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        _write_private(OUTPUT / "dataset.local.json", payload)
        print(
            json.dumps(
                {
                    "version": 1,
                    "status": "created",
                    "cases": len(cases),
                    "variants": 2,
                    "household_media": False,
                    "uploads_performed": False,
                    "paths_disclosed": False,
                },
                separators=(",", ":"),
            )
        )
        return 0
    except Exception:
        shutil.rmtree(OUTPUT)
        raise


class VisualDefinition:
    def __init__(self, predicate: str, expected_answer: str):
        self.predicate = predicate
        self.expected_answer = expected_answer


if __name__ == "__main__":
    raise SystemExit(main())
