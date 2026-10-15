from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .object_tracker import (
    DetectionFrame,
    DetectionFrameBatch,
    FrameDetection,
)
from .source_admission import ExternalSourceAdmission
from .tracker_evaluation import (
    TrackerAssociationReview,
    TrackerReviewAssignment,
)

CAVIAR_BOX_SOURCE_ID = "caviar_hand_labelled_person_boxes_v1"
CAVIAR_FRAME_WIDTH = 384
CAVIAR_FRAME_HEIGHT = 288
CAVIAR_FPS = 25
MAX_CAVIAR_XML_BYTES = 16 * 1024 * 1024
MAX_CAVIAR_FRAMES = 10_000
MAX_CAVIAR_OBJECTS = 10_000

_EVALUATION_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)
_FRAME_MICROSECONDS = 1_000_000 // CAVIAR_FPS
_INTEGER = re.compile(rb"(?:0|[1-9][0-9]*)")
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_UNSAFE_DATASET_ID_CHARACTER = re.compile(r"[^a-z0-9]+")
_OBJECT_CHILD_TAGS = {
    "appearance",
    "body",
    "box",
    "hypothesislist",
    "orientation",
}


class CaviarTrackingError(ValueError):
    """A CAVIAR annotation import is unsafe or outside the supported profile."""


@dataclass(frozen=True, slots=True)
class CaviarTrackingImport:
    batch: DetectionFrameBatch
    review: TrackerAssociationReview
    dataset_name: str
    source_frame_count: int
    imported_frame_count: int
    imported_detection_count: int
    ignored_group_count: int
    first_frame: int
    last_frame: int
    timestamp_semantics: str = "synthetic_frame_index_at_25fps"
    source_media_included: bool = False

    def to_public_dict(self) -> dict[str, object]:
        return {
            "valid": True,
            "dataset_name": self.dataset_name,
            "source_frame_count": self.source_frame_count,
            "imported_frame_count": self.imported_frame_count,
            "imported_detection_count": self.imported_detection_count,
            "ignored_group_count": self.ignored_group_count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "timestamp_semantics": self.timestamp_semantics,
            "detection_batch_sha256": self.batch.fingerprint,
            "review_sha256": self.review.fingerprint,
            "source_admission_sha256": self.review.source_admission_sha256,
            "ground_truth_provenance": self.review.ground_truth_provenance,
            "source_media_included": False,
            "paths_disclosed": False,
        }


@dataclass(frozen=True, slots=True)
class _ObjectBox:
    upstream_id: int
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _SourceFrame:
    number: int
    objects: tuple[_ObjectBox, ...]
    group_count: int


def _camera_id(dataset_name: str) -> str:
    normalized = _CAMEL_CASE_BOUNDARY.sub("_", dataset_name).lower()
    component = _UNSAFE_DATASET_ID_CHARACTER.sub(
        "_",
        normalized,
    ).strip("_")
    if not component:
        raise CaviarTrackingError("CAVIAR dataset name cannot produce a safe camera identifier")
    candidate = f"caviar_{component}"
    if len(candidate) <= 80:
        return candidate
    suffix = hashlib.sha256(dataset_name.encode("utf-8")).hexdigest()[:12]
    return f"caviar_{component[:60]}_{suffix}"


def _read_xml(path: str | Path) -> bytes:
    source = Path(path).expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        size = source.stat().st_size
        if not 1 <= size <= MAX_CAVIAR_XML_BYTES:
            raise CaviarTrackingError("CAVIAR XML is empty or exceeds the 16 MiB limit")
        raw = source.read_bytes()
    except CaviarTrackingError:
        raise
    except OSError as exc:
        raise CaviarTrackingError("CAVIAR XML is missing or unsafe") from exc
    upper = raw.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise CaviarTrackingError("CAVIAR XML declarations are unsafe")
    return raw


def _integer(value: str | None, field: str) -> int:
    if value is None:
        raise CaviarTrackingError(f"CAVIAR {field} is missing")
    encoded = value.encode("ascii", errors="ignore")
    if encoded.decode("ascii") != value or _INTEGER.fullmatch(encoded) is None:
        raise CaviarTrackingError(f"CAVIAR {field} must be a non-negative integer")
    parsed = int(value)
    if parsed > 1_000_000:
        raise CaviarTrackingError(f"CAVIAR {field} exceeds the supported range")
    return parsed


def _parse_box(
    element: ET.Element,
    *,
    frame_number: int,
) -> tuple[float, float, float, float]:
    if set(element.attrib) != {"h", "w", "xc", "yc"} or list(element):
        raise CaviarTrackingError("CAVIAR box fields do not match the supported CVML")
    h = _integer(element.attrib.get("h"), "box h")
    w = _integer(element.attrib.get("w"), "box w")
    xc = _integer(element.attrib.get("xc"), "box xc")
    yc = _integer(element.attrib.get("yc"), "box yc")
    if h == 0 or w == 0:
        raise CaviarTrackingError("CAVIAR boxes must have positive area")
    x1 = xc - (w / 2.0)
    y1 = yc - (h / 2.0)
    x2 = xc + (w / 2.0)
    y2 = yc + (h / 2.0)
    if not (0.0 <= x1 < x2 <= CAVIAR_FRAME_WIDTH and 0.0 <= y1 < y2 <= CAVIAR_FRAME_HEIGHT):
        raise CaviarTrackingError(f"CAVIAR frame {frame_number} contains an out-of-bounds box")
    return (
        x1 / CAVIAR_FRAME_WIDTH,
        y1 / CAVIAR_FRAME_HEIGHT,
        x2 / CAVIAR_FRAME_WIDTH,
        y2 / CAVIAR_FRAME_HEIGHT,
    )


def _parse_source_frames(raw: bytes) -> tuple[str, tuple[_SourceFrame, ...]]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise CaviarTrackingError("CAVIAR XML is invalid") from exc
    if root.tag != "dataset" or set(root.attrib) != {"name"}:
        raise CaviarTrackingError("CAVIAR XML root does not match the supported CVML")
    dataset_name = root.attrib["name"]
    if (
        not dataset_name
        or dataset_name != dataset_name.strip()
        or len(dataset_name) > 80
        or any(character in dataset_name for character in "\x00\r\n")
    ):
        raise CaviarTrackingError("CAVIAR dataset name is invalid")
    if not 1 <= len(root) <= MAX_CAVIAR_FRAMES:
        raise CaviarTrackingError("CAVIAR frame count is outside the supported range")

    frames: list[_SourceFrame] = []
    object_total = 0
    for frame_element in root:
        if frame_element.tag != "frame" or set(frame_element.attrib) != {"number"}:
            raise CaviarTrackingError("CAVIAR frame fields do not match CVML")
        frame_number = _integer(frame_element.attrib.get("number"), "frame number")
        object_lists = frame_element.findall("objectlist")
        group_lists = frame_element.findall("grouplist")
        if (
            len(frame_element) != 2
            or len(object_lists) != 1
            or len(group_lists) != 1
            or {child.tag for child in frame_element} != {"objectlist", "grouplist"}
        ):
            raise CaviarTrackingError("CAVIAR frame must contain one objectlist and one grouplist")
        objects: list[_ObjectBox] = []
        seen_ids: set[int] = set()
        for object_element in object_lists[0]:
            if object_element.tag != "object" or set(object_element.attrib) != {"id"}:
                raise CaviarTrackingError("CAVIAR object fields do not match CVML")
            if any(child.tag not in _OBJECT_CHILD_TAGS for child in object_element):
                raise CaviarTrackingError("CAVIAR object contains an unsupported tag")
            upstream_id = _integer(object_element.attrib.get("id"), "object id")
            if upstream_id in seen_ids:
                raise CaviarTrackingError("CAVIAR object IDs must be unique per frame")
            seen_ids.add(upstream_id)
            boxes = object_element.findall("box")
            if len(boxes) != 1:
                raise CaviarTrackingError("CAVIAR object must contain exactly one box")
            objects.append(
                _ObjectBox(
                    upstream_id=upstream_id,
                    bbox_xyxy=_parse_box(boxes[0], frame_number=frame_number),
                )
            )
        object_total += len(objects)
        if object_total > MAX_CAVIAR_OBJECTS:
            raise CaviarTrackingError("CAVIAR object count exceeds the supported range")
        frames.append(
            _SourceFrame(
                number=frame_number,
                objects=tuple(objects),
                group_count=len(group_lists[0]),
            )
        )

    numbers = [frame.number for frame in frames]
    if numbers != list(range(numbers[0], numbers[-1] + 1)):
        raise CaviarTrackingError("CAVIAR frames must be ordered and consecutive")
    return dataset_name, tuple(frames)


def import_caviar_tracking_ground_truth(
    xml_path: str | Path,
    source_admission_path: str | Path,
    *,
    first_frame: int | None = None,
    last_frame: int | None = None,
) -> CaviarTrackingImport:
    admission = ExternalSourceAdmission.read(source_admission_path)
    if admission.asset_kind != "dataset_archive":
        raise CaviarTrackingError("CAVIAR admission must describe a dataset archive")
    if (
        not admission.labels.ground_truth
        or not admission.labels.human_verified
        or admission.labels.origin not in {"human_expert", "human_nonexpert"}
    ):
        raise CaviarTrackingError("CAVIAR admission must describe verified human ground truth")
    admission.verify_asset(xml_path)
    dataset_name, source_frames = _parse_source_frames(_read_xml(xml_path))

    source_first = source_frames[0].number
    source_last = source_frames[-1].number
    selected_first = source_first if first_frame is None else first_frame
    selected_last = source_last if last_frame is None else last_frame
    if (
        type(selected_first) is not int
        or type(selected_last) is not int
        or not source_first <= selected_first <= selected_last <= source_last
    ):
        raise CaviarTrackingError("CAVIAR frame window is outside the source range")
    selected = tuple(
        frame for frame in source_frames if selected_first <= frame.number <= selected_last
    )

    all_subject_ids = sorted({box.upstream_id for frame in source_frames for box in frame.objects})
    subject_ids = {
        upstream_id: f"subject_{index:03d}"
        for index, upstream_id in enumerate(all_subject_ids, start=1)
    }
    frames: list[DetectionFrame] = []
    subject_by_detection: dict[tuple[str, str], str] = {}
    for source_frame in selected:
        frame_id = f"frame_{source_frame.number:04d}"
        ordered = sorted(
            source_frame.objects,
            key=lambda item: (*item.bbox_xyxy, item.upstream_id),
        )
        detections: list[FrameDetection] = []
        for index, item in enumerate(ordered, start=1):
            detection_id = f"det_f{source_frame.number:04d}_{index:02d}"
            detections.append(
                FrameDetection(
                    detection_id=detection_id,
                    label="person",
                    confidence=1.0,
                    bbox_xyxy=item.bbox_xyxy,
                )
            )
            subject_by_detection[(frame_id, detection_id)] = subject_ids[item.upstream_id]
        frames.append(
            DetectionFrame(
                frame_id=frame_id,
                occurred_at=_EVALUATION_EPOCH
                + timedelta(microseconds=source_frame.number * _FRAME_MICROSECONDS),
                detections=tuple(detections),
            )
        )

    batch = DetectionFrameBatch(
        camera_id=_camera_id(dataset_name),
        detector_adapter=CAVIAR_BOX_SOURCE_ID,
        frames=tuple(frames),
    )
    assignments = tuple(
        TrackerReviewAssignment(
            frame_id=frame.frame_id,
            detection_id=detection.detection_id,
            review_state="assigned",
            subject_id=subject_by_detection[(frame.frame_id, detection.detection_id)],
        )
        for frame in batch.frames
        for detection in frame.detections
    )
    if not assignments:
        raise CaviarTrackingError("CAVIAR frame window contains no person boxes")
    review = TrackerAssociationReview(
        camera_id=batch.camera_id,
        detector_adapter=batch.detector_adapter,
        detection_batch_sha256=batch.fingerprint,
        source_admission_sha256=admission.fingerprint,
        review_status="approved",
        assignments=assignments,
        reviews=(),
        human_labels=True,
        ground_truth_provenance="admitted_publisher_human",
    )
    review.bind(batch)
    return CaviarTrackingImport(
        batch=batch,
        review=review,
        dataset_name=dataset_name,
        source_frame_count=len(source_frames),
        imported_frame_count=len(batch.frames),
        imported_detection_count=len(assignments),
        ignored_group_count=sum(frame.group_count for frame in selected),
        first_frame=selected_first,
        last_frame=selected_last,
    )
