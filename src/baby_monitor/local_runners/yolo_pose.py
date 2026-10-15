from __future__ import annotations

import argparse
import gc
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from baby_monitor.frame_transform import (
    inverse_normalized_box,
    inverse_normalized_point,
    rotate_frame,
)
from baby_monitor.pose_protocol import (
    COCO_PERSON_KEYPOINTS,
    MAX_TOTAL_POSES,
    PoseFrame,
    PoseFrameBatch,
    PoseInstance,
    PoseKeypoint,
    PoseProtocolError,
    PoseScanRequest,
)
from baby_monitor.video_timeline import decoded_frame_timestamp_seconds
from baby_monitor.yolo_pose_model import YoloPoseModelManifest


def _tolist(value: Any, field: str) -> Any:
    try:
        return value.detach().cpu().tolist()
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise PoseProtocolError(f"pose runner {field} tensor is invalid") from exc


def _loaded_names(model: object) -> tuple[str, ...]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        try:
            return tuple(str(names[index]) for index in range(len(names)))
        except (KeyError, TypeError) as exc:
            raise PoseProtocolError("pose model class map is invalid") from exc
    if isinstance(names, (list, tuple)):
        return tuple(str(item) for item in names)
    raise PoseProtocolError("pose model does not expose a class map")


def _validate_loaded_model(model: object, manifest: YoloPoseModelManifest) -> None:
    if getattr(model, "task", None) != "pose":
        raise PoseProtocolError("reviewed YOLO model is not a pose model")
    if _loaded_names(model) != (manifest.class_name,):
        raise PoseProtocolError("pose model class map does not match the manifest")
    inner = getattr(model, "model", None)
    shape = getattr(inner, "kpt_shape", None)
    if not isinstance(shape, (list, tuple)) or tuple(shape) != (
        len(manifest.keypoint_names),
        manifest.keypoint_dimensions,
    ):
        raise PoseProtocolError("pose model keypoint shape does not match the manifest")


def _keypoint(
    name: str,
    coordinate: Any,
    confidence: Any,
    *,
    rotation_degrees: int,
) -> PoseKeypoint:
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
        raise PoseProtocolError("pose runner keypoint coordinate is invalid")
    try:
        x = float(coordinate[0])
        y = float(coordinate[1])
        score = float(confidence)
    except (TypeError, ValueError) as exc:
        raise PoseProtocolError("pose runner keypoint value is invalid") from exc
    in_frame = math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    if in_frame:
        try:
            x, y = inverse_normalized_point(
                x,
                y,
                rotation_degrees=rotation_degrees,
            )
        except ValueError as exc:
            raise PoseProtocolError("pose runner keypoint value is invalid") from exc
    return PoseKeypoint(
        name=name,
        x=x if in_frame else None,
        y=y if in_frame else None,
        confidence=score,
        in_frame=in_frame,
    )


def _poses_from_result(
    result: object,
    *,
    rotation_degrees: int,
) -> tuple[PoseInstance, ...]:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None and keypoints is None:
        return ()
    if boxes is None or keypoints is None:
        raise PoseProtocolError("pose runner returned incomplete pose tensors")
    coordinates = _tolist(getattr(boxes, "xyxyn", None), "bbox")
    scores = _tolist(getattr(boxes, "conf", None), "bbox confidence")
    points = _tolist(getattr(keypoints, "xyn", None), "keypoint coordinate")
    point_confidences = _tolist(getattr(keypoints, "conf", None), "keypoint confidence")
    if not all(
        isinstance(value, list) for value in (coordinates, scores, points, point_confidences)
    ):
        raise PoseProtocolError("pose runner returned malformed pose tensors")
    if not len(coordinates) == len(scores) == len(points) == len(point_confidences):
        raise PoseProtocolError("pose runner pose tensor counts do not align")

    poses: list[PoseInstance] = []
    for index, (bbox, score, pose_points, pose_confidences) in enumerate(
        zip(coordinates, scores, points, point_confidences, strict=True),
        start=1,
    ):
        if (
            not isinstance(pose_points, list)
            or not isinstance(pose_confidences, list)
            or len(pose_points) != len(COCO_PERSON_KEYPOINTS)
            or len(pose_confidences) != len(COCO_PERSON_KEYPOINTS)
        ):
            raise PoseProtocolError("pose runner did not return the COCO-17 keypoint set")
        try:
            normalized_bbox = inverse_normalized_box(
                bbox,
                rotation_degrees=rotation_degrees,
            )
        except ValueError as exc:
            raise PoseProtocolError("pose runner bbox is invalid") from exc
        poses.append(
            PoseInstance(
                instance_id=f"p{index:04d}",
                confidence=float(score),
                bbox_xyxy=normalized_bbox,
                keypoints=tuple(
                    _keypoint(
                        name,
                        coordinate,
                        point_confidence,
                        rotation_degrees=rotation_degrees,
                    )
                    for name, coordinate, point_confidence in zip(
                        COCO_PERSON_KEYPOINTS,
                        pose_points,
                        pose_confidences,
                        strict=True,
                    )
                ),
            )
        )
    return tuple(poses)


def _release_accelerator_cache(device: str) -> None:
    gc.collect()
    if device == "cpu":
        return
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        return


def run_request(request: PoseScanRequest) -> PoseFrameBatch:
    manifest = YoloPoseModelManifest.read(request.manifest_path)
    if (
        manifest.model_id != request.model_id
        or manifest.revision != request.model_revision
        or manifest.sha256 != request.model_sha256
    ):
        raise PoseProtocolError("pose request does not match the reviewed manifest")
    manifest.verify_weight(request.model_path)

    try:
        import cv2  # type: ignore[import-not-found]
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as exc:
        raise PoseProtocolError("pose runtime dependencies are missing") from exc

    capture = cv2.VideoCapture(request.source_path)
    frames: list[PoseFrame] = []
    total_poses = 0
    frame_index = 0
    model: object | None = None
    results: object | None = None
    try:
        if not capture.isOpened():
            raise PoseProtocolError("pose runner could not open the local video")
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(native_fps) or native_fps <= 0:
            native_fps = 25.0
        frame_step = max(1, round(native_fps / request.sample_fps))
        model = YOLO(request.model_path)
        _validate_loaded_model(model, manifest)
        while len(frames) < request.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or getattr(frame, "size", None) == 0:
                raise PoseProtocolError("pose runner decoded an empty frame")
            if frame_index % frame_step != 0:
                frame_index += 1
                continue
            try:
                inference_frame = rotate_frame(
                    frame,
                    cv2_module=cv2,
                    rotation_degrees=request.rotation_degrees,
                )
            except RuntimeError as exc:
                raise PoseProtocolError(str(exc)) from exc
            results = model.predict(
                source=inference_frame,
                conf=request.confidence,
                device=request.device,
                imgsz=request.image_size,
                verbose=False,
            )
            if not isinstance(results, (list, tuple)) or len(results) != 1:
                raise PoseProtocolError("pose runner expected one result per sampled frame")
            poses = _poses_from_result(
                results[0],
                rotation_degrees=request.rotation_degrees,
            )
            total_poses += len(poses)
            if total_poses > MAX_TOTAL_POSES:
                raise PoseProtocolError("pose runner exceeded the total pose limit")
            frames.append(
                PoseFrame(
                    frame_id=f"f{len(frames) + 1:06d}",
                    timestamp_seconds=decoded_frame_timestamp_seconds(
                        capture,
                        position_property=getattr(cv2, "CAP_PROP_POS_MSEC", None),
                        frame_index=frame_index,
                        native_fps=native_fps,
                    ),
                    poses=poses,
                )
            )
            frame_index += 1
    finally:
        capture.release()
        results = None
        model = None
        _release_accelerator_cache(request.device)

    if not frames:
        raise PoseProtocolError("pose runner did not sample any frames")
    return PoseFrameBatch(
        camera_id=request.camera_id,
        pose_adapter=request.pose_adapter,
        model_id=manifest.model_id,
        model_revision=manifest.revision,
        model_sha256=manifest.sha256,
        frames=tuple(frames),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("request")
    args = parser.parse_args(argv)
    try:
        request = PoseScanRequest.read(Path(args.request))
        # The parent expects one strict JSON response and suppresses stderr.
        # Third-party import/model banners must not corrupt that response.
        with redirect_stdout(sys.stderr):
            result = run_request(request)
    except (OSError, RuntimeError, ValueError):
        print("pose_runner_failed", file=sys.stderr)
        return 2
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
