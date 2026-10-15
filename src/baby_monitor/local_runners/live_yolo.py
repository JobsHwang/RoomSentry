"""Persistent YOLO runner. Camera frames arrive over an inherited socket offline."""

from __future__ import annotations

import argparse
import hashlib
import socket
import time

from baby_monitor.adapters.detector import UltralyticsYoloDetector
from baby_monitor.live_detector_protocol import receive_packet, send_packet
from baby_monitor.live_review_cadence import (
    AlternateReviewCadence,
    validate_empty_review_intervals,
)
from baby_monitor.yolo_model import YoloModelManifest


def run(channel: socket.socket) -> None:
    if {name for _, name in socket.if_nameindex()} != {"lo"}:
        raise RuntimeError("offline model namespace required")
    request, payload = receive_packet(channel)
    if request.get("type") != "initialize" or request.get("version") != 1 or payload:
        raise ValueError("invalid live detector initialization")
    manifest = YoloModelManifest.read(request["manifest_path"])
    half_precision = request.get("half_precision", False)
    if type(half_precision) is not bool or (half_precision and request["device"] == "cpu"):
        raise ValueError("invalid live detector precision")
    admission = UltralyticsYoloDetector(
        model_path=request["model_path"],
        manifest=manifest,
        labels=("person",),
        confidence=request["confidence"],
        sample_fps=1,
        device=request["device"],
        image_size=request["image_size"],
    )
    image_height = request.get("image_height")
    if image_height is not None and (
        type(image_height) is not int
        or not 320 <= image_height <= admission.image_size
        or image_height % 32
    ):
        raise ValueError("invalid live input shape")
    image_shape = (image_height or admission.image_size, admission.image_size)
    review_rotation_180 = request.get("review_rotation_180", False)
    if type(review_rotation_180) is not bool:
        raise ValueError("invalid alternate review view")
    empty_intervals = validate_empty_review_intervals(
        request.get("review_empty_interval_seconds_by_camera"), enabled=review_rotation_180
    )
    cadence = AlternateReviewCadence(empty_intervals)
    review_empty_scale_fallback = request.get("review_empty_scale_fallback", False)
    if type(review_empty_scale_fallback) is not bool or (
        review_empty_scale_fallback and (not review_rotation_180 or image_shape != (384, 640))
    ):
        raise ValueError("scale fallback requires admitted primary view")
    review_nonempty_shadow = request.get("review_nonempty_shadow", False)
    if type(review_nonempty_shadow) is not bool or (
        review_nonempty_shadow and not review_rotation_180
    ):
        raise ValueError("nonempty shadow review requires alternate review view")
    import numpy as np
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    model = YOLO(str(admission.model_path))
    admission._validate_loaded_model(model)
    # Complete GPU initialization before opening any live source, avoiding a warm-up backlog.
    # Available sources may arrive asynchronously. Prewarm every allowed batch size,
    # not only four, before any decoder begins feeding the bounded FIFO.
    for batch_size in range(1, 5):
        model.predict(
            [np.zeros((480, 864, 3), dtype=np.uint8) for _ in range(batch_size)],
            device=admission.device,
            imgsz=image_shape,
            conf=admission.confidence,
            classes=[manifest.labels.index("person")],
            max_det=100,
            rect=False,
            half=half_precision,
            verbose=False,
        )
    if admission.device != "cpu":
        torch.cuda.empty_cache()
    if review_empty_scale_fallback:
        for batch_size in range(1, 5):
            model.predict(
                [np.zeros((480, 864, 3), dtype=np.uint8) for _ in range(batch_size)],
                device=admission.device,
                imgsz=(576, 960),
                conf=admission.confidence,
                classes=[manifest.labels.index("person")],
                max_det=100,
                rect=False,
                half=half_precision,
                verbose=False,
            )
        if admission.device != "cpu":
            torch.cuda.empty_cache()
    send_packet(
        channel,
        {
            "type": "ready",
            "model_id": manifest.model_id,
            "model_sha256": admission.model_sha256,
            "device": admission.device,
            "image_size": admission.image_size,
            "image_shape": list(image_shape),
            "confidence": admission.confidence,
            "network_enabled": False,
            "model_loads": 1,
            "weak_trace_protocol": 1,
            "precision": "fp16" if half_precision else "fp32",
            "review_rotation_180": review_rotation_180,
            "review_nonempty_shadow": review_nonempty_shadow,
            "review_empty_scale_fallback": review_empty_scale_fallback,
            "review_interval_seconds": 1.0 if review_rotation_180 else None,
            "review_empty_interval_seconds_by_camera": empty_intervals,
            "review_shadow_interval_seconds": 1.0 if review_rotation_180 else None,
            "review_fallback_interval_seconds": 1.0 if review_rotation_180 else None,
        },
    )
    while True:
        request, payload = receive_packet(channel)
        if request.get("type") == "stop" and not payload:
            return
        entries = request.get("frames")
        if (
            request.get("type") != "predict"
            or not isinstance(entries, list)
            or not 1 <= len(entries) <= 4
        ):
            raise ValueError("invalid live frame batch")
        if len({entry["camera_id"] for entry in entries}) != len(entries):
            raise ValueError("duplicate live inference camera")
        images, offset = [], 0
        for entry in entries:
            width, height = entry["width"], entry["height"]
            if (
                type(width) is not int
                or type(height) is not int
                or not 1 <= width <= 4096
                or not 1 <= height <= 4096
                or width * height > 4_000_000
            ):
                raise ValueError("invalid live frame dimensions")
            size = width * height * 3
            data = payload[offset : offset + size]
            if (
                entry["byte_count"] != size
                or len(data) != size
                or hashlib.sha256(data).hexdigest() != entry["sha256"]
            ):
                raise ValueError("live frame bytes mismatch")
            images.append(np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3))
            offset += size
        if offset != len(payload):
            raise ValueError("unexpected live frame bytes")
        weak_requested = request.get("weak_trace", False)
        if type(weak_requested) is not bool or (weak_requested and admission.confidence != 0.35):
            raise ValueError("invalid weak trace request")
        if weak_requested and admission.device != "cpu":
            torch.cuda.synchronize()
        primary_started = time.monotonic() if weak_requested else None
        predictions = model.predict(
            images,
            device=admission.device,
            imgsz=image_shape,
            conf=admission.confidence,
            classes=[manifest.labels.index("person")],
            max_det=100,
            rect=False,
            half=half_precision,
            verbose=False,
        )
        if len(predictions) != len(entries):
            raise RuntimeError("live detector batch mismatch")
        rows = []
        for entry, result in zip(entries, predictions, strict=True):
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    detections.append(
                        {
                            "confidence": float(box.conf.item()),
                            "bbox_xyxy": [float(item) for item in box.xyxyn[0].tolist()],
                        }
                    )
            rows.append(
                {
                    "camera_id": entry["camera_id"],
                    "sequence": entry["sequence"],
                    "sha256": entry["sha256"],
                    "detections": detections,
                    "review_checked": False,
                    "shadow_checked": False,
                    "shadow_skip_reason": "disabled"
                    if not review_nonempty_shadow
                    else "rate_limited"
                    if detections
                    else "native_empty",
                }
            )
        diagnostic = None
        if weak_requested:
            if admission.device != "cpu":
                torch.cuda.synchronize()
            primary_ms = (time.monotonic() - primary_started) * 1000
            weak_started = time.monotonic()
            weak_rows = []
            weak_status = "completed"
            try:
                weak_predictions = model.predict(
                    images,
                    device=admission.device,
                    imgsz=image_shape,
                    conf=0.20,
                    classes=[manifest.labels.index("person")],
                    max_det=100,
                    rect=False,
                    half=half_precision,
                    verbose=False,
                )
                if len(weak_predictions) != len(entries):
                    raise RuntimeError("weak detector batch mismatch")
                for entry, result in zip(entries, weak_predictions, strict=True):
                    weak_rows.append(
                        {
                            "camera_id": entry["camera_id"],
                            "sequence": entry["sequence"],
                            "sha256": entry["sha256"],
                            "detections": []
                            if result.boxes is None
                            else [
                                {
                                    "confidence": float(box.conf.item()),
                                    "bbox_xyxy": [float(v) for v in box.xyxyn[0].tolist()],
                                }
                                for box in result.boxes
                            ],
                        }
                    )
            except Exception:
                # Extra diagnostic failure must not replace valid primary boxes
                # with an empty frame. Raw third-party errors are never exported.
                weak_rows, weak_status = [], "inference_failed"
            if admission.device != "cpu":
                torch.cuda.synchronize()
            diagnostic = {
                "version": 1,
                "status": weak_status,
                "frames": weak_rows,
                "primary_ms": primary_ms,
                "weak_ms": (time.monotonic() - weak_started) * 1000,
            }
        if review_rotation_180:
            now = time.monotonic()
            indices = [
                index
                for index, row in enumerate(rows)
                if (not row["detections"] or review_nonempty_shadow)
                and cadence.rotation_due(
                    row["camera_id"], native_present=bool(row["detections"]), now=now
                )
            ]
            if indices:
                alternate_images = [
                    np.ascontiguousarray(images[index][::-1, ::-1]) for index in indices
                ]
                rotation_started = time.monotonic()
                alternate = model.predict(
                    alternate_images,
                    device=admission.device,
                    imgsz=image_shape,
                    conf=admission.confidence,
                    classes=[manifest.labels.index("person")],
                    max_det=100,
                    rect=False,
                    half=half_precision,
                    verbose=False,
                )
                if len(alternate) != len(indices):
                    raise RuntimeError("alternate review batch mismatch")
                for index, result in zip(indices, alternate, strict=True):
                    hints = []
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = (float(value) for value in box.xyxyn[0].tolist())
                            hints.append(
                                {
                                    "confidence": float(box.conf.item()),
                                    "bbox_xyxy": [1 - x2, 1 - y2, 1 - x1, 1 - y1],
                                }
                            )
                    if rows[index]["detections"]:
                        rows[index].update(
                            shadow_checked=True, shadow_skip_reason=None, shadow_detections=hints
                        )
                    else:
                        rows[index].update(review_checked=True, review_detections=hints)
                    cadence.mark_rotation(rows[index]["camera_id"], rotation_started)
                # Preserve both original views. Only an empty native AND empty
                # rotated baseline may request the admitted larger review view.
                double_empty_indices = [
                    index
                    for index in indices
                    if not rows[index]["detections"] and not rows[index].get("review_detections")
                ]
                fallback_now = time.monotonic()
                fallback_indices = [
                    index
                    for index in double_empty_indices
                    if cadence.fallback_due(rows[index]["camera_id"], fallback_now)
                ]
                if review_empty_scale_fallback and fallback_indices:
                    fallback_images = [
                        np.ascontiguousarray(images[index][::-1, ::-1])
                        for index in fallback_indices
                    ]
                    fallback_started = time.monotonic()
                    fallback_results = model.predict(
                        fallback_images,
                        device=admission.device,
                        imgsz=(576, 960),
                        conf=admission.confidence,
                        classes=[manifest.labels.index("person")],
                        max_det=100,
                        rect=False,
                        half=half_precision,
                        verbose=False,
                    )
                    if len(fallback_results) != len(fallback_indices):
                        raise RuntimeError("scale fallback batch mismatch")
                    for index, result in zip(fallback_indices, fallback_results, strict=True):
                        hints = []
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = (float(v) for v in box.xyxyn[0].tolist())
                                hints.append(
                                    {
                                        "confidence": float(box.conf.item()),
                                        "bbox_xyxy": [1 - x2, 1 - y2, 1 - x1, 1 - y1],
                                    }
                                )
                        rows[index].update(review_fallback_checked=True, review_detections=hints)
                        cadence.mark_fallback(rows[index]["camera_id"], fallback_started)
        send_packet(
            channel,
            {
                "type": "predictions",
                "frames": rows,
                **({"weak_trace": diagnostic} if diagnostic is not None else {}),
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", required=True, type=int)
    args = parser.parse_args()
    try:
        with socket.socket(fileno=args.fd) as channel:
            run(channel)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
