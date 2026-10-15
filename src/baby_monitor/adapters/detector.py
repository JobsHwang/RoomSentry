from __future__ import annotations

import gc
import math
import os
import re
import shutil
import signal as signal_module
import stat
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

from baby_monitor.adapters.camera import CameraSource
from baby_monitor.detector_protocol import (
    DetectorProtocolError,
    DetectorScanRequest,
    DetectorScanResult,
)
from baby_monitor.detector_sampling import DetectorSampleFrame, DetectorSamplingLedger
from baby_monitor.domain import CandidateEvent, Detection
from baby_monitor.frame_transform import (
    inverse_normalized_box,
    rotate_frame,
    validate_image_size,
    validate_rotation_degrees,
)
from baby_monitor.video_timeline import decoded_frame_timestamp
from baby_monitor.yolo_model import YoloModelManifest


class DetectorError(RuntimeError):
    pass


class Detector(ABC):
    @abstractmethod
    def scan(self, source: CameraSource) -> Sequence[Detection]:
        raise NotImplementedError


class StaticDetector(Detector):
    """Deterministic detector for tests and explicit smoke demos only."""

    def __init__(self, detections: Iterable[Detection]):
        self.detections = tuple(detections)

    def scan(self, source: CameraSource) -> Sequence[Detection]:
        del source
        return self.detections


_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_PTS_TIME = re.compile(rf"\bpts_time:(?P<value>{_NUMBER_PATTERN})")
_SCENE_SCORE = re.compile(rf"\blavfi\.scene_score=(?P<value>{_NUMBER_PATTERN})")


class FFmpegSceneDetector(Detector):
    """Use FFmpeg scene scores as a cheap, model-free candidate trigger.

    A scene change is only a system-detected full-frame change. It does not
    identify a person, baby, pet, action, or risk and must not be presented as
    semantic recognition.
    """

    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        threshold: float = 0.15,
        max_detections: int = 200,
        timeout_seconds: float = 600.0,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("scene threshold must be greater than 0 and at most 1")
        if not 1 <= max_detections <= 10_000:
            raise ValueError("scene max detections must be between 1 and 10000")
        if not 1.0 <= timeout_seconds <= 3600.0:
            raise ValueError("scene detector timeout must be between 1 and 3600 seconds")
        self.ffmpeg = ffmpeg
        self.threshold = threshold
        self.max_detections = max_detections
        self.timeout_seconds = timeout_seconds

    def scan(self, source: CameraSource) -> Sequence[Detection]:
        if source.kind != "file":
            raise DetectorError(
                "the FFmpeg scene detector supports bounded local files; "
                "RTSP needs the ring buffer worker"
            )
        if shutil.which(self.ffmpeg) is None and not Path(self.ffmpeg).is_file():
            raise DetectorError("FFmpeg is not installed or configured")

        filter_graph = f"select=gt(scene\\,{self.threshold:.6f}),metadata=print:file=-"
        try:
            completed = subprocess.run(
                [
                    self.ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    source.read_uri,
                    "-an",
                    "-vf",
                    filter_graph,
                    "-frames:v",
                    str(self.max_detections),
                    "-f",
                    "null",
                    "-",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DetectorError("FFmpeg scene detection timed out") from exc
        if completed.returncode != 0:
            raise DetectorError("FFmpeg could not analyze the local video")

        detections: list[Detection] = []
        timestamp: float | None = None
        for line in completed.stdout.splitlines():
            timestamp_match = _PTS_TIME.search(line)
            if timestamp_match:
                timestamp = float(timestamp_match.group("value"))
                continue
            score_match = _SCENE_SCORE.search(line)
            if score_match and timestamp is not None:
                score = float(score_match.group("value"))
                if math.isfinite(timestamp) and math.isfinite(score) and timestamp >= 0:
                    detections.append(
                        Detection(
                            timestamp_seconds=timestamp,
                            label="scene_change",
                            confidence=min(1.0, max(0.0, score)),
                            bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
                        )
                    )
                timestamp = None
        return tuple(detections)


class UltralyticsYoloDetector(Detector):
    """Low-sampling-rate YOLO adapter for local video files.

    A concrete local weight file is mandatory. This deliberately prevents the
    Ultralytics library from treating a model name as permission to download it.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        manifest: YoloModelManifest,
        labels: Iterable[str],
        confidence: float,
        sample_fps: float,
        device: str,
        image_size: int = 640,
        rotation_degrees: int = 0,
    ):
        self.model_path = Path(model_path).expanduser()
        try:
            verification = manifest.verify_weight(self.model_path)
        except RuntimeError as exc:
            raise DetectorError(f"{exc}; automatic downloads are disabled") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.01 <= sample_fps <= 10.0:
            raise ValueError("sample_fps must be between 0.01 and 10")
        selected_labels = tuple(labels)
        if not selected_labels or len(set(selected_labels)) != len(selected_labels):
            raise ValueError("YOLO labels must be non-empty and unique")
        unknown_labels = sorted(set(selected_labels) - set(manifest.labels))
        if unknown_labels:
            raise DetectorError("configured YOLO labels are absent from the model manifest")
        if device != "cpu" and re.fullmatch(r"\d+", device) is None:
            raise ValueError("YOLO device must be 'cpu' or one numeric GPU index")
        validate_image_size(image_size)
        validate_rotation_degrees(rotation_degrees)
        self.manifest = manifest
        self.model_sha256 = verification.sha256
        self.labels = frozenset(selected_labels)
        self.confidence = confidence
        self.sample_fps = sample_fps
        self.device = device
        self.image_size = image_size
        self.rotation_degrees = rotation_degrees

    def _validate_loaded_model(self, model: object) -> None:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            try:
                loaded = tuple(str(names[index]) for index in range(len(names)))
            except (KeyError, TypeError) as exc:
                raise DetectorError("YOLO model class map is invalid") from exc
        elif isinstance(names, (list, tuple)):
            loaded = tuple(str(value) for value in names)
        else:
            raise DetectorError("YOLO model does not expose a class map")
        if loaded != self.manifest.labels:
            raise DetectorError("YOLO model class map does not match the reviewed manifest")

    def _release_accelerator_cache(self) -> None:
        """Best-effort CUDA cleanup before a later local VLM is started.

        The bounded file pipeline loads the detector and VLM sequentially.  A
        CUDA allocator may otherwise retain detector memory after the YOLO
        object is released, so explicitly return unused blocks to the driver.
        """

        gc.collect()
        if self.device == "cpu":
            return
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            # Cleanup must never mask the detector result or its original error.
            return

    def scan(self, source: CameraSource) -> Sequence[Detection]:
        return list(self.scan_result(source).detections)

    def scan_result(self, source: CameraSource) -> DetectorScanResult:
        if source.kind != "file":
            raise DetectorError(
                "the first-stage YOLO scanner supports bounded local files; "
                "RTSP needs the ring buffer worker"
            )
        try:
            import cv2  # type: ignore[import-not-found]
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DetectorError(
                "YOLO dependencies are missing; install the optional 'vision' extra after approval"
            ) from exc

        capture = cv2.VideoCapture(source.read_uri)
        detections: list[Detection] = []
        samples: list[DetectorSampleFrame] = []
        frame_index = 0
        fallback_count = 0
        previous_pts: float | None = None
        maximum_pts_gap: float | None = None
        regression_count = 0
        model: object | None = None
        results: object | None = None
        result: object | None = None
        boxes: object | None = None
        try:
            if not capture.isOpened():
                raise DetectorError("could not open the local video")
            native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            native_fps_source = "reported"
            if not math.isfinite(native_fps) or native_fps <= 0:
                native_fps = 25.0
                native_fps_source = "fallback_25"
            if not 0.01 <= native_fps <= 1000:
                raise DetectorError("detector native frame rate exceeds the supported bound")
            frame_step = max(1, round(native_fps / self.sample_fps))
            model = YOLO(str(self.model_path))
            self._validate_loaded_model(model)
            while True:
                ok, frame = capture.read()
                if not ok:
                    if frame_index == 0:
                        raise DetectorError("detector did not decode any frames")
                    break
                # A missing source can activate a model runtime's default
                # input. Never infer from anything but an actual decoded frame.
                if frame is None or getattr(frame, "size", None) == 0:
                    raise DetectorError("detector decoded an empty frame")
                if frame_index >= 10_000_000:
                    raise DetectorError("detector decoded-frame evidence exceeds the bound")
                timestamp, timestamp_source = decoded_frame_timestamp(
                    capture,
                    position_property=getattr(cv2, "CAP_PROP_POS_MSEC", None),
                    frame_index=frame_index,
                    native_fps=native_fps,
                )
                if timestamp_source == "nominal_fps":
                    fallback_count += 1
                    previous_pts = None
                else:
                    if previous_pts is not None:
                        gap = timestamp - previous_pts
                        if gap < 0:
                            regression_count += 1
                        else:
                            maximum_pts_gap = max(maximum_pts_gap or 0.0, gap)
                    previous_pts = timestamp
                if frame_index % frame_step != 0:
                    frame_index += 1
                    continue
                if len(samples) >= 100_000:
                    raise DetectorError("detector sampled-frame evidence exceeds the bound")
                detection_start = len(detections)

                try:
                    inference_frame = rotate_frame(
                        frame,
                        cv2_module=cv2,
                        rotation_degrees=self.rotation_degrees,
                    )
                except RuntimeError as exc:
                    raise DetectorError(str(exc)) from exc
                results = model.predict(
                    source=inference_frame,
                    conf=self.confidence,
                    device=self.device,
                    imgsz=self.image_size,
                    verbose=False,
                )
                if not isinstance(results, (list, tuple)) or len(results) != 1:
                    raise DetectorError(
                        "YOLO did not return exactly one result for the sampled frame"
                    )
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    classes = boxes.cls.detach().cpu().tolist()
                    scores = boxes.conf.detach().cpu().tolist()
                    coordinates = boxes.xyxyn.detach().cpu().tolist()
                    for class_index, score, coordinate in zip(
                        classes, scores, coordinates, strict=True
                    ):
                        class_value = float(class_index)
                        if not math.isfinite(class_value) or not class_value.is_integer():
                            raise DetectorError("YOLO returned an invalid class index")
                        class_id = int(class_value)
                        if not 0 <= class_id < len(self.manifest.labels):
                            raise DetectorError("YOLO returned an out-of-range class index")
                        label = self.manifest.labels[class_id]
                        if self.labels and label not in self.labels:
                            continue
                        score_value = float(score)
                        if not math.isfinite(score_value):
                            raise DetectorError("YOLO returned a non-finite confidence")
                        try:
                            bbox = inverse_normalized_box(
                                coordinate,
                                rotation_degrees=self.rotation_degrees,
                            )
                        except ValueError as exc:
                            raise DetectorError("YOLO returned an invalid normalized box") from exc
                        detections.append(
                            Detection(
                                timestamp_seconds=timestamp,
                                label=label,
                                confidence=score_value,
                                bbox_xyxy=bbox,
                            )
                        )
                        if len(detections) > 100_000:
                            raise DetectorError("detector detections exceed the bound")
                samples.append(
                    DetectorSampleFrame(
                        decoded_frame_index=frame_index,
                        timestamp_seconds=timestamp,
                        timestamp_source=timestamp_source,
                        detection_start=detection_start,
                        detection_count=len(detections) - detection_start,
                    )
                )
                frame_index += 1
        finally:
            capture.release()
            # Drop the last inference result as well as the model before asking
            # PyTorch to release cached CUDA blocks.
            boxes = None
            result = None
            results = None
            model = None
            self._release_accelerator_cache()
        return DetectorScanResult(
            camera_id=source.camera_id,
            detector_adapter="ultralytics_yolo_inprocess_v2",
            model_id=self.manifest.model_id,
            model_revision=self.manifest.revision,
            model_sha256=self.model_sha256,
            detections=tuple(detections),
            version=2,
            sampling=DetectorSamplingLedger(
                native_fps=native_fps,
                native_fps_source=native_fps_source,
                requested_sample_fps=self.sample_fps,
                frame_step=frame_step,
                decoded_frame_count=frame_index,
                samples=tuple(samples),
                timestamp_fallback_frame_count=fallback_count,
                max_adjacent_container_pts_gap_seconds=maximum_pts_gap,
                container_pts_regression_count=regression_count,
            ),
        )


class CommandYoloDetector(Detector):
    """Run the Ultralytics adapter in a short-lived, bounded subprocess.

    The parent process never imports torch, OpenCV or Ultralytics.  Only the
    path-bearing request file crosses the process boundary; stdout is parsed as
    a strict, path-free DetectorScanResult before detections enter the core.
    """

    runner_adapter = "ultralytics_yolo_subprocess_v1"
    _execution_lock_poll_seconds = 0.1
    _execution_lock_handoff_seconds = 0.12

    def __init__(
        self,
        *,
        model_path: str | Path,
        manifest_path: str | Path,
        manifest: YoloModelManifest,
        labels: Iterable[str],
        confidence: float,
        sample_fps: float,
        device: str,
        command: Sequence[str],
        image_size: int = 640,
        rotation_degrees: int = 0,
        timeout_seconds: float = 600.0,
        execution_lock_path: str | Path | None = None,
        execution_lock_timeout_seconds: float = 600.0,
    ):
        self.model_path = Path(model_path).expanduser()
        try:
            verification = manifest.verify_weight(self.model_path)
        except RuntimeError as exc:
            raise DetectorError(f"{exc}; automatic downloads are disabled") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.01 <= sample_fps <= 10.0:
            raise ValueError("sample_fps must be between 0.01 and 10")
        selected_labels = tuple(labels)
        if not selected_labels or len(set(selected_labels)) != len(selected_labels):
            raise ValueError("YOLO labels must be non-empty and unique")
        unknown_labels = sorted(set(selected_labels) - set(manifest.labels))
        if unknown_labels:
            raise DetectorError("configured YOLO labels are absent from the model manifest")
        if device != "cpu" and re.fullmatch(r"\d+", device) is None:
            raise ValueError("YOLO device must be 'cpu' or one numeric GPU index")
        validate_image_size(image_size)
        validate_rotation_degrees(rotation_degrees)
        if not command or not all(isinstance(part, str) and part for part in command):
            raise ValueError("YOLO runner command cannot be empty")
        if not Path(command[0]).expanduser().is_absolute():
            raise ValueError("YOLO runner command must start with an absolute executable")
        if sum(part.count("{request}") for part in command) != 1:
            raise ValueError("YOLO runner command must contain {request} exactly once")
        if any("\x00" in part for part in command):
            raise ValueError("YOLO runner command contains an invalid character")
        if not 1.0 <= timeout_seconds <= 3_600.0:
            raise ValueError("YOLO runner timeout must be between 1 and 3600 seconds")
        if not 1.0 <= execution_lock_timeout_seconds <= 3_600.0:
            raise ValueError("YOLO execution lock timeout must be between 1 and 3600 seconds")
        lock_path = (
            Path(execution_lock_path).expanduser() if execution_lock_path is not None else None
        )
        if lock_path is not None and (
            not lock_path.is_absolute()
            or lock_path.name in {"", ".", ".."}
            or os.name == "posix"
            and len(lock_path.parts) >= 3
            and tuple(part.casefold() for part in lock_path.parts[1:3]) == ("mnt", "c")
        ):
            raise ValueError("YOLO execution lock path is invalid")
        self.manifest = manifest
        self.manifest_path = Path(manifest_path).expanduser()
        if self.manifest_path.is_symlink() or not self.manifest_path.is_file():
            raise DetectorError("YOLO subprocess manifest must be a regular local file")
        self.model_path = self.model_path.resolve()
        self.manifest_path = self.manifest_path.resolve()
        self.model_sha256 = verification.sha256
        self.labels = frozenset(selected_labels)
        self.confidence = confidence
        self.sample_fps = sample_fps
        self.device = device
        self.image_size = image_size
        self.rotation_degrees = rotation_degrees
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.execution_lock_path = lock_path
        self.execution_lock_timeout_seconds = execution_lock_timeout_seconds
        self.last_execution_lock_wait_seconds: float | None = None

    @contextmanager
    def _execution_gate(self) -> Iterator[None]:
        path = self.execution_lock_path
        if path is None:
            self.last_execution_lock_wait_seconds = 0.0
            yield
            return
        if os.name != "posix":
            raise DetectorError("YOLO execution lock is unavailable")
        try:
            import fcntl

            parent = path.parent
            parent_metadata = parent.lstat()
            if (
                parent.is_symlink()
                or not parent.is_dir()
                or parent_metadata.st_uid != os.geteuid()
                or (parent_metadata.st_mode & 0o777) != 0o700
                or parent.resolve(strict=True) != parent
            ):
                raise DetectorError("YOLO execution lock is unsafe")
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
        except (OSError, DetectorError):
            raise DetectorError("YOLO execution lock is unsafe") from None
        acquired = False
        wait_started = time.monotonic()
        self.last_execution_lock_wait_seconds = None
        try:
            try:
                metadata = os.fstat(descriptor)
                path_metadata = path.lstat()
            except OSError:
                raise DetectorError("YOLO execution lock is unsafe") from None
            if (
                not stat.S_ISREG(metadata.st_mode)
                or path.is_symlink()
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_mode & 0o777 != 0o600
                or (metadata.st_dev, metadata.st_ino)
                != (path_metadata.st_dev, path_metadata.st_ino)
            ):
                raise DetectorError("YOLO execution lock is unsafe")
            deadline = time.monotonic() + self.execution_lock_timeout_seconds
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    self.last_execution_lock_wait_seconds = round(
                        time.monotonic() - wait_started,
                        6,
                    )
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise DetectorError("YOLO execution lock timed out") from None
                    time.sleep(self._execution_lock_poll_seconds)
                except OSError:
                    raise DetectorError("YOLO execution lock is unavailable") from None
            yield
        finally:
            if acquired:
                with suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            with suppress(OSError):
                os.close(descriptor)
            if acquired:
                # A worker can otherwise release and immediately reacquire the
                # lock before processes already sleeping in the bounded poll
                # loop get CPU time. This is a cooperative handoff window, not
                # a strict FIFO guarantee.
                time.sleep(self._execution_lock_handoff_seconds)

    @staticmethod
    def _subprocess_environment() -> dict[str, str]:
        allowed = {
            "CUDA_VISIBLE_DEVICES",
            "HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "LD_LIBRARY_PATH",
            "PATH",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "TMPDIR",
            "WINDIR",
        }
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        environment.update(
            {
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                "YOLO_AUTOINSTALL": "false",
                "YOLO_OFFLINE": "true",
            }
        )
        return environment

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[str]) -> None:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal_module.SIGKILL)
        else:
            with suppress(OSError):
                process.kill()

    def _request(self, source: CameraSource) -> DetectorScanRequest:
        source_path = Path(source.read_uri).expanduser()
        if not source_path.is_absolute() or source_path.is_symlink() or not source_path.is_file():
            raise DetectorError("YOLO subprocess source must be a regular local file")
        return DetectorScanRequest(
            camera_id=source.camera_id,
            detector_adapter=self.runner_adapter,
            source_path=str(source_path),
            model_path=str(self.model_path),
            manifest_path=str(self._manifest_path),
            model_id=self.manifest.model_id,
            model_revision=self.manifest.revision,
            model_sha256=self.model_sha256,
            labels=tuple(sorted(self.labels)),
            confidence=self.confidence,
            sample_fps=self.sample_fps,
            device=self.device,
            image_size=self.image_size,
            rotation_degrees=self.rotation_degrees,
        )

    @property
    def _manifest_path(self) -> Path:
        # The runner must re-read the reviewed manifest, not receive a
        # serialized object whose class map could be changed in transit.
        return self.manifest_path

    def scan(self, source: CameraSource) -> Sequence[Detection]:
        return self.scan_result(source).detections

    def scan_result(self, source: CameraSource) -> DetectorScanResult:
        if source.kind != "file":
            raise DetectorError(
                "the YOLO subprocess scanner supports bounded local files; "
                "RTSP needs the ring buffer worker"
            )
        request = self._request(source)
        with tempfile.TemporaryDirectory(prefix="baby-monitor-detector-") as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(request.to_json(), encoding="utf-8")
            with suppress(OSError):
                request_path.chmod(0o600)
            command = [part.replace("{request}", str(request_path)) for part in self.command]
            with self._execution_gate():
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=self._subprocess_environment(),
                        start_new_session=os.name == "posix",
                    )
                except OSError:
                    raise DetectorError("YOLO detector subprocess could not start") from None
                try:
                    stdout, _ = process.communicate(timeout=self.timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._kill_process_tree(process)
                    process.communicate()
                    raise DetectorError("YOLO detector subprocess timed out") from None
                except BaseException:
                    self._kill_process_tree(process)
                    process.communicate()
                    raise
                if process.returncode != 0:
                    raise DetectorError("YOLO detector subprocess failed")
            if len(stdout.encode("utf-8")) > 8 * 1024 * 1024:
                raise DetectorError("YOLO detector subprocess returned too much output")
            try:
                result = DetectorScanResult.from_json(stdout.strip())
            except DetectorProtocolError:
                raise DetectorError("YOLO detector subprocess returned invalid JSON") from None

        if (
            result.camera_id != source.camera_id
            or result.detector_adapter != self.runner_adapter
            or result.model_id != self.manifest.model_id
            or result.model_revision != self.manifest.revision
            or result.model_sha256 != self.model_sha256
        ):
            raise DetectorError(
                "YOLO detector subprocess metadata does not match the reviewed model"
            )
        if any(detection.label not in self.labels for detection in result.detections):
            raise DetectorError("YOLO detector subprocess returned an unselected label")
        if result.sampling is not None and result.sampling.requested_sample_fps != self.sample_fps:
            raise DetectorError("YOLO detector sampling evidence conflicts with request")
        return result


def group_detections(
    detections: Iterable[Detection],
    *,
    camera_id: str,
    gap_seconds: float,
    pre_event_seconds: float,
    post_event_seconds: float,
) -> tuple[CandidateEvent, ...]:
    if gap_seconds < 0 or pre_event_seconds < 0 or post_event_seconds < 0:
        raise ValueError("event window values cannot be negative")
    ordered = sorted(detections, key=lambda item: item.timestamp_seconds)
    if not ordered:
        return ()

    groups: list[list[Detection]] = [[ordered[0]]]
    for detection in ordered[1:]:
        if detection.timestamp_seconds - groups[-1][-1].timestamp_seconds <= gap_seconds:
            groups[-1].append(detection)
        else:
            groups.append([detection])

    candidates: list[CandidateEvent] = []
    for group in groups:
        candidates.append(
            CandidateEvent(
                event_id=uuid.uuid4().hex,
                camera_id=camera_id,
                start_seconds=max(0.0, group[0].timestamp_seconds - pre_event_seconds),
                end_seconds=group[-1].timestamp_seconds + post_event_seconds,
                detections=tuple(group),
            )
        )
    return tuple(candidates)
