from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import re
import stat
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from baby_monitor.domain import SchemaError
from baby_monitor.local_runners.common import report_runner_failure
from baby_monitor.local_runners.mage_vl_4b import (
    OFFICIAL_REVISION,
    MageVLRunnerError,
    MageVLSettings,
    TrustedModelCodeManifest,
    _build_quantization_config,
    _verify_int8_layout,
    isolated_transformers_module_cache,
)
from baby_monitor.visual_predicate import (
    VisualPredicate,
    VisualPredicateAnswer,
    VisualPredicateResult,
)

PROMPT_VERSION = "mage_atomic_visible_v1"
RESULT_CONTRACT = "visual_predicate_result_v1"
ATOMIC_MODEL_ID = "microsoft.Mage-VL"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MEDIA_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG"}
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_FRAME_BYTES = 25 * 1024 * 1024
_MAX_TOTAL_FRAME_BYTES = 100 * 1024 * 1024
_MAX_IMAGE_PIXELS = 50_000_000
_VIDEO_CANVAS_SIZE = (512, 512)
_VIDEO_CANVAS_COLOR = (127, 127, 127)


class MagePredicateRunnerError(MageVLRunnerError):
    pass


@dataclass(frozen=True, slots=True)
class MagePredicateFrame:
    frame_id: str
    sha256: str
    media_type: str
    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if _SAFE_ID.fullmatch(self.frame_id) is None:
            raise MagePredicateRunnerError("predicate frame ID is invalid")
        if _SHA256.fullmatch(self.sha256) is None:
            raise MagePredicateRunnerError("predicate frame hash is invalid")
        if self.media_type not in _MEDIA_TYPES:
            raise MagePredicateRunnerError("predicate frame media type is invalid")
        if (
            not isinstance(self.payload, bytes)
            or not 0 < len(self.payload) <= _MAX_FRAME_BYTES
            or hashlib.sha256(self.payload).hexdigest() != self.sha256
        ):
            raise MagePredicateRunnerError("predicate frame admission failed")


@dataclass(frozen=True, slots=True)
class MagePredicateRequest:
    event_id: str
    predicate: VisualPredicate
    frames: tuple[MagePredicateFrame, ...]
    model_id: str
    model_revision: str
    prompt_version: str
    version: int = 1
    result_contract: str = RESULT_CONTRACT

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise MagePredicateRunnerError("predicate request version is invalid")
        if _SAFE_ID.fullmatch(self.event_id) is None:
            raise MagePredicateRunnerError("predicate event ID is invalid")
        if not isinstance(self.predicate, VisualPredicate):
            raise MagePredicateRunnerError("predicate name is invalid")
        if (
            not isinstance(self.frames, tuple)
            or not 1 <= len(self.frames) <= 12
            or any(not isinstance(frame, MagePredicateFrame) for frame in self.frames)
            or len({frame.frame_id for frame in self.frames}) != len(self.frames)
            or sum(len(frame.payload) for frame in self.frames) > _MAX_TOTAL_FRAME_BYTES
        ):
            raise MagePredicateRunnerError("predicate frames are invalid")
        if self.model_id != ATOMIC_MODEL_ID or self.model_revision != OFFICIAL_REVISION:
            raise MagePredicateRunnerError("predicate model identity is invalid")
        if self.prompt_version != PROMPT_VERSION:
            raise MagePredicateRunnerError("predicate prompt version is invalid")
        if self.result_contract != RESULT_CONTRACT:
            raise MagePredicateRunnerError("predicate result contract is invalid")

    @property
    def frame_ids(self) -> tuple[str, ...]:
        return tuple(frame.frame_id for frame in self.frames)


def _strict_json(raw: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise MagePredicateRunnerError("predicate request contains duplicate keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                MagePredicateRunnerError("predicate request contains invalid numbers")
            ),
        )
    except (json.JSONDecodeError, UnicodeError):
        raise MagePredicateRunnerError("predicate request JSON is invalid") from None


def _private_root(request_path: Path) -> Path:
    if not request_path.is_absolute() or request_path.name != "request.json":
        raise MagePredicateRunnerError("predicate request path is invalid")
    root = request_path.parent
    try:
        root_metadata = root.stat(follow_symlinks=False)
    except OSError:
        raise MagePredicateRunnerError("predicate staging root is unsafe") from None
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or root.is_symlink()
        or (os.name == "posix" and root_metadata.st_uid != os.getuid())
        or (os.name == "posix" and stat.S_IMODE(root_metadata.st_mode) != 0o700)
    ):
        raise MagePredicateRunnerError("predicate staging root is unsafe")
    return root.resolve(strict=True)


def _read_private_file(path: Path, *, root: Path, limit: int) -> bytes:
    try:
        if path.parent.resolve(strict=True) != root or path.name in {"", ".", ".."}:
            raise MagePredicateRunnerError("predicate private file is unsafe")
    except OSError:
        raise MagePredicateRunnerError("predicate private file is unsafe") from None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise MagePredicateRunnerError("predicate private file is unsafe") from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= limit
            or (os.name == "posix" and metadata.st_uid != os.getuid())
            or (os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600)
        ):
            raise MagePredicateRunnerError("predicate private file is unsafe")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise MagePredicateRunnerError("predicate private file is unsafe")
            chunks.append(chunk)
        if total != metadata.st_size:
            raise MagePredicateRunnerError("predicate private file changed during admission")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_predicate_request(path: str | Path) -> MagePredicateRequest:
    request_path = Path(path).expanduser()
    root = _private_root(request_path)
    raw = _read_private_file(request_path, root=root, limit=_MAX_REQUEST_BYTES)
    try:
        value = _strict_json(raw.decode("utf-8"))
    except UnicodeError:
        raise MagePredicateRunnerError("predicate request JSON is invalid") from None
    expected = {
        "version",
        "event_id",
        "predicate",
        "frames",
        "model_id",
        "model_revision",
        "prompt_version",
        "result_contract",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise MagePredicateRunnerError("predicate request fields are invalid")
    if not isinstance(value["frames"], list) or not 1 <= len(value["frames"]) <= 12:
        raise MagePredicateRunnerError("predicate frames are invalid")
    frames: list[MagePredicateFrame] = []
    for item in value["frames"]:
        frame_fields = {"frame_id", "sha256", "media_type", "path"}
        if not isinstance(item, dict) or set(item) != frame_fields:
            raise MagePredicateRunnerError("predicate frame fields are invalid")
        if any(not isinstance(item[key], str) for key in frame_fields):
            raise MagePredicateRunnerError("predicate frame fields are invalid")
        frame_path = Path(item["path"])
        if not frame_path.is_absolute():
            raise MagePredicateRunnerError("predicate frame path is invalid")
        payload = _read_private_file(frame_path, root=root, limit=_MAX_FRAME_BYTES)
        frames.append(
            MagePredicateFrame(
                frame_id=item["frame_id"],
                sha256=item["sha256"],
                media_type=item["media_type"],
                payload=payload,
            )
        )
    try:
        predicate = VisualPredicate(value["predicate"])
    except (TypeError, ValueError):
        raise MagePredicateRunnerError("predicate name is invalid") from None
    return MagePredicateRequest(
        version=value["version"],
        event_id=value["event_id"],
        predicate=predicate,
        frames=tuple(frames),
        model_id=value["model_id"],
        model_revision=value["model_revision"],
        prompt_version=value["prompt_version"],
        result_contract=value["result_contract"],
    )


_QUESTIONS = {
    VisualPredicate.FACE_VISIBILITY: (
        "Judge the best directly visible human face. clear means eyes, nose, and mouth are "
        "clearly visible at useful resolution; partially_obscured means a face is visible but "
        "important parts are obscured or too small; not_visible means no human face is visible; "
        "unknown means the evidence is unreadable or genuinely ambiguous. Allowed labels: "
        "clear, partially_obscured, not_visible, unknown."
    ),
    VisualPredicate.SUPPORT_CONTACT_VISIBLE: (
        "Is direct contact between an anonymous person and a supporting surface clearly visible? "
        "Answer yes only for visible contact, no only for clearly visible separation, and unknown "
        "when contact is hidden or ambiguous. Allowed labels: yes, no, unknown."
    ),
    VisualPredicate.POSTURE_TRANSITION_VISIBLE: (
        "Across time, is a clear change in an anonymous person's body posture, height, or support "
        "position directly visible? Answer yes only for a visible transition, no only when enough "
        "temporal evidence shows no transition, and unknown when association or visibility is "
        "ambiguous. Allowed labels: yes, no, unknown."
    ),
    VisualPredicate.COVERING_NEAR_FACE_VISIBLE: (
        "Is an object directly visible covering or immediately against an anonymous person's face? "
        "Answer yes only when visibly present, no only when a sufficiently clear face area is not "
        "covered, and unknown when the face area is hidden or ambiguous. Allowed labels: yes, no, "
        "unknown."
    ),
}


def build_atomic_prompt(request: MagePredicateRequest) -> tuple[str, str]:
    order = ", ".join(request.frame_ids)
    system = (
        "You are an offline visual verifier. Return exactly one lowercase allowed label and "
        "nothing else. Use only directly visible evidence. Never infer identity, age, intent, "
        "risk, or a notification decision. Detector crops are anonymous hints and may be false "
        "positives."
    )
    user = (
        f"Input order: {order}. IDs ending in _full are full-scene frames. IDs containing _crop "
        "are detector-suggested crops from the same time index, not additional moments. "
        f"{_QUESTIONS[request.predicate]}"
    )
    return system, user


def normalize_model_answer(
    request: MagePredicateRequest,
    raw: str,
) -> VisualPredicateResult:
    allowed = {
        VisualPredicate.FACE_VISIBILITY: {
            VisualPredicateAnswer.CLEAR,
            VisualPredicateAnswer.PARTIALLY_OBSCURED,
            VisualPredicateAnswer.NOT_VISIBLE,
            VisualPredicateAnswer.UNKNOWN,
        },
        VisualPredicate.SUPPORT_CONTACT_VISIBLE: {
            VisualPredicateAnswer.YES,
            VisualPredicateAnswer.NO,
            VisualPredicateAnswer.UNKNOWN,
        },
        VisualPredicate.POSTURE_TRANSITION_VISIBLE: {
            VisualPredicateAnswer.YES,
            VisualPredicateAnswer.NO,
            VisualPredicateAnswer.UNKNOWN,
        },
        VisualPredicate.COVERING_NEAR_FACE_VISIBLE: {
            VisualPredicateAnswer.YES,
            VisualPredicateAnswer.NO,
            VisualPredicateAnswer.UNKNOWN,
        },
    }[request.predicate]
    try:
        answer = VisualPredicateAnswer(raw.strip())
    except (TypeError, ValueError):
        answer = VisualPredicateAnswer.UNKNOWN
    if answer not in allowed:
        answer = VisualPredicateAnswer.UNKNOWN
    evidence = () if answer is VisualPredicateAnswer.UNKNOWN else request.frame_ids
    return VisualPredicateResult(
        event_id=request.event_id,
        predicate=request.predicate,
        answer=answer,
        confidence=0.0 if answer is VisualPredicateAnswer.UNKNOWN else 0.5,
        evidence_frames=evidence,
        model_id=request.model_id,
        model_revision=request.model_revision,
        prompt_version=request.prompt_version,
    )


class MagePredicateSession:
    """One loaded Mage model; callers must bound jobs/lifetime outside this class."""

    def __init__(self, settings: MageVLSettings):
        self.settings = settings
        self._torch: Any = None
        self._image_type: Any = None
        self._processor: Any = None
        self._model: Any = None

    def __enter__(self) -> MagePredicateSession:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise MagePredicateRunnerError(
                "Mage predicate runtime dependencies are not installed"
            ) from exc
        if not torch.cuda.is_available():
            raise MagePredicateRunnerError("CUDA is not available to the Mage predicate runner")
        self._torch = torch
        self._image_type = Image
        self._processor = AutoProcessor.from_pretrained(
            str(self.settings.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        quantization_config = _build_quantization_config(
            BitsAndBytesConfig,
            self.settings.language_linear_precision,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "max_memory": {
                0: f"{self.settings.max_gpu_memory_mib}MiB",
                "cpu": f"{self.settings.max_cpu_memory_mib}MiB",
            },
            "attn_implementation": "sdpa",
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.settings.model_path),
            **model_kwargs,
        ).eval()
        if self.settings.language_linear_precision == "int8":
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise MagePredicateRunnerError(
                    "Mage predicate INT8 dependency is not installed"
                ) from exc
            _verify_int8_layout(self._model, bnb.nn.Linear8bitLt)
        return self

    def assess(self, request: MagePredicateRequest) -> VisualPredicateResult:
        if self._model is None or self._processor is None:
            raise MagePredicateRunnerError("Mage predicate session is not loaded")
        images: list[Any] = []
        inputs: dict[str, Any] | None = None
        generated: Any = None
        try:
            for frame in request.frames:
                with self._image_type.open(io.BytesIO(frame.payload)) as source:
                    source.load()
                    if source.format != _MEDIA_TYPES[frame.media_type]:
                        raise MagePredicateRunnerError("predicate image type is invalid")
                    width, height = source.size
                    if (
                        isinstance(width, bool)
                        or isinstance(height, bool)
                        or not 1 <= width <= 16_384
                        or not 1 <= height <= 16_384
                        or width * height > _MAX_IMAGE_PIXELS
                    ):
                        raise MagePredicateRunnerError("predicate image dimensions are invalid")
                    converted = source.convert("RGB")
                    try:
                        converted.thumbnail(
                            _VIDEO_CANVAS_SIZE,
                            self._image_type.Resampling.LANCZOS,
                        )
                        canvas = self._image_type.new(
                            "RGB",
                            _VIDEO_CANVAS_SIZE,
                            _VIDEO_CANVAS_COLOR,
                        )
                        offset = (
                            (_VIDEO_CANVAS_SIZE[0] - converted.width) // 2,
                            (_VIDEO_CANVAS_SIZE[1] - converted.height) // 2,
                        )
                        canvas.paste(converted, offset)
                        images.append(canvas)
                    finally:
                        converted.close()
            system_prompt, user_prompt = build_atomic_prompt(request)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self._processor(
                text=[text],
                videos=[images],
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: (value.to(self._model.device) if hasattr(value, "to") else value)
                for key, value in inputs.items()
            }
            for key in ("pixel_values", "pixel_values_videos"):
                if key in inputs:
                    inputs[key] = inputs[key].to(self._model.dtype)
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.settings.max_new_tokens,
                    do_sample=False,
                )
            raw = self._processor.tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
            return normalize_model_answer(request, raw)
        finally:
            for image in images:
                image.close()
            del generated
            del inputs

    def __exit__(self, *_args: object) -> None:
        self._model = None
        self._processor = None
        self._image_type = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baby-monitor-mage-vl-predicate")
    parser.add_argument("--request", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-manifest", required=True)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--confirm-reviewed-remote-code", action="store_true")
    parser.add_argument("--language-linear-precision", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-gpu-memory-mib", type=int, default=10_240)
    parser.add_argument("--max-cpu-memory-mib", type=int, default=12_288)
    return parser


def _run(args: argparse.Namespace) -> int:
    if args.revision != OFFICIAL_REVISION:
        raise MagePredicateRunnerError("Mage predicate revision has not been reviewed")
    if args.confirm_reviewed_remote_code is not True:
        raise MagePredicateRunnerError(
            "Mage predicate remote code requires explicit one-shot confirmation"
        )
    manifest = TrustedModelCodeManifest.read(args.code_manifest)
    settings = MageVLSettings(
        model_path=Path(args.model_path).expanduser(),
        code_manifest=manifest,
        confirm_reviewed_remote_code=True,
        language_linear_precision=args.language_linear_precision,
        max_new_tokens=args.max_new_tokens,
        max_gpu_memory_mib=args.max_gpu_memory_mib,
        max_cpu_memory_mib=args.max_cpu_memory_mib,
    )
    request = load_predicate_request(args.request)
    try:
        with (
            open(os.devnull, "w", encoding="utf-8") as output_sink,
            redirect_stdout(output_sink),
            redirect_stderr(output_sink),
            isolated_transformers_module_cache(),
            MagePredicateSession(settings) as session,
        ):
            result = session.assess(request)
    except MagePredicateRunnerError:
        raise
    except Exception as exc:
        raise MagePredicateRunnerError("Mage predicate inference failed") from exc
    print(result.to_json())
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(build_parser().parse_args(argv))
    except (MagePredicateRunnerError, OSError, SchemaError, ValueError) as exc:
        report_runner_failure(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
