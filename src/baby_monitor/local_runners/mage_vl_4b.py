from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baby_monitor.domain import SchemaError
from baby_monitor.local_runners.common import (
    JSON_SYSTEM_PROMPT,
    LocalVLMRequest,
    LocalVLMRunnerError,
    build_prompt,
    load_request,
    report_runner_failure,
    validate_result,
)

MODEL_ID = "microsoft/Mage-VL"
OFFICIAL_REVISION = "5c78cab61938e73859b63724d9bf5cb88c477eaa"
_REVISION = re.compile(r"[0-9a-f]{40}")
_SAFE_ROOT_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_AUDITED_SUFFIXES = {".jinja", ".json", ".py"}
_REQUIRED_FILES = {
    "chat_template.jinja",
    "codec_video_processing_mage_vl.py",
    "config.json",
    "configuration_mage_vl.py",
    "modeling_mage_vl.py",
    "preprocessor_config.json",
    "processing_mage_vl.py",
    "video_preprocessor_config.json",
    "video_processing_mage_vl.py",
}
_EXPECTED_AUTO_MAP = {
    "AutoConfig": "configuration_mage_vl.MageVLConfig",
    "AutoModel": "modeling_mage_vl.MageVLModel",
    "AutoModelForCausalLM": "modeling_mage_vl.MageVLForConditionalGeneration",
    "AutoModelForImageTextToText": ("modeling_mage_vl.MageVLForConditionalGeneration"),
    "AutoProcessor": "processing_mage_vl.MageVLProcessor",
    "AutoVideoProcessor": "video_processing_mage_vl.MageVLVideoProcessor",
}
_MAX_AUDITED_FILES = 64
_MAX_AUDITED_FILE_BYTES = 32 * 1024 * 1024
_MAX_AUDITED_TOTAL_BYTES = 96 * 1024 * 1024
_LANGUAGE_LINEAR_PRECISIONS = frozenset({"bf16", "int8"})
_INT8_SKIPPED_MODULES = ("model.visual",)


class MageVLRunnerError(LocalVLMRunnerError):
    pass


@contextmanager
def isolated_transformers_module_cache() -> Iterator[Path]:
    if "transformers" in sys.modules:
        raise MageVLRunnerError("Transformers was loaded before Mage cache isolation")
    previous = os.environ.get("HF_MODULES_CACHE")
    with tempfile.TemporaryDirectory(prefix="baby-monitor-mage-modules-") as directory:
        cache = Path(directory).resolve(strict=True)
        os.environ["HF_MODULES_CACHE"] = str(cache)
        try:
            yield cache
        finally:
            if previous is None:
                os.environ.pop("HF_MODULES_CACHE", None)
            else:
                os.environ["HF_MODULES_CACHE"] = previous


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise MageVLRunnerError(f"{label} fields are invalid")


def _model_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        raise MageVLRunnerError("model path must be absolute")
    if not root.is_dir() or root.is_symlink():
        raise MageVLRunnerError("pinned local model directory is missing or unsafe")
    try:
        return root.resolve(strict=True)
    except OSError as exc:
        raise MageVLRunnerError("pinned local model directory is missing or unsafe") from exc


@dataclass(frozen=True, slots=True)
class TrustedCodeFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not isinstance(self.sha256, str):
            raise MageVLRunnerError("trusted code manifest file fields are invalid")
        suffix = Path(self.path).suffix.lower()
        if (
            _SAFE_ROOT_FILE.fullmatch(self.path) is None
            or "/" in self.path
            or "\\" in self.path
            or suffix not in _AUDITED_SUFFIXES
        ):
            raise MageVLRunnerError("trusted code manifest contains an unsafe file")
        if type(self.size) is not int or not 0 <= self.size <= _MAX_AUDITED_FILE_BYTES:
            raise MageVLRunnerError("trusted code manifest file size is invalid")
        if _SHA256.fullmatch(self.sha256) is None:
            raise MageVLRunnerError("trusted code manifest hash is invalid")

    @classmethod
    def from_dict(cls, value: object) -> TrustedCodeFile:
        if not isinstance(value, dict):
            raise MageVLRunnerError("trusted code manifest file must be an object")
        _exact_keys(value, {"path", "size", "sha256"}, "trusted code manifest file")
        return cls(path=value["path"], size=value["size"], sha256=value["sha256"])

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class TrustedModelCodeManifest:
    revision: str
    files: tuple[TrustedCodeFile, ...]
    model_id: str = MODEL_ID
    version: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.version) is not int
            or not isinstance(self.model_id, str)
            or not isinstance(self.revision, str)
            or not isinstance(self.files, tuple)
            or any(not isinstance(item, TrustedCodeFile) for item in self.files)
        ):
            raise MageVLRunnerError("trusted code manifest fields are invalid")
        if self.version != 1 or self.model_id != MODEL_ID:
            raise MageVLRunnerError("trusted code manifest identity is invalid")
        if self.revision != OFFICIAL_REVISION:
            raise MageVLRunnerError("trusted code manifest revision is invalid")
        if not 1 <= len(self.files) <= _MAX_AUDITED_FILES:
            raise MageVLRunnerError("trusted code manifest file count is invalid")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise MageVLRunnerError("trusted code manifest files must be unique and sorted")
        if not _REQUIRED_FILES.issubset(paths):
            raise MageVLRunnerError("trusted code manifest is missing required runtime files")
        if sum(item.size for item in self.files) > _MAX_AUDITED_TOTAL_BYTES:
            raise MageVLRunnerError("trusted code manifest is too large")

    @classmethod
    def from_dict(cls, value: object) -> TrustedModelCodeManifest:
        if not isinstance(value, dict):
            raise MageVLRunnerError("trusted code manifest must be an object")
        _exact_keys(value, {"version", "model_id", "revision", "files"}, "manifest")
        if not isinstance(value["files"], list):
            raise MageVLRunnerError("trusted code manifest files must be an array")
        return cls(
            version=value["version"],
            model_id=value["model_id"],
            revision=value["revision"],
            files=tuple(TrustedCodeFile.from_dict(item) for item in value["files"]),
        )

    @classmethod
    def read(cls, path: str | Path) -> TrustedModelCodeManifest:
        manifest_path = Path(path).expanduser()
        try:
            metadata = manifest_path.stat(follow_symlinks=False)
        except OSError:
            raise MageVLRunnerError("trusted code manifest is missing or unsafe") from None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 256 * 1024:
            raise MageVLRunnerError("trusted code manifest is missing or unsafe")
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise MageVLRunnerError("trusted code manifest is invalid") from None
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "model_id": self.model_id,
            "revision": self.revision,
            "files": [item.to_dict() for item in self.files],
        }

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.files)

    def verify(self, model_path: str | Path) -> None:
        observed = audit_model_code(model_path, revision=self.revision)
        if observed != self:
            raise MageVLRunnerError("trusted model code changed after review")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise MageVLRunnerError(f"{label} is invalid") from None
    if not isinstance(value, dict):
        raise MageVLRunnerError(f"{label} must be an object")
    return value


def _verify_official_runtime_shape(root: Path) -> None:
    config = _read_json_object(root / "config.json", label="Mage model config")
    if (
        config.get("model_type") != "mage_vl"
        or config.get("architectures") != ["MageVLForConditionalGeneration"]
        or config.get("auto_map") != _EXPECTED_AUTO_MAP
    ):
        raise MageVLRunnerError("Mage model config does not match the reviewed API")
    preprocessor = _read_json_object(
        root / "preprocessor_config.json",
        label="Mage preprocessor config",
    )
    video_preprocessor = _read_json_object(
        root / "video_preprocessor_config.json",
        label="Mage video preprocessor config",
    )
    expected_processor_map = {
        "AutoProcessor": "processing_mage_vl.MageVLProcessor",
        "AutoVideoProcessor": "video_processing_mage_vl.MageVLVideoProcessor",
    }
    if (
        preprocessor.get("auto_map") != expected_processor_map
        or video_preprocessor.get("auto_map") != expected_processor_map
        or preprocessor.get("processor_class") != "MageVLProcessor"
        or video_preprocessor.get("processor_class") != "MageVLProcessor"
    ):
        raise MageVLRunnerError("Mage processor config does not match the reviewed API")


def audit_model_code(
    model_path: str | Path,
    *,
    revision: str = OFFICIAL_REVISION,
) -> TrustedModelCodeManifest:
    root = _model_root(model_path)
    if _REVISION.fullmatch(revision) is None:
        raise MageVLRunnerError("Mage model revision must be a full lowercase commit hash")
    if revision != OFFICIAL_REVISION:
        raise MageVLRunnerError("Mage model revision has not been reviewed by this release")

    files: list[TrustedCodeFile] = []
    total_bytes = 0
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.suffix.lower() not in _AUDITED_SUFFIXES:
            continue
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError:
            raise MageVLRunnerError("Mage runtime control file is missing or unsafe") from None
        if not stat.S_ISREG(metadata.st_mode):
            raise MageVLRunnerError("Mage runtime control file is missing or unsafe")
        if _SAFE_ROOT_FILE.fullmatch(path.name) is None:
            raise MageVLRunnerError("Mage runtime control filename is unsafe")
        size = metadata.st_size
        if size > _MAX_AUDITED_FILE_BYTES:
            raise MageVLRunnerError("Mage runtime control file is too large")
        if len(files) >= _MAX_AUDITED_FILES:
            raise MageVLRunnerError("Mage runtime control file count is invalid")
        total_bytes += size
        if total_bytes > _MAX_AUDITED_TOTAL_BYTES:
            raise MageVLRunnerError("Mage runtime control files are too large")
        try:
            payload = path.read_bytes()
        except OSError:
            raise MageVLRunnerError("Mage runtime control file is unreadable") from None
        if len(payload) != size:
            raise MageVLRunnerError("Mage runtime control file changed during audit")
        files.append(
            TrustedCodeFile(
                path=path.name,
                size=size,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    manifest = TrustedModelCodeManifest(revision=revision, files=tuple(files))
    _verify_official_runtime_shape(root)
    return manifest


def write_code_manifest(
    model_path: str | Path,
    output_path: str | Path,
    *,
    revision: str = OFFICIAL_REVISION,
) -> TrustedModelCodeManifest:
    manifest = audit_model_code(model_path, revision=revision)
    output = Path(output_path).expanduser()
    if not output.is_absolute():
        raise MageVLRunnerError("trusted code manifest output must be absolute")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise MageVLRunnerError("trusted code manifest parent is missing or unsafe")
    payload = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n"
    created = False
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        output.chmod(0o600)
    except Exception:
        if created:
            with suppress(FileNotFoundError):
                output.unlink()
        raise
    return manifest


@dataclass(frozen=True, slots=True)
class MageVLSettings:
    model_path: Path
    code_manifest: TrustedModelCodeManifest
    confirm_reviewed_remote_code: bool
    language_linear_precision: str = "bf16"
    max_new_tokens: int = 512
    max_gpu_memory_mib: int = 10_240
    max_cpu_memory_mib: int = 12_288

    def __post_init__(self) -> None:
        root = _model_root(self.model_path)
        if self.code_manifest.revision != OFFICIAL_REVISION:
            raise MageVLRunnerError("Mage model revision has not been reviewed")
        if self.confirm_reviewed_remote_code is not True:
            raise MageVLRunnerError("Mage remote code requires explicit one-shot confirmation")
        if (
            not isinstance(self.language_linear_precision, str)
            or self.language_linear_precision not in _LANGUAGE_LINEAR_PRECISIONS
        ):
            raise MageVLRunnerError("Mage language linear precision is invalid")
        if type(self.max_new_tokens) is not int or not 64 <= self.max_new_tokens <= 1024:
            raise MageVLRunnerError("max_new_tokens must be between 64 and 1024")
        if (
            type(self.max_gpu_memory_mib) is not int
            or not 6_144 <= self.max_gpu_memory_mib <= 10_752
        ):
            raise MageVLRunnerError("Mage GPU memory budget is invalid")
        if (
            type(self.max_cpu_memory_mib) is not int
            or not 4_096 <= self.max_cpu_memory_mib <= 16_384
        ):
            raise MageVLRunnerError("Mage CPU memory budget is invalid")
        self.code_manifest.verify(root)


def _build_quantization_config(factory: Any, precision: str) -> Any | None:
    if precision == "bf16":
        return None
    if precision != "int8":
        raise MageVLRunnerError("Mage language linear precision is invalid")
    return factory(
        load_in_8bit=True,
        llm_int8_skip_modules=list(_INT8_SKIPPED_MODULES),
    )


def _verify_int8_layout(model: Any, int8_linear_type: type[Any]) -> None:
    visual_count = sum(isinstance(module, int8_linear_type) for module in model.visual.modules())
    language_count = sum(
        isinstance(module, int8_linear_type) for module in model.language_model.modules()
    )
    if visual_count != 0 or language_count == 0:
        raise MageVLRunnerError("Mage selective INT8 layout verification failed")


def generate(settings: MageVLSettings, request: LocalVLMRequest) -> str:
    with isolated_transformers_module_cache():
        return _generate_with_isolated_cache(settings, request)


def _generate_with_isolated_cache(
    settings: MageVLSettings,
    request: LocalVLMRequest,
) -> str:
    # This runner is a one-event subprocess. Override inherited values before any
    # third-party import so a permissive parent environment cannot weaken the boundary.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        import torch
        from PIL import Image
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            BitsAndBytesConfig,
        )
    except ImportError as exc:
        raise MageVLRunnerError("Mage-VL runtime dependencies are not installed") from exc
    if not torch.cuda.is_available():
        raise MageVLRunnerError("CUDA is not available to the Mage-VL runner")

    images = []
    model = None
    processor = None
    inputs = None
    generated = None
    try:
        for path in request.frame_paths:
            with Image.open(path) as source:
                source.load()
                images.append(source.convert("RGB"))

        processor = AutoProcessor.from_pretrained(
            str(settings.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        quantization_config = _build_quantization_config(
            BitsAndBytesConfig,
            settings.language_linear_precision,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "max_memory": {
                0: f"{settings.max_gpu_memory_mib}MiB",
                "cpu": f"{settings.max_cpu_memory_mib}MiB",
            },
            "attn_implementation": "sdpa",
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        model = AutoModelForCausalLM.from_pretrained(
            str(settings.model_path),
            **model_kwargs,
        ).eval()
        if settings.language_linear_precision == "int8":
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise MageVLRunnerError("Mage INT8 runtime dependency is not installed") from exc
            _verify_int8_layout(model, bnb.nn.Linear8bitLt)
        messages = [
            {"role": "system", "content": JSON_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": build_prompt(request)},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(
            text=[text],
            videos=[images],
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: (value.to(model.device) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }
        for key in ("pixel_values", "pixel_values_videos"):
            if key in inputs:
                inputs[key] = inputs[key].to(model.dtype)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
            )
        return processor.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
    finally:
        for image in images:
            image.close()
        del generated
        del inputs
        del model
        del processor
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baby-monitor-mage-vl-4b")
    commands = parser.add_subparsers(dest="action", required=True)

    audit = commands.add_parser(
        "audit-code",
        help="hash a pinned local Mage runtime without executing third-party code",
    )
    audit.add_argument("--model-path", required=True)
    audit.add_argument("--revision", default=OFFICIAL_REVISION)
    audit.add_argument("--output", required=True)

    run = commands.add_parser(
        "run",
        help="run one offline Mage event after explicit remote-code review",
    )
    run.add_argument("--request", required=True)
    run.add_argument("--model-path", default=os.environ.get("BABY_MONITOR_MAGE_MODEL_PATH"))
    run.add_argument(
        "--code-manifest",
        default=os.environ.get("BABY_MONITOR_MAGE_CODE_MANIFEST"),
    )
    run.add_argument("--revision", default=OFFICIAL_REVISION)
    run.add_argument("--confirm-reviewed-remote-code", action="store_true")
    run.add_argument("--max-new-tokens", type=int, default=512)
    run.add_argument(
        "--language-linear-precision",
        choices=sorted(_LANGUAGE_LINEAR_PRECISIONS),
        default="bf16",
    )
    run.add_argument("--max-gpu-memory-mib", type=int, default=10_240)
    run.add_argument("--max-cpu-memory-mib", type=int, default=12_288)
    return parser


def _audit_command(args: argparse.Namespace) -> int:
    manifest = write_code_manifest(
        args.model_path,
        args.output,
        revision=args.revision,
    )
    print(
        json.dumps(
            {
                "version": 1,
                "status": "created",
                "model_id": manifest.model_id,
                "revision": manifest.revision,
                "file_count": len(manifest.files),
                "total_bytes": manifest.total_bytes,
                "review_required_before_run": True,
                "remote_code_executed": False,
                "network_used": False,
                "path_disclosed": False,
                "hashes_disclosed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_command(args: argparse.Namespace) -> int:
    if not args.model_path or not args.code_manifest:
        raise MageVLRunnerError("pinned model and trusted code manifest are required")
    if args.revision != OFFICIAL_REVISION:
        raise MageVLRunnerError("Mage model revision has not been reviewed")
    if args.confirm_reviewed_remote_code is not True:
        raise MageVLRunnerError("Mage remote code requires explicit one-shot confirmation")
    request = load_request(args.request)
    manifest = TrustedModelCodeManifest.read(args.code_manifest)
    settings = MageVLSettings(
        model_path=Path(args.model_path).expanduser(),
        code_manifest=manifest,
        confirm_reviewed_remote_code=args.confirm_reviewed_remote_code,
        language_linear_precision=args.language_linear_precision,
        max_new_tokens=args.max_new_tokens,
        max_gpu_memory_mib=args.max_gpu_memory_mib,
        max_cpu_memory_mib=args.max_cpu_memory_mib,
    )
    try:
        # Reviewed model code may print progress or diagnostics. Keep stdout/stderr
        # reserved for the single-object command protocol without retaining them.
        with (
            open(os.devnull, "w", encoding="utf-8") as output_sink,
            redirect_stdout(output_sink),
            redirect_stderr(output_sink),
        ):
            raw = generate(settings, request)
    except MageVLRunnerError:
        raise
    except Exception as exc:
        raise MageVLRunnerError("Mage inference failed") from exc
    result = validate_result(raw, allowed_frame_ids=request.frame_ids)
    if not result.summary_zh.strip():
        raise SchemaError("Mage result summary must be non-empty")
    print(result.to_json(indent=None))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _audit_command(args) if args.action == "audit-code" else _run_command(args)
    except (
        MageVLRunnerError,
        OSError,
        SchemaError,
        ValueError,
    ) as exc:
        report_runner_failure(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
