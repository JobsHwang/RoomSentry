from __future__ import annotations

import argparse
import gc
import importlib
import os
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from baby_monitor.domain import SchemaError
from baby_monitor.local_runners.common import (
    JSON_SYSTEM_PROMPT,
    load_request,
    report_runner_failure,
    validate_result,
)
from baby_monitor.local_runners.common import (
    LocalVLMRequest as MiniCPMRequest,
)
from baby_monitor.local_runners.common import (
    LocalVLMRunnerError as MiniCPMRunnerError,
)
from baby_monitor.local_runners.common import (
    build_constrained_schema as build_constrained_schema,
)
from baby_monitor.local_runners.common import (
    build_prompt as build_shared_prompt,
)

MINICPM_PROMPT_VERSION = "vlm_event_minicpm_v4_20261006_unicode_constraints"


def build_prompt(request: MiniCPMRequest) -> str:
    return build_shared_prompt(request, include_default_template=True)


@dataclass(frozen=True, slots=True)
class MiniCPMSettings:
    model_path: Path
    max_new_tokens: int = 512
    downsample_mode: str = "16x"

    def __post_init__(self) -> None:
        if not self.model_path.is_absolute():
            raise ValueError("model path must be absolute")
        if not self.model_path.is_dir():
            raise FileNotFoundError("pinned local model directory does not exist")
        if type(self.max_new_tokens) is not int or not 64 <= self.max_new_tokens <= 2048:
            raise ValueError("max_new_tokens must be between 64 and 2048")
        if not isinstance(self.downsample_mode, str) or self.downsample_mode not in {
            "4x",
            "16x",
        }:
            raise ValueError("downsample_mode must be 4x or 16x")


def _build_json_prefix_fn(
    tokenizer,
    output_schema,
    *,
    tokenizer_data_builder,
    config_factory,
    parser_factory,
    prefix_builder,
):
    """Freeze the vocabulary alphabet before JSON parser context initialization.

    LMFE 0.11.3 stores alphabet_without_quotes in the parser's context during
    construction. Updating parser.config later does not update that context;
    non-ASCII free text can then discard the parse stack and allow early EOS.
    Dependency factories keep this assembly testable without model extras.
    """
    tokenizer_data = tokenizer_data_builder(tokenizer)
    config = config_factory(alphabet=tokenizer_data.tokenizer_alphabet)
    parser = parser_factory(output_schema, config=config)
    return prefix_builder(tokenizer_data, parser)


def generate(settings: MiniCPMSettings, request: MiniCPMRequest) -> str:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        from PIL import Image
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            PreTrainedTokenizerBase,
        )
    except ImportError as exc:
        raise MiniCPMRunnerError("MiniCPM runtime dependencies are not installed") from exc
    tokenizer_utils = importlib.import_module("transformers.tokenization_utils")
    tokenizer_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    try:
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
            build_transformers_prefix_allowed_tokens_fn,
        )
    except ImportError as exc:
        raise MiniCPMRunnerError("structured generation integration is unavailable") from exc
    if not torch.cuda.is_available():
        raise MiniCPMRunnerError("CUDA is not available to the MiniCPM runner")

    images = []
    model = None
    processor = None
    try:
        for path in request.frame_paths:
            with Image.open(path) as source:
                source.load()
                images.append(source.convert("RGB"))

        processor = AutoProcessor.from_pretrained(
            str(settings.model_path),
            local_files_only=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(settings.model_path),
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()
        model.to("cuda")

        content: list[dict[str, object]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": build_prompt(request)})
        messages = [
            {"role": "system", "content": JSON_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            processor_kwargs={
                "downsample_mode": settings.downsample_mode,
                "max_slice_nums": 1,
            },
        ).to(model.device)
        prefix_allowed_tokens_fn = _build_json_prefix_fn(
            processor.tokenizer,
            build_constrained_schema(request),
            tokenizer_data_builder=build_token_enforcer_tokenizer_data,
            config_factory=CharacterLevelParserConfig,
            parser_factory=JsonSchemaParser,
            prefix_builder=build_transformers_prefix_allowed_tokens_fn,
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                downsample_mode=settings.downsample_mode,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        trimmed = [
            output[len(source) :]
            for source, output in zip(inputs.input_ids, generated, strict=True)
        ]
        return processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
    finally:
        for image in images:
            image.close()
        del model
        del processor
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baby-monitor-minicpm-v46")
    parser.add_argument("--request", required=True)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("BABY_MONITOR_MINICPM_MODEL_PATH"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--downsample-mode", choices=["4x", "16x"], default="16x")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.model_path:
            raise ValueError("a pinned local model path is required")
        request = load_request(args.request)
        settings = MiniCPMSettings(
            model_path=Path(args.model_path).expanduser(),
            max_new_tokens=args.max_new_tokens,
            downsample_mode=args.downsample_mode,
        )
        try:
            with (
                open(os.devnull, "w", encoding="utf-8") as output_sink,
                redirect_stdout(output_sink),
            ):
                raw = generate(settings, request)
        except MiniCPMRunnerError:
            raise
        except Exception as exc:
            raise MiniCPMRunnerError("MiniCPM inference failed") from exc
        validated = validate_result(raw, allowed_frame_ids=request.frame_ids)
        if not validated.summary_zh.strip():
            raise SchemaError("MiniCPM result summary must be non-empty")
    except (FileNotFoundError, MiniCPMRunnerError, SchemaError, ValueError) as exc:
        report_runner_failure(exc)
        return 2
    print(validated.to_json(indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
