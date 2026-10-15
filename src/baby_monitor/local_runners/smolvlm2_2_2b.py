from __future__ import annotations

import argparse
import gc
import json
import os
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from baby_monitor.domain import SchemaError
from baby_monitor.local_runners.common import (
    LocalVLMRequest,
    LocalVLMRunnerError,
    load_request,
    report_runner_failure,
    validate_result,
)

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"
PROMPT_VERSION = "vlm_event_smolvlm2_v3_20260912_coverage_unverified"
RUNTIME_ID = "py312_torch2.11_tf5.7_bf16_sdpa_num2words0.5.14_v1"


@dataclass(frozen=True, slots=True)
class SmolVLM2Settings:
    model_path: Path
    max_new_tokens: int = 768

    def __post_init__(self) -> None:
        if not self.model_path.is_absolute():
            raise ValueError("model path must be absolute")
        if self.model_path.is_symlink() or not self.model_path.is_dir():
            raise FileNotFoundError("pinned local model directory does not exist")
        required = (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        )
        if any(not (self.model_path / name).is_file() for name in required):
            raise FileNotFoundError("pinned local model snapshot is incomplete")
        if type(self.max_new_tokens) is not int or not 64 <= self.max_new_tokens <= 1536:
            raise ValueError("max_new_tokens must be between 64 and 1536")


def build_prompt(request: LocalVLMRequest) -> str:
    template = {
        "event_type": "unknown",
        "subjects": [],
        "observations": [],
        "inferences": [],
        "risk_level": "none",
        "confidence": 0.0,
        "needs_remote_review": True,
        "summary_zh": "证据不足，无法确定家庭事件。",
        "evidence_frames": [request.frame_ids[0]],
    }
    detections = request.candidate.get("detections", [])
    signals = request.candidate.get("signals", [])
    coverage = {
        "coverage_status": request.evidence.coverage_status,
        "continuous_coverage_verified": False,
        "coverage_gap_reasons": [gap.reason for gap in request.evidence.coverage_gaps],
        "selected_segments": request.evidence.selected_segments,
    }
    return "\n".join(
        (
            "Analyze the local event frames in chronological order.",
            request.question,
            "Return exactly one JSON object and no Markdown or other text.",
            "Start from this exact field structure and replace only its values:",
            json.dumps(template, ensure_ascii=False, separators=(",", ":")),
            "The template values are placeholders, not visual facts.",
            "event_type must be sleeping, awake, crying, movement, "
            "adult_interaction, possible_risk, or unknown.",
            "risk_level must be none, low, medium, or high. confidence must be "
            "between 0 and 1. needs_remote_review must be true or false.",
            "subjects is an array. Every subject may contain only kind, confidence, "
            "and evidence_frames. kind must be baby, adult, person, pet, or unknown.",
            "observations, inferences, and evidence_frames must be native JSON arrays.",
            "Write only directly visible facts or supplied detector facts in observations.",
            "Put uncertain identity, sleep, waking, crying, movement, leaving a bed, "
            "being picked up, and risk interpretations in inferences.",
            "A silent image cannot prove crying. A single or repeated frame cannot prove "
            "motion. Generic people must not be promoted to baby or adult.",
            "If evidence is insufficient, use event_type unknown, low confidence, and "
            "needs_remote_review true. Do not invent missing events.",
            "summary_zh must be a non-empty short Chinese summary of visible content and "
            "uncertainty.",
            "Every root and subject evidence_frames entry must use only these frame IDs: "
            + json.dumps(request.frame_ids, ensure_ascii=False),
            "Detector facts: " + json.dumps(detections, ensure_ascii=False),
            "Signal facts: " + json.dumps(signals, ensure_ascii=False),
            "Coverage metadata: " + json.dumps(coverage, ensure_ascii=False),
            "Coverage unverified means continuous coverage was not verified; unknown does not "
            "mean recorded missing footage or continuous coverage. gaps_recorded means known "
            "gaps exist. No recorded gaps does not prove no gaps. Do not invent missing duration "
            "or events in gaps.",
        )
    )


def generate(settings: SmolVLM2Settings, request: LocalVLMRequest) -> str:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch
        from PIL import Image
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )
    except ImportError as exc:
        raise LocalVLMRunnerError("SmolVLM2 runtime dependencies are not installed") from exc
    if not torch.cuda.is_available():
        raise LocalVLMRunnerError("CUDA is not available to the SmolVLM2 runner")

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
            trust_remote_code=False,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(settings.model_path),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).eval()
        model.to("cuda")

        content: list[dict[str, object]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": build_prompt(request)})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
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
    parser = argparse.ArgumentParser(prog="baby-monitor-smolvlm2-2.2b")
    parser.add_argument("--request", required=True)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("BABY_MONITOR_SMOLVLM2_MODEL_PATH"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.model_path:
            raise ValueError("a pinned local model path is required")
        request = load_request(args.request)
        settings = SmolVLM2Settings(
            model_path=Path(args.model_path).expanduser(),
            max_new_tokens=args.max_new_tokens,
        )
        try:
            with (
                open(os.devnull, "w", encoding="utf-8") as output_sink,
                redirect_stdout(output_sink),
            ):
                raw = generate(settings, request)
        except LocalVLMRunnerError:
            raise
        except Exception as exc:
            raise LocalVLMRunnerError("SmolVLM2 inference failed") from exc
        validated = validate_result(raw, allowed_frame_ids=request.frame_ids)
        if not validated.summary_zh.strip():
            raise SchemaError("SmolVLM2 result summary must be non-empty")
    except (FileNotFoundError, LocalVLMRunnerError, SchemaError, ValueError) as exc:
        report_runner_failure(exc)
        return 2
    print(validated.to_json(indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
