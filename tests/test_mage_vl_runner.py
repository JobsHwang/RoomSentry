from __future__ import annotations

import copy
import json
import os
import stat
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor.domain import EventMedia, VLMEvent
from baby_monitor.local_runners.common import load_request as common_load_request
from baby_monitor.local_runners.mage_vl_4b import (
    MODEL_ID,
    OFFICIAL_REVISION,
    MageVLRunnerError,
    MageVLSettings,
    TrustedModelCodeManifest,
    _build_quantization_config,
    _verify_int8_layout,
    audit_model_code,
    isolated_transformers_module_cache,
    load_request,
    main,
)

EXPECTED_AUTO_MAP = {
    "AutoConfig": "configuration_mage_vl.MageVLConfig",
    "AutoModel": "modeling_mage_vl.MageVLModel",
    "AutoModelForCausalLM": "modeling_mage_vl.MageVLForConditionalGeneration",
    "AutoModelForImageTextToText": "modeling_mage_vl.MageVLForConditionalGeneration",
    "AutoProcessor": "processing_mage_vl.MageVLProcessor",
    "AutoVideoProcessor": "video_processing_mage_vl.MageVLVideoProcessor",
}
REQUIRED_CONTROL_FILES = {
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


def _write_model_root(root: Path) -> Path:
    model_root = root / "model"
    model_root.mkdir(parents=True)
    for name in REQUIRED_CONTROL_FILES:
        if name.endswith(".py"):
            (model_root / name).write_text(f"# synthetic {name}\n", encoding="utf-8")
        elif name.endswith(".jinja"):
            (model_root / name).write_text("{{ messages }}\n", encoding="utf-8")
    (model_root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mage_vl",
                "architectures": ["MageVLForConditionalGeneration"],
                "auto_map": EXPECTED_AUTO_MAP,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    processor_config = {
        "auto_map": {
            "AutoProcessor": "processing_mage_vl.MageVLProcessor",
            "AutoVideoProcessor": "video_processing_mage_vl.MageVLVideoProcessor",
        },
        "processor_class": "MageVLProcessor",
    }
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        (model_root / name).write_text(
            json.dumps(processor_config, sort_keys=True),
            encoding="utf-8",
        )
    return model_root


def _write_request(root: Path) -> Path:
    frame = root / "frame_01.jpg"
    frame.write_bytes(b"synthetic-frame-placeholder")
    request = root / "request.json"
    request.write_text(
        json.dumps(
            {
                "candidate": {"detections": [], "signals": []},
                "evidence": EventMedia(
                    clip_path=str(root / "clip.mp4"),
                    frame_paths=(str(frame),),
                    selected_segments=1,
                ).to_dict(),
                "frame_paths": [str(frame)],
                "question": "证据不足时输出 unknown。",
                "output_schema": VLMEvent.json_schema(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return request


def _valid_result(frame_id: str) -> str:
    return json.dumps(
        {
            "event_type": "unknown",
            "subjects": [],
            "observations": ["画面中可见纯色测试图。"],
            "inferences": [],
            "risk_level": "none",
            "confidence": 0.1,
            "needs_remote_review": True,
            "summary_zh": "测试图不足以判断宝宝事件。",
            "evidence_frames": [frame_id],
        },
        ensure_ascii=False,
    )


def _write_manifest(root: Path, model_root: Path) -> Path:
    manifest = audit_model_code(model_root)
    path = root / "mage-code-manifest.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path


class MageVLRunnerTests(unittest.TestCase):
    def test_uses_shared_request_contract(self) -> None:
        self.assertIs(common_load_request, load_request)

    def test_transformers_remote_code_cache_is_ephemeral_and_restores_environment(self) -> None:
        original = os.environ.get("HF_MODULES_CACHE")
        sentinel = str(Path(tempfile.gettempdir()) / "existing-private-cache")
        os.environ["HF_MODULES_CACHE"] = sentinel
        try:
            with isolated_transformers_module_cache() as cache:
                self.assertTrue(cache.is_dir())
                self.assertNotEqual(sentinel, str(cache))
                self.assertEqual(str(cache), os.environ["HF_MODULES_CACHE"])
            self.assertFalse(cache.exists())
            self.assertEqual(sentinel, os.environ["HF_MODULES_CACHE"])
        finally:
            if original is None:
                os.environ.pop("HF_MODULES_CACHE", None)
            else:
                os.environ["HF_MODULES_CACHE"] = original

    def test_audit_is_deterministic_and_covers_runtime_control_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            first = audit_model_code(model_root)
            second = audit_model_code(model_root)
            self.assertEqual(first, second)
            self.assertEqual(MODEL_ID, first.model_id)
            self.assertEqual(OFFICIAL_REVISION, first.revision)
            paths = tuple(item.path for item in first.files)
            self.assertEqual(tuple(sorted(paths)), paths)
            self.assertTrue(REQUIRED_CONTROL_FILES.issubset(paths))
            self.assertNotIn(str(root), json.dumps(first.to_dict()))

    def test_audit_cli_creates_private_path_free_manifest_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            manifest_path = root / "private-manifest.json"
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "audit-code",
                        "--model-path",
                        str(model_root),
                        "--output",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            summary = json.loads(stdout.getvalue())
            self.assertEqual("created", summary["status"])
            self.assertFalse(summary["remote_code_executed"])
            self.assertFalse(summary["network_used"])
            self.assertFalse(summary["path_disclosed"])
            self.assertFalse(summary["hashes_disclosed"])
            self.assertNotIn(str(root), stdout.getvalue())
            manifest = TrustedModelCodeManifest.read(manifest_path)
            for item in manifest.files:
                self.assertNotIn(item.sha256, stdout.getvalue())
            if os.name != "nt":
                self.assertEqual(0o600, stat.S_IMODE(manifest_path.stat().st_mode))

    def test_audit_cli_never_overwrites_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            manifest_path = root / "private-manifest.json"
            manifest_path.write_text("keep-existing", encoding="utf-8")
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "audit-code",
                        "--model-path",
                        str(model_root),
                        "--output",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertEqual("keep-existing", manifest_path.read_text(encoding="utf-8"))
            self.assertNotIn(str(manifest_path), stderr.getvalue())

    def test_manifest_detects_tampering_and_added_control_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            manifest = audit_model_code(model_root)
            (model_root / "modeling_mage_vl.py").write_text("# changed\n", encoding="utf-8")
            with self.assertRaisesRegex(MageVLRunnerError, "changed after review"):
                manifest.verify(model_root)

            model_root = _write_model_root(root / "second")
            manifest = audit_model_code(model_root)
            (model_root / "unexpected_runtime.py").write_text("raise SystemExit\n")
            with self.assertRaisesRegex(MageVLRunnerError, "changed after review"):
                manifest.verify(model_root)

    def test_audit_rejects_symlinked_control_code_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            control = model_root / "modeling_mage_vl.py"
            target = root / "outside.py"
            target.write_text("# outside\n", encoding="utf-8")
            control.unlink()
            try:
                control.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {type(exc).__name__}")
            with self.assertRaisesRegex(MageVLRunnerError, "missing or unsafe"):
                audit_model_code(model_root)

    def test_audit_rejects_unreviewed_revision_and_runtime_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            with self.assertRaisesRegex(MageVLRunnerError, "has not been reviewed"):
                audit_model_code(model_root, revision="0" * 40)
            config_path = model_root / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["auto_map"]["AutoProcessor"] = "unexpected.Module"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(MageVLRunnerError, "reviewed API"):
                audit_model_code(model_root)

    def test_manifest_parser_rejects_extra_fields_wrong_types_and_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = audit_model_code(_write_model_root(Path(directory))).to_dict()
            variants = []
            extra = copy.deepcopy(manifest)
            extra["unexpected"] = True
            variants.append(extra)
            wrong_version = copy.deepcopy(manifest)
            wrong_version["version"] = True
            variants.append(wrong_version)
            wrong_revision = copy.deepcopy(manifest)
            wrong_revision["revision"] = 123
            variants.append(wrong_revision)
            wrong_path = copy.deepcopy(manifest)
            wrong_path["files"][0]["path"] = None
            variants.append(wrong_path)
            reversed_files = copy.deepcopy(manifest)
            reversed_files["files"].reverse()
            variants.append(reversed_files)
            for value in variants:
                with self.subTest(value=value), self.assertRaises(MageVLRunnerError):
                    TrustedModelCodeManifest.from_dict(value)

    def test_settings_reject_boolean_or_fractional_resource_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_root = _write_model_root(Path(directory))
            manifest = audit_model_code(model_root)
            variants = (
                {"max_new_tokens": 512.5},
                {"max_gpu_memory_mib": True},
                {"max_cpu_memory_mib": 12_288.5},
            )
            for values in variants:
                with self.subTest(values=values), self.assertRaises(MageVLRunnerError):
                    MageVLSettings(  # type: ignore[arg-type]
                        model_path=model_root,
                        code_manifest=manifest,
                        confirm_reviewed_remote_code=True,
                        **values,
                    )

    def test_settings_rejects_unknown_language_linear_precision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_root = _write_model_root(Path(directory))
            manifest = audit_model_code(model_root)
            with self.assertRaisesRegex(MageVLRunnerError, "precision is invalid"):
                MageVLSettings(
                    model_path=model_root,
                    code_manifest=manifest,
                    confirm_reviewed_remote_code=True,
                    language_linear_precision="fp8",
                )

    def test_int8_config_excludes_visual_tree_and_bf16_has_no_config(self) -> None:
        captured: dict[str, object] = {}

        def factory(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return kwargs

        self.assertIsNone(_build_quantization_config(factory, "bf16"))
        config = _build_quantization_config(factory, "int8")
        self.assertEqual(config, captured)
        self.assertTrue(captured["load_in_8bit"])
        self.assertEqual(["model.visual"], captured["llm_int8_skip_modules"])

    def test_int8_layout_requires_language_only_quantized_linears(self) -> None:
        class QuantizedLinear:
            pass

        class ModuleTree:
            def __init__(self, modules: list[object]) -> None:
                self._modules = modules

            def modules(self):
                return iter(self._modules)

        valid = SimpleNamespace(
            visual=ModuleTree([object()]),
            language_model=ModuleTree([object(), QuantizedLinear()]),
        )
        _verify_int8_layout(valid, QuantizedLinear)

        visual_quantized = SimpleNamespace(
            visual=ModuleTree([QuantizedLinear()]),
            language_model=ModuleTree([QuantizedLinear()]),
        )
        with self.assertRaisesRegex(MageVLRunnerError, "layout verification"):
            _verify_int8_layout(visual_quantized, QuantizedLinear)

        language_unquantized = SimpleNamespace(
            visual=ModuleTree([object()]),
            language_model=ModuleTree([object()]),
        )
        with self.assertRaisesRegex(MageVLRunnerError, "layout verification"):
            _verify_int8_layout(language_unquantized, QuantizedLinear)

    def test_run_refuses_without_remote_code_confirmation_before_request_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            manifest_path = _write_manifest(root, model_root)
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch("baby_monitor.local_runners.mage_vl_4b.generate") as generate,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "run",
                        "--request",
                        str(root / "missing-request.json"),
                        "--model-path",
                        str(model_root),
                        "--code-manifest",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertEqual("runner_failed", json.loads(stderr.getvalue())["error_code"])
            generate.assert_not_called()

    def test_run_prints_only_validated_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            manifest_path = _write_manifest(root, model_root)
            request_path = _write_request(root)
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.mage_vl_4b.generate",
                    return_value=_valid_result("frame_01.jpg"),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "run",
                        "--request",
                        str(request_path),
                        "--model-path",
                        str(model_root),
                        "--code-manifest",
                        str(manifest_path),
                        "--confirm-reviewed-remote-code",
                    ]
                )
            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            self.assertEqual("unknown", VLMEvent.from_json(stdout.getvalue()).event_type.value)

    def test_run_discards_model_output_before_printing_validated_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            manifest_path = _write_manifest(root, model_root)
            request_path = _write_request(root)
            stdout = StringIO()
            stderr = StringIO()

            def noisy_generate(*_args: object, **_kwargs: object) -> str:
                print("untrusted model stdout diagnostic")
                print("untrusted model stderr diagnostic", file=sys.stderr)
                return _valid_result("frame_01.jpg")

            with (
                patch(
                    "baby_monitor.local_runners.mage_vl_4b.generate",
                    side_effect=noisy_generate,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "run",
                        "--request",
                        str(request_path),
                        "--model-path",
                        str(model_root),
                        "--code-manifest",
                        str(manifest_path),
                        "--confirm-reviewed-remote-code",
                    ]
                )
            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            self.assertNotIn("untrusted model stdout diagnostic", stdout.getvalue())
            self.assertNotIn("untrusted model stderr diagnostic", stderr.getvalue())
            self.assertEqual("unknown", VLMEvent.from_json(stdout.getvalue()).event_type.value)

    def test_run_fails_closed_without_echoing_invalid_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            manifest_path = _write_manifest(root, model_root)
            request_path = _write_request(root)
            private_output = "private hallucinated model output"
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.mage_vl_4b.generate",
                    return_value=private_output,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "run",
                        "--request",
                        str(request_path),
                        "--model-path",
                        str(model_root),
                        "--code-manifest",
                        str(manifest_path),
                        "--confirm-reviewed-remote-code",
                    ]
                )
            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertNotIn(private_output, stderr.getvalue())
            self.assertNotIn(str(root), stderr.getvalue())
            self.assertEqual(
                "invalid_structured_output",
                json.loads(stderr.getvalue())["error_code"],
            )

    def test_run_rejects_an_empty_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            manifest_path = _write_manifest(root, model_root)
            request_path = _write_request(root)
            payload = json.loads(_valid_result("frame_01.jpg"))
            payload["summary_zh"] = ""
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.mage_vl_4b.generate",
                    return_value=json.dumps(payload, ensure_ascii=False),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "run",
                        "--request",
                        str(request_path),
                        "--model-path",
                        str(model_root),
                        "--code-manifest",
                        str(manifest_path),
                        "--confirm-reviewed-remote-code",
                    ]
                )
            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertEqual(
                {
                    "protocol": "baby_monitor_local_runner_failure_v1",
                    "error_code": "invalid_structured_output",
                    "oom_status": "not_observed",
                },
                json.loads(stderr.getvalue()),
            )

    def test_run_fails_closed_without_echoing_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_root = _write_model_root(root)
            manifest_path = _write_manifest(root, model_root)
            request_path = _write_request(root)
            private_error = f"CUDA failed while opening {root}"
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.mage_vl_4b.generate",
                    side_effect=RuntimeError(private_error),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "run",
                        "--request",
                        str(request_path),
                        "--model-path",
                        str(model_root),
                        "--code-manifest",
                        str(manifest_path),
                        "--confirm-reviewed-remote-code",
                    ]
                )
            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertNotIn(private_error, stderr.getvalue())
            self.assertNotIn(str(root), stderr.getvalue())
            self.assertEqual("unknown", json.loads(stderr.getvalue())["oom_status"])

    def test_runner_source_has_no_download_or_remote_api_path(self) -> None:
        source = Path("src/baby_monitor/local_runners/mage_vl_4b.py").read_text(encoding="utf-8")
        for forbidden in ("snapshot_download", "http://", "https://", "openai"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source.lower())
        for setting in (
            'os.environ["HF_HUB_OFFLINE"] = "1"',
            'os.environ["TRANSFORMERS_OFFLINE"] = "1"',
            'os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"',
        ):
            with self.subTest(setting=setting):
                self.assertIn(setting, source)
        self.assertNotIn('setdefault("HF_HUB_OFFLINE"', source)

    def test_public_manifest_schema_pins_identity_and_limits(self) -> None:
        schema = json.loads(
            Path("schemas/trusted-model-code-manifest-v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(1, schema["properties"]["version"]["const"])
        self.assertEqual(MODEL_ID, schema["properties"]["model_id"]["const"])
        self.assertEqual(OFFICIAL_REVISION, schema["properties"]["revision"]["const"])
        self.assertEqual(64, schema["properties"]["files"]["maxItems"])
        self.assertTrue(schema["properties"]["files"]["uniqueItems"])
        required_paths = {
            rule["contains"]["properties"]["path"]["const"]
            for rule in schema["properties"]["files"]["allOf"]
        }
        self.assertEqual(REQUIRED_CONTROL_FILES, required_paths)
        self.assertEqual(
            32 * 1024 * 1024,
            schema["properties"]["files"]["items"]["properties"]["size"]["maximum"],
        )
        self.assertEqual(
            160,
            schema["properties"]["files"]["items"]["properties"]["path"]["maxLength"],
        )

    def test_ab_example_routes_mage_through_the_shared_command_interface(self) -> None:
        config = tomllib.loads(Path("config/vlm-ab.example.toml").read_text(encoding="utf-8"))
        mage = config["backends"]["mage_vl_4b"]
        command = mage["command"]
        self.assertFalse(mage["enabled"])
        self.assertEqual("command", mage["kind"])
        self.assertEqual("baby_monitor.local_runners.mage_vl_4b", command[2])
        self.assertEqual("run", command[3])
        self.assertEqual(1, command.count("{request}"))
        self.assertIn(OFFICIAL_REVISION, command)
        self.assertIn("--confirm-reviewed-remote-code", command)
        self.assertEqual(MODEL_ID, mage["model_id"])
        self.assertEqual(OFFICIAL_REVISION, mage["revision"])
        self.assertEqual("vlm_event_v5_20260912_coverage_unverified", mage["prompt_version"])
        self.assertEqual("transformers5.7_bf16_sdpa_jsonpost_v1", mage["runtime_id"])

    def test_remote_code_controls_are_reflected_in_the_threat_model(self) -> None:
        threat_model = Path("docs/security/threat-model.md").read_text(encoding="utf-8")
        self.assertIn("TM-05", threat_model)
        self.assertIn("materialized", threat_model)
        self.assertIn("HF_MODULES_CACHE", threat_model)
        self.assertIn("manifest 只证明审查后文件未变", threat_model)
        self.assertIn("不证明代码安全", threat_model)


if __name__ == "__main__":
    unittest.main()
