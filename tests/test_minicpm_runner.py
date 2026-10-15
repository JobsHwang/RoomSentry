from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from baby_monitor.domain import EventMedia, EvidenceGap, SchemaError, VLMEvent
from baby_monitor.local_runners.common import build_prompt as common_build_prompt
from baby_monitor.local_runners.mage_vl_4b import build_prompt as mage_build_prompt
from baby_monitor.local_runners.minicpm_v46 import (
    JSON_SYSTEM_PROMPT,
    MiniCPMRunnerError,
    MiniCPMSettings,
    _build_json_prefix_fn,
    build_constrained_schema,
    build_prompt,
    load_request,
    main,
    validate_result,
)


def _valid_result(frame: str) -> str:
    return json.dumps(
        {
            "event_type": "unknown",
            "subjects": [],
            "observations": ["画面中可见纯色测试图。"],
            "inferences": [],
            "risk_level": "none",
            "confidence": 0.2,
            "needs_remote_review": True,
            "summary_zh": "测试图不足以判断宝宝事件。",
            "evidence_frames": [frame],
        },
        ensure_ascii=False,
    )


class MiniCPMRunnerTests(unittest.TestCase):
    def test_parser_context_gets_unicode_alphabet_before_prefix_builder(self):
        tokenizer = object()
        schema = {"type": "object"}
        alphabet = 'abc012婴儿"'
        data = SimpleNamespace(tokenizer_alphabet=alphabet)
        steps = []
        captured = {}

        def data_builder(received):
            self.assertIs(tokenizer, received)
            steps.append("data")
            return data

        def config_factory(*, alphabet):
            self.assertEqual(["data"], steps)
            steps.append("config")
            return SimpleNamespace(alphabet=alphabet)

        def parser_factory(received_schema, *, config):
            self.assertIs(schema, received_schema)
            self.assertEqual(["data", "config"], steps)
            steps.append("parser")
            # Model the library's constructor-time alphabet snapshot.
            parser = SimpleNamespace(context_alphabet=config.alphabet)
            captured["parser"] = parser
            return parser

        sentinel = object()

        def prefix_builder(received_data, parser):
            self.assertIs(data, received_data)
            self.assertIs(captured["parser"], parser)
            self.assertEqual(alphabet, parser.context_alphabet)
            self.assertEqual(["data", "config", "parser"], steps)
            steps.append("prefix")
            return sentinel

        result = _build_json_prefix_fn(
            tokenizer,
            schema,
            tokenizer_data_builder=data_builder,
            config_factory=config_factory,
            parser_factory=parser_factory,
            prefix_builder=prefix_builder,
        )
        self.assertIs(sentinel, result)
        self.assertEqual(["data", "config", "parser", "prefix"], steps)
        self.assertEqual({"type": "object"}, schema)

    def test_prefix_assembly_keeps_tokenizer_initialization_failure_closed(self):
        from unittest.mock import Mock

        parser_factory = Mock()
        prefix_builder = Mock()

        def unavailable(_tokenizer):
            raise ValueError("synthetic unavailable tokenizer")

        with self.assertRaisesRegex(ValueError, "unavailable tokenizer"):
            _build_json_prefix_fn(
                object(),
                {"type": "object"},
                tokenizer_data_builder=unavailable,
                config_factory=Mock(),
                parser_factory=parser_factory,
                prefix_builder=prefix_builder,
            )
        parser_factory.assert_not_called()
        prefix_builder.assert_not_called()

    def _request(self, root: Path, *, frame_values: list[str] | None = None) -> Path:
        frame = root / "frame_01.jpg"
        frame.write_bytes(b"synthetic-frame-placeholder")
        frames = frame_values or [str(frame)]
        request = root / "request.json"
        request.write_text(
            json.dumps(
                {
                    "candidate": {"detections": []},
                    "evidence": EventMedia(
                        clip_path=str(root / "clip.mp4"),
                        frame_paths=tuple(frames),
                        selected_segments=1,
                    ).to_dict(),
                    "frame_paths": frames,
                    "question": "证据不足时输出 unknown。",
                    "output_schema": VLMEvent.json_schema(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return request

    def test_request_rejects_non_local_frame_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._request(root, frame_values=["https://example.invalid/frame.jpg"])
            with self.assertRaises(MiniCPMRunnerError):
                load_request(request)

    def test_settings_reject_boolean_or_fractional_generation_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            for value in (True, 512.5):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    MiniCPMSettings(  # type: ignore[arg-type]
                        model_path=model_path,
                        max_new_tokens=value,
                    )

    def test_request_rejects_a_different_output_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = self._request(root)
            value = json.loads(request_path.read_text(encoding="utf-8"))
            value["output_schema"] = {"type": "object"}
            request_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(MiniCPMRunnerError):
                load_request(request_path)

    def test_request_rejects_non_utf8_json_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_bytes(b"\xff\xfe\x00")
            with self.assertRaisesRegex(MiniCPMRunnerError, "missing or invalid"):
                load_request(request_path)

    def test_prompt_separates_visible_facts_and_inferences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = load_request(self._request(Path(directory)))
            prompt = build_prompt(request)
            self.assertIn("静态无声关键帧不能证明哭闹", prompt)
            self.assertIn("静态或重复关键帧不能证明 walking、movement", prompt)
            self.assertIn("没有可见跨帧变化时，event_type 应为 unknown", prompt)
            self.assertIn("zone_enter、zone_dwell", prompt)
            self.assertIn("observations", prompt)
            self.assertIn("inferences", prompt)
            self.assertIn("不能增加、删除、重复或改写字段名", prompt)
            self.assertIn('"event_type":"unknown"', prompt)
            self.assertIn("模板中的值只是格式占位", prompt)
            self.assertIn("summary_zh 必须用非空中文短句", prompt)
            self.assertIn("绝对禁止 ```json", prompt)
            self.assertIn("数组必须是原生 JSON 数组", prompt)
            self.assertIn("不要把 observations、inferences 等字段名当成数组内容", prompt)
            self.assertNotIn("additionalProperties", prompt)
            self.assertNotIn(str(request.frame_paths[0].parent), prompt)
            self.assertIn("证据不足时保持 unknown", JSON_SYSTEM_PROMPT)
            self.assertIn("系统信号事实", prompt)
            self.assertIn("证据覆盖元数据", prompt)
            self.assertIn('"coverage_status": "unverified"', prompt)
            self.assertIn('"continuous_coverage_verified": false', prompt)
            self.assertNotIn('"complete_coverage"', prompt)

    def test_shared_minicpm_and_mage_coverage_hints_never_upgrade_legacy_no_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            request = load_request(self._request(Path(directory)))
            started = datetime(2026, 9, 12, tzinfo=UTC)
            gap = EvidenceGap(started, started + timedelta(seconds=3), "sampling_gap")
            for gaps, status in (((), "unverified"), ((gap,), "gaps_recorded")):
                bounded = replace(request, evidence=replace(request.evidence, coverage_gaps=gaps))
                for name, builder in (
                    ("minicpm", build_prompt),
                    ("mage", mage_build_prompt),
                    ("common", common_build_prompt),
                ):
                    with self.subTest(runner=name, status=status):
                        prompt = builder(bounded)
                        metadata = json.loads(
                            next(
                                line
                                for line in prompt.splitlines()
                                if line.startswith("证据覆盖元数据：")
                            ).removeprefix("证据覆盖元数据：")
                        )
                        self.assertEqual(status, metadata["coverage_status"])
                        self.assertIs(False, metadata["continuous_coverage_verified"])
                        self.assertEqual(
                            [item.reason for item in gaps], metadata["coverage_gap_reasons"]
                        )
                        self.assertEqual(1, metadata["selected_segments"])
                        self.assertNotIn("complete", metadata)
                        self.assertNotIn("complete_coverage", metadata)
                        self.assertIn("未知不等于已记录缺失", prompt)
                        self.assertIn("未记录缺口不能证明没有缺口", prompt)
                        self.assertNotIn(str(request.frame_paths[0].parent), prompt)

    def test_request_rejects_mismatched_evidence_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request_path = self._request(Path(directory))
            value = json.loads(request_path.read_text(encoding="utf-8"))
            value["evidence"]["frame_paths"] = ["/different/frame.jpg"]
            request_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(MiniCPMRunnerError, "do not match"):
                load_request(request_path)

    def test_unknown_evidence_frame_is_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            validate_result(
                _valid_result("invented.jpg"),
                allowed_frame_ids=("frame_01.jpg",),
            )

    def test_empty_root_evidence_is_rejected(self) -> None:
        payload = json.loads(_valid_result("frame_01.jpg"))
        payload["evidence_frames"] = []
        with self.assertRaisesRegex(SchemaError, "did not cite"):
            validate_result(
                json.dumps(payload),
                allowed_frame_ids=("frame_01.jpg",),
            )

    def test_constrained_schema_enumerates_only_requested_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            request = load_request(self._request(Path(directory)))
            schema = build_constrained_schema(request)
            root_items = schema["properties"]["evidence_frames"]["items"]
            subject_items = schema["properties"]["subjects"]["items"]["properties"][
                "evidence_frames"
            ]["items"]
            self.assertEqual(["frame_01.jpg"], root_items["enum"])
            self.assertEqual(["frame_01.jpg"], subject_items["enum"])
            self.assertEqual(1, schema["properties"]["evidence_frames"]["minItems"])
            self.assertEqual(
                1,
                schema["properties"]["subjects"]["items"]["properties"]["evidence_frames"][
                    "minItems"
                ],
            )
            self.assertNotIn(
                "enum",
                request.output_schema["properties"]["evidence_frames"]["items"],
            )

    def test_main_prints_only_validated_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = self._request(root)
            model_path = root / "model"
            model_path.mkdir()
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.minicpm_v46.generate",
                    return_value=_valid_result("frame_01.jpg"),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "--request",
                        str(request_path),
                        "--model-path",
                        str(model_path),
                    ]
                )
            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            parsed = VLMEvent.from_json(stdout.getvalue())
            self.assertEqual("unknown", parsed.event_type.value)

    def test_main_fails_closed_without_echoing_invalid_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = self._request(root)
            model_path = root / "model"
            model_path.mkdir()
            invalid = "private hallucinated model text"
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.minicpm_v46.generate",
                    return_value=invalid,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "--request",
                        str(request_path),
                        "--model-path",
                        str(model_path),
                    ]
                )
            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertNotIn(invalid, stderr.getvalue())
            self.assertEqual(
                {
                    "protocol": "baby_monitor_local_runner_failure_v1",
                    "error_code": "invalid_structured_output",
                    "oom_status": "not_observed",
                },
                json.loads(stderr.getvalue()),
            )

    def test_main_fails_closed_without_echoing_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = self._request(root)
            model_path = root / "model"
            model_path.mkdir()
            private_error = f"CUDA failed while opening {root}"
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.minicpm_v46.generate",
                    side_effect=RuntimeError(private_error),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "--request",
                        str(request_path),
                        "--model-path",
                        str(model_path),
                    ]
                )
            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertNotIn(private_error, stderr.getvalue())
            self.assertNotIn(str(root), stderr.getvalue())
            self.assertEqual("unknown", json.loads(stderr.getvalue())["oom_status"])

    def test_main_reports_only_an_explicit_torch_oom_as_observed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = self._request(root)
            model_path = root / "model"
            model_path.mkdir()
            torch_oom = type(
                "OutOfMemoryError",
                (RuntimeError,),
                {"__module__": "torch.cuda"},
            )("private CUDA allocation detail")
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.minicpm_v46.generate",
                    side_effect=torch_oom,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(
                    [
                        "--request",
                        str(request_path),
                        "--model-path",
                        str(model_path),
                    ]
                )
            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertEqual(
                {
                    "protocol": "baby_monitor_local_runner_failure_v1",
                    "error_code": "runner_oom",
                    "oom_status": "observed",
                },
                json.loads(stderr.getvalue()),
            )


if __name__ == "__main__":
    unittest.main()
