from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from baby_monitor.ab_eval import (
    ABBackend,
    _build_backends,
    _score,
    load_dataset,
    run_ab,
)
from baby_monitor.adapters.vlm import VLMAdapter, VLMError, VLMRequest
from baby_monitor.cli import main
from baby_monitor.domain import (
    EventType,
    RecognitionRunMetrics,
    RiskLevel,
    SubjectKind,
    VLMEvent,
)


class _PrivateFailureAdapter(VLMAdapter):
    name = "private_failure"

    def __init__(self, message: str):
        self.message = message

    def analyze(self, request: VLMRequest) -> VLMEvent:
        raise RuntimeError(self.message)


class _ReportedFailureAdapter(VLMAdapter):
    name = "reported_failure"

    def analyze(self, request: VLMRequest) -> VLMEvent:
        raise VLMError(
            "sanitized local runner rejection",
            code="invalid_structured_output",
            oom_status="not_observed",
            metrics=RecognitionRunMetrics(
                duration_ms=123,
                json_compliant=False,
                process_rss_peak_kib=456,
                gpu_device_memory_peak_mib=789,
            ),
        )


class ABEvaluationTests(unittest.TestCase):
    @staticmethod
    def _case(root: Path) -> dict[str, object]:
        frame = root / "frame.ppm"
        frame.write_text("P3\n1 1\n255\n64 64 64\n", encoding="ascii")
        return {
            "version": 1,
            "case_id": "case_001",
            "frames": [
                {
                    "frame_id": "f01",
                    "path": frame.name,
                    "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                }
            ],
            "question": "只写可见事实。",
            "expected": {
                "event_type": "unknown",
                "risk_level": "none",
                "allowed_subjects": [],
                "allowed_observations": [],
                "allowed_inferences": [],
            },
        }

    def test_mock_backend_produces_metrics_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(self._case(root), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            config = root / "ab.toml"
            config.write_text(
                "\n".join(
                    [
                        "[run]",
                        f'dataset = "{dataset.as_posix()}"',
                        f'output_dir = "{(root / "out").as_posix()}"',
                        "poll_interval_seconds = 0.05",
                        "",
                        "[backends.mock]",
                        "enabled = true",
                        'kind = "mock"',
                    ]
                ),
                encoding="utf-8",
            )
            artifacts = run_ab(config)
            self.assertEqual(1, artifacts.backend_count)
            self.assertEqual(1, artifacts.case_count)
            self.assertTrue(artifacts.results_path.is_file())
            self.assertTrue(artifacts.summary_path.is_file())
            raw_result = artifacts.results_path.read_text(encoding="utf-8")
            result = json.loads(raw_result)
            self.assertEqual(1, result["version"])
            self.assertEqual(1, result["case_schema_version"])
            self.assertRegex(result["dataset_sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(result["json_compliant"])
            self.assertIsNone(result["error_code"])
            self.assertEqual("not_observed", result["oom_status"])
            self.assertEqual(
                {
                    "version": 1,
                    "duration_ms": result["recognition_metrics"]["duration_ms"],
                    "json_compliant": True,
                    "process_rss_peak_kib": None,
                    "gpu_device_memory_peak_mib": None,
                },
                result["recognition_metrics"],
            )
            self.assertEqual(
                {
                    "version": 1,
                    "model_id": "builtin/mock",
                    "model_revision": "builtin_mock_v1",
                    "prompt_version": "vlm_event_v5_20260912_coverage_unverified",
                    "runtime_id": "stdlib",
                },
                result["provenance"],
            )
            self.assertTrue(result["score"]["event_type_match"])
            self.assertTrue(result["score"]["evidence_compliant"])
            self.assertEqual(["f01"], result["result"]["evidence_frames"])
            self.assertNotIn(str(root), raw_result)
            self.assertIn("gpu_used_peak_mb", result["resources"])
            summary = artifacts.summary_path.read_text(encoding="utf-8")
            self.assertIn("子进程RSS峰值(MiB)", summary)
            self.assertIn("失败数/OOM未知数", summary)

    def test_dataset_rejects_hash_mismatch_and_extra_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            value = self._case(root)
            value["frames"][0]["sha256"] = "0" * 64  # type: ignore[index]
            dataset.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                load_dataset(dataset)

            value = self._case(root)
            value["private_note"] = "must be rejected"
            dataset.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly the v1 fields"):
                load_dataset(dataset)

            dataset.write_bytes(b"\xff\xfe\x00")
            with self.assertRaisesRegex(ValueError, "readable UTF-8"):
                load_dataset(dataset)

    def test_dataset_rejects_duplicate_cases_and_symlink_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            value = self._case(root)
            line = json.dumps(value) + "\n"
            dataset.write_text(line + line, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicated"):
                load_dataset(dataset)

            target = root / "frame.ppm"
            link = root / "linked.ppm"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {type(exc).__name__}")
            value = self._case(root)
            value["frames"][0]["path"] = link.name  # type: ignore[index]
            dataset.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                load_dataset(dataset)

    def test_dataset_validator_reports_counts_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(self._case(root), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(["validate-ab-dataset", str(dataset)])
            self.assertEqual(0, exit_code)
            payload = json.loads(output.getvalue())
            self.assertEqual(1, payload["cases"])
            self.assertEqual(1, payload["frames"])
            self.assertEqual([1], payload["case_versions"])
            self.assertTrue(payload["hashes_verified"])
            self.assertFalse(payload["paths_disclosed"])
            self.assertNotIn(str(root), output.getvalue())

    def test_public_schema_enums_match_domain_contract(self) -> None:
        schema = json.loads(
            Path("schemas/vlm-eval-case-v1.schema.json").read_text(encoding="utf-8")
        )
        expected = schema["properties"]["expected"]["properties"]
        self.assertEqual(1, schema["properties"]["version"]["const"])
        self.assertEqual({item.value for item in EventType}, set(expected["event_type"]["enum"]))
        self.assertEqual({item.value for item in RiskLevel}, set(expected["risk_level"]["enum"]))
        self.assertEqual(
            {item.value for item in SubjectKind},
            set(expected["allowed_subjects"]["items"]["enum"]),
        )
        schema_v2 = json.loads(
            Path("schemas/vlm-eval-case-v2.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(2, schema_v2["properties"]["version"]["const"])

    def test_v2_required_fact_groups_measure_coverage_without_losing_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self._case(root)
            case["version"] = 2
            expected = case["expected"]
            assert isinstance(expected, dict)
            expected["allowed_observations"] = ["red rectangle", "red box"]
            expected["required_observation_groups"] = [["red rectangle", "red box"]]
            expected["required_inference_groups"] = []
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps(case) + "\n", encoding="utf-8")
            parsed = load_dataset(dataset)[0]
            self.assertEqual(2, parsed.schema_version)
            result = VLMEvent(
                event_type=EventType.UNKNOWN,
                subjects=(),
                observations=("red rectangle",),
                inferences=(),
                risk_level=RiskLevel.NONE,
                confidence=0.0,
                needs_remote_review=True,
                summary_zh="",
                evidence_frames=(parsed.frames[0].path,),
            )
            score = _score(result, parsed.expected, evidence_compliant=True)
            self.assertEqual(1, score["required_observation_group_hits"])
            self.assertEqual(1, score["required_observation_group_total"])
            self.assertEqual(0, score["unsupported_observation_count"])

            expected["required_observation_groups"] = [["missing alternative"]]
            dataset.write_text(json.dumps(case) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must also be allowed"):
                load_dataset(dataset)

    def test_command_backend_requires_path_free_reproducibility_metadata(self) -> None:
        backend = {
            "enabled": True,
            "kind": "command",
            "model_id": "microsoft/Mage-VL",
            "revision": "5c78cab61938e73859b63724d9bf5cb88c477eaa",
            "prompt_version": "vlm_event_v5_20260912_coverage_unverified",
            "runtime_id": "transformers5.7_bf16_sdpa_lmfe0.11.3",
            "command": [sys.executable, "-c", "pass", "{request}"],
        }
        built = _build_backends({"mage_vl_4b": backend})
        self.assertEqual(
            {
                "version": 1,
                "model_id": "microsoft/Mage-VL",
                "model_revision": "5c78cab61938e73859b63724d9bf5cb88c477eaa",
                "prompt_version": "vlm_event_v5_20260912_coverage_unverified",
                "runtime_id": "transformers5.7_bf16_sdpa_lmfe0.11.3",
            },
            built["mage_vl_4b"].provenance(),
        )

        for field, invalid in (
            ("model_id", "/private/model"),
            ("revision", "/private/revision"),
            ("revision", ".."),
            ("prompt_version", "prompt with spaces"),
            ("runtime_id", "../../runtime"),
        ):
            value = dict(backend)
            value[field] = invalid
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "path-free"):
                _build_backends({"mage_vl_4b": value})

        extra = dict(backend)
        extra["private_path"] = "/must/not/persist"
        with self.assertRaisesRegex(ValueError, "fields are invalid"):
            _build_backends({"mage_vl_4b": extra})

    def test_failed_case_never_persists_private_exception_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(self._case(root), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            config = root / "ab.toml"
            config.write_text(
                "\n".join(
                    [
                        "[run]",
                        f'dataset = "{dataset.as_posix()}"',
                        f'output_dir = "{(root / "out").as_posix()}"',
                    ]
                ),
                encoding="utf-8",
            )
            private_message = f"failed while reading {root / 'private-frame.jpg'}"
            backend = ABBackend(
                adapter=_PrivateFailureAdapter(private_message),
                model_id="fixture/failure",
                model_revision="fixture_revision",
                prompt_version="fixture_prompt",
                runtime_id="fixture_runtime",
            )
            with patch(
                "baby_monitor.ab_eval._build_backends",
                return_value={"failure": backend},
            ):
                artifacts = run_ab(config)
            raw = artifacts.results_path.read_text(encoding="utf-8")
            result = json.loads(raw)
            self.assertEqual("RuntimeError", result["error"])
            self.assertIsNone(result["error_code"])
            self.assertEqual("unknown", result["oom_status"])
            self.assertIsNone(result["recognition_metrics"])
            self.assertNotIn(private_message, raw)
            self.assertNotIn(str(root), raw)
            summary = artifacts.summary_path.read_text(encoding="utf-8")
            self.assertIn("0.0%/0.0%/0.0%", summary)

    def test_failed_case_uses_only_strict_runner_oom_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text(
                json.dumps(self._case(root), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            config = root / "ab.toml"
            config.write_text(
                "\n".join(
                    [
                        "[run]",
                        f'dataset = "{dataset.as_posix()}"',
                        f'output_dir = "{(root / "out").as_posix()}"',
                    ]
                ),
                encoding="utf-8",
            )
            backend = ABBackend(
                adapter=_ReportedFailureAdapter(),
                model_id="fixture/failure",
                model_revision="fixture_revision",
                prompt_version="fixture_prompt",
                runtime_id="fixture_runtime",
            )
            with patch(
                "baby_monitor.ab_eval._build_backends",
                return_value={"failure": backend},
            ):
                artifacts = run_ab(config)
            result = json.loads(artifacts.results_path.read_text(encoding="utf-8"))
            self.assertEqual("invalid_structured_output", result["error_code"])
            self.assertEqual("not_observed", result["oom_status"])
            self.assertEqual(123, result["recognition_metrics"]["duration_ms"])
            self.assertEqual(456, result["recognition_metrics"]["process_rss_peak_kib"])
            self.assertEqual(789, result["recognition_metrics"]["gpu_device_memory_peak_mib"])
            self.assertIn(
                "| failure | 1 | 0.0% | 1/0 |",
                artifacts.summary_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
