from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from baby_monitor.domain import EventMedia, SchemaError, VLMEvent
from baby_monitor.local_runners import bonsai_27b as runner
from baby_monitor.local_runners.common import LocalVLMRequest, LocalVLMRunnerError


def event(frame="frame.png"):
    return json.dumps(
        {
            "event_type": "unknown",
            "subjects": [],
            "observations": ["可见红色矩形。"],
            "inferences": [],
            "risk_level": "none",
            "confidence": 0.6,
            "needs_remote_review": False,
            "summary_zh": "可见红色矩形。",
            "evidence_frames": [frame],
        }
    )


def envelope(raw=None, finish="stop"):
    return {
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "content": event() if raw is None else raw,
                },
            }
        ]
    }


class BonsaiRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.models, self.runtime, self.cuda = (
            self.root / p for p in ("models", "runtime", "cuda")
        )
        for directory in (self.models, self.runtime, self.cuda):
            directory.mkdir()
        (self.runtime / "llama-server").write_bytes(b"synthetic binary")
        self.manifest = self.root / "runtime.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "version": 1,
                    "revision": runner.RUNTIME_REVISION,
                    "files": {
                        "llama-server": {
                            "bytes": 16,
                            "sha256": hashlib.sha256(b"synthetic binary").hexdigest(),
                        }
                    },
                    "symlinks": {},
                }
            )
        )
        self.settings = runner.BonsaiSettings(
            self.models, self.runtime, self.manifest, (self.cuda,)
        )
        self.frame = self.root / "frame.png"
        # Header fixture, not a production decodable image.
        self.frame.write_bytes(
            b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 32, 32)
        )
        self.request = LocalVLMRequest(
            candidate={"detections": []},
            evidence=EventMedia("", (str(self.frame),)),
            frame_paths=(self.frame,),
            question="描述画面。",
            output_schema=VLMEvent.json_schema(),
        )

    def test_payload_is_single_inline_image_with_bounded_generation(self):
        payload = runner.build_payload(self.settings, self.request)
        self.assertEqual(512, payload["max_tokens"])
        self.assertEqual("none", payload["reasoning_effort"])
        content = payload["messages"][0]["content"]
        self.assertEqual(2, len(content))
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn(str(self.root), json.dumps(payload))
        self.assertIn("不转录招牌文字", content[1]["text"])
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(["frame.png"], schema["properties"]["evidence_frames"]["items"]["enum"])

    def test_refuses_multiple_frames_instead_of_silently_dropping_them(self):
        with self.assertRaises(LocalVLMRunnerError):
            runner.build_payload(
                self.settings, replace(self.request, frame_paths=(self.frame, self.frame))
            )

    def test_frame_rejects_unsupported_bytes_oversize_and_symlink(self):
        self.frame.write_bytes(b"not image")
        with self.assertRaises(LocalVLMRunnerError):
            runner._frame(self.request)
        with self.frame.open("wb") as stream:
            stream.truncate(runner.MAX_FRAME_BYTES + 1)
        with self.assertRaises(LocalVLMRunnerError):
            runner._frame(self.request)
        link = self.root / "link.png"
        link.symlink_to(self.frame)
        with self.assertRaises(OSError):
            runner._frame(replace(self.request, frame_paths=(link,)))

    def test_rejects_png_pixel_budget(self):
        self.frame.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", 13)
            + b"IHDR"
            + struct.pack(">II", 10000, 10000)
        )
        with self.assertRaises(LocalVLMRunnerError):
            runner._frame(self.request)

    def test_jpeg_dimensions_and_budget_before_native_decoder(self):
        for marker in (0xC0, 0xC2):
            for width, height in ((640, 480), (65535, 65535), (0, 100)):
                header = b"\xff\xd8\xff\xe0\x00\x04ab\xff" + bytes([marker])
                header += struct.pack(">HBHHB", 8, 8, height, width, 1)
                self.frame.write_bytes(header)
                if width == 640:
                    self.assertEqual((header, "image/jpeg"), runner._frame(self.request))
                else:
                    with self.assertRaises(LocalVLMRunnerError):
                        runner._frame(self.request)

    def test_jpeg_rejects_truncated_invalid_or_missing_dimensions(self):
        for data in (
            b"\xff\xd8\xff",
            b"\xff\xd8\xff\xda",
            b"\xff\xd8\xff\xe0\x00\x00",
            b"\xff\xd8\xff\xc0\x00\x08\x08",
            b"\xff\xd8\xff\xe0\xff\xff",
        ):
            self.frame.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(LocalVLMRunnerError):
                runner._frame(self.request)

    def test_settings_reject_invalid_budgets(self):
        for value in (True, 0, 1025, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.settings, max_new_tokens=value)

    def test_settings_reject_relative_or_symlink_asset_directories(self):
        link = self.root / "runtime-link"
        link.symlink_to(self.runtime)
        for path in (Path("relative"), link):
            with self.assertRaises(LocalVLMRunnerError):
                replace(self.settings, runtime_directory=path)

    def test_manifest_must_be_a_regular_non_symlink_file(self):
        fifo = self.root / "manifest-fifo"
        os.mkfifo(fifo)
        link = self.root / "manifest-link"
        link.symlink_to(self.manifest)
        for path in (fifo, link, self.root, self.root / "missing"):
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                replace(self.settings, runtime_manifest=path)

    def test_manifest_swapped_for_fifo_fails_without_blocking(self):
        self.manifest.unlink()
        os.mkfifo(self.manifest)
        with patch.object(runner, "MODEL_ASSETS", ()), self.assertRaises(LocalVLMRunnerError):
            self.settings.verify_assets()

    def test_strict_json_rejects_duplicate_nonfinite_and_fences(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', "```json\n{}\n```"):
            with self.assertRaises(ValueError):
                runner.strict_json(raw)

    def test_response_preserves_raw_unknown_review_flag_for_policy_layer(self):
        result = VLMEvent.from_json(runner.parse_response(envelope(), self.request))
        self.assertEqual("unknown", result.event_type.value)
        self.assertFalse(result.needs_remote_review)

    def test_response_rejects_incomplete_even_when_content_is_valid(self):
        for finish in ("length", "tool_calls", None):
            with self.assertRaises(LocalVLMRunnerError):
                runner.parse_response(envelope(finish=finish), self.request)

    def test_incomplete_and_bad_json_have_path_free_diagnostic_codes(self):
        for response, code in (
            (envelope(finish="length"), "bonsai_output_token_limit"),
            (envelope(finish="tool_calls"), "bonsai_response_incomplete"),
            (envelope("private invalid JSON"), "bonsai_invalid_response_json"),
        ):
            with self.assertRaises(runner.BonsaiResponseError) as caught:
                runner.parse_response(response, self.request)
            self.assertEqual(code, caught.exception.code)
            self.assertNotIn("private", str(caught.exception))
        with self.assertRaises(ValueError):
            runner.BonsaiResponseError("private arbitrary text")

    def test_response_rejects_wrong_evidence(self):
        with self.assertRaises(SchemaError):
            runner.parse_response(envelope(event("unseen.png")), self.request)

    def test_response_rejects_bad_envelopes(self):
        for value in ({}, {"error": "private"}, {"choices": []}, {"choices": [0]}, []):
            with self.assertRaises(LocalVLMRunnerError):
                runner.parse_response(value, self.request)

    def test_response_rejects_tool_calls(self):
        value = envelope()
        value["choices"][0]["message"]["tool_calls"] = [{"name": "unexpected"}]
        with self.assertRaises(LocalVLMRunnerError):
            runner.parse_response(value, self.request)

    def test_response_rejects_oversized_content(self):
        with self.assertRaises(LocalVLMRunnerError):
            runner.parse_response(envelope("x" * (runner.MAX_RESPONSE_BYTES + 1)), self.request)

    def test_environment_drops_credentials_proxy_and_model_overrides(self):
        with patch.dict(
            os.environ,
            {
                "FIXTURE_API_KEY": "secret",
                "https_proxy": "fixture",
                "LLAMA_ARG_HOST": "0.0.0.0",
            },
        ):
            env = runner._environment(self.settings)
        self.assertNotIn("FIXTURE_API_KEY", env)
        self.assertNotIn("https_proxy", env)
        self.assertNotIn("LLAMA_ARG_HOST", env)
        self.assertEqual(f"{self.runtime}:{self.cuda}", env["LD_LIBRARY_PATH"])

    def test_server_flags_force_offline_single_slot_and_no_webui(self):
        command = runner._server_command(self.settings)
        self.assertIn("--offline", command)
        self.assertIn("--no-webui", command)
        self.assertEqual("127.0.0.1", command[command.index("--host") + 1])
        self.assertEqual("4096", command[command.index("-c") + 1])
        self.assertEqual("1", command[command.index("-np") + 1])

    def test_namespace_refused_before_changing_host_interface(self):
        with (
            patch.object(runner.socket, "if_nameindex", return_value=[(1, "lo"), (2, "eth0")]),
            patch.object(runner.subprocess, "run") as run,
        ):
            with self.assertRaises(LocalVLMRunnerError):
                runner._isolated_loopback()
            run.assert_not_called()

    def test_redirect_is_refused(self):
        with self.assertRaises(LocalVLMRunnerError):
            runner._NoRedirect().redirect_request(None, None, 302, "", {}, "https://invalid/")

    def test_runtime_manifest_verifies_local_files_without_model_loading(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["files"]["llama-server"]["bytes"] = len(b"synthetic binary")
        self.manifest.write_text(json.dumps(manifest))
        with patch.object(runner, "MODEL_ASSETS", ()):
            self.settings.verify_assets()
            (self.runtime / "extra.so").write_bytes(b"extra")
            with self.assertRaises(LocalVLMRunnerError):
                self.settings.verify_assets()

    def test_runtime_hash_mismatch_is_rejected(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["files"]["llama-server"]["bytes"] = len(b"synthetic binary")
        manifest["files"]["llama-server"]["sha256"] = "0" * 64
        self.manifest.write_text(json.dumps(manifest))
        with patch.object(runner, "MODEL_ASSETS", ()), self.assertRaises(LocalVLMRunnerError):
            self.settings.verify_assets()

    def test_cleanup_on_success_exception_and_interruption(self):
        for failure in (None, ValueError("fixture"), KeyboardInterrupt()):
            process = Mock()
            process.poll.return_value = None
            http_results = [{"status": "ok"}, envelope() if failure is None else failure]
            with (
                patch.object(runner, "_isolated_loopback"),
                patch.object(runner.BonsaiSettings, "verify_assets"),
                patch.object(runner.subprocess, "Popen", return_value=process) as popen,
                patch.object(runner, "_http", side_effect=http_results),
            ):
                if failure is None:
                    runner.generate(self.settings, self.request)
                else:
                    with self.assertRaises(type(failure)):
                        runner.generate(self.settings, self.request)
            self.assertFalse(popen.call_args.kwargs["start_new_session"])
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=10)

    def test_slow_server_cleanup_escalates_to_kill(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 10), 0]
        runner._stop_server(process)
        process.kill.assert_called_once()
        self.assertEqual(2, process.wait.call_count)

    def test_cli_does_not_echo_private_failure(self):
        args = [
            "--request",
            str(self.root / "missing.json"),
            "--model-directory",
            str(self.models),
            "--runtime-directory",
            str(self.runtime),
            "--runtime-manifest",
            str(self.manifest),
            "--cuda-library-directory",
            str(self.cuda),
        ]
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(1, runner.main(args))
        self.assertEqual("", stdout.getvalue())
        self.assertNotIn(str(self.root), stderr.getvalue())
        self.assertEqual("runner_failed", json.loads(stderr.getvalue())["error_code"])


if __name__ == "__main__":
    unittest.main()
