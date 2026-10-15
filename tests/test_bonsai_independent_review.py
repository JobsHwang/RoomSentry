from __future__ import annotations

import base64
import copy
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from unittest.mock import Mock, patch

from baby_monitor.local_runners import bonsai_27b as runner
from baby_monitor.local_runners import live_bonsai as live
from tests import test_bonsai_runner as fixtures


class BonsaiIndependentReviewTests(unittest.TestCase):
    setUp = fixtures.BonsaiRunnerTests.setUp

    def hinted_request(self):
        return replace(
            self.request,
            candidate={
                "opaque_id": "opaque-candidate-17",
                "opaque_metadata": {"preserve": ["value"]},
                "detections": [
                    {
                        "label": "detector-person-sentinel",
                        "confidence": 0.591237,
                        "box": [0.112345, 0.212345, 0.712345, 0.912345],
                    }
                ],
                "signals": [{"type": "signal-sentinel", "zone": "zone-sentinel"}],
            },
        )

    def test_independent_payload_hides_hints_without_mutating_original_request(self):
        request = self.hinted_request()
        original = copy.deepcopy(request.candidate)
        settings = replace(self.settings, independent_visual_review=True)
        with patch.object(runner, "build_prompt", wraps=runner.build_prompt) as build:
            payload = runner.build_payload(settings, request)
        prompt_request = build.call_args.args[0]
        self.assertIsNot(request, prompt_request)
        self.assertIsNot(request.candidate, prompt_request.candidate)
        self.assertEqual([], prompt_request.candidate["detections"])
        self.assertEqual([], prompt_request.candidate["signals"])
        self.assertEqual("opaque-candidate-17", prompt_request.candidate["opaque_id"])
        self.assertEqual(original["opaque_metadata"], prompt_request.candidate["opaque_metadata"])
        self.assertEqual(original, request.candidate)
        self.assertIs(request.evidence, prompt_request.evidence)
        self.assertIs(request.frame_paths, prompt_request.frame_paths)
        self.assertIs(request.output_schema, prompt_request.output_schema)
        self.assertEqual(request.question, prompt_request.question)
        serialized = json.dumps(payload, ensure_ascii=False)
        for sentinel in (
            "detector-person-sentinel",
            "0.591237",
            "0.112345",
            "0.212345",
            "0.712345",
            "0.912345",
            "signal-sentinel",
            "zone-sentinel",
        ):
            with self.subTest(sentinel=sentinel):
                self.assertNotIn(sentinel, serialized)

    def test_original_pixels_ids_schema_and_generation_limits_are_preserved(self):
        request = self.hinted_request()
        original = runner.build_payload(self.settings, request)
        independent_settings = replace(self.settings, independent_visual_review=True)
        independent = runner.build_payload(independent_settings, request)
        content = independent["messages"][0]["content"]
        self.assertEqual(2, len(content))
        self.assertEqual(original["messages"][0]["content"][0], content[0])
        image_data = content[0]["image_url"]["url"].split(",", 1)[1]
        self.assertEqual(self.frame.read_bytes(), base64.b64decode(image_data))
        self.assertIn('帧 ID（输入顺序）：["frame.png"]', content[1]["text"])
        self.assertEqual(original["response_format"], independent["response_format"])
        for key in original.keys() - {"messages"}:
            self.assertEqual(original[key], independent[key])
        self.assertEqual(
            runner._server_command(self.settings), runner._server_command(independent_settings)
        )

    def test_independent_guidance_is_explicit_about_missing_hints_and_uncertainty(self):
        payload = runner.build_payload(
            replace(self.settings, independent_visual_review=True), self.hinted_request()
        )
        prompt = payload["messages"][0]["content"][1]["text"]
        for required in (
            "未提供任何上游检测或系统信号",
            "空列表不表示无人",
            "只依原图像素独立判断",
            "不能复述、猜测或引用检测器",
            "显示器中的人物、倒影、衣物",
            "subject.kind 使用 person",
            "睡眠状态保持 unknown",
        ):
            with self.subTest(required=required):
                self.assertIn(required, prompt)

    def test_default_keeps_existing_hints_prompt_and_revision(self):
        request = self.hinted_request()
        with patch.object(runner, "build_prompt", wraps=runner.build_prompt) as build:
            payload = runner.build_payload(self.settings, request)
        self.assertIs(request, build.call_args.args[0])
        prompt = payload["messages"][0]["content"][1]["text"]
        self.assertIn("detector-person-sentinel", prompt)
        self.assertIn("0.591237", prompt)
        self.assertIn("signal-sentinel", prompt)
        self.assertNotIn("本轮是独立视觉复核", prompt)
        self.assertIs(False, self.settings.independent_visual_review)
        self.assertEqual("bonsai_single_frame_v1_20260929", runner.PROMPT_VERSION)
        self.assertEqual(runner.PROMPT_VERSION, runner.prompt_revision(self.settings))

    def test_settings_require_real_boolean(self):
        for value in (0, 1, "true", "false", None, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.settings, independent_visual_review=value)

    def cli_arguments(self):
        return [
            "--model-directory",
            str(self.models),
            "--runtime-directory",
            str(self.runtime),
            "--runtime-manifest",
            str(self.manifest),
            "--cuda-library-directory",
            str(self.cuda),
            "--max-new-tokens",
            "768",
        ]

    def test_one_shot_cli_flag_selects_independent_settings_and_preserves_default(self):
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(runner, "load_request", return_value=self.request),
                patch.object(runner, "generate", return_value="{}") as generate,
                redirect_stdout(StringIO()),
            ):
                args = self.cli_arguments() + ["--request", str(self.root / "request.json")]
                if enabled:
                    args.append("--independent-visual-review")
                self.assertEqual(0, runner.main(args))
                settings, request = generate.call_args.args
                self.assertIs(enabled, settings.independent_visual_review)
                self.assertIs(self.request, request)
                self.assertEqual(768, settings.max_new_tokens)

    def test_live_cli_flag_selects_independent_settings_and_preserves_default(self):
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(live.socket, "socket"),
                patch.object(live, "run") as run,
            ):
                args = self.cli_arguments() + ["--fd", "19", "--execution-lock-fd", "20"]
                if enabled:
                    args.append("--independent-visual-review")
                self.assertEqual(0, live.main(args))
                settings = run.call_args.args[1]
                self.assertIs(enabled, settings.independent_visual_review)
                self.assertEqual(768, settings.max_new_tokens)
                self.assertEqual(20, run.call_args.kwargs["execution_lock_fd"])

    def test_ready_reports_revision_for_selected_prompt(self):
        for enabled, revision in (
            (False, runner.PROMPT_VERSION),
            (True, runner.INDEPENDENT_PROMPT_VERSION),
        ):
            with (
                self.subTest(enabled=enabled),
                patch.object(live, "BonsaiSession") as session,
                patch.object(live, "receive_packet", return_value=({"type": "stop"}, b"")),
                patch.object(live, "send_packet") as send,
            ):
                settings = replace(self.settings, independent_visual_review=enabled)
                channel = Mock()
                live.run(channel, settings)
                self.assertEqual(revision, runner.prompt_revision(settings))
                self.assertEqual(revision, send.call_args.args[1]["prompt_revision"])
                self.assertEqual(1, send.call_args.args[1]["model_loads"])
                self.assertIs(False, send.call_args.args[1]["network_enabled"])
                session.assert_called_once_with(settings, execution_lock_fd=None)
                session.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
