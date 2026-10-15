from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from baby_monitor.local_runners import bonsai_27b as runner
from tests import test_bonsai_runner as fixtures


class BonsaiInputBudgetTests(unittest.TestCase):
    setUp = fixtures.BonsaiRunnerTests.setUp

    def test_defaults_and_explicit_higher_detail_server_arguments(self):
        for context, images in ((4096, 1024), (8192, 1024), (8192, 2048), (8192, 4096)):
            with self.subTest(context=context, images=images):
                settings = replace(self.settings, context_tokens=context, image_tokens=images)
                command = runner._server_command(settings)
                self.assertEqual(str(context), command[command.index("-c") + 1])
                for flag in ("--image-min-tokens", "--image-max-tokens"):
                    self.assertEqual(str(images), command[command.index(flag) + 1])
                # Changing representation budgets leaves the actual prompt/pixels/schema intact.
                self.assertEqual(
                    runner.build_payload(self.settings, self.request),
                    runner.build_payload(settings, self.request),
                )

    def test_budget_types_ranges_and_generation_room_rejected(self):
        for field, values in (
            ("context_tokens", (True, False, None, 4096.0, "4096", 0, 2048, 16384)),
            ("image_tokens", (True, False, None, 1024.0, "1024", 0, 512, 8192)),
        ):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.settings, **{field: value})
        with self.assertRaises(ValueError):
            replace(self.settings, context_tokens=4096, image_tokens=4096)
        with self.assertRaises(ValueError):
            replace(self.settings, context_tokens=4096, image_tokens=2048, max_new_tokens=1024)
        for value in (None, "768", True):
            with self.subTest(output=value), self.assertRaises(ValueError):
                replace(self.settings, max_new_tokens=value)

    def test_larger_explicit_output_only_changes_request_limit_not_prompt_or_server(self):
        larger = replace(self.settings, max_new_tokens=1024)
        expected = runner.build_payload(self.settings, self.request)
        self.assertEqual(512, expected["max_tokens"])
        expected["max_tokens"] = 1024
        self.assertEqual(expected, runner.build_payload(larger, self.request))
        self.assertEqual(runner._server_command(self.settings), runner._server_command(larger))
        for invalid in (1025, 10**400):
            with self.assertRaises(ValueError):
                replace(self.settings, max_new_tokens=invalid)

    def test_numeric_diagnostics_do_not_copy_raw_response_or_prompts(self):
        response = {
            **fixtures.envelope(),
            "prompt": "private prompt text",
            "truncated": False,
            "usage": {
                "prompt_tokens": 2750,
                "completion_tokens": 490,
                "total_tokens": 3240,
                "prompt_tokens_details": {"cached_tokens": 0, "image_tokens": 1008},
            },
            "timings": {
                "prompt_n": 2750,
                "predicted_n": 490,
                "prompt_ms": 1234.5,
                "predicted_ms": 8000,
            },
        }
        result = runner.response_diagnostics(response, self.settings)
        self.assertEqual(2750, result["prompt_tokens_reported"])
        self.assertEqual(1008, result["image_tokens_reported"])
        self.assertEqual(0, result["cached_tokens_reported"])
        self.assertIs(False, result["context_truncated_reported"])
        self.assertEqual("stop", result["finish_reason_reported"])
        self.assertEqual(self.settings.max_new_tokens, result["max_new_tokens_configured"])
        self.assertEqual(1234.5, result["prompt_ms_reported"])
        self.assertTrue(
            all(type(value) in (int, float, bool, str, type(None)) for value in result.values())
        )

    def test_missing_and_malformed_metadata_remain_unknown(self):
        for value in (None, [], {}, {"usage": "private", "timings": []}):
            result = runner.response_diagnostics(value, self.settings)
            self.assertIsNone(result["image_tokens_reported"])
            self.assertIsNone(result["prompt_tokens_reported"])
            self.assertIsNone(result["context_truncated_reported"])
            self.assertIsNone(result["finish_reason_reported"])
        for invalid in (True, -1, 1_000_001, 12.0, "12", {}, None):
            result = runner.response_diagnostics(
                {"usage": {"prompt_tokens": invalid}}, self.settings
            )
            self.assertIsNone(result["prompt_tokens_reported"])
        for invalid in (True, -1, 600_001, 10**400, -(10**400), math.nan, math.inf, "12", None):
            result = runner.response_diagnostics({"timings": {"prompt_ms": invalid}}, self.settings)
            self.assertIsNone(result["prompt_ms_reported"])

    def test_diagnostic_overflow_and_untrusted_finish_reason_do_not_break_generation(self):
        session = runner.BonsaiSession(self.settings)
        session.process = Mock()
        session.process.poll.return_value = None
        response = {**fixtures.envelope(), "timings": {"prompt_ms": 10**400}}
        with patch.object(runner, "_http", return_value=response):
            self.assertIsInstance(session.generate(self.request), str)
        self.assertIsNone(session.last_diagnostics["prompt_ms_reported"])
        for reason in ([], {}, "private model text", 1):
            response = {"choices": [{"finish_reason": reason}]}
            self.assertIsNone(
                runner.response_diagnostics(response, self.settings)["finish_reason_reported"]
            )

    def test_counter_uses_identical_payload_without_token_or_prompt_dump(self):
        session = runner.BonsaiSession(self.settings)
        session.process = Mock()
        session.process.poll.return_value = None
        with patch.object(
            runner, "_http", return_value={"input_tokens": 2750, "object": "response.input_tokens"}
        ) as http:
            self.assertEqual(2750, session.count_input_tokens(self.request))
        self.assertEqual("/v1/chat/completions/input_tokens", http.call_args.args[0])
        payload = runner.build_payload(self.settings, self.request)
        payload["cache_prompt"] = False
        self.assertEqual(payload, http.call_args.args[1])

    def test_counter_rejects_closed_server_and_invalid_counter_responses(self):
        session = runner.BonsaiSession(self.settings)
        with self.assertRaises(runner.BonsaiResponseError):
            session.count_input_tokens(self.request)
        session.process = Mock()
        session.process.poll.return_value = None
        for value in (
            {},
            {"object": "wrong", "input_tokens": 12},
            {"object": "response.input_tokens", "input_tokens": True},
            {"object": "response.input_tokens", "input_tokens": 0},
        ):
            with (
                patch.object(runner, "_http", return_value=value),
                self.assertRaises(runner.LocalVLMRunnerError),
            ):
                session.count_input_tokens(self.request)

    def test_generation_failure_keeps_numeric_metadata_and_next_request_resets_it(self):
        session = runner.BonsaiSession(self.settings)
        session.process = Mock()
        session.process.poll.return_value = None
        response = {**fixtures.envelope(finish="length"), "usage": {"completion_tokens": 768}}
        with (
            patch.object(runner, "_http", return_value=response),
            self.assertRaises(runner.BonsaiResponseError),
        ):
            session.generate(self.request)
        self.assertEqual(768, session.last_diagnostics["completion_tokens_reported"])
        with (
            patch.object(runner, "_http", side_effect=OSError("synthetic transport error")),
            self.assertRaises(OSError),
        ):
            session.generate(self.request)
        self.assertIsNone(session.last_diagnostics)


if __name__ == "__main__":
    unittest.main()
