from __future__ import annotations

import math
import unittest
from collections import UserDict
from types import MappingProxyType

from baby_monitor.live_vlm_protocol import VLMDiagnostics, validate_vlm_diagnostics


def diagnostics() -> VLMDiagnostics:
    return {
        "context_tokens_configured": 4096,
        "image_tokens_configured": 1024,
        "max_new_tokens_configured": 512,
        "prompt_tokens_reported": None,
        "completion_tokens_reported": None,
        "total_tokens_reported": None,
        "cached_tokens_reported": None,
        "image_tokens_reported": None,
        "prompt_tokens_timed": None,
        "cached_tokens_timed": None,
        "predicted_tokens_timed": None,
        "prompt_ms_reported": None,
        "predicted_ms_reported": None,
        "context_truncated_reported": None,
        "finish_reason_reported": None,
    }


class LiveVLMDiagnosticsProtocolTests(unittest.TestCase):
    def assert_invalid(self, value):
        with self.assertRaises(ValueError) as caught:
            validate_vlm_diagnostics(value)
        self.assertEqual("invalid VLM diagnostics", str(caught.exception))

    def test_none_stays_unknown(self):
        self.assertIsNone(validate_vlm_diagnostics(None))

    def test_valid_unknown_record_is_copied_without_mutating_source(self):
        source = diagnostics()
        original = source.copy()
        result = validate_vlm_diagnostics(source)
        self.assertEqual(original, source)
        self.assertEqual(source, result)
        self.assertIs(type(result), dict)
        self.assertIsNot(source, result)
        result["prompt_tokens_reported"] = 10
        self.assertIsNone(source["prompt_tokens_reported"])
        source["completion_tokens_reported"] = 20
        self.assertIsNone(result["completion_tokens_reported"])

    def test_non_plain_dictionary_records_are_rejected(self):
        class DictionarySubclass(dict):
            pass

        for value in (
            False,
            0,
            0.0,
            "private raw response",
            [],
            (),
            UserDict(diagnostics()),
            MappingProxyType(diagnostics()),
            DictionarySubclass(diagnostics()),
        ):
            with self.subTest(kind=type(value).__name__):
                self.assert_invalid(value)

    def test_unknown_and_private_fields_are_rejected_without_echoing_data(self):
        for field in ("prompt", "content", "raw_response", "token_ids", "unknown", 123):
            source = diagnostics()
            source[field] = "private prompt or response text"
            original = source.copy()
            with self.subTest(field=field):
                self.assert_invalid(source)
                self.assertEqual(original, source)

    def test_every_missing_field_is_rejected(self):
        for field in diagnostics():
            source = diagnostics()
            del source[field]
            with self.subTest(field=field):
                self.assert_invalid(source)
        self.assert_invalid({})

    def test_all_admitted_budget_combinations_and_output_boundaries(self):
        for context, image in (
            (4096, 1024),
            (4096, 2048),
            (8192, 1024),
            (8192, 2048),
            (8192, 4096),
        ):
            for output in (64, 512, 768, 1024):
                source = diagnostics()
                source.update(
                    context_tokens_configured=context,
                    image_tokens_configured=image,
                    max_new_tokens_configured=output,
                )
                with self.subTest(context=context, image=image, output=output):
                    if image + output + 1024 >= context:
                        with self.assertRaises(ValueError):
                            validate_vlm_diagnostics(source)
                    else:
                        self.assertEqual(source, validate_vlm_diagnostics(source))

    def test_malformed_configured_budgets_are_rejected(self):
        for field, invalid_values in (
            ("context_tokens_configured", (None, True, 4096.0, "4096", 0, 16384, 10**400)),
            ("image_tokens_configured", (None, False, 1024.0, "1024", 0, 512, 8192)),
            ("max_new_tokens_configured", (None, True, 512.0, "512", -1, 63, 1025, 10**400)),
        ):
            for invalid in invalid_values:
                source = diagnostics()
                source[field] = invalid
                with self.subTest(field=field, kind=type(invalid).__name__):
                    self.assert_invalid(source)

    def test_configuration_without_coarse_generation_room_is_rejected(self):
        source = diagnostics()
        source["image_tokens_configured"] = 4096
        self.assert_invalid(source)

    def test_all_count_fields_accept_only_bounded_exact_integers_or_none(self):
        count_fields = (
            "prompt_tokens_reported",
            "completion_tokens_reported",
            "total_tokens_reported",
            "cached_tokens_reported",
            "image_tokens_reported",
            "prompt_tokens_timed",
            "cached_tokens_timed",
            "predicted_tokens_timed",
        )
        for field in count_fields:
            for accepted in (None, 0, 1, 1_000_000):
                source = diagnostics()
                source[field] = accepted
                with self.subTest(field=field, accepted=accepted):
                    self.assertEqual(source, validate_vlm_diagnostics(source))
            for invalid in (True, False, -1, 1_000_001, 10**400, 1.0, "1", [], {}):
                source = diagnostics()
                source[field] = invalid
                with self.subTest(field=field, kind=type(invalid).__name__):
                    self.assert_invalid(source)

    def test_all_timings_accept_only_bounded_finite_exact_numbers_or_none(self):
        for field in ("prompt_ms_reported", "predicted_ms_reported"):
            for accepted in (None, 0, 0.0, 1.5, 600_000, 600_000.0):
                source = diagnostics()
                source[field] = accepted
                with self.subTest(field=field, accepted=accepted):
                    self.assertEqual(source, validate_vlm_diagnostics(source))
            for invalid in (
                True,
                False,
                -1,
                -0.1,
                600_001,
                600_000.1,
                10**400,
                -(10**400),
                math.nan,
                math.inf,
                -math.inf,
                "1",
                [],
                {},
            ):
                source = diagnostics()
                source[field] = invalid
                with self.subTest(field=field, kind=type(invalid).__name__):
                    self.assert_invalid(source)

    def test_context_truncation_remains_boolean_or_unknown(self):
        for accepted in (None, True, False):
            source = diagnostics()
            source["context_truncated_reported"] = accepted
            self.assertIs(accepted, validate_vlm_diagnostics(source)["context_truncated_reported"])
        for invalid in (0, 1, 0.0, "false", [], {}):
            source = diagnostics()
            source["context_truncated_reported"] = invalid
            self.assert_invalid(source)

    def test_finish_reason_is_a_closed_enum_or_unknown(self):
        for accepted in (None, "stop", "length", "tool_calls", "content_filter", "function_call"):
            source = diagnostics()
            source["finish_reason_reported"] = accepted
            self.assertEqual(accepted, validate_vlm_diagnostics(source)["finish_reason_reported"])
        for invalid in ("", "private model text", True, 1, [], {}):
            source = diagnostics()
            source["finish_reason_reported"] = invalid
            self.assert_invalid(source)

    def test_primitive_subclasses_are_not_protocol_values(self):
        class IntegerSubclass(int):
            pass

        class FloatSubclass(float):
            pass

        class StringSubclass(str):
            pass

        for field, invalid in (
            ("context_tokens_configured", IntegerSubclass(4096)),
            ("image_tokens_configured", IntegerSubclass(1024)),
            ("max_new_tokens_configured", IntegerSubclass(512)),
            ("prompt_tokens_reported", IntegerSubclass(12)),
            ("prompt_ms_reported", FloatSubclass(12.5)),
            ("finish_reason_reported", StringSubclass("stop")),
        ):
            source = diagnostics()
            source[field] = invalid
            with self.subTest(field=field):
                self.assert_invalid(source)
        source = diagnostics()
        source[StringSubclass("prompt_tokens_reported")] = source.pop("prompt_tokens_reported")
        self.assert_invalid(source)


if __name__ == "__main__":
    unittest.main()
