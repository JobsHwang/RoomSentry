from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import patch

from baby_monitor.adapters.live_vlm import PersistentLocalVLM
from baby_monitor.adapters.vlm import VLMError, VLMRequest
from baby_monitor.domain import EventMedia
from baby_monitor.live_detector_protocol import send_packet
from baby_monitor.local_runners import bonsai_27b as runner
from tests import test_bonsai_runner as fixtures
from tests import test_live_bonsai_runner as worker_fixtures
from tests import test_live_vlm as adapter_fixtures


def diagnostics(*, finish="stop"):
    return {
        "context_tokens_configured": 4096,
        "image_tokens_configured": 1024,
        "max_new_tokens_configured": 768,
        "prompt_tokens_reported": 2750,
        "completion_tokens_reported": 768 if finish == "length" else 490,
        "total_tokens_reported": 3518 if finish == "length" else 3240,
        "cached_tokens_reported": 0,
        "image_tokens_reported": None,
        "prompt_tokens_timed": 2750,
        "cached_tokens_timed": 0,
        "predicted_tokens_timed": 768 if finish == "length" else 490,
        "prompt_ms_reported": 1234.5,
        "predicted_ms_reported": 8000.0,
        "context_truncated_reported": None,
        "finish_reason_reported": finish,
    }


class LiveBonsaiDiagnosticsTests(unittest.TestCase):
    setUp = fixtures.BonsaiRunnerTests.setUp
    start_runner = worker_fixtures.LiveBonsaiProtocolTests.start_runner
    analyze = worker_fixtures.LiveBonsaiProtocolTests.analyze

    def test_success_and_length_error_keep_request_bound_numeric_diagnostics(self):
        parent, thread, _, errors = self.start_runner([])
        attempts = iter(("stop", "length", None))

        def generate(_request):
            finish = next(attempts)
            if finish is not None:
                self.session.last_diagnostics = diagnostics(finish=finish)
            if finish == "length":
                raise runner.BonsaiResponseError("bonsai_output_token_limit")
            return fixtures.event()

        self.session.generate.side_effect = generate
        for finish in ("stop", "length", None):
            request, reply = self.analyze(parent)
            self.assertEqual(request["request_id"], reply["request_id"])
            if finish is not None:
                self.assertEqual(diagnostics(finish=finish), reply["diagnostics"])
            else:
                self.assertNotIn("diagnostics", reply)
            if finish == "length":
                self.assertEqual("bonsai_output_token_limit", reply["error_code"])
                self.assertIs(False, reply["fatal"])
            else:
                self.assertEqual(json.loads(fixtures.event()), reply["result"])
        send_packet(parent, {"type": "stop"})
        thread.join(3)
        self.assertEqual([], errors)

    def test_legacy_session_cannot_reuse_previous_metadata(self):
        parent, thread, _, errors = self.start_runner([fixtures.event()])
        self.session.last_diagnostics = diagnostics()
        _, reply = self.analyze(parent)
        self.assertNotIn("diagnostics", reply)
        self.assertIsNone(self.session.last_diagnostics)
        send_packet(parent, {"type": "stop"})
        thread.join(3)
        self.assertEqual([], errors)

    def test_fatal_generation_failure_also_keeps_only_valid_metadata(self):
        parent, thread, _, errors = self.start_runner([])

        def generate(_request):
            self.session.last_diagnostics = diagnostics()
            raise RuntimeError("private model output must not be echoed")

        self.session.generate.side_effect = generate
        _, reply = self.analyze(parent)
        self.assertEqual("runner_failed", reply["error_code"])
        self.assertIs(True, reply["fatal"])
        self.assertEqual(diagnostics(), reply["diagnostics"])
        self.assertNotIn("private", json.dumps(reply))
        thread.join(3)
        self.assertEqual([], errors)

    def test_invalid_session_metadata_fails_closed_without_echoing_it(self):
        for bad in (
            {**diagnostics(), "prompt": "private prompt"},
            {**diagnostics(), "prompt_tokens_reported": True},
            {**diagnostics(), "prompt_ms_reported": 10**400},
            {**diagnostics(), "finish_reason_reported": "private model output"},
            {"prompt_tokens_reported": 1},
            "private raw HTTP response",
        ):
            with self.subTest(kind=type(bad).__name__):
                parent, thread, _, errors = self.start_runner([])

                def generate(_request, metadata=bad):
                    self.session.last_diagnostics = metadata
                    return fixtures.event()

                self.session.generate.side_effect = generate
                _, reply = self.analyze(parent)
                self.assertEqual("runner_failed", reply["error_code"])
                self.assertIs(True, reply["fatal"])
                self.assertNotIn("diagnostics", reply)
                self.assertNotIn("private", json.dumps(reply))
                thread.join(3)
                self.assertEqual([], errors)


@unittest.skipUnless(os.name == "posix", "resident IPC requires POSIX")
class PersistentVLMDiagnosticsTests(unittest.TestCase):
    setUp = adapter_fixtures.PersistentVLMTests.setUp
    request = staticmethod(adapter_fixtures.PersistentVLMTests.request)
    assert_gate_free = adapter_fixtures.PersistentVLMTests.assert_gate_free

    def adapter(self, replies):
        program = """
import socket,sys
from pathlib import Path
from baby_monitor.live_detector_protocol import receive_packet,send_packet
channel=socket.socket(fileno=int(sys.argv[-1]))
send_packet(channel,dict(type='ready',model_revision='revision',prompt_revision='prompt',
                        model_loads=1,network_enabled=False))
for template in REPLIES:
 request,payload=receive_packet(channel)
 if request['type']=='stop': break
 reply=dict(template)
 reply['request_id']='wrong' if reply.pop('wrong_id',False) else request['request_id']
 if reply.get('type')=='result':
  import json
  value=json.loads(Path(request['request_path']).read_text())
  result=dict(EVENT)
  result['evidence_frames']=['wrong.png' if reply.pop('wrong_evidence',False)
                             else Path(value['frame_paths'][0]).name]
  reply['result']=result
 send_packet(channel,reply)
request,payload=receive_packet(channel)
channel.close()
"""
        program = f"REPLIES={replies!r}\nEVENT={json.loads(fixtures.event())!r}\n" + program
        adapter = PersistentLocalVLM(
            command=(sys.executable, "-c", program),
            model_revision="revision",
            prompt_revision="prompt",
            startup_timeout_seconds=2,
            request_timeout_seconds=2,
        )
        self.addCleanup(adapter.close)
        return adapter

    def test_result_and_recoverable_length_metadata_are_reset_for_legacy_reply(self):
        adapter = self.adapter(
            [
                {"type": "result", "diagnostics": diagnostics()},
                {
                    "type": "error",
                    "error_code": "bonsai_output_token_limit",
                    "fatal": False,
                    "diagnostics": diagnostics(finish="length"),
                },
                {"type": "result"},
            ]
        )
        first = adapter.analyze(self.request())
        self.assertEqual(diagnostics(), adapter.last_diagnostics)
        previous = adapter.last_diagnostics
        process = adapter.process
        with self.assertRaises(VLMError) as failure:
            adapter.analyze(self.request("second.png"))
        self.assertEqual("bonsai_output_token_limit", failure.exception.code)
        self.assertEqual(diagnostics(finish="length"), adapter.last_diagnostics)
        self.assertIsNot(previous, adapter.last_diagnostics)
        self.assertIs(process, adapter.process)
        third = adapter.analyze(self.request())
        self.assertEqual(first.to_dict(), third.to_dict())
        self.assertIsNone(adapter.last_diagnostics)

    def test_fatal_error_keeps_valid_metadata_after_process_cleanup(self):
        adapter = self.adapter(
            [
                {
                    "type": "error",
                    "error_code": "runner_oom",
                    "fatal": True,
                    "diagnostics": diagnostics(),
                }
            ]
        )
        with self.assertRaises(VLMError) as failure:
            adapter.analyze(self.request())
        self.assertEqual("runner_oom", failure.exception.code)
        self.assertEqual(diagnostics(), adapter.last_diagnostics)
        self.assertIsNone(adapter.process)
        self.assert_gate_free()

    def test_invalid_diagnostics_reject_result_and_recoverable_error(self):
        for bad in (
            {**diagnostics(), "prompt": "private prompt"},
            {**diagnostics(), "context_tokens_configured": True},
            {**diagnostics(), "completion_tokens_reported": True},
            {**diagnostics(), "predicted_ms_reported": 10**400},
            {**diagnostics(), "finish_reason_reported": "private worker text"},
            [],
        ):
            for kind in ("result", "error"):
                with self.subTest(kind=kind, bad_type=type(bad).__name__):
                    adapter = self.adapter(
                        [
                            {
                                "type": kind,
                                "diagnostics": bad,
                                "error_code": "bonsai_output_token_limit",
                                "fatal": False,
                            }
                        ]
                    )
                    with self.assertRaises(VLMError) as failure:
                        adapter.analyze(self.request())
                    self.assertEqual("resident_response_invalid", failure.exception.code)
                    self.assertNotIn("private", str(failure.exception))
                    self.assertIsNone(adapter.last_diagnostics)
                    self.assertFalse(adapter.ready)
                    self.assert_gate_free()

    def test_mismatched_request_or_evidence_never_publishes_metadata(self):
        for binding in ("wrong_id", "wrong_evidence"):
            with self.subTest(binding=binding):
                adapter = self.adapter(
                    [
                        {"type": "result", "diagnostics": diagnostics()},
                        {"type": "result", "diagnostics": diagnostics(), binding: True},
                    ]
                )
                adapter.analyze(self.request())
                self.assertIsNotNone(adapter.last_diagnostics)
                with self.assertRaises(VLMError) as failure:
                    adapter.analyze(self.request("second.png"))
                self.assertEqual("resident_binding_failed", failure.exception.code)
                self.assertIsNone(adapter.last_diagnostics)
                self.assertFalse(adapter.ready)

    def test_invalid_new_request_or_startup_failure_clears_old_diagnostics(self):
        adapter = self.adapter([{"type": "result", "diagnostics": diagnostics()}])
        adapter.analyze(self.request())
        bad_request = VLMRequest(self.request().candidate, EventMedia("", ("/a.png", "/b.png")))
        with self.assertRaises(VLMError):
            adapter.analyze(bad_request)
        self.assertIsNone(adapter.last_diagnostics)
        adapter.close()
        adapter.last_diagnostics = diagnostics()
        with (
            patch.object(adapter, "start", side_effect=VLMError("synthetic startup failure")),
            self.assertRaises(VLMError),
        ):
            adapter.analyze(self.request())
        self.assertIsNone(adapter.last_diagnostics)


if __name__ == "__main__":
    unittest.main()
