from __future__ import annotations

import json
import os
import socket
import threading
import unittest
import uuid
from contextlib import ExitStack, redirect_stderr
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from baby_monitor.domain import EventMedia, SchemaError
from baby_monitor.live_detector_protocol import receive_packet, send_packet
from baby_monitor.local_runners import bonsai_27b as runner
from baby_monitor.local_runners import live_bonsai as live
from tests import test_bonsai_runner as fixtures


class BonsaiSessionTests(unittest.TestCase):
    setUp = fixtures.BonsaiRunnerTests.setUp

    def mocked_server(self, responses):
        stack = ExitStack()
        self.addCleanup(stack.close)
        process = Mock()
        process.poll.return_value = None
        isolate = stack.enter_context(patch.object(runner, "_isolated_loopback"))
        verify = stack.enter_context(patch.object(runner.BonsaiSettings, "verify_assets"))
        popen = stack.enter_context(patch.object(runner.subprocess, "Popen", return_value=process))
        http = stack.enter_context(patch.object(runner, "_http", side_effect=responses))
        return process, isolate, verify, popen, http

    def test_one_server_and_verification_for_independent_requests(self):
        other_frame = self.root / "other.png"
        other_frame.write_bytes(self.frame.read_bytes() + b"different synthetic pixels")
        other = replace(
            self.request,
            frame_paths=(other_frame,),
            evidence=EventMedia("", (str(other_frame),)),
            question="第二个独立问题。",
        )
        process, isolate, verify, popen, http = self.mocked_server(
            [{"status": "ok"}, fixtures.envelope(), fixtures.envelope(fixtures.event("other.png"))]
        )
        with runner.BonsaiSession(self.settings) as session:
            first = json.loads(session.generate(self.request))
            second = json.loads(session.generate(other))
        self.assertEqual(["frame.png"], first["evidence_frames"])
        self.assertEqual(["other.png"], second["evidence_frames"])
        isolate.assert_called_once()
        verify.assert_called_once()
        popen.assert_called_once()
        self.assertEqual(3, http.call_count)
        first_payload, second_payload = [call.args[1] for call in http.call_args_list[1:]]
        for payload in (first_payload, second_payload):
            self.assertIs(payload["cache_prompt"], False)
            self.assertEqual(1, len(payload["messages"]))
            self.assertEqual(self.settings.max_new_tokens, payload["max_tokens"])
        self.assertNotEqual(first_payload["messages"], second_payload["messages"])
        self.assertNotIn("描述画面。", json.dumps(second_payload, ensure_ascii=False))
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10)

    def test_one_shot_retains_payload_and_loads_once(self):
        _, _, verify, popen, http = self.mocked_server([{"status": "ok"}, fixtures.envelope()])
        runner.generate(self.settings, self.request)
        verify.assert_called_once()
        popen.assert_called_once()
        self.assertNotIn("cache_prompt", http.call_args.args[1])

    def test_token_limit_can_be_followed_by_new_request(self):
        self.mocked_server(
            [{"status": "ok"}, fixtures.envelope(finish="length"), fixtures.envelope()]
        )
        with runner.BonsaiSession(self.settings) as session:
            with self.assertRaises(runner.BonsaiResponseError) as caught:
                session.generate(self.request)
            self.assertEqual("bonsai_output_token_limit", caught.exception.code)
            self.assertIsInstance(json.loads(session.generate(self.request)), dict)

    def test_server_exit_between_requests_detected_without_http(self):
        process, _, _, _, http = self.mocked_server([{"status": "ok"}])
        with runner.BonsaiSession(self.settings) as session:
            process.poll.return_value = 1
            with self.assertRaises(runner.BonsaiResponseError) as caught:
                session.generate(self.request)
            self.assertEqual("bonsai_server_exited", caught.exception.code)
        self.assertEqual(1, http.call_count)

    def test_startup_failure_and_interrupt_reap_server(self):
        for failure in (RuntimeError("private failure"), KeyboardInterrupt()):
            with self.subTest(kind=type(failure).__name__):
                process, *_ = self.mocked_server([failure])
                with self.assertRaises(type(failure)), runner.BonsaiSession(self.settings):
                    self.fail("startup should fail")
                process.terminate.assert_called_once()

    def test_close_idempotent_and_closed_session_cannot_generate_or_reenter(self):
        process, *_ = self.mocked_server([{"status": "ok"}])
        with runner.BonsaiSession(self.settings) as session:
            pass
        session.close()
        process.terminate.assert_called_once()
        with self.assertRaises(runner.BonsaiResponseError):
            session.generate(self.request)
        with self.assertRaises(runner.LocalVLMRunnerError):
            session.__enter__()

    def test_bad_cache_setting_rejected(self):
        for value in (0, 1, "false", None):
            with self.assertRaises(ValueError):
                runner.BonsaiSession(self.settings, disable_prompt_cache=value)

    def test_server_exit_during_response_is_detected(self):
        process, _, _, _, http = self.mocked_server([{"status": "ok"}])
        with runner.BonsaiSession(self.settings) as session:

            def exited(*_args):
                process.poll.return_value = 1
                return fixtures.envelope()

            http.side_effect = exited
            with self.assertRaises(runner.BonsaiResponseError) as caught:
                session.generate(self.request)
            self.assertEqual("bonsai_server_exited", caught.exception.code)

    def test_verification_failure_never_starts_process(self):
        _, _, verify, popen, _ = self.mocked_server([])
        verify.side_effect = ValueError("synthetic mismatch")
        with self.assertRaises(ValueError), runner.BonsaiSession(self.settings):
            self.fail("must not enter")
        popen.assert_not_called()

    def test_previous_evidence_name_is_invalid_for_next_request(self):
        other_frame = self.root / "other.png"
        other_frame.write_bytes(self.frame.read_bytes())
        other = replace(self.request, frame_paths=(other_frame,))
        process, *_ = self.mocked_server(
            [{"status": "ok"}, fixtures.envelope(), fixtures.envelope()]
        )
        with self.assertRaises(SchemaError), runner.BonsaiSession(self.settings) as session:
            session.generate(self.request)
            session.generate(other)
        process.terminate.assert_called_once()

    def test_health_timeout_closes_server(self):
        process, *_ = self.mocked_server([{"status": "loading"}])
        with (
            patch.object(runner.time, "monotonic", side_effect=[0, 0, 61]),
            patch.object(runner.time, "sleep"),
            self.assertRaises(runner.BonsaiResponseError) as caught,
            runner.BonsaiSession(self.settings),
        ):
            self.fail("timeout must not enter")
        self.assertEqual("bonsai_server_load_timeout", caught.exception.code)
        process.terminate.assert_called_once()

    def test_execution_lease_passed_to_server_and_not_closed_by_session(self):
        path = self.root / "execution.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        self.addCleanup(os.close, descriptor)
        _, _, _, popen, _ = self.mocked_server([{"status": "ok"}])
        with runner.BonsaiSession(self.settings, execution_lock_fd=descriptor):
            pass
        self.assertEqual((descriptor,), popen.call_args.kwargs["pass_fds"])
        self.assertEqual(1, os.fstat(descriptor).st_nlink)

    def test_execution_lease_rejects_invalid_types_or_metadata(self):
        for value in (True, -1, "3", 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.BonsaiSession(self.settings, execution_lock_fd=value)
        safe = {"st_mode": 0o100600, "st_uid": os.getuid(), "st_nlink": 1}
        for field, value in (
            ("st_mode", 0o100644),
            ("st_mode", 0o100400),
            ("st_mode", 0o100700),
            ("st_mode", 0o010600),
            ("st_uid", os.getuid() + 1),
            ("st_nlink", 2),
        ):
            with (
                self.subTest(field=field, value=value),
                patch.object(
                    runner.os, "fstat", return_value=SimpleNamespace(**(safe | {field: value}))
                ),
                self.assertRaises(ValueError),
            ):
                runner.BonsaiSession(self.settings, execution_lock_fd=17)

    def test_execution_lease_closed_after_construction_prevents_server_start(self):
        path = self.root / "execution.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        session = runner.BonsaiSession(self.settings, execution_lock_fd=descriptor)
        os.close(descriptor)
        _, _, _, popen, _ = self.mocked_server([])
        with self.assertRaises(OSError), session:
            self.fail("closed lease must fail")
        popen.assert_not_called()


class LiveBonsaiProtocolTests(unittest.TestCase):
    setUp = fixtures.BonsaiRunnerTests.setUp

    def start_runner(self, responses, *, execution_lock_fd=None):
        parent, child = socket.socketpair()
        parent.settimeout(3)
        child.settimeout(3)
        self.addCleanup(parent.close)
        self.addCleanup(child.close)
        stack = ExitStack()
        self.addCleanup(stack.close)
        session_class = stack.enter_context(patch.object(live, "BonsaiSession"))
        self.session = session_class.return_value.__enter__.return_value
        self.session.generate.side_effect = responses
        stack.enter_context(patch.object(live, "_load_private_request", return_value=self.request))
        errors = []

        def worker():
            try:
                with child:
                    live.run(child, self.settings, execution_lock_fd=execution_lock_fd)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        self.addCleanup(thread.join, 4)
        ready, payload = receive_packet(parent)
        self.assertEqual(b"", payload)
        self.assertEqual(
            {
                "type": "ready",
                "model_revision": runner.MODEL_REVISION,
                "prompt_revision": runner.PROMPT_VERSION,
                "model_loads": 1,
                "network_enabled": False,
                "subject_probe_revision": live.SUBJECT_PROBE_VERSION,
                "subject_state_revision": live.SUBJECT_STATE_VERSION,
            },
            ready,
        )
        return parent, thread, session_class, errors

    def analyze(self, parent, **overrides):
        header = {
            "type": "analyze",
            "request_id": uuid.uuid4().hex,
            "request_path": str(self.root / "request.json"),
        }
        header.update(overrides)
        send_packet(parent, header)
        result, payload = receive_packet(parent)
        self.assertEqual(b"", payload)
        return header, result

    def test_two_requests_matched_then_stop_closes(self):
        parent, thread, session_class, errors = self.start_runner([fixtures.event()] * 2)
        for _ in range(2):
            request, result = self.analyze(parent)
            self.assertEqual(request["request_id"], result["request_id"])
            self.assertEqual("result", result["type"])
            self.assertEqual(json.loads(fixtures.event()), result["result"])
        send_packet(parent, {"type": "stop"})
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)
        session_class.return_value.__exit__.assert_called_once()

    def test_explicit_probe_has_separate_result_type_and_no_implicit_event_call(self):
        parent, thread, session_class, errors = self.start_runner([fixtures.event()])
        self.session.probe_subjects.return_value = {"claims": {"people": [], "uncertainties": []}}
        _, result = self.analyze(parent, type="probe_subjects")
        self.assertEqual("subject_result", result["type"])
        self.session.generate.assert_not_called()
        self.session.probe_subjects.assert_called_once_with(self.request)
        _, result = self.analyze(parent)
        self.assertEqual("result", result["type"])
        send_packet(parent, {"type": "stop"})
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)
        session_class.return_value.__exit__.assert_called_once()

    def test_state_probe_uses_same_resident_session_and_explicit_capability(self):
        parent, thread, session_class, errors = self.start_runner([])
        self.session.probe_subject_state.return_value = {
            "version": 2,
            "claims": {"people": [], "uncertainties": []},
        }
        _, result = self.analyze(parent, type="probe_subject_state")
        self.assertEqual("subject_result", result["type"])
        self.assertEqual(2, result["result"]["version"])
        self.session.probe_subject_state.assert_called_once_with(self.request)
        self.session.generate.assert_not_called()
        self.session.probe_subjects.assert_not_called()
        send_packet(parent, {"type": "stop"})
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)
        session_class.return_value.__exit__.assert_called_once()

    def test_token_limit_safe_error_then_continues(self):
        parent, thread, _, errors = self.start_runner(
            [runner.BonsaiResponseError("bonsai_output_token_limit"), fixtures.event()]
        )
        _, result = self.analyze(parent)
        self.assertEqual("bonsai_output_token_limit", result["error_code"])
        self.assertIs(False, result["fatal"])
        _, result = self.analyze(parent)
        self.assertEqual("result", result["type"])
        send_packet(parent, {"type": "stop"})
        thread.join(3)
        self.assertEqual([], errors)

    def test_fatal_model_exception_sanitized_and_closes(self):
        parent, thread, session_class, errors = self.start_runner(
            [RuntimeError("private household pathname and model text")]
        )
        _, result = self.analyze(parent)
        self.assertEqual("runner_failed", result["error_code"])
        self.assertIs(True, result["fatal"])
        self.assertNotIn("private", json.dumps(result))
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)
        session_class.return_value.__exit__.assert_called_once()

    def test_malformed_request_id_not_echoed(self):
        parent, thread, _, errors = self.start_runner([])
        _, result = self.analyze(parent, request_id="private arbitrary value")
        self.assertEqual("bonsai_invalid_protocol", result["error_code"])
        self.assertIsNone(result["request_id"])
        self.assertNotIn("private", json.dumps(result))
        thread.join(3)
        self.assertEqual([], errors)
        self.session.generate.assert_not_called()

    def test_nonempty_payload_refused(self):
        parent, thread, _, errors = self.start_runner([])
        send_packet(
            parent,
            {"type": "analyze", "request_id": uuid.uuid4().hex, "request_path": "/synthetic"},
            b"unexpected",
        )
        result, _ = receive_packet(parent)
        self.assertEqual("bonsai_invalid_protocol", result["error_code"])
        thread.join(3)
        self.assertEqual([], errors)

    def test_eof_reaps_session(self):
        parent, thread, session_class, errors = self.start_runner([])
        parent.close()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual([], errors)
        session_class.return_value.__exit__.assert_called_once()

    def test_run_forwards_execution_lock_descriptor(self):
        parent, thread, session_class, errors = self.start_runner([], execution_lock_fd=17)
        session_class.assert_called_once_with(self.settings, execution_lock_fd=17)
        send_packet(parent, {"type": "stop"})
        thread.join(3)
        self.assertEqual([], errors)

    def test_cli_requires_and_forwards_execution_lock_descriptor(self):
        args = [
            "--fd",
            "18",
            "--model-directory",
            str(self.models),
            "--runtime-directory",
            str(self.runtime),
            "--runtime-manifest",
            str(self.manifest),
            "--cuda-library-directory",
            str(self.cuda),
        ]
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
            live.main(args)
        self.assertEqual(2, caught.exception.code)
        with patch.object(live.socket, "socket") as channel, patch.object(live, "run") as run:
            self.assertEqual(0, live.main(args + ["--execution-lock-fd", "17"]))
        self.assertEqual(17, run.call_args.kwargs["execution_lock_fd"])
        self.assertIs(channel.return_value.__enter__.return_value, run.call_args.args[0])


class PrivateRequestTests(unittest.TestCase):
    setUp = fixtures.BonsaiRunnerTests.setUp

    def request_file(self):
        path = self.root / "request.json"
        path.write_text(
            json.dumps(
                {
                    "candidate": self.request.candidate,
                    "evidence": self.request.evidence.to_dict(),
                    "frame_paths": [str(self.frame)],
                    "question": self.request.question,
                    "output_schema": self.request.output_schema,
                }
            )
        )
        path.chmod(0o600)
        return path

    def test_private_request_loads_through_admitted_fd(self):
        self.assertEqual(self.request, live._load_private_request(str(self.request_file())))

    def test_public_symlink_fifo_hardlink_and_oversize_rejected(self):
        path = self.request_file()
        path.chmod(0o644)
        with self.assertRaises(ValueError):
            live._load_private_request(str(path))
        path.chmod(0o600)
        link = self.root / "link.json"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            live._load_private_request(str(link))
        hardlink = self.root / "hard.json"
        os.link(path, hardlink)
        with self.assertRaises(ValueError):
            live._load_private_request(str(hardlink))
        fifo = self.root / "fifo"
        os.mkfifo(fifo, 0o600)
        with self.assertRaises(ValueError):
            live._load_private_request(str(fifo))
        large = self.root / "large.json"
        large.write_bytes(b" " * (live.MAX_REQUEST_BYTES + 1))
        large.chmod(0o600)
        with self.assertRaises(ValueError):
            live._load_private_request(str(large))

    def test_noncanonical_relative_nonstrings_rejected(self):
        for value in ("request.json", "/tmp/../request.json", None, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                live._load_private_request(value)


if __name__ == "__main__":
    unittest.main()
