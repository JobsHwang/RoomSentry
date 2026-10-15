from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baby_monitor.adapters import vlm_execution as gate
from baby_monitor.adapters.live_vlm import PersistentLocalVLM
from baby_monitor.adapters.vlm import VLMError, VLMRequest
from baby_monitor.domain import CandidateEvent, EventMedia
from tests.test_bonsai_runner import event


@unittest.skipUnless(os.name == "posix", "resident IPC requires POSIX")
class PersistentVLMTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root.chmod(0o700)
        patched = patch.object(gate, "_execution_root", return_value=self.root)
        patched.start()
        self.addCleanup(patched.stop)

    def adapter(self, mode="ok", timeout=2):
        program = """
import hashlib,json,os,socket,sys,time
from pathlib import Path
from baby_monitor.live_detector_protocol import receive_packet,send_packet
channel=socket.socket(fileno=int(sys.argv[-1]))
ready=dict(type='ready',model_revision=REVISION,prompt_revision='prompt',
                         model_loads=1,network_enabled=False,
                         subject_probe_revision='subject_probe_v1_20261005')
if MODE!='state_no_capability': ready['subject_state_revision']='subject_state_probe_v2_20261012'
send_packet(channel,ready)
count=0
while True:
 request,payload=receive_packet(channel)
 if request['type']=='stop': break
 count+=1
 path=Path(request['request_path'])
 assert path.stat().st_mode & 0o777 == 0o600
 assert path.parent.stat().st_mode & 0o777 == 0o700
 value=json.loads(path.read_text())
 frame=Path(value['frame_paths'][0]).name
 if count==1: first_frame=frame
 if MODE=='timeout': time.sleep(10)
 if MODE=='oom':
  send_packet(channel,dict(type='error',request_id=request['request_id'],
              error_code='runner_oom',fatal=True))
  break
 if MODE=='length' and count==1:
  send_packet(channel,dict(type='error',request_id=request['request_id'],
              error_code='bonsai_output_token_limit',fatal=False))
  continue
 result=EVENT.copy()
 result['evidence_frames']=[first_frame if MODE=='stale' else frame]
 result['summary_zh']=str(count)
 result_id='wrong' if MODE=='binding' else request['request_id']
 response_type='result'
 if request['type'] in ('probe_subjects','probe_subject_state'):
  response_type='subject_result'
  result=dict(version=1,prompt_revision='subject_probe_v1_20261005',
    semantics='unverified_model_claims',frame_id=frame,
    input_sha256=hashlib.sha256(Path(value['frame_paths'][0]).read_bytes()).hexdigest(),
    coordinate_space='submitted_image_thousandths',claims=dict(people=[],uncertainties=[]))
  if request['type']=='probe_subject_state':
   result['version']=2
   result['prompt_revision']='subject_state_probe_v2_20261012'
  if MODE=='probe_hash': result['input_sha256']='0'*64
  if MODE=='probe_type': response_type='result'
  if MODE=='probe_change': Path(value['frame_paths'][0]).write_bytes(b'changed')
 send_packet(channel,dict(type=response_type,request_id=result_id,
             result=result),b'extra' if MODE=='payload' else b'')
channel.close()
"""
        program = (
            f"MODE={mode!r}\nREVISION={'wrong' if mode == 'identity' else 'revision'!r}\n"
            f"EVENT={json.loads(event())!r}\n" + program
        )
        adapter = PersistentLocalVLM(
            command=(sys.executable, "-c", program),
            model_revision="revision",
            prompt_revision="prompt",
            startup_timeout_seconds=2,
            request_timeout_seconds=timeout,
        )
        self.addCleanup(adapter.close)
        return adapter

    @staticmethod
    def request(name="frame.png"):
        return VLMRequest(
            CandidateEvent("fixture", "cam_01", 0, 0, ()), EventMedia("", ("/" + name,))
        )

    def assert_gate_free(self):
        with gate.local_vlm_execution("0"):
            pass

    def probe_request(self):
        path = self.root / "probe.png"
        path.write_bytes(b"synthetic image bytes")
        return VLMRequest(CandidateEvent("probe", "cam_01", 0, 0, ()), EventMedia("", (str(path),)))

    def test_explicit_probe_reuses_same_model_and_event_contract_stays_separate(self):
        adapter = self.adapter()
        request = self.probe_request()
        adapter.analyze(request)
        process = adapter.process
        result = adapter.probe_subjects(request)
        self.assertEqual("unverified_model_claims", result["semantics"])
        self.assertIs(process, adapter.process)
        self.assertEqual("3", adapter.analyze(request).summary_zh)
        self.assertIs(process, adapter.process)

    def test_probe_wrong_hash_type_or_changed_source_closes_session(self):
        for mode in ("probe_hash", "probe_type", "probe_change"):
            with self.subTest(mode=mode):
                adapter = self.adapter(mode)
                request = self.probe_request()
                with self.assertRaises(VLMError):
                    adapter.probe_subjects(request)
                self.assertFalse(adapter.ready)
                self.assert_gate_free()

    def test_state_probe_reuses_single_process_and_rejects_missing_capability(self):
        adapter = self.adapter()
        request = self.probe_request()
        adapter.analyze(request)
        process = adapter.process
        result = adapter.probe_subject_state(request)
        self.assertEqual(2, result["version"])
        self.assertIs(process, adapter.process)
        adapter.close()
        incompatible = self.adapter("state_no_capability")
        with self.assertRaises(VLMError):
            incompatible.probe_subject_state(request)
        self.assertFalse(incompatible.ready)
        self.assert_gate_free()

    def test_state_probe_rejects_hash_type_and_changed_source(self):
        for mode in ("probe_hash", "probe_type", "probe_change"):
            adapter = self.adapter(mode)
            with self.subTest(mode=mode), self.assertRaises(VLMError):
                adapter.probe_subject_state(self.probe_request())
            self.assertFalse(adapter.ready)
            self.assert_gate_free()

    def test_probe_unsafe_input_does_not_start_model(self):
        adapter = self.adapter()
        with self.assertRaises(VLMError):
            adapter.probe_subjects(self.request())
        self.assertIsNone(adapter.process)

    def test_two_requests_reuse_process_and_hold_gate_until_close(self):
        adapter = self.adapter()
        self.assertEqual("1", adapter.analyze(self.request()).summary_zh)
        process = adapter.process
        with self.assertRaises(gate.VLMExecutionError), gate.local_vlm_execution("0"):
            self.fail("resident model must keep its gate")
        analysis = adapter.analyze_with_metrics(self.request("second.png"))
        self.assertEqual("2", analysis.result.summary_zh)
        self.assertTrue(analysis.metrics.json_compliant)
        self.assertIs(process, adapter.process)
        self.assertTrue(adapter.ready)
        adapter.close()
        self.assertIsNotNone(process.poll())
        self.assertFalse(adapter.ready)
        self.assert_gate_free()

    def test_output_limit_is_failed_not_success_and_session_can_continue(self):
        adapter = self.adapter("length")
        with self.assertRaises(VLMError) as failure:
            adapter.analyze(self.request())
        self.assertEqual("bonsai_output_token_limit", failure.exception.code)
        self.assertTrue(adapter.ready)
        self.assertEqual("2", adapter.analyze(self.request()).summary_zh)

    def test_wrong_id_or_payload_closes_process_before_releasing_gate(self):
        for mode in ("binding", "payload"):
            with self.subTest(mode=mode):
                adapter = self.adapter(mode)
                with self.assertRaises(VLMError) as failure:
                    adapter.analyze(self.request())
                self.assertEqual("resident_binding_failed", failure.exception.code)
                self.assertFalse(adapter.ready)
                self.assertIsNone(adapter.process)
                self.assert_gate_free()

    def test_previous_request_evidence_cannot_leak_into_next_result(self):
        adapter = self.adapter("stale")
        adapter.analyze(self.request())
        with self.assertRaises(VLMError) as failure:
            adapter.analyze(self.request("different.png"))
        self.assertEqual("resident_binding_failed", failure.exception.code)
        self.assertFalse(adapter.ready)
        self.assert_gate_free()

    def test_identity_mismatch_does_not_leave_resident_or_gate(self):
        adapter = self.adapter("identity")
        with self.assertRaises(VLMError) as failure:
            adapter.start()
        self.assertEqual("resident_identity_mismatch", failure.exception.code)
        self.assertFalse(adapter.ready)
        self.assert_gate_free()

    def test_closed_set_failure_keeps_diagnostic_and_closes_session(self):
        adapter = self.adapter("oom")
        with self.assertRaises(VLMError) as failure:
            adapter.analyze(self.request())
        self.assertEqual("runner_oom", failure.exception.code)
        self.assertEqual("observed", failure.exception.oom_status)
        self.assertFalse(adapter.ready)
        self.assert_gate_free()

    def test_request_timeout_cleans_up_and_releases_gate(self):
        adapter = self.adapter("timeout", timeout=1)
        with self.assertRaises(VLMError) as failure:
            adapter.analyze(self.request())
        self.assertEqual("resident_request_failed", failure.exception.code)
        self.assertFalse(adapter.ready)
        self.assert_gate_free()

    def test_busy_gate_does_not_start_process(self):
        adapter = self.adapter()
        with (
            gate.local_vlm_execution("0"),
            patch("baby_monitor.adapters.live_vlm.subprocess.Popen") as popen,
            self.assertRaises(VLMError) as failure,
        ):
            adapter.start()
        self.assertEqual("local_vlm_busy", failure.exception.code)
        popen.assert_not_called()

    def test_invalid_configuration_is_rejected(self):
        for values in (
            {"command": ()},
            {"command": ("fixture", "{request}")},
            {"startup_timeout_seconds": True},
            {"request_timeout_seconds": 121},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PersistentLocalVLM(
                    **{
                        "command": ("fixture",),
                        "model_revision": "revision",
                        "prompt_revision": "prompt",
                        **values,
                    }
                )

    def test_multiframe_request_is_rejected_before_startup(self):
        adapter = self.adapter()
        request = VLMRequest(
            CandidateEvent("fixture", "cam_01", 0, 0, ()),
            EventMedia("", ("/one.png", "/two.png")),
        )
        with patch.object(adapter, "start") as start, self.assertRaises(VLMError):
            adapter.analyze(request)
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
