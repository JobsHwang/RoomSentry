from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from baby_monitor.adapters.visual_predicate import (
    CommandVisualPredicateAdapter,
    ConservativeVisualPredicateAdapter,
    ResidentCommandVisualPredicateAdapter,
    ResidentVisualPredicateHealth,
    VisualPredicateAdapterError,
    VisualPredicateFrame,
    VisualPredicateHealthError,
    VisualPredicateRequest,
)
from baby_monitor.visual_predicate import (
    VisualPredicate,
    VisualPredicateAnswer,
)

_REVISION = "5c78cab61938e73859b63724d9bf5cb88c477eaa"


def _write_private_frame(root: Path, name: str = "input.jpg") -> VisualPredicateFrame:
    payload = b"public-test-image-bytes"
    path = root / name
    path.write_bytes(payload)
    path.chmod(0o600)
    return VisualPredicateFrame(
        frame_id="frame_01",
        sha256=hashlib.sha256(payload).hexdigest(),
        media_type="image/jpeg",
        path=str(path),
    )


def _request(frame: VisualPredicateFrame) -> VisualPredicateRequest:
    return VisualPredicateRequest(
        event_id="event_01",
        predicate=VisualPredicate.FACE_VISIBILITY,
        frames=(frame,),
    )


def _adapter(code: str, *, timeout_seconds: int = 10) -> CommandVisualPredicateAdapter:
    return CommandVisualPredicateAdapter(
        command=(sys.executable, "-c", code, "{request}"),
        model_id="microsoft.Mage-VL",
        model_revision=_REVISION,
        prompt_version="atomic_face_v1",
        timeout_seconds=timeout_seconds,
    )


_SUCCESS_RUNNER = r"""
import json, os, pathlib, sys
request = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
frame = request["frames"][0]
path = pathlib.Path(frame["path"])
assert path.parent == pathlib.Path(sys.argv[1]).parent
assert path.name == "frame_00.jpg"
assert path.stat().st_mode & 0o077 == 0
assert "BABY_MONITOR_TEST_SECRET" not in os.environ
print(json.dumps({
    "version": 1,
    "event_id": request["event_id"],
    "predicate": request["predicate"],
    "answer": "not_visible",
    "confidence": 0.8,
    "evidence_frames": [frame["frame_id"]],
    "model_id": request["model_id"],
    "model_revision": request["model_revision"],
    "prompt_version": request["prompt_version"],
    "subject_identity_inference": False,
    "risk_inference": False,
    "notification_decision": False,
}, separators=(",", ":")))
"""

_RESIDENT_RUNNER = r"""
import json, os, pathlib, sys
counter = pathlib.Path(sys.argv[1])
with counter.open("a", encoding="utf-8") as handle:
    handle.write("start\n")
print(json.dumps({
    "version": 1,
    "status": "ready",
    "model_id": "microsoft.Mage-VL",
    "model_revision": "5c78cab61938e73859b63724d9bf5cb88c477eaa",
    "prompt_version": "atomic_face_v1",
    "max_jobs": 2,
    "idle_exit_seconds": 30.0,
    "max_runtime_seconds": 300.0,
}, separators=(",", ":")), flush=True)
for line in sys.stdin:
    control = json.loads(line)
    request_path = pathlib.Path(control["request_path"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    frame = request["frames"][0]
    staged = pathlib.Path(frame["path"])
    assert staged.parent == request_path.parent
    assert staged.stat().st_mode & 0o077 == 0
    assert "BABY_MONITOR_TEST_SECRET" not in os.environ
    result = {
        "version": 1,
        "event_id": request["event_id"],
        "predicate": request["predicate"],
        "answer": "not_visible",
        "confidence": 0.8,
        "evidence_frames": [frame["frame_id"]],
        "model_id": request["model_id"],
        "model_revision": request["model_revision"],
        "prompt_version": request["prompt_version"],
        "subject_identity_inference": False,
        "risk_inference": False,
        "notification_decision": False,
    }
    print(json.dumps({
        "version": 1,
        "status": "ok",
        "result": result,
    }, separators=(",", ":")), flush=True)
"""


def _resident_adapter(
    code: str,
    counter: Path,
    *,
    request_timeout_seconds: float = 10.0,
    health_path: Path | None = None,
) -> ResidentCommandVisualPredicateAdapter:
    return ResidentCommandVisualPredicateAdapter(
        command=(sys.executable, "-c", code, str(counter)),
        model_id="microsoft.Mage-VL",
        model_revision=_REVISION,
        prompt_version="atomic_face_v1",
        max_jobs=2,
        idle_exit_seconds=30.0,
        health_path=health_path,
        max_runtime_seconds=300.0,
        startup_timeout_seconds=10.0,
        request_timeout_seconds=request_timeout_seconds,
    )


class VisualPredicateAdapterTests(unittest.TestCase):
    def test_conservative_adapter_is_explicit_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            frame = _write_private_frame(root)
            adapter = ConservativeVisualPredicateAdapter(
                model_id="builtin.conservative",
                model_revision="f" * 40,
                prompt_version="unknown_v1",
            )
            result = adapter.assess(_request(frame))
            self.assertIs(result.answer, VisualPredicateAnswer.UNKNOWN)
            self.assertEqual(0.0, result.confidence)
            self.assertEqual((), result.evidence_frames)

    def test_command_adapter_stages_private_copy_and_filters_environment(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            frame = _write_private_frame(root)
            os.environ["BABY_MONITOR_TEST_SECRET"] = "must-not-leak"
            try:
                result = _adapter(_SUCCESS_RUNNER).assess(_request(frame))
            finally:
                os.environ.pop("BABY_MONITOR_TEST_SECRET", None)
            self.assertIs(result.answer, VisualPredicateAnswer.NOT_VISIBLE)
            self.assertEqual(("frame_01",), result.evidence_frames)
            self.assertNotIn(str(root), result.to_json())

    def test_frame_repr_hides_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            frame = _write_private_frame(Path(raw))
            self.assertNotIn(frame.path, repr(frame))

    def test_rejects_hash_mode_and_symlink_before_runner(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            frame = _write_private_frame(root)
            bad_hash = VisualPredicateFrame(
                frame_id=frame.frame_id,
                sha256="0" * 64,
                media_type=frame.media_type,
                path=frame.path,
            )
            with self.assertRaisesRegex(VisualPredicateAdapterError, "admission"):
                _adapter(_SUCCESS_RUNNER).assess(_request(bad_hash))

            Path(frame.path).chmod(0o644)
            with self.assertRaisesRegex(VisualPredicateAdapterError, "admission"):
                _adapter(_SUCCESS_RUNNER).assess(_request(frame))

            target = root / "target.jpg"
            target.write_bytes(b"target")
            target.chmod(0o600)
            link = root / "link.jpg"
            link.symlink_to(target)
            linked = VisualPredicateFrame(
                frame_id="frame_link",
                sha256=hashlib.sha256(b"target").hexdigest(),
                media_type="image/jpeg",
                path=str(link),
            )
            with self.assertRaisesRegex(VisualPredicateAdapterError, "admission"):
                _adapter(_SUCCESS_RUNNER).assess(_request(linked))

    def test_rejects_unbound_or_path_leaking_result(self) -> None:
        unbound = _SUCCESS_RUNNER.replace(
            '"event_id": request["event_id"]',
            '"event_id": "different_event"',
        )
        leaking = _SUCCESS_RUNNER.replace(
            '"notification_decision": False,',
            '"notification_decision": False, "source_path": frame["path"],',
        )
        with tempfile.TemporaryDirectory() as raw:
            frame = _write_private_frame(Path(raw))
            with self.assertRaisesRegex(VisualPredicateAdapterError, "binding"):
                _adapter(unbound).assess(_request(frame))
            with self.assertRaisesRegex(VisualPredicateAdapterError, "output"):
                _adapter(leaking).assess(_request(frame))

    def test_nonzero_failure_never_discloses_runner_stderr(self) -> None:
        code = "import sys; print('private/path.jpg', file=sys.stderr); raise SystemExit(7)"
        with tempfile.TemporaryDirectory() as raw:
            frame = _write_private_frame(Path(raw))
            with self.assertRaises(VisualPredicateAdapterError) as raised:
                _adapter(code).assess(_request(frame))
            self.assertEqual("runner_failed", raised.exception.code)
            self.assertNotIn("private", str(raised.exception))

    def test_timeout_is_bounded(self) -> None:
        code = "import time; time.sleep(30)"
        with tempfile.TemporaryDirectory() as raw:
            frame = _write_private_frame(Path(raw))
            with self.assertRaises(VisualPredicateAdapterError) as raised:
                _adapter(code, timeout_seconds=1).assess(_request(frame))
            self.assertEqual("runner_timeout", raised.exception.code)

    def test_request_rejects_duplicate_frame_ids(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = _write_private_frame(root, "first.jpg")
            second_path = root / "second.jpg"
            second_path.write_bytes(b"second")
            second_path.chmod(0o600)
            second = VisualPredicateFrame(
                frame_id=first.frame_id,
                sha256=hashlib.sha256(b"second").hexdigest(),
                media_type="image/jpeg",
                path=str(second_path),
            )
            with self.assertRaisesRegex(ValueError, "frames"):
                VisualPredicateRequest(
                    event_id="event_01",
                    predicate=VisualPredicate.FACE_VISIBILITY,
                    frames=(first, second),
                )

    def test_resident_adapter_reuses_then_restarts_at_the_job_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            frame = _write_private_frame(root)
            counter = root / "worker-starts.local"
            os.environ["BABY_MONITOR_TEST_SECRET"] = "must-not-leak"
            adapter = _resident_adapter(_RESIDENT_RUNNER, counter)
            initial = adapter.health_snapshot()
            self.assertEqual(
                ("stopped", 0, 0), (initial.state, initial.worker_starts, initial.successful_jobs)
            )
            self.assertFalse(initial.worker_running)
            try:
                answers = tuple(adapter.assess(_request(frame)).answer for _ in range(3))
            finally:
                adapter.close()
                os.environ.pop("BABY_MONITOR_TEST_SECRET", None)
            self.assertEqual((VisualPredicateAnswer.NOT_VISIBLE,) * 3, answers)
            self.assertEqual(["start", "start"], counter.read_text().splitlines())
            self.assertIsNone(adapter._process)
            health = adapter.health_snapshot()
            self.assertEqual("stopped", health.state)
            self.assertFalse(health.worker_running)
            self.assertEqual(2, health.worker_starts)
            self.assertEqual(1, health.worker_restarts)
            self.assertEqual(3, health.successful_jobs)
            self.assertEqual(0, health.failed_jobs)
            self.assertEqual(0, health.timeout_jobs)
            self.assertEqual(0, health.current_worker_jobs)
            self.assertIsNone(health.last_error_code)
            public = health.to_dict()
            self.assertFalse(public["paths_disclosed"])
            self.assertFalse(public["media_metadata_included"])
            self.assertNotIn(str(root), str(public))

    def test_resident_adapter_timeout_kills_worker_and_hides_output(self) -> None:
        slow = _RESIDENT_RUNNER.replace(
            "for line in sys.stdin:\n",
            "for line in sys.stdin:\n    import time; time.sleep(30)\n",
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            adapter = _resident_adapter(
                slow,
                root / "worker-starts.local",
                request_timeout_seconds=1.0,
                health_path=root / "timeout-health.json",
            )
            with self.assertRaises(VisualPredicateAdapterError) as raised:
                adapter.assess(_request(_write_private_frame(root)))
            self.assertEqual("worker_timeout", raised.exception.code)
            self.assertIsNone(adapter._process)
            health = adapter.health_snapshot()
            self.assertEqual("failed", health.state)
            self.assertFalse(health.worker_running)
            self.assertEqual(1, health.worker_starts)
            self.assertEqual(0, health.successful_jobs)
            self.assertEqual(1, health.failed_jobs)
            self.assertEqual(1, health.timeout_jobs)
            self.assertEqual("worker_timeout", health.last_error_code)
            persisted = ResidentVisualPredicateHealth.read_private(root / "timeout-health.json")
            self.assertEqual("failed", persisted.state)
            self.assertEqual("worker_timeout", persisted.last_error_code)
            self.assertEqual(1, persisted.failed_jobs)
            self.assertEqual(1, persisted.timeout_jobs)

    def test_resident_adapter_rejects_unbound_result_and_bad_handshake(self) -> None:
        unbound = _RESIDENT_RUNNER.replace(
            '"event_id": request["event_id"]',
            '"event_id": "different_event"',
        )
        bad_ready = _RESIDENT_RUNNER.replace('"max_jobs": 2', '"max_jobs": 99')
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            frame = _write_private_frame(root)
            with self.assertRaises(VisualPredicateAdapterError) as raised:
                _resident_adapter(unbound, root / "unbound.local").assess(_request(frame))
            self.assertEqual("result_binding_failed", raised.exception.code)

            with self.assertRaises(VisualPredicateAdapterError) as raised:
                _resident_adapter(bad_ready, root / "bad-ready.local").assess(_request(frame))
            self.assertEqual("invalid_worker_output", raised.exception.code)

    def test_resident_health_is_private_atomic_and_tracks_stable_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            health_path = root / "resident-health.json"
            adapter = _resident_adapter(
                _RESIDENT_RUNNER,
                root / "worker-starts.local",
                health_path=health_path,
            )
            initial = ResidentVisualPredicateHealth.read_private(health_path)
            self.assertEqual("stopped", initial.state)
            self.assertFalse(initial.worker_running)
            self.assertEqual(0o600, health_path.stat().st_mode & 0o777)

            adapter.assess(_request(_write_private_frame(root)))
            ready = ResidentVisualPredicateHealth.read_private(health_path)
            self.assertEqual("ready", ready.state)
            self.assertTrue(ready.worker_running)
            self.assertEqual(1, ready.worker_starts)
            self.assertEqual(1, ready.successful_jobs)
            self.assertEqual(1, ready.current_worker_jobs)
            self.assertNotIn(str(root), str(ready.to_dict()))

            adapter.close()
            payload = json.loads(health_path.read_text(encoding="utf-8"))
            schema = json.loads(
                Path("schemas/resident-visual-predicate-health-v1.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(set(schema["required"]), set(payload))
            self.assertEqual(1, schema["properties"]["version"]["const"])
            stopped = ResidentVisualPredicateHealth.read_private(health_path)
            self.assertEqual("stopped", stopped.state)
            self.assertFalse(stopped.worker_running)
            self.assertEqual(1, stopped.successful_jobs)
            self.assertEqual(0, stopped.current_worker_jobs)

    def test_resident_health_rejects_weak_roots_links_and_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            root.chmod(0o755)
            with self.assertRaisesRegex(VisualPredicateHealthError, "path"):
                _resident_adapter(
                    _RESIDENT_RUNNER,
                    root / "weak-counter.local",
                    health_path=root / "health.json",
                )

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            health_path = root / "health.json"
            health_path.write_text('{"version":1,"version":1}\n', encoding="utf-8")
            health_path.chmod(0o600)
            with self.assertRaisesRegex(VisualPredicateHealthError, "invalid"):
                ResidentVisualPredicateHealth.read_private(health_path)

            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            target.chmod(0o600)
            link = root / "linked.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(VisualPredicateHealthError, "unsafe"):
                ResidentVisualPredicateHealth.read_private(link)

            hardlink = root / "hardlink.json"
            os.link(target, hardlink)
            with self.assertRaisesRegex(VisualPredicateHealthError, "unsafe"):
                ResidentVisualPredicateHealth.read_private(target)

    def test_health_persistence_failure_stops_resident_worker(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            adapter = _resident_adapter(
                _RESIDENT_RUNNER,
                root / "worker-starts.local",
                health_path=root / "health.json",
            )
            frame = _write_private_frame(root)
            root.chmod(0o755)
            try:
                with self.assertRaises(VisualPredicateAdapterError) as raised:
                    adapter.assess(_request(frame))
            finally:
                root.chmod(0o700)
            self.assertEqual("health_persistence_failed", raised.exception.code)
            self.assertIsNone(adapter._process)


if __name__ == "__main__":
    unittest.main()
