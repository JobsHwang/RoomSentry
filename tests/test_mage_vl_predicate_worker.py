from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from baby_monitor.local_runners.mage_vl_4b import OFFICIAL_REVISION
from baby_monitor.local_runners.mage_vl_predicate import ATOMIC_MODEL_ID, PROMPT_VERSION
from baby_monitor.local_runners.mage_vl_predicate_worker import (
    MagePredicateWorkerError,
    _serve_one,
    _strict_control,
    main,
    serve,
)
from baby_monitor.visual_predicate import (
    VisualPredicate,
    VisualPredicateAnswer,
    VisualPredicateResult,
)


def _result(event_id: str = "public_event_01") -> VisualPredicateResult:
    return VisualPredicateResult(
        event_id=event_id,
        predicate=VisualPredicate.FACE_VISIBILITY,
        answer=VisualPredicateAnswer.CLEAR,
        confidence=0.5,
        evidence_frames=("t00_full",),
        model_id=ATOMIC_MODEL_ID,
        model_revision=OFFICIAL_REVISION,
        prompt_version=PROMPT_VERSION,
    )


def _control(path: str = "/private/request.json") -> bytes:
    return json.dumps(
        {"version": 1, "request_path": path},
        separators=(",", ":"),
    ).encode("utf-8")


class MageVLPredicateWorkerTests(unittest.TestCase):
    def test_control_is_strict_absolute_and_rejects_duplicates(self) -> None:
        self.assertEqual("/private/request.json", _strict_control(_control())["request_path"])
        for invalid in (
            b'{"version":1,"version":1,"request_path":"/private/request.json"}',
            b'{"version":1,"request_path":"relative.json"}',
            b'{"version":1,"request_path":"/private/request.json","extra":1}',
            b"not-json",
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(MagePredicateWorkerError, "control"),
            ):
                _strict_control(invalid)

    def test_one_request_returns_only_bound_result_or_fixed_error(self) -> None:
        session = unittest.mock.Mock()
        session.assess.return_value = _result()
        with patch(
            "baby_monitor.local_runners.mage_vl_predicate_worker.load_predicate_request",
            return_value=object(),
        ):
            response = _serve_one(session, _control())
        self.assertEqual("ok", response["status"])
        self.assertEqual(_result().to_dict(), response["result"])

        with patch(
            "baby_monitor.local_runners.mage_vl_predicate_worker.load_predicate_request",
            side_effect=MagePredicateWorkerError("private path must not escape"),
        ):
            error = _serve_one(session, _control())
        self.assertEqual(
            {"version": 1, "status": "error", "error_code": "runner_failed"},
            error,
        )
        self.assertNotIn("path", json.dumps(error))

    def test_serve_is_bounded_and_emits_one_ready_then_one_result_per_job(self) -> None:
        output = io.StringIO()
        session = unittest.mock.Mock()
        session.assess.return_value = _result()
        with (
            patch(
                "baby_monitor.local_runners.mage_vl_predicate_worker.load_predicate_request",
                return_value=object(),
            ),
            patch(
                "baby_monitor.local_runners.mage_vl_predicate_worker._readline_with_timeout",
                side_effect=(_control(), _control(), _control()),
            ),
        ):
            completed = serve(
                session,
                input_stream=io.BytesIO(),
                output_stream=output,
                max_jobs=2,
                idle_exit_seconds=30.0,
                max_runtime_seconds=300.0,
            )
        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(2, completed)
        self.assertEqual(("ready", "ok", "ok"), tuple(item["status"] for item in lines))
        self.assertEqual(2, session.assess.call_count)
        self.assertEqual(2, lines[0]["max_jobs"])

    def test_main_requires_confirmation_before_manifest_or_model_access(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch(
                "baby_monitor.local_runners.mage_vl_predicate_worker.TrustedModelCodeManifest.read"
            ) as manifest_read,
            patch("sys.stdout", stdout),
            patch("sys.stderr", stderr),
        ):
            code = main(
                [
                    "--model-path",
                    "/missing/model",
                    "--code-manifest",
                    "/missing/manifest.json",
                ]
            )
        self.assertEqual(2, code)
        self.assertEqual("", stdout.getvalue())
        self.assertEqual("runner_failed", json.loads(stderr.getvalue())["error_code"])
        manifest_read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
