from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from baby_monitor.domain import EventMedia, EvidenceGap, VLMEvent
from baby_monitor.local_runners.common import load_request
from baby_monitor.local_runners.smolvlm2_2_2b import (
    MODEL_ID,
    MODEL_REVISION,
    SmolVLM2Settings,
    build_prompt,
    main,
)


def _valid_result(frame: str) -> str:
    return json.dumps(
        {
            "event_type": "unknown",
            "subjects": [],
            "observations": ["A red rectangle is visible on a light background."],
            "inferences": [],
            "risk_level": "none",
            "confidence": 0.2,
            "needs_remote_review": True,
            "summary_zh": "画面中可见一个红色矩形，无法判断家庭事件。",
            "evidence_frames": [frame],
        },
        ensure_ascii=False,
    )


class SmolVLM2RunnerTests(unittest.TestCase):
    def _model(self, root: Path) -> Path:
        model = root / "model"
        model.mkdir()
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ):
            (model / name).write_bytes(b"fixture")
        return model

    def _request(self, root: Path) -> Path:
        frame = root / "frame_01.ppm"
        frame.write_bytes(b"synthetic-frame-placeholder")
        request = root / "request.json"
        request.write_text(
            json.dumps(
                {
                    "candidate": {"detections": [], "signals": []},
                    "evidence": EventMedia(
                        clip_path="",
                        frame_paths=(str(frame),),
                        selected_segments=0,
                    ).to_dict(),
                    "frame_paths": [str(frame)],
                    "question": "Only report directly visible facts.",
                    "output_schema": VLMEvent.json_schema(),
                }
            ),
            encoding="utf-8",
        )
        return request

    def test_identity_and_settings_are_pinned_and_bounded(self) -> None:
        self.assertEqual("HuggingFaceTB/SmolVLM2-2.2B-Instruct", MODEL_ID)
        self.assertEqual(40, len(MODEL_REVISION))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self._model(root)
            self.assertEqual(model, SmolVLM2Settings(model).model_path)
            for value in (True, 63, 1537):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    SmolVLM2Settings(model, max_new_tokens=value)  # type: ignore[arg-type]

    def test_settings_reject_incomplete_or_symlinked_model_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = root / "incomplete"
            incomplete.mkdir()
            with self.assertRaises(FileNotFoundError):
                SmolVLM2Settings(incomplete)
            target = self._model(root)
            link = root / "model-link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileNotFoundError):
                SmolVLM2Settings(link)

    def test_prompt_preserves_fact_inference_and_identity_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = load_request(self._request(root))
            prompt = build_prompt(request)
            self.assertIn("证据不足，无法确定家庭事件。", prompt)
            self.assertIn("silent image cannot prove crying", prompt)
            self.assertIn("repeated frame cannot prove motion", prompt)
            self.assertIn("must not be promoted to baby or adult", prompt)
            self.assertIn("observations", prompt)
            self.assertIn("inferences", prompt)
            self.assertIn("frame_01.ppm", prompt)
            self.assertNotIn(str(root), prompt)

    def test_coverage_hints_distinguish_unverified_from_recorded_gaps_without_certifying_continuity(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            request = load_request(self._request(Path(directory)))
            started = datetime(2026, 9, 12, tzinfo=UTC)
            gap = EvidenceGap(started, started + timedelta(seconds=3), "sampling_gap")
            for gaps, status in (((), "unverified"), ((gap,), "gaps_recorded")):
                with self.subTest(status=status):
                    bounded = replace(
                        request, evidence=replace(request.evidence, coverage_gaps=gaps)
                    )
                    prompt = build_prompt(bounded)
                    metadata = json.loads(
                        next(
                            line
                            for line in prompt.splitlines()
                            if line.startswith("Coverage metadata: ")
                        ).removeprefix("Coverage metadata: ")
                    )
                    self.assertEqual(status, metadata["coverage_status"])
                    self.assertIs(False, metadata["continuous_coverage_verified"])
                    self.assertEqual(
                        [item.reason for item in gaps], metadata["coverage_gap_reasons"]
                    )
                    self.assertEqual(0, metadata["selected_segments"])
                    self.assertNotIn("complete", metadata)
                    self.assertNotIn("complete_coverage", metadata)
                    self.assertIn("unknown does not mean recorded missing footage", prompt)
                    self.assertIn("No recorded gaps does not prove no gaps", prompt)
                    self.assertNotIn("duration_seconds", metadata)

    def test_main_prints_only_validated_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._request(root)
            model = self._model(root)
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.smolvlm2_2_2b.generate",
                    return_value=_valid_result("frame_01.ppm"),
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(["--request", str(request), "--model-path", str(model)])
            self.assertEqual(0, code)
            self.assertEqual("", stderr.getvalue())
            self.assertEqual(
                "unknown",
                VLMEvent.from_json(stdout.getvalue()).event_type.value,
            )

    def test_main_fails_closed_without_echoing_model_or_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self._request(root)
            model = self._model(root)
            private_output = f"invalid output mentioning {root}"
            stdout = StringIO()
            stderr = StringIO()
            with (
                patch(
                    "baby_monitor.local_runners.smolvlm2_2_2b.generate",
                    return_value=private_output,
                ),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                code = main(["--request", str(request), "--model-path", str(model)])
            self.assertEqual(2, code)
            self.assertEqual("", stdout.getvalue())
            self.assertNotIn(private_output, stderr.getvalue())
            self.assertNotIn(str(root), stderr.getvalue())
            self.assertEqual(
                "invalid_structured_output",
                json.loads(stderr.getvalue())["error_code"],
            )


if __name__ == "__main__":
    unittest.main()
