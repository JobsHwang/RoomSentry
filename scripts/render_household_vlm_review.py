#!/usr/bin/env python3
"""Render an offline, no-upload household VLM labeling page."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Any

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CASE_KEYS = {
    "version",
    "review_status",
    "case_id",
    "frames",
    "question",
    "expected",
    "reviews",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pending_cases(manifest_path: Path) -> tuple[dict[str, Any], ...]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("review manifest must be a regular file, not a symlink")
    root = manifest_path.resolve().parent
    if not 0 < manifest_path.stat().st_size <= 10 * 1024 * 1024:
        raise ValueError("review manifest size is outside the allowed range")
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError("review manifest must be readable UTF-8") from exc
    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid review JSON on line {line_number}") from exc
        if not isinstance(case, dict) or set(case) != _CASE_KEYS:
            raise ValueError(f"review fields are invalid on line {line_number}")
        case_id = case["case_id"]
        frames = case["frames"]
        if (
            case["version"] != 1
            or case["review_status"] != "pending"
            or case["expected"] is not None
            or case["reviews"] != []
            or not isinstance(case_id, str)
            or _SAFE_ID.fullmatch(case_id) is None
            or case_id in case_ids
            or not isinstance(case["question"], str)
            or not 1 <= len(case["question"]) <= 4000
            or not isinstance(frames, list)
            or not 1 <= len(frames) <= 8
        ):
            raise ValueError(f"pending review case is invalid on line {line_number}")
        frame_ids: set[str] = set()
        for frame in frames:
            if not isinstance(frame, dict) or set(frame) != {"frame_id", "path", "sha256"}:
                raise ValueError(f"review frame is invalid on line {line_number}")
            frame_id = frame["frame_id"]
            raw_path = frame["path"]
            expected_hash = frame["sha256"]
            if (
                not isinstance(frame_id, str)
                or _SAFE_ID.fullmatch(frame_id) is None
                or frame_id in frame_ids
                or not isinstance(raw_path, str)
                or not raw_path
                or Path(raw_path).is_absolute()
                or ".." in Path(raw_path).parts
                or not isinstance(expected_hash, str)
                or _SHA256.fullmatch(expected_hash) is None
            ):
                raise ValueError(f"review frame is invalid on line {line_number}")
            path = (root / raw_path).resolve(strict=True)
            if (
                not path.is_relative_to(root)
                or path.is_symlink()
                or not path.is_file()
                or not 0 < path.stat().st_size <= 50 * 1024 * 1024
                or _sha256(path) != expected_hash
            ):
                raise ValueError(f"review frame failed integrity on line {line_number}")
            frame_ids.add(frame_id)
        cases.append(case)
        case_ids.add(case_id)
    if not 10 <= len(cases) <= 30:
        raise ValueError("household review manifest must contain 10 to 30 cases")
    return tuple(cases)


def _case_markup(case: dict[str, Any]) -> str:
    case_id = html.escape(case["case_id"], quote=True)
    images = "".join(
        f'<figure><img src="{html.escape(frame["path"], quote=True)}" '
        f'alt="{case_id} {html.escape(frame["frame_id"], quote=True)}">'
        f"<figcaption>{html.escape(frame['frame_id'])}</figcaption></figure>"
        for frame in case["frames"]
    )
    subject_boxes = "".join(
        f'<label><input type="checkbox" name="subjects" value="{subject}"> {subject}</label>'
        for subject in ("baby", "adult", "person", "pet", "unknown")
    )
    event_options = "".join(
        f'<option value="{value}">{value}</option>'
        for value in (
            "sleeping",
            "awake",
            "crying",
            "movement",
            "adult_interaction",
            "possible_risk",
            "unknown",
        )
    )
    risk_options = "".join(
        f'<option value="{value}">{value}</option>' for value in ("none", "low", "medium", "high")
    )
    return f"""
<section class="case" data-case-id="{case_id}">
  <h2>{case_id}</h2>
  <div class="frames">{images}</div>
  <label>event_type<select name="event_type" required>
    <option value="">请选择</option>{event_options}
  </select></label>
  <label>risk_level<select name="risk_level" required>
    <option value="">请选择</option>{risk_options}
  </select></label>
  <fieldset><legend>allowed_subjects（只按可见证据）</legend>{subject_boxes}</fieldset>
  <label>allowed_observations（每行一条可见事实）
    <textarea name="observations" rows="4" required></textarea>
  </label>
  <label>required_observation_groups（每行一组，同义表达用 || 分隔）
    <textarea name="observation_groups" rows="3"></textarea>
  </label>
  <label>allowed_inferences（每行一条，必须表达不确定性）
    <textarea name="inferences" rows="3"></textarea>
  </label>
  <label>required_inference_groups（每行一组，同义表达用 || 分隔）
    <textarea name="inference_groups" rows="2"></textarea>
  </label>
  <label><input type="checkbox" name="reviewed" required> 已按顺序复核全部帧</label>
</section>"""


def _page(cases: tuple[dict[str, Any], ...]) -> str:
    case_data = json.dumps(cases, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    sections = "".join(_case_markup(case) for case in cases)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline';
               script-src 'unsafe-inline'; connect-src 'none'; media-src 'none';
               frame-src 'none'">
<meta name="referrer" content="no-referrer">
<title>家庭 VLM 本地人工复核</title>
<style>
body {{ font-family: sans-serif; margin: 1.5rem; max-width: 1280px; }}
.notice {{ background: #fff4cc; border: 1px solid #a77b00; padding: 1rem; }}
.case {{ border-top: 2px solid #555; margin-top: 2rem; padding-top: 1rem; }}
.frames {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
figure {{ margin: 0; }} img {{ width: 360px; max-width: 100%; height: auto; }}
label, fieldset {{ display: block; margin: .8rem 0; }}
textarea {{ display: block; width: min(100%, 900px); }}
fieldset label {{ display: inline-block; margin-right: 1rem; }}
button {{ font-size: 1rem; padding: .7rem 1rem; }}
</style>
</head>
<body>
<h1>家庭 VLM 本地人工复核</h1>
<p class="notice">此页没有网络请求，不会上传画面。普通 person 不能证明宝宝、成人或具体身份；
证据不足请选择 unknown。中高风险或 possible_risk 会自动标记为 needs_second_review。</p>
<label>reviewer_id（匿名标识）
  <input id="reviewer" value="local_reviewer_1" pattern="[A-Za-z0-9_.-]{{1,80}}" required>
</label>
{sections}
<button id="export" type="button">导出本地 JSONL</button>
<p id="status" role="status"></p>
<script>
const cases = {case_data};
const lines = value => [...new Set(value.split('\n').map(v => v.trim()).filter(Boolean))];
const groups = value => lines(value)
  .map(line => [...new Set(line.split('||').map(v => v.trim()).filter(Boolean))])
  .filter(group => group.length);
const union = (values, nested) => [...new Set([...values, ...nested.flat()])];
document.getElementById('export').addEventListener('click', async () => {{
  const reviewer = document.getElementById('reviewer');
  if (!reviewer.reportValidity()) return;
  const output = [];
  for (const original of cases) {{
    const section = document.querySelector(`[data-case-id="${{original.case_id}}"]`);
    const required = section.querySelectorAll('[required]');
    if (![...required].every(field => field.reportValidity())) return;
    const eventType = section.querySelector('[name=event_type]').value;
    const riskLevel = section.querySelector('[name=risk_level]').value;
    const observationGroups = groups(section.querySelector('[name=observation_groups]').value);
    const inferenceGroups = groups(section.querySelector('[name=inference_groups]').value);
    const observations = union(
      lines(section.querySelector('[name=observations]').value), observationGroups
    );
    const inferences = union(
      lines(section.querySelector('[name=inferences]').value), inferenceGroups
    );
    const safety = eventType === 'possible_risk' || riskLevel === 'medium' || riskLevel === 'high';
    output.push({{
      ...original,
      review_status: safety ? 'needs_second_review' : 'approved',
      expected: {{
        event_type: eventType,
        risk_level: riskLevel,
        allowed_subjects: [
          ...section.querySelectorAll('[name=subjects]:checked')
        ].map(v => v.value),
        allowed_observations: observations,
        allowed_inferences: inferences,
        required_observation_groups: observationGroups,
        required_inference_groups: inferenceGroups
      }},
      reviews: [{{
        reviewer_id: reviewer.value,
        verdict: 'approve',
        reviewed_at: new Date().toISOString()
      }}]
    }});
  }}
  const payload = output.map(item => JSON.stringify(item)).join('\n') + '\n';
  const blob = new Blob([payload], {{type: 'application/x-ndjson'}});
  if ('showSaveFilePicker' in window) {{
    const handle = await window.showSaveFilePicker({{suggestedName: 'review.completed.jsonl'}});
    const writable = await handle.createWritable();
    await writable.write(blob); await writable.close();
  }} else {{
    const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
    link.download = 'review.completed.jsonl'; link.click(); URL.revokeObjectURL(link.href);
  }}
  document.getElementById('status').textContent = '已在本机导出；画面没有上传。';
}});
</script>
</body>
</html>
"""


def render(manifest_path: Path, output_path: Path) -> tuple[int, int]:
    cases = _pending_cases(manifest_path)
    if output_path.exists():
        raise ValueError("output file already exists; refusing to overwrite")
    if output_path.parent.resolve() != manifest_path.resolve().parent:
        raise ValueError("offline review page must stay beside the private manifest")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output_path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(_page(cases))
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return len(cases), sum(len(case["frames"]) for case in cases)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    cases, frames = render(args.manifest, args.output)
    print(
        json.dumps(
            {
                "version": 1,
                "cases": cases,
                "frames": frames,
                "network_requests": 0,
                "uploads_performed": False,
                "pixels_disclosed": False,
                "paths_disclosed": False,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
