#!/usr/bin/env python3
"""Prepare an offline, no-upload tracker association review package."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from baby_monitor.object_tracker import DetectionFrameBatch
from baby_monitor.source_admission import ExternalSourceAdmission
from baby_monitor.tracker_evaluation import (
    TrackerAssociationReview,
    TrackerPredictionComparison,
)

_MAX_REVIEW_FRAMES = 120
_MAX_FRAME_BYTES = 50 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ffmpeg_executable(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else shutil.which(str(value))
    if resolved is None:
        raise ValueError("ffmpeg executable is unavailable")
    path = Path(resolved)
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("ffmpeg executable must be an executable regular file")
    return path


def _extract_frame(
    *,
    ffmpeg: Path,
    source: Path,
    offset_seconds: float,
    output: Path,
) -> None:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{offset_seconds:.6f}",
        "-i",
        str(source),
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-n",
        str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("tracker review frame extraction failed") from exc
    if completed.returncode != 0:
        raise RuntimeError("tracker review frame extraction failed")
    try:
        size = output.stat().st_size
    except OSError as exc:
        raise RuntimeError("tracker review frame extraction produced no file") from exc
    if output.is_symlink() or not output.is_file() or not 0 < size <= _MAX_FRAME_BYTES:
        raise RuntimeError("tracker review frame is missing or outside the size limit")
    output.chmod(0o600)


def _media_manifest(
    batch: DetectionFrameBatch,
    *,
    frame_paths: dict[str, Path],
    root: Path,
) -> dict[str, object]:
    return {
        "version": 1,
        "detection_batch_sha256": batch.fingerprint,
        "frames": [
            {
                "frame_id": frame.frame_id,
                "path": frame_paths[frame.frame_id].relative_to(root).as_posix(),
                "sha256": _sha256(frame_paths[frame.frame_id]),
            }
            for frame in batch.frames
        ],
        "private_relative_paths_included": True,
        "source_path_included": False,
        "uploads_performed": False,
    }


def _frame_markup(
    frame: Any,
    *,
    image_path: str,
    disagreement: bool,
) -> str:
    boxes = []
    rows = []
    for detection in frame.detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        detection_id = html.escape(detection.detection_id, quote=True)
        label = html.escape(detection.label)
        boxes.append(
            '<div class="box" '
            f'style="left:{x1 * 100:.6f}%;top:{y1 * 100:.6f}%;'
            f'width:{(x2 - x1) * 100:.6f}%;height:{(y2 - y1) * 100:.6f}%">'
            f"<span>{detection_id} · {label}</span></div>"
        )
        rows.append(
            f'<div class="assignment" data-frame-id="{html.escape(frame.frame_id, quote=True)}" '
            f'data-detection-id="{detection_id}">'
            f"<code>{detection_id}</code> <span>{label}</span> "
            '<label>状态 <select name="review_state" required>'
            '<option value="">请选择</option>'
            '<option value="assigned">assigned：本帧主体主框</option>'
            '<option value="duplicate">duplicate：同主体重复框</option>'
            '<option value="false_positive">false_positive：不是有效主体框</option>'
            '<option value="unreviewable">unreviewable：无法可靠判断</option>'
            "</select></label> "
            '<label>匿名主体 <input name="subject_id" '
            'placeholder="subject_001" pattern="subject_[A-Za-z0-9_.-]{1,64}"></label>'
            "</div>"
        )
    disagreement_note = (
        '<p class="disagreement-note">两个匿名 tracker 在此帧的相邻关联 link 不一致；'
        "这里只提高人工复核优先级，不表示任一 adapter 正确或错误。</p>"
        if disagreement
        else ""
    )
    frame_class = "frame disagreement" if disagreement else "frame"
    return f"""
<section class="{frame_class}" data-frame-id="{html.escape(frame.frame_id, quote=True)}">
  <h2>{html.escape(frame.frame_id)}</h2>
  {disagreement_note}
  <div class="image-wrap"><img src="{html.escape(image_path, quote=True)}"
       alt="{html.escape(frame.frame_id, quote=True)}">{"".join(boxes)}</div>
  <div class="assignments">{"".join(rows)}</div>
</section>"""


def _review_page(
    batch: DetectionFrameBatch,
    review: TrackerAssociationReview,
    media: dict[str, object],
    disagreement_frame_ids: frozenset[str],
) -> str:
    media_frames = media["frames"]
    assert isinstance(media_frames, list)
    paths = {item["frame_id"]: item["path"] for item in media_frames}
    sections = "".join(
        _frame_markup(
            frame,
            image_path=paths[frame.frame_id],
            disagreement=frame.frame_id in disagreement_frame_ids,
        )
        for frame in batch.frames
    )
    review_json = json.dumps(
        review.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline';
               script-src 'unsafe-inline'; connect-src 'none'; media-src 'none';
               frame-src 'none'">
<meta name="referrer" content="no-referrer">
<title>Tracker 匿名关联本地复核</title>
<style>
body {{ font-family: sans-serif; margin: 1.5rem; max-width: 1400px; }}
.notice {{ background:#fff4cc; border:1px solid #9b7600; padding:1rem; }}
.frame {{ border-top:2px solid #555; margin-top:2rem; padding-top:1rem; }}
.frame.disagreement {{ border-top-color:#a40000; background:#fff4f4; padding:1rem; }}
.disagreement-note {{ color:#7a0000; font-weight:bold; }}
.image-wrap {{ position:relative; display:inline-block; max-width:1100px; }}
.image-wrap img {{ display:block; width:100%; height:auto; }}
.box {{ position:absolute; border:3px solid #00ff70; box-sizing:border-box; }}
.box span {{ position:absolute; left:0; top:0; color:#000; background:#00ff70;
             font:12px monospace; white-space:nowrap; }}
.assignment {{ margin:.6rem 0; padding:.5rem; background:#f3f3f3; }}
.assignment label {{ margin-left:1rem; }}
button {{ font-size:1rem; padding:.7rem 1rem; margin-top:1rem; }}
</style>
</head>
<body>
<h1>Tracker 匿名关联本地复核</h1>
<p class="notice">页面完全离线，不会上传图片。只判断 detector 框在不同帧是否属于同一个可见主体；
匿名编号只能使用 subject_001 这类格式，不填写宝宝、爸爸、成人或任何姓名。每帧同一主体只选一个
assigned 主框，其余重叠重复框标 duplicate；不能判断时选 unreviewable。</p>
<label>reviewer_id（匿名）
  <input id="reviewer" value="reviewer_01"
         pattern="reviewer_[A-Za-z0-9_.-]{{1,64}}" required>
</label>
{sections}
<button id="export" type="button">导出已完成人工复核 JSON</button>
<p id="status" role="status"></p>
<script>
const pending = {review_json};
document.getElementById('export').addEventListener('click', async () => {{
  const reviewer = document.getElementById('reviewer');
  if (!reviewer.reportValidity()) return;
  const assignments = [];
  for (const row of document.querySelectorAll('.assignment')) {{
    const state = row.querySelector('[name=review_state]');
    const subject = row.querySelector('[name=subject_id]');
    if (!state.reportValidity()) return;
    const needsSubject = state.value === 'assigned' || state.value === 'duplicate';
    if (needsSubject) {{
      subject.required = true;
      if (!subject.reportValidity()) return;
    }} else {{
      subject.required = false; subject.value = '';
    }}
    assignments.push({{
      frame_id: row.dataset.frameId,
      detection_id: row.dataset.detectionId,
      review_state: state.value,
      subject_id: needsSubject ? subject.value : null
    }});
  }}
  for (const item of assignments.filter(v => v.review_state === 'duplicate')) {{
    const primary = assignments.some(v => v.frame_id === item.frame_id &&
      v.review_state === 'assigned' && v.subject_id === item.subject_id);
    if (!primary) {{
      document.getElementById('status').textContent =
        `${{item.frame_id}} 的 duplicate 缺少同主体 assigned 主框。`;
      return;
    }}
  }}
  const completed = {{
    ...pending,
    review_status: 'approved',
    assignments,
    reviews: [{{reviewer_id: reviewer.value,
               reviewed_at: new Date().toISOString(), verdict: 'approve'}}],
    human_labels: true
  }};
  const blob = new Blob([JSON.stringify(completed, null, 2) + '\n'],
                        {{type:'application/json'}});
  if ('showSaveFilePicker' in window) {{
    const handle = await window.showSaveFilePicker(
      {{suggestedName:'review.completed.json'}});
    const writable = await handle.createWritable();
    await writable.write(blob); await writable.close();
  }} else {{
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob); link.download = 'review.completed.json';
    link.click(); URL.revokeObjectURL(link.href);
  }}
  document.getElementById('status').textContent =
    '已在本机导出；图片没有上传。请用 validate-tracker-review 校验。';
}});
</script>
</body>
</html>
"""


def prepare(
    *,
    frames_path: Path,
    source_path: Path,
    source_admission_path: Path,
    prediction_comparison_path: Path | None,
    output_dir: Path,
    ffmpeg: str | Path,
    first_frame_offset_seconds: float,
    confirm_private_media: bool,
) -> dict[str, object]:
    if not confirm_private_media:
        raise ValueError("--confirm-private-media is required")
    if (
        isinstance(first_frame_offset_seconds, bool)
        or not isinstance(first_frame_offset_seconds, (int, float))
        or not 0.0 <= float(first_frame_offset_seconds) <= 3600.0
    ):
        raise ValueError("first frame offset must be between 0 and 3600 seconds")
    batch = DetectionFrameBatch.read(frames_path)
    if not 1 <= len(batch.frames) <= _MAX_REVIEW_FRAMES:
        raise ValueError("tracker review must contain 1 to 120 frames")
    admission = ExternalSourceAdmission.read(source_admission_path)
    if admission.asset_kind != "media_asset":
        raise ValueError("tracker review source admission must describe media")
    admission.verify_asset(source_path)
    disagreement_frame_ids: frozenset[str] = frozenset()
    if prediction_comparison_path is not None:
        comparison = TrackerPredictionComparison.read(prediction_comparison_path)
        valid_frame_ids = {frame.frame_id for frame in batch.frames}
        if comparison.detection_batch_sha256 != batch.fingerprint or any(
            frame_id not in valid_frame_ids for frame_id in comparison.disagreement_frame_ids
        ):
            raise ValueError("tracker prediction comparison belongs to a different batch")
        disagreement_frame_ids = frozenset(comparison.disagreement_frame_ids)
    executable = _ffmpeg_executable(ffmpeg)
    if output_dir.exists():
        raise ValueError("output directory already exists; refusing to overwrite")
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
        raise ValueError("output parent is unsafe")

    old_umask = os.umask(0o077)
    try:
        output_dir.mkdir(mode=0o700)
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(mode=0o700)
        try:
            first_time = batch.frames[0].occurred_at
            frame_paths: dict[str, Path] = {}
            for frame in batch.frames:
                offset = (
                    float(first_frame_offset_seconds)
                    + (frame.occurred_at - first_time).total_seconds()
                )
                if not 0.0 <= offset <= 7200.0:
                    raise ValueError("tracker review frame offset is outside the limit")
                output = frame_dir / f"{frame.frame_id}.jpg"
                _extract_frame(
                    ffmpeg=executable,
                    source=source_path.resolve(),
                    offset_seconds=offset,
                    output=output,
                )
                frame_paths[frame.frame_id] = output

            review = TrackerAssociationReview.pending(
                batch,
                source_admission_sha256=admission.fingerprint,
            )
            review_path = output_dir / "review.pending.json"
            review.write_new(review_path)
            media = _media_manifest(
                batch,
                frame_paths=frame_paths,
                root=output_dir,
            )
            media_path = output_dir / "frame-media.local.json"
            media_path.write_text(
                json.dumps(media, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            media_path.chmod(0o600)
            page_path = output_dir / "review.local.html"
            page_path.write_text(
                _review_page(
                    batch,
                    review,
                    media,
                    disagreement_frame_ids,
                ),
                encoding="utf-8",
                newline="\n",
            )
            page_path.chmod(0o600)
            marker = output_dir / "PRIVATE_MEDIA_DO_NOT_COMMIT"
            marker.write_text(
                "Local tracker review frames. Do not upload or commit.\n",
                encoding="utf-8",
                newline="\n",
            )
            marker.chmod(0o600)
            return {
                "version": 1,
                "frames": len(batch.frames),
                "detections": len(review.assignments),
                "review_status": "pending",
                "detection_batch_sha256": batch.fingerprint,
                "source_admission_sha256": admission.fingerprint,
                "review_sha256": review.fingerprint,
                "private_media_used": True,
                "network_requests": 0,
                "uploads_performed": False,
                "model_generated_labels": False,
                "human_labels": False,
                "ready_for_evaluation": False,
                "prediction_comparison_used": prediction_comparison_path is not None,
                "disagreement_frames": len(disagreement_frame_ids),
                "accuracy_evaluated": False,
                "better_adapter_selected": False,
                "paths_disclosed": False,
            }
        except BaseException:
            shutil.rmtree(output_dir)
            raise
    finally:
        os.umask(old_umask)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-admission", required=True, type=Path)
    parser.add_argument("--prediction-comparison", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--first-frame-offset-seconds",
        type=float,
        required=True,
        help=(
            "source-media offset represented by the first DetectionFrame; "
            "pass 0 explicitly when the first frame is at media start"
        ),
    )
    parser.add_argument("--confirm-private-media", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare(
        frames_path=args.frames,
        source_path=args.source,
        source_admission_path=args.source_admission,
        prediction_comparison_path=args.prediction_comparison,
        output_dir=args.output_dir,
        ffmpeg=args.ffmpeg,
        first_frame_offset_seconds=args.first_frame_offset_seconds,
        confirm_private_media=args.confirm_private_media,
    )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
