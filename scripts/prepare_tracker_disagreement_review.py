#!/usr/bin/env python3
"""Prepare a minimal offline same-person review for tracker disagreements."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from baby_monitor.object_tracker import DetectionFrameBatch, FrameDetection
from baby_monitor.tracker_evaluation import (
    TrackerDisagreementLinkReview,
    TrackerDisagreementReview,
    TrackerPredictionBatch,
    TrackerPredictionComparison,
)

_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_FRAME_BYTES = 50 * 1024 * 1024
_MAX_LINK_QUESTIONS = 200
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _strict_json(raw: str, context: str) -> Any:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{context} contains duplicate JSON keys")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError(f"{context} contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} is invalid JSON") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_media_manifest(
    path: Path,
    batch: DetectionFrameBatch,
) -> tuple[dict[str, Path], dict[str, str]]:
    source = path.expanduser()
    try:
        if source.is_symlink() or not source.is_file():
            raise OSError
        if not 0 < source.stat().st_size <= _MAX_MANIFEST_BYTES:
            raise ValueError("frame media manifest exceeds the 1 MiB limit")
        value = _strict_json(
            source.read_text(encoding="utf-8"),
            "frame media manifest",
        )
    except ValueError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ValueError("frame media manifest is missing or unsafe") from exc
    fields = {
        "version",
        "detection_batch_sha256",
        "frames",
        "private_relative_paths_included",
        "source_path_included",
        "uploads_performed",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("frame media manifest fields do not match v1")
    if (
        value["version"] != 1
        or value["detection_batch_sha256"] != batch.fingerprint
        or value["private_relative_paths_included"] is not True
        or value["source_path_included"] is not False
        or value["uploads_performed"] is not False
    ):
        raise ValueError("frame media manifest provenance is invalid")
    items = value["frames"]
    if not isinstance(items, list) or len(items) != len(batch.frames):
        raise ValueError("frame media manifest must cover every frame")

    root = source.parent
    frame_root = root / "frames"
    if root.is_symlink() or frame_root.is_symlink() or not frame_root.is_dir():
        raise ValueError("frame media root is unsafe")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for frame, item in zip(batch.frames, items, strict=True):
        if not isinstance(item, dict) or set(item) != {"frame_id", "path", "sha256"}:
            raise ValueError("frame media entry fields do not match v1")
        frame_id = item["frame_id"]
        relative = item["path"]
        digest = item["sha256"]
        expected_relative = PurePosixPath("frames", f"{frame.frame_id}.jpg")
        if (
            frame_id != frame.frame_id
            or not isinstance(relative, str)
            or PurePosixPath(relative) != expected_relative
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError("frame media entry is invalid or out of order")
        candidate = root / Path(*expected_relative.parts)
        try:
            size = candidate.stat().st_size
        except OSError as exc:
            raise ValueError("frame media file is missing or unsafe") from exc
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or candidate.parent.resolve() != frame_root.resolve()
            or not 0 < size <= _MAX_FRAME_BYTES
            or _sha256(candidate) != digest
        ):
            raise ValueError("frame media file does not match its manifest")
        paths[frame_id] = candidate
        hashes[frame_id] = digest
    return paths, hashes


def _detection_map(
    batch: DetectionFrameBatch,
) -> dict[tuple[str, str], FrameDetection]:
    return {
        (frame.frame_id, detection.detection_id): detection
        for frame in batch.frames
        for detection in frame.detections
    }


def _canvas_markup(
    *,
    caption: str,
    frame_id: str,
    detection: FrameDetection,
    image_path: str,
) -> str:
    x1, y1, x2, y2 = detection.bbox_xyxy
    box_style = (
        f"left:{x1 * 100:.6f}%;top:{y1 * 100:.6f}%;"
        f"width:{(x2 - x1) * 100:.6f}%;height:{(y2 - y1) * 100:.6f}%"
    )
    return f"""
<figure class="panel">
  <figcaption>{html.escape(caption)}</figcaption>
  <div class="source-fallback">
    <img src="{html.escape(image_path, quote=True)}"
      alt="{html.escape(caption, quote=True)}中的完整本地画面">
    <span class="fallback-box" style="{box_style}" aria-hidden="true"></span>
  </div>
  <canvas class="crop" role="img"
    aria-label="{html.escape(caption, quote=True)}中的目标框"
    data-src="{html.escape(image_path, quote=True)}"
    data-x1="{x1:.9f}" data-y1="{y1:.9f}"
    data-x2="{x2:.9f}" data-y2="{y2:.9f}"></canvas>
  <small>{html.escape(frame_id)}</small>
</figure>"""


def _question_markup(
    index: int,
    total: int,
    link: TrackerDisagreementLinkReview,
    *,
    detections: dict[tuple[str, str], FrameDetection],
    image_paths: dict[str, str],
) -> str:
    previous = detections[(link.previous_frame_id, link.previous_detection_id)]
    current = detections[(link.current_frame_id, link.current_detection_id)]
    link_id = html.escape(link.link_id, quote=True)
    previous_markup = _canvas_markup(
        caption="较早画面",
        frame_id=link.previous_frame_id,
        detection=previous,
        image_path=image_paths[link.previous_frame_id],
    )
    current_markup = _canvas_markup(
        caption="较晚画面",
        frame_id=link.current_frame_id,
        detection=current,
        image_path=image_paths[link.current_frame_id],
    )
    return f"""
<section class="question" id="question-{link_id}" data-link-id="{link_id}">
  <h2>问题 {index} / {total}</h2>
  <p>先检查两个黄色框：每个框都必须只对应<strong>一名可辨认的人</strong>。
    满足后，再判断框中是不是同一个人。</p>
  <div class="pair">
    {previous_markup}
    {current_markup}
  </div>
  <fieldset>
    <legend>请选择一个答案</legend>
    <label class="choice same"><input type="radio" name="{link_id}"
      value="same_subject">是，同一个人</label>
    <label class="choice different"><input type="radio" name="{link_id}"
      value="different_subject">不是，不同的人</label>
    <label class="choice unknown"><input type="radio" name="{link_id}"
      value="unreviewable">任一框含多人 / 重叠 / 看不清（本题不计分）</label>
  </fieldset>
</section>"""


def _review_page(
    batch: DetectionFrameBatch,
    review: TrackerDisagreementReview,
    media: dict[str, object],
) -> str:
    items = media["frames"]
    assert isinstance(items, list)
    image_paths = {item["frame_id"]: item["path"] for item in items}
    detections = _detection_map(batch)
    questions = "".join(
        _question_markup(
            index,
            len(review.links),
            link,
            detections=detections,
            image_paths=image_paths,
        )
        for index, link in enumerate(review.links, start=1)
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
               frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<meta name="referrer" content="no-referrer">
<title>Tracker 分歧快速复核（{len(review.links)}题）</title>
<style>
:root {{ color-scheme:light; font-family:system-ui,sans-serif; }}
body {{ margin:0 auto; padding:1.25rem; max-width:1280px; background:#f6f7fb; }}
.top {{ position:sticky; top:0; z-index:5; padding:1rem; border-radius:12px;
        background:#172033; color:white; box-shadow:0 4px 16px #0003; }}
.top h1 {{ margin:.1rem 0 .5rem; font-size:1.35rem; }}
.top p {{ margin:.35rem 0; }}
#progress {{ font-weight:700; color:#ffe27a; }}
.question {{ margin:1.5rem 0; padding:1rem; border:2px solid #ccd2df;
             border-radius:14px; background:white; }}
.question:has(input:checked) {{ border-color:#2f7d55; }}
.question h2 {{ margin-top:0; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
.panel {{ margin:0; min-width:0; }}
.panel figcaption {{ font-weight:700; margin-bottom:.35rem; }}
  .source-fallback {{ position:relative; overflow:hidden; width:100%;
                      border-radius:10px; background:#20242c; }}
  .source-fallback img {{ display:block; width:100%; height:auto; }}
  .fallback-box {{ position:absolute; box-sizing:border-box;
                   border:5px solid #ffe000; box-shadow:0 0 0 2px #0008; }}
  .crop {{ display:none; width:100%; min-height:240px; border-radius:10px;
          background:#20242c; }}
  .panel.rendered .source-fallback {{ display:none; }}
  .panel.rendered .crop {{ display:block; }}
.panel small {{ display:block; overflow-wrap:anywhere; color:#596070; }}
fieldset {{ margin-top:1rem; border:0; padding:0; display:grid;
            grid-template-columns:repeat(3,1fr); gap:.7rem; }}
legend {{ font-weight:700; margin-bottom:.55rem; }}
.choice {{ display:flex; align-items:center; justify-content:center; gap:.45rem;
           min-height:3.2rem; padding:.4rem; border:2px solid #aab2c0;
           border-radius:10px; cursor:pointer; font-weight:700; text-align:center; }}
.choice:has(input:checked) {{ border-color:#172033; background:#e8eefc; }}
.choice input {{ width:1.25rem; height:1.25rem; }}
.same {{ color:#17613f; }} .different {{ color:#9b2727; }} .unknown {{ color:#5d5967; }}
#export {{ width:100%; padding:1rem; font-size:1.05rem; font-weight:700;
           color:white; background:#17613f; border:0; border-radius:10px; cursor:pointer; }}
#status {{ font-weight:700; }}
@media (max-width:760px) {{
  .pair, fieldset {{ grid-template-columns:1fr; }}
  .crop {{ min-height:180px; }}
}}
</style>
</head>
<body>
<header class="top">
  <h1>只需回答 {len(review.links)} 个“是不是同一个人”</h1>
  <p>不用填写任何内容，也不用提供姓名、宝宝/成人或 subject 编号；
    每个黄框都只对应一名可辨认的人时才判断。任一框含多人、重叠或看不清，
    直接选“本题不计分”。</p>
  <p id="progress">已完成 0 / {len(review.links)}</p>
</header>
<main>{questions}</main>
<button id="export" type="button">完成并导出 review.completed.json</button>
<p id="status" role="status"></p>
<script>
const pending = {review_json};
const statusNode = document.getElementById('status');
function updateProgress() {{
  const answered = document.querySelectorAll('.question input:checked').length;
  document.getElementById('progress').textContent =
    `已完成 ${{answered}} / ${{pending.links.length}}`;
}}
for (const input of document.querySelectorAll('.question input')) {{
  input.addEventListener('change', updateProgress);
}}
for (const canvas of document.querySelectorAll('canvas.crop')) {{
  const image = new Image();
  image.addEventListener('load', () => {{
    const x1 = Number(canvas.dataset.x1), y1 = Number(canvas.dataset.y1);
    const x2 = Number(canvas.dataset.x2), y2 = Number(canvas.dataset.y2);
    const padX = Math.max((x2 - x1) * .45, .055);
    const padY = Math.max((y2 - y1) * .25, .055);
    const cropX1 = Math.max(0, x1 - padX), cropY1 = Math.max(0, y1 - padY);
    const cropX2 = Math.min(1, x2 + padX), cropY2 = Math.min(1, y2 + padY);
    const sourceX = cropX1 * image.naturalWidth;
    const sourceY = cropY1 * image.naturalHeight;
    const sourceWidth = (cropX2 - cropX1) * image.naturalWidth;
    const sourceHeight = (cropY2 - cropY1) * image.naturalHeight;
    canvas.width = Math.max(1, Math.round(sourceWidth));
    canvas.height = Math.max(1, Math.round(sourceHeight));
    const context = canvas.getContext('2d');
    context.drawImage(image, sourceX, sourceY, sourceWidth, sourceHeight,
                      0, 0, canvas.width, canvas.height);
    context.strokeStyle = '#ffe000';
    context.lineWidth = Math.max(5, Math.round(canvas.width / 180));
    context.strokeRect(
      (x1 - cropX1) / (cropX2 - cropX1) * canvas.width,
      (y1 - cropY1) / (cropY2 - cropY1) * canvas.height,
      (x2 - x1) / (cropX2 - cropX1) * canvas.width,
      (y2 - y1) / (cropY2 - cropY1) * canvas.height
    );
    canvas.closest('.panel').classList.add('rendered');
  }});
  image.addEventListener('error', () => {{
    statusNode.textContent = '本地图片加载失败，请停止复核。';
  }});
  image.src = canvas.dataset.src;
}}
document.getElementById('export').addEventListener('click', async () => {{
  const completedLinks = [];
  for (const link of pending.links) {{
    const selected = document.querySelector(`input[name="${{link.link_id}}"]:checked`);
    if (!selected) {{
      statusNode.textContent = '还有问题未回答。';
      document.getElementById(`question-${{link.link_id}}`).scrollIntoView(
        {{behavior:'smooth', block:'center'}});
      return;
    }}
    completedLinks.push({{...link, review_state:selected.value}});
  }}
  const completed = {{
    ...pending,
    review_status:'approved',
    links:completedLinks,
    reviews:[{{reviewer_id:'reviewer_01',
               reviewed_at:new Date().toISOString(), verdict:'approve'}}],
    human_labels:true
  }};
  const blob = new Blob([
    JSON.stringify(completed, null, 2) + String.fromCharCode(10)
  ],
                        {{type:'application/json'}});
  if ('showSaveFilePicker' in window) {{
    const handle = await window.showSaveFilePicker(
      {{suggestedName:'review.completed.json'}});
    const writable = await handle.createWritable();
    await writable.write(blob); await writable.close();
  }} else {{
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'review.completed.json';
    link.click(); URL.revokeObjectURL(link.href);
  }}
  statusNode.textContent = '已在本机导出；图片没有上传。';
}});
</script>
</body>
</html>
"""


def _copy_frames(
    review: TrackerDisagreementReview,
    *,
    source_paths: dict[str, Path],
    source_hashes: dict[str, str],
    output_dir: Path,
) -> dict[str, object]:
    frame_ids = tuple(
        dict.fromkeys(
            frame_id
            for link in review.links
            for frame_id in (link.previous_frame_id, link.current_frame_id)
        )
    )
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(mode=0o700)
    items = []
    for frame_id in frame_ids:
        target = frame_dir / f"{frame_id}.jpg"
        try:
            with source_paths[frame_id].open("rb") as source, target.open("xb") as out:
                shutil.copyfileobj(source, out, length=1024 * 1024)
        except (KeyError, OSError) as exc:
            raise RuntimeError("tracker disagreement frame copy failed") from exc
        target.chmod(0o600)
        if _sha256(target) != source_hashes[frame_id]:
            raise RuntimeError("tracker disagreement frame copy failed integrity check")
        items.append(
            {
                "frame_id": frame_id,
                "path": f"frames/{frame_id}.jpg",
                "sha256": source_hashes[frame_id],
            }
        )
    return {
        "version": 1,
        "detection_batch_sha256": review.detection_batch_sha256,
        "comparison_sha256": review.comparison_sha256,
        "frames": items,
        "private_relative_paths_included": True,
        "source_path_included": False,
        "uploads_performed": False,
    }


def prepare(
    *,
    frames_path: Path,
    left_path: Path,
    right_path: Path,
    comparison_path: Path,
    media_manifest_path: Path,
    output_dir: Path,
    confirm_private_media: bool,
) -> dict[str, object]:
    if not confirm_private_media:
        raise ValueError("--confirm-private-media is required")
    batch = DetectionFrameBatch.read(frames_path)
    left = TrackerPredictionBatch.read(left_path)
    right = TrackerPredictionBatch.read(right_path)
    comparison = TrackerPredictionComparison.read(comparison_path)
    review = TrackerDisagreementReview.pending(batch, left, right, comparison)
    if not 1 <= len(review.links) <= _MAX_LINK_QUESTIONS:
        raise ValueError("tracker disagreement review requires 1 to 200 links")
    source_paths, source_hashes = _read_media_manifest(
        media_manifest_path,
        batch,
    )
    if output_dir.exists():
        raise ValueError("output directory already exists; refusing to overwrite")
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
        raise ValueError("output parent is unsafe")

    old_umask = os.umask(0o077)
    try:
        output_dir.mkdir(mode=0o700)
        try:
            media = _copy_frames(
                review,
                source_paths=source_paths,
                source_hashes=source_hashes,
                output_dir=output_dir,
            )
            review.write_new(output_dir / "review.pending.json")
            media_path = output_dir / "frame-media.local.json"
            media_path.write_text(
                json.dumps(media, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            media_path.chmod(0o600)
            page_path = output_dir / "review.local.html"
            page_path.write_text(
                _review_page(batch, review, media),
                encoding="utf-8",
                newline="\n",
            )
            page_path.chmod(0o600)
            marker = output_dir / "PRIVATE_MEDIA_DO_NOT_COMMIT"
            marker.write_text(
                "Local tracker disagreement crops. Do not upload or commit.\n",
                encoding="utf-8",
                newline="\n",
            )
            marker.chmod(0o600)
            return {
                "version": 1,
                "questions": len(review.links),
                "unique_frames": len(media["frames"]),  # type: ignore[arg-type]
                "review_status": "pending",
                "detection_batch_sha256": batch.fingerprint,
                "comparison_sha256": comparison.fingerprint,
                "review_sha256": review.fingerprint,
                "private_media_used": True,
                "network_requests": 0,
                "uploads_performed": False,
                "human_labels": False,
                "model_generated_labels": False,
                "overall_accuracy_evaluated": False,
                "global_adapter_selected": False,
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
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--media-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--confirm-private-media", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare(
        frames_path=args.frames,
        left_path=args.left,
        right_path=args.right,
        comparison_path=args.comparison,
        media_manifest_path=args.media_manifest,
        output_dir=args.output_dir,
        confirm_private_media=args.confirm_private_media,
    )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
