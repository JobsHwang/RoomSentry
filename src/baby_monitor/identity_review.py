# ruff: noqa: E501
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import secrets
import stat
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


class IdentityReviewError(ValueError):
    """A private household identity review artifact is unsafe or malformed."""


REVIEW_VERSION = 1
MIN_GROUP_IMAGES = 3
MAX_GROUP_IMAGES = 12
MAX_REVIEW_CANDIDATES = 64
MAX_SELECTION_BYTES = 64 * 1024
MAX_REVIEW_HTML_BYTES = 32 * 1024 * 1024
_CANDIDATE_ID = re.compile(r"candidate_[0-9a-f]{16}")
_REVIEW_ID = re.compile(r"review_[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_LABELS = frozenset({"baby", "adult", "reject"})
_CONFIRMATIONS = frozenset(
    {
        "baby_same_subject",
        "adult_same_subject",
        "live_camera_not_replay",
        "single_clear_face",
    }
)


def _strict_json(raw: bytes, *, context: str, maximum_bytes: int) -> Any:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= maximum_bytes:
        raise IdentityReviewError(f"{context} exceeds the size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise IdentityReviewError(f"{context} is not valid UTF-8") from exc

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise IdentityReviewError(f"{context} contains duplicate JSON keys")
            value[key] = item
        return value

    try:
        return json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                IdentityReviewError(f"{context} contains a non-standard number")
            ),
        )
    except json.JSONDecodeError as exc:
        raise IdentityReviewError(f"{context} is not valid JSON") from exc


def admit_private_review_root(value: str | Path) -> Path:
    """Admit one existing owner-only WSL directory outside the Git tree."""

    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise IdentityReviewError("identity review requires a private POSIX directory")
    try:
        candidate = Path(value)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise OSError
        resolved = candidate.resolve(strict=True)
        metadata = candidate.lstat()
        project_root = Path(__file__).resolve(strict=True).parents[2]
        home_root = Path.home().resolve(strict=True)
    except (OSError, TypeError, ValueError):
        raise IdentityReviewError("identity review directory is unsafe") from None
    if (
        resolved != candidate
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
        or resolved in {Path("/"), home_root, project_root}
        or resolved.parts[:3] == ("/", "mnt", "c")
    ):
        raise IdentityReviewError("identity review directory is unsafe")
    return resolved


def write_private_file(path: Path, payload: bytes, *, maximum_bytes: int) -> None:
    """Create one direct private file without following links or overwriting data."""

    if not isinstance(payload, bytes) or not 0 < len(payload) <= maximum_bytes:
        raise IdentityReviewError("identity review output exceeds the size limit")
    root = admit_private_review_root(path.parent)
    if path.parent != root or _SAFE_FILENAME.fullmatch(path.name) is None:
        raise IdentityReviewError("identity review output path is unsafe")
    root_descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_descriptor,
        )
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise OSError
    except FileExistsError:
        raise
    except OSError as exc:
        raise IdentityReviewError("identity review output could not be written safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(root_descriptor)


def write_private_json(path: Path, value: object, *, maximum_bytes: int) -> None:
    try:
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise IdentityReviewError("identity review JSON is invalid") from exc
    write_private_file(path, payload, maximum_bytes=maximum_bytes)


@dataclass(frozen=True, slots=True)
class ReviewCandidate:
    candidate_id: str
    ordinal: int
    source_filename: str = field(repr=False)
    source_sha256: str
    preview_filename: str = field(repr=False)
    preview_sha256: str
    width: int
    height: int
    bbox_xyxy: tuple[float, float, float, float]
    detector_confidence: float
    mean_luma: float
    luma_stddev: float

    def __post_init__(self) -> None:
        if _CANDIDATE_ID.fullmatch(self.candidate_id) is None:
            raise IdentityReviewError("identity review candidate ID is invalid")
        if type(self.ordinal) is not int or not 1 <= self.ordinal <= MAX_REVIEW_CANDIDATES:
            raise IdentityReviewError("identity review candidate ordinal is invalid")
        for filename in (self.source_filename, self.preview_filename):
            if _SAFE_FILENAME.fullmatch(filename) is None or not filename.endswith(".jpg"):
                raise IdentityReviewError("identity review candidate filename is invalid")
        if self.source_filename == self.preview_filename:
            raise IdentityReviewError("identity review source and preview must be distinct")
        for digest in (self.source_sha256, self.preview_sha256):
            if _SHA256.fullmatch(digest) is None:
                raise IdentityReviewError("identity review candidate hash is invalid")
        if type(self.width) is not int or type(self.height) is not int:
            raise IdentityReviewError("identity review candidate dimensions are invalid")
        if not 1 <= self.width <= 65_535 or not 1 <= self.height <= 65_535:
            raise IdentityReviewError("identity review candidate dimensions are invalid")
        if not isinstance(self.bbox_xyxy, tuple) or len(self.bbox_xyxy) != 4:
            raise IdentityReviewError("identity review candidate bbox is invalid")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not all(type(item) is float and 0.0 <= item <= 1.0 for item in self.bbox_xyxy):
            raise IdentityReviewError("identity review candidate bbox is invalid")
        if x1 >= x2 or y1 >= y2:
            raise IdentityReviewError("identity review candidate bbox is invalid")
        for value in (self.detector_confidence, self.mean_luma, self.luma_stddev):
            if type(value) is not float or not 0.0 <= value <= 255.0:
                raise IdentityReviewError("identity review candidate metric is invalid")
        if self.detector_confidence > 1.0:
            raise IdentityReviewError("identity review detector confidence is invalid")

    def to_private_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "ordinal": self.ordinal,
            "source_filename": self.source_filename,
            "source_sha256": self.source_sha256,
            "preview_filename": self.preview_filename,
            "preview_sha256": self.preview_sha256,
            "width": self.width,
            "height": self.height,
            "bbox_xyxy": list(self.bbox_xyxy),
            "detector_confidence": self.detector_confidence,
            "mean_luma": self.mean_luma,
            "luma_stddev": self.luma_stddev,
        }

    @classmethod
    def from_private_dict(cls, value: Any) -> ReviewCandidate:
        expected = {
            "candidate_id",
            "ordinal",
            "source_filename",
            "source_sha256",
            "preview_filename",
            "preview_sha256",
            "width",
            "height",
            "bbox_xyxy",
            "detector_confidence",
            "mean_luma",
            "luma_stddev",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise IdentityReviewError("identity review candidate fields are invalid")
        bbox = value["bbox_xyxy"]
        if not isinstance(bbox, list):
            raise IdentityReviewError("identity review candidate bbox is invalid")
        return cls(
            candidate_id=value["candidate_id"],
            ordinal=value["ordinal"],
            source_filename=value["source_filename"],
            source_sha256=value["source_sha256"],
            preview_filename=value["preview_filename"],
            preview_sha256=value["preview_sha256"],
            width=value["width"],
            height=value["height"],
            bbox_xyxy=tuple(bbox),
            detector_confidence=value["detector_confidence"],
            mean_luma=value["mean_luma"],
            luma_stddev=value["luma_stddev"],
        )


@dataclass(frozen=True, slots=True)
class ReviewBundle:
    review_id: str
    candidates: tuple[ReviewCandidate, ...]
    created_at_utc: str
    version: int = REVIEW_VERSION

    def __post_init__(self) -> None:
        if self.version != REVIEW_VERSION or _REVIEW_ID.fullmatch(self.review_id) is None:
            raise IdentityReviewError("identity review bundle identity is invalid")
        if not 1 <= len(self.candidates) <= MAX_REVIEW_CANDIDATES:
            raise IdentityReviewError("identity review candidate count is invalid")
        if any(not isinstance(item, ReviewCandidate) for item in self.candidates):
            raise IdentityReviewError("identity review candidates are invalid")
        if len({item.candidate_id for item in self.candidates}) != len(self.candidates):
            raise IdentityReviewError("identity review candidate IDs must be unique")
        if tuple(item.ordinal for item in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise IdentityReviewError("identity review candidate ordinals must be contiguous")
        try:
            parsed = datetime.fromisoformat(self.created_at_utc.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise IdentityReviewError("identity review timestamp is invalid") from exc
        if parsed.tzinfo is None:
            raise IdentityReviewError("identity review timestamp is invalid")

    def to_private_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "review_id": self.review_id,
            "created_at_utc": self.created_at_utc,
            "candidates": [item.to_private_dict() for item in self.candidates],
            "identity_inference": False,
            "risk_decision_input": False,
            "notification_input": False,
            "remote_upload": False,
            "training": False,
        }

    @classmethod
    def from_private_dict(cls, value: Any) -> ReviewBundle:
        expected = {
            "version",
            "review_id",
            "created_at_utc",
            "candidates",
            "identity_inference",
            "risk_decision_input",
            "notification_input",
            "remote_upload",
            "training",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise IdentityReviewError("identity review bundle fields are invalid")
        if any(
            value[field] is not False
            for field in (
                "identity_inference",
                "risk_decision_input",
                "notification_input",
                "remote_upload",
                "training",
            )
        ):
            raise IdentityReviewError("identity review scope flags are invalid")
        candidates = value["candidates"]
        if not isinstance(candidates, list):
            raise IdentityReviewError("identity review candidates must be an array")
        return cls(
            version=value["version"],
            review_id=value["review_id"],
            created_at_utc=value["created_at_utc"],
            candidates=tuple(ReviewCandidate.from_private_dict(item) for item in candidates),
        )


def read_review_bundle(root: str | Path) -> ReviewBundle:
    admitted = admit_private_review_root(root)
    path = admitted / "candidates.json"
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= 4 * 1024 * 1024
        ):
            raise OSError
        raw = path.read_bytes()
    except OSError as exc:
        raise IdentityReviewError("identity review bundle could not be read safely") from exc
    return ReviewBundle.from_private_dict(
        _strict_json(raw, context="identity review bundle", maximum_bytes=4 * 1024 * 1024)
    )


def build_review_html(bundle: ReviewBundle, previews: dict[str, bytes]) -> bytes:
    if set(previews) != {item.candidate_id for item in bundle.candidates}:
        raise IdentityReviewError("identity review previews do not match candidates")
    cards: list[str] = []
    for candidate in bundle.candidates:
        preview = previews[candidate.candidate_id]
        if not isinstance(preview, bytes) or not preview.startswith(b"\xff\xd8"):
            raise IdentityReviewError("identity review preview is not a JPEG")
        encoded = base64.b64encode(preview).decode("ascii")
        candidate_id = html.escape(candidate.candidate_id, quote=True)
        cards.append(
            f'''<article class="card" data-candidate="{candidate_id}">
  <img alt="候选 {candidate.ordinal:02d}" src="data:image/jpeg;base64,{encoded}">
  <div class="card-body">
    <strong>候选 {candidate.ordinal:02d}</strong>
    <div class="choices" role="radiogroup" aria-label="候选 {candidate.ordinal:02d} 分类">
      <label><input type="radio" name="{candidate_id}" value="baby"> 宝宝</label>
      <label><input type="radio" name="{candidate_id}" value="adult"> 大人</label>
      <label><input type="radio" name="{candidate_id}" value="reject" checked> 不用</label>
    </div>
  </div>
</article>'''
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'">
<title>本地人脸录入审核</title>
<style>
:root {{ color-scheme: dark; font-family: ui-sans-serif,system-ui,-apple-system,"Microsoft YaHei",sans-serif; background:#0b1020; color:#eef2ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:linear-gradient(145deg,#09101f,#151a2c); min-height:100vh; }}
main {{ max-width:1260px; margin:auto; padding:28px 20px 80px; }}
.hero {{ background:#121a30; border:1px solid #293556; border-radius:18px; padding:22px; box-shadow:0 18px 45px #0006; }}
h1 {{ margin:0 0 10px; font-size:clamp(24px,4vw,38px); }}
p {{ margin:8px 0; color:#cbd5e1; line-height:1.65; }}
.scope {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
.pill {{ background:#172554; color:#bfdbfe; border:1px solid #1d4ed8; border-radius:999px; padding:6px 10px; font-size:13px; }}
.counter {{ position:sticky; top:10px; z-index:3; display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin:18px 0; padding:12px 16px; border:1px solid #334155; background:#0f172aeF; border-radius:14px; backdrop-filter:blur(10px); }}
.count {{ font-weight:800; }}
.baby {{ color:#f9a8d4; }} .adult {{ color:#93c5fd; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:16px; }}
.card {{ overflow:hidden; border:2px solid #26334f; border-radius:16px; background:#111827; transition:.15s; }}
.card:has(input[value="baby"]:checked) {{ border-color:#ec4899; box-shadow:0 0 0 2px #ec489944; }}
.card:has(input[value="adult"]:checked) {{ border-color:#3b82f6; box-shadow:0 0 0 2px #3b82f644; }}
.card img {{ display:block; width:100%; aspect-ratio:16/10; object-fit:contain; background:#020617; }}
.card-body {{ padding:12px; }}
.choices {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:10px; }}
.choices label {{ display:flex; justify-content:center; gap:5px; border:1px solid #334155; border-radius:10px; padding:9px 4px; cursor:pointer; background:#0f172a; }}
.confirm {{ margin-top:24px; padding:18px; border-radius:16px; border:1px solid #334155; background:#111827; }}
.confirm label {{ display:block; margin:10px 0; line-height:1.5; }}
button {{ margin-top:14px; min-height:48px; border:0; border-radius:12px; padding:0 22px; color:white; background:#2563eb; font-size:16px; font-weight:800; cursor:pointer; }}
button:disabled {{ opacity:.55; cursor:wait; }}
#status {{ min-height:28px; margin-top:12px; font-weight:700; }}
.error {{ color:#fca5a5; }} .ok {{ color:#86efac; }}
</style>
</head>
<body>
<main>
  <section class="hero">
    <h1>本地人脸录入审核</h1>
    <p><b>宝宝一组，大人一组。</b>同一组不能混人；每张图只应有一个清楚的脸。其余全部保持“不用”。</p>
    <p>每组选择 3～12 张。绿色框只是本地模型找到的那张脸，不代表系统已经知道是谁。</p>
    <div class="scope"><span class="pill">仅本机</span><span class="pill">不上传</span><span class="pill">不训练</span><span class="pill">只进入 identity shadow</span><span class="pill">不影响风险与通知</span></div>
  </section>
  <div class="counter"><span>宝宝：<span id="babyCount" class="count baby">0</span>/3～12</span><span>大人：<span id="adultCount" class="count adult">0</span>/3～12</span><span>不用：<span id="rejectCount" class="count">{len(bundle.candidates)}</span></span></div>
  <section class="grid">{"".join(cards)}</section>
  <section class="confirm">
    <h2>保存前确认</h2>
    <label><input id="babySame" type="checkbox"> 我确认“宝宝”组里的每张图都是同一个宝宝。</label>
    <label><input id="adultSame" type="checkbox"> 我确认“大人”组里的每张图都是同一位大人。</label>
    <label><input id="liveCamera" type="checkbox"> 这些都是本次本地摄像头画面，不是照片或屏幕翻拍。</label>
    <label><input id="singleFace" type="checkbox"> 每张选中的图都只看到一个清楚的人脸。</label>
    <button id="save" type="button">保存本次选择</button>
    <div id="status" role="status" aria-live="polite"></div>
  </section>
</main>
<script>
const cards=[...document.querySelectorAll('[data-candidate]')];
const statusNode=document.getElementById('status');
function labels(){{const out={{}};for(const card of cards){{out[card.dataset.candidate]=card.querySelector('input:checked').value;}}return out;}}
function update(){{const values=Object.values(labels());document.getElementById('babyCount').textContent=values.filter(v=>v==='baby').length;document.getElementById('adultCount').textContent=values.filter(v=>v==='adult').length;document.getElementById('rejectCount').textContent=values.filter(v=>v==='reject').length;}}
for(const input of document.querySelectorAll('input[type=radio]')) input.addEventListener('change',update);
document.getElementById('save').addEventListener('click',async()=>{{
  const selected=labels(); const values=Object.values(selected); const baby=values.filter(v=>v==='baby').length; const adult=values.filter(v=>v==='adult').length;
  if(baby<3||baby>12||adult<3||adult>12){{statusNode.className='error';statusNode.textContent='宝宝和大人都必须各选 3～12 张。';return;}}
  const confirmations={{baby_same_subject:document.getElementById('babySame').checked,adult_same_subject:document.getElementById('adultSame').checked,live_camera_not_replay:document.getElementById('liveCamera').checked,single_clear_face:document.getElementById('singleFace').checked}};
  if(Object.values(confirmations).some(v=>!v)){{statusNode.className='error';statusNode.textContent='请先勾选四项确认。';return;}}
  const button=document.getElementById('save');button.disabled=true;statusNode.className='';statusNode.textContent='正在保存到本机…';
  try{{const token=new URLSearchParams(location.search).get('token')||'';const response=await fetch('/selection?token='+encodeURIComponent(token),{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{version:1,review_id:{json.dumps(bundle.review_id)},labels:selected,confirmations}})}});const result=await response.json();if(!response.ok)throw new Error(result.error||'保存失败');statusNode.className='ok';statusNode.textContent='已安全保存到本机。可以回到 Codex，我会继续完成录入。';}}
  catch(error){{button.disabled=false;statusNode.className='error';statusNode.textContent='没有保存成功：'+error.message;}}
}});
update();
</script>
</body>
</html>"""
    encoded_page = page.encode("utf-8")
    if len(encoded_page) > MAX_REVIEW_HTML_BYTES:
        raise IdentityReviewError("identity review page exceeds the size limit")
    return encoded_page


def validate_selection(value: Any, bundle: ReviewBundle) -> dict[str, object]:
    expected = {"version", "review_id", "labels", "confirmations"}
    if not isinstance(value, dict) or set(value) != expected:
        raise IdentityReviewError("identity review selection fields are invalid")
    if value["version"] != REVIEW_VERSION or value["review_id"] != bundle.review_id:
        raise IdentityReviewError("identity review selection does not match this review")
    labels = value["labels"]
    expected_ids = {item.candidate_id for item in bundle.candidates}
    if not isinstance(labels, dict) or set(labels) != expected_ids:
        raise IdentityReviewError("identity review labels do not match candidates")
    if any(not isinstance(label, str) or label not in _LABELS for label in labels.values()):
        raise IdentityReviewError("identity review label is invalid")
    confirmations = value["confirmations"]
    if (
        not isinstance(confirmations, dict)
        or set(confirmations) != _CONFIRMATIONS
        or any(confirmations[key] is not True for key in _CONFIRMATIONS)
    ):
        raise IdentityReviewError("identity review confirmations are incomplete")
    baby_count = sum(label == "baby" for label in labels.values())
    adult_count = sum(label == "adult" for label in labels.values())
    if not MIN_GROUP_IMAGES <= baby_count <= MAX_GROUP_IMAGES:
        raise IdentityReviewError("identity review baby selection must contain 3 to 12 images")
    if not MIN_GROUP_IMAGES <= adult_count <= MAX_GROUP_IMAGES:
        raise IdentityReviewError("identity review adult selection must contain 3 to 12 images")
    return {
        "version": REVIEW_VERSION,
        "review_id": bundle.review_id,
        "labels": {key: labels[key] for key in sorted(labels)},
        "confirmations": {key: True for key in sorted(_CONFIRMATIONS)},
        "baby_count": baby_count,
        "adult_count": adult_count,
        "saved_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "identity_shadow_only": True,
        "risk_decision_input": False,
        "notification_input": False,
        "remote_upload": False,
        "training": False,
    }


def new_review_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"review_{timestamp}_{secrets.token_hex(4)}"


def run_review_server(root: str | Path, *, port: int = 0) -> None:
    """Serve one token-gated loopback review and stop after its first valid save."""

    admitted = admit_private_review_root(root)
    bundle = read_review_bundle(admitted)
    page_path = admitted / "review.local.html"
    try:
        page_metadata = page_path.lstat()
        if (
            not stat.S_ISREG(page_metadata.st_mode)
            or stat.S_ISLNK(page_metadata.st_mode)
            or stat.S_IMODE(page_metadata.st_mode) != 0o600
            or page_metadata.st_uid != os.geteuid()
            or page_metadata.st_nlink != 1
            or not 0 < page_metadata.st_size <= MAX_REVIEW_HTML_BYTES
        ):
            raise OSError
        page = page_path.read_bytes()
    except OSError as exc:
        raise IdentityReviewError("identity review page could not be read safely") from exc
    if not page.startswith(b"<!doctype html>"):
        raise IdentityReviewError("identity review page is invalid")
    if type(port) is not int or not 0 <= port <= 65_535:
        raise IdentityReviewError("identity review port is invalid")
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        server_version = "BabyMonitorIdentityReview/1"

        def log_message(self, _format: str, *args: object) -> None:
            return

        def _authorized(self) -> bool:
            query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            supplied = query.get("token", [""])
            return len(supplied) == 1 and secrets.compare_digest(supplied[0], token)

        def _json_response(self, status: HTTPStatus, value: dict[str, object]) -> None:
            payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            split = urlsplit(self.path)
            if split.path != "/" or not self._authorized():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'",
            )
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802
            split = urlsplit(self.path)
            if split.path != "/selection" or not self._authorized():
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if self.headers.get_content_type() != "application/json":
                self._json_response(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "json_required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_SELECTION_BYTES:
                self._json_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid_size"})
                return
            try:
                value = _strict_json(
                    self.rfile.read(length),
                    context="identity review selection",
                    maximum_bytes=MAX_SELECTION_BYTES,
                )
                normalized = validate_selection(value, bundle)
                write_private_json(
                    admitted / "selection.json",
                    normalized,
                    maximum_bytes=MAX_SELECTION_BYTES,
                )
            except FileExistsError:
                self._json_response(HTTPStatus.CONFLICT, {"error": "already_saved"})
                return
            except IdentityReviewError:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid_selection"})
                return
            self._json_response(HTTPStatus.OK, {"status": "saved"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    try:
        host, selected_port = server.server_address[:2]
        url = f"http://{host}:{selected_port}/?token={token}\n".encode("ascii")
        write_private_file(admitted / "review.url", url, maximum_bytes=4096)
        print(f"identity_review_ready candidates={len(bundle.candidates)} port={selected_port}")
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()


def sha256_bytes(value: bytes) -> str:
    if not isinstance(value, bytes):
        raise IdentityReviewError("identity review bytes are invalid")
    return hashlib.sha256(value).hexdigest()
