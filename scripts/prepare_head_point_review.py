"""Build a self-contained local visible-head review; no capture or model calls."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
import secrets
import struct
from pathlib import Path

from baby_monitor.head_point_review import (
    HeadPointReviewError,
    fingerprint,
    parse_json,
    validate_manifest,
    validate_review,
)
from scripts.household_debug_common import new_output_directory, private_directory, private_file


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_blob(path: Path, *, limit: int) -> bytes:
    """Check ownership/privacy/symlinks plus descriptor identity before/after."""
    private_file(path, limit=limit)
    before = path.stat()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if _identity(opened) != _identity(before):
            raise HeadPointReviewError("review input changed")
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    private_file(path, limit=limit)
    if (
        _identity(after) != _identity(before)
        or _identity(path.stat()) != _identity(before)
        or not 0 < len(data) <= limit
        or len(data) != before.st_size
    ):
        raise HeadPointReviewError("review input changed")
    return data


def _write(path: Path, data: bytes) -> None:
    private_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[8:16] != b"\x00\x00\x00\x0dIHDR":
        raise HeadPointReviewError("review requires a bound PNG input")
    return struct.unpack(">II", data[16:24])


def build_page(manifest: dict, images: list[bytes], *, synthetic: bool = False) -> str:
    validate_manifest(manifest)
    if len(images) != len(manifest["items"]):
        raise HeadPointReviewError("review images do not match manifest")
    cards = []
    for item, data in zip(manifest["items"], images, strict=True):
        if hashlib.sha256(data).hexdigest() != item["frame_sha256"] or _png_dimensions(data) != (
            item["width"],
            item["height"],
        ):
            raise HeadPointReviewError("review image binding mismatch")
        cards.append(
            f'<section class="card" data-item="{html.escape(item["item_id"])}">'
            f"<h2>{html.escape(item['item_id'])} · {html.escape(item['camera_id'])}</h2>"
            f'<p class="stamp">历史收帧：{html.escape(item["frame_received_at"])}；不是当前状态</p>'
            '<label>画面判断 <select class="decision"><option value="">请选择</option>'
            '<option value="marked">能定位可见头部</option>'
            '<option value="no_visible_head">没有可辨认的头部（不代表无人）</option>'
            '<option value="unknown">整张看不清／不确定，跳过</option></select></label>'
            '<label>下一次点击标记为 <select class="role"><option value="">请选择</option>'
            '<option value="baby">宝宝</option><option value="adult">成人</option>'
            '<option value="unknown">年龄不明</option></select></label>'
            '<div class="tools"><button type="button" class="zoom" data-scale="1">适合窗口</button>'
            '<button type="button" class="zoom" data-scale="2">放大 2×</button>'
            '<button type="button" class="zoom" data-scale="4">放大 4×</button>'
            '<button type="button" class="undo">撤销最后一个点</button></div>'
            '<div class="viewport"><div class="stage">'
            f'<img draggable="false" width="{item["width"]}" height="{item["height"]}" '
            f'alt="{html.escape(item["item_id"])} 历史原画面，点击可见头部中心" '
            f'src="data:image/png;base64,{base64.b64encode(data).decode()}">'
            '<svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true"></svg>'
            '</div></div><p class="load-status" aria-live="polite">正在加载本机画面…</p>'
            '<p class="points" aria-live="polite">尚未标记头部</p>'
            '<label><input type="checkbox" class="complete">已标完这张图中所有能辨认的头部；'
            "若可能漏了别的头部，不勾选</label>"
            '<label><input type="checkbox" class="reviewed">'
            "我已查看本张画面／无法加载提示，确认以上回答</label>"
            "</section>"
        )
    embedded = json.dumps(manifest, separators=(",", ":"), ensure_ascii=True).replace(
        "<", "\\u003c"
    )
    provenance = "synthetic" if synthetic else "human_user_review"
    reviewer = "reviewer_synthetic" if synthetic else "reviewer_local"
    replacements = {
        "__CARDS__": "\n".join(cards),
        "__MANIFEST__": embedded,
        "__FINGERPRINT__": fingerprint(manifest),
        "__PROVENANCE__": provenance,
        "__REVIEWER__": reviewer,
    }
    # Replace original template tokens once; inserted legal IDs are never reprocessed.
    return re.sub(
        r"__(?:CARDS|MANIFEST|FINGERPRINT|PROVENANCE|REVIEWER)__",
        lambda match: replacements[match.group()],
        PAGE,
    )


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:;
style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none';
media-src 'none'; frame-src 'none'; form-action 'none'; base-uri 'none'">
<title>本地头部复核 · 历史开发画面</title>
<style>
*{box-sizing:border-box}
body{font:16px/1.6 system-ui,sans-serif;color:#1b283b;background:#f4f6fa;
margin:0;padding:24px}
main{max-width:1100px;margin:auto}
.notice,.card{background:white;padding:20px;border:1px solid #cad3df;
border-radius:12px;margin-bottom:20px}
h1,h2{line-height:1.3}h1{font-size:27px}.notice{border-left:5px solid #2b64c7}
.stamp{color:#526078;font-size:14px}label{display:block;margin:12px 0}
select,button{font:inherit;padding:8px 12px;border:1px solid #91a3be;
border-radius:6px;background:white;color:inherit}button{cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible{
outline:3px solid #235cce;outline-offset:3px}
button:disabled{opacity:.45;cursor:not-allowed}
.tools{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
.viewport{max-height:650px;overflow:auto;border:1px solid #7789a0;background:#182534}
.stage{width:100%;position:relative}
.stage img{display:block;width:100%;height:auto;cursor:crosshair;user-select:none}
.stage svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.points{min-height:26px}.error{color:#a62224;font-weight:600}
.success{color:#126747;font-weight:600}
.export{position:sticky;bottom:0;background:#eaf0fc;padding:16px;
border:1px solid #94acd8;border-radius:10px}
#download{background:#265dba;color:white}.note{font-size:14px;color:#526078}
@media(max-width:650px){body{padding:10px}.card,.notice{padding:12px}
select{max-width:100%}}
</style></head><body><main>
<h1>只点头部，不画身体框</h1>
<section class="notice"><p>这是少量历史开发画面，不是当前监控或新的独立测试集。</p>
<ol><li>选择“能定位可见头部”，再选宝宝、成人或年龄不明。</li>
<li>点击每个可见头部的中心；可以放大、撤销。小脸看不清时，不要猜。</li>
<li>不确定整张图就选“跳过”；遮住头部不等于没有人。</li>
<li>确认各张回答后，点底部“导出本地回答”。</li></ol>
<p class="note">本页隐藏模型检测框、分数和以前的答案。
画面只在本机显示，没有上传、后台提交或自动保存到服务器。</p></section>
__CARDS__
<section class="export"><p id="progress" aria-live="polite"></p>
<button id="download" type="button">导出本地回答</button>
<p id="message" role="status" aria-live="polite"></p></section>
</main><script>
'use strict';
const manifest=__MANIFEST__;
const answers=manifest.items.map(item=>({item_id:item.item_id,
frame_sha256:item.frame_sha256,decision:'',points:[],
inventory_complete:false,reviewed:false}));
const cards=[...document.querySelectorAll('.card')];
const roles={baby:'宝宝',adult:'成人',unknown:'年龄不明'};
function imageReady(index){
 const image=cards[index].querySelector('img'),item=manifest.items[index];
 return image.complete && image.naturalWidth===item.width && image.naturalHeight===item.height;
}
function resetAcknowledgment(card,a){
 a.reviewed=false;card.querySelector('.reviewed').checked=false;
}
function redraw(card,a){
 const svg=card.querySelector('svg');svg.replaceChildren();
 for(const [i,p] of a.points.entries()){
  const circle=document.createElementNS('http://www.w3.org/2000/svg','circle');
  circle.setAttribute('cx',p.point_xy[0]);circle.setAttribute('cy',p.point_xy[1]);circle.setAttribute('r',.013);
  circle.setAttribute('fill','#ffdf47');circle.setAttribute('stroke','#172844');circle.setAttribute('stroke-width',.002);svg.append(circle);
  const text=document.createElementNS('http://www.w3.org/2000/svg','text');text.setAttribute('x',p.point_xy[0]+.015);text.setAttribute('y',p.point_xy[1]);text.setAttribute('fill','#ffdf47');text.setAttribute('font-size',.028);text.textContent=String(i+1);svg.append(text);
 }
 card.querySelector('.points').textContent=a.points.length?a.points.map((p,i)=>
  `${i+1}：${roles[p.role]}（x ${(p.point_xy[0]*100).toFixed(2)}%，`+
  `y ${(p.point_xy[1]*100).toFixed(2)}%）`).join('；'):'尚未标记头部';
 update();
}
function valid(a,index){return a.reviewed && (a.decision==='unknown'||imageReady(index)) && (
 (a.decision==='marked' && a.points.length>0) ||
 (a.decision==='no_visible_head' && !a.points.length && a.inventory_complete) ||
 (a.decision==='unknown' && !a.points.length && !a.inventory_complete));
}
function update(){
 const n=answers.filter((a,index)=>valid(a,index)).length;
 document.querySelector('#progress').textContent=
  `已确认 ${n}/${answers.length} 张；看不清可以跳过。`;
 document.querySelector('#download').disabled=n!==answers.length;
}
for(const [index,card] of cards.entries()){
 const a=answers[index],image=card.querySelector('img');
 const decision=card.querySelector('.decision'),complete=card.querySelector('.complete');
 function showImageStatus(){
  const status=card.querySelector('.load-status'),ready=imageReady(index);
  status.className=ready?'load-status success':'load-status error';
  status.textContent=ready?'原画面已加载，尺寸与绑定一致。':'画面未加载或尺寸不符；仅可确认“看不清／跳过”，不能确认没有头部。';
  if(!ready){a.points=[];a.inventory_complete=a.decision==='no_visible_head';complete.checked=a.inventory_complete;resetAcknowledgment(card,a);redraw(card,a);}
  else update();
 }
 image.addEventListener('load',showImageStatus);image.addEventListener('error',showImageStatus);
 if(image.complete)showImageStatus();
 decision.addEventListener('change',()=>{
  a.decision=decision.value;a.points=[];a.inventory_complete=a.decision==='no_visible_head';complete.checked=a.inventory_complete;
  complete.disabled=a.decision!=='marked';card.querySelector('.role').disabled=a.decision!=='marked';resetAcknowledgment(card,a);redraw(card,a);
 });
 complete.disabled=true;card.querySelector('.role').disabled=true;
 image.addEventListener('click',event=>{
  const role=card.querySelector('.role').value;
  if(a.decision!=='marked'||!role||a.points.length>=16||!imageReady(index)){document.querySelector('#message').textContent='请先选择能定位头部和年龄类别；画面未加载或点数到限时不能标记。';return;}
  const r=image.getBoundingClientRect();
  const x=(event.clientX-r.left)/r.width,y=(event.clientY-r.top)/r.height;
  if(x<0||x>1||y<0||y>1||!Number.isFinite(x)||!Number.isFinite(y))return;
  if(a.points.some(p=>p.point_xy[0]===x&&p.point_xy[1]===y))return;
  a.points.push({point_id:`head_${String(a.points.length+1).padStart(2,'0')}`,point_xy:[x,y],role});
  a.inventory_complete=false;complete.checked=false;resetAcknowledgment(card,a);document.querySelector('#message').textContent='';redraw(card,a);
 });
 for(const button of card.querySelectorAll('.zoom'))button.addEventListener('click',()=>{
  card.querySelector('.stage').style.width=`${Number(button.dataset.scale)*100}%`;
 });
 card.querySelector('.undo').addEventListener('click',()=>{a.points.pop();a.inventory_complete=false;complete.checked=false;resetAcknowledgment(card,a);redraw(card,a);});
 complete.addEventListener('change',()=>{a.inventory_complete=complete.checked;resetAcknowledgment(card,a);update();});
 card.querySelector('.reviewed').addEventListener('change',event=>{a.reviewed=event.target.checked;update();});
}
document.querySelector('#download').addEventListener('click',()=>{
 if(!answers.every((a,index)=>valid(a,index)))return;
 const result={version:1,scope:manifest.scope,manifest_sha256:'__FINGERPRINT__',
 reviewer_id:'__REVIEWER__',provenance:'__PROVENANCE__',answers};
 const blob=new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'});
 const url=URL.createObjectURL(blob),anchor=document.createElement('a');
 anchor.href=url;anchor.download=manifest.review_id+'-answers.json';anchor.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
 document.querySelector('#message').className='success';
 document.querySelector('#message').textContent=
  '已生成本地下载；没有上传或提交到服务器。请保留下载的 JSON。';
});update();
</script></body></html>"""


def prepare(source: Path, output: Path, *, synthetic: bool = False) -> dict:
    raw = read_blob(source, limit=128 * 1024)
    value = parse_json(raw)
    if (
        type(value) is not dict
        or set(value) != {"version", "items"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or type(value["items"]) is not list
    ):
        raise HeadPointReviewError("invalid review source manifest")
    items, paths = [], []
    expected = {
        "item_id",
        "camera_id",
        "frame_sha256",
        "frame_received_at",
        "width",
        "height",
        "image_path",
    }
    for item in value["items"]:
        if type(item) is not dict or set(item) != expected or type(item["image_path"]) is not str:
            raise HeadPointReviewError("invalid review source item")
        items.append({key: val for key, val in item.items() if key != "image_path"})
        paths.append(Path(item["image_path"]))
    manifest = validate_manifest(
        {
            "version": 1,
            "scope": "frame_visible_head_points",
            "review_id": "review_" + secrets.token_hex(12),
            "selection_purpose": "development_counterexamples",
            "blinding_scope": "page_hides_models_and_previous_answers",
            "items": items,
        }
    )
    images = [read_blob(path, limit=4 * 1024 * 1024) for path in paths]
    if sum(map(len, images)) > 8 * 1024 * 1024:
        raise HeadPointReviewError("review images exceed package limit")
    page = build_page(manifest, images, synthetic=synthetic).encode()
    if read_blob(source, limit=128 * 1024) != raw:
        raise HeadPointReviewError("review source manifest changed")
    new_output_directory(output)
    output.mkdir(mode=0o700)
    _write(output / "manifest.json", _json(manifest))
    _write(output / "review.local.html", page)
    _write(
        output / "package.json",
        _json(
            {
                "version": 1,
                "manifest_sha256": fingerprint(manifest),
                "page_sha256": hashlib.sha256(page).hexdigest(),
                "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "synthetic": synthetic,
                "media_uploads_performed": False,
                "model_calls_performed": False,
                "item_count": len(items),
            }
        ),
    )
    return {
        "item_count": len(items),
        "manifest_sha256": fingerprint(manifest),
        "synthetic": synthetic,
    }


def read_bundle(root: Path) -> tuple[dict, dict, bytes]:
    manifest = validate_manifest(parse_json(read_blob(root / "manifest.json", limit=128 * 1024)))
    package = parse_json(read_blob(root / "package.json", limit=128 * 1024))
    page = read_blob(root / "review.local.html", limit=16 * 1024 * 1024)
    fields = {
        "version",
        "manifest_sha256",
        "page_sha256",
        "source_manifest_sha256",
        "synthetic",
        "media_uploads_performed",
        "model_calls_performed",
        "item_count",
    }
    if (
        type(package) is not dict
        or set(package) != fields
        or type(package["version"]) is not int
        or package["version"] != 1
        or type(package["synthetic"]) is not bool
        or package["media_uploads_performed"] is not False
        or package["model_calls_performed"] is not False
        or type(package["item_count"]) is not int
        or package["item_count"] != len(manifest["items"])
        or package["manifest_sha256"] != fingerprint(manifest)
        or package["page_sha256"] != hashlib.sha256(page).hexdigest()
        or type(package["source_manifest_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", package["source_manifest_sha256"]) is None
    ):
        raise HeadPointReviewError("review package binding mismatch")
    return manifest, package, page


def admit_answers(root: Path, answer_path: Path, output: Path) -> dict:
    manifest, package, _page = read_bundle(root)
    review = validate_review(parse_json(read_blob(answer_path, limit=128 * 1024)), manifest)
    if review["provenance"] != ("synthetic" if package["synthetic"] else "human_user_review"):
        raise HeadPointReviewError("review package provenance mismatch")
    # Browser downloads must be copied into a private local input directory first.
    _write(output, _json(review))
    return {
        "answer_count": len(review["answers"]),
        "review_sha256": fingerprint(review),
        "provenance": review["provenance"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare")
    make.add_argument("--source", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--synthetic", action="store_true")
    admit = commands.add_parser("admit")
    admit.add_argument("--review-root", type=Path, required=True)
    admit.add_argument("--answers", type=Path, required=True)
    admit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            prepare(args.source, args.output, synthetic=args.synthetic)
            if args.command == "prepare"
            else admit_answers(args.review_root, args.answers, args.output)
        )
        print(json.dumps(result, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError):
        print(json.dumps({"ok": False, "error_code": "head_point_review_rejected"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
