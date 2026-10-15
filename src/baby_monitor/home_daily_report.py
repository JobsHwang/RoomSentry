from __future__ import annotations

import html
import os
import secrets
import stat
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .home_event_digest import HomeEventDigest, HomeEventDigestError, read_home_event_digest
from .presence_history import (
    PresenceDaySummary,
    PresenceFactMode,
    PresenceHistoryError,
    read_presence_day_summary,
)
from .sampling_history import (
    SamplingDaySummary,
    SamplingHistoryError,
    read_sampling_day_summary,
)

_MAX_REPORT_BYTES = 256 * 1024
_EVENT_LABELS = {
    "unknown": "未知",
    "sleeping": "睡眠分类",
    "awake": "清醒分类",
    "crying": "哭闹分类",
    "movement": "活动分类",
    "adult_interaction": "成人互动分类",
    "possible_risk": "可能风险分类",
}
_RISK_LABELS = {
    "none": "无",
    "low": "低",
    "medium": "中",
    "high": "高",
}


class HomeDailyReportError(RuntimeError):
    """Path-free failure for the aggregate local home report."""


@dataclass(frozen=True, slots=True)
class HomeDailyReportReceipt:
    local_date: str
    event_count: int
    presence_sample_count: int
    shadow_presence_sample_count: int
    presence_history_available: bool
    generated_at: str

    def to_public_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "status": "generated",
            "local_date": self.local_date,
            "event_count": self.event_count,
            "presence_sample_count": self.presence_sample_count,
            "shadow_presence_sample_count": self.shadow_presence_sample_count,
            "presence_history_available": self.presence_history_available,
            "generated_at": self.generated_at,
            "markdown_written": True,
            "html_written": True,
            "media_read": False,
            "paths_disclosed": False,
            "event_ids_disclosed": False,
            "camera_ids_disclosed": False,
            "summaries_disclosed": False,
            "risk_authority": False,
            "notification_decision": False,
            "external_delivery_used": False,
        }


def _admit_database(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    parent = candidate.parent
    try:
        metadata = candidate.lstat()
        parent_metadata = parent.lstat()
        resolved = candidate.resolve(strict=True)
        resolved_parent = parent.resolve(strict=True)
    except OSError:
        raise HomeDailyReportError("home daily report database is unavailable") from None
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or resolved != candidate
        or parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        or resolved_parent != parent
    ):
        raise HomeDailyReportError("home daily report database is unavailable")
    return resolved


def _admit_output_root(
    database: Path,
    output_root: str | Path,
    *,
    create: bool,
) -> Path | None:
    target = Path(output_root).expanduser()
    expected = database.parent / "daily-reports"
    if not target.is_absolute() or target != expected:
        raise HomeDailyReportError("home daily report output is unavailable")
    if not target.exists() and not create:
        return None
    try:
        if create:
            target.mkdir(mode=0o700, exist_ok=True)
        metadata = target.lstat()
        resolved = target.resolve(strict=True)
    except OSError:
        raise HomeDailyReportError("home daily report output is unavailable") from None
    if (
        target.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or resolved != target
    ):
        raise HomeDailyReportError("home daily report output is unavailable")
    return resolved


def _report_paths(root: Path, local_date: str) -> tuple[Path, Path]:
    stem = f"{local_date}_daily_report"
    return root / f"{stem}.md", root / f"{stem}.html"


def _verify_private_file(path: Path, *, root: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        raise HomeDailyReportError("home daily report is unavailable") from None
    if (
        path.parent != root
        or path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or resolved != path
        or not 0 < metadata.st_size <= _MAX_REPORT_BYTES
    ):
        raise HomeDailyReportError("home daily report is unavailable")
    return metadata


def _atomic_replace(path: Path, payload: bytes, *, root: Path) -> None:
    if not payload or len(payload) > _MAX_REPORT_BYTES:
        raise HomeDailyReportError("home daily report generation failed")
    if path.exists() or path.is_symlink():
        _verify_private_file(path, root=root)
    temporary = root / f".{path.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError:
        raise HomeDailyReportError("home daily report generation failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    _verify_private_file(path, root=root)


def _count_lines(values: tuple[tuple[str, int], ...], labels: dict[str, str]) -> list[str]:
    result = [f"- {labels.get(key, key)}：{count}" for key, count in values if count]
    return result or ["- 无已落库记录。"]


def _duration_label(seconds: float | None) -> str:
    if seconds is None:
        return "无可计算样本"
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    if seconds < 3600:
        return f"{seconds / 60:.1f} 分钟"
    return f"{seconds / 3600:.1f} 小时"


def _presence_markdown_lines(summary: PresenceDaySummary | None) -> list[str]:
    if summary is None:
        return [
            "- presence 历史表尚不可用；不能据此判断当天是否持续监督。",
            "- 旧事件仍可汇总，但没有 person/clear/coverage-gap 的日内事实。",
        ]
    if not summary.streams:
        return [
            "- 当天没有已落库 presence 事实；这既不表示房间无人，也不表示系统持续在线。",
            "- person/clear 均为 0 不能解释为安全结论。",
        ]
    timezone = ZoneInfo(summary.timezone_name)
    first = min(item.first_observed_at for item in summary.streams).astimezone(timezone)
    last = max(item.last_observed_at for item in summary.streams).astimezone(timezone)
    return [
        f"- 已落库 presence 样本：{summary.sample_count}（{len(summary.streams)} 路）",
        f"- 通用 person 检测样本：{summary.person_detected_count}",
        f"- clear 样本：{summary.clear_count}",
        f"- coverage-gap 样本：{summary.coverage_gap_count}",
        f"- 首末事实时间（含缺口评估）：{first.strftime('%H:%M:%S')}–{last.strftime('%H:%M:%S')}",
        "- 单路最大事实记录间隔（包含日界空档）："
        + _duration_label(summary.maximum_observation_gap_seconds),
        "- presence 样本是业务事实条数，不是模型调用次数；"
        "同一片段可同时保留 person 和 coverage-gap。",
        "- 新主链路 person/clear 使用实际检查时间；"
        "coverage-gap 时间是缺口评估边界，不是成功检查时间。",
        "- 历史记录未自动迁移；旧记录的时间不因此获得新的采样证据。",
        "- person 只表示画面中检测到通用人物，不能据此确认是宝宝、成人或具体身份。",
    ]


def _shadow_presence_markdown_lines(summary: PresenceDaySummary | None) -> list[str]:
    if summary is None or not summary.streams:
        return []
    return [
        "## Detector + Pose shadow（实验旁路）",
        "",
        f"- shadow 样本：{summary.sample_count}（{len(summary.streams)} 路）",
        f"- shadow person：{summary.person_detected_count}",
        f"- shadow clear：{summary.clear_count}",
        f"- shadow coverage-gap：{summary.coverage_gap_count}",
        "- shadow 与主 presence 分开计数，不改变事件、房间状态、风险、通知或照护结论。",
        "- 该旁路绑定冻结 profile、时间 holdout 与单人盲审，仍不是官方准确率或部署批准。",
        "",
    ]


def _sampling_markdown_lines(summary: SamplingDaySummary | None) -> list[str]:
    boundaries = [
        "- 调用次数不是人数、独立画面数或连续观察时长；同一画面可能被重复扫描。",
        "- 零框只表示这次模型调用未检出目标，不表示画面无人、宝宝不在场或环境安全。",
        "- 未验证源视频是否完整读尽；这些记录并非连续监护，也不能替代成人照护。",
    ]
    if summary is None:
        return [
            "- 实际检查历史表尚不可用；旧扫描不能补记为已经检查，调用次数未知。",
            *boundaries,
        ]
    gap = summary.max_adjacent_container_pts_gap_seconds
    gap_label = "无可计算记录" if gap is None else f"{gap:.3f} 秒"
    total_scans = summary.recorded_scan_count + summary.unknown_scan_count
    lines = [
        f"- 已记录扫描：{total_scans}",
        f"- 实际模型调用（已记录）：{summary.sampled_frame_count}",
        f"- 其中零框调用：{summary.zero_detection_sample_count}",
        f"- 缺少实际采样证据的扫描：{summary.unknown_scan_count}",
        f"- 使用回退时间戳的采样：{summary.timestamp_fallback_sample_count}",
        f"- 源容器 PTS 回退次数：{summary.container_pts_regression_count}",
        f"- 相邻源帧最大容器 PTS 间隔：{gap_label}",
        "- PTS 间隔只是已记录扫描的时间轴线索，不是确认丢失时长，也不覆盖扫描之间的空档。",
    ]
    if total_scans == 0:
        lines.append("- 当天没有已落库扫描；0 次不表示系统在线或全天无事。")
    elif summary.unknown_scan_count:
        lines.append("- 已记录调用数不包含缺少采样证据的扫描，不能当作当天完整调用总数。")
    return [*lines, *boundaries]


def _markdown(
    digest: HomeEventDigest,
    presence: PresenceDaySummary | None,
    shadow_presence: PresenceDaySummary | None,
    generated_at: datetime,
    sampling: SamplingDaySummary | None = None,
) -> str:
    lines = [
        f"# 本地隐私日报 — {digest.local_date}",
        "",
        "> 本报告只汇总已落库记录，不读取或附带媒体。没有记录不代表环境安全，"
        "分类与风险字段也不构成医疗或照护结论。",
        "",
        "## 今日概览",
        "",
        f"- 已落库事件：{digest.today_total}",
        f"- 生成时间（UTC）：{generated_at.astimezone(UTC).isoformat()}",
        "- 摄像头、事件 ID、模型摘要、媒体和本机路径：未包含",
        "",
        "## 事件分类",
        "",
        *_count_lines(digest.event_type_counts, _EVENT_LABELS),
        "",
        "## 风险字段分布",
        "",
        *_count_lines(digest.risk_level_counts, _RISK_LABELS),
        "",
        "## 宝宝监督覆盖（系统事实）",
        "",
        *_presence_markdown_lines(presence),
        "",
        *_shadow_presence_markdown_lines(shadow_presence),
        "## 实际检查记录",
        "",
        *_sampling_markdown_lines(sampling),
        "",
        "## 当天最近记录（最多 16 条）",
        "",
        "| 时间 | 分类 | 风险字段 | 置信度 | 待复核 |",
        "|---|---|---|---:|---|",
    ]
    for entry in digest.recent:
        lines.append(
            f"| {entry.local_time} | {_EVENT_LABELS[entry.event_type]} | "
            f"{_RISK_LABELS[entry.risk_level]} | {entry.confidence:.0%} | "
            f"{'是' if entry.needs_review else '否'} |"
        )
    if not digest.recent:
        lines.append("| — | — | — | — | — |")
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 本报告不证明全天摄像头、HA、缓冲或模型持续在线。",
            "- 本报告不读取图片或视频，不进行身份推断，也不改变任何风险等级。",
            "- presence 采样与事件分类都不是连续生命体征监护，也不能替代成人照护。",
            "- 生成本报告不会触发通知规则、外部投递、上传、训练或删除。",
            "",
        ]
    )
    return "\n".join(lines)


def _presence_html(summary: PresenceDaySummary | None) -> str:
    return "".join(
        f"<li>{html.escape(line[2:])}</li>" for line in _presence_markdown_lines(summary)
    )


def _shadow_presence_html(summary: PresenceDaySummary | None) -> str:
    lines = _shadow_presence_markdown_lines(summary)
    if not lines:
        return ""
    items = "".join(f"<li>{html.escape(line[2:])}</li>" for line in lines if line.startswith("- "))
    return f"<h2>Detector + Pose shadow（实验旁路）</h2><ul>{items}</ul>"


def _sampling_html(summary: SamplingDaySummary | None) -> str:
    return "".join(
        f"<li>{html.escape(line[2:])}</li>" for line in _sampling_markdown_lines(summary)
    )


def _html(
    digest: HomeEventDigest,
    presence: PresenceDaySummary | None,
    shadow_presence: PresenceDaySummary | None,
    generated_at: datetime,
    sampling: SamplingDaySummary | None = None,
) -> str:
    event_items = (
        "".join(
            f"<li>{html.escape(_EVENT_LABELS[key])}：{count}</li>"
            for key, count in digest.event_type_counts
            if count
        )
        or "<li>无已落库记录。</li>"
    )
    risk_items = (
        "".join(
            f"<li>{html.escape(_RISK_LABELS[key])}：{count}</li>"
            for key, count in digest.risk_level_counts
            if count
        )
        or "<li>无已落库记录。</li>"
    )
    rows = (
        "".join(
            "<tr>"
            f"<td>{html.escape(entry.local_time)}</td>"
            f"<td>{html.escape(_EVENT_LABELS[entry.event_type])}</td>"
            f"<td>{html.escape(_RISK_LABELS[entry.risk_level])}</td>"
            f"<td>{entry.confidence:.0%}</td>"
            f"<td>{'是' if entry.needs_review else '否'}</td>"
            "</tr>"
            for entry in digest.recent
        )
        or '<tr><td colspan="5">当天没有已落库事件；不能据此判断安全。</td></tr>'
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>本地隐私日报 {html.escape(digest.local_date)}</title>
<link rel="stylesheet" href="/style.css"></head><body><main>
<div class="eyebrow">不含媒体、身份和路径</div>
<h1>本地隐私日报 — {html.escape(digest.local_date)}</h1>
<p class="notice">本报告只汇总已落库记录。没有记录不代表环境安全，
分类与风险字段也不构成医疗或照护结论。</p>
<section class="grid"><article class="card"><h2>今日概览</h2>
<div class="metric">{digest.today_total} 条事件</div>
<p class="muted">生成时间（UTC）：{html.escape(generated_at.astimezone(UTC).isoformat())}</p>
<p class="muted">未包含摄像头、事件 ID、模型摘要、媒体或本机路径。</p></article>
<article class="card"><h2>事件分类</h2><ul>{event_items}</ul></article>
<article class="card"><h2>风险字段分布</h2><ul>{risk_items}</ul></article></section>
<h2>宝宝监督覆盖（系统事实）</h2><ul>{_presence_html(presence)}</ul>
{_shadow_presence_html(shadow_presence)}
<h2>实际检查记录</h2><ul>{_sampling_html(sampling)}</ul>
<h2>当天最近记录（最多 16 条）</h2>
<table><thead><tr><th>时间</th><th>分类</th><th>风险字段</th><th>置信度</th><th>待复核</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>使用边界</h2><ul>
<li>不证明全天摄像头、HA、缓冲或模型持续在线。</li>
<li>不读取图片或视频，不进行身份推断，也不改变任何风险等级。</li>
<li>presence 采样与事件分类都不是连续生命体征监护，也不能替代成人照护。</li>
<li>不会触发通知规则、外部投递、上传、训练或删除。</li>
</ul><a class="control-link" href="/">返回状态页</a>
</main></body></html>"""


def generate_home_daily_report(
    *,
    database_path: str | Path,
    output_root: str | Path,
    timezone_name: str = "Asia/Shanghai",
    now: datetime | None = None,
) -> HomeDailyReportReceipt:
    observed_at = now or datetime.now(UTC)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise HomeDailyReportError("home daily report timestamp is invalid")
    database = _admit_database(database_path)
    try:
        digest = read_home_event_digest(
            database,
            timezone_name=timezone_name,
            recent_limit=16,
            recent_day_only=True,
            now=observed_at,
        )
        presence = read_presence_day_summary(
            database,
            local_date=date.fromisoformat(digest.local_date),
            timezone_name=timezone_name,
        )
        shadow_presence = read_presence_day_summary(
            database,
            local_date=date.fromisoformat(digest.local_date),
            timezone_name=timezone_name,
            mode=PresenceFactMode.SHADOW,
        )
        sampling = read_sampling_day_summary(
            database,
            local_date=date.fromisoformat(digest.local_date),
            timezone_name=timezone_name,
        )
    except (HomeEventDigestError, PresenceHistoryError, SamplingHistoryError, ValueError):
        raise HomeDailyReportError("home daily report database is unavailable") from None
    root = _admit_output_root(database, output_root, create=True)
    assert root is not None
    markdown_path, html_path = _report_paths(root, digest.local_date)
    _atomic_replace(
        markdown_path,
        _markdown(digest, presence, shadow_presence, observed_at, sampling).encode("utf-8"),
        root=root,
    )
    _atomic_replace(
        html_path,
        _html(digest, presence, shadow_presence, observed_at, sampling).encode("utf-8"),
        root=root,
    )
    return HomeDailyReportReceipt(
        local_date=digest.local_date,
        event_count=digest.today_total,
        presence_sample_count=presence.sample_count if presence is not None else 0,
        shadow_presence_sample_count=(
            shadow_presence.sample_count if shadow_presence is not None else 0
        ),
        presence_history_available=presence is not None,
        generated_at=observed_at.astimezone(UTC).isoformat(),
    )


def inspect_home_daily_report(
    *,
    database_path: str | Path,
    output_root: str | Path,
    timezone_name: str = "Asia/Shanghai",
    now: datetime | None = None,
) -> dict[str, object]:
    common = {
        "version": 1,
        "media_read": False,
        "paths_disclosed": False,
        "event_ids_disclosed": False,
        "camera_ids_disclosed": False,
        "summaries_disclosed": False,
        "risk_authority": False,
        "notification_decision": False,
    }
    try:
        database = _admit_database(database_path)
        digest = read_home_event_digest(
            database,
            timezone_name=timezone_name,
            recent_limit=1,
            recent_day_only=True,
            now=now,
        )
        root = _admit_output_root(database, output_root, create=False)
        if root is None:
            return {
                **common,
                "status": "not_generated",
                "local_date": digest.local_date,
                "event_count": digest.today_total,
                "available": False,
                "operational": True,
                "attention_required": False,
            }
        markdown_path, html_path = _report_paths(root, digest.local_date)
        markdown_metadata = _verify_private_file(markdown_path, root=root)
        html_metadata = _verify_private_file(html_path, root=root)
    except (HomeDailyReportError, HomeEventDigestError, ValueError):
        return {
            **common,
            "status": "unavailable",
            "local_date": None,
            "event_count": 0,
            "available": False,
            "operational": False,
            "attention_required": True,
        }
    generated_at = datetime.fromtimestamp(
        max(markdown_metadata.st_mtime, html_metadata.st_mtime),
        tz=UTC,
    ).isoformat()
    return {
        **common,
        "status": "available",
        "local_date": digest.local_date,
        "event_count": digest.today_total,
        "available": True,
        "generated_at": generated_at,
        "operational": True,
        "attention_required": False,
    }


def read_home_daily_report_html(
    *,
    database_path: str | Path,
    output_root: str | Path,
    timezone_name: str = "Asia/Shanghai",
    now: datetime | None = None,
) -> bytes:
    database = _admit_database(database_path)
    try:
        digest = read_home_event_digest(
            database,
            timezone_name=timezone_name,
            recent_limit=1,
            recent_day_only=True,
            now=now,
        )
    except HomeEventDigestError:
        raise HomeDailyReportError("home daily report database is unavailable") from None
    root = _admit_output_root(database, output_root, create=False)
    if root is None:
        raise HomeDailyReportError("home daily report is unavailable")
    _markdown_path, html_path = _report_paths(root, digest.local_date)
    before = _verify_private_file(html_path, root=root)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(html_path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise HomeDailyReportError("home daily report is unavailable")
            payload = os.read(descriptor, _MAX_REPORT_BYTES + 1)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise HomeDailyReportError("home daily report is unavailable") from None
    if (
        (after.st_dev, after.st_ino, after.st_size)
        != (before.st_dev, before.st_ino, before.st_size)
        or not payload.startswith(b"<!doctype html>")
        or len(payload) != before.st_size
    ):
        raise HomeDailyReportError("home daily report is unavailable")
    return payload
