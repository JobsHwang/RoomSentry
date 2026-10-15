"""Regenerate docs/configuration.md from the settings implementation.

The reference is derived directly from ``baby_monitor.config``: section
dataclasses, field types and defaults come from runtime introspection, and
environment-variable names are extracted from the ``_env(...)`` calls inside
each section constructor via AST. Run with ``--check`` in CI so the document
cannot drift from the code.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import sys
from pathlib import Path

from baby_monitor.config import Settings

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_SOURCE = REPO_ROOT / "src" / "baby_monitor" / "config.py"

SECTION_INTRO = {
    "general": "全局设置（当前仅时区）。",
    "storage": "本地/NAS 存储与保留期；删除类开关默认必须保持 false。",
    "camera": "默认摄像头适配器；实时流由部署 profile 显式提供。",
    "detector": "低成本检测器；YOLO 需要本地权重与已复核 manifest，禁止自动下载。",
    "media": "事件窗口、关键帧数量与 FFmpeg 可执行文件。",
    "buffer": "循环缓冲目录、分段与重连参数；淘汰必须双重确认。",
    "media_health": "媒体健康快照的新鲜度判定。",
    "candidate_worker": "有界候选扫描 worker 的运行限制与候选启动门限。",
    "analysis_queue": "持久分析队列容量、租约与重试。",
    "analysis_worker": "顺序分析 worker 生命周期与可选 GPU 门槛。",
    "archive_queue": "归档队列容量、租约与重试。",
    "archive_worker": "顺序归档 worker 生命周期。",
    "local_vlm": "主本地 VLM 后端；同一时间只允许一个 VLM 驻留。",
    "review_policy": "复核升级策略；频率只提升复核层级。",
    "secondary_vlm": "可选第二本地复核器；必须在主模型之后顺序加载。",
    "remote_vlm": "远端复核；功能开关与上传同意缺一不可，默认全关。",
    "identity": "身份能力总开关；默认全部关闭。",
    "notification": "通知 outbox 与外部投递；外部投递独立开关且默认关闭。",
}

EXAMPLE_FILES = {
    "default.toml": "核心 TOML 默认值（本参考的数据来源之一）",
    "household-event-policy.example.json": "家庭事件包路由策略示例（规则引擎输入）",
    "ha-signal-routes.example.json": "HA 信号到本地媒体路由的私有绑定示例",
    "home-assistant-signals.example.json": "HA entity 到 SignalEvent 的显式映射示例",
    "iou-tracker.example.json": "IoU 跟踪器配置示例",
    "mediamtx.loopback.yml": "回环媒体路由器配置",
    "motion-two-stage-tracker.example.json": "motion-two-stage 跟踪器配置示例",
    "notification-smtp.example.json": "SMTP 外部投递私有 profile 示例",
    "notification-webhook.example.json": "HTTPS webhook 私有 profile 示例",
    "opencv-sface-2021dec.reviewed.example.json": "SFace 人脸模型 manifest 示例",
    "opencv-yunet-2023mar.reviewed.example.json": "YuNet 人脸检测模型 manifest 示例",
    "room-registry.example.json": "房间/物理摄像头/逻辑流注册表示例",
    "temporal-zones.example.json": "时序区域配置示例",
    "temporal-zones.person-public-smoke.example.json": "公开 smoke 用的区域配置",
    "temporal-zones.scene-smoke.example.json": "场景检测 smoke 用的区域配置",
    "vlm-ab.example.toml": "VLM A/B 评测配置示例",
    "yolo-model.example.json": "YOLO 权重 manifest 示例",
    "yolo11n-coco.reviewed.example.json": "已复核 YOLO11n 权重 manifest",
    "yolo11n-pose-coco.reviewed.example.json": "已复核 YOLO11n-Pose 权重 manifest",
    "yolo11n-subprocess.example.toml": "YOLO 子进程运行配置示例",
    "yolo26m-coco.reviewed.example.json": "已复核 YOLO26m 权重 manifest",
    "yolo26m-pose-coco.reviewed.example.json": "已复核 YOLO26m-Pose 权重 manifest",
}


def _section_env_map() -> dict[str, dict[str, str]]:
    """Map section class name -> {field name -> environment variable}."""

    tree = ast.parse(CONFIG_SOURCE.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if not name.endswith("Settings"):
            continue
        for keyword in node.keywords:
            for sub in ast.walk(keyword.value):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "_env"
                    and sub.args
                    and isinstance(sub.args[0], ast.Constant)
                    and isinstance(sub.args[0].value, str)
                ):
                    mapping.setdefault(name, {})[keyword.arg] = sub.args[0].value
                    break
    return mapping


def _format_default(value: object) -> str:
    if isinstance(value, tuple):
        text = "(" + ", ".join(repr(item) for item in value) + ")"
    elif isinstance(value, Path):
        text = str(value)
    else:
        text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"`{text}`"


def _render() -> str:
    env_map = _section_env_map()
    lines = [
        "# 配置参考",
        "",
        "本页由 `scripts/generate_config_reference.py` 从 `baby_monitor.config` 的实现"
        "自动生成，请勿手工编辑。",
        "",
        "## 配置加载顺序",
        "",
        "1. 代码内置默认值（本页表格中的默认列）；",
        "2. TOML 配置文件：CLI `--config` 指定，缺省 `config/default.toml`；",
        "3. 环境变量覆盖：仅表格中列出环境变量列的设置支持，前缀固定为 `BABY_MONITOR_`。",
        "",
        "层级只增不改：环境变量只覆盖同名 TOML 键，TOML 只覆盖内置默认值。",
        "",
        "## 隐私约定",
        "",
        "- `config/*.example.*` 是可公开的示例；真实部署使用 Git 忽略的"
        "`*.local.*` 文件或私有路径。",
        "- 密码、token、API key、RTSP 凭据只允许存在于环境变量或仓库外 `0600` 私有"
        "文件，禁止写入任何被跟踪文件。",
        "- `release-audit` 会在发布前扫描凭据、私有媒体扩展名和机器专属 home 路径。",
        "",
        "## 设置项",
        "",
    ]
    settings_fields = dataclasses.fields(Settings)
    simple_fields = [
        field
        for field in settings_fields
        if field.default is dataclasses.MISSING
        or not (dataclasses.is_dataclass(field.default) and not isinstance(field.default, type))
    ]
    if simple_fields:
        lines.append("### `[general]`")
        lines.append("")
        lines.append("| TOML 键 | 类型 | 默认值 | 环境变量 |")
        lines.append("|---|---|---|---|")
        for field in simple_fields:
            field_type = (field.type if isinstance(field.type, str) else str(field.type)).replace(
                "|", "\\|"
            )
            env = env_map.get("Settings", {}).get(field.name, "")
            lines.append(
                f"| `{field.name}` | {field_type} | "
                f"{_format_default(field.default)} | "
                f"{'`' + env + '`' if env else '—'} |"
            )
        lines.append("")
    for settings_field in settings_fields:
        default = settings_field.default
        if default is dataclasses.MISSING or not (
            dataclasses.is_dataclass(default) and not isinstance(default, type)
        ):
            simple_fields.append(settings_field.name)
            continue
        section_name = type(default).__name__
        section_fields = dataclasses.fields(type(default))
        env_fields = env_map.get(section_name, {})
        toml_section = settings_field.name
        lines.append(f"### `[{toml_section}]`")
        lines.append("")
        intro = SECTION_INTRO.get(toml_section, "")
        if intro:
            lines.append(intro)
            lines.append("")
        lines.append("| TOML 键 | 类型 | 默认值 | 环境变量 |")
        lines.append("|---|---|---|---|")
        for field in section_fields:
            field_type = field.type if isinstance(field.type, str) else str(field.type)
            field_type = (
                field_type.replace("typing.", "")
                .replace("baby_monitor.config.", "")
                .replace("|", "\\|")
            )
            env = env_fields.get(field.name, "")
            lines.append(
                f"| `{field.name}` | {field_type} | "
                f"{_format_default(field.default)} | "
                f"{'`' + env + '`' if env else '—'} |"
            )
        lines.append("")
    lines.append("## `config/` 示例文件")
    lines.append("")
    lines.append("| 文件 | 用途 |")
    lines.append("|---|---|")
    for name in sorted(EXAMPLE_FILES):
        lines.append(f"| `{name}` | {EXAMPLE_FILES[name]} |")
    lines.append("")
    lines.append("## 私有运行器环境变量")
    lines.append("")
    lines.append(
        "模型运行器还有少量独立便利变量（如 `BABY_MONITOR_MAGE_MODEL_PATH`、"
        "`BABY_MONITOR_MINICPM_MODEL_PATH`、`BABY_MONITOR_SMOLVLM2_MODEL_PATH`、"
        "`BABY_MONITOR_MAGE_CODE_MANIFEST`），只被对应 runner 读取，不参与核心 "
        "Settings；路径属于私有部署信息，不应提交。"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/configuration.md"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the file matches the generated content instead of writing",
    )
    args = parser.parse_args(argv)

    content = _render()
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if current != content:
            print(
                "error: docs/configuration.md is out of date; regenerate it "
                "with scripts/generate_config_reference.py",
                file=sys.stderr,
            )
            return 1
        print("configuration reference is current")
        return 0

    args.output.write_text(content, encoding="utf-8")
    print(f"wrote {len(content.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
