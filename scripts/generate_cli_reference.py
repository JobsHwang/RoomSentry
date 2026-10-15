"""Regenerate docs/cli-reference.md from the current CLI parser definition.

The script imports ``baby_monitor.cli.build_parser`` and renders one
deterministic Markdown row per registered subcommand. Run it with ``--check``
in CI so the reference cannot drift from the actual command surface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baby_monitor.cli import build_parser

_HEADER = """# CLI 参考

本页由 `scripts/generate_cli_reference.py` 从当前 CLI 定义自动生成，请勿手工编辑。
命令按字典序排列；隐私、凭据与授权边界以 [SPEC.md](../SPEC.md) 为准。

| 命令 | 说明 |
|---|---|
"""


def _collect_commands() -> list[tuple[str, str]]:
    parser = build_parser()
    rows: list[tuple[str, str]] = []
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        helps = {item.dest: item.help for item in action._choices_actions}
        for name in action.choices:
            help_text = helps.get(name) or ""
            rows.append((name, help_text.replace("|", "\\|")))
        break
    rows.sort(key=lambda row: row[0])
    return rows


def _render(rows: list[tuple[str, str]]) -> str:
    lines = [_HEADER]
    lines.extend(f"| `{name}` | {help_text} |" for name, help_text in rows)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/cli-reference.md"),
    )
    argument_parser.add_argument(
        "--check",
        action="store_true",
        help="verify the file matches the generated content instead of writing",
    )
    args = argument_parser.parse_args(argv)

    content = _render(_collect_commands())
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if current != content:
            print(
                "error: docs/cli-reference.md is out of date; "
                "regenerate it with scripts/generate_cli_reference.py",
                file=sys.stderr,
            )
            return 1
        print("cli reference is current")
        return 0

    args.output.write_text(content, encoding="utf-8")
    print(f"wrote {len(content.splitlines())} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
