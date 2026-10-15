#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
python_bin="${BABY_MONITOR_RELEASE_PYTHON:-python3}"
work_dir="$(mktemp -d -p /tmp baby-monitor-release.XXXXXXXX)"

cleanup() {
  case "$work_dir" in
    /tmp/baby-monitor-release.*)
      rm -rf -- "$work_dir"
      ;;
    *)
      printf '%s\n' 'Refusing to clean an unexpected release workspace.' >&2
      ;;
  esac
}
trap cleanup EXIT

cd "$repo_root"
"$python_bin" -c \
  'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'

export PYTHONPATH="$repo_root/src"
"$python_bin" -m baby_monitor release-audit --root "$repo_root" \
  >"$work_dir/release-audit.json"
"$python_bin" -m baby_monitor release-sbom --root "$repo_root" \
  >"$work_dir/standalone.cdx.json"
"$python_bin" -m baby_monitor release-bundle \
  --root "$repo_root" \
  --output "$work_dir/source-a.tar.gz" \
  >"$work_dir/source-a-summary.json"
"$python_bin" -m baby_monitor release-bundle \
  --root "$repo_root" \
  --output "$work_dir/source-b.tar.gz" \
  >"$work_dir/source-b-summary.json"
cmp --silent "$work_dir/source-a.tar.gz" "$work_dir/source-b.tar.gz"

mkdir "$work_dir/extracted"
tar -xzf "$work_dir/source-a.tar.gz" --no-same-owner -C "$work_dir/extracted"
mapfile -t source_roots < <(
  find "$work_dir/extracted" -mindepth 1 -maxdepth 1 -type d -name 'baby-monitor-*'
)
if ((${#source_roots[@]} != 1)); then
  printf '%s\n' 'Release archive has an unexpected root.' >&2
  exit 1
fi
source_root="${source_roots[0]}"
if find "$source_root" -type l -print -quit | grep -q .; then
  printf '%s\n' 'Release archive contains a symlink.' >&2
  exit 1
fi
cmp --silent "$work_dir/standalone.cdx.json" "$source_root/SBOM.cdx.json"

# An alternate release index is valid for audit/bundle creation, but must not
# override repositories created by the extracted test suite.
unset GIT_INDEX_FILE
cd "$source_root"
bash scripts/install_core_venv.sh --python "$python_bin"
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t . -v
PYTHONPATH=src .venv/bin/python -m compileall -q src tests
PYTHONPATH=src .venv/bin/python -m baby_monitor validate-json \
  tests/fixtures/valid_vlm_event.json >/dev/null

if [[ -x "$repo_root/.venv/bin/ruff" ]]; then
  "$repo_root/.venv/bin/ruff" check src tests
  lint_status="passed"
else
  lint_status="skipped_not_installed"
fi

"$python_bin" - "$work_dir/source-a-summary.json" "$lint_status" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(
    json.dumps(
        {
            "version": 1,
            "status": "passed",
            "archive_files": summary["archive_files"],
            "staged_source_files": summary["staged_source_files"],
            "includes_source_sbom": summary["includes_source_sbom"],
            "deterministic_archive": True,
            "extracted_core_tests": "passed",
            "lint": sys.argv[2],
            "network_used": False,
            "model_loaded": False,
            "path_disclosed": False,
            "content_disclosed": False,
        },
        ensure_ascii=False,
        indent=2,
    )
)
PY
