#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: install_core_venv.sh [--venv PATH] [--python COMMAND]

Create or refresh a Python virtual environment for the standard-library core.
The installer does not invoke pip, access PyPI, or install optional vision/model
dependencies. It links this source checkout into the venv with a .pth file.
EOF
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
venv_dir="${repo_root}/.venv"
python_bin="python3"

while (($#)); do
  case "$1" in
    --venv)
      if (($# < 2)); then
        printf '%s\n' 'Missing value for --venv.' >&2
        exit 2
      fi
      venv_dir="$2"
      shift 2
      ;;
    --python)
      if (($# < 2)); then
        printf '%s\n' 'Missing value for --python.' >&2
        exit 2
      fi
      python_bin="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$venv_dir" != /* ]]; then
  venv_dir="${repo_root}/${venv_dir}"
fi
if [[ -L "$venv_dir" ]]; then
  printf '%s\n' 'Refusing a symlink virtual-environment target.' >&2
  exit 2
fi
venv_dir="$(realpath -m -- "$venv_dir")"

case "$venv_dir" in
  /|"$repo_root"|"$repo_root/src")
    printf '%s\n' 'Refusing an unsafe virtual-environment target.' >&2
    exit 2
    ;;
esac

if [[ -e "$venv_dir" && ! -f "$venv_dir/pyvenv.cfg" ]]; then
  printf '%s\n' 'Refusing to reuse a non-venv target.' >&2
  exit 2
fi

"$python_bin" -c \
  'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'

if [[ ! -e "$venv_dir" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi

venv_python="$venv_dir/bin/python"
if [[ ! -x "$venv_python" ]]; then
  printf '%s\n' 'The target does not contain an executable venv Python.' >&2
  exit 2
fi

source_root="$repo_root/src"
case "$source_root" in
  *$'\n'*)
    printf '%s\n' 'The source path is unsafe for a .pth file.' >&2
    exit 2
    ;;
esac

site_packages="$($venv_python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
pth_path="$site_packages/baby_monitor_source.pth"
tmp_path="$site_packages/.baby_monitor_source.pth.tmp.$$"
cleanup() {
  rm -f -- "$tmp_path"
}
trap cleanup EXIT
umask 022
printf '%s\n' "$source_root" >"$tmp_path"
chmod 0644 "$tmp_path"
mv -f -- "$tmp_path" "$pth_path"
trap - EXIT

"$venv_python" -I -m baby_monitor validate-json \
  "$repo_root/tests/fixtures/valid_vlm_event.json" >/dev/null
"$venv_python" -I -m baby_monitor schema >/dev/null

printf '%s\n' 'Core venv ready from the local source checkout.'
printf '%s\n' 'No PyPI package, optional vision dependency, or model weight was downloaded.'
