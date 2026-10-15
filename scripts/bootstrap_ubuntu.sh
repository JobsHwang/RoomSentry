#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

if [[ ! -r /etc/os-release ]]; then
  printf 'Cannot identify the Linux distribution.\n' >&2
  exit 1
fi
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
  printf 'Expected Ubuntu 24.04, found %s %s.\n' "${ID:-unknown}" "${VERSION_ID:-unknown}" >&2
  exit 1
fi
if [[ ! -e /dev/dxg ]]; then
  printf 'WSL GPU device /dev/dxg is absent. Fix WSL/GPU passthrough before model setup.\n' >&2
  exit 1
fi

printf 'This installs only the lightweight development toolchain (no model weights).\n'
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git \
  python3 \
  python3-pip \
  python3-venv \
  ffmpeg \
  build-essential

bash "$repo_root/scripts/install_core_venv.sh" --venv "$repo_root/.venv" --python python3

printf 'Core environment ready. No PyPI package, YOLO weight, or VLM model was downloaded.\n'
