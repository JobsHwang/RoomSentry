#!/usr/bin/env bash
set -u

command_status() {
  local command_name="$1"
  if command -v "$command_name" >/dev/null 2>&1; then
    printf '%-14s %s\n' "$command_name" "$(command -v "$command_name")"
  else
    printf '%-14s %s\n' "$command_name" "NOT FOUND"
  fi
}

printf 'OS\n'
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  printf 'name=%s version=%s\n' "${PRETTY_NAME:-unknown}" "${VERSION_ID:-unknown}"
fi
printf 'kernel=%s\n' "$(uname -r)"
printf 'wsl_interop=%s\n' "${WSL_INTEROP:+present}"
printf 'dev_dxg=%s\n' "$(test -e /dev/dxg && echo present || echo absent)"

printf '\nTools\n'
for command_name in git python3 pip3 node npm docker ffmpeg ffprobe nvidia-smi; do
  command_status "$command_name"
done

printf '\nMemory\n'
free -h

printf '\nDisk\n'
df -hT / "$PWD"

printf '\nGPU\n'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu --format=csv,noheader
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu --format=csv,noheader
else
  printf 'nvidia-smi NOT FOUND\n'
fi

printf '\nDocker (read-only daemon capability check)\n'
if command -v docker >/dev/null 2>&1; then
  docker version --format 'client={{.Client.Version}} server={{.Server.Version}}' 2>/dev/null || true
  docker info --format 'runtimes={{json .Runtimes}} default={{.DefaultRuntime}}' 2>/dev/null || true
else
  printf 'docker NOT FOUND\n'
fi

printf '\nNo images, packages, models, or private files were downloaded or inspected.\n'

