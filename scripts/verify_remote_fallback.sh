#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-${project_root}/data/verification/remote-fallback-e2e-v1}"

if [[ -e "${output_root}" ]]; then
  echo "error: output target already exists: ${output_root}" >&2
  exit 2
fi

mkdir -p "${output_root}"

ffmpeg -hide_banner -loglevel error \
  -f lavfi -i testsrc2=size=320x180:rate=5 \
  -t 15 -c:v libx264 -pix_fmt yuv420p \
  "${output_root}/synthetic.mp4"

export BABY_MONITOR_REMOTE_ENABLED=true
export BABY_MONITOR_REMOTE_UPLOAD_CONSENT=true
export BABY_MONITOR_REMOTE_BASE_URL=http://127.0.0.1:18081/v1
export BABY_MONITOR_REMOTE_MODEL=local-fixture
export BABY_MONITOR_REMOTE_API_KEY=local-fixture-key
export BABY_MONITOR_REMOTE_MIN_INTERVAL_SECONDS=0

"${project_root}/.venv/bin/python" -m baby_monitor remote-mock-server \
  --host 127.0.0.1 \
  --port 18081 \
  --max-requests 1 \
  >"${output_root}/server.json" &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT

ready=false
for _ in {1..50}; do
  if grep -q '"status": "ready"' "${output_root}/server.json"; then
    ready=true
    break
  fi
  sleep 0.1
done
if [[ "${ready}" != true ]]; then
  echo "error: loopback mock did not become ready" >&2
  exit 2
fi

cd "${project_root}"
/usr/bin/time -v .venv/bin/python -m baby_monitor process-file \
  --source "${output_root}/synthetic.mp4" \
  --detector mock \
  --camera-id synthetic_cam \
  --database "${output_root}/events.sqlite3" \
  --output-root "${output_root}/output" \
  >"${output_root}/process.json" \
  2>"${output_root}/process.time"

wait "${server_pid}"
trap - EXIT

.venv/bin/python -m baby_monitor remote-audit-status \
  --database "${output_root}/events.sqlite3" \
  >"${output_root}/audit.json"

.venv/bin/python - "${output_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
process = json.loads((root / "process.json").read_text(encoding="utf-8"))
audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
event = process["events"][0]
print(
    json.dumps(
        {
            "candidates": process["candidates"],
            "remote_reviews": process["remote_reviews"],
            "pending_remote_reviews": process["pending_remote_reviews"],
            "final_source": event["decision"]["final_source"],
            "event_type": event["result"]["event_type"],
            "evidence_frames": len(event["result"]["evidence_frames"]),
            "audit_counts": audit["counts"],
            "audit_frame_count": audit["attempts"][0]["frame_count"],
            "audit_error_code": audit["attempts"][0]["error_code"],
        },
        ensure_ascii=False,
    )
)
PY

grep -E 'Elapsed|Maximum resident set size' "${output_root}/process.time"
