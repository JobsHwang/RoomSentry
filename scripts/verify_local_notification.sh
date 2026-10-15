#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-${project_root}/data/verification/local-notification-e2e-v1}"
output_root="$(realpath -m "${output_root}")"

if [[ -e "${output_root}" ]]; then
  echo "error: output target already exists: ${output_root}" >&2
  exit 2
fi

mkdir -p "${output_root}/file-sink"
cd "${project_root}"

.venv/bin/python -m baby_monitor smoke-demo \
  --output-root "${output_root}/pipeline" \
  >"${output_root}/smoke.json"

event_id=$(
  .venv/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["pipeline"]["stored_event_ids"][0])' \
    "${output_root}/smoke.json"
)
database="${output_root}/pipeline/events.sqlite3"

export BABY_MONITOR_NOTIFICATION_ENABLED=true
export BABY_MONITOR_NOTIFICATION_RETRY_DELAY_SECONDS=0

.venv/bin/python -m baby_monitor notification-enqueue-event \
  --database "${database}" \
  --event-id "${event_id}" \
  --sink file \
  --sink-id local_file \
  >"${output_root}/enqueue-file.json"

.venv/bin/python -m baby_monitor notification-enqueue-event \
  --database "${database}" \
  --event-id "${event_id}" \
  --sink stdout \
  --sink-id local_stdout \
  >"${output_root}/enqueue-stdout.json"

.venv/bin/python -m baby_monitor notification-run-once \
  --database "${database}" \
  --sink file \
  --sink-id local_file \
  --output-dir "${output_root}/file-sink" \
  >"${output_root}/delivery-file.json"

.venv/bin/python -m baby_monitor notification-run-once \
  --database "${database}" \
  --sink stdout \
  --sink-id local_stdout \
  >"${output_root}/stdout-message.json" \
  2>"${output_root}/delivery-stdout.json"

.venv/bin/python -m baby_monitor notification-queue-status \
  --database "${database}" \
  >"${output_root}/status.json"

.venv/bin/python -m baby_monitor validate-notification-json \
  "${output_root}/stdout-message.json" \
  >"${output_root}/validate-stdout.json"

file_message=$(find "${output_root}/file-sink" -maxdepth 1 -type f -name '*.json')
if [[ -z "${file_message}" || "${file_message}" == *$'\n'* ]]; then
  echo "error: expected exactly one file notification" >&2
  exit 2
fi
.venv/bin/python -m baby_monitor validate-notification-json \
  "${file_message}" \
  >"${output_root}/validate-file.json"

.venv/bin/python - "${output_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
message = json.loads((root / "stdout-message.json").read_text(encoding="utf-8"))
status = json.loads((root / "status.json").read_text(encoding="utf-8"))
serialized = json.dumps(message, ensure_ascii=False)
for forbidden in ("clip_path", "frame_paths", "evidence_frames", "source_ref"):
    if forbidden in serialized:
        raise SystemExit(f"forbidden notification field: {forbidden}")
print(
    json.dumps(
        {
            "review_status": message["review_status"],
            "event_type": message["event_type"],
            "risk_level": message["risk_level"],
            "observations": len(message["observations"]),
            "inferences": len(message["inferences"]),
            "queue": status["queue"],
            "file_notifications": len(list((root / "file-sink").glob("*.json"))),
            "external_delivery": False,
        },
        ensure_ascii=False,
    )
)
PY
