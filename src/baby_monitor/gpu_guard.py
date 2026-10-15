from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass


class GPUProbeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GPUCapacity:
    allowed: bool
    reason: str
    device_index: int | None
    memory_total_mib: int | None
    memory_used_mib: int | None
    memory_free_mib: int | None
    utilization_percent: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_gpu_capacity(
    *,
    device: str,
    min_free_vram_mib: int,
    max_utilization_percent: int,
    timeout_seconds: int,
) -> GPUCapacity:
    if not device.isdigit():
        raise ValueError("GPU device must be a numeric index")
    if min_free_vram_mib < 1:
        raise ValueError("minimum free VRAM must be positive")
    if not 0 <= max_utilization_percent <= 100:
        raise ValueError("maximum GPU utilization must be between 0 and 100")
    if not 1 <= timeout_seconds <= 30:
        raise ValueError("GPU probe timeout must be between 1 and 30 seconds")

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return GPUCapacity(False, "probe_unavailable", None, None, None, None, None)
    environment = {
        key: value for key, value in os.environ.items() if key in {"PATH", "SYSTEMROOT", "WINDIR"}
    }
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--id",
                device,
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return GPUCapacity(False, "probe_unavailable", None, None, None, None, None)
    if completed.returncode != 0 or len(completed.stdout) > 4_096:
        return GPUCapacity(False, "probe_unavailable", None, None, None, None, None)
    rows = list(csv.reader(io.StringIO(completed.stdout.strip())))
    if len(rows) != 1 or len(rows[0]) != 5:
        return GPUCapacity(False, "probe_unavailable", None, None, None, None, None)
    try:
        index, total, used, free, utilization = (int(value.strip()) for value in rows[0])
    except ValueError:
        return GPUCapacity(False, "probe_unavailable", None, None, None, None, None)
    if min(index, total, used, free, utilization) < 0 or utilization > 100:
        return GPUCapacity(False, "probe_unavailable", None, None, None, None, None)
    if free < min_free_vram_mib:
        reason = "insufficient_free_vram"
        allowed = False
    elif utilization > max_utilization_percent:
        reason = "gpu_busy"
        allowed = False
    else:
        reason = "gpu_capacity_available"
        allowed = True
    return GPUCapacity(allowed, reason, index, total, used, free, utilization)
