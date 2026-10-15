"""Measure daily presence aggregation using only a temporary synthetic database.

Run from the repository with the core virtualenv. Peak memory reports Python
allocations during the query/aggregation, not SQLite native memory or process RSS.
"""

from __future__ import annotations

import argparse
import gc
import json
import sqlite3
import tempfile
import time
import tracemalloc
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from baby_monitor.presence_history import PresenceFactStore, read_presence_day_summary


def benchmark(*, samples: int, cameras: int) -> dict[str, object]:
    if not 1 <= cameras <= samples <= 1_000_000:
        raise ValueError("require 1 <= cameras <= samples <= 1000000")
    day = date(2026, 9, 1)
    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    states = ("person_detected", "clear", "coverage_gap")
    with tempfile.TemporaryDirectory(prefix="baby-monitor-presence-benchmark-") as directory:
        database = Path(directory) / "synthetic.sqlite3"
        PresenceFactStore(database).initialize()
        connection = sqlite3.connect(database)
        try:
            with connection:
                connection.executemany(
                    "INSERT INTO presence_facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        (
                            f"sample_{index:07d}",
                            "primary",
                            f"camera_{index % cameras:04d}",
                            (start + timedelta(seconds=index * 86400 / samples)).isoformat(),
                            states[index % len(states)],
                            "synthetic_scanner",
                            None if index % len(states) == 2 else "synthetic_detector",
                            "a" * 64,
                            "worker_not_running" if index % len(states) == 2 else None,
                            start.isoformat(),
                        )
                        for index in range(samples)
                    ),
                )
        finally:
            connection.close()

        gc.collect()
        tracemalloc.start()
        try:
            began = time.perf_counter()
            summary = read_presence_day_summary(database, local_date=day, timezone_name="UTC")
            elapsed = time.perf_counter() - began
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if summary is None or summary.sample_count != samples or len(summary.streams) != cameras:
            raise RuntimeError("synthetic summary counts differ")
        counts = (summary.person_detected_count, summary.clear_count, summary.coverage_gap_count)
        expected = tuple((samples + 2 - index) // 3 for index in range(3))
        if counts != expected:
            raise RuntimeError("synthetic summary states differ")
        return {
            "samples": samples,
            "cameras": cameras,
            "state_counts": counts,
            "python_peak_bytes": peak,
            "elapsed_seconds": round(elapsed, 6),
            "media_read": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--cameras", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.cameras <= args.samples <= 1_000_000:
        parser.error("require 1 <= cameras <= samples <= 1000000")
    print(json.dumps(benchmark(samples=args.samples, cameras=args.cameras), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
