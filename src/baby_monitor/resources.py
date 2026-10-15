from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    gpu_used_baseline_mb: float | None
    gpu_used_peak_mb: float | None
    gpu_delta_peak_mb: float | None
    system_used_baseline_mb: float | None
    system_used_peak_mb: float | None
    process_rss_peak_mb: float | None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _gpu_used_mb() -> float | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        wsl_path = "/usr/lib/wsl/lib/nvidia-smi"
        executable = wsl_path if PathLike.is_file(wsl_path) else None
    if executable is None:
        return None
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            creationflags=creation_flags,
        )
        first_line = completed.stdout.strip().splitlines()[0]
        return float(first_line.strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class PathLike:
    @staticmethod
    def is_file(path: str) -> bool:
        try:
            return os.path.isfile(path)
        except OSError:
            return False


def _linux_memory() -> tuple[float | None, float | None]:
    try:
        values: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0])
        total_mb = values["MemTotal"] / 1024
        available_mb = values["MemAvailable"] / 1024
        with open("/proc/self/status", encoding="ascii") as handle:
            rss_line = next(line for line in handle if line.startswith("VmRSS:"))
        rss_mb = int(rss_line.split()[1]) / 1024
        return total_mb - available_mb, rss_mb
    except (OSError, KeyError, StopIteration, ValueError):
        return None, None


def _windows_memory() -> tuple[float | None, float | None]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("page_fault_count", ctypes.c_ulong),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None, None
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None, None
        used_mb = (status.total_physical - status.available_physical) / (1024 * 1024)
        rss_mb = counters.working_set_size / (1024 * 1024)
        return used_mb, rss_mb
    except (AttributeError, OSError):
        return None, None


def _memory() -> tuple[float | None, float | None]:
    return _windows_memory() if os.name == "nt" else _linux_memory()


class ResourceSampler:
    def __init__(self, poll_interval_seconds: float = 0.25):
        self.poll_interval_seconds = poll_interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_baseline: float | None = None
        self._system_baseline: float | None = None
        self._gpu_peak: float | None = None
        self._system_peak: float | None = None
        self._rss_peak: float | None = None

    def _sample(self) -> None:
        gpu = _gpu_used_mb()
        system, rss = _memory()
        if self._gpu_baseline is None:
            self._gpu_baseline = gpu
        if self._system_baseline is None:
            self._system_baseline = system
        if gpu is not None:
            self._gpu_peak = gpu if self._gpu_peak is None else max(self._gpu_peak, gpu)
        if system is not None:
            self._system_peak = (
                system if self._system_peak is None else max(self._system_peak, system)
            )
        if rss is not None:
            self._rss_peak = rss if self._rss_peak is None else max(self._rss_peak, rss)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_interval_seconds):
            self._sample()

    def __enter__(self) -> ResourceSampler:
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="resource-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def result(self) -> ResourceUsage:
        delta = None
        if self._gpu_baseline is not None and self._gpu_peak is not None:
            delta = max(0.0, self._gpu_peak - self._gpu_baseline)
        return ResourceUsage(
            gpu_used_baseline_mb=self._gpu_baseline,
            gpu_used_peak_mb=self._gpu_peak,
            gpu_delta_peak_mb=delta,
            system_used_baseline_mb=self._system_baseline,
            system_used_peak_mb=self._system_peak,
            process_rss_peak_mb=self._rss_peak,
        )
