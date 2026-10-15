from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from baby_monitor.gpu_guard import probe_gpu_capacity


class GPUGuardTests(unittest.TestCase):
    def test_idle_gpu_with_enough_memory_is_allowed(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, 12282, 1380, 10902, 7\n",
            stderr="",
        )
        with (
            patch(
                "baby_monitor.gpu_guard.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            patch("baby_monitor.gpu_guard.subprocess.run", return_value=completed),
        ):
            result = probe_gpu_capacity(
                device="0",
                min_free_vram_mib=6_000,
                max_utilization_percent=20,
                timeout_seconds=5,
            )
        self.assertTrue(result.allowed)
        self.assertEqual("gpu_capacity_available", result.reason)
        self.assertEqual(10_902, result.memory_free_mib)

    def test_busy_or_memory_constrained_gpu_is_deferred(self) -> None:
        cases = [
            ("0, 12282, 7000, 5282, 5\n", "insufficient_free_vram"),
            ("0, 12282, 2000, 10282, 80\n", "gpu_busy"),
        ]
        for stdout, reason in cases:
            with self.subTest(reason=reason):
                completed = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=stdout, stderr=""
                )
                with (
                    patch(
                        "baby_monitor.gpu_guard.shutil.which",
                        return_value="/usr/bin/nvidia-smi",
                    ),
                    patch("baby_monitor.gpu_guard.subprocess.run", return_value=completed),
                ):
                    result = probe_gpu_capacity(
                        device="0",
                        min_free_vram_mib=6_000,
                        max_utilization_percent=20,
                        timeout_seconds=5,
                    )
                self.assertFalse(result.allowed)
                self.assertEqual(reason, result.reason)

    def test_missing_or_malformed_probe_fails_closed(self) -> None:
        with patch("baby_monitor.gpu_guard.shutil.which", return_value=None):
            missing = probe_gpu_capacity(
                device="0",
                min_free_vram_mib=6_000,
                max_utilization_percent=20,
                timeout_seconds=5,
            )
        self.assertFalse(missing.allowed)
        self.assertEqual("probe_unavailable", missing.reason)

        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not private but malformed", stderr=""
        )
        with (
            patch("baby_monitor.gpu_guard.shutil.which", return_value="nvidia-smi"),
            patch("baby_monitor.gpu_guard.subprocess.run", return_value=completed),
        ):
            malformed = probe_gpu_capacity(
                device="0",
                min_free_vram_mib=6_000,
                max_utilization_percent=20,
                timeout_seconds=5,
            )
        self.assertFalse(malformed.allowed)
        self.assertEqual("probe_unavailable", malformed.reason)


if __name__ == "__main__":
    unittest.main()
