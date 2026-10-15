from __future__ import annotations

import unittest
from pathlib import Path


class CoreInstallPolicyTests(unittest.TestCase):
    def test_bootstrap_does_not_invoke_pip_or_model_installers(self) -> None:
        bootstrap = Path("scripts/bootstrap_ubuntu.sh").read_text(encoding="utf-8")

        self.assertIn("scripts/install_core_venv.sh", bootstrap)
        self.assertNotIn("pip install", bootstrap)
        self.assertNotIn("huggingface", bootstrap.casefold())
        self.assertNotIn("snapshot_download", bootstrap)

    def test_core_installer_is_offline_and_rejects_unsafe_targets(self) -> None:
        installer = Path("scripts/install_core_venv.sh").read_text(encoding="utf-8")

        self.assertNotIn("pip install", installer)
        self.assertNotIn("curl ", installer)
        self.assertNotIn("wget ", installer)
        self.assertIn("Refusing an unsafe virtual-environment target", installer)
        self.assertIn("Refusing a symlink virtual-environment target", installer)
        self.assertIn("Refusing to reuse a non-venv target", installer)
        self.assertIn("python -m baby_monitor", installer.replace('"$venv_python" -I', "python"))


if __name__ == "__main__":
    unittest.main()
