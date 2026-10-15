from __future__ import annotations

import shlex
import subprocess
import sys
import unittest
from pathlib import Path


class ReferenceCommandTests(unittest.TestCase):
    def test_ci_reference_commands_accept_the_configured_arguments_without_writing(self) -> None:
        root = Path(__file__).resolve().parent.parent
        workflow = (root / ".github/workflows/core.yml").read_text(encoding="utf-8")
        commands = [
            shlex.split(line.strip().removeprefix("run: "))
            for line in workflow.splitlines()
            if line.strip().startswith("run: python scripts/generate_")
        ]
        self.assertEqual(
            {"scripts/generate_cli_reference.py", "scripts/generate_config_reference.py"},
            {command[1] for command in commands},
        )
        self.assertEqual(2, len(commands))
        for command in commands:
            with self.subTest(script=command[1]):
                self.assertIn("--check", command)
                self.assertIn("--output", command)
                output = root / command[command.index("--output") + 1]
                before = output.read_bytes()
                result = subprocess.run(
                    [sys.executable, *command[1:]],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertIn("is current", result.stdout)
                self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
