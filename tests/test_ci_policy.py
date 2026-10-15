from __future__ import annotations

import re
import unittest
from pathlib import Path


class CIPolicyTests(unittest.TestCase):
    def test_core_workflow_is_bounded_read_only_and_sha_pinned(self) -> None:
        path = Path(".github/workflows/core.yml")
        workflow = path.read_text(encoding="utf-8")

        self.assertIn("permissions:\n  contents: read\n", workflow)
        self.assertNotIn("pull_request_target", workflow)
        self.assertNotIn("self-hosted", workflow)
        self.assertNotIn("secrets.", workflow)
        self.assertNotRegex(workflow, r"(?m)^\s+[A-Za-z0-9_-]+:\s+write\s*$")
        self.assertIn("runs-on: ubuntu-24.04", workflow)
        self.assertIn("timeout-minutes: 15", workflow)
        self.assertIn('python-version: ["3.11", "3.12"]', workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("ruff==0.16.1", workflow)
        self.assertIn("python -m unittest discover -s tests -t . -v", workflow)
        self.assertIn("python -m baby_monitor release-audit --root .", workflow)
        self.assertIn("python -m baby_monitor release-sbom --root .", workflow)
        self.assertIn("python -m baby_monitor release-bundle --root .", workflow)
        self.assertIn("cmp --silent", workflow)

        uses = re.findall(r"(?m)^\s*uses:\s*([^@\s]+)@([^\s]+)\s*$", workflow)
        self.assertEqual(
            [
                (
                    "actions/checkout",
                    "de0fac2e4500dabe0009e67214ff5f5447ce83dd",
                ),
                (
                    "actions/setup-python",
                    "a309ff8b426b58ec0e2a45f0f869d46889d02405",
                ),
            ],
            uses,
        )
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", revision) for _, revision in uses))

    def test_all_workflows_obey_the_same_action_and_trigger_boundary(self) -> None:
        workflows = sorted(Path(".github/workflows").glob("*.y*ml"))
        self.assertEqual([Path(".github/workflows/core.yml")], workflows)
        for path in workflows:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("pull_request_target", text)
            self.assertNotIn("workflow_run", text)
            self.assertNotIn("schedule:", text)
            for revision in re.findall(r"(?m)^\s*uses:\s*[^@\s]+@([^\s]+)\s*$", text):
                self.assertRegex(revision, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
