from __future__ import annotations

import unittest
from pathlib import Path


class ReleaseScriptPolicyTests(unittest.TestCase):
    def test_release_verifier_is_bounded_offline_and_cleans_only_its_temp(self) -> None:
        script = Path("scripts/verify_source_release.sh").read_text(encoding="utf-8")

        self.assertIn("mktemp -d -p /tmp baby-monitor-release.", script)
        self.assertIn("/tmp/baby-monitor-release.*)", script)
        self.assertIn("release-audit", script)
        self.assertIn("release-sbom", script)
        self.assertEqual(2, script.count("release-bundle"))
        self.assertIn("cmp --silent", script)
        self.assertIn("install_core_venv.sh", script)
        self.assertIn("unittest discover", script)
        unset_index = script.index("unset GIT_INDEX_FILE")
        self.assertGreater(
            unset_index,
            script.index('cmp --silent "$work_dir/standalone.cdx.json"'),
        )
        self.assertLess(unset_index, script.index('cd "$source_root"'))
        self.assertNotIn("pip install", script)
        self.assertNotIn("curl ", script)
        self.assertNotIn("wget ", script)
        self.assertNotIn("git push", script)
        self.assertNotIn("release create", script)


if __name__ == "__main__":
    unittest.main()
