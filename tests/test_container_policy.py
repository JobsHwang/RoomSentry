from __future__ import annotations

import re
import unittest
from pathlib import Path


def _service_blocks(compose: str) -> dict[str, str]:
    lines = compose.splitlines()
    try:
        start = lines.index("services:") + 1
    except ValueError as exc:
        raise AssertionError("compose.yaml must define services") from exc
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[start:]:
        if line and not line.startswith(" "):
            break
        match = re.fullmatch(r"  ([a-z0-9_-]+):", line)
        if match:
            current = match.group(1)
            blocks[current] = [line]
        elif current is not None:
            blocks[current].append(line)
    return {name: "\n".join(block) for name, block in blocks.items()}


class ContainerPolicyTests(unittest.TestCase):
    def test_compose_exposes_only_a_hardened_profile_gated_cli(self) -> None:
        compose = Path("compose.yaml").read_text(encoding="utf-8")
        services = _service_blocks(compose)

        self.assertEqual({"cli"}, set(services))
        self.assertIn('profiles: ["tools"]', services["cli"])
        self.assertIn('command: ["--help"]', services["cli"])
        self.assertNotIn("worker-placeholder", compose)
        self.assertNotIn("env_file:", compose)
        self.assertNotIn("ports:", compose)
        self.assertNotIn("network_mode: host", compose)
        self.assertNotIn("privileged:", compose)
        self.assertNotIn("/var/run/docker.sock", compose)
        self.assertIn("read_only: true", compose)
        self.assertIn("no-new-privileges:true", compose)
        self.assertIn("cap_drop:\n    - ALL", compose)
        self.assertIn("internal: true", compose)

    def test_image_build_does_not_resolve_python_or_model_dependencies(self) -> None:
        dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
        lowered = dockerfile.casefold()

        self.assertIn("pythonpath=/app/src", lowered)
        self.assertIn("user app", lowered)
        self.assertIn('entrypoint ["python", "-m", "baby_monitor"]', lowered)
        self.assertNotIn("pip install", lowered)
        self.assertNotIn("curl ", lowered)
        self.assertNotIn("wget ", lowered)
        self.assertNotIn("huggingface", lowered)
        self.assertNotIn("minicpm", lowered)
        self.assertNotIn("mage-vl", lowered)
        self.assertNotRegex(dockerfile, r"(?im)^\s*ADD\s")

    def test_build_context_is_an_explicit_source_allowlist(self) -> None:
        dockerignore = Path(".dockerignore").read_text(encoding="utf-8").splitlines()

        self.assertEqual("**", dockerignore[0])
        self.assertEqual(
            {
                "!Dockerfile",
                "!.dockerignore",
                "!pyproject.toml",
                "!README.md",
                "!LICENSE",
                "!src/",
                "!src/**",
            },
            set(dockerignore[1:]),
        )
        self.assertFalse(any(".env" in line for line in dockerignore[1:]))
        self.assertFalse(any("data" in line for line in dockerignore[1:]))


if __name__ == "__main__":
    unittest.main()
