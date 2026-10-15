#!/usr/bin/env python3
"""Verify one reviewed local face model without importing or loading it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baby_monitor.identity_model import FaceModelManifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    verification = FaceModelManifest.read(args.manifest).verify_model(args.model)
    print(json.dumps(verification.to_public_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
