#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from baby_monitor.identity_review import IdentityReviewError, run_review_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve one private identity review on loopback.")
    parser.add_argument("--review-root", required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        run_review_server(args.review_root, port=args.port)
    except IdentityReviewError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
