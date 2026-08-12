#!/usr/bin/env python3
"""Compatibility entry point for exact CBOFS public-AWS estimates."""

from __future__ import annotations

import argparse
import json

try:
    from .cbofs_fetcher import plan_request
except ImportError:
    from cbofs_fetcher import plan_request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = plan_request(args.request, args.run_dir, output=args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
