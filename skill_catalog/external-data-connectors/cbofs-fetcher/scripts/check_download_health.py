#!/usr/bin/env python3
"""Create source-aware health JSON for a CBOFS fetch/extraction run."""

from __future__ import annotations

import argparse
import json

try:
    from .cbofs_fetcher import evaluate_health
except ImportError:
    from cbofs_fetcher import evaluate_health


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--compact", action="append", default=[])
    args = parser.parse_args()
    report = evaluate_health(args.request, args.run_dir, args.compact)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "healthy" else 2


if __name__ == "__main__":
    raise SystemExit(main())
