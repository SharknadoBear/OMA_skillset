#!/usr/bin/env python3
"""Compatibility entry point for the CFSv2 timed estimate gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .cfsv2_fetcher import build_cfsv2_plan
    from .download_monitor import atomic_write_json
except ImportError:
    from cfsv2_fetcher import build_cfsv2_plan
    from download_monitor import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-repeats", type=int, default=2)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    plan = build_cfsv2_plan(request, run_dir=args.run_dir, probe_repeats=args.probe_repeats)
    atomic_write_json(args.output, plan)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
