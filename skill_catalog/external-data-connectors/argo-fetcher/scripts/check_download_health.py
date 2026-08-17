#!/usr/bin/env python3
"""Standard health hook for a native Argo profile run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .argo_fetcher import health_check
except ImportError:
    from argo_fetcher import health_check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", help="Download plan; defaults to <run-dir>/download_plan.json")
    parser.add_argument("--request", help="Accepted for connector-hook compatibility; the plan remains authoritative")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plots-dir")
    args = parser.parse_args()
    plan_path = Path(args.plan) if args.plan else Path(args.run_dir) / "download_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError("An immutable Argo download plan is required for health validation")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    result = health_check(plan, args.run_dir, output=args.output, plots_dir=args.plots_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
