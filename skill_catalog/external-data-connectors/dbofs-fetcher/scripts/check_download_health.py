#!/usr/bin/env python3
"""Evaluate transfer, time, C-grid, ROMS-transform, and field health for DBOFS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from .dbofs_fetcher import evaluate_health  # type: ignore[attr-defined]
except ImportError:
    from dbofs_fetcher import evaluate_health  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--plots-dir")
    args = parser.parse_args()
    try:
        report = evaluate_health(
            args.request,
            args.run_dir,
            output=args.output,
            plots_dir=args.plots_dir,
        )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": report["status"],
        "critical_count": len(report["critical_findings"]),
        "warning_count": len(report["warnings"]),
    }, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
