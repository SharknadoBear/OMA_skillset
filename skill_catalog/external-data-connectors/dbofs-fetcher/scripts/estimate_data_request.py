#!/usr/bin/env python3
"""Write an exact-byte NOAA-archive plan for a bounded DBOFS request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from .dbofs_fetcher import plan_request  # type: ignore[attr-defined]
except ImportError:
    from dbofs_fetcher import plan_request  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        report = plan_request(args.request, args.run_dir, output=args.output)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "ok",
        "object_count": report["object_count"],
        "total_bytes": report["total_bytes"],
        "routing_decision": report["routing_decision"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
