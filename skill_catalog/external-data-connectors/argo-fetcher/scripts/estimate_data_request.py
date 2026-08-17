#!/usr/bin/env python3
"""Standard estimate hook for bounded native Argo profile requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .argo_fetcher import atomic_write_json, build_download_plan
except ImportError:
    from argo_fetcher import atomic_write_json, build_download_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--index-dir")
    parser.add_argument("--refresh-indexes", action="store_true")
    parser.add_argument("--allow-stale-offline", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    plan = build_download_plan(
        request,
        args.run_dir,
        cache_dir=args.cache_dir,
        index_dir=args.index_dir,
        refresh_indexes=args.refresh_indexes,
        allow_stale_offline=args.allow_stale_offline,
        timeout=args.timeout,
    )
    atomic_write_json(args.output, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 2 if plan["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
