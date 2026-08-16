#!/usr/bin/env python3
"""Compatibility entry point for CFSv2 subset health validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .cfsv2_fetcher import health_cfsv2
    from .download_monitor import atomic_write_json
except ImportError:
    from cfsv2_fetcher import health_cfsv2
    from download_monitor import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Fetched NetCDF; otherwise discover one in --run-dir")
    parser.add_argument("--request")
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--plots-dir", help="Accepted for connector-hook compatibility")
    args = parser.parse_args()
    path = Path(args.input) if args.input else next(
        (item for item in Path(args.run_dir).glob("*.nc") if item.is_file()), None
    )
    if path is None:
        raise FileNotFoundError("No CFSv2 NetCDF supplied or found in the run directory")
    request = json.loads(Path(args.request).read_text(encoding="utf-8")) if args.request else None
    health = health_cfsv2(path, request)
    atomic_write_json(args.output, health)
    print(json.dumps(health, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
