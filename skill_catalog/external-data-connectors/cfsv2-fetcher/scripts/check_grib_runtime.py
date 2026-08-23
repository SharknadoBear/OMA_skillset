#!/usr/bin/env python3
"""Write the machine-readable CFS GRIB runtime preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .cfs_grib_core import runtime_preflight
    from .download_monitor import atomic_write_json
except ImportError:
    from cfs_grib_core import runtime_preflight
    from download_monitor import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = runtime_preflight()
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
