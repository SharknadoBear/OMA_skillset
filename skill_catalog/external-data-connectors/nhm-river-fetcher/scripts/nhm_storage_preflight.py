#!/usr/bin/env python3
"""Check whether a local or df-reported path has enough space for selected NHM files."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "selected"}


def bytes_from_manifest(path: Path, include_unselected: bool = False) -> int:
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    total = 0
    for row in rows:
        if include_unselected or truthy(row.get("selected")):
            total += int(float(row.get("size_bytes") or 0))
    return total


def parse_df_output(text: str) -> int:
    """Parse `df -PB1 <path>` and return available bytes from the last data row."""
    lines = [line.split() for line in text.splitlines() if line.strip()]
    data_rows = [line for line in lines if len(line) >= 4 and line[0].lower() != "filesystem"]
    if not data_rows:
        raise ValueError("no df data row found")
    return int(data_rows[-1][3])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True, help="Local path to check.")
    parser.add_argument("--manifest", type=Path, help="CSV or JSON manifest with size_bytes and selected columns.")
    parser.add_argument("--bytes", type=int, dest="requested_bytes", help="Requested bytes, if no manifest is used.")
    parser.add_argument("--include-unselected", action="store_true", help="Count all manifest rows.")
    parser.add_argument("--multiplier", type=float, default=4.0)
    parser.add_argument("--df-output", type=Path, help="Parse a saved `df -PB1` output instead of local disk usage.")
    parser.add_argument("--out-json", type=Path, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    if args.manifest:
        requested = bytes_from_manifest(args.manifest, args.include_unselected)
    elif args.requested_bytes is not None:
        requested = int(args.requested_bytes)
    else:
        parser.error("provide --manifest or --bytes")

    required = int(requested * args.multiplier)
    if args.df_output:
        available = parse_df_output(args.df_output.read_text(encoding="utf-8"))
        checked_path = str(args.path)
    else:
        args.path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(args.path)
        available = int(usage.free)
        checked_path = str(args.path.resolve())

    passed = available > required
    report = {
        "path": checked_path,
        "requested_bytes": requested,
        "required_bytes": required,
        "available_bytes": available,
        "multiplier": args.multiplier,
        "passed": passed,
        "requested_gb": requested / 1e9,
        "required_gb": required / 1e9,
        "available_gb": available / 1e9,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
