#!/usr/bin/env python3
"""Inventory TPXO9v5 NetCDF roles, variables, constituents, and grid spans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tpxo9v5.io import inventory_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", help="Directory containing TPXO NetCDF files.")
    parser.add_argument("--file", action="append", default=[], help="Explicit NetCDF path; may be repeated.")
    parser.add_argument("--output", required=True, help="Inventory JSON path.")
    args = parser.parse_args()
    paths = [Path(item).expanduser().resolve() for item in args.file]
    if args.source_dir:
        paths.extend(sorted(Path(args.source_dir).expanduser().resolve().glob("*.nc")))
    paths = list(dict.fromkeys(paths))
    if not paths:
        parser.error("Provide --source-dir or at least one --file.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing)}")
    result = inventory_sources(paths)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
