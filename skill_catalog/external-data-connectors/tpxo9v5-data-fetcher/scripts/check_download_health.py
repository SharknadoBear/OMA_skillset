#!/usr/bin/env python3
"""Validate a model-neutral TPXO9v5 harmonic NetCDF product."""

from __future__ import annotations

import argparse
import json

from tpxo9v5.outputs import validate_product, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Generated TPXO harmonic NetCDF.")
    parser.add_argument("--output", required=True, help="Health report JSON path.")
    args = parser.parse_args()
    result = validate_product(args.input)
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
