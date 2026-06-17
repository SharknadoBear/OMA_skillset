#!/usr/bin/env python
"""Build a combined CUDEM, CRM, and ETOPO source index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cudem_bathy.sources import (  # noqa: E402
    build_bathy_source_index,
    save_bathy_source_index,
    summarize_bathy_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output JSON source index path.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--no-cudem", action="store_true", help="Skip CUDEM sources.")
    parser.add_argument("--no-crm", action="store_true", help="Skip NOAA CRM sources.")
    parser.add_argument("--no-etopo", action="store_true", help="Skip ETOPO 2022 sources.")
    args = parser.parse_args()

    index = build_bathy_source_index(
        include_cudem=not args.no_cudem,
        include_crm=not args.no_crm,
        include_etopo=not args.no_etopo,
        timeout=args.timeout,
    )
    path = save_bathy_source_index(index, args.output)
    print(f"Wrote {path}")
    print(json.dumps(summarize_bathy_index(index), indent=2, sort_keys=True))
    if index.get("warnings"):
        print("Warnings:")
        for warning in index["warnings"]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
