#!/usr/bin/env python3
"""Estimate GSHHG/GSHHS request size and choose local or Kestrel routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gshhs_coastline.sources import (  # noqa: E402
    GSHHG_ZIP_ESTIMATED_BYTES,
    GSHHG_ZIP_URL,
    find_gshhs_cache,
    local_free_bytes,
    write_json,
)


SCRATCH_TEMPLATE = "/scratch/yhuang168/oma_external_data_connectors/{skill_name}/{run_id}"


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="Request JSON with optional bbox/cache/resolution fields.")
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skill-name", default="gshhs-coastline")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    request = _read_json(args.request)
    cache_dir = args.cache_dir or request.get("cache_dir")
    run_id = args.run_id or request.get("name") or request.get("run_id") or "default"
    cache = find_gshhs_cache(cache_dir)
    local_free = local_free_bytes(args.run_dir)

    if cache:
        estimated_bytes = 0
        recommendation = "local"
        reason = "Usable local GSHHS cache was found; no download is required."
        cache_info = {
            "status": "found",
            "gshhs_dir": str(cache.gshhs_dir),
            "source_kind": cache.source_kind,
            "available_resolutions": list(cache.available_resolutions),
        }
    else:
        estimated_bytes = GSHHG_ZIP_ESTIMATED_BYTES
        cache_info = {"status": "missing"}
        if local_free > 4 * estimated_bytes:
            recommendation = "local"
            reason = "Local free space is more than four times the estimated SOEST ZIP size."
        else:
            recommendation = "kestrel"
            reason = "Local free space is not more than four times the estimated SOEST ZIP size."

    result = {
        "schema_version": "external_data_estimate_v1",
        "skill_name": args.skill_name,
        "request_path": str(args.request) if args.request else None,
        "run_dir": str(args.run_dir),
        "source_url": GSHHG_ZIP_URL,
        "cache": cache_info,
        "estimated_requested_bytes": int(estimated_bytes),
        "estimated_requested_mb": round(estimated_bytes / 1024**2, 3),
        "local_free_bytes": local_free,
        "local_free_gb": round(local_free / 1024**3, 3),
        "routing_recommendation": recommendation,
        "routing_reason": reason,
        "kestrel_scratch_path": SCRATCH_TEMPLATE.format(skill_name=args.skill_name, run_id=run_id),
        "download_gate": {
            "rule": "download locally only when local_free_bytes > 4 * estimated_requested_bytes",
            "cache_policy": "local cache means estimated_requested_bytes is zero",
        },
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
