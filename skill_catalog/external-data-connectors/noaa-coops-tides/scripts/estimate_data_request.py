#!/usr/bin/env python3
"""Estimate requested data volume and choose local or Kestrel routing.

This script is intentionally generic. It reads a request/manifest JSON and
looks for explicit size fields first. If no reliable size is present, it blocks
download execution by returning review_required instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


SIZE_KEYS_BYTES = ("estimated_bytes", "requested_bytes", "download_bytes", "size_bytes", "bytes")
SIZE_KEYS_MB = ("estimated_mb", "requested_mb", "download_mb", "size_mb")
SIZE_KEYS_GB = ("estimated_gb", "requested_gb", "download_gb", "size_gb")
SCRATCH_TEMPLATE = "/scratch/yhuang168/oma_external_data_connectors/{skill_name}/{run_id}"


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _explicit_size_bytes(obj: Any) -> float | None:
    if isinstance(obj, dict):
        total = 0.0
        found = False
        for key in SIZE_KEYS_BYTES:
            value = _number(obj.get(key))
            if value is not None:
                total += value
                found = True
        for key in SIZE_KEYS_MB:
            value = _number(obj.get(key))
            if value is not None:
                total += value * 1024**2
                found = True
        for key in SIZE_KEYS_GB:
            value = _number(obj.get(key))
            if value is not None:
                total += value * 1024**3
                found = True
        for list_key in ("sources", "tiles", "files", "chunks", "requests"):
            if isinstance(obj.get(list_key), list):
                for item in obj[list_key]:
                    item_size = _explicit_size_bytes(item)
                    if item_size is not None:
                        total += item_size
                        found = True
        return total if found else None
    if isinstance(obj, list):
        total = 0.0
        found = False
        for item in obj:
            item_size = _explicit_size_bytes(item)
            if item_size is not None:
                total += item_size
                found = True
        return total if found else None
    return None


def _local_free_bytes(path: Path) -> int:
    target = path if path.exists() else path.parent
    target.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(target).free)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="Request or manifest JSON with bbox/time/source/size fields.")
    parser.add_argument("--run-dir", default=".", help="Local run/cache directory used for free-space checks.")
    parser.add_argument("--output", required=True, help="Estimate JSON to write.")
    parser.add_argument("--skill-name", default=None, help="Override detected skill name.")
    parser.add_argument("--run-id", default=None, help="Run identifier for Kestrel scratch planning.")
    args = parser.parse_args()

    request = _read_json(args.request)
    run_dir = Path(args.run_dir)
    skill_name = args.skill_name or Path(__file__).resolve().parents[1].name
    run_id = args.run_id or request.get("name") or request.get("run_id") or "default"
    estimated_bytes = _explicit_size_bytes(request)
    local_free = _local_free_bytes(run_dir)

    if estimated_bytes is None:
        recommendation = "review_required"
        reason = "No explicit size estimate was found in the request manifest."
    elif local_free > 4 * estimated_bytes:
        recommendation = "local"
        reason = "Local free space is more than four times the estimated request size."
    else:
        recommendation = "kestrel"
        reason = "Local free space is not more than four times the estimated request size."

    result = {
        "schema_version": "external_data_estimate_v1",
        "skill_name": skill_name,
        "request_path": str(args.request) if args.request else None,
        "run_dir": str(run_dir),
        "estimated_requested_bytes": int(estimated_bytes) if estimated_bytes is not None else None,
        "estimated_requested_mb": round(estimated_bytes / 1024**2, 3) if estimated_bytes is not None else None,
        "local_free_bytes": local_free,
        "local_free_gb": round(local_free / 1024**3, 3),
        "routing_recommendation": recommendation,
        "routing_reason": reason,
        "kestrel_scratch_path": SCRATCH_TEMPLATE.format(skill_name=skill_name, run_id=run_id),
        "download_gate": {
            "rule": "download locally only when local_free_bytes > 4 * estimated_requested_bytes",
            "unknown_estimate_policy": "review_required",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
