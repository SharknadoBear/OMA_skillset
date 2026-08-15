#!/usr/bin/env python3
"""Estimate TPXO9v5 staging requirements before an authenticated download."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_fields(value: str) -> list[str]:
    fields = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(fields) - {"elevation", "transport"})
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported field(s): {', '.join(invalid)}")
    return fields


def infer_role(item: dict[str, Any]) -> str | None:
    explicit = str(item.get("role", "")).strip().lower()
    if explicit in {"grid", "elevation", "transport"}:
        return explicit
    name = str(item.get("name") or item.get("title") or item.get("basename") or "").lower()
    if "grid" in name:
        return "grid"
    if name.startswith("h_") or "elevation" in name:
        return "elevation"
    if name.startswith("u_") or "transport" in name or "current" in name:
        return "transport"
    return None


def manifest_files(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("files", "sources", "items"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
    raise ValueError("Manifest must be a list or contain a files/sources/items list.")


def file_size(item: dict[str, Any]) -> int:
    for key in ("size_bytes", "size", "bytes"):
        if key in item and item[key] not in (None, ""):
            value = int(item[key])
            if value < 0:
                raise ValueError("File sizes must be non-negative.")
            return value
    raise ValueError(f"No byte size found for {item.get('name') or item.get('title') or 'manifest item'}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Size-only JSON manifest from Drive metadata or local files.")
    parser.add_argument("--fields", default="elevation", type=parse_fields, help="Comma-separated elevation and/or transport.")
    parser.add_argument("--run-dir", required=True, help="Planned local staging/output directory.")
    parser.add_argument("--output", required=True, help="Estimate JSON path.")
    args = parser.parse_args()

    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    required_roles = {"grid", *args.fields}
    selected: dict[str, dict[str, Any]] = {}
    for item in manifest_files(payload):
        role = infer_role(item)
        if role in required_roles:
            if role in selected:
                raise ValueError(f"Manifest contains multiple candidates for {role}.")
            selected[role] = item
    missing = sorted(required_roles - selected.keys())
    if missing:
        raise ValueError(f"Manifest is missing required role(s): {', '.join(missing)}")
    required_bytes = sum(file_size(item) for item in selected.values())
    run_dir = Path(args.run_dir).expanduser().resolve()
    probe = run_dir if run_dir.exists() else run_dir.parent
    probe.mkdir(parents=True, exist_ok=True)
    free_bytes = int(shutil.disk_usage(probe).free)
    required_working_bytes = 4 * required_bytes
    status = "pass" if free_bytes > required_working_bytes else "fail"
    result = {
        "schema_version": "tpxo9v5_download_estimate_v1",
        "status": status,
        "requested_fields": args.fields,
        "required_roles": sorted(required_roles),
        "selected_files": [
            {
                "role": role,
                "basename": str(item.get("name") or item.get("title") or item.get("basename") or ""),
                "size_bytes": file_size(item),
            }
            for role, item in sorted(selected.items())
        ],
        "required_raw_bytes": required_bytes,
        "estimated_staging_bytes": required_bytes,
        "estimated_output_upper_bound_bytes": required_bytes,
        "required_working_bytes": required_working_bytes,
        "local_free_bytes": free_bytes,
        "rule": "local_free_bytes > 4 * required_raw_bytes",
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
