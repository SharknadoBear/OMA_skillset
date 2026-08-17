#!/usr/bin/env python3
"""Download and health-check exactly one current core, B, and S profile."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import argo_fetcher as af


def _iso(compact: str) -> str:
    return datetime.strptime(compact, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def singleton_request(frame, product: str) -> dict:
    """Choose a recent row whose time/WMO/DAC/mode tuple is unique."""
    keys = ["date_compact", "wmo", "dac", "file_mode"]
    counts = frame.groupby(keys, dropna=False).size().rename("tuple_count")
    merged = frame.join(counts, on=keys)
    candidates = merged.loc[merged["tuple_count"].eq(1)].sort_values("date_compact", ascending=False)
    if candidates.empty:
        raise af.ArgoError(f"No unique bounded smoke-test row found for {product}")
    row = candidates.iloc[0]
    instant = _iso(str(row["date_compact"]))
    request = {
        "schema": af.REQUEST_SCHEMA,
        "products": [product],
        "start": instant,
        "end": instant,
        "global": True,
        "wmos": [str(row["wmo"])],
        "dacs": [str(row["dac"])],
        "file_modes": [str(row["file_mode"])],
    }
    if product in {"bio", "synthetic"}:
        available = [value for value in str(row.get("parameters", "")).split() if value not in {"PRES", "TEMP", "PSAL"}]
        if not available:
            raise af.ArgoError(f"Selected {product} row has no BGC parameter")
        request["parameters"] = [available[0]]
        request["parameter_match"] = "all"
    return request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--refresh-indexes", action="store_true")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    root = Path(args.run_dir)
    cache = root / "cache"
    result = {"schema": "argo_live_smoke_v1", "created_utc": af.utc_now(), "products": []}
    for product in ("core", "bio", "synthetic"):
        index_path, index_meta = af.ensure_index(product, cache, refresh=args.refresh_indexes, timeout=args.timeout)
        request = singleton_request(af.load_index(index_path, product), product)
        product_run = root / product
        plan = af.build_download_plan(request, product_run, cache_dir=cache, timeout=args.timeout)
        af.atomic_write_json(product_run / "download_plan.json", plan)
        if plan["selection_count"] != 1 or plan["blocked"]:
            raise af.ArgoError(f"{product} smoke plan is not one executable file: count={plan['selection_count']} blocked={plan['blocked']}")
        manifest = af.fetch_plan(plan, product_run, cache_dir=cache, timeout=args.timeout, workers=1)
        health = af.health_check(plan, product_run)
        if health["status"] != "pass" or manifest[0]["status"] == "failed":
            raise af.ArgoError(f"{product} live smoke failed health or transfer")
        result["products"].append(
            {
                "product": product,
                "file": plan["selected_rows"][0]["file"],
                "bytes": manifest[0]["bytes"],
                "sha256": manifest[0]["sha256"],
                "mirror": manifest[0]["mirror"],
                "index_sha256": index_meta["sha256"],
                "health": health["status"],
                "plots": health["plots"],
            }
        )
    result["status"] = "pass"
    af.atomic_write_json(root / "live_smoke_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
