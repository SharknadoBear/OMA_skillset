from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

try:
    from .noaa_tides import fetch_noaa_waterlevel
except ImportError:
    from noaa_tides import fetch_noaa_waterlevel


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch paired NOAA observed and predicted water levels for validation")
    p.add_argument("--station", action="append", required=True)
    p.add_argument("--period-start", required=True); p.add_argument("--period-end", required=True)
    p.add_argument("--cache-dir", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--manifest", required=True)
    args = p.parse_args()
    cache = Path(args.cache_dir); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    products = []
    for station in args.station:
        observed = fetch_noaa_waterlevel(station, args.period_start, args.period_end, cache / station, product="water_level", datum="MSL", time_zone="GMT")
        predicted = fetch_noaa_waterlevel(station, args.period_start, args.period_end, cache / station, product="predictions", datum="MSL", time_zone="GMT")
        a = observed[["time", "water_level"]].rename(columns={"water_level": "observed"})
        b = predicted[["time", "water_level"]].rename(columns={"water_level": "predicted"})
        merged = a.merge(b, on="time", how="outer").sort_values("time").drop_duplicates("time")
        path = output / f"{station}_noaa_waterlevel.csv"
        merged.to_csv(path, index=False)
        products.append({"station": station, "path": str(path), "sha256": sha256(path), "rows": len(merged), "observed_finite": int(merged["observed"].notna().sum()), "predicted_finite": int(merged["predicted"].notna().sum()), "datum": "MSL", "time_zone": "GMT"})
    manifest = {"schema": "noaa_coops_waterlevel_validation_bundle_v1", "period_start": args.period_start, "period_end": args.period_end, "products": products}
    manifest_path = Path(args.manifest); manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
