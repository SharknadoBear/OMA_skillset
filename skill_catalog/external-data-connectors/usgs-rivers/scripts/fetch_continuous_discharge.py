"""Fetch one regular USGS continuous discharge series with its original qualifiers."""
import argparse
import csv
import hashlib
import json
import math
import tempfile
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


ENDPOINT = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"
CFS_TO_M3S = 0.028316846592


def utc(value):
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.utcoffset() is None:
        raise ValueError("explicit UTC offset required")
    return instant.astimezone(timezone.utc)


def iso(instant):
    return instant.isoformat().replace("+00:00", "Z")


def observations(features, site, parameter, start, end, interval):
    rows, series = [], set()
    for feature in features:
        p = feature["properties"]
        if p["monitoring_location_id"] != site or p["parameter_code"] != parameter:
            raise ValueError("unexpected USGS site or parameter")
        if p["unit_of_measure"] != "ft^3/s":
            raise ValueError(f"unexpected discharge unit: {p['unit_of_measure']}")
        series.add((p["time_series_id"], p["statistic_id"]))
        value = float(p["value"])
        if not math.isfinite(value):
            raise ValueError("nonfinite discharge")
        rows.append((utc(p["time"]), p["value"], value * CFS_TO_M3S,
                     p["approval_status"], p.get("qualifier")))
    rows.sort(key=lambda row: row[0])
    if len(series) != 1 or not rows:
        raise ValueError("expected exactly one nonempty discharge series")
    if rows[0][0] != start or rows[-1][0] != end:
        raise ValueError("source does not cover requested endpoints")
    if any(b[0] - a[0] != interval for a, b in zip(rows, rows[1:])):
        raise ValueError("missing, duplicated, or irregular source timestamp")
    return rows, next(iter(series))


def download(site, parameter, start, end, minutes, output):
    if parameter not in {"72137", "00060"} or not site.startswith("USGS-"):
        raise ValueError("supply a USGS site ID and a discharge parameter")
    if end <= start or minutes <= 0:
        raise ValueError("positive sampling interval and increasing dates required")
    interval = timedelta(minutes=minutes)
    count, remainder = divmod(end - start, interval)
    if remainder or count >= 10000:
        raise ValueError("request must fit one bounded page of at most 10,000 aligned samples")
    query = urllib.parse.urlencode({"f": "json", "monitoring_location_id": site,
        "parameter_code": parameter, "datetime": f"{iso(start)}/{iso(end)}", "limit": 10000})
    url = ENDPOINT + "?" + query
    if output.exists():
        raise FileExistsError(f"select a new source directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        product = Path(tmp) / "series"
        product.mkdir()
        with urllib.request.urlopen(url, timeout=60) as response:
            raw = response.read()
        payload = json.loads(raw)
        if any(link.get("rel") == "next" for link in payload.get("links", [])):
            raise ValueError("response is paginated; narrow the request or select a unique series")
        rows, series = observations(payload["features"], site, parameter, start, end, interval)
        (product / "response.geojson").write_bytes(raw)
        table = product / "discharge.csv"
        with table.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_utc", "discharge_ft3s", "discharge_m3s", "approval_status", "qualifier"])
            for t, value, q, status, qualifier in rows:
                writer.writerow([iso(t), value, q, status, qualifier])
        manifest = {"schema": "usgs_continuous_discharge_v1", "site": site,
            "parameter_code": parameter, "time_series_id": series[0], "statistic_id": series[1],
            "start_utc": iso(start), "end_utc": iso(end), "interval_seconds": minutes * 60,
            "records": len(rows), "units": "m3/s", "original_units": "ft^3/s",
            "conversion_factor": CFS_TO_M3S, "url": url, "retrieved_at": iso(datetime.now(timezone.utc)),
            "approval_counts": dict(Counter(row[3] for row in rows)),
            "raw": {"path": "response.geojson", "sha256": hashlib.sha256(raw).hexdigest()},
            "series": {"path": table.name, "sha256": hashlib.sha256(table.read_bytes()).hexdigest()}}
        (product / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        product.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, help="e.g. USGS-14246900")
    parser.add_argument("--parameter", required=True, choices=["72137", "00060"])
    parser.add_argument("--start", required=True, type=utc)
    parser.add_argument("--end", required=True, type=utc)
    parser.add_argument("--interval-minutes", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(download(args.site, args.parameter, args.start, args.end,
                              args.interval_minutes, args.output_dir), indent=2))
