from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests


MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
DATA_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seven_day_chunks(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    a = pd.Timestamp(start)
    b = pd.Timestamp(end)
    if a.tzinfo is None or b.tzinfo is None:
        raise ValueError("period timestamps must include UTC timezone")
    a, b = a.tz_convert("UTC"), b.tz_convert("UTC")
    if b <= a:
        raise ValueError("period_end must be after period_start")
    chunks = []
    cur = a
    while cur < b:
        nxt = min(cur + pd.Timedelta(days=7) - pd.Timedelta(minutes=1), b - pd.Timedelta(minutes=1))
        chunks.append((cur, nxt))
        cur = nxt + pd.Timedelta(minutes=1)
    return chunks


def api_get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload or "errorMsg" in payload:
        raise RuntimeError(str(payload.get("error") or payload.get("errorMsg")))
    return payload


def parse_mesh(path: Path) -> tuple[dict[int, tuple[float, float]], list[tuple[int, int, int]]]:
    nodes, tris = {}, []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = raw.split()
        if p and p[0] == "ND" and len(p) >= 5:
            nodes[int(p[1])] = (float(p[2]), float(p[3]))
        elif p and p[0] == "E3T" and len(p) >= 5:
            tris.append((int(p[2]), int(p[3]), int(p[4])))
    if not nodes or not tris:
        raise ValueError("mesh has no nodes/triangles")
    return nodes, tris


def point_in_mesh(x: float, y: float, nodes: dict[int, tuple[float, float]], tris: list[tuple[int, int, int]]) -> bool:
    for ids in tris:
        a, b, c = (nodes[i] for i in ids)
        if x < min(a[0], b[0], c[0]) or x > max(a[0], b[0], c[0]) or y < min(a[1], b[1], c[1]) or y > max(a[1], b[1], c[1]):
            continue
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if den == 0:
            continue
        u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / den
        v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / den
        w = 1 - u - v
        if min(u, v, w) >= -1e-10:
            return True
    return False


def metadata_stations(kind: str) -> list[dict[str, Any]]:
    return api_get_json(f"{MDAPI}/stations.json", {"type": kind, "units": "metric"}).get("stations", [])


def current_metadata(station: str) -> dict[str, Any]:
    deployments = api_get_json(f"{MDAPI}/stations/{station}/deployments.json", {"units": "metric"})
    bins = api_get_json(f"{MDAPI}/stations/{station}/bins.json", {"units": "metric"})
    return {"deployments": deployments, "bins": bins.get("bins", []), "units": bins.get("units")}


def current_period_coverage(metadata: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    """Screen documented deployment dates against the requested half-open period."""
    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    if begin.tzinfo is None or finish.tzinfo is None or finish <= begin:
        raise ValueError("current period must have explicit timezones and increasing endpoints")
    begin, finish = begin.tz_convert("UTC"), finish.tz_convert("UTC")
    def stamp(value):
        if value is None or str(value).strip().lower() in ("", "none", "null", "nan"):
            return None
        try:
            value = pd.Timestamp(value)
            if pd.isna(value):
                return None
            return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        except (TypeError, ValueError):
            return None
    payload = metadata.get("deployments", metadata)
    if not isinstance(payload, dict):
        return {"status": "unknown", "overlap": None, "intervals": []}
    first, last = stamp(payload.get("first_good_data")), stamp(payload.get("last_good_data"))
    intervals, uncertain = [], False
    for row in payload.get("deployments", []):
        lower, upper = stamp(row.get("deployed")), stamp(row.get("retrieved"))
        if lower is None:
            uncertain = True
            continue
        if upper is None:
            upper = last
        if upper is None or upper < lower:
            uncertain = True
            continue
        intervals.append((lower, upper))
    if not intervals and first is not None and last is not None and last >= first:
        intervals = [(first, last)]
        uncertain = False
    overlap = any(lower < finish and upper >= begin for lower, upper in intervals)
    status = "overlap" if overlap else "unknown" if uncertain or not intervals else "no_overlap"
    return {"status": status, "overlap": True if overlap else None if status == "unknown" else False,
            "period_start": begin.isoformat(), "period_end": finish.isoformat(),
            "intervals": [{"start": lower.isoformat(), "end": upper.isoformat()} for lower, upper in intervals],
            "method": "deployment intervals, falling back to documented first/last good data; metadata dates interpreted in UTC"}


def discover(mesh: Path, mesh_crs: str, start: str, end: str,
             water_level_mapping_policy: str = "strict_inside") -> dict[str, Any]:
    from pyproj import Transformer
    if water_level_mapping_policy not in {"strict_inside", "containing_cell_or_nearest_wet_cell"}:
        raise ValueError("Unsupported water-level mapping policy")
    nodes, tris = parse_mesh(mesh)
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    mesh_bounds = (min(xs), min(ys), max(xs), max(ys))
    tx = Transformer.from_crs("EPSG:4326", mesh_crs, always_xy=True)
    inverse = Transformer.from_crs(mesh_crs, "EPSG:4326", always_xy=True)
    corners = [inverse.transform(x, y) for x in (mesh_bounds[0], mesh_bounds[2]) for y in (mesh_bounds[1], mesh_bounds[3])]
    geographic_bounds = (min(p[0] for p in corners), min(p[1] for p in corners), max(p[0] for p in corners), max(p[1] for p in corners))
    inventory = []
    seen = set()
    for kind, role in (("waterlevels", "water_level"), ("currents", "current"), ("historiccurrents", "current")):
        for station in metadata_stations(kind):
            sid = str(station.get("id"))
            key = (sid, role)
            if key in seen:
                continue
            seen.add(key)
            try:
                lon, lat = float(station["lng"]), float(station["lat"])
                x, y = tx.transform(lon, lat)
            except (KeyError, TypeError, ValueError):
                continue
            if not (geographic_bounds[0] <= lon <= geographic_bounds[2] and geographic_bounds[1] <= lat <= geographic_bounds[3]):
                continue
            inside = point_in_mesh(x, y, nodes, tris)
            item = {"id": sid, "name": station.get("name"), "role": role, "latitude": lat, "longitude": lon, "inside_geographic_envelope": True, "inside_wet_mesh": inside, "eligible": inside, "exclusion_reason": None if inside else "outside_actual_wet_mesh"}
            if role == "water_level" and water_level_mapping_policy != "strict_inside":
                item.update(eligible=True, exclusion_reason=None,
                            mapping_required="containing_cell" if inside else "nearest_wet_cell_proxy",
                            source_data_eligibility="pending_period_scalar_data_checks")
            if role == "current":
                try:
                    meta = current_metadata(sid)
                    orientation = str(meta["deployments"].get("orientation", "")).lower()
                    item.update({"orientation": orientation, "water_depth_m": meta["deployments"].get("measured_depth") or meta["deployments"].get("depth"), "sensor_depth_m": meta["deployments"].get("sensor_depth"), "bins": meta["bins"], "metadata_units": meta["units"]})
                    coverage = current_period_coverage(meta, start, end)
                    item["period_coverage"] = coverage
                    if orientation != "down":
                        item["eligible"] = False
                        item["exclusion_reason"] = f"orientation_{orientation or 'unknown'}_not_depth_integrated"
                    elif inside and coverage["status"] != "overlap":
                        item["eligible"] = False
                        item["exclusion_reason"] = ("no_deployment_overlap_requested_period" if coverage["status"] == "no_overlap"
                                                    else "current_period_metadata_unknown_requires_data_probe")
                except Exception as exc:
                    item["eligible"] = False
                    item["exclusion_reason"] = f"metadata_error:{type(exc).__name__}:{exc}"
            inventory.append(item)
    counts = {
        "water_level_envelope": sum(x["role"] == "water_level" for x in inventory),
        "water_level_strict_wet": sum(x["role"] == "water_level" and x["inside_wet_mesh"] for x in inventory),
        "water_level_eligible": sum(x["role"] == "water_level" and x["eligible"] for x in inventory),
        "current_envelope": sum(x["role"] == "current" for x in inventory),
        "current_downward_strict_wet": sum(x["role"] == "current" and x["eligible"] for x in inventory),
    }
    return {"schema": "noaa_coops_validation_station_inventory_v1", "generated_at": now(), "mesh": str(mesh), "mesh_sha256": sha256(mesh), "mesh_crs": mesh_crs, "geographic_envelope": geographic_bounds, "period_start": start, "period_end": end, "selection_policy": "discover within regional mesh envelope; water mapping policy explicit; current eligibility requires wet containment, downward profiles and period overlap", "water_level_mapping_policy": water_level_mapping_policy, "counts": counts, "stations": sorted(inventory, key=lambda x: (x["role"], x["id"]))}


def speed_direction_to_uv(speed_cm_s: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = speed_cm_s.astype(float) / 100.0
    theta = np.deg2rad(direction_deg.astype(float))
    return speed * np.sin(theta), speed * np.cos(theta)


def depth_average(group: pd.DataFrame, bins: dict[int, float], water_depth: float) -> dict[str, Any]:
    rows = []
    for _, row in group.iterrows():
        bid = int(row["bin"])
        z = bins.get(bid)
        if z is not None and 0 <= z <= water_depth and np.isfinite(row["east_m_s"]) and np.isfinite(row["north_m_s"]):
            rows.append((z, float(row["east_m_s"]), float(row["north_m_s"]), bid))
    rows.sort()
    if not rows:
        return {"east_m_s": math.nan, "north_m_s": math.nan, "valid_bins": 0, "weight_sum_m": 0.0, "used_bins": ""}
    z = np.asarray([r[0] for r in rows])
    edges = np.empty(len(z) + 1)
    edges[0], edges[-1] = 0.0, water_depth
    if len(z) > 1:
        edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    thickness = np.diff(edges)
    if np.any(thickness < 0):
        raise ValueError("bin depths are not compatible with water depth")
    east = float(np.average([r[1] for r in rows], weights=thickness))
    north = float(np.average([r[2] for r in rows], weights=thickness))
    return {"east_m_s": east, "north_m_s": north, "valid_bins": len(rows), "weight_sum_m": float(thickness.sum()), "used_bins": ",".join(str(r[3]) for r in rows), "surface_extrapolation": "shallowest_valid_bin_to_surface", "bed_extrapolation": "deepest_valid_bin_to_bed"}


def fetch_currents(station: str, start: str, end: str, cache_dir: Path, output: Path, manifest_path: Path,
                   getter: Callable[..., dict[str, Any]] = api_get_json) -> dict[str, Any]:
    meta = current_metadata(station)
    dep = meta["deployments"]
    orientation = str(dep.get("orientation", "")).lower()
    if orientation != "down":
        raise ValueError(f"station {station} orientation is {orientation!r}, not downward-looking")
    water_depth = float(dep.get("measured_depth") or dep.get("depth"))
    metadata_bins = {int(b["num"]): float(b["depth"]) for b in meta["bins"] if b.get("depth") is not None and str(b.get("qc_flag", "1")) in {"", "1"}}
    valid_bins = {bid: depth for bid, depth in metadata_bins.items() if 0 <= depth <= water_depth}
    excluded_bins = sorted(set(metadata_bins) - set(valid_bins))
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames, chunks_meta = [], []
    for a, b in seven_day_chunks(start, end):
        label = f"{a.strftime('%Y%m%dT%H%M')}_{b.strftime('%Y%m%dT%H%M')}"
        cache = cache_dir / f"{station}_{label}.csv"
        if cache.exists():
            frame = pd.read_csv(cache)
            source = "cache"
        else:
            params = {"begin_date": a.strftime("%Y%m%d %H:%M"), "end_date": b.strftime("%Y%m%d %H:%M"), "station": station, "product": "currents", "time_zone": "gmt", "units": "metric", "application": "FVCOM_Simulation", "format": "json", "bin": 0}
            payload = getter(DATA_API, params=params, timeout=90)
            records = payload.get("data", [])
            frame = pd.DataFrame({"time": [r.get("t") for r in records], "speed_cm_s": [pd.to_numeric(r.get("s"), errors="coerce") for r in records], "direction_deg": [pd.to_numeric(r.get("d"), errors="coerce") for r in records], "bin": [pd.to_numeric(r.get("b"), errors="coerce") for r in records]})
            frame.to_csv(cache, index=False)
            source = "api"
            time.sleep(0.2)
        frames.append(frame)
        chunks_meta.append({"start": a.isoformat(), "end": b.isoformat(), "cache": str(cache), "source": source, "rows": len(frame), "sha256": sha256(cache)})
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "speed_cm_s", "direction_deg", "bin"])
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    raw["bin"] = pd.to_numeric(raw["bin"], errors="coerce")
    raw = raw.dropna(subset=["time", "speed_cm_s", "direction_deg", "bin"])
    raw = raw[(raw["speed_cm_s"] >= 0) & raw["direction_deg"].between(0, 360) & raw["bin"].astype(int).isin(valid_bins)]
    raw = raw.sort_values(["time", "bin"]).drop_duplicates(["time", "bin"], keep="last")
    raw["east_m_s"], raw["north_m_s"] = speed_direction_to_uv(raw["speed_cm_s"].to_numpy(), raw["direction_deg"].to_numpy())
    rows = []
    for timestamp, group in raw.groupby("time"):
        item = depth_average(group, valid_bins, water_depth)
        item["time"] = timestamp.isoformat()
        rows.append(item)
    depth_mean = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    depth_mean.to_csv(output, index=False)
    raw_path = output.with_name(output.stem + "_all_bins.csv")
    raw.to_csv(raw_path, index=False)
    manifest = {"schema": "noaa_coops_current_profile_v1", "generated_at": now(), "station": station, "period_start": start, "period_end": end, "orientation": orientation, "water_depth_m": water_depth, "metadata_bin_depths_m": metadata_bins, "used_bin_depths_m": valid_bins, "excluded_bins_outside_water_column": excluded_bins, "api_speed_units": "cm/s", "output_velocity_units": "m/s", "direction_convention": "flow-toward degrees clockwise from true north", "vertical_weighting": "midpoint layer interfaces; shallowest to surface and deepest to bed", "chunks": chunks_meta, "all_bins_csv": str(raw_path), "all_bins_sha256": sha256(raw_path), "depth_mean_csv": str(output), "depth_mean_sha256": sha256(output), "row_count": len(depth_mean), "warnings": [f"excluded metadata bins outside 0..water_depth: {excluded_bins}"] if excluded_bins else []}
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description="NOAA CO-OPS current profile discovery and download")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("discover")
    q.add_argument("--mesh", required=True); q.add_argument("--mesh-crs", required=True)
    q.add_argument("--water-level-mapping-policy", choices=["strict_inside", "containing_cell_or_nearest_wet_cell"], default="strict_inside")
    q.add_argument("--period-start", required=True); q.add_argument("--period-end", required=True); q.add_argument("--output", required=True)
    q = sub.add_parser("fetch")
    q.add_argument("--station", required=True); q.add_argument("--period-start", required=True); q.add_argument("--period-end", required=True)
    q.add_argument("--cache-dir", required=True); q.add_argument("--output", required=True); q.add_argument("--manifest", required=True)
    args = p.parse_args()
    if args.command == "discover":
        result = discover(Path(args.mesh), args.mesh_crs, args.period_start, args.period_end, args.water_level_mapping_policy); write_json(Path(args.output), result)
    else:
        result = fetch_currents(args.station, args.period_start, args.period_end, Path(args.cache_dir), Path(args.output), Path(args.manifest))
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
