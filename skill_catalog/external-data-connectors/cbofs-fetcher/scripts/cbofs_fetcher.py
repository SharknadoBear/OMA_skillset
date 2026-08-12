#!/usr/bin/env python3
"""Plan, fetch, inspect, and extract NOAA CBOFS ROMS data from public AWS."""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from . import roms_aws_core as core
    from . import roms_processing as roms
except ImportError:
    import roms_aws_core as core
    import roms_processing as roms

CONFIG = core.ModelConfig(
    model="cbofs",
    schema_version="cbofs_request_v2",
    compact_filename="cbofs_fields.nc",
    display_name="CBOFS",
)
SCHEMA_VERSION = CONFIG.schema_version
COMPACT_SCHEMA_VERSION = roms.COMPACT_SCHEMA_VERSION
BUCKET = core.BUCKET
S3_ENDPOINT = core.S3_ENDPOINT
ALL_CYCLES: tuple[str, ...] = ("t00z", "t06z", "t12z", "t18z")
CANAL_LON_DEFAULT = -75.81
CANAL_LAT_DEFAULT = 39.53
CANAL_LON = CANAL_LON_DEFAULT
CANAL_LAT = CANAL_LAT_DEFAULT
CBOFS_VARIABLE_GRIDS = {"salt": "rho", "temp": "rho", "zeta": "rho", "u": "u", "v": "v"}
CBOFS_VARIABLE_3D = {"salt": True, "temp": True, "zeta": False, "u": True, "v": True}


def validate_request(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return core.validate_request(CONFIG, mapping)


def load_request(path: str | Path) -> dict[str, Any]:
    return core.load_request(CONFIG, path)


def parse_object_key(key: str) -> dict[str, Any] | None:
    return core.parse_object_key(CONFIG, key)


def discovery_prefixes(request: Mapping[str, Any]) -> list[str]:
    return core.discovery_prefixes(CONFIG, request)


def discover_objects(request: Mapping[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
    return core.discover_objects(CONFIG, request, **kwargs)


def select_objects(request: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return core.select_objects(request, objects)


def inventory_request(request: Mapping[str, Any] | str | Path, run_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    return core.inventory_request(CONFIG, request, run_dir, **kwargs)


def plan_request(request: Mapping[str, Any] | str | Path, run_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    return core.plan_request(CONFIG, request, run_dir, **kwargs)


def download_object(item: Mapping[str, Any], destination: str | Path, **kwargs: Any) -> dict[str, Any]:
    return core.download_object(CONFIG, item, destination, **kwargs)


def fetch_plan(plan: Mapping[str, Any] | str | Path, run_dir: str | Path,
               **kwargs: Any) -> dict[str, Any]:
    """Transfer only from a previously written and reviewed local plan."""
    return core.fetch_from_plan(CONFIG, plan, run_dir, **kwargs)


def fetch_request(request: Mapping[str, Any] | str | Path, run_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Disabled compatibility name; request-to-transfer bypasses plan review."""
    return core.fetch_request(CONFIG, request, run_dir, **kwargs)


def inspect_file(path: str | Path) -> dict[str, Any]:
    return core.inspect_file(path)


def extract_request(request: Mapping[str, Any] | str | Path, paths: Sequence[str | Path],
                    output: str | Path, *,
                    provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return roms.extract_fields(CONFIG, request, paths, output, provenance=provenance)


def evaluate_health(request: Mapping[str, Any] | str | Path, run_dir: str | Path,
                    compact_paths: Sequence[str | Path] = ()) -> dict[str, Any]:
    return roms.evaluate_health(CONFIG, request, run_dir, compact_paths)


def _legacy_warning(name: str) -> None:
    warnings.warn(
        f"{name} is a deprecated cbofs-canal compatibility wrapper; use the "
        "cbofs-fetcher inventory/plan/fetch/extract workflow instead",
        DeprecationWarning,
        stacklevel=2,
    )


def _cycle_hour(cycle: str) -> int:
    if cycle not in ALL_CYCLES:
        raise ValueError(f"cycle must be one of {ALL_CYCLES}")
    return int(cycle[1:3])


def _legacy_code(fhour: int) -> tuple[str, int]:
    if isinstance(fhour, bool) or not isinstance(fhour, int) or fhour < 0:
        raise ValueError("fhour must be a non-negative integer")
    # The legacy API called the cycle-time analysis fhour=0. The public AWS
    # archive has no f000/n000; its explicit cycle-time object is n006.
    return ("n", 6) if fhour == 0 else ("f", fhour)


def _legacy_field_url(date_str: str, cycle: str = "t00z", fhour: int = 0) -> str:
    run = datetime.strptime(date_str, "%Y-%m-%d")
    _cycle_hour(cycle)
    code, lead = _legacy_code(fhour)
    key = f"cbofs/netcdf/{run:%Y/%m/%d}/cbofs.{cycle}.{run:%Y%m%d}.fields.{code}{lead:03d}.nc"
    return f"{S3_ENDPOINT}/{key}"


def _request_for_single(date_str: str, cycle: str, fhour: int,
                        product: str = "fields") -> dict[str, Any]:
    run = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=_cycle_hour(cycle), tzinfo=timezone.utc)
    code, lead = _legacy_code(fhour)
    valid = run + timedelta(hours=(lead - 6 if code == "n" else lead))
    request: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "start_utc": core.iso(valid),
        "end_utc_exclusive": core.iso(valid + (timedelta(minutes=6) if product == "stations" else timedelta(hours=1))),
        "product": product,
        "guidance": "nowcast" if code == "n" else "forecast",
        "missing_policy": "error",
        "cache_policy": "keep",
        "max_workers": 1,
    }
    if code == "f":
        request["run_cycle_utc"] = core.iso(run)
    if product == "fields":
        request["variables"] = ["zeta", "salt", "u", "v"]
        request["vertical_views"] = ["surface"]
    return request


def _legacy_fetch_one(date_str: str, cycle: str, fhour: int, work_dir: Path,
                      product: str = "fields") -> Path:
    request = validate_request(_request_for_single(date_str, cycle, fhour, product))
    objects = discover_objects(request)
    run = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=_cycle_hour(cycle), tzinfo=timezone.utc)
    matches = [item for item in objects if item["run_time"] == core.iso(run)]
    plan_path = work_dir / "download_estimate.json"
    plan_request(request, work_dir, objects=matches, output=plan_path)
    manifest = fetch_plan(plan_path, work_dir)
    if len(manifest["outcomes"]) != 1:
        raise RuntimeError(f"expected one legacy compatibility object, found {len(manifest['outcomes'])}")
    return Path(manifest["outcomes"][0]["local_path"])


def get_cbofs_canal_nodes(canal_lon: float = CANAL_LON_DEFAULT,
                          canal_lat: float = CANAL_LAT_DEFAULT,
                          sample_date: str = "2026-07-20",
                          cycle: str = "t00z",
                          work_dir: str | Path | None = None) -> dict[str, Any]:
    """Deprecated nearest-C-grid-node helper backed by the public AWS archive."""
    _legacy_warning("get_cbofs_canal_nodes")
    import numpy as np
    import netCDF4
    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="cbofs_nodes_"))
    root.mkdir(parents=True, exist_ok=True)
    path = _legacy_fetch_one(sample_date, cycle, 0, root)
    try:
        with netCDF4.Dataset(path) as dataset:
            result: dict[str, Any] = {}
            for grid in ("rho", "u", "v"):
                lon = np.ma.filled(dataset.variables[f"lon_{grid}"][:], np.nan)
                lat = np.ma.filled(dataset.variables[f"lat_{grid}"][:], np.nan)
                distance = np.hypot(lon - canal_lon, lat - canal_lat)
                flat = int(np.nanargmin(distance))
                j, i = np.unravel_index(flat, lon.shape)
                result[f"i_{grid}"] = int(i)
                result[f"j_{grid}"] = int(j)
                if grid == "rho":
                    result.update(lon_found=float(lon[j, i]), lat_found=float(lat[j, i]),
                                  dist_deg=float(distance[j, i]),
                                  h_found=float(dataset.variables["h"][j, i]))
            result["N"] = len(dataset.dimensions["s_rho"])
            return result
    finally:
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".download.json").unlink(missing_ok=True)


def roms_station_depths(Cs_r: Any, s_rho: Any, hc: float, h_station: float,
                        zeta: float = 0.0, vtransform: int = 2):
    """Deprecated point-depth helper; production extraction reads file metadata."""
    _legacy_warning("roms_station_depths")
    import numpy as np
    result = roms.roms_depths(s_rho, Cs_r,
                              np.asarray(h_station), np.asarray(zeta), hc, vtransform)
    return np.asarray(result).reshape(-1)


def fetch_cbofs_canal(t_start: str, t_end: str, canal_nodes: dict[str, Any],
                      variables: Sequence[str] = ("salt", "temp"),
                      cycles: Sequence[str] = ALL_CYCLES, fhour: int = 0,
                      work_dir: str | Path | None = None, parallel: bool = False,
                      n_workers: int = 4, progress: bool = True) -> dict[str, Any]:
    """Deprecated point sampler preserving the old array-oriented return shape."""
    _legacy_warning("fetch_cbofs_canal")
    import numpy as np
    import netCDF4
    unknown = sorted(set(variables) - set(CBOFS_VARIABLE_GRIDS))
    if unknown:
        raise ValueError(f"unknown CBOFS variables: {unknown}")
    for cycle in cycles:
        _cycle_hour(cycle)
    if isinstance(n_workers, bool) or not isinstance(n_workers, int) or n_workers < 1:
        raise ValueError("n_workers must be a positive integer")
    if parallel:
        warnings.warn(
            "deprecated fetch_cbofs_canal runs serially so each transfer retains an independent reviewed plan",
            RuntimeWarning, stacklevel=2)
    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="cbofs_canal_"))
    d0, d1 = datetime.strptime(t_start, "%Y-%m-%d"), datetime.strptime(t_end, "%Y-%m-%d")
    if d1 < d0:
        raise ValueError("t_end must be on or after t_start")
    jobs = [((d0 + timedelta(days=day)).strftime("%Y-%m-%d"), cycle)
            for day in range((d1 - d0).days + 1) for cycle in cycles]
    rows: list[dict[str, Any]] = []
    for date_str, cycle in jobs:
        row: dict[str, Any] = {"time_dt64": np.datetime64("NaT", "s")}
        has_3d = any(CBOFS_VARIABLE_3D.get(name, False) for name in variables)
        if has_3d:
            row["z_levels"] = np.full(int(canal_nodes.get("N", 20)), np.nan)
        for name in variables:
            row[name] = (np.full(int(canal_nodes.get("N", 20)), np.nan)
                         if CBOFS_VARIABLE_3D[name] else np.asarray(np.nan))
        path: Path | None = None
        try:
            path = _legacy_fetch_one(date_str, cycle, fhour, root)
            with netCDF4.Dataset(path) as dataset:
                times = core.decode_times(dataset)
                row["time_dt64"] = np.datetime64(times[0]["normalized_time_utc"].replace("Z", ""), "s")
                geometry = roms.read_geometry(dataset)
                if has_3d:
                    zeta = float(dataset.variables["zeta"][0, canal_nodes["j_rho"], canal_nodes["i_rho"]])
                    row["z_levels"] = roms.roms_depths(geometry["s_rho"], geometry["Cs_r"],
                                                       np.asarray(geometry["h"][canal_nodes["j_rho"], canal_nodes["i_rho"]]),
                                                       np.asarray(zeta), geometry["hc"], geometry["Vtransform"]).reshape(-1)
                for name in variables:
                    grid = CBOFS_VARIABLE_GRIDS[name]
                    variable = dataset.variables[name]
                    index = [0]
                    if CBOFS_VARIABLE_3D[name]:
                        index += [slice(None)]
                    index += [canal_nodes[f"j_{grid}"], canal_nodes[f"i_{grid}"]]
                    row[name] = np.ma.filled(variable[tuple(index)], np.nan).astype(float)
        except Exception as exc:
            warnings.warn(f"CBOFS compatibility snapshot failed for {date_str} {cycle}: {exc}",
                          RuntimeWarning, stacklevel=2)
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
                path.with_name(path.name + ".download.json").unlink(missing_ok=True)
        rows.append(row)
    rows.sort(key=lambda row: row["time_dt64"])
    output: dict[str, Any] = {"time_dt64": np.asarray([row["time_dt64"] for row in rows], dtype="datetime64[s]")}
    if rows and "z_levels" in rows[0]:
        output["z_levels"] = np.stack([row["z_levels"] for row in rows])
    for name in variables:
        output[name] = np.asarray([row[name] for row in rows])
    return output


def _station_names(dataset: Any):
    import netCDF4
    if "station_name" not in dataset.variables:
        return []
    return [str(value).strip() for value in netCDF4.chartostring(dataset.variables["station_name"][:])]


def probe_cbofs_station_file(date_str: str = "2026-07-20", cycle: str = "t00z",
                             station_match: str = "8573927",
                             work_dir: str | Path | None = None) -> dict[str, Any]:
    """Deprecated station metadata probe backed by one AWS station-nowcast file."""
    _legacy_warning("probe_cbofs_station_file")
    import numpy as np
    import netCDF4
    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="cbofs_station_probe_"))
    run = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=_cycle_hour(cycle), tzinfo=timezone.utc)
    request = validate_request({"schema_version": SCHEMA_VERSION,
                                "start_utc": core.iso(run - timedelta(minutes=1)),
                                "end_utc_exclusive": core.iso(run + timedelta(minutes=1)),
                                "product": "stations", "guidance": "nowcast",
                                "missing_policy": "skip", "cache_policy": "keep", "max_workers": 1})
    path = _legacy_fetch_one(date_str, cycle, 0, root, "stations")
    sidecar = core.read_json(path.with_name(path.name + ".download.json"))
    try:
        with netCDF4.Dataset(path) as dataset:
            names = _station_names(dataset)
            station_idx = next((i for i, name in enumerate(names) if station_match in name), None)
            if station_idx is None:
                raise RuntimeError(f"station {station_match!r} was not found")
            records = core.decode_times(dataset)
            cs = np.asarray(dataset.variables["Cs_r"][:], dtype=float)
            sigma = np.asarray(dataset.variables["s_rho"][:], dtype=float)
            hvar = dataset.variables["h"]
            h_station = float(hvar[station_idx] if hvar.ndim == 1 else np.asarray(hvar[:]).reshape(-1)[station_idx])
            return {"station_idx": station_idx, "station_name": names[station_idx],
                    "n_sigma": len(sigma), "n_stations": len(names), "n_time": len(records),
                    "dt_minutes": 6 if len(records) > 1 else 0,
                    "time_start": np.datetime64(records[0]["normalized_time_utc"].replace("Z", ""), "s"),
                    "time_end": np.datetime64(records[-1]["normalized_time_utc"].replace("Z", ""), "s"),
                    "variables": list(dataset.variables), "Cs_r": cs, "s_rho": sigma,
                    "hc": float(roms._scalar(dataset, "hc")),
                    "vtransform": int(roms._scalar(dataset, "Vtransform")),
                    "h_station": h_station, "probe_url": sidecar["url"]}
    finally:
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".download.json").unlink(missing_ok=True)


def fetch_cbofs_station_ts(station_match: str, t_start: str, t_end: str,
                           cache_dir: str | Path, cycles: Sequence[str] = ALL_CYCLES,
                           work_dir: str | Path | None = None) -> dict[str, Any]:
    """Deprecated station T/S extractor; retains an NPZ-compatible result."""
    _legacy_warning("fetch_cbofs_station_ts")
    import numpy as np
    import netCDF4
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cache_path = cache / f"cbofs_sta_{station_match}_{t_start.replace('-', '')}_{t_end.replace('-', '')}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as saved:
            return {name: (saved[name].astype("datetime64[s]") if name == "time_dt64"
                           else float(np.asarray(saved[name]).reshape(-1)[0])
                           if name in {"hc", "h_station"}
                           else int(np.asarray(saved[name]).reshape(-1)[0])
                           if name == "vtransform"
                           else np.asarray(saved[name], dtype=np.float64)
                           if name in {"salt", "temp", "zeta", "Cs_r", "s_rho"}
                           else np.asarray(saved[name]))
                    for name in saved.files}
    start = datetime.strptime(t_start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(t_end, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    if end <= start:
        raise ValueError("t_end must be on or after t_start")
    request = validate_request({"schema_version": SCHEMA_VERSION, "start_utc": core.iso(start),
                                "end_utc_exclusive": core.iso(end), "product": "stations",
                                "guidance": "nowcast", "missing_policy": "skip",
                                "cache_policy": "keep", "max_workers": 4})
    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="cbofs_station_"))
    allowed_hours = {_cycle_hour(cycle) for cycle in cycles}
    objects = [item for item in discover_objects(request) if item["cycle_hour"] in allowed_hours]
    plan = plan_request(request, root, objects=objects)
    manifest = fetch_plan(plan, root)
    candidates: dict[np.datetime64, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    for outcome in manifest["outcomes"]:
        with netCDF4.Dataset(outcome["local_path"]) as dataset:
            names = _station_names(dataset)
            station_idx = next((i for i, name in enumerate(names) if station_match in name), None)
            if station_idx is None:
                continue
            records = core.decode_times(dataset)
            def station_series(variable: Any):
                axis = next(i for i, dim in enumerate(variable.dimensions) if "stat" in dim.lower())
                index = [slice(None)] * variable.ndim
                index[axis] = station_idx
                values = np.ma.filled(variable[tuple(index)], np.nan)
                remaining = [dim for i, dim in enumerate(variable.dimensions) if i != axis]
                time_axis = next(i for i, dim in enumerate(remaining) if dim in {"time", "ocean_time"})
                return np.moveaxis(values, time_axis, 0)
            salt_values = station_series(dataset.variables["salt"])
            temp_values = station_series(dataset.variables["temp"])
            zeta_values = station_series(dataset.variables["zeta"])
            run_time = core.parse_utc(outcome["source"]["run_time"])
            for index, item in enumerate(records):
                stamp = np.datetime64(item["normalized_time_utc"].replace("Z", ""), "s")
                candidates.setdefault(stamp, []).append({
                    "run_time": run_time, "path": outcome["local_path"],
                    "salt": salt_values[index], "temp": temp_values[index],
                    "zeta": zeta_values[index],
                })
            metadata = {"Cs_r": np.asarray(dataset.variables["Cs_r"][:]),
                        "s_rho": np.asarray(dataset.variables["s_rho"][:]),
                        "hc": float(roms._scalar(dataset, "hc")),
                        "vtransform": int(roms._scalar(dataset, "Vtransform")),
                        "h_station": float(np.asarray(dataset.variables["h"][:]).reshape(-1)[station_idx])}
    lower, upper = np.datetime64(start.replace(tzinfo=None), "s"), np.datetime64(end.replace(tzinfo=None), "s")
    selected = [(stamp, min(group, key=lambda item: (item["run_time"], item["path"])))
                for stamp, group in sorted(candidates.items()) if lower <= stamp < upper]
    if not selected:
        raise RuntimeError(f"station {station_match!r} had no records in the requested window")
    output = {"time_dt64": np.asarray([stamp for stamp, _ in selected], dtype="datetime64[s]"),
              "salt": np.asarray([item["salt"] for _, item in selected]),
              "temp": np.asarray([item["temp"] for _, item in selected]),
              "zeta": np.asarray([item["zeta"] for _, item in selected]), **metadata}
    np.savez_compressed(cache_path, **output)
    try:
        roms.delete_raw_cache(root, time_audit=core.audit_time_records(CONFIG, request, [
            item["local_path"] for item in manifest["outcomes"]]))
    except Exception as exc:
        warnings.warn(f"deprecated station wrapper could not clean its raw cache: {exc}",
                      RuntimeWarning, stacklevel=2)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="discover matching public S3 objects")
    inventory.add_argument("--request", required=True)
    inventory.add_argument("--run-dir", default=".")
    inventory.add_argument("--output")
    plan = sub.add_parser("plan", help="select objects and write an exact estimate")
    plan.add_argument("--request", required=True)
    plan.add_argument("--run-dir", required=True)
    plan.add_argument("--output")
    fetch = sub.add_parser("fetch", help="download an approved plan")
    fetch.add_argument("--plan", required=True)
    fetch.add_argument("--run-dir", required=True)
    inspect = sub.add_parser("inspect", help="inspect downloaded NetCDF metadata")
    inspect.add_argument("--input", action="append", required=True)
    inspect.add_argument("--output")
    extract = sub.add_parser("extract", help="create a compact ROMS field product")
    extract.add_argument("--request", required=True)
    source = extract.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", action="append")
    source.add_argument("--manifest")
    source.add_argument("--run-dir")
    extract.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        report = inventory_request(args.request, args.run_dir, output=args.output)
    elif args.command == "plan":
        report = plan_request(args.request, args.run_dir, output=args.output)
    elif args.command == "fetch":
        report = fetch_plan(args.plan, args.run_dir)
    elif args.command == "inspect":
        report = {"schema_version": "cbofs_inspection_set_v1",
                  "files": [inspect_file(path) for path in args.input]}
        if args.output:
            core.write_json_atomic(args.output, report)
    else:
        provenance = None
        if args.input:
            paths = [Path(path) for path in args.input]
        elif args.manifest:
            manifest_path = Path(args.manifest).resolve()
            paths, provenance = core.verified_manifest_inputs(
                CONFIG, manifest_path, request=args.request, run_dir=manifest_path.parent)
        else:
            run_dir = Path(args.run_dir).resolve()
            paths, provenance = core.verified_manifest_inputs(
                CONFIG, run_dir / "fetch_manifest.json", request=args.request,
                run_dir=run_dir)
        report = extract_request(args.request, paths, args.output, provenance=provenance)
    print(json.dumps(core.json_clean(report), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
