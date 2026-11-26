"""Build and independently reopen node-distributed, volume-only FVCOM river forcing."""
import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np

from fvcom_writer import write_river_nc, write_river_nml
from grid_utils import datetime64_to_mjd, read_obc_nodes_dat


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def second(text):
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.utcoffset() is None or dt.microsecond:
        raise ValueError("whole-second timestamp with UTC offset required")
    return int(dt.timestamp())


def stamp(value):
    return datetime.fromtimestamp(int(value), timezone.utc).isoformat().replace("+00:00", "Z")


def series_window(manifest_path, start, end, step):
    manifest = load(manifest_path)
    if manifest.get("schema") != "usgs_continuous_discharge_v1" or manifest.get("units") != "m3/s":
        raise ValueError("expected a verified USGS discharge series in m3/s")
    products = {key: manifest_path.parent / manifest[key]["path"] for key in ("raw", "series")}
    for key, product in products.items():
        if digest(product) != manifest[key]["sha256"]:
            raise ValueError(f"{key} hash differs from source manifest")
    with products["series"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.array([second(row["time_utc"]) for row in rows], dtype=np.int64)
    q = np.array([float(row["discharge_m3s"]) for row in rows])
    original = np.array([float(row["discharge_ft3s"]) for row in rows])
    if not np.all(np.isfinite(q)) or np.any(q < 0) or not np.any(q > 0):
        raise ValueError("finite nonnegative inflow with positive flow required")
    if manifest["original_units"] != "ft^3/s" or not np.allclose(q, original * 0.028316846592, rtol=1e-12, atol=0):
        raise ValueError("USGS discharge conversion mismatch")
    if len(times) < 2 or len(times) != manifest["records"] or manifest["interval_seconds"] <= 0:
        raise ValueError("invalid source record count or cadence")
    if not np.all(np.diff(times) == manifest["interval_seconds"]):
        raise ValueError("source series has missing, unordered or duplicated records")
    if times[0] > start or times[-1] < end:
        raise ValueError("source series does not bracket the run")
    if end <= start or step <= 0 or (end - start) % step:
        raise ValueError("run endpoints must align to positive forcing cadence")
    grid = np.arange(start, end + 1, step, dtype=np.int64)
    knots = np.r_[start, times[(times > start) & (times < end)], end]
    if np.any((knots - start) % step):
        raise ValueError("forcing cadence must represent all source breakpoints")
    q_knots = np.interp(knots, times, q)
    integrated = float(np.sum(np.diff(knots) * (q_knots[1:] + q_knots[:-1]) * 0.5))
    return grid, np.interp(grid, times, q), integrated, manifest


def source_nodes(mapping, mesh, depths, ocean):
    if mapping["mesh_sha256"] != digest(mesh) or mapping["obc_sha256"] != digest(ocean):
        raise ValueError("mapping grid or ocean boundary hash differs from submitted inputs")
    separation = mapping["gauge_to_cutoff_km"]
    if not mapping["cutoff_description"].strip() or not math.isfinite(separation) or separation < 0:
        raise ValueError("document the upstream cutoff and gauge separation")
    floor = float(mapping["wet_depth_floor_m"])
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("invalid wet depth floor")
    with mesh.open(encoding="ascii") as f:
        node_count = int(f.readline().split("=")[1])
        triangle_count = int(f.readline().split("=")[1])
        faces = np.loadtxt(f, dtype=int, max_rows=triangle_count, ndmin=2)[:, 1:4]
        vertices = np.loadtxt(f, ndmin=2)
    with depths.open(encoding="ascii") as f:
        if int(f.readline().split("=")[1]) != node_count:
            raise ValueError("depth header does not match grid")
        dep = np.loadtxt(f, ndmin=2)
    if vertices.shape[0] != node_count or not np.array_equal(vertices[:, 0], np.arange(1, node_count + 1)):
        raise ValueError("runtime grid node ID/order mismatch")
    if dep.shape != (node_count, 3) or not np.array_equal(vertices[:, 1:4], dep):
        raise ValueError("runtime depth coordinates or positive-down depths differ from grid")
    if not np.all(np.isfinite(dep)) or faces.min() < 1 or faces.max() > node_count:
        raise ValueError("invalid grid connectivity or depth data")
    ocean_nodes = set(map(int, read_obc_nodes_dat(ocean)))
    edge_counts = Counter()
    adjacent = defaultdict(set)
    for a, b, c in faces:
        for left, right in [(a, b), (b, c), (c, a)]:
            edge_counts[tuple(sorted((left, right)))] += 1
            adjacent[left].add(right)
            adjacent[right].add(left)
    boundary = {n for edge, count in edge_counts.items() if count == 1 for n in edge}
    if not ocean_nodes or not ocean_nodes <= boundary:
        raise ValueError("ocean OBC nodes missing or off boundary")
    wet = set(np.flatnonzero(dep[:, 2] > floor) + 1)
    reached = set(ocean_nodes & wet)
    todo = deque(reached)
    while todo:
        for neighbor in adjacent[todo.popleft()]:
            if neighbor in wet and neighbor not in reached:
                reached.add(neighbor)
                todo.append(neighbor)

    entries = mapping["nodes"]
    ids = [item["node_id"] for item in entries]
    if not ids or any(type(n) is not int for n in ids) or len(set(ids)) != len(ids):
        raise ValueError("unique integer receiving node IDs required")
    if not set(ids) <= ((reached & boundary) - ocean_nodes):
        raise ValueError("receiving nodes must be wet, solid-boundary and connected to ocean")
    widths = np.array([float(item["width_m"]) for item in entries])
    if not np.all(np.isfinite(widths) & (widths > 0)):
        raise ValueError("positive finite tributary widths required at cutoff")
    h = dep[np.array(ids) - 1, 2]
    areas = widths * h
    weights = areas / areas.sum()
    if not np.all(np.isfinite(weights)) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("invalid cross-sectional weights")
    return [{"name": f"inflow_{n}", "node_id": n, "width_m": w, "depth_m": d, "weight": p}
            for n, w, d, p in zip(ids, widths, h, weights)]


def freshwater_temperatures(startup, nodes, depth_path, start):
    """Recover the inactive river temperature placeholder from the matching frozen water column."""
    ids = np.array([row["node_id"] for row in nodes], dtype=int) - 1
    target_depth = np.loadtxt(depth_path, skiprows=1, ndmin=2)[:, 2]
    with netCDF4.Dataset(startup) as ds:
        if ds["temp"].dimensions != ("time", "siglay", "node") or len(ds.dimensions["time"]) != 1:
            raise ValueError("one native startup temperature profile per node is required")
        if not np.allclose(ds["h"][:], target_depth, rtol=0, atol=.001):
            raise ValueError("startup bathymetry differs from runtime depths")
        ticks = np.asarray(ds["Itime"][:], dtype="i8") * 86400000 + np.asarray(ds["Itime2"][:], dtype="i8")
        if ticks.shape != (1,) or ticks[0] != 40587*86400000 + start*1000:
            raise ValueError("startup clock must equal the first river forcing timestamp")
        sigma = ds["siglev"]
        if sigma.dimensions == ("siglev",):
            levels = np.broadcast_to(sigma[:][:, None], (len(sigma), len(ids)))
        elif sigma.dimensions == ("siglev", "node"):
            levels = sigma[:, ids]
        else:
            raise ValueError("unsupported startup sigma interface dimensions")
        thickness = -np.diff(levels, axis=0)
        temps = np.ma.asarray(ds["temp"][0, :, ids])
        if (np.ma.is_masked(temps) or np.ma.is_masked(thickness) or not np.isfinite(temps).all()
            or not np.isfinite(thickness).all() or np.any(thickness <= 0)
            or not np.allclose(levels[[0,-1]], [[0.],[-1.]], rtol=0., atol=1e-6)
            or np.any((temps < -5) | (temps > 45))):
            raise ValueError("invalid startup temperature or physical layer thickness")
        return np.average(np.asarray(temps, dtype=float), axis=0, weights=np.asarray(thickness, dtype=float))


def build(source, mapping_path, grid_path, depth_path, obc_path, start, end, step, output,
          *, freshwater_temperature_startup=None):
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    seconds, q, expected_volume, manifest = series_window(source, start, end, step)
    mapping = load(mapping_path)
    nodes = source_nodes(mapping, grid_path, depth_path, obc_path)
    names = [n["name"] for n in nodes]
    distributed = q[:, None] * np.array([n["weight"] for n in nodes])
    temperature = np.full(len(nodes),20.)
    salinity = 30.
    info = ("Barotropic river volume forcing", "Uniform inactive scalars", "barotropic-tide-river")
    if freshwater_temperature_startup is not None:
        temperature = freshwater_temperatures(freshwater_temperature_startup, nodes, depth_path, start)
        salinity = 0.
        info = ("Freshwater inflow for active salinity", "Fixed temperature from startup receiver depth averages", "estuarine-salinity")
    mjd = datetime64_to_mjd(seconds.astype("datetime64[s]"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as stage:
        products = Path(stage) / "river"
        products.mkdir()
        nc = products / "river.nc"
        nml = products / "river.nml"
        write_river_nc(nc, names, mjd, distributed, np.broadcast_to(temperature,distributed.shape),
                       np.full(distributed.shape, salinity), info1=info[0],
                       info2=info[1], casename=info[2], precise_clock=True)
        write_river_nml(nml, names, [n["node_id"] for n in nodes], nc.name, named_distribution=True)
        with netCDF4.Dataset(nc) as ds:
            received = ds["river_flux"][:].astype(np.float64).sum(axis=1)
            dates = np.char.decode(netCDF4.chartostring(ds["Times"][:], encoding="bytes"), "ascii")
            expected_dates = np.datetime_as_string(seconds.astype("datetime64[s]"), unit="us")
            clock_ms = ds["Itime"][:].astype(np.int64) * 86400000 + ds["Itime2"][:]
            if not np.array_equal(clock_ms, np.rint(mjd * 86400000)) or not np.array_equal(dates, expected_dates):
                raise ValueError("inconsistent integer or character FVCOM clock")
            if not np.array_equal(ds["time"][:], mjd):
                raise ValueError("inconsistent floating-point FVCOM clock")
            file_names = [v.strip() for v in netCDF4.chartostring(ds["river_names"][:])]
            if names != file_names:
                raise ValueError("river entry order mismatch")
            if not np.all(ds["river_salt"][:]==salinity) or not np.all(ds["river_temp"][:]==temperature.astype("f4")):
                raise ValueError("river scalar readback differs from the declared freshwater or barotropic policy")
            for name, unit in [("river_flux", "m^3s^-1"), ("river_temp", "Celsius"), ("river_salt", "PSU")]:
                if ds[name].units != unit:
                    raise ValueError(f"FVCOM reader unit mismatch for {name}")
        written_volume = float(np.sum(0.5 * (received[1:] + received[:-1]) * np.diff(seconds)))
        if not np.allclose(q, received, rtol=2e-7, atol=1e-9) or not math.isclose(written_volume, expected_volume, rel_tol=2e-7, abs_tol=1e-7):
            raise ValueError("distributed NetCDF does not conserve source discharge or integrated volume")
        result = {"schema": "fvcom_river_forcing_v1", "status": "ready",
            "river_number": len(nodes), "nodes": nodes, "start_utc": stamp(start), "end_utc": stamp(end),
            "cadence_seconds": step, "records": len(seconds), "mesh_sha256": digest(grid_path),
            "obc_sha256": digest(obc_path), "depth_sha256": digest(depth_path), "mapping_sha256": digest(mapping_path),
            "cutoff_description": mapping["cutoff_description"], "gauge_to_cutoff_km": mapping["gauge_to_cutoff_km"],
            "source": {"provider": "usgs", "station_id": manifest["site"].removeprefix("USGS-"),
                       "parameter_code": manifest["parameter_code"], "source_manifest_sha256": digest(source),
                       "series_sha256": manifest["series"]["sha256"], "approval_counts": manifest["approval_counts"]},
            "files": {k: {"path": f.name, "sha256": digest(f)} for k, f in [("netcdf", nc), ("namelist", nml)]},
            "checks": {"max_discharge_error_m3s": float(np.max(np.abs(q - received))),
                       "source_volume_m3": expected_volume, "written_volume_m3": written_volume,
                       "volume_basis": "unramped forcing over inclusive piecewise-linear sample intervals",
                       "minimum_receiving_depth_m": min(node["depth_m"] for node in nodes),
                       "wet_depth_floor_m": mapping["wet_depth_floor_m"]}}
        if freshwater_temperature_startup is not None:
            result["tracers"] = {"mode":"active_salinity_freshwater", "salinity_psu":0.,
                "temperature_role":"inactive reader companion, sigma-thickness-weighted frozen startup profile",
                "temperature_degC_by_node":dict(zip(names, temperature.tolist())),
                "startup_sha256":digest(freshwater_temperature_startup), "source_salt_transport_psu_m3_s":0.}
        (products / "river_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        products.rename(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source-manifest", "mapping", "grd", "dep", "obc", "output-dir"):
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--start", required=True, type=second)
    parser.add_argument("--end", required=True, type=second)
    parser.add_argument("--interval-seconds", default=360, type=int)
    parser.add_argument("--freshwater-temperature-startup", type=Path,
                        help="Set active river salinity to zero and use the single-record frozen startup for the inactive temperature companion")
    args = parser.parse_args()
    print(json.dumps(build(args.source_manifest, args.mapping, args.grd, args.dep, args.obc,
                           args.start, args.end, args.interval_seconds, args.output_dir,
                           freshwater_temperature_startup=args.freshwater_temperature_startup)["checks"], indent=2))
