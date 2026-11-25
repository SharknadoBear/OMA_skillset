"""Compare a matched FVCOM pair on a frozen, bank-to-bank control section."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import netCDF4 as nc
import numpy as np
import pandas as pd

from condense_fvcom_station import load_stack
from run_case_validation import parse_namelist, read, sha256, verify_prerequisites


def records(project, attempt):
    return verify_prerequisites(argparse.Namespace(
        project=project, attempt=attempt, grid_case="fresh_reproduction",
        input_freeze=None, production_audit=None, station_netcdf=None,
        observation_manifest=None, forcing_manifest=None))


def check_pair(control, river):
    from fvcom_profiles import FACTORIAL_PROFILE, PAIR_PROFILE, WIND_PROFILE, has_river, has_wind
    a, b = control["request"], river["request"]
    if b.get("profile") == FACTORIAL_PROFILE:
        profiles_match = a.get("profile") in {WIND_PROFILE, FACTORIAL_PROFILE}
    else:
        profiles_match = a.get("profile") == b.get("profile") == PAIR_PROFILE
    if not profiles_match or has_river(a) or not has_river(b) or has_wind(a) != has_wind(b):
        raise ValueError("expected matched river-off and river-on treatments with the same wind setting")
    if control["executable_hash"] != river["executable_hash"]:
        raise ValueError("different executables in paired experiment")
    left, right = (parse_namelist(c["namelist"]) for c in (control, river))
    for key in set(left) | set(right):
        if key.startswith("RIVER_") or key in {"CASE_TITLE", "INPUT_DIR", "OUTPUT_DIR"}:
            continue
        if left.get(key) != right.get(key):
            raise ValueError(f"paired namelist setting differs: {key}")
    products = read(river["input_dir"] / "river_manifest.json")["files"]
    additional = {"river_manifest.json", products["namelist"]["path"], products["netcdf"]["path"]}
    right_files = {k: v for k, v in river["freeze"]["files"].items() if k not in additional}
    if right_files != control["freeze"]["files"]:
        raise ValueError("paired runtime inputs differ beyond the river forcing")
    if sha256(control["inventory_path"]) != sha256(river["inventory_path"]):
        raise ValueError("observation station inventories differ")


def section_geometry(tri, x, y, geometry, river_nodes=(), ocean_nodes=()):
    """Check the complete inner edge cut and derive normals from metric coordinates."""
    cells = np.asarray(geometry["upstream_element_ids"], dtype=int) - 1
    if len(cells) == 0 or len(np.unique(cells)) != len(cells) or np.any((cells < 0) | (cells >= len(tri))):
        raise ValueError("invalid upstream control volume cells")
    inside = set(cells.tolist())
    if set(np.unique(tri[cells])).intersection(ocean_nodes):
        raise ValueError("upstream control volume intersects the ocean OBC")
    for node in river_nodes:
        incident = np.flatnonzero(np.any(tri == node, axis=1))
        if not len(incident) or any(idx not in inside for idx in incident):
            raise ValueError("section must enclose all cells incident on each river source")
    edge_cells = {}
    for idx, nodes in enumerate(tri):
        for a, b in zip(nodes, np.roll(nodes, -1)):
            edge_cells.setdefault(tuple(sorted((a, b))), []).append(idx)
    connected = {idx: [] for idx in inside}
    for neighbors in edge_cells.values():
        if len(neighbors) == 2 and all(i in inside for i in neighbors):
            a, b = neighbors
            connected[a].append(b)
            connected[b].append(a)
    visited = {cells[0]}
    queue = [cells[0]]
    while queue:
        for other in connected[queue.pop()]:
            if other not in visited:
                queue.append(other)
                visited.add(other)
    if visited != inside:
        raise ValueError("upstream control volume must be connected")
    cut = {edge for edge, neighbors in edge_cells.items()
           if len(neighbors) == 2 and sum(i in inside for i in neighbors) == 1}
    seen = set()
    faces = []
    for row in geometry["section"]:
        a, b = (int(n) - 1 for n in row["edge_nodes"])
        cell = int(row["cell_id"]) - 1
        edge = tuple(sorted((a, b)))
        if edge in seen or edge not in cut or cell not in inside or cell not in edge_cells[edge]:
            raise ValueError("section face does not bound the upstream cells")
        seen.add(edge)
        dx, dy = x[b] - x[a], y[b] - y[a]
        length = np.hypot(dx, dy)
        if not length > 0:
            raise ValueError("zero length section face")
        normal = np.array([dy, -dx]) / length
        midpoint = np.array([(x[a] + x[b])/2, (y[a] + y[b])/2])
        center = np.array([x[tri[cell]].mean(), y[tri[cell]].mean()])
        if np.dot(normal, midpoint - center) < 0:
            normal = -normal
        faces.append((row["station_id"], cell, a, b, length, normal))
    if not cut or cut != seen:
        raise ValueError("section does not cover the complete open face of the control volume")
    return cells, faces


def sample_member(case, geometry, river_nodes=(), ocean_nodes=()):
    run = parse_namelist(case["namelist"])
    if geometry["mesh_sha256"] != case["freeze"]["files"][run["GRID_FILE"]]:
        raise ValueError("section geometry is bound to a different runtime grid")
    names, iints, fields, _ = load_stack(case["station_paths"])
    step = float(run["EXTSTEP_SECONDS"]) * int(run["ISPLIT"])
    times = pd.DatetimeIndex(case["run_start"] + pd.to_timedelta(iints * step, unit="s"))
    stations = {str(s["station_id"]): s for s in read(case["mapping_path"])["stations"]}
    storage = {}
    faces = None
    for row in read(case["audit_path"])["full_grid_netcdf"]:
        path = Path(row["path"])
        if not path.is_file():
            path = case["attempt"] / Path(row["path"].replace("\\", "/")).name
        if sha256(path) != row["sha256"]:
            raise ValueError("grid output differs from production audit")
        with nc.Dataset(path) as ds:
            if faces is None:
                nv = ds["nv"]
                tri = np.moveaxis(np.asarray(nv[:], dtype=int), nv.dimensions.index("nele"), 0) - 1
                x, y, h = (np.asarray(ds[v][:], dtype=float) for v in ("x", "y", "h"))
                cells, faces = section_geometry(tri, x, y, geometry, river_nodes, ocean_nodes)
                vertices = tri[cells]
                p, q, r = vertices.T
                area = abs((x[q]-x[p])*(y[r]-y[p]) - (x[r]-x[p])*(y[q]-y[p])) / 2
                if np.any(area <= 0):
                    raise ValueError("zero-area control volume element")
            for i, clock in enumerate(ds["iint"][:]):
                zeta = np.ma.asarray(ds["zeta"][i, :])
                if np.ma.getmaskarray(zeta)[vertices].any():
                    raise ValueError("masked control volume elevations")
                column = np.maximum(h[vertices] + np.asarray(zeta)[vertices], 0.0)
                value = np.sum(area * column.mean(axis=1))
                if int(clock) in storage and storage[int(clock)] != value:
                    raise ValueError("inconsistent overlapping storage records")
                storage[int(clock)] = value
    if faces is None or not storage:
        raise ValueError("full-grid output missing")
    discharge = np.zeros(len(times))
    for name, cell, a, b, length, normal in faces:
        if name not in stations or stations[name]["cell_id"] != cell+1:
            raise ValueError(f"diagnostic station does not sample its transect cell: {name}")
        j = names.index(name)
        depth = (h[a]+h[b])/2 + fields["zeta"][:, j]
        discharge += length * np.maximum(depth, 0) * (normal[0]*fields["ua"][:, j] + normal[1]*fields["va"][:, j])
    volume = pd.Series(storage).sort_index()
    volume.index = case["run_start"] + pd.to_timedelta(volume.index.to_numpy()*step, unit="s")
    return pd.Series(discharge, index=times), volume, fields["zeta"], names


def applied_flux(case, times):
    nml = parse_namelist(case["namelist"])
    product = read(case["input_dir"] / "river_manifest.json")["files"]["netcdf"]["path"]
    with nc.Dataset(case["input_dir"] / product) as ds:
        epoch = pd.Timestamp("1858-11-17", tz="UTC")
        source = (np.asarray(ds["Itime"][:], dtype=float)*86400 + np.asarray(ds["Itime2"][:])/1000)
        flux = np.ma.asarray(ds["river_flux"][:])
        if np.ma.getmaskarray(flux).any() or not np.isfinite(flux).all() or np.any(flux < 0):
            raise ValueError("invalid river source values")
        axis = ds["river_flux"].dimensions.index("rivers")
        total = np.asarray(flux, dtype=float).sum(axis=axis)
    t = np.asarray((times - epoch).total_seconds())
    if not np.all(np.diff(source) > 0) or t[0] < source[0] or t[-1] > source[-1]:
        raise ValueError("river forcing does not bracket the diagnostic clock")
    ramp_seconds = int(nml["IRAMP"])*int(nml["ISPLIT"])*float(nml["EXTSTEP_SECONDS"])
    elapsed = np.asarray((times - case["run_start"]).total_seconds())
    ramp = np.tanh(elapsed/ramp_seconds) if ramp_seconds else np.ones(len(times))
    return np.interp(t, source, total)*ramp


def integral(values, seconds):
    return float(np.sum(0.5*(values[1:]+values[:-1])*np.diff(seconds)))


def water_level_metrics(control, river, gauges, times, names, control_zeta, river_zeta):
    results = []
    for station in gauges:
        rel = Path("water_level") / f"{station}_noaa_waterlevel.csv"
        raw = control["observation_path"].parent / rel
        if sha256(raw) != sha256(river["observation_path"].parent / rel):
            raise ValueError(f"paired NOAA series differ at {station}")
        reference = pd.read_csv(raw)
        reference.index = pd.to_datetime(reference["time"], utc=True)
        if not reference.index.is_unique:
            raise ValueError("duplicate NOAA comparison timestamps")
        j = names.index(station)
        model = pd.DataFrame({"control": control_zeta[:, j], "river": river_zeta[:, j]}, index=times)
        joined = model.join(reference[["observed", "predicted"]], how="inner")
        joined = joined.loc[(joined.index >= river["start"]) & (joined.index < river["end"])]
        for product in ("observed", "predicted"):
            common = joined[["control", "river", product]].dropna()
            if common.empty and product == "predicted":
                raise ValueError(f"missing common prediction samples for {station}")
            for mode in ("control", "river"):
                diff = common[mode] - common[product]
                results.append({"station": station, "reference": product, "member": mode,
                                "samples": len(common), "reference_sha256": sha256(raw),
                                "mean_difference_m": diff.mean(),
                                "centered_rmse_m": np.sqrt(((diff-diff.mean())**2).mean())})
    return pd.DataFrame(results)


def compare(control, river, geometry_path, output):
    check_pair(control, river)
    geometry = read(geometry_path)
    river_nodes = [row["node_id"] - 1 for row in read(river["input_dir"] / "river_manifest.json")["nodes"]]
    nml = parse_namelist(river["namelist"])
    ocean_nodes = np.loadtxt(river["input_dir"] / nml["OBC_NODE_LIST_FILE"],
                             skiprows=1, usecols=1, dtype=int, ndmin=1) - 1
    cq, cv, cz, names = sample_member(control, geometry, river_nodes, ocean_nodes)
    rq, rv, rz, rnames = sample_member(river, geometry, river_nodes, ocean_nodes)
    if not cq.index.equals(rq.index) or not cv.index.equals(rv.index) or names != rnames:
        raise ValueError("paired clocks or stations differ")
    if not ((cq.index[1:] - cq.index[:-1]) == pd.Timedelta(minutes=6)).all():
        raise ValueError("section sampling must cover every six minutes")
    times = rq.index
    data = pd.DataFrame({"applied_river_m3_s": applied_flux(river, times),
                         "control_export_m3_s": cq, "river_export_m3_s": rq}, index=times)
    volumes = pd.DataFrame({"control_storage_m3": cv, "river_storage_m3": rv})
    if not np.isfinite(data).all().all() or not np.isfinite(volumes).all().all():
        raise ValueError("non-finite paired budget")
    budgets = {}
    for label, start in (("full_run", river["run_start"]), ("analysis", river["start"])):
        end = river["end"]
        window = data.loc[start:end]
        if window.index[0] != start or window.index[-1] != end:
            raise ValueError("missing budget endpoints")
        seconds = np.asarray((window.index-start).total_seconds())
        applied, control_out, river_out = [integral(window[c].to_numpy(), seconds) for c in data.columns]
        change = volumes.loc[end] - volumes.loc[start]
        storage = float(change["river_storage_m3"] - change["control_storage_m3"])
        residual = applied - (river_out-control_out) - storage
        if applied <= 0:
            raise ValueError("no positive river volume delivered")
        budgets[label] = dict(applied_m3=applied, excess_export_m3=river_out-control_out,
                              excess_storage_change_m3=storage, residual_m3=residual,
                              residual_fraction=residual/applied)
    mask = (times >= river["start"]) & (times < river["end"])
    gauges = [sid for sid, role in river["station_keys"] if role == "water_level"]
    offsets = {sid: float(np.mean(rz[mask, names.index(sid)] - cz[mask, names.index(sid)])) for sid in gauges}
    metrics = water_level_metrics(control, river, gauges, times, names, cz, rz)
    result = dict(schema="fvcom_paired_river_budget_v1", geometry_sha256=sha256(geometry_path),
                  control_input_sha256=control["freeze"]["input_bundle_sha256"],
                  river_input_sha256=river["freeze"]["input_bundle_sha256"], budgets=budgets,
                  river_minus_control_mean_water_level_m=offsets,
                  method="Six-minute trapezoidal integration of ramped input and excess section export. "
                         "Normals use model x/y. Transport uses upstream cell mean velocity, station elevation "
                         "and edge-mean bed depth. Storage integrates nodal positive water columns on upstream "
                         "triangles at the audited endpoints. Residual includes spatial and temporal quadrature.")
    output.mkdir(parents=True, exist_ok=False)
    data.to_csv(output/"section.csv", index_label="time_utc")
    volumes.to_csv(output/"storage.csv", index_label="time_utc")
    metrics.to_csv(output/"common_noaa_levels.csv", index=False)
    (output/"budget.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(times, data["applied_river_m3_s"], label="Applied river")
    axes[0].plot(times, rq-cq, label="Additional seaward transport", alpha=.75)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=times.tz))
    axes[0].set_xlabel("Date (UTC)")
    axes[0].set_ylabel("Discharge (m³/s)")
    axes[0].legend()
    station_names = {str(row["id"]): row.get("name") for row in read(river["inventory_path"])["stations"]}
    labels = [textwrap.fill(station_names.get(sid) or sid, 17) for sid in offsets]
    axes[1].bar(range(len(offsets)), list(offsets.values()), tick_label=labels)
    axes[1].set_ylabel("River − control mean level (m)")
    axes[1].set_xlabel("NOAA station")
    fig.savefig(output/"paired_response.png", dpi=160)
    plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control-project", "control-attempt", "river-project", "river-attempt", "geometry", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    result = compare(records(args.control_project, args.control_attempt),
                     records(args.river_project, args.river_attempt), args.geometry, args.output)
    print(json.dumps(result, indent=2))
