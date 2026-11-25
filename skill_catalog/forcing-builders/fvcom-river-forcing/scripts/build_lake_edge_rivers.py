"""Construct native signed edge-river forcing from a NOS river control and prepared gauged series."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import tempfile

import netCDF4
import numpy as np
import pandas as pd

from fvcom_writer import write_river_nc, write_river_nml


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_control(path):
    lines = path.read_text().splitlines()
    counts = next(line.split("!!")[0].split() for line in lines if "NIJ NRIVERS" in line)
    n_edges, n_gauges = map(int, counts[:2])
    stations = {}
    edges = []
    for line in lines:
        if not re.match(r"^\s*\d+\s+", line) or "!!" in line:
            continue
        f = shlex.split(line)
        if len(f) >= 13 and f[3] in {"USGS", "ECCC"}:
            stations[int(f[0])] = {"id": f[1], "name": f[12], "agency": f[3]}
        elif len(f) == 10:
            edges.append({"index": int(f[0]), "cell_id": int(f[2]), "flow_source": int(f[5]),
                          "flow_scale": float(f[6]), "temperature_source": int(f[7]),
                          "temperature_scale": float(f[8])})
    require(len(stations) == n_gauges and len(edges) == n_edges, "control counts differ")
    require([e["index"] for e in edges] == list(range(1, n_edges + 1)), "nonsequential edges")
    require(len({e["cell_id"] for e in edges}) == n_edges, "duplicate river element")
    require(all(e["flow_source"] in stations and e["temperature_source"] in stations for e in edges),
            "control references unknown gauge")
    return stations, edges


def edge_faces(edges, tri, nbe, depth):
    for edge in edges:
        cell = edge["cell_id"]
        require(1 <= cell <= len(tri), "invalid native river element")
        missing = np.flatnonzero(nbe[cell - 1] == 0)
        require(len(missing) == 1, f"river element {cell} lacks one identifiable shoreline face")
        nodes = np.delete(tri[cell - 1], missing[0])
        h = depth[nodes - 1]
        require(np.all(np.isfinite(h) & (h > 0)), "dry receiving edge")
        edge.update(edge_nodes=nodes.tolist(), static_depth_m=h.tolist())


def build(series, control, source_grid, grid_manifest, out, prefix):
    grid = json.loads(grid_manifest.read_text())
    require(grid["status"] == "ready" and grid["geometry"] == "spherical" and
            grid["source_sha256"] == sha(source_grid), "river grid/source mismatch")
    for name in ("grd", "dep", "sig"):
        item = grid["files"][name]
        require(sha(grid_manifest.parent / item["name"]) == item["sha256"], f"altered runtime {name}")
    stations, edges = read_control(control)
    with netCDF4.Dataset(source_grid) as ds:
        ds.set_auto_mask(False)
        edge_faces(edges, ds["nv"][:].T, ds["nbe"][:].T, ds["h"][:])
    data = pd.read_csv(series, dtype={"gauge_id":str})
    data["epoch"] = pd.to_datetime(data.timestamp_utc, utc=True).to_numpy(dtype="datetime64[s]").astype("i8")
    require(set(data.gauge_id) == {station["id"] for station in stations.values()}, "gauges differ from control")
    require(not data.duplicated(["gauge_id", "epoch"]).any(), "duplicate gauge/time")
    clocks = np.sort(data.epoch.unique())
    require(len(clocks) >= 2 and np.all(np.diff(clocks) == clocks[1] - clocks[0]), "irregular river clock")
    series_by_station = {}
    for station in stations.values():
        frame = data[data.gauge_id == station["id"]].sort_values("epoch")
        require(np.array_equal(frame.epoch, clocks), f"incomplete times for {station['id']}")
        require(np.isfinite(frame.discharge_m3s).all() and (frame.discharge_m3s >= 0).all(),
                f"expected nonsigned gauge discharge at {station['id']}; signs belong to control")
        series_by_station[station["id"]] = frame
    q, t = [], []
    names = []
    budgets = {}
    for e in edges:
        qid = stations[e["flow_source"]]["id"]
        tid = stations[e["temperature_source"]]["id"]
        frame = series_by_station[qid]
        values = frame.discharge_m3s.to_numpy() * e["flow_scale"]
        temp = np.asarray(series_by_station[tid].temperature_degC) * e["temperature_scale"]
        require(e["flow_scale"] != 0 and e["temperature_scale"] > 0, "zero scale/negative temperature multiplier")
        if e["flow_scale"] < 0:
            temp = np.zeros(len(clocks))
        require(np.all(np.isfinite(temp) & (temp >= -2) & (temp <= 40)), "invalid inlet temperature")
        names.append(f"{qid}_e{e['cell_id']}")
        q.append(values)
        t.append(temp)
        e.update(flow_gauge=qid, temperature_gauge=tid, river_name=names[-1])
        item = budgets.setdefault(qid, dict(gauge=stations[e["flow_source"]]["name"],
                                   source_volume_m3=float(np.trapezoid(frame.discharge_m3s, clocks)),
                                   applied_scale=0.0, applied_volume_m3=0.0,
                                   temperature_source=sorted(set(series_by_station[tid].temperature_source))))
        item["applied_scale"] += e["flow_scale"]
        item["applied_volume_m3"] += float(np.trapezoid(values, clocks))
    flux, temp = np.array(q).T, np.array(t).T
    salt = np.zeros_like(flux)
    mjd = clocks / 86400 + 40587
    out.parent.mkdir(parents=True, exist_ok=True)
    require(not out.exists(), "output already exists; use a fresh output directory")
    with tempfile.TemporaryDirectory(dir=out.parent) as work:
        dst = Path(work) / "products"
        dst.mkdir()
        river_nc = dst / f"{prefix}_river.nc"
        river_nml = dst / f"{prefix}_river.nml"
        write_river_nc(river_nc, names, mjd, flux, temp, salt, info1="Observed freshwater river forcing",
                       info2="Signed native FVCOM edge sources and withdrawals", casename=prefix, precise_clock=True)
        write_river_nml(river_nml, names, [e["cell_id"] for e in edges], river_nc.name,
                        n_sigma=grid["sigma_layers"], named_distribution=True)
        with netCDF4.Dataset(river_nc) as ds:
            require(np.array_equal(ds["river_flux"][:], flux.astype('f4')) and
                    np.array_equal(ds["river_temp"][:], temp.astype('f4')) and
                    np.array_equal(ds["river_salt"][:], salt), "river forcing readback differs")
            actual_names = [s.strip() for s in netCDF4.chartostring(ds["river_names"][:])]
            require(actual_names == names and np.array_equal(ds["time"][:], mjd), "river names/clock mismatch")
            ms = ds["Itime"][:].astype('i8') * 86400000 + ds["Itime2"][:]
            require(np.array_equal(ms, (clocks + 40587 * 86400) * 1000), "native integer clock mismatch")
            actual_text = np.char.decode(netCDF4.chartostring(ds["Times"][:], encoding="bytes"), "ascii")
            require(np.array_equal(actual_text, np.datetime_as_string(clocks.astype("datetime64[s]"), unit="us")),
                    "native character clock mismatch")
            written_flux = ds["river_flux"][:].astype(float)
        read_nml = river_nml.read_text()
        require(re.findall(r"RIVER_NAME\s*=\s*'([^']+)'", read_nml) == names and
                list(map(int, re.findall(r"RIVER_GRID_LOCATION\s*=\s*(\d+)", read_nml))) ==
                [e["cell_id"] for e in edges], "river companion mismatch")
        for qid, frame in series_by_station.items():
            ix = [j for j,e in enumerate(edges) if e["flow_gauge"] == qid]
            require(np.allclose(written_flux[:, ix].sum(axis=1), frame.discharge_m3s * budgets[qid]["applied_scale"],
                                rtol=2e-7, atol=1e-6), f"failed source conservation at {qid}")
        result = {"schema":"fvcom_lake_signed_edge_river_v1", "status":"ready", "mesh_sha256":grid["files"]["grd"]["sha256"],
                  "source_grid_sha256":grid["source_sha256"], "preconfiguration_sha256":sha(grid_manifest),
                  "control_sha256":sha(control), "source_series_sha256":sha(series), "gauge_budgets":budgets,
                  "location":"edge", "positive_heat":"prescribed incoming freshwater temperature",
                  "negative_heat":"local receiving edge node/layer donor; requires signed-outflow reader; file placeholder 0 degC",
                  "outflow_heat_integration":"integrate with the simulated donor temperature",
                  "net_flux_range_m3s":[float(written_flux.sum(axis=1).min()),float(written_flux.sum(axis=1).max())],
                  "net_volume_m3": float(np.trapezoid(written_flux.sum(axis=1), clocks)), "entries":edges,
                  "count":len(clocks), "start_time":str(np.datetime64(int(clocks[0]), 's'))+"Z",
                  "end_time":str(np.datetime64(int(clocks[-1]), 's'))+"Z", "interval_seconds":int(np.diff(clocks)[0]),
                  "files":{p.name:sha(p) for p in (river_nml, river_nc)}}
        (dst / "river_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        dst.rename(out)
    return out / "river_manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["series", "control", "source-grid", "grid-manifest", "out"]:
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    print(build(**vars(parser.parse_args())))
