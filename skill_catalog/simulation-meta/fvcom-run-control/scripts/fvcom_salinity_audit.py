"""Read back native clocks, fixed temperature, active salt, stations and the MPI salt ledger."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import netCDF4 as nc
import numpy as np

from fvcom_salinity import STAGES, digest, load


def arr(var, region=slice(None)):
    v = np.ma.asarray(var[region])
    if np.any(np.ma.getmaskarray(v)) or not np.isfinite(v).all():
        raise ValueError("Incomplete/nonfinite data in " + var.name)
    return np.asarray(v)


def strings(var):
    return ["".join(x.astype("U1")).strip("\x00 ") for x in np.ma.filled(var[:], b" ")]


def clocks(ds, steps):
    if not np.array_equal(arr(ds["iint"]), steps):
        raise ValueError("Unexpected native IINT in " + ds.filepath())
    start = datetime(2025, 3, 18)
    if "Times" in ds.variables and [
        datetime.fromisoformat(x) for x in strings(ds["Times"])
    ] != [start + timedelta(seconds=int(i * 1.2)) for i in steps]:
        raise ValueError("Unexpected native UTC Times in " + ds.filepath())
    if "Itime" in ds.variables:
        absolute = (arr(ds["Itime"]).astype(np.int64) - 60752) * 86400000 + arr(
            ds["Itime2"]
        ).astype(np.int64)
        if not np.array_equal(absolute, steps * 1200):
            raise ValueError("Unexpected native integer UTC")
    else:
        reference = nc.date2num(start, ds["time"].units)
        native = arr(ds["time"])
        if not np.array_equal(
            native, (reference + steps * 1.2 / 86400).astype(native.dtype)
        ):
            raise ValueError(
                "Station date/IINT disagree with representable native time"
            )


def audit(args):
    a = Path(args.attempt)
    inputs = Path(args.inputs)
    m = load(a / "attempt_manifest.json")
    r = load(a / "slurm.json")
    job = r["job_id"]
    launched = load(a / "launch_identity.json")
    if (
        launched.get("site") != "perlmutter"
        or str(launched.get("job_id")) != job
        or launched.get("executable_sha256") != m["executable_sha256"]
        or launched.get("mpi_ranks") != 52
    ):
        raise ValueError("Mismatched launch identity")
    rows = [x.split("|") for x in (a / "accounting.psv").read_text().splitlines()]
    if len(
        [x for x in rows if x[:3] == [job, "COMPLETED", "0:0"]]
    ) != 1 or "TADA" not in (a / "stdout.log").read_text(errors="replace"):
        raise ValueError(
            "Missing successful parent Slurm row or native terminal marker"
        )
    if any(digest(inputs / k) != v for k, v in m["inputs"].items()):
        raise ValueError("Attempt input identity changed")
    if digest(a / "columbia_run.nml") != m["namelist_sha256"]:
        raise ValueError("Run namelist changed")
    products = {
        name: sha
        for sha, name in [
            x.split(None, 1)
            for x in (a / "native_files.sha256").read_text().splitlines()
        ]
    }
    used = [
        "launch_identity.json",
        "accounting.psv",
        "stdout.log",
        "stderr.log",
        "columbia_0001.nc",
        "columbia_station_timeseries.nc",
        "columbia_restart_0001.nc",
    ]
    ledgers = []
    for stem in ("salt_stage_rank_", "salt_boundary_rank_"):
        paths = sorted(a.glob(stem + "*.dat"))
        if len(paths) != 52 or {int(x.stem.split("_")[-1]) for x in paths} != set(
            range(1, 53)
        ):
            raise ValueError("Expected native ledger from each rank")
        ledgers.append(paths)
        used += [x.name for x in paths]
    if any(products.get(k) != digest(a / k) for k in used):
        raise ValueError("Closed native output hash mismatch")
    seconds = STAGES[m["stage"]]
    ngrid = 3600 if seconds == 86400 else 360
    grid_steps = np.rint(np.arange(0, seconds + 1, ngrid) / 1.2).astype(int)
    station_steps = np.rint(np.arange(0, seconds + 1, 360) / 1.2).astype(int)
    failures = []
    with (
        nc.Dataset(inputs / "columbia_start.nc") as init,
        nc.Dataset(inputs / "columbia_elevtide.nc") as tide,
        nc.Dataset(a / "columbia_0001.nc") as g,
        nc.Dataset(a / "columbia_station_timeseries.nc") as s,
        nc.Dataset(a / "columbia_restart_0001.nc") as rst,
    ):
        clocks(g, grid_steps)
        clocks(s, station_steps)
        clocks(rst, np.array([station_steps[-1]]))
        mapped = load(inputs / "station_mapping.json")["stations"]
        if strings(s["name_station"]) != [str(x["station_id"]) for x in mapped]:
            raise ValueError("Station names disagree with mapping")
        cells = np.array([x["cell_id"] for x in mapped]) - 1
        mesh = arr(g["nv"]).T - 1
        if (
            np.count_nonzero(cells == 19423) != 1
            or mapped[np.where(cells == 19423)[0][0]]["name"] != "SATURN03"
        ):
            raise ValueError("SATURN03 mapping")
        obc = arr(tide["obc_nodes"]) - 1
        t0 = arr(init["temp"], 0)
        s0 = arr(init["salinity"], 0)
        tt = (arr(tide["Itime"]).astype(np.int64) - 60752) * 86400 + arr(
            tide["Itime2"]
        ) / 1000
        hist = []
        z_error = []
        temp_error = 0.0
        salt_change = 0.0
        station_error = 0.0
        for i, step in enumerate(grid_steps):
            st_idx = int(step // 300)
            secs = step * 1.2
            it = int(np.searchsorted(tt, secs))
            if tt[it] != secs:
                raise ValueError("Missing retained forcing knot at native output UTC")
            wet = arr(g["wet_nodes"], i).astype(bool)
            wc = arr(g["wet_cells"], i)
            if not np.array_equal(arr(s["wet_cells"], st_idx), wc[cells]):
                failures.append("station_wet_mask")
            T = arr(g["temp"], i)
            S = arr(g["salinity"], i)
            temp_error = max(temp_error, float(np.abs(T - t0).max()))
            salt_change = max(salt_change, float(np.abs(S[:, wet] - s0[:, wet]).max()))
            for name in ("u", "v", "ww", "ua", "va", "zeta"):
                arr(g[name], i)
            h = arr(g["h"])[obc]
            expected = np.maximum(arr(tide["elevation"], it), 0.05 - h)
            z_error.append(float(np.max(abs(arr(g["zeta"], i)[obc] - expected))))
            salt_at_station = np.mean(S[:, mesh[cells]], axis=-1)
            station_error = max(
                station_error,
                float(np.max(abs(salt_at_station - arr(s["salinity"], st_idx)))),
            )
            neg = (S < 0) & wet[None, :]
            stat = np.where(wet[None, :], S, np.inf)
            layer, node = np.unravel_index(np.argmin(stat), S.shape)
            thickness = np.abs(np.diff(arr(g["siglev"]), axis=0))
            volumes = (
                thickness
                * arr(g["art1"])[None, :]
                * (arr(g["h"]) + arr(g["zeta"], i))[None, :]
            )
            hist.append(
                {
                    "seconds": int(secs),
                    "wet_nodes": int(wet.sum()),
                    "wet_cells": int(wc.sum()),
                    "wet_salinity_range": [
                        float(S[:, wet].min()),
                        float(S[:, wet].max()),
                    ],
                    "minimum_node_id": int(node + 1),
                    "minimum_layer": int(layer + 1),
                    "negative_wet_layers": int(neg.sum()),
                    "negative_wet_volume_m3": float(volumes[neg].sum(dtype=np.float64)),
                    "negative_salt_psu_m3": float(
                        (volumes[neg] * S[neg]).sum(dtype=np.float64)
                    ),
                    "obc_elevation_max_error_m": z_error[-1],
                }
            )
        for key in ("salinity", "temp", "u", "v", "ww", "zeta", "wet_cells"):
            arr(s[key])
        startup_salt_error = float(np.max(abs(arr(g["salinity"], 0) - s0)))
        for k in (
            "temp",
            "salinity",
            "zeta",
            "wet_nodes",
            "wet_cells",
            "u",
            "v",
            "ua",
            "va",
        ):
            if not np.array_equal(arr(rst[k], -1), arr(g[k], -1)):
                failures.append("terminal_restart_" + k)
    stage = np.stack([np.loadtxt(path, ndmin=2) for path in ledgers[0]])
    boundary = np.stack([np.loadtxt(path, ndmin=2) for path in ledgers[1]])
    for group in (stage, boundary):
        if (
            not np.isfinite(group).all()
            or not np.all(group[:, :, 0] == station_steps[None, 1:])
            or not np.all(group[:, :, 1] == station_steps[None, 1:])
        ):
            failures.append("ledger_clock_or_nonfinite")
    sums = stage[:, -1, :].sum(axis=0)
    b = boundary[:, -1, :].sum(axis=0)
    closure = abs(sum(sums[6:]) - (sums[5] - sums[4]))
    if closure > max(1e-5, abs(sums[4]) * 1e-9):
        failures.append("native_stage_closure")
    if max(z_error) > 1e-4:
        failures.append("open_boundary_elevation")
    if (
        temp_error > 1e-6
        or startup_salt_error > 1e-5
        or station_error > 1e-5
        or salt_change <= 0
    ):
        failures.append("tracer_readback")
    if b[11] <= 0 or b[12] != 0 or boundary[:, -1, -1].max() > 1e-8:
        failures.append("river_or_ocean_source")
    result = {
        "schema": "fvcom_estuary_salinity_native_audit_v1",
        "status": "verified" if not failures else "failed",
        "failed_checks": sorted(set(failures)),
        "stage": m["stage"],
        "job_id": job,
        "attempt_id": m["attempt_id"],
        "namelist_sha256": m["namelist_sha256"],
        "executable_sha256": m["executable_sha256"],
        "inputs": m["inputs"],
        "grid_records": len(grid_steps),
        "station_records": len(station_steps),
        "terminal_seconds": seconds,
        "station_count": len(mapped),
        "startup_salt_max_error": startup_salt_error,
        "fixed_temperature_max_error_C": temp_error,
        "max_salinity_change": salt_change,
        "native_station_interpolation_max_error": station_error,
        "open_boundary_max_error_m": max(z_error),
        "post_start_obc_max_error_m": max(z_error[1:]),
        "history": hist,
        "closed_output_sha256": products,
        "stage_salt_closure_psu_m3": float(closure),
        "river_volume_m3": float(b[11]),
        "river_salt_psu_m3": float(b[12]),
        "max_obc_inward_salinity_error": float(boundary[:, -1, -1].max()),
    }
    path = Path(args.output)
    raw = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != raw:
        raise FileExistsError(path)
    path.write_text(raw, newline="\n")
    return result
