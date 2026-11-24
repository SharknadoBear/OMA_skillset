"""Compare native OBC zeta with the staged type-1 elevation and FVCOM's cold-start ramp."""
import argparse
import json
import re
from pathlib import Path

import netCDF4 as nc
import numpy as np

from audit_fvcom_production import parse_namelist, sha256


def check_boundary(namelist, inputs, grid_paths):
    nml = parse_namelist(namelist)
    if nml.get("STARTUP_TYPE", "").lower() != "coldstart" or int(nml["ISPLIT"]) != 1:
        raise ValueError("this native boundary audit requires the Columbia cold start and ISPLIT=1")
    obc = np.loadtxt(inputs/nml["OBC_NODE_LIST_FILE"], skiprows=1, dtype=int, ndmin=2)
    if np.any(obc[:,2] != 1) or len(set(obc[:,1])) != len(obc):
        raise ValueError("expected unique prescribed type-1 boundary nodes")
    nodes = obc[:,1]
    path = inputs/nml["OBC_ELEVATION_FILE"]
    peak = 0.
    counts = {}
    with nc.Dataset(path) as forcing:
        if not np.array_equal(nodes, forcing["obc_nodes"][:]):
            raise ValueError("elevation file has a different ordered boundary")
        utc = forcing["Itime"][:].astype(np.int64)*86400000 + forcing["Itime2"][:]
        if np.any(np.diff(utc) <= 0):
            raise ValueError("forcing clock is not strictly increasing")
        for grid_path in grid_paths:
            with nc.Dataset(grid_path) as ds:
                axis = ds["Itime"][:].astype(np.int64)*86400000 + ds["Itime2"][:]
                idx = np.searchsorted(utc,axis)
                if np.any(idx >= len(utc)) or not np.array_equal(utc[idx],axis):
                    raise ValueError("saved grid time disagrees with the boundary forcing")
                for j, k in enumerate(idx):
                    actual = np.ma.asarray(ds["zeta"][j, nodes-1])
                    wet = np.ma.asarray(ds["wet_nodes"][j, nodes-1])
                    if np.ma.getmaskarray(wet).any() or not np.all(wet == 1):
                        raise ValueError("dry or masked Columbia ocean boundary node")
                    ramp = np.tanh(float(ds["iint"][j])/int(nml["IRAMP"])) if int(nml["IRAMP"]) else 1.
                    expected = np.ma.asarray(forcing["elevation"][k]) * ramp
                    error = np.ma.abs(actual-expected)
                    if np.ma.getmaskarray(error).any() or not np.isfinite(error).all():
                        raise ValueError("masked or non-finite OBC elevation")
                    peak = max(peak, float(error.max()))
                    counts[int(axis[j])] = len(nodes)
    if not counts or peak > 1.e-4:
        raise ValueError(f"native boundary differs from ramped prescribed file, max error {peak:g} m")
    return {"status":"passed","saved_times":len(counts),"ocean_nodes":len(nodes),
            "max_abs_error_m":peak,"tolerance_m":1.e-4,"forcing_sha256":sha256(path),
            "run_namelist_sha256":sha256(namelist),
            "native_grid_files":{p.name:sha256(p) for p in grid_paths}}


def check_readers(namelist, stdout):
    nml = parse_namelist(namelist)
    wind = nml["WIND_ON"].strip(".").upper() in {"T", "TRUE"}
    if wind:
        signatures = ["SETTING UP WIND STRESS FORCING FROM A 'FVCOM GRID' FILE",
                      "SETTING UP AIR PRESSURE FORCING FROM A 'FVCOM GRID' FILE"]
    else:
        signatures = ["SURFACE WIND FORCING IS OFF", "SETTING UP CONSTANT AIR PRESSURE FORCING"]
    log = stdout.read_bytes().replace(b"\0", b"").decode("utf-8", errors="replace").upper()
    normalized = re.sub(r"\s+", " ", log)
    if any(marker not in normalized for marker in signatures):
        raise ValueError(f"missing native forcing reader evidence in {stdout}")
    return {"stdout_sha256":sha256(stdout), "native_messages":signatures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namelist", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--grid", type=Path, action="append", required=True)
    parser.add_argument("--stdout", type=Path, help="Also verify the wind-pair native reader messages")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = check_boundary(args.namelist, args.inputs, args.grid)
    if args.stdout:
        evidence["readers"] = check_readers(args.namelist, args.stdout)
    with args.output.open("x", encoding="utf-8") as report:
        json.dump(evidence, report, indent=2)
