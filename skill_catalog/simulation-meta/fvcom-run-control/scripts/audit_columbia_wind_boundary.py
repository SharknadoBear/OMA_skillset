"""Compare native OBC zeta with the staged type-1 elevation and FVCOM's cold-start ramp."""
import argparse
import json
import re
from pathlib import Path

import netCDF4 as nc
import numpy as np

from audit_fvcom_production import parse_namelist, sha256


def finite(values, name):
    if np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
        raise ValueError(f"masked or non-finite {name}")
    return np.asarray(values)


def native_wet_state(ds, nodes, target, depth, minimum):
    native_depth = finite(ds["h"][nodes-1], "native boundary depth")
    if not np.array_equal(native_depth, depth):
        raise ValueError("native boundary depths differ from the staged depth file")
    nv = ds["nv"]
    triangles = finite(nv[:], "native triangle connectivity")
    if nv.dimensions == ("three", "nele"):
        triangles = triangles.T
    elif nv.dimensions != ("nele", "three"):
        raise ValueError("unexpected native triangle dimensions")
    cells = np.flatnonzero(np.isin(triangles, nodes).any(axis=1))
    incident = triangles[cells]
    cell_flags = finite(ds["wet_cells"][:, cells], "adjacent wet-cell flags")
    if not np.isin(cell_flags, [0, 1]).all():
        raise ValueError("invalid adjacent wet-cell flag")
    has_wet_cell = np.empty(target.shape, dtype=bool)
    for column, node in enumerate(nodes):
        adjacent = (incident == node).any(axis=1)
        if not adjacent.any():
            raise ValueError(f"boundary node {node} has no incident cells")
        has_wet_cell[:, column] = cell_flags[:, adjacent].any(axis=1)
    # WET_JUDGE classifies nodes before flooring ELF, in single precision.
    above = (depth + target.astype(np.float32) - minimum) >= np.float32(1.e-5)
    return above & has_wet_cell, np.maximum(target, minimum-depth)


def check_boundary(namelist, inputs, grid_paths):
    nml = parse_namelist(namelist)
    if nml.get("STARTUP_TYPE", "").lower() != "coldstart" or int(nml["ISPLIT"]) != 1:
        raise ValueError("this native boundary audit requires the Columbia cold start and ISPLIT=1")
    obc = np.loadtxt(inputs/nml["OBC_NODE_LIST_FILE"], skiprows=1, dtype=int, ndmin=2)
    if np.any(obc[:,2] != 1) or len(set(obc[:,1])) != len(obc):
        raise ValueError("expected unique prescribed type-1 boundary nodes")
    nodes = obc[:,1]
    path = inputs/nml["OBC_ELEVATION_FILE"]
    wetting = nml.get("WETTING_DRYING_ON", "F").strip(".").upper() in {"T", "TRUE"}
    minimum = np.float32(nml["MIN_DEPTH"]) if wetting else None
    if wetting:
        depth = np.loadtxt(inputs/nml["DEPTH_FILE"], skiprows=1, usecols=2, dtype=np.float32)[nodes-1]
        if not np.isfinite(depth).all() or not np.isfinite(minimum) or minimum <= 0:
            raise ValueError("invalid staged depth or MIN_DEPTH")
    dry_nodes = set()
    samples = {}
    peak = wet_peak = dry_peak = floor_peak = 0.
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
                actual = finite(ds["zeta"][:, nodes-1], "OBC elevation")
                flags = finite(ds["wet_nodes"][:, nodes-1], "boundary wet-node flags")
                steps = finite(ds["iint"][:], "IINT").astype(np.float64)
                iramp = int(nml["IRAMP"])
                ramp = np.tanh(steps/iramp)[:, None] if iramp else 1.
                target = finite(forcing["elevation"][idx,:], "prescribed elevation") * ramp
                if wetting:
                    wet, expected = native_wet_state(ds, nodes, target, depth, minimum)
                else:
                    wet, expected = np.ones(target.shape, dtype=bool), target
                if not np.array_equal(flags, wet):
                    raise ValueError("saved boundary wet-node flags disagree with the native depth and adjacent-cell rules")
                error = np.abs(actual-expected)
                peak = max(peak, float(error.max()))
                wet_peak = max(wet_peak, float(error[wet].max(initial=0.)))
                dry_peak = max(dry_peak, float(error[~wet].max(initial=0.)))
                floor_peak = max(floor_peak, float((expected-target).max()))
                dry_nodes.update(nodes[(~wet).any(axis=0)].tolist())
                for j, stamp in enumerate(axis):
                    samples[int(stamp)] = (int((~wet[j]).sum()), int((expected[j]>target[j]).sum()))
    if not samples or peak > 1.e-4:
        raise ValueError(f"native boundary differs from the expected saved elevation, max error {peak:g} m")
    return {"status":"passed","saved_times":len(samples),"ocean_nodes":len(nodes),
            "checked_node_samples":len(samples)*len(nodes),"minimum_depth_m":float(nml["MIN_DEPTH"]) if wetting else None,
            "dry_node_samples":sum(row[0] for row in samples.values()),"dry_node_ids":sorted(dry_nodes),
            "floor_limited_node_samples":sum(row[1] for row in samples.values()),"max_floor_adjustment_m":floor_peak,
            "max_abs_wet_error_m":wet_peak,"max_abs_dry_error_m":dry_peak if dry_nodes else None,
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
