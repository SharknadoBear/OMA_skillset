"""Prepare a closed, native spherical FVCOM grid from a dated NOAA nowcast and its operational fix files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import netCDF4
import numpy as np
from pyproj import CRS, Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def demand(condition, message):
    if not condition:
        raise ValueError(message)


def topology(tri, neighbors, x, y):
    """Verify each native opposite-node neighbor and derive the closed shore loops."""
    count = len(tri)
    pairs = np.sort(tri[:, [[1, 2], [2, 0], [0, 1]]], axis=2).reshape(-1, 2)
    unique, inverse, uses = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    demand(np.max(uses) == 2, "non-manifold edge")
    cell = np.repeat(np.arange(1, count + 1), 3)
    owner_sum = np.bincount(inverse, weights=cell).astype(int)
    demand(np.array_equal(neighbors.ravel(), owner_sum[inverse] - cell), "native neighbor mismatch")
    edges = unique[uses == 1]
    adj = {}
    for a, b in edges.tolist():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    demand(all(len(ns) == 2 for ns in adj.values()), "shore is not a set of simple loops")
    loops = []
    while adj:
        start = next(iter(adj))
        last, point = None, start
        loop = []
        while point in adj:
            loop.append(point)
            ns = adj.pop(point)
            nxt = min(ns - {last})
            last, point = point, nxt
        demand(point == start, "shore loop did not close")
        loops.append(loop)
    ii, jj = (unique - 1).T
    graph = coo_matrix((np.ones(len(ii), dtype='i1'), (ii, jj)), shape=(len(x), len(x)))
    groups = connected_components(graph, directed=False, return_labels=False)
    demand(groups == 1, "disconnected lake mesh")
    demand(len(x) - len(unique) + count == 2 - len(loops), "Euler characteristic disagrees with shoreline")
    xx, yy = x[tri - 1], y[tri - 1]
    areas = ((xx[:, 1] - xx[:, 0]) * (yy[:, 2] - yy[:, 0]) -
             (xx[:, 2] - xx[:, 0]) * (yy[:, 1] - yy[:, 0])) / 2
    demand(np.all(areas < 0), "expected clockwise, nondegenerate native triangles")
    def loop_area(loop):
        xx, yy = x[np.asarray(loop) - 1], y[np.asarray(loop) - 1]
        return float(abs(np.dot(xx, np.roll(yy, 1)) - np.dot(yy, np.roll(xx, 1))) / 2)
    loops.sort(key=loop_area, reverse=True)
    return loops, {"connected_components": int(groups), "shoreline_edges": len(edges),
                   "shoreline_loops": len(loops), "islands": len(loops) - 1,
                   "clockwise_elements": count, "min_projected_cell_area_m2": float(-areas.max()),
                   "max_projected_cell_area_m2": float(-areas.min())}


def prepare(source, fix, out, prefix, counts, sms=None):
    out.mkdir(parents=True, exist_ok=True)
    native_grid = fix / f"{prefix}_grd.dat"
    dep_path = fix / f"{prefix}_dep.dat"
    sig_path = fix / f"{prefix}_sigma.dat"
    cor_path = fix / f"{prefix}_cor.dat"
    with netCDF4.Dataset(source) as ds:
        ds.set_auto_mask(False)
        demand(ds.CoordinateSystem == "GeoReferenced", "expected spherical NOAA FVCOM output")
        n, m, layers = (ds.dimensions[k].size for k in ("node", "nele", "siglay"))
        demand((n, m, layers) == tuple(counts), "wrong native node, element or layer count")
        tri = ds["nv"][:].T.copy()
        neighbor = ds["nbe"][:].T
        demand(tri.min() == 1 and tri.max() == n, "invalid one-based connectivity")
        h = ds["h"][:]
        demand(np.isfinite(h).all() and h.min() > 0, "nonpositive or invalid wet bathymetry")
        native_lon = ds["lon"][:].astype(float)
        lat = ds["lat"][:].astype(float)
        lon = (native_lon + 180) % 360 - 180
        x, y = (ds[k][:].astype(float) for k in ("x", "y"))
        crs = CRS.from_user_input(ds.CoordinateProjection)
        xt, yt = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
        projected_error = float(np.max(np.hypot(xt - x, yt - y)))
        demand(projected_error < 3, "published x/y disagree with projected coordinates")
        interface = ds["siglev"][:]
        middle = ds["siglay"][:]
        uniform = np.linspace(0, -1, layers + 1).astype('f4')
        demand(np.allclose(interface, uniform[:, None], atol=1e-7, rtol=0) and
               np.allclose(middle, (interface[1:] + interface[:-1]) / 2, atol=1e-7, rtol=0),
               "nonuniform native sigma is unsupported")
        native_area = ds["art1"][:]
        demand(native_area.shape == (n,) and np.all(np.isfinite(native_area) & (native_area > 0)),
               "invalid native node control areas")
        valid_time = str(netCDF4.num2date(ds["time"][0], ds["time"].units))

    with native_grid.open() as stream:
        demand(int(next(stream).split("=")[1]) == n and int(next(stream).split("=")[1]) == m, "grid headers differ")
        cells = np.loadtxt(stream, max_rows=m, dtype=int)
        points = np.loadtxt(stream)
    # READ_COLDSTART_CONNECTIVITY loads the ASCII N1,N2,N3 as NVG(I,:) = N1,N3,N2.
    demand(np.array_equal(cells[:, 0], np.arange(1, m + 1)) and np.array_equal(cells[:, [1, 3, 2]], tri),
           "fix connectivity differs from dated field")
    demand(np.array_equal(points[:, 0], np.arange(1, n + 1)), "fix node order differs")
    dep = np.loadtxt(dep_path, skiprows=1)
    cor = np.loadtxt(cor_path, skiprows=1)
    # Operational fix retains more longitude precision than the 0..360 float32 history file.
    demand(np.allclose(points[:, 1:3], dep[:, :2], atol=1e-6, rtol=0), "grid and depth coordinates differ")
    demand(np.allclose(dep[:, 0], lon, atol=2e-5, rtol=0) and np.allclose(dep[:, 1], lat, atol=5e-6, rtol=0),
           "operational fix node coordinates differ from dated field")
    demand(np.array_equal(dep[:, 2].astype('f4'), h), "operational fix bathymetry differs from dated field")
    demand(np.allclose(cor[:, :2], dep[:, :2], atol=1e-5, rtol=0) and
           np.allclose(cor[:, 2], lat, atol=1e-4, rtol=0), "Coriolis latitude differs")
    sigma_lines = [line.replace(" ", "").upper() for line in sig_path.read_text().splitlines()]
    demand(f"NUMBEROFSIGMALEVELS={layers+1}" in sigma_lines and
           "SIGMACOORDINATETYPE=UNIFORM" in sigma_lines, "source sigma fix does not match")
    checks = {}
    if sms:
        j = k = 0
        with sms.open() as stream:
            for line in stream:
                f = line.split()
                if not f:
                    continue
                if f[0] == "E3T":
                    demand(int(f[1]) == j + 1 and list(map(int, f[2:5])) == cells[j, 1:4].tolist(),
                           "published 2DM topology differs")
                    j += 1
                if f[0] == "ND":
                    demand(int(f[1]) == k + 1 and abs(float(f[2]) - dep[k, 0]) <= 2e-5 and
                           abs(float(f[3]) - dep[k, 1]) <= 5e-6, "published 2DM order/coordinates differ")
                    k += 1
        demand((k, j) == (n, m), "published 2DM size differs")
        checks["sms_2dm_sha256"] = digest(sms)
    loops, summary = topology(tri, neighbor, x, y)
    outputs = {}
    for key, src in {"grd": native_grid, "dep": dep_path, "cor": cor_path, "sig": sig_path}.items():
        target = out / f"{prefix}_{key}.dat"
        shutil.copyfile(src, target)
        outputs[key] = target
    for suffix, label in [("obc", "OBC"), ("spg", "Sponge")]:
        path = out / f"{prefix}_{suffix}.dat"
        path.write_bytes(f"{label} Node Number = 0\n".encode("ascii"))
        outputs[suffix] = path
    metrics = out / "native_horizontal_metrics.npz"
    np.savez_compressed(metrics, node_id=np.arange(1, n + 1, dtype='i4'), x_m=x, y_m=y,
                        lon=dep[:, 0], lat=dep[:, 1], node_control_area_m2=native_area)
    shore = out / "native_shorelines.json"
    shore.write_text(json.dumps({"node_numbering": "one-based native", "exterior": loops[0], "islands": loops[1:]}))
    outputs.update(metrics=metrics, shorelines=shore)
    summary.update(schema="fvcom_retained_spherical_grid_v1", status="ready", prefix=prefix,
                   nodes=n, elements=m, sigma_layers=layers, sigma_interfaces=layers+1, valid_time_utc=valid_time,
                   geometry="spherical", grid_file_units="degrees", auxiliary_projection=crs.to_string(),
                   ascii_to_native_nv_one_based=[1, 3, 2],
                   projected_roundtrip_error_m=projected_error, obc_nodes=0, sponge_nodes=0,
                   sigma_interface_values=uniform.tolist(), min_depth_m=float(h.min()), max_depth_m=float(h.max()),
                   nodal_control_area_sum_m2=float(native_area.sum(dtype=float)), source_sha256=digest(source),
                   files={k:{"name":p.name, "sha256":digest(p)} for k,p in outputs.items()}, **checks)
    manifest = out / f"{prefix}_fvcom_dat_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "fix", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--counts", nargs=3, type=int, required=True, metavar=("NODES", "CELLS", "LAYERS"))
    parser.add_argument("--sms", type=Path)
    print(prepare(**vars(parser.parse_args())))
