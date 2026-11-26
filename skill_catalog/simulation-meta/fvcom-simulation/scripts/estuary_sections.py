"""Sample fixed transects on an FVCOM mesh and measure the connected near-bed salt front."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from pyproj import Transformer
from shapely import STRtree, points, polygons


def numeric(var, key=slice(None)):
    return np.ma.asarray(var[key], dtype="float64").filled(np.nan)


def native_times(dataset):
    if "Times" in dataset.variables:
        strings = [
            row.tobytes().decode("ascii").strip("\x00 ").replace(" ", "T")
            for row in dataset["Times"][:]
        ]
        return np.asarray(strings, dtype="datetime64[ms]")
    days = np.asarray(dataset["Itime"][:], dtype="timedelta64[D]")
    millis = np.asarray(dataset["Itime2"][:], dtype="timedelta64[ms]")
    return np.datetime64("1858-11-17", "ms") + days + millis


def mesh(dataset, crs):
    lon, lat = numeric(dataset["lon"]), numeric(dataset["lat"])
    if crs:
        x, y = numeric(dataset["x"]), numeric(dataset["y"])
    else:
        if np.ptp(lon) == 0 or np.ptp(lat) == 0:
            raise ValueError("Cartesian model requires its actual projected CRS")
        x, y = Transformer.from_crs(4326, 32610, always_xy=True).transform(
            (lon + 180) % 360 - 180, lat
        )
    nv = np.asarray(dataset["nv"][:], dtype=np.int64).T - 1
    return np.column_stack([x, y]), nv


def weights_at(xy, triangles, query):
    vertices = xy[triangles]
    # Discard distant cells before constructing the spatial index.
    keep = np.flatnonzero(
        np.all(vertices.max(axis=1) >= query.min(axis=0), axis=1)
        & np.all(vertices.min(axis=1) <= query.max(axis=0), axis=1)
    )
    tree = STRtree(polygons(vertices[keep]))
    station, local = tree.query(points(query), predicate="intersects")
    cell = np.full(len(query), -1, dtype=int)
    for i, c in sorted(zip(station, keep[local]), reverse=True):
        cell[i] = c
    verts = triangles[np.maximum(cell, 0)]
    a, b, c = vertices[np.maximum(cell, 0)].transpose(1, 0, 2)
    mat = np.stack([b - a, c - a], axis=2)
    uv = np.linalg.solve(mat, (query - a)[:, :, None])[:, :, 0]
    coeff = np.column_stack([1 - uv.sum(axis=1), uv])
    coeff[cell < 0] = np.nan
    return cell, verts, coeff


def sample_vertical(z, s, target, bed, eta):
    """Linear in physical elevation; extend the endpoint layer only to the physical boundaries."""
    if (
        not np.isfinite(target)
        or not (bed <= target <= eta)
        or not np.all(np.isfinite(z))
        or not np.all(np.isfinite(s))
    ):
        return float("nan")
    if np.all(np.diff(z) < 0):
        z, s = z[::-1], s[::-1]
    if not np.all(np.diff(z) > 0):
        raise ValueError("Non-monotonic vertical layer positions")
    return float(np.interp(target, z, s))


def salt_front(distance, salt, threshold):
    """Return the first downstream-to-upstream transition with continuous valid support."""
    if len(distance) == 0 or np.any(np.diff(distance) <= 0):
        raise ValueError("Distances must increase from the mouth")
    if not np.isfinite(salt[0]):
        return float("nan"), "mouth_uncovered"
    if salt[0] < threshold:
        return 0.0, "mouth_below_threshold"
    for i in range(1, len(salt)):
        if not np.isfinite(salt[i]):
            return float(distance[i - 1]), "lower_bound_gap"
        if salt[i] < threshold:
            fraction = (threshold - salt[i - 1]) / (salt[i] - salt[i - 1])
            return float(
                distance[i - 1] + fraction * (distance[i] - distance[i - 1])
            ), "crossing"
    return float(distance[-1]), "lower_bound_end"


def topology_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sample(args):
    features = json.loads(args.channels.read_text())["features"]
    ll, channel, distance = [], [], []
    for f in features:
        coords, km = f["geometry"]["coordinates"], f["properties"]["distance_km"]
        if len(km) != len(coords) or np.any(np.diff(km) <= 0) or km[0] != 0:
            raise ValueError("Channel distance and geometry disagree")
        ll.extend(coords)
        distance.extend(km)
        channel.extend([f["properties"]["channel"]] * len(km))
    ll, channel, distance = np.array(ll), np.array(channel), np.array(distance)
    projected = args.crs or "EPSG:32610"
    target = np.column_stack(
        Transformer.from_crs(4326, projected, always_xy=True).transform(*ll.T)
    )
    results = {}
    identity = None
    for path in args.files:
        with Dataset(path) as d:
            xy, tri = mesh(d, args.crs)
            sigma, h = numeric(d["siglay"]), numeric(d["h"])
            signature = topology_digest(xy, tri, sigma, h)
            if identity is None:
                identity = signature
                cell, nodes, w = weights_at(xy, tri, target)
                unique, inverse = np.unique(nodes, return_inverse=True)
                inverse = inverse.reshape(nodes.shape)
                weights = w[None, :, :]
                nodal_h = h[unique][inverse]
                h_at = np.sum(nodal_h * w, axis=-1)
                if sigma.ndim == 1:
                    sigma = np.broadcast_to(sigma[:, None], (len(sigma), len(h)))
                sigma_nodes = sigma[:, unique][:, inverse]
            elif signature != identity:
                raise ValueError(f"Static mesh changed at {path}")
            for i, when in enumerate(native_times(d)):
                if when in results:
                    raise ValueError(f"Duplicate native clock {when}")
                eta = numeric(d["zeta"], (i, unique))[inverse]
                surface = np.sum(eta * w, axis=-1)
                elev = np.sum(
                    (eta[None, :, :] + sigma_nodes * (nodal_h + eta)[None, :, :])
                    * weights,
                    axis=-1,
                )
                sal = np.sum(
                    numeric(d["salinity"], (i, slice(None), unique))[:, inverse]
                    * weights,
                    axis=-1,
                )
                wet = (cell >= 0) & (
                    np.asarray(d["wet_cells"][i])[np.maximum(cell, 0)] == 1
                )
                wet &= np.all(
                    np.asarray(d["wet_nodes"][i, unique])[inverse] == 1, axis=1
                )
                sal[:, ~wet] = np.nan
                near = [
                    sample_vertical(z, s, bed + args.height, bed, top)
                    for z, s, bed, top in zip(elev.T, sal.T, -h_at, surface)
                ]
                results[when] = (elev, sal, near, surface)
    if not results:
        raise ValueError("No native records")
    times = sorted(results)
    z, s, near, eta = (np.asarray(a) for a in zip(*(results[t] for t in times)))
    args.output.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        args.output / "sections.npz",
        time=times,
        channel=channel,
        distance_km=distance,
        lonlat=ll,
        h=h_at,
        cell=cell + 1,
        weight=w,
        z=z,
        salt=s,
        near=near,
        eta=eta,
        height_m=args.height,
        mesh_hash=identity,
    )
    with (args.output / "fronts.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "channel", "threshold", "distance_km", "status"])
        for k, t in enumerate(times):
            for name in dict.fromkeys(channel):
                use = channel == name
                for threshold in args.threshold:
                    front, status = salt_front(distance[use], near[k, use], threshold)
                    writer.writerow([str(t), name, threshold, front, status])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files", nargs="+", type=Path, required=True)
    p.add_argument("--channels", type=Path, required=True)
    p.add_argument(
        "--crs",
        help="Projected model CRS when native x/y are metres; omit for geographic FVCOM",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--height", type=float, default=1.0)
    p.add_argument("--threshold", nargs="+", type=float, default=[1.0, 2.0])
    args = p.parse_args()
    if args.height <= 0:
        p.error("sampling height above bed must be positive")
    sample(args)


if __name__ == "__main__":
    main()
