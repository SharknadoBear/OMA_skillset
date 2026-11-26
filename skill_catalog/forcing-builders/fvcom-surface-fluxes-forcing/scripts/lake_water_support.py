"""Select same-lake HRRR donors for surface energy and the associated vapour flux."""
import hashlib
import json

import netCDF4 as nc
import numpy as np
from pyproj import Transformer
from scipy.ndimage import label
from scipy.spatial import cKDTree

from prepare_wind_pressure import array, require
from surface_fluxes_core import sha256_file


def bracket(axis, targets):
    axis = np.asarray(axis, dtype=float)
    if np.all(np.diff(axis) < 0):
        return bracket(-axis, -targets)
    require(np.all(np.diff(axis) > 0), "Source axis must be strictly monotone")
    require(np.all((axis[0] <= targets) & (targets <= axis[-1])), "Model lies outside the source grid")
    left = np.clip(np.searchsorted(axis, targets, side="right") - 1, 0, len(axis) - 2)
    return left, (targets - axis[left]) / (axis[left + 1] - axis[left])


class LakeWaterSupport:
    def __init__(self, axes, nodes, water, seed, maximum_m):
        y, x = axes
        require(water.shape == (len(y), len(x)) and np.isfinite(maximum_m) and maximum_m > 0,
                "Invalid water mask extent or donor radius")
        sy, sx = (int(abs(axis - p).argmin()) for axis, p in zip(axes, seed))
        require(all(min(axis) <= p <= max(axis) for axis, p in zip(axes, seed)) and water[sy, sx],
                "Choose a seed inside a water pixel of the target lake")
        components, _ = label(water)  # Shared edges connect pixels.
        basin = components == components[sy, sx]
        j, dy = bracket(y, nodes[:, 0])
        i, dx = bracket(x, nodes[:, 1])
        idx = np.column_stack((j*len(x)+i, j*len(x)+i+1, (j+1)*len(x)+i, (j+1)*len(x)+i+1))
        w = np.column_stack(((1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx))
        same = basin.ravel()[idx]
        self.replace = np.flatnonzero(np.any((w > 0) & ~same, axis=1))
        self.index = idx[self.replace].copy()
        self.weights = (w*same)[self.replace]
        sums = self.weights.sum(axis=1)
        nearest = sums == 0
        self.weights[~nearest] /= sums[~nearest, None]
        if np.any(nearest):
            by, bx = np.nonzero(basin)
            donors = cKDTree(np.column_stack((y[by], x[bx])))
            _, pos = donors.query(nodes[self.replace[nearest]])
            self.index[nearest, 0] = by[pos]*len(x)+bx[pos]
            self.weights[nearest, 0] = 1.
        row, corner = np.nonzero(self.weights > 0)
        dyidx, dxidx = np.unravel_index(self.index[row, corner], basin.shape)
        target = nodes[self.replace[row]]
        distance = np.hypot(y[dyidx]-target[:, 0], x[dxidx]-target[:, 1])
        max_used = float(distance.max(initial=0.))
        require(max_used <= maximum_m and np.all(basin[dyidx, dxidx]), "Same-lake donor exceeds the approved support")
        self.info = dict(seed_yx=[sy, sx], source_component_pixels=int(basin.sum()), fully_supported=len(nodes)-len(sums),
                         renormalized=int((~nearest).sum()), nearest_same_lake=int(nearest.sum()),
                         maximum_used_distance_m=max_used, maximum_allowed_distance_m=maximum_m,
                         replacement_donors_over_6000m_nodes=int(len(np.unique(row[distance > 6000]))),
                         replacement_donors_over_9000m_nodes=int(len(np.unique(row[distance > 9000]))),
                         component_sha256=hashlib.sha256(basin.astype("u1").tobytes()).hexdigest(),
                         components="four_connected", fields="six direct surface heat components and derived vapour flux")

    def apply(self, field, bilinear):
        bilinear[self.replace] = np.sum(field.ravel()[self.index] * self.weights, axis=1)
        return bilinear


def from_checked_masks(args, atmosphere, axes, nodes):
    masks = getattr(args, "lake_water_mask", None)
    seed = getattr(args, "lake_seed_lonlat", None)
    radius = getattr(args, "max_water_distance_km", None)
    if not masks and seed is None and radius is None:
        return None
    require(masks and seed is not None and radius is not None, "Supply water mask, lake seed and donor radius together")
    checks = []
    water = None
    for path in masks:
        health = json.loads((path.parent/"health_check.json").read_bytes())
        digest = sha256_file(path)
        require(health["passed"] and any(c.get("sha256") == digest for c in health["checks"]), "Unchecked HRRR mask")
        with nc.Dataset(path) as ds:
            require(ds["crs"].crs_wkt == atmosphere["crs"].crs_wkt and
                    all(np.array_equal(array(ds[name]), axis) for name, axis in zip(("y", "x"), axes)),
                    "HRRR mask grid differs from the forcing")
            found = []
            for var in ds.variables.values():
                records = json.loads(getattr(var, "source_grib_metadata", "[]"))
                if records and all(r["idx_short_name"] == "LAND" and r["source_idx_level"] == "surface" and
                                   r["step_type"] == "instant" for r in records):
                    found.append((var, records))
            require(len(found) == 1, "Expected one native surface LAND field")
            var, records = found[0]
            require(var.dimensions == ("time", "y", "x") and len(records) == len(var), "Unmatched HRRR LAND records")
            for frame, record in zip(var[:], records):
                require(not np.ma.getmaskarray(frame).any() and np.isin(frame, [0, 1]).all(), "Nonbinary HRRR LAND")
                actual = np.asarray(frame) == 0
                require(water is None or np.array_equal(water, actual), "Water mask changed between source dates")
                water = actual
                checks.append(dict(valid_time=record["valid_time"], grib_sha256=record["message_sha256"], file_sha256=digest))
    px, py = Transformer.from_crs(4326, atmosphere["crs"].crs_wkt, always_xy=True).transform(*seed, errcheck=True)
    support = LakeWaterSupport(axes, nodes, water, (py, px), 1000.*radius)
    support.info.update(seed_lonlat=list(seed), source_masks=checks,
                        binary_water_sha256=hashlib.sha256(water.astype("u1").tobytes()).hexdigest())
    return support
