"""Structured regional grid mechanics. No external mesher or compilation."""

from __future__ import annotations
import numpy as np
from pyproj import CRS, Transformer, Geod
from shapely.geometry import Polygon
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset

GEOD = Geod(ellps="WGS84")


def fit_domain(
    vertices, spacing=500.0, padding=2000.0, rotation=None, max_cells=1000000
):
    pts = np.asarray(vertices, float)
    if np.array_equal(pts[0], pts[-1]):
        pts = pts[:-1]
    center = pts.mean(axis=0)
    crs = CRS.from_proj4(
        f"+proj=tmerc +lat_0={center[1]} +lon_0={center[0]} +k=1 +datum=WGS84 +units=m"
    )
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    # Densify geographic edges so fitting preserves the full mission envelope.
    samples = np.concatenate(
        [
            a[None, :] + np.linspace(0, 1, 101)[:, None] * (b - a)
            for a, b in zip(pts, np.roll(pts, -1, axis=0))
        ]
    )
    xy = np.column_stack(forward.transform(samples[:, 0], samples[:, 1]))
    if not np.isfinite(xy).all() or np.ptp(xy, axis=0).max() > 1500000:
        raise ValueError(
            "v1 projected regional domain exceeds supported 1500 km extent"
        )
    if (
        not np.isfinite(spacing)
        or spacing <= 0
        or not np.isfinite(padding)
        or padding < 0
    ):
        raise ValueError("Spacing must be positive and padding nonnegative")
    if rotation is None:
        corners = np.asarray(Polygon(xy).minimum_rotated_rectangle.exterior.coords)[:4]
        edges = np.roll(corners, -1, axis=0) - corners
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        rotation = float(np.rad2deg(np.arctan2(edge[1], edge[0]))) % 180
        if rotation > 90:
            rotation -= 180
    if not np.isfinite(rotation):
        raise ValueError("Invalid rotation")
    a = np.deg2rad(rotation)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    uv = xy @ rot
    lower = np.floor((uv.min(axis=0) - padding) / spacing) * spacing
    upper = np.ceil((uv.max(axis=0) + padding) / spacing) * spacing
    nx, ny = np.rint((upper - lower) / spacing).astype(int)
    if min(nx, ny) < 4 or nx * ny > max_cells:
        raise ValueError("Grid dimensions below four cells or above cell budget")
    footprint = np.array(
        [
            [0, 0],
            [nx * spacing, 0],
            [nx * spacing, ny * spacing],
            [0, ny * spacing],
            [0, 0],
        ]
    )
    projected = (footprint + lower) @ rot.T
    lo, la = inverse.transform(projected[:, 0], projected[:, 1])
    return {
        "crs_wkt": crs.to_wkt(),
        "center_lonlat": center.tolist(),
        "rotation_deg": float(rotation),
        "origin_uv_m": lower.tolist(),
        "spacing_m": float(spacing),
        "nx": int(nx),
        "ny": int(ny),
        "footprint_lonlat": np.column_stack((lo, la)).tolist(),
        "padding_m": float(padding),
    }


def coordinates(fit):
    nx, ny = fit["nx"], fit["ny"]
    d = fit["spacing_m"]
    a = np.deg2rad(fit["rotation_deg"])
    inverse = Transformer.from_crs(CRS.from_wkt(fit["crs_wkt"]), 4326, always_xy=True)
    arrays = {}
    # rho halo is outside the physical domain; physical edges are psi endpoints.
    axes = {
        "rho": (np.arange(nx + 2) - 0.5, np.arange(ny + 2) - 0.5),
        "u": (np.arange(nx + 1), np.arange(ny + 2) - 0.5),
        "v": (np.arange(nx + 2) - 0.5, np.arange(ny + 1)),
        "psi": (np.arange(nx + 1), np.arange(ny + 1)),
    }
    for tag, (xx, yy) in axes.items():
        u, v = np.meshgrid(xx * d, yy * d)
        U = u + fit["origin_uv_m"][0]
        V = v + fit["origin_uv_m"][1]
        x = U * np.cos(a) - V * np.sin(a)
        y = U * np.sin(a) + V * np.cos(a)
        lon, lat = inverse.transform(x, y)
        arrays.update(
            {f"lon_{tag}": lon, f"lat_{tag}": lat, f"x_{tag}": u, f"y_{tag}": v}
        )
    lon = arrays["lon_rho"]
    lat = arrays["lat_rho"]
    # Distances between symmetric half-cell positions define cell-centered metrics.
    u = arrays["x_rho"] + fit["origin_uv_m"][0]
    v = arrays["y_rho"] + fit["origin_uv_m"][1]

    def at(du, dv):
        return inverse.transform(
            (u + du) * np.cos(a) - (v + dv) * np.sin(a),
            (u + du) * np.sin(a) + (v + dv) * np.cos(a),
        )

    lx, ly = at(-d / 2, 0)
    rx, ry = at(d / 2, 0)
    bx, by = at(0, -d / 2)
    tx, ty = at(0, d / 2)
    _, _, dx = GEOD.inv(lx, ly, rx, ry)
    _, _, dy = GEOD.inv(bx, by, tx, ty)
    azx, _, _ = GEOD.inv(lon, lat, rx, ry)
    azy, _, _ = GEOD.inv(lon, lat, tx, ty)
    error = np.abs(np.abs((azx - azy + 180) % 360 - 180) - 90)
    arrays.update(
        pm=1 / dx,
        pn=1 / dy,
        angle=(np.pi / 2 - np.deg2rad(azx) + np.pi) % (2 * np.pi) - np.pi,
        f=2 * 7.292115e-5 * np.sin(np.deg2rad(lat)),
        orthogonality_error_deg=error,
    )
    if not all(np.isfinite(v).all() for v in arrays.values()) or np.max(error) > 0.05:
        raise ValueError("Nonfinite geometry or excessive orthogonality error")
    return arrays


def sample_netcdf(path, variable, lon, lat, positive):
    if positive not in ("up", "down"):
        raise ValueError("Declare elevation sign as up or down")
    with Dataset(path) as ds:
        z = ds[variable]
        declared = getattr(z, "positive", positive).lower()
        if declared != positive:
            raise ValueError("Bathymetry sign conflicts with source metadata")
        if getattr(z, "units", "").lower() not in (
            "m",
            "meter",
            "meters",
            "metre",
            "metres",
        ):
            raise ValueError("Bathymetry must explicitly use metres")
        x = np.asarray(ds["lon"][:], float)
        y = np.asarray(ds["lat"][:], float)
        data = np.asarray(np.ma.filled(z[:], np.nan), float)
        if data.shape != (len(y), len(x)):
            raise ValueError("Expected a rectilinear (lat,lon) bathymetry field")
        if np.all(np.diff(x) < 0):
            x = x[::-1]
            data = data[:, ::-1]
        if np.all(np.diff(y) < 0):
            y = y[::-1]
            data = data[::-1, :]
        if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
            raise ValueError("Source axes are not monotone")
        zi = RegularGridInterpolator(
            (y, x), data, bounds_error=False, fill_value=np.nan
        )(np.column_stack((lat.ravel(), lon.ravel()))).reshape(lon.shape)
    return zi if positive == "up" else -zi


def connected_mask(candidate, anchors=()):
    labels, n = ndimage.label(
        candidate, structure=ndimage.generate_binary_structure(2, 1)
    )
    if not n:
        raise ValueError("No wet cells")
    if anchors:
        ids = [int(labels[j, i]) for j, i in anchors]
        if 0 in ids or len(set(ids)) != 1:
            raise ValueError(
                "Protected wet features are dry or disconnected; refine or revise using source evidence"
            )
        keep = ids[0]
    else:
        boundary = np.unique(
            np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        )
        boundary = boundary[boundary > 0]
        counts = np.bincount(labels.ravel())
        keep = (
            int(boundary[np.argmax(counts[boundary])])
            if len(boundary)
            else int(1 + np.argmax(counts[1:]))
        )
    final = labels == keep
    return final, {
        "components_before": int(n),
        "components_after": 1,
        "removed_cells": int(np.count_nonzero(candidate & ~final)),
        "method": "retain the component containing all protected wet features, otherwise largest boundary-connected component",
    }


def roughness(h, mask):
    vals = []
    for axis in (0, 1):
        a = np.take(h, range(h.shape[axis] - 1), axis=axis)
        b = np.take(h, range(1, h.shape[axis]), axis=axis)
        wet = np.take(mask, range(h.shape[axis] - 1), axis=axis) & np.take(
            mask, range(1, h.shape[axis]), axis=axis
        )
        vals.append(
            float(np.max(np.abs(a - b)[wet] / (a + b)[wet])) if wet.any() else 0.0
        )
    return max(vals)


def smooth_depth(h, mask, area, rmax=0.2, max_iterations=4000):
    if not 0 < rmax < 1:
        raise ValueError("rx0 target must lie between zero and one")
    h = np.array(h, float, copy=True)
    start = float(np.sum(h[mask] * area[mask]))
    q = (1 + rmax) / (1 - rmax)
    for iteration in range(max_iterations + 1):
        r = roughness(h, mask)
        if r <= rmax + 1e-8:
            return h, {
                "iterations": iteration,
                "rx0": r,
                "volume_relative_change": float(
                    (np.sum(h[mask] * area[mask]) - start) / start
                ),
            }
        # Disjoint red/black pairs; each redistribution preserves physical cell volume.
        for axis in (0, 1):
            for parity in (0, 1):
                left = [slice(None), slice(None)]
                right = left.copy()
                left[axis] = slice(parity, h.shape[axis] - 1, 2)
                right[axis] = slice(parity + 1, h.shape[axis], 2)
                left = tuple(left)
                right = tuple(right)
                a = h[left]
                b = h[right]
                A = area[left]
                B = area[right]
                bad = mask[left] & mask[right] & (np.abs(a - b) / (a + b) > rmax)
                total = A * a + B * b
                up = b > a
                na = np.where(up, total / (A + q * B), q * total / (B + q * A))
                nb = np.where(up, q * na, total / (B + q * A))
                a[bad] = na[bad]
                b[bad] = nb[bad]
    raise ValueError(
        f"Smoothing failed to reach rx0={rmax}; achieved {roughness(h,mask)}"
    )


def stretching(N, theta_s, theta_b):
    if (
        not np.isfinite([N, theta_s, theta_b]).all()
        or int(N) != N
        or N < 1
        or theta_s < 0
        or theta_b < 0
        or max(theta_s, theta_b) > 20
    ):
        raise ValueError("Invalid vertical stretching parameters")
    sr = (np.arange(N) + 0.5) / N - 1
    sw = np.arange(N + 1) / N - 1

    def curve(s):
        c = (
            (1 - np.cosh(theta_s * s)) / (np.cosh(theta_s) - 1)
            if theta_s > 0
            else -s * s
        )
        return np.expm1(theta_b * c) / (-np.expm1(-theta_b)) if theta_b > 0 else c

    return sr, sw, curve(sr), curve(sw)


def depths(h, N=40, theta_s=6.0, theta_b=2.0, hc=20.0, zeta=0.0):
    if not np.isfinite(hc) or hc < 0 or np.any(h <= 0) or not np.isfinite(h).all():
        raise ValueError("Invalid positive depths or hc")
    sr, sw, cr, cw = stretching(N, theta_s, theta_b)

    def z(s, c):
        return (
            zeta
            + (zeta + h)[None, ...]
            * (hc * s[:, None, None] + c[:, None, None] * h[None, ...])
            / (hc + h)[None, ...]
        )

    return z(sr, cr), z(sw, cw), (sr, sw, cr, cw)


def haney(z_w, mask):
    worst = 0.0
    for axis in (1, 2):
        a = np.take(z_w, range(z_w.shape[axis] - 1), axis=axis)
        b = np.take(z_w, range(1, z_w.shape[axis]), axis=axis)
        wet = np.take(mask, range(mask.shape[axis - 1] - 1), axis=axis - 1) & np.take(
            mask, range(1, mask.shape[axis - 1]), axis=axis - 1
        )
        den = np.diff(a, axis=0) + np.diff(b, axis=0)
        values = np.abs(a[1:] - b[1:] + a[:-1] - b[:-1]) / den
        if wet.any():
            worst = max(worst, float(np.max(values[:, wet])))
    return worst


def staggered_masks(mask):
    m = mask.astype("i4")
    u = m[:, :-1] * m[:, 1:]
    v = m[:-1] * m[1:]
    # Pinned REMORA calculate_nodal_masks: 1 for three/four wet neighbors,
    # 2 for exactly two adjacent wet neighbors, 0 for diagonal-only pairs.
    count = m[:-1, :-1] + m[1:, :-1] + m[:-1, 1:] + m[1:, 1:]
    adjacent = (
        m[:-1, :-1] * m[:-1, 1:]
        + m[1:, :-1] * m[1:, 1:]
        + m[:-1, :-1] * m[1:, :-1]
        + m[:-1, 1:] * m[1:, 1:]
    ) > 0
    psi = np.where(count >= 3, 1, np.where((count == 2) & adjacent, 2, 0)).astype("i4")
    return {"mask_rho": m, "mask_u": u, "mask_v": v, "mask_psi": psi}
