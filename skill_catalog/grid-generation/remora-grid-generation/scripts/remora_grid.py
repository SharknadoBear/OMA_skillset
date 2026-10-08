#!/usr/bin/env python3
"""Build, inspect and validate single-level REMORA grid packages."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import box
from scipy.spatial import cKDTree
from netCDF4 import Dataset

REGION_SCRIPTS = Path(__file__).resolve().parents[2] / "remora-region-bpoly" / "scripts"
if not REGION_SCRIPTS.is_dir():
    raise RuntimeError("Install remora-region-bpoly beside this skill")
sys.path.insert(0, str(REGION_SCRIPTS))
from remora_regions import (
    read_json,
    write_json,
    sha,
    resolve,
    file_record,
    polygon,
    land_frame,
    setup_map,
    validate_delivery as validate_region,
    build as build_regions,
)
from grid_math import (
    fit_domain,
    coordinates,
    sample_netcdf,
    connected_mask,
    roughness,
    smooth_depth,
    depths,
    haney,
    staggered_masks,
    GEOD,
)


def parameters(req):
    p = dict(
        spacing_m=500.0,
        padding_m=2000.0,
        hmin_m=2.0,
        rx0_max=0.2,
        N=40,
        theta_s=6.0,
        theta_b=2.0,
        hc_m=20.0,
        max_cells=1000000,
    )
    p.update(req.get("parameters", {}))
    if not np.isfinite(p["hmin_m"]) or p["hmin_m"] <= 0:
        raise ValueError("hmin must be positive")
    return p


def load_request(path):
    path = Path(path).resolve()
    req = read_json(path)
    if "region_delivery" not in req:
        raise ValueError(
            "Run prepare for a complete region request, then inspect and review its returned package"
        )
    region_path = resolve(path.parent, req["region_delivery"])
    rd, rp = validate_region(region_path, True)
    roots = [r for r in rp["regions"] if r.get("parent_id") is None]
    if "region_id" not in req and len(roots) != 1:
        raise ValueError("Select region_id for multi-root plans")
    rid = req.get("region_id", roots[0]["id"] if roots else None)
    region = next((r for r in rp["regions"] if r["id"] == rid), None)
    if region is None:
        raise ValueError("Unknown region_id")
    if region.get("parent_id"):
        raise ValueError(
            "v1 generates a root grid only; nested numerical grids are deferred"
        )
    return req, parameters(req), region_path, rd, rp, region


def prepare(path, out):
    path = Path(path).resolve()
    req = read_json(path)
    out = Path(out).resolve()
    if "region_delivery" in req:
        return resolve(path.parent, req["region_delivery"])
    if "region_request" not in req:
        raise ValueError("Supply region_delivery or region_request")
    target = out / "01_regions" / "region_delivery.json"
    if target.exists():
        existing, _ = validate_region(target)
        if existing.get("request", {}).get("sha256") != sha(
            resolve(path.parent, req["region_request"])
        ):
            raise ValueError(
                "Supplied region request changed; use a new case directory"
            )
    else:
        build_regions(resolve(path.parent, req["region_request"]), target.parent)
    req["region_delivery"] = str(target)
    # Source paths retain their original meaning in the resumed request.
    for section in ["bathymetry"]:
        for key in ["path", "manifest"]:
            if key in req.get(section, {}):
                req[section][key] = str(resolve(path.parent, req[section][key]))
    write_json(out / "grid_request.json", req)
    return target


def fit(path, out):
    path = Path(path).resolve()
    out = Path(out).resolve()
    req, p, rpath, rd, rp, region = load_request(path)
    if (out / "grid_fit.json").exists():
        raise ValueError("Use a new fit directory for changed geometry")
    geom = fit_domain(
        region["vertices_lonlat"],
        p["spacing_m"],
        p["padding_m"],
        p.get("rotation_deg"),
        p["max_cells"],
    )
    a = coordinates(geom)
    coast = resolve(rpath.parent, rd["source"]["path"])
    bounds = [
        float(a["lon_rho"].min()),
        float(a["lat_rho"].min()),
        float(a["lon_rho"].max()),
        float(a["lat_rho"].max()),
    ]
    if not box(*rd["source"]["bbox_wsen"]).covers(box(*bounds)):
        raise ValueError(
            "Fetch a larger coastline footprint including the fitted grid halo"
        )
    land = land_frame(coast)
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    setup_map(
        ax,
        land,
        [bounds[0] - 0.03, bounds[1] - 0.03, bounds[2] + 0.03, bounds[3] + 0.03],
    )
    for r in rp["regions"]:
        x, y = polygon(r["vertices_lonlat"]).exterior.xy
        ax.plot(x, y, lw=1.5, label=r["id"])
    lo = a["lon_psi"]
    la = a["lat_psi"]
    step = max(1, min(geom["nx"], geom["ny"]) // 22)
    for j in range(0, lo.shape[0], step):
        ax.plot(lo[j], la[j], color="#486176", lw=0.35, alpha=0.6)
    for i in range(0, lo.shape[1], step):
        ax.plot(lo[:, i], la[:, i], color="#486176", lw=0.35, alpha=0.6)
    f = np.asarray(geom["footprint_lonlat"])
    ax.plot(f[:, 0], f[:, 1], "r-", lw=2, label="Physical grid footprint")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"Fitted grid: {geom['nx']} x {geom['ny']} cells, {p['spacing_m']:g} m\nGeographic regions remain separate from the numerical footprint"
    )
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "grid_fit.png", dpi=150)
    plt.close(fig)
    doc = {
        "schema": "remora_grid_fit_v1",
        "geometry": geom,
        "parameters": p,
        "request": file_record(path, out),
        "region_delivery": file_record(rpath, out),
        "region_review": file_record(resolve(rpath.parent, rd["review"]), out),
        "region_id": region["id"],
        "maps": [file_record(out / "grid_fit.png", out)],
    }
    write_json(out / "grid_fit.json", doc)
    return out / "grid_fit.json"


def check_fit(path, reviewed=True):
    path = Path(path).resolve()
    f = read_json(path)
    if f.get("schema") != "remora_grid_fit_v1":
        raise ValueError("Unsupported grid fit")
    for rec in [f["request"], f["region_delivery"], f["region_review"], *f["maps"]]:
        if sha(resolve(path.parent, rec["path"])) != rec["sha256"]:
            raise ValueError("Fit input or map changed")
    validate_region(resolve(path.parent, f["region_delivery"]["path"]), True)
    if reviewed:
        r = read_json(path.parent / "fit_review.json")
        if r.get("decision") != "accepted" or r.get("artifact_sha256") != sha(path):
            raise ValueError("Current visual fit review required")
    return f


def write_review(path, decision, rationale, kind):
    path = Path(path).resolve()
    d = check_fit(path, False) if kind == "fit" else validate_grid(path, False)
    if not rationale.strip():
        raise ValueError("Physical rationale required")
    review = {
        "decision": decision,
        "rationale": rationale,
        "reviewer": "agent",
        "artifact_sha256": sha(path),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "inspected_maps": d["maps"],
    }
    write_json(
        path.parent / ("fit_review.json" if kind == "fit" else "grid_review.json"),
        review,
    )


def anchor_indices(a, mask, features, spacing):
    jj, ii = np.where(mask)
    lon = a["lon_rho"]
    lat = a["lat_rho"]
    scale = np.cos(np.deg2rad(lat.mean()))
    tree = cKDTree(np.column_stack((lon[mask] * scale, lat[mask])))
    result = []
    records = []
    for f in features:
        x, y = f["point_lonlat"]
        _, k = tree.query([x * scale, y])
        j, i = int(jj[k]), int(ii[k])
        _, _, distance = GEOD.inv(x, y, float(lon[j, i]), float(lat[j, i]))
        if distance > f.get("max_snap_m", spacing * 1.5):
            raise ValueError("Required wet feature unsupported: " + f["name"])
        result.append((j, i))
        records.append({"name": f["name"], "j": j, "i": i, "snap_distance_m": distance})
    return result, records


def export_grid(path, a, h, mask, p, fitdoc, raw):
    ny, nx = h.shape
    sr, sw, cr, cw = depths(h, p["N"], p["theta_s"], p["theta_b"], p["hc_m"])[2]
    dims = {
        "eta_rho": ny,
        "xi_rho": nx,
        "eta_u": ny,
        "xi_u": nx - 1,
        "eta_v": ny - 1,
        "xi_v": nx,
        "eta_psi": ny - 1,
        "xi_psi": nx - 1,
        "s_rho": p["N"],
        "s_w": p["N"] + 1,
    }
    with Dataset(path, "w", format="NETCDF3_64BIT_DATA") as ds:
        for name, n in dims.items():
            ds.createDimension(name, n)
        ds.title = "REMORA regional grid"
        ds.Conventions = "CF-1.8"
        ds.grid_schema = "remora_grid_v1"
        ds.projection_wkt = fitdoc["geometry"]["crs_wkt"]
        ds.coordinate_note = "x/y are logical projected-grid metres; lon/lat describe geographic location"
        ds.numerical_levels = 1
        ds.source_policy = "Real source data; no missing wet bathymetry substitution"
        for key, val in p.items():
            if isinstance(val, (int, float)):
                ds.setncattr(key, val)
        for tag in ["rho", "u", "v", "psi"]:
            for prefix in ["lon", "lat", "x", "y"]:
                name = f"{prefix}_{tag}"
                v = ds.createVariable(name, "f8", (f"eta_{tag}", f"xi_{tag}"))
                v[:] = a[name]
                v.units = (
                    "degrees_east"
                    if prefix == "lon"
                    else "degrees_north" if prefix == "lat" else "m"
                )
        for name, data in {
            **{k: a[k] for k in ["pm", "pn", "angle", "f"]},
            "h": h,
            "hraw": raw,
        }.items():
            v = ds.createVariable(name, "f8", ("eta_rho", "xi_rho"))
            v[:] = data
            v.units = (
                "m-1"
                if name in ("pm", "pn")
                else "radian" if name == "angle" else "s-1" if name == "f" else "m"
            )
            if name in ("h", "hraw"):
                v.positive = "down"
        for name, data in staggered_masks(mask).items():
            tag = name[5:]
            v = ds.createVariable(name, "i4", (f"eta_{tag}", f"xi_{tag}"))
            v[:] = data
        for name, data, dim in [
            ("s_rho", sr, "s_rho"),
            ("s_w", sw, "s_w"),
            ("Cs_r", cr, "s_rho"),
            ("Cs_w", cw, "s_w"),
        ]:
            ds.createVariable(name, "f8", (dim,))[:] = data
        for name, value in [
            ("theta_s", p["theta_s"]),
            ("theta_b", p["theta_b"]),
            ("hc", p["hc_m"]),
            ("Vtransform", 2),
            ("Vstretching", 4),
            ("spherical", 1),
            ("xl", fitdoc["geometry"]["nx"] * p["spacing_m"]),
            ("el", fitdoc["geometry"]["ny"] * p["spacing_m"]),
        ]:
            ds.createVariable(name, "f8", ())[...] = value


def diagnostic_maps(out, a, h, raw, mask, initial, land, features, z_w):
    out = Path(out)
    paths = []
    lo = a["lon_rho"]
    la = a["lat_rho"]
    full = [lo.min(), la.min(), lo.max(), la.max()]
    views = [("overview", full)]
    for k, f in enumerate(features):
        x, y = f["point_lonlat"]
        dy = f.get("review_radius_km", 12) / 111.32
        dx = dy / np.cos(np.deg2rad(y))
        views.append((f"feature_{k}", [x - dx, y - dy, x + dx, y + dy]))
    for name, bounds in views:
        fig, axs = plt.subplots(1, 3, figsize=(17, 8), layout="constrained")
        panels = [
            (initial.astype(float), "Initial wet mask", "Blues", 0, 1),
            (
                np.where(mask, h, np.nan),
                "Final bathymetry (m)",
                "viridis",
                0,
                min(60, float(h[mask].max())),
            ),
            (
                np.where(mask, h - raw, np.nan),
                "Smoothing change (m)",
                "RdBu_r",
                -max(1, float(np.max(np.abs(h[mask] - raw[mask])))),
                max(1, float(np.max(np.abs(h[mask] - raw[mask])))),
            ),
        ]
        for ax, (data, title, cmap, vmin, vmax) in zip(axs, panels):
            setup_map(ax, land, bounds)
            mesh = ax.pcolormesh(
                lo,
                la,
                data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
                zorder=2,
            )
            local_land = land.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]]
            if not local_land.empty:
                local_land.boundary.plot(ax=ax, color="black", linewidth=0.3, zorder=3)
            changed = initial & ~mask
            if changed.any():
                ax.scatter(
                    lo[changed],
                    la[changed],
                    s=4,
                    c="#f000b0",
                    label="Closed disconnected cells",
                    zorder=4,
                )
            ax.set_title(title)
            fig.colorbar(mesh, ax=ax, shrink=0.6)
        fig.suptitle(
            "Mask and bathymetry review: "
            + name
            + "\nMagenta: disconnected water cells changed to dry"
        )
        p = out / (name + "_mask_bathy.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for ax, (k, title) in zip(
        axs,
        [
            ("pm", "Cell width in xi (m)"),
            ("orthogonality_error_deg", "Orthogonality deviation (degrees)"),
        ],
    ):
        setup_map(ax, land, full)
        value = 1 / a[k] if k == "pm" else a[k]
        mesh = ax.pcolormesh(lo, la, value, shading="nearest")
        fig.colorbar(mesh, ax=ax)
        ax.set_title(title)
    p = out / "geometry_quality.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    # Sections through the wettest row/column; dry segments are broken.
    j = int(np.argmax(mask.sum(axis=1)))
    i = int(np.argmax(mask.sum(axis=0)))
    for ax, section, wet, x, title in [
        (axs[0], z_w[:, j, :], mask[j], a["x_rho"][j] / 1000, f"eta={j}"),
        (axs[1], z_w[:, :, i], mask[:, i], a["y_rho"][:, i] / 1000, f"xi={i}"),
    ]:
        for k in range(0, len(z_w), max(1, (len(z_w) - 1) // 20)):
            ax.plot(x, np.where(wet, section[k], np.nan), color="#13688c", lw=0.6)
        ax.set(
            title="Vertical interfaces: " + title,
            xlabel="Logical distance (km)",
            ylabel="z (m)",
        )
        ax.grid(alpha=0.2)
    p = out / "vertical_sections.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def build(fit_path, out):
    fit_path = Path(fit_path).resolve()
    f = check_fit(fit_path)
    out = Path(out).resolve()
    if (out / "grid_delivery.json").exists():
        raise ValueError("Use a new build directory for a changed grid")
    request_path = resolve(fit_path.parent, f["request"]["path"])
    req, p, rpath, rd, rp, region = load_request(request_path)
    a = coordinates(f["geometry"])
    landpath = resolve(rpath.parent, rd["source"]["path"])
    land = land_frame(landpath)
    shore_water = ~shapely.intersects_xy(
        land.geometry.union_all(), a["lon_rho"], a["lat_rho"]
    )
    bathy = req["bathymetry"]
    bp = resolve(request_path.parent, bathy["path"])
    elev = sample_netcdf(
        bp,
        bathy.get("variable", "elevation_m"),
        a["lon_rho"],
        a["lat_rho"],
        bathy["positive"],
    )
    missing = shore_water & ~np.isfinite(elev)
    if missing.any():
        raise ValueError(
            f"{missing.sum()} coastline-water cells lack bathymetry; fetch source coverage"
        )
    initial = shore_water & (elev < 0)
    if not initial.any():
        raise ValueError("No wet cells after coastline/bathymetry classification")
    anchors, anchor_report = anchor_indices(
        a, initial, req.get("protected_wet_features", []), p["spacing_m"]
    )
    mask, corrections = connected_mask(initial, anchors)
    raw = np.where(mask, np.maximum(-elev, p["hmin_m"]), p["hmin_m"])
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite classified depth")
    area = 1 / (a["pm"] * a["pn"])
    h, smoothing = smooth_depth(raw, mask, area, p["rx0_max"])
    z_r, z_w, _ = depths(h, p["N"], p["theta_s"], p["theta_b"], p["hc_m"])
    if np.any(np.diff(z_w, axis=0) <= 0):
        raise ValueError("Nonpositive vertical layer thickness")
    out.mkdir(parents=True, exist_ok=True)
    mapsdir = out / "maps"
    mapsdir.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "mask_changes.npz",
        j=np.where(initial != mask)[0],
        i=np.where(initial != mask)[1],
        old=initial[initial != mask],
        new=mask[initial != mask],
    )
    corrections["protected_features"] = anchor_report
    corrections["sign_policy"] = (
        "wet only if outside shoreline land and sampled elevation < 0"
    )
    write_json(out / "mask_edits.json", corrections)
    grid = out / "remora_grid.nc"
    export_grid(grid, a, h, mask, p, f, raw)
    maps = diagnostic_maps(
        mapsdir,
        a,
        h,
        raw,
        mask,
        initial,
        land,
        req.get("protected_wet_features", []),
        z_w,
    )
    report = readback(grid)
    core = mask.copy()
    core[[0, -1], :] = False
    core[:, [0, -1]] = False
    smoothing["volume_domain"] = "all rho cells including the one-cell halo"
    smoothing["physical_volume_relative_change"] = float(
        np.sum((h - raw)[core] * area[core]) / np.sum(raw[core] * area[core])
    )
    report.update(
        smoothing=smoothing,
        haney_rx1_diagnostic=haney(z_w, mask),
        maximum_depth_change_m=float(np.max(np.abs(h[mask] - raw[mask]))),
        rms_depth_change_m=float(np.sqrt(np.mean((h[mask] - raw[mask]) ** 2))),
        wet_cells=int(mask.sum()),
        physical_wet_cells=int(mask[1:-1, 1:-1].sum()),
        orthogonality_error_max_deg=float(a["orthogonality_error_deg"].max()),
        source_vertical_datum=bathy.get("vertical_datum", "unspecified"),
        warnings=list(bathy.get("warnings", [])),
    )
    report["edge_wet_cells"] = {
        k: int(v.sum())
        for k, v in {
            "xlo": mask[1:-1, 1],
            "xhi": mask[1:-1, -2],
            "ylo": mask[1, 1:-1],
            "yhi": mask[-2, 1:-1],
        }.items()
    }
    # Grid direction names are explicit. Production boundary conditions are not inferred from FVCOM sides.
    report["boundary_roles"] = {
        k: ("water_boundary_requires_forcing" if n else "land")
        for k, n in report["edge_wet_cells"].items()
    }
    if corrections["removed_cells"]:
        report["warnings"].append(
            f"Closed {corrections['removed_cells']} disconnected water cells; inspect lagoon and tributary consequences before scientific use."
        )
    write_json(out / "validation.json", report)
    nx, ny = f["geometry"]["nx"], f["geometry"]["ny"]
    snippet = f"""# Grid-only snippet; merge with a complete REMORA simulation configuration.
remora.n_cell = {nx} {ny} {p['N']}
remora.prob_lo = 0 0 {-float(h.max()):.8g}
remora.prob_hi = {nx*p['spacing_m']:.8g} {ny*p['spacing_m']:.8g} 0
remora.use_curvilinear_grid = true
remora.nc_grid_file_0 = remora_grid.nc
remora.ic_type = netcdf
remora.mask_type = netcdf
remora.coriolis_type = netcdf
remora.theta_s = {p['theta_s']}
remora.theta_b = {p['theta_b']}
remora.tcline = {p['hc_m']}
amr.max_level = 0
"""
    (out / "inputs.grid").write_text(snippet, encoding="utf-8")
    sources = {
        "bathymetry": file_record(bp, out),
        "coastline": file_record(landpath, out),
    }
    if bathy.get("manifest"):
        sources["bathymetry_manifest"] = file_record(
            resolve(request_path.parent, bathy["manifest"]), out
        )
    doc = {
        "schema": "remora_grid_delivery_v1",
        "grid": file_record(grid, out),
        "fit": file_record(fit_path, out),
        "fit_review": file_record(fit_path.parent / "fit_review.json", out),
        "sources": sources,
        "validation": file_record(out / "validation.json", out),
        "mask_edits": file_record(out / "mask_edits.json", out),
        "mask_changes": file_record(out / "mask_changes.npz", out),
        "inputs": file_record(out / "inputs.grid", out),
        "maps": [file_record(x, out) for x in maps],
        "numerical_levels": 1,
        "reader_check": "reader_check.json",
    }
    doc["software"] = {
        "scripts_sha256": {x.name: sha(x) for x in Path(__file__).parent.glob("*.py")},
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
    }
    write_json(out / "grid_delivery.json", doc)
    return out / "grid_delivery.json"


def readback(path):
    with Dataset(path) as d:
        if d.data_model != "NETCDF3_64BIT_DATA":
            raise ValueError("Expected CDF5 output")
        h = np.asarray(d["h"][:])
        m = np.asarray(d["mask_rho"][:])
        ny, nx = h.shape
        if not np.isin(m, [0, 1]).all():
            raise ValueError("Nonbinary rho mask")
        shapes = {
            "rho": (ny, nx),
            "u": (ny, nx - 1),
            "v": (ny - 1, nx),
            "psi": (ny - 1, nx - 1),
        }
        for tag, shape in shapes.items():
            for prefix in ["lon", "lat", "x", "y", "mask"]:
                v = d[f"{prefix}_{tag}"]
                if (
                    v.shape != shape
                    or np.ma.getmaskarray(v[:]).any()
                    or not np.isfinite(v[:]).all()
                ):
                    raise ValueError("Invalid staggered field " + v.name)
        for name in ["h", "pm", "pn", "angle", "f"]:
            v = d[name][:]
            if (
                v.shape != h.shape
                or np.ma.getmaskarray(v).any()
                or not np.isfinite(v).all()
            ):
                raise ValueError("Invalid field " + name)
            if name in ("h", "pm", "pn") and np.any(v <= 0):
                raise ValueError("Nonpositive " + name)
        for name, expected in staggered_masks(m.astype(bool)).items():
            if not np.array_equal(d[name][:], expected):
                raise ValueError("Inconsistent mask " + name)
        r = roughness(h, m.astype(bool))
        if r > d.rx0_max + 2e-8:
            raise ValueError("Excessive bathymetric roughness")
        zr, zw, _ = depths(h, int(d.N), d.theta_s, d.theta_b, d.hc_m)
        if (
            not np.allclose(zw[0], -h)
            or not np.allclose(zw[-1], 0)
            or np.any(np.diff(zw, axis=0) <= 0)
        ):
            raise ValueError("Invalid vertical transform")
        return {
            "local_validation": "pass",
            "format": d.data_model,
            "rho_shape": [ny, nx],
            "n_cell": [nx - 2, ny - 2, int(d.N)],
            "rx0": r,
            "minimum_layer_thickness_m": float(np.min(np.diff(zw, axis=0))),
            "reader_validation": "not_run",
        }


def validate_grid(path, reviewed=False, reader=False):
    path = Path(path).resolve()
    d = read_json(path)
    if d.get("schema") != "remora_grid_delivery_v1":
        raise ValueError("Unsupported grid delivery")
    records = (
        [
            d[k]
            for k in [
                "grid",
                "fit",
                "fit_review",
                "validation",
                "mask_edits",
                "mask_changes",
                "inputs",
            ]
        ]
        + list(d["sources"].values())
        + d["maps"]
    )
    for rec in records:
        if sha(resolve(path.parent, rec["path"])) != rec["sha256"]:
            raise ValueError("Changed grid evidence: " + rec["path"])
    check_fit(resolve(path.parent, d["fit"]["path"]))
    readback(resolve(path.parent, d["grid"]["path"]))
    if reviewed:
        r = read_json(path.parent / "grid_review.json")
        if r.get("decision") != "accepted" or r.get("artifact_sha256") != sha(path):
            raise ValueError("Current visual grid review required")
    if reader:
        r = read_json(path.parent / d["reader_check"])
        if (
            r.get("schema") != "remora_reader_check_v1"
            or r.get("status") != "pass"
            or r.get("grid_sha256") != d["grid"]["sha256"]
            or not r.get("comparisons")
            or not r.get("evidence")
        ):
            raise ValueError("Passing executable reader check required")
        for rec in r["evidence"]:
            if sha(resolve(path.parent, rec["path"])) != rec["sha256"]:
                raise ValueError("Reader evidence changed")
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    for name in ["prepare", "fit"]:
        b = s.add_parser(name)
        b.add_argument("--request", required=True)
        b.add_argument("--output-dir", required=True)
    b = s.add_parser("build")
    b.add_argument("--fit", required=True)
    b.add_argument("--output-dir", required=True)
    for name in ["review-fit", "review"]:
        r = s.add_parser(name)
        r.add_argument("--artifact", required=True)
        r.add_argument(
            "--decision", required=True, choices=["accepted", "needs_revision"]
        )
        r.add_argument("--rationale", required=True)
    v = s.add_parser("validate")
    v.add_argument("--delivery", required=True)
    v.add_argument("--require-reviewed", action="store_true")
    v.add_argument("--require-reader", action="store_true")
    a = p.parse_args()
    if a.command == "prepare":
        print(prepare(a.request, a.output_dir))
    elif a.command == "fit":
        print(fit(a.request, a.output_dir))
    elif a.command == "build":
        print(build(a.fit, a.output_dir))
    elif a.command in ("review-fit", "review"):
        write_review(
            a.artifact,
            a.decision,
            a.rationale,
            "fit" if a.command == "review-fit" else "grid",
        )
        print("Review recorded")
    else:
        validate_grid(a.delivery, a.require_reviewed, a.require_reader)
        print("Grid delivery valid")


if __name__ == "__main__":
    main()
