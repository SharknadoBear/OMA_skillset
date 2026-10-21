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
    sample_netcdf,
    connected_mask,
    roughness,
    smooth_depth,
    vertical_diagnostics,
    stretching,
    staggered_masks,
    GEOD,
)
from curvilinear import coordinates, fit_curvilinear, geometry_report


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
    if req.get("science") or req.get("geometry"):
        science = req.get("science", {})
        if science.get("regime") not in ("barotropic_tide", "baroclinic"):
            raise ValueError("New geometry requires explicit science.regime")
        if not science.get("rationale", "").strip():
            raise ValueError("Scientific regime rationale required")
        for key in ("coverage_rationale", "resolution_rationale", "cell_budget_rationale"):
            if not req.get("design", {}).get(key, "").strip():
                raise ValueError("Explicit design."+key+" required")
        intents=req["design"].get("boundary_intents", {})
        if set(intents) != {"xlo","xhi","ylo","yhi"}:
            raise ValueError("Record design.boundary_intents for all four logical sides")
        for side,value in intents.items():
            if (not isinstance(value,dict) or value.get("intent") not in
                    ("open","land","mixed","undetermined")
                    or not value.get("rationale","").strip()):
                raise ValueError("Invalid boundary intent for "+side)
        if "N" not in req.get("parameters", {}) or not p.get("vertical_rationale", "").strip():
            raise ValueError("Choose N explicitly with vertical_rationale")
        if science["regime"] == "barotropic_tide" and not 8 <= p["N"] <= 20:
            raise ValueError("Barotropic-tide policy requires 8–20 justified layers")
        if isinstance(p["N"], bool) or int(p["N"]) != p["N"]:
            raise ValueError("N must be an integer")
        p["N"] = int(p["N"])
        p.setdefault("rx1_max", 9.0)
        p.setdefault("rx1_preferred", 7.0)
        p.setdefault("rx1_target", p["rx1_preferred"])
        if not (0 < p["rx1_target"] <= p["rx1_max"]
                and 0 < p["rx1_preferred"] <= p["rx1_max"]):
            raise ValueError("Invalid rx1 target/preferred/maximum")
        p["scientific_regime"] = science["regime"]
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
    if (rp.get("modeling_purpose") and req.get("science", {}).get("regime")
            and rp["modeling_purpose"] != req["science"]["regime"]):
        raise ValueError("Region purpose and grid scientific regime disagree")
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
    args = (region["vertices_lonlat"], p["spacing_m"], p["padding_m"],
            p.get("rotation_deg"), p["max_cells"])
    geom = (fit_curvilinear(*args, req["geometry"]) if req.get("geometry")
            else fit_domain(*args))
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
        f"Fitted grid: {geom['nx']} x {geom['ny']} cells\n"
        f"xi spacing {1/a['pm'].max():.0f}–{1/a['pm'].min():.0f} m; "
        f"eta spacing {1/a['pn'].max():.0f}–{1/a['pn'].min():.0f} m"
    )
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "grid_fit.png", dpi=150)
    plt.close(fig)
    doc = {
        "schema": "remora_grid_fit_v1",
        "geometry": geom,
        "parameters": p,
        "science": req.get("science", {}),
        "design": req.get("design", {}),
        "geometry_validation": geometry_report(a, geom),
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
        if f.get("priority", "essential") != "essential":
            raise ValueError("protected_wet_features must contain essential features only")
        x, y = f["point_lonlat"]
        _, k = tree.query([x * scale, y])
        j, i = int(jj[k]), int(ii[k])
        _, _, distance = GEOD.inv(x, y, float(lon[j, i]), float(lat[j, i]))
        local_spacing = max(1/a["pm"][j,i], 1/a["pn"][j,i])
        if distance > f.get("max_snap_m", local_spacing * 1.5):
            raise ValueError("Required wet feature unsupported: " + f["name"])
        result.append((j, i))
        records.append({"name": f["name"], "j": j, "i": i, "snap_distance_m": distance})
    return result, records


def require_scientific_features(req, region):
    """Bind essential water geography to mask protection for new requests."""
    if not req.get("science"):
        return
    protected=req.get("protected_wet_features",[])
    lookup={f["name"]:f for f in protected}
    if len(lookup)!=len(protected):
        raise ValueError("Duplicate protected wet feature names")
    features=region.get("features",[])+region.get("required_features",[])
    for feature in features:
        kind=feature.get("feature_type","water" if "point_lonlat" in feature else "coverage")
        if feature.get("priority","essential") != "essential" or kind!="water":
            continue
        anchor=lookup.get(feature["name"])
        if anchor is None:
            raise ValueError("Essential water feature needs a named protected anchor: "+feature["name"])
        point=np.asarray(anchor.get("point_lonlat"),dtype=float)
        if point.shape!=(2,) or not np.isfinite(point).all():
            raise ValueError("Protected feature needs finite lon/lat")
        if "point_lonlat" in feature:
            if not np.allclose(point,feature["point_lonlat"],rtol=0,atol=1e-8):
                raise ValueError("Protected anchor moved from declared water feature: "+feature["name"])
        else:
            west,south,east,north=feature["bbox_wsen"]
            if not (west<=point[0]<=east and south<=point[1]<=north):
                raise ValueError("Protected anchor outside its essential water feature")


def export_grid(path, a, h, mask, p, fitdoc, raw):
    ny, nx = h.shape
    sr, sw, cr, cw = stretching(p["N"], p["theta_s"], p["theta_b"])
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
        ds.geometry_kind = fitdoc["geometry"].get("geometry_kind", "rotated_uniform")
        ds.geometry_definition = json.dumps(fitdoc["geometry"], separators=(",", ":"))
        ds.science = json.dumps(fitdoc.get("science", {}), separators=(",", ":"))
        ds.design_rationale = json.dumps(fitdoc.get("design", {}), separators=(",", ":"))
        ds.numerical_levels = 1
        ds.smoothing_method = ("area_weighted_log_rx0_rx1_pair_projection_v2"
                               if "rx1_max" in p else "area_weighted_log_pair_projection_v1")
        ds.smoothing_space = "natural_log_depth"
        ds.smoothing_volume_policy = "report_only_no_rescaling"
        ds.source_policy = "Real source data; no missing wet bathymetry substitution"
        for key, val in p.items():
            if isinstance(val, (int, float, str)):
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
            for prefix in ("projected_x", "projected_y"):
                name = f"{prefix}_{tag}"
                if name in a:
                    v = ds.createVariable(name, "f8", (f"eta_{tag}", f"xi_{tag}"))
                    v[:] = a[name]
                    v.units = "m"
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
            ("xl", fitdoc["geometry"].get("logical_extent_m", [fitdoc["geometry"]["nx"] * p["spacing_m"], fitdoc["geometry"]["ny"] * p["spacing_m"]])[0]),
            ("el", fitdoc["geometry"].get("logical_extent_m", [fitdoc["geometry"]["nx"] * p["spacing_m"], fitdoc["geometry"]["ny"] * p["spacing_m"]])[1]),
        ]:
            ds.createVariable(name, "f8", ())[...] = value


from review_maps import diagnostic_maps, feature_sections


def build(fit_path, out):
    fit_path = Path(fit_path).resolve()
    f = check_fit(fit_path)
    out = Path(out).resolve()
    if (out / "grid_delivery.json").exists():
        raise ValueError("Use a new build directory for a changed grid")
    request_path = resolve(fit_path.parent, f["request"]["path"])
    req, p, rpath, rd, rp, region = load_request(request_path)
    require_scientific_features(req,region)
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
    candidate = initial.copy()
    explicit_edits = []
    declared_features={feature["name"]:feature
                       for feature in region.get("features",[])+region.get("required_features",[])}
    for edit in req.get("mask_adjustments", []):
        if (edit.get("action") != "close_optional" or not edit.get("rationale", "").strip()
                or not edit.get("evidence")):
            raise ValueError("Mask adjustment must close optional water with rationale and evidence")
        feature=declared_features.get(edit.get("feature_name"))
        if not feature or feature.get("priority","essential") != "optional":
            raise ValueError("Explicit closure must identify a declared optional region feature")
        if any(not isinstance(e,dict) or not e.get("source") or not e.get("finding")
               for e in edit["evidence"]):
            raise ValueError("Closure evidence needs source and finding")
        shape = polygon(edit["vertices_lonlat"])
        affected = shapely.intersects_xy(shape, a["lon_rho"], a["lat_rho"]) & candidate
        candidate[affected] = False
        explicit_edits.append(dict(edit, changed_cells=int(affected.sum())))
    # Snap in the original source mask, so a closure cannot move an essential
    # anchor into a different surviving basin and falsely claim preservation.
    anchors, anchor_report = anchor_indices(
        a, initial, req.get("protected_wet_features", []), p["spacing_m"]
    )
    mask, corrections = connected_mask(candidate, anchors)
    if req.get("science"):
        for feature in declared_features.values():
            if feature.get("priority","essential")!="essential" or feature.get("feature_type")!="land":
                continue
            if "point_lonlat" in feature:
                x,y=feature["point_lonlat"]
                if not shapely.intersects_xy(land.geometry.union_all(),x,y):
                    raise ValueError("Essential land feature lacks shoreline support: "+feature["name"])
                distance=(a["lon_rho"]-x)**2*np.cos(np.deg2rad(y))**2+(a["lat_rho"]-y)**2
                j,i=np.unravel_index(distance.argmin(),mask.shape)
                if mask[j,i]:
                    raise ValueError("Essential island/land point is not represented by the mask: "+feature["name"])
    corrections["explicit_optional_closures"] = explicit_edits
    corrections["removed_cells"] = int((initial & ~mask).sum())
    raw = np.where(mask, np.maximum(-elev, p["hmin_m"]), p["hmin_m"])
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite classified depth")
    area = 1 / (a["pm"] * a["pn"])
    if "rx1_max" in p:
        from conditioning import smooth_depth_vertical
        h, smoothing = smooth_depth_vertical(
            raw, mask, area, p["rx0_max"], p["rx1_target"], p["N"],
            p["theta_s"], p["theta_b"], p["hc_m"], hmin=p["hmin_m"],
            max_iter=p.get("smoothing_max_iter", 4000))
    else:
        h, smoothing = smooth_depth(raw, mask, area, p["rx0_max"])
    vertical = vertical_diagnostics(
        h, p["N"], p["theta_s"], p["theta_b"], p["hc_m"], mask=mask)
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
        p,
    )
    section_report, section_maps = feature_sections(mapsdir, a, raw, h, mask, anchors, req.get("protected_wet_features", []))
    maps.extend(section_maps)
    report = readback(grid)
    core = mask.copy()
    core[[0, -1], :] = False
    core[:, [0, -1]] = False
    smoothing["volume_domain"] = "all rho cells including the one-cell halo"
    smoothing["physical_volume_relative_change"] = float(
        np.sum((h - raw)[core] * area[core]) / np.sum(raw[core] * area[core])
    )
    source_depth = np.where(mask, -elev, p["hmin_m"])
    floor_added_volume = float(np.sum((raw-source_depth)[core]*area[core]))
    floor_report = {
        "wet_cells_clipped": int((mask & (source_depth<p["hmin_m"])).sum()),
        "physical_volume_added_m3": floor_added_volume,
        "physical_volume_relative_change": floor_added_volume/float(np.sum(source_depth[core]*area[core])),
        "interpretation": "depth-floor changes before log smoothing; hraw already includes this floor"}
    report.update(
        smoothing=smoothing,
        feature_sections=section_report,
        haney_rx1_diagnostic=vertical["haney_rx1_diagnostic"],
        maximum_depth_change_m=float(np.max(np.abs(h[mask] - raw[mask]))),
        rms_depth_change_m=float(np.sqrt(np.mean((h[mask] - raw[mask]) ** 2))),
        wet_cells=int(mask.sum()),
        physical_wet_cells=int(mask[1:-1, 1:-1].sum()),
        orthogonality_error_max_deg=float(a["orthogonality_error_deg"].max()),
        source_vertical_datum=bathy.get("vertical_datum", "unspecified"),
        warnings=list(bathy.get("warnings", [])),
        geometry=geometry_report(a, f["geometry"]),
        design=req.get("design", {}),
        science=req.get("science", {}),
        depth_floor=floor_report,
        computational_size={
            "physical_dimensions": [f["geometry"]["nx"], f["geometry"]["ny"]],
            "rho_dimensions": [h.shape[1], h.shape[0]],
            "physical_horizontal_cells": f["geometry"]["nx"]*f["geometry"]["ny"],
            "allocated_3d_cells_without_halo": f["geometry"]["nx"]*f["geometry"]["ny"]*p["N"],
            "wet_3d_cells_without_halo": int(mask[1:-1,1:-1].sum())*p["N"],
            "interpretation": "counts only; no measured runtime or speedup"},
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
    if req.get("science"):
        for side, intent in req["design"]["boundary_intents"].items():
            if intent["intent"] == "land" and report["edge_wet_cells"][side]:
                report["warnings"].append(
                    f"{side} was intended as land but has wet boundary cells; revise or explain boundary intent.")
    if corrections["removed_cells"]:
        report["warnings"].append(
            f"Closed {corrections['removed_cells']} disconnected water cells; inspect lagoon and tributary consequences before scientific use."
        )
    write_json(out / "validation.json", report)
    nx, ny = f["geometry"]["nx"], f["geometry"]["ny"]
    lx, ly = f["geometry"].get("logical_extent_m", [nx*p["spacing_m"],ny*p["spacing_m"]])
    snippet = f"""# Grid-only snippet; merge with a complete REMORA simulation configuration.
remora.n_cell = {nx} {ny} {p['N']}
remora.prob_lo = 0 0 {-float(h.max()):.8g}
remora.prob_hi = {lx:.8g} {ly:.8g} 0
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
        if r > d.rx0_max + (0 if hasattr(d,"rx1_max") else 2e-8):
            raise ValueError("Excessive bathymetric roughness")
        vertical = vertical_diagnostics(h, int(d.N), d.theta_s, d.theta_b, d.hc_m,
                                        mask=m.astype(bool))
        if hasattr(d, "rx1_max") and vertical["haney_rx1_diagnostic"] >= d.rx1_max:
            raise ValueError("Excessive vertical-coordinate rx1")
        geometry = {}
        if getattr(d, "geometry_kind", "") == "conformal_graded":
            definition = json.loads(d.geometry_definition)
            expected_geometry = coordinates(definition)
            for name in ["pm","pn","angle","f"] + [
                    f"{prefix}_{tag}" for tag in ("rho","u","v","psi")
                    for prefix in ("lon","lat","x","y","projected_x","projected_y")]:
                np.testing.assert_allclose(d[name][:],expected_geometry[name],
                                           rtol=1e-12,atol=1e-10,
                                           err_msg="Grid geometry readback mismatch: "+name)
            geometry = geometry_report(expected_geometry,definition)
        return {
            "local_validation": "pass",
            "format": d.data_model,
            "rho_shape": [ny, nx],
            "n_cell": [nx - 2, ny - 2, int(d.N)],
            "rx0": r,
            "minimum_layer_thickness_m": vertical["minimum_layer_thickness_m"],
            "haney_rx1_diagnostic": vertical["haney_rx1_diagnostic"],
            "rx1_preferred_met": bool(vertical["haney_rx1_diagnostic"] <
                                      getattr(d,"rx1_preferred",7.0)),
            "geometry": geometry,
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
    q = s.add_parser("finalize")
    q.add_argument("--case-root", required=True)
    q.add_argument("--delivery", required=True)
    q.add_argument("--release-revision", required=True)
    q = s.add_parser("validate-case")
    q.add_argument("--delivery", required=True)
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
    elif a.command == "finalize":
        from case_delivery import finalize
        print(finalize(a.case_root, a.delivery, a.release_revision))
    elif a.command == "validate-case":
        from case_delivery import validate_case
        validate_case(a.delivery)
        print("Case delivery valid")
    else:
        validate_grid(a.delivery, a.require_reviewed, a.require_reader)
        print("Grid delivery valid")


if __name__ == "__main__":
    main()
