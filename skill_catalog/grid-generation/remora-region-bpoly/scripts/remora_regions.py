#!/usr/bin/env python3
"""Geographic REMORA region plans, independent of numerical grid topology."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, box, mapping


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(base, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base) / p).resolve()


def file_record(path, base):
    p = Path(path).resolve()
    return {
        "path": Path(os.path.relpath(p, Path(base).resolve())).as_posix(),
        "sha256": sha(p),
    }


def polygon(vertices):
    a = np.asarray(vertices, dtype=float)
    if a.shape == (5, 2) and np.array_equal(a[0], a[-1]):
        a = a[:-1]
    if a.shape != (4, 2) or not np.isfinite(a).all():
        raise ValueError("Each region needs exactly four finite lon/lat vertices")
    if (
        np.any(np.abs(a[:, 0]) > 180)
        or np.any(np.abs(a[:, 1]) >= 80)
        or np.ptp(a[:, 0]) >= 180
    ):
        raise ValueError(
            "v1 requires compact non-polar geography without an antimeridian crossing"
        )
    p = Polygon(a)
    if not p.is_valid or p.area <= 1e-10:
        raise ValueError("Invalid, self-intersecting, or degenerate polygon")
    return p


def feature_shape(f):
    if "point_lonlat" in f:
        a = np.asarray(f["point_lonlat"], float)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("Invalid feature point")
        return Point(a)
    a = np.asarray(f["bbox_wsen"], float)
    if a.shape != (4,) or not np.isfinite(a).all() or a[0] >= a[2] or a[1] >= a[3]:
        raise ValueError("Invalid feature bbox")
    return box(*a)


def validate_regions(regions):
    by_id = {}
    for r in regions:
        rid = r["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", rid) or rid in by_id:
            raise ValueError("Region IDs must be unique and filesystem-safe")
        if not str(r.get("purpose", "")).strip():
            raise ValueError(f"Missing purpose: {rid}")
        p = polygon(r["vertices_lonlat"])
        for f in r.get("required_features", []):
            if not p.covers(feature_shape(f)):
                raise ValueError(f"Required feature {f['name']} outside {rid}")
        by_id[rid] = (r, p)
    if not by_id:
        raise ValueError("Empty region plan")
    for rid, (r, p) in by_id.items():
        seen = {rid}
        parent = r.get("parent_id")
        while parent is not None:
            if parent in seen or parent not in by_id:
                raise ValueError("Missing parent or cyclic relationship")
            seen.add(parent)
            parent = by_id[parent][0].get("parent_id")
        parent = r.get("parent_id")
        if parent and not by_id[parent][1].covers(p):
            raise ValueError(f"Child {rid} extends outside parent")
    return by_id


def land_frame(path):
    kwargs = {"layer": "land_polygons"} if Path(path).suffix == ".gpkg" else {}
    g = gpd.read_file(path, **kwargs)
    if g.crs is None:
        raise ValueError("Coastline CRS missing")
    g = g.to_crs(4326)
    if g.empty or not g.is_valid.all():
        raise ValueError("Empty or invalid coastline")
    return g


def setup_map(ax, land, bounds):
    w, s, e, n = bounds
    clipped = land.cx[w:e, s:n]
    if not clipped.empty:
        clipped.plot(ax=ax, facecolor="#e2e0d7", edgecolor="#6a7473", linewidth=0.35)
    ax.set(xlim=(w, e), ylim=(s, n), xlabel="Longitude", ylabel="Latitude")
    ax.set_aspect(1 / np.cos(np.deg2rad((s + n) / 2)))
    ax.set_facecolor("#e8f3f9")
    ax.grid(alpha=0.2)


def map_regions(plan, land, output):
    polygons = [(r, polygon(r["vertices_lonlat"])) for r in plan["regions"]]
    b = np.array([p.bounds for _, p in polygons])
    full = [
        b[:, 0].min() - 0.06,
        b[:, 1].min() - 0.06,
        b[:, 2].max() + 0.06,
        b[:, 3].max() + 0.06,
    ]
    views = [("region_overview", "Region plan and parent-child relationships", full)]
    for r, p in polygons:
        if r.get("parent_id"):
            w, s, e, n = p.bounds
            views.append(
                (
                    f"child_{r['id']}",
                    f"{r['id']} inside {r['parent_id']}",
                    [w - 0.04, s - 0.04, e + 0.04, n + 0.04],
                )
            )
        for i, f in enumerate(r.get("required_features", [])):
            w, s, e, n = feature_shape(f).bounds
            dy = max(float(f.get("review_radius_km", 8)) / 111.32, 0.01)
            dx = dy / np.cos(np.deg2rad((s + n) / 2))
            views.append(
                (f"feature_{r['id']}_{i}", f["name"], [w - dx, s - dy, e + dx, n + dy])
            )
    paths = []
    for name, title, bounds in views:
        fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
        setup_map(ax, land, bounds)
        for idx, (r, p) in enumerate(polygons):
            x, y = p.exterior.xy
            ax.plot(
                x,
                y,
                color=f"C{idx%10}",
                lw=2,
                label=f"{r['id']} ({r.get('role','region')})",
            )
            ax.scatter(x[:-1], y[:-1], color=f"C{idx%10}", s=20)
            for f in r.get("required_features", []):
                c = feature_shape(f).centroid
                ax.plot(c.x, c.y, "k+", ms=7)
                if bounds[0] <= c.x <= bounds[2] and bounds[1] <= c.y <= bounds[3]:
                    ax.annotate(
                        f["name"],
                        (c.x, c.y),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        p = Path(output) / "maps" / (name + ".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def build(request_path, output):
    request_path = Path(request_path).resolve()
    req = read_json(request_path)
    out = Path(output).resolve()
    if (out / "region_delivery.json").exists():
        raise ValueError("Use a new directory for a changed plan")
    validate_regions(req["regions"])
    coast = req["coastline"]
    cp = resolve(request_path.parent, coast["path"])
    coverage = box(*coast["bbox_wsen"])
    if not all(coverage.covers(polygon(r["vertices_lonlat"])) for r in req["regions"]):
        raise ValueError("Coastline source coverage does not contain regions")
    plan = {
        "schema": "remora_region_plan_v1",
        "name": req["name"],
        "objective": req["objective"],
        "crs": "EPSG:4326",
        "regions": req["regions"],
        "geographic_sources": req.get("geographic_sources", []),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "region_plan.json", plan)
    write_json(
        out / "regions.geojson",
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        k: v for k, v in r.items() if k != "vertices_lonlat"
                    },
                    "geometry": mapping(polygon(r["vertices_lonlat"])),
                }
                for r in req["regions"]
            ],
        },
    )
    maps = map_regions(plan, land_frame(cp), out)
    source = file_record(cp, out)
    source["bbox_wsen"] = coast["bbox_wsen"]
    if coast.get("manifest"):
        source["manifest"] = file_record(
            resolve(request_path.parent, coast["manifest"]), out
        )
    d = {
        "schema": "remora_region_delivery_v1",
        "request": file_record(request_path, out),
        "plan": file_record(out / "region_plan.json", out),
        "polygons": file_record(out / "regions.geojson", out),
        "source": source,
        "maps": [file_record(p, out) for p in maps],
        "review": "scientific_review.json",
    }
    write_json(out / "region_delivery.json", d)
    return out / "region_delivery.json"


def validate_delivery(path, require_reviewed=False):
    path = Path(path).resolve()
    d = read_json(path)
    base = path.parent
    if d.get("schema") != "remora_region_delivery_v1":
        raise ValueError("Unsupported region schema")
    records = [d["plan"], d["polygons"], d["source"], *d["maps"]]
    if "request" in d:
        records.append(d["request"])
    if "manifest" in d["source"]:
        records.append(d["source"]["manifest"])
    for rec in records:
        if sha(resolve(base, rec["path"])) != rec["sha256"]:
            raise ValueError("Changed evidence: " + rec["path"])
    plan = read_json(resolve(base, d["plan"]["path"]))
    validate_regions(plan["regions"])
    if require_reviewed:
        r = read_json(resolve(base, d["review"]))
        if r.get("decision") != "accepted" or r.get("delivery_sha256") != sha(path):
            raise ValueError("Region plan lacks a current accepted visual review")
    return d, plan


def review(path, decision, rationale):
    path = Path(path).resolve()
    d, _ = validate_delivery(path)
    if not rationale.strip():
        raise ValueError("A physical rationale is required")
    write_json(
        resolve(path.parent, d["review"]),
        {
            "schema": "remora_region_review_v1",
            "decision": decision,
            "rationale": rationale,
            "reviewer": "agent",
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
            "delivery_sha256": sha(path),
            "inspected_maps": d["maps"],
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    b = s.add_parser("build")
    b.add_argument("--request", required=True)
    b.add_argument("--output-dir", required=True)
    r = s.add_parser("review")
    r.add_argument("--delivery", required=True)
    r.add_argument("--decision", choices=["accepted", "needs_revision"], required=True)
    r.add_argument("--rationale", required=True)
    v = s.add_parser("validate")
    v.add_argument("--delivery", required=True)
    v.add_argument("--require-reviewed", action="store_true")
    a = p.parse_args()
    if a.command == "build":
        print(build(a.request, a.output_dir))
    elif a.command == "review":
        review(a.delivery, a.decision, a.rationale)
        print("Visual review recorded")
    else:
        validate_delivery(a.delivery, a.require_reviewed)
        print("Region delivery valid")


if __name__ == "__main__":
    main()
