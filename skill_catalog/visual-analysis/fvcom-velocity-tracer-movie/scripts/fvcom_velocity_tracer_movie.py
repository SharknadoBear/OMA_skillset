#!/usr/bin/env python3
"""Build an offline native-mesh FVCOM current particle viewer."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
from netCDF4 import Dataset, chartostring, num2date
from pyproj import CRS, Transformer, Geod

PACKAGE = Path(__file__).resolve().parents[1]
UPSTREAM = "5f091f0c3b60aa38a996a886985bacb3673d16c3"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def epoch(value):
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed).timestamp()


def utc(value):
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat().replace("+00:00", "Z")


def values(v):
    return np.asarray(np.ma.filled(np.ma.asarray(v[:], dtype=float), np.nan))


def decode_times(ds):
    if "Times" in ds.variables:
        result = np.array([epoch(s.replace("\x00", "")) for s in chartostring(ds["Times"][:]).tolist()])
    elif "Itime" in ds.variables and "Itime2" in ds.variables:
        result = (values(ds["Itime"]) - 40587) * 86400 + values(ds["Itime2"]) / 1000
    elif "time" in ds.variables and hasattr(ds["time"], "units"):
        v = ds["time"]
        result = np.array([epoch(t.isoformat()) for t in num2date(v[:], v.units, getattr(v, "calendar", "standard"))])
    else:
        raise ValueError("No supported UTC time coordinate (Times, Itime/Itime2, or CF time)")
    if not np.isfinite(result).all() or not len(result):
        raise ValueError("Empty or invalid timestamps")
    return result


def native_mesh(ds, coordinate_crs=None, vector_basis=None):
    if "nv" not in ds.variables:
        raise ValueError("Missing native FVCOM triangle connectivity nv")
    raw = np.asarray(ds["nv"][:], dtype=np.int64)
    tri = raw.T if ds["nv"].dimensions[0] == "three" or raw.shape[0] == 3 else raw
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("nv must have three vertices per element")
    start_index = getattr(ds["nv"], "start_index", 1 if tri.min() == 1 else 0)
    if start_index not in (0, 1):
        raise ValueError("Unsupported connectivity start_index")
    tri = tri - start_index
    geographic_valid = False
    if "lon" in ds.variables and "lat" in ds.variables:
        lon, lat = values(ds["lon"]).ravel(), values(ds["lat"]).ravel()
        geographic_valid = bool(np.isfinite(lon).all() and np.isfinite(lat).all()
                                and np.ptp(lon) > 1e-8 and np.ptp(lat) > 1e-8
                                and np.abs(lat).max() <= 90 and np.abs(lon).max() <= 360)
    crs_text = coordinate_crs or getattr(ds, "coordinate_crs", None) or getattr(ds, "output_grid_crs", None)
    # Read explicit CF grid mappings, never guess projected units from coordinate names.
    if not crs_text:
        for v in ds.variables.values():
            if hasattr(v, "crs_wkt"):
                crs_text = v.crs_wkt
                break
            if hasattr(v, "spatial_ref"):
                crs_text = v.spatial_ref
                break
    basis = vector_basis or getattr(ds, "vector_basis", None)
    if basis is None:
        a = ds.variables.get("ua", ds.variables.get("u"))
        b = ds.variables.get("va", ds.variables.get("v"))
        if a is not None and b is not None and "eastward" in getattr(a, "standard_name", "") and "northward" in getattr(b, "standard_name", ""):
            basis = "east-north"
    if basis not in ("grid", "east-north"):
        raise ValueError("Vector basis is ambiguous; supply --vector-basis grid or east-north")
    if crs_text and CRS.from_user_input(crs_text).is_projected and "x" in ds.variables and "y" in ds.variables:
        crs = CRS.from_user_input(crs_text)
        if any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in crs.axis_info[:2]):
            raise ValueError("Projected x/y must use metres")
        x, y = values(ds["x"]).ravel(), values(ds["y"]).ravel()
        recovered_lon, recovered_lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
        if geographic_valid:
            _, _, error = Geod(ellps="WGS84").inv(lon, lat, recovered_lon, recovered_lat)
            if np.max(error) > 20:
                raise ValueError("Projected and geographic coordinates disagree by more than 20 m")
        lon, lat = np.asarray(recovered_lon), np.asarray(recovered_lat)
        coordinate_method = "projected_xy_with_explicit_crs"
    elif geographic_valid:
        if basis == "grid":
            raise ValueError("Grid-relative vectors require a projected x/y CRS")
        # A local azimuthal equidistant projection avoids a UTM zone assumption.
        crs = CRS.from_proj4(f"+proj=aeqd +lat_0={np.mean(lat)} +lon_0={np.mean(lon)} +datum=WGS84 +units=m")
        x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
        x, y = np.asarray(x), np.asarray(y)
        coordinate_method = "valid_lonlat_to_local_aeqd"
    else:
        raise ValueError("Invalid/missing lon/lat and no projected x/y CRS; supply --crs")
    if not all(np.isfinite(a).all() for a in (x, y, lon, lat)) or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("Invalid or collapsed coordinates")
    if tri.min() < 0 or tri.max() >= len(x) or len(np.unique(tri)) != len(x):
        raise ValueError("Invalid connectivity or unused mesh nodes")
    det = (x[tri[:, 1]] - x[tri[:, 0]]) * (y[tri[:, 2]] - y[tri[:, 0]]) - (y[tri[:, 1]] - y[tri[:, 0]]) * (x[tri[:, 2]] - x[tri[:, 0]])
    if np.any(np.abs(det) < 1e-8):
        raise ValueError("Degenerate triangles")
    reversed_cells = det < 0
    tri[reversed_cells] = tri[reversed_cells][:, [0, 2, 1]]
    area = np.abs(det) * .5
    # Neighbors are indexed by the opposite vertex; the walker uses this convention.
    neighbors = np.full(tri.shape, -1, dtype=np.int32)
    edge_owner = {}
    boundary = []
    for cell, nodes in enumerate(tri):
        for opposite in range(3):
            edge = tuple(sorted((int(nodes[(opposite + 1) % 3]), int(nodes[(opposite + 2) % 3]))))
            if edge not in edge_owner:
                edge_owner[edge] = (cell, opposite)
            else:
                previous, side = edge_owner[edge]
                if previous < 0:
                    raise ValueError("Nonmanifold edge shared by more than two triangles")
                neighbors[cell, opposite] = previous
                neighbors[previous, side] = cell
                edge_owner[edge] = (-1, -1)
    for edge, owner in edge_owner.items():
        if owner[0] >= 0:
            boundary.append(edge)
    origin = np.array([float(x.min()), float(y.min())])
    xy = np.column_stack((x, y)) - origin
    edge_lengths = np.linalg.norm(xy[tri[:, [1, 2, 0]]] - xy[tri[:, [2, 0, 1]]], axis=2)
    heights = 2 * area / edge_lengths.max(axis=1)
    transform = None
    if basis == "east-north":
        centers = np.column_stack((x[tri].mean(axis=1), y[tri].mean(axis=1)))
        cl, ct = Transformer.from_crs(crs, 4326, always_xy=True).transform(centers[:, 0], centers[:, 1])
        geod = Geod(ellps="WGS84")
        target = Transformer.from_crs(4326, crs, always_xy=True)
        columns = []
        for azimuth in (90, 0):
            pl, pt, _ = geod.fwd(cl, ct, np.full(len(cl), azimuth), np.ones(len(cl)))
            px, py = target.transform(pl, pt)
            columns.append(np.column_stack((px, py)) - centers)
        transform = np.stack(columns, axis=2)
    return dict(xy=xy, geo=np.column_stack((lon, lat)), tri=tri.astype(np.int32), area=area,
                height=heights, neighbors=neighbors, boundary=np.array(boundary, dtype=np.int32),
                origin=origin, crs=crs.to_string(), basis=basis, coordinate_method=coordinate_method,
                invalid_geographic_arrays=not geographic_valid, transform=transform)


def read_velocity(ds, time_index, layer, count):
    names = ("ua", "va") if layer == "depth_average" else ("u", "v")
    result = []
    for name in names:
        if name not in ds.variables:
            raise ValueError(f"Layer {layer} requires native {names[0]}/{names[1]}; {name} is absent")
        v = ds[name]
        if getattr(v, "units", "").strip().lower() not in ("meters s-1", "metres s-1", "m s-1", "m/s", "meter second-1", "meters/second"):
            raise ValueError(f"Unsupported velocity units for {name}: {getattr(v, 'units', None)}")
        if "nele" not in v.dimensions or "time" not in v.dimensions:
            raise ValueError("Velocity must be native element-centered, with time and nele dimensions")
        slices = [slice(None)] * v.ndim
        slices[v.dimensions.index("time")] = time_index
        if layer != "depth_average":
            if "siglay" not in v.dimensions:
                raise ValueError("Native u/v must have siglay dimension")
            n = len(ds.dimensions["siglay"])
            if layer.startswith("index:"):
                idx = int(layer.split(":", 1)[1])
                if not 0 <= idx < n:
                    raise ValueError("Sigma layer index is out of bounds")
                slices[v.dimensions.index("siglay")] = idx
            elif layer not in ("surface", "bottom"):
                raise ValueError("Layer must be depth_average, surface, bottom, or index:N")
        data = np.asarray(np.ma.filled(np.ma.asarray(v[tuple(slices)], dtype=float), np.nan))
        remaining_dims = [d for d, s in zip(v.dimensions, slices) if not isinstance(s, (int, np.integer))]
        if layer in ("surface", "bottom"):
            sigma_name = "siglay_center" if "siglay_center" in ds.variables else "siglay"
            if sigma_name not in ds.variables:
                raise ValueError("Surface/bottom selection requires sigma metadata")
            sv = ds[sigma_name]
            sigma = values(sv)
            axis = sv.dimensions.index("siglay")
            sigma = np.moveaxis(sigma, axis, 0)
            if sigma.ndim == 1:
                sigma = np.repeat(sigma[:, None], count, axis=1)
            elif "node" in sv.dimensions:
                raw = np.asarray(ds["nv"][:], dtype=int)
                tri = raw.T if raw.shape[0] == 3 else raw
                tri -= getattr(ds["nv"], "start_index", 1 if tri.min() == 1 else 0)
                sigma = sigma[:, tri].mean(axis=2)
            if sigma.shape != (n, count) or not np.isfinite(sigma).all() or np.max(np.abs(sigma)) > 1.001:
                raise ValueError("Unsupported sigma geometry")
            idx = np.argmin(np.abs(sigma), axis=0) if layer == "surface" else np.argmax(np.abs(sigma), axis=0)
            data = np.moveaxis(data, remaining_dims.index("siglay"), 0)[idx, np.arange(count)]
        if data.shape != (count,):
            raise ValueError(f"Unexpected {name} shape {data.shape}")
        result.append(data)
    vector = np.column_stack(result)
    wet = np.ones(count, dtype=bool)
    if "wet_cells" in ds.variables:
        mask = ds["wet_cells"]
        if mask.dimensions != ("time", "nele"):
            raise ValueError("Unsupported wet_cells dimensions")
        a = np.asarray(np.ma.filled(np.ma.asarray(mask[time_index], dtype=float), np.nan))
        if not np.isin(a, [0, 1]).all():
            raise ValueError("wet_cells must contain only 0/1")
        wet = a == 1
    finite = np.isfinite(vector).all(axis=1)
    valid = wet & finite
    if not valid.any():
        raise ValueError("No finite wet vectors in selected frame")
    vector[~valid] = 0
    return vector, valid, int((wet & ~finite).sum())


def prepare(inputs, start=None, end=None, layer="depth_average", crs=None, vector_basis=None, obc=None, max_gap_hours=None):
    first = None
    records = {}
    sources = []
    dropped = 0
    for path in map(Path, inputs):
        with Dataset(path) as ds:
            mesh = native_mesh(ds, crs, vector_basis)
            if first is None:
                first = mesh
            elif mesh["crs"] != first["crs"] or mesh["basis"] != first["basis"] or any(not np.array_equal(mesh[k], first[k]) for k in ("xy", "tri", "origin")):
                raise ValueError("Input stack contains inconsistent mesh geometry or vector basis")
            times = decode_times(ds)
            selected = 0
            for i, t in enumerate(times):
                if (start and t < epoch(start)) or (end and t > epoch(end)):
                    continue
                velocity, wet, invalid = read_velocity(ds, i, layer, len(mesh["tri"]))
                source_velocity = velocity.astype("<f4")
                if mesh["transform"] is not None:
                    velocity = np.einsum("nij,nj->ni", mesh["transform"], velocity)
                velocity = velocity.astype("<f4")
                if t in records:
                    if not np.array_equal(records[t][0], velocity) or not np.array_equal(records[t][1], wet) or not np.array_equal(records[t][3], source_velocity):
                        raise ValueError(f"Conflicting duplicate record at {utc(t)}")
                else:
                    records[t] = (velocity, wet, invalid, source_velocity)
                dropped += invalid
                selected += 1
            sources.append(dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path), selected_records=selected))
    if not records:
        raise ValueError("No records in selected time interval")
    times = np.array(sorted(records))
    if start and times[0] != epoch(start) or end and times[-1] != epoch(end):
        raise ValueError("Requested endpoints must match available source timestamps; no extrapolation")
    steps = np.diff(times)
    nominal = float(np.min(steps)) if len(steps) else 0.0
    gap_limit = max_gap_hours * 3600 if max_gap_hours is not None else nominal * 1.5
    gaps = [i for i, step in enumerate(steps) if step > gap_limit]
    uv = np.stack([records[t][0] for t in times])
    wet = np.stack([records[t][1] for t in times]).astype("u1")
    boundary_type = np.zeros(len(first["boundary"]), dtype="u1")
    obc_source = None
    if obc:
        lines = Path(obc).read_text().strip().splitlines()
        expected = int(lines[0].split("=")[-1])
        nodes = [int(line.split()[1]) - 1 for line in lines[1:] if line.strip()]
        if len(nodes) != expected or len(set(nodes)) != expected or any(n < 0 or n >= len(first["xy"]) for n in nodes):
            raise ValueError("Invalid OBC node file")
        if not set(nodes).issubset(set(first["boundary"].ravel().tolist())):
            raise ValueError("OBC nodes are not on the mesh boundary")
        node_set = set(nodes)
        boundary_type = np.array([int(a in node_set and b in node_set) for a, b in first["boundary"]], dtype="u1")
        obc_source = dict(path=str(Path(obc).resolve()), sha256=digest(obc), node_count=len(nodes))
    arrays = {"xy": first["xy"].astype("<f8"), "geo": first["geo"].astype("<f8"),
              "tri": first["tri"].astype("<i4"), "neighbors": first["neighbors"].astype("<i4"),
              "area": first["area"].astype("<f8"), "height": first["height"].astype("<f4"),
              "boundary": first["boundary"].astype("<i4"), "boundaryType": boundary_type,
              "uv": uv, "wet": wet}
    if first["transform"] is not None:
        arrays["sourceUv"] = np.stack([records[t][3] for t in times])
    speed = np.linalg.norm(uv, axis=2)[wet.astype(bool)]
    info = dict(schema_version="fvcom_velocity_tracer_v1", nodes=len(first["xy"]), elements=len(first["tri"]),
                times=times.tolist(), timestamps=[utc(t) for t in times], layer=layer, units="m/s",
                crs=first["crs"], vector_basis=first["basis"], origin=first["origin"].tolist(),
                coordinate_method=first["coordinate_method"], invalid_geographic_arrays=first["invalid_geographic_arrays"],
                sources=sources, obc=obc_source, boundary_label="Solid / open mesh boundary" if obc else "Mesh boundary (open edges unclassified)",
                nominal_cadence_seconds=nominal, gap_limit_seconds=gap_limit, gap_after_indices=gaps,
                invalid_wet_vectors_excluded=dropped, finite_wet_counts=wet.sum(axis=1).tolist(),
                speed_range_mps=[float(speed.min()), float(speed.max())],
                smoothing="area_weighted_connected_wet_vertex_fans_then_native_barycentric",
                temporal_interpolation="linear_components_with_common_wet_mask",
                upstream_revision=UPSTREAM, renderer_revision="snapshot_gif_v004", display_defaults=dict(particles=3000, tail_seconds=3, continuous_tail_seconds=1, visual_speed=1, lifetime_seconds=8, playback_seconds=60, trail_history_limit_bytes=64*1024*1024, continuous_motion="independent_visual_multiplier"))
    return arrays, info


def reconstruct(mesh, vectors, valid):
    """Reference corner vectors, separating disconnected wet sectors at a node."""
    tri, area, adj = mesh["tri"], mesh["area"], mesh["neighbors"]
    incident = [[] for _ in range(len(mesh["xy"]))]
    for c, nodes in enumerate(tri):
        if valid[c]:
            for k, n in enumerate(nodes):
                incident[n].append((c, k))
    output = np.zeros((len(tri), 3, 2), dtype=np.float32)
    for fan in incident:
        pending = dict(fan)
        while pending:
            cell = next(iter(pending))
            stack, component = [cell], []
            while stack:
                c = stack.pop()
                if c not in pending:
                    continue
                k = pending.pop(c)
                component.append((c, k))
                stack.extend(int(adj[c, side]) for side in ((k + 1) % 3, (k + 2) % 3) if adj[c, side] in pending)
            ids = [c for c, _ in component]
            mean = np.average(vectors[ids], axis=0, weights=area[ids])
            for c, k in component:
                output[c, k] = mean
    return output


def write_html(arrays, info, output, title="FVCOM current atlas", vmax=None):
    output = Path(output)
    if output.resolve() in {Path(source["path"]).resolve() for source in info["sources"]}:
        raise ValueError("Output must not overwrite a source NetCDF")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(info)
    metadata["title"] = title
    metadata["vmax"] = float(vmax if vmax is not None else max(info["speed_range_mps"][1], .01))
    if not np.isfinite(metadata["vmax"]) or metadata["vmax"] <= 0:
        raise ValueError("vmax must be positive and finite")
    metadata["gif_export"] = dict(mode="snapshot", width=2400, duration_seconds=5, fps=20,
        max_pixels=8000000, max_duration_seconds=10, intended_ppi=300,
        encoder=json.loads((PACKAGE / "references/gifenc_provenance.json").read_text(encoding="utf-8")))
    metadata["arrays"] = {}
    blocks = []
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        payload = a.tobytes()
        metadata["arrays"][name] = dict(dtype=a.dtype.str, shape=list(a.shape), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        blocks.append(f'<script type="application/octet-stream" id="data-{name}">{base64.b64encode(payload).decode("ascii")}</script>')
    template = (PACKAGE / "assets/viewer.html").read_text(encoding="utf-8")
    replacements = {"@@TITLE@@": html.escape(title), "@@STYLE@@": (PACKAGE / "assets/viewer.css").read_text(encoding="utf-8"),
                    "@@DATA@@": "\n".join(blocks), "@@META@@": json.dumps(metadata, separators=(",", ":")).replace("<", "\\u003c"),
                    "@@ENGINE@@": (PACKAGE / "assets/tracer_engine.js").read_text(encoding="utf-8"),
                    "@@TRAILS@@": (PACKAGE / "assets/trail_renderer.js").read_text(encoding="utf-8"),
                    "@@FIELD@@": (PACKAGE / "assets/field_renderer.js").read_text(encoding="utf-8"),
                    "@@GIFENC@@": (PACKAGE / "assets/gifenc.js").read_text(encoding="utf-8"),
                    "@@GIF_WORKER@@": (PACKAGE / "assets/gif_worker.js").read_text(encoding="utf-8"),
                    "@@GIF_EXPORT@@": (PACKAGE / "assets/gif_export.js").read_text(encoding="utf-8"),
                    "@@GIF_LICENSE@@": html.escape((PACKAGE / "assets/GIFENC_LICENSE.txt").read_text(encoding="utf-8")),
                    "@@VIEWER@@": (PACKAGE / "assets/viewer.js").read_text(encoding="utf-8"),
                    "@@LICENSE@@": html.escape((PACKAGE / "assets/EARTH_LICENSE.txt").read_text(encoding="utf-8"))}
    for key, content in replacements.items():
        template = template.replace(key, content)
    output.write_text(template, encoding="utf-8")
    metadata["output"] = dict(path=str(output.resolve()), bytes=output.stat().st_size, sha256=digest(output))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "build"])
    parser.add_argument("--input", nargs="+", action="extend", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end", help="Inclusive final native timestamp")
    parser.add_argument("--layer", default="depth_average")
    parser.add_argument("--crs", help="CRS of native projected x/y, e.g. EPSG:32615")
    parser.add_argument("--vector-basis", choices=["grid", "east-north"])
    parser.add_argument("--obc", help="Optional FVCOM obc.dat, same node IDs as output")
    parser.add_argument("--max-gap-hours", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--title", default="FVCOM current atlas")
    parser.add_argument("--vmax", type=float)
    a = parser.parse_args()
    try:
        arrays, info = prepare(a.input, a.start, a.end, a.layer, a.crs, a.vector_basis, a.obc, a.max_gap_hours)
        report_path = Path(a.output) if a.command == "inspect" else Path(a.report or str(Path(a.output).with_suffix(".json")))
        protected = {Path(p).resolve() for p in a.input}
        if report_path.resolve() in protected or (a.command == "build" and report_path.resolve() == Path(a.output).resolve()):
            raise ValueError("Report path must be distinct from input files and output HTML")
        if a.command == "build":
            info = write_html(arrays, info, a.output, a.title, a.vmax)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(json.dumps(dict(status="passed", command=a.command, frames=len(info["times"]), nodes=info["nodes"], elements=info["elements"], report=str(report_path))))
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
