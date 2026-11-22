"""Bind a river intake and a downstream transport section to a metric FVCOM mesh."""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyproj import CRS, Geod, Transformer

from mesh_io import sha256_file


def read_grid(path):
    with path.open(encoding="ascii") as stream:
        counts = []
        for label in ("Node Number", "Cell Number"):
            key, value = stream.readline().split("=")
            if key.strip() != label:
                raise ValueError(f"Expected {label} in runtime grid")
            counts.append(int(value))
        nn, ne = counts
        elements = np.array([list(map(int, stream.readline().split())) for _ in range(ne)])
        nodes = np.array([list(map(float, stream.readline().split())) for _ in range(nn)])
        if stream.read().strip() or elements.shape != (ne, 4) or nodes.shape != (nn, 4):
            raise ValueError("Grid rows differ from the FVCOM four-column format")
    if not np.array_equal(elements[:, 0], np.arange(1, ne + 1)) or not np.array_equal(nodes[:, 0], np.arange(1, nn + 1)):
        raise ValueError("Grid IDs must match the contiguous FVCOM file order")
    triangles = elements[:, 1:] - 1
    if triangles.min() < 0 or triangles.max() >= nn or np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        raise ValueError("Element references an invalid or repeated node")
    if not np.all(np.isfinite(nodes)) or not np.all(nodes[:, 3] > 0):
        raise ValueError("Expected finite coordinates and positive reference depths")
    return nodes[:, 1:3], nodes[:, 3], triangles


def read_ocean_nodes(path, node_count):
    with path.open(encoding="ascii") as stream:
        key, value = stream.readline().split("=")
        rows = [list(map(int, line.split())) for line in stream if line.strip()]
    if key.strip() != "OBC Node Number" or not rows or len(rows) != int(value) or any(len(r) != 3 for r in rows):
        raise ValueError("Invalid ocean OBC header or row count")
    ids = {r[1] - 1 for r in rows}
    if len(ids) != len(rows) or min(ids) < 0 or max(ids) >= node_count:
        raise ValueError("Invalid or duplicate ocean OBC node")
    return ids


def connected(seeds, neighbors):
    seen = set(seeds)
    todo = list(seen)
    while todo:
        for other in neighbors[todo.pop()]:
            if other not in seen:
                seen.add(other)
                todo.append(other)
    return seen


def project_points(transform, points, count):
    ll = np.asarray(points, dtype=float)
    if ll.shape != (count, 2) or not np.all(np.isfinite(ll)) or np.any(abs(ll) > [180, 90]):
        raise ValueError(f"Expected {count} finite longitude/latitude pairs")
    return np.column_stack(transform.transform(*ll.T, errcheck=True))


def source_from_edges(xy, h, triangles, edge_cells, ocean, line, tolerance, floor):
    start, end = line
    tangent = end - start
    span = np.linalg.norm(tangent)
    if not 0 < 2 * tolerance < span:
        raise ValueError("Cutoff matching tolerance must be positive and below half the source width")
    along = (xy - start) @ (tangent / span)
    near = xy - (start + np.clip(along, 0., span)[:, None] * tangent / span)
    on_cut = np.linalg.norm(near, axis=1) <= tolerance
    edges = [edge for edge, cells in edge_cells.items() if len(cells) == 1 and all(on_cut[n] for n in edge)]
    widths = defaultdict(float)
    chain = defaultdict(list)
    for a, b in edges:
        width = float(np.linalg.norm(xy[b] - xy[a]))
        widths[a] += width / 2
        widths[b] += width / 2
        chain[a].append(b)
        chain[b].append(a)
    if not edges or connected([edges[0][0]], chain) != set(chain) or sum(len(v) == 1 for v in chain.values()) != 2 or max(map(len, chain.values())) > 2:
        raise ValueError("Cutoff does not select one connected exterior edge chain")
    if any(min(np.linalg.norm(xy[n] - point) for n in chain) > tolerance for point in line) or set(chain) & ocean:
        raise ValueError("Cutoff misses its endpoints or overlaps the ocean OBC")
    receiving = sorted(n for n in widths if h[n] > floor)
    if len(receiving) < 2:
        raise ValueError("Fewer than two receiving nodes meet the low-water depth floor")
    wet_neighbors = defaultdict(list)
    for a, b, c in triangles[np.min(h[triangles], axis=1) > floor]:
        wet_neighbors[a].extend((b, c))
        wet_neighbors[b].extend((a, c))
        wet_neighbors[c].extend((a, b))
    if not set(receiving).issubset(connected([n for n in ocean if h[n] > floor], wet_neighbors)):
        raise ValueError("Receiving nodes lack a connected low-water triangle path to the ocean OBC")
    return edges, widths, receiving


def budget_section(xy, triangles, edge_cells, source_edges, ocean, line):
    centers = xy[triangles].mean(axis=1)
    direction = line[1] - line[0]
    length = np.linalg.norm(direction)
    if length <= 0:
        raise ValueError("Section endpoints coincide")
    tangent = direction / length
    normal = np.array([-tangent[1], tangent[0]])
    distance = (centers - line[0]) @ normal
    seeds = {edge_cells[e][0] for e in source_edges}
    side = np.sign(distance[next(iter(seeds))])
    if not all(distance[i] * side > 0 for i in seeds):
        raise ValueError("Section intersects the upstream cutoff")
    neighbors = defaultdict(list)
    for adjacent in edge_cells.values():
        if len(adjacent) == 2 and all(distance[i] * side > 0 for i in adjacent):
            a, b = adjacent
            neighbors[a].append(b)
            neighbors[b].append(a)
    interior = connected([next(iter(seeds))], neighbors)
    if not seeds.issubset(interior) or set(triangles[list(interior)].ravel()) & ocean:
        raise ValueError("Section does not isolate the river sources from the ocean OBC")
    edges = []
    for (a, b), adjacent in sorted(edge_cells.items()):
        if len(adjacent) != 2 or (adjacent[0] in interior) == (adjacent[1] in interior):
            continue
        inside, outside = adjacent if adjacent[0] in interior else reversed(adjacent)
        # Reject accidental cuts through remote water bodies along the line's extension.
        crossing = centers[inside] + (centers[outside] - centers[inside]) * distance[inside] / (distance[inside] - distance[outside])
        along = np.dot(crossing - line[0], tangent)
        if not 0 <= along <= length:
            raise ValueError("Section endpoints do not span every seaward outlet of the upstream volume")
        edge = xy[b] - xy[a]
        width = float(np.linalg.norm(edge))
        outward = np.array([edge[1], -edge[0]]) / width
        if np.dot(outward, centers[outside] - centers[inside]) < 0:
            outward = -outward
        edges.append(dict(cell_id=int(inside + 1), edge_nodes=[int(a + 1), int(b + 1)],
                          outward_normal_east=float(outward[0]), outward_normal_north=float(outward[1]),
                          length_m=width, station_id=f"B{inside + 1}"))
    if not edges:
        raise ValueError("No mesh edges divide the upstream and seaward volumes")
    return sorted(int(n + 1) for n in interior), edges


def prepare(grid, obc, request):
    xy, h, triangles = read_grid(grid)
    ocean = read_ocean_nodes(obc, len(xy))
    crs = CRS.from_user_input(request["horizontal_crs"])
    if not crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in crs.axis_info[:2]):
        raise ValueError("A projected horizontal CRS in meters is required")
    to_xy = Transformer.from_crs(4326, crs, always_xy=True)
    to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
    line = project_points(to_xy, request["cutoff_line_lonlat"], 2)
    downstream = project_points(to_xy, request["section_line_lonlat"], 2)
    gauge = request["gauge_lonlat"]
    project_points(to_xy, [gauge], 1)
    floor = float(request["wet_depth_floor_m"])
    if not np.isfinite(floor) or floor < 0 or not str(request["cutoff_description"]).strip():
        raise ValueError("A nonnegative wet-depth floor and a cutoff description are required")
    edges = defaultdict(list)
    for i, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((int(u), int(v))))].append(i)
    if any(len(adjacent) > 2 for adjacent in edges.values()):
        raise ValueError("Nonmanifold edge in runtime grid")
    source_edges, widths, ids = source_from_edges(xy, h, triangles, edges, ocean, line,
                                                  float(request["cutoff_tolerance_m"]), floor)
    upstream, section = budget_section(xy, triangles, edges, source_edges, ocean, downstream)
    area = float(sum(widths[i] * h[i] for i in ids))
    rows = []
    for i in ids:
        lon, lat = to_ll.transform(*xy[i])
        rows.append(dict(node_id=int(i + 1), width_m=widths[i], depth_m=float(h[i]),
                         weight=float(widths[i] * h[i] / area), lon=lon, lat=lat))
    cutoff = to_ll.transform(*np.mean(line, axis=0))
    _, _, distance = Geod(ellps="WGS84").inv(*gauge, *cutoff)
    hashes = dict(mesh_sha256=sha256_file(grid), obc_sha256=sha256_file(obc))
    mapping = dict(schema="fvcom_river_node_mapping_v1", **hashes, wet_depth_floor_m=floor,
                   cutoff_description=request["cutoff_description"], gauge_lonlat=gauge,
                   gauge_to_cutoff_km=distance / 1000, gauge_distance_method="straight-line WGS84 geodesic",
                   source_area_m2=area, face_total_width_m=sum(widths.values()),
                   face_receiving_width_m=sum(widths[i] for i in ids), nodes=rows)
    geometry = dict(schema="fvcom_river_budget_geometry_v1", **hashes, model_crs=crs.to_string(),
                    normal_frame="projected_cartesian_xy", upstream_element_ids=upstream, section=section)
    return mapping, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("grid", "obc", "request", "out-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Select a new geometry output directory")
    mapping, geometry = prepare(args.grid, args.obc, json.loads(args.request.read_text(encoding="utf-8-sig")))
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out_dir.parent) as work:
        stage = Path(work) / "geometry"
        stage.mkdir()
        for name, doc in (("river_mapping.json", mapping), ("river_budget_geometry.json", geometry)):
            doc["request_sha256"] = sha256_file(args.request)
            (stage / name).write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(args.out_dir)
    print(f"{len(mapping['nodes'])} river nodes, {len(geometry['section'])} downstream section edges")


if __name__ == "__main__":
    main()
