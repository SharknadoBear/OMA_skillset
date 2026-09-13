"""Bounded single-island mesh transactions; never an unchanged-retained join.

Indices in this API are zero based. Serialized SMS identifiers remain one based.
The caller supplies a scientifically reviewed ring, source bindings, and a strict
bathymetry sampler. No shoreline discovery, datum conversion or model tuning is
performed here. The independent audit rereads both serialized meshes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import shapely
from shapely.geometry import LineString, Point, Polygon

from refinement_core import (classify_patch, deterministic_new_ids,
    graph_lower_envelope, order_boundary_loops, signed_areas,
    stitch_field_audit, unique_edges)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and np.isfinite(value)


def bind(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "binding needs a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def verify_binding(binding, base=Path(".")):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"},
            "invalid file binding")
    path = Path(binding["path"])
    if not path.is_absolute():
        path = base / path
    actual = bind(path)
    require(actual["sha256"] == binding["sha256"], "input hash changed: " + str(path))
    return path


def dump_new(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


@dataclass
class IdentityMesh:
    xy: np.ndarray
    depth: np.ndarray
    triangles: np.ndarray
    material: np.ndarray
    node_lines: tuple[str, ...]
    element_lines: tuple[str, ...]
    other_lines: tuple[str, ...]

    @classmethod
    def read(cls, path):
        nodes, elements, other = {}, {}, []
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            tag = parts[0] if parts else ""
            if tag in {"ND", "E3T"}:
                count = 5 if tag == "ND" else 6
                require(len(parts) == count, "unsupported entity record")
                ident = int(parts[1])
                table = nodes if tag == "ND" else elements
                require(ident > 0 and ident not in table, "duplicate/invalid entity ID")
                table[ident] = (parts, line)
            else:
                require(tag in {"", "MESH2D", "MESHNAME", "NS"} or tag.startswith("#"),
                        "unhandled SMS record: " + tag)
                other.append(line)
        require(nodes and elements, "empty mesh")
        for table in (nodes, elements):
            require(sorted(table) == list(range(1, len(table) + 1)), "noncontiguous IDs")
        n = [nodes[i] for i in range(1, len(nodes) + 1)]
        e = [elements[i] for i in range(1, len(elements) + 1)]
        xyz = np.asarray([[float(x) for x in p[2:]] for p, _ in n])
        tri = np.asarray([[int(x) - 1 for x in p[2:5]] for p, _ in e])
        require(np.isfinite(xyz).all() and (xyz[:, 2] > 0).all(), "invalid coordinates/depth")
        require(tri.min() >= 0 and tri.max() < len(n), "invalid connectivity range")
        return cls(xyz[:, :2], xyz[:, 2], tri,
                   np.asarray([int(p[5]) for p, _ in e]),
                   tuple(line for _, line in n), tuple(line for _, line in e), tuple(other))

    def projected(self, source_crs, metric_crs):
        metric = CRS.from_user_input(metric_crs)
        require(metric.is_projected and all(abs(a.unit_conversion_factor - 1) < 1e-12
                for a in metric.axis_info), "metric CRS must use metres")
        transformer = Transformer.from_crs(source_crs, metric, always_xy=True)
        result = np.column_stack(transformer.transform(*self.xy.T))
        require(np.isfinite(result).all(), "projection failed")
        return result

    def write(self, path):
        # Keep nonentity records, including complete NS ordering, verbatim.
        # Entity lines are already keyed by contiguous IDs and retain materials.
        split = next((i for i, line in enumerate(self.other_lines)
                      if line.split() and line.split()[0] == "NS"), len(self.other_lines))
        before, after = self.other_lines[:split], self.other_lines[split:]
        with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join([*before, *self.element_lines, *self.node_lines, *after]) + "\n")


def mesh_domain(xy, triangles):
    edges, incidence = unique_edges(triangles)
    require((incidence <= 2).all(), "nonmanifold source")
    loops = order_boundary_loops(xy, edges[incidence == 1])
    wet = Polygon(xy[list(loops[0])], [xy[list(x)] for x in loops[1:]])
    require(wet.is_valid and wet.area > 0, "invalid wet domain")
    return wet, loops, edges, incidence


def select_patch(xy, triangles, ring_xy, buffer_m, protected_nodes=()):
    """Select intersecting elements, including a hole containing no old vertex."""
    ring = np.asarray(ring_xy, dtype=float)
    require(ring.ndim == 2 and ring.shape[1] == 2 and len(ring) >= 3
            and np.isfinite(ring).all(), "invalid island coordinates")
    require(len(np.unique(ring, axis=0)) == len(ring), "ring must be open and unique")
    island = Polygon(ring)
    require(island.is_valid and island.area > 0, "invalid island ring")
    wet, _, _, _ = mesh_domain(xy, triangles)
    require(wet.contains(island) and wet.boundary.disjoint(island),
            "island must be strictly inside existing wet domain")
    require(np.isfinite(buffer_m) and buffer_m > 0, "invalid patch buffer")
    polygons = shapely.polygons(xy[triangles])
    selection = shapely.intersects(polygons, island.buffer(buffer_m))
    patch = classify_patch(xy, triangles, selection)
    polygon = Polygon(xy[list(patch.loops[0])], [xy[list(x)] for x in patch.loops[1:]])
    require(polygon.is_valid and polygon.contains(island)
            and polygon.boundary.disjoint(island), "island touches patch boundary")
    interior = np.setdiff1d(patch.selected_nodes, np.unique(patch.boundary_edges))
    require(not np.intersect1d(interior, np.asarray(protected_nodes, dtype=int)).size,
            "patch would replace protected nodes")
    return patch, polygon.difference(island), interior


def incumbent_sizes(xy, triangles):
    """Explicit geometric proxy for an unavailable historical target field."""
    edges, _ = unique_edges(triangles)
    lengths = np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1)
    values = [[] for _ in xy]
    for (a, b), length in zip(edges, lengths):
        values[a].append(length)
        values[b].append(length)
    require(all(values), "unused source node")
    return np.asarray([np.median(v) for v in values])


class WetSizeField:
    """Visibility-graph lower envelope, with a continuous visible-cone extension.

    This is an exact envelope on its finite graph, not a claim of exact continuous
    geodesics. All obstacle vertices are graph points. No cone crosses a land
    segment. Source samples and the polygon distance retain the continuous core.
    """
    def __init__(self, wet, sample_xy, sample_h, ring_xy, core_h, gradation):
        require(0 < gradation <= .10 and core_h > 0, "invalid size intent")
        self.wet = wet
        self.visible_domain = wet.buffer(1e-7)
        self.xy = np.vstack((sample_xy, ring_xy))
        self.gradation = float(gradation)
        self.core_h = float(core_h)
        self.island = Polygon(ring_xy)
        sample_h = np.asarray(sample_h, dtype=float)
        require(wet.is_valid and np.isfinite(self.xy).all()
                and sample_h.shape == (len(sample_xy),) and np.isfinite(sample_h).all()
                and (sample_h > 0).all(), "invalid field inputs")
        self.incumbent = {tuple(p): float(h) for p, h in zip(sample_xy, sample_h)}
        require(shapely.covers(self.visible_domain, shapely.points(self.xy)).all(),
                "field sample outside wet domain")
        source_points = shapely.points(sample_xy)
        distance = shapely.distance(source_points, self.island)
        visible_core = shapely.covers(self.visible_domain, shapely.shortest_line(source_points, self.island))
        core_cost = np.where(visible_core, core_h + gradation * distance, np.inf)
        initial = np.r_[np.minimum(sample_h, core_cost),
                        np.full(len(ring_xy), core_h)]
        rows = []
        # Bounded local graph; callers must not supply a full regional grid.
        require(len(self.xy) <= 5000, "visibility graph exceeds local sample cap")
        for a in range(len(self.xy) - 1):
            b = np.arange(a + 1, len(self.xy))
            lines = shapely.linestrings(np.stack((np.repeat(self.xy[a:a+1], len(b), axis=0), self.xy[b]), axis=1))
            b = b[shapely.covers(self.visible_domain, lines)]
            rows.extend((a, int(i)) for i in b)
        self.edges = np.asarray(rows, dtype=int).reshape(-1, 2)
        self.values = graph_lower_envelope(len(self.xy), self.edges, self.xy, initial, gradation)

    def __call__(self, x, y):
        point = np.array([float(x), float(y)])
        require(self.visible_domain.covers(Point(point)), "size query outside wet domain")
        distance = np.linalg.norm(self.xy - point, axis=1)
        costs = self.values + self.gradation * distance
        query = Point(point)
        core_cost = np.inf
        core_distance = query.distance(self.island)
        if core_distance <= 1e-8 or self.visible_domain.covers(shapely.shortest_line(query, self.island)):
            core_cost = self.core_h + self.gradation * core_distance
        # Every first visible sample is a valid graph-plus-visible-segment path.
        for i in np.argsort(costs, kind="stable"):
            if distance[i] <= 1e-8 or self.visible_domain.covers(LineString([point, self.xy[i]])):
                return float(min(costs[i], core_cost))
        raise ValueError("size graph has no visible sample")

    def values_at(self, points):
        return np.asarray([self(*p) for p in points])


def generate_patch(source_xy, patch, ring_xy, field, msh_path):
    from fvcom_grid_generation.gmsh_backend import (
        BoundaryLoopGeometry, GmshGeometry, GmshMeshingConfig, generate_gmsh_mesh)
    seam = np.unique(patch.stitch_edges)
    if len(seam):
        old_h = np.asarray([field.incumbent[tuple(source_xy[n])] for n in seam])
        new_h = field.values_at(source_xy[seam])
        ratios = np.maximum(old_h / new_h, new_h / old_h)
        require(np.quantile(ratios, .95) <= 1.5 and ratios.max() <= 2,
                "stitch size field failed; expand bounded patch")
    loops = [BoundaryLoopGeometry(f"retained_{i}", source_xy[list(loop)],
                role="exterior" if i == 0 else "island",
                source_vertex_ids=tuple(f"source:{n}" for n in loop))
             for i, loop in enumerate(patch.loops)]
    loops.append(BoundaryLoopGeometry("inserted_island", ring_xy,
        source_vertex_ids=tuple(f"island:{i}" for i in range(len(ring_xy)))))
    result = generate_gmsh_mesh(GmshGeometry(loops[0], tuple(loops[1:])),
        GmshMeshingConfig(uniform_target_m=field.core_h, algorithm=6, random_seed=1,
            smoothing_steps=8, preserve_source_boundary_discretization=True,
            canonical_size_callback=field), msh_path)
    fixed = {int(result.source_vertex_node_ids[f"source:{n}"]) - 1: n
             for loop in patch.loops for n in loop}
    require(len(fixed) == len(np.unique(patch.boundary_edges)), "seam identity incomplete")
    require(all(np.linalg.norm(result.nodes_xy[a] - source_xy[b]) <= 1e-7
                for a, b in fixed.items()), "mesher moved source boundary")
    return result, fixed


def merge_patch(source, source_xy, patch, patch_xy, patch_triangles, fixed,
                source_crs, metric_crs, sample_new_depth):
    """Deterministic replacement with exact retained records; fail on ID shrinkage."""
    removed = np.setdiff1d(patch.selected_nodes, np.unique(patch.boundary_edges))
    new_local = np.setdiff1d(np.arange(len(patch_xy)), np.asarray(list(fixed), dtype=int))
    assigned = deterministic_new_ids(len(source.xy), removed, patch_xy[new_local])
    mapping = np.empty(len(patch_xy), dtype=int)
    for a, b in fixed.items():
        mapping[a] = b
    mapping[new_local] = assigned
    triangles = mapping[np.asarray(patch_triangles, dtype=int)]
    # New elements sort by sorted GLOBAL connectivity, with CCW order retained.
    require((signed_areas(patch_xy, patch_triangles) > 0).all(), "nonpositive patch")
    vacant_elements = np.flatnonzero(patch.selection)
    require(len(triangles) >= len(vacant_elements), "new patch has fewer elements than vacated IDs")
    order = np.lexsort(np.sort(triangles, axis=1).T[::-1])
    eid = np.r_[vacant_elements, np.arange(len(source.triangles),
        len(source.triangles) + len(triangles) - len(vacant_elements))]
    transform = Transformer.from_crs(metric_crs, source_crs, always_xy=True)
    serialized_xy = np.column_stack(transform.transform(*patch_xy[new_local].T))
    serialized_xy = np.asarray([[float(f"{v:.12f}") for v in row] for row in serialized_xy])
    depth = np.asarray(sample_new_depth(serialized_xy), dtype=float)
    require(depth.shape == (len(new_local),) and np.isfinite(depth).all() and (depth > 0).all(),
            "new-node bathymetry invalid or uncovered")
    nodes = list(source.node_lines) + [""] * (len(new_local) - len(removed))
    for ident, xy, h in zip(assigned, serialized_xy, depth):
        require(float(f"{h:.4f}") > 0, "serialized depth nonpositive")
        nodes[ident] = f"ND {ident+1} {xy[0]:.12f} {xy[1]:.12f} {h:.4f}"
    material = np.unique(source.material[patch.selection])
    require(len(material) == 1, "patch spans material interfaces; explicit mapping required")
    elements = list(source.element_lines) + [""] * (len(triangles) - len(vacant_elements))
    for ident, row in zip(eid, triangles[order]):
        elements[ident] = f"E3T {ident+1} {row[0]+1} {row[1]+1} {row[2]+1} {material[0]}"
    return nodes, elements, {"replaced_source_node_ids": (removed + 1).tolist(),
        "replaced_source_element_ids": (vacant_elements + 1).tolist(),
        "new_node_ids": (assigned + 1).tolist(), "new_element_ids": (eid + 1).tolist(),
        "local_to_global_node_ids": (mapping + 1).tolist()}


def audit_transaction(source_path, candidate_path, source_crs, metric_crs,
                      ring_xy, selection, protected_nodes=(), area_tolerance_m2=.01):
    """Independent full-serialized structural certificate (not runtime readiness)."""
    old, new = IdentityMesh.read(source_path), IdentityMesh.read(candidate_path)
    xy, nxt = old.projected(source_crs, metric_crs), new.projected(source_crs, metric_crs)
    require(len(new.xy) >= len(old.xy) and len(new.triangles) >= len(old.triangles), "ID shrinkage")
    selection = np.asarray(selection)
    require(selection.dtype == bool and selection.shape == (len(old.triangles),), "invalid selection")
    patch = classify_patch(xy, old.triangles, selection)
    require(np.array_equal(patch.selection, selection), "selection lacks topology closure")
    replaced = np.setdiff1d(patch.selected_nodes, np.unique(patch.boundary_edges))
    retained = np.setdiff1d(np.arange(len(old.xy)), replaced)
    require(not np.intersect1d(replaced, np.asarray(protected_nodes, dtype=int)).size,
            "protected node replaced")
    require(all(old.node_lines[i] == new.node_lines[i] for i in retained), "retained node record changed")
    require(all(old.element_lines[i] == new.element_lines[i] for i in np.flatnonzero(~selection)),
            "outside element record changed")
    require(old.other_lines == new.other_lines, "header/OBC records changed")
    require((signed_areas(xy, old.triangles) > 0).all(), "source orientation invalid")
    require((signed_areas(nxt, new.triangles) > 0).all(), "serialized nonpositive elements")
    require(len(np.unique(new.triangles)) == len(nxt), "unused delivered nodes")
    require(len(np.unique(nxt, axis=0)) == len(nxt), "duplicate delivered coordinates")
    require(len(np.unique(np.sort(new.triangles, axis=1), axis=0)) == len(new.triangles), "duplicate elements")
    oldwet, oldloops, oldedges, oldcounts = mesh_domain(xy, old.triangles)
    newwet, newloops, edges, counts = mesh_domain(nxt, new.triangles)
    island = Polygon(ring_xy)
    require(island.is_valid and oldwet.contains(island) and oldwet.boundary.disjoint(island), "invalid source island")
    expected = oldwet.difference(island)
    require(area_tolerance_m2 > 0 and np.isfinite(area_tolerance_m2), "invalid area tolerance")
    require(newwet.symmetric_difference(expected).area <= area_tolerance_m2, "unexpected wet-domain change")
    require(len(newloops) == len(oldloops) + 1, "wrong boundary-loop delta")
    oldboundary = {tuple(e) for e in oldedges[oldcounts == 1]}
    boundary = {tuple(e) for e in edges[counts == 1]}
    require(oldboundary <= boundary, "original physical chord changed")
    expected_new = boundary - oldboundary
    extra_loops = order_boundary_loops(nxt, np.asarray(sorted(expected_new), dtype=int))
    require(len(extra_loops) == 1, "new boundary must be exactly one ring")
    ring_nodes = np.asarray(extra_loops[0], dtype=int)
    require(len(ring_nodes) == len(ring_xy), "new ring discretization changed")
    expected_ring = Polygon(nxt[ring_nodes])
    require(expected_ring.boundary.hausdorff_distance(island.boundary) <= 1e-5,
            "new island chords differ from frozen ring")
    incidence = {tuple(edge): int(count) for edge, count in zip(edges, counts)}
    require(all(incidence.get(tuple(edge)) == 2 for edge in patch.stitch_edges), "stitch incidence changed")
    graph = coo_matrix((np.ones(len(edges) * 2),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(nxt), len(nxt)))
    components = connected_components(graph, directed=False, return_labels=False)
    require(components == 1, "disconnected wet mesh")
    oldeuler = len(xy) - len(oldedges) + len(old.triangles)
    euler = len(nxt) - len(edges) + len(new.triangles)
    require(euler == oldeuler - 1, "wrong Euler characteristic delta")
    changed_ids = np.r_[np.flatnonzero(selection), np.arange(len(old.triangles), len(new.triangles))]
    polygons = shapely.polygons(nxt[new.triangles[changed_ids]])
    patch_domain = Polygon(xy[list(patch.loops[0])], [xy[list(x)] for x in patch.loops[1:]])
    patch_expected = patch_domain.difference(island)
    union = shapely.union_all(polygons)
    # Sum-vs-union catches folded/overlapping triangles; union-vs-target catches gaps.
    overlap = float(shapely.area(polygons).sum() - union.area)
    require(abs(overlap) <= area_tolerance_m2, "overlapping patch triangles")
    require(union.symmetric_difference(patch_expected).area <= area_tolerance_m2,
            "patch triangles leave gaps, cover land or escape patch")
    require(shapely.area(shapely.intersection(polygons, island)).sum() <= area_tolerance_m2,
            "wet triangles overlap island")
    require(not shapely.contains_xy(island.buffer(-1e-5), nxt[:, 0], nxt[:, 1]).any(), "active node inside land")
    require(np.all(new.material[changed_ids] == old.material[np.flatnonzero(selection)[0]])
            and len(np.unique(old.material[selection])) == 1, "patch material changed")
    return {"schema_version": "fvcom_single_interior_island_structural_audit_v1",
        "passed": True, "source_mesh": bind(source_path), "candidate_mesh": bind(candidate_path),
        "retained_nodes_checked": len(retained), "retained_elements_checked": int((~selection).sum()),
        "boundary_loops_before": len(oldloops), "boundary_loops_after": len(newloops),
        "wet_components": int(components), "euler_delta": int(euler - oldeuler),
        "intended_land_area_m2": island.area, "wet_area_loss_m2": oldwet.area - newwet.area,
        "patch_overlap_m2": overlap, "area_tolerance_m2": area_tolerance_m2,
        "added_island_node_ids": (ring_nodes + 1).tolist(),
        "complete_boundary_loops_zero_based": [list(loop) for loop in newloops],
        "benchmark_baseline_ready": False, "submission_eligible": False,
        "remaining_gates": ["owning_quality", "source_bound_TGE", "roundtrip", "bathymetry_provenance",
                            "delivery_contract", "preconfiguration", "forcing_rebind"]}


def validate_request(path):
    """Read a separate, strict island request; old refinement requests stay intact."""
    path = Path(path)
    request = json.loads(path.read_text(encoding="utf-8-sig"))
    require(set(request) == {"schema_version", "inputs", "source_crs", "metric_crs",
        "buffer_candidates_m", "core_target_m", "gradation", "maximum_patch_nodes",
        "area_tolerance_m2", "scientific_definition"}, "unknown or missing island request fields")
    require(request["schema_version"] == "fvcom_single_interior_island_request_v1", "wrong request mode")
    required = {"source_mesh", "source_grid_contract", "source_certificate", "island_ring",
        "source_evidence", "bathymetry_receipt", "protected_contract", "tge_source", "quality_policy"}
    require(isinstance(request["inputs"], dict) and set(request["inputs"]) == required,
            "island request binding inventory differs")
    paths = {k: verify_binding(v, path.parent) for k, v in request["inputs"].items()}
    require(isinstance(request["scientific_definition"], str) and request["scientific_definition"].strip(),
            "missing explicit scientific land definition")
    require(finite_number(request["gradation"]) and finite_number(request["core_target_m"])
            and 0 < request["gradation"] <= .10 and request["core_target_m"] > 0,
            "invalid core target or gradation")
    buffers = request["buffer_candidates_m"]
    require(isinstance(buffers, list) and 1 <= len(buffers) <= 3
            and all(finite_number(x) for x in buffers) and min(buffers) > 0
            and buffers == sorted(set(buffers)), "use one to three increasing bounded patch buffers")
    cap = request["maximum_patch_nodes"]
    require(type(cap) is int and 0 < cap <= 1000000, "invalid patch node cap")
    require(finite_number(request["area_tolerance_m2"]) and 0 < request["area_tolerance_m2"] <= .1,
            "area tolerance must be positive and at most0.1m2")
    source = IdentityMesh.read(paths["source_mesh"])
    xy = source.projected(request["source_crs"], request["metric_crs"])
    ring = json.loads(paths["island_ring"].read_text(encoding="utf-8-sig"))
    require(set(ring) == {"crs", "coordinates"} and CRS(ring["crs"]) == CRS(request["metric_crs"]),
            "ring must declare exact metric CRS and coordinates")
    protected = json.loads(paths["protected_contract"].read_text(encoding="utf-8-sig"))
    require(set(protected) == {"protected_node_ids", "open_boundaries"}, "invalid protected contract")
    ids = protected["protected_node_ids"]
    require(isinstance(ids, list) and all(type(x) is int and 1 <= x <= len(xy) for x in ids)
            and len(ids) == len(set(ids)), "invalid protected IDs")
    contract = json.loads(paths["source_grid_contract"].read_text(encoding="utf-8-sig"))
    require(contract.get("schema_version") == "fvcom_grid_delivery_contract_v1", "invalid parent grid contract")
    mesh_binding = {k: contract["mesh"][k] for k in ("path", "sha256")}
    require(verify_binding(mesh_binding, paths["source_grid_contract"].parent).resolve()
            == paths["source_mesh"].resolve(), "source mesh does not match parent contract")
    metadata = contract["mesh"]
    require(metadata["node_count"] == len(source.xy) and metadata["triangle_count"] == len(source.triangles)
            and CRS(metadata["coordinate_crs"]) == CRS(request["source_crs"])
            and CRS(metadata["horizontal_crs"]) == CRS(request["metric_crs"])
            and metadata["horizontal_units"] == "m", "parent mesh metadata differs")
    require(contract["open_boundaries"] == protected["open_boundaries"], "protected OBC contract differs")
    from fvcom_grid_generation.sms_2dm import read_2dm
    from fvcom_grid_generation.quality_policy import load_quality_policy, policy_path
    sms = read_2dm(paths["source_mesh"])
    boundaries = protected["open_boundaries"]
    require([list(row["node_ids"]) for row in boundaries]
            == [row.tolist() for row in sms.open_boundary_chains]
            and [row["nodestring_id"] for row in boundaries] == list(sms.open_boundary_ids),
            "source SMS OBC identity differs")
    require(all(type(row["cyclic"]) is bool and set(row["node_ids"]) <= set(ids)
                for row in boundaries), "unprotected OBC or unspecified cyclicity")
    certificate = json.loads(paths["source_certificate"].read_text(encoding="utf-8-sig"))
    require(certificate.get("schema_version") == "fvcom_retained_grid_join_certificate_v1"
            and certificate.get("audit", {}).get("passed") is True,
            "this repair route requires a passed retained parent certificate")
    deps = certificate["audit"]["dependencies"]
    for key, path_key in (("terminal_mesh", "source_mesh"), ("grid_contract", "source_grid_contract"),
                          ("tge_source", "tge_source")):
        require(deps[key]["sha256"] == bind(paths[path_key])["sha256"], "parent certificate binding differs: " + key)
    load_quality_policy(paths["quality_policy"])
    require(bind(paths["quality_policy"])["sha256"] == bind(policy_path())["sha256"],
            "quality policy differs from active owner")
    return request, paths, source, xy, np.asarray(ring["coordinates"], dtype=float), protected


def audit_quality_and_tge(candidate_path, source_crs, metric_crs, obc_chains,
                          obc_cyclic, tge_source_path, output_directory):
    """Recompute owning gates; do not trust a supplied passing report."""
    from fvcom_grid_generation.quality import evaluate_mesh_quality
    from fvcom_grid_generation.tge_topology import audit_tge_boundary_junctions
    candidate = IdentityMesh.read(candidate_path)
    xy = candidate.projected(source_crs, metric_crs)
    _, loops, _, _ = mesh_domain(xy, candidate.triangles)
    require((signed_areas(xy, candidate.triangles) > 0).all(), "nonpositive serialized mesh")
    chains = [list(map(int, chain)) for chain in obc_chains]
    require(len(chains) == len(obc_cyclic) and all(type(v) is bool for v in obc_cyclic),
            "OBC cyclicity must be explicit")
    # Independently parse SMS ordered nodestrings using the owning parser.
    from fvcom_grid_generation.sms_2dm import read_2dm
    sms = read_2dm(candidate_path)
    require([chain.tolist() for chain in sms.open_boundary_chains] == chains,
            "serialized OBC order differs from supplied contract")
    quality = evaluate_mesh_quality(xy, candidate.depth, candidate.triangles + 1,
        np.asarray([n for chain in chains for n in chain], dtype=int),
        {"boundary_constraint_recovered": True}, constraint_chains=[list(x) for x in loops],
        open_boundary_chains=chains, open_boundary_cyclic=obc_cyclic,
        require_open_boundary=bool(chains), expected_open_boundary_count=len(chains),
        enforce_no_unused_nodes=True)
    tge = audit_tge_boundary_junctions(len(xy), candidate.triangles,
        [[n - 1 for n in chain] for chain in chains], tge_source_path=tge_source_path)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    roundtrip = directory / "roundtrip.2dm"
    candidate.write(roundtrip)
    reread = IdentityMesh.read(roundtrip)
    exact = (candidate.node_lines == reread.node_lines
        and candidate.element_lines == reread.element_lines and candidate.other_lines == reread.other_lines)
    require(exact and bind(candidate_path)["sha256"] == bind(roundtrip)["sha256"], "roundtrip not exact")
    dump_new(directory / "quality.json", quality)
    dump_new(directory / "tge.json", tge)
    return {"quality": bind(directory / "quality.json"), "tge": bind(directory / "tge.json"),
        "roundtrip": bind(roundtrip), "roundtrip_exact": True,
        "benchmark_baseline_ready": bool(quality["benchmark_grid_baseline_ready"] and tge["passed"]),
        "submission_eligible": False}


def certify_geometry(request_path, candidate_path, field_npz, raw_patch_npz,
                     numbering_path, sample_new_depth, selected_buffer_m,
                     evidence, output_directory):
    """Replay an unconditioned single-island transaction before geometry admission.

    The caller binds scientific/visual/boundary/sampler-code review receipts in
    evidence. Its sampler must be the actual declared, verified owning adapter.
    This v1 replays RAW Gmsh only: conditioned candidates require a separately
    audited conditioning replay, never an equality waiver here.
    """
    request, paths, source, xy, ring, protected = validate_request(request_path)
    require(selected_buffer_m in request['buffer_candidates_m'], "undeclared patch buffer")
    require(isinstance(evidence, dict) and {'source_review', 'physical_review', 'boundary_review',
        'boundary_sidecar', 'sampler_code', 'independent_transaction_review'} <= set(evidence),
        "geometry certificate review inventory incomplete")
    evidence = {key: bind(verify_binding(value, Path(request_path).parent))
                for key, value in evidence.items()}
    patch, wet, removed = select_patch(xy, source.triangles, ring, selected_buffer_m,
                                      np.asarray(protected['protected_node_ids']) - 1)
    sample = patch.selected_nodes
    sample = sample[~shapely.contains_xy(Polygon(ring), xy[sample, 0], xy[sample, 1])]
    incumbent = incumbent_sizes(xy, source.triangles)
    field = WetSizeField(wet, xy[sample], incumbent[sample], ring,
                         request['core_target_m'], request['gradation'])
    with np.load(field_npz, allow_pickle=False) as frozen:
        for name, expected in {'selection': patch.selection, 'source_xy': xy,
            'selected_source_nodes': patch.selected_nodes, 'removed': removed, 'sample': sample,
            'ring': ring, 'graph_xy': field.xy, 'graph_h': field.values, 'graph_edges': field.edges,
            'stitch_edges': patch.stitch_edges, 'physical_edges': patch.physical_edges}.items():
            require(np.array_equal(frozen[name], expected), 'frozen patch/field differs: ' + name)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    regenerated, fixed = generate_patch(xy, patch, ring, field, output / 'replayed_patch.msh')
    require(len(regenerated.nodes_xy) <= request['maximum_patch_nodes'], 'patch node cap exceeded')
    with np.load(raw_patch_npz, allow_pickle=False) as raw:
        require(np.array_equal(raw['xy'], regenerated.nodes_xy)
                and np.array_equal(raw['triangles'], regenerated.triangles - 1),
                'Gmsh RAW replay differs; no conditioned/reordered substitution')
    nodes, elements, lineage = merge_patch(source, xy, patch, regenerated.nodes_xy,
        regenerated.triangles - 1, fixed, request['source_crs'], request['metric_crs'], sample_new_depth)
    require(json.loads(Path(numbering_path).read_text(encoding='utf-8-sig')) == lineage,
            'deterministic numbering replay differs')
    replay_mesh = output / 'replayed_mesh.2dm'
    IdentityMesh(source.xy, source.depth, source.triangles, source.material,
        tuple(nodes), tuple(elements), source.other_lines).write(replay_mesh)
    require(bind(replay_mesh)['sha256'] == bind(candidate_path)['sha256'],
            'full serialized generator/merge/bathymetry replay differs')
    structure = audit_transaction(paths['source_mesh'], candidate_path, request['source_crs'],
        request['metric_crs'], ring, patch.selection, np.asarray(protected['protected_node_ids']) - 1,
        request['area_tolerance_m2'])
    dump_new(output / 'structural_audit.json', structure)
    source_contract = json.loads(paths['source_grid_contract'].read_text(encoding='utf-8-sig'))
    old_sidecar = verify_binding(source_contract['delivery_artifacts']['boundary_nodes'], paths['source_grid_contract'].parent)
    boundary_audit = audit_boundary_sidecar(old_sidecar, verify_binding(evidence['boundary_sidecar']),
        candidate_path, request['source_crs'], request['metric_crs'], protected['open_boundaries'],
        structure['added_island_node_ids'], paths['island_ring'], paths['source_evidence'])
    dump_new(output / 'boundary_audit.json', boundary_audit)
    gates = audit_quality_and_tge(candidate_path, request['source_crs'], request['metric_crs'],
        [row['node_ids'] for row in protected['open_boundaries']],
        [row['cyclic'] for row in protected['open_boundaries']], paths['tge_source'], output / 'owning_audits')
    require(gates['benchmark_baseline_ready'], 'replayed full quality/TGE baseline failed')
    certificate = {'schema_version': 'fvcom_single_interior_island_geometry_certificate_v1',
        'candidate_mesh': bind(candidate_path), 'request': bind(request_path),
        'audit': {'passed': True, 'scope': 'retained_source_plus_single_interior_Gmsh6_island_patch',
            'source_inputs': {k: bind(v) for k, v in paths.items()},
            'field': bind(field_npz), 'raw_patch': bind(raw_patch_npz), 'numbering': bind(numbering_path),
            'selected_buffer_m': selected_buffer_m, 'structural_audit': bind(output / 'structural_audit.json'),
            'boundary_audit': bind(output / 'boundary_audit.json'),
            'replayed_mesh': bind(replay_mesh), 'replayed_msh': bind(output / 'replayed_patch.msh'),
            'owning_gates': gates, 'evidence': evidence, 'certificate_code': bind(__file__)},
        'benchmark_grid_baseline_ready': True, 'submission_eligible': False,
        'scientific_definition': request['scientific_definition'],
        'remaining_gates': ['derived_delivery_contract', 'preconfiguration', 'grid_edge',
                            'station_mapping', 'exact_OBC_forcing_rebind', 'runtime_controls_and_stages']}
    dump_new(output / 'geometry_certificate.json', certificate)
    return certificate


def audit_boundary_sidecar(old_path, new_path, mesh_path, source_crs, metric_crs,
                           open_boundaries, new_island_node_ids, ring_path, source_review_path):
    """Verify actual complete GeoJSON Point inventory and its closed chain edges."""
    import collections
    old = json.loads(Path(old_path).read_text(encoding='utf-8-sig'))
    new = json.loads(Path(new_path).read_text(encoding='utf-8-sig'))
    require(old['type'] == new['type'] == 'FeatureCollection', 'boundary sidecar must be GeoJSON')
    source_features = {f['properties']['node_id_1based']: f for f in old['features']}
    features = {f['properties']['node_id_1based']: f for f in new['features']}
    require(len(source_features) == len(old['features']) and len(features) == len(new['features']),
            'duplicate boundary feature identity')
    require(all(features.get(n) == f for n, f in source_features.items()), 'source boundary feature changed')
    require(set(features) - set(source_features) == set(new_island_node_ids), 'unexpected added boundary inventory')
    mesh = IdentityMesh.read(mesh_path)
    xy = mesh.projected(source_crs, metric_crs)
    _, _, edges, incidence = mesh_domain(xy, mesh.triangles)
    boundary = {tuple(e) for e in edges[incidence == 1]}
    require(set(features) == set((np.unique(edges[incidence == 1]) + 1).tolist()), 'incomplete boundary node coverage')
    chains = collections.defaultdict(list)
    transformer = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
    declared = {n: [] for n in features}
    for row in open_boundaries:
        for node in row['node_ids']:
            require(node in declared, 'OBC node absent from boundary sidecar')
            declared[node].append((row['nodestring_id'], row['cyclic']))
    maximum_error = 0.
    for node, feature in features.items():
        prop, geometry = feature['properties'], feature['geometry']
        require(type(node) is int and prop['node_index_zero_based'] == node - 1, 'boundary node indexing mismatch')
        require(geometry['type'] == 'Point' and len(geometry['coordinates']) == 2, 'boundary feature must be point')
        mapped = np.array(transformer.transform(*geometry['coordinates']))
        error = float(np.linalg.norm(mapped - xy[node - 1]))
        require(np.isfinite(error) and error <= .01, 'boundary coordinate differs from serialized mesh')
        maximum_error = max(maximum_error, error)
        require(type(prop['is_open_boundary']) is bool and prop['is_open_boundary'] == bool(declared[node]),
                'boundary open/solid role differs')
        require(prop['open_boundary_nodestring_ids'] == [x[0] for x in declared[node]]
            and prop['open_boundary_cyclic'] == [x[1] for x in declared[node]], 'boundary OBC identity/cyclicity differs')
        position = prop['constraint_chain_position']
        require(type(position) is int and position >= 0, 'invalid boundary chain position')
        chains[prop['constraint_chain_id']].append((position, node - 1))
        if node in new_island_node_ids:
            require(prop.get('source_type') == 'supplied_interior_island'
                    and prop.get('source_node_id_1based') in (None, 0)
                    and prop.get('source_node_index_zero_based') is None,
                    'new boundary mislabelled as retained source')
            require(prop.get('boundary_kind') == 'island' and prop.get('is_hard_anchor') is True,
                    'new boundary must be a locked island')
            require(verify_binding(prop['source_geometry'], Path(new_path).parent).resolve() == Path(ring_path).resolve()
                and verify_binding(prop['source_review'], Path(new_path).parent).resolve() == Path(source_review_path).resolve(),
                'new island source binding differs')
    covered = []
    for rows in chains.values():
        rows.sort()
        require([p for p, _ in rows] == list(range(len(rows))), 'boundary chain positions incomplete/duplicated')
        require(len(rows) >= 3, 'boundary loop too short')
        nodes = [n for _, n in rows]
        covered.extend(tuple(sorted((a, nodes[(i + 1) % len(nodes)]))) for i, a in enumerate(nodes))
    require(len(covered) == len(set(covered)) and set(covered) == boundary, 'boundary chords not covered exactly once')
    return {'schema_version': 'fvcom_interior_island_boundary_sidecar_audit_v1', 'passed': True,
        'source_sidecar': bind(old_path), 'delivered_sidecar': bind(new_path), 'mesh': bind(mesh_path),
        'source_features_preserved': len(source_features), 'new_island_nodes': len(new_island_node_ids),
        'complete_boundary_edges': len(boundary), 'maximum_coordinate_error_m': maximum_error}
