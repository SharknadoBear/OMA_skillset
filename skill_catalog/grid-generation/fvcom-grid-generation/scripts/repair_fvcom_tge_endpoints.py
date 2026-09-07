#!/usr/bin/env python3
"""Bounded, source-bound reclassification of diagnosed FVCOM OBC endpoints.

Grid-owned repair of OBC node roles. Mesh geometry remains exact.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

from fvcom_grid_generation.quality import evaluate_mesh_quality
from fvcom_grid_generation.sms_2dm import read_2dm
from fvcom_grid_generation.tge_topology import audit_tge_boundary_junctions


class RepairRejected(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def require(condition, code, detail):
    if not condition:
        raise RepairRejected(code, detail)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order_sha(nodes):
    return hashlib.sha256(json.dumps(list(map(int, nodes)), separators=(",", ":")).encode("ascii")).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_nodestrings(raw, replacements):
    """Preserve all non-NS bytes and untouched NS blocks, including plural IDs."""
    lines = raw.splitlines(keepends=True)
    output, seen = [], set()
    index = 0
    while index < len(lines):
        if not lines[index].lstrip().upper().startswith(b"NS "):
            output.append(lines[index])
            index += 1
            continue
        block, nodes, ns_id = [], [], None
        while index < len(lines) and lines[index].lstrip().upper().startswith(b"NS "):
            line = lines[index]
            block.append(line)
            values = list(map(int, line.split()[1:]))
            index += 1
            for pos, value in enumerate(values):
                if value < 0:
                    require(pos + 2 == len(values), "nodestring_id_missing_or_ambiguous", "NS terminator requires one explicit ID")
                    nodes.append(-value)
                    ns_id = values[pos + 1]
                    break
                nodes.append(value)
            if ns_id is not None:
                break
        require(ns_id is not None and ns_id not in seen, "invalid_nodestring_serialization", "Unterminated or duplicate NS block")
        seen.add(ns_id)
        if ns_id not in replacements:
            output.extend(block)
            continue
        old, new = replacements[ns_id]
        require(nodes == old, "nodestring_parser_disagreement", f"NS {ns_id} does not match validated order")
        newline = b"\r\n" if block[-1].endswith(b"\r\n") else b"\n"
        for start in range(0, len(new), 10):
            chunk = new[start:start + 10]
            if start + 10 >= len(new):
                chunk = chunk[:-1] + [-chunk[-1], ns_id]
            output.append(b"NS " + " ".join(map(str, chunk)).encode("ascii") + newline)
    require(set(replacements) <= seen, "missing_nodestring", "Requested NS was not serialized")
    return b"".join(output)


def repair(grid_contract, boundary_nodes, obc_remap, tge_source, output_dir):
    paths = {"grid_contract": Path(grid_contract).resolve(), "boundary_nodes": Path(boundary_nodes).resolve(),
        "obc_remap": Path(obc_remap).resolve(), "tge_source": Path(tge_source).resolve()}
    output = Path(output_dir).resolve()
    require(not output.exists(), "output_exists", "Use a new immutable output directory")
    output.mkdir(parents=True)
    result = {"schema_version": "fvcom_tge_endpoint_repair_v1", "status": "blocked", "changed": False,
        "input_paths": {key: str(path) for key, path in paths.items()}, "input_hashes": {}, "blocking_reasons": []}
    try:
        result["input_hashes"] = {key: sha(path) for key, path in paths.items()}
        contract = json.loads(paths["grid_contract"].read_text(encoding="utf-8-sig"))
        require(contract.get("schema_version") == "fvcom_grid_delivery_contract_v1", "unsupported_contract", "Expected fvcom_grid_delivery_contract_v1")
        for name, record in contract.get("delivery_artifacts", {}).items():
            bound_path = Path(record["path"])
            if not bound_path.is_absolute():
                bound_path = paths["grid_contract"].parent / bound_path
            require(sha(bound_path) == record["sha256"], "delivery_artifact_hash_mismatch", f"Bound artifact {name} is stale")
            if name in paths:
                require(sha(paths[name]) == record["sha256"], "supplied_sidecar_hash_mismatch", f"Supplied sidecar {name} differs from contract")
        spec = contract["mesh"]
        mesh_path = Path(spec["path"])
        mesh_path = (paths["grid_contract"].parent / mesh_path).resolve() if not mesh_path.is_absolute() else mesh_path.resolve()
        paths["mesh"] = mesh_path
        result["input_paths"]["mesh"] = str(mesh_path)
        result["input_hashes"]["mesh"] = sha(mesh_path)
        require(sha(mesh_path) == spec["sha256"].lower(), "mesh_hash_mismatch", "Contract mesh hash is stale")
        mesh = read_2dm(mesh_path)
        require(len(mesh.nodes_lonlat) == spec["node_count"] and len(mesh.triangles) == spec["triangle_count"], "mesh_count_mismatch", "Contract counts differ")
        raw = mesh_path.read_bytes()
        nd_ids = [int(line.split()[1]) for line in raw.splitlines() if line.startswith(b"ND ")]
        element_ids = [int(line.split()[1]) for line in raw.splitlines() if line.startswith(b"E3T ")]
        require(sorted(nd_ids) == list(range(1, len(nd_ids) + 1)) and element_ids == list(range(1, len(element_ids) + 1)),
            "noncanonical_mesh_ids", "This bounded tool requires unique contiguous node IDs and ordered contiguous element IDs")
        boundary = json.loads(paths["boundary_nodes"].read_text(encoding="utf-8-sig"))
        remap = json.loads(paths["obc_remap"].read_text(encoding="utf-8-sig"))
        declared = contract["open_boundaries"]
        require(len({r["obc_id"] for r in declared}) == len(declared), "duplicate_obc_id", "OBC IDs must be unique")
        all_ns = dict(zip(mesh.open_boundary_ids, [c.tolist() for c in mesh.open_boundary_chains]))
        declared_ids = [int(r["nodestring_id"]) for r in declared]
        require(len(set(declared_ids)) == len(declared_ids), "duplicate_nodestring_id", "OBC NS IDs must be unique")
        require(set(declared_ids) | set(contract.get("excluded_nodestring_ids", [])) == set(all_ns), "undeclared_nodestring", "Every serialized NS must have an explicit role")
        remap_rows = {int(r["nodestring_id"]): r for r in remap["chains"]}
        require(len(remap_rows) == len(remap["chains"]) and set(remap_rows) == set(declared_ids), "sidecar_chain_mismatch", "Remap chains must match OBC NS IDs")
        chains, cyclic, memberships = [], [], collections.defaultdict(list)
        for row in declared:
            ns = int(row["nodestring_id"])
            chain = list(map(int, row["node_ids"]))
            require(all_ns.get(ns) == chain and row["node_order_sha256"] == order_sha(chain), "obc_order_mismatch", f"OBC NS {ns} order or hash is stale")
            require(len(chain) >= 2 and len(set(chain)) == len(chain), "invalid_obc_chain", f"OBC NS {ns} is short or duplicated")
            sidecar = remap_rows[ns]
            require(sidecar["delivered_node_ids_1based"] == chain and sidecar["delivered_node_count"] == len(chain), "sidecar_order_mismatch", f"Remap NS {ns} is stale")
            flag = row.get("cyclic", sidecar.get("cyclic"))
            require(type(flag) is bool and sidecar.get("cyclic") == flag, "unknown_or_conflicting_cyclicity", f"OBC NS {ns} requires consistent explicit cyclicity")
            chains.append(chain)
            cyclic.append(flag)
            for node in chain:
                memberships[node].append(ns)
        require(all(len(v) == 1 for v in memberships.values()), "overlapping_obcs", "Different OBCs cannot share nodes")
        groups, boundary_ids = collections.defaultdict(list), set()
        for feature in boundary["features"]:
            p = feature["properties"]
            node = int(p["node_id_1based"])
            require(node not in boundary_ids and 1 <= node <= len(mesh.nodes_lonlat), "invalid_boundary_node_id", "Boundary node IDs must be unique and valid")
            boundary_ids.add(node)
            require(np.max(np.abs(np.asarray(feature["geometry"]["coordinates"]) - mesh.nodes_lonlat[node-1])) <= 1e-10,
                "boundary_coordinate_mismatch", f"Boundary coordinate differs at node {node}")
            require(bool(p["is_open_boundary"]) == (node in memberships) and p["open_boundary_nodestring_ids"] == memberships.get(node, []),
                "boundary_role_mismatch", f"Boundary role differs at node {node}")
            groups[p["constraint_chain_id"]].append((p["constraint_chain_position"], node - 1))
        constraints = []
        for rows in groups.values():
            require(sorted(pos for pos, _ in rows) == list(range(len(rows))), "invalid_constraint_chain_order", "Constraint positions must be contiguous")
            constraints.append([node for _, node in sorted(rows)])
        xy = mesh.nodes_lonlat
        target_crs = CRS.from_user_input(spec["horizontal_crs"])
        require(target_crs.is_projected and all(abs(axis.unit_conversion_factor - 1) < 1e-12 for axis in target_crs.axis_info),
            "nonmetric_crs", "Horizontal output CRS must be projected in meters")
        if CRS.from_user_input(spec["coordinate_crs"]) != target_crs:
            transform = Transformer.from_crs(spec["coordinate_crs"], target_crs, always_xy=True)
            xy = np.column_stack(transform.transform(xy[:, 0], xy[:, 1]))
        require(np.all(np.isfinite(xy)), "nonfinite_coordinates", "Metric coordinates must be finite")

        def audit(open_chains):
            source_audit = audit_tge_boundary_junctions(len(xy), mesh.triangles-1,
                [np.asarray(c)-1 for c in open_chains], tge_source_path=paths["tge_source"])
            quality = evaluate_mesh_quality(xy, mesh.depths, mesh.triangles, np.empty(0, dtype=int),
                {"boundary_constraint_recovered": True, "source": "validated_delivered_boundary_sidecar"},
                constraint_chains=constraints, open_boundary_chains=open_chains, open_boundary_cyclic=cyclic,
                expected_open_boundary_count=len(open_chains), enforce_no_unused_nodes=True)
            return source_audit, quality

        before, before_quality = audit(chains)
        dump(output / "before_tge_audit.json", before)
        dump(output / "before_mesh_quality.json", before_quality)
        require(before["source_binding"]["passed"], "tge_source_mismatch", "Exact TGE source expressions are missing")
        require(set(before["exterior_node_ids_1based"]) == boundary_ids, "boundary_coverage_mismatch", "Sidecar must cover every and only topological boundary node")
        replacements, reclassified = {}, set()
        if not before["passed"]:
            endpoint_owner = {}
            for i, chain in enumerate(chains):
                for node in (chain[0], chain[-1]):
                    endpoint_owner[node] = i
            for cell in before["fatal_cells"]:
                require(cell["isonb_sum"] == 5 and sorted(cell["isonb_values"]) == [1, 2, 2], "unsupported_tge_cell", "Only endpoint cells with flags 1, 2, 2 may be reclassified")
                open_nodes = set(cell["open_boundary_node_ids_1based"])
                candidates = open_nodes & set(endpoint_owner)
                require(len(candidates) == 1, "ambiguous_or_nonendpoint_fatal_cell", "Fatal cell must identify exactly one OBC endpoint")
                node = next(iter(candidates))
                chain_index = endpoint_owner[node]
                require(not cyclic[chain_index], "cyclic_endpoint_repair_unsupported", "Never trim a cyclic OBC")
                require(open_nodes <= set(chains[chain_index]), "cross_chain_fatal_cell", "Fatal cell crosses distinct OBC chains")
                reclassified.add(node)
            for row, chain in zip(declared, chains):
                retained = [node for node in chain if node not in reclassified]
                require(len(retained) >= 2, "endpoint_trim_too_short", "Repair must retain at least 2 nodes per OBC")
                if retained != chain:
                    replacements[int(row["nodestring_id"])] = (chain, retained)
        other_failures = set(before_quality["failure_taxonomy"]) - {"fvcom_tge_boundary_cell_sum_above_four"}
        require(not other_failures, "preexisting_mesh_failure", f"Unrelated hard mesh failures: {sorted(other_failures)}")
        if not reclassified:
            result.update(status="ready", changed=False, decision="already_source_bound_ready", output_paths=result["input_paths"], output_hashes=result["input_hashes"])
        else:
            revised = [replacements.get(int(row["nodestring_id"]), (chain, chain))[1] for row, chain in zip(declared, chains)]
            after, after_quality = audit(revised)
            dump(output / "after_tge_audit.json", after)
            dump(output / "after_mesh_quality.json", after_quality)
            require(after["passed"] and after_quality["benchmark_grid_baseline_ready"], "postrepair_gate_failure", "Bounded endpoint correction did not pass all global gates")
            candidate_path = output / "fvcom_grid.2dm"
            candidate_path.write_bytes(rewrite_nodestrings(raw, replacements))
            delivered = read_2dm(candidate_path)
            require(np.array_equal(mesh.nodes_lonlat, delivered.nodes_lonlat) and np.array_equal(mesh.depths, delivered.depths) and np.array_equal(mesh.triangles, delivered.triangles),
                "geometry_or_connectivity_changed", "Endpoint correction changed geometric mesh content")
            delivered_ns = dict(zip(delivered.open_boundary_ids, [c.tolist() for c in delivered.open_boundary_chains]))
            require(delivered_ns == {ns: replacements.get(ns, (chain, chain))[1] for ns, chain in all_ns.items()}, "serialized_obc_mismatch", "Serialized chains differ from complete expected set")
            for feature in boundary["features"]:
                p = feature["properties"]
                if p["node_id_1based"] in reclassified:
                    p.update(is_open_boundary=False, open_boundary_nodestring_ids=[], open_boundary_cyclic=[])
                    p["fvcom_endpoint_reclassification"] = "prescribed_open_to_solid_exact_TGE"
            for row, old, new in zip(declared, chains, revised):
                ns = int(row["nodestring_id"])
                row["node_ids"], row["node_order_sha256"] = new, order_sha(new)
                sidecar = remap_rows[ns]
                sidecar.update(source_node_ids_1based=old, source_node_count=len(old), delivered_node_ids_1based=new,
                    delivered_node_count=len(new), node_order_sha256=order_sha(new), orientation_preserved=True,
                    source_sequence_retained_by_lineage=(old == new), reclassified_endpoint_nodes_1based=[n for n in old if n not in new])
            remap.update(input_mesh=str(mesh_path), input_mesh_sha256=sha(mesh_path), derived_mesh_sha256=sha(candidate_path),
                source_remap_sha256=sha(paths["obc_remap"]), forcing_compatible=False, source_boundary_forcing_compatible=False,
                forcing_interpolation_performed=False, forcing_regeneration_required=True)
            cyc_contract = remap.get("cyclicity_contract", {})
            if "boundary_contract_sha256" in cyc_contract:
                cyc_contract["parent_boundary_contract_sha256"] = cyc_contract.pop("boundary_contract_sha256")
            for row in cyc_contract.get("chains", []):
                row["node_count"] = len(delivered_ns[int(row["nodestring_id"])])
            dump(output / "boundary_nodes.geojson", boundary)
            dump(output / "obc_remap_manifest.json", remap)
            parent_delivery = copy.deepcopy(contract.get("source_delivery"))
            contract["mesh"].update(path="fvcom_grid.2dm", sha256=sha(candidate_path))
            contract["source_delivery"] = {"parent_contract_path": str(paths["grid_contract"]), "parent_contract_sha256": sha(paths["grid_contract"]),
                "parent_source_delivery": parent_delivery, "correction": "diagnosed_noncyclic_TGE_endpoint_reclassification",
                "geometry_and_connectivity_unchanged": True, "forcing_compatibility": "invalidated_until_regenerated", "submission_eligible": False}
            contract["delivery_artifacts"] = {name: {"path": filename, "sha256": sha(output / filename)} for name, filename in
                {"boundary_nodes": "boundary_nodes.geojson", "obc_remap": "obc_remap_manifest.json", "tge_audit": "after_tge_audit.json", "mesh_quality": "after_mesh_quality.json"}.items()}
            dump(output / "grid_delivery_contract.json", contract)
            # Repeat the source-bound gate on the serialized result, not only arrays.
            serialized_audit = audit_tge_boundary_junctions(len(delivered.nodes_lonlat), delivered.triangles-1,
                [np.asarray(delivered_ns[int(row["nodestring_id"])])-1 for row in declared], tge_source_path=paths["tge_source"])
            require(serialized_audit == after, "serialized_tge_mismatch", "Serialized TGE audit differs")
            result.update(status="ready", changed=True, decision="one_bounded_endpoint_reclassification",
                reclassified_node_ids_1based=sorted(reclassified), geometry_depth_connectivity_exact=True, all_other_nodestrings_exact=True,
                forcing_regeneration_required=True, submission_eligible=False,
                output_paths={name: str(output / filename) for name, filename in {"mesh": "fvcom_grid.2dm", "grid_contract": "grid_delivery_contract.json",
                    "boundary_nodes": "boundary_nodes.geojson", "obc_remap": "obc_remap_manifest.json", "tge_audit": "after_tge_audit.json", "mesh_quality": "after_mesh_quality.json"}.items()})
            result["output_hashes"] = {name: sha(path) for name, path in result["output_paths"].items()}
        require(all(sha(path) == result["input_hashes"][name] for name, path in paths.items()), "inputs_changed_during_repair", "An input changed during the transaction")
    except (RepairRejected, KeyError, ValueError, OSError) as exc:
        result.update(status="blocked", blocking_reasons=[{"code": getattr(exc, "code", "invalid_input"), "detail": str(exc)}])
    dump(output / "repair_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("grid-contract", "boundary-nodes", "obc-remap", "tge-source", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = repair(args.grid_contract, args.boundary_nodes, args.obc_remap, args.tge_source, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
