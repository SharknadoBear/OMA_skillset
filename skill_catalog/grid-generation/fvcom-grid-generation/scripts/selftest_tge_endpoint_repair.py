#!/usr/bin/env python3
"""Meaningful endpoint-repair regressions, with optional immutable regional cases."""
import argparse
import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from pyproj import Transformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-scripts", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regional-fixtures", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.grid_scripts.resolve()))
    module_spec = importlib.util.spec_from_file_location("endpoint_repair_proposal", Path(__file__).with_name("repair_fvcom_tge_endpoints.py"))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    from fvcom_grid_generation.sms_2dm import read_2dm, write_2dm
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / "tge.F"
    source.write_text("""ISONB = 0
ISONB(NV(I,2)) = 1
ISONB(I_OBC_N(I))=2
IF(SUM(ISONB(NV(I,1:3))) == 4) THEN
ELSE IF(SUM(ISONB(NV(I,1:3))) > 4) THEN
END IF
""", encoding="utf-8")
    results = []

    def fixture(name, chains, cyclic=None):
        location = root / name
        location.mkdir()
        coordinates = np.asarray([[0,0],[1,0],[2,0],[2,1],[2,2],[1,2],[0,2],[0,1],[1,1]], dtype=float) * 100 + [450000, 5100000]
        transformer = Transformer.from_crs(32610, 4326, always_xy=True)
        ll = np.column_stack(transformer.transform(coordinates[:, 0], coordinates[:, 1]))
        triangles = np.asarray([[8,1,2],[2,9,8],[2,3,9],[3,4,9],[4,5,9],[5,6,9],[6,7,9],[7,8,9]])
        ids = list(range(11, 11 + len(chains)))
        cyclic = cyclic or [False] * len(chains)
        write_2dm(location / "mesh.2dm", ll, np.full(9, 10.), triangles, np.empty(0, dtype=int),
            open_boundary_chains=chains, open_boundary_ids=ids)
        mesh = read_2dm(location / "mesh.2dm")
        contract = {"schema_version":"fvcom_grid_delivery_contract_v1", "mesh":{"path":"mesh.2dm", "sha256":module.sha(location / "mesh.2dm"),
            "node_count":9,"triangle_count":8,"coordinate_crs":"EPSG:4326","horizontal_crs":"EPSG:32610","horizontal_units":"m"},
            "open_boundaries":[{"obc_id":f"ocean_{ns}","nodestring_id":ns,"node_ids":chain,"node_order_sha256":module.order_sha(chain),"cyclic":cycle,"obc_type":"prescribed"}
                for ns,chain,cycle in zip(ids,chains,cyclic)], "excluded_nodestring_ids":[], "sigma":{"levels":10,"type":"UNIFORM"}}
        features=[]
        for node in range(1,9):
            membership=[ns for ns,c in zip(ids,chains) if node in c]
            features.append({"type":"Feature","geometry":{"type":"Point","coordinates":mesh.nodes_lonlat[node-1].tolist()},
                "properties":{"node_id_1based":node,"node_index_zero_based":node-1,"constraint_chain_id":0,"constraint_chain_position":node-1,
                    "is_hard_anchor":True,"is_open_boundary":bool(membership),"open_boundary_nodestring_ids":membership,
                    "open_boundary_cyclic":[cyclic[ids.index(ns)] for ns in membership]}})
        sidecar={"schema_version":"fvcom_obc_remap_manifest_v1","chains":[{"nodestring_id":ns,"delivered_node_ids_1based":chain,
            "delivered_node_count":len(chain),"cyclic":cycle,"orientation_preserved":True} for ns,chain,cycle in zip(ids,chains,cyclic)],
            "cyclicity_contract":{"chains":[{"nodestring_id":ns,"node_count":len(chain),"cyclic":cycle} for ns,chain,cycle in zip(ids,chains,cyclic)]}}
        for filename,data in [("grid_delivery_contract.json",contract),("boundary_nodes.geojson",{"type":"FeatureCollection","features":features}),("obc_remap_manifest.json",sidecar)]:
            module.dump(location / filename,data)
        return location

    def run_case(name, location, expected, *, source_path=source):
        paths=[location / "grid_delivery_contract.json",location / "boundary_nodes.geojson",location / "obc_remap_manifest.json"]
        before={str(p):module.sha(p) for p in paths+[source_path]}
        result=module.repair(*paths,source_path,root / (name+"_result"))
        assert before=={str(p):module.sha(p) for p in paths+[source_path]}, "Inputs mutated"
        actual=result["status"] if result["status"]=="ready" else result["blocking_reasons"][0]["code"]
        assert actual==expected,(name,actual,result)
        results.append({"test":name,"status":"pass","decision":result.get("decision",actual)})
        return result

    multi=fixture("plural_input",[[1,2,3],[5,6]])
    plural=run_case("plural_preservation",multi,"ready")
    assert plural["changed"] and plural["reclassified_node_ids_1based"]==[1]
    delivered=read_2dm(plural["output_paths"]["mesh"])
    assert delivered.open_boundary_ids==(11,12)
    assert [chain.tolist() for chain in delivered.open_boundary_chains]==[[2,3],[5,6]]
    delivered_contract=json.loads(Path(plural["output_paths"]["grid_contract"]).read_text())
    assert [r["obc_id"] for r in delivered_contract["open_boundaries"]]==["ocean_11","ocean_12"]
    assert json.loads(Path(plural["output_paths"]["obc_remap"]).read_text())["chains"][1]["delivered_node_ids_1based"]==[5,6]
    no_op=run_case("already_passing_noop",root / "plural_preservation_result","ready")
    assert not no_op["changed"] and no_op["decision"]=="already_source_bound_ready"
    tampered=root / "bound_sidecar_input"
    shutil.copytree(root / "plural_preservation_result",tampered)
    tampered_sidecar=json.loads((tampered / "boundary_nodes.geojson").read_text())
    tampered_sidecar["unexpected_change"]="stale input"
    module.dump(tampered / "boundary_nodes.geojson",tampered_sidecar)
    run_case("bound_sidecar_hash_rejection",tampered,"delivery_artifact_hash_mismatch")
    run_case("cyclic_rejection",fixture("cyclic_input",[[1,2,3]],[True]),"cyclic_endpoint_repair_unsupported")
    run_case("ambiguous_endpoint_rejection",fixture("ambiguous_input",[[1,2]]),"ambiguous_or_nonendpoint_fatal_cell")
    run_case("interior_open_fan_rejection",fixture("interior_input",[[3,2,1,8,7]]),"unsupported_tge_cell")
    stale=fixture("stale_input",[[1,2,3]])
    doc=json.loads((stale / "grid_delivery_contract.json").read_text())
    doc["mesh"]["sha256"]="0"*64
    module.dump(stale / "grid_delivery_contract.json",doc)
    run_case("stale_hash_rejection",stale,"mesh_hash_mismatch")
    bad_source=root / "bad_tge.F"
    bad_source.write_text("ISONB=0\n")
    run_case("source_mismatch_rejection",fixture("source_input",[[1,2,3]]),"tge_source_mismatch",source_path=bad_source)
    overlap=fixture("overlap_input",[[1,2,3],[3,4]])
    run_case("overlapping_obc_rejection",overlap,"overlapping_obcs")
    if args.regional_fixtures:
        regional=json.loads(args.regional_fixtures.read_text())
        for row in regional["cases"]:
            original_contract=Path(row["grid_contract"]).resolve()
            location=root / (row["name"]+"_input")
            location.mkdir()
            doc=json.loads(original_contract.read_text())
            mesh_path=Path(doc["mesh"]["path"])
            if not mesh_path.is_absolute():
                mesh_path=(original_contract.parent/mesh_path).resolve()
            doc["mesh"]["path"]=str(mesh_path)
            module.dump(location / "grid_delivery_contract.json",doc)
            for key,filename in [("boundary_nodes","boundary_nodes.geojson"),("obc_remap","obc_remap_manifest.json")]:
                (location / filename).write_bytes(Path(row[key]).read_bytes())
            output=run_case(row["name"],location,"ready",source_path=Path(regional["tge_source"]).resolve())
            assert output["reclassified_node_ids_1based"]==row["expected_reclassified_nodes"]
            current=read_2dm(output["output_paths"]["mesh"])
            original=read_2dm(mesh_path)
            assert np.array_equal(original.nodes_lonlat,current.nodes_lonlat) and np.array_equal(original.depths,current.depths) and np.array_equal(original.triangles,current.triangles)
            assert sum(len(c) for c in current.open_boundary_chains)==row["expected_obc_count"]
            # Preserve original non-NS bytes. Historical helpers normalized line
            # endings, so compare their scientific serialization separately.
            non_ns=lambda p: b"".join(line for line in Path(p).read_bytes().splitlines(keepends=True) if not line.lstrip().upper().startswith(b"NS "))
            assert non_ns(mesh_path)==non_ns(output["output_paths"]["mesh"])
            if "reference_corrected_mesh" in row:
                reference=read_2dm(Path(row["reference_corrected_mesh"]))
                assert np.array_equal(reference.nodes_lonlat,current.nodes_lonlat) and np.array_equal(reference.depths,current.depths) and np.array_equal(reference.triangles,current.triangles)
                assert reference.open_boundary_ids==current.open_boundary_ids
                assert all(np.array_equal(a,b) for a,b in zip(reference.open_boundary_chains,current.open_boundary_chains))
    module.dump(root / "selftest_results.json",{"status":"pass","test_count":len(results),"tests":results})
    print(json.dumps({"status":"pass","test_count":len(results),"tests":results},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
