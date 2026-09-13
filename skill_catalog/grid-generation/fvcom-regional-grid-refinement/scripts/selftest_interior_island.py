"""Real Gmsh patch and adversarial serialized-identity regression fixtures."""
from pathlib import Path
from copy import deepcopy
import json
import tempfile
import sys
import numpy as np
from shapely.geometry import Polygon

GRID = Path(__file__).resolve().parents[2] / "fvcom-grid-generation" / "scripts"
sys.path.insert(0, str(GRID))
from fvcom_grid_generation.gmsh_backend import (BoundaryLoopGeometry, GmshGeometry,
    GmshMeshingConfig, generate_gmsh_mesh)
from interior_island import (IdentityMesh, WetSizeField, select_patch, incumbent_sizes,
    generate_patch, merge_patch, audit_transaction, deterministic_new_ids, finite_number)
from interior_island import audit_quality_and_tge, audit_boundary_sidecar, bind, mesh_domain
from pyproj import Transformer

checks = 0


def check(condition, message):
    global checks
    assert condition, message
    checks += 1


def rejects(action, text):
    try:
        action()
    except ValueError as exc:
        check(text in str(exc), f"wrong rejection: {exc}; expected {text}")
    else:
        raise AssertionError("did not reject " + text)


def source_mesh(directory):
    xy = np.array([[0., 0.], [1000., 0.], [1000., 1000.], [0., 1000.]])
    result = generate_gmsh_mesh(GmshGeometry(BoundaryLoopGeometry("outer", xy, "exterior")),
        GmshMeshingConfig(uniform_target_m=100., constant_field=True), directory / "source.msh")
    nodes = tuple(f"ND {i+1} {p[0]:.12f} {p[1]:.12f} 5.0000" for i, p in enumerate(result.nodes_xy))
    triangles = tuple(f"E3T {i+1} {t[0]} {t[1]} {t[2]} 7" for i, t in enumerate(result.triangles))
    path = directory / "source.2dm"
    path.write_text("\n".join(["MESH2D", 'MESHNAME "test"', *triangles, *nodes]) + "\n")
    return path, IdentityMesh.read(path)


def candidate(directory, source, ring, buffer, target=20.):
    patch, wet, removed = select_patch(source.xy, source.triangles, ring, buffer)
    incumbent = incumbent_sizes(source.xy, source.triangles)
    sample = patch.selected_nodes
    sample = sample[~np.array([Polygon(ring).contains(__import__('shapely').Point(p)) for p in source.xy[sample]])]
    field = WetSizeField(wet, source.xy[sample], incumbent[sample], ring, target, .1)
    result, fixed = generate_patch(source.xy, patch, ring, field, directory / "patch.msh")
    lines, cells, lineage = merge_patch(source, source.xy, patch, result.nodes_xy,
        result.triangles - 1, fixed, "EPSG:32610", "EPSG:32610",
        lambda xy: np.full(len(xy), 5.))
    output = directory / "candidate.2dm"
    IdentityMesh(source.xy, source.depth, source.triangles, source.material,
                 tuple(lines), tuple(cells), source.other_lines).write(output)
    return output, patch, field, result, fixed, lineage


def sidecar_tests(root, source_path, output, ring):
    transform = Transformer.from_crs('EPSG:32610', 'EPSG:4326', always_xy=True)
    ring_path, review_path = root / 'ring.json', root / 'source_review.json'
    ring_path.write_text(json.dumps({'crs': 'EPSG:32610', 'coordinates': ring.tolist()}))
    review_path.write_text('{"synthetic_fixture": true}')
    def features(mesh, loops, start=0):
        return [{'type': 'Feature', 'geometry': {'type': 'Point',
                'coordinates': list(transform.transform(*mesh.xy[n]))},
            'properties': {'node_id_1based': int(n + 1), 'node_index_zero_based': int(n),
                'constraint_chain_id': c + start, 'constraint_chain_position': p,
                'is_open_boundary': False, 'open_boundary_nodestring_ids': [],
                'open_boundary_cyclic': []}}
            for c, loop in enumerate(loops) for p, n in enumerate(loop)]
    source, delivered = IdentityMesh.read(source_path), IdentityMesh.read(output)
    _, old_loops, _, _ = mesh_domain(source.xy, source.triangles)
    _, new_loops, _, _ = mesh_domain(delivered.xy, delivered.triangles)
    original = features(source, old_loops)
    old_ids = {f['properties']['node_id_1based'] for f in original}
    added_loops = [loop for loop in new_loops if not ({n + 1 for n in loop} & old_ids)]
    extra = features(delivered, added_loops, len(old_loops))
    for f in extra:
        f['properties'].update(source_type='supplied_interior_island', source_node_id_1based=None,
            source_node_index_zero_based=None, boundary_kind='island', is_hard_anchor=True,
            source_geometry={**bind(ring_path), 'path': ring_path.name},
            source_review={**bind(review_path), 'path': review_path.name})
    old_doc = {'type': 'FeatureCollection', 'features': original}
    new_doc = {'type': 'FeatureCollection', 'features': deepcopy(original) + extra}
    old_path, new_path = root / 'old_boundary.json', root / 'new_boundary.json'
    old_path.write_text(json.dumps(old_doc))
    def audit(doc):
        new_path.write_text(json.dumps(doc))
        return audit_boundary_sidecar(old_path, new_path, output, 'EPSG:32610', 'EPSG:32610', [],
            [f['properties']['node_id_1based'] for f in extra], ring_path, review_path)
    result = audit(new_doc)
    check(result['passed'] and result['new_island_nodes'] == len(ring), 'complete sidecar failed')
    for kind, expected in [('old', 'source boundary feature changed'),
        ('missing', 'unexpected added boundary inventory'), ('duplicate', 'duplicate boundary feature'),
        ('role', 'open/solid role differs'), ('coordinate', 'coordinate differs'),
        ('position', 'positions incomplete'), ('source', 'mislabelled'), ('binding', 'hash changed'),
        ('source_index', 'mislabelled'), ('kind', 'locked island'), ('anchor', 'locked island')]:
        altered = deepcopy(new_doc)
        row = altered['features'][-1]
        if kind == 'old':
            altered['features'][0]['properties']['unexpected'] = True
        elif kind == 'missing':
            altered['features'].pop()
        elif kind == 'duplicate':
            altered['features'].append(deepcopy(row))
        elif kind == 'role':
            row['properties']['is_open_boundary'] = True
        elif kind == 'coordinate':
            row['geometry']['coordinates'][0] += .01
        elif kind == 'position':
            row['properties']['constraint_chain_position'] = 100
        elif kind == 'source':
            row['properties']['source_node_id_1based'] = 1
        elif kind == 'source_index':
            row['properties']['source_node_index_zero_based'] = 0
        elif kind == 'kind':
            row['properties']['boundary_kind'] = 'mainland'
        elif kind == 'anchor':
            row['properties']['is_hard_anchor'] = False
        else:
            row['properties']['source_geometry']['sha256'] = '0' * 64
        rejects(lambda: audit(altered), expected)
    audit(new_doc)


def main():
    barrier = [[-1., -5.], [1., -5.], [1., 5.], [-1., 5.]]
    outer = [[-10., -10.], [10., -10.], [10., 10.], [-10., 10.]]
    small_ring = np.array([[-5., -.5], [-4., -.5], [-4., .5], [-5., .5]])
    samples = np.array(outer + barrier + [[4., 0.]])
    field = WetSizeField(Polygon(outer, [barrier, small_ring]), samples,
                         np.full(len(samples), 100.), small_ring, 1., .1)
    check(abs(field(4., 0.) - (1 + .1 * (np.sqrt(34) + 2 + np.sqrt(29.25)))) < 1e-10,
          "visibility envelope took shortcut across old island")
    check(field(-5., 0.) == 1., "continuous island-edge core target lost")
    for invalid in (True, False, float('inf'), float('nan'), '20', None):
        check(not finite_number(invalid), "nonfinite/nonnumeric scalar accepted")
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        source_path, source = source_mesh(root)
        mixed = IdentityMesh(source.xy, source.depth, source.triangles, source.material,
            source.node_lines, source.element_lines,
            ('MESH2D', 'MESHNAME "mixed"', 'NS\t1 2', '# middle', ' NS -3 1', '# trailing'))
        mixed.write(root / 'mixed.2dm')
        check(IdentityMesh.read(root / 'mixed.2dm').other_lines == mixed.other_lines,
              "nonentity/NS order changed")
        # Hole strictly inside one source cell: element intersection must find it
        # even with no source vertex in the requested core.
        centroids = source.xy[source.triangles].mean(axis=1)
        index = np.argmin(np.linalg.norm(centroids - [500, 500], axis=1))
        center = centroids[index]
        theta = np.arange(8) * 2 * np.pi / 8
        ring = center + 6 * np.c_[np.cos(theta), np.sin(theta)]
        patch, wet, removed = select_patch(source.xy, source.triangles, ring, 5.)
        check(patch.selection[index], "did not select containing source element")
        check(not any(Polygon(ring).contains(__import__('shapely').Point(p)) for p in source.xy), "fixture contains old node")
        tiny = root / 'no_source_vertex'
        tiny.mkdir()
        tiny_output, tiny_patch, *_ = candidate(tiny, source, ring, 1000., target=3.)
        check(audit_transaction(source_path, tiny_output, 'EPSG:32610', 'EPSG:32610',
            ring, tiny_patch.selection)['passed'], 'no-source-vertex Gmsh hole integration failed')
        # Resolved island fixture and a repeated generation.
        ring = center + 36 * np.c_[np.cos(theta), np.sin(theta)]
        outputs = []
        for repeat in range(2):
            directory = root / f"repeat{repeat}"
            directory.mkdir()
            output, patch, field, generated, fixed, lineage = candidate(directory, source, ring, 500.)
            audit = audit_transaction(source_path, output, "EPSG:32610", "EPSG:32610", ring, patch.selection)
            check(audit["passed"] and audit["euler_delta"] == -1, "topology audit failed")
            check(audit["boundary_loops_after"] == 2, "wrong island count")
            outputs.append(output)
        check(outputs[0].read_bytes() == outputs[1].read_bytes(), "nondeterministic serialized result")
        tge = root / 'tge.F'
        tge.write_text('ISONB=0\nISONB(NV(I,1))=1\nISONB(I_OBC_N(I))=2\n'
            'IF(SUM(ISONB(NV(I,1:3)))==4) THEN\nENDIF\nIF(SUM(ISONB(NV(I,1:3)))>4) STOP\n')
        gate = audit_quality_and_tge(outputs[0], 'EPSG:32610', 'EPSG:32610', [], [], tge, root / 'quality')
        quality = json.loads((root / 'quality/quality.json').read_text())
        check(quality['constraint_integrity']['missing_protected_edge_count'] == 0,
              'closed-loop API introduced false self edges')
        check(gate['roundtrip_exact'], 'roundtrip gate failed')
        sidecar_tests(root, source_path, outputs[0], ring)
        check(all(field.visible_domain.covers(__import__('shapely').LineString(field.xy[e])) for e in field.edges),
              "field graph crosses land")
        rejects(lambda: field(*center), "outside wet domain")
        # Exact record mutation tests independently reread the altered file.
        delivered = IdentityMesh.read(outputs[0])
        retained = sorted(set(range(len(source.xy))) - set(np.array(lineage["replaced_source_node_ids"]) - 1))
        outside = np.flatnonzero(~patch.selection)
        require_outside = len(outside) > 0
        check(require_outside, "fixture must contain outside elements")
        for kind in ("coordinate", "depth", "material", "connectivity", "header", "duplicate", "reversed"):
            nodes = list(delivered.node_lines)
            cells = list(delivered.element_lines)
            other = list(delivered.other_lines)
            if kind in {"coordinate", "depth"}:
                row = nodes[retained[0]].split()
                k = 2 if kind == "coordinate" else 4
                row[k] = str(float(row[k]) + .125)
                nodes[retained[0]] = " ".join(row)
            elif kind in {"material", "connectivity"}:
                row = cells[outside[0]].split()
                k = 5 if kind == "material" else 2
                row[k] = str(int(row[k]) + 1)
                cells[outside[0]] = " ".join(row)
            elif kind == "header":
                other.append("# unexpected metadata")
            elif kind == "duplicate":
                nodes.append(nodes[0])
            else:
                row = cells[np.flatnonzero(patch.selection)[0]].split()
                row[2], row[3] = row[3], row[2]
                cells[np.flatnonzero(patch.selection)[0]] = " ".join(row)
            mutated = root / (kind + ".2dm")
            IdentityMesh(delivered.xy, delivered.depth, delivered.triangles, delivered.material,
                tuple(nodes), tuple(cells), tuple(other)).write(mutated)
            expected = {"coordinate": "retained node", "depth": "retained node",
                "material": "outside element", "connectivity": "outside element", "header": "header/OBC",
                "duplicate": "duplicate", "reversed": "nonpositive"}[kind]
            rejects(lambda: audit_transaction(source_path, mutated, "EPSG:32610", "EPSG:32610", ring, patch.selection), expected)
        rejects(lambda: deterministic_new_ids(20, np.arange(4), np.zeros((3, 2))), "fewer interior nodes")
        rejects(lambda: merge_patch(source, source.xy, patch, generated.nodes_xy, generated.triangles[:1] - 1,
            fixed, "EPSG:32610", "EPSG:32610", lambda xy: np.full(len(xy), 5)), "fewer elements")
        rejects(lambda: merge_patch(source, source.xy, patch, generated.nodes_xy, generated.triangles - 1,
            fixed, "EPSG:32610", "EPSG:32610", lambda xy: np.full(len(xy), np.nan)), "bathymetry invalid")
        rejects(lambda: select_patch(source.xy, source.triangles, np.array([[0, 0], [10, 0], [0, 10]]), 5), "strictly inside")
        rejects(lambda: select_patch(source.xy, source.triangles, np.array([[400, 400], [500, 500], [400, 500], [500, 400]]), 5), "invalid island")
        if len(removed):
            rejects(lambda: select_patch(source.xy, source.triangles, ring, 310, [int(removed[0])]), "protected")
        rejects(lambda: select_patch(delivered.xy, delivered.triangles, ring * .999 + center * .001, 50), "strictly inside")
        print({"status": "PASS", "checks": checks})


if __name__ == "__main__":
    main()
