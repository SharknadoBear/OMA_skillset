"""Replayable forcing certificate for an explicitly retained, unchanged mesh.

This route does not certify fresh generation or reconstruct removed intermediates.
Only geometry-preserving noncyclic endpoint trims are admitted.
"""
from __future__ import annotations
import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon

from fvcom_grid_generation.forcing_join import audit_terminal_forcing, _read, _hash, _require
from fvcom_grid_generation.grid_project import _portable_boundary_audit, _portable_tge_audit
from fvcom_grid_generation.open_exterior import validate_grid_boundary_gate
from fvcom_grid_generation.quality import evaluate_mesh_quality
from fvcom_grid_generation.sms_2dm import read_2dm
from fvcom_grid_generation.tge_topology import audit_tge_boundary_junctions

RULE = 'fvcom_retained_grid_join_certificate_v1'
REQUIRED = {'grid_contract', 'original_mesh', 'original_boundary_nodes', 'original_publication_status',
    'source_case_manifest', 'open_exterior_source', 'boundary_resolution_source', 'tge_source',
    'retention_evidence', 'obc_points', 'forcing', 'forcing_manifest', 'preconfiguration_manifest',
    'source_snapshot_manifest'}


def _regular(path):
    path = Path(path).absolute()
    # Retained Windows provenance can exceed MAX_PATH. Use one normalized form
    # for reads and containment checks instead of treating long paths as absent.
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        path = Path('\\\\?\\' + str(path))
    _require(path.is_file() and not path.is_symlink(), 'retained input must be a regular nonsymlink file: '+str(path))
    return path.resolve()


def bind(path):
    path = _regular(path)
    return {'path': str(path), 'sha256': _hash(path)}


def resolve(binding, base):
    path = Path(binding['path'])
    if not path.is_absolute(): path = base/path
    path = _regular(path)
    _require(_hash(path) == binding['sha256'], 'retained input hash changed: ' + str(path))
    return path


def identity_remap(contract, original, source_case, resolution):
    """Normalize only a proven historical display alias; preserve every remap array."""
    normalized = copy.deepcopy(original)
    bounds = contract['open_boundaries']
    cycles = normalized['cyclicity_contract']['chains']
    _require(len(bounds) == len(cycles), 'retained identity chain count differs')
    case_rows = source_case['boundary']['open_boundaries']
    source_rows = resolution['open_boundary_chains']
    aliases = []
    for b, cycle in zip(bounds, cycles):
        current = str(cycle.get('declared_open_boundary_id', b['obc_id']))
        source_id = b.get('source_obc_id')
        display = str(b.get('display_name', b['obc_id']))
        matches = [x for x in case_rows if str(x.get('id')) == display]
        sources = [x for x in source_rows if str(x.get('obc_id')) == str(source_id)]
        _require(source_id is not None and str(b['obc_id']) == str(source_id)
            and current in (display, str(source_id)) and len(matches) == len(sources) == 1,
            'retained source/display identity is not a unique bijection')
        selection = matches[0].get('selection')
        _require(selection == f'exact finalized Adaptive-v2 obc_id {source_id}'
            or str(matches[0].get('source_obc_id', '')) == str(source_id),
            'historical case does not explicitly select the source OBC identity')
        _require(cycle['nodestring_id'] == b['nodestring_id'], 'alias nodestring identity differs')
        if current == str(b['obc_id']): continue
        cycle['declared_open_boundary_id'] = str(b['obc_id'])
        aliases.append({'nodestring_id': b['nodestring_id'], 'source_obc_id': source_id,
            'before': current, 'after': str(b['obc_id'])})
    return normalized, aliases


def audit_sponge_values(pre, boundaries, chains, sponge):
    """Compare serialized per-node controls with their declared per-OBC values."""
    rows = pre['sponge_by_obc']
    _require(len(rows) == len(boundaries), 'preconfiguration sponge chain count differs')
    expected = []
    for row, boundary, chain in zip(rows, boundaries, chains):
        _require(str(row['obc_id']) == str(boundary['obc_id'])
            and row['nodestring_id'] == boundary['nodestring_id'], 'preconfiguration sponge identity differs')
        radius, coefficient = float(row['radius_m']), float(row['coefficient'])
        _require(np.isfinite([radius, coefficient]).all() and min(radius, coefficient) > 0,
            'preconfiguration sponge controls are invalid')
        if pre['sponge_mode'] == 'estimate' and not row.get('override'):
            base = row['baseline_estimate']
            _require(radius == base['sponge_radius_m'] and coefficient == base['sponge_coefficient'],
                'preconfiguration sponge controls differ from declared baseline')
        expected.extend((int(node), float(f'{radius:.6f}'), float(f'{coefficient:.6f}')) for node in chain)
    _require(np.array_equal(sponge, np.asarray(expected)), 'preconfiguration sponge DAT differs from declared per-OBC controls')


def audit(request_path):
    request_path = _regular(request_path)
    request = _read(request_path)
    _require(request.get('schema_version') == 'fvcom_retained_grid_join_request_v1'
        and request.get('workflow_kind') == 'accepted_retained', 'retained route is only for an explicitly accepted retained mesh')
    _require(REQUIRED <= set(request.get('inputs', {})), 'retained join request is incomplete')
    paths = {k: resolve(v, request_path.parent) for k, v in request['inputs'].items()}
    dependencies = {k: bind(v) for k, v in paths.items()}
    retention = _read(paths['retention_evidence'])
    _require(retention.get('schema_version') == 'fvcom_grid_post_cleanup_integrity_v1'
        and retention.get('final_mesh_audit_passed') is True
        and retention.get('protected_t6v6_files_checked', 0) > 0
        and retention.get('protected_t6v6_files_matching') == retention['protected_t6v6_files_checked']
        and retention.get('protected_t6v6_files_missing') == 0
        and retention.get('protected_t6v6_files_mismatching') == 0, 'retained cleanup integrity evidence is unsupported or failing')
    if request.get('supplementary_retention_evidence'):
        dependencies['supplementary_retention_evidence'] = bind(resolve(request['supplementary_retention_evidence'], request_path.parent))
    contract = _read(paths['grid_contract'])
    _require(contract.get('schema_version') == 'fvcom_grid_delivery_contract_v1', 'unsupported retained grid contract')
    mesh_path = resolve(contract['mesh'], paths['grid_contract'].parent)
    dependencies['terminal_mesh'] = bind(mesh_path)
    companions = {k: resolve(v, paths['grid_contract'].parent) for k, v in contract['delivery_artifacts'].items()}
    _require({'boundary_nodes', 'mesh_quality', 'obc_remap', 'tge_audit'} <= companions.keys(), 'missing retained delivery companions')
    dependencies.update({k: bind(v) for k, v in companions.items()})
    mesh, old = read_2dm(mesh_path), read_2dm(paths['original_mesh'])
    _require(np.array_equal(mesh.nodes_lonlat, old.nodes_lonlat) and np.array_equal(mesh.depths, old.depths)
        and np.array_equal(mesh.triangles, old.triangles), 'retained route cannot certify changed geometry/depth/connectivity')
    strip_ns = lambda p: b''.join(s for s in p.read_bytes().splitlines(keepends=True) if not s.startswith(b'NS '))
    _require(strip_ns(mesh_path) == strip_ns(paths['original_mesh']), 'non-nodestring serialization changed')
    _require(mesh.open_boundary_ids == old.open_boundary_ids, 'retained route cannot change OBC chain identities')
    boundaries = contract['open_boundaries']
    _require(len(boundaries) == len(mesh.open_boundary_chains), 'retained OBC chain count differs')
    endpoint_changes = []
    for b, old_chain, new_chain in zip(boundaries, old.open_boundary_chains, mesh.open_boundary_chains):
        a, z = list(map(int, old_chain)), list(map(int, new_chain))
        if a == z: continue
        _require(b['cyclic'] is False and len(z) >= 2 and z in (a[1:], a[:-1], a[1:-1]),
            'retained route only admits bounded noncyclic endpoint trims')
        endpoint_changes.append({'nodestring_id': b['nodestring_id'], 'removed_node_ids': sorted(set(a)-set(z))})
    publication = _read(paths['original_publication_status'])
    _require(publication.get('mesh', {}).get('sha256') == _hash(paths['original_mesh']), 'retained source publication does not bind original mesh')
    _require(publication.get('obc_status') == 'pass', 'retained source OBC placement was not passing')
    boundary = _portable_boundary_audit(validate_grid_boundary_gate(paths['open_exterior_source'], paths['boundary_resolution_source'],
        policy=publication['boundary_gate_policy'], strict_contract_required=True))
    _require(boundary['passed'] and boundary == publication['open_exterior_audit'], 'retained boundary evidence is stale or failing')
    # Preserve every extant historical source; absence is disclosed, never filled by invented payloads.
    snapshot = _read(paths['source_snapshot_manifest'])
    _require(snapshot.get('all_extant_files_copied_and_hash_verified') is True and snapshot.get('files'), 'retained source snapshot is incomplete')
    snapshot_base = paths['source_snapshot_manifest'].parent.parent
    for i, record in enumerate(snapshot['files']):
        path = resolve({'path': record['retained'], 'sha256': record['sha256']}, snapshot_base)
        _require(path.is_relative_to(snapshot_base), 'retained snapshot path escapes its package')
        dependencies[f'source_snapshot_{i}'] = bind(path)
    retained_hashes = {v['sha256'] for k, v in dependencies.items() if k.startswith('source_snapshot_')}
    _require(all(_hash(paths[k]) in retained_hashes for k in ('original_publication_status', 'original_mesh', 'original_boundary_nodes',
        'source_case_manifest', 'open_exterior_source', 'boundary_resolution_source')),
        'original publication/mesh/boundary/case sources must match retained snapshot bytes')
    # Current geometry, boundary loops and quality are re-audited independently of historical flags.
    n, tri = len(mesh.depths), mesh.triangles-1
    _require(0 < n <= 5000000 and np.isfinite(mesh.nodes_lonlat).all() and np.isfinite(mesh.depths).all()
        and (mesh.depths > 0).all(), 'invalid retained node/depth data')
    _require(tri.min() >= 0 and tri.max() < n and len(np.unique(tri)) == n
        and (np.diff(np.sort(tri, axis=1), axis=1) > 0).all()
        and len(np.unique(np.sort(tri, axis=1), axis=0)) == len(tri), 'invalid retained connectivity')
    xy = np.column_stack(Transformer.from_crs('EPSG:4326', contract['mesh']['horizontal_crs'], always_xy=True).transform(*mesh.nodes_lonlat.T))
    a, b, c = (xy[tri[:, k]] for k in range(3))
    area = ((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])-(b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0]))/2
    _require((area > 0).all(), 'nonpositive retained cell area')
    edges, counts = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0, return_counts=True)
    boundary_edges = edges[counts == 1]
    boundary_nodes = np.unique(boundary_edges)
    _require((counts <= 2).all() and (np.bincount(boundary_edges.ravel(), minlength=n)[boundary_nodes] == 2).all(), 'invalid retained boundary topology')
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    _require(connected_components(graph, directed=False)[0] == 1, 'retained mesh has disconnected wet components')
    side, original_side = _read(companions['boundary_nodes']), _read(paths['original_boundary_nodes'])
    original_by_id = {f['properties']['node_id_1based']: f for f in original_side['features']}
    groups = collections.defaultdict(list)
    seen = set()
    for feature in side['features']:
        prop = feature['properties']; node = prop['node_id_1based']
        _require(node not in seen and node in original_by_id and 1 <= node <= n, 'invalid retained boundary node identity')
        seen.add(node); before = original_by_id[node]
        _require(feature['geometry'] == before['geometry'] and np.allclose(feature['geometry']['coordinates'][:2], mesh.nodes_lonlat[node-1], rtol=0, atol=1e-10)
            and all(prop[k] == before['properties'][k] for k in ('constraint_chain_id', 'constraint_chain_position', 'is_hard_anchor')), 'physical boundary or hard anchor changed')
        groups[prop['constraint_chain_id']].append((prop['constraint_chain_position'], node-1))
    _require(seen == set((boundary_nodes+1).tolist()) == set(original_by_id), 'boundary sidecar is incomplete')
    chains = [[node for _, node in sorted(group)] for group in groups.values()]
    covered = [tuple(sorted((a, b))) for chain in chains for a, b in zip(chain, chain[1:]+chain[:1])]
    _require(len(covered) == len(boundary_edges) and set(covered) == set(map(tuple, boundary_edges.tolist())), 'sidecar does not cover every boundary edge exactly once')
    polygons = [Polygon(xy[chain]) for chain in chains]
    exterior = max(polygons, key=lambda p: p.area); holes = [p for p in polygons if p is not exterior]
    _require(exterior.is_valid and all(p.is_valid and exterior.contains(p) for p in holes)
        and all(not p.intersects(q) for i, p in enumerate(holes) for q in holes[i+1:]), 'invalid retained exterior/islands')
    quality = evaluate_mesh_quality(xy, mesh.depths, mesh.triangles, np.empty(0, dtype=int),
        {'boundary_constraint_recovered': True, 'source': 'validated_delivered_boundary_sidecar'},
        constraint_chains=chains, open_boundary_chains=[list(map(int, x)) for x in mesh.open_boundary_chains],
        open_boundary_cyclic=[x['cyclic'] for x in boundaries], expected_open_boundary_count=len(boundaries), enforce_no_unused_nodes=True)
    _require(quality == _read(companions['mesh_quality']) and quality['benchmark_grid_baseline_ready'] and not quality['failure_taxonomy'], 'current retained quality gate fails or differs from frozen evidence')
    tge = audit_tge_boundary_junctions(n, tri, [np.asarray(x)-1 for x in mesh.open_boundary_chains], tge_source_path=paths['tge_source'])
    _require(tge['passed'] and tge['source_binding']['passed'] and _portable_tge_audit(tge) == _portable_tge_audit(_read(companions['tge_audit'])), 'retained source-bound TGE failed or changed')
    original_tge = audit_tge_boundary_junctions(n, tri, [np.asarray(x)-1 for x in old.open_boundary_chains], tge_source_path=paths['tge_source'])
    if endpoint_changes:
        removed = {node for row in endpoint_changes for node in row['removed_node_ids']}
        fatal = original_tge['fatal_cells']
        _require(not original_tge['passed'] and fatal
            and all(row['isonb_sum'] == 5 for row in fatal)
            and all(any(node in row['node_ids_1based'] for row in fatal) for node in removed),
            'endpoint trim lacks a reproduced original TGE endpoint failure')
    pre = _read(paths['preconfiguration_manifest'])
    _require(resolve(pre['grid_contract'], paths['preconfiguration_manifest'].parent) == paths['grid_contract'],
        'preconfiguration grid-contract binding differs')
    _require(resolve(pre['obc_topology_audit']['tge_source_binding'], paths['preconfiguration_manifest'].parent) == paths['tge_source'],
        'preconfiguration TGE-source binding differs')
    _require(pre['status'] == 'ready' and pre['mesh_sha256'] == _hash(mesh_path)
        and pre['obc_topology_audit']['status'] == 'ready'
        and pre['obc_topology_audit']['accepted_open_boundary_cell_ids'] == tge['open_cell_ids_1based'], 'preconfiguration and source TGE disagree')
    dat = {}
    for key, value in pre['generated_files'].items():
        path = Path(value)
        if not path.is_absolute(): path = paths['preconfiguration_manifest'].parent/path.name
        path = _regular(path)
        _require(_hash(path) == pre['generated_file_sha256'][key], 'preconfiguration file changed: '+key)
        dependencies['preconfiguration_'+key] = bind(path)
        dat[key] = path
    _require({'grd', 'dep', 'cor', 'obc', 'spg', 'sig'} <= dat.keys(), 'preconfiguration package is incomplete')
    with dat['grd'].open() as stream:
        headers = [stream.readline().strip(), stream.readline().strip()]
        cells = np.asarray([list(map(int, stream.readline().split())) for _ in range(len(tri))])
        nodes = np.asarray([list(map(float, row.split())) for row in stream if row.strip()])
    _require(headers == [f'Node Number = {n}', f'Cell Number = {len(tri)}']
        and cells.shape == (len(tri), 4) and np.array_equal(cells[:, 0], np.arange(1, len(tri)+1))
        and np.array_equal(cells[:, 1:], mesh.triangles) and nodes.shape == (n, 4)
        and np.array_equal(nodes[:, 0], np.arange(1, n+1))
        and np.allclose(nodes[:, 1:3], xy, rtol=0, atol=.001) and np.array_equal(nodes[:, 3], mesh.depths), 'preconfiguration grid differs from retained mesh')
    dep, cor = np.loadtxt(dat['dep'], skiprows=1), np.loadtxt(dat['cor'], skiprows=1)
    _require(dep.shape == cor.shape == (n, 3) and np.array_equal(dep[:, :2], nodes[:, 1:3])
        and np.array_equal(cor[:, :2], nodes[:, 1:3]) and np.array_equal(dep[:, 2], mesh.depths)
        and np.allclose(cor[:, 2], mesh.nodes_lonlat[:, 1], rtol=0, atol=1e-6), 'preconfiguration depth or latitude differs')
    ids = np.concatenate(mesh.open_boundary_chains)
    obc, sponge = np.atleast_2d(np.loadtxt(dat['obc'], skiprows=1)), np.atleast_2d(np.loadtxt(dat['spg'], skiprows=1))
    _require(obc.shape == sponge.shape == (len(ids), 3) and np.array_equal(obc[:, 1], ids)
        and np.array_equal(obc[:, 0], np.arange(1, len(ids)+1)) and (obc[:, 2] == 1).all()
        and np.array_equal(sponge[:, 0], ids) and np.isfinite(sponge).all() and (sponge[:, 1:] > 0).all(), 'preconfiguration OBC/sponge order or values differ')
    audit_sponge_values(pre, boundaries, mesh.open_boundary_chains, sponge)
    _require(dat['sig'].read_text().splitlines()[:2] == [f"NUMBER OF SIGMA LEVELS = {contract['sigma']['levels']}",
        'SIGMA COORDINATE TYPE = '+contract['sigma']['type'].upper()], 'preconfiguration sigma differs from contract')
    normalized, aliases = identity_remap(contract, _read(companions['obc_remap']), _read(paths['source_case_manifest']), _read(paths['boundary_resolution_source']))
    with tempfile.TemporaryDirectory(prefix='retained_obc_identity_') as temp:
        path = Path(temp)/'remap.json'; path.write_text(json.dumps(normalized, indent=2)+'\n', encoding='utf-8')
        forcing = audit_terminal_forcing(mesh_path, paths['grid_contract'], paths['obc_points'], paths['forcing'], paths['forcing_manifest'], path)
    return {'passed': True, 'scope': 'accepted_retained_geometry_preserved', 'dependencies': dependencies,
        'endpoint_changes': endpoint_changes, 'identity_aliases': aliases, 'terminal_remap': normalized,
        'tge': tge, 'original_tge': original_tge, 'quality': quality, 'boundary': boundary, 'forcing': forcing,
        'limitations': ['Historical generation intermediates may be absent under the bound retention evidence; fresh-generation validation is not claimed.',
                       'Geometry, depths and connectivity are identical to retained original; only proven noncyclic OBC endpoint trims are admitted.']}


def verify(certificate_path):
    cert = _read(_regular(certificate_path))
    _require(cert.get('schema_version') == RULE, 'unsupported retained certificate')
    auditor = resolve(cert['auditor'], Path(certificate_path).parent)
    _require(auditor == _regular(Path(certificate_path).parent/'auditor.py')
        and _hash(auditor) == _hash(__file__), 'retained certificate requires its exact archived auditor version')
    _require(cert.get('submission_ready') is True, 'retained certificate is not submission ready')
    request_path = resolve(cert['request'], Path(certificate_path).parent)
    result = audit(request_path)
    _require(result == cert['audit'], 'retained certificate replay differs')
    return {'passed': True, 'submission_ready': True, 'route': RULE, 'certificate': bind(certificate_path)}


def certify(request_path, output_dir):
    output_dir = Path(output_dir)
    _require(not output_dir.exists(), 'immutable retained certificate output already exists')
    result = audit(request_path)
    output_dir.mkdir(parents=True)
    auditor = output_dir/'auditor.py'
    auditor.write_bytes(Path(__file__).read_bytes())
    certificate = {'schema_version': RULE, 'request': bind(request_path), 'audit': result,
        'auditor': bind(auditor), 'submission_ready': True}
    path = output_dir/'certificate.json'
    path.write_text(json.dumps(certificate, indent=2)+'\n', encoding='utf-8')
    receipt = verify(path)
    (output_dir/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--request', type=Path); group.add_argument('--verify', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.verify: result = verify(args.verify)
    else:
        if not args.output_dir: parser.error('--request requires --output-dir')
        result = certify(args.request, args.output_dir)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
