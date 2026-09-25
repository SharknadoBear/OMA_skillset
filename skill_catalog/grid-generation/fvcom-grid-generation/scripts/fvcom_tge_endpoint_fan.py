"""Opt-in bounded, replayable TGE endpoint-fan membership planning.

No geometry is written here. The repair owner applies only a complete passing plan.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from fvcom_grid_generation.tge_topology import audit_tge_boundary_junctions

RULE = 'fvcom_tge_sequential_endpoint_fan_v1'
MAX_STEPS = 4


class FanRejected(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def require(ok, code, detail):
    if not ok:
        raise FanRejected(code, detail)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()


def audit_digest(audit):
    normalized = dict(audit)
    normalized['source_binding'] = dict(audit['source_binding'])
    normalized['source_binding'].pop('source_path', None)
    return digest(normalized)


def plan_endpoint_fan(node_count, triangles_zero_based, chains, cyclic, tge_source, mesh_sha256, max_steps):
    """Prove initial anchored fans, then replay uniquely caused current endpoints.

    The small bound is total node demotions over the whole delivery. Every step
    strictly reduces the original fatal-cell set; disconnected/nonendpoint debt
    cannot be reached by opportunistically shortening an unrelated arc.
    """
    require(type(max_steps) is int and 1 <= max_steps <= MAX_STEPS,
            'invalid_endpoint_fan_bound', f'Opt-in endpoint fan bound must be 1..{MAX_STEPS}')
    chains = [list(map(int, chain)) for chain in chains]
    original = [chain[:] for chain in chains]
    require(len(chains) == len(cyclic) and all(type(x) is bool for x in cyclic), 'invalid_endpoint_fan_cyclicity', 'Exact per-chain cyclicity is required')
    require(all(len(c) >= 2 and len(c) == len(set(c)) for c in chains), 'invalid_endpoint_fan_chain', 'Chains must contain at least two unique nodes')
    ownership = {}
    for i, chain in enumerate(chains):
        for node in chain:
            require(node not in ownership, 'cross_chain_endpoint_fan', 'Different OBC chains may not overlap')
            ownership[node] = i
    triangles = np.asarray(triangles_zero_based, dtype=int)
    edges, count = np.unique(np.sort(np.vstack([triangles[:, [0,1]], triangles[:, [1,2]], triangles[:, [2,0]]]), axis=1), axis=0, return_counts=True)
    exterior = set(map(tuple, (edges[count == 1] + 1).tolist()))
    for chain, cycle in zip(chains, cyclic):
        pairs = list(zip(chain, chain[1:])) + ([(chain[-1], chain[0])] if cycle else [])
        require(all(tuple(sorted(pair)) in exterior for pair in pairs), 'disconnected_endpoint_fan_chain', 'Every OBC edge must be a topological exterior edge')

    def audit(current):
        result = audit_tge_boundary_junctions(node_count, triangles,
                    [np.asarray(c, dtype=int)-1 for c in current], tge_source_path=tge_source)
        require(result['source_binding']['passed'], 'tge_source_mismatch', 'Exact source TGE expressions are required')
        return result

    before = audit(chains)
    fatal = before['fatal_cells']
    # Classify every original fatal cell before making any membership proposal.
    owner_by_cell = {}
    for cell in fatal:
        require(cell['isonb_sum'] == 5 and sorted(cell['isonb_values']) == [1,2,2], 'unsupported_tge_cell', 'Endpoint fans only admit original sum-five cells with flags 1,2,2')
        owners = {ownership[node] for node in cell['open_boundary_node_ids_1based']}
        require(len(owners) == 1, 'cross_chain_endpoint_fan', 'Fatal cells may not cross OBC chains')
        owner = next(iter(owners)); owner_by_cell[cell['element_id_1based']] = owner
        require(not cyclic[owner], 'cyclic_endpoint_repair_unsupported', 'Never demote a cyclic OBC')
    # Edge adjacency, not point contact: a detached interior fan cannot piggyback.
    remaining = {c['element_id_1based']: c for c in fatal}
    components = []
    while remaining:
        first = min(remaining); component = [remaining.pop(first)]; frontier = component[:]
        while frontier:
            cell = frontier.pop()
            neighbors = [key for key, other in remaining.items()
                         if len(set(cell['node_ids_1based']) & set(other['node_ids_1based'])) >= 2]
            for key in neighbors:
                other = remaining.pop(key); component.append(other); frontier.append(other)
        owners = {owner_by_cell[c['element_id_1based']] for c in component}
        require(len(owners) == 1, 'cross_chain_endpoint_fan', 'A connected fatal fan must have one OBC owner')
        owner = next(iter(owners)); chain = chains[owner]
        open_nodes = {node for c in component for node in c['open_boundary_node_ids_1based']}
        anchors = open_nodes & {chain[0], chain[-1]}
        require(len(anchors) == 1, 'unanchored_or_ambiguous_endpoint_fan', 'Every original connected fatal fan needs exactly one initial noncyclic endpoint')
        anchor = next(iter(anchors))
        components.append({'initial_fatal_cell_ids': sorted(c['element_id_1based'] for c in component),
                           'chain_index': owner, 'initial_endpoint': anchor,
                           'side': 'start' if anchor == chain[0] else 'end'})
    components.sort(key=lambda c: c['initial_fatal_cell_ids'])
    original_fatal = {c['element_id_1based'] for c in fatal}
    trace = []
    current_audit = before
    while not current_audit['passed']:
        require(len(trace) < max_steps, 'endpoint_fan_bound_exhausted', 'Bound exhausted before every source-TGE fatal cell cleared; no corrected delivery is permitted')
        current_fatal = {c['element_id_1based'] for c in current_audit['fatal_cells']}
        require(current_fatal <= original_fatal, 'endpoint_fan_new_fatal_cell', 'Sequential demotion created unrelated fatal debt')
        choices = {}
        for cell in current_audit['fatal_cells']:
            owner = owner_by_cell[cell['element_id_1based']]; chain = chains[owner]
            candidates = set(cell['open_boundary_node_ids_1based']) & {chain[0], chain[-1]}
            require(len(candidates) <= 1, 'ambiguous_current_endpoint_fan', 'A current fatal cell has multiple endpoint candidates')
            if not candidates: continue
            node = next(iter(candidates)); side = 'start' if node == chain[0] else 'end'
            component = next(c for c in components if cell['element_id_1based'] in c['initial_fatal_cell_ids'])
            require(component['chain_index'] == owner and component['side'] == side,
                    'endpoint_fan_anchor_changed', 'Current endpoint must descend from the proven original fan anchor')
            choices.setdefault((owner, side, node), []).append(cell['element_id_1based'])
        require(bool(choices), 'endpoint_fan_no_progress', 'No uniquely identified current endpoint can resolve remaining anchored debt')
        owner, side, node = sorted(choices, key=lambda x: (x[0], x[1] != 'start', x[2]))[0]
        require(len(chains[owner]) > 2, 'endpoint_trim_too_short', 'A repair must preserve at least two nodes per OBC')
        revised = [c[:] for c in chains]
        revised[owner] = revised[owner][1:] if side == 'start' else revised[owner][:-1]
        after = audit(revised)
        after_fatal = {c['element_id_1based'] for c in after['fatal_cells']}
        require(after_fatal < current_fatal, 'endpoint_fan_no_progress', 'Every source-bound step must strictly reduce fatal cells')
        trace.append({'step': len(trace)+1, 'chain_index': owner, 'side': side, 'node_id_1based': node,
                      'current_endpoint_cell_ids': sorted(choices[(owner,side,node)]),
                      'before_fatal_cell_ids': sorted(current_fatal), 'after_fatal_cell_ids': sorted(after_fatal),
                      'before_chains_sha256': digest(chains), 'after_chains_sha256': digest(revised),
                      'before_source_tge_audit_sha256': audit_digest(current_audit),
                      'after_source_tge_audit_sha256': audit_digest(after)})
        chains, current_audit = revised, after
    return {'schema_version': RULE, 'max_steps': max_steps, 'mesh_sha256': mesh_sha256,
            'tge_source_sha256': sha(tge_source), 'planner_source_sha256': sha(__file__),
            'original_chains': original, 'cyclic': list(cyclic), 'initial_components': components,
            'steps': trace, 'delivered_chains': chains,
            'initial_source_tge_audit_sha256': audit_digest(before), 'final_source_tge_audit_sha256': audit_digest(current_audit),
            'passed': current_audit['passed']}


def replay_endpoint_fan(trace, original_mesh, terminal_mesh, cyclic, tge_source, original_sha, terminal_sha, expected_max_steps):
    """Independently recompute the complete trace and exact terminal memberships."""
    require(isinstance(trace, dict) and trace.get('schema_version') == RULE, 'invalid_endpoint_fan_trace', 'Missing or unsupported endpoint-fan trace')
    require(type(expected_max_steps) is int and 1 <= expected_max_steps <= MAX_STEPS
            and trace.get('max_steps') == expected_max_steps,
            'endpoint_fan_bound_binding_mismatch', 'Trace bound must equal the explicit bound in the terminal delivery contract')
    require(all(np.array_equal(getattr(original_mesh, k), getattr(terminal_mesh, k)) for k in ('nodes_lonlat','depths','triangles')),
            'endpoint_fan_geometry_changed', 'Endpoint fan cannot change coordinates, depths or triangles')
    require(original_mesh.open_boundary_ids == terminal_mesh.open_boundary_ids, 'endpoint_fan_identity_changed', 'Plural nodestring IDs must be exact')
    plan = plan_endpoint_fan(len(original_mesh.depths), original_mesh.triangles-1,
                    original_mesh.open_boundary_chains, cyclic, tge_source, original_sha, trace.get('max_steps'))
    expected = dict(plan, terminal_mesh_sha256=terminal_sha)
    require(trace == expected, 'endpoint_fan_trace_replay_mismatch', 'Complete source/mesh-bound endpoint trace differs from independent replay')
    require([c.tolist() for c in terminal_mesh.open_boundary_chains] == plan['delivered_chains'],
            'endpoint_fan_terminal_order_mismatch', 'Terminal plural OBC order differs from replay')
    return {'passed': True, 'rule': RULE, 'trace_sha256': digest(trace), 'step_count': len(plan['steps']),
            'reclassified_node_ids_1based': [s['node_id_1based'] for s in plan['steps']],
            'planner_source_sha256': sha(__file__)}
