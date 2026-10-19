"""Focused retained-identity and admission regressions; real-grid replay is separate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import numpy as np
from certify_retained_grid_join import audit, certify, identity_remap, audit_sponge_values


class RetainedJoinTests(unittest.TestCase):
    def setUp(self):
        self.contract = {'open_boundaries': [{'obc_id': '0', 'source_obc_id': 0,
            'display_name': 'ocean', 'nodestring_id': 1}]}
        self.remap = {'chains': [{'delivered_node_ids_1based': [2, 3, 4]}],
            'cyclicity_contract': {'chains': [{'declared_open_boundary_id': 'ocean', 'nodestring_id': 1}]}}
        self.case = {'boundary': {'open_boundaries': [{'id': 'ocean', 'selection': 'exact finalized Adaptive-v2 obc_id 0'}]}}
        self.resolution = {'open_boundary_chains': [{'obc_id': 0}]}

    def normalize(self):
        return identity_remap(self.contract, self.remap, self.case, self.resolution)

    def test_only_alias_normalized_original_unchanged(self):
        before = copy.deepcopy(self.remap)
        result, aliases = self.normalize()
        self.assertEqual(self.remap, before)
        self.assertEqual(result['chains'], before['chains'])
        self.assertEqual(result['cyclicity_contract']['chains'][0]['declared_open_boundary_id'], '0')
        self.assertEqual(len(aliases), 1)

    def test_matching_ids_still_need_source_proof(self):
        self.contract['open_boundaries'][0]['obc_id'] = 'invented'
        self.remap['cyclicity_contract']['chains'][0]['declared_open_boundary_id'] = 'invented'
        with self.assertRaises(ValueError): self.normalize()

    def test_valid_already_source_named_identity(self):
        self.remap['cyclicity_contract']['chains'][0]['declared_open_boundary_id'] = '0'
        normalized, aliases = self.normalize()
        self.assertEqual(normalized, self.remap)
        self.assertFalse(aliases)

    def test_unrelated_case_selection_rejected(self):
        self.case['boundary']['open_boundaries'][0]['selection'] = 'some other source'
        with self.assertRaises(ValueError): self.normalize()

    def implicit_single_cycle(self):
        ids=[1,2,3,4]
        self.contract['open_boundaries'][0].update(cyclic=True,node_ids=ids,
            node_order_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
        self.case['boundary']['open_boundaries'][0]={'id':'ocean','cyclic':True}
        self.resolution['open_boundary_chains'][0]['is_closed']=True
        self.remap['chains'][0].update(cyclic=True,nodestring_id=1,source_node_ids_1based=ids,
            delivered_node_ids_1based=ids,source_node_count=4,delivered_node_count=4,
            orientation_preserved=True,source_sequence_retained_by_lineage=True)
        self.remap['cyclicity_contract']['chains'][0].update(cyclic=True,node_count=4)

    def test_implicit_single_cycle_preserves_all_original_bytes(self):
        self.implicit_single_cycle()
        before=copy.deepcopy((self.contract,self.remap,self.case,self.resolution))
        normalized,aliases=self.normalize()
        self.assertEqual(before,(self.contract,self.remap,self.case,self.resolution))
        self.assertEqual(normalized['chains'],self.remap['chains'])
        self.assertEqual(aliases,[{'nodestring_id':1,'source_obc_id':0,'before':'ocean','after':'0'}])

    def test_implicit_single_cycle_rejects_ambiguous_or_tampered_authorities(self):
        mutations=[
            lambda:self.case['boundary']['open_boundaries'].append({'id':'other','cyclic':True}),
            lambda:self.resolution['open_boundary_chains'].append({'obc_id':1,'is_closed':True}),
            lambda:self.contract['open_boundaries'].append(copy.deepcopy(self.contract['open_boundaries'][0])),
            lambda:self.remap['cyclicity_contract']['chains'].append(copy.deepcopy(self.remap['cyclicity_contract']['chains'][0])),
            lambda:self.remap['chains'].append(copy.deepcopy(self.remap['chains'][0])),
            lambda:self.contract['open_boundaries'][0].update(node_order_sha256='0'*64),
            lambda:self.contract['open_boundaries'][0].update(node_ids=[2,3,4,1]),
            lambda:self.remap['chains'][0].update(delivered_node_ids_1based=[2,3,4,1]),
            lambda:self.remap['chains'][0].update(source_node_ids_1based=[2,3,4,1]),
            lambda:self.remap['chains'][0].update(source_node_count=5),
            lambda:self.remap['chains'][0].update(orientation_preserved=False),
            lambda:self.remap['chains'][0].update(source_sequence_retained_by_lineage=False),
            lambda:self.contract['open_boundaries'][0].update(cyclic=False),
            lambda:self.case['boundary']['open_boundaries'][0].update(cyclic=False),
            lambda:self.resolution['open_boundary_chains'][0].update(is_closed=False),
            lambda:self.remap['chains'][0].update(cyclic=False),
            lambda:self.remap['cyclicity_contract']['chains'][0].update(cyclic=False),
            lambda:self.case['boundary']['open_boundaries'][0].update(source_obc_id=1),
            lambda:self.case['boundary']['open_boundaries'][0].update(selection='wrong source'),
            lambda:self.case['boundary']['open_boundaries'][0].update(source_obc_id=0,selection='wrong source'),
            lambda:self.case['boundary']['open_boundaries'][0].update(source_obc_id=1,selection='exact finalized Adaptive-v2 obc_id 0'),
        ]
        for i,mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                self.setUp(); self.implicit_single_cycle(); mutate()
                with self.assertRaises(ValueError): self.normalize()

    def test_duplicate_source_or_case_identity_rejected(self):
        for where in ('source', 'case'):
            with self.subTest(where=where):
                rows = self.resolution['open_boundary_chains'] if where == 'source' else self.case['boundary']['open_boundaries']
                rows.append(copy.deepcopy(rows[0]))
                with self.assertRaises(ValueError): self.normalize()
                rows.pop()

    def test_nodestring_change_rejected(self):
        self.remap['cyclicity_contract']['chains'][0]['nodestring_id'] = 2
        with self.assertRaises(ValueError): self.normalize()

    def test_fresh_cannot_use_retained_route_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'request.json'; out = Path(tmp)/'certificate'
            p.write_text(json.dumps({'schema_version':'fvcom_retained_grid_join_request_v1','workflow_kind':'fresh_generation','inputs':{}}))
            with self.assertRaisesRegex(ValueError, 'accepted retained'): certify(p, out)
            self.assertFalse(out.exists())

    def test_existing_revision_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); marker = p/'untouched'; marker.write_bytes(b'original')
            with self.assertRaisesRegex(ValueError, 'already exists'): certify(p/'missing.json', p)
            self.assertEqual(marker.read_bytes(), b'original')

    def test_positive_sponge_change_is_not_authorized_by_a_new_file_hash(self):
        pre = {'sponge_mode': 'estimate', 'sponge_by_obc': [{'obc_id': '0', 'nodestring_id': 1,
            'radius_m': 12.34567891, 'coefficient': .0025, 'override': None,
            'baseline_estimate': {'sponge_radius_m': 12.34567891, 'sponge_coefficient': .0025}}]}
        values = np.asarray([[2, 12.345679, .0025], [3, 12.345679, .0025]])
        audit_sponge_values(pre, self.contract['open_boundaries'], [[2, 3]], values)
        values[0, 1] *= 2
        with self.assertRaisesRegex(ValueError, 'DAT differs'):
            audit_sponge_values(pre, self.contract['open_boundaries'], [[2, 3]], values)
        values[0, 1] /= 2
        pre['sponge_by_obc'][0]['radius_m'] *= 2
        with self.assertRaisesRegex(ValueError, 'baseline'):
            audit_sponge_values(pre, self.contract['open_boundaries'], [[2, 3]], values)

if __name__ == '__main__': unittest.main()
