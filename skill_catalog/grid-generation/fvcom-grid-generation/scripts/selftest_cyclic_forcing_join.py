#!/usr/bin/env python3
"""Prove cyclic package certification without weakening standalone SMS evidence."""
import json,unittest
import numpy as np
import selftest_forcing_join as fixtures
from fvcom_grid_generation import forcing_join
from fvcom_grid_generation.grid_project import validate
from fvcom_grid_generation.sms_2dm import read_2dm,write_2dm
read,write,digest=fixtures.read,fixtures.write,fixtures.digest
CYCLIC='cyclic_obc_not_self_describing_in_sms_2dm'

class CyclicForcingTests(unittest.TestCase):
    cyclic=True
    setUpClass=classmethod(fixtures.ForcingJoinTests.setUpClass.__func__)
    tearDownClass=classmethod(fixtures.ForcingJoinTests.tearDownClass.__func__)
    setUp=fixtures.ForcingJoinTests.setUp
    tearDown=fixtures.ForcingJoinTests.tearDown
    join=fixtures.ForcingJoinTests.join
    rejection=fixtures.ForcingJoinTests.rejection
    rehash_forcing=fixtures.ForcingJoinTests.rehash_forcing

    def test_cyclic_package_passes_but_original_finding_remains(self):
        before={p.name:digest(p) for p in (self.p/'final').iterdir() if p.name!='fvcom_grid_status.json'}
        self.assertFalse(validate(self.p,require_submission_ready=True)['passed'])
        result=self.join();self.assertTrue(result['passed'],result)
        cert=read(self.p/'08_audit/forcing_joins/v001/certificate.json')
        self.assertEqual(cert['schema_version'],'fvcom_terminal_forcing_certificate_v2')
        self.assertIn(CYCLIC,cert['resolved_submission_preconditions'])
        proof=cert['audit']['cyclic_sidecar_audit']
        self.assertTrue(proof['passed']);self.assertEqual(proof['chains'][0]['closing_edge_1based'],[6,1])
        self.assertEqual(proof['chains'][0]['boundary_edge_count'],6)
        self.assertFalse(proof['standalone_sms_self_describing'])
        self.assertIn(CYCLIC,read(self.p/'final/mesh_quality.json')['all_quality_findings'])
        self.assertEqual(before,{name:digest(self.p/'final'/name) for name in before})
        self.assertTrue(validate(self.p,require_submission_ready=True)['passed'])

    def proof(self,chain=None):
        contract=read(self.args['grid_contract']);remap=read(self.p/'final/obc_remap_manifest.json')
        if chain is not None:contract['open_boundaries'][0]['node_ids']=chain
        return forcing_join._audit_cyclic_sidecar(read_2dm(self.p/'final/fvcom_grid.2dm'),contract['open_boundaries'],remap['chains'],remap['cyclicity_contract']['chains'])
    def test_closing_edge_must_be_actual_boundary(self):
        with self.assertRaisesRegex(ValueError,'boundary edge'):self.proof([1,2,3,4,5])
    def test_interior_edges_cannot_make_a_cyclic_obc(self):
        with self.assertRaisesRegex(ValueError,'boundary edge'):self.proof([1,2,7])
    def test_agreeing_sidecars_cannot_hide_missing_closing_edge(self):
        meshpath=self.p/'final/fvcom_grid.2dm';mesh=read_2dm(meshpath);chain=[1,2,3,4,5]
        write_2dm(meshpath,mesh.nodes_lonlat,np.full(7,5.),mesh.triangles,np.empty(0,dtype=int),open_boundary_chains=[chain],open_boundary_ids=[11])
        c=read(self.args['grid_contract']);c['mesh']['sha256']=digest(meshpath);b=c['open_boundaries'][0]
        b['node_ids']=chain;b['node_order_sha256']=fixtures.hashlib.sha256(json.dumps(chain,separators=(',',':')).encode()).hexdigest();write(self.args['grid_contract'],c)
        path=self.p/'final/obc_remap_manifest.json';d=read(path)
        d['chains'][0].update(delivered_node_ids_1based=chain,delivered_node_count=5);d['cyclicity_contract']['chains'][0]['node_count']=5;write(path,d)
        with self.assertRaisesRegex(ValueError,'boundary edge'):
            forcing_join.audit_terminal_forcing(meshpath,self.args['grid_contract'],self.args['obc_points'],self.args['forcing'],self.args['forcing_manifest'],path,require_cyclic_sidecar=True)
    def test_two_node_loop_rejected(self):
        with self.assertRaisesRegex(ValueError,'three distinct'):self.proof([1,2])
    def test_repeated_seam_rejected(self):
        with self.assertRaisesRegex(ValueError,'three distinct'):self.proof([1,2,3,4,5,6,1])
    def test_numeric_cyclicity_is_not_a_boolean(self):
        path=self.p/'final/obc_remap_manifest.json';d=read(path);d['chains'][0]['cyclic']=1;write(path,d)
        with self.assertRaisesRegex(ValueError,'boolean'):self.proof()
    def test_legacy_schema_cannot_claim_cyclic_resolution(self):
        self.assertTrue(self.join()['passed']);p=self.p/'08_audit/forcing_joins/v001/certificate.json'
        cert=read(p);cert['schema_version']='fvcom_terminal_forcing_certificate_v1';write(p,cert)
        for name in ['project_status.json','final/fvcom_grid_status.json','08_audit/forcing_joins/v001/delivery_status.json']:
            d=read(self.p/name);d['terminal_forcing_certificate']['sha256']=digest(p);write(self.p/name,d)
        self.assertIn('terminal_forcing_certificate_invalid',validate(self.p,require_submission_ready=True)['failure_taxonomy'])
    def test_actual_cyclic_sidecar_changes_rejected(self):
        self.assertTrue(self.join()['passed']);p=self.p/'final/obc_remap_manifest.json';d=read(p);d['cyclicity_contract']['chains'][0]['cyclic']=False;write(p,d)
        self.assertIn('terminal_forcing_certificate_invalid',validate(self.p,require_submission_ready=True)['failure_taxonomy'])
    def test_proof_tampering_rejected_even_with_new_certificate_hash(self):
        self.assertTrue(self.join()['passed']);p=self.p/'08_audit/forcing_joins/v001/certificate.json';d=read(p)
        d['audit']['cyclic_sidecar_audit']['chains'][0]['closing_edge_1based']=[6,2];write(p,d)
        for name in ['project_status.json','final/fvcom_grid_status.json','08_audit/forcing_joins/v001/delivery_status.json']:
            d=read(self.p/name);d['terminal_forcing_certificate']['sha256']=digest(p);write(self.p/name,d)
        self.assertIn('terminal_forcing_certificate_invalid',validate(self.p,require_submission_ready=True)['failure_taxonomy'])
    test_nonfinite_forcing_still_rejected=fixtures.ForcingJoinTests.test_nan_actual_forcing
    test_unrelated_gate_still_rejected=fixtures.ForcingJoinTests.test_other_gate_never_waived
    test_bad_order_still_rejected=fixtures.ForcingJoinTests.test_wrong_contract_order
    def test_disagreeing_cyclicity_still_rejected(self):
        c=read(self.args['grid_contract']);c['open_boundaries'][0]['cyclic']=False;write(self.args['grid_contract'],c);self.rejection()
    test_wrong_identity_still_rejected=fixtures.ForcingJoinTests.test_wrong_nodestring_identity
    test_missing_tge_still_rejected=fixtures.ForcingJoinTests.test_missing_tge
    test_removed_certificate_still_rejected=fixtures.ForcingJoinTests.test_certificate_removal_cannot_bypass

if __name__=='__main__':unittest.main(verbosity=2)
