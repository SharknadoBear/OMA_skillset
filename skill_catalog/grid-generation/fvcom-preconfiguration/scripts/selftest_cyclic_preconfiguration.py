"""Cyclic OBC topology/spacing regression; unequal edge lengths expose seam loss."""
from pathlib import Path
import json, math, tempfile
from mesh_io import parse_2dm, write_fvcom_dat, ordered_nodes_sha256
from selftest_contract_preconfiguration import MESH_TEXT

FAN = '''MESH2D
E3T 1 1 2 5 1
E3T 2 2 3 5 1
E3T 3 3 4 5 1
E3T 4 4 1 5 1
ND 1 0 0 5
ND 2 1 0 5
ND 3 1 2 5
ND 4 0 5 5
ND 5 0.5 1 5
'''

def main():
    with tempfile.TemporaryDirectory(prefix='cyclic_preconfiguration_') as raw:
        root=Path(raw)
        def run(nodes, declared=None, label='case', order=None):
            path=root/(label+'.2dm')
            path.write_text(FAN+'NS '+' '.join(map(str,nodes[:-1]))+' '+str(-nodes[-1])+' 1\n')
            boundary={'obc_id':'offshore_cycle','nodestring_id':1,'node_ids':nodes if order is None else order,
                      'node_order_sha256':ordered_nodes_sha256(nodes if order is None else order)}
            if declared is not None: boundary['cyclic']=declared
            return write_fvcom_dat(parse_2dm(path), root/label, label, open_boundaries=[boundary])
        expected=3*(2+math.sqrt(10))/2
        for label,nodes in [('cycle',[1,2,3,4]),('rotate',[3,4,1,2]),('reverse',[4,3,2,1])]:
            result=run(nodes,True,label)
            assert result['status']=='ready'
            assert result['open_boundaries'][0]['cyclic'] is True
            assert result['open_boundaries'][0]['node_ids']==nodes
            assert result['open_boundaries'][0]['node_order_sha256']==ordered_nodes_sha256(nodes)
            assert result['obc_topology_audit']['declared_open_edge_count']==4
            assert result['obc_topology_audit']['accepted_open_boundary_cell_ids']==[1,2,3,4]
            sponge=result['sponge_by_obc'][0]['baseline_estimate']
            assert sponge['edge_count']==4 and sponge['cyclic'] is True
            assert math.isclose(sponge['sponge_radius_m'],expected,rel_tol=0,abs_tol=1e-12)
            assert len((root/label/(label+'_obc.dat')).read_text().splitlines())==5
        legacy=run([1,2,3,4],None,'legacy')
        explicit_open=run([1,2,3,4],False,'open')
        assert 'cyclic' not in legacy['open_boundaries'][0]
        assert explicit_open['open_boundaries'][0]['cyclic'] is False
        assert legacy['sponge_by_obc'][0]['baseline_estimate']['edge_count']==3
        assert legacy['sponge_by_obc'][0]['radius_m']==6.0
        for key in legacy['generated_files']:
            assert Path(legacy['generated_files'][key]).read_bytes()==Path(explicit_open['generated_files'][key]).read_bytes()
        def rejects(fn,message):
            try: fn()
            except ValueError as e: assert message in str(e),str(e)
            else: raise AssertionError('Invalid cyclic contract accepted: '+message)
        rejects(lambda:run([1,2,3],True,'bad_closure'),'FVCOM-invalid open-boundary topology')
        rejects(lambda:run([1,2],True,'short'),'at least three')
        rejects(lambda:run([1,2,3,4,1],True,'duplicate'),'unique nodes')
        rejects(lambda:run([1,2,3,4],'true','string_bool'),'must be a boolean')
        rejects(lambda:run([1,2,3,4],True,'bad_order',order=[2,3,4,1]),'node order mismatch')
        # Two separate noncyclic chains retain independent order and all DAT bytes.
        plural=root/'plural.2dm'; plural.write_text(MESH_TEXT)
        a=write_fvcom_dat(parse_2dm(plural),root/'plural_legacy','plural',open_ns=[1,2])
        b=write_fvcom_dat(parse_2dm(plural),root/'plural_explicit','plural',open_boundaries=[
            {'obc_id':'obc_001','nodestring_id':1,'cyclic':False},
            {'obc_id':'obc_002','nodestring_id':2,'cyclic':False}])
        assert [x['node_ids'] for x in b['open_boundaries']]==[[2,3,4],[12,13,14]]
        for key in a['generated_files']: assert Path(a['generated_files'][key]).read_bytes()==Path(b['generated_files'][key]).read_bytes()
    print(json.dumps({'passed':True,'cases':['unequal_edge_cyclic','seam_rotation','source_reversal','legacy_noncyclic','plural_noncyclic_byte_parity','nonexterior_closure_rejection','cyclic_two_nodes_rejection','repeated_closure_rejection','nonboolean_rejection','changed_order_rejection']}))
    return 0

if __name__=='__main__': raise SystemExit(main())
