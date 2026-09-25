"""Focused opt-in endpoint-fan owner/replay regressions; no production adoption."""
import argparse,copy,json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pyproj import Transformer
import repair_fvcom_tge_endpoints as owner
import fvcom_tge_endpoint_fan as fan
from fvcom_grid_generation.sms_2dm import read_2dm,write_2dm

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--mobile-contract',type=Path)
    parser.add_argument('--mobile-boundary',type=Path)
    parser.add_argument('--mobile-remap',type=Path)
    parser.add_argument('--tge-source',type=Path)
    args=parser.parse_args();root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    source=root/'tge.F';source.write_text('ISONB = 0\nISONB(NV(I,2)) = 1\nISONB(I_OBC_N(I))=2\nIF(SUM(ISONB(NV(I,1:3))) == 4) THEN\nELSE IF(SUM(ISONB(NV(I,1:3))) > 4) THEN\nEND IF\n')
    results=[]
    def record(name,action):
        action();results.append({'name':name,'passed':True})
    def reject(call,code):
        try:call()
        except (ValueError,KeyError,TypeError) as exc:
            actual=getattr(exc,'code',type(exc).__name__)
            assert actual==code,(actual,code,str(exc))
        else:raise AssertionError('Expected rejection: '+code)
    def fixture(name,chains,cyclic=None):
        place=root/name;place.mkdir();cyclic=cyclic or [False]*len(chains);ids=list(range(11,11+len(chains)))
        xy=np.array([[0,0],[1,0],[2,0],[3,0],[3,3],[2,3],[1,3],[0,3],[-1,1.5],[1.5,1.5]])*100+[450000,5100000]
        ll=np.column_stack(Transformer.from_crs(32610,4326,always_xy=True).transform(*xy.T))
        tri=np.array([[9,1,2],[9,2,3],[9,3,10],[3,4,10],[4,5,10],[5,6,10],[6,7,10],[7,8,10],[8,9,10]])
        write_2dm(place/'mesh.2dm',ll,np.full(10,10.),tri,np.empty(0,dtype=int),open_boundary_chains=chains,open_boundary_ids=ids)
        mesh=read_2dm(place/'mesh.2dm')
        contract={'schema_version':'fvcom_grid_delivery_contract_v1','mesh':{'path':'mesh.2dm','sha256':owner.sha(place/'mesh.2dm'),'node_count':10,'triangle_count':9,'coordinate_crs':'EPSG:4326','horizontal_crs':'EPSG:32610','horizontal_units':'m'},
            'open_boundaries':[{'obc_id':f'arc_{ns}','nodestring_id':ns,'node_ids':c,'node_order_sha256':owner.order_sha(c),'cyclic':cyc,'obc_type':'prescribed'} for ns,c,cyc in zip(ids,chains,cyclic)],'excluded_nodestring_ids':[],'sigma':{'levels':10,'type':'UNIFORM'}}
        features=[]
        for node in range(1,10):
            membership=[ns for ns,c in zip(ids,chains) if node in c]
            features.append({'type':'Feature','geometry':{'type':'Point','coordinates':mesh.nodes_lonlat[node-1].tolist()},'properties':{'node_id_1based':node,'node_index_zero_based':node-1,'constraint_chain_id':0,'constraint_chain_position':node-1,'is_hard_anchor':True,'is_open_boundary':bool(membership),'open_boundary_nodestring_ids':membership,'open_boundary_cyclic':[cyclic[ids.index(ns)] for ns in membership]}})
        remap={'schema_version':'fvcom_obc_remap_manifest_v1','chains':[{'nodestring_id':ns,'delivered_node_ids_1based':c,'delivered_node_count':len(c),'cyclic':cyc,'orientation_preserved':True} for ns,c,cyc in zip(ids,chains,cyclic)],'cyclicity_contract':{'chains':[{'nodestring_id':ns,'node_count':len(c),'cyclic':cyc} for ns,c,cyc in zip(ids,chains,cyclic)]}}
        for name,value in [('grid_delivery_contract.json',contract),('boundary_nodes.geojson',{'type':'FeatureCollection','features':features}),('obc_remap_manifest.json',remap)]:owner.dump(place/name,value)
        return place
    def run(name,loc,expected,steps=0):
        paths=[loc/n for n in ['grid_delivery_contract.json','boundary_nodes.geojson','obc_remap_manifest.json']]
        before={str(p):owner.sha(p) for p in paths+[source,loc/'mesh.2dm']}
        out=root/(name+'_result');result=owner.repair(*paths,source,out,endpoint_fan_max_steps=steps)
        actual=result['status'] if result['status']=='ready' else result['blocking_reasons'][0]['code']
        assert actual==expected,(name,actual,result)
        assert before=={str(p):owner.sha(p) for p in paths+[source,loc/'mesh.2dm']}
        if actual!='ready':assert not (out/'delivery').exists() and not any((out/n).exists() for n in ['fvcom_grid.2dm','boundary_nodes.geojson','obc_remap_manifest.json','grid_delivery_contract.json'])
        results.append({'name':name,'passed':True,'decision':actual});return result
    two=fixture('two_input',[[1,2,3,4]])
    run('legacy_default_rejects_two_step',two,'ambiguous_or_nonendpoint_fatal_cell')
    run('exhausted_bound_atomic',two,'endpoint_fan_bound_exhausted',1)
    for bound in [-1,5,1.5,True]:run('invalid_bound_'+str(bound),two,'invalid_endpoint_fan_bound',bound)
    passing=run('sequential_two_step',two,'ready',2)
    original=read_2dm(two/'mesh.2dm');terminal=read_2dm(passing['output_paths']['mesh']);trace=json.loads(Path(passing['endpoint_fan']['trace_path']).read_text())
    assert passing['reclassified_node_ids_1based']==[1,2] and terminal.open_boundary_chains[0].tolist()==[3,4]
    assert [s['node_id_1based'] for s in trace['steps']]==[1,2]
    non_ns=lambda p:b''.join(line for line in Path(p).read_bytes().splitlines(keepends=True) if not line.startswith(b'NS '))
    assert non_ns(two/'mesh.2dm')==non_ns(passing['output_paths']['mesh'])
    def replay(t=trace,old=original,new=terminal,s=source):
        return fan.replay_endpoint_fan(t,old,new,[False],s,owner.sha(two/'mesh.2dm'),owner.sha(passing['output_paths']['mesh']),2)
    record('complete_trace_independent_replay',lambda:replay())
    mutations={
        'missing_trace':lambda t:t.clear(),
        'missing_steps':lambda t:t.pop('steps'),
        'altered_node':lambda t:t['steps'][0].update(node_id_1based=2),
        'missing_step':lambda t:t['steps'].pop(),
        'reordered_steps':lambda t:t['steps'].reverse(),
        'changed_current_endpoint_cells':lambda t:t['steps'][0].update(current_endpoint_cell_ids=[]),
        'changed_source_hash':lambda t:t.update(tge_source_sha256='0'*64),
        'changed_mesh_hash':lambda t:t.update(mesh_sha256='0'*64),
        'changed_terminal_hash':lambda t:t.update(terminal_mesh_sha256='0'*64),
        'changed_planner_hash':lambda t:t.update(planner_source_sha256='0'*64),
        'changed_bound':lambda t:t.update(max_steps=1),
        'increased_bound':lambda t:t.update(max_steps=3),
        'changed_chain_order':lambda t:t['delivered_chains'][0].reverse(),
        'fabricated_extra_step':lambda t:t['steps'].append(copy.deepcopy(t['steps'][0])),
    }
    for name,mutate in mutations.items():
        altered=copy.deepcopy(trace);mutate(altered)
        expected='invalid_endpoint_fan_trace' if name=='missing_trace' else ('endpoint_fan_bound_binding_mismatch' if name in ('changed_bound','increased_bound') else 'endpoint_fan_trace_replay_mismatch')
        record('trace_'+name,lambda t=altered,e=expected:reject(lambda:replay(t),e))
    changed=copy.deepcopy(terminal);changed.depths[0]+=1
    record('tampered_terminal_depth',lambda:reject(lambda:replay(new=changed),'endpoint_fan_geometry_changed'))
    changed_xy=copy.deepcopy(terminal);changed_xy.nodes_lonlat[0,0]+=.001
    record('tampered_terminal_geometry',lambda:reject(lambda:replay(new=changed_xy),'endpoint_fan_geometry_changed'))
    changed_tri=copy.deepcopy(terminal);changed_tri.triangles[0]=changed_tri.triangles[1]
    record('tampered_terminal_connectivity',lambda:reject(lambda:replay(new=changed_tri),'endpoint_fan_geometry_changed'))
    bad_source=root/'bad_tge.F';bad_source.write_text(source.read_text()+'! source changed\n')
    record('tampered_valid_source_bytes',lambda:reject(lambda:replay(s=bad_source),'endpoint_fan_trace_replay_mismatch'))
    plural=run('plural_untouched_order',fixture('plural_input',[[1,2,3,4],[6,7]]),'ready',2)
    pm=read_2dm(plural['output_paths']['mesh']);assert pm.open_boundary_ids==(11,12) and pm.open_boundary_chains[1].tolist()==[6,7]
    run('cross_chain_preflight',fixture('cross_input',[[1,2],[3,4]]),'cross_chain_endpoint_fan',2)
    run('ambiguous_initial_anchor',fixture('ambiguous_input',[[1,2,3]]),'unanchored_or_ambiguous_endpoint_fan',2)
    run('disconnected_chain_preflight',fixture('disconnected_chain',[[1,3,4]]),'preexisting_mesh_failure',2)
    run('unsupported_sum_six',fixture('unsupported_input',[[9,1,2,3,4]]),'unsupported_tge_cell',2)
    # A cyclic chain with incomplete serialized coverage is refused before any
    # demotion; helper-level preflight checks a fully cyclic chain's sum-six debt.
    cyclic=fixture('cyclic_input',[[1,2,3,4]],[True])
    run('cyclic_blocked_without_partial',cyclic,'preexisting_mesh_failure',2)
    # Targeted causal-graph fixtures isolate branches not naturally expressible
    # by this small triangulation. Their mocked audits are explicitly test-only.
    base_audit=fan.audit_tge_boundary_junctions(10,original.triangles-1,[np.array([1,2,3,4])-1],tge_source_path=source)
    synthetic=copy.deepcopy(base_audit)
    synthetic['fatal_cells'].append(dict(synthetic['fatal_cells'][1],element_id_1based=99,node_ids_1based=[8,3,4],open_boundary_node_ids_1based=[3,4]))
    # Anchor4 would make a second valid component. Omit it from the declared arc
    # using a longer valid boundary chain so3,4 are interior to the membership.
    record('detached_interior_fatal_component',lambda:detached_test(original,source,synthetic))
    # Force unchanged debt to prove the planner refuses a source audit that does
    # not strictly decrease after the unique endpoint proposal.
    with patch.object(fan,'audit_tge_boundary_junctions',return_value=base_audit):
        record('no_progress_atomic_plan',lambda:reject(lambda:fan.plan_endpoint_fan(10,original.triangles-1,[[1,2,3,4]],[False],source,owner.sha(two/'mesh.2dm'),2),'endpoint_fan_no_progress'))
    owning_audit=owner.audit_tge_boundary_junctions
    for mutation,expected in [('serialized_mismatch','serialized_tge_mismatch'),('late_input_change','inputs_changed_during_repair')]:
        calls=[0];saved_source=source.read_bytes()
        def changed_audit(*a,**kw):
            result=owning_audit(*a,**kw);calls[0]+=1
            if calls[0]==3:
                if mutation=='serialized_mismatch':result=dict(result,passed=False)
                else:source.write_bytes(saved_source+b'\n! simulated external mutation\n')
            return result
        try:
            with patch.object(owner,'audit_tge_boundary_junctions',side_effect=changed_audit):
                out=root/(mutation+'_result')
                result=owner.repair(two/'grid_delivery_contract.json',two/'boundary_nodes.geojson',two/'obc_remap_manifest.json',source,out,endpoint_fan_max_steps=2)
            assert result['status']=='blocked' and result['blocking_reasons'][0]['code']==expected,result
            assert not (out/'delivery').exists() and not (out/'grid_delivery_contract.json').exists()
            assert (out/'_rejected_delivery_diagnostic/DO_NOT_USE.json').is_file()
            results.append({'name':mutation+'_keeps_corrected_delivery_unpublished','passed':True,'decision':expected})
        finally:source.write_bytes(saved_source)
    if args.mobile_contract:
        out=root/'mobile_diag'
        result=owner.repair(args.mobile_contract,args.mobile_boundary,args.mobile_remap,args.tge_source,out,endpoint_fan_max_steps=2)
        assert result['status']=='ready' and result['reclassified_node_ids_1based']==[1,2],result
        old_contract=json.loads(args.mobile_contract.read_text());oldpath=Path(old_contract['mesh']['path']);old=read_2dm(oldpath);new=read_2dm(result['output_paths']['mesh'])
        tr=json.loads(Path(result['endpoint_fan']['trace_path']).read_text())
        receipt=fan.replay_endpoint_fan(tr,old,new,[False],args.tge_source,owner.sha(oldpath),owner.sha(result['output_paths']['mesh']),2)
        assert non_ns(oldpath)==non_ns(result['output_paths']['mesh']) and new.open_boundary_chains[0].tolist()==list(range(3,251))
        owner.dump(out/'DIAGNOSTIC_ONLY.json',{'production_adoption_authorized':False,'scope':'isolated prospective test only','replay':receipt})
        results.append({'name':'mobile_prospective_diagnostic_only','passed':True,'reclassified_nodes':[1,2],'terminal_mesh_sha256':owner.sha(result['output_paths']['mesh']),'production_adoption_authorized':False})
    owner.dump(root/'test_results.json',{'status':'pass','test_count':len(results),'tests':results})
    print(json.dumps({'status':'pass','test_count':len(results),'output_dir':str(root)}))

def detached_test(mesh,source,synthetic):
    with patch.object(fan,'audit_tge_boundary_junctions',return_value=synthetic):
        try:fan.plan_endpoint_fan(10,mesh.triangles-1,[[1,2,3,4,5]],[False],source,'0'*64,2)
        except fan.FanRejected as exc:assert exc.code=='unanchored_or_ambiguous_endpoint_fan',exc.code
        else:raise AssertionError('Detached interior fatal component was admitted')

if __name__=='__main__':main()
