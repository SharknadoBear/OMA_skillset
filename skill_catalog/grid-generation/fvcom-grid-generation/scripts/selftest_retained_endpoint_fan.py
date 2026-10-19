"""Retained-certificate admission/archived-planner checks with isolated fixtures."""
import argparse,copy,json
from pathlib import Path
from unittest.mock import patch
import certify_retained_grid_join as cert
import repair_fvcom_tge_endpoints as owner

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fan-fixtures',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False);fixtures=args.fan_fixtures.resolve()
    results=[]
    original=fixtures/'two_input';terminal=fixtures/'sequential_two_step_result/delivery';source=fixtures/'tge.F'
    base_contract=json.loads((terminal/'grid_delivery_contract.json').read_text())
    for entry in [base_contract['mesh'],*base_contract['delivery_artifacts'].values()]:
        entry['path']=str((terminal/entry['path']).resolve())
    original_mesh=original/'mesh.2dm'
    dummy=root/'dummy.json';owner.dump(dummy,{})
    retention=root/'retention.json';owner.dump(retention,{'schema_version':'fvcom_grid_post_cleanup_integrity_v1','final_mesh_audit_passed':True,'protected_t6v6_files_checked':1,'protected_t6v6_files_matching':1,'protected_t6v6_files_missing':0,'protected_t6v6_files_mismatching':0})
    def request(name,modify):
        place=root/name;place.mkdir();doc=copy.deepcopy(base_contract)
        trace=json.loads((terminal/'endpoint_fan_trace.json').read_text());modify(doc,trace)
        if 'endpoint_fan_trace' in doc['delivery_artifacts']:
            owner.dump(place/'endpoint_fan_trace.json',trace)
            doc['delivery_artifacts']['endpoint_fan_trace']=cert.bind(place/'endpoint_fan_trace.json')
        owner.dump(place/'contract.json',doc)
        inputs={k:cert.bind(dummy) for k in cert.REQUIRED}
        inputs.update(grid_contract=cert.bind(place/'contract.json'),original_mesh=cert.bind(original_mesh),original_boundary_nodes=cert.bind(original/'boundary_nodes.geojson'),retention_evidence=cert.bind(retention),tge_source=cert.bind(source))
        path=place/'request.json';owner.dump(path,{'schema_version':'fvcom_retained_grid_join_request_v1','workflow_kind':'accepted_retained','inputs':inputs})
        return path
    def expect(name,mutate,expected):
        path=request(name,mutate)
        try:cert.audit(path)
        except (ValueError,KeyError) as exc:
            # This deliberate fixture stops at original-publication admission
            # after successful endpoint replay; no forcing certificate is made.
            actual=getattr(exc,'code',str(exc))
            assert expected in actual,(name,expected,actual)
        else:raise AssertionError('Expected fixture stop/rejection')
        results.append({'name':name,'passed':True,'expected_gate':expected})
    expect('valid_extended_trace_reaches_next_gate',lambda d,t:None,'retained source publication does not bind original mesh')
    expect('missing_trace_rejected',lambda d,t:d['delivery_artifacts'].pop('endpoint_fan_trace'),'complete endpoint-fan replay trace')
    expect('empty_trace_rejected',lambda d,t:t.clear(),'invalid_endpoint_fan_trace')
    expect('missing_step_rejected',lambda d,t:t['steps'].pop(),'endpoint_fan_trace_replay_mismatch')
    expect('reordered_steps_rejected',lambda d,t:t['steps'].reverse(),'endpoint_fan_trace_replay_mismatch')
    expect('altered_source_hash_rejected',lambda d,t:t.update(tge_source_sha256='0'*64),'endpoint_fan_trace_replay_mismatch')
    expect('altered_geometry_hash_rejected',lambda d,t:t.update(mesh_sha256='0'*64),'endpoint_fan_trace_replay_mismatch')
    expect('increased_trace_bound_rejected',lambda d,t:t.update(max_steps=3),'endpoint_fan_bound_binding_mismatch')
    expect('missing_contract_bound_rejected',lambda d,t:d['source_delivery'].pop('endpoint_fan_max_steps'),'endpoint_fan_bound_binding_mismatch')
    # Exercise archive plumbing independent of real forcing/science: a mocked
    # audit result is explicitly labeled synthetic and never used operationally.
    archive=root/'synthetic_archive_plumbing_only';request_path=root/'synthetic_request.json';owner.dump(request_path,{'synthetic_fixture_only':True})
    synthetic={'passed':True,'synthetic_fixture_only':True,'endpoint_fan_replay':{'passed':True}}
    with patch.object(cert,'audit',return_value=synthetic):
        receipt=cert.certify(request_path,archive);assert receipt['passed']
        cpath=archive/'certificate.json';saved=cpath.read_bytes();doc=json.loads(saved)
        planner=archive/'fvcom_tge_endpoint_fan.py';planner_saved=planner.read_bytes()
        assert cert._hash(planner)==cert._hash(cert.fvcom_tge_endpoint_fan.__file__)
        results.append({'name':'planner_archived_and_bound','passed':True})
        for name,mutate in [
            ('missing_planner_binding',lambda d:d.pop('endpoint_fan_planner')),
            ('stale_planner_hash',lambda d:d['endpoint_fan_planner'].update(sha256='0'*64)),
        ]:
            changed=copy.deepcopy(doc);mutate(changed);owner.dump(cpath,changed)
            try:cert.verify(cpath)
            except ValueError:pass
            else:raise AssertionError(name)
            cpath.write_bytes(saved);results.append({'name':name,'passed':True})
        planner.write_bytes(planner_saved+b'\n# tampered planner fixture\n')
        changed=copy.deepcopy(doc);changed['endpoint_fan_planner']=cert.bind(planner);owner.dump(cpath,changed)
        try:cert.verify(cpath)
        except ValueError as exc:assert 'exact archived endpoint-fan planner version' in str(exc)
        else:raise AssertionError('A rehashed different planner was accepted')
        planner.write_bytes(planner_saved);cpath.write_bytes(saved)
        results.append({'name':'rehashed_changed_planner_rejected','passed':True})
        assert cert.verify(cpath)['passed']
    owner.dump(root/'test_results.json',{'status':'pass','test_count':len(results),'tests':results,'synthetic_archive_is_not_scientific_certificate':True})
    print(json.dumps({'status':'pass','test_count':len(results)}))

if __name__=='__main__':main()
