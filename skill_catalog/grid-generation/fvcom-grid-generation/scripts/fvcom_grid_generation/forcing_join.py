"""Certify new terminal-order TPXO forcing without republishing immutable grids."""
from __future__ import annotations
import csv,hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
import netCDF4 as nc
import numpy as np

RULE='fvcom_terminal_forcing_certificate_v1'
RESOLVABLE={'open_boundary_forcing_missing','open_boundary_forcing_incompatible'}

def _read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def _hash(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def _require(condition,message):
    if not condition:raise ValueError(message)
def _regular(path):
    path=Path(path)
    _require(path.is_file() and not path.is_symlink(),'forcing join requires regular nonsymlink inputs')
    return path.resolve()

def _audit_prerequisites(status,mesh,tge_source,open_exterior_source,boundary_resolution_source):
    from .grid_project import _audit_mesh_tge,_portable_tge_audit,_portable_boundary_audit
    from .open_exterior import validate_grid_boundary_gate
    from .sms_2dm import read_2dm
    _require(status.get('obc_status')=='pass','published OBC placement status must pass')
    _require(_hash(tge_source)==status['fvcom_tge_source_bound_audit']['source_sha256'],'TGE source differs from frozen publication')
    tge=_portable_tge_audit(_audit_mesh_tge(read_2dm(mesh),tge_source));tge['mesh_sha256']=_hash(mesh)
    _require(tge['passed'] and tge['source_binding']['passed'],'actual source-bound TGE gate failed')
    boundary=_portable_boundary_audit(validate_grid_boundary_gate(open_exterior_source,boundary_resolution_source,policy=status['boundary_gate_policy'],strict_contract_required=True))
    _require(boundary['passed'] and boundary==status['open_exterior_audit'],'published boundary evidence is missing, stale or changed')
    return tge

def audit_terminal_forcing(mesh_path,contract_path,points_path,forcing_path,manifest_path,remap_path):
    """Read actual IDs, geographic coordinates, time, finite data and owner hashes."""
    from .sms_2dm import read_2dm
    paths=[_regular(p) for p in (mesh_path,contract_path,points_path,forcing_path,manifest_path,remap_path)]
    mesh_path,contract_path,points_path,forcing_path,manifest_path,remap_path=paths
    mesh=read_2dm(mesh_path);contract=_read(contract_path);manifest=_read(manifest_path)
    _require(contract.get('schema_version')=='fvcom_grid_delivery_contract_v1','unsupported grid contract')
    _require(contract.get('mesh',{}).get('sha256')==_hash(mesh_path),'terminal mesh and contract hash disagree')
    _require(contract['mesh'].get('coordinate_crs')=='EPSG:4326','forcing join requires geographic terminal coordinates')
    _require(contract['mesh'].get('node_count')==len(mesh.nodes_lonlat) and contract['mesh'].get('triangle_count')==len(mesh.triangles),'terminal contract mesh counts disagree')
    chains=[list(map(int,b['node_ids'])) for b in contract.get('open_boundaries',[])]
    _require(chains==[list(map(int,c)) for c in mesh.open_boundary_chains],'contract chain order differs from terminal mesh')
    boundaries=contract['open_boundaries'];remap=_read(remap_path)
    nsids=[b['nodestring_id'] for b in boundaries]
    _require(nsids==list(mesh.open_boundary_ids),'contract nodestring identity/order differs from terminal mesh')
    stable_ids=[str(b['obc_id']) for b in boundaries]
    _require(all(stable_ids) and len(stable_ids)==len(set(stable_ids)),'missing or duplicate stable OBC IDs')
    mapped=remap['chains'];cyclicity=remap['cyclicity_contract']['chains']
    _require([c['nodestring_id'] for c in mapped]==nsids and [c['nodestring_id'] for c in cyclicity]==nsids,'remap chain identity/order differs from terminal contract')
    for boundary,chain,side,cycle in zip(boundaries,chains,mapped,cyclicity):
        order=hashlib.sha256(json.dumps(chain,separators=(',',':')).encode('ascii')).hexdigest()
        _require(len(chain)>=2 and boundary['node_order_sha256']==order,'invalid OBC size or stale chain order hash')
        _require(side['delivered_node_ids_1based']==chain and side['delivered_node_count']==len(chain) and cycle['node_count']==len(chain),'remap terminal chain order/count differs')
        _require(type(boundary['cyclic']) is bool and boundary['cyclic']==side['cyclic']==cycle['cyclic'],'contract cyclicity differs from immutable remap')
        if 'declared_open_boundary_id' in cycle:
            _require(str(cycle['declared_open_boundary_id'])==str(boundary['obc_id']),'contract stable OBC identity differs from remap')
    ids=np.asarray([i for chain in chains for i in chain],dtype=np.int64)
    _require(len(ids)>0 and len(set(ids.tolist()))==len(ids) and ids.min()>0 and ids.max()<=len(mesh.nodes_lonlat),'invalid or duplicate terminal OBC IDs')
    with points_path.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    actual_ids=np.asarray([int(r['node_id']) for r in rows],dtype=np.int64)
    xy=np.asarray([[float(r['longitude']),float(r['latitude'])] for r in rows])
    _require(np.array_equal(ids,actual_ids),'point IDs differ from terminal contract order')
    _require(xy.shape==(len(ids),2) and np.isfinite(xy).all() and np.allclose(xy,mesh.nodes_lonlat[ids-1],rtol=0,atol=1e-10),'point coordinates differ from terminal mesh')
    hashes=manifest.get('hashes',{});provenance=manifest.get('provenance',{})
    _require(manifest.get('status')=='ready' and not manifest.get('blocking_reasons') and manifest.get('health',{}).get('status')=='pass' and not manifest['health'].get('problems'),'forcing owner manifest is not ready')
    _require(manifest.get('interpolation',{}).get('unresolved_count')==0,'forcing has unresolved interpolation')
    _require(hashes.get('forcing_sha256')==_hash(forcing_path) and hashes.get('obc_points_sha256')==_hash(points_path),'forcing or point file hash differs from owner manifest')
    canonical=''.join(f'{int(i)},{x:.12f},{y:.12f}\n' for i,(x,y) in zip(ids,xy))
    order_hash=hashlib.sha256(canonical.encode('ascii')).hexdigest()
    _require(hashes.get('obc_order_sha256')==order_hash,'owner geographic OBC order hash disagrees')
    _require(provenance.get('reconstruction_rule_version')=='utide_exact_time_nodal_v1' and provenance.get('utide_ngflags')==[False]*4,'terminal forcing lacks exact-time nodal provenance')
    for key in ('source_harmonics_sha256','builder_sha256'):
        _require(re.fullmatch('[0-9a-f]{64}',str(hashes.get(key,''))) is not None,'missing forcing provenance hash: '+key)
    names=manifest.get('constituents',[])
    _require(len(names)>0 and len(names)==len(set(names))==manifest.get('constituent_count'),'invalid forcing constituent inventory')
    with nc.Dataset(forcing_path) as ds:
        _require(ds['obc_nodes'].dimensions==('nobc',) and ds['obc_nodes'].dtype.kind in 'iu','invalid NetCDF OBC variable')
        _require(np.array_equal(np.ma.asarray(ds['obc_nodes'][:]),ids),'actual NetCDF OBC IDs disagree')
        _require(not np.ma.getmaskarray(ds['obc_nodes'][:]).any(),'masked NetCDF OBC IDs')
        t=np.ma.asarray(ds['time'][:]);eta=ds['elevation']
        _require(ds['time'].dimensions==('time',) and getattr(ds['time'],'units',None)=='days since 1858-11-17 00:00:00' and getattr(ds['time'],'time_zone',None)=='UTC','forcing time must use UTC modified Julian days')
        _require(ds['time'].dtype.itemsize>=8 and not np.ma.getmaskarray(t).any() and np.isfinite(t.data).all() and len(t)>1 and np.all(np.diff(t)>0),'invalid terminal forcing time axis')
        _require(eta.dimensions==('time','nobc') and eta.shape==(len(t),len(ids)),'terminal forcing dimension/order mismatch')
        _require(getattr(eta,'units',None)=='meters','forcing elevation must be in meters')
        for begin in range(0,len(t),1024):
            values=np.ma.asarray(eta[begin:begin+1024])
            _require(not np.ma.getmaskarray(values).any() and np.isfinite(values.data).all(),'terminal forcing contains masked or nonfinite elevation')
        _require(len(t)==manifest.get('time_count') and len(ids)==manifest.get('obc_node_count'),'owner forcing counts disagree')
        interval=float(provenance['interval_minutes'])*60
        _require(interval>0 and np.allclose(np.diff(t)*86400,interval,rtol=0,atol=1e-5),'terminal forcing cadence differs from manifest')
        epoch=datetime(1858,11,17,tzinfo=timezone.utc)
        ends=[datetime.fromisoformat(provenance[k].replace('Z','+00:00')) for k in ('time_coverage_start','time_coverage_end')]
        _require(all(d.tzinfo is not None and d.utcoffset().total_seconds()==0 for d in ends),'forcing manifest times must be UTC')
        expected=np.asarray([(d-epoch).total_seconds()/86400 for d in ends])
        _require(np.allclose(np.asarray(t)[[0,-1]],expected,rtol=0,atol=1e-10),'terminal forcing endpoint mismatch')
        integer_time=np.ma.asarray(ds['Itime'][:]);milliseconds=np.ma.asarray(ds['Itime2'][:])
        _require(ds['Itime'].dtype.kind in 'iu' and ds['Itime2'].dtype.kind in 'iu','FVCOM integer time variables must be integers')
        _require(integer_time.shape==milliseconds.shape==t.shape and not np.ma.getmaskarray(integer_time).any() and not np.ma.getmaskarray(milliseconds).any(),'masked or invalid FVCOM integer times')
        _require(np.all((milliseconds>=0)&(milliseconds<86400000)) and np.allclose(integer_time+milliseconds/86400000,t,rtol=0,atol=1e-10),'FVCOM integer time encodings disagree')
        characters=np.ma.asarray(ds['Times'][:]);_require(not np.ma.getmaskarray(characters).any(),'masked FVCOM string times')
        strings=nc.chartostring(characters)
        string_times=np.asarray([(datetime.strptime(str(s),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)-epoch).total_seconds()/86400 for s in strings])
        _require(string_times.shape==t.shape and np.allclose(string_times,t,rtol=0,atol=1e-10),'FVCOM string time encoding disagrees')
        for attribute,expected_value in {
            'reconstruction_rule_version':'utide_exact_time_nodal_v1','utide_ngflags_json':json.dumps([False]*4),
            'utide_version':provenance.get('utide_version'),'builder_sha256':hashes['builder_sha256'],
            'source_harmonics_sha256':hashes['source_harmonics_sha256'],'obc_points_sha256':hashes['obc_points_sha256'],'obc_order_sha256':order_hash}.items():
            _require(expected_value is not None and getattr(ds,attribute,None)==expected_value,'NetCDF forcing provenance mismatch: '+attribute)
    return {'passed':True,'node_count':len(ids),'time_count':len(t),'obc_order_sha256':order_hash,
            'forcing_sha256':hashes['forcing_sha256'],'mesh_sha256':_hash(mesh_path),'constituents':names,
            'source_order_compatibility_reinterpreted':False,'terminal_order_forcing_compatible':True}

def verify_terminal_forcing_join(root,status):
    """Return resolved preconditions only after immutable evidence is rechecked."""
    root=Path(root).resolve();record=status['terminal_forcing_certificate']
    path=(root/record['path']).resolve()
    _require(path.is_relative_to(root) and not path.is_symlink() and _hash(path)==record['sha256'],'terminal forcing certificate missing, unsafe or stale')
    cert=_read(path);_require(cert.get('schema_version')==RULE,'unsupported terminal forcing certificate')
    inputs={}
    for key,binding in cert['inputs'].items():
        target=(root/binding['path']).resolve()
        _require(target.is_relative_to(root) and not target.is_symlink() and target.is_file() and _hash(target)==binding['sha256'],'terminal forcing certificate input changed: '+key)
        inputs[key]=target
    _require(inputs['mesh']==(root/status['mesh']['path']).resolve(),'certificate belongs to another terminal mesh')
    from .grid_project import FINAL_COMPANIONS
    for key,name in FINAL_COMPANIONS.items():
        _require(inputs[key]==(root/'final'/name).resolve(),'certificate does not bind the immutable final companion: '+key)
    _require(inputs['project_manifest']==root/'project_manifest.json','certificate project manifest mismatch')
    _require(inputs['source_tge_audit']==(root/status['fvcom_tge_source_bound_audit']['path']).resolve(),'certificate source TGE binding mismatch')
    tge=_audit_prerequisites(status,inputs['mesh'],inputs['tge_source'],inputs.get('open_exterior_source'),inputs.get('boundary_resolution_source'))
    _require(tge==_read(inputs['source_tge_audit']),'recomputed TGE audit differs from immutable publication')
    result=audit_terminal_forcing(inputs['mesh'],inputs['grid_contract'],inputs['obc_points'],inputs['forcing'],inputs['forcing_manifest'],inputs['obc_remap_manifest'])
    _require(result==cert['audit'],'terminal forcing certificate readback disagrees')
    previous=_read(inputs['previous_status'])
    permitted={'created_utc','forcing_status','submission_failure_taxonomy','submission_eligible','terminal_forcing_certificate'}
    _require({k:v for k,v in status.items() if k not in permitted}=={k:v for k,v in previous.items() if k not in permitted},'forcing join changed unrelated published status fields')
    from .quality_policy import classify_failure_codes,load_quality_policy
    original=classify_failure_codes(_read(inputs['mesh_quality']).get('all_quality_findings',[]),load_quality_policy())
    findings=set(original['submission_preconditions'])|set(previous['submission_failure_taxonomy'])
    resolved=sorted(findings & RESOLVABLE)
    effective=sorted(findings-RESOLVABLE)
    _require(cert['resolved_submission_preconditions']==resolved,'certificate attempts to resolve a different precondition')
    _require(not original['benchmark_baseline'] and previous['benchmark_grid_baseline_ready'] is True,'certificate cannot waive geometric baseline failures')
    _require(status['submission_failure_taxonomy']==effective and status['submission_eligible']==(not effective) and status['forcing_status']=='compatible','status disagrees with certificate-derived submission decision')
    _require(_read(path.with_name('delivery_status.json'))==status,'mutable status disagrees with immutable forcing status revision')
    return resolved

def certify_terminal_forcing(project,*,grid_contract,obc_points,forcing,forcing_manifest,tge_source,open_exterior_source=None,boundary_resolution_source=None,revision):
    """Create one evidence revision and update only mutable delivery status files."""
    from .grid_project import validate,FINAL_COMPANIONS,_atomic_json,_verified_copy,_append_command
    root=Path(project).resolve();_require(int(revision)>0,'forcing revision must be positive')
    folder=root/'08_audit/forcing_joins'/f'v{int(revision):03d}'
    _require(not folder.exists(),'immutable forcing certificate revision already exists')
    baseline=validate(root,require_benchmark_ready=True,require_submission_ready=True)
    _require(not (set(baseline['failure_taxonomy'])-{'project_not_submission_ready'}),'grid/source/TGE validation must pass before forcing join: '+','.join(baseline['failure_taxonomy']))
    status=baseline['status'];_require(status['benchmark_grid_baseline_ready'] is True,'forcing join requires benchmark-ready grid')
    _require(not (set(status['submission_failure_taxonomy'])-RESOLVABLE),'forcing join cannot waive other submission failures')
    contract=_regular(grid_contract);_require(contract.is_relative_to(root),'grid contract must be project-local')
    mesh=root/status['mesh']['path']
    tge_source=_regular(tge_source)
    boundary_inputs={key:_regular(value) for key,value in [('open_exterior_source',open_exterior_source),('boundary_resolution_source',boundary_resolution_source)] if value is not None}
    _require(all(path.is_relative_to(root) for path in boundary_inputs.values()),'boundary evidence manifests must remain project-local at their original locations')
    tge=_audit_prerequisites(status,mesh,tge_source,boundary_inputs.get('open_exterior_source'),boundary_inputs.get('boundary_resolution_source'))
    _require(tge==_read(root/status['fvcom_tge_source_bound_audit']['path']),'recomputed TGE differs from original audit')
    result=audit_terminal_forcing(mesh,contract,obc_points,forcing,forcing_manifest,root/'final'/FINAL_COMPANIONS['obc_remap_manifest'])
    # Every rejecting science/structural gate above runs before creating a revision.
    folder.mkdir(parents=True)
    inputs={'mesh':mesh,'grid_contract':contract,'project_manifest':root/'project_manifest.json',
            'source_tge_audit':root/status['fvcom_tge_source_bound_audit']['path']}
    inputs.update({key:root/'final'/name for key,name in FINAL_COMPANIONS.items()})
    inputs.update(boundary_inputs)
    for key,source,name in [('obc_points',obc_points,'obc_points.csv'),('forcing',forcing,'elevation.nc'),('forcing_manifest',forcing_manifest,'forcing_manifest.json'),('tge_source',tge_source,'tge.F')]:
        target=folder/name;_verified_copy(Path(source).resolve(),target);inputs[key]=target
    previous=folder/'previous_status.json';_atomic_json(previous,status);inputs['previous_status']=previous
    from .quality_policy import classify_failure_codes,load_quality_policy
    original=classify_failure_codes(_read(inputs['mesh_quality']).get('all_quality_findings',[]),load_quality_policy())
    findings=set(original['submission_preconditions'])|set(status['submission_failure_taxonomy'])
    certificate={'schema_version':RULE,'revision':int(revision),'created_utc':datetime.now(timezone.utc).isoformat(),
        'inputs':{key:{'path':path.relative_to(root).as_posix(),'sha256':_hash(path)} for key,path in inputs.items()},
        'audit':result,'resolved_submission_preconditions':sorted(findings&RESOLVABLE),
        'preservation':'Original mesh, quality, source-order remap, maps and source TGE companions remain unchanged.'}
    path=folder/'certificate.json';_atomic_json(path,certificate)
    updated=dict(status);updated.update(created_utc=certificate['created_utc'],forcing_status='compatible',
        submission_failure_taxonomy=sorted(findings-RESOLVABLE),submission_eligible=not bool(findings-RESOLVABLE),
        terminal_forcing_certificate={'path':path.relative_to(root).as_posix(),'sha256':_hash(path)})
    _atomic_json(folder/'delivery_status.json',updated)
    verify_terminal_forcing_join(root,updated)
    _atomic_json(root/'final/fvcom_grid_status.json',updated);_atomic_json(root/'project_status.json',updated)
    _append_command(root,'certify_terminal_forcing',{'revision':int(revision),'certificate_sha256':_hash(path),'forcing_sha256':result['forcing_sha256']})
    return validate(root,require_benchmark_ready=True,require_submission_ready=True)
