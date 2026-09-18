#!/usr/bin/env python3
"""Bounded, resumable NOAA CIOFS point subsets; no project or credential dependency."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import xml.etree.ElementTree as ET

import netCDF4
import numpy as np
import requests

import roms_points as roms

SKILL=Path(__file__).resolve().parents[1]
CATALOG='https://opendap.co-ops.nos.noaa.gov/thredds/catalog/NOAA/CIOFS/MODELS'
DODS='https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/'
SCHEMA='ciofs_point_request_v1'
TLS=threading.local()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.part')
    temp.write_bytes(data);temp.replace(path)


def write_json(path,value):
    atomic_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def utc(text):
    dt=datetime.fromisoformat(str(text).replace('Z','+00:00'))
    if dt.tzinfo is None:raise ValueError('Time must include UTC offset or Z')
    return dt.astimezone(timezone.utc)


def iso(dt):return dt.astimezone(timezone.utc).isoformat().replace('+00:00','Z')


def run_root(path):
    root=Path(path).resolve()
    if root==SKILL or SKILL in root.parents:raise ValueError('Run artifacts must be outside the skill package')
    root.mkdir(parents=True,exist_ok=True)
    return root


def contained(root,relative):
    path=(Path(root)/relative).resolve()
    if Path(root).resolve() not in path.parents:raise ValueError('Artifact path escapes run directory')
    return path


def normalize_request(mapping):
    if not isinstance(mapping,dict):raise ValueError('Request must be a JSON object')
    allowed={'schema_version','start_utc','end_utc_exclusive','guidance','run_cycle_utc','variables','vertical_views','points',
             'minimum_depth_m','missing_policy','min_completeness','max_gap_hours','max_workers','timeout_seconds','retries'}
    if set(mapping)-allowed:raise ValueError(f'Unknown request fields: {sorted(set(mapping)-allowed)}')
    r=json.loads(json.dumps(mapping))
    if r.get('schema_version')!=SCHEMA:raise ValueError(f'Expected {SCHEMA}')
    start,end=utc(r['start_utc']),utc(r['end_utc_exclusive'])
    if end<=start or any(t.minute or t.second or t.microsecond for t in [start,end]):raise ValueError('Require increasing, hour-aligned [start,end) UTC bounds')
    r.update(start_utc=iso(start),end_utc_exclusive=iso(end))
    if r.get('guidance') not in ['nowcast','forecast']:raise ValueError('guidance must be nowcast or forecast')
    if r['guidance']=='forecast':
        cycle=utc(r['run_cycle_utc'])
        if cycle.hour not in [0,6,12,18] or cycle.minute or cycle.second or cycle.microsecond:raise ValueError('Forecast requires an exact 00/06/12/18 UTC run cycle')
        if start<=cycle or end>cycle+timedelta(hours=49):raise ValueError('Forecast valid times must be cycle+1 through cycle+48 hours')
        r['run_cycle_utc']=iso(cycle)
    elif 'run_cycle_utc' in r:raise ValueError('Do not specify a forecast cycle for a nowcast')
    aliases={'currents':['u','v'],'salinity':['salt'],'temperature':['temp'],'water_level':['zeta']}
    variables=r.get('variables',['u','v'])
    if not isinstance(variables,list) or not variables or not all(isinstance(v,str) for v in variables):raise ValueError('variables must be a nonempty string list')
    variables=[v for name in variables for v in aliases.get(name,[name])]
    if set(variables)-{'u','v','salt','temp','zeta'} or ('u' in variables)!=('v' in variables):raise ValueError('Supported: paired u/v, salt, temp, zeta; U/V must be requested together')
    r['variables']=list(dict.fromkeys(variables))
    views=r.get('vertical_views',['surface'])
    if not isinstance(views,list) or not views:raise ValueError('vertical_views must be a nonempty list')
    views=['surface' if v=='near_surface' else v for v in views]
    if any(not isinstance(v,str) or (v not in ['surface','bottom'] and not re.fullmatch(r'index:\d+',v)) for v in views):
        raise ValueError('Supported vertical views: surface, bottom, index:N (depth_average is not implemented)')
    r['vertical_views']=list(dict.fromkeys(views))
    for name,default,lo,hi in [('minimum_depth_m',0.,0.,11000.),('min_completeness',.95,0.,1.)]:
        val=r.get(name,default)
        if isinstance(val,bool) or not isinstance(val,(int,float)) or not math.isfinite(val) or not lo<=val<=hi:raise ValueError(f'Invalid {name}')
        r[name]=float(val)
    r.setdefault('max_gap_hours',None)
    if r['max_gap_hours'] is not None and (isinstance(r['max_gap_hours'],bool) or not isinstance(r['max_gap_hours'],(int,float)) or not math.isfinite(r['max_gap_hours']) or r['max_gap_hours']<1):raise ValueError('max_gap_hours must be null or >=1')
    for name,default,lo,hi in [('max_workers',4,1,8),('timeout_seconds',60,1,300),('retries',4,1,8)]:
        value=r.get(name,default)
        if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:raise ValueError(f'Invalid {name}')
        r[name]=value
    r.setdefault('missing_policy','error')
    if r['missing_policy'] not in ['error','skip']:raise ValueError('missing_policy must be error or skip')
    if not isinstance(r.get('points'),list) or not r['points']:raise ValueError('points must be a nonempty list')
    seen=set()
    for p in r['points']:
        if not isinstance(p,dict) or set(p)-{'id','latitude','longitude','sampling','max_distance_m','required'}:raise ValueError('Invalid point fields')
        pid=p.get('id')
        if not isinstance(pid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',pid) or pid in seen:raise ValueError('Point IDs must be unique safe strings')
        seen.add(pid)
        for name,lo,hi in [('latitude',-90,90),('longitude',-180,180)]:
            val=p.get(name)
            if isinstance(val,bool) or not isinstance(val,(int,float)) or not math.isfinite(val) or not lo<=val<=hi:raise ValueError(f'Invalid point {name}')
            p[name]=float(val)
        p.setdefault('sampling','nearest_wet');p.setdefault('max_distance_m',5000.);p.setdefault('required',True)
        if p['sampling'] not in ['nearest_wet','bilinear'] or not isinstance(p['required'],bool):raise ValueError('Invalid point sampling or required flag')
        v=p['max_distance_m']
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise ValueError('Invalid point max_distance_m')
        p['max_distance_m']=float(v)
    return r


def request_file(path):return normalize_request(read_json(path))


def http(url,r,query=None,allow_missing=False):
    if not hasattr(TLS,'session'):
        TLS.session=requests.Session();TLS.session.headers['User-Agent']='CIOFS-point-fetcher/1.0 (scientific bounded subsets)'
    for attempt in range(r['retries']):
        try:
            response=TLS.session.get(url,params=query,timeout=r['timeout_seconds'],allow_redirects=False)
            if allow_missing and response.status_code==404:return response
            if response.status_code!=200:raise requests.HTTPError(f'HTTP {response.status_code}: {response.url}',response=response)
            return response
        except requests.RequestException as exc:
            status=exc.response.status_code if getattr(exc,'response',None) is not None else None
            if attempt+1==r['retries'] or status in [400,401,403,404]:raise
            time.sleep(min(2**attempt,8))


def parse_source(path):
    m=re.fullmatch(r'NOAA/CIOFS/MODELS/(\d{4})/(\d{2})/(\d{2})/ciofs\.t(00|06|12|18)z\.(\d{8})\.fields\.([nf])(\d{3})\.nc',path)
    if not m:return None
    year,month,day,hour,stamp,kind,lead=m.groups();lead=int(lead)
    if stamp!=year+month+day or (kind=='n' and not 1<=lead<=6) or (kind=='f' and not 1<=lead<=48):return None
    cycle=datetime.strptime(stamp+hour,'%Y%m%d%H').replace(tzinfo=timezone.utc)
    valid=cycle+timedelta(hours=lead-6 if kind=='n' else lead)
    return {'path':path,'url':DODS+path,'run_cycle_utc':iso(cycle),'nominal_valid_utc':iso(valid),
            'guidance':'nowcast' if kind=='n' else 'forecast','lead_index':lead}


def inventory(r):
    start,end=utc(r['start_utc']),utc(r['end_utc_exclusive'])
    if r['guidance']=='forecast':days=[utc(r['run_cycle_utc']).date()]
    else:
        # A 19--23 UTC nowcast hour belongs to the following day's 00 UTC cycle.
        last_cycle_day=(end+timedelta(hours=4)).date()
        days=[start.date()+timedelta(days=k) for k in range((last_cycle_day-start.date()).days+1)]
    def day_files(day):
        response=http(f'{CATALOG}/{day:%Y/%m/%d}/catalog.xml',r,allow_missing=True)
        if response.status_code==404:return []
        return sorted({e.attrib['urlPath'] for e in ET.fromstring(response.content).iter() if 'urlPath' in e.attrib})
    items=[]
    with futures.ThreadPoolExecutor(max_workers=min(4,r['max_workers'])) as pool:
        for paths in pool.map(day_files,days):
            for path in paths:
                item=parse_source(path)
                if item and item['guidance']==r['guidance'] and start<=utc(item['nominal_valid_utc'])<end:
                    if r['guidance']=='nowcast' or item['run_cycle_utc']==r['run_cycle_utc']:items.append(item)
    items.sort(key=lambda v:v['nominal_valid_utc'])
    if len({i['nominal_valid_utc'] for i in items})!=len(items):raise ValueError('Conflicting source candidates for the same valid hour')
    expected=[iso(start+timedelta(hours=k)) for k in range(int((end-start).total_seconds()/3600))]
    available={i['nominal_valid_utc'] for i in items}
    return {'schema_version':'ciofs_inventory_v1','request':r,'request_sha256':digest(r),'objects':items,
            'expected_hours':len(expected),'available_hours':len(items),'missing_hours':[t for t in expected if t not in available],
            'source':'NOAA CO-OPS THREDDS rolling native-field archive','dates_shifted':False}


def inspect_source(url,r):
    if not url.startswith(DODS) or not parse_source(url[len(DODS):]):raise ValueError('Not a canonical CIOFS source URL')
    dds=http(url+'.dds',r).content;das=http(url+'.das',r).content
    return {'url':url,'dds':dds.decode(),'das':das.decode(),'metadata':roms.parse_metadata(dds.decode(),das.decode(),r['variables']),
            'dds_sha256':hashlib.sha256(dds).hexdigest(),'das_sha256':hashlib.sha256(das).hexdigest()}


def bind_run(root,r):
    path=root/'request.json'
    if path.exists() and digest(read_json(path))!=digest(r):raise ValueError('Run directory already belongs to a different request; use a new run directory')
    write_json(path,r)


def plan_request(r,root):
    bind_run(root,r)
    planpath=root/'download_estimate.json'
    if planpath.exists():
        return load_plan(planpath,root)
    inv=inventory(r);write_json(root/'inventory.json',inv)
    if not inv['objects']:raise ValueError('No requested hours are available; dates were not shifted')
    if inv['missing_hours'] and r['missing_policy']=='error':raise ValueError(f'{len(inv["missing_hours"])} unavailable hours; inspect inventory.json or explicitly choose missing_policy=skip')
    probe=inspect_source(inv['objects'][-1]['url'],r)
    write_json(root/'source_probe.json',probe)
    grid_query=','.join(roms.STATIC)
    grid_dds=http(probe['url']+'.dds',r,grid_query).text
    grid_bytes=roms.payload_size(grid_dds)+len(grid_dds)+8
    free=shutil.disk_usage(root).free
    if free<grid_bytes*4:raise ValueError('Insufficient local space for the one-time planning geometry probe')
    print(f'Planning geometry probe: {grid_bytes/1e6:.1f} MB decoded (wire compression unknown)',file=sys.stderr,flush=True)
    payload=http(probe['url']+'.dods',r,grid_query).content
    grid=roms.decode_dods(payload);roms.validate_grid(grid)
    atomic_bytes(root/'grid.dods',payload)
    points=roms.select_points(r,grid);indices=roms.view_indices(r['vertical_views'],grid)
    write_json(root/'points.json',points)
    subsets=[];per_hour=0
    for point in points:
        query,bounds=roms.point_query(point,r,indices)
        dds=http(probe['url']+'.dds',r,query).text
        decoded=roms.payload_size(dds)+len(dds.encode())+8
        subsets.append({'point_id':point['id'],'query':query,'bounds':bounds,'estimated_decoded_response_bytes':decoded})
        per_hour+=decoded
    meta_per_hour=len(probe['dds'].encode())+len(probe['das'].encode())
    transfer_estimate=(per_hour+meta_per_hour)*len(inv['objects'])
    expected_rows=inv['expected_hours']*len(points)*len(indices)
    storage_budget=4*(transfer_estimate+grid_bytes+expected_rows*1000)
    estimate={'schema_version':'ciofs_download_plan_v1','request':r,'request_sha256':digest(r),'objects':inv['objects'],
              'missing_hours':inv['missing_hours'],'expected_hours':inv['expected_hours'],'points':points,'view_indices':indices,'subsets':subsets,
              'source_probe_sha256':file_hash(root/'source_probe.json'),'grid_sha256':file_hash(root/'grid.dods'),
              'points_sha256':file_hash(root/'points.json'),'inventory_sha256':file_hash(root/'inventory.json'),
              'estimate':{'geometry_probe_decoded_bytes':len(payload),'remaining_transfer_decoded_bytes_estimate':transfer_estimate,
                          'metadata_included':True,'http_requests':len(inv['objects'])*(2+len(points)),
                          'wire_bytes':'unknown: server compression and headers vary','storage_budget_bytes':storage_budget,
                          'free_space_bytes':free,'routing':'local' if free>storage_budget else 'insufficient_local_space'},
              'created_utc':iso(datetime.now(timezone.utc))}
    estimate['plan_sha256']=digest(estimate)
    write_json(planpath,estimate)
    write_csv(root/'points.csv',[{k:v for k,v in p.items() if k!='support'} for p in points])
    write_csv(root/'interpolation_support.csv',[{'point_id':p['id'],**c} for p in points for c in p['support']])
    return estimate


def load_plan(path,root):
    plan=read_json(path);claimed=plan.get('plan_sha256');body={k:v for k,v in plan.items() if k!='plan_sha256'}
    if plan.get('schema_version')!='ciofs_download_plan_v1' or claimed!=digest(body):raise ValueError('Plan digest/schema mismatch')
    r=normalize_request(plan['request'])
    if plan['request_sha256']!=digest(r) or digest(read_json(root/'request.json'))!=digest(r):raise ValueError('Request binding mismatch')
    for name,key in [('source_probe.json','source_probe_sha256'),('grid.dods','grid_sha256'),('points.json','points_sha256'),('inventory.json','inventory_sha256')]:
        if file_hash(root/name)!=plan[key]:raise ValueError(f'Changed planning artifact: {name}')
    for item in plan['objects']:
        parsed=parse_source(item['path'])
        if parsed!=item:raise ValueError('Noncanonical/modified source identity')
        if parsed['guidance']!=r['guidance'] or not utc(r['start_utc'])<=utc(parsed['nominal_valid_utc'])<utc(r['end_utc_exclusive']):raise ValueError('Source outside request scope')
        if r['guidance']=='forecast' and parsed['run_cycle_utc']!=r['run_cycle_utc']:raise ValueError('Wrong forecast cycle')
    if len({x['path'] for x in plan['objects']})!=len(plan['objects']):raise ValueError('Duplicate source identity')
    grid=roms.decode_dods((root/'grid.dods').read_bytes());roms.validate_grid(grid)
    if plan['view_indices']!=roms.view_indices(r['vertical_views'],grid):raise ValueError('Modified vertical selection')
    if plan['points']!=read_json(root/'points.json'):raise ValueError('Point binding mismatch')
    roms.validate_points(plan['points'],r,grid)
    if len(plan['subsets'])!=len(plan['points']):raise ValueError('Missing/duplicate subset definitions')
    for point,subset in zip(plan['points'],plan['subsets']):
        query,bounds=roms.point_query(point,r,plan['view_indices'])
        if subset['point_id']!=point['id'] or subset['query']!=query or subset['bounds']!=bounds:raise ValueError('Changed subset query')
    return plan


def cache_path(root,item,point=None):
    filename=Path(item['path']).name
    return root/'cache'/('metadata' if point is None else 'subsets')/(filename+'.json' if point is None else point['id']+'__'+filename+'.dods')


def verified_subset(root,item,point,subset,plan):
    path=cache_path(root,item,point);side=path.with_suffix(path.suffix+'.json')
    if not path.exists() and not side.exists():return None
    if not path.exists() or not side.exists():return None  # interrupted atomic pair: re-fetch
    record=read_json(side)
    for key,expected in [('plan_sha256',plan['plan_sha256']),('source_url',item['url']),('query',subset['query'])]:
        if record.get(key)!=expected:raise ValueError(f'Cache binding mismatch: {path.name}')
    if record.get('sha256')!=file_hash(path) or record.get('bytes')!=path.stat().st_size:raise ValueError(f'Corrupt cache: {path.name}')
    return record


def verified_meta(root,item,plan):
    path=cache_path(root,item)
    if not path.exists():return None
    record=read_json(path);body={k:v for k,v in record.items() if k!='record_sha256'}
    if record.get('record_sha256')!=digest(body) or record.get('plan_sha256')!=plan['plan_sha256'] or record.get('url')!=item['url']:
        raise ValueError('Metadata cache integrity mismatch')
    recomputed=roms.parse_metadata(record['dds'],record['das'],plan['request']['variables'])
    if recomputed!=record['metadata']:raise ValueError('Modified parsed metadata')
    return record


def fetch_one(root,item,plan,grid,baseline):
    r=plan['request'];meta=verified_meta(root,item,plan);cached=0;downloaded=0
    if meta is None:
        meta=inspect_source(item['url'],r)
        if roms.signature(meta['metadata'])!=roms.signature(baseline):raise ValueError('Source metadata or dimensions drift')
        meta['plan_sha256']=plan['plan_sha256'];meta['record_sha256']=digest(meta)
        write_json(cache_path(root,item),meta)
    if roms.signature(meta['metadata'])!=roms.signature(baseline):raise ValueError('Cached metadata drift')
    for point,subset in zip(plan['points'],plan['subsets']):
        if verified_subset(root,item,point,subset,plan) is not None:
            cached+=1;continue
        response=http(item['url']+'.dods',r,subset['query'])
        array=roms.decode_dods(response.content);roms.validate_subset(array,subset['bounds'],grid)
        path=cache_path(root,item,point);atomic_bytes(path,response.content)
        side={'schema_version':'ciofs_subset_cache_v1','plan_sha256':plan['plan_sha256'],'source_url':item['url'],
              'query':subset['query'],'sha256':file_hash(path),'bytes':len(response.content),
              'source_metadata_record_sha256':meta['record_sha256'],'retrieved_utc':iso(datetime.now(timezone.utc)),
              'etag':response.headers.get('ETag'),'last_modified':response.headers.get('Last-Modified')}
        write_json(path.with_suffix(path.suffix+'.json'),side);downloaded+=1
    return {'source_url':item['url'],'status':'ok','cached_subsets':cached,'downloaded_subsets':downloaded}


def fetch_plan(planpath,root):
    plan=load_plan(planpath,root)
    if plan['estimate']['routing']!='local' or shutil.disk_usage(root).free<=plan['estimate']['storage_budget_bytes']:
        raise ValueError('Insufficient local space for planned run; choose a larger run volume and replan')
    grid=roms.decode_dods((root/'grid.dods').read_bytes());baseline=read_json(root/'source_probe.json')['metadata']
    outcomes=[];t0=time.monotonic();last=0
    with futures.ThreadPoolExecutor(max_workers=plan['request']['max_workers']) as pool:
        pending={pool.submit(fetch_one,root,item,plan,grid,baseline):item for item in plan['objects']}
        for future in futures.as_completed(pending):
            try:outcomes.append(future.result())
            except Exception as exc:outcomes.append({'source_url':pending[future]['url'],'status':'failed','error':str(exc)})
            elapsed=time.monotonic()-t0
            if elapsed-last>=30:
                print(f'{len(outcomes)}/{len(pending)} source hours processed',file=sys.stderr,flush=True);last=elapsed
    manifest={'schema_version':'ciofs_fetch_manifest_v1','plan_sha256':plan['plan_sha256'],'request_sha256':plan['request_sha256'],
              'outcomes':sorted(outcomes,key=lambda v:v['source_url']),'elapsed_seconds':time.monotonic()-t0}
    manifest['manifest_sha256']=digest(manifest);write_json(root/'fetch_manifest.json',manifest)
    if any(o['status']!='ok' for o in outcomes):raise RuntimeError('Some source hours failed; inspect fetch_manifest.json, then rerun fetch to resume')
    return manifest


def decode_time(arrays,meta):
    attrs=meta['ocean_time']['attributes'];calendar=attrs.get('calendar','standard')
    if calendar=='gregorian_proleptic':calendar='proleptic_gregorian'
    if calendar not in ['standard','gregorian','proleptic_gregorian']:raise ValueError('Non-UTC-compatible CF calendar')
    decoded=netCDF4.num2date(float(arrays['ocean_time'][0]),attrs['units'],calendar=calendar,only_use_cftime_datetimes=False)
    source=datetime(decoded.year,decoded.month,decoded.day,decoded.hour,decoded.minute,decoded.second,decoded.microsecond,tzinfo=timezone.utc)
    normalized=datetime.fromtimestamp(round(source.timestamp()/3600)*3600,tz=timezone.utc)
    delta=(normalized-source).total_seconds()
    if abs(delta)>60:raise ValueError('Decoded CF time is not within 60 seconds of hourly cadence')
    return source,normalized,delta


def verified_manifest(root,plan):
    manifest=read_json(root/'fetch_manifest.json');body={k:v for k,v in manifest.items() if k!='manifest_sha256'}
    if manifest.get('manifest_sha256')!=digest(body) or manifest.get('plan_sha256')!=plan['plan_sha256'] or manifest.get('request_sha256')!=plan['request_sha256']:
        raise ValueError('Fetch manifest binding mismatch')
    expected={i['url'] for i in plan['objects']}
    actual={i['source_url'] for i in manifest['outcomes']}
    if expected!=actual or len(actual)!=len(manifest['outcomes']) or any(i['status']!='ok' for i in manifest['outcomes']):raise ValueError('Incomplete fetch manifest; resume retrieval')
    return manifest


def collected_records(plan,root):
    grid=roms.decode_dods((root/'grid.dods').read_bytes());baseline=read_json(root/'source_probe.json')['metadata']
    records=[];seen=set();audit=[]
    for item in plan['objects']:
        meta=verified_meta(root,item,plan)
        if meta is None:raise ValueError('Missing source metadata cache')
        if roms.signature(meta['metadata'])!=roms.signature(baseline):raise ValueError('Metadata signature drift')
        for point,subset in zip(plan['points'],plan['subsets']):
            side=verified_subset(root,item,point,subset,plan)
            if side is None or side.get('source_metadata_record_sha256')!=meta['record_sha256']:raise ValueError('Missing/unbound subset cache')
            a=roms.decode_dods(cache_path(root,item,point).read_bytes());roms.validate_subset(a,subset['bounds'],grid)
            source,normalized,delta=decode_time(a,meta['metadata'])
            mismatch=iso(normalized)!=item['nominal_valid_utc']
            if not utc(plan['request']['start_utc'])<=normalized<utc(plan['request']['end_utc_exclusive']):
                audit.append({'source_url':item['url'],'point_id':point['id'],'reason':'decoded_time_outside_request','decoded_time':iso(source)})
                continue
            if delta or mismatch:audit.append({'source_url':item['url'],'point_id':point['id'],'source_time':iso(source),'normalized_time':iso(normalized),'adjustment_seconds':delta,'filename_mismatch':mismatch})
            for record in roms.extract_point(a,point,plan['request'],plan['view_indices'],subset['bounds'],meta['metadata']):
                key=(iso(normalized),point['id'],record['view'])
                if key in seen:raise ValueError('Duplicate decoded point/time/view')
                seen.add(key)
                records.append({'time':iso(normalized),'source_time':iso(source),'time_adjustment_seconds':delta,'source_url':item['url'],**record})
    return records,audit


def write_csv(path,rows):
    rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError('No rows to write')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def coverage(plan,records):
    r=plan['request'];start=utc(r['start_utc']);n=plan['expected_hours']
    lookup={(rec['point_id'],rec['view'],rec['time']):rec for rec in records};qa=[]
    names=(['currents'] if 'u' in r['variables'] else [])+[v for v in r['variables'] if v not in ['u','v']]
    for point in plan['points']:
        for view in r['vertical_views']:
            for name in names:
                good=[];reasons={}
                for k in range(n):
                    row=lookup.get((point['id'],view,iso(start+timedelta(hours=k))))
                    reason=row['quality'][name] if row else 'missing_source_hour'
                    good.append(reason=='ok');reasons[reason]=reasons.get(reason,0)+1
                indices=np.flatnonzero(good)
                max_missing=run=0
                for ok in good:
                    run=0 if ok else run+1;max_missing=max(max_missing,run)
                gap=float(max(np.diff(indices),default=0)) if len(indices) else float(n)
                edge=max(int(indices[0]),int(n-1-indices[-1])) if len(indices) else n
                gap=max(gap,float(edge));complete=float(np.mean(good))
                passed=bool(len(indices)>0 and complete>=r['min_completeness'] and (r['max_gap_hours'] is None or gap<=r['max_gap_hours']))
                qa.append({'point_id':point['id'],'view':view,'variable':name,'required':point['required'],'expected':n,'valid':int(sum(good)),
                           'completeness':complete,'maximum_gap_hours':gap,'longest_missing_run_hours':max_missing,
                           'valid_span_hours':float(indices[-1]-indices[0]) if len(indices)>1 else 0.,'accepted':passed,'reasons':reasons})
    return qa


def write_netcdf(path,plan,records,manifest_hash):
    r=plan['request'];points=plan['points'];views=r['vertical_views'];n=plan['expected_hours']
    source_meta=read_json(Path(path).parent/'source_probe.json')['metadata']
    with netCDF4.Dataset(path,'w',format='NETCDF4') as ds:
        ds.schema_version='ciofs_point_series_v1';ds.Conventions='CF-1.8';ds.title='NOAA CIOFS model point time series'
        ds.plan_sha256=plan['plan_sha256'];ds.fetch_manifest_sha256=manifest_hash;ds.vector_reference='earth_relative_true_east_north'
        ds.velocity_processing='Strict wet-face destagger to rho; rotate XI-to-east; interpolate east/north at bilinear points'
        ds.request_json=json.dumps(r,sort_keys=True);ds.point_support_json=json.dumps(points,sort_keys=True)
        for name,size in [('time',n),('point',len(points)),('view',len(views))]:ds.createDimension(name,size)
        t=ds.createVariable('time','f8',('time',));t.units='seconds since 1970-01-01 00:00:00 UTC';t.calendar='proleptic_gregorian'
        t.standard_name='time';t[:]=utc(r['start_utc']).timestamp()+np.arange(n)*3600
        for name,values,dim in [('point_id',[p['id'] for p in points],'point'),('view_name',views,'view')]:
            var=ds.createVariable(name,str,(dim,));var[:]=np.array(values,dtype=object)
        for name,key,unit in [('longitude','longitude','degrees_east'),('latitude','latitude','degrees_north'),('depth','depth_m','m')]:
            var=ds.createVariable(name,'f8',('point',));var.units=unit;var[:]=[p[key] for p in points]
        ds['depth'].positive='down';ds['depth'].long_name='Static model bathymetric depth at sampling point'
        sigma=roms.decode_dods((Path(path).parent/'grid.dods').read_bytes())['s_rho']
        vi=ds.createVariable('sigma_index','i4',('view',));vi[:]=[plan['view_indices'][v] for v in views]
        sv=ds.createVariable('sigma_coordinate','f8',('view',));sv.units='1';sv[:]=[sigma[plan['view_indices'][v]] for v in views]
        names=(['east_m_s','north_m_s','speed_m_s'] if 'u' in r['variables'] else [])+[v for v in r['variables'] if v not in ['u','v']]
        arrays={name:np.full((n,len(points),len(views)),np.nan) for name in names}
        flags={name:np.zeros((n,len(points),len(views)),dtype='i1') for name in names}
        pidx={p['id']:k for k,p in enumerate(points)};vidx={v:k for k,v in enumerate(views)}
        for rec in records:
            ti=int((utc(rec['time'])-utc(r['start_utc'])).total_seconds()/3600);pi=pidx[rec['point_id']];vi=vidx[rec['view']]
            for name in names:
                if rec[name] is not None:arrays[name][ti,pi,vi]=rec[name];flags[name][ti,pi,vi]=1
        for name in names:
            v=ds.createVariable(name,'f8',('time','point','view'),zlib=True,fill_value=np.nan)
            v[:]=arrays[name];v.coordinates='longitude latitude';v.ancillary_variables=name+'_valid'
            if name.endswith('_m_s'):v.units='m s-1'
            else:
                attrs=source_meta[name]['attributes'];v.source_units=attrs.get('units','not declared');v.units=attrs.get('units','')
            if name=='east_m_s':v.standard_name='eastward_sea_water_velocity'
            if name=='north_m_s':v.standard_name='northward_sea_water_velocity'
            if name=='zeta':v.comment='Water level is independent of view; repeated across views for a uniform point-series interface.'
            flag=ds.createVariable(name+'_valid','i1',('time','point','view'),zlib=True);flag[:]=flags[name]
            flag.flag_values=np.array([0,1],dtype='i1');flag.flag_meanings='missing_or_masked valid'


def extract_run(root):
    plan=load_plan(root/'download_estimate.json',root);manifest=verified_manifest(root,plan)
    records,audit=collected_records(plan,root)
    if not records:raise ValueError('No in-window decoded samples')
    records.sort(key=lambda x:(x['time'],x['point_id'],x['view']))
    write_json(root/'point_records.json',records);write_csv(root/'point_series.csv',records)
    write_netcdf(root/'point_series.nc',plan,records,manifest['manifest_sha256'])
    qa=coverage(plan,records);write_csv(root/'coverage.csv',qa);write_json(root/'time_audit.json',audit)
    report={'schema_version':'ciofs_extraction_v1','plan_sha256':plan['plan_sha256'],'fetch_manifest_sha256':manifest['manifest_sha256'],
            'coverage':qa,'time_audit':audit,'records':len(records),
            'outputs':{name:file_hash(root/name) for name in ['point_records.json','point_series.csv','point_series.nc','coverage.csv','time_audit.json']}}
    write_json(root/'extraction_manifest.json',report)
    return report


def health(root):
    plan=load_plan(root/'download_estimate.json',root);manifest=verified_manifest(root,plan);report=read_json(root/'extraction_manifest.json')
    if report.get('plan_sha256')!=plan['plan_sha256'] or report.get('fetch_manifest_sha256')!=manifest['manifest_sha256']:raise ValueError('Extraction provenance mismatch')
    if set(report['outputs'])!={'point_records.json','point_series.csv','point_series.nc','coverage.csv','time_audit.json'}:
        raise ValueError('Missing/unexpected extraction output entries')
    for name,expected in report['outputs'].items():
        if file_hash(contained(root,name))!=expected:raise ValueError(f'Output integrity mismatch: {name}')
    records,audit=collected_records(plan,root)
    records.sort(key=lambda x:(x['time'],x['point_id'],x['view']))
    if records!=read_json(root/'point_records.json'):raise ValueError('Re-extraction disagrees with saved point records')
    qa=coverage(plan,records)
    if qa!=report['coverage'] or audit!=report['time_audit']:raise ValueError('Coverage/time audit mismatch')
    r=plan['request'];points=plan['points'];views=r['vertical_views']
    if report['records']!=len(records):raise ValueError('Record count mismatch')
    with netCDF4.Dataset(root/'point_series.nc') as ds:
        if ds.plan_sha256!=plan['plan_sha256'] or ds.fetch_manifest_sha256!=manifest['manifest_sha256']:raise ValueError('NetCDF provenance mismatch')
        if {n:len(d) for n,d in ds.dimensions.items()}!={'time':plan['expected_hours'],'point':len(points),'view':len(views)}:
            raise ValueError('NetCDF dimensions mismatch')
        if json.loads(ds.request_json)!=r or json.loads(ds.point_support_json)!=points:raise ValueError('NetCDF request/support mismatch')
        if list(ds['point_id'][:])!=[p['id'] for p in points] or list(ds['view_name'][:])!=views:raise ValueError('NetCDF point/view order mismatch')
        np.testing.assert_array_equal(ds['time'][:],utc(r['start_utc']).timestamp()+np.arange(plan['expected_hours'])*3600)
        for name,key in [('longitude','longitude'),('latitude','latitude'),('depth','depth_m')]:
            np.testing.assert_array_equal(ds[name][:],[p[key] for p in points])
        np.testing.assert_array_equal(ds['sigma_index'][:],[plan['view_indices'][v] for v in views])
        sigma=roms.decode_dods((root/'grid.dods').read_bytes())['s_rho']
        np.testing.assert_array_equal(ds['sigma_coordinate'][:],[sigma[plan['view_indices'][v]] for v in views])
        names=(['east_m_s','north_m_s','speed_m_s'] if 'u' in r['variables'] else [])+[v for v in r['variables'] if v not in ['u','v']]
        expected={name:np.full((plan['expected_hours'],len(points),len(views)),np.nan) for name in names}
        for rec in records:
            ti=int((utc(rec['time'])-utc(r['start_utc'])).total_seconds()/3600)
            pi=[p['id'] for p in points].index(rec['point_id']);vi=views.index(rec['view'])
            for name in names:
                if rec[name] is not None:expected[name][ti,pi,vi]=rec[name]
        for name in names:
            val=np.ma.filled(ds[name][:],np.nan);flags=ds[name+'_valid'][:]
            np.testing.assert_allclose(val,expected[name],rtol=0,atol=0,equal_nan=True)
            if not np.isin(flags,[0,1]).all() or not np.array_equal(np.isfinite(val),flags==1):raise ValueError('NetCDF validity mismatch')
        if 'speed_m_s' in ds.variables:
            np.testing.assert_allclose(ds['speed_m_s'][:],np.hypot(ds['east_m_s'][:],ds['north_m_s'][:]),atol=1e-12)
    ok=all(row['accepted'] for row in qa if row['required']) and any(row['accepted'] for row in qa)
    result={'schema_version':'ciofs_health_v1','status':'ok' if ok else 'insufficient_coverage','plan_sha256':plan['plan_sha256'],
            'fetch_manifest_sha256':manifest['manifest_sha256'],'coverage':qa,'time_audit':audit,
            'validated_raw_subsets':len(plan['objects'])*len(plan['points']),
            'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','netCDF4','pyproj','requests']}}
    write_json(root/'health_check.json',result)
    if not ok:raise RuntimeError('Required point/view/variable failed coverage; inspect health_check.json')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['inventory','plan']:
        p=sub.add_parser(name);p.add_argument('--request',required=True);p.add_argument('--run-dir',required=True)
    p=sub.add_parser('fetch');p.add_argument('--plan',required=True);p.add_argument('--run-dir',required=True)
    for name in ['extract','health']:
        p=sub.add_parser(name);p.add_argument('--run-dir',required=True)
    p=sub.add_parser('inspect');p.add_argument('--url',required=True);p.add_argument('--request',required=True);p.add_argument('--output',required=True)
    a=parser.parse_args(argv)
    try:
        if a.command=='inspect':
            output=Path(a.output).resolve();run_root(output.parent)
            result=inspect_source(a.url,request_file(a.request));write_json(output,result)
        else:
            root=run_root(a.run_dir)
            if a.command=='inventory':
                r=request_file(a.request);bind_run(root,r);result=inventory(r);write_json(root/'inventory.json',result)
            elif a.command=='plan':result=plan_request(request_file(a.request),root)
            elif a.command=='fetch':result=fetch_plan(a.plan,root)
            elif a.command=='extract':result=extract_run(root)
            else:result=health(root)
        summary={k:v for k,v in result.items() if k not in ['objects','points','subsets','coverage','outcomes','metadata','dds','das','request','view_indices']}
        print(json.dumps(summary,indent=2));return 0
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
