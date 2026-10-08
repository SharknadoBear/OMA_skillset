"""Bounded NOAA multi_1 acquisition. Portable: no user-specific paths."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import tarfile
import time

import numpy as np
from netCDF4 import Dataset
import rasterio
from rasterio.windows import Window
import requests

BASE = 'https://polar.ncep.noaa.gov/waves/hindcasts/multi_1/'
GRIDS = {'glo_30m','ao_30m','ak_10m','at_10m','wc_10m','ep_10m','ak_4m','at_4m','wc_4m'}
FIELDS = {'hs','tp','dp','wind','phs','ptp','pdir'}

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            md5.update(chunk); sha.update(chunk)
    return {'md5':md5.hexdigest(), 'sha256':sha.hexdigest()}

def epoch(value):
    d = datetime.fromisoformat(value.replace('Z','+00:00'))
    if d.tzinfo is None or d.utcoffset().total_seconds() != 0:
        raise ValueError('Dates must explicitly use UTC')
    return int(d.timestamp())

def request(path):
    r = read(path)
    required = {'schema','source_family','product','grid','stations','start','end','bbox','fields','missing_policy','raw_retention'}
    if set(r) != required or r['schema']!='ww3_request_v1' or r['source_family']!='multi_1':
        raise ValueError('Invalid ww3_request_v1 keys/source')
    if epoch(r['start'])>=epoch(r['end']): raise ValueError('Empty time interval')
    if len(set(r['stations']))!=len(r['stations']) or len(set(r['fields']))!=len(r['fields']):raise ValueError('Duplicate selections')
    if r['missing_policy'] not in ('error','skip') or r['raw_retention'] not in ('keep','delete_after_health'):
        raise ValueError('Invalid missing/retention policy')
    if r['product']=='fields':
        b = r['bbox']
        if r['grid'] not in GRIDS or not r['fields'] or not set(r['fields'])<=FIELDS or r['stations']:
            raise ValueError('Invalid fields request')
        if len(b)!=4 or not (-180<=b[0]<b[2]<=180 and -90<=b[1]<b[3]<=90):
            raise ValueError('Invalid non-dateline bbox')
    elif r['product']=='point_spectra':
        if r['grid'] is not None or r['bbox'] is not None or r['fields'] or not r['stations']:
            raise ValueError('Invalid point request')
        if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,16}', s) for s in r['stations']):
            raise ValueError('Invalid station ID')
    else: raise ValueError('Unsupported product')
    return r

def months(r):
    d = datetime.fromtimestamp(epoch(r['start']),timezone.utc).replace(day=1,hour=0,minute=0,second=0)
    while d.timestamp()<epoch(r['end']):
        yield d.strftime('%Y%m')
        d = d.replace(year=d.year+1,month=1) if d.month==12 else d.replace(month=d.month+1)

def get(url, **kwargs):
    last=None
    for attempt in range(3):
        try:
            response=requests.get(url,timeout=(30,180),headers={'Accept-Encoding':'identity'},**kwargs)
            response.raise_for_status(); return response
        except requests.RequestException as exc:
            last=exc
            if getattr(exc.response,'status_code',None)==404: raise
            time.sleep(2**attempt)
    raise last

def identity(url):
    response=requests.head(url,timeout=(30,180),headers={'Accept-Encoding':'identity'})
    response.raise_for_status()
    return {'bytes':int(response.headers['Content-Length']), 'etag':response.headers.get('ETag'),
            'last_modified':response.headers.get('Last-Modified'), 'content_encoding':response.headers.get('Content-Encoding')}

def inventory(r, root):
    available=sorted(set(re.findall(r'href=[\"\'](?:[^\"\']*/)?(\d{6})/?[\"\']',get(BASE).text)))
    records=[]; missing=[]
    for month in months(r):
        if r['product']=='fields':
            names=[('gribs',f'multi_1.{r["grid"]}.{f}.{month}.grb2',f) for f in sorted(set(r['fields']))]
        else:
            names=[('points',f'multi_1_base.buoys_{f}.{month}.tar.gz',f) for f in ('spec','wmo')]
        for folder,name,code in names:
            url=BASE+month+'/'+folder+'/'+name
            try:
                info=identity(url)
                response=requests.get(url+'.MD5',timeout=(30,120),stream=True,headers={'Accept-Encoding':'identity'})
                if response.status_code==404: md5=None
                else:
                    response.raise_for_status()
                    checksum_text=response.raw.read(decode_content=False).decode('ascii')
                    found=re.search(r'\b[0-9a-fA-F]{32}\b',checksum_text)
                    if not found: raise ValueError('Malformed publisher MD5')
                    md5=found.group().lower()
                records.append({'month':month,'code':code,'filename':name,'url':url,'identity':info,'publisher_md5':md5})
            except requests.HTTPError as exc:
                if exc.response.status_code!=404: raise
                missing.append(url)
    result={'created_utc':now(),'request':r,'request_sha256':digest(r),'available_months':available,
            'files':records,'missing':missing,'total_source_bytes':sum(f['identity']['bytes'] for f in records)}
    save(root/'inventory.json',result)
    (root/'source_README.txt').write_text(get(BASE+'README.txt').text,encoding='utf-8')
    if missing and r['missing_policy']=='error': raise ValueError(f'Missing files: {missing}')
    return result

def download(record, raw, retries=3):
    raw=Path(raw); raw.mkdir(parents=True,exist_ok=True)
    if Path(record['filename']).name!=record['filename'] or '\\' in record['filename']:raise ValueError('Unsafe source filename')
    dest=raw/record['filename']; part=dest.with_name(dest.name+'.part'); side=part.with_name(part.name+'.json')
    verified=dest.with_name(dest.name+'.verified.json')
    expected=record['identity']
    if identity(record['url'])!=expected: raise ValueError('Changed source identity; inventory again in a new run')
    if dest.exists():
        h=hashes(dest)
        if dest.stat().st_size!=expected['bytes'] or (record['publisher_md5'] and h['md5']!=record['publisher_md5']):
            raise ValueError('Corrupt cached source')
        if verified.exists():
            v=read(verified)
            if v['record']!=record or v['sha256']!=h['sha256']:raise ValueError('Changed cached identity or SHA-256')
        elif not record['publisher_md5']:raise ValueError('Cache lacks independent SHA-256 evidence')
        save(verified,{'record':record,**h})
        return dict(record,**h,cache_reused=True)
    if part.exists():
        if not side.exists() or read(side)!=record: raise ValueError('Unbound or changed partial source identity')
        if part.stat().st_size>expected['bytes']: raise ValueError('Partial source exceeds expected length')
    else: save(side,record)
    for attempt in range(retries):
        try:
            offset=part.stat().st_size if part.exists() else 0
            if offset<expected['bytes']:
                headers={'Accept-Encoding':'identity'}
                if offset: headers['Range']=f'bytes={offset}-'
                with requests.get(record['url'],headers=headers,stream=True,timeout=(30,180)) as response:
                    response.raise_for_status()
                    if offset:
                        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {offset}-{expected["bytes"]-1}/{expected["bytes"]}':
                            raise ValueError('Server did not honor validated range')
                    elif response.status_code!=200: raise ValueError('Unexpected full-transfer status')
                    for header,key in [('ETag','etag'),('Last-Modified','last_modified')]:
                        if response.headers.get(header)!=expected[key]: raise ValueError('GET source identity changed')
                    with part.open('ab' if offset else 'wb') as stream:
                        for chunk in response.raw.stream(1024*1024,decode_content=False):
                            stream.write(chunk)
                            if stream.tell()>expected['bytes']: raise ValueError('Source exceeds expected length')
            if part.stat().st_size!=expected['bytes']: raise IOError('Truncated source')
            h=hashes(part)
            if record['publisher_md5'] and h['md5']!=record['publisher_md5']: raise ValueError('Publisher checksum failure')
            part.replace(dest); side.unlink();save(verified,{'record':record,**h})
            print(json.dumps({'download_complete':dest.name,'bytes':expected['bytes']}),flush=True)
            return dict(record,**h,cache_reused=False)
        except ValueError: raise
        except Exception:
            if attempt==retries-1: raise
            if identity(record['url'])!=expected: raise ValueError('Source identity changed during retry')
            time.sleep(2**attempt)

def unique_records(times, arrays, start, end):
    times=np.asarray(times,dtype=np.int64); unique,indices=np.unique(times,return_index=True)
    duplicates=[]
    for t in unique:
        ii=np.flatnonzero(times==t)
        if len(ii)>1:
            for a in arrays:
                if any(not np.array_equal(a[ii[0]],a[i],equal_nan=True) for i in ii[1:]):
                    raise ValueError(f'Conflicting duplicate {t}')
            duplicates.append(int(t))
    selected=indices[(unique>=start)&(unique<end)]
    return times[selected],[a[selected] for a in arrays],{'identical_duplicates_removed':duplicates,'boundary_records_excluded':int(((times<start)|(times>=end)).sum())}

def variable(nc,name,values,dims,units='',**attrs):
    a=np.asarray(values)
    v=nc.createVariable(name,a.dtype,dims,zlib=True,complevel=4,fill_value=np.nan if a.dtype.kind=='f' else False)
    v[:]=np.ma.masked_invalid(a) if a.dtype.kind=='f' else a
    v.units=units
    for key,value in attrs.items(): setattr(v,key,value)
    return v

def check_roundtrip(path, arrays):
    with Dataset(path) as nc:
        for name,a in arrays.items():
            b=nc[name][:]
            if np.ma.isMaskedArray(b): b=b.filled(np.nan)
            if not np.array_equal(a,b,equal_nan=True): raise ValueError('Lossless roundtrip failed: '+name)

def packing_quanta(path):
    """GRIB2 common section-5 packing increment, indexed like GDAL source bands."""
    output={};band=0
    with Path(path).open('rb') as stream:
        while True:
            header=stream.read(16)
            if not header:break
            if len(header)!=16 or header[:4]!=b'GRIB' or header[7]!=2:raise ValueError('Invalid GRIB2 framing')
            length=int.from_bytes(header[8:16],'big');message=header+stream.read(length-16)
            if len(message)!=length or message[-4:]!=b'7777':raise ValueError('Truncated GRIB2 message')
            offset=16
            while offset<length-4:
                size=int.from_bytes(message[offset:offset+4],'big');sec=message[offset:offset+size]
                if size<5 or len(sec)!=size:raise ValueError('Invalid GRIB2 section length')
                if sec[4]==5:
                    band+=1;template=int.from_bytes(sec[9:11],'big')
                    if template in (0,2,3,40,41,42):
                        signed=lambda v:-(v&32767) if v&32768 else v
                        e=signed(int.from_bytes(sec[15:17],'big'));d=signed(int.from_bytes(sec[17:19],'big'))
                        output[band]=2.**e/10.**d
                    else:output[band]=np.nan
                offset+=size
    return output

def extract_fields(r, root, files, month):
    arrays={}; attrs={}; lineage={}; record_metadata={}; coords=None
    start=max(epoch(r['start']),epoch(month[:4]+'-'+month[4:]+'-01T00:00:00Z'))
    d=datetime.fromtimestamp(start,timezone.utc).replace(day=1,hour=0,minute=0,second=0)
    nextd=d.replace(year=d.year+1,month=1) if d.month==12 else d.replace(month=d.month+1)
    end=min(epoch(r['end']),int(nextd.timestamp()))
    time_ref=None
    for f in files:
        quanta=packing_quanta(root/'raw'/f['filename'])
        with rasterio.open(root/'raw'/f['filename']) as ds:
            tr=ds.transform
            if tr.b or tr.d or not ds.crs.is_geographic: raise ValueError('Unsupported rotated/non-geographic grid')
            lon=tr.c+(np.arange(ds.width)+.5)*tr.a; lon=(lon+180)%360-180
            lat=tr.f+(np.arange(ds.height)+.5)*tr.e
            b=r['bbox']; ii=np.flatnonzero((lat>=b[1])&(lat<=b[3])); jj=np.flatnonzero((lon>=b[0])&(lon<=b[2]))
            if not len(ii) or not len(jj) or np.any(np.diff(jj)!=1): raise ValueError('Empty/wrapped native crop')
            window=Window(int(jj[0]),int(ii[0]),len(jj),len(ii))
            ll=lon[jj]; la=lat[ii]; oi=np.argsort(la); oj=np.argsort(ll); la=la[oi];ll=ll[oj]
            c=(la,ll,ds.crs.to_wkt())
            if coords is not None and (not np.array_equal(coords[0],la) or not np.array_equal(coords[1],ll) or coords[2]!=c[2]): raise ValueError('Field grids differ')
            coords=c
            groups={}
            for band in ds.indexes:
                tags=ds.tags(band); key=(tags['GRIB_ELEMENT'],ds.descriptions[band-1] or tags.get('GRIB_SHORT_NAME',''))
                groups.setdefault(key,[]).append((band,tags))
            for (element,level),records in groups.items():
                a=ds.read([v[0] for v in records],window=window,masked=True).filled(np.nan).astype('f4')[:,oi][:,:,oj]
                t=[int(v[1]['GRIB_VALID_TIME'].split()[0]) for v in records]
                tt,aa,info=unique_records(t,[a],start,end); a=aa[0]
                expected=np.arange(((start+10799)//10800)*10800,end,10800,dtype='i8')
                gaps=sorted(set(expected.tolist())-set(tt.tolist()))
                if gaps and r['missing_policy']=='error': raise ValueError('Missing three-hour fields')
                if time_ref is not None and not np.array_equal(tt,time_ref): raise ValueError('Field times differ; separate products required')
                time_ref=tt
                aliases={'hs':'significant_wave_height','tp':'peak_wave_period','dp':'peak_wave_direction'}
                name=aliases[f['code']] if f['code'] in aliases and len(groups)==1 else re.sub('[^a-zA-Z0-9_]','_',f['code']+'_'+element+'_'+level).strip('_')
                arrays[name]=a; tag=records[0][1]
                unit=tag['GRIB_UNIT'].strip('[]')
                attrs[name]={'grib_element':element,'native_description':tag['GRIB_COMMENT'],'grib_level':level,'units':'degree' if unit=='Degree true' else unit,'native_units_label':tag['GRIB_UNIT'],'source_sha256':f['sha256'],'source_url':f['url'],'native_grib_tags_json':json.dumps(tag)}
                if element=='PERPW': attrs[name]['interpretation']='Peak wave period per NOAA product table; original GRIB label retained'
                lineage[name]=dict(info,missing_times=gaps,native_bands=ds.count)
                first_by_time={}
                for band,metadata in records:first_by_time.setdefault(int(metadata['GRIB_VALID_TIME'].split()[0]),(band,metadata))
                record_metadata[name]=[(first_by_time[int(v)][0],int(first_by_time[int(v)][1]['GRIB_REF_TIME'].split()[0]),int(first_by_time[int(v)][1]['GRIB_FORECAST_SECONDS'].split()[0]),quanta[first_by_time[int(v)][0]]) for v in tt]
    if not arrays: raise ValueError('No fields extracted')
    la,ll,crs=coords
    wet=np.logical_or.reduce([np.isfinite(a).any(axis=0) for a in arrays.values()]).astype('u1')
    dest=root/'products'/f'ww3_{r["grid"]}_{month}_native.nc';dest.parent.mkdir(exist_ok=True)
    temp=dest.with_suffix('.nc.tmp')
    with Dataset(temp,'w') as nc:
        for dim,length in [('time',len(time_ref)),('latitude',len(la)),('longitude',len(ll)),('bounds',2)]:nc.createDimension(dim,length)
        variable(nc,'time',time_ref,('time',),'seconds since 1970-01-01 00:00:00 UTC',calendar='standard')
        variable(nc,'latitude',la,('latitude',),'degrees_north',standard_name='latitude')
        variable(nc,'longitude',ll,('longitude',),'degrees_east',standard_name='longitude')
        variable(nc,'source_wet_mask',wet,('latitude','longitude'),'1')
        variable(nc,'latitude_bounds',np.column_stack((la-abs(tr.e)/2,la+abs(tr.e)/2)),('latitude','bounds'),'degrees_north')
        variable(nc,'longitude_bounds',np.column_stack((ll-abs(tr.a)/2,ll+abs(tr.a)/2)),('longitude','bounds'),'degrees_east')
        nc['latitude'].bounds='latitude_bounds';nc['longitude'].bounds='longitude_bounds'
        nc.native_latitude_spacing=abs(tr.e);nc.native_longitude_spacing=abs(tr.a)
        for name,a in arrays.items():
            variable(nc,name,a,('time','latitude','longitude'),**attrs[name])
            source_metadata=np.array(record_metadata[name],dtype='f8')
            variable(nc,name+'_source_band',source_metadata[:,0].astype('i8'),('time',),'1')
            variable(nc,name+'_reference_time',source_metadata[:,1].astype('i8'),('time',),'seconds since 1970-01-01 00:00:00 UTC')
            variable(nc,name+'_forecast_seconds',source_metadata[:,2].astype('i8'),('time',),'s')
            variable(nc,name+'_packing_quantum',source_metadata[:,3],('time',),attrs[name]['units'])
        nc.Conventions='CF-1.8';nc.native_grid_crs=crs;nc.request_json=json.dumps(r);nc.request_sha256=digest(r)
        nc.processing='Native center crop; no interpolation; equal regional duplicates deduplicated; [start,end)'
        nc.lineage_json=json.dumps(lineage);nc.history=now()
    check_roundtrip(temp,dict(arrays,time=time_ref,latitude=la,longitude=ll,source_wet_mask=wet));temp.replace(dest)
    return {'path':str(dest.relative_to(root)),'sha256':hashes(dest)['sha256'],'product':'fields','times':len(time_ref),'wet_points':int(wet.sum()),'lineage':lineage}

def selected_members(path,stations,kind,month):
    wanted={f'multi_1.{s}.{kind}.{month}':s for s in stations}; result={}
    with tarfile.open(path,'r:gz') as tar:
        for member in tar:
            p=PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or '\\' in member.name: raise ValueError('Unsafe archive member path')
            if p.name not in wanted: continue
            if not member.isfile() or member.issym() or member.islnk() or member.size>100*1024*1024: raise ValueError('Unsafe selected member')
            station=wanted[p.name]
            if station in result: raise ValueError('Duplicate station member')
            result[station]=tar.extractfile(member).read().decode('ascii')
    if set(result)!=set(stations): raise ValueError('Selected station missing from archive')
    return result

def parse_spec(text,station):
    lines=iter(text.splitlines()); header=shlex.split(next(lines)); nf,nd,npnt=map(int,header[1:4])
    if npnt!=1 or not (2<=nf<=1000 and 2<=nd<=1000): raise ValueError('Unsupported spectral dimensions')
    def numbers(n):
        a=[]
        while len(a)<n: a.extend(float(v.replace('D','E')) for v in next(lines).split())
        if len(a)!=n: raise ValueError('Spectral count overflow')
        return np.array(a,dtype='f8')
    freq=numbers(nf); direction=numbers(nd); times=[]; env=[]; energy=[]
    for line in lines:
        if not line.strip(): continue
        date,hour=line.split()
        times.append(int(datetime.strptime(date+hour,'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc).timestamp()))
        point=shlex.split(next(lines))
        if point[0].strip()!=station: raise ValueError('Wrong spectral station')
        env.append(list(map(float,point[1:])))
        energy.append(numbers(nf*nd).reshape((nd,nf)).T)
    e=np.array(energy,dtype='f4');e[e<0]=np.nan
    if not np.all(np.diff(freq)>0) or len(np.unique(direction))!=nd: raise ValueError('Invalid spectral axes')
    return np.array(times,dtype='i8'),freq,direction,np.array(env),e

def parse_wmo(text):
    data=[]
    for line in text.splitlines():
        words=line.split()
        if len(words)>=8 and re.fullmatch(r'\d{4}',words[0]):data.append(list(map(float,words[:8])))
    data=np.array(data)
    times=np.array([int(datetime(*map(int,row[:4]),tzinfo=timezone.utc).timestamp()) for row in data],dtype='i8')
    return times,data[:,4:]

def integrate(freq,direction,e):
    freq=np.asarray(freq);direction=np.asarray(direction);e=np.asarray(e)
    # ASCII axes have only 3 significant figures: estimate the logarithmic ratio
    # across the complete axis instead of demanding exact adjacent-bin ratios.
    slope,intercept=np.polyfit(np.arange(len(freq)),np.log(freq),1)
    ratio=float(np.exp(slope));df=.5*(ratio-1/ratio)*freq
    # Periodic midpoint weights, mapped back to the native unsorted axis.
    angle=np.mod(direction,2*np.pi);order=np.argsort(angle);sorted_a=angle[order]
    widths=(np.mod(np.roll(sorted_a,-1)-sorted_a,2*np.pi)+np.mod(sorted_a-np.roll(sorted_a,1),2*np.pi))/2
    dw=np.empty_like(widths);dw[order]=widths
    fitted=np.exp(intercept+slope*np.arange(len(freq)))
    if not np.allclose(freq,fitted,rtol=.006):raise ValueError('Non-geometric spectral frequency axis beyond printed precision')
    return 4*np.sqrt(np.sum(e*df[None,:,None]*dw[None,None,:],axis=(1,2))),df,dw

def extract_points(r,root,files,month):
    sources={f['code']:f for f in files}
    if set(sources)!={'spec','wmo'}: raise ValueError('Point source pair incomplete')
    spec=selected_members(root/'raw'/sources['spec']['filename'],r['stations'],'SPEC',month)
    wmo=selected_members(root/'raw'/sources['wmo']['filename'],r['stations'],'WMO',month)
    output=[]
    for s in r['stations']:
        t,f,d,env,e=parse_spec(spec[s],s);bt,bulk=parse_wmo(wmo[s])
        raw_t=t.copy();raw_bt=bt.copy()
        t,aa,info=unique_records(t,[env,e],epoch(r['start']),epoch(r['end']));env,e=aa
        bt,aa,binfo=unique_records(bt,[bulk],epoch(r['start']),epoch(r['end']));bulk=aa[0]
        gaps={}
        for label,raw,filtered in [('spectra',raw_t,t),('point_bulk',raw_bt,bt)]:
            unique=np.unique(raw)
            if len(unique)<2:raise ValueError('Too few point records to determine cadence')
            cadence=int(np.min(np.diff(unique)));anchor=int(unique[0]);start=epoch(r['start']);end=epoch(r['end'])
            expected=np.arange(anchor+((start-anchor+cadence-1)//cadence)*cadence,end,cadence)
            gaps[label]=sorted(set(expected.tolist())-set(filtered.tolist()))
            if gaps[label] and r['missing_policy']=='error':raise ValueError('Point cadence gap: '+label)
        hs,df,dw=integrate(f,d,e)
        common,ii,jj=np.intersect1d(t,bt,return_indices=True)
        if not len(common) or len(common)!=len(t):raise ValueError('Missing coincident WMO records')
        error=hs[ii]-bulk[jj,2];tolerance=np.maximum(.1,.05*bulk[jj,2])
        passed=np.isfinite(error)&(np.abs(error)<=tolerance)
        qa={'paired_records':len(common),'maximum_absolute_error_m':float(np.abs(error).max()),'mean_absolute_error_m':float(np.abs(error).mean()),'within_tolerance':bool(passed.all()),'tolerance':'max(0.1m,5% rounded WMO Hs)','spectral_cadence_seconds':np.unique(np.diff(t)).tolist(),'bulk_cadence_seconds':np.unique(np.diff(bt)).tolist(),'missing_times':gaps,'spectral_records':len(t),'point_bulk_records':len(bt)}
        if not qa['within_tolerance']: raise ValueError('Spectral height integration failed: '+json.dumps(qa))
        dest=root/'products'/f'ww3_{s}_{month}_spectra.nc';dest.parent.mkdir(exist_ok=True);temp=dest.with_suffix('.nc.tmp')
        arrays={'time':t,'bulk_time':bt,'frequency':f,'direction':d,'frequency_bandwidth':df,'direction_bandwidth':dw,'energy_density':e,'integrated_significant_wave_height':hs,'point_environment':env,'point_bulk':bulk}
        with Dataset(temp,'w') as nc:
            for dim,n in [('time',len(t)),('bulk_time',len(bt)),('frequency',len(f)),('direction',len(d)),('environment_component',env.shape[1]),('bulk_component',4)]:nc.createDimension(dim,n)
            for name,dim in [('time','time'),('bulk_time','bulk_time')]:variable(nc,name,arrays[name],(dim,),'seconds since 1970-01-01 00:00:00 UTC',calendar='standard')
            variable(nc,'frequency',f,('frequency',),'Hz');variable(nc,'direction',d,('direction',),'radian',convention='Cartesian travel direction; native order retained')
            variable(nc,'frequency_bandwidth',df,('frequency',),'Hz');variable(nc,'direction_bandwidth',dw,('direction',),'radian')
            variable(nc,'energy_density',e,('time','frequency','direction'),'m2 Hz-1 radian-1',native_header=spec[s].splitlines()[0])
            variable(nc,'integrated_significant_wave_height',hs,('time',),'m')
            variable(nc,'point_environment',env,('time','environment_component'),component_order='latitude,longitude,depth,wind_speed,wind_direction,current_speed,current_direction',component_units='degree_north,degree_east,m,m/s,degree,m/s,degree')
            variable(nc,'point_bulk',bulk,('bulk_time','bulk_component'),component_order='wind_speed,wind_direction,significant_wave_height,peak_wave_period',component_units='m/s,degree,m,s')
            nc.station_id=s;nc.source_json=json.dumps(files);nc.request_json=json.dumps(r);nc.request_sha256=digest(r)
            nc.integration_qa_json=json.dumps(qa);nc.lineage_json=json.dumps({'spec':info,'wmo':binfo});nc.history=now()
        check_roundtrip(temp,arrays);temp.replace(dest)
        output.append({'path':str(dest.relative_to(root)),'sha256':hashes(dest)['sha256'],'product':'point_spectra','station':s,'times':len(t),'integration':qa})
    return output

def extract(r,root):
    m=read(root/'fetch_manifest.json')
    if m['request_sha256']!=digest(r):raise ValueError('Request changed after fetching')
    for f in m['files']:
        p=root/'raw'/f['filename']
        if not p.exists() or hashes(p)['sha256']!=f['sha256']:raise ValueError('Validated raw source missing or changed')
    products=[]
    for month in sorted(set(f['month'] for f in m['files'])):
        files=[f for f in m['files'] if f['month']==month]
        if r['product']=='fields':products.append(extract_fields(r,root,files,month))
        else:products.extend(extract_points(r,root,files,month))
    result={'created_utc':now(),'request':r,'request_sha256':digest(r),'products':products,'missing_sources':m['missing']}
    save(root/'extraction_manifest.json',result);return result

def health(root,cleanup=False):
    m=read(root/'extraction_manifest.json');f=read(root/'fetch_manifest.json');r=m['request'];checks=[];gaps=[];rounding_samples=0
    if m['request_sha256']!=f['request_sha256']:raise ValueError('Lineage request mismatch')
    for p in m['products']:
        path=(root/p['path']).resolve()
        if not path.is_relative_to(root.resolve()) or hashes(path)['sha256']!=p['sha256']:raise ValueError('Compact product changed')
        with Dataset(path) as nc:
            t=nc['time'][:]
            if not len(t) or np.any(np.diff(t)<=0) or np.any((t<epoch(r['start']))|(t>=epoch(r['end']))):raise ValueError('Invalid product times')
            if nc.request_sha256!=digest(r):raise ValueError('Product request mismatch')
            if p['product']=='fields':
                gaps.extend(g for v in p['lineage'].values() for g in v['missing_times'])
                hs=nc.variables.get('significant_wave_height');tp=nc.variables.get('peak_wave_period')
                if hs is not None and np.ma.min(hs[:])<0:raise ValueError('Negative Hs')
                if tp is not None and hs is not None and np.any((hs[:]>0)&(tp[:]<=0)):raise ValueError('Invalid active-wave period')
                if hs is not None and tp is not None:
                    heights=hs[:].filled(np.nan);periods=tp[:].filled(np.nan)
                    if np.any(np.isfinite(heights)&(heights>0)&(~np.isfinite(periods)|(periods<=0))):raise ValueError('Missing active-wave period')
                    dp=nc.variables.get('peak_wave_direction')
                    if dp is not None:
                        direction=dp[:].filled(np.nan)
                        if np.any((heights>0)&~np.isfinite(direction)):raise ValueError('Missing active-wave direction')
                        quantum=nc['peak_wave_direction_packing_quantum'][:].filled(np.nan)
                        tolerance=.5*quantum[:,None,None]+np.finfo('f4').eps*np.maximum(np.abs(direction),1)
                        outside=np.isfinite(direction)&((direction<0)|(direction>360))
                        if np.any(outside&(~np.isfinite(tolerance)|(direction< -tolerance)|(direction>360+tolerance))):raise ValueError('Direction outside range plus native packing precision')
                        rounding_samples+=int(outside.sum())
            else:
                gaps.extend(g for values in p['integration']['missing_times'].values() for g in values)
                bt=nc['bulk_time'][:]
                if len(bt)<2 or np.any(np.diff(bt)<=0) or np.any((bt<epoch(r['start']))|(bt>=epoch(r['end']))):raise ValueError('Invalid bulk times')
                e=nc['energy_density'][:].filled(np.nan);reconstructed,_,_=integrate(nc['frequency'][:],nc['direction'][:],e)
                if not np.array_equal(reconstructed,nc['integrated_significant_wave_height'][:]):raise ValueError('Spectral moment changed')
                common,ii,jj=np.intersect1d(t,nc['bulk_time'][:],return_indices=True);bulk=nc['point_bulk'][:,2]
                if np.any(np.abs(reconstructed[ii]-bulk[jj])>np.maximum(.1,.05*bulk[jj])):raise ValueError('Independent spectral QA failure')
        checks.append({'path':p['path'],'hash_and_structure':True})
    removed=[]
    for source in f['files']:
        path=(root/'raw'/source['filename']).resolve()
        if not path.is_relative_to((root/'raw').resolve()):raise ValueError('Unsafe manifest source path')
        if path.exists():
            if path.stat().st_size!=source['identity']['bytes'] or hashes(path)['sha256']!=source['sha256']:raise ValueError('Raw source changed')
        elif not (root/'cleanup.json').exists() or source['filename'] not in read(root/'cleanup.json')['removed_sources']:
            raise ValueError('Raw missing without health-gated cleanup record')
    status='healthy' if not m['missing_sources'] and not gaps and checks else 'incomplete'
    result={'created_utc':now(),'status':status,'checks':checks,'products':m['products'],'missing_sources':m['missing_sources'],'missing_times':sorted(set(gaps)),'native_direction_boundary_rounding_samples_retained':rounding_samples,'direction_range_tolerance':'half native GRIB packing quantum plus float32 rounding'}
    save(root/'health.json',result)
    if cleanup and r['raw_retention']=='delete_after_health':
        if status!='healthy':raise ValueError('Incomplete run cannot be cleaned')
        if (root/'cleanup.json').exists() and not any((root/'raw'/source['filename']).exists() for source in f['files']):return result
        save(root/'health_before_cleanup.json',result)
        for source in f['files']:
            path=(root/'raw'/source['filename']).resolve()
            if path.exists():path.unlink()
            verified=path.with_name(path.name+'.verified.json')
            if verified.exists():verified.unlink()
            removed.append(source['filename'])
        save(root/'cleanup.json',{'created_utc':now(),'health_snapshot':'health_before_cleanup.json','health_sha256':hashes(root/'health_before_cleanup.json')['sha256'],'removed_sources':removed,'retained':'compact products, request, checksums and provenance'})
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['inventory','plan','fetch','extract','inspect','health']);parser.add_argument('--request');parser.add_argument('--output',required=True);parser.add_argument('--cleanup',action='store_true')
    a=parser.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    r=request(a.request) if a.request else None
    if a.command in ('inventory','plan','fetch','extract') and r is None:parser.error('--request required')
    if a.command=='inventory':result=inventory(r,root)
    elif a.command=='plan':
        result=read(root/'inventory.json') if (root/'inventory.json').exists() else inventory(r,root)
        if result['request_sha256']!=digest(r):raise ValueError('Inventory request changed')
        result=dict(result,free_bytes=shutil.disk_usage(root).free,minimum_working_space_bytes=4*result['total_source_bytes'])
        if result['free_bytes']<result['minimum_working_space_bytes']:raise ValueError('Insufficient working space')
        save(root/'plan.json',result)
    elif a.command=='fetch':
        plan=read(root/'plan.json')
        if plan['request_sha256']!=digest(r):raise ValueError('Plan request changed')
        with ThreadPoolExecutor(max_workers=3) as pool:files=list(pool.map(lambda f:download(f,root/'raw'),plan['files']))
        result=dict(plan,files=files,completed_utc=now());save(root/'fetch_manifest.json',result)
    elif a.command=='extract':result=extract(r,root)
    elif a.command=='health':
        try:result=health(root,a.cleanup)
        except Exception as exc:
            save(root/'health.json',{'created_utc':now(),'status':'failed','error':str(exc)});raise
    else:
        result={name:read(root/name) for name in ('inventory.json','extraction_manifest.json','health.json','cleanup.json') if (root/name).exists()}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
