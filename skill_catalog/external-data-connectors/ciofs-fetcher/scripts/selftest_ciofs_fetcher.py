#!/usr/bin/env python3
"""Offline regression suite; synthetic DAP2/ROMS fixtures, no live downloads."""
from __future__ import annotations

import copy
from datetime import timedelta
import json
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest.mock import patch

import netCDF4
import numpy as np

import ciofs_fetcher as cf
import roms_points as rp


def request(**updates):
    r={'schema_version':cf.SCHEMA,'start_utc':'2026-09-16T19:00:00Z',
       'end_utc_exclusive':'2026-09-16T21:00:00Z','guidance':'nowcast',
       'variables':['currents','salt','temp','zeta'],'vertical_views':['surface','bottom','index:1'],
       'points':[{'id':'Exact','longitude':-151.977,'latitude':60.024,'sampling':'bilinear'},
                 {'id':'Nearest','longitude':-151.977,'latitude':60.024}]}
    r.update(updates)
    return cf.normalize_request(r)


class Fixture:
    """Six-by-six C-grid, three sigma levels, two hours with different CF epochs."""
    def __init__(self):
        self.calls=[]
        self.paths=[f'NOAA/CIOFS/MODELS/2026/09/17/ciofs.t00z.20260917.fields.n00{k}.nc' for k in [1,2]]
        j,i=np.mgrid[:6,:6]
        self.grid={'lon_rho':-152+i*.01,'lat_rho':60+j*.01,'h':20+j+i,
                   'angle':np.full((6,6),np.pi/2),'mask_rho':np.ones((6,6)),
                   'mask_u':np.ones((6,5)),'mask_v':np.ones((5,6)),'s_rho':np.array([-.9,-.5,-.1])}
        self.dims={key:('eta_rho','xi_rho') for key in ['lon_rho','lat_rho','h','angle','mask_rho']}
        self.dims.update(mask_u=('eta_u','xi_u'),mask_v=('eta_v','xi_v'),s_rho=('s_rho',),ocean_time=('ocean_time',),
                         wetdry_mask_rho=('ocean_time','eta_rho','xi_rho'),wetdry_mask_u=('ocean_time','eta_u','xi_u'),
                         wetdry_mask_v=('ocean_time','eta_v','xi_v'),u=('ocean_time','s_rho','eta_u','xi_u'),
                         v=('ocean_time','s_rho','eta_v','xi_v'),salt=('ocean_time','s_rho','eta_rho','xi_rho'),
                         temp=('ocean_time','s_rho','eta_rho','xi_rho'),zeta=('ocean_time','eta_rho','xi_rho'))
        self.kinds={key:'Float32' if key in ['u','v','salt','temp','zeta'] else 'Float64' for key in self.dims}

    def arrays(self,hour=0):
        a={k:np.array(v,dtype=float,copy=True) for k,v in self.grid.items()}
        # Both epochs decode to their own requested UTC hour.
        a['ocean_time']=np.array([19*3600 if hour==0 else 20.])
        for name,shape in [('u',(1,3,6,5)),('v',(1,3,5,6)),('salt',(1,3,6,6)),('temp',(1,3,6,6))]:
            base={'u':2.,'v':1.,'salt':30.,'temp':8.}[name]+hour
            a[name]=np.broadcast_to(base+np.arange(3)[None,:,None,None],shape).copy()
        a['zeta']=np.full((1,6,6),.4+hour)
        for name in ['rho','u','v']:a['wetdry_mask_'+name]=a['mask_'+name][None,...].copy()
        return a

    def encoded(self,a,query=None):
        parts=query.split(',') if query else list(a)
        lines=[];chunks=[]
        for part in parts:
            match=re.fullmatch(r'(\w+)((?:\[[^\]]+\])*)',part)
            name,selection=match.groups();arr=a[name]
            slices=[]
            for dimension in re.findall(r'\[([^\]]+)\]',selection):
                bits=list(map(int,dimension.split(':')))
                if len(bits)==1:slices.append(slice(bits[0],bits[0]+1))
                else:slices.append(slice(bits[0],bits[2]+1,bits[1]))
            if slices:arr=arr[tuple(slices)]
            kind=self.kinds[name]
            dimensions=''.join(f'[{dim} = {size}]' for dim,size in zip(self.dims[name],arr.shape))
            lines.append(f'    {kind} {name}{dimensions};')
            chunks.append(struct.pack('>II',arr.size,arr.size)+np.asarray(arr,dtype=rp.KINDS[kind]).tobytes())
        dds='Dataset {\n'+'\n'.join(lines)+'\n} fixture;\n'
        return dds.encode()+b'Data:\n'+b''.join(chunks)

    def das(self,hour=0):
        blocks=[]
        for name in self.dims:
            attrs=[]
            if name=='angle':attrs=['String units "radians";','String long_name "angle between XI-axis and EAST";']
            elif name=='ocean_time':
                units='seconds' if hour==0 else 'hours'
                attrs=[f'String units "{units} since 2026-09-16 00:00:00";','String calendar "proleptic_gregorian";']
            elif name in ['u','v']:attrs=['String units "meter second-1";','Float32 _FillValue 1e37;']
            elif name in ['salt','temp','zeta']:
                attrs=[f'String units "{dict(salt="1e-3",temp="Celsius",zeta="meter")[name]}";','Float32 _FillValue 1e37;']
            blocks.append('  '+name+' {\n    '+'\n    '.join(attrs)+'\n  }')
        return 'Attributes {\n'+'\n'.join(blocks)+'\n}\n'

    def meta(self,hour=0):
        dds=self.encoded(self.arrays(hour)).split(b'Data:\n')[0].decode()
        return rp.parse_metadata(dds,self.das(hour),['u','v','salt','temp','zeta'])

    def http(self,url,r,query=None,allow_missing=False):
        self.calls.append((url,query))
        response=cf.requests.Response();response.status_code=200;response.url=url
        if url.endswith('catalog.xml'):
            paths=self.paths if '/2026/09/17/' in url else []
            response._content=('<catalog>'+''.join(f'<dataset urlPath="{p}" />' for p in paths)+'</catalog>').encode()
        else:
            hour=0 if '.n001.nc' in url else 1
            if url.endswith('.das'):response._content=self.das(hour).encode()
            else:
                body=self.encoded(self.arrays(hour),query)
                response._content=body.split(b'Data:\n')[0] if url.endswith('.dds') else body
        return response


class ScienceTests(unittest.TestCase):
    def setUp(self):self.fixture=Fixture();self.r=request();self.grid=self.fixture.grid

    def processed(self,a=None,point=0):
        p=rp.select_points(self.r,self.grid)[point];indices=rp.view_indices(self.r['vertical_views'],self.grid)
        query,bounds=rp.point_query(p,self.r,indices)
        arr=rp.decode_dods(self.fixture.encoded(self.fixture.arrays() if a is None else a,query))
        rp.validate_subset(arr,bounds,self.grid)
        return rp.extract_point(arr,p,self.r,indices,bounds,self.fixture.meta())

    def test_dap_roundtrip(self):
        payload=self.fixture.encoded(self.fixture.arrays());decoded=rp.decode_dods(payload)
        np.testing.assert_array_equal(decoded['u'],self.fixture.arrays()['u'])
        header,binary=payload.split(b'Data:\n');self.assertEqual(rp.payload_size(header.decode()),len(binary))

    def test_dap_corruption_rejected(self):
        payload=self.fixture.encoded(self.fixture.arrays())
        for bad in [payload[:-1],payload+b'X',payload.replace(b'Data:\n\x00\x00\x00\x24',b'Data:\n\x00\x00\x00\x25',1)]:
            with self.subTest(length=len(bad)),self.assertRaises((ValueError,struct.error)):rp.decode_dods(bad)

    def test_dap_scalar(self):
        blob=b'Dataset {\n Float64 scalar;\n} x;\nData:\n'+struct.pack('>d',3.25)
        self.assertEqual(rp.decode_dods(blob)['scalar'],3.25)

    def test_exact_coordinate_weights(self):
        points=rp.select_points(self.r,self.grid);p=points[0]
        np.testing.assert_allclose([c['weight'] for c in p['support']],[.42,.18,.28,.12],atol=1e-9)
        self.assertEqual(p['offset_m'],0)
        changed=copy.deepcopy(points);changed[0]['support'][0]['weight']+=.01
        with self.assertRaises(ValueError):rp.validate_points(changed,self.r,self.grid)

    def test_nearest_point_is_not_exact_target(self):
        p=rp.select_points(self.r,self.grid)[1]
        self.assertGreater(p['offset_m'],0);self.assertEqual(p['support'][0]['weight'],1)

    def test_no_bilinear_extrapolation(self):
        r=request(points=[{'id':'Outside','longitude':-152.1,'latitude':60.03,'sampling':'bilinear','max_distance_m':50000}])
        with self.assertRaises(ValueError):rp.select_points(r,self.grid)

    def test_static_wet_faces_enforced(self):
        grid=copy.deepcopy(self.grid);grid['mask_u'][:]=0
        with self.assertRaises(ValueError):rp.select_points(self.r,grid)

    def test_surface_bottom_reversed_sigma(self):
        grid=copy.deepcopy(self.grid);grid['s_rho']=grid['s_rho'][::-1]
        self.assertEqual(rp.view_indices(['surface','bottom'],grid),{'surface':0,'bottom':2})
        with self.assertRaises(ValueError):rp.view_indices(['index:3'],grid)

    def test_rotation_and_all_vertical_views(self):
        rows=self.processed()
        for row in rows:
            k={'surface':2,'bottom':0,'index:1':1}[row['view']]
            self.assertAlmostEqual(row['east_m_s'],-(1+k));self.assertAlmostEqual(row['north_m_s'],2+k)
            self.assertAlmostEqual(row['salt'],30+k);self.assertAlmostEqual(row['zeta'],.4,places=7)

    def test_destagger_nonuniform_faces(self):
        a=self.fixture.arrays();a['u'][:]=np.arange(5)[None,None,None,:];a['v'][:]=np.arange(5)[None,None,:,None]
        row=self.processed(a,point=1)[0]
        self.assertAlmostEqual(row['east_m_s'],-1.5);self.assertAlmostEqual(row['north_m_s'],1.5)

    def test_interpolate_vectors_before_speed(self):
        a=self.fixture.arrays();a['u'][:]=np.arange(5)[None,None,None,:];a['v'][:]=np.arange(5)[None,None,:,None]
        row=self.processed(a)[0]
        self.assertAlmostEqual(row['east_m_s'],-1.9,places=8);self.assertAlmostEqual(row['north_m_s'],1.8,places=8)
        self.assertAlmostEqual(row['speed_m_s'],np.hypot(1.9,1.8),places=8)

    def test_rotation_per_contributor(self):
        self.grid['angle'][2:4,2:4]=[[0,np.pi/2],[np.pi,np.pi*1.5]]
        row=self.processed()[0]
        expected=np.array([.42,.18,.28,.12])@np.array([[4,3],[-3,4],[-4,-3],[3,-4]])
        np.testing.assert_allclose([row['east_m_s'],row['north_m_s']],expected,atol=1e-8)

    def test_dynamic_face_masks_do_not_zero_or_destroy_scalars(self):
        a=self.fixture.arrays();a['wetdry_mask_u'][0,2,1]=0
        row=self.processed(a)[0]
        self.assertIsNone(row['east_m_s']);self.assertIsNone(row['north_m_s'])
        self.assertEqual(row['quality']['currents'],'dry_velocity_face');self.assertEqual(row['salt'],32)

    def test_dynamic_rho_masks(self):
        a=self.fixture.arrays();a['wetdry_mask_rho'][0,2,2]=0
        row=self.processed(a)[0]
        self.assertTrue(all(v=='dry_rho' for v in row['quality'].values()));self.assertIsNone(row['zeta'])

    def test_float32_fill_value(self):
        a=self.fixture.arrays();a['u'][0,2,2,1]=np.float32(1e37)
        row=self.processed(a)[0]
        self.assertIsNone(row['east_m_s']);self.assertEqual(row['quality']['currents'],'source_fill_or_nonfinite')

    def test_scalar_only(self):
        self.r=request(variables=['salt','zeta']);row=self.processed()[0]
        self.assertNotIn('east_m_s',row);self.assertEqual(row['salt'],32)

    def test_metadata_gates(self):
        dds=self.fixture.encoded(self.fixture.arrays()).split(b'Data:\n')[0].decode();das=self.fixture.das()
        for bad in [das.replace('radians','degrees'),das.replace('angle between XI-axis and EAST','ambiguous'),das.replace('Float32 _FillValue 1e37;','Float32 scale_factor 0.01;')]:
            with self.assertRaises(ValueError):rp.parse_metadata(dds,bad,['u','v'])

    def test_time_per_file_epochs_and_tolerance(self):
        for hour in [0,1]:
            original,normalized,delta=cf.decode_time(self.fixture.arrays(hour),self.fixture.meta(hour))
            self.assertEqual(normalized.hour,19+hour);self.assertEqual(delta,0)
        a=self.fixture.arrays();a['ocean_time']+=10
        self.assertEqual(cf.decode_time(a,self.fixture.meta())[2],-10)
        a['ocean_time']+=100
        with self.assertRaises(ValueError):cf.decode_time(a,self.fixture.meta())


class ContractTests(unittest.TestCase):
    def test_normalization_and_defaults(self):
        r=request(start_utc='2026-09-16T12:00:00-07:00',variables=['currents','salinity'])
        self.assertEqual(r['start_utc'],'2026-09-16T19:00:00Z');self.assertEqual(r['variables'],['u','v','salt'])
        self.assertEqual(r['max_workers'],4)

    def test_invalid_requests(self):
        for change in [dict(start_utc='2026-09-16T19:00:00'),dict(start_utc='2026-09-17T19:00:00Z'),
                       dict(start_utc='2026-09-16T19:10:00Z'),dict(variables=['u']),dict(vertical_views=['depth_average']),
                       dict(max_workers=True),dict(min_completeness=float('nan')),dict(unknown=1),
                       dict(run_cycle_utc='2026-09-16T18:00:00Z'),dict(points=[{'id':'x','latitude':100,'longitude':0}])]:
            with self.subTest(change=change),self.assertRaises((ValueError,KeyError)):request(**change)
        r=request();r['points'][1]['id']='Exact'
        with self.assertRaises(ValueError):cf.normalize_request(r)

    def test_forecast_contract(self):
        r=request(guidance='forecast',run_cycle_utc='2026-09-16T18:00:00Z');self.assertEqual(r['guidance'],'forecast')
        for updates in [dict(run_cycle_utc='2026-09-16T17:00:00Z'),dict(run_cycle_utc='2026-09-14T18:00:00Z'),dict(start_utc='2026-09-16T18:00:00Z')]:
            with self.assertRaises(ValueError):cf.normalize_request({**r,**updates})

    def test_source_mapping(self):
        prefix='NOAA/CIOFS/MODELS/2026/09/17/ciofs.t00z.20260917.fields.'
        self.assertEqual(cf.parse_source(prefix+'n001.nc')['nominal_valid_utc'],'2026-09-16T19:00:00Z')
        self.assertEqual(cf.parse_source(prefix+'n006.nc')['nominal_valid_utc'],'2026-09-17T00:00:00Z')
        self.assertEqual(cf.parse_source(prefix+'f048.nc')['nominal_valid_utc'],'2026-09-19T00:00:00Z')
        self.assertIsNone(cf.parse_source(prefix+'n000.nc'))

    def test_inventory_cross_midnight_no_shift(self):
        fixture=Fixture()
        with patch.object(cf,'http',fixture.http):inv=cf.inventory(request())
        self.assertEqual(inv['available_hours'],2);self.assertFalse(inv['dates_shifted'])
        self.assertTrue(any('/2026/09/17/' in url for url,_ in fixture.calls))
        fixture.paths=[]
        with patch.object(cf,'http',fixture.http):inv=cf.inventory(request())
        self.assertEqual(inv['available_hours'],0);self.assertEqual(len(inv['missing_hours']),2)

    def test_coverage_no_gap_fill(self):
        r=request(end_utc_exclusive='2026-09-16T23:00:00Z',variables=['currents'],vertical_views=['surface'],min_completeness=.5,max_gap_hours=2)
        p=rp.select_points(r,Fixture().grid)
        plan={'request':r,'points':p,'expected_hours':4}
        rows=[{'time':cf.iso(cf.utc(r['start_utc'])+timedelta(hours=k)),'point_id':point['id'],'view':'surface','quality':{'currents':'ok'}} for point in p for k in [0,2,3]]
        qa=cf.coverage(plan,rows)
        self.assertEqual(qa[0]['maximum_gap_hours'],2);self.assertEqual(qa[0]['longest_missing_run_hours'],1)
        self.assertEqual(qa[0]['valid'],3);self.assertTrue(qa[0]['accepted'])

    def test_example_and_schema_contract(self):
        root=Path(__file__).resolve().parents[1]
        example=cf.read_json(root/'assets/request.example.json');cf.normalize_request(example)
        schema=cf.read_json(root/'references/request.schema.json')
        self.assertEqual(schema['properties']['schema_version']['const'],cf.SCHEMA)
        self.assertTrue(set(example)<=set(schema['properties']))


class WorkflowTests(unittest.TestCase):
    def run_fixture(self,root):
        fixture=Fixture()
        with patch.object(cf,'http',fixture.http):
            plan=cf.plan_request(request(),root)
            manifest=cf.fetch_plan(root/'download_estimate.json',root)
        self.assertTrue(all(o['downloaded_subsets']==2 for o in manifest['outcomes']))
        cf.extract_run(root);self.assertEqual(cf.health(root)['status'],'ok')
        return plan

    def test_end_to_end_and_offline_resume(self):
        with tempfile.TemporaryDirectory(prefix='ciofs-test-') as directory:
            root=Path(directory);self.run_fixture(root)
            with patch.object(cf,'http',side_effect=AssertionError('Network forbidden')):
                manifest=cf.fetch_plan(root/'download_estimate.json',root)
                self.assertTrue(all(o['cached_subsets']==2 and o['downloaded_subsets']==0 for o in manifest['outcomes']))
                cf.extract_run(root);self.assertEqual(cf.health(root)['validated_raw_subsets'],4)
            with self.assertRaises(ValueError):cf.bind_run(root,request(min_completeness=.5))

    def test_corrupt_raw_cache_fails(self):
        with tempfile.TemporaryDirectory(prefix='ciofs-test-') as directory:
            root=Path(directory);plan=self.run_fixture(root)
            path=cf.cache_path(root,plan['objects'][0],plan['points'][0])
            with path.open('ab') as stream:stream.write(b'corrupt')
            with self.assertRaises(ValueError):cf.health(root)

    def test_changed_netcdf_values_fail_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory(prefix='ciofs-test-') as directory:
            root=Path(directory);self.run_fixture(root)
            with netCDF4.Dataset(root/'point_series.nc','a') as ds:ds['east_m_s'][0,0,0]+=1
            report=cf.read_json(root/'extraction_manifest.json');report['outputs']['point_series.nc']=cf.file_hash(root/'point_series.nc')
            cf.write_json(root/'extraction_manifest.json',report)
            with self.assertRaises(AssertionError):cf.health(root)

    def test_changed_plan_fails(self):
        with tempfile.TemporaryDirectory(prefix='ciofs-test-') as directory:
            root=Path(directory);plan=self.run_fixture(root)
            plan['request']['min_completeness']=.5;cf.write_json(root/'download_estimate.json',plan)
            with self.assertRaises(ValueError):cf.load_plan(root/'download_estimate.json',root)


if __name__=='__main__':unittest.main(verbosity=2)
