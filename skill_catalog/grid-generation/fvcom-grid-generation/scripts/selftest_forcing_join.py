#!/usr/bin/env python3
"""Exercise late forcing publication, unchanged grid evidence and rejection gates."""
from __future__ import annotations
import hashlib,json,shutil,tempfile,unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
import netCDF4 as nc
import numpy as np
from selftest_grid_project import (init_project,promote,publish,validate,mesh_fixture,
    gmsh6_candidate,companions,contract_fixture,tge_source_fixture)
from fvcom_grid_generation.grid_project import FINAL_COMPANIONS
from fvcom_grid_generation.forcing_join import certify_terminal_forcing,audit_terminal_forcing
from fvcom_grid_generation.sms_2dm import write_2dm
from fvcom_grid_generation.tge_topology import audit_tge_boundary_junctions

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v),encoding='utf-8')

class ForcingJoinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage=tempfile.TemporaryDirectory();cls.base=Path(cls.storage.name)/'base'
        p=cls.base;init_project(p,'late_forcing')
        angles=np.arange(6)*np.pi/3
        xy=np.vstack([np.column_stack([-75+.1*np.cos(angles),39+.1*np.sin(angles)]),[-75,39]])
        triangles=np.asarray([[i+1,(i+1)%6+1,7] for i in range(6)])
        chains=[[1,2],[4,5]];nsids=[11,22];ids=[1,2,4,5]
        def mesh(path):
            path.parent.mkdir(parents=True,exist_ok=True)
            return write_2dm(path,xy,np.full(7,5.),triangles,np.empty(0,dtype=int),open_boundary_chains=chains,open_boundary_ids=nsids)
        raw=mesh(p/'06_raw_mesh/_work/gmsh6/raw_mesh.2dm')
        promote(p,'06_raw_mesh',raw,'raw_mesh.2dm',generator_manifest=gmsh6_candidate(raw))
        conditioned=mesh(p/'07_conditioning/_work/mesh.2dm')
        promote(p,'07_conditioning',conditioned,'conditioned_mesh.2dm')
        cls.companion_sources=companions(p,findings=['open_boundary_forcing_incompatible'])
        quality=read(cls.companion_sources['mesh_quality'])
        quality['fvcom_tge_boundary_junction_gate']=audit_tge_boundary_junctions(7,triangles-1,[[0,1],[3,4]])
        write(cls.companion_sources['mesh_quality'],quality)
        write(cls.companion_sources['obc_remap_manifest'],{'chains':[{'nodestring_id':ns,'delivered_node_ids_1based':chain,'delivered_node_count':len(chain),'cyclic':False} for ns,chain in zip(nsids,chains)],
            'cyclicity_contract':{'chains':[{'nodestring_id':ns,'node_count':len(chain),'cyclic':False,'declared_open_boundary_id':str(ns)} for ns,chain in zip(nsids,chains)]}})
        publish(p,mesh=p/'07_conditioning/conditioned_mesh.2dm',companions=cls.companion_sources,
            fvcom_ready=True,submission_eligible=False,obc_status='pass',forcing_status='missing',
            failures=[],open_exterior_source=contract_fixture(p/'evidence'),
            tge_source=tge_source_fixture(p/'source'),basemap_provider='offline')
        write(p/'final/grid_delivery_contract.json',{'schema_version':'fvcom_grid_delivery_contract_v1',
            'mesh':{'sha256':digest(p/'final/fvcom_grid.2dm'),'coordinate_crs':'EPSG:4326','node_count':7,'triangle_count':6},
            'open_boundaries':[{'node_ids':chain,'nodestring_id':ns,'obc_id':str(ns),'cyclic':False,
                'node_order_sha256':hashlib.sha256(json.dumps(chain,separators=(',',':')).encode()).hexdigest()} for ns,chain in zip(nsids,chains)]})
        f=p/'forcing';f.mkdir();points=f/'obc_points.csv'
        point_lines=''.join(f'{i},{xy[i-1,0]:.12f},{xy[i-1,1]:.12f}\n' for i in ids)
        points.write_text('node_id,longitude,latitude\n'+point_lines,encoding='utf-8')
        hashes={'obc_points_sha256':digest(points),'obc_order_sha256':hashlib.sha256(point_lines.encode()).hexdigest(),
            'source_harmonics_sha256':'1'*64,'builder_sha256':'2'*64}
        begin=datetime(2025,4,1,tzinfo=timezone.utc);dates=[begin+timedelta(minutes=6*i) for i in range(3)]
        times=np.asarray([(d-datetime(1858,11,17,tzinfo=timezone.utc)).total_seconds()/86400 for d in dates])
        provenance={'reconstruction_rule_version':'utide_exact_time_nodal_v1','utide_ngflags':[False]*4,
            'utide_version':'0.3.1','interval_minutes':6,'time_coverage_start':dates[0].isoformat(),'time_coverage_end':dates[-1].isoformat()}
        with nc.Dataset(f/'elevation.nc','w') as ds:
            for name,size in [('time',3),('nobc',4),('DateStrLen',26)]:ds.createDimension(name,size)
            ds.createVariable('obc_nodes','i4',('nobc',))[:]=ids
            time=ds.createVariable('time','f8',('time',));time[:]=times;time.units='days since 1858-11-17 00:00:00';time.time_zone='UTC'
            ds.createVariable('Itime','i4',('time',))[:]=np.floor(times)
            ds.createVariable('Itime2','i4',('time',))[:]=[0,360000,720000]
            ds.createVariable('Times','S1',('time','DateStrLen'))[:]=np.asarray([d.strftime('%Y/%m/%d %H:%M:%S.%f') for d in dates],dtype='S26').view('S1').reshape(3,26)
            eta=ds.createVariable('elevation','f4',('time','nobc'),fill_value=-9999);eta[:]=np.arange(12).reshape(3,4)*.1;eta.units='meters'
            ds.setncatts({**hashes,'reconstruction_rule_version':'utide_exact_time_nodal_v1','utide_ngflags_json':json.dumps([False]*4),'utide_version':'0.3.1'})
        hashes['forcing_sha256']=digest(f/'elevation.nc')
        write(f/'forcing_manifest.json',{'status':'ready','blocking_reasons':[],'health':{'status':'pass','problems':[]},
            'interpolation':{'unresolved_count':0},'hashes':hashes,'provenance':provenance,
            'constituents':['M2'],'constituent_count':1,'time_count':3,'obc_node_count':4})

    @classmethod
    def tearDownClass(cls):cls.storage.cleanup()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)/'grid';shutil.copytree(self.base,self.p)
        self.args={'grid_contract':self.p/'final/grid_delivery_contract.json','obc_points':self.p/'forcing/obc_points.csv',
            'forcing':self.p/'forcing/elevation.nc','forcing_manifest':self.p/'forcing/forcing_manifest.json',
            'tge_source':self.p/'source/tge.F','open_exterior_source':self.p/'evidence/contract.json','revision':1}
    def tearDown(self):self.tmp.cleanup()
    def join(self):return certify_terminal_forcing(self.p,**self.args)
    def rehash_forcing(self):
        m=read(self.args['forcing_manifest']);m['hashes']['forcing_sha256']=digest(self.args['forcing']);write(self.args['forcing_manifest'],m)
    def rejection(self):
        with self.assertRaises((ValueError,KeyError)):self.join()
        self.assertFalse((self.p/'08_audit/forcing_joins/v001').exists())
    def test_join_preserves_frozen_evidence(self):
        before={str(p.relative_to(self.p)):digest(p) for p in (self.p/'final').iterdir() if p.name!='fvcom_grid_status.json'}
        self.assertFalse(validate(self.p,require_submission_ready=True)['passed'])
        result=self.join();self.assertTrue(result['passed'],result)
        self.assertEqual(before,{name:digest(self.p/name) for name in before})
        self.assertEqual(result['status']['submission_failure_taxonomy'],[])
        self.assertEqual(read(self.p/'final/mesh_quality.json')['all_quality_findings'],['open_boundary_forcing_incompatible'])
        with self.assertRaisesRegex(ValueError,'already exists'):self.join()
    def test_wrong_mesh(self):
        c=read(self.args['grid_contract']);c['mesh']['sha256']='0'*64;write(self.args['grid_contract'],c);self.rejection()
    def test_wrong_contract_order(self):
        c=read(self.args['grid_contract']);c['open_boundaries'][0]['node_ids']=[2,1];write(self.args['grid_contract'],c);self.rejection()
    def test_wrong_nodestring_identity(self):
        c=read(self.args['grid_contract']);c['open_boundaries'][0]['nodestring_id']=999;write(self.args['grid_contract'],c);self.rejection()
    def test_wrong_cyclicity(self):
        c=read(self.args['grid_contract']);c['open_boundaries'][0]['cyclic']=True;write(self.args['grid_contract'],c);self.rejection()
    def test_duplicate_stable_identity(self):
        c=read(self.args['grid_contract']);c['open_boundaries'][0]['obc_id']=c['open_boundaries'][1]['obc_id'];write(self.args['grid_contract'],c);self.rejection()
    def test_stale_chain_hash(self):
        c=read(self.args['grid_contract']);c['open_boundaries'][0]['node_order_sha256']='0'*64;write(self.args['grid_contract'],c);self.rejection()
    def test_wrong_chain_segmentation(self):
        c=read(self.args['grid_contract']);c['open_boundaries'][0]['node_ids']=[1,2,4];c['open_boundaries'][1]['node_ids']=[5];write(self.args['grid_contract'],c);self.rejection()
    def test_wrong_geographic_points(self):
        self.args['obc_points'].write_text('node_id,longitude,latitude\n2,-74.0,39.0\n');self.rejection()
    def test_wrong_point_order(self):
        self.args['obc_points'].write_text('node_id,longitude,latitude\n1,-75.0,39.0\n');self.rejection()
    def test_stale_forcing_hash(self):
        with nc.Dataset(self.args['forcing'],'a') as ds:ds['elevation'][0,0]=1
        self.rejection()
    def test_masked_actual_forcing(self):
        with nc.Dataset(self.args['forcing'],'a') as ds:ds['elevation'][0,0]=-9999
        self.rehash_forcing();self.rejection()
    def test_nan_actual_forcing(self):
        with nc.Dataset(self.args['forcing'],'a') as ds:ds['elevation'][0,0]=np.nan
        self.rehash_forcing();self.rejection()
    def test_wrong_actual_node_id(self):
        with nc.Dataset(self.args['forcing'],'a') as ds:ds['obc_nodes'][0]=7
        self.rehash_forcing();self.rejection()
    def test_wrong_time_units(self):
        with nc.Dataset(self.args['forcing'],'a') as ds:ds['time'].units='hours since 1858-11-17 00:00:00'
        self.rehash_forcing();self.rejection()
    def test_bad_cadence(self):
        with nc.Dataset(self.args['forcing'],'a') as ds:ds['time'][1]+=.001
        self.rehash_forcing();self.rejection()
    def test_integer_time_disagreement(self):
        with nc.Dataset(self.args['forcing'],'a') as ds:ds['Itime2'][1]=0
        self.rehash_forcing();self.rejection()
    def test_string_time_disagreement(self):
        with nc.Dataset(self.args['forcing'],'a') as ds:ds['Times'][1,:]=ds['Times'][0,:]
        self.rehash_forcing();self.rejection()
    def test_old_nodal_protocol(self):
        m=read(self.args['forcing_manifest']);m['provenance']['utide_ngflags']=[True,False,False,False];write(self.args['forcing_manifest'],m);self.rejection()
    def test_missing_tge(self):
        s=read(self.p/'project_status.json');s.pop('fvcom_tge_source_bound_audit');write(self.p/'project_status.json',s);self.rejection()
    def test_changed_actual_tge_source(self):
        self.args['tge_source'].write_text('ISONB=0\n');self.rejection()
    def test_failed_obc_status(self):
        for name in ['project_status.json','final/fvcom_grid_status.json']:
            s=read(self.p/name);s['obc_status']='fail';write(self.p/name,s)
        self.rejection()
    def test_changed_boundary_map(self):
        path=Path(read(self.args['open_exterior_source'])['map']['path']);before=path.read_bytes()
        try:
            path.write_bytes(b'changed map');self.rejection()
        finally:path.write_bytes(before)
    def test_changed_archived_source_rejected(self):
        self.assertTrue(self.join()['passed']);(self.p/'08_audit/forcing_joins/v001/tge.F').write_text('changed')
        self.assertIn('terminal_forcing_certificate_invalid',validate(self.p,require_submission_ready=True)['failure_taxonomy'])
    def test_status_copies_cannot_change_unrelated_fields(self):
        self.assertTrue(self.join()['passed'])
        for name in ['project_status.json','final/fvcom_grid_status.json','08_audit/forcing_joins/v001/delivery_status.json']:
            s=read(self.p/name);s['regional_refinement_debt']=[];s['unrelated_new_claim']=True;write(self.p/name,s)
        self.assertIn('terminal_forcing_certificate_invalid',validate(self.p,require_submission_ready=True)['failure_taxonomy'])
    def test_other_gate_never_waived(self):
        s=read(self.p/'project_status.json');s['submission_failure_taxonomy'].append('open_exterior_contract_missing');write(self.p/'project_status.json',s);self.rejection()
    def test_false_status_cannot_clear_original_findings(self):
        s=read(self.p/'project_status.json');s.update(submission_eligible=True,submission_failure_taxonomy=[],forcing_status='compatible');write(self.p/'project_status.json',s)
        self.assertIn('submission_findings_unresolved',validate(self.p,require_submission_ready=True)['failure_taxonomy'])
    def test_stale_certificate(self):
        self.assertTrue(self.join()['passed']);p=self.p/'08_audit/forcing_joins/v001/certificate.json';p.write_text(p.read_text()+' ')
        self.assertIn('terminal_forcing_certificate_invalid',validate(self.p,require_submission_ready=True)['failure_taxonomy'])
    def test_changed_certified_forcing(self):
        self.assertTrue(self.join()['passed'])
        with nc.Dataset(self.p/'08_audit/forcing_joins/v001/elevation.nc','a') as ds:ds['elevation'][0,0]=1
        self.assertIn('terminal_forcing_certificate_invalid',validate(self.p,require_submission_ready=True)['failure_taxonomy'])
    def test_status_disagreement(self):
        self.assertTrue(self.join()['passed']);s=read(self.p/'project_status.json');s['forcing_status']='pending';write(self.p/'project_status.json',s)
        self.assertIn('terminal_forcing_certificate_invalid',validate(self.p,require_submission_ready=True)['failure_taxonomy'])
    def test_certificate_removal_cannot_bypass(self):
        self.assertTrue(self.join()['passed']);s=read(self.p/'project_status.json');s.pop('terminal_forcing_certificate');write(self.p/'project_status.json',s)
        self.assertIn('submission_findings_unresolved',validate(self.p,require_submission_ready=True)['failure_taxonomy'])

if __name__=='__main__':unittest.main(verbosity=2)
