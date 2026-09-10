"""Exercise outside-water proxies without weakening current or lineage contracts."""
from pathlib import Path
import copy
import hashlib
import json
import tempfile
from pyproj import Transformer
from prepare_station_mapping import build

def main():
    with tempfile.TemporaryDirectory(prefix='fvcom-proxy-') as tmp:
        root=Path(tmp)
        mesh=root/'grid.2dm'
        mesh.write_text('MESH2D\nE3T 7 1 2 3 1\nND 1 500000 3000000 -2\nND 2 500100 3000000 -4\nND 3 500000 3000100 -6\n')
        digest=hashlib.sha256(mesh.read_bytes()).hexdigest()
        contract=root/'grid.json'
        contract.write_text(json.dumps({'schema_version':'fvcom_grid_delivery_contract_v1','mesh':{'path':'grid.2dm','sha256':digest,'node_count':3,'triangle_count':1,'coordinate_crs':'EPSG:32617'}}))
        geo=Transformer.from_crs(32617,4326,always_xy=True)
        inside=geo.transform(500010,3000010); outside=geo.transform(499990,3000040)
        def row(sid,role,point,eligible):
            return {'id':sid,'role':role,'longitude':point[0],'latitude':point[1],'eligible':eligible,'inside_wet_mesh':eligible,'exclusion_reason':None if eligible else 'outside_actual_wet_mesh'}
        data={'schema':'noaa_coops_validation_station_inventory_v1','mesh_sha256':digest,'stations':[row('I','water_level',inside,True),row('O','water_level',outside,False),row('C','current',outside,False)]}
        inv=root/'inventory.json'; inv.write_text(json.dumps(data))
        strict=build(contract,inv,root/'strict.json')
        assert [r['station_id'] for r in strict['stations']]==['I']
        args={'water_level_mapping_policy':'containing_cell_or_nearest_wet_cell','metric_crs':'EPSG:32617'}
        candidate=build(contract,inv,root/'candidate.json',**args)
        assert candidate['status']=='needs_proxy_review' and candidate['proxy_review_pending']==['O']
        out=next(r for r in candidate['stations'] if r['station_id']=='O')
        assert out['cell_id']==7 and not out['inside_wet_mesh'] and out['depth_m']==4
        assert out['observation_longitude']==outside[0] and out['longitude']!=outside[0]
        assert out['spatial_mapping']['node_weights']==[1/3]*3
        assert 43 < out['spatial_mapping']['distance_m'] < 45
        record = dict(out['spatial_mapping'], station_id='O', selection_approved=True)
        review=root/'review.json'; review.write_text(json.dumps({'schema':'fvcom_waterlevel_proxy_review_v1','status':'passed','mesh_sha256':digest,'metric_crs':'EPSG:32617','stations':[record]}))
        diagnostic={'station_id':'D','longitude':inside[0],'latitude':inside[1],'cell_id':7,'depth_m':4,'role':'model_diagnostic','validation_eligible':False}
        prior=root/'diagnostic.json'; prior.write_text(json.dumps({'mesh_sha256':digest,'stations':[diagnostic]}))
        ready=build(contract,inv,root/'ready.json',proxy_review=review,diagnostic_mapping=prior,**args)
        assert ready['status']=='ready' and ready['stations'][0]==diagnostic and ready['station_count']==3
        assert [r['station_id'] for r in ready['stations']]==['D','I','O']
        assert ready['stations'][2]['spatial_mapping']['geometry_review_sha256']==hashlib.sha256(review.read_bytes()).hexdigest()
        bad=copy.deepcopy(data); bad['stations'][-1]['eligible']=True
        inv.write_text(json.dumps(bad))
        try: build(contract,inv,root/'bad_current.json',proxy_review=review,**args)
        except ValueError as exc: assert 'not inside' in str(exc)
        else: raise AssertionError('Outside current silently received water-level proxy policy')
        moved = copy.deepcopy(data); moved['stations'][1]['longitude'] += 1e-6
        inv.write_text(json.dumps(moved))
        try: build(contract,inv,root/'moved_gauge.json',proxy_review=review,**args)
        except ValueError as exc: assert 'disagree' in str(exc)
        else: raise AssertionError('Stale geometry review accepted changed original gauge coordinate')
        inv.write_text(json.dumps(data)); wrong=json.loads(review.read_text()); wrong['stations'][0]['cell_id']=8; review.write_text(json.dumps(wrong))
        try: build(contract,inv,root/'bad_review.json',proxy_review=review,**args)
        except ValueError as exc: assert 'disagree' in str(exc)
        else: raise AssertionError('Wrong reviewed cell accepted')
    print(json.dumps({'status':'passed','scope':'strict compatibility; outside water proxy; unchanged currents; diagnostic preservation; centroid geometry; review lineage'}))

if __name__=='__main__': main()
