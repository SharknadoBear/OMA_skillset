"""Wide-domain UTM projection must meet millimetre delivery-join tolerance."""
from pathlib import Path
import builtins, json, math, tempfile
from unittest.mock import patch
from pyproj import Transformer
from mesh_io import Mesh2DM, Node, projected_xy_values

def main():
    with tempfile.TemporaryDirectory(prefix='metric_projection_') as raw:
        root=Path(raw)
        def mesh(name,coords):
            p=root/(name+'.2dm')
            return Mesh2DM(p,name,{i:Node(i,x,y,5) for i,(x,y) in enumerate(coords,1)},{},{})
        north=mesh('wide_hawaii',[(-162.2,17.395),(-153.4,17.395),(-153.4,24.105),(-162.2,24.105)])
        for m,target in [(north,'EPSG:32604'),(mesh('south',[(167,-42),(175,-38)]),'EPSG:32759')]:
            evidence={}; got=projected_xy_values(m,'EPSG:4326',target,provenance=evidence)
            transformer=Transformer.from_crs('EPSG:4326',target,always_xy=True)
            assert evidence['backend']=='pyproj' and evidence['always_xy'] is True
            error=max(abs(a-b) for node in m.sorted_nodes for a,b in zip(got[node.id],transformer.transform(node.x,node.y)))
            assert error<1e-9
        with patch('pyproj.Transformer.from_crs',side_effect=RuntimeError('fixture transform failure')):
            try: projected_xy_values(north,'EPSG:4326','EPSG:32604')
            except RuntimeError as e: assert 'fixture transform failure' in str(e)
            else: raise AssertionError('Transform failure silently fell back')
        with patch('pyproj.Transformer.from_crs') as factory:
            factory.return_value.transform.return_value=(math.inf,0)
            try: projected_xy_values(north,'EPSG:4326','EPSG:32604')
            except ValueError as e: assert 'Nonfinite projected coordinate' in str(e)
            else: raise AssertionError('Nonfinite transform accepted')
        # Force the optional-dependency branch, proving it remains explicitly
        # approximate and cannot claim the strict wide-domain delivery tolerance.
        real_import=builtins.__import__
        def no_pyproj(name,*args,**kwargs):
            if name=='pyproj' or name.startswith('pyproj.'): raise ImportError('fixture pyproj unavailable')
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',no_pyproj):
            evidence={}; legacy=projected_xy_values(north,'EPSG:4326','EPSG:32604',provenance=evidence)
            assert evidence['backend']=='legacy_wgs84_utm_series'
            assert evidence['accuracy_certification']=='requires_independent_delivery_audit'
            native=mesh('native',[(123,456),(789,321)]); unchanged=projected_xy_values(native,'EPSG:32604','EPSG:32604')
            assert unchanged=={1:(123,456),2:(789,321)}
            try: projected_xy_values(north,'EPSG:4326','EPSG:3857')
            except ValueError as e: assert 'requires pyproj' in str(e)
            else: raise AssertionError('Non-UTM transform accepted without pyproj')
        reference=Transformer.from_crs('EPSG:4326','EPSG:32604',always_xy=True)
        legacy_error=max(abs(a-b) for node in north.sorted_nodes for a,b in zip(legacy[node.id],reference.transform(node.x,node.y)))
        assert legacy_error>0.001
        # Reproduce legacy numbers frozen before this correction; fallback does
        # not silently switch formulas or modify already projected inputs.
        expected_legacy=(1095603.6279995497,1931974.2950954635)
        assert all(math.isclose(a,b,rel_tol=0,abs_tol=1e-6) for a,b in zip(legacy[2],expected_legacy)),legacy[2]
    print(json.dumps({'passed':True,'cases':['wide_northern_utm','southern_utm','axis_order','transform_error_no_fallback','nonfinite_rejection','optional_dependency_provenance','legacy_fallback_values','metric_identity_unchanged','unsupported_missing_dependency_rejected'],'legacy_wide_domain_error_m':legacy_error}))
    return 0

if __name__=='__main__': raise SystemExit(main())
