"""Regression: exact-time nodal reconstruction is independent of window/chunk size."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
import utide
from utide._time_conversion import _python_gregorian_datenum
from utide._ut_constants import constit_index_dict
from utide.harmonics import FUV

NAMES = ['M2','S2','N2','K2','K1','O1','P1','Q1','MM','MF','M4','MN4','MS4',
         '2N2','S1','2Q1','J1','L2','M3','MU2','NU2','OO1']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builder-path',type=Path,default=Path(__file__).with_name('build_fvcom_tides.py'))
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    spec=importlib.util.spec_from_file_location('nodal_test_builder',args.builder_path.resolve())
    builder=importlib.util.module_from_spec(spec);sys.modules[spec.name]=builder;spec.loader.exec_module(builder)
    times,_=builder.build_time_axis(builder.parse_utc('2025-03-25T00:00:00Z'),builder.parse_utc('2025-05-01T00:00:00Z'),6)
    latitude=np.asarray([28.85,46.25,-21.0])
    amplitudes=np.linspace(.025,.25,22)[:,None]*np.asarray([1.,.8,1.2])[None,:]
    phase=np.deg2rad(np.arange(22)[:,None]*17+np.asarray([359.,1.,45.])[None,:])
    coefficient=amplitudes*np.exp(-1j*phase)
    actual=builder.reconstruct_utide(coefficient,NAMES,times,latitude)
    t=_python_gregorian_datenum(times)
    indices=np.asarray([constit_index_dict[n] for n in NAMES])
    expected=np.empty_like(actual)
    f_span=[]
    for node,lat in enumerate(latitude):
        f,u,v=FUV(t,float(t.mean()),indices,float(lat),[False,False,False,False])
        assert f.shape==u.shape==v.shape==(8881,22)
        f_span.append(float(np.max(np.ptp(f,axis=0))))
        # Independent real cosine expansion, rather than builder's complex sum.
        expected[node]=np.sum(f*amplitudes[:,node][None,:]*np.cos(2*np.pi*(u+v)-phase[:,node][None,:]),axis=1).astype(np.float32)
    reference_error=float(np.max(np.abs(actual.astype(float)-expected)))
    windows=[slice(0,721),slice(3120,3361),slice(8640,8881)]
    window_error=0.
    for window in windows:
        partial=builder.reconstruct_utide(coefficient,NAMES,times[window],latitude)
        window_error=max(window_error,float(np.max(np.abs(partial.astype(float)-actual[:,window]))))
    tests={'matches_independent_exact_time_cosine':reference_error<=1e-6,
           'same_utc_same_tide_across_three_windows':window_error<=1e-6,
           'nodal_amplitude_changes_with_time':all(v>1e-5 for v in f_span)}
    report={'status':'passed' if all(tests.values()) else 'failed','tests':tests,'records':len(times),
            'constituents':NAMES,'latitudes':latitude.tolist(),'max_reference_error_m':reference_error,
            'max_window_error_m':window_error,'max_nodal_amplitude_spans':f_span,
            'builder_path':str(args.builder_path.resolve()),'builder_sha256':hashlib.sha256(args.builder_path.read_bytes()).hexdigest(),
            'utide_version':utide.__version__,'primary_source':'https://github.com/wesleybowman/UTide/blob/v0.3.1/utide/harmonics.py'}
    if args.report: args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not all(tests.values()): raise SystemExit(1)

if __name__=='__main__':main()
