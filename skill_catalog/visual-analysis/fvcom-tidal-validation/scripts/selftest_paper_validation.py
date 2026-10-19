"""Offline 17-day cold-start audit and 14-day exclusive validation regression."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc
import run_case_validation as runner
import audit_fvcom_production as audit_module
from selftest_run_case_validation import fixture, state_file, sha256, write, rejects

def main():
    start = dt.datetime(2025,3,29,tzinfo=dt.timezone.utc)
    analysis = dt.datetime(2025,4,1,tzinfo=dt.timezone.utc)
    end = dt.datetime(2025,4,15,tzinfo=dt.timezone.utc)
    skill = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='paper_validation_test_') as td:
        args, inputs, old_command = fixture(Path(td)/'project', True, runner, Path(audit_module.__file__), skill)
        request = {'case_id':'synthetic_regional', 'profile':'paper_barotropic', 'tpxo':{'expected_constituent_count':22}}
        write(args.project/'request.json', request)
        nml = args.attempt/'regional_run.nml'
        text = nml.read_text().replace('2024-06-01 00:00:00','2025-03-29 00:00:00').replace('2024-06-04 00:00:00','2025-04-15 00:00:00').replace("STARTUP_TYPE='hotstart'", "STARTUP_TYPE='coldstart'").replace("STARTUP_FILE='startup.nc'", "STARTUP_FILE='none'")
        nml.write_text(text)
        (inputs/'startup.nc').unlink()
        times = pd.date_range(start,end,freq='6min')
        hours = np.arange(len(times))/10
        eta = 0.8*np.cos(2*np.pi*hours*0.0805114007)
        station = args.attempt/'regional_station_timeseries.nc'
        station.unlink()
        with nc.Dataset(station,'w') as ds:
            ds.createDimension('time',len(times));ds.createDimension('station',2);ds.createDimension('clen',20)
            names=ds.createVariable('name_station','S1',('station','clen'))
            for i,name in enumerate(['profileX9','9933333']):
                names[i]=np.asarray(list(name.ljust(20)),dtype='S1')
            ds.createVariable('iint','i4',('time',))[:]=np.arange(len(times))*75
            clock=ds.createVariable('time','f4',('time',));clock.units='days since 1858-11-17 00:00:00'
            clock[:]=nc.date2num(times.to_pydatetime().tolist(),clock.units)
            for name in ('zeta','ua','va'):
                ds.createVariable(name,'f4',('time','station'))[:]=np.repeat(eta[:,None],2,axis=1)
        full=args.attempt/'regional_0001.nc';full.unlink()
        state_file(full,np.arange(137)*2250,[start+dt.timedelta(hours=3*i) for i in range(137)],complete=False)
        last=int((end-start).total_seconds()/4.8)
        restart=args.attempt/'regional_restart.nc';restart.unlink()
        state_file(restart,[last],[end],complete=True)
        (args.attempt/'stdout.log').write_text(f' ! {last} 2025-04-15T00:00:00.000000\nTADA!\n')
        freeze_path=args.attempt/'input_freeze.json'
        freeze=json.loads(freeze_path.read_text())
        freeze['files']={p.name:sha256(p) for p in inputs.iterdir()}
        freeze['input_bundle_sha256']=hashlib.sha256(json.dumps(freeze['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        freeze['run_namelist_sha256']=sha256(nml);write(freeze_path,freeze)
        manifest_path=args.attempt/'attempt_manifest.json';manifest=json.loads(manifest_path.read_text())
        manifest['attempt_namelist_sha256']=sha256(nml);write(manifest_path,manifest)
        observation=args.project/'analysis/observations'
        # Include spin-up and terminal samples on purpose; neither may reach metrics.
        pd.DataFrame({'time':times,'observed':eta+0.3,'predicted':eta+0.1}).to_csv(observation/'water_level/9933333_noaa_waterlevel.csv',index=False)
        current_times=times[:-1]+pd.Timedelta(minutes=3)
        pd.DataFrame({'time':current_times,'east_m_s':eta[:-1],'north_m_s':eta[:-1]}).to_csv(observation/'currents/profileX9/depth_mean.csv',index=False)
        command=[];skip=False
        for i,value in enumerate(old_command):
            if skip:skip=False;continue
            if value in ('--startup-restart-netcdf','--expected-station-records','--expected-full-grid-records'):
                skip=True;continue
            command.append(value)
        completed=subprocess.run(command,capture_output=True,text=True)
        assert completed.returncode==0,completed.stderr
        audit=json.loads((args.attempt/'production_audit.json').read_text())
        assert audit['time_anchor']['iint_at_start']==0
        assert audit['station_time']['record_count']==4081
        assert audit['full_grid_time']['record_count']==137
        result=runner.run_validation(args)
        assert result['workflow_status']=='validation_complete',result
        output=args.project/'analysis'/args.grid_case/f'validation_{args.attempt.name}'
        water=pd.read_csv(output/'tables/9933333_water.csv')
        current=pd.read_csv(output/'tables/profileX9_current.csv')
        for table in (water,current):
            clock=pd.to_datetime(table['time'],utc=True)
            assert len(table)==3360 and clock.min()>=analysis and clock.max()<end
        assert pd.to_datetime(current['time'],utc=True).max()==end-dt.timedelta(minutes=3)
        assert json.loads((output/'validation_lineage.json').read_text())['analysis_end_exclusive'] is True
        nml.write_text(text.replace("STARTUP_TYPE='coldstart'", "STARTUP_TYPE='hotstart'"))
        rejects(lambda:runner.verify_prerequisites(args),'namelist')
    print('paper_barotropic 17-day audit / 14-day validation selftest: PASS')

if __name__=='__main__':
    main()
